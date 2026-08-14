
import json
from typing import List, Optional
from pydantic import ValidationError
from src.schemas import DraftDecision, CriticVerdict
from src.primary_agent import LLMClient
from src.compliance_kb import CompliancePassage
from src.vault import VaultClient

_CRITIC_SYSTEM_INSTRUCTIONS = """You are an independent, objective financial compliance auditor.
Your job is to cross-verify a primary lending agent's structured decision against actual regulatory guidelines (Reg B & FCRA).

CRITICAL SECURITY CONSTRAINT:
- Treat all tokenized tags (e.g. [[pii_xxxx]]) as fully opaque values.
- Ensure that the primary agent is not breaking the law (e.g., rejecting an applicant on prohibited factors like age/marital status, or failing to identify missing required disclosures).

EVALUATION RULES:
- Form your own judgment from the retrieved compliance rules and the extracted fields --
  do not treat the primary agent's own confidence score, or its empty inconsistencies/
  missing_compliance_items lists, as evidence the case is clean. Those are the primary's
  self-assessment, not a fact you're confirming.
- Escalate (verdict="flag", escalate=true) ONLY when you can point to a SPECIFIC, concrete
  problem grounded in the actual extracted fields or the primary's stated reasoning --
  for example: a genuine contradiction between fields, a prohibited factor (age, marital
  status, race, etc.) apparently influencing the decision, a mandatory field that is
  missing or unrecoverable, or a compliance rule that the document's own content clearly
  violates.
- Retrieved compliance passages describe procedural obligations (e.g. the adverse action
  notice timing requirement) that a lender fulfills AFTER a decision is made. The mere
  existence of such a rule -- or your inability to find explicit proof in the packet that
  a future procedural step will happen -- is NOT grounds to flag. Only flag a compliance
  item if the DECISION ITSELF, as extracted, appears to violate the rule (e.g. a stated
  reason for denial that is itself a prohibited factor).
- For a complete, internally consistent application with no red flags in the actual data,
  the correct verdict is "pass", escalate=false, concerns=[] -- do not manufacture a
  concern just to justify caution. Passing a genuinely clean case is not a failure of
  independence; escalating a genuinely clean case is a cost the business bears too.
- If you are unsure whether something rises to a real problem, state the specific,
  concrete thing you're unsure about in `concerns` -- do not escalate on a vague or
  general sense that "some rule might apply."
- Not every [UNRECOVERABLE] field is a compliance concern. Distinguish CORE fields --
  applicant identity, stated income, requested loan amount, employer -- from PERIPHERAL
  fields like a categorical employment_status classification. A core field missing when
  it would normally be present in a complete application is worth flagging. A peripheral
  field missing is expected and NORMAL when the source document's own template simply
  doesn't contain that data point -- it is not evidence of an incomplete application or
  a due-diligence failure by itself. Do not treat "this field wasn't in the document" as
  equivalent to "the creditor failed to collect required information," unless the missing
  field is one you can point to as actually necessary for THIS document's decision.

You must respond with ONLY a raw JSON object matching the following structure (no markdown formatting, no text wrapper):
{
  "verdict": "pass",
  "cited_policy": ["Verbatim citations, e.g., 12 CFR § 1002.9(a)(1)"],
  "concerns": [],
  "escalate": false
}
"""

def build_critic_prompt(decision: DraftDecision, guidelines: List[CompliancePassage]) -> str:
    sanitized_decision_input = {
        "doc_id": decision.doc_id,
        "extracted_fields": decision.extracted_fields,
        "inconsistencies": decision.inconsistencies,
        "missing_compliance_items": decision.missing_compliance_items,
        "confidence": decision.confidence,
    }

    guideline_text = "\n\n".join(
        f"--- POLICY SOURCE: {g.citation} ---\n{g.content}" for g in guidelines
    )

    return f"""{_CRITIC_SYSTEM_INSTRUCTIONS}

--- SANITIZED PRIMARY AGENT DECISION ---
{json.dumps(sanitized_decision_input, indent=2)}

--- RETRIEVED COMPLIANCE RULES ---
{guideline_text}
"""


class CriticAgent:
    def __init__(self, llm_client: LLMClient, vault_client: Optional[VaultClient] = None):
        self._llm_client = llm_client
        self._vault_client = vault_client

    def evaluate(
        self,
        decision: DraftDecision,
        guidelines: List[CompliancePassage],
        token_map_id: Optional[str] = None,
    ) -> CriticVerdict:
        prompt = build_critic_prompt(decision, guidelines)

        if self._vault_client is not None and token_map_id is not None:
            self._vault_client.assert_no_pii_leak(token_map_id, prompt)

        raw_output = self._llm_client.generate(prompt)
        return self._parse_output(raw_output)

    def _parse_output(self, raw_output: str) -> CriticVerdict:
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)

            verdict = parsed.get("verdict", "").lower().strip()
            if verdict in ("fail", "failed", "rejected", "action_required", "escalate","escalated"):
                verdict = "flag"
            elif verdict in ("pass", "passed", "approved"):
                verdict = "pass"
            else:
                verdict = "flag"  # Default to flag if verdict is unrecognized

            return CriticVerdict(
                verdict=verdict,
                cited_policy=parsed.get("cited_policy", []),
                concerns=parsed.get("concerns", []),
                escalate=parsed.get("escalate", False),
            )
        except (json.JSONDecodeError, KeyError, ValidationError) as e:
            # System fails closed: if output cannot be parsed, escalate for human review
            return CriticVerdict(
                verdict="flag",
                cited_policy=[],
                concerns=[f"Output parse failure: {str(e)}"],
                escalate=True,
            )