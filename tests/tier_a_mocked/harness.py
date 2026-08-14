

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.compliance_kb import CompliancePassage
from src.pipeline import SecureAuditPipeline
from src.schemas import DocumentType
from src.vault import VaultClient

from tests.tier_a_mocked.fixtures import get_scripted_responses
from tests.tier_a_mocked.personas import Persona, get_pdf_paths
from tests.tier_a_mocked.scripted_llm_client import ScriptedLLMClient


class FakeComplianceKB:
    def __init__(self, canned_passages: Optional[List[CompliancePassage]] = None):
        self._canned = canned_passages or [
            CompliancePassage(
                id="stub_reg_b_adverse_action_notice",
                content="Stub passage standing in for a real retrieved policy excerpt.",
                citation="12 CFR § 1002.9(a)(1)",
                metadata={"statute": "Reg B", "topic": "stub"},
            )
        ]

    def query_relevant_policies(self, draft_reasoning: str, n_results: int = 3) -> List[CompliancePassage]:
        return self._canned[:n_results]


@dataclass
class PersonaResult:
    persona_id: str
    category: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    pii_leak_detected: bool = False
    routing: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None


class PersonaTestHarness:
    def __init__(self, pdfs_dir: "Path | str", escalation_threshold: float = 0.85):
        self._pdfs_dir = Path(pdfs_dir)
        self._escalation_threshold = escalation_threshold

    async def run_persona(self, persona: Persona) -> PersonaResult:
        failures: List[str] = []

        primary_response, critic_response = get_scripted_responses(persona)

        llm = ScriptedLLMClient([primary_response, critic_response])
        vault = VaultClient()
        kb = FakeComplianceKB()

        pipeline = SecureAuditPipeline(
            llm_client=llm,
            vault_client=vault,
            kb_client=kb,
            audit_log_writer=None,
            threshold=self._escalation_threshold,
        )

        pdf_paths = get_pdf_paths(persona, self._pdfs_dir)
        # execute_pdf takes one doc_type for the combined document; use the
        # first document's type as the overall packet type (matches how
        # execute_pdf merges all pages into a single Document today).
        doc_type = DocumentType(persona.documents[0].doc_type)

        payload, routing = await pipeline.execute_pdf(pdf_paths, doc_type)

        # --- Independent PII leak check ---
        pii_leak_detected = False
        for span in persona.ground_truth.pii_spans:
            if any(span.raw_value in prompt for prompt in llm.prompts_sent):
                pii_leak_detected = True
                failures.append(f"Raw PII value leaked into an outbound prompt: {span.raw_value!r}")

        # --- Routing / escalation check ---
        expected_route = "human_queue" if persona.ground_truth.expected_escalate else "auto_resolve"
        if routing.route != expected_route:
            failures.append(f"Routing mismatch: expected '{expected_route}', got '{routing.route}'")

        # --- Detokenization check ---
# --- Detokenization check ---
        degraded_extraction_expected = (
            "Document quality insufficient for full field verification"
            in persona.ground_truth.expected_missing_compliance_items
        )

        for field_name, expected_value in persona.ground_truth.extracted_fields.items():
            actual_value = payload["extracted_fields"].get(field_name)

            if expected_value == "[[TOKEN]]":
                known_raw_values = {span.raw_value for span in persona.ground_truth.pii_spans}

                if degraded_extraction_expected and actual_value == "[[NO_TOKEN_FOUND]]":
                    # Extraction failure is the correct outcome for a document flagged as
                    # low quality — what matters is that nothing wrong got detokenized,
                    # and the routing check above already confirms it was escalated.
                    continue

                if actual_value is None or actual_value.startswith("[[PII_"):
                    failures.append(f"Field '{field_name}': still tokenized in final payload: {actual_value!r}")
                elif actual_value not in known_raw_values:
                    failures.append(
                        f"Field '{field_name}': detokenized value {actual_value!r} doesn't match "
                        f"any known raw PII value for this persona"
                    )
            else:
                if actual_value != str(expected_value):
                    failures.append(f"Field '{field_name}': expected {str(expected_value)!r}, got {actual_value!r}")
                # --- Inconsistency detection check ---
        for expected_item in persona.ground_truth.expected_inconsistencies:
            if expected_item not in payload["inconsistencies"]:
                failures.append(f"Expected inconsistency not present in payload: {expected_item!r}")

        return PersonaResult(
            persona_id=persona.persona_id,
            category=persona.category.value,
            passed=not failures,
            failures=failures,
            pii_leak_detected=pii_leak_detected,
            routing=routing.route,
            raw_payload=payload,
        )

    async def run_batch(self, personas: List[Persona]) -> List[PersonaResult]:
        return [await self.run_persona(p) for p in personas]

    @staticmethod
    def summarize(results: List[PersonaResult]) -> Dict[str, Any]:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        leaks = sum(1 for r in results if r.pii_leak_detected)
        by_category: Dict[str, Dict[str, int]] = {}
        for r in results:
            bucket = by_category.setdefault(r.category, {"total": 0, "passed": 0})
            bucket["total"] += 1
            bucket["passed"] += int(r.passed)
        return {
            "total": total,
            "passed": passed,
            "pass_rate": (passed / total) if total else 0.0,
            "pii_leak_count": leaks,
            "pii_leak_rate": (leaks / total) if total else 0.0,
            "by_category": by_category,
        }