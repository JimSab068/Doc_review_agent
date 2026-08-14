

# src/primary_agent.py

from __future__ import annotations

import json
import os
from typing import Protocol

from src.schemas import Document, DraftDecision, TokenizedDocument
from src.vault import VaultClient
from src.secret_redaction import safe_exception_message


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Return raw text output from the model (expected to be JSON)."""
        ...


class LLMOutputParseError(Exception):
    """Raised when the model's output can't be parsed into a DraftDecision."""


class LLMAPIError(Exception):
    """Raised when the outbound call to the model provider itself fails."""


_SYSTEM_INSTRUCTIONS = """You are reviewing a loan/KYC document packet.
The document has had personally identifying fields replaced with opaque
tokens of the form [[PII_TYPE_xxxxxxxx]] -- treat these as opaque
identifiers, never attempt to guess or reconstruct their real values.

Extract key loan and applicant fields into `extracted_fields`.

CRITICAL EXTRACTION RULES:
1. EXTRACT VALUES VERBATIM: Do NOT reformat or alter extracted values. Preserve exact text as written in the source document, including currency symbols ($), commas ($45,000), decimals, percentages, and exact PII tokens (e.g., [[PII_NAME_...]]).
2. KEY NAMES: Use exact field keys corresponding to standard document labels (e.g., "applicant_name", "employer", "stated_annual_income", "loan_amount_requested", "requested_amount", "ssn", "date_of_birth", "email", "employment_status").
3. All values in `extracted_fields` MUST be strings.

EVALUATION RULES:
1. If the document packet is complete, coherent, and consistent across all pages, set "inconsistencies": [] and "missing_compliance_items": [].

2. CONFIDENCE CALIBRATION (three tiers — do not treat this as a binary choice):

- 0.95–1.0 ("clean"): All fields present and unambiguous, values consistent across
  every document, no plausible alternative reading of any field.

- 0.80–0.94 ("genuinely uncertain — this tier exists and you should use it"):
  Use this tier whenever a reasonable, informed reviewer could disagree about the
  right conclusion. Concrete triggers include (non-exhaustive):
    - A field is present but only partially legible, or has one plausible value
      among two or more reasonable readings
    - Two documents differ in a way that MIGHT be the same fact stated differently
      (e.g. employer name formatted two ways) rather than a clear contradiction
    - A compliance item is referenced but not clearly satisfied or clearly absent
    - The applicant's situation is edge-case-y in a way policy doesn't obviously
      resolve (borderline income ratio, ambiguous employment status)
  When you use this tier, `inconsistencies` or `missing_compliance_items` should
  usually be non-empty — name the specific thing you're uncertain about, even if
  you don't think it rises to a hard problem. This tier is not a punishment score;
  it is the CORRECT output for a case that is genuinely hard, and defaulting away
  from it because the case "isn't THAT bad" is itself an error.

- <0.80 ("clear problem"): Confirmed contradiction, confirmed missing mandatory
  document, or a compliance violation you can point to directly.

3. FIELD ABSENCE RULE: If a field's value cannot be determined from the document (illegible,
missing page, truncated, or genuinely absent), set that field's value to the exact string
"[UNRECOVERABLE]" — nothing else. Never write "null", "N/A", "Not stated", "Not provided",
"unknown", or leave it as an empty string. This exact sentinel is required so downstream
scoring can distinguish "field missing" from "field extracted as an empty value."

4. KEY NAMES: Use exact field keys corresponding to standard document labels (e.g.,
   "applicant_name", "employer", "stated_annual_income", "loan_amount_requested",
   "requested_amount", "ssn", "date_of_birth", "email"). Only extract "employment_status"
   if the document contains an EXPLICIT categorical label for it (e.g. "Employment Status:
   Full-time") -- do not infer it from employer/income fields being present, and do not
   report it as [UNRECOVERABLE] if the document simply never has this field at all;
   omit the key entirely in that case rather than reporting it as missing.

Do not skip the middle tier. A case that doesn't cleanly qualify for "clean" is
NOT automatically high-confidence just because it also fails to qualify as a
"genuine" red flag — most real ambiguity lives in the middle tier by definition.

Respond ONLY with a valid JSON object matching this schema, no other text:

{
  "extracted_fields": {
    "applicant_name": "[[PII_NAME_...]]",
    "employer": "...",
    "stated_annual_income": "$...",
    "loan_amount_requested": "$..."
  },
  "inconsistencies": [],
  "missing_compliance_items": [],
  "confidence": 0.95,
  "reasoning": "Short explanation of the extraction and confidence score"
}
"""


def build_prompt(tokenized_doc: TokenizedDocument) -> str:
    doc_text = "\n\n".join(f"--- Page {p.page_number} ---\n{p.text}" for p in tokenized_doc.pages)
    return f"{_SYSTEM_INSTRUCTIONS}\n\nDOCUMENT ({tokenized_doc.doc_type.value}):\n{doc_text}"


class PrimaryAgent:
    def __init__(self, llm_client: LLMClient, vault_client: VaultClient, model_name: str = "stub"):
        self._llm_client = llm_client
        self._vault_client = vault_client
        self._model_name = model_name

    def run(self, tokenized_doc: TokenizedDocument) -> DraftDecision:
        prompt = build_prompt(tokenized_doc)

        # Hard privacy gate: abort rather than call the model if any raw
        # PII value for this request is somehow present in the prompt.
        self._vault_client.assert_no_pii_leak(tokenized_doc.token_map_id, prompt)

        raw_output = self._llm_client.generate(prompt)
        return self._parse_output(tokenized_doc.doc_id, raw_output)

    def _parse_output(self, doc_id: str, raw_output: str) -> DraftDecision:
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
        except json.JSONDecodeError as exc:
            raise LLMOutputParseError(f"Model output was not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise LLMOutputParseError(
                f"Model output was valid JSON but not an object: got {type(parsed).__name__}"
        )

        try:
            extracted = parsed.get("extracted_fields", {})
            if isinstance(extracted, dict):
                normalized = {}
                for k, v in extracted.items():
            # Preserve exact string representation verbatim without stripping symbols
                    normalized[str(k)] = str(v) if v is not None else ""
                extracted = normalized

            return DraftDecision(
        doc_id=doc_id,
        extracted_fields=extracted,
        inconsistencies=parsed.get("inconsistencies", []),
        missing_compliance_items=parsed.get("missing_compliance_items", []),
        confidence=float(parsed["confidence"]),
        reasoning=parsed["reasoning"],
        model_used=self._model_name,
    )
        except (KeyError, ValueError, TypeError) as exc:
            raise LLMOutputParseError(f"Model output JSON had invalid/missing fields: {exc}") from exc

class GeminiClient:
    def __init__(self, model_name: str = "gemini-3.1-flash-lite"):
        if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
            raise KeyError(
                "Neither GOOGLE_API_KEY nor GEMINI_API_KEY were found in the environment variables. "
                "Please export your credential string prior to initializing the GeminiClient."
            )

        from google import genai  # noqa: F401

        try:
            self._client = genai.Client()
        except Exception as exc:
            raise LLMAPIError(
                f"Failed to initialize Gemini client: {safe_exception_message(exc)}"
            ) from None

        self._model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
        except Exception as exc:
            raise LLMAPIError(
                f"Gemini API call failed: {safe_exception_message(exc)}"
            ) from None

        return response.text