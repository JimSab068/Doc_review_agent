"""
Tier B live test harness.

Deliberately mirrors tests/tier_a_mocked/harness.py's PersonaResult shape
and overall flow so results are easy to compare side by side -- the
difference is entirely in *what's real* and *how strictly outputs are
compared*, not in the harness's structure.

Key differences from Tier A, and why:

- LLM/KB/audit-log clients are real (injected via live_clients.py), not
  fakes -- this is the whole point of Tier B.
- Detokenization checks for extracted_fields values are exact-match only
  where the ground truth value is fixed and unambiguous ([[TOKEN]]
  fields resolving to a known raw PII value). Free-text/numeric fields
  (employer, income, loan amount) are compared after light
  normalization, since a real model may format "$77,862" as "77862" or
  vice versa -- that's immaterial formatting variance, not a bug.
- A parse/API failure (bad JSON, network error, auth error) is recorded
  as its own failure category distinct from a PII/extraction mismatch,
  since those need different triage (infra/prompt issue vs. model
  extraction quality).
- PII leak detection is identical to Tier A and is never relaxed --
  this is the one invariant that must hold with real calls exactly as
  strictly as with mocked ones.
"""

from __future__ import annotations

import re
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline import SecureAuditPipeline
from src.primary_agent import LLMAPIError, LLMOutputParseError
from src.schemas import DocumentType

from tests.tier_a_mocked.personas import Persona, get_pdf_paths


_MONEY_STRIP_RE = re.compile(r"[,$\s]")

_QUOTED_RE = re.compile(r"'([^']+)'")


def _inconsistency_matches(expected: str, actual_list: List[str]) -> bool:
    """A real LLM won't reproduce a hand-written ground-truth sentence
    verbatim even when it correctly identifies the same underlying
    problem -- exact string equality is too strict a bar. Instead,
    pull out the quoted entities in the expected sentence (e.g. the two
    names in a name-mismatch case) and check that some actual
    inconsistency string mentions all of them together. Falls back to
    exact match if the expected string has no quoted entities at all
    (nothing to loosen)."""
    quoted = _QUOTED_RE.findall(expected)
    if not quoted:
        return expected in actual_list
    return any(all(q in actual for q in quoted) for actual in actual_list)


def _normalize_money(value: str) -> str:
    """'$77,862' / '77,862' / '77862' / '77862.0' all normalize to the
    same comparable string, so formatting variance from a real model
    doesn't register as a false failure."""
    stripped = _MONEY_STRIP_RE.sub("", value)
    try:
        return str(int(float(stripped)))
    except ValueError:
        return stripped


@dataclass
class TierBResult:
    persona_id: str
    category: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    pii_leak_detected: bool = False
    routing: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    api_error: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None
    expected_route: Optional[str] = None
    extracted_field_checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fairness_pair_id: Optional[str] = None
    latency_ms: Optional[float] = None


class TierBLiveHarness:
    """Runs personas through the real pipeline. One persona at a time,
    strictly sequential -- required for the rate limiter in
    rate_limiter.py to reason correctly about call spacing, and cheap
    enough given how small the Tier B subset is."""

    def __init__(self, pdfs_dir: "Path | str", escalation_threshold: float = 0.85):
        self._pdfs_dir = Path(pdfs_dir)
        self._escalation_threshold = escalation_threshold

    async def run_persona(
        self,
        persona: Persona,
        llm_client: Any,
        vault_client: Any,
        kb_client: Any,
        audit_log_writer: Any,
        audit_log_reader: Optional[Any] = None,
    ) -> TierBResult:
        started_at = time.perf_counter()
        failures: List[str] = []
        expected_route = "human_queue" if persona.ground_truth.expected_escalate else "auto_resolve"
        extracted_field_checks: Dict[str, Dict[str, Any]] = {}

        pipeline = SecureAuditPipeline(
            llm_client=llm_client,
            vault_client=vault_client,
            kb_client=kb_client,
            audit_log_writer=audit_log_writer,
            threshold=self._escalation_threshold,
        )

        pdf_paths = get_pdf_paths(persona, self._pdfs_dir)
        doc_type = DocumentType(persona.documents[0].doc_type)

        # A parse/API failure is its own category -- record it and bail
        # out of this persona rather than letting an AttributeError on
        # `None` mask the real problem three lines down.
        try:
            payload, routing = await pipeline.execute_pdf(pdf_paths, doc_type)
        except (LLMOutputParseError, LLMAPIError) as exc:
            return TierBResult(
                persona_id=persona.persona_id,
                category=persona.category.value,
                passed=False,
                failures=[f"Live API/parse failure: {exc}"],
                api_error=str(exc),
                expected_route=expected_route,
                fairness_pair_id=persona.ground_truth.fairness_pair_id,
                latency_ms=(time.perf_counter() - started_at) * 1000,
            )
        except ValueError as exc:
    # Ingestion-layer failure (e.g. image-only PDF, no text layer --
    # see ingestion.py). Distinct from an API/parse failure: the model
    # was never called. Recorded, not raised, so one bad persona can't
    # take down the rest of a sequential batch run.
            return TierBResult(
                persona_id=persona.persona_id,
                category=persona.category.value,
                passed=False,
                failures=[f"Ingestion failure: {exc}"],
                api_error=str(exc),
                expected_route=expected_route,
                fairness_pair_id=persona.ground_truth.fairness_pair_id,
                latency_ms=(time.perf_counter() - started_at) * 1000,
            )

        # --- PII leak check (never relaxed vs. Tier A) ---
        pii_leak_detected = False
        prompts_sent = getattr(llm_client, "prompts_sent", None)
        if prompts_sent is not None:
            for span in persona.ground_truth.pii_spans:
                if any(span.raw_value in prompt for prompt in prompts_sent):
                    pii_leak_detected = True
                    failures.append(f"Raw PII value leaked into an outbound prompt: {span.raw_value!r}")
        # If the wrapped live client doesn't expose .prompts_sent (real
        # GeminiClient doesn't, only ScriptedLLMClient does), we instead
        # rely on the hard vault gate inside PrimaryAgent.run --
        # assert_no_pii_leak already would have raised VaultSecurityError
        # before the API call happened, which surfaces as a raised
        # exception rather than a soft pii_leak_detected flag. Either
        # way, a leak can't reach this point silently.

        # --- Routing / escalation check ---
        if routing.route != expected_route:
            failures.append(
                f"Routing mismatch: expected '{expected_route}', got '{routing.route}' "
                f"(confidence={routing.primary_confidence})"
            )

        # --- Detokenization / extraction checks ---
        extracted_fields = payload.get("extracted_fields", {})
        for field_name, expected_value in persona.ground_truth.extracted_fields.items():
            actual_value = extracted_fields.get(field_name)
            check = {"expected": expected_value, "actual": actual_value, "passed": True, "message": None}

            if expected_value == "[[TOKEN]]":
                known_raw_values = {span.raw_value for span in persona.ground_truth.pii_spans}
                still_tokenized = actual_value is None or str(actual_value).startswith("[[PII_")

                if still_tokenized:
                    check.update(passed=False, message=f"still tokenized in final payload: {actual_value!r}")
                    failures.append(f"Field '{field_name}': {check['message']}")
                elif field_name in {"ssn", "date_of_birth", "phone", "email", "account_number"}:
                    # These fields have real pii_spans to check against -- keep
                    # the strict exact-match verification.
                    if actual_value not in known_raw_values:
                        check.update(passed=False, message=f"detokenized value {actual_value!r} does not match known PII")
                        failures.append(
                            f"Field '{field_name}': detokenized value {actual_value!r} doesn't match "
                            f"any known raw PII value for this persona"
                        )
                # else: field (e.g. applicant_name) has no tracked pii_span --
                # detokenization succeeding (checked above) is all we can verify.

                # if actual_value is None or str(actual_value).startswith("[[PII_"):
                #     failures.append(f"Field '{field_name}': still tokenized in final payload: {actual_value!r}")
                # elif actual_value not in known_raw_values:
                #     failures.append(
                #         f"Field '{field_name}': detokenized value {actual_value!r} doesn't match "
                #         f"any known raw PII value for this persona"
                #    )
            else:
                actual_str = "" if actual_value is None else str(actual_value)
                expected_str = str(expected_value)
                if _normalize_money(actual_str) != _normalize_money(expected_str):
                    check.update(passed=False, message=f"expected {expected_str!r}, got {actual_str!r}")
                    failures.append(
                        f"Field '{field_name}': expected {expected_str!r} (~{_normalize_money(expected_str)}), "
                        f"got {actual_str!r} (~{_normalize_money(actual_str)})"
                    )
            extracted_field_checks[field_name] = check

        # --- Inconsistency detection check ---
        # --- Inconsistency detection check ---
        for expected_item in persona.ground_truth.expected_inconsistencies:
            if not _inconsistency_matches(expected_item, payload.get("inconsistencies", [])):
                failures.append(f"Expected inconsistency not present in payload: {expected_item!r}")
        # --- Audit log readback check (real Mongo write actually landed) ---
        # Deliberately reads back via AuditLogReader rather than trusting
        # that no exception was raised during pipeline.execute_pdf --
        # pipeline.py currently swallows audit-log write failures with a
        # printed warning (see the AuditLogIntegrityError fix suggested
        # alongside this harness), so "no exception" does not by itself
        # prove the primary Mongo write succeeded rather than silently
        # degrading to the local fallback file.
        if audit_log_reader is not None:
            entry = await audit_log_reader.get_by_doc_id(payload["doc_id"])
            if entry is None:
                failures.append(
                    f"Audit log entry for doc_id={payload['doc_id']!r} was not found in "
                    f"MongoDB -- write may have silently degraded to the local fallback "
                    f"log instead of the primary store."
                )

        return TierBResult(
            persona_id=persona.persona_id,
            category=persona.category.value,
            passed=not failures,
            failures=failures,
            pii_leak_detected=pii_leak_detected,
            routing=routing.route,
            confidence=routing.primary_confidence,
            reasoning=payload.get("escalation_reason"),
            raw_payload=payload,
            expected_route=expected_route,
            extracted_field_checks=extracted_field_checks,
            fairness_pair_id=persona.ground_truth.fairness_pair_id,
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )

    async def run_batch(
        self,
        personas: List[Persona],
        llm_client: Any,
        vault_client: Any,
        kb_client: Any,
        audit_log_writer: Any,
        audit_log_reader: Optional[Any] = None,
        on_progress: Optional[Any] = None,
    ) -> List[TierBResult]:
        """Strictly sequential -- do not asyncio.gather this. The rate
        limiter assumes calls happen one at a time; running personas
        concurrently would let multiple `generate()` calls race past
        RateLimiter.acquire() simultaneously and burst well past quota."""
        results: List[TierBResult] = []
        for persona in personas:
            result = await self.run_persona(
                persona, llm_client, vault_client, kb_client, audit_log_writer, audit_log_reader
            )
            results.append(result)
            if on_progress:
                on_progress(result)
        return results

    @staticmethod
    def summarize(results: List[TierBResult]) -> Dict[str, Any]:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        leaks = sum(1 for r in results if r.pii_leak_detected)
        api_errors = sum(1 for r in results if r.api_error)
        return {
            "total": total,
            "passed": passed,
            "pass_rate": (passed / total) if total else 0.0,
            "pii_leak_count": leaks,
            "api_error_count": api_errors,
        }
