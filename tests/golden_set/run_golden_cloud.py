"""
tests/golden_set/run_golden_cloud.py

Runs the frozen 25-case golden set against a DEPLOYED cloud endpoint
(ECS/ALB), not the in-process pipeline tier_b_live.harness uses. This is
the one test in the repo that actually exercises the full deployed
stack -- ALB routing, ECS task, Secrets Manager injection, and the real
network path to Atlas/Chroma/Gemini -- rather than pipeline logic
running in the same process as the test.

Ground truth is loaded from the frozen golden_personas.json (via the
same load_personas() everything else uses -- select_golden_set.py wrote
that file preserving the original corpus's raw shape specifically so
this works with zero format drift). manifest.json supplies frozen run
metadata (reviewer, date frozen, threshold) for the report, but scoring
itself is against golden_personas.json's ground_truth directly, since
that's the complete Persona object (including pii_spans, which
manifest.json intentionally omits as a human-readable summary only).

Output artifact/report shape is IDENTICAL to tier_b_live's
run_evaluation.py (same TierBResult fields, same calculate_metrics(),
same render_markdown()) so a cloud golden-set report and a local Tier B
report are directly comparable side by side -- only the metadata block
differs (endpoint URL instead of a model/git_sha assumption, since a
git_sha isn't knowable from an HTTP response alone).

Usage (PowerShell):
  $env:TIER_B_LIVE = "1"
  $env:AUDIT_MONGO_URI = "mongodb+srv://..."
  python tests/golden_set/run_golden_cloud.py `
      --endpoint http://loan-kyc-alb-199452990.us-east-2.elb.amazonaws.com `
      --golden tests/golden_set/golden_personas.json `
      --manifest tests/golden_set/manifest.json `
      --pdfs-dir tests/generated_personas/pdfs

Client-side rate limiting: src/api.py's lifespan() constructs GeminiClient
directly with no RateLimitedLLMClient wrapper -- the DEPLOYED service has
no proactive pacing of its own, only whatever Gemini's API enforces via
429s after the fact. This script paces its own requests using the same
RateLimiter class tier_b_live uses, so a golden-set run doesn't blow
through your Gemini quota. Note the multiplier: each /review call costs
~3 Gemini calls server-side (primary agent generate, critic agent
generate, KB embedding query), not 1 -- --max-requests-per-minute is
requests to /review, already accounted for that multiplier in its
default, not raw Gemini calls.

Requires:
  - httpx  (pip install httpx --break-system-packages if not already present)
  - AUDIT_MONGO_URI pointed at the SAME Atlas cluster the deployed ECS
    task writes to -- this script reads back from golden_set_audit (see
    tests/golden_set/conftest.py's GOLDEN_SET_MONGO_DB) to confirm each
    request's audit entry actually landed, same as tier_b_live does
    locally. Per that conftest's docstring, this database is never torn
    down -- golden-set runs accumulate as build-over-build history.
  - TIER_B_LIVE=1 -- same opt-in gate as every other live/cost-incurring
    script in this repo. This one hits your live ECS deployment AND its
    real Gemini/Chroma/Mongo backends, so the gate matters here more
    than anywhere else -- an accidental run isn't just quota, it's a
    real request against production infrastructure.

Known gap vs. tier_b_live's PII-leak check: TierBLiveHarness's leak
check inspects llm_client.prompts_sent, which doesn't exist here -- an
HTTP client has no visibility into what was sent to Gemini inside the
deployed container. Instead this script checks (a) 'extracted_raw_text'
is absent from the response, matching deployment_smoke_test.sh's guard,
and (b) no raw PII value from any golden persona's ground_truth.pii_spans
appears anywhere in the raw response body. This is a WEAKER guarantee
than tier_b_live's -- it catches a leak that reaches the HTTP boundary,
not one that reaches Gemini but gets scrubbed before the response is
built. Treat a clean run here as "no leak observed at the API boundary,"
not as equivalent proof to tier_b_live's stronger, prompt-level check.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx

from tests.tier_a_mocked.personas import Persona, get_pdf_paths, load_personas
from tests.tier_b_live.harness import TierBResult, _inconsistency_matches, _normalize_money
from tests.tier_b_live.live_clients import build_live_audit_log_reader, build_live_audit_log_store
from tests.tier_b_live.metrics import calculate_metrics
from tests.tier_b_live.rate_limiter import RateLimiter
from tests.tier_b_live.report import render_markdown

# Mirrors src/api.py's _INTERNAL_ONLY_FIELDS -- fields that must never
# leave the process boundary. If this ever shows up in a cloud response,
# that's a deployed-code regression, not a test bug.
_INTERNAL_ONLY_FIELDS = {"extracted_raw_text"}


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


async def _post_review(
    client: httpx.AsyncClient,
    endpoint: str,
    pdf_paths: List[str],
    doc_type: str,
    timeout: float,
) -> tuple[int, Dict[str, Any], str]:
    """Returns (status_code, parsed_json_or_error_dict, raw_text).
    raw_text is kept separately for the PII substring scan, since a
    malformed/non-JSON error response should still be scannable."""
    file_handles = [open(p, "rb") for p in pdf_paths]
    try:
        files = [
            ("files", (Path(p).name, fh, "application/pdf"))
            for p, fh in zip(pdf_paths, file_handles)
        ]
        resp = await client.post(
            f"{endpoint.rstrip('/')}/review",
            files=files,
            data={"doc_type": doc_type},
            timeout=timeout,
        )
    finally:
        for fh in file_handles:
            fh.close()

    raw_text = resp.text
    try:
        body = resp.json()
    except json.JSONDecodeError:
        body = {"detail": raw_text}
    return resp.status_code, body, raw_text


def _early_result(persona: Persona, expected_route: str, failure: str, started_at: float) -> TierBResult:
    return TierBResult(
        persona_id=persona.persona_id,
        category=persona.category.value,
        passed=False,
        failures=[failure],
        api_error=failure,
        expected_route=expected_route,
        fairness_pair_id=persona.ground_truth.fairness_pair_id,
        latency_ms=(time.perf_counter() - started_at) * 1000,
    )


async def run_persona_cloud(
    persona: Persona,
    endpoint: str,
    pdfs_dir: str,
    client: httpx.AsyncClient,
    audit_log_reader,
    timeout: float,
    rate_limiter: RateLimiter,
) -> TierBResult:
    started_at = time.perf_counter()
    failures: List[str] = []
    expected_route = "human_queue" if persona.ground_truth.expected_escalate else "auto_resolve"

    try:
        pdf_paths = get_pdf_paths(persona, pdfs_dir)
    except FileNotFoundError as exc:
        return _early_result(persona, expected_route, f"Fixture PDF missing: {exc}", started_at)

    # The pipeline packet has one doc_type context, not one per file --
    # deployment_smoke_test.sh always sends loan_application; mirror
    # that preference here, falling back to whatever the persona's
    # first document actually is for edge cases.
    doc_type = next(
        (d.doc_type for d in persona.documents if d.doc_type == "loan_application"),
        persona.documents[0].doc_type,
    )

    # RateLimiter.acquire() is a blocking time.sleep, not asyncio.sleep --
    # fine here because run() below calls this strictly one persona at a
    # time (a plain for-loop with await, no gather), same reasoning as
    # rate_limiter.py's own docstring for why tier_b_live does the same.
    rate_limiter.acquire()

    try:
        status_code, body, raw_text = await _post_review(client, endpoint, pdf_paths, doc_type, timeout)
    except httpx.HTTPError as exc:
        return _early_result(
            persona, expected_route, f"HTTP transport failure calling {endpoint}/review: {exc}", started_at
        )

    if status_code != 200:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        return _early_result(persona, expected_route, f"/review returned HTTP {status_code}: {detail}", started_at)

    payload = body

    # --- PII leak checks (weaker than tier_b_live's -- see module docstring) ---
    pii_leak_detected = False
    for field_name in _INTERNAL_ONLY_FIELDS:
        if field_name in payload:
            pii_leak_detected = True
            failures.append(f"'{field_name}' present in cloud API response -- process-boundary leak")

    # Scope the substring scan to everything EXCEPT 'extracted_fields'.
    # Stage 3.8 (late-bound detokenization) is *supposed* to return real
    # SSN/DOB/email/etc. values inside extracted_fields -- that's the
    # product's whole purpose, and it's separately verified above (the
    # "still tokenized" / "detokenized value ... unrecognized" checks).
    # Scanning the full raw response body for any known raw PII value
    # would flag that correct, expected behavior as a leak on every
    # persona whose extraction is supposed to succeed. What we actually
    # want to catch here is a raw value showing up somewhere it should
    # NEVER appear -- reasoning, inconsistencies, missing_compliance_items,
    # escalation_reason, or any other field that only ever carries tokens
    # or agent-generated text.
    leak_scan_payload = {k: v for k, v in payload.items() if k != "extracted_fields"}
    try:
        leak_scan_text = json.dumps(leak_scan_payload, default=str)
    except TypeError:
        leak_scan_text = str(leak_scan_payload)

    for span in persona.ground_truth.pii_spans:
        if span.raw_value and span.raw_value in leak_scan_text:
            pii_leak_detected = True
            failures.append(
                f"Raw PII value found outside extracted_fields in cloud "
                f"response body: {span.raw_value!r}"
            )

    # --- Routing check ---
    routing = payload.get("routing")
    if routing != expected_route:
        failures.append(f"Routing mismatch: expected '{expected_route}', got '{routing!r}'")

    # --- Extraction checks (same normalization rules as tier_b_live.harness) ---
    extracted_fields = payload.get("extracted_fields", {}) or {}
    extracted_field_checks: Dict[str, Dict[str, Any]] = {}
    known_raw_values = {span.raw_value for span in persona.ground_truth.pii_spans}

    for field_name, expected_value in persona.ground_truth.extracted_fields.items():
        actual_value = extracted_fields.get(field_name)
        check = {"expected": expected_value, "actual": actual_value, "passed": True, "message": None}

        if expected_value == "[[TOKEN]]":
            still_tokenized = actual_value is None or str(actual_value).startswith("[[PII_")
            if still_tokenized:
                check.update(passed=False, message=f"still tokenized: {actual_value!r}")
                failures.append(f"Field '{field_name}': still tokenized in cloud response: {actual_value!r}")
            elif field_name in {"ssn", "date_of_birth", "phone", "email", "account_number"}:
                if actual_value not in known_raw_values:
                    check.update(passed=False, message=f"detokenized value {actual_value!r} unrecognized")
                    failures.append(
                        f"Field '{field_name}': detokenized value {actual_value!r} doesn't match "
                        f"any known raw PII value for this persona"
                    )
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

    # --- Inconsistency checks ---
    for expected_item in persona.ground_truth.expected_inconsistencies:
        if not _inconsistency_matches(expected_item, payload.get("inconsistencies", [])):
            failures.append(f"Expected inconsistency not present in cloud response: {expected_item!r}")

    # --- Audit log readback (direct Atlas connection, golden_set_audit db) ---
    doc_id = payload.get("doc_id")
    if audit_log_reader is not None and doc_id:
        entry = await audit_log_reader.get_by_doc_id(doc_id)
        if entry is None:
            failures.append(
                f"Audit log entry for doc_id={doc_id!r} not found in golden_set_audit -- "
                f"deployed ECS task's write may have silently degraded to a local "
                f"fallback log inside the container instead of reaching Atlas."
            )

    return TierBResult(
        persona_id=persona.persona_id,
        category=persona.category.value,
        passed=not failures,
        failures=failures,
        pii_leak_detected=pii_leak_detected,
        routing=routing,
        confidence=None,  # /review's response payload doesn't expose primary_confidence -- see src/api.py
        reasoning=payload.get("escalation_reason"),
        raw_payload=payload,
        expected_route=expected_route,
        extracted_field_checks=extracted_field_checks,
        fairness_pair_id=persona.ground_truth.fairness_pair_id,
        latency_ms=(time.perf_counter() - started_at) * 1000,
    )


async def run(args) -> tuple[Path, Path]:
    if os.environ.get("TIER_B_LIVE") != "1":
        raise RuntimeError(
            "Set TIER_B_LIVE=1 to opt in. This script calls your live cloud "
            "deployment and its real Gemini/Chroma/Mongo backends -- same "
            "opt-in gate as every other live script in this repo."
        )
    if "AUDIT_MONGO_URI" not in os.environ:
        raise RuntimeError(
            "AUDIT_MONGO_URI is not set. This must point at the SAME Atlas "
            "cluster your deployed ECS task writes to, or the audit-log "
            "readback check will always fail to find anything."
        )

    golden_path = Path(args.golden)
    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    personas = load_personas(golden_path)

    manifest_ids = set(manifest.get("personas", {}).keys())
    loaded_ids = {p.persona_id for p in personas}
    if manifest_ids != loaded_ids:
        only_manifest = manifest_ids - loaded_ids
        only_golden = loaded_ids - manifest_ids
        raise ValueError(
            f"golden_personas.json and manifest.json disagree on persona IDs. "
            f"Only in manifest: {only_manifest or None}. Only in golden set: "
            f"{only_golden or None}. These two files should always be "
            f"regenerated together by select_golden_set.py."
        )

    db_name = os.environ.get("GOLDEN_SET_MONGO_DB", "golden_set_audit")
    coll_name = os.environ.get("GOLDEN_SET_MONGO_COLLECTION", "audit_log")
    store = build_live_audit_log_store(db_name=db_name, collection_name=coll_name)
    reader = build_live_audit_log_reader(store)

    rate_limiter = RateLimiter(max_calls=args.max_requests_per_minute, period_seconds=60.0)
    print(
        f"Rate limiting to {args.max_requests_per_minute} /review requests/min "
        f"(~{args.max_requests_per_minute * 3} Gemini calls/min at the ~3-calls-per-request estimate)"
    )

    results: List[TierBResult] = []
    async with httpx.AsyncClient() as client:
        for persona in personas:
            result = await run_persona_cloud(
                persona, args.endpoint, args.pdfs_dir, client, reader, args.timeout, rate_limiter
            )
            status = "PASS" if result.passed else "FAIL"
            print(
                f"[{status}] {result.persona_id} ({result.category}) route={result.routing} "
                f"expected={result.expected_route} latency_ms={result.latency_ms:.0f}"
            )
            if result.failures:
                for f in result.failures:
                    print(f"    - {f}")
            results.append(result)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / f"golden-cloud-{stamp}.json"
    md = out_dir / f"golden-cloud-{stamp}.md"

    serialized = []
    for r in results:
        item = asdict(r)
        # Same redaction as run_evaluation.py's checkpointing -- final
        # payloads can hold detokenized PII and must never be persisted.
        item.pop("raw_payload", None)
        item["extracted_field_checks"] = {
            name: {"passed": bool(check.get("passed"))}
            for name, check in item.get("extracted_field_checks", {}).items()
        }
        serialized.append(item)

    artifact = {
        "metadata": {
            "timestamp": stamp,
            "git_sha": manifest["metadata"].get("source_corpus_sha256", "unknown")[:12],
            "model": f"cloud:{args.endpoint}",
            "threshold": manifest["metadata"].get("threshold"),
            "persona_count": len(personas),
            "golden_set_reviewer": manifest["metadata"].get("reviewer"),
            "golden_set_frozen_date": manifest["metadata"].get("date_frozen"),
        },
        "results": serialized,
    }
    artifact["metrics"] = calculate_metrics(artifact)

    with js.open("x", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    with md.open("x", encoding="utf-8") as f:
        f.write(render_markdown(artifact, artifact["metrics"]))

    return js, md


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--endpoint",
        required=True,
        help="Base URL of the deployed service, e.g. http://loan-kyc-alb-....elb.amazonaws.com",
    )
    p.add_argument("--golden", default="tests/golden_set/golden_personas.json")
    p.add_argument("--manifest", default="tests/golden_set/manifest.json")
    p.add_argument("--pdfs-dir", default=os.environ.get("TIER_B_PDFS_DIR", "tests/generated_personas/pdfs"))
    p.add_argument("--output-dir", default="artifacts/golden_cloud")
    p.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Per-request HTTP timeout in seconds -- cloud calls include real Gemini latency plus network hops",
    )
    p.add_argument(
        "--max-requests-per-minute",
        type=int,
        default=int(os.environ.get("GOLDEN_CLOUD_MAX_REQUESTS_PER_MINUTE", "3")),
        help=(
            "Requests to /review, not raw Gemini calls -- each /review call costs "
            "~3 Gemini calls server-side (primary + critic + KB embedding), so this "
            "defaults conservatively low (3/min ~= 9 Gemini calls/min) to stay under "
            "a typical 10 RPM free-tier ceiling with headroom."
        ),
    )
    args = p.parse_args()
    js, md = asyncio.run(run(args))
    print(f"\nSaved: {js}\nReport: {md}")


if __name__ == "__main__":
    main()