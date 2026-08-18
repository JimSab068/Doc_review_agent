"""
tests/golden_set/run_golden_cloud_smoke.py

One-persona smoke test for the deployed cloud endpoint. Run this after a
redeploy, BEFORE burning Gemini quota on the full 25-case golden set --
it reuses run_golden_cloud.py's own run_persona_cloud() scoring (routing
check, extracted-field checks, the fixed PII-leak scan, and the Atlas
audit-log readback) against a single persona, so a pass here means the
same checks the full run does would also pass for this case.

Usage (PowerShell):
  $env:TIER_B_LIVE = "1"
  $env:AUDIT_MONGO_URI = "mongodb+srv://..."
  python tests/golden_set/run_golden_cloud_smoke.py `
      --endpoint http://loan-kyc-alb-199452990.us-east-2.elb.amazonaws.com `
      --golden tests/golden_set/golden_personas.json `
      --pdfs-dir tests/generated_personas/pdfs `
      --persona-id p_0001

Defaults to p_0001 (clean_approve) since that's the case that surfaced
both the false-positive PII-leak scan and the swallowed audit-write
failure -- it's the best single case to confirm both fixes landed.

Exit code is 0 on pass, 1 on failure, so this can gate a deploy script
or CI step without parsing output.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx

from tests.tier_a_mocked.personas import load_personas
from tests.tier_b_live.live_clients import build_live_audit_log_reader, build_live_audit_log_store
from tests.tier_b_live.rate_limiter import RateLimiter
from tests.golden_set.run_golden_cloud import run_persona_cloud


async def run(args) -> bool:
    if os.environ.get("TIER_B_LIVE") != "1":
        raise RuntimeError(
            "Set TIER_B_LIVE=1 to opt in. This hits your live cloud deployment "
            "and its real Gemini/Chroma/Mongo backends."
        )
    if "AUDIT_MONGO_URI" not in os.environ:
        raise RuntimeError(
            "AUDIT_MONGO_URI is not set. Must point at the SAME Atlas cluster "
            "the deployed ECS task writes to, or the audit-log readback check "
            "will always fail to find anything."
        )

    personas = load_personas(Path(args.golden))
    matches = [p for p in personas if p.persona_id == args.persona_id]
    if not matches:
        raise ValueError(
            f"persona_id={args.persona_id!r} not found in {args.golden}. "
            f"Available ids: {[p.persona_id for p in personas][:10]}..."
        )
    persona = matches[0]

    db_name = os.environ.get("GOLDEN_SET_MONGO_DB", "golden_set_audit")
    coll_name = os.environ.get("GOLDEN_SET_MONGO_COLLECTION", "audit_log")
    store = build_live_audit_log_store(db_name=db_name, collection_name=coll_name)
    reader = build_live_audit_log_reader(store)

    # A single call, so the rate limiter is really just a safety net --
    # max_calls=1 means it can never trigger a wait on the first request.
    rate_limiter = RateLimiter(max_calls=1, period_seconds=60.0)

    print(f"Smoke-testing {args.persona_id!r} against {args.endpoint} ...")
    async with httpx.AsyncClient() as client:
        result = await run_persona_cloud(
            persona, args.endpoint, args.pdfs_dir, client, reader, args.timeout, rate_limiter
        )

    status = "PASS" if result.passed else "FAIL"
    print(
        f"[{status}] {result.persona_id} ({result.category}) route={result.routing} "
        f"expected={result.expected_route} pii_leak_detected={result.pii_leak_detected} "
        f"latency_ms={result.latency_ms:.0f}"
    )
    if result.failures:
        for f in result.failures:
            print(f"    - {f}")
    if result.api_error:
        print(f"    - api_error: {result.api_error}")

    if result.passed:
        print(
            "\nSingle-persona smoke test passed: routing correct, no PII leak "
            "outside extracted_fields, and the audit entry landed in Atlas. "
            "Safe to run the full golden set now."
        )
    else:
        print(
            "\nSmoke test failed -- do NOT burn quota on the full golden-set "
            "run until this is resolved. See failures above."
        )

    return result.passed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--golden", default="tests/golden_set/golden_personas.json")
    p.add_argument("--pdfs-dir", default=os.environ.get("TIER_B_PDFS_DIR", "tests/generated_personas/pdfs"))
    p.add_argument("--persona-id", default="p_0001")
    p.add_argument("--timeout", type=float, default=90.0)
    args = p.parse_args()

    passed = asyncio.run(run(args))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()