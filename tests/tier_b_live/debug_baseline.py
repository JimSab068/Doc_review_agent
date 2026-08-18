"""
Quick, cheap diagnostic: run a handful of personas live and print full
failure detail (routing, confidence, reasoning, and every check that
failed) instead of just [PASS]/[FAIL].

Usage (PowerShell):
  $env:TIER_B_LIVE = "1"
  python tests/tier_b_live/debug_baseline.py --category clean_approve --limit 5
  python tests/tier_b_live/debug_baseline.py --ids p_0001 p_0002 p_0003

Kept separate from run_evaluation.py on purpose: this never writes an
artifact and is meant for fast iteration while diagnosing a specific
regression, not for producing a reusable evaluation result.
"""
from __future__ import annotations
import argparse
import asyncio
import os

from tests.tier_a_mocked.personas import load_personas
from tests.tier_b_live.harness import TierBLiveHarness
from tests.tier_b_live.live_clients import (
    build_live_audit_log_reader,
    build_live_audit_log_store,
    build_live_audit_log_writer,
    build_live_kb_client,
    build_live_llm_client,
    build_live_vault_client,
    build_shared_rate_limiter,
    teardown_live_audit_log,
)


async def run(args):
    if os.environ.get("TIER_B_LIVE") != "1":
        raise RuntimeError("Set TIER_B_LIVE=1 to opt in.")

    all_personas = load_personas(args.personas)

    if args.ids:
        selected = [p for p in all_personas if p.persona_id in set(args.ids)]
        missing = set(args.ids) - {p.persona_id for p in selected}
        if missing:
            raise ValueError(f"Persona IDs not found: {missing}")
    else:
        selected = [p for p in all_personas if p.category.value == args.category][: args.limit]
        if not selected:
            raise ValueError(f"No personas found for category '{args.category}'")

    print(f"Running {len(selected)} persona(s): {[p.persona_id for p in selected]}\n")

    limiter = build_shared_rate_limiter(int(os.environ.get("TIER_B_MAX_CALLS_PER_MINUTE", "10")))
    db = os.environ.get("TIER_B_MONGO_DB", "tier_b_test")
    coll = os.environ.get("TIER_B_MONGO_COLLECTION", "audit_log")
    store = build_live_audit_log_store(db, coll)

    reader = build_live_audit_log_reader(store)

    def _on_progress(r):
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.persona_id}  route={r.routing}  confidence={r.confidence}  expected_route={r.expected_route}")
        if r.reasoning:
            print(f"    routing reason: {r.reasoning}")
        if r.api_error:
            print(f"    api_error: {r.api_error}")
        if r.failures:
            print("    failures:")
            for f in r.failures:
                print(f"      - {f}")

    async def _print_critic_detail(doc_id: str) -> None:
        # The critic's actual concerns/cited_policy never make it into
        # TierBResult.raw_payload -- pipeline.py's final_payload only
        # carries the primary's inconsistencies/missing_compliance_items,
        # not the critic_verdict. It IS written to the audit log though,
        # so pull it back from there -- this is the only place the real
        # "why did the critic flag this" text actually lives.
        entry = await reader.get_by_doc_id(doc_id)
        if entry is None:
            print("    (no audit entry found -- can't inspect critic verdict)")
            return
        critic = entry.get("critic_verdict", {})
        print(f"    critic verdict: {critic.get('verdict')}  escalate={critic.get('escalate')}")
        print(f"    critic cited_policy: {critic.get('cited_policy')}")
        print(f"    critic concerns: {critic.get('concerns')}")
        print()

    try:
        harness = TierBLiveHarness(
            os.environ.get("TIER_B_PDFS_DIR", "tests/generated_personas/pdfs"), args.threshold
        )
        results = await harness.run_batch(
            selected,
            build_live_llm_client(limiter, args.model),
            build_live_vault_client(),
            build_live_kb_client(limiter, os.environ.get("TIER_B_CHROMA_COLLECTION", "compliance_rules")),
            build_live_audit_log_writer(store),
            reader,
            on_progress=_on_progress,
        )
        for r in results:
            if r.raw_payload:
                await _print_critic_detail(r.raw_payload["doc_id"])
    finally:
        teardown_live_audit_log(db, coll)

    summary = harness.summarize(results)
    print(f"Summary: {summary}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--personas", default=os.environ.get("TIER_B_PERSONAS_JSON", "tests/generated_personas/personas.json"))
    p.add_argument("--category", default="clean_approve")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--ids", nargs="*", default=None, help="Explicit persona IDs, overrides --category/--limit")
    p.add_argument("--model", default=os.environ.get("TIER_B_MODEL", "gemini-3.1-flash-lite"))
    p.add_argument("--threshold", type=float, default=float(os.environ.get("TIER_B_ESCALATION_THRESHOLD", "0.85")))
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()