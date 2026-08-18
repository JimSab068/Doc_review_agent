

"""Run Tier B once and write immutable reusable JSON/Markdown artifacts.

Checkpointing: each persona's result is appended to a .jsonl checkpoint
file the moment it completes, not just at the very end. Previously the
final JSON/MD artifact was only written after the entire batch finished,
so any interrupt, crash, or exhausted quota partway through a 143-persona
run threw away every already-completed result -- including the API calls
that had already been paid for in quota. Now, even an interrupted run
leaves a usable .jsonl of whatever finished before the interrupt, and the
final artifact-writing step can resume from a checkpoint instead of
re-running personas that already have a recorded result.
"""
from __future__ import annotations
import argparse, asyncio, json, os, subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from tests.tier_b_live.metrics import calculate_metrics
from tests.tier_b_live.report import render_markdown
from tests.tier_b_live.tier_b_personas import load_tier_b_personas


def _sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _checkpoint_path(output_dir: Path, stamp: str, sha: str) -> Path:
    ckpt_dir = output_dir / "_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"{stamp}-{sha}.jsonl"


def _load_checkpoint(path: Path) -> Dict[str, dict]:
    """Load any already-completed results from a prior interrupted run.
    Keyed by persona_id so a resumed run can skip personas that already
    have a recorded result -- never re-spend quota on a persona that
    already finished."""
    if not path.exists():
        return {}
    completed: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # A partial/truncated last line from an interrupt mid-write.
                # Skip it rather than fail the whole load -- everything
                # before it is still valid.
                continue
            completed[item["persona_id"]] = item
    return completed


def _append_checkpoint(path: Path, result) -> None:
    item = asdict(result)
    # Same redaction as the final artifact: final payloads can contain
    # detokenized PII, so the checkpoint must never hold raw_payload
    # either -- it's not a lower-trust file, it's an on-disk artifact
    # like any other.
    item.pop("raw_payload", None)
    item["extracted_field_checks"] = {
        name: {"passed": bool(check.get("passed"))}
        for name, check in item.get("extracted_field_checks", {}).items()
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, default=str) + "\n")


async def run(args):
    if os.environ.get("TIER_B_LIVE") != "1":
        raise RuntimeError("Set TIER_B_LIVE=1 to opt in to the live evaluation.")

    persons = (
        load_tier_b_personas(args.personas) if args.smoke else load_personas(args.personas)
    )

    stamp = args.resume_stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = _sha()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    js = out / f"{stamp}-{sha}.json"
    md = out / f"{stamp}-{sha}.md"
    ckpt_path = _checkpoint_path(out, stamp, sha)

    already_done = _load_checkpoint(ckpt_path)
    if already_done:
        print(f"[resume] {len(already_done)} persona(s) already completed in {ckpt_path}")

    remaining = [p for p in persons if p.persona_id not in already_done]
    if not remaining:
        print("[resume] All personas already completed -- nothing left to run.")
    else:
        limiter = build_shared_rate_limiter(int(os.environ.get("TIER_B_MAX_CALLS_PER_MINUTE", "5")))
        db = os.environ.get("TIER_B_MONGO_DB", "tier_b_test")
        coll = os.environ.get("TIER_B_MONGO_COLLECTION", "audit_log")
        store = build_live_audit_log_store(db, coll)

        def _on_progress(r):
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] {r.persona_id} route={r.routing} confidence={r.confidence}")
            if r.failures:
                for f in r.failures:
                    print(f"    - {f}")
            # Write through immediately -- this is the whole point.
            _append_checkpoint(ckpt_path, r)

        try:
            await TierBLiveHarness(
                os.environ.get("TIER_B_PDFS_DIR", "tests/generated_personas/pdfs"),
                args.threshold,
            ).run_batch(
                remaining,
                build_live_llm_client(limiter, args.model),
                build_live_vault_client(),
                build_live_kb_client(limiter, os.environ.get("TIER_B_CHROMA_COLLECTION", "compliance_rules")),
                build_live_audit_log_writer(store),
                build_live_audit_log_reader(store),
                on_progress=_on_progress,
            )
        finally:
            teardown_live_audit_log(db, coll)

    # Re-load the checkpoint fresh (covers both a completed run and a
    # resumed one) and assemble the final immutable artifact from it,
    # in original persona order -- not batch-completion order.
    all_completed = _load_checkpoint(ckpt_path)
    missing = [p.persona_id for p in persons if p.persona_id not in all_completed]
    if missing:
        raise RuntimeError(
            f"{len(missing)} persona(s) never completed (interrupted run?): {missing}. "
            f"Re-run the same command -- it will resume from {ckpt_path} "
            f"and only process what's missing."
        )

    serialized = [all_completed[p.persona_id] for p in persons]
    a = {
        "metadata": {
            "timestamp": stamp,
            "git_sha": sha,
            "model": args.model,
            "threshold": args.threshold,
            "persona_count": len(persons),
        },
        "results": serialized,
    }
    a["metrics"] = calculate_metrics(a)

    with js.open("x", encoding="utf-8") as f:
        json.dump(a, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    with md.open("x", encoding="utf-8") as f:
        f.write(render_markdown(a, a["metrics"]))

    return js, md


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--personas", default=os.environ.get("TIER_B_PERSONAS_JSON", "tests/generated_personas/personas.json"))
    p.add_argument("--output-dir", default="artifacts/evaluations")
    p.add_argument("--model", default=os.environ.get("TIER_B_MODEL", "gemini-3.1-flash-lite"))
    p.add_argument("--threshold", type=float, default=float(os.environ.get("TIER_B_ESCALATION_THRESHOLD", "0.85")))
    p.add_argument(
        "--resume-stamp",
        default=None,
        help=(
            "Reuse an existing timestamp (from a prior interrupted run's "
            "checkpoint filename, e.g. 20260813T120000Z) to resume instead "
            "of starting a fresh run. Omit to start a new run."
        ),
    )
    args = p.parse_args()
    js, md = asyncio.run(run(args))
    print(f"Saved: {js}\nReport: {md}")


if __name__ == "__main__":
    main()