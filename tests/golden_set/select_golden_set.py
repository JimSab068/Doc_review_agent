"""
tests/golden_set/select_golden_set.py

Selects a frozen 25-case golden set from the EXISTING 143-persona corpus
and writes tests/golden_set/golden_personas.json + manifest.json. This
does not generate new personas or new ground truth -- it selects a
subset and freezes the corpus's already-established ground_truth as the
golden set's regression expectations.

Run once (or any time you deliberately want to re-derive the set --
re-running does NOT overwrite an existing manifest.json without --force,
since silently regenerating frozen expectations defeats the point of a
golden set):

    python tests/golden_set/select_golden_set.py \
        --personas tests/generated_personas/personas.json \
        --reviewer "Jimmy Sab" \
        --threshold 0.85

Selection principles (Step 3 spec):
  - Prefer stable/representative cases over edge-case-y ones -- same
    reasoning as tier_b_personas.py's docstring for why Tier B itself
    avoids genuinely ambiguous canaries.
  - Exclude personas tied to open/unresolved investigations: the
    [UNRECOVERABLE]-over-use trio (p_0115/p_0119/p_0123) and the known
    image-only-PDF ingestion gap (p_0114/p_0118/p_0122). A golden set
    freezes RESOLVED expectations, not open questions.
  - Exclude the fp_008 fairness pair (p_0140/p_0141) -- both are known,
    accepted false-escalations against expected_route=auto_resolve.
    Freezing a case whose own expected_route is an accepted mismatch
    would make the regression gate check against a value you already
    know is "wrong" by the corpus's own accounting.

ADAPT -- this script assumes Category enum values matching the strings
already used elsewhere in this codebase (metrics.py's category keys,
tier_b_personas.py's category references): clean_approve, clean_deny,
cross_doc_inconsistency_obvious, cross_doc_inconsistency_inferential,
ambiguous_escalate, injection_direct, injection_indirect,
degraded_document. If tests/tier_a_mocked/personas.py's actual Category
enum uses different names, update CATEGORY_BUCKETS below to match --
the script will raise loudly (not silently under-select) if a bucket
comes up empty after exclusions, so a name mismatch surfaces immediately
rather than producing a golden set with fewer than 25 cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tests.tier_a_mocked.personas import load_personas

# Personas excluded from consideration regardless of category -- see
# module docstring for why each group is excluded.
_KNOWN_OPEN_ITEMS = {
    "p_0115", "p_0119", "p_0123",  # [UNRECOVERABLE]-over-use, unresolved
    "p_0114", "p_0118", "p_0122",  # image-only PDFs, no OCR stage
    "p_0140", "p_0141",            # fp_008 pair, accepted false-escalation
}

# category_value -> (target_count, is_fairness_bucket)
# ADAPT: match your real Category enum values (see module docstring).
CATEGORY_BUCKETS: Dict[str, int] = {
    "clean_approve": 3,
    "clean_deny": 2,
    "inconsistency_obvious": 3,
    "inconsistency_inferential": 2,
    "ambiguous_escalate": 4,
    "injection_direct": 2,
    "injection_indirect": 2,
    "degraded_document": 3,
}
# Fairness-pair members are selected separately (by fairness_pair_id,
# not category) since a persona's category and its fairness-pair
# membership are independent attributes.
_FAIRNESS_TARGET_COUNT = 4

_TOTAL_TARGET = sum(CATEGORY_BUCKETS.values()) + _FAIRNESS_TARGET_COUNT  # should be 25


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select(personas, category_buckets: Dict[str, int], fairness_target: int):
    eligible = [p for p in personas if p.persona_id not in _KNOWN_OPEN_ITEMS]

    selected: List[Any] = []
    selected_ids: set[str] = set()

    # 1. Fill category buckets first, deterministically (sorted by ID so
    #    re-running with the same corpus always picks the same personas).
    for category_value, count in category_buckets.items():
        candidates = sorted(
            (p for p in eligible if getattr(p.category, "value", p.category) == category_value
             and p.persona_id not in selected_ids),
            key=lambda p: p.persona_id,
        )
        if len(candidates) < count:
            raise ValueError(
                f"Category '{category_value}' needs {count} personas but only "
                f"{len(candidates)} eligible candidates exist after exclusions. "
                f"Check CATEGORY_BUCKETS matches your real Category enum values, "
                f"or lower the target count for this category."
            )
        chosen = candidates[:count]
        selected.extend(chosen)
        selected_ids.update(p.persona_id for p in chosen)

    # 2. Fill fairness-pair slots -- prefer whole pairs (2 personas each)
    #    not already selected, until the target count is reached.
    pairs: Dict[str, List[Any]] = {}
    for p in eligible:
        pair_id = getattr(p.ground_truth, "fairness_pair_id", None)
        if pair_id and p.persona_id not in selected_ids:
            pairs.setdefault(pair_id, []).append(p)

    fairness_selected: List[Any] = []
    for pair_id in sorted(pairs):
        members = pairs[pair_id]
        if len(members) < 2:
            continue  # incomplete pair in the corpus, skip
        if len(fairness_selected) + 2 > fairness_target:
            break
        fairness_selected.extend(sorted(members, key=lambda p: p.persona_id)[:2])

    if len(fairness_selected) < fairness_target:
        raise ValueError(
            f"Needed {fairness_target} fairness-pair personas but only found "
            f"{len(fairness_selected)} after exclusions. Check fairness_pair_id "
            f"coverage in the corpus."
        )

    selected.extend(fairness_selected)
    selected_ids.update(p.persona_id for p in fairness_selected)

    return selected


def _build_manifest(selected, threshold: float, reviewer: str, source_path: Path) -> Dict[str, Any]:
    entries: Dict[str, Any] = {}
    for p in selected:
        expected_route = "human_queue" if p.ground_truth.expected_escalate else "auto_resolve"
        entries[p.persona_id] = {
            "category": getattr(p.category, "value", p.category),
            "fairness_pair_id": p.ground_truth.fairness_pair_id,
            "expected_route": expected_route,
            "expected_critical_fields": dict(p.ground_truth.extracted_fields),
            "expected_inconsistencies": list(p.ground_truth.expected_inconsistencies),
            "verification_note": (
                "Frozen from corpus ground_truth at selection time -- not "
                "independently re-verified against a live model run. If you "
                "hand-verify this case against a live run before freezing, "
                "replace this note with what you checked and how."
            ),
        }

    return {
        "metadata": {
            "date_frozen": datetime.now(timezone.utc).isoformat(),
            "reviewer": reviewer,
            "threshold": threshold,
            "source_corpus_path": str(source_path),
            "source_corpus_sha256": _file_sha256(source_path),
            "persona_count": len(selected),
        },
        "personas": entries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", required=True, help="Path to the full 143-persona corpus JSON")
    ap.add_argument("--out-dir", default="tests/golden_set")
    ap.add_argument("--reviewer", required=True, help="Who is freezing this set -- goes in manifest.json")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--force", action="store_true", help="Overwrite an existing manifest.json/golden_personas.json")
    args = ap.parse_args()

    personas_path = Path(args.personas)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    golden_path = out_dir / "golden_personas.json"
    manifest_path = out_dir / "manifest.json"

    if not args.force and (golden_path.exists() or manifest_path.exists()):
        raise SystemExit(
            f"{golden_path} or {manifest_path} already exists. Re-running this "
            f"selection would silently change frozen expectations -- pass --force "
            f"only if you deliberately intend to re-derive the golden set from "
            f"scratch (and understand this discards any hand-verification notes "
            f"already recorded in manifest.json)."
        )

    all_personas = load_personas(personas_path)
    selected = _select(all_personas, CATEGORY_BUCKETS, _FAIRNESS_TARGET_COUNT)

    if len(selected) != 25:
        raise AssertionError(
            f"Internal error: selected {len(selected)} personas, expected exactly "
            f"25. CATEGORY_BUCKETS + fairness target sums to {_TOTAL_TARGET} -- "
            f"fix the constants at the top of this file."
        )

    # Write golden_personas.json preserving the SOURCE corpus's own raw
    # JSON shape for each selected entry (not a re-serialization of the
    # parsed Persona dataclass), so downstream code that already knows
    # how to read personas.json (load_personas) can read this file too
    # with zero format drift.
    raw = json.loads(personas_path.read_text(encoding="utf-8"))
    raw_entries = raw["personas"] if isinstance(raw, dict) and "personas" in raw else raw
    selected_ids = {p.persona_id for p in selected}
    raw_selected = [e for e in raw_entries if e.get("persona_id") in selected_ids]

    if len(raw_selected) != 25:
        raise AssertionError(
            f"Selected 25 Persona objects but only matched {len(raw_selected)} raw "
            f"JSON entries by persona_id -- personas.json's raw shape may not use "
            f"'persona_id' as the ID key. Check the source file's schema."
        )

    golden_payload = raw_selected if isinstance(raw, list) else {"personas": raw_selected}
    manifest = _build_manifest(selected, args.threshold, args.reviewer, personas_path)

    golden_path.write_text(json.dumps(golden_payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {golden_path} (25 personas)")
    print(f"Wrote {manifest_path}")
    print("\nBy category:")
    from collections import Counter
    counts = Counter(getattr(p.category, "value", p.category) for p in selected)
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"  fairness-pair members: {_FAIRNESS_TARGET_COUNT}")


if __name__ == "__main__":
    main()