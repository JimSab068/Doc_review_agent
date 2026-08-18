"""
Tier B runs against a small, deliberately curated subset of the full
persona set -- not all 143. Each real external call costs money/quota
and is rate-limited to a handful of requests per minute, so the subset
is chosen to exercise each real system at least once rather than
exhaustively re-testing structural logic Tier A already covers for free.

# ADAPT: the IDs below are placeholders illustrating the *shape* of a
# good subset (a couple from each category that matters for live
# testing). Swap in real persona_ids from your personas.json once you've
# picked which ones to use -- ideally ones that are stable/representative
# rather than edge-case-y, since Tier B assertions are structural, not
# exact-match, and a genuinely ambiguous persona makes a bad live-test
# canary.
"""

# top of each tier_b_live test file with async tests
from __future__ import annotations

from pathlib import Path
from typing import List

from tests.tier_a_mocked.personas import Persona, load_personas


TIER_B_PERSONA_IDS: List[str] = [
    # -- Baseline: does a real Gemini call even return valid, schema-
    #    conforming JSON for a clean, unambiguous document?
    "p_0001",
    "p_0002",
    # -- Adversarial PII: the one thing Tier A structurally *cannot*
    #    test, since a scripted response never has the opportunity to
    #    try to reconstruct a token. This is Tier B's most important job.
    "p_0107",
    "p_0033",
    # -- Cross-document inconsistency: exercises real KB retrieval
    #    (does Chroma return a genuinely relevant Reg B/FCRA passage for
    #    reasoning the real model actually produced, not a stub).
    "p_0018",
    "p_0044",
    # -- Prompt injection: tests real model robustness, which Tier A's
    #    routing-logic checks can't touch at all.
    "p_0061",
    "p_0075",
    # -- Degraded document: confirms the real model (not scripted logic)
    #    correctly reports low confidence / missing-field extraction
    #    rather than confabulating a value.
    "p_0114",
    # -- Fairness pair: only meaningful against a real model, since a
    #    script can't accidentally exhibit disparate treatment.
    "p_0090",
    "p_0091",
]


def load_tier_b_personas(personas_json_path: "Path | str") -> List[Persona]:
    """Load the full persona set, then filter down to the curated Tier B
    subset. Raises loudly if any curated ID isn't present in the source
    data, so a stale ID list fails at collection time, not silently."""
    all_personas = load_personas(personas_json_path)
    by_id = {p.persona_id: p for p in all_personas}

    missing = [pid for pid in TIER_B_PERSONA_IDS if pid not in by_id]
    if missing:
        raise ValueError(
            f"Tier B persona subset references IDs not found in "
            f"{personas_json_path}: {missing}. Update TIER_B_PERSONA_IDS "
            f"in tier_b_personas.py."
        )

    return [by_id[pid] for pid in TIER_B_PERSONA_IDS]