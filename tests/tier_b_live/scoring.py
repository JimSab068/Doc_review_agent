"""
Scoring functions for Tier B (real API) persona runs.

Kept separate from harness.py so these can be unit-tested with plain
dicts/strings, with no pipeline, vault, or API call involved at all.
"""


from __future__ import annotations

from typing import Dict, List, Tuple

from src.compliance_kb import CompliancePassage


def score_extraction(
    actual_fields: Dict[str, str], expected_fields: Dict[str, str]
) -> Tuple[float, List[str]]:
    """Per-field exact-match accuracy against ground truth.

    Returns (accuracy, mismatched_field_names). Exact string match is
    intentionally strict -- a model that reformats "$35,000" as "35000"
    counts as a mismatch here. That's a deliberate choice: this number
    feeds a per-field accuracy report (spec 4.3), and silently accepting
    reformatted values would hide real extraction drift. Mismatches are
    a list to review, not proof of a bug on their own.
    """
    if not expected_fields:
        return 1.0, []

    mismatched: List[str] = []
    correct = 0
    for field_name, expected_value in expected_fields.items():
        actual_value = actual_fields.get(field_name)
        if actual_value == expected_value:
            correct += 1
        else:
            mismatched.append(field_name)

    return correct / len(expected_fields), mismatched


def check_groundedness(actual_fields: Dict[str, str], source_text: str) -> List[str]:
    """Return field names whose value cannot be found (case-insensitive
    substring) anywhere in the source document text.

    This is a strict, automated FIRST-PASS hallucination flag, not a
    final verdict -- per spec 4.4, any field flagged here should go to
    human/LLM-judge review, not be treated as a confirmed hallucination.
    A model that reformats a value (e.g. "$35,000" -> "35000") will be
    flagged here even though it isn't actually inventing anything; the
    cost of a few false positives is much lower than the cost of a
    hallucination silently passing because the check was too lenient.
    """
    if not source_text:
        return list(actual_fields.keys())

    haystack = source_text.lower()
    ungrounded: List[str] = []
    for field_name, value in actual_fields.items():
        if not value or value.lower() not in haystack:
            ungrounded.append(field_name)
    return ungrounded


def check_citation_grounding(
    cited_policy: List[str], retrieved_passages: List[CompliancePassage]
) -> bool:
    """True if every citation the critic gave was actually among the
    passages RETRIEVED for this specific call -- not just a real,
    valid Reg B/FCRA section somewhere in the corpus. A citation can be
    100% real and still be fabricated grounding if it wasn't in what the
    critic was actually given to work with.
    """
    if not cited_policy:
        return True
    retrieved_citations = {p.citation for p in retrieved_passages}
    return all(citation in retrieved_citations for citation in cited_policy)