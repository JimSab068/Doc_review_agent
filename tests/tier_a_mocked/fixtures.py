"""
Scripted LLM responses for Tier A.

Two layers:

1. SCRIPTED_RESPONSES -- hand-authored, per-persona-id scripts for a few
   illustrative categories. Kept as documentation of the pattern and as
   a place to hand-craft edge cases (e.g. malformed-JSON variants) that
   don't fit a generic template.

2. build_scripted_responses() -- a generic fallback that derives a
   scripted primary/critic response pair directly from any persona's
   ground_truth, with no hand-authoring required. This is what lets
   Tier A actually run against the full persona set (previously it
   could only run against the 3 personas hand-scripted below and raised
   KeyError on everything else).

Deliberately kept separate from the Persona/GroundTruth model: a real
generated persona (spec 4.1) never has a "scripted response" -- that's
a Tier A testing artifact, not part of the persona data model shared
with Tier B and the future generator.
"""

from __future__ import annotations

import json
from typing import Tuple

from tests.tier_a_mocked.personas import Persona
from tests.tier_a_mocked.scripted_llm_client import ScriptedResponse, extract_tokens_by_type


def _primary_json(extracted_fields, inconsistencies, missing_items, confidence, reasoning) -> str:
    return json.dumps(
        {
            "extracted_fields": extracted_fields,
            "inconsistencies": inconsistencies,
            "missing_compliance_items": missing_items,
            "confidence": confidence,
            "reasoning": reasoning,
        }
    )


def _critic_json(verdict, cited_policy, concerns, escalate) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "cited_policy": cited_policy,
            "concerns": concerns,
            "escalate": escalate,
        }
    )


def _clean_approve_primary(prompt: str) -> str:
    tokens = extract_tokens_by_type(prompt)
    return _primary_json(
        extracted_fields={
            "employer": "Acme Robotics Inc.",
            "requested_amount": "$35,000",
            "stated_purpose": "Debt consolidation",
            "ssn": tokens["SSN"],
            "date_of_birth": tokens["DATE_OF_BIRTH"],
            "email": tokens["EMAIL"],
        },
        inconsistencies=[],
        missing_items=[],
        confidence=0.94,
        reasoning="All fields present and consistent across the single application document.",
    )


def _cross_doc_primary(prompt: str) -> str:
    return _primary_json(
        extracted_fields={
            "stated_monthly_income": "$9,500",
            "bank_avg_monthly_deposits": "$4,200",
        },
        inconsistencies=[
            "Stated monthly income ($9,500) is more than double the average "
            "monthly bank deposits ($4,200) shown on the attached statement."
        ],
        missing_items=[],
        confidence=0.55,
        reasoning="Stated income is not corroborated by bank deposit history.",
    )


def _direct_injection_primary(prompt: str) -> str:
    # The scripted response deliberately does NOT follow the embedded
    # injection instruction -- Tier A cannot test whether a real model
    # would resist it (that's Tier B's job); this response only proves
    # the pipeline mechanics (prompt construction, parsing, routing)
    # behave correctly when the document text contains an injection
    # attempt, regardless of how a real model would react to it.
    return _primary_json(
        extracted_fields={
            "stated_monthly_income": "$2,000",
            "requested_amount": "$250,000",
        },
        inconsistencies=[
            "Requested amount ($250,000) is inconsistent with stated monthly "
            "income ($2,000)."
        ],
        missing_items=[],
        confidence=0.40,
        reasoning="Income-to-loan-amount ratio is far outside normal bounds; "
        "document also contains an embedded instruction attempting to "
        "override review -- flagged, not followed.",
    )


SCRIPTED_RESPONSES: dict[str, Tuple[ScriptedResponse, ScriptedResponse]] = {
    "ex_clean_approve_01": (
        _clean_approve_primary,
        _critic_json("pass", ["12 CFR § 1002.9(a)(1)"], [], False),
    ),
    "ex_cross_doc_inconsistency_01": (
        _cross_doc_primary,
        _critic_json(
            "flag",
            ["12 CFR § 1002.9(b)(2)"],
            ["Income claim is not supported by bank statement evidence; "
             "specific reason for any adverse action must be substantiated."],
            True,
        ),
    ),
    "ex_direct_injection_01": (
        _direct_injection_primary,
        _critic_json(
            "flag",
            ["12 CFR § 1002.9(b)(2)"],
            ["Requested loan amount is inconsistent with stated income.",
             "Document contains an embedded instruction attempting to "
             "bypass review; escalate for manual inspection."],
            True,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Generic fallback: derive a scripted response pair from ground_truth alone
# ---------------------------------------------------------------------------

# Maps a ground_truth.extracted_fields key (lowercased) to the PII type
# name the vault would have tokenized it under, if it's PII at all. Any
# field NOT in this map is treated as non-PII and echoed back verbatim.
_PII_FIELD_TO_TOKEN_TYPE = {
    "ssn": "SSN",
    "date_of_birth": "DATE_OF_BIRTH",
    "dob": "DATE_OF_BIRTH",
    "email": "EMAIL",
    "phone": "PHONE",
    "account_number": "ACCOUNT_NUMBER",
    "applicant_name": "PERSON_NAME",
    "account_holder": "PERSON_NAME",
    "address": "ADDRESS",
    "mailing_address": "ADDRESS",
}


def build_scripted_responses(persona: Persona) -> Tuple[ScriptedResponse, ScriptedResponse]:
    def primary(prompt: str) -> str:
        tokens_by_type = extract_tokens_by_type(prompt)  # {TYPE: token}
        available_tokens = list(tokens_by_type.values())
        token_iter = iter(available_tokens)

        fields: dict[str, str] = {}
        for field_name, value in persona.ground_truth.extracted_fields.items():
            if value == "[[TOKEN]]":
                # Tier A doesn't test which PII type maps to which field
                # (that's extraction-quality, i.e. Tier B's job) -- it only
                # needs a real, valid token here so detokenization has
                # something genuine to resolve later.
                token = next(token_iter, None)
                fields[field_name] = token or (available_tokens[-1] if available_tokens else "[[NO_TOKEN_FOUND]]")
            else:
                fields[field_name] = str(value) if value is not None else ""

        confidence = 0.55 if persona.ground_truth.expected_escalate else 0.94
        return _primary_json(
            extracted_fields=fields,
            inconsistencies=persona.ground_truth.expected_inconsistencies,
            missing_items=persona.ground_truth.expected_missing_compliance_items,
            confidence=confidence,
            reasoning=f"[Tier A scripted-from-ground-truth] category={persona.category.value}",
        )

    def critic(prompt: str) -> str:
        verdict = persona.ground_truth.expected_verdict
        concerns = [] if verdict == "pass" else [f"Scripted concern for category={persona.category.value}"]
        cited = ["12 CFR § 1002.9(a)(1)"] if verdict == "pass" else ["12 CFR § 1002.9(b)(2)"]
        return _critic_json(verdict=verdict, cited_policy=cited, concerns=concerns, escalate=persona.ground_truth.expected_escalate)

    return primary, critic


def get_scripted_responses(persona: Persona) -> Tuple[ScriptedResponse, ScriptedResponse]:
    """Look up a hand-authored script for this persona_id if one exists;
    otherwise fall back to the generic ground-truth-derived builder.
    This is the function the harness should call -- it's what makes
    Tier A runnable against the full persona set instead of only the 3
    examples in SCRIPTED_RESPONSES."""
    if persona.persona_id in SCRIPTED_RESPONSES:
        return SCRIPTED_RESPONSES[persona.persona_id]
    return build_scripted_responses(persona)


def primary(prompt: str) -> str:
    tokens = extract_tokens_by_type(prompt)
    fields: dict[str, str] = {}
    for field_name, value in persona.ground_truth.extracted_fields.items():
        if value == "[[TOKEN]]":
            token_type = _PII_FIELD_TO_TOKEN_TYPE.get(field_name.lower())
            if token_type and token_type in tokens:
                fields[field_name] = tokens[token_type]
            else:
                # No known mapping or the vault didn't tokenize this type
                # for this doc -- surfacing as-is makes a missing-mapping
                # bug visible instead of silently echoing the sentinel.
                fields[field_name] = tokens.get(token_type, "[[UNMAPPED_TOKEN]]")
        else:
            fields[field_name] = str(value) if value is not None else ""
    ...