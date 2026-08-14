
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest

from tests.tier_a_mocked.harness import PersonaTestHarness, PersonaResult
from tests.tier_a_mocked.personas import Persona, load_personas

GENERATED_DIR = Path(__file__).parent.parent / "generated_personas"
PERSONAS_JSON = GENERATED_DIR / "personas.json"
PDFS_DIR = GENERATED_DIR / "pdfs"


@pytest.fixture(scope="module")
def all_personas() -> List[Persona]:
    return load_personas(PERSONAS_JSON)


@pytest.fixture(scope="module")
def harness() -> PersonaTestHarness:
    return PersonaTestHarness(pdfs_dir=PDFS_DIR, escalation_threshold=0.85)


@pytest.mark.asyncio
async def test_all_personas_pass_structurally(harness: PersonaTestHarness, all_personas: List[Persona]):
    results: List[PersonaResult] = await harness.run_batch(all_personas)
    summary = harness.summarize(results)

    failing = [r for r in results if not r.passed]
    leaking = [r for r in results if r.pii_leak_detected]

    if os.getenv("TIER_A_VERBOSE"):
        print(f"\nTier A summary: {summary}")

    assert not leaking, (
        f"SECURITY FAILURE: {len(leaking)} persona(s) leaked raw PII into an "
        f"outbound prompt: {[r.persona_id for r in leaking]}"
    )

    if failing:
        detail = "\n".join(f"- {r.persona_id} ({r.category}): {r.failures}" for r in failing)
        pytest.fail(
            f"{len(failing)}/{summary['total']} personas failed Tier A structural checks:\n{detail}"
        )


@pytest.mark.asyncio
async def test_pii_leak_rate_is_zero(harness: PersonaTestHarness, all_personas: List[Persona]):
    results = await harness.run_batch(all_personas)
    leaks = sum(1 for r in results if r.pii_leak_detected)
    assert leaks == 0, f"PII leak rate: {leaks}/{len(results)} (target: 0)"