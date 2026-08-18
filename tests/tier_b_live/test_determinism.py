# tests/tier_b_live/test_determinism.py
from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.mark.tier_b
async def test_determinism_p0075(
    harness,
    tier_b_personas,
    live_llm_client,
    live_vault_client,
    live_kb_client,
    live_audit_log_writer,
    live_audit_log_reader,
):
    """p_0075 (ambiguous_escalate) swung 0.98 -> 0.75 -> 0.98 across
    manual reruns in this session. This confirms whether that's real
    run-to-run variance (a calibration finding to report per spec
    4.3) or a fluke -- 5 reruns, sequential (rate limiter requires it).
    """
    persona = next(p for p in tier_b_personas if p.persona_id == "p_0075")
    runs = 5
    results = []

    for i in range(runs):
        result = await harness.run_persona(
            persona,
            live_llm_client,
            live_vault_client,
            live_kb_client,
            live_audit_log_writer,
            live_audit_log_reader,
        )
        results.append(result)
        print(f"[run {i+1}/{runs}] route={result.routing} confidence={result.confidence}")

    routes = {r.routing for r in results}
    confidences = [r.confidence for r in results if r.confidence is not None]

    print(f"\nRoutes across {runs} runs: {routes}")
    print(f"Confidences: {confidences}")

    # Don't assert away real variance -- report it. If this fails, that
    # IS the finding: document it in eval_report.md per spec 4.6 rather
    # than tuning the threshold until it happens to pass.
    assert len(routes) == 1, (
        f"Determinism check: routing varied across {runs} identical runs: "
        f"{routes} (confidences: {confidences})"
    )