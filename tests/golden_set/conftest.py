"""
tests/golden_set/conftest.py

Wires the same real (live) clients tier_b_live uses, via the same
factories in tests/tier_b_live/live_clients.py -- deliberately NOT
importing tier_b_live's fixtures directly (pytest doesn't auto-share
fixtures across sibling conftest.py files without root-level plugin
registration), so this file re-declares thin fixture wrappers around
the same underlying builders instead of duplicating any client logic.

Writes to a SEPARATE Mongo database (golden_set_audit by default, not
tier_b_test) so golden set runs don't mix into -- or get dropped by --
tier_b_live's per-session teardown. Deliberately NOT torn down after
each run: unlike tier_b_test's throwaway data, a golden set's audit
trail is useful build-over-build history, so entries accumulate unless
you clean them up manually.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from tests.tier_b_live.harness import TierBLiveHarness
from tests.tier_b_live.live_clients import (
    build_live_audit_log_reader,
    build_live_audit_log_store,
    build_live_audit_log_writer,
    build_live_kb_client,
    build_live_llm_client,
    build_live_vault_client,
    build_shared_rate_limiter,
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "golden: frozen golden-set regression gate (live, opt-in via TIER_B_LIVE=1)"
    )


@pytest.fixture(scope="session")
def pdfs_dir() -> str:
    return os.environ.get("TIER_B_PDFS_DIR", "tests/generated_personas/pdfs")


@pytest.fixture(scope="session")
def harness(pdfs_dir):
    threshold = float(os.environ.get("TIER_B_ESCALATION_THRESHOLD", "0.85"))
    return TierBLiveHarness(pdfs_dir=pdfs_dir, escalation_threshold=threshold)


@pytest.fixture(scope="session")
def shared_rate_limiter():
    max_rpm = int(os.environ.get("TIER_B_MAX_CALLS_PER_MINUTE", "6"))
    return build_shared_rate_limiter(max_calls_per_minute=max_rpm)


@pytest.fixture(scope="session")
def live_llm_client(shared_rate_limiter):
    return build_live_llm_client(rate_limiter=shared_rate_limiter)


@pytest.fixture(scope="session")
def live_vault_client():
    return build_live_vault_client()


@pytest.fixture(scope="session")
def live_kb_client(shared_rate_limiter):
    collection_name = os.environ.get("TIER_B_CHROMA_COLLECTION", "compliance_rules")
    return build_live_kb_client(rate_limiter=shared_rate_limiter, collection_name=collection_name)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def live_audit_log_store():
    # See conftest.py's docstring on loop binding in tier_b_live -- same
    # reasoning applies here: must be async-fixture-constructed so the
    # underlying AsyncIOMotorClient binds to the session-scoped loop.
    db_name = os.environ.get("GOLDEN_SET_MONGO_DB", "golden_set_audit")
    collection_name = os.environ.get("GOLDEN_SET_MONGO_COLLECTION", "audit_log")
    store = build_live_audit_log_store(db_name=db_name, collection_name=collection_name)
    yield store
    # Deliberately no teardown here -- see module docstring.


@pytest.fixture(scope="session")
def live_audit_log_writer(live_audit_log_store):
    return build_live_audit_log_writer(live_audit_log_store)


@pytest.fixture(scope="session")
def live_audit_log_reader(live_audit_log_store):
    return build_live_audit_log_reader(live_audit_log_store)