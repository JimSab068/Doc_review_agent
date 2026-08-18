

"""
Tier B fixtures.

Two safety guards worth calling out explicitly:

1. Tier B is SKIPPED BY DEFAULT. A bare `pytest` run, or even
   `pytest tests/tier_b_live`, will not spend API quota or touch Mongo/
   Chroma unless you explicitly opt in with `TIER_B_LIVE=1`.

2. Every real client here is pointed at a dedicated *_test database/
   collection where the test genuinely writes (audit log), with
   teardown after the session. The compliance KB is the one exception,
   used read-only against your real collection -- see the comment in
   live_clients.py's build_live_kb_client() for why.

Required environment variables (see README.md for the full list and
example PowerShell commands):
  GEMINI_API_KEY or GOOGLE_API_KEY   -- LLM + embeddings
  AUDIT_MONGO_URI                    -- MongoDB Atlas connection string
  CHROMA_HOST (+ CHROMA_PORT/SSL)    -- deployed Chroma instance
  TIER_B_LIVE=1                      -- opt-in flag, required to un-skip
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from .harness import TierBLiveHarness
from .tier_b_personas import load_tier_b_personas

from .live_clients import (
    build_live_audit_log_reader,
    build_live_audit_log_store,
    build_live_audit_log_writer,
    build_live_kb_client,
    build_live_llm_client,
    build_live_vault_client,
    build_shared_rate_limiter,
    teardown_live_audit_log,
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "tier_b: live-integration test hitting real Gemini/Chroma/Mongo (costs quota, opt-in only)"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip every tier_b-marked test unless TIER_B_LIVE=1 is set."""
    if os.environ.get("TIER_B_LIVE") == "1":
        return
    skip_marker = pytest.mark.skip(
        reason="Tier B live tests are opt-in. Set TIER_B_LIVE=1 to run them (see README)."
    )
    for item in items:
        if "tier_b" in item.keywords:
            item.add_marker(skip_marker)


PERSONAS_JSON_PATH = os.environ.get("TIER_B_PERSONAS_JSON", "tests/generated_personas/personas.json")

# conftest.py — add this fixture, alongside the others

@pytest.fixture(scope="session")
def tier_b_personas():
    return load_tier_b_personas(PERSONAS_JSON_PATH)

@pytest.fixture(scope="session")
def harness(pdfs_dir):
    return TierBLiveHarness(pdfs_dir=pdfs_dir)


@pytest.fixture(scope="session")
def pdfs_dir() -> str:
    # ADAPT: point this at wherever Tier B's persona PDFs live on disk.
    return os.environ.get("TIER_B_PDFS_DIR", "tests/generated_personas/pdfs")


@pytest.fixture(scope="session")
def shared_rate_limiter():
    """One limiter shared by the LLM client AND the KB's embedding
    function -- both draw against the same Gemini API key's quota, so
    they need to draw against the same budget, not two independent ones
    that could each individually stay under 10/min while their sum
    blows past it."""
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
    # Must be an async fixture, not a plain sync one: AsyncIOMotorClient
    # binds to whatever event loop is "current" at construction time.
    # A sync fixture runs during pytest's setup phase, outside any
    # asyncio context, so it grabs a throwaway default loop instead of
    # the session-scoped loop our tier_b tests actually run under --
    # that mismatched loop gets closed after the first test, which is
    # exactly the "Event loop is closed" failure this fixes.
    db_name = os.environ.get("TIER_B_MONGO_DB", "tier_b_test")
    collection_name = os.environ.get("TIER_B_MONGO_COLLECTION", "audit_log")
    store = build_live_audit_log_store(db_name=db_name, collection_name=collection_name)
    yield store
    teardown_live_audit_log(db_name=db_name, collection_name=collection_name)


@pytest.fixture(scope="session")
def live_audit_log_writer(live_audit_log_store):
    return build_live_audit_log_writer(live_audit_log_store)


@pytest.fixture(scope="session")
def live_audit_log_reader(live_audit_log_store):
    return build_live_audit_log_reader(live_audit_log_store)