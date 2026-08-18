"""
Factories for the real (non-mocked) clients Tier B swaps in for Tier A's
fakes.

Wired against your actual src/compliance_kb.py and src/audit_log.py.

One non-obvious thing this file handles: Chroma's
GoogleGeminiEmbeddingFunction calls the Gemini embeddings endpoint
directly, on the same API key/quota as GeminiClient.generate(). A rate
limiter that only wraps .generate() has a blind spot -- every KB query
during a Tier B run (one per persona, inside the critic-agent step)
makes an untracked Gemini call. This file shares ONE RateLimiter
instance between the LLM client and the embedding function so both
count against the same 10-RPM budget.
"""

from __future__ import annotations

import os

from src.vault import VaultClient
from src.primary_agent import GeminiClient
from src.compliance_kb import ComplianceKB
from src.audit_log import AuditLogStore, AuditLogWriter, AuditLogReader

from .rate_limiter import RateLimitedLLMClient, RateLimiter


# ---------------------------------------------------------------------------
# Shared rate budget
# ---------------------------------------------------------------------------

def build_shared_rate_limiter(max_calls_per_minute: int = 10) -> RateLimiter:
    """One limiter, shared across generate() calls AND embedding calls,
    since both draw from the same Gemini API key's quota. Call this once
    per test session and pass the same instance into both
    build_live_llm_client() and build_live_kb_client()."""
    return RateLimiter(max_calls=max_calls_per_minute, period_seconds=60.0)


class _RateLimitedEmbeddingFunction:
    """Wraps a Chroma embedding function so each __call__ (one Chroma
    query or upsert, which may embed a batch of texts in a single
    underlying API call) counts as one slot against the shared limiter.

    Approximation: if Chroma's embedding function ever splits a large
    batch into multiple underlying API calls internally, this only
    accounts for one. Not a real concern for Tier B's one-persona-at-a-
    time usage, but worth knowing if this wrapper is ever reused for
    bulk corpus seeding.
    """

    def __init__(self, inner_embedding_function, rate_limiter: RateLimiter):
        self._inner = inner_embedding_function
        self._limiter = rate_limiter

    def __call__(self, input):  # noqa: A002 -- name required by Chroma's EmbeddingFunction protocol
        self._limiter.acquire()
        return self._inner(input)

    def __getattr__(self, name):
        # Chroma sometimes introspects embedding functions (e.g. for
        # name/config serialization on collection creation) -- forward
        # anything we don't explicitly implement to the wrapped instance.
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Gemini (real LLM)
# ---------------------------------------------------------------------------

def build_live_llm_client(
    rate_limiter: RateLimiter,
    model_name: str = "gemini-3.1-flash-lite",
) -> RateLimitedLLMClient:
    """Real GeminiClient wrapped with rate limiting + backoff retry.

    Requires GOOGLE_API_KEY or GEMINI_API_KEY in the environment --
    GeminiClient itself raises a clear KeyError if neither is set.
    """
    inner = GeminiClient(model_name=model_name)
    return RateLimitedLLMClient(inner, rate_limiter=rate_limiter)


# ---------------------------------------------------------------------------
# Vault (no adaptation needed -- same real VaultClient as Tier A/production)
# ---------------------------------------------------------------------------

def build_live_vault_client() -> VaultClient:
    return VaultClient()


# ---------------------------------------------------------------------------
# ChromaDB compliance KB
# ---------------------------------------------------------------------------
#
# Read-only against your real dev/prod compliance collection. Requires
# CHROMA_HOST to be set so this hits a real deployed Chroma instance
# rather than the ephemeral in-process default -- see compliance_kb.py's
# docstring for the full env var list (CHROMA_HOST, CHROMA_PORT,
# CHROMA_SSL).

def build_live_kb_client(rate_limiter: RateLimiter, collection_name: str = "compliance_rules") -> ComplianceKB:
    if "CHROMA_HOST" not in os.environ:
        raise RuntimeError(
            "CHROMA_HOST is not set -- Tier B is meant to test against a "
            "real deployed Chroma instance, not the ephemeral in-process "
            "default. Set CHROMA_HOST (and CHROMA_PORT / CHROMA_SSL if "
            "not using defaults) before running, or use Tier A if you "
            "specifically want the in-memory behavior."
        )

    kb = ComplianceKB(collection_name=collection_name)
    rate_limited_ef = _RateLimitedEmbeddingFunction(kb.embedding_function, rate_limiter)
    kb.embedding_function = rate_limited_ef
    # Chroma resolves the embedding function per-collection-handle, so
    # swapping the attribute alone isn't enough -- get_or_create_collection
    # must be called again with the wrapped one attached.
    kb.collection = kb.client.get_or_create_collection(
        name=collection_name,
        embedding_function=rate_limited_ef,
    )
    return kb


# ---------------------------------------------------------------------------
# MongoDB Atlas audit log writer
# ---------------------------------------------------------------------------

def build_live_audit_log_store(
    db_name: str = "tier_b_test",
    collection_name: str = "audit_log",
) -> AuditLogStore:
    """AuditLogStore reads its connection string from AUDIT_MONGO_URI
    directly (never as a constructor arg, by design -- see audit_log.py)
    so there's nothing to pass through here except db/collection names.

    Built once and shared between the writer and reader below, so the
    reader is guaranteed to be looking at the same collection the
    writer just wrote to."""
    if "AUDIT_MONGO_URI" not in os.environ:
        raise RuntimeError(
            "AUDIT_MONGO_URI is not set. Export your Atlas connection "
            "string before running Tier B, e.g. (PowerShell):\n"
            '  $env:AUDIT_MONGO_URI = "mongodb+srv://user:pass@your-cluster.mongodb.net/?retryWrites=true&w=majority"'
        )
    return AuditLogStore(db_name=db_name, collection_name=collection_name)


def build_live_audit_log_writer(store: AuditLogStore) -> AuditLogWriter:
    """Real audit log writer, pointed at a dedicated tier_b_test database
    -- never the same database/collection your production audit trail
    writes to."""
    return AuditLogWriter(store=store)


def build_live_audit_log_reader(store: AuditLogStore) -> AuditLogReader:
    """Used by the Tier B harness to verify a write actually landed in
    Mongo, rather than trusting that no exception means success -- see
    the comment in harness.py's run_persona for why that distinction
    matters given pipeline.py's current swallow-and-warn behavior."""
    return AuditLogReader(store)


def teardown_live_audit_log(db_name: str = "tier_b_test", collection_name: str = "audit_log") -> None:
    """Drops the tier_b_test audit log collection after a run.

    Deliberately uses plain (synchronous) pymongo here rather than
    motor -- AuditLogStore uses motor because AuditLogWriter/Reader are
    async in the main app, but teardown is a one-shot fire-and-forget
    cleanup step with no async context to fit into, so a sync client is
    simpler and avoids spinning up an event loop just to drop a
    collection.
    """
    uri = os.environ.get("AUDIT_MONGO_URI")
    if not uri:
        return
    try:
        from pymongo import MongoClient
        import certifi

        client = MongoClient(uri, tlsCAFile=certifi.where())
        client[db_name][collection_name].drop()
        client.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[tier_b teardown] WARNING: failed to drop Mongo test collection: {exc}")