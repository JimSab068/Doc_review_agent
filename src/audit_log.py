"""
Stage 3.9: Audit log.

README 3.9 calls this "the artifact a regulator or bank compliance
officer would actually want to see" -- this module is what makes that
true in code, not just in the spec. Previously the pipeline did a bare
`await self.mongo_collection.insert_one(...)` inline, which has three
problems this module fixes:

1. No durability guarantee if the Mongo write fails -- the entry was
   simply lost. An audit trail that can silently drop entries is not
   an audit trail.
2. No defense-in-depth check that the entry itself is safe to persist.
   AuditLogEntry.primary_decision / critic_verdict are built from
   pre-detokenization objects (DraftDecision, CriticVerdict), so they
   should never contain raw PII -- but "should never" is an assumption,
   not a guarantee, and this is exactly the kind of assumption that's
   worth checking mechanically rather than trusting silently.
3. No connection-credential hygiene. Mongo URIs commonly carry inline
   credentials (mongodb+srv://user:pass@...). Threading that through a
   constructor argument risks it ending up in a stack trace, a repr, or
   a log line -- the same class of risk secret_redaction.py documents
   for the Gemini API key.

Design follows the pattern already established by GeminiClient and
ComplianceKB elsewhere in this codebase: credentials are read directly
from the environment at the point of use, never accepted as a function
or constructor argument.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.schemas import AuditLogEntry
from src.secret_redaction import safe_exception_message


class AuditLogWriteError(Exception):
    """Raised when an entry could not be durably persisted to the
    primary store. The entry itself is never lost when this is raised --
    see AuditLogWriter.write, which always appends to the local fallback
    log before raising this."""


class AuditLogIntegrityError(Exception):
    """Raised when an audit entry appears to contain raw PII-shaped data.
    This should never fire in normal operation (AuditLogEntry is built
    from pre-detokenization objects upstream) -- if it does fire, treat
    it as a signal that the token boundary was violated somewhere
    upstream, not as a bug in this check."""


# ---------------------------------------------------------------------------
# Defense-in-depth PII shape scan
# ---------------------------------------------------------------------------
# Intentionally narrow, high-precision patterns -- this is a last-line
# backstop on the *serialized audit entry*, not a replacement for the
# vault's tokenization boundary or the PII detector's recall-oriented
# scan (3.2). False positives here just mean an audit write is rejected
# and retried after investigation, which is the safe failure direction.

_PII_SHAPE_PATTERNS: Dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}


def _assert_no_raw_pii_shapes(serialized_entry: str) -> None:
    for name, pattern in _PII_SHAPE_PATTERNS.items():
        if pattern.search(serialized_entry):
            raise AuditLogIntegrityError(
                f"Audit entry appears to contain raw {name}-shaped data; "
                f"refusing to persist. This indicates a token-boundary "
                f"violation upstream, not a fault in this check."
            )


class AuditLogStore:
    """Owns the MongoDB connection used for audit persistence.

    Reads the connection string from the `AUDIT_MONGO_URI` environment
    variable at construction time and nowhere else. Never accepts a URI
    as a constructor argument -- Mongo connection strings frequently
    embed credentials inline, and keeping it out of any parameter list
    is what keeps it out of stack traces, reprs, and log lines from
    calling code.

    Import of `motor` is deferred to `__init__` so any module that only
    needs `AuditLogWriter`'s fallback-log behavior (e.g. unit tests)
    never needs the package installed.
    """

    def __init__(self, db_name: str = "loan_kyc_audit", collection_name: str = "audit_log"):
        if "AUDIT_MONGO_URI" not in os.environ:
            raise KeyError(
                "Missing critical environment variable: 'AUDIT_MONGO_URI'. "
                "Please export or set it in your environment prior to "
                "constructing AuditLogStore."
            )

        from motor.motor_asyncio import AsyncIOMotorClient  # noqa: F401  (deferred import)

        try:
            import certifi
            self._client = AsyncIOMotorClient(
                os.environ["AUDIT_MONGO_URI"],
                tls=True,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=10_000,
            )
            self._collection = self._client[db_name][collection_name]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize audit log store: {safe_exception_message(exc)}"
            ) from None

    @property
    def collection(self):
        """Exposed only for query helpers in this module -- callers
        outside audit_log.py should use AuditLogWriter / AuditLogReader,
        not this property directly, to keep write/read paths consistent
        and auditable in one place."""
        return self._collection


class AuditLogWriter:
    """Public write interface for the audit trail. This is the ONLY
    component that should ever call insert on the audit collection --
    routing all writes through here keeps the integrity check and the
    fallback-durability guarantee unconditional, rather than something
    every caller has to remember to apply.

    Append-only by design: this class intentionally exposes no update
    or delete method. An audit trail that can be edited after the fact
    is not an audit trail.
    """

    def __init__(
        self,
        store: Optional[AuditLogStore] = None,
        fallback_path: Optional[str] = None,
    ):
        """`store` is optional so this can run in local/dev environments
        with no MongoDB configured at all -- entries then go only to the
        fallback log, which is a valid (if degraded) mode, not an error
        state. `fallback_path` defaults to the `AUDIT_LOG_FALLBACK_PATH`
        env var, falling back further to a local file -- never hardcoded
        to a path outside the caller's control.
        """
        self._store = store
        self._fallback_path = fallback_path or os.environ.get(
            "AUDIT_LOG_FALLBACK_PATH", "./audit_log_fallback.jsonl"
        )

    async def write(self, entry: AuditLogEntry) -> None:
        """Persist one audit entry. Always append-only, always durable:

        1. Serialize and integrity-check the entry (raises
        AuditLogIntegrityError and does NOT write anywhere if the
        entry looks like it contains raw PII -- fail closed).
        2. If a primary store is configured, try to write there.
        3. If the primary write fails (or no store is configured),
        append to the local fallback log so the entry is never
        silently dropped, then surface the failure by raising
        AuditLogWriteError -- callers/ops should alert on this, since
        it means the durable audit trail is currently degraded to a
        local file rather than the primary store.
        """
        payload = entry.model_dump(mode="json")
        serialized = json.dumps(payload, default=str)

        # Fail closed: never write anywhere -- not even the fallback --
        # if the entry itself looks compromised.
        _assert_no_raw_pii_shapes(serialized)

        if self._store is None:
            self._append_fallback(serialized)
            return

        try:
            await self._store.collection.insert_one(payload)
        except Exception as exc:
            try:
                self._append_fallback(serialized)
            except AuditLogWriteError as fallback_exc:
                raise AuditLogWriteError(
                    f"Primary audit store write failed AND fallback write failed. "
                    f"Primary error: {safe_exception_message(exc)}. "
                    f"Fallback error: {safe_exception_message(fallback_exc)}"
                ) from exc
            raise AuditLogWriteError(
                f"Primary audit store write failed; entry was preserved "
                f"in the local fallback log ({self._fallback_path}). "
                f"Underlying error: {safe_exception_message(exc)}"
            ) from None

    def _append_fallback(self, serialized_entry: str) -> None:
        # Append-only, one JSON object per line. Opened and closed per
        # call rather than held open, so a crash between writes can't
        # corrupt a previous entry.
        try:
            with open(self._fallback_path, "a", encoding="utf-8") as f:
                f.write(serialized_entry + "\n")
        except OSError as exc:
            raise AuditLogWriteError(
                f"Fallback log write failed ({self._fallback_path}): "
                f"{safe_exception_message(exc)}"
            ) from exc


class AuditLogReader:
    """Read-side interface for the metrics dashboard (3.10) and for
    incident investigation. Separate from AuditLogWriter on purpose --
    callers that only need to read never get a handle that could write.
    """

    def __init__(self, store: AuditLogStore):
        self._store = store

    async def get_by_doc_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return await self._store.collection.find_one({"doc_id": doc_id})

    async def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = self._store.collection.find().sort("timestamps.received", -1).limit(limit)
        return [doc async for doc in cursor]

    async def get_escalated(self, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = (
            self._store.collection.find({"routing_decision.route": "human_queue"})
            .sort("timestamps.received", -1)
            .limit(limit)
        )
        return [doc async for doc in cursor]

    async def compute_escalation_rate(self, since: Optional[datetime] = None) -> float:
        query: Dict[str, Any] = {}
        if since is not None:
            query["timestamps.received"] = {"$gte": since.isoformat()}
        total = await self._store.collection.count_documents(query)
        if total == 0:
            return 0.0
        escalated_query = {**query, "routing_decision.route": "human_queue"}
        escalated = await self._store.collection.count_documents(escalated_query)
        return escalated / total