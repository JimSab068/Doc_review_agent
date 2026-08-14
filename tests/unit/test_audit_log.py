from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.audit_log import (
    AuditLogIntegrityError,
    AuditLogWriteError,
    AuditLogWriter,
)
from src.schemas import AuditLogEntry


def _sample_entry(**overrides) -> AuditLogEntry:
    defaults = dict(
        doc_id="doc-123",
        input_hash="abc123",
        pii_types_detected=["ssn", "email"],
        primary_decision={"extracted_fields": {"employer": "Acme"}, "confidence": 0.9},
        critic_verdict={"verdict": "pass", "cited_policy": [], "concerns": [], "escalate": False},
        routing_decision={"route": "auto_resolve", "escalation_reason": None,
                           "primary_confidence": 0.9, "critic_escalated": False},
        timestamps={"received": "2026-07-25T00:00:00", "completed": "2026-07-25T00:00:01"},
        latency_ms={"total": 12.3},
    )
    defaults.update(overrides)
    return AuditLogEntry(**defaults)


class FakeFailingCollection:
    async def insert_one(self, doc):
        raise ConnectionError("simulated Mongo outage")


class FakeFailingStore:
    """Duck-types AuditLogStore's public surface without a real Mongo connection."""
    def __init__(self):
        self.collection = FakeFailingCollection()


@pytest.mark.asyncio
async def test_no_store_configured_writes_only_to_fallback(tmp_path):
    fallback_path = tmp_path / "fallback.jsonl"
    writer = AuditLogWriter(store=None, fallback_path=str(fallback_path))

    await writer.write(_sample_entry())

    assert fallback_path.exists()
    lines = fallback_path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["doc_id"] == "doc-123"


@pytest.mark.asyncio
async def test_store_failure_falls_back_and_raises(tmp_path):
    fallback_path = tmp_path / "fallback.jsonl"
    writer = AuditLogWriter(store=FakeFailingStore(), fallback_path=str(fallback_path))

    with pytest.raises(AuditLogWriteError):
        await writer.write(_sample_entry())

    # The entry must not be lost even though the primary store failed.
    assert fallback_path.exists()
    lines = fallback_path.read_text().strip().splitlines()
    assert len(lines) == 1


@pytest.mark.asyncio
async def test_raw_pii_shape_blocks_write_entirely(tmp_path):
    fallback_path = tmp_path / "fallback.jsonl"
    writer = AuditLogWriter(store=None, fallback_path=str(fallback_path))

    # Simulates a token-boundary violation upstream: a raw SSN-shaped
    # string ending up in a field that should only ever hold tokens.
    tainted = _sample_entry(
        primary_decision={"extracted_fields": {"ssn": "489-09-1234"}, "confidence": 0.9}
    )

    with pytest.raises(AuditLogIntegrityError):
        await writer.write(tainted)

    # Fail CLOSED: not even the fallback log should receive this entry.
    assert not fallback_path.exists()

class FakeWorkingCollection:
    def __init__(self):
        self.inserted_docs = []

    async def insert_one(self, doc):
        self.inserted_docs.append(doc)
        return True


class FakeWorkingStore:
    def __init__(self):
        self.collection = FakeWorkingCollection()


@pytest.mark.asyncio
async def test_successful_primary_store_write(tmp_path):
    """Verifies that a healthy store receives the inserted document."""
    fallback_path = tmp_path / "fallback.jsonl"
    fake_store = FakeWorkingStore()
    writer = AuditLogWriter(store=fake_store, fallback_path=str(fallback_path))

    entry = _sample_entry()
    await writer.write(entry)

    assert len(fake_store.collection.inserted_docs) == 1
    assert fake_store.collection.inserted_docs[0]["doc_id"] == "doc-123"


@pytest.mark.asyncio
async def test_raw_email_leak_blocks_write_entirely(tmp_path):
    """Verifies that raw email addresses trigger AuditLogIntegrityError."""
    fallback_path = tmp_path / "fallback.jsonl"
    writer = AuditLogWriter(store=None, fallback_path=str(fallback_path))

    tainted_email = _sample_entry(
        critic_verdict={
            "verdict": "pass",
            "cited_policy": [],
            "concerns": ["Leak: john.doe@example.com"],
            "escalate": False,
        }
    )

    with pytest.raises(AuditLogIntegrityError):
        await writer.write(tainted_email)

    assert not fallback_path.exists()


@pytest.mark.asyncio
async def test_fallback_appends_multiple_entries_sequentially(tmp_path):
    """Verifies that fallback logging appends lines without overwriting prior records."""
    fallback_path = tmp_path / "fallback.jsonl"
    writer = AuditLogWriter(store=None, fallback_path=str(fallback_path))

    await writer.write(_sample_entry(doc_id="doc-1"))
    await writer.write(_sample_entry(doc_id="doc-2"))

    lines = fallback_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["doc_id"] == "doc-1"
    assert json.loads(lines[1])["doc_id"] == "doc-2"


@pytest.mark.asyncio
async def test_double_fault_raises_audit_log_write_error(tmp_path):
    """Verifies proper exception handling when both primary store AND fallback file write fail."""
    # Invalid path where directory doesn't exist to force OS write error
    invalid_fallback = tmp_path / "non_existent_dir" / "fallback.jsonl"
    writer = AuditLogWriter(store=FakeFailingStore(), fallback_path=str(invalid_fallback))

    with pytest.raises(AuditLogWriteError):
        await writer.write(_sample_entry())