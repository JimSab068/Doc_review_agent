"""
FastAPI service wrapping SecureAuditPipeline.

Client construction (Gemini, Chroma, Mongo, Vault) happens ONCE at startup
via the lifespan handler, not per-request -- rebuilding these on every call
would open a new Chroma/Mongo connection and re-init the Gemini SDK client
on every single /review request.
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.audit_log import AuditLogStore, AuditLogWriter
from src.compliance_kb import ComplianceKB
from src.config import settings
from src.critic_agent import CriticAgent
from src.pipeline import SecureAuditPipeline
from src.primary_agent import GeminiClient
from src.schemas import DocumentType
from src.vault import VaultClient

# Fields on SecureAuditPipeline's final_payload that must never leave the
# process boundary. extracted_raw_text carries the full untokenized
# document text -- see Step 4 spec ("audit carefully for anything that
# shouldn't leave the process boundary").
_INTERNAL_ONLY_FIELDS = {"extracted_raw_text"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm_client = GeminiClient(model_name=settings.model_name)

    # Backend selection + the staging/production fail-closed check both
    # live inside VaultClient/_build_store already (vault.py) -- driven by
    # the VAULT_BACKEND and APP_ENV env vars, not by anything passed here.
    vault_client = VaultClient()

    kb_client = ComplianceKB(collection_name=settings.chroma_collection)

    # Wire vault_client into the critic explicitly. SecureAuditPipeline's
    # own default construction (pipeline.py) builds CriticAgent(llm_client)
    # with no vault_client, which silently skips the critic's
    # assert_no_pii_leak check (see critic_agent.py: that check only runs
    # `if self._vault_client is not None`). Passing it in here closes that
    # gap without touching critic_agent.py/pipeline.py.
    critic_agent = CriticAgent(llm_client=llm_client, vault_client=vault_client)

    # Fail loudly at startup if the audit store can't be reached, rather
    # than silently degrading every request to the local fallback log
    # with nobody noticing (see AuditLogWriter's docstring on that mode).
    try:
        audit_store = AuditLogStore(
            db_name=settings.mongo_db,
            collection_name=settings.mongo_collection,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not initialize audit log store: {exc}") from exc

    audit_writer = AuditLogWriter(store=audit_store)

    app.state.pipeline = SecureAuditPipeline(
        llm_client=llm_client,
        vault_client=vault_client,
        kb_client=kb_client,
        critic_agent=critic_agent,
        audit_log_writer=audit_writer,
    )
    app.state.audit_store = audit_store
    app.state.kb_client = kb_client

    yield

    # No explicit teardown required today -- motor/chromadb/boto3 clients
    # in this codebase don't need an explicit close() call.


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/healthz")
async def health_check():
    """Process is running; no dependency calls."""
    return {"status": "ok"}


@app.get("/readyz")
async def readiness_check():
    """Confirms connectivity to Atlas and Chroma (vault connectivity is
    validated at startup by VaultClient() construction -- if that failed,
    the process wouldn't have come up in the first place)."""
    try:
        await app.state.audit_store._client.admin.command("ping")
        app.state.kb_client.client.heartbeat()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}") from None


@app.post("/review")
async def review_document(
    files: List[UploadFile] = File(
        ..., description="One or more pages/documents of a loan/KYC packet, as PDFs"
    ),
    doc_type: DocumentType = Form(DocumentType.LOAN_APPLICATION),
):
    """Accepts a PDF packet, runs it through SecureAuditPipeline, and
    returns only the fields safe to leave the process boundary."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_paths: List[Path] = []
            for f in files:
                if not (f.filename or "").lower().endswith(".pdf"):
                    raise HTTPException(status_code=400, detail=f"'{f.filename}' is not a PDF")
                dest = Path(tmp_dir) / f.filename
                dest.write_bytes(await f.read())
                tmp_paths.append(dest)

            final_payload, _routing = await app.state.pipeline.execute_pdf(tmp_paths, doc_type)

        for key in _INTERNAL_ONLY_FIELDS:
            final_payload.pop(key, None)

        return final_payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None