import asyncio
import hashlib
import inspect
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
# top of file, alongside other imports
from src.primary_agent import PrimaryAgent
from src.ingestion import ingest_pdf
from src.audit_log import AuditLogIntegrityError
from src.schemas import (
    AuditLogEntry,
    Document as PipelineDoc,
    DocumentType,
    RoutingDecision,
    Page
)


class SecureAuditPipeline:
    """
    Main orchestration engine for processing loan/KYC documents safely.
    Handles ingestion, PII tokenization, primary/critic LLM analysis,
    compliance policy checking, escalation routing, detokenization, and auditing.
    """

    def __init__(
        self,
        llm_client: Any,
        vault_client: Any,
        kb_client: Any,
        detector: Optional[Any] = None,
        critic_agent: Optional[Any] = None,
        audit_log_writer: Optional[Any] = None,
        threshold: float = 0.80,
    ):
        self.vault = vault_client
        self.kb = kb_client
        self.threshold = threshold
        self.audit_log_writer = audit_log_writer

        # 1. Primary Agent Setup
        # If a raw LLM client (e.g. GeminiClient) is passed instead of PrimaryAgent,
        # wrap it in a PrimaryAgent using self.vault.
        model_name = getattr(llm_client, "_model_name", "gemini-3.1-flash-lite")
        if not isinstance(model_name, str):
            model_name = "gemini-3.1-flash-lite"
        self.primary_agent = PrimaryAgent(
            llm_client=llm_client,
            vault_client=self.vault,
            model_name=model_name,
        )
                # 2. PII Detector Setup
        if detector is None:
            try:
                from src.pii_detector import RegexPIIDetector
                self.detector = RegexPIIDetector()
            except ImportError:
                from src.pii_detector import PIIDetector
                self.detector = PIIDetector()
        else:
            self.detector = detector

        # 3. Critic Agent Setup
        if critic_agent is None:
            from src.critic_agent import CriticAgent
            self.critic_agent = CriticAgent(llm_client=llm_client)
        else:
            self.critic_agent = critic_agent

    async def _safe_call(self, func_or_coro_func, *args, **kwargs):
        """Executes a function regardless of whether it returns a coroutine or immediate result."""
        res = func_or_coro_func(*args, **kwargs)
        if inspect.iscoroutine(res):
            return await res
        return res

    async def execute_doc(
        self, doc: PipelineDoc
    ) -> Tuple[Dict[str, Any], RoutingDecision]:
        """Executes the pipeline directly from an existing Document object."""
        start_total = time.perf_counter()
        timestamps = {"received": datetime.now(timezone.utc).isoformat()}
        latencies = {}

        raw_content = "\n\n--- DOCUMENT BREAK ---\n\n".join(p.text for p in doc.pages)
        input_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()

        # 1. Secure Vault Tokenization
        t_start = time.perf_counter()
        spans = self.detector.detect(doc)
        pii_types = list(set(
            s.field_type.value if hasattr(s.field_type, "value") else str(s.field_type)
            for s in spans
        ))
        tokenized_doc = self.vault.tokenize_document(doc, spans)
        latencies["tokenization"] = (time.perf_counter() - t_start) * 1000

        # Everything from here on operates on a persisted token map. If any
        # of steps 2-7 raises -- primary agent, KB, critic, audit write,
        # anything -- the finally below still runs, so the map never
        # outlives this request. Don't add steps after this try/finally
        # without moving them inside it.
        try:
            # 2. Primary Agent Execution
            t_start = time.perf_counter()
            primary_decision = await asyncio.to_thread(self.primary_agent.run, tokenized_doc)
            latencies["primary_agent"] = (time.perf_counter() - t_start) * 1000

            # 3. KB Policy Retrieval
            t_start = time.perf_counter()
            policies = await self._safe_call(
                self.kb.query_relevant_policies, primary_decision.reasoning, n_results=2
            )
            latencies["kb_query"] = (time.perf_counter() - t_start) * 1000

            # 4. Critic Agent Evaluation
            t_start = time.perf_counter()
            critic_verdict = await asyncio.to_thread(
            self.critic_agent.evaluate, primary_decision, policies, tokenized_doc.token_map_id
            )
            latencies["critic_agent"] = (time.perf_counter() - t_start) * 1000

            # 5. Safety Escalation Routing
            escalate = (
                critic_verdict.escalate
                or getattr(critic_verdict, "verdict", None) == "flag"
                or primary_decision.confidence < self.threshold
            )

            reason = None
            if escalate:
                reasons = []
                if critic_verdict.escalate:
                    reasons.append("Critic requested escalation")
                if getattr(critic_verdict, "verdict", None) == "flag":
                    reasons.append("Compliance policy flagged warning")
                if primary_decision.confidence < self.threshold:
                    reasons.append(
                        f"Primary confidence ({primary_decision.confidence}) below threshold ({self.threshold})"
                    )
                reason = " | ".join(reasons)

            routing = RoutingDecision(
                route="human_queue" if escalate else "auto_resolve",
                escalation_reason=reason,
                primary_confidence=primary_decision.confidence,
                critic_escalated=critic_verdict.escalate,
            )

            # 6. Late-Bound Detokenization
            t_start = time.perf_counter()
            detokenized_fields = {}
            extracted = getattr(primary_decision, "extracted_fields", {})
            if isinstance(extracted, dict):
                for field, value in extracted.items():
                    if isinstance(value, str):
                        detokenized_fields[field] = self.vault.detokenize_text(
                            tokenized_doc.token_map_id, value
                        )
                    else:
                        detokenized_fields[field] = value
            latencies["detokenization"] = (time.perf_counter() - t_start) * 1000

            timestamps["completed"] = datetime.now(timezone.utc).isoformat()
            latencies["total"] = (time.perf_counter() - start_total) * 1000

            # 7. Audit Logging
            primary_dump = primary_decision.model_dump() if hasattr(primary_decision, "model_dump") else primary_decision.__dict__
            critic_dump = critic_verdict.model_dump() if hasattr(critic_verdict, "model_dump") else critic_verdict.__dict__
            routing_dump = routing.model_dump() if hasattr(routing, "model_dump") else routing.__dict__

            audit_doc = AuditLogEntry(
                doc_id=doc.doc_id,
                input_hash=input_hash,
                pii_types_detected=pii_types,
                primary_decision=primary_dump,
                critic_verdict=critic_dump,
                routing_decision=routing_dump,
                timestamps=timestamps,
                latency_ms=latencies,
            )

            if self.audit_log_writer is not None:
                try:
                    await self._safe_call(self.audit_log_writer.write, audit_doc)
                except AuditLogIntegrityError:
                    # Security invariant, not a durability hiccup -- never
                    # swallow this one. Re-raising surfaces it to whatever's
                    # calling execute_pdf/execute_doc, same fail-closed
                    # posture as the vault's PII leak gate.
                    raise
                except Exception as exc:
                    print(f"[audit_log] WARNING: {exc}")

            final_payload = {
                "doc_id": doc.doc_id,
                "routing": routing.route,
                "escalation_reason": routing.escalation_reason,
                "extracted_fields": detokenized_fields,
                "inconsistencies": getattr(primary_decision, "inconsistencies", []),
                "missing_compliance_items": getattr(primary_decision, "missing_compliance_items", []),
                "extracted_raw_text": raw_content,
            }

            return final_payload, routing
        finally:
            self.vault.discard(tokenized_doc.token_map_id)

    async def execute_pdf(
        self,
        pdf_paths: Union[str, Path, List[Union[str, Path]]],
        doc_type: DocumentType,
    ) -> Tuple[Dict[str, Any], RoutingDecision]:
        """
        Executes the full pipeline starting from one or more PDF files on disk.
        Combines multi-document PDF packets into a unified tokenized Document object.
        """
        if isinstance(pdf_paths, (str, Path)):
            paths = [Path(pdf_paths)]
        else:
            paths = [Path(p) for p in pdf_paths]

        start_total = time.perf_counter()
        timestamps = {"received": datetime.now(timezone.utc).isoformat()}
        latencies = {}

        # 1. Read all PDF files, combine raw bytes for hashing, and ingest pages
        t_start = time.perf_counter()
        combined_bytes = bytearray()
        all_pages = []
        doc_id = None

        for p in paths:
            with open(p, "rb") as f:
                content = f.read()
                combined_bytes.extend(content)

            ingested_doc = await asyncio.to_thread(ingest_pdf, pdf_path=str(p), doc_type=doc_type)
            if doc_id is None:
                doc_id = ingested_doc.doc_id
            for pg in ingested_doc.pages:
                all_pages.append(Page(page_number=len(all_pages) + 1, text=pg.text))
        input_hash = hashlib.sha256(combined_bytes).hexdigest()
        combined_filenames = ", ".join(p.name for p in paths)

        doc = PipelineDoc(
            doc_id=doc_id or "multi_doc",
            doc_type=doc_type,
            source_filename=combined_filenames,
            pages=all_pages,
        )

        latencies["ingestion"] = (time.perf_counter() - t_start) * 1000
        raw_content = "\n\n--- DOCUMENT BREAK ---\n\n".join(p.text for p in doc.pages)

        # 2. Secure Vault Tokenization
        t_start = time.perf_counter()
        spans = self.detector.detect(doc)
        pii_types = list(set(
            s.field_type.value if hasattr(s.field_type, "value") else str(s.field_type)
            for s in spans
        ))
        tokenized_doc = self.vault.tokenize_document(doc, spans)
        latencies["tokenization"] = (time.perf_counter() - t_start) * 1000

        # Same rule as execute_doc: everything downstream of tokenization
        # goes inside this try, and discard() runs in finally regardless of
        # where it fails. This was previously only wrapping the final
        # `return` -- a failure in primary_agent/kb/critic/audit above that
        # point left the token map alive in DynamoDB until TTL expiry.
        try:
            # 3. Primary Agent Execution
            t_start = time.perf_counter()
            primary_decision = await self._safe_call(self.primary_agent.run, tokenized_doc)
            latencies["primary_agent"] = (time.perf_counter() - t_start) * 1000

            # 4. KB Policy Retrieval
            t_start = time.perf_counter()
            policies = await asyncio.to_thread(self.kb.query_relevant_policies, primary_decision.reasoning, n_results=2)
            latencies["kb_query"] = (time.perf_counter() - t_start) * 1000

            # 5. Critic Agent Evaluation
            t_start = time.perf_counter()
            critic_verdict = await self._safe_call(
                self.critic_agent.evaluate,
                primary_decision,
                policies,
                tokenized_doc.token_map_id,
            )
            latencies["critic_agent"] = (time.perf_counter() - t_start) * 1000

            # 6. Safety Escalation Routing
            escalate = (
                critic_verdict.escalate
                or getattr(critic_verdict, "verdict", None) == "flag"
                or primary_decision.confidence < self.threshold
            )

            reason = None
            if escalate:
                reasons = []
                if critic_verdict.escalate:
                    reasons.append("Critic requested escalation")
                if getattr(critic_verdict, "verdict", None) == "flag":
                    reasons.append("Compliance policy flagged warning")
                if primary_decision.confidence < self.threshold:
                    reasons.append(
                        f"Primary confidence ({primary_decision.confidence}) below threshold ({self.threshold})"
                    )
                reason = " | ".join(reasons)

            routing = RoutingDecision(
                route="human_queue" if escalate else "auto_resolve",
                escalation_reason=reason,
                primary_confidence=primary_decision.confidence,
                critic_escalated=critic_verdict.escalate,
            )

            # 7. Late-Bound Detokenization
            t_start = time.perf_counter()
            detokenized_fields = {}
            extracted = getattr(primary_decision, "extracted_fields", {})
            if isinstance(extracted, dict):
                for field, value in extracted.items():
                    if isinstance(value, str):
                        detokenized_fields[field] = self.vault.detokenize_text(
                            tokenized_doc.token_map_id, value
                        )
                    else:
                        detokenized_fields[field] = value
            latencies["detokenization"] = (time.perf_counter() - t_start) * 1000

            timestamps["completed"] = datetime.now(timezone.utc).isoformat()
            latencies["total"] = (time.perf_counter() - start_total) * 1000

            # 8. Audit Logging
            primary_dump = primary_decision.model_dump() if hasattr(primary_decision, "model_dump") else primary_decision.__dict__
            critic_dump = critic_verdict.model_dump() if hasattr(critic_verdict, "model_dump") else critic_verdict.__dict__
            routing_dump = routing.model_dump() if hasattr(routing, "model_dump") else routing.__dict__

            audit_doc = AuditLogEntry(
                doc_id=doc.doc_id,
                input_hash=input_hash,
                pii_types_detected=pii_types,
                primary_decision=primary_dump,
                critic_verdict=critic_dump,
                routing_decision=routing_dump,
                timestamps=timestamps,
                latency_ms=latencies,
            )

            if self.audit_log_writer is not None:
                try:
                    await self._safe_call(self.audit_log_writer.write, audit_doc)
                except AuditLogIntegrityError:
                    # Same fail-closed posture as execute_doc -- an audit
                    # integrity failure is a security invariant violation,
                    # not a durability hiccup, so it must propagate rather
                    # than be logged and swallowed.
                    raise
                except Exception as exc:
                    print(f"[audit_log] WARNING: {exc}")

            final_payload = {
                "doc_id": doc.doc_id,
                "routing": routing.route,
                "escalation_reason": routing.escalation_reason,
                "extracted_fields": detokenized_fields,
                "inconsistencies": getattr(primary_decision, "inconsistencies", []),
                "missing_compliance_items": getattr(primary_decision, "missing_compliance_items", []),
                "extracted_raw_text": raw_content,
            }

            return final_payload, routing
        finally:
            self.vault.discard(tokenized_doc.token_map_id)