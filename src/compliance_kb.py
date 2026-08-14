
import os
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.secret_redaction import safe_exception_message


class CompliancePassage(BaseModel):
    id: str
    content: str
    citation: str
    metadata: Dict[str, Any]


class ComplianceKB:

    def __init__(
        self,
        collection_name: str = "compliance_rules",
        embedding_function: Optional[Any] = None,
    ):
        """Initializes ChromaDB with Google's gemini-embedding-001.

        Connection target is env-driven, same pattern as AuditLogStore's
        AUDIT_MONGO_URI: nothing about *where* Chroma lives is accepted
        as a constructor argument, so call sites don't need to change
        between local dev, Tier B live tests, and an AWS deployment --
        only environment variables do.

          - No CHROMA_HOST set (default):
                chromadb.Client() -- ephemeral, in-memory, local. This is
                what every existing call site gets today; behavior is
                unchanged unless you opt in.
          - CHROMA_HOST set:
                chromadb.HttpClient(host=..., port=..., ssl=...) --
                points at a real Chroma server (self-hosted on AWS, or
                anywhere else reachable over the network).

        `embedding_function` is now injectable (defaults to the same
        GoogleGeminiEmbeddingFunction as before if not supplied). This
        exists so tests -- or any caller sharing a Gemini API key's
        quota across multiple call sites -- can inject a rate-limited
        wrapper instead of hitting the Gemini embeddings endpoint
        completely unthrottled. Embedding calls bill against the same
        API key as generation calls; without this, a rate limiter
        wrapped only around GeminiClient.generate() has a blind spot.

        Reads secrets implicitly from the shell environment to maximize
        execution safety -- the key is never accepted as a constructor
        argument, so it can't end up in a stack trace, a repr, or a log
        line further down the call chain.
        """
        if "GEMINI_API_KEY" not in os.environ:
            raise KeyError(
                "Missing critical environment variable: 'GEMINI_API_KEY'. "
                "Please export or set it in your terminal prior to execution."
            )

        chroma_host = os.environ.get("CHROMA_HOST")
        if chroma_host:
            chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
            chroma_ssl = os.environ.get("CHROMA_SSL", "false").lower() == "true"
            self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port, ssl=chroma_ssl)
        else:
            self.client = chromadb.Client()

        try:
            self.embedding_function = embedding_function or embedding_functions.GoogleGeminiEmbeddingFunction(
                model_name="gemini-embedding-001"
            )

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize compliance knowledge base: {safe_exception_message(exc)}"
            ) from None

    def seed_initial_compliance_data(self, passages: List[CompliancePassage]):
        """Seeds the vector DB with real regulatory excerpts. Uses upsert so
        re-seeding an existing id updates its content instead of erroring
        or silently no-op'ing."""
        if not passages:
            return

        ids = [p.id for p in passages]
        documents = [p.content for p in passages]
        metadatas = [{"citation": p.citation, **p.metadata} for p in passages]

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to seed compliance knowledge base: {safe_exception_message(exc)}"
            ) from None

    def query_relevant_policies(self, draft_reasoning: str, n_results: int = 3) -> List[CompliancePassage]:
        """Queries the knowledge base using the primary agent's reasoning text."""
        if not isinstance(draft_reasoning, str):
            draft_reasoning = str(draft_reasoning)

        try:
            results = self.collection.query(
                query_texts=[draft_reasoning],
                n_results=n_results
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to query compliance knowledge base: {safe_exception_message(exc)}"
            ) from None

        extracted_passages = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                extracted_passages.append(
                    CompliancePassage(
                        id=results["ids"][0][i],
                        content=results["documents"][0][i],
                        citation=results["metadatas"][0][i]["citation"],
                        metadata=results["metadatas"][0][i]
                    )
                )
        return extracted_passages