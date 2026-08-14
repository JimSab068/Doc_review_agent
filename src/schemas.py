"""
Shared data models for the loan/KYC review pipeline.

These are the contracts between stages 3.1-3.4. Every stage consumes and
produces one of these types -- nothing passes raw dicts or strings between
components, so each stage can be tested and swapped independently.


Make sure it covers all the exhaustives reps 
it covers the usual ones not exhaustive

"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


from datetime import datetime
from typing import Literal, Optional, Any, Dict, List
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 3.1 Ingestion layer
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    LOAN_APPLICATION = "loan_application"
    BANK_STATEMENT = "bank_statement"
    UNKNOWN = "unknown"


class Page(BaseModel):
    page_number: int
    text: str


class Document(BaseModel):
    """Normalized output of the ingestion layer. Nothing downstream should
    ever touch a raw file path or PDF bytes again after this point."""

    doc_id: str
    doc_type: DocumentType
    source_filename: str
    pages: list[Page]

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


# ---------------------------------------------------------------------------
# 3.2 PII detector
# ---------------------------------------------------------------------------

class PIIType(str, Enum):
    SSN = "ssn"
    ACCOUNT_NUMBER = "account_number"
    DATE_OF_BIRTH = "date_of_birth"
    PHONE = "phone"
    EMAIL = "email"
    PERSON_NAME = "person_name"
    ADDRESS = "address"


class PIISpan(BaseModel):
    """One detected instance of sensitive data, located precisely enough
    that the vault can replace it without disturbing surrounding text."""

    field_type: PIIType
    raw_value: str
    doc_id: str
    page_number: int
    start_char: int
    end_char: int
    confidence: float = Field(ge=0.0, le=1.0)
    detector: str  # which detector found it, e.g. "regex:ssn" or "ner:spacy"


# ---------------------------------------------------------------------------
# 3.3 Tokenization vault
# ---------------------------------------------------------------------------

class TokenizedDocument(BaseModel):
    """A Document with every PIISpan replaced by an opaque token. This is
    the ONLY representation of the document that may ever be sent to an
    external LLM API."""

    doc_id: str
    doc_type: DocumentType
    pages: list[Page]
    token_map_id: str  # opaque reference to the vault's request-scoped map


# ---------------------------------------------------------------------------
# 3.4 Primary agent
# ---------------------------------------------------------------------------

class DraftDecision(BaseModel):
    """Structured output of the primary agent. Field values here may
    contain tokens (e.g. 'PII_TOKEN_ab12') rather than raw PII -- callers
    detokenize only after this object leaves the agent boundary."""

    doc_id: str
    extracted_fields: dict[str, str]
    inconsistencies: list[str] = Field(default_factory=list)
    missing_compliance_items: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    model_used: str

# critic
class CriticVerdict(BaseModel):
    verdict: Literal["pass", "flag"]
    cited_policy: List[str] = Field(
        description="Verbatim short citations and policy sources only (e.g., '12 CFR § 1002.9(a)(1)')"
    )
    concerns: List[str] = Field(
        description="Specific compliance issues, gaps, or violations flagged by the critic."
    )
    escalate: bool = Field(
        description="True if human compliance officer intervention is recommended."
    )

# routing decision
class RoutingDecision(BaseModel):
    route: Literal["auto_resolve", "human_queue"]
    escalation_reason: Optional[str] = None
    primary_confidence: float
    critic_escalated: bool

# log
class AuditLogEntry(BaseModel):
    doc_id: str
    input_hash: str
    pii_types_detected: List[str]
    primary_decision: Dict[str, Any]  # Serialized DraftDecision
    critic_verdict: Dict[str, Any]    # Serialized CriticVerdict
    routing_decision: Dict[str, Any]   # Serialized RoutingDecision
    timestamps: Dict[str, str]        # ISO strings for received, processed
    latency_ms: Dict[str, float]       # Breakdown of pipeline execution times
