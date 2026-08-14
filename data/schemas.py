"""
Persona / ground-truth schemas.

These mirror the JSON shape specified in func_testing_synthetic.md §4.1.2
exactly, so eval_report.py (Tier A/B/C harnesses) can load personas
without any translation layer. Kept deliberately separate from
src/schemas.py (the pipeline's own contracts) -- a persona is test
*input* + hand-authored *truth*, never a pipeline object itself.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    CLEAN_APPROVE = "clean_approve"
    CLEAN_DENY = "clean_deny"
    INCONSISTENCY_OBVIOUS = "inconsistency_obvious"
    INCONSISTENCY_INFERENTIAL = "inconsistency_inferential"
    LOOKS_LIKE_CONFLICT = "looks_like_conflict_but_isnt"
    AMBIGUOUS_ESCALATE = "ambiguous_escalate"
    INJECTION_DIRECT = "injection_direct"
    INJECTION_INDIRECT = "injection_indirect"
    PII_OBFUSCATED_ADVERSARIAL = "pii_obfuscated_adversarial"
    DEGRADED_DOCUMENT = "degraded_document"
    FAIRNESS_PAIR = "fairness_pair"
    # golden_regression is not a distinct generation category -- it's a
    # frozen flag applied to a hand-picked subset drawn from the other
    # categories (see build_personas.py). A persona can be both e.g.
    # CLEAN_APPROVE and golden=True.


class PersonaDocument(BaseModel):
    doc_type: str  # "loan_application" | "bank_statement"
    raw_text: str


class GroundTruthPIISpan(BaseModel):
    field_type: str
    raw_value: str
    location_hint: str


class InjectionPlanted(BaseModel):
    present: bool = False
    type: Optional[str] = None  # "direct" | "indirect"
    payload: Optional[str] = None


class GroundTruth(BaseModel):
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)
    pii_spans: List[GroundTruthPIISpan] = Field(default_factory=list)
    expected_inconsistencies: List[str] = Field(default_factory=list)
    expected_missing_compliance_items: List[str] = Field(default_factory=list)
    expected_escalate: bool = False
    expected_verdict: str = "pass"  # "pass" | "flag"
    injection_planted: InjectionPlanted = Field(default_factory=InjectionPlanted)
    fairness_pair_id: Optional[str] = None


class Persona(BaseModel):
    persona_id: str
    category: Category
    documents: List[PersonaDocument]
    ground_truth: GroundTruth
    golden: bool = False  # True if part of the frozen 25-case regression set
    render_degraded: bool = False  # True -> pdf_renderer produces a genuinely
    # low-res / cropped PDF instead of a clean one (only for DEGRADED_DOCUMENT)
    degraded_mode: Optional[str] = None  # "low_res" | "cropped" | "missing_pages" | "truncated"