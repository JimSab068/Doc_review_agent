

"""
Persona / ground-truth schemas, matching spec section 4.1.2 exactly.

This is the ONE schema shared by Tier A (mocked), Tier B (live API), and
the synthetic persona generator -- ground truth is authored once and
reused across all three tiers. Kept deliberately separate from
src/schemas.py (the pipeline's own contracts): a persona is test *input*
+ hand-authored *truth*, never a pipeline object itself.

pdf_paths is deliberately NOT a field on Persona -- personas.json (the
generator's output) never contains file paths, only raw_text per
document. PDF paths are derived on demand from persona_id + doc_type via
get_pdf_paths(), once the renderer's naming convention is known, rather
than being something the model claims to "have" out of the box.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
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
    golden: bool = False
    render_degraded: bool = False
    degraded_mode: Optional[str] = None  # "low_res" | "cropped" | "missing_pages" | "truncated"


def get_pdf_paths(persona: Persona, pdfs_dir: "Path | str") -> list[str]:
    """Resolve the on-disk PDF file for each document in this persona.
    Missing files are skipped rather than raising -- for DEGRADED_DOCUMENT
    personas with missing_pages, a document intentionally may not exist;
    the pipeline is expected to handle the resulting partial packet."""
    pdfs_dir = Path(pdfs_dir)
    paths: list[str] = []
    missing: list[str] = []
    for doc in persona.documents:
        path = pdfs_dir / f"{persona.persona_id}_{doc.doc_type}.pdf"
        if path.exists():
            paths.append(str(path))
        else:
            missing.append(str(path))

    if missing and persona.degraded_mode != "missing_pages":
        raise FileNotFoundError(
            f"Persona '{persona.persona_id}' (category={persona.category.value}, "
            f"degraded_mode={persona.degraded_mode}) is missing expected PDF(s): {missing}"
        )

    if not paths:
        raise FileNotFoundError(f"Persona '{persona.persona_id}' has no renderable PDFs at all.")

    return paths


def load_personas(json_path: "Path | str") -> list[Persona]:
    """Load personas from the generator's personas.json.

    Handles both a bare top-level list and a dict wrapping the list
    under a common key -- adjust KNOWN_WRAPPER_KEYS if your generator
    output uses a different key name.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    if isinstance(data, dict):
        KNOWN_WRAPPER_KEYS = ("personas", "data", "items", "results")
        for key in KNOWN_WRAPPER_KEYS:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError(
                f"personas.json is a dict but none of {KNOWN_WRAPPER_KEYS} "
                f"found as a list key. Top-level keys: {list(data.keys())}"
            )

    if not isinstance(data, list):
        raise ValueError(f"Expected personas.json to contain a list, got {type(data).__name__}")

    return [Persona.model_validate(item) for item in data]

