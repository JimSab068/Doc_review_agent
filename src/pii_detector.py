

"""
Stage 3.2: PII detector.

Layered detection: regex for structured formats (SSN, account numbers,
DOB, phone, email, name headers) plus a pluggable NER detector for names/addresses.

Design note: the regex layer ships as the working implementation because
it's deterministic, fast, and needs no model download -- good for a
free-tier / offline-friendly build. The NER layer is defined as a Protocol
so a real model (spaCy, Presidio, a fine-tuned transformer) can be dropped
in later without touching any caller. Recall is the metric that matters
most here -- a missed PII span is a leak, not a minor accuracy hit -- so
the eval suite must score this layer on recall specifically.
"""

from __future__ import annotations

import re
from typing import Protocol

from src.schemas import Document, PIISpan, PIIType

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Patterns use named capture group `(?P<target>...)` where prefixes (e.g., "Acct #")
# need to be stripped from the raw PII value. Patterns are tolerant to line breaks
# and whitespace artifacts introduced by OCR or adversarial formatting.

_PATTERNS: dict[PIIType, re.Pattern] = {}

# Header-Aware Name Matching
if hasattr(PIIType, "NAME"):
    _PATTERNS[PIIType.NAME] = re.compile(
        r"\b(?:Applicant\s+Name|Borrower\s+Name|Account\s+Holder|Co-Borrower\s+Name|Full\s+Name|Primary\s+Applicant|Applicant|Borrower|Name)\s*[:\-\=]?[ \t]*(?P<target>[A-Za-z]['A-Za-z\.\-']*(?:[ \t]+[A-Za-z]['A-Za-z\.\-']*)+)",
        re.IGNORECASE,
    )

# SSN: Tolerant to spaces, hyphens, or embedded newlines (e.g. 265\n51\n4694)
if hasattr(PIIType, "SSN"):
    _PATTERNS[PIIType.SSN] = re.compile(
        r"\b\d{3}[\s\n-]*\d{2}[\s\n-]*\d{4}\b"
    )

# Account Number: Strips prefix using target capture group
if hasattr(PIIType, "ACCOUNT_NUMBER"):
    _PATTERNS[PIIType.ACCOUNT_NUMBER] = re.compile(
        r"\b(?:Account|Acct\.?)\s*(?:No\.?|Number|#)?\s*:?\s*(?P<target>\d{8,17})\b",
        re.IGNORECASE,
    )

# Date of Birth
if hasattr(PIIType, "DATE_OF_BIRTH"):
    _PATTERNS[PIIType.DATE_OF_BIRTH] = re.compile(
        r"\b(?:DOB|Date of Birth)\s*:?\s*(?P<target>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        re.IGNORECASE,
    )

# Phone Number: Tolerant to multiline breaks and various delimiters
if hasattr(PIIType, "PHONE"):
    _PATTERNS[PIIType.PHONE] = re.compile(
        r"(?:\+?1[\s\n.-]*)?\(?\d{3}\)?[\s\n.-]*\d{3}[\s\n.-]*\d{4}\b"
    )

# Email Address
if hasattr(PIIType, "EMAIL"):
    _PATTERNS[PIIType.EMAIL] = re.compile(
        r"\b[\w.+-]+[\s\n]*@[\s\n]*[\w-]+\.[\w.-]+\b"
    )


class NERDetector(Protocol):
    """Interface for a pluggable name/address detector. Swap in spaCy,
    Presidio, or a fine-tuned model by implementing this protocol --
    RegexPIIDetector.detect() will call it if one is provided."""

    def detect(self, text: str) -> list[tuple[PIIType, str, int, int]]:
        """Return (field_type, raw_value, start_char, end_char) tuples."""
        ...


class RegexPIIDetector:
    """Working PII detector: regex layer with obfuscation tolerance, plus an
    optional NER detector for names/addresses if one is injected."""

    def __init__(self, ner_detector: NERDetector | None = None):
        self._ner_detector = ner_detector

    def detect(self, document: Document) -> list[PIISpan]:
        spans: list[PIISpan] = []

        for page in document.pages:
            page_spans = self._detect_regex(document.doc_id, page.page_number, page.text)
            if self._ner_detector is not None:
                page_spans.extend(
                    self._detect_ner(document.doc_id, page.page_number, page.text)
                )

            # Resolve duplicate or overlapping span boundaries per page
            spans.extend(self._resolve_overlaps(page_spans))

        return spans

    def _detect_regex(self, doc_id: str, page_number: int, text: str) -> list[PIISpan]:
        spans: list[PIISpan] = []
        for field_type, pattern in _PATTERNS.items():
            for match in pattern.finditer(text):
                # Safely resolve target group ('target' named group > group 1 > full match)
                if "target" in pattern.groupindex:
                    group_key: str | int = "target"
                elif match.groups():
                    group_key = 1
                else:
                    group_key = 0

                try:
                    value = match.group(group_key)
                    start = match.start(group_key)
                    end = match.end(group_key)
                except (IndexError, KeyError):
                    value = match.group(0)
                    start = match.start()
                    end = match.end()

                if not value:
                    continue
                
                if field_type == PIIType.SSN:
                    digits = re.sub(r"\D", "", value)
                    if len(digits) == 9:
                        value = f"{digits[0:3]}-{digits[3:5]}-{digits[5:9]}"



                spans.append(
                    PIISpan(
                        field_type=field_type,
                        raw_value=value,
                        doc_id=doc_id,
                        page_number=page_number,
                        start_char=start,
                        end_char=end,
                        confidence=0.95,
                        detector=f"regex:{field_type.value if hasattr(field_type, 'value') else field_type}",
                    )
                )
        return spans

    def _detect_ner(self, doc_id: str, page_number: int, text: str) -> list[PIISpan]:
        spans: list[PIISpan] = []
        for field_type, value, start, end in self._ner_detector.detect(text):
            spans.append(
                PIISpan(
                    field_type=field_type,
                    raw_value=value,
                    doc_id=doc_id,
                    page_number=page_number,
                    start_char=start,
                    end_char=end,
                    confidence=0.75,
                    detector="ner",
                )
            )
        return spans

    def _resolve_overlaps(self, spans: list[PIISpan]) -> list[PIISpan]:
        """Greedy overlap resolution (Non-Maximum Suppression).
        Higher confidence spans take priority; if tied, longer spans take priority.
        """
        if not spans:
            return []

        # Sort by confidence descending, then span character length descending
        sorted_spans = sorted(
            spans,
            key=lambda s: (s.confidence, s.end_char - s.start_char),
            reverse=True,
        )

        kept_spans: list[PIISpan] = []
        for candidate in sorted_spans:
            # Keep span if it doesn't overlap with any higher-priority span already kept
            has_overlap = any(
                max(candidate.start_char, kept.start_char) < min(candidate.end_char, kept.end_char)
                for kept in kept_spans
            )
            if not has_overlap:
                kept_spans.append(candidate)

        # Return sorted by document character position
        return sorted(kept_spans, key=lambda s: s.start_char)