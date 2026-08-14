"""
Stage 3.1: Ingestion layer.

Converts raw input (PDF bytes, or plain text for testing/synthetic personas)
into normalized Document objects. This is the only layer allowed to touch
raw file bytes -- everything downstream works with Document/Page objects.
"""

from __future__ import annotations

import uuid

from src.schemas import Document, DocumentType, Page

import pdfplumber


"""
From text or PDF bytes it creates a document object and it contains multiple pages.

"""

def ingest_text(
    text: str,
    doc_type: DocumentType,
    source_filename: str,
    doc_id: str | None = None,
) -> Document:
    """Build a Document from plain text. Used for synthetic personas and
    for any input that's already text (as opposed to a PDF)."""
    if not text or not text.strip():
        raise ValueError("Cannot ingest an empty document")

    return Document(
        doc_id=doc_id or str(uuid.uuid4()),
        doc_type=doc_type,
        source_filename=source_filename,
        pages=[Page(page_number=1, text=text)],
    )


def ingest_pdf(pdf_path: str, doc_type: DocumentType, doc_id: str | None = None) -> Document:
    """Build a Document from a PDF file on disk, one Page per PDF page.

    Requires pdfplumber. Kept as a thin wrapper so the rest of the system
    never depends on pdfplumber directly -- swapping the PDF library later
    only touches this function.
    """
    pages: list[Page] = []
    empty_page_numbers: list[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, pdf_page in enumerate(pdf.pages, start=1):
            text = pdf_page.extract_text() or ""
            if not text.strip():
                empty_page_numbers.append(i)
            pages.append(Page(page_number=i, text=text))

    if not pages:
        raise ValueError(f"No pages extracted from {pdf_path}")

    if len(empty_page_numbers) == len(pages):
        # Every page came back with no text layer at all -- almost
        # certainly an image-only/scanned PDF with no OCR applied.
        # Distinct failure mode from "model struggled with noisy text":
        # surface it loudly here rather than silently handing downstream
        # an all-empty Document that then gets blamed on the model.
        raise ValueError(
            f"{pdf_path}: no text layer found on any of {len(pages)} page(s) -- "
            f"likely an image-only PDF with no OCR applied. This ingestion "
            f"pipeline has no OCR step (known limitation)."
        )

    return Document(
        doc_id=doc_id or str(uuid.uuid4()),
        doc_type=doc_type,
        source_filename=pdf_path,
        pages=pages,
    )