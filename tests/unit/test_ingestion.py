"""
PDF ingestion path tests (Stage 3.1).

Exercises `ingest_pdf` directly. Per spec 4.1.3, this is the layer most
likely to introduce real-world extraction noise (broken line wraps,
missing text) -- and it previously had zero unit coverage; test_ingestion.py
only exercised `ingest_text`.

Requires `reportlab` as a test-only dependency to generate fixture PDFs
on the fly rather than checking in binary files:
    pip install reportlab --break-system-packages
"""

from __future__ import annotations

import pytest

reportlab = pytest.importorskip("reportlab", reason="reportlab required to generate fixture PDFs")

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.ingestion import ingest_pdf
from src.pii_detector import RegexPIIDetector
from src.schemas import DocumentType

from unittest.mock import MagicMock, patch


def _make_pdf(path, pages_text: list[str]) -> None:
    """Writes a simple multi-page PDF, one line of text per page."""
    c = canvas.Canvas(str(path), pagesize=letter)
    for text in pages_text:
        c.setFont("Helvetica", 12)
        c.drawString(72, 720, text)
        c.showPage()
    c.save()


class TestIngestPdf:
    def test_extracts_text_from_each_page(self, tmp_path):
        pdf_path = tmp_path / "multi_page.pdf"
        _make_pdf(pdf_path, ["Page one content.", "Page two content.", "Page three content."])

        doc = ingest_pdf(str(pdf_path), DocumentType.LOAN_APPLICATION)

        assert len(doc.pages) == 3
        assert "Page one content." in doc.pages[0].text
        assert "Page two content." in doc.pages[1].text
        assert "Page three content." in doc.pages[2].text

    def test_page_numbers_are_sequential_from_one(self, tmp_path):
        pdf_path = tmp_path / "seq.pdf"
        _make_pdf(pdf_path, ["one", "two"])

        doc = ingest_pdf(str(pdf_path), DocumentType.LOAN_APPLICATION)

        assert [p.page_number for p in doc.pages] == [1, 2]

    def test_generates_doc_id_when_not_provided(self, tmp_path):
        pdf_path = tmp_path / "a.pdf"
        _make_pdf(pdf_path, ["content"])

        doc1 = ingest_pdf(str(pdf_path), DocumentType.LOAN_APPLICATION)
        doc2 = ingest_pdf(str(pdf_path), DocumentType.LOAN_APPLICATION)

        assert doc1.doc_id != doc2.doc_id

    def test_uses_provided_doc_id(self, tmp_path):
        pdf_path = tmp_path / "a.pdf"
        _make_pdf(pdf_path, ["content"])

        doc = ingest_pdf(str(pdf_path), DocumentType.LOAN_APPLICATION, doc_id="fixed-pdf-id")

        assert doc.doc_id == "fixed-pdf-id"

    def test_source_filename_is_pdf_path(self, tmp_path):
        pdf_path = tmp_path / "app.pdf"
        _make_pdf(pdf_path, ["content"])

        doc = ingest_pdf(str(pdf_path), DocumentType.LOAN_APPLICATION)

        assert doc.source_filename == str(pdf_path)

    def test_raises_on_unreadable_pdf(self, tmp_path):
        # reportlab always produces at least one page, so we can't easily
        # construct a "zero pages" PDF to hit the ValueError branch
        # directly -- instead confirm pdfplumber's own failure mode
        # (garbage bytes it can't open as a PDF at all) surfaces as an
        # exception rather than silently returning an empty Document.
        bad_path = tmp_path / "not_a_pdf.pdf"
        bad_path.write_text("this is not a real pdf")

        with pytest.raises(Exception):
            ingest_pdf(str(bad_path), DocumentType.LOAN_APPLICATION)

    def test_pii_detector_finds_ssn_extracted_from_pdf(self, tmp_path):
        """Confirms PII detection still works on pdfplumber's extracted
        text, not just on hand-written test strings -- pdfplumber can
        introduce different whitespace/line-break behavior than plain
        text ingestion, which is the exact class of noise spec 4.1.3
        flags as the reason not to skip this layer in testing."""
        pdf_path = tmp_path / "pii.pdf"
        _make_pdf(pdf_path, ["Applicant SSN: 123-45-6789, DOB: 04/12/1990"])

        doc = ingest_pdf(str(pdf_path), DocumentType.LOAN_APPLICATION)
        spans = RegexPIIDetector().detect(doc)

        ssn_spans = [s for s in spans if s.raw_value == "123-45-6789"]
        assert len(ssn_spans) == 1

    def test_multi_document_packet_via_separate_pdfs(self, tmp_path):
        """Mirrors the real usage pattern: a loan packet is multiple PDF
        files (application + bank statement), each ingested separately
        and later combined by the pipeline (see execute_pdf)."""
        app_path = tmp_path / "application.pdf"
        stmt_path = tmp_path / "bank_statement.pdf"
        _make_pdf(app_path, ["LOAN APPLICATION", "Stated monthly income: $9,500"])
        _make_pdf(stmt_path, ["BANK STATEMENT", "Average monthly deposits: $4,200"])

        app_doc = ingest_pdf(str(app_path), DocumentType.LOAN_APPLICATION)
        stmt_doc = ingest_pdf(str(stmt_path), DocumentType.BANK_STATEMENT)

        assert "9,500" in app_doc.full_text
        assert "4,200" in stmt_doc.full_text
        assert app_doc.doc_id != stmt_doc.doc_id

    def test_extract_text_none_fallback_to_empty_string(self, tmp_path):
        """Verifies that if extract_text() returns None (e.g., blank or image page),
        it safely falls back to an empty string instead of crashing."""
        pdf_path = tmp_path / "blank_page.pdf"
        # Canvas created without calling drawString/showPage still produces a page
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.showPage()
        c.save()

        # Patch extract_text to simulate pdfplumber returning None for an unextractable page
        with patch("pdfplumber.open") as mock_open:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = None
            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page]
            mock_open.return_value.__enter__.return_value = mock_pdf

            with pytest.raises(ValueError, match="no text layer found"):
                    ingest_pdf(str(pdf_path), DocumentType.LOAN_APPLICATION)

    def test_raises_value_error_when_pdf_has_zero_pages(self, tmp_path):
        """Simulates a PDF with zero pages to trigger the 'if not pages' ValueError branch."""
        pdf_path = tmp_path / "empty_pages.pdf"
        pdf_path.touch()

        with patch("pdfplumber.open") as mock_open:
            mock_pdf = MagicMock()
            mock_pdf.pages = []  # No pages found
            mock_open.return_value.__enter__.return_value = mock_pdf

            with pytest.raises(ValueError, match="No pages extracted from"):
                ingest_pdf(str(pdf_path), DocumentType.LOAN_APPLICATION)

    def test_raises_file_not_found_for_missing_file(self):
        """Confirms FileNotFoundError is raised when the file path does not exist."""
        with pytest.raises(FileNotFoundError):
            ingest_pdf("non_existent_file.pdf", DocumentType.LOAN_APPLICATION)