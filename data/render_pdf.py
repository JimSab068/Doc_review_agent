"""
Renders persona documents to actual PDF files so ingestion.py's
pdfplumber-based path (3.1) is genuinely exercised, per
func_testing_synthetic.md §4.1.3.

Degradation modes for the "degraded_document" category are implemented
as real PDF-level defects, not text edited before rendering:

- low_res:       text is rasterized to a small PIL image at low DPI
                  (default 45) and embedded as an image with NO text
                  layer -- pdfplumber.extract_text() genuinely gets
                  little-to-nothing back, the way a real scanned/OCR'd
                  low-quality document would behave against a
                  text-extraction-only pipeline (this system has no OCR
                  step, so "poor OCR text" is modeled as "no recoverable
                  text layer" rather than faking OCR output).
- cropped:       page height is set smaller than the content, so text
                  drawn past the bottom margin is physically outside the
                  page and pdfplumber never sees it -- a real crop, not
                  a string slice.
- missing_pages: one of the two source documents for the persona is
                  simply not rendered/returned at all.
- truncated:     the bank statement's transaction history is cut before
                  rendering (this one *is* a content-level truncation,
                  since "truncated statement" is inherently about missing
                  content rather than a PDF-mechanics defect).
"""

from __future__ import annotations

import os
import textwrap
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from schemas import Persona

PAGE_W, PAGE_H = letter
MARGIN = 54
LINE_HEIGHT = 14
FONT_SIZE = 10
WRAP_CHARS = 95


def _wrapped_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw_line, width=WRAP_CHARS) or [""])
    return lines


def render_clean_pdf(text: str, out_path: str) -> None:
    """Standard, fully-legible single/multi-page PDF with a real text
    layer -- what every non-degraded category uses."""
    c = canvas.Canvas(out_path, pagesize=letter)
    c.setFont("Helvetica", FONT_SIZE)
    y = PAGE_H - MARGIN
    for line in _wrapped_lines(text):
        if y < MARGIN:
            c.showPage()
            c.setFont("Helvetica", FONT_SIZE)
            y = PAGE_H - MARGIN
        c.drawString(MARGIN, y, line)
        y -= LINE_HEIGHT
    c.save()


def render_low_res_pdf(text: str, out_path: str, dpi_scale: float = 0.28) -> None:
    """Rasterize the page as a small bitmap (simulating a low-DPI scan)
    and embed only that image -- no text layer -- so pdfplumber's
    extract_text() genuinely returns little to nothing."""
    lines = _wrapped_lines(text)
    small_w = int(PAGE_W * dpi_scale)
    small_h = int(max(PAGE_H * dpi_scale, (len(lines) + 4) * LINE_HEIGHT * dpi_scale))

    img = Image.new("L", (small_w, small_h), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(6, int(FONT_SIZE * dpi_scale * 1.4)))
    except Exception:
        font = ImageFont.load_default()

    y = int(6 * dpi_scale) + 4
    for line in lines:
        draw.text((int(MARGIN * dpi_scale * 0.5), y), line, fill=0, font=font)
        y += max(3, int(LINE_HEIGHT * dpi_scale))

    # Upscale back to page size with a blur-like resample to mimic the
    # blocky, illegible-to-OCR look of a genuinely low-resolution scan.
    img_full = img.resize((int(PAGE_W), int(PAGE_H)), Image.BILINEAR)

    c = canvas.Canvas(out_path, pagesize=letter)
    from reportlab.lib.utils import ImageReader
    c.drawImage(ImageReader(img_full), 0, 0, width=PAGE_W, height=PAGE_H)
    c.save()


def render_cropped_pdf(text: str, out_path: str, visible_fraction: float = 0.45) -> None:
    """Draw full content but with a page height that only covers the top
    `visible_fraction` of it -- everything past that y-coordinate is
    outside the physical page and unrecoverable by pdfplumber, same as a
    real mis-scanned/cropped document."""
    lines = _wrapped_lines(text)
    full_content_h = MARGIN * 2 + len(lines) * LINE_HEIGHT
    cropped_h = max(120, full_content_h * visible_fraction)

    c = canvas.Canvas(out_path, pagesize=(PAGE_W, cropped_h))
    c.setFont("Helvetica", FONT_SIZE)
    y = cropped_h - MARGIN
    for line in lines:
        if y < 0:
            break  # physically off the page -- this is the crop
        c.drawString(MARGIN, y, line)
        y -= LINE_HEIGHT
    c.save()


def render_truncated_text_pdf(text: str, out_path: str, keep_fraction: float = 0.5) -> None:
    """Cut the source content short before rendering -- models a
    statement that was only partially captured/uploaded."""
    lines = text.split("\n")
    keep = max(3, int(len(lines) * keep_fraction))
    truncated = "\n".join(lines[:keep]) + "\n[... document truncated ...]"
    render_clean_pdf(truncated, out_path)


def render_persona(persona: Persona, out_dir: str) -> Dict[str, Optional[str]]:
    """Render every document in `persona` to a PDF under out_dir, applying
    the persona's degraded_mode if render_degraded is set. Returns
    {doc_type: pdf_path_or_None}; None means the document was deliberately
    omitted (missing_pages mode)."""
    os.makedirs(out_dir, exist_ok=True)
    result: Dict[str, Optional[str]] = {}

    docs = list(persona.documents)
    drop_index = None
    if persona.render_degraded and persona.degraded_mode == "missing_pages" and len(docs) > 1:
        drop_index = 1  # drop the bank statement, keep the loan application

    for i, doc in enumerate(docs):
        out_path = os.path.join(out_dir, f"{persona.persona_id}_{doc.doc_type}.pdf")

        if drop_index is not None and i == drop_index:
            result[doc.doc_type] = None
            continue

        if not persona.render_degraded:
            render_clean_pdf(doc.raw_text, out_path)
        elif persona.degraded_mode == "low_res":
            render_low_res_pdf(doc.raw_text, out_path)
        elif persona.degraded_mode == "cropped":
            render_cropped_pdf(doc.raw_text, out_path)
        elif persona.degraded_mode == "truncated" and doc.doc_type == "bank_statement":
            render_truncated_text_pdf(doc.raw_text, out_path)
        else:
            render_clean_pdf(doc.raw_text, out_path)

        result[doc.doc_type] = out_path

    return result