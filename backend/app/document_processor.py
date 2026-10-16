from __future__ import annotations

from typing import Any
import io
import os
import shutil
from pathlib import Path
import fitz


def _ocr_page(page: fitz.Page) -> tuple[str, str | None]:
    """OCR a rendered PDF page with Tesseract; return text and optional warning."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return "", "OCR dependencies are missing. Install pytesseract and Pillow."

    # Support a custom executable path and common Windows installation paths.
    # This fixes Windows setups where Tesseract is installed but not on PATH.
    configured_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    candidates = [
        configured_cmd,
        shutil.which("tesseract") or "",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            pytesseract.pytesseract.tesseract_cmd = candidate
            break

    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        text = pytesseract.image_to_string(image, config="--oem 3 --psm 3").strip()
        return text, None
    except pytesseract.pytesseract.TesseractNotFoundError:
        return "", "Tesseract OCR executable was not found. Install Tesseract and ensure it is on PATH."
    except Exception as exc:
        return "", f"OCR could not process this page ({type(exc).__name__})."


def _page_is_image_heavy(page: fitz.Page) -> bool:
    """Return True when raster images cover a substantial part of the page."""
    try:
        area = max(float(page.rect.width * page.rect.height), 1.0)
        image_area = 0.0
        for info in page.get_image_info(xrefs=True):
            bbox = info.get("bbox")
            if bbox:
                rect = fitz.Rect(bbox) & page.rect
                image_area += max(0.0, rect.width * rect.height)
        return min(image_area / area, 1.0) >= 0.25
    except Exception:
        return False


def _choose_text(native_text: str, ocr_text: str) -> tuple[str, bool]:
    """Prefer OCR only when it returns meaningfully more readable content."""
    native_text = native_text.strip()
    ocr_text = ocr_text.strip()
    if not ocr_text:
        return native_text, False
    if not native_text or len(ocr_text) >= len(native_text) * 1.15:
        return ocr_text, True
    return native_text, False


def process_pdf(filename: str, content: bytes) -> dict[str, Any]:
    """Extract PDF text and OCR scanned/image-heavy pages without changing API keys."""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open {filename} as a valid PDF.") from exc

    pages = []
    ocr_warnings: list[str] = []
    try:
        for index, page in enumerate(doc):
            native_text = page.get_text("text").strip()
            image_heavy = _page_is_image_heavy(page)
            # Trigger OCR for image-only pages, low-text pages, and image-heavy
            # pages whose text layer may be incomplete. This is the key fix for
            # scanned PDFs that contain a small hidden/selectable text layer.
            should_ocr = len(native_text) < 80 or (image_heavy and len(native_text) < 1800)
            page_text = native_text
            used_ocr = False
            ocr_attempted = False

            if should_ocr:
                ocr_attempted = True
                ocr_text, warning = _ocr_page(page)
                if warning and warning not in ocr_warnings:
                    ocr_warnings.append(warning)
                page_text, used_ocr = _choose_text(native_text, ocr_text)

            low_text = len(page_text.strip()) < 25
            pages.append({
                "page": index + 1,
                "text": page_text,
                "character_count": len(page_text),
                "word_count": len(page_text.split()),
                "likely_scanned": low_text,
                "used_ocr": used_ocr,
                "ocr_attempted": ocr_attempted,
                "extraction_status": "ocr" if used_ocr else "low_text" if low_text else "good",
            })
    finally:
        doc.close()

    combined = "\n\n".join(
        f"[Page {page['page']}]\n{page['text']}"
        for page in pages if page["text"]
    )
    scanned_pages = [p["page"] for p in pages if p["likely_scanned"]]
    ocr_pages = [p["page"] for p in pages if p["used_ocr"]]
    attempted_pages = [p["page"] for p in pages if p["ocr_attempted"]]

    if ocr_warnings:
        status = "ocr_unavailable" if not ocr_pages and attempted_pages else "partial"
    elif scanned_pages:
        status = "partial" if combined.strip() else "no_text"
    elif ocr_pages:
        status = "ocr_processed"
    else:
        status = "good"

    return {
        "filename": filename,
        "page_count": len(pages),
        "pages": pages,
        "text": combined,
        "has_extractable_text": bool(combined.strip()),
        "character_count": len(combined),
        "likely_scanned_pages": scanned_pages,
        "ocr_pages": ocr_pages,
        "ocr_attempted_pages": attempted_pages,
        "ocr_warnings": ocr_warnings,
        "extraction_status": status,
        "preview": combined[:1200],
    }
