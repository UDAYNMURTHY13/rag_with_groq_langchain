import pdfplumber
import os
import logging
import re
from pathlib import Path
from typing import Dict, List

# Prefer pdfplumber for more reliable extraction, fall back to PyPDF2
try:
    import pdfplumber as _pdfplumber
except Exception:
    _pdfplumber = None

try:
    from PyPDF2 import PdfReader as _PdfReader
except Exception:
    _PdfReader = None

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _extract_text_pdfplumber(pdf_path: Path) -> str:
    text_parts: List[str] = []
    with _pdfplumber.open(str(pdf_path)) as doc:
        for p in doc.pages:
            t = p.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)


def _extract_text_pypdf(pdf_path: Path) -> str:
    reader = _PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages).strip()


def _extract_full_text(pdf_path: Path) -> str:
    if _pdfplumber:
        try:
            return _extract_text_pdfplumber(pdf_path)
        except Exception as e:
            logging.debug(f"pdfplumber extraction failed: {e}")
    if _PdfReader:
        try:
            return _extract_text_pypdf(pdf_path)
        except Exception as e:
            logging.debug(f"PyPDF2 extraction failed: {e}")
    logging.error("No PDF parser available (install pdfplumber or PyPDF2).")
    return ""


def _find_section_bounds(text: str, start_keys: List[str], end_keys: List[str] = None) -> str:
    """
    Heuristic: find first occurrence of any start_keys, optional end_keys to stop.
    Returns substring or empty string.
    """
    lower = text.lower()
    start_idx = -1
    for k in start_keys:
        try:
            start_idx = lower.index(k.lower())
            break
        except ValueError:
            continue
    if start_idx == -1:
        return ""

    if end_keys:
        for ek in end_keys:
            try:
                end_idx = lower.index(ek.lower(), start_idx + 1)
                return text[start_idx:end_idx].strip()
            except ValueError:
                continue
    return text[start_idx:].strip()


def _first_n_words(text: str, n: int = 250) -> str:
    words = re.findall(r"\S+", text)
    return " ".join(words[:n])


def _last_n_words(text: str, n: int = 250) -> str:
    words = re.findall(r"\S+", text)
    return " ".join(words[-n:])


def extract_sections_from_pdf(pdf_path: str) -> Dict[str, str]:
    """
    Extract common sections (abstract, introduction, conclusion) from a PDF.
    Returns dict with keys present. Falls back to full_text if headers not found.
    """
    p = Path(pdf_path)
    if not p.exists():
        logging.error(f"PDF not found: {pdf_path}")
        return {}

    full_text = _extract_full_text(p)
    if not full_text:
        return {}

    # Try header-based extraction
    abstract = _find_section_bounds(full_text, ["abstract"], ["introduction", "1. introduction", "introduction\n"])
    introduction = _find_section_bounds(full_text, ["introduction", "1 introduction", "1. introduction"], ["method", "methods", "related work", "conclusion"])
    conclusion = _find_section_bounds(full_text, ["conclusion", "conclusions"], ["references", "acknowledgements", "references\n"])

    # If headers not detected, fallback heuristics
    if not abstract:
        abstract = _first_n_words(full_text, 200)
    if not introduction:
        introduction = _first_n_words(full_text.replace(abstract, ""), 400)
    if not conclusion:
        conclusion = _last_n_words(full_text, 250)

    # Clean up whitespace
    sections = {
        "abstract": abstract.strip(),
        "introduction": introduction.strip(),
        "conclusion": conclusion.strip(),
        "full_text": full_text.strip()
    }
    # Remove empty entries except full_text
    return {k: v for k, v in sections.items() if v}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Chunk text by approximate word counts for embedding. Returns list[str].
    """
    words = re.findall(r"\S+", text)
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_scraper.py <path-to-pdf>")
        sys.exit(1)
    path = sys.argv[1]
    secs = extract_sections_from_pdf(path)
    print("Extracted sections:")
    for k, v in secs.items():
        print(f"\n=== {k.upper()} ===\n")
        print(v[:1000])