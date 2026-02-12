from __future__ import annotations

import hashlib
from pathlib import Path

from .models import ParsedBlock, ParsedDoc


def _compute_doc_id(text: str, file_path: Path) -> str:
    seed = text if text.strip() else file_path.name
    digest = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return digest[:16]


def _parse_txt(file_path: Path) -> tuple[str, list[ParsedBlock], list[str]]:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return text, [], []


def _parse_pdf(file_path: Path) -> tuple[str, list[ParsedBlock], list[str]]:
    warnings: list[str] = []
    blocks: list[ParsedBlock] = []

    try:
        from pypdf import PdfReader
    except Exception:
        warnings.append("pypdf is unavailable; returning empty text for PDF.")
        return "", blocks, warnings

    try:
        reader = PdfReader(str(file_path))
        text_parts: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                text_parts.append(page_text)
                blocks.append(ParsedBlock(page=index, bbox=None, text=page_text))
        if not text_parts:
            warnings.append("No extractable text found in PDF.")
        return "\n".join(text_parts), blocks, warnings
    except Exception as exc:
        warnings.append(f"PDF parsing failed: {exc}")
        return "", blocks, warnings


def _parse_docx(file_path: Path) -> tuple[str, list[ParsedBlock], list[str]]:
    warnings: list[str] = []
    blocks: list[ParsedBlock] = []

    try:
        from docx import Document
    except Exception:
        warnings.append("python-docx is unavailable; returning empty text for DOCX.")
        return "", blocks, warnings

    try:
        document = Document(str(file_path))
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
        for paragraph_text in paragraphs:
            blocks.append(ParsedBlock(page=None, bbox=None, text=paragraph_text))
        if not paragraphs:
            warnings.append("No extractable text found in DOCX.")
        return "\n".join(paragraphs), blocks, warnings
    except Exception as exc:
        warnings.append(f"DOCX parsing failed: {exc}")
        return "", blocks, warnings


def parse_document(file_path: str) -> ParsedDoc:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input document not found: '{path}'")

    extension = path.suffix.lower()
    if extension == ".txt":
        source_type = "txt"
        text, blocks, warnings = _parse_txt(path)
    elif extension == ".pdf":
        source_type = "pdf"
        text, blocks, warnings = _parse_pdf(path)
    elif extension == ".docx":
        source_type = "docx"
        text, blocks, warnings = _parse_docx(path)
    else:
        raise NotImplementedError(
            f"Unsupported file type '{extension}'. Supported types: .txt, .pdf, .docx"
        )

    doc_id = _compute_doc_id(text=text, file_path=path)
    return ParsedDoc(
        doc_id=doc_id,
        source_type=source_type,
        language=None,
        text=text,
        blocks=blocks,
        parsing_warnings=warnings,
        layout_flags={},
    )

