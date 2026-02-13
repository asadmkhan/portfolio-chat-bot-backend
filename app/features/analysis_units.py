from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from app.parsing.models import ParsedDoc
from app.schemas.normalized import EvidenceSpan, NormalizedResume

UnitType = Literal[
    "header",
    "contact",
    "url",
    "section_title",
    "objective",
    "experience_bullet",
    "skills",
    "education",
    "other",
]

_BULLET_RE = re.compile(r"^\s*(?:[-*•·▪◦‣]|(?:\d+[\.\)]))\s+")
_SECTION_RE = re.compile(
    r"^\s*(summary|objective|profile|experience|work experience|employment|skills|education|projects|certifications)\s*:?\s*$",
    re.IGNORECASE,
)
_CONTACT_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\+?\d[\d\s().-]{7,}\d")
_URL_RE = re.compile(r"(?:https?://|www\.|linkedin\.com|github\.com)", re.IGNORECASE)
_ACTION_START_RE = re.compile(
    r"^\s*(built|led|managed|designed|developed|implemented|created|delivered|improved|reduced|scaled|launched|optimized)\b",
    re.IGNORECASE,
)


class AnalysisUnit(BaseModel):
    unit_id: str
    unit_type: UnitType
    text: str
    line_start: int | None = None
    line_end: int | None = None
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)


def _make_unit(
    *,
    idx: int,
    unit_type: UnitType,
    text: str,
    line_start: int,
    line_end: int,
    doc_id: str | None,
) -> AnalysisUnit:
    snippet = text.strip()
    return AnalysisUnit(
        unit_id=f"u{idx}",
        unit_type=unit_type,
        text=snippet,
        line_start=line_start,
        line_end=line_end,
        evidence_spans=[
            EvidenceSpan(
                doc_id=doc_id,
                line_start=line_start,
                line_end=line_end,
                text_snippet=snippet[:400],
            )
        ],
    )


def _is_section_title(line: str) -> bool:
    return bool(_SECTION_RE.match(line))


def _is_contact(line: str) -> bool:
    return bool(_CONTACT_RE.search(line))


def _is_url(line: str) -> bool:
    return bool(_URL_RE.search(line))


def _normalize_section_name(line: str) -> str:
    lowered = line.strip().lower().rstrip(":")
    if lowered in {"summary", "objective", "profile"}:
        return "objective"
    if lowered in {"experience", "work experience", "employment"}:
        return "experience"
    if lowered == "skills":
        return "skills"
    if lowered == "education":
        return "education"
    return "other"


def _bullet_text(line: str) -> str:
    return _BULLET_RE.sub("", line).strip()


def _line_kind(line: str, section: str, *, line_no: int) -> UnitType:
    stripped = line.strip()
    if _is_section_title(stripped):
        return "section_title"
    if _is_url(stripped):
        return "url"
    if _is_contact(stripped):
        return "contact"
    if section == "objective":
        return "objective"
    if section == "skills":
        return "skills"
    if section == "education":
        return "education"
    if section == "experience":
        if _BULLET_RE.match(stripped) or _ACTION_START_RE.match(stripped):
            return "experience_bullet"
        return "other"
    # Header lines are short identity/role lines near the top, not prose sentences.
    if (
        line_no <= 4
        and len(stripped.split()) <= 8
        and "." not in stripped
        and not re.search(r"\b(?:have|built|led|managed|designed|developed|created|improved|reduced)\b", stripped, re.IGNORECASE)
    ):
        return "header"
    return "other"


def build_analysis_units(parsed_doc: ParsedDoc, normalized_resume: NormalizedResume | None = None) -> list[AnalysisUnit]:
    lines = parsed_doc.text.splitlines()
    units: list[AnalysisUnit] = []
    current_section = "other"
    idx = 1
    pending_bullet: dict[str, int | str] | None = None

    def flush_pending() -> None:
        nonlocal idx, pending_bullet
        if pending_bullet is None:
            return
        units.append(
            _make_unit(
                idx=idx,
                unit_type="experience_bullet",
                text=str(pending_bullet["text"]),
                line_start=int(pending_bullet["line_start"]),
                line_end=int(pending_bullet["line_end"]),
                doc_id=parsed_doc.doc_id,
            )
        )
        idx += 1
        pending_bullet = None

    for line_no, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped:
            flush_pending()
            continue

        if _is_section_title(stripped):
            flush_pending()
            current_section = _normalize_section_name(stripped)
            units.append(
                _make_unit(
                    idx=idx,
                    unit_type="section_title",
                    text=stripped,
                    line_start=line_no,
                    line_end=line_no,
                    doc_id=parsed_doc.doc_id,
                )
            )
            idx += 1
            continue

        kind = _line_kind(stripped, current_section, line_no=line_no)
        if kind == "experience_bullet":
            bullet = _bullet_text(stripped) or stripped
            if pending_bullet is not None:
                flush_pending()
            pending_bullet = {"text": bullet, "line_start": line_no, "line_end": line_no}
            continue

        if pending_bullet is not None:
            continuation = (
                kind in {"other", "objective", "skills", "education"}
                and not _is_contact(stripped)
                and not _is_url(stripped)
            )
            if continuation and len(stripped.split()) >= 2:
                pending_bullet["text"] = f"{pending_bullet['text']} {stripped}"
                pending_bullet["line_end"] = line_no
                continue
            flush_pending()

        units.append(
            _make_unit(
                idx=idx,
                unit_type=kind,
                text=stripped,
                line_start=line_no,
                line_end=line_no,
                doc_id=parsed_doc.doc_id,
            )
        )
        idx += 1

    flush_pending()

    return units


def summarize_analysis_units(units: list[AnalysisUnit]) -> dict[str, int]:
    counts = Counter(unit.unit_type for unit in units)
    summary: dict[str, int] = {"total_units": len(units)}
    for key in ("header", "contact", "url", "section_title", "objective", "experience_bullet", "skills", "education", "other"):
        summary[key] = int(counts.get(key, 0))
    return summary
