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

_BULLET_RE = re.compile(
    r"^\s*(?:[-*]|[\u2022\u00b7\u25aa\u25ab\u25cf\u25e6\u2043\u2023\u2219\uf0b7]|(?:\d+[\.\)]))\s+"
)
_CONTACT_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\+?\d[\d\s().-]{7,}\d")
_URL_RE = re.compile(r"(?:https?://|www\.|linkedin\.com|github\.com)", re.IGNORECASE)
_ACTION_START_RE = re.compile(
    r"^\s*(built|led|managed|designed|developed|implemented|created|delivered|improved|reduced|scaled|launched|optimized|engineered|owned|drove|architected)\b",
    re.IGNORECASE,
)
_SECTION_KEYWORDS = (
    "summary",
    "objective",
    "profile",
    "professional summary",
    "experience",
    "work experience",
    "professional experience",
    "employment",
    "career history",
    "skills",
    "technical skills",
    "core skills",
    "education",
    "projects",
    "certifications",
)


class AnalysisUnit(BaseModel):
    unit_id: str
    unit_type: UnitType
    text: str
    line_start: int | None = None
    line_end: int | None = None
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)


def _normalize_line(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _is_section_title(line: str) -> bool:
    stripped = _normalize_line(line)
    if not stripped:
        return False
    lowered = stripped.lower().rstrip(":")
    if any(lowered == keyword for keyword in _SECTION_KEYWORDS):
        return True
    if any(keyword in lowered for keyword in ("experience", "skills", "education", "summary", "objective", "profile")) and len(stripped.split()) <= 4:
        return True
    return bool(stripped.isupper() and len(stripped.split()) <= 5 and 3 <= len(stripped) <= 40)


def _is_contact(line: str) -> bool:
    stripped = _normalize_line(line)
    return bool(stripped and _CONTACT_RE.search(stripped))


def _is_url(line: str) -> bool:
    stripped = _normalize_line(line)
    return bool(stripped and _URL_RE.search(stripped))


def _normalize_section_name(line: str) -> str:
    lowered = _normalize_line(line).lower().rstrip(":")
    if lowered in {"summary", "objective", "profile", "professional summary"}:
        return "objective"
    if lowered in {"experience", "work experience", "professional experience", "employment", "career history"}:
        return "experience"
    if lowered in {"skills", "technical skills", "core skills"}:
        return "skills"
    if lowered == "education":
        return "education"
    return "other"


def _bullet_text(line: str) -> str:
    return _BULLET_RE.sub("", line).strip()


def _line_kind(line: str, section: str, *, line_no: int) -> UnitType:
    stripped = _normalize_line(line)
    if not stripped:
        return "other"
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

    is_bullet = bool(_BULLET_RE.match(stripped))
    if section == "experience":
        if is_bullet or _ACTION_START_RE.match(stripped):
            return "experience_bullet"
        return "other"

    # If section could not be inferred, still classify obvious action bullets as experience.
    if is_bullet and (_ACTION_START_RE.match(_bullet_text(stripped)) or len(_bullet_text(stripped).split()) >= 6):
        return "experience_bullet"

    # Header lines are short identity/role lines near top, not prose.
    if (
        line_no <= 4
        and len(stripped.split()) <= 10
        and "." not in stripped
        and not _ACTION_START_RE.match(stripped)
    ):
        return "header"
    return "other"


def _make_unit(
    *,
    idx: int,
    unit_type: UnitType,
    text: str,
    line_start: int,
    line_end: int,
    doc_id: str | None,
) -> AnalysisUnit:
    snippet = _normalize_line(text)
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


def build_analysis_units(parsed_doc: ParsedDoc, normalized_resume: NormalizedResume | None = None) -> list[AnalysisUnit]:
    _ = normalized_resume
    lines = parsed_doc.text.splitlines()
    units: list[AnalysisUnit] = []
    current_section = "other"
    idx = 1
    pending: dict[str, int | str] | None = None

    def flush_pending() -> None:
        nonlocal pending, idx
        if pending is None:
            return
        units.append(
            _make_unit(
                idx=idx,
                unit_type=pending["unit_type"],  # type: ignore[arg-type]
                text=str(pending["text"]),
                line_start=int(pending["line_start"]),
                line_end=int(pending["line_end"]),
                doc_id=parsed_doc.doc_id,
            )
        )
        idx += 1
        pending = None

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = _normalize_line(raw_line)
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
        is_bullet = bool(_BULLET_RE.match(stripped))

        if kind == "experience_bullet":
            bullet_text = _bullet_text(stripped) or stripped
            if pending is not None:
                flush_pending()
            pending = {
                "unit_type": "experience_bullet",
                "text": bullet_text,
                "line_start": line_no,
                "line_end": line_no,
            }
            continue

        if pending is not None:
            # Stitch wrapped line fragments into previous bullet/paragraph.
            continuation = (
                kind in {"other", "objective", "skills", "education"}
                and not is_bullet
                and not _is_contact(stripped)
                and not _is_url(stripped)
                and len(stripped.split()) >= 2
                and (
                    pending["unit_type"] == "experience_bullet"
                    or stripped[0].islower()
                    or str(pending["text"]).endswith((",", ";", ":", "-", "/", "+", "("))
                )
            )
            if continuation:
                pending["text"] = f"{pending['text']} {stripped}"
                pending["line_end"] = line_no
                continue
            flush_pending()

        # Start paragraph-like pending units for prose areas to reduce wrapped-line fragmentation.
        if kind in {"objective", "other", "skills", "education"} and not _is_contact(stripped) and not _is_url(stripped):
            pending = {
                "unit_type": kind,
                "text": stripped,
                "line_start": line_no,
                "line_end": line_no,
            }
            continue

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
