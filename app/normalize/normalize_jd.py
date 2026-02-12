from __future__ import annotations

from app.parsing.models import ParsedDoc
from app.schemas.normalized import EvidenceSpan, JDRequirement, NormalizedJD

from .utils import contains_any, enumerate_lines, strip_bullet_prefix

_MUST_MARKERS = ("must", "required", "you have")
_NICE_MARKERS = ("nice", "preferred", "plus")
_REQUIREMENT_HEADERS = ("requirements", "requirement", "qualifications")
_RESPONSIBILITY_HEADERS = ("responsibilities", "what you'll do", "what you will do")


def _normalize_header(line: str) -> str:
    return line.strip().lower().rstrip(":")


def _classify_priority(line: str, in_requirements_section: bool) -> str | None:
    if contains_any(line, _NICE_MARKERS):
        return "nice"
    if in_requirements_section or contains_any(line, _MUST_MARKERS):
        return "must"
    return None


def normalize_jd(parsed: ParsedDoc) -> NormalizedJD:
    title: str | None = None
    requirements: list[JDRequirement] = []
    responsibilities: list[str] = []
    in_requirements_section = False
    in_responsibilities_section = False

    for line_no, raw_line in enumerate_lines(parsed.text):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if title is None:
            title = stripped

        header = _normalize_header(stripped)
        if any(header.startswith(item) for item in _REQUIREMENT_HEADERS):
            in_requirements_section = True
            in_responsibilities_section = False
            continue
        if any(header.startswith(item) for item in _RESPONSIBILITY_HEADERS):
            in_requirements_section = False
            in_responsibilities_section = True
            continue

        if in_responsibilities_section:
            responsibilities.append(strip_bullet_prefix(stripped))

        priority = _classify_priority(stripped, in_requirements_section=in_requirements_section)
        if priority is None:
            continue

        requirements.append(
            JDRequirement(
                req_id=f"r{len(requirements) + 1}",
                text=strip_bullet_prefix(stripped),
                priority=priority,
                evidence=EvidenceSpan(
                    doc_id=parsed.doc_id,
                    page=None,
                    line_start=line_no,
                    line_end=line_no,
                    bbox=None,
                    text_snippet=stripped,
                ),
            )
        )

    return NormalizedJD(
        source_language=parsed.language,
        title=title,
        responsibilities=responsibilities,
        requirements=requirements,
    )

