from __future__ import annotations

from app.parsing.models import ParsedDoc
from app.schemas.normalized import EvidenceSpan, NormalizedResume, ResumeClaim

from .utils import (
    enumerate_lines,
    is_bullet_like,
    is_claim_like,
    is_contact_or_url,
    is_continuation_line,
    is_section_heading,
    normalize_line,
    strip_bullet_prefix,
)

_EXPERIENCE_SECTION_HINTS = {"experience", "work experience", "employment history", "projects", "professional experience"}


def _section_key(line: str) -> str:
    lowered = normalize_line(line).lower().rstrip(":")
    if lowered in {"summary", "objective", "profile"}:
        return "summary"
    if lowered in _EXPERIENCE_SECTION_HINTS:
        return "experience"
    if lowered == "skills":
        return "skills"
    if lowered == "education":
        return "education"
    return "other"


def _build_claim(
    *,
    index: int,
    text: str,
    doc_id: str,
    line_start: int,
    line_end: int,
    snippet: str,
) -> ResumeClaim:
    return ResumeClaim(
        claim_id=f"c{index}",
        text=text,
        evidence=EvidenceSpan(
            doc_id=doc_id,
            page=None,
            line_start=line_start,
            line_end=line_end,
            bbox=None,
            text_snippet=snippet[:400],
        ),
    )


def normalize_resume(parsed: ParsedDoc) -> NormalizedResume:
    claims: list[ResumeClaim] = []
    current_section = "other"
    pending_claim: dict[str, int | str] | None = None

    def flush_pending() -> None:
        nonlocal pending_claim
        if pending_claim is None:
            return
        claim_text = normalize_line(str(pending_claim["text"]))
        if claim_text:
            claims.append(
                _build_claim(
                    index=len(claims) + 1,
                    text=claim_text,
                    doc_id=parsed.doc_id,
                    line_start=int(pending_claim["line_start"]),
                    line_end=int(pending_claim["line_end"]),
                    snippet=str(pending_claim["snippet"]),
                )
            )
        pending_claim = None

    for line_no, raw_line in enumerate_lines(parsed.text):
        stripped = normalize_line(raw_line)
        if not stripped:
            flush_pending()
            continue

        if is_section_heading(stripped):
            flush_pending()
            current_section = _section_key(stripped)
            continue

        if is_contact_or_url(stripped):
            flush_pending()
            continue

        in_experience = current_section == "experience"
        bullet_line = is_bullet_like(raw_line) or is_bullet_like(stripped)
        cleaned = strip_bullet_prefix(stripped) if bullet_line else stripped
        claim_line = bullet_line or is_claim_like(cleaned, in_experience_section=in_experience)

        if claim_line:
            if pending_claim is not None and not bullet_line and is_continuation_line(stripped):
                pending_claim["text"] = f"{pending_claim['text']} {cleaned}"
                pending_claim["line_end"] = line_no
                pending_claim["snippet"] = f"{pending_claim['snippet']} {stripped}"
                continue
            if pending_claim is not None:
                flush_pending()
            pending_claim = {
                "text": cleaned,
                "line_start": line_no,
                "line_end": line_no,
                "snippet": stripped,
            }
            continue

        if pending_claim is not None and is_continuation_line(stripped):
            pending_claim["text"] = f"{pending_claim['text']} {stripped}"
            pending_claim["line_end"] = line_no
            pending_claim["snippet"] = f"{pending_claim['snippet']} {stripped}"
            continue

        flush_pending()

    flush_pending()

    # Deterministic fallback for dense PDFs with no explicit bullets:
    # keep meaningful non-contact prose lines as claims to avoid 0-claim regressions.
    if not claims:
        for line_no, raw_line in enumerate_lines(parsed.text):
            stripped = normalize_line(raw_line)
            if not stripped or is_contact_or_url(stripped) or is_section_heading(stripped):
                continue
            words = stripped.split()
            if len(words) < 8:
                continue
            claims.append(
                _build_claim(
                    index=len(claims) + 1,
                    text=stripped,
                    doc_id=parsed.doc_id,
                    line_start=line_no,
                    line_end=line_no,
                    snippet=stripped,
                )
            )
            if len(claims) >= 20:
                break

    return NormalizedResume(
        source_language=parsed.language,
        claims=claims,
    )
