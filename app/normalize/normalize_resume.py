from __future__ import annotations

from app.parsing.models import ParsedDoc
from app.schemas.normalized import EvidenceSpan, NormalizedResume, ResumeClaim

from .utils import enumerate_lines, is_bullet_like, strip_bullet_prefix


def normalize_resume(parsed: ParsedDoc) -> NormalizedResume:
    claims: list[ResumeClaim] = []

    for line_no, raw_line in enumerate_lines(parsed.text):
        if not is_bullet_like(raw_line):
            continue
        cleaned = strip_bullet_prefix(raw_line)
        if not cleaned:
            continue
        claims.append(
            ResumeClaim(
                claim_id=f"c{len(claims) + 1}",
                text=cleaned,
                evidence=EvidenceSpan(
                    doc_id=parsed.doc_id,
                    page=None,
                    line_start=line_no,
                    line_end=line_no,
                    bbox=None,
                    text_snippet=raw_line.strip(),
                ),
            )
        )

    return NormalizedResume(
        source_language=parsed.language,
        claims=claims,
    )

