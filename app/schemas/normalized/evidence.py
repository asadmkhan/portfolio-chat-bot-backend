from __future__ import annotations

from pydantic import BaseModel


class EvidenceSpan(BaseModel):
    doc_id: str | None = None
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    bbox: list[float] | None = None
    text_snippet: str | None = None

