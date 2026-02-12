from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ParsedBlock(BaseModel):
    page: int | None = None
    bbox: list[float] | None = None
    text: str


class ParsedDoc(BaseModel):
    doc_id: str
    source_type: str
    language: str | None = None
    text: str
    blocks: list[ParsedBlock] = Field(default_factory=list)
    parsing_warnings: list[str] = Field(default_factory=list)
    layout_flags: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_type")
    @classmethod
    def _validate_source_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"pdf", "docx", "txt"}:
            raise ValueError("source_type must be one of: pdf, docx, txt")
        return normalized

