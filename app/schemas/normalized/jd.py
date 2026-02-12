from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .evidence import EvidenceSpan


class JDRequirement(BaseModel):
    req_id: str
    text: str
    priority: str
    evidence: EvidenceSpan
    extracted_skills: list[str] = Field(default_factory=list)
    canonical_skill_ids: list[str] = Field(default_factory=list)

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"must", "nice"}:
            raise ValueError("priority must be 'must' or 'nice'")
        return normalized


class NormalizedJD(BaseModel):
    source_language: str | None = None
    title: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[JDRequirement]

