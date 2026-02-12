from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .evidence import EvidenceSpan


class ResumeClaim(BaseModel):
    claim_id: str
    text: str
    evidence: EvidenceSpan
    extracted_skills: list[str] = Field(default_factory=list)
    canonical_skill_ids: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    ownership_markers: list[str] = Field(default_factory=list)
    scope_markers: list[str] = Field(default_factory=list)


class NormalizedResume(BaseModel):
    source_language: str | None = None
    profile: dict[str, Any] = Field(default_factory=dict)
    experience: list[dict[str, Any]] = Field(default_factory=list)
    education: list[dict[str, Any]] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    claims: list[ResumeClaim]

