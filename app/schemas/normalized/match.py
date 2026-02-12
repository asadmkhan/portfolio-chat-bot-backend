from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MatchHit(BaseModel):
    claim_id: str
    similarity: float
    evidence_strength: int

    @field_validator("evidence_strength")
    @classmethod
    def _validate_evidence_strength(cls, value: int) -> int:
        if value < 0 or value > 3:
            raise ValueError("evidence_strength must be between 0 and 3")
        return value


class MatchMatrix(BaseModel):
    matches: dict[str, list[MatchHit]] = Field(default_factory=dict)
    must_req_ids: list[str] = Field(default_factory=list)
    nice_req_ids: list[str] = Field(default_factory=list)

