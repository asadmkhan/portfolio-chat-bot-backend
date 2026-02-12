from __future__ import annotations

import re

from pydantic import BaseModel

from app.schemas.normalized import NormalizedResume

_NUMBER_PATTERN = re.compile(r"\d")


class ResumeFeatures(BaseModel):
    metric_density: float
    repetition_similarity: float
    generic_phrase_density: float


def build_resume_features(resume: NormalizedResume) -> ResumeFeatures:
    claim_count = len(resume.claims)
    if claim_count == 0:
        metric_density = 0.0
    else:
        metric_claims = sum(1 for claim in resume.claims if _NUMBER_PATTERN.search(claim.text))
        metric_density = metric_claims / claim_count

    return ResumeFeatures(
        metric_density=metric_density,
        repetition_similarity=0.0,
        generic_phrase_density=0.0,
    )

