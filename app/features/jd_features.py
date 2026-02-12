from __future__ import annotations

from pydantic import BaseModel

from app.schemas.normalized import NormalizedJD

_VAGUE_WORDS = (
    "various",
    "etc",
    "dynamic",
    "fast-paced",
    "other duties",
    "as needed",
)


class JDFeatures(BaseModel):
    vagueness_density: float
    cluster_breadth: float


def build_jd_features(jd: NormalizedJD) -> JDFeatures:
    lines = [req.text for req in jd.requirements] + list(jd.responsibilities)
    if not lines:
        vagueness_density = 0.0
    else:
        vague_hits = 0
        for line in lines:
            lowered = line.lower()
            if any(word in lowered for word in _VAGUE_WORDS):
                vague_hits += 1
        vagueness_density = vague_hits / len(lines)

    return JDFeatures(vagueness_density=vagueness_density, cluster_breadth=0.0)

