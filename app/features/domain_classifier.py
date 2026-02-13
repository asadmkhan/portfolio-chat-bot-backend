from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from app.core.config.scoring import get_scoring_value
from app.schemas.normalized import NormalizedJD, NormalizedResume

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tech": ("python", "java", "api", "backend", "frontend", "aws", "sql", "docker", "kubernetes", ".net", "sap"),
    "sales": ("sales", "quota", "pipeline", "crm", "prospecting", "deal", "revenue", "account executive"),
    "marketing": ("marketing", "seo", "campaign", "brand", "content", "funnel", "growth", "copywriting"),
    "finance": ("finance", "financial", "reporting", "budget", "forecast", "audit", "gaap", "fp&a"),
    "hr": ("hr", "recruiting", "talent", "onboarding", "benefits", "payroll", "hris"),
    "healthcare": ("healthcare", "patient", "clinical", "emr", "ehr", "nursing", "medical", "hospital"),
}


class DomainClassification(BaseModel):
    domain_primary: str
    domain_secondary: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_units: list[str] = Field(default_factory=list)
    using_general_expectations: bool = False


def _domain_scores(text: str) -> dict[str, int]:
    lowered = (text or "").lower()
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            score += lowered.count(keyword)
        scores[domain] = score
    return scores


def classify_domain(text: str, *, evidence_units: list[str] | None = None) -> DomainClassification:
    scores = _domain_scores(text)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    total_hits = sum(scores.values())
    if total_hits <= 0 or not ranked:
        return DomainClassification(
            domain_primary="other",
            domain_secondary=None,
            confidence=0.35,
            evidence_units=(evidence_units or [])[:5],
            using_general_expectations=True,
        )

    primary_domain, primary_hits = ranked[0]
    secondary_domain = ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0 else None
    confidence = primary_hits / total_hits if total_hits else 0.35
    low_conf_threshold = float(get_scoring_value("domains.classifier.low_confidence_threshold", 0.55))
    use_general = confidence < low_conf_threshold
    return DomainClassification(
        domain_primary=primary_domain,
        domain_secondary=secondary_domain,
        confidence=confidence,
        evidence_units=(evidence_units or [])[:5],
        using_general_expectations=use_general,
    )


def classify_domain_from_jd(jd: NormalizedJD) -> DomainClassification:
    chunks: list[str] = []
    if jd.title:
        chunks.append(jd.title)
    chunks.extend(jd.responsibilities)
    chunks.extend(req.text for req in jd.requirements)
    return classify_domain("\n".join(chunks), evidence_units=chunks[:5])


def classify_domain_from_resume(
    resume: NormalizedResume,
    *,
    analysis_units: list[Any] | None = None,
) -> DomainClassification:
    chunks: list[str] = [claim.text for claim in resume.claims if claim.text]
    chunks.extend(resume.skills)
    if analysis_units:
        for unit in analysis_units:
            unit_text = getattr(unit, "text", "")
            if unit_text:
                chunks.append(str(unit_text))
    return classify_domain("\n".join(chunks), evidence_units=chunks[:5])

