from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.normalized import NormalizedJD, NormalizedResume

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tech": ("python", "java", "api", "backend", "frontend", "aws", "sql", "docker", "kubernetes"),
    "sales": ("sales", "quota", "pipeline", "crm", "prospecting", "deal", "revenue", "account executive"),
    "marketing": ("marketing", "seo", "campaign", "brand", "content", "funnel", "growth"),
    "finance": ("finance", "financial", "reporting", "budget", "forecast", "audit", "gaap", "fp&a"),
    "hr": ("hr", "recruiting", "talent", "onboarding", "benefits", "payroll", "hris"),
    "healthcare": ("healthcare", "patient", "clinical", "emr", "ehr", "nursing", "medical", "hospital"),
}


class DomainClassification(BaseModel):
    domain_primary: str
    confidence: float = Field(ge=0.0, le=1.0)


def _domain_scores(text: str) -> dict[str, int]:
    lowered = text.lower()
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            score += lowered.count(keyword)
        scores[domain] = score
    return scores


def classify_domain(text: str) -> DomainClassification:
    scores = _domain_scores(text)
    total_hits = sum(scores.values())
    if total_hits <= 0:
        return DomainClassification(domain_primary="other", confidence=0.4)

    primary_domain = max(scores, key=lambda key: scores[key])
    primary_hits = scores[primary_domain]
    confidence = primary_hits / total_hits if total_hits else 0.4
    if primary_hits <= 0:
        return DomainClassification(domain_primary="other", confidence=0.4)
    return DomainClassification(domain_primary=primary_domain, confidence=confidence)


def classify_domain_from_jd(jd: NormalizedJD) -> DomainClassification:
    chunks = []
    if jd.title:
        chunks.append(jd.title)
    chunks.extend(jd.responsibilities)
    chunks.extend(req.text for req in jd.requirements)
    return classify_domain("\n".join(chunks))


def classify_domain_from_resume(resume: NormalizedResume) -> DomainClassification:
    chunks = [claim.text for claim in resume.claims]
    chunks.extend(resume.skills)
    return classify_domain("\n".join(chunks))

