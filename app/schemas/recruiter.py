from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

LocaleCode = Literal["en", "de", "es", "fr", "it", "ar"]
RiskLevel = Literal["Low", "Medium", "High"]
FitLevel = Literal["Strong", "Moderate", "Weak"]
RatingLevel = Literal["Clear", "Risky", "Problematic"]
ImpactLevel = Literal["Low", "Med", "High"]
RoleLevelInference = Literal["Junior", "Mid", "Senior", "Lead", "Staff", "Unclear"]


class RecruiterBaseRequest(BaseModel):
    locale: LocaleCode = "en"
    session_id: str = Field(min_length=8, max_length=200)


class ResumeAuthenticityRequest(RecruiterBaseRequest):
    resume_text: str = Field(min_length=30, max_length=120000)
    jd_text: str = Field(default="", max_length=120000)


class ResumeAuthenticitySignal(BaseModel):
    name: str
    severity: RiskLevel
    explanation: str
    examples: list[str] = Field(default_factory=list, max_length=3)
    suggested_fix: str


class ResumeAuthenticityResponse(BaseModel):
    risk_level: RiskLevel
    overall_summary: str
    signals: list[ResumeAuthenticitySignal] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)
    generated_at: datetime


class ClaimVerificationRequest(RecruiterBaseRequest):
    resume_text: str = Field(min_length=30, max_length=120000)
    jd_text: str = Field(default="", max_length=120000)


class VerificationClaim(BaseModel):
    claim_text: str
    why_verify: str
    questions: list[str] = Field(default_factory=list, min_length=3, max_length=5)
    strong_answer_signals: list[str] = Field(default_factory=list, min_length=3, max_length=6)
    weak_answer_signals: list[str] = Field(default_factory=list, min_length=3, max_length=6)


class ClaimVerificationResponse(BaseModel):
    summary: str
    claims: list[VerificationClaim] = Field(default_factory=list)
    red_flag_follow_ups: list[str] = Field(default_factory=list)
    generated_at: datetime


class JDQualityAnalyzerRequest(RecruiterBaseRequest):
    jd_text: str = Field(min_length=30, max_length=120000)


class JDQualityIssue(BaseModel):
    category: str
    severity: RiskLevel
    evidence: str
    why_it_hurts: str
    improvement_suggestion: str


class JDQualityResponse(BaseModel):
    rating: RatingLevel
    summary: str
    issues: list[JDQualityIssue] = Field(default_factory=list)
    role_level_inference: RoleLevelInference
    role_level_reasoning: str
    generated_at: datetime


class ATSHumanRiskSplitRequest(RecruiterBaseRequest):
    resume_text: str = Field(min_length=30, max_length=120000)
    jd_text: str = Field(default="", max_length=120000)


class RiskBlock(BaseModel):
    level: RiskLevel
    explanation: str
    top_drivers: list[str] = Field(default_factory=list, min_length=1, max_length=5)


class QuickWin(BaseModel):
    item: str
    effort_minutes: int = Field(ge=5, le=240)
    impact: ImpactLevel


class ATSHumanRiskSplitResponse(BaseModel):
    ats_risk: RiskBlock
    human_risk: RiskBlock
    quick_wins: list[QuickWin] = Field(default_factory=list)
    where_to_focus: str
    hard_filter_gaps: list[str] = Field(default_factory=list, max_length=5)
    generated_at: datetime


class ResumeCompareCandidateInput(BaseModel):
    candidate_label: str = Field(min_length=1, max_length=30)
    resume_text: str = Field(min_length=30, max_length=120000)


class ResumeCompareRequest(RecruiterBaseRequest):
    jd_text: str = Field(min_length=30, max_length=120000)
    resumes: list[ResumeCompareCandidateInput] = Field(min_length=2, max_length=3)


class ResumeCompareCandidateOutput(BaseModel):
    label: str
    fit_level: FitLevel
    ats_risk: RiskLevel
    human_risk: RiskLevel
    strengths: list[str] = Field(default_factory=list, max_length=5)
    risks: list[str] = Field(default_factory=list, max_length=5)
    interview_focus: list[str] = Field(default_factory=list, max_length=5)


class ResumeCompareResponse(BaseModel):
    comparison_summary: str
    candidates: list[ResumeCompareCandidateOutput] = Field(default_factory=list)
    recommendation: str
    decision_log: str
    generated_at: datetime


class RecruiterShareCreateRequest(BaseModel):
    tool_slug: str = Field(min_length=2, max_length=80)
    locale: LocaleCode = "en"
    result_payload: dict[str, Any] = Field(default_factory=dict)


class RecruiterShareCreateResponse(BaseModel):
    share_id: str
    expires_at: datetime


class RecruiterShareGetResponse(BaseModel):
    tool_slug: str
    locale: LocaleCode
    result_payload: dict[str, Any]
    created_at: datetime
    expires_at: datetime

