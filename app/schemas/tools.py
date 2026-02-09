from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

LocaleCode = Literal["en", "de", "es", "fr", "it", "ar"]
TargetRegion = Literal["US", "EU", "UK", "Other"]
Seniority = Literal["junior", "mid", "senior", "lead", "career-switcher"]
Recommendation = Literal["apply", "fix", "skip"]
RiskType = Literal["hard_filter", "keyword_gap", "parsing", "seniority", "evidence_gap"]
Severity = Literal["low", "medium", "high"]
SummarizerSource = Literal["text", "word", "pdf", "ppt", "youtube", "video", "image"]
SummarizerMode = Literal["summary", "key_points", "action_items"]
SummarizerLength = Literal["short", "medium", "long"]
VpnToolVerdict = Literal["good", "attention", "critical"]
VpnCardStatus = Literal["pass", "warn", "fail", "info"]
ResumeLayoutType = Literal["single_column", "multi_column", "hybrid", "unknown"]
ResumeLayoutSourceType = Literal["pdf", "word", "text", "image", "unknown"]


class ResumeLayoutProfile(BaseModel):
    detected_layout: ResumeLayoutType = "unknown"
    column_count: int = Field(default=1, ge=1, le=4)
    confidence: float = Field(default=0.2, ge=0.0, le=1.0)
    table_count: int = Field(default=0, ge=0, le=200)
    header_link_density: float = Field(default=0.0, ge=0.0, le=1.0)
    complexity_score: int = Field(default=20, ge=0, le=100)
    source_type: ResumeLayoutSourceType = "unknown"
    signals: list[str] = Field(default_factory=list, max_length=40)


class ResumeFileMeta(BaseModel):
    filename: str = Field(default="", max_length=255)
    extension: str = Field(default="", max_length=20)
    source_type: str = Field(default="unknown", max_length=30)


class CandidateProfile(BaseModel):
    target_region: TargetRegion = "Other"
    seniority: Seniority = "mid"


class ToolRequest(BaseModel):
    locale: LocaleCode = "en"
    resume_text: str = Field(default="", min_length=1, max_length=50000)
    job_description_text: str = Field(default="", min_length=1, max_length=50000)
    candidate_profile: CandidateProfile = Field(default_factory=CandidateProfile)
    session_id: str = Field(min_length=8, max_length=200)
    tool_inputs: dict[str, Any] = Field(default_factory=dict)
    resume_layout_profile: ResumeLayoutProfile | None = None
    resume_file_meta: ResumeFileMeta | None = None


class RiskItem(BaseModel):
    type: RiskType
    severity: Severity
    message: str


class FixPlanItem(BaseModel):
    id: str
    title: str
    impact_score: int = Field(ge=0, le=100)
    effort_minutes: int = Field(ge=1, le=240)
    reason: str


class ScoreCard(BaseModel):
    job_match: int = Field(ge=0, le=100)
    ats_readability: int = Field(ge=0, le=100)


class ToolResponse(BaseModel):
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    scores: ScoreCard
    risks: list[RiskItem]
    fix_plan: list[FixPlanItem]
    generated_at: datetime
    details: dict[str, Any] = Field(default_factory=dict)


class LeadCaptureRequest(BaseModel):
    locale: LocaleCode = "en"
    session_id: str = Field(min_length=8, max_length=200)
    email: str = Field(min_length=5, max_length=320)
    tool: str = Field(min_length=2, max_length=100)
    consent: bool = True


class LeadCaptureResponse(BaseModel):
    status: Literal["ok"]
    message: str


class ExtractTextResponse(BaseModel):
    filename: str
    source_type: SummarizerSource
    text: str
    characters: int = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class ExtractJobRequest(BaseModel):
    locale: LocaleCode = "en"
    session_id: str = Field(min_length=8, max_length=200)
    job_url: str = Field(min_length=8, max_length=3000)


class ExtractJobResponse(BaseModel):
    job_url: str
    normalized_url: str
    domain: str
    title: str = ""
    company: str = ""
    location: str = ""
    job_description_text: str = ""
    characters: int = Field(ge=0)
    extraction_mode: Literal["json_ld", "domain_parser", "readability"]
    warnings: list[str] = Field(default_factory=list)
    blocked: bool = False


class ExtractResumeUrlRequest(BaseModel):
    locale: LocaleCode = "en"
    session_id: str = Field(min_length=8, max_length=200)
    resume_url: str = Field(min_length=8, max_length=3000)


class ExtractResumeUrlResponse(BaseModel):
    resume_url: str
    normalized_url: str
    domain: str
    filename: str
    content_type: str
    resume_text: str
    characters: int = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
    blocked: bool = False
    warnings: list[str] = Field(default_factory=list)
    content_base64: str | None = None


class SummarizerRequest(BaseModel):
    locale: LocaleCode = "en"
    source_type: SummarizerSource = "text"
    content: str = Field(default="", max_length=300000)
    source_url: str | None = Field(default=None, max_length=3000)
    mode: SummarizerMode = "summary"
    output_language: str = Field(default="default", max_length=50)
    length: SummarizerLength = "medium"
    session_id: str = Field(min_length=8, max_length=200)


class SummarizerResponse(BaseModel):
    summary: str
    key_points: list[str]
    action_items: list[str]
    word_count_in: int = Field(ge=0)
    word_count_out: int = Field(ge=0)
    generated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class VpnToolRequest(BaseModel):
    locale: LocaleCode = "en"
    session_id: str = Field(min_length=8, max_length=200)
    input: dict[str, Any] = Field(default_factory=dict)


class VpnToolCard(BaseModel):
    title: str
    status: VpnCardStatus
    value: str
    detail: str = ""


class VpnToolResponse(BaseModel):
    tool: str
    headline: str
    verdict: VpnToolVerdict
    score: int = Field(ge=0, le=100)
    cards: list[VpnToolCard] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime


class VpnProbeEnrichRequest(BaseModel):
    locale: LocaleCode = "en"
    session_id: str = Field(min_length=8, max_length=200)
    public_ip: str | None = Field(default=None, max_length=120)
    baseline_public_ip: str | None = Field(default=None, max_length=120)
    webrtc_ips: list[str] = Field(default_factory=list, max_length=40)
    baseline_webrtc_ips: list[str] = Field(default_factory=list, max_length=40)
    dns_resolver_ips: list[str] = Field(default_factory=list, max_length=40)
    expected_country: str | None = Field(default=None, max_length=120)


class VpnProbeGeoRecord(BaseModel):
    ip: str
    country: str | None = None
    country_code: str | None = None
    region: str | None = None
    city: str | None = None
    isp: str | None = None
    is_private: bool = False
    source: str = "unknown"


class VpnProbeEnrichResponse(BaseModel):
    records: list[VpnProbeGeoRecord] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime
