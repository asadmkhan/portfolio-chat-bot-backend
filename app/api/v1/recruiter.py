from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.core.config import settings
from app.core.recruiter_share_store import create_recruiter_share, get_recruiter_share
from app.core.tools_rate_limit import ToolsRateLimitExceeded, enforce_tools_rate_limit
from app.schemas.recruiter import (
    ATSHumanRiskSplitRequest,
    ATSHumanRiskSplitResponse,
    ClaimVerificationRequest,
    ClaimVerificationResponse,
    HiringBiasRiskDetectorRequest,
    HiringBiasRiskDetectorResponse,
    JDMarketRealityRequest,
    JDMarketRealityResponse,
    JDQualityAnalyzerRequest,
    JDQualityResponse,
    RecruiterShareCreateRequest,
    RecruiterShareCreateResponse,
    RecruiterShareGetResponse,
    ResumeSignalStrengthRequest,
    ResumeSignalStrengthResponse,
    ResumeAuthenticityRequest,
    ResumeAuthenticityResponse,
    ResumeCompareRequest,
    ResumeCompareResponse,
    RoleSeniorityDefinitionRequest,
    RoleSeniorityDefinitionResponse,
    ShortlistJustificationRequest,
    ShortlistJustificationResponse,
)
from app.services.recruiter_service import (
    RecruiterQualityError,
    redact_share_payload,
    run_ats_vs_human,
    run_hiring_bias_risk_detector,
    run_jd_market_reality,
    run_claim_verification,
    run_jd_quality,
    run_resume_signal_strength,
    run_resume_authenticity,
    run_resume_compare,
    run_role_seniority_definition,
    run_shortlist_justification,
    validate_share_payload,
)

router = APIRouter()


def _client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if settings.trust_x_forwarded_for and forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_rate_limit(request: Request, limit: int, window_seconds: int = 60) -> None:
    client = _client_key(request)
    route = request.url.path
    try:
        enforce_tools_rate_limit(client_key=client, route_key=route, limit=limit, window_seconds=window_seconds)
    except ToolsRateLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait and try again.",
        ) from exc


def _raise_quality_error(exc: RecruiterQualityError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/recruiter/resume-authenticity", response_model=ResumeAuthenticityResponse)
async def recruiter_resume_authenticity(request: Request, payload: ResumeAuthenticityRequest):
    _enforce_rate_limit(request, limit=20)
    try:
        return run_resume_authenticity(payload)
    except RecruiterQualityError as exc:
        _raise_quality_error(exc)


@router.post("/recruiter/claim-verification", response_model=ClaimVerificationResponse)
async def recruiter_claim_verification(request: Request, payload: ClaimVerificationRequest):
    _enforce_rate_limit(request, limit=20)
    try:
        return run_claim_verification(payload)
    except RecruiterQualityError as exc:
        _raise_quality_error(exc)


@router.post("/recruiter/jd-quality", response_model=JDQualityResponse)
async def recruiter_jd_quality(request: Request, payload: JDQualityAnalyzerRequest):
    _enforce_rate_limit(request, limit=20)
    try:
        return run_jd_quality(payload)
    except RecruiterQualityError as exc:
        _raise_quality_error(exc)


@router.post("/recruiter/ats-vs-human", response_model=ATSHumanRiskSplitResponse)
async def recruiter_ats_vs_human(request: Request, payload: ATSHumanRiskSplitRequest):
    _enforce_rate_limit(request, limit=20)
    try:
        return run_ats_vs_human(payload)
    except RecruiterQualityError as exc:
        _raise_quality_error(exc)


@router.post("/recruiter/resume-compare", response_model=ResumeCompareResponse)
async def recruiter_resume_compare(request: Request, payload: ResumeCompareRequest):
    _enforce_rate_limit(request, limit=20)
    try:
        return run_resume_compare(payload)
    except RecruiterQualityError as exc:
        _raise_quality_error(exc)


@router.post("/recruiter/resume-signal-strength", response_model=ResumeSignalStrengthResponse)
async def recruiter_resume_signal_strength(request: Request, payload: ResumeSignalStrengthRequest):
    _enforce_rate_limit(request, limit=20)
    try:
        return run_resume_signal_strength(payload)
    except RecruiterQualityError as exc:
        _raise_quality_error(exc)


@router.post("/recruiter/jd-market-reality", response_model=JDMarketRealityResponse)
async def recruiter_jd_market_reality(request: Request, payload: JDMarketRealityRequest):
    _enforce_rate_limit(request, limit=20)
    try:
        return run_jd_market_reality(payload)
    except RecruiterQualityError as exc:
        _raise_quality_error(exc)


@router.post("/recruiter/role-seniority-definition", response_model=RoleSeniorityDefinitionResponse)
async def recruiter_role_seniority_definition(request: Request, payload: RoleSeniorityDefinitionRequest):
    _enforce_rate_limit(request, limit=20)
    try:
        return run_role_seniority_definition(payload)
    except RecruiterQualityError as exc:
        _raise_quality_error(exc)


@router.post("/recruiter/shortlist-justification", response_model=ShortlistJustificationResponse)
async def recruiter_shortlist_justification(request: Request, payload: ShortlistJustificationRequest):
    _enforce_rate_limit(request, limit=20)
    try:
        return run_shortlist_justification(payload)
    except RecruiterQualityError as exc:
        _raise_quality_error(exc)


@router.post("/recruiter/bias-risk-detector", response_model=HiringBiasRiskDetectorResponse)
async def recruiter_bias_risk_detector(request: Request, payload: HiringBiasRiskDetectorRequest):
    _enforce_rate_limit(request, limit=20)
    try:
        return run_hiring_bias_risk_detector(payload)
    except RecruiterQualityError as exc:
        _raise_quality_error(exc)


@router.post("/recruiter/share", response_model=RecruiterShareCreateResponse)
async def recruiter_create_share(request: Request, payload: RecruiterShareCreateRequest):
    _enforce_rate_limit(request, limit=30)
    try:
        validate_share_payload(payload)
        result_payload = payload.result_payload
        if settings.share_redact_sensitive_fields:
            result_payload = redact_share_payload(result_payload)
        share_id, expires_at = create_recruiter_share(
            tool_slug=payload.tool_slug,
            locale=payload.locale,
            result_payload=result_payload,
        )
        return RecruiterShareCreateResponse(share_id=share_id, expires_at=expires_at)
    except RecruiterQualityError as exc:
        _raise_quality_error(exc)


@router.get("/recruiter/share/{share_id}", response_model=RecruiterShareGetResponse)
async def recruiter_get_share(request: Request, share_id: str):
    _enforce_rate_limit(request, limit=60)
    record = get_recruiter_share(share_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Share link not found or expired.")
    return RecruiterShareGetResponse(
        tool_slug=record["tool_slug"],
        locale=record["locale"],
        result_payload=record["result_payload"],
        created_at=record["created_at"],
        expires_at=record["expires_at"],
    )
