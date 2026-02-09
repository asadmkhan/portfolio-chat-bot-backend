from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.core.recruiter_share_store import create_recruiter_share, get_recruiter_share
from app.core.tools_rate_limit import ToolsRateLimitExceeded, enforce_tools_rate_limit
from app.schemas.recruiter import (
    ATSHumanRiskSplitRequest,
    ATSHumanRiskSplitResponse,
    ClaimVerificationRequest,
    ClaimVerificationResponse,
    JDQualityAnalyzerRequest,
    JDQualityResponse,
    RecruiterShareCreateRequest,
    RecruiterShareCreateResponse,
    RecruiterShareGetResponse,
    ResumeAuthenticityRequest,
    ResumeAuthenticityResponse,
    ResumeCompareRequest,
    ResumeCompareResponse,
)
from app.services.recruiter_service import (
    RecruiterQualityError,
    run_ats_vs_human,
    run_claim_verification,
    run_jd_quality,
    run_resume_authenticity,
    run_resume_compare,
    validate_share_payload,
)

router = APIRouter()


def _client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
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


@router.post("/recruiter/share", response_model=RecruiterShareCreateResponse)
async def recruiter_create_share(request: Request, payload: RecruiterShareCreateRequest):
    _enforce_rate_limit(request, limit=30)
    try:
        validate_share_payload(payload)
        share_id, expires_at = create_recruiter_share(
            tool_slug=payload.tool_slug,
            locale=payload.locale,
            result_payload=payload.result_payload,
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

