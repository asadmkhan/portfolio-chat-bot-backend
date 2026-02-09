import asyncio
import json
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.core.tools_rate_limit import ToolsRateLimitExceeded, enforce_tools_rate_limit
from app.schemas.tools import (
    ExtractJobRequest,
    ExtractJobResponse,
    ExtractResumeUrlRequest,
    ExtractResumeUrlResponse,
    ExtractTextResponse,
    LeadCaptureRequest,
    LeadCaptureResponse,
    SummarizerRequest,
    SummarizerResponse,
    ToolRequest,
    ToolResponse,
    VpnProbeEnrichRequest,
    VpnProbeEnrichResponse,
    VpnToolRequest,
    VpnToolResponse,
)
from app.services.tools_service import (
    ADDITIONAL_TOOL_SLUGS,
    QualityEnforcementError,
    SUMMARIZER_TOOL_SLUGS,
    VPN_TOOL_SLUGS,
    extract_job_from_url,
    extract_resume_from_url,
    extract_text_from_file,
    run_additional_tool,
    run_ats_checker,
    run_cover_letter,
    run_interview_predictor,
    run_job_match,
    run_missing_keywords,
    run_vpn_probe_enrich,
    run_summarizer,
    run_vpn_tool,
    save_lead,
)
from app.services.tools_llm import ToolsLLMError

router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {
    "txt", "md", "rtf", "pdf", "docx", "doc",
    "pptx", "ppt", "png", "jpg", "jpeg", "webp",
    "gif", "bmp", "mp4", "mov", "avi", "mkv", "webm", "m4v",
}


def _client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_tools_rate_limit(request: Request, limit: int, window_seconds: int = 60) -> None:
    client = _client_key(request)
    route = request.url.path
    try:
        enforce_tools_rate_limit(client_key=client, route_key=route, limit=limit, window_seconds=window_seconds)
    except ToolsRateLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait a minute and try again.",
        ) from exc


def _raise_quality_http_error(exc: Exception) -> None:
    if isinstance(exc, QualityEnforcementError):
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if isinstance(exc, ToolsLLMError):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    raise exc


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


@router.post("/tools/job-match", response_model=ToolResponse)
async def tools_job_match(request: Request, payload: ToolRequest):
    _enforce_tools_rate_limit(request, limit=20)
    _ = request
    try:
        return run_job_match(payload)
    except (QualityEnforcementError, ToolsLLMError) as exc:
        _raise_quality_http_error(exc)


@router.post("/tools/missing-keywords", response_model=ToolResponse)
async def tools_missing_keywords(request: Request, payload: ToolRequest):
    _enforce_tools_rate_limit(request, limit=20)
    _ = request
    try:
        return run_missing_keywords(payload)
    except (QualityEnforcementError, ToolsLLMError) as exc:
        _raise_quality_http_error(exc)


@router.post("/tools/ats-checker", response_model=ToolResponse)
async def tools_ats_checker(request: Request, payload: ToolRequest):
    _enforce_tools_rate_limit(request, limit=20)
    _ = request
    try:
        return run_ats_checker(payload)
    except (QualityEnforcementError, ToolsLLMError) as exc:
        _raise_quality_http_error(exc)


@router.post("/tools/ats-checker/stream")
async def tools_ats_checker_stream(request: Request, payload: ToolRequest):
    _enforce_tools_rate_limit(request, limit=20)

    async def event_stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def push_progress(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, {"kind": "progress", "payload": event})

        def worker() -> None:
            try:
                result = run_ats_checker(payload, progress_callback=push_progress)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"kind": "result", "payload": result.model_dump(mode="json")},
                )
            except QualityEnforcementError as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "kind": "error",
                        "payload": {"message": str(exc), "status": exc.status_code},
                    },
                )
            except ToolsLLMError as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "kind": "error",
                        "payload": {
                            "message": str(exc),
                            "status": status.HTTP_503_SERVICE_UNAVAILABLE,
                        },
                    },
                )
            except Exception as exc:  # pragma: no cover - guard rail
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "kind": "error",
                        "payload": {"message": str(exc), "status": status.HTTP_500_INTERNAL_SERVER_ERROR},
                    },
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, {"kind": "done", "payload": {}})

        task = asyncio.create_task(asyncio.to_thread(worker))

        try:
            yield _sse_event("connected", {"ok": True})
            while True:
                if await request.is_disconnected():
                    break
                event = await queue.get()
                kind = event.get("kind")
                payload_data = event.get("payload", {})
                if kind == "progress":
                    yield _sse_event("progress", payload_data)
                    continue
                if kind == "result":
                    yield _sse_event("result", payload_data)
                    continue
                if kind == "error":
                    yield _sse_event("error", payload_data)
                    continue
                if kind == "done":
                    break
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/tools/cover-letter", response_model=ToolResponse)
async def tools_cover_letter(request: Request, payload: ToolRequest):
    _enforce_tools_rate_limit(request, limit=20)
    _ = request
    try:
        return run_cover_letter(payload)
    except (QualityEnforcementError, ToolsLLMError) as exc:
        _raise_quality_http_error(exc)


@router.post("/tools/interview-predictor", response_model=ToolResponse)
async def tools_interview_predictor(request: Request, payload: ToolRequest):
    _enforce_tools_rate_limit(request, limit=20)
    _ = request
    try:
        return run_interview_predictor(payload)
    except (QualityEnforcementError, ToolsLLMError) as exc:
        _raise_quality_http_error(exc)


@router.post("/tools/lead-capture", response_model=LeadCaptureResponse)
async def tools_lead_capture(request: Request, payload: LeadCaptureRequest):
    _enforce_tools_rate_limit(request, limit=10)
    _ = request
    message = save_lead(payload)
    return LeadCaptureResponse(status="ok", message=message)


@router.post("/tools/extract-text", response_model=ExtractTextResponse)
async def tools_extract_text(request: Request, file: UploadFile = File(...)):
    _enforce_tools_rate_limit(request, limit=20)
    filename = file.filename or "uploaded-file"

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 64)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)
    payload = b"".join(chunks)

    try:
        return extract_text_from_file(filename=filename, content=payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/tools/extract-job", response_model=ExtractJobResponse)
async def tools_extract_job(request: Request, payload: ExtractJobRequest):
    _enforce_tools_rate_limit(request, limit=20)
    try:
        return extract_job_from_url(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/tools/extract-resume-url", response_model=ExtractResumeUrlResponse)
async def tools_extract_resume_url(request: Request, payload: ExtractResumeUrlRequest):
    _enforce_tools_rate_limit(request, limit=20)
    try:
        return extract_resume_from_url(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/tools/summarize", response_model=SummarizerResponse)
async def tools_summarize(request: Request, payload: SummarizerRequest):
    _enforce_tools_rate_limit(request, limit=30)
    try:
        return run_summarizer(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/tools/{tool_slug}", response_model=ToolResponse)
async def tools_additional(request: Request, tool_slug: str, payload: ToolRequest):
    if tool_slug in {
        "job-match",
        "missing-keywords",
        "ats-checker",
        "cover-letter",
        "interview-predictor",
        "lead-capture",
        "extract-text",
        "extract-job",
        "extract-resume-url",
        "summarize",
    }:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Use dedicated endpoint for this tool.")

    if tool_slug in SUMMARIZER_TOOL_SLUGS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Use /v1/tools/summarize for summarizer tools.")

    if tool_slug not in ADDITIONAL_TOOL_SLUGS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown tool.")

    _enforce_tools_rate_limit(request, limit=20)
    try:
        return run_additional_tool(payload, tool_slug)
    except (QualityEnforcementError, ToolsLLMError) as exc:
        _raise_quality_http_error(exc)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/tools/vpn/probe-enrich", response_model=VpnProbeEnrichResponse)
async def tools_vpn_probe_enrich(request: Request, payload: VpnProbeEnrichRequest):
    _enforce_tools_rate_limit(request, limit=40)
    try:
        return run_vpn_probe_enrich(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/tools/vpn/{tool_slug}", response_model=VpnToolResponse)
async def tools_vpn(request: Request, tool_slug: str, payload: VpnToolRequest):
    if tool_slug not in VPN_TOOL_SLUGS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown VPN tool.")
    _enforce_tools_rate_limit(request, limit=30)
    try:
        return run_vpn_tool(tool_slug, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
