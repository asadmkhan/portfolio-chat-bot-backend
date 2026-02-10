from __future__ import annotations

import base64
from io import BytesIO
import json
import logging
import os
import time
import uuid
from functools import lru_cache
from typing import Any

from openai import OpenAI

from app.analytics.db import log_ai_analysis_run

logger = logging.getLogger(__name__)


class ToolsLLMError(RuntimeError):
    def __init__(self, message: str, *, code: str = "llm_unavailable"):
        super().__init__(message)
        self.code = code


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _looks_like_placeholder(value: str) -> bool:
    lower = value.strip().lower()
    return lower.startswith("your_") or lower.startswith("replace_") or lower in {"changeme", "todo"}


def tools_llm_enabled() -> bool:
    enabled = _env_bool("TOOLS_LLM_ENABLED", True)
    if not enabled:
        return False
    provider = (os.getenv("AI_PROVIDER") or "openai").strip().lower()
    if provider != "openai":
        return False
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key or _looks_like_placeholder(api_key):
        return False
    return True


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
        base_url=(os.getenv("OPENAI_BASE_URL") or None),
        timeout=float(os.getenv("TOOLS_LLM_TIMEOUT_S", "20")),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
    )


def _model() -> str:
    return (os.getenv("AI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()


def _strict_quality_mode() -> bool:
    value = (os.getenv("TOOLS_STRICT_LLM") or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _log_ai_run(
    *,
    run_id: str,
    tool_slug: str,
    schema_valid: bool,
    status: str,
    latency_ms: int | None,
    error_code: str | None = None,
) -> None:
    try:
        log_ai_analysis_run(
            run_id=run_id,
            tool_slug=tool_slug or "unknown",
            model=_model(),
            strict_mode=_strict_quality_mode(),
            schema_valid=schema_valid,
            status=status,
            error_code=error_code,
            latency_ms=latency_ms,
        )
    except Exception:  # pragma: no cover - analytics must not break AI responses
        logger.debug("ai_run_logging_failed", exc_info=True)


def json_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 900,
    tool_slug: str = "unknown",
) -> dict[str, Any] | None:
    run_id = uuid.uuid4().hex
    started = time.perf_counter()
    if not tools_llm_enabled():
        _log_ai_run(
            run_id=run_id,
            tool_slug=tool_slug,
            schema_valid=False,
            status="skipped",
            error_code="llm_disabled",
            latency_ms=0,
        )
        return None

    try:
        response = _client().chat.completions.create(
            model=_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_output_tokens,
        )
        content = response.choices[0].message.content if response.choices else ""
        if not content:
            _log_ai_run(
                run_id=run_id,
                tool_slug=tool_slug,
                schema_valid=False,
                status="empty",
                error_code="empty_response",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            return None
        parsed = json.loads(content)
        schema_valid = isinstance(parsed, dict)
        _log_ai_run(
            run_id=run_id,
            tool_slug=tool_slug,
            schema_valid=schema_valid,
            status="success" if schema_valid else "invalid_schema",
            error_code=None if schema_valid else "invalid_schema",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return parsed if schema_valid else None
    except Exception as exc:  # noqa: BLE001 - deterministic fallback is expected
        logger.warning("tools_llm_json_failed model=%s prompt_len=%s: %s", _model(), len(user_prompt), exc)
        _log_ai_run(
            run_id=run_id,
            tool_slug=tool_slug,
            schema_valid=False,
            status="error",
            error_code="llm_exception",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return None


def json_completion_required(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 900,
    tool_slug: str = "unknown",
) -> dict[str, Any]:
    if not tools_llm_enabled():
        raise ToolsLLMError("AI quality mode is enabled but OpenAI is not configured.", code="llm_disabled")

    payload = json_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        tool_slug=tool_slug,
    )
    if not payload:
        raise ToolsLLMError("AI quality mode could not produce a valid response. Try again.", code="llm_invalid")
    return payload


def vision_extract_text(*, content: bytes, filename: str) -> str | None:
    if not tools_llm_enabled():
        return None

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
        "bmp": "image/bmp",
    }.get(ext, "image/png")

    try:
        encoded = base64.b64encode(content).decode("utf-8")
        response = _client().chat.completions.create(
            model=_model(),
            messages=[
                {
                    "role": "system",
                    "content": "Extract concise, factual text from this image. Return plain text only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read this image and return a concise text extraction summary."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=500,
        )
        content_text = response.choices[0].message.content if response.choices else ""
        if not content_text:
            return None
        return str(content_text).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("tools_llm_vision_failed model=%s file=%s: %s", _model(), filename, exc)
        return None


def transcribe_media(*, content: bytes, filename: str) -> str | None:
    if not tools_llm_enabled():
        return None

    model = (os.getenv("OPENAI_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe").strip()
    try:
        file_obj = BytesIO(content)
        file_obj.name = filename
        response = _client().audio.transcriptions.create(
            model=model,
            file=file_obj,
        )
        text = getattr(response, "text", None)
        if not text:
            return None
        return str(text).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("tools_llm_transcribe_failed model=%s file=%s: %s", model, filename, exc)
        return None
