from __future__ import annotations

import os
from typing import AsyncGenerator, Optional, Sequence

from openai import AsyncOpenAI

from app.ai.types import ChatMessage


class OpenAIProvider:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.2,
    ):
        self._model = model
        self._temperature = temperature
        self._response_format = (os.getenv("OPENAI_RESPONSE_FORMAT") or "").strip().lower()
        key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is missing")

        self._client = AsyncOpenAI(
            api_key=key,
            base_url=(base_url or os.getenv("OPENAI_BASE_URL") or None),
            timeout=float(os.getenv("OPENAI_TIMEOUT_S", str(timeout_s))),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", str(max_retries))),
        )

    async def stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        payload = [{"role": m.role, "content": m.content} for m in messages]

        create_kwargs = {
            "model": self._model,
            "messages": payload,
            "temperature": self._temperature,
            "stream": True,
        }
        if self._response_format == "json":
            create_kwargs["response_format"] = {"type": "json_object"}

        stream = await self._client.chat.completions.create(**create_kwargs)

        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
            except Exception:
                continue


def from_env() -> OpenAIProvider:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    timeout_s = float(os.getenv("OPENAI_TIMEOUT_S", "30"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

    return OpenAIProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
