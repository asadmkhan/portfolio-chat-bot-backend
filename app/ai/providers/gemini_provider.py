from typing import AsyncGenerator, Sequence

from app.ai.types import ChatMessage


class GeminiProvider:
    def __init__(self, model: str):
        self._model = model

    async def stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        raise RuntimeError(
            "GeminiProvider not configured yet. Set up Google SDK + GEMINI_API_KEY."
        )
