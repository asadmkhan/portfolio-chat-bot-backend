from typing import AsyncGenerator, Sequence

from app.ai.types import ChatMessage


class ClaudeProvider:
    def __init__(self, model: str):
        self._model = model

    async def stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        raise RuntimeError(
            "ClaudeProvider not configured yet. Set up Anthropic SDK + ANTHROPIC_API_KEY."
        )
