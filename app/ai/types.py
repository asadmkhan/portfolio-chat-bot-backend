from dataclasses import dataclass
from typing import AsyncGenerator, Literal, Protocol, Sequence


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class AIClient(Protocol):
    async def stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncGenerator[str, None]: ...
