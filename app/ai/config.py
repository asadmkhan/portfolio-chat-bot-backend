import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AIConfig:
    provider: str
    model: str


def load_ai_config() -> AIConfig:
    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    model = os.getenv("AI_MODEL", "gpt-4o-mini").strip()
    return AIConfig(provider=provider, model=model)
