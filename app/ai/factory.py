from app.ai.config import load_ai_config
from app.ai.types import AIClient

from app.ai.providers.openai_provider import OpenAIProvider
from app.ai.providers.claude_provider import ClaudeProvider
from app.ai.providers.gemini_provider import GeminiProvider


def get_ai_client() -> AIClient:
    cfg = load_ai_config()

    if cfg.provider == "openai":
        return OpenAIProvider(model=cfg.model)

    if cfg.provider == "claude":
        return ClaudeProvider(model=cfg.model)

    if cfg.provider == "gemini":
        return GeminiProvider(model=cfg.model)

    raise ValueError(f"Unsupported AI_PROVIDER='{cfg.provider}'")
