from __future__ import annotations

from pathlib import Path
from typing import Any

_SCORING_CONFIG_CACHE: dict[str, Any] | None = None
_SCORING_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "scoring.yaml"


def get_scoring_config() -> dict[str, Any]:
    """Load scoring config from repo-level config/scoring.yaml and cache it."""
    global _SCORING_CONFIG_CACHE

    if _SCORING_CONFIG_CACHE is not None:
        return _SCORING_CONFIG_CACHE

    if not _SCORING_CONFIG_PATH.exists():
        raise RuntimeError(
            f"Scoring config not found at '{_SCORING_CONFIG_PATH}'. "
            "Expected file: config/scoring.yaml"
        )

    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Unable to parse scoring config because PyYAML is unavailable. "
            "Install dependency: PyYAML."
        ) from exc

    try:
        raw = _SCORING_CONFIG_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read scoring config '{_SCORING_CONFIG_PATH}': {exc}"
        ) from exc

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        raise RuntimeError(
            f"Invalid YAML in scoring config '{_SCORING_CONFIG_PATH}': {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Invalid scoring config '{_SCORING_CONFIG_PATH}': expected a top-level mapping."
        )

    _SCORING_CONFIG_CACHE = parsed
    return _SCORING_CONFIG_CACHE


def get_scoring_value(path: str, default: Any = None) -> Any:
    """Get nested config value using dot path notation, e.g. 'matching.weights.must_have'."""
    if not path:
        return default

    current: Any = get_scoring_config()
    for key in path.split("."):
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    return current

