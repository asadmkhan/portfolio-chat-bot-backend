from __future__ import annotations

import re

_BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|(?:\d+[\.\)]))\s+")


def enumerate_lines(text: str) -> list[tuple[int, str]]:
    return [(index + 1, line) for index, line in enumerate(text.splitlines())]


def is_bullet_like(line: str) -> bool:
    return bool(_BULLET_PATTERN.match(line))


def strip_bullet_prefix(line: str) -> str:
    return _BULLET_PATTERN.sub("", line).strip()


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)

