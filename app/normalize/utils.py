from __future__ import annotations

import re

_BULLET_CHARS = "•◦▪▫●○■□◆◇▶►-–—*·"
_BULLET_PATTERN = re.compile(rf"^\s*(?:[{re.escape(_BULLET_CHARS)}]|(?:\d+[\.\)]))\s+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_URL_RE = re.compile(r"(?:https?://|www\.|linkedin\.com|github\.com)", re.IGNORECASE)
_SECTION_RE = re.compile(
    r"^\s*(summary|objective|profile|experience|work experience|employment history|skills|education|projects|certifications)\s*:?\s*$",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"^\s*(built|led|managed|designed|developed|implemented|created|delivered|improved|reduced|scaled|launched|optimized|engineered|owned|drove|supported)\b",
    re.IGNORECASE,
)
_METRIC_RE = re.compile(r"\d|%|[$€£]")


def enumerate_lines(text: str) -> list[tuple[int, str]]:
    return [(index + 1, line) for index, line in enumerate(text.splitlines())]


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_bullet_like(line: str) -> bool:
    return bool(_BULLET_PATTERN.match(line))


def strip_bullet_prefix(line: str) -> str:
    return _BULLET_PATTERN.sub("", line).strip()


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def is_section_heading(line: str) -> bool:
    stripped = normalize_line(line)
    if not stripped:
        return False
    if _SECTION_RE.match(stripped):
        return True
    return bool(stripped.isupper() and len(stripped.split()) <= 5 and len(stripped) <= 36)


def is_contact_or_url(line: str) -> bool:
    stripped = normalize_line(line)
    if not stripped:
        return False
    return bool(_EMAIL_RE.search(stripped) or _PHONE_RE.search(stripped) or _URL_RE.search(stripped))


def is_claim_like(line: str, *, in_experience_section: bool = False) -> bool:
    stripped = normalize_line(line)
    if not stripped:
        return False
    if is_section_heading(stripped) or is_contact_or_url(stripped):
        return False
    words = stripped.split()
    if len(words) < 4:
        return False
    if _ACTION_RE.match(stripped):
        return True
    if not in_experience_section:
        return False
    if _METRIC_RE.search(stripped) and len(words) >= 5:
        return True
    if in_experience_section and len(words) >= 6 and any(
        token.endswith(("ed", "ing"))
        for token in re.findall(r"[A-Za-z]+", stripped.lower())
    ):
        return True
    return False


def is_continuation_line(line: str) -> bool:
    stripped = normalize_line(line)
    if not stripped:
        return False
    if is_section_heading(stripped) or is_contact_or_url(stripped) or is_bullet_like(stripped):
        return False
    words = stripped.split()
    if len(words) < 2:
        return False
    return stripped[0].islower() or len(words) <= 12
