from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.parsing.models import ParsedDoc


class ParsingReport(BaseModel):
    layout_flags: dict[str, Any] = Field(default_factory=dict)
    missing_contact: bool = False


def build_parsing_report(parsed: ParsedDoc, profile: dict[str, Any] | None = None) -> ParsingReport:
    if profile is None:
        missing_contact = False
    else:
        has_contact = bool(profile.get("email") or profile.get("phone") or profile.get("contact"))
        missing_contact = not has_contact

    return ParsingReport(layout_flags=parsed.layout_flags, missing_contact=missing_contact)

