from __future__ import annotations

from typing import Protocol


class TaxonomyProvider(Protocol):
    def normalize_skill(self, raw: str) -> tuple[str, str | None]:
        """Return normalized text and optional canonical skill ID."""

