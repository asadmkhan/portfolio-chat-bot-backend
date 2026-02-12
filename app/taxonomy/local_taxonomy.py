from __future__ import annotations

import json
from pathlib import Path

from .provider import TaxonomyProvider


class LocalTaxonomy(TaxonomyProvider):
    def __init__(self, synonyms_path: str | Path | None = None) -> None:
        path = Path(synonyms_path) if synonyms_path else Path(__file__).with_name("synonyms.json")
        self._synonyms = self._load_synonyms(path)

    @staticmethod
    def _load_synonyms(path: Path) -> dict[str, str]:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {str(key).strip().lower(): str(value) for key, value in raw.items()}

    def normalize_skill(self, raw: str) -> tuple[str, str | None]:
        normalized = raw.strip().lower()
        canonical_skill_id = self._synonyms.get(normalized)
        return normalized, canonical_skill_id

