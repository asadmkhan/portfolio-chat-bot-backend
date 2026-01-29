from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    score: float


class FaissRetriever:
    _model_cache: dict[str, SentenceTransformer] = {}
    _retriever_cache: dict[str, "FaissRetriever"] = {}

    def __init__(self, lang: str, data_dir: Path | None = None):
        self.lang = lang
        self.data_dir = data_dir or Path("data")
        self.documents_dir = self.data_dir / "documents" / lang
        self.faiss_dir = self.data_dir / "faiss" / lang

        self.meta_path = self.faiss_dir / "meta.json"
        self.index_path = self.faiss_dir / "index.faiss"

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.model_name: str = meta["model_name"]

        self._model = self._get_model(self.model_name)
        self._index = faiss.read_index(str(self.index_path))
        self._chunks: list[dict[str, Any]] = meta["chunks"]

    @classmethod
    def get(cls, lang: str) -> "FaissRetriever":
        lang = (lang or "en").lower()
        if lang not in cls._retriever_cache:
            cls._retriever_cache[lang] = cls(lang=lang)
        return cls._retriever_cache[lang]

    @classmethod
    def warmup(cls, langs: list[str]) -> None:
        for lang in langs:
            _ = cls.get(lang)

    @classmethod
    def _get_model(cls, model_name: str) -> SentenceTransformer:
        if model_name not in cls._model_cache:
            cls._model_cache[model_name] = SentenceTransformer(model_name)
        return cls._model_cache[model_name]

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        emb = self._model.encode([q], normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")

        scores, idxs = self._index.search(emb, k)
        results: list[dict[str, Any]] = []

        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self._chunks):
                continue
            c = self._chunks[idx]
            results.append(
                {
                    "id": c.get("id"),
                    "source": c.get("source"),
                    "text": c.get("text"),
                    "score": float(score),
                }
            )

        return results
