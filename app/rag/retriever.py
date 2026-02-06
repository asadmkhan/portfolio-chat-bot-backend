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
        self.embeddings_path = self.faiss_dir / "embeddings.npy"

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.model_name: str = meta["model_name"]

        self._model = self._get_model(self.model_name)
        self._index = faiss.read_index(str(self.index_path))
        self._chunks: list[dict[str, Any]] = meta["chunks"]
        self._embeddings: np.ndarray | None = None
        if self.embeddings_path.exists():
            self._embeddings = np.load(self.embeddings_path, mmap_mode="r")

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

    def _mmr_select(
        self,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        k: int,
        lambda_mult: float,
    ) -> list[int]:
        n = doc_embs.shape[0]
        if n == 0:
            return []
        k = min(k, n)
        query_vec = query_emb.reshape(-1)
        sim_to_query = doc_embs @ query_vec

        selected: list[int] = []
        candidate_idxs = list(range(n))

        while len(selected) < k and candidate_idxs:
            if not selected:
                next_idx = int(np.argmax(sim_to_query))
                selected.append(next_idx)
                candidate_idxs.remove(next_idx)
                continue

            selected_embs = doc_embs[selected]
            sim_to_selected = doc_embs @ selected_embs.T
            max_sim = sim_to_selected.max(axis=1)
            mmr_scores = lambda_mult * sim_to_query - (1 - lambda_mult) * max_sim

            for idx in selected:
                mmr_scores[idx] = -np.inf

            next_idx = int(np.argmax(mmr_scores))
            if next_idx in selected:
                break
            selected.append(next_idx)
            candidate_idxs.remove(next_idx)

        return selected

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        use_mmr: bool = True,
        fetch_k: int = 20,
        mmr_lambda: float = 0.6,
    ) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        emb = self._model.encode([q], normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")

        search_k = max(k, fetch_k) if use_mmr else k
        scores, idxs = self._index.search(emb, search_k)
        results: list[dict[str, Any]] = []

        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self._chunks):
                continue
            c = self._chunks[idx]
            results.append(
                {
                    "index": idx,
                    "id": c.get("id"),
                    "source": c.get("source"),
                    "text": c.get("text"),
                    "score": float(score),
                }
            )

        if not use_mmr or len(results) <= k:
            return results[:k]

        if self._embeddings is not None:
            doc_idxs = [r.get("index") for r in results]
            doc_embs = np.asarray(self._embeddings[doc_idxs], dtype="float32")
        else:
            doc_texts = [r.get("text", "") for r in results]
            doc_embs = self._model.encode(
                doc_texts, normalize_embeddings=True, show_progress_bar=False
            )
            doc_embs = np.asarray(doc_embs, dtype="float32")

        selected_idxs = self._mmr_select(emb[0], doc_embs, k, mmr_lambda)
        reranked = [results[i] for i in selected_idxs]
        return reranked
