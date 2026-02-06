import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    start: int
    end: int


def _read_markdown_files(doc_dir: Path) -> List[Tuple[str, str]]:
    files = sorted([p for p in doc_dir.rglob("*.md") if p.is_file()])
    items: List[Tuple[str, str]] = []
    for p in files:
        items.append((p.name, p.read_text(encoding="utf-8")))
    return items


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[Tuple[int, int, str]]:
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_faiss_index(
    lang: str,     
    documents_dir: str = "data/documents",
    out_dir: str = "data/faiss",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 450,
    overlap: int = 80,
) -> Dict:
    doc_dir = Path(documents_dir)/lang
    out_path = Path(out_dir)/lang
    out_path.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name)

    md_files = _read_markdown_files(doc_dir)
    chunks: List[Chunk] = []

    for source_name, content in md_files:
        for i, (s, e, chunk_text) in enumerate(_chunk_text(content, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{source_name}::chunk::{i}",
                    text=chunk_text,
                    source=source_name,
                    start=s,
                    end=e,
                )
            )

    if not chunks:
        raise RuntimeError(f"No chunks found. Put .md files into {documents_dir}")

    texts = [c.text for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(out_path / "index.faiss"))
    np.save(out_path / "embeddings.npy", vectors)

    meta = {
        "lang": lang,
        "model_name": model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "count": len(chunks),
        "embeddings_path": "embeddings.npy",
        "chunks": [
            {"id": c.id, "source": c.source, "start": c.start, "end": c.end, "text": c.text}
            for c in chunks
        ],
    }
    (out_path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    return {"index_path": str(out_path / "index.faiss"), "meta_path": str(out_path / "meta.json"), "count": len(chunks)}
