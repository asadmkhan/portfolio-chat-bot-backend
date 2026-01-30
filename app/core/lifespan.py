from contextlib import asynccontextmanager
from pathlib import Path

from app.rag.ingest import build_faiss_index
from app.rag.retriever import FaissRetriever


@asynccontextmanager
async def lifespan(app):
    en_meta = Path("data/faiss/en/meta.json")
    de_meta = Path("data/faiss/de/meta.json")

    if not en_meta.exists():
        build_faiss_index(lang="en")

    if not de_meta.exists():
        build_faiss_index(lang="de")

    FaissRetriever.warmup(["en", "de"])
    yield
