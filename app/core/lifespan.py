from contextlib import asynccontextmanager

from app.rag.retriever import FaissRetriever


@asynccontextmanager
async def lifespan(app):
    FaissRetriever.warmup(["en", "de"])
    yield
