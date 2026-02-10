import contextlib
from contextlib import asynccontextmanager
import asyncio
import logging
from pathlib import Path

from app.rag.ingest import build_faiss_index
from app.rag.retriever import FaissRetriever
from app.analytics.db import init_db, purge_old_records

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    en_meta = Path("data/faiss/en/meta.json")
    de_meta = Path("data/faiss/de/meta.json")

    if not en_meta.exists():
        build_faiss_index(lang="en")

    if not de_meta.exists():
        build_faiss_index(lang="de")

    FaissRetriever.warmup(["en", "de"])
    init_db()
    purge_old_records()

    stop_event = asyncio.Event()

    async def periodic_purge() -> None:
        while not stop_event.is_set():
            try:
                deleted = purge_old_records()
                if any(deleted.values()):
                    logger.info("analytics_retention_purge deleted=%s", deleted)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("analytics_retention_purge_failed: %s", exc)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=3600)
            except asyncio.TimeoutError:
                continue

    purge_task = asyncio.create_task(periodic_purge())
    yield
    stop_event.set()
    if not purge_task.done():
        purge_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await purge_task
