"""FastAPI entrypoint for the AI Workout Coach."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.analysis.repository import HistoryRepository
from app.api.routes import agent as agent_routes
from app.api.routes import analyze as analyze_routes
from app.api.routes import ask as ask_routes
from app.config import get_settings
from app.llm.client import LLMClient
from app.rag.store import VectorStore

settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# LOG_LEVEL=debug is for *our* code. Third-party clients at DEBUG dump full
# request bodies and response headers, which buries application logs and risks
# echoing prompt content into stdout.
for noisy in ("httpx", "httpcore", "openai", "chromadb"):
    logging.getLogger(noisy).setLevel(max(logging.INFO, logger.getEffectiveLevel()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = LLMClient(settings)
    app.state.store = VectorStore(
        directory=settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )
    app.state.history = HistoryRepository(settings.workout_history_path)
    logger.info(
        "started | model=%s embedding=%s chunks=%d",
        settings.llm_model_name,
        settings.text_embedding_model_name,
        app.state.store.count(),
    )
    yield
    await app.state.llm.aclose()


app = FastAPI(
    title="AI Workout Coach",
    version="0.1.0",
    description="Fitness knowledge RAG, workout history analysis, and a coach-assist agent.",
    lifespan=lifespan,
)


app.include_router(ask_routes.router)
app.include_router(analyze_routes.router)
app.include_router(agent_routes.router)


@app.get("/health", tags=["ops"])
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": settings.llm_model_name,
        "indexed_chunks": app.state.store.count(),
    }
