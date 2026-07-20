"""FastAPI entrypoint for the AI Workout Coach."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.llm.client import LLMClient

settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = LLMClient(settings)
    logger.info(
        "started | model=%s embedding=%s base_url=%s",
        settings.llm_model_name,
        settings.text_embedding_model_name,
        settings.base_url,
    )
    yield
    await app.state.llm.aclose()


app = FastAPI(
    title="AI Workout Coach",
    version="0.1.0",
    description="Fitness knowledge RAG, workout history analysis, and a coach-assist agent.",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.llm_model_name}


# Feature routers are mounted in later phases:
#   /ask      -> Feature 1 (RAG + guardrails)
#   /analyze  -> Feature 2 (workout history analysis)
#   /agent    -> Feature 3 (coach assist agent)
