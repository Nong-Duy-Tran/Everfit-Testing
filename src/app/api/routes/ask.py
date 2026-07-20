"""Feature 1 — fitness knowledge RAG endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import AskRequest, AskResponse
from app.config import get_settings
from app.rag.answer import RagAnswerer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["knowledge"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Answer a fitness question from the knowledge base",
)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    settings = get_settings()
    store = request.app.state.store

    if store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge base is empty. Run `python scripts/ingest.py` "
                "(or `docker compose run --rm api python scripts/ingest.py`) first."
            ),
        )

    answerer = RagAnswerer(client=request.app.state.llm, store=store, settings=settings)
    try:
        result = await answerer.answer(payload.question)
    except Exception:
        # Upstream gateway failures must not leak internals to the caller.
        logger.exception("failed to answer question")
        raise HTTPException(
            status_code=502, detail="Upstream model request failed. Please retry."
        )

    return AskResponse.model_validate(result.as_dict(settings))
