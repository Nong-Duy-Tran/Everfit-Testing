"""Feature 3 — coach-assist agent endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.agent.loop import CoachAssistAgent
from app.agent.tools import build_registry
from app.analysis.insight import HistoryAnalyzer
from app.api.schemas import AgentRequest, AgentResponse
from app.config import get_settings
from app.guardrail.classifier import Guardrail
from app.llm.client import Usage
from app.rag.answer import RagAnswerer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent"])


@router.post(
    "/agent",
    response_model=AgentResponse,
    summary="Answer a multi-step coaching question using the RAG and analysis tools",
)
async def agent(payload: AgentRequest, request: Request) -> AgentResponse:
    settings = get_settings()
    llm = request.app.state.llm
    usage = Usage()

    # rag_search reuses the full Feature 1 pipeline, guardrail included, so unsafe
    # sub-queries the agent issues are refused the same way a direct /ask is.
    guardrail = Guardrail(client=llm, settings=settings) if settings.guardrail_enabled else None
    answerer = RagAnswerer(
        client=llm, store=request.app.state.store, settings=settings, guardrail=guardrail
    )
    analyzer = HistoryAnalyzer(client=llm, settings=settings)

    registry = build_registry(
        answerer=answerer,
        analyzer=analyzer,
        history=request.app.state.history,
        usage=usage,
    )
    coach_agent = CoachAssistAgent(
        client=llm, registry=registry, usage=usage, settings=settings
    )

    try:
        result = await coach_agent.run(payload.question)
    except Exception:
        logger.exception("agent run failed")
        raise HTTPException(
            status_code=502, detail="Upstream model request failed. Please retry."
        )

    return AgentResponse.model_validate(result.as_dict(settings))
