"""Feature 2 — workout history analysis endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.analysis.insight import HistoryAnalyzer
from app.analysis.repository import UnknownUser
from app.api.schemas import AnalyzeRequest, AnalyzeResponse
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse a user's workout history and answer a question about it",
)
async def analyze(payload: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    settings = get_settings()

    # Presence, not truthiness: an explicitly empty `workouts: []` is a valid
    # request (analyse an empty history -> insufficient_data), not a missing one.
    if (payload.user_id is not None) == (payload.workouts is not None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of `user_id` or `workouts`.",
        )

    if payload.user_id is not None:
        try:
            # Isolation boundary: get() returns only this user's workouts.
            workouts = request.app.state.history.get(payload.user_id)
        except UnknownUser:
            raise HTTPException(
                status_code=404, detail=f"Unknown user_id: {payload.user_id!r}"
            )
    else:
        workouts = [w.model_dump() for w in payload.workouts or []]

    analyzer = HistoryAnalyzer(client=request.app.state.llm, settings=settings)
    try:
        result = await analyzer.analyze(workouts, payload.question)
    except Exception:
        logger.exception("failed to analyze history")
        raise HTTPException(
            status_code=502, detail="Upstream model request failed. Please retry."
        )

    return AnalyzeResponse.model_validate(result.as_dict(settings))
