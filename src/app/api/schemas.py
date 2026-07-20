"""Request/response models for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="A natural-language fitness question.",
        examples=["How do I bench press with proper form?"],
    )


class SourceModel(BaseModel):
    number: int = Field(..., description="Citation number referenced as [n] in the answer.")
    chunk_id: str = Field(..., description="Stable chunk identifier, e.g. 01-bench-press#proper-form.")
    document: str
    section: str
    similarity: float
    used: bool = Field(..., description="Whether the model actually drew on this source.")


class UsageModel(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    embedding_tokens: int
    api_calls: int
    estimated_cost_usd: float


class WorkoutSet(BaseModel):
    reps: int
    weight: float
    unit: str = "kg"


class WorkoutEntry(BaseModel):
    date: str = Field(..., description="ISO date, e.g. 2026-03-20.")
    exercise: str
    sets: list[WorkoutSet]


class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    user_id: str | None = Field(
        default=None,
        description="Load a stored user's history. Mutually exclusive with `workouts`.",
    )
    workouts: list[WorkoutEntry] | None = Field(
        default=None,
        description="Inline workout history. Mutually exclusive with `user_id`.",
    )


class AnalyzeResponse(BaseModel):
    status: Literal["answered", "insufficient_data"]
    insight: str
    data_points_used: list[str]
    summary: dict | None = Field(
        default=None,
        description="The deterministic pre-computed summary the insight was based on.",
    )
    usage: "UsageModel"


class AgentRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="A coaching question, possibly multi-step, possibly naming a user_id.",
        examples=[
            "Based on user_b's recent history, is he ready to increase bench "
            "press weight? What does proper progressive overload look like for him?"
        ],
    )


class AgentToolCall(BaseModel):
    name: str
    arguments: dict
    status: str = Field(..., description="Outcome status the tool returned.")


class AgentResponse(BaseModel):
    answer: str
    tool_calls: list[AgentToolCall] = Field(
        ..., description="Tools the agent chose to call, in order, with outcomes."
    )
    iterations: int
    hit_iteration_cap: bool
    usage: "UsageModel"


class AskResponse(BaseModel):
    status: Literal["answered", "out_of_scope", "refused"] = Field(
        ...,
        description=(
            "answered: grounded answer with sources. "
            "out_of_scope: not a fitness question, no generation ran. "
            "refused: blocked by the safety guardrail (see refusal_category)."
        ),
    )
    answer: str
    sources: list[SourceModel]
    in_scope: bool = Field(
        ...,
        description="True only when status is 'answered'. Kept for convenience.",
    )
    refusal_category: str | None = Field(
        default=None,
        description="medical_diagnosis | injury_rehab | eating_disorder when refused, else null.",
    )
    usage: UsageModel
