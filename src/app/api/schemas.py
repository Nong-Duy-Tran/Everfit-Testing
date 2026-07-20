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
