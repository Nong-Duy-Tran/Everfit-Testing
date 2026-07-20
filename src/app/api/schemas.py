"""Request/response models for the public API."""

from __future__ import annotations

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
    answer: str
    sources: list[SourceModel]
    in_scope: bool = Field(
        ...,
        description="False when the question was out of scope; no generation ran and sources is empty.",
    )
    usage: UsageModel
