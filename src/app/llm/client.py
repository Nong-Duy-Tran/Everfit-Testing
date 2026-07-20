"""Async wrapper around the OpenAI-compatible gateway.

Every call routes through here so that token usage — and therefore cost — is
measured rather than estimated. The README's cost-per-query figures come from
`UsageTracker`, not from arithmetic on guessed prompt sizes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

Message = dict[str, Any]
ToolSchema = dict[str, Any]


@dataclass
class Usage:
    """Token counts for a single logical operation (may span several API calls)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    calls: int = 0

    def add_completion(self, completion: ChatCompletion) -> None:
        self.calls += 1
        if completion.usage:
            self.prompt_tokens += completion.usage.prompt_tokens
            self.completion_tokens += completion.usage.completion_tokens

    def add_embedding(self, tokens: int) -> None:
        self.calls += 1
        self.embedding_tokens += tokens

    def merge(self, other: "Usage") -> None:
        """Fold another operation's usage in — e.g. a tool call's cost into the
        agent's running total, so agent usage reflects the whole tool chain."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.embedding_tokens += other.embedding_tokens
        self.calls += other.calls

    def cost_usd(self, s: Settings) -> float:
        return (
            self.prompt_tokens / 1e6 * s.usd_per_1m_input
            + self.completion_tokens / 1e6 * s.usd_per_1m_output
            + self.embedding_tokens / 1e6 * s.usd_per_1m_embedding
        )

    def as_dict(self, s: Settings) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "api_calls": self.calls,
            "estimated_cost_usd": round(self.cost_usd(s), 6),
        }


@dataclass
class LLMClient:
    """Thin, provider-agnostic surface over the chat + embedding endpoints."""

    settings: Settings = field(default_factory=get_settings)
    _client: AsyncOpenAI = field(init=False)

    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.request_timeout_s,
            max_retries=self.settings.max_retries,
        )

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        usage: Usage | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        completion = await self._client.chat.completions.create(
            model=self.settings.llm_model_name,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if usage is not None:
            usage.add_completion(completion)
        return completion.choices[0].message.content or ""

    async def chat_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        usage: Usage | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ChatCompletion:
        """Raw completion — the agent loop needs `tool_calls`, not just text."""
        completion = await self._client.chat.completions.create(
            model=self.settings.llm_model_name,
            messages=list(messages),
            tools=list(tools),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if usage is not None:
            usage.add_completion(completion)
        return completion

    async def structured(
        self,
        messages: Sequence[Message],
        schema: dict[str, Any],
        *,
        schema_name: str = "response",
        usage: Usage | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Strict-mode JSON. Gateway support verified in the Phase 0 probe."""
        completion = await self._client.chat.completions.create(
            model=self.settings.llm_model_name,
            messages=list(messages),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if usage is not None:
            usage.add_completion(completion)
        return completion.choices[0].message.content or ""

    async def embed(
        self, texts: Sequence[str], *, usage: Usage | None = None
    ) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self.settings.text_embedding_model_name,
            input=list(texts),
        )
        if usage is not None:
            usage.add_embedding(response.usage.total_tokens if response.usage else 0)
        return [item.embedding for item in response.data]

    async def aclose(self) -> None:
        await self._client.close()