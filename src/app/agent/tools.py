"""Tool registry for the coach-assist agent.

Each tool is a thin wrapper over an existing feature — rag_search over Feature 1,
analyze_history over Feature 2 — plus its OpenAI function schema. The agent loop
knows nothing about individual tools; it iterates this registry. Adding a third
tool is a new entry here and zero changes to the loop (the brief asks how a third
tool would be added without rewriting agent logic — this is the answer).

Tools return a structured dict with an explicit `status`. A tool that finds no
usable data returns `{"status": "insufficient_data", ...}` rather than an empty
result, so the model cannot mistake "no data" for "nothing to report" and
confidently answer from a void. That is the graceful-degradation path the brief
requires.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.analysis.insight import HistoryAnalyzer
from app.analysis.repository import HistoryRepository, UnknownUser
from app.llm.client import Usage
from app.rag.answer import RagAnswerer

logger = logging.getLogger(__name__)

ToolFn = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class Tool:
    name: str
    schema: dict[str, Any]  # OpenAI function-tool schema
    run: ToolFn


class ToolRegistry:
    """Holds the agent's tools and exposes them by name and as schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)


def build_registry(
    *,
    answerer: RagAnswerer,
    analyzer: HistoryAnalyzer,
    history: HistoryRepository,
    usage: Usage,
) -> ToolRegistry:
    """Wire the two feature tools into a registry, sharing one usage accumulator."""

    async def rag_search(query: str) -> dict[str, Any]:
        """Feature 1: answer a general fitness question from the knowledge base."""
        result = await answerer.answer(query)
        usage.merge(result.usage)  # roll the tool's cost into the agent total
        # The agent needs the grounded text plus which sources backed it.
        return {
            "status": "ok" if result.status == "answered" else result.status,
            "answer": result.answer,
            "sources": [
                {"chunk_id": s.chunk_id, "section": s.section}
                for s in result.sources
                if s.used
            ],
        }

    async def analyze_history(user_id: str, question: str) -> dict[str, Any]:
        """Feature 2: analyse a specific user's training history."""
        try:
            workouts = history.get(user_id)
        except UnknownUser:
            # Explicit, not an empty result — the model must be told the user
            # doesn't exist rather than inferring an answer from nothing.
            return {
                "status": "unknown_user",
                "message": (
                    f"No workout history found for user_id {user_id!r}. "
                    f"Known users: {history.user_ids()}."
                ),
            }
        result = await analyzer.analyze(workouts, question)
        usage.merge(result.usage)
        return {
            "status": result.status,  # answered | insufficient_data
            "insight": result.insight,
            "data_points_used": result.data_points_used,
            "summary": result.summary,
        }

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="rag_search",
            run=rag_search,
            schema={
                "type": "function",
                "function": {
                    "name": "rag_search",
                    "description": (
                        "Search the fitness knowledge base for general training "
                        "guidance — technique, programming, progressive overload, "
                        "recovery, nutrition. Use for questions about how training "
                        "works in general, not about a specific user's history."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A self-contained fitness question.",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
        )
    )
    registry.register(
        Tool(
            name="analyze_history",
            run=analyze_history,
            schema={
                "type": "function",
                "function": {
                    "name": "analyze_history",
                    "description": (
                        "Analyse a specific user's logged workout history — their "
                        "strength trends, training balance, what they're "
                        "neglecting, and readiness to progress. Use when the "
                        "question is about a particular user's own data."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "The user whose history to analyse.",
                            },
                            "question": {
                                "type": "string",
                                "description": "What to determine from their history.",
                            },
                        },
                        "required": ["user_id", "question"],
                        "additionalProperties": False,
                    },
                },
            },
        )
    )
    return registry
