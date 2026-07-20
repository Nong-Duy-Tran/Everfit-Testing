"""Coach-assist agent — a registry-driven tool-calling loop.

No framework (LangGraph/CrewAI): for two tools and a single level of delegation,
native function-calling via the OpenAI-compatible gateway is enough, and it keeps
the reasoning the brief grades visible rather than hidden behind a library. Native
+ parallel tool-calling was verified against the gateway in Phase 0.

The loop is deliberately tool-agnostic. It asks the model which tools to call,
executes whatever it asks for from the registry, feeds the results back, and
repeats until the model stops calling tools or the iteration cap is hit. It does
not hardcode the call order — the model decides, per the brief.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.agent.tools import ToolRegistry
from app.config import Settings, get_settings
from app.llm.client import LLMClient, Usage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an assistant that helps a fitness coach answer questions about their \
clients. You have two tools:

- rag_search: general training knowledge (technique, programming, progressive \
overload, recovery, nutrition).
- analyze_history: a specific user's logged workout data (trends, balance, \
what they're neglecting, readiness to progress).

Decide which tools to use and in what order based on the question. Many coaching \
questions need both: check what the client's data shows, then ground the advice \
in training principles. You may call tools in parallel when they are independent.

Rules:
- When you use data from a tool, base your answer on it. Do not invent numbers \
or facts the tools didn't return.
- If a tool reports insufficient_data or unknown_user, say so plainly and work \
with what you have — do not fabricate the missing part.
- When you use both a user's data and general knowledge, make the connection \
explicit (what their data shows, and what principle applies).
- Give the coach a clear, practical answer they can act on."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result_status: str


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCall]
    iterations: int
    usage: Usage
    hit_iteration_cap: bool = False

    def as_dict(self, settings: Settings) -> dict[str, object]:
        return {
            "answer": self.answer,
            "tool_calls": [
                {"name": c.name, "arguments": c.arguments, "status": c.result_status}
                for c in self.tool_calls
            ],
            "iterations": self.iterations,
            "hit_iteration_cap": self.hit_iteration_cap,
            "usage": self.usage.as_dict(settings),
        }


class CoachAssistAgent:
    def __init__(
        self,
        *,
        client: LLMClient,
        registry: ToolRegistry,
        usage: Usage,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._registry = registry
        self._usage = usage
        self._settings = settings or get_settings()

    async def run(self, question: str) -> AgentResult:
        settings = self._settings
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        executed: list[ToolCall] = []

        for iteration in range(1, settings.agent_max_iterations + 1):
            completion = await self._client.chat_with_tools(
                messages,
                self._registry.schemas(),
                usage=self._usage,
                max_tokens=1200,
            )
            choice = completion.choices[0].message
            tool_calls = choice.tool_calls or []

            if not tool_calls:
                # Model produced a final answer.
                return AgentResult(
                    answer=choice.content or "",
                    tool_calls=executed,
                    iterations=iteration,
                    usage=self._usage,
                )

            # Echo the assistant's tool-call turn back into the conversation.
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # Execute each requested tool and append its result.
            for tc in tool_calls:
                result = await self._execute(tc.function.name, tc.function.arguments)
                executed.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=_safe_json(tc.function.arguments),
                        result_status=str(result.get("status", "ok")),
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

        # Iteration cap hit — ask once more for a final answer with tools off,
        # so the coach still gets a coherent response instead of a dangling loop.
        logger.warning("agent hit iteration cap (%d)", settings.agent_max_iterations)
        final = await self._client.chat(
            messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Give your best final answer now using what the tools "
                        "have already returned. Do not call any more tools."
                    ),
                }
            ],
            usage=self._usage,
            max_tokens=1200,
        )
        return AgentResult(
            answer=final,
            tool_calls=executed,
            iterations=settings.agent_max_iterations,
            usage=self._usage,
            hit_iteration_cap=True,
        )

    async def _execute(self, name: str, raw_arguments: str) -> dict[str, Any]:
        tool = self._registry.get(name)
        if tool is None:
            # The model hallucinated a tool name; tell it, don't crash.
            return {"status": "error", "message": f"unknown tool {name!r}"}

        args = _safe_json(raw_arguments)
        try:
            return await tool.run(**args)
        except TypeError as exc:
            # Wrong/missing arguments — surface to the model to retry.
            logger.warning("tool %s bad arguments %r: %s", name, args, exc)
            return {"status": "error", "message": f"invalid arguments for {name}: {exc}"}
        except Exception:
            logger.exception("tool %s failed", name)
            return {"status": "error", "message": f"{name} failed to execute"}


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
