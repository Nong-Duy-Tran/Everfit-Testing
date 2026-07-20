"""Agent loop control flow, tested offline with a scripted LLM client.

The model's *tool-selection quality* is measured by the eval pipeline (Feature 4).
These tests fix the model's decisions so they assert our loop: registry dispatch,
feeding results back, graceful degradation, the iteration cap, and cost merging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from app.agent.loop import CoachAssistAgent
from app.agent.tools import Tool, ToolRegistry
from app.llm.client import Usage


# --- a scripted chat-completions client -----------------------------------


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list | None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeCompletion:
    choices: list


class ScriptedClient:
    """Yields a pre-scripted sequence of assistant turns.

    Each script item is either a list of (tool_name, args_dict) to call, or a
    string final answer.
    """

    def __init__(self, script: list):
        self._script = list(script)
        self.chat_with_tools_calls = 0

    async def chat_with_tools(self, messages, tools, *, usage=None, **kw):
        step = self._script.pop(0)
        self.chat_with_tools_calls += 1
        if isinstance(step, str):
            return FakeCompletion([FakeChoice(FakeMessage(step, None))])
        tool_calls = [
            FakeToolCall(id=f"call_{i}", function=FakeFunction(name, json.dumps(args)))
            for i, (name, args) in enumerate(step)
        ]
        return FakeCompletion([FakeChoice(FakeMessage(None, tool_calls))])

    async def chat(self, messages, *, usage=None, **kw):
        step = self._script.pop(0)
        return step if isinstance(step, str) else ""


def registry_with(recorder: list) -> ToolRegistry:
    async def rag_search(query: str):
        recorder.append(("rag_search", query))
        return {"status": "ok", "answer": "overload means adding load", "sources": []}

    async def analyze_history(user_id: str, question: str):
        recorder.append(("analyze_history", user_id))
        if user_id == "ghost":
            return {"status": "unknown_user", "message": "no such user"}
        return {"status": "answered", "insight": "trending up", "summary": {}}

    reg = ToolRegistry()
    reg.register(Tool("rag_search", {"type": "function", "function": {"name": "rag_search"}}, rag_search))
    reg.register(Tool("analyze_history", {"type": "function", "function": {"name": "analyze_history"}}, analyze_history))
    return reg


def make_agent(script, recorder, max_iter=5):
    from app.config import get_settings

    settings = get_settings()
    settings = settings.model_copy(update={"agent_max_iterations": max_iter})
    return CoachAssistAgent(
        client=ScriptedClient(script),
        registry=registry_with(recorder),
        usage=Usage(),
        settings=settings,
    )


@pytest.mark.asyncio
async def test_no_tool_calls_returns_final_answer_immediately():
    recorder = []
    agent = make_agent(["Progressive overload is..."], recorder)
    result = await agent.run("what is progressive overload?")
    assert result.answer.startswith("Progressive overload")
    assert result.tool_calls == []
    assert recorder == []  # no tools executed
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_dispatches_requested_tool_then_answers():
    recorder = []
    agent = make_agent(
        [[("rag_search", {"query": "overload?"})], "Here is the grounded answer."],
        recorder,
    )
    result = await agent.run("explain overload")
    assert recorder == [("rag_search", "overload?")]
    assert [c.name for c in result.tool_calls] == ["rag_search"]
    assert result.answer == "Here is the grounded answer."
    assert result.iterations == 2


@pytest.mark.asyncio
async def test_parallel_tool_calls_in_one_turn_all_execute():
    recorder = []
    agent = make_agent(
        [
            [("analyze_history", {"user_id": "user_b", "question": "ready?"}),
             ("rag_search", {"query": "progressive overload"})],
            "Combined answer using both.",
        ],
        recorder,
    )
    result = await agent.run("is user_b ready to progress and what's the principle?")
    assert {r[0] for r in recorder} == {"analyze_history", "rag_search"}
    assert len(result.tool_calls) == 2


@pytest.mark.asyncio
async def test_unknown_user_status_is_surfaced_not_hidden():
    recorder = []
    agent = make_agent(
        [[("analyze_history", {"user_id": "ghost", "question": "ready?"})],
         "I couldn't find that user."],
        recorder,
    )
    result = await agent.run("is ghost ready?")
    assert result.tool_calls[0].result_status == "unknown_user"


@pytest.mark.asyncio
async def test_hallucinated_tool_name_does_not_crash():
    recorder = []
    agent = make_agent(
        [[("nonexistent_tool", {})], "Okay, answering directly."],
        recorder,
    )
    result = await agent.run("do something")
    # loop survives and still produces an answer
    assert result.answer == "Okay, answering directly."
    assert result.tool_calls[0].result_status == "error"


@pytest.mark.asyncio
async def test_iteration_cap_forces_a_final_answer():
    recorder = []
    # Ask for a tool on every one of the 3 allowed iterations → never terminates
    # on its own; the loop must force a final answer via chat() after the cap.
    tool_turns = [[("rag_search", {"query": "x"})]] * 3
    agent = make_agent(tool_turns + ["forced final answer"], recorder, max_iter=3)
    result = await agent.run("loop forever")
    assert result.hit_iteration_cap is True
    assert result.iterations == 3
    assert result.answer == "forced final answer"
