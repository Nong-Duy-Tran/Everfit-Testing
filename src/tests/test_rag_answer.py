"""Answer-layer behaviour, tested without hitting the gateway.

Retrieval scores and model output are stubbed so these assert *our* control
flow — the out-of-scope gate, citation mapping, and malformed-payload
degradation — rather than model quality. Model quality is measured separately
by the evaluation pipeline (Feature 4).
"""

from __future__ import annotations

import json

import pytest

from app.config import get_settings
from app.rag.answer import OUT_OF_SCOPE_MESSAGE, RagAnswerer
from app.rag.store import RetrievedChunk


def chunk(idx: int, similarity: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"doc-{idx}#section",
        text=f"body {idx}",
        doc_id=f"doc-{idx}",
        doc_title=f"Doc {idx}",
        section="Section",
        source=f"doc-{idx}.md",
        similarity=similarity,
    )


class StubClient:
    def __init__(self, structured_response: str = '{"answer": "a", "used_sources": [1]}'):
        self.structured_response = structured_response
        self.structured_calls = 0

    async def embed(self, texts, usage=None):
        return [[0.0] * 1024 for _ in texts]

    async def structured(self, messages, schema, *, schema_name="", usage=None, **kw):
        self.structured_calls += 1
        self.last_messages = messages
        return self.structured_response


class StubStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, embedding, *, top_k):
        return self._chunks[:top_k]


@pytest.mark.asyncio
async def test_out_of_scope_question_never_generates():
    """The brief requires that "What's the weather today?" must not produce a
    fitness answer. Guaranteed structurally: below threshold, no completion is
    requested at all."""
    client = StubClient()
    answerer = RagAnswerer(client=client, store=StubStore([chunk(1, 0.29)]))

    result = await answerer.answer("What's the weather today?")

    assert result.in_scope is False
    assert result.answer == OUT_OF_SCOPE_MESSAGE
    assert result.sources == []
    assert client.structured_calls == 0, "no generation may happen out of scope"


@pytest.mark.asyncio
async def test_in_scope_question_returns_cited_sources():
    client = StubClient('{"answer": "Grip wider [1].", "used_sources": [1, 3]}')
    answerer = RagAnswerer(
        client=client,
        store=StubStore([chunk(1, 0.81), chunk(2, 0.65), chunk(3, 0.60)]),
    )

    result = await answerer.answer("How do I bench press?")

    assert result.in_scope is True
    assert result.answer == "Grip wider [1]."
    assert [s.number for s in result.sources] == [1, 2, 3]
    assert [s.used for s in result.sources] == [True, False, True]
    assert result.sources[0].chunk_id == "doc-1#section"


@pytest.mark.asyncio
async def test_sources_are_always_returned_for_attribution():
    """Every in-scope answer must carry references, per Feature 1."""
    answerer = RagAnswerer(client=StubClient(), store=StubStore([chunk(1, 0.9)]))
    result = await answerer.answer("How do I squat?")
    assert len(result.sources) >= 1


@pytest.mark.asyncio
async def test_malformed_model_payload_degrades_instead_of_raising():
    answerer = RagAnswerer(
        client=StubClient("not json at all"),
        store=StubStore([chunk(1, 0.9), chunk(2, 0.7)]),
    )
    result = await answerer.answer("How do I squat?")
    assert result.answer == "not json at all"
    assert all(s.used for s in result.sources)


@pytest.mark.asyncio
async def test_prompt_contains_only_retrieved_context():
    """Guards against leaking unretrieved knowledge-base content into the prompt."""
    client = StubClient()
    answerer = RagAnswerer(client=client, store=StubStore([chunk(1, 0.9)]))
    await answerer.answer("How do I squat?")

    user_message = client.last_messages[-1]["content"]
    assert "body 1" in user_message
    assert "body 2" not in user_message


@pytest.mark.asyncio
async def test_empty_retrieval_is_treated_as_out_of_scope():
    client = StubClient()
    answerer = RagAnswerer(client=client, store=StubStore([]))
    result = await answerer.answer("anything")
    assert result.in_scope is False
    assert client.structured_calls == 0


def test_threshold_sits_between_measured_in_and_out_of_scope_scores():
    """Empirically measured on the real corpus: out-of-scope questions top out
    at 0.292 (weather) and 0.260 (world cup); the weakest genuine question
    ("How much protein should I eat?") scores 0.434."""
    threshold = get_settings().relevance_threshold
    assert 0.292 < threshold < 0.434
