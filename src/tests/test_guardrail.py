"""Guardrail classification logic and its integration into the answer flow.

The classifier's *accuracy* against real questions is measured by the eval
pipeline (Feature 4). These tests stub the model so they assert our control
flow: category -> refusal mapping, fail-open on malformed output, and that
safety runs before the relevance check.
"""

from __future__ import annotations

import pytest

from app.guardrail.classifier import (
    REFUSAL_MESSAGES,
    Guardrail,
    GuardrailCategory,
)
from app.rag.answer import RagAnswerer
from app.rag.store import RetrievedChunk


class StubClient:
    """Returns a canned structured payload; records whether generation ran."""

    def __init__(self, classification: str):
        self._classification = classification
        self.structured_calls: list[str] = []

    async def embed(self, texts, usage=None):
        return [[0.0] * 1024 for _ in texts]

    async def structured(self, messages, schema, *, schema_name="", usage=None, **kw):
        self.structured_calls.append(schema_name)
        if schema_name == "safety_classification":
            return self._classification
        # answer generation
        return '{"answer": "generated", "used_sources": [1]}'


class StubStore:
    def __init__(self, sim: float):
        self._sim = sim

    def search(self, embedding, *, top_k):
        return [
            RetrievedChunk(
                id="03-deadlift#common-mistakes",
                text="Rounding the lower back",
                doc_id="03-deadlift",
                doc_title="Deadlift",
                section="Common Mistakes",
                source="03-deadlift.md",
                similarity=self._sim,
            )
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category",
    [
        GuardrailCategory.MEDICAL_DIAGNOSIS,
        GuardrailCategory.INJURY_REHAB,
        GuardrailCategory.EATING_DISORDER,
    ],
)
async def test_unsafe_categories_are_refused_with_redirect(category):
    client = StubClient(f'{{"category": "{category.value}", "reason": "x"}}')
    verdict = await Guardrail(client=client).classify("q")

    assert verdict.allowed is False
    assert verdict.category is category
    assert verdict.message == REFUSAL_MESSAGES[category]
    # Redirects, not dead-ends: each names a professional.
    assert any(w in verdict.message.lower() for w in ("doctor", "physio", "dietitian"))


@pytest.mark.asyncio
async def test_allow_passes_through():
    client = StubClient('{"category": "allow", "reason": "technique"}')
    verdict = await Guardrail(client=client).classify("how do I squat?")
    assert verdict.allowed is True
    assert verdict.message is None


@pytest.mark.asyncio
async def test_malformed_classification_fails_open():
    """Blocking every question on a parse error is worse over-restriction than
    the brief warns against; downstream answers stay grounded and cited."""
    client = StubClient("not json")
    verdict = await Guardrail(client=client).classify("how do I squat?")
    assert verdict.allowed is True


@pytest.mark.asyncio
async def test_guardrail_runs_before_relevance_check():
    """A medical question is topically in scope (high similarity), so the
    relevance gate would let it through. Safety must fire first — and no
    generation may happen."""
    client = StubClient('{"category": "injury_rehab", "reason": "rehab"}')
    answerer = RagAnswerer(
        client=client,
        store=StubStore(sim=0.8),  # comfortably above threshold
        guardrail=Guardrail(client=client),
    )

    result = await answerer.answer("How do I rehab my torn rotator cuff?")

    assert result.status == "refused"
    assert result.refusal_category == "injury_rehab"
    assert result.sources == []
    assert "grounded_answer" not in client.structured_calls, "must not generate"


@pytest.mark.asyncio
async def test_allowed_question_proceeds_to_grounded_answer():
    client = StubClient('{"category": "allow", "reason": "technique"}')
    answerer = RagAnswerer(
        client=client,
        store=StubStore(sim=0.8),
        guardrail=Guardrail(client=client),
    )

    result = await answerer.answer("How do I deadlift safely?")

    assert result.status == "answered"
    assert result.in_scope is True
    assert "grounded_answer" in client.structured_calls


@pytest.mark.asyncio
async def test_no_guardrail_configured_skips_classification():
    client = StubClient('{"category": "allow", "reason": ""}')
    answerer = RagAnswerer(client=client, store=StubStore(sim=0.8), guardrail=None)
    result = await answerer.answer("How do I deadlift?")
    assert result.status == "answered"
    assert "safety_classification" not in client.structured_calls
