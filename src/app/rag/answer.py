"""Grounded answer generation over retrieved knowledge-base chunks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.llm.client import LLMClient, Usage
from app.rag.store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a knowledgeable strength-training assistant for a fitness coaching \
platform. You answer using ONLY the numbered sources provided.

Rules:
- Ground every claim in the sources. Never add facts, numbers, or formulas that \
are not present in them.
- Cite inline with the source number, e.g. "keep elbows at 45-75 degrees [2]".
- If the sources only partially cover the question, answer what they support and \
state plainly what they do not cover. Do not fill the gap from memory.
- Be concise and practical, in the voice of a coach talking to a client.
- List in `used_sources` only the numbers you actually drew on.\
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The grounded answer, with inline [n] citations.",
        },
        "used_sources": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "1-based numbers of the sources actually used.",
        },
    },
    "required": ["answer", "used_sources"],
    "additionalProperties": False,
}

OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions about strength training — exercise technique, "
    "programming, recovery, and nutrition basics. That question falls outside "
    "the material I have, so I'd rather not guess. Ask me about lifting form, "
    "workout splits, progressive overload, or recovery and I can help."
)


@dataclass
class Source:
    number: int
    chunk_id: str
    document: str
    section: str
    similarity: float
    used: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "chunk_id": self.chunk_id,
            "document": self.document,
            "section": self.section,
            "similarity": round(self.similarity, 4),
            "used": self.used,
        }


@dataclass
class AnswerResult:
    answer: str
    sources: list[Source]
    in_scope: bool
    usage: Usage

    def as_dict(self, settings: Settings) -> dict[str, object]:
        return {
            "answer": self.answer,
            "sources": [s.as_dict() for s in self.sources],
            "in_scope": self.in_scope,
            "usage": self.usage.as_dict(settings),
        }


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.source} — {c.section})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )


class RagAnswerer:
    def __init__(
        self,
        *,
        client: LLMClient,
        store: VectorStore,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._settings = settings or get_settings()

    async def answer(self, question: str) -> AnswerResult:
        settings = self._settings
        usage = Usage()

        query_vector = (await self._client.embed([question], usage=usage))[0]
        chunks = self._store.search(query_vector, top_k=settings.retrieval_top_k)

        best = chunks[0].similarity if chunks else 0.0
        if best < settings.relevance_threshold:
            # Refuse before spending a completion. An out-of-scope question can
            # never produce a fitness answer, because no generation happens.
            logger.info(
                "out of scope | best_similarity=%.3f threshold=%.2f | %r",
                best, settings.relevance_threshold, question,
            )
            return AnswerResult(
                answer=OUT_OF_SCOPE_MESSAGE,
                sources=[],
                in_scope=False,
                usage=usage,
            )

        raw = await self._client.structured(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Sources:\n\n{_format_context(chunks)}\n\nQuestion: {question}",
                },
            ],
            ANSWER_SCHEMA,
            schema_name="grounded_answer",
            usage=usage,
        )

        try:
            payload = json.loads(raw)
            answer = str(payload["answer"]).strip()
            used = {int(n) for n in payload.get("used_sources", [])}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Strict schema makes this near-impossible, but a malformed payload
            # must not 500 — degrade to the raw text and mark all sources used.
            logger.warning("could not parse structured answer, falling back to raw text")
            answer = raw.strip()
            used = set(range(1, len(chunks) + 1))

        sources = [
            Source(
                number=i,
                chunk_id=c.id,
                document=c.source,
                section=c.section,
                similarity=c.similarity,
                used=i in used,
            )
            for i, c in enumerate(chunks, start=1)
        ]
        return AnswerResult(answer=answer, sources=sources, in_scope=True, usage=usage)
