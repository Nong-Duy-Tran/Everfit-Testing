"""Chroma-backed vector store.

Chroma is embedded rather than a separate service, which keeps
`docker compose up` a single command. At ~111 chunks its scaling ceiling is
irrelevant; the persistence and cosine search are all that is needed.

Embeddings are computed by the project gateway and passed in explicitly — no
Chroma embedding function is registered, so the store never silently falls back
to its bundled default model (which would produce a different vector space
from the one used at query time).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.rag.chunking import Chunk

logger = logging.getLogger(__name__)

# chromadb 0.6.x raises inside its own telemetry hook even when telemetry is
# disabled, emitting an ERROR line per collection access. Nothing is broken and
# nothing is sent; silence it so real errors stay visible.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from a similarity search."""

    id: str
    text: str
    doc_id: str
    doc_title: str
    section: str
    source: str
    similarity: float

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.section}"


class VectorStore:
    def __init__(self, *, directory: Path, collection_name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_name = collection_name

    @property
    def _collection(self):
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop the collection so ingest is idempotent rather than additive."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # collection may not exist yet
            logger.debug("no existing collection %s to delete", self._collection_name)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            documents=[c.embed_text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.to_metadata() for c in chunks],
        )

    def search(self, embedding: list[float], *, top_k: int) -> list[RetrievedChunk]:
        """Return the `top_k` most similar chunks, ranked by cosine similarity.

        Plain top-k, deliberately. Earlier iterations added a per-document cap
        and then MMR to stop one document from taking every slot on multi-topic
        questions ("should I deload or eat more?"). Both were measured and both
        failed: no fixed setting is right for both that question and a
        single-topic deep-dive ("tell me all about deload") — the two produce
        near-identical candidate lists but want opposite results. The signal
        that separates them is query intent, which lives in the question, not in
        the chunks, so no retrieval-time reranking can recover it. Query
        decomposition is the real fix; it's out of scope for Feature 1. See
        EVALUATION.md.
        """
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, max(self.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            RetrievedChunk(
                id=cid,
                text=doc,
                doc_id=str(meta.get("doc_id", "")),
                doc_title=str(meta.get("doc_title", "")),
                section=str(meta.get("section", "")),
                source=str(meta.get("source", "")),
                similarity=1.0 - float(dist),
            )
            for cid, doc, meta, dist in zip(ids, documents, metadatas, distances)
        ]
