"""Knowledge-base ingestion: load -> chunk -> embed -> persist."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.llm.client import LLMClient, Usage
from app.rag.chunking import Chunk, load_knowledge_base
from app.rag.store import VectorStore

logger = logging.getLogger(__name__)

# The gateway accepts batched input; batching keeps ingest to a handful of
# round trips instead of one per chunk.
EMBED_BATCH_SIZE = 32


@dataclass
class IngestReport:
    documents: int
    chunks: int
    usage: Usage

    def as_dict(self, settings: Settings) -> dict[str, object]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            **self.usage.as_dict(settings),
        }


async def ingest(
    *,
    client: LLMClient,
    store: VectorStore,
    settings: Settings | None = None,
    rebuild: bool = True,
) -> IngestReport:
    settings = settings or get_settings()
    chunks: list[Chunk] = load_knowledge_base(settings.knowledge_base_dir)
    documents = len({c.doc_id for c in chunks})
    logger.info("chunked %d documents into %d sections", documents, len(chunks))

    if rebuild:
        # Ingest is idempotent: re-running replaces the collection rather than
        # appending duplicate ids with stale text.
        store.reset()

    usage = Usage()
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        vectors = await client.embed([c.embed_text for c in batch], usage=usage)

        if vectors and len(vectors[0]) != settings.embedding_dim:
            raise ValueError(
                f"embedding dimension mismatch: gateway returned "
                f"{len(vectors[0])}, config expects {settings.embedding_dim}"
            )

        store.add(batch, vectors)
        logger.info("embedded %d/%d chunks", min(start + len(batch), len(chunks)), len(chunks))

    return IngestReport(documents=documents, chunks=len(chunks), usage=usage)
