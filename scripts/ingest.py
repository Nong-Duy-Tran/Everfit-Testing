"""Build the vector store from the knowledge base.

Run:  PYTHONPATH=src python scripts/ingest.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

from app.config import get_settings
from app.llm.client import LLMClient
from app.rag.ingest import ingest
from app.rag.store import VectorStore


async def main() -> int:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    client = LLMClient(settings)
    store = VectorStore(
        directory=settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )
    try:
        report = await ingest(client=client, store=store, settings=settings)
    finally:
        await client.aclose()

    print(f"\ningested: {report.as_dict(settings)}")
    print(f"collection now holds {store.count()} chunks at {settings.chroma_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
