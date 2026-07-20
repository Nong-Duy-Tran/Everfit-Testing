"""Chunking behaviour that retrieval quality depends on."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.rag.chunking import chunk_markdown, load_knowledge_base

SAMPLE = """# Bench Press

Intro prose about the lift.

## Proper Form

1. Lie flat on a bench
2. Grip the bar

## Common Mistakes

- Flaring elbows
"""


def test_splits_on_h2_sections():
    chunks = chunk_markdown(SAMPLE, doc_id="01-bench-press", source="01-bench-press.md")
    sections = [c.section for c in chunks]
    assert sections == ["Overview", "Proper Form", "Common Mistakes"]


def test_chunk_ids_are_stable_and_readable():
    chunks = chunk_markdown(SAMPLE, doc_id="01-bench-press", source="01-bench-press.md")
    assert chunks[1].id == "01-bench-press#proper-form"


def test_embed_text_carries_document_title():
    """The load-bearing property: a bare 'Common Mistakes' body is ~30 words and
    reads near-identically across bench/squat/deadlift. Without the title prefix
    those chunks collapse together in embedding space."""
    chunks = chunk_markdown(SAMPLE, doc_id="01-bench-press", source="01-bench-press.md")
    mistakes = next(c for c in chunks if c.section == "Common Mistakes")
    assert mistakes.embed_text.startswith("Bench Press — Common Mistakes")


def test_duplicate_headings_get_unique_ids():
    text = "# Doc\n\n## Notes\n\nfirst\n\n## Notes\n\nsecond\n"
    ids = [c.id for c in chunk_markdown(text, doc_id="d", source="d.md")]
    assert len(ids) == len(set(ids))


def test_empty_sections_are_dropped():
    text = "# Doc\n\n## Empty\n\n## Real\n\nbody\n"
    sections = [c.section for c in chunk_markdown(text, doc_id="d", source="d.md")]
    assert sections == ["Real"]


def test_missing_knowledge_base_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_knowledge_base(tmp_path)


def test_real_knowledge_base_chunks_cleanly():
    chunks = load_knowledge_base(get_settings().knowledge_base_dir)
    assert len(chunks) == 111
    assert len({c.id for c in chunks}) == len(chunks)
    assert len({c.doc_id for c in chunks}) == 20
    # No section should be large enough to need sub-splitting.
    assert max(len(c.embed_text.split()) for c in chunks) < 400
