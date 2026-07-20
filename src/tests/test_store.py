"""VectorStore search contract, exercised against a real (tiny) Chroma index."""

from __future__ import annotations

import pytest

from app.rag.chunking import Chunk
from app.rag.store import VectorStore


def make_chunk(doc: str, section: str) -> Chunk:
    return Chunk(
        id=f"{doc}#{section}",
        doc_id=doc,
        doc_title=doc.title(),
        section=section,
        body=f"body of {doc} {section}",
        source=f"{doc}.md",
    )


@pytest.fixture
def store(tmp_path):
    s = VectorStore(directory=tmp_path / "chroma", collection_name="test")
    chunks = [
        make_chunk("deload", "what"),
        make_chunk("deload", "when"),
        make_chunk("nutrition", "protein"),
    ]
    # Deterministic 3-dim vectors: deload chunks near [1,0,0], nutrition near [0,1,0].
    vectors = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]
    store_dim_patch(s)
    s.add(chunks, vectors)
    return s


def store_dim_patch(_s):
    # The fixture uses 3-dim vectors; nothing to patch, kept for clarity.
    return None


def test_search_ranks_by_similarity(store):
    hits = store.search([1.0, 0.0, 0.0], top_k=3)
    assert hits[0].doc_id == "deload"
    assert hits[0].similarity >= hits[1].similarity >= hits[2].similarity


def test_search_respects_top_k(store):
    assert len(store.search([1.0, 0.0, 0.0], top_k=1)) == 1


def test_search_returns_metadata_for_attribution(store):
    top = store.search([0.0, 1.0, 0.0], top_k=1)[0]
    assert top.doc_id == "nutrition"
    assert top.source == "nutrition.md"
    assert top.section == "protein"
    assert top.citation == "nutrition.md#protein"


def test_similarity_is_higher_for_closer_vectors(store):
    hits = store.search([1.0, 0.0, 0.0], top_k=3)
    by_id = {h.doc_id: h.similarity for h in hits}
    assert by_id["deload"] > by_id["nutrition"]


def test_reset_empties_the_collection(store):
    assert store.count() == 3
    store.reset()
    assert store.count() == 0


def test_add_rejects_length_mismatch(tmp_path):
    s = VectorStore(directory=tmp_path / "c", collection_name="t")
    with pytest.raises(ValueError):
        s.add([make_chunk("a", "b")], [[1.0], [2.0]])
