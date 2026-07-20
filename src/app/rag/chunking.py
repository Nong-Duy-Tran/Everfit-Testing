from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
H2_SPLIT_RE = re.compile(r"^##\s+", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """One retrievable section of a knowledge-base document."""

    id: str
    doc_id: str
    doc_title: str
    section: str
    body: str
    source: str

    @property
    def embed_text(self) -> str:
        """Text actually embedded and shown to the model.

        The `title — section` prefix is what disambiguates otherwise
        near-identical sections across documents.
        """
        return f"{self.doc_title} — {self.section}\n\n{self.body}"

    def to_metadata(self) -> dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section": self.section,
            "source": self.source,
        }


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def chunk_markdown(text: str, *, doc_id: str, source: str) -> list[Chunk]:
    """Split one markdown document into section chunks."""
    h1 = H1_RE.search(text)
    doc_title = h1.group(1).strip() if h1 else doc_id

    # Everything before the first `##` is the document preamble. Usually just
    # the title, but keep it as an "Overview" chunk when it carries prose.
    parts = H2_SPLIT_RE.split(text)
    preamble = H1_RE.sub("", parts[0]).strip()

    chunks: list[Chunk] = []
    seen: set[str] = set()

    def add(section: str, body: str) -> None:
        body = body.strip()
        if not body:
            return
        base = f"{doc_id}#{_slug(section)}"
        chunk_id = base
        suffix = 2
        while chunk_id in seen:  # duplicate headings within a document
            chunk_id = f"{base}-{suffix}"
            suffix += 1
        seen.add(chunk_id)
        chunks.append(
            Chunk(
                id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                section=section,
                body=body,
                source=source,
            )
        )

    if preamble:
        add("Overview", preamble)

    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        add(heading.strip(), body)

    return chunks


def load_knowledge_base(directory: Path) -> list[Chunk]:
    """Chunk every markdown document in `directory`, sorted for deterministic ids."""
    files = sorted(directory.glob("*.md"))
    if not files:
        raise FileNotFoundError(
            f"No markdown documents found in {directory}. "
            "The knowledge base is required for retrieval."
        )
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(
            chunk_markdown(
                path.read_text(encoding="utf-8"),
                doc_id=path.stem,
                source=path.name,
            )
        )
    return chunks
