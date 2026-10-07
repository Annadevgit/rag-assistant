"""Ingest raw documents (Markdown/text) from the corpus directory."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """A raw document, before being split into chunks."""
    doc_id: str
    source_path: str
    title: str
    content: str


def load_corpus(corpus_dir: str = "data/corpus") -> list[Document]:
    """Load all .md and .txt files from a directory as documents."""
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}. "
            "Place your documents (.md or .txt) in this directory."
        )

    documents = []
    for file_path in sorted(corpus_path.glob("**/*")):
        if file_path.suffix.lower() not in (".md", ".txt"):
            continue

        content = file_path.read_text(encoding="utf-8")
        title = _extract_title(content, fallback=file_path.stem)

        documents.append(
            Document(
                doc_id=file_path.stem,
                source_path=str(file_path),
                title=title,
                content=content,
            )
        )

    if not documents:
        raise ValueError(
            f"No .md or .txt documents found in {corpus_dir}. "
            "Make sure the directory contains the required files."
        )

    return documents


def _extract_title(content: str, fallback: str) -> str:
    """Extract the title from the first Markdown heading (# Title), otherwise use the filename."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback.replace("_", " ").title()


if __name__ == "__main__":
    docs = load_corpus()
    print(f"{len(docs)} documents loaded:")
    for doc in docs:
        print(f'  - {doc.doc_id}: "{doc.title}" ({len(doc.content)} characters)')
