import re
from dataclasses import dataclass

from src.ingest import Document

CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30


@dataclass
class Chunk:
    """A document fragment, serving as the basic unit for indexing and retrieval."""
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str
    text: str


def chunk_document(doc: Document) -> list[Chunk]:
    """Split a document into chunks, using Markdown sections when possible."""
    sections = _split_by_markdown_headers(doc.content)

    chunks = []
    chunk_index = 0
    for section_title, section_text in sections:
        section_text = section_text.strip()
        if not section_text:
            continue

        word_count = len(section_text.split())
        if word_count <= CHUNK_SIZE_WORDS:
            # The section fits into a single chunk
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::chunk_{chunk_index}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    section_title=section_title,
                    text=section_text,
                )
            )
            chunk_index += 1
        else:
            # Section is too long: fall back to word-based splitting with overlap
            for sub_text in _split_with_overlap(section_text):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::chunk_{chunk_index}",
                        doc_id=doc.doc_id,
                        doc_title=doc.title,
                        section_title=section_title,
                        text=sub_text,
                    )
                )
                chunk_index += 1

    return chunks


def _split_by_markdown_headers(content: str) -> list[tuple[str, str]]:
    """Split Markdown text into (section_title, content) pairs at ## headings."""
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(content))

    if not matches:
        # No ## headings: treat the entire document as a single "section"
        return [("", content)]

    sections = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append((title, content[start:end]))

    return sections


def _split_with_overlap(text: str) -> list[str]:
    """Split long text into chunks of CHUNK_SIZE_WORDS words with overlap."""
    words = text.split()
    chunks = []
    start = 0
    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

    while start < len(words):
        chunk_words = words[start : start + CHUNK_SIZE_WORDS]
        chunks.append(" ".join(chunk_words))
        start += step

    return chunks


def chunk_corpus(documents: list[Document]) -> list[Chunk]:
    """Split all documents in a corpus into chunks."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


if __name__ == "__main__":
    from src.ingest import load_corpus

    docs = load_corpus()
    chunks = chunk_corpus(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    for c in chunks[:5]:
        print(
            f'  [{c.chunk_id}] section="{c.section_title}" '
            f"({len(c.text.split())} words)"
        )
