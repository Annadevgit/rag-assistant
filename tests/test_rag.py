"""Unit tests for the RAG pipeline."""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.build_index import build_index
from src.chunking import chunk_document, chunk_corpus
from src.embeddings import get_backend
from src.ingest import Document, load_corpus
from src.rag_pipeline import RagPipeline
from src.vector_store import VectorStore


@pytest.fixture
def sample_document():
    return Document(
        doc_id="test_doc",
        source_path="test_doc.md",
        title="Test Document",
        content=(
            "# Test Document\n\n"
            "## First section\n"
            "This is the content of the first section, with enough "
            "text to be a valid and testable chunk.\n\n"
            "## Second section\n"
            "This is the content of the second section, completely "
            "different from the first one to test chunk separation.\n"
        ),
    )


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory with a test corpus and clean it up after the test."""
    tmp_dir = tempfile.mkdtemp()
    corpus_dir = Path(tmp_dir) / "corpus"
    index_dir = Path(tmp_dir) / "index"
    corpus_dir.mkdir()

    (corpus_dir / "doc1.md").write_text(
        "# Shipping\n\n## Delivery times\nStandard delivery takes 3 to 5 business days.\n",
        encoding="utf-8",
    )
    (corpus_dir / "doc2.md").write_text(
        "# Returns\n\n## Return period\nYou have 30 days to return an item.\n",
        encoding="utf-8",
    )

    yield str(corpus_dir), str(index_dir)

    shutil.rmtree(tmp_dir)


def test_load_corpus_reads_markdown_files(temp_project_dir):
    corpus_dir, _ = temp_project_dir
    docs = load_corpus(corpus_dir)
    assert len(docs) == 2
    assert {d.doc_id for d in docs} == {"doc1", "doc2"}


def test_load_corpus_raises_on_missing_dir():
    with pytest.raises(FileNotFoundError):
        load_corpus("directory_that_does_not_exist")


def test_extract_title_from_markdown_header(temp_project_dir):
    corpus_dir, _ = temp_project_dir
    docs = load_corpus(corpus_dir)
    titles = {d.title for d in docs}
    assert "Shipping" in titles
    assert "Returns" in titles


def test_chunk_document_splits_by_section(sample_document):
    chunks = chunk_document(sample_document)
    assert len(chunks) == 2
    assert chunks[0].section_title == "First section"
    assert chunks[1].section_title == "Second section"


def test_chunk_document_produces_unique_ids(sample_document):
    chunks = chunk_document(sample_document)
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_corpus_aggregates_all_documents(temp_project_dir):
    corpus_dir, _ = temp_project_dir
    docs = load_corpus(corpus_dir)
    chunks = chunk_corpus(docs)
    assert len(chunks) >= 2
    doc_ids_in_chunks = {c.doc_id for c in chunks}
    assert doc_ids_in_chunks == {"doc1", "doc2"}


def test_tfidf_backend_fit_and_encode():
    backend = get_backend("tfidf")
    texts = [
        "the cat eats a mouse",
        "the dog barks loudly",
        "the mouse runs away from the cat",
    ]
    backend.fit(texts)
    embeddings = backend.encode(texts)

    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] > 0


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError):
        get_backend("backend_that_does_not_exist")


def test_vector_store_search_returns_most_similar_first():
    import numpy as np
    from src.chunking import Chunk

    chunks = [
        Chunk("c1", "d1", "Doc 1", "Section A", "text about cats"),
        Chunk("c2", "d1", "Doc 1", "Section B", "text about dogs"),
    ]

    # Toy embeddings: c1 is very close to the query, c2 is orthogonal
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    store = VectorStore()
    store.build(chunks, embeddings)

    results = store.search(
        query_embedding=np.array([1.0, 0.0]),
        top_k=2,
    )

    assert results[0][0].chunk_id == "c1"
    assert results[0][1] > results[1][1]


def test_vector_store_raises_on_mismatched_lengths():
    import numpy as np
    from src.chunking import Chunk

    chunks = [
        Chunk("c1", "d1", "Doc 1", "Section A", "text")
    ]
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]]
    )  # 2 embeddings for 1 chunk

    store = VectorStore()

    with pytest.raises(ValueError):
        store.build(chunks, embeddings)


def test_end_to_end_pipeline_retrieves_correct_chunk(temp_project_dir):
    """Integration test: build a real index and verify that retrieval works."""
    corpus_dir, index_dir = temp_project_dir

    build_index(
        corpus_dir=corpus_dir,
        index_dir=index_dir,
        backend_name="tfidf",
    )

    pipeline = RagPipeline(
        index_dir=index_dir,
        generation_mode="extractive",
    )
    pipeline.load()

    response = pipeline.ask(
        "How many days do I have to return an item?",
        top_k=1,
    )

    assert response.sources[0][0].doc_id == "doc2"


def test_generate_extractive_handles_empty_results():
    from src.generation import generate_extractive

    result = generate_extractive("some question", [])
    assert "No relevant passage" in result