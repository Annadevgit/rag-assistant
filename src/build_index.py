"""Build the complete vector index: ingestion -> chunking -> embeddings -> vector store."""

import argparse
import pickle
from pathlib import Path

from src.chunking import chunk_corpus
from src.embeddings import get_backend
from src.ingest import load_corpus
from src.vector_store import VectorStore


def build_index(
    corpus_dir: str = "data/corpus",
    index_dir: str = "data/index",
    backend_name: str = "tfidf",
):
    print(f"1/4 — Loading corpus from {corpus_dir}...")
    documents = load_corpus(corpus_dir)
    print(f"     {len(documents)} documents loaded")

    print("2/4 — Splitting into chunks...")
    chunks = chunk_corpus(documents)
    print(f"     {len(chunks)} chunks generated")

    print(f"3/4 — Computing embeddings (backend: {backend_name})...")
    backend = get_backend(backend_name)
    texts = [c.text for c in chunks]
    backend.fit(texts)
    embeddings = backend.encode(texts)
    print(f"     Embeddings with dimension {embeddings.shape[1]}")

    print(f"4/4 — Building and saving index to {index_dir}...")
    store = VectorStore()
    store.build(chunks, embeddings)
    store.save(index_dir)

    # The TF-IDF backend must also be saved: it is used to encode
    # future queries, and it must use exactly the same vocabulary
    # learned from the corpus; otherwise, the dimensions will no longer match.
    with open(Path(index_dir) / "backend.pkl", "wb") as f:
        pickle.dump(backend, f)

    print("\nIndex built successfully.")
    return store, backend


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument(
        "--backend",
        default="tfidf",
        choices=["tfidf", "sentence-transformers"],
    )
    args = parser.parse_args()

    build_index(args.corpus_dir, args.index_dir, args.backend)