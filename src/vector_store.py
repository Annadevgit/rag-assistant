"""
Minimal in-memory vector store with disk persistence.

Deliberate choice: use a NumPy index instead of FAISS/Chroma/Qdrant.

For a corpus of a few thousand chunks (which covers the vast majority of
enterprise RAG projects built on a targeted document collection), brute-force
cosine similarity search with NumPy is more than fast enough (a few
milliseconds) and avoids a heavy dependency that needs to be compiled or
run as a separate service.

For a corpus of several million chunks, FAISS or a dedicated vector database
(Chroma, Qdrant, Pinecone) would become necessary for large-scale approximate
search (ANN — Approximate Nearest Neighbors). The extension point is clearly
identified: `VectorStore.search()` is the only method that needs to be
replaced with a FAISS/Chroma call if the corpus grows.
"""

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.chunking import Chunk


class VectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Build the index from chunks and their pre-computed embeddings."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Number of chunks ({len(chunks)}) != number of embeddings ({embeddings.shape[0]})"
            )
        self.chunks = chunks
        # L2 normalization so that the dot product = cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # prevent division by zero for a zero vector
        self.embeddings = embeddings / norms

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks most similar to the query, with their scores."""
        if self.embeddings is None or len(self.chunks) == 0:
            raise RuntimeError(
                "The vector store is empty. Call .build() or .load() first."
            )

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        scores = (
            self.embeddings @ query_norm
        )  # cosine similarity, since everything is already normalized

        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def save(self, path: str = "data/index") -> None:
        """Save the index to disk (embeddings in .npy, metadata in .json)."""
        index_dir = Path(path)
        index_dir.mkdir(parents=True, exist_ok=True)

        np.save(index_dir / "embeddings.npy", self.embeddings)
        with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(
                [asdict(c) for c in self.chunks],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: str = "data/index") -> None:
        """Load a previously saved index."""
        index_dir = Path(path)
        embeddings_path = index_dir / "embeddings.npy"
        chunks_path = index_dir / "chunks.json"

        if not embeddings_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Index not found in {path}. "
                "Run `python -m src.build_index` first."
            )

        self.embeddings = np.load(embeddings_path)
        with open(chunks_path, encoding="utf-8") as f:
            raw_chunks = json.load(f)
        self.chunks = [Chunk(**c) for c in raw_chunks]