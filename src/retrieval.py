"""Retrieval: encode a query and retrieve the most relevant chunks."""

import pickle
from pathlib import Path

from src.chunking import Chunk
from src.embeddings import EmbeddingBackend
from src.vector_store import VectorStore


class Retriever:
    def __init__(self, index_dir: str = "data/index"):
        self.index_dir = index_dir
        self.store = VectorStore()
        self.backend: EmbeddingBackend | None = None

    def load(self) -> None:
        self.store.load(self.index_dir)
        backend_path = Path(self.index_dir) / "backend.pkl"
        if not backend_path.exists():
            raise FileNotFoundError(
                f"Embedding backend not found in {self.index_dir}. "
                "Run `python -m src.build_index` first."
            )
        with open(backend_path, "rb") as f:
            self.backend = pickle.load(f)

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k most relevant chunks for a query, with their scores."""
        if self.backend is None:
            raise RuntimeError("Call .load() before .retrieve()")

        query_embedding = self.backend.encode([query])[0]
        return self.store.search(query_embedding, top_k=top_k)


if __name__ == "__main__":
    retriever = Retriever()
    retriever.load()

    test_query = "How long do I have to return an item?"
    results = retriever.retrieve(test_query, top_k=3)

    print(f'Query: "{test_query}"\n')
    for chunk, score in results:
        print(f"[{score:.3f}] {chunk.doc_title} > {chunk.section_title}")
        print(f"  {chunk.text[:150]}...\n")