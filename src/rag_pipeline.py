"""Complete RAG pipeline: question -> retrieval -> answer generation."""

from dataclasses import dataclass

from src.chunking import Chunk
from src.generation import generate_answer
from src.retrieval import Retriever


@dataclass
class RagResponse:
    query: str
    answer: str
    sources: list[tuple[Chunk, float]]


class RagPipeline:
    def __init__(
        self,
        index_dir: str = "data/index",
        generation_mode: str = "extractive",
    ):
        self.retriever = Retriever(index_dir)
        self.generation_mode = generation_mode
        self._loaded = False

    def load(self) -> None:
        self.retriever.load()
        self._loaded = True

    def ask(self, query: str, top_k: int = 5) -> RagResponse:
        if not self._loaded:
            raise RuntimeError("Call .load() before .ask()")

        retrieved = self.retriever.retrieve(query, top_k=top_k)
        answer = generate_answer(
            query,
            retrieved,
            mode=self.generation_mode,
        )

        return RagResponse(
            query=query,
            answer=answer,
            sources=retrieved,
        )


if __name__ == "__main__":
    pipeline = RagPipeline()
    pipeline.load()

    test_questions = [
        "How long do I have to return an item?",
        "How does the loyalty programme work?",
        "What are the standard delivery times?",
    ]

    for question in test_questions:
        response = pipeline.ask(question, top_k=3)
        print(f"Q: {response.query}")
        print(f"R: {response.answer}")
        print(f"Sources: {[c.chunk_id for c, _ in response.sources]}")
        print("-" * 80)