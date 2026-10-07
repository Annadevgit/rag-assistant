"""
Generate an answer from the retrieved chunks.

Two modes, just like for embeddings:

- `extractive` (default, offline): does not use an LLM; instead, it directly
  returns the most relevant passages with light formatting.
  This allows the entire pipeline (retrieval + display) to be tested without
  an API key. It is an intentional degradation: less fluent than true
  generation, but guarantees that no information is invented (no hallucination
  is possible by design, since we only quote the source text).

- `llm`: calls a real LLM (Claude via the Anthropic API here) with the chunks
  as context, in order to produce a written and synthesized answer.
  Requires the ANTHROPIC_API_KEY environment variable.
"""

import os

from src.chunking import Chunk

SYSTEM_PROMPT = """You are an assistant that answers questions ONLY from the provided context.

Strict rules:
- If the answer is not found in the context, explicitly say that you do not know; do not make anything up.
- Cite the source section of your answer in parentheses, e.g. (source: Return Policy > Return Period).
- Answer concisely and directly, in 2-4 sentences maximum unless the question requires more detail.
"""


def generate_extractive(query: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
    """Extractive generation: directly return the most relevant passages."""
    if not retrieved_chunks:
        return "No relevant passage was found in the document database for this question."

    best_chunk, best_score = retrieved_chunks[0]

    lines = [
        f'According to "{best_chunk.doc_title}" (section: {best_chunk.section_title}):',
        "",
        best_chunk.text.strip(),
    ]

    if len(retrieved_chunks) > 1:
        lines.append("")
        lines.append("Additional potentially relevant passages:")
        for chunk, score in retrieved_chunks[1:3]:
            lines.append(
                f"- {chunk.doc_title} > {chunk.section_title} (score: {score:.2f})"
            )

    return "\n".join(lines)


def generate_llm(
    query: str,
    retrieved_chunks: list[tuple[Chunk, float]],
    model: str = "claude-sonnet-4-6",
) -> str:
    """Generate an answer via the Anthropic API using the retrieved chunks as context."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export your API key or use "
            "generate_extractive() to test without an API key."
        )

    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "The 'anthropic' package is not installed. Run `pip install anthropic`."
        ) from e

    if not retrieved_chunks:
        return "No relevant passage was found in the document database for this question."

    context = "\n\n---\n\n".join(
        f"[{chunk.doc_title} > {chunk.section_title}]\n{chunk.text}"
        for chunk, _ in retrieved_chunks
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {query}",
            }
        ],
    )

    return response.content[0].text


def generate_answer(
    query: str,
    retrieved_chunks: list[tuple[Chunk, float]],
    mode: str = "extractive",
) -> str:
    """Single entry point that selects the generation mode."""
    if mode == "extractive":
        return generate_extractive(query, retrieved_chunks)
    if mode == "llm":
        return generate_llm(query, retrieved_chunks)
    raise ValueError(
        f"Unknown generation mode: {mode}. "
        "Available choices: 'extractive', 'llm'."
    )