import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.rag_pipeline import RagPipeline


@dataclass
class EvalQuestion:
    question: str
    expected_chunk_id: str  # the chunk_id that MUST appear in the top-k for the test to succeed


# Minimal but realistic evaluation set, manually built from the demo corpus.
# In production, this set would grow with real user questions, annotated
# retrospectively based on which document actually answered their need.
EVAL_SET = [
    EvalQuestion("How long do I have to return an item?", "Refund_Policy::chunk_0"),
    EvalQuestion("Which items cannot be returned?", "Refund_Policy::chunk_1"),
    EvalQuestion("How long does it take to receive a refund?", "Refund_Policy::chunk_2"),
    EvalQuestion("How much does express shipping cost?", "Shipping_Policy::chunk_0"),
    EvalQuestion("What is the minimum amount for free shipping?", "Shipping_Policy::chunk_1"),
    EvalQuestion("What should I do if my package is lost?", "Shipping_Policy::chunk_2"),
    EvalQuestion("How many points do I earn per euro spent?", "Loyalty_Program::chunk_0"),
    EvalQuestion("What are the benefits of Gold status?", "Loyalty_Program::chunk_1"),
    EvalQuestion("How much do I earn from a referral?", "Loyalty_Program::chunk_3"),
]


def evaluate_retrieval(pipeline: RagPipeline, eval_set: list[EvalQuestion], top_k: int = 3) -> dict:
    """Calculate Recall@k: proportion of questions where the correct chunk is found in the top-k retrieved results."""
    results = []
    hits = 0

    for item in eval_set:
        retrieved = pipeline.retriever.retrieve(item.question, top_k=top_k)
        retrieved_ids = [chunk.chunk_id for chunk, _ in retrieved]
        hit = item.expected_chunk_id in retrieved_ids

        if hit:
            hits += 1

        results.append({
            "question": item.question,
            "expected_chunk_id": item.expected_chunk_id,
            "retrieved_ids": retrieved_ids,
            "hit": hit,
            "rank": retrieved_ids.index(item.expected_chunk_id) + 1 if hit else None,
        })

    recall_at_k = hits / len(eval_set)

    return {
        "top_k": top_k,
        "recall_at_k": round(recall_at_k, 3),
        "hits": hits,
        "total": len(eval_set),
        "details": results,
    }


def evaluate_faithfulness(pipeline: RagPipeline, eval_set: list[EvalQuestion]) -> dict:
    """
    Simple faithfulness heuristic: does the generated answer share
    significant vocabulary with the expected source chunk?

    Known limitation: this does not detect subtle hallucinations
    (invented numbers that "sound right"), only off-topic answers.
    For a rigorous production evaluation, an LLM judge comparing
    the generated answer with the source context would be required.
    """
    results = []

    for item in eval_set:
        response = pipeline.ask(item.question, top_k=3)
        answer_words = set(response.answer.lower().split())

        source_chunk = next(
            (c for c, _ in response.sources if c.chunk_id == item.expected_chunk_id), None
        )
        if source_chunk is None:
            results.append({
                "question": item.question,
                "faithfulness_score": 0.0,
                "reason": "expected chunk was not retrieved",
            })
            continue

        source_words = set(source_chunk.text.lower().split())
        overlap = len(answer_words & source_words) / max(len(source_words), 1)

        results.append({
            "question": item.question,
            "faithfulness_score": round(overlap, 3),
        })

    avg_score = sum(r["faithfulness_score"] for r in results) / len(results)

    return {"avg_faithfulness_score": round(avg_score, 3), "details": results}


def main():
    pipeline = RagPipeline(generation_mode="extractive")
    pipeline.load()

    print("=== Retrieval evaluation ===\n")
    retrieval_results = evaluate_retrieval(pipeline, EVAL_SET, top_k=3)
    print(f"Recall@{retrieval_results['top_k']} : {retrieval_results['recall_at_k']:.1%} "
          f"({retrieval_results['hits']}/{retrieval_results['total']})")

    for detail in retrieval_results["details"]:
        status = f"✓ (rank {detail['rank']})" if detail["hit"] else "✗ MISSED"
        print(f"  {status} — {detail['question']}")

    print("\n=== Generation faithfulness evaluation ===\n")
    faithfulness_results = evaluate_faithfulness(pipeline, EVAL_SET)
    print(f"Average faithfulness score: {faithfulness_results['avg_faithfulness_score']:.3f}")

    output = {"retrieval": retrieval_results, "faithfulness": faithfulness_results}
    Path("data/eval_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nResults saved: data/eval_results.json")


if __name__ == "__main__":
    main()