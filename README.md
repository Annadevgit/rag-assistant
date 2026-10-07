RAG Document Assistant

A question-answering system (RAG — Retrieval-Augmented Generation) that answers questions using only an indexed document database, with systematic source citation and rigorous evaluation of both retrieval and generation.

Context and Problem Statement

A chatbot based solely on an LLM can generate plausible but sometimes incorrect answers (hallucinations) when asked about specific information it has never seen (internal documentation, company policies, proprietary knowledge bases). RAG addresses this issue by forcing the model to answer based on passages actually retrieved from a document database, with source citations — and allows it to honestly answer "I don't know" when no relevant passage exists instead of making something up.

This project implements the complete pipeline end to end, with a particular focus on evaluation: a RAG system that "seems to work" on three examples is not sufficient — this project separately measures retrieval quality and generation faithfulness on an annotated test set.

Data

The demonstration corpus consists of 3 fictional but realistic Markdown documents (return policy, shipping policy, and loyalty program for an e-commerce website) — see data/corpus/. The pipeline is domain-agnostic: replace these files with your own .md or .txt documents (technical documentation, reports, articles) to adapt it to another use case.

Methodology

Ingestion (src/ingest.py): loads all .md/.txt files from a folder and extracts the title from the first Markdown heading.

Chunking (src/chunking.py): splits documents by Markdown section (##) when available — preserving natural semantic boundaries — with a fallback to word-based splitting with overlap (150 words, 30-word overlap) for sections that are too long. This size is an intentional trade-off: chunks that are too small lose context, while chunks that are too large make retrieval less precise.

Embeddings (src/embeddings.py): two interchangeable backends.

tfidf (default): offline, instant, lexical similarity.

sentence-transformers (optional): higher-quality semantic embeddings, requiring an internet connection when the model is loaded for the first time.

Vector Store (src/vector_store.py): an in-memory index (NumPy) with cosine similarity search and disk persistence. This is an intentional choice over FAISS/Chroma/Qdrant: it is more than sufficient for a corpus of a few thousand chunks, without requiring a heavy dependency to run.

Retrieval (src/retrieval.py): encodes the query using the same backend as the corpus and returns the top-k most similar chunks.

Generation (src/generation.py): two modes.

extractive (default): directly returns the relevant passages, with no LLM — zero hallucination by construction, allowing the entire pipeline to be tested without an API key.

llm: generates a written response through the Anthropic API (Claude), using a system prompt that explicitly prohibits inventing information outside the provided context.

Evaluation (src/evaluate.py): see the dedicated section below.

Interface (app/streamlit_app.py): chat interface with source display and relevance scores, with a switch between the two generation modes.

Evaluation

**Recall@3 (retrieval)**: on a hand-labeled set of 9 questions (question → chunk that
should be retrieved), **88.9% (8/9) of expected chunks are found in the top-3**
(see `src/evaluate.py`, `EVAL_SET`).

The one miss is informative rather than a bug, and worth detailing: for the question
*"Which items cannot be returned?"*, the expected chunk (`Refund_Policy::chunk_1`,
"Eligibility Conditions") shares no exact vocabulary with the question — it talks about
items that are "not eligible for return", not "cannot be returned". Instead, the top
result retrieved was `Loyalty_Program::chunk_4` ("Account Closure"), which contains the
literal phrase "points... **cannot** be refunded". This is TF-IDF's core limitation made
concrete: it matches exact words, not meaning, so a strong lexical overlap on "cannot"
outweighs actual topical relevance. This is exactly the kind of failure a semantic
embeddings backend (`sentence-transformers`) would avoid — see Limitations below.

**Faithfulness score (generation)**: measured via lexical overlap between the generated
answer and the expected source chunk — **0.710** on average on the test set in extractive
mode. This is a deliberately simple heuristic (documented as such); a production
evaluation would use an LLM-as-judge comparing answer and source context.

To reproduce these results:

python -m src.build_index
python -m src.evaluate

Limitations and Areas for Improvement

- **Default TF-IDF backend**: captures lexical similarity (shared words), not semantic
  similarity. Concrete example observed in evaluation: the question "Which items cannot
  be returned?" fails to retrieve the correct "Eligibility Conditions" section (which
  describes ineligible items without using the words "cannot" or "returned" together) and
  instead retrieves an unrelated loyalty-program section that happens to contain the
  literal word "cannot". The `sentence-transformers` backend would capture this kind of
  paraphrase correctly, at the cost of a network dependency.

Heuristic faithfulness evaluation: lexical overlap does not detect subtle hallucinations (such as invented but plausible numbers). An LLM judge would be required for rigorous production evaluation.

No conversation history management: each question is handled independently, without memory of previous exchanges — a true multi-turn chat interface would need to rewrite the query based on the conversation context.

Fixed-size fallback chunking: does not detect sentence boundaries and may cut a sentence in half in sections without Markdown subheadings.

Installation and Usage
git clone <repository-url>
cd rag-assistant
pip install -r requirements.txt

# 1. Build the index from the corpus (data/corpus/)
python -m src.build_index

# 2. Test the pipeline from the command line
python -m src.rag_pipeline

# 3. Run the evaluation (retrieval + faithfulness)
python -m src.evaluate

# 4. Launch the Streamlit interface
streamlit run app/streamlit_app.py


To enable generation through a real LLM (Claude):

pip install anthropic
export ANTHROPIC_API_KEY="your-api-key"


Then select the "llm" mode in the interface, or pass generation_mode="llm" to RagPipeline in code.

With Docker:

docker build -t rag-assistant .
docker run -p 8501:8501 -d 8501


Tests:

python -m pytest tests/ -v

Project Structure
rag-assistant/
├── data/
│   ├── corpus/               # Source documents (.md / .txt)
│   └── index/                # Generated index (not versioned, reconstructible)
├── src/
│   ├── ingest.py             # Document loading
│   ├── chunking.py           # Chunking
│   ├── embeddings.py         # Embedding backends (TF-IDF / sentence-transformers)
│   ├── vector_store.py       # In-memory vector index + persistence
│   ├── build_index.py        # Complete ingestion -> index pipeline
│   ├── retrieval.py          # Relevant chunk retrieval
│   ├── generation.py         # Answer generation (extractive / LLM)
│   ├── rag_pipeline.py       # Retrieval + generation orchestration
│   └── evaluate.py           # Retrieval (Recall@k) + faithfulness evaluation
├── app/
│   └── streamlit_app.py      # Chat interface
├── tests/
│   └── test_rag.py
├── .github/workflows/ci.yml  # Automated tests + evaluation on every push
├── Dockerfile
├── requirements.txt
└── README.md

Tech Stack

Python · scikit-learn · NumPy · Streamlit · (optional) sentence-transformers · (optional) Anthropic API · pytest · GitHub Actions