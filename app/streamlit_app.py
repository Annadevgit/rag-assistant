"""Streamlit interface for querying the document database through the RAG pipeline."""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.rag_pipeline import RagPipeline

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Document Assistant")
st.write(
    "Ask a question about the return policy, delivery policy, or loyalty "
    "program. The assistant answers using only the indexed document database "
    "— no information is ever made up."
)

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"

if not (INDEX_DIR / "embeddings.npy").exists():
    st.error(
        "Index not found. Run this first:\n\n"
        "```\npython -m src.build_index\n```"
    )
    st.stop()


@st.cache_resource
def load_pipeline(generation_mode: str):
    pipeline = RagPipeline(
        index_dir=str(INDEX_DIR),
        generation_mode=generation_mode,
    )
    pipeline.load()
    return pipeline


with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Generation mode",
        options=["extractive", "llm"],
        help=(
            "extractive: directly returns the relevant passages; no API key required.\n\n"
            "llm: generates a written answer using Claude "
            "(requires ANTHROPIC_API_KEY)."
        ),
    )

    top_k = st.slider(
        "Number of retrieved passages",
        1,
        10,
        3,
    )


pipeline = load_pipeline(mode)

query = st.text_input(
    "Your question",
    placeholder="e.g. How long do I have to return an item?",
)

if st.button("Ask question", type="primary") and query:
    with st.spinner("Searching the document database..."):
        try:
            response = pipeline.ask(query, top_k=top_k)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    st.divider()
    st.markdown("### Answer")
    st.write(response.answer)

    st.markdown("### Sources used")

    for chunk, score in response.sources:
        with st.expander(
            f"{chunk.doc_title} > {chunk.section_title} "
            f"(score: {score:.3f})"
        ):
            st.write(chunk.text)


st.divider()

st.caption(
    "Demo project — fictional document database (e-commerce policies). "
    "See the repository README for the complete methodology."
)