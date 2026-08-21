import streamlit as st

import malaria_embed_query as mq
import rag_core


def render() -> None:
    st.title("About this app")

    st.markdown("""
This is a retrieval-augmented Q&A tool over a fixed set of malaria treatment
PDFs. Every answer is grounded in retrieved passages — the model is told to
say so explicitly when the content doesn't contain an answer, rather than
fall back on its own training knowledge.
    """)

    st.subheader("Pipeline")
    st.markdown(f"""
1. **Embed** — query encoded with `{mq.MODEL}` (384-dim, normalized)
2. **Retrieve** — cosine similarity search over a FAISS `IndexFlatIP` index, top-k candidates
3. **Filter** — candidates below the similarity threshold (default `{mq.SIMILARITY_THRESHOLD}`) are dropped
4. **Generate** — surviving passages are stuffed into a prompt and sent to `{rag_core.GEN_MODEL}`
    """)

    st.subheader("Configuration")
    st.code(
        f"Model: {mq.MODEL}\n"
        f"Generation model: {rag_core.GEN_MODEL}\n"
        f"Default threshold: {mq.SIMILARITY_THRESHOLD}\n"
        f"Index path: {mq.MALARIA_FAISS_INDEX}",
        language="text",
    )

    st.subheader("Same core, two front doors")
    st.markdown(
        "This Streamlit app and the `Malator.py` FastAPI `/ask` endpoint both call "
        "into `rag_core.py` — the retrieval and prompting logic is written once."
    )
