import streamlit as st

import rag_core
from ui.components import render_no_index_notice


def render() -> None:
    st.title("Retrieval inspector")
    st.caption("Runs retrieval only, no LLM call. Shows every candidate FAISS returned, including ones the similarity threshold would reject, so you can see the threshold's effect directly.")

    if not rag_core.index_exists():
        render_no_index_notice()
        return

    query = st.text_input("Query")
    # Reads the sidebar's value rather than defining a competing one. There
    # used to be two top_k sliders with different ranges, while the threshold
    # came from the sidebar either way.
    top_k = st.slider("top_k", 1, 20, st.session_state.top_k)

    if not query:
        return

    results = rag_core.retrieve_only(query, top_k=top_k, threshold=0.0)
    if not results:
        st.info("FAISS returned no candidates at all (empty index?).")
        return

    threshold = st.session_state.threshold
    for r in results:
        passes = r["Score"] >= threshold  # matches search(), which uses >=
        marker = "✅ above threshold" if passes else "❌ below threshold"
        with st.container(border=True):
            st.markdown(f"**{r['Source']}** · page {r['Page Number']} · score `{r['Score']:.3f}` · {marker}")
            st.caption(r["Content"])
