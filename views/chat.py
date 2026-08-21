import streamlit as st

import rag_core
from ui.components import render_sources, render_no_index_notice, render_no_results_notice, render_drift_banner
from ui.state import cached_index_status


def render() -> None:
    st.title("Ask about malaria treatment")
    st.caption("Answers are grounded only in the PDFs in M_pdfs/ — nothing is answered from the model's own knowledge.")

    if not rag_core.index_exists():
        render_no_index_notice()
        return

    render_drift_banner(cached_index_status())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []))

    query = st.chat_input("Ask a question grounded in the malaria guidelines...")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query, "sources": []})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        chunks, pieces = rag_core.answer_stream(
            query, top_k=st.session_state.top_k, threshold=st.session_state.threshold
        )
        if not chunks:
            render_no_results_notice(query)
            answer_text = "No relevant content found."
        else:
            answer_text = st.write_stream(pieces)
            render_sources(chunks)

    st.session_state.messages.append({"role": "assistant", "content": answer_text, "sources": chunks})
