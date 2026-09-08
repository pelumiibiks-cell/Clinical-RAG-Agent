"""Session state init and cached resource loaders, shared by every view."""

import streamlit as st


def mq_threshold() -> float:
    """Single source of truth: this used to be a second hardcoded 0.4 that
    silently drifted from the retriever's own constant."""
    import malaria_embed_query as mq

    return mq.SIMILARITY_THRESHOLD

import rag_core


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str, "sources": list}]
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "threshold" not in st.session_state:
        st.session_state.threshold = mq_threshold()


@st.cache_resource(show_spinner="Loading embedding model and index (first run only)...")
def load_pipeline():
    """Warms the embedding model + FAISS index once per server process.
    Streamlit reruns the whole script on every interaction, but @cache_resource
    survives reruns -- this is what stops the 5-15s cold start from repeating
    on every chat message."""
    if not rag_core.index_exists():
        return None
    import malaria_embed_query as mq
    mq.get_model()
    mq.get_index()
    mq.get_metadata()
    return True


@st.cache_data(ttl=None, show_spinner=False)
def cached_index_status():
    return rag_core.index_status()


def invalidate_index_status() -> None:
    cached_index_status.clear()
    load_pipeline.clear()
