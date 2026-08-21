import streamlit as st

from ui.state import init_session_state, load_pipeline
from views import chat, knowledge_base, inspector, about

st.set_page_config(page_title="Malaria RAG", page_icon="🩺", layout="wide")

init_session_state()

try:
    load_pipeline()
except Exception as e:
    st.error(f"Could not start: {e}")
    st.stop()

with st.sidebar:
    st.header("Retrieval settings")
    st.session_state.top_k = st.slider("Passages to retrieve (top_k)", 1, 10, st.session_state.top_k)
    st.session_state.threshold = st.slider(
        "Similarity threshold", 0.0, 1.0, st.session_state.threshold, step=0.05,
        help="Higher = stricter. Passages below this are never used, even if they're the closest match available."
    )
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

chat_page = st.Page(chat.render, title="Chat", icon="💬", url_path="chat", default=True)
kb_page = st.Page(knowledge_base.render, title="Knowledge Base", icon="📚", url_path="knowledge-base")
inspector_page = st.Page(inspector.render, title="Inspector", icon="🔍", url_path="inspector")
about_page = st.Page(about.render, title="About", icon="ℹ️", url_path="about")

pg = st.navigation([chat_page, kb_page, inspector_page, about_page])
pg.run()
