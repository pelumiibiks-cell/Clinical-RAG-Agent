import streamlit as st

import rag_core
from ui.state import cached_index_status, invalidate_index_status


def _rebuild_with_progress() -> None:
    with st.status("Rebuilding index...", expanded=True) as status:
        bar = st.progress(0.0)

        def on_progress(stage, current, total, label):
            fraction = current / total if total else 0.0
            if stage == "reading":
                bar.progress(fraction, text=f"Reading {label} ({current}/{total})")
            elif stage == "embedding":
                bar.progress(0.9, text=f"Embedding {label}...")
            elif stage == "done":
                bar.progress(1.0, text="Saved")

        rag_core.rebuild_index(progress_callback=on_progress)
        status.update(label="Index rebuilt", state="complete", expanded=False)

    invalidate_index_status()
    st.rerun()


def render() -> None:
    st.title("Knowledge base")
    st.caption("Every PDF in M_pdfs/ should be reflected in the index. If it isn't, the app will answer confidently from the wrong (or no) source.")

    status = cached_index_status()

    if not status["index_exists"]:
        st.warning("No index built yet.", icon="📚")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("PDFs on disk", len(status["on_disk"]))
        c2.metric("PDFs indexed", len(status["indexed"]))
        c3.metric("Total chunks", status["total_chunks"])

    st.subheader("Documents")
    rows = []
    for name in sorted(status["on_disk"]):
        indexed = name in status["indexed"]
        rows.append({
            "File": name,
            "Status": "✅ Indexed" if indexed else "❌ Not indexed",
            "Chunks": status["chunk_counts"].get(name, 0),
        })
    for name in sorted(status["orphaned"]):
        rows.append({"File": name, "Status": "⚠️ Indexed, missing from M_pdfs/", "Chunks": status["chunk_counts"].get(name, 0)})

    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No PDFs found in M_pdfs/.")

    st.subheader("Add documents")
    uploaded = st.file_uploader("Upload a PDF to add to M_pdfs/", type="pdf", accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            rag_core.save_uploaded_pdf(f.name, f.getvalue())
        st.success(f"Saved {len(uploaded)} file(s) to M_pdfs/. Rebuild the index to include them.")
        invalidate_index_status()
        st.rerun()

    st.subheader("Rebuild")
    st.caption("Re-reads every PDF in M_pdfs/, re-chunks, re-embeds, and overwrites the FAISS index. Takes a while the first time (model download + encoding all chunks).")
    if st.button("Rebuild index", type="primary"):
        _rebuild_with_progress()
