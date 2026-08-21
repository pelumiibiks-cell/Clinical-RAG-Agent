"""Reusable UI fragments: source citations, index-freshness banner, empty states."""

import streamlit as st


def render_sources(chunks: list) -> None:
    if not chunks:
        return
    with st.expander(f"Sources ({len(chunks)})"):
        for c in chunks:
            score = c.get("Score")
            score_str = f" · similarity {score:.2f}" if score is not None else ""
            st.markdown(f"**{c['Source']}** · page {c['Page Number']}{score_str}")
            st.caption(c["Content"])
            st.divider()


def render_no_index_notice() -> None:
    st.warning(
        "No knowledge base found yet. Go to **Knowledge Base** and click "
        "**Rebuild index** before asking questions.",
        icon="📚",
    )


def render_no_results_notice(query: str) -> None:
    st.info(
        f"No passage in the knowledge base cleared the similarity threshold for "
        f"\"{query}\". Rather than guess, no answer was generated. Try lowering "
        f"the threshold in the sidebar, or check Inspector to see what almost matched.",
        icon="🔍",
    )


def render_drift_banner(status: dict) -> None:
    """Warns whenever the index doesn't match what's actually in M_pdfs/ --
    this is the direct fix for the stale-index finding in EVALUATION_REPORT.md."""
    if not status["index_exists"]:
        return
    missing = status["not_indexed"]
    orphaned = status["orphaned"]
    if not missing and not orphaned:
        return

    parts = []
    if missing:
        parts.append(f"{len(missing)} PDF(s) on disk are not indexed: {', '.join(sorted(missing))}")
    if orphaned:
        parts.append(f"{len(orphaned)} indexed source(s) no longer exist in M_pdfs/: {', '.join(sorted(orphaned))}")
    st.warning(
        "Index is out of sync with M_pdfs/. " + " ".join(parts) + " Rebuild on the Knowledge Base page.",
        icon="⚠️",
    )
