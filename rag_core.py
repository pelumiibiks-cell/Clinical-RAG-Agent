"""Front-end-agnostic RAG logic shared by Malator.py (FastAPI) and streamlit_app.py.

Nothing here imports streamlit or fastapi -- both front ends call into this module
instead of reimplementing retrieval or prompting.
"""

from pathlib import Path

from config_mal import client
import malaria_embed_query as mq
import malaria_embed_context as mc

GEN_MODEL = "gemini-3.1-flash-lite"


def prompt(query: str, retrieved_chunks: list) -> str:
    content = "\n\n".join(
        f"""Source: {c['Source']}
Page: {c['Page Number']}
Content: {c['Content']}"""
        for c in retrieved_chunks
    )

    return f"""
Use only the content below to answer the query.
If the answer is not contained in the content, say so.

Content:
{content}

Query:
{query}

Answer:
"""


def answer(query: str, top_k: int = 5, threshold: float = mq.SIMILARITY_THRESHOLD, model: str = GEN_MODEL):
    """Retrieve then generate. Returns (answer_text, chunks). chunks is [] when
    nothing cleared the threshold -- callers should treat that as "no grounded
    answer" rather than calling the model with an empty content block."""
    chunks = mq.search(query, top_results=top_k, threshold=threshold)
    if not chunks:
        return None, chunks

    final_prompt = prompt(query, chunks)
    response = client.models.generate_content(model=model, contents=final_prompt)
    return response.text, chunks


def answer_stream(query: str, top_k: int = 5, threshold: float = mq.SIMILARITY_THRESHOLD, model: str = GEN_MODEL):
    """Same as answer() but yields text pieces as they arrive. Retrieval happens
    up front, so chunks are known before the first piece is yielded -- call
    retrieve_only() first if you need the sources before streaming starts."""
    chunks = mq.search(query, top_results=top_k, threshold=threshold)
    if not chunks:
        return chunks, iter(())

    final_prompt = prompt(query, chunks)

    def _pieces():
        for piece in client.models.generate_content_stream(model=model, contents=final_prompt):
            if piece.text:
                yield piece.text

    return chunks, _pieces()


def retrieve_only(query: str, top_k: int = 5, threshold: float = 0.0) -> list:
    """Retrieval with no LLM call, no threshold filtering by default -- used by
    the Inspector view to show every candidate, including ones a stricter
    threshold would reject."""
    return mq.search(query, top_results=top_k, threshold=threshold)


def index_exists() -> bool:
    return mq.index_exists()


def index_status() -> dict:
    """Compares M_pdfs/ against what's actually indexed. Direct fix for the
    'stale index' finding in EVALUATION_REPORT.md -- six PDFs on disk, only
    three ever embedded, and nothing noticed until now."""
    pdf_folder = mc.locate_dir()
    on_disk = {p.name for p in sorted(pdf_folder.iterdir()) if p.suffix.lower() == ".pdf"}

    if not mq.index_exists():
        return {
            "index_exists": False,
            "on_disk": on_disk,
            "indexed": set(),
            "not_indexed": on_disk,
            "orphaned": set(),
            "chunk_counts": {},
            "total_chunks": 0,
            "index_mtime": None,
        }

    metadata = mq.get_metadata()
    indexed = {m["Source"] for m in metadata}

    chunk_counts = {}
    for m in metadata:
        chunk_counts[m["Source"]] = chunk_counts.get(m["Source"], 0) + 1

    return {
        "index_exists": True,
        "on_disk": on_disk,
        "indexed": indexed,
        "not_indexed": on_disk - indexed,
        "orphaned": indexed - on_disk,  # indexed but no longer in M_pdfs/
        "chunk_counts": chunk_counts,
        "total_chunks": len(metadata),
        "index_mtime": mq.MALARIA_FAISS_INDEX.stat().st_mtime,
    }


def rebuild_index(progress_callback=None) -> None:
    mc.embed_context(progress_callback=progress_callback)
    mq.reset_cache()  # next search() re-reads the freshly written index/metadata


def save_uploaded_pdf(filename: str, data: bytes) -> Path:
    pdf_folder = mc.locate_dir()
    dest = pdf_folder / filename
    dest.write_bytes(data)
    return dest
