"""Front-end-agnostic RAG logic shared by Malator.py (FastAPI) and streamlit_app.py.

Nothing here imports streamlit or fastapi -- both front ends call into this module
instead of reimplementing retrieval or prompting.
"""

import time
from pathlib import Path

from google.genai import types

from config_mal import client
import malaria_embed_query as mq
import malaria_embed_context as mc

GEN_MODEL = "gemini-3.1-flash-lite"

# Deterministic by default. The eval scores a single sample per question, so a
# sampled model means the reported faithfulness and correctness numbers are of
# one draw, not of the system.
TEMPERATURE = 0.0

# One definition instead of six. top_k was hardcoded as 5 in six places across
# the retriever, this module, the UI and the eval, with nothing keeping them in
# step.
TOP_K = 5

# Retry on transient API failures. The eval harness had 429 handling and
# production had none, so a rate limit or a hung call took out a user request
# while the offline script sailed through.
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 2.0
_REQUEST_TIMEOUT_MS = 60_000

# The standing rules belong in system_instruction, not glued to the top of every
# user turn. The whole thesis of this project is that a confidently wrong
# antimalarial dose is a different class of failure from a wrong answer about a
# recipe, and the entire previous instruction was "Use only the content below
# to answer the query. If the answer is not contained in the content, say so."
SYSTEM_INSTRUCTION = """You answer questions about malaria diagnosis and treatment for clinicians, using only the excerpts from official clinical guidelines supplied in each request.

Rules, in order of precedence:

1. Ground every clinical claim in the supplied excerpts. Never use knowledge from outside them, even if you are confident it is correct.
2. If the excerpts do not contain the answer, say so plainly and stop. Do not guess, do not reason toward a likely answer, and do not offer a general-knowledge answer as a substitute. An honest "the guidelines provided do not cover this" is a correct response.
3. Never extrapolate a dose, a dosing interval, a duration, or a weight band that is not stated in the excerpts. Do not convert, scale, or infer paediatric doses from adult ones.
4. Cite the source and page for each clinical claim, as (source, page N), using the Source and Page values given with each excerpt.
5. If the excerpts disagree with each other, say so and cite both rather than silently picking one.
6. Quote dose figures, drug names and durations exactly as written. Do not round or reformat them.
7. Be brief. Answer the question asked, without preamble or a summary of these rules."""


def prompt(query: str, retrieved_chunks: list) -> str:
    """The per-request content. Standing instructions live in
    SYSTEM_INSTRUCTION, so this carries only the excerpts and the question."""
    content = "\n\n".join(
        f"""Source: {c['Source']}
Page: {c['Page Number']}
Content: {c['Content']}"""
        for c in retrieved_chunks
    )

    return f"""Excerpts from the clinical guidelines:

{content}

Question:
{query}

Answer:"""


def _config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
    )


def _call_with_retry(fn, *args, **kwargs):
    """Retry transient failures with linear backoff. A hung or rate-limited
    call used to hang the caller indefinitely with no timeout at all."""
    last = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # google-genai raises several distinct types
            last = exc
            transient = any(
                marker in str(exc)
                for marker in ("429", "500", "502", "503", "504", "timeout", "deadline")
            )
            if not transient or attempt == _MAX_ATTEMPTS - 1:
                raise
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))
    raise last


def answer(query: str, top_k: int = TOP_K, threshold: float = mq.SIMILARITY_THRESHOLD, model: str = GEN_MODEL):
    """Retrieve then generate. Returns (answer_text, chunks). chunks is [] when
    nothing cleared the threshold -- callers should treat that as "no grounded
    answer" rather than calling the model with an empty content block."""
    chunks = mq.search(query, top_results=top_k, threshold=threshold)
    if not chunks:
        return None, chunks

    final_prompt = prompt(query, chunks)
    response = _call_with_retry(
        client.models.generate_content,
        model=model,
        contents=final_prompt,
        config=_config(),
    )
    return response.text, chunks


def answer_stream(query: str, top_k: int = TOP_K, threshold: float = mq.SIMILARITY_THRESHOLD, model: str = GEN_MODEL):
    """Same as answer() but yields text pieces as they arrive. Retrieval happens
    up front, so chunks are known before the first piece is yielded -- call
    retrieve_only() first if you need the sources before streaming starts."""
    chunks = mq.search(query, top_results=top_k, threshold=threshold)
    if not chunks:
        return chunks, iter(())

    final_prompt = prompt(query, chunks)

    def _pieces():
        stream = _call_with_retry(
            client.models.generate_content_stream,
            model=model,
            contents=final_prompt,
            config=_config(),
        )
        for piece in stream:
            if piece.text:
                yield piece.text

    return chunks, _pieces()


def retrieve_only(query: str, top_k: int = TOP_K, threshold: float = 0.0) -> list:
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
