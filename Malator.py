from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

import malaria_embed_query as mq
import rag_core


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model and index at startup.

    Importing faiss and sentence_transformers takes ~34s and loading the model
    another ~39s cold. With no startup hook the first /ask paid all of it inside
    the request. Streamlit already warmed up via @st.cache_resource; the API did
    not.
    """
    if mq.index_exists():
        mq.get_model()
        mq.get_index()
        mq.get_metadata()
    yield


app = FastAPI(title="Malator", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    # Exposed because the Streamlit sidebar could tune these and API callers
    # could not.
    top_k: int = Field(default=rag_core.TOP_K, ge=1, le=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


@app.get("/health")
def health():
    status = rag_core.index_status()
    return {
        "ok": True,
        "index_exists": status["index_exists"],
        "total_chunks": status["total_chunks"],
        "model_loaded": mq._model is not None,
    }


@app.post("/ask")
def ask_question(request: QueryRequest):
    threshold = request.threshold if request.threshold is not None else mq.SIMILARITY_THRESHOLD
    answer_text, chunks = rag_core.answer(
        request.query, top_k=request.top_k, threshold=threshold
    )

    return {
        "query": request.query,
        "answer": answer_text if answer_text is not None else "No relevant content found.",
        # The chunks were retrieved, unpacked, then thrown away, so the API
        # returned ungrounded prose while the Streamlit UI showed citations for
        # the same answer. In a clinical tool the provenance is the point.
        "sources": [
            {
                "source": c["Source"],
                "page": c["Page Number"],
                "page_end": c.get("Page End", c["Page Number"]),
                "score": round(c["Score"], 4),
                "content": c["Content"],
            }
            for c in chunks
        ],
    }
