from fastapi import FastAPI
from pydantic import BaseModel

import rag_core

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


@app.post("/ask")
def ask_question(request: QueryRequest):
    answer_text, chunks = rag_core.answer(request.query)

    return {
        "query": request.query,
        "answer": answer_text if answer_text is not None else "No relevant content found.",
    }
