from fastapi import FastAPI
from pydantic import BaseModel

from config_mal import client
from malaria_embed_query import search

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


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


@app.post("/ask")
def ask_question(request: QueryRequest):

    results = search(request.query)

    final_prompt = prompt(request.query, results)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=final_prompt
    )

    return {
        "query": request.query,
        "answer": response.text
    }