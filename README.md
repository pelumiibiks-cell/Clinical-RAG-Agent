# Malaria RAG — Q&A Pipeline over Clinical Treatment Guidelines

A retrieval-augmented generation (RAG) system that answers malaria treatment questions grounded in official clinical guideline PDFs (WHO, national essential medicines lists, standard treatment guidelines), paired with an independently-built evaluation layer that scores retrieval and generation separately.

## Why this repo is more than a RAG demo

Most RAG side projects stop at "it works." This one includes a full **before-any-fixes evaluation** of the baseline pipeline: a 24-question golden set (including 2 deliberately-planted negative controls), a retrieval scorer, an LLM-judged generation scorer, and a root-cause drill-down that traced a "chunk overlap dilution" hypothesis down to a specific mechanism — three unrelated clinical facts packed into a single embedding. Full write-up: [`EVALUATION_REPORT.md`](./EVALUATION_REPORT.md).

## Architecture

```
PDF guidelines (M_pdfs/)
   │  malaria_embed_context.py — PyMuPDF extraction → sentence-aware chunking → all-MiniLM-L6-v2 embeddings
   ▼
FAISS IndexFlatIP + metadata pickle (Malaria_db/)
   │  malaria_embed_query.py — embed query → top-k search → similarity-threshold filter
   ▼
Retrieved chunks
   │  Malator.py — FastAPI /ask endpoint → prompt-stuffing → Gemini call
   ▼
Answer
```

## Status

- [x] Baseline retrieval + generation pipeline
- [x] Independent evaluation layer (golden set, retrieval scorer, LLM-judged generation scorer, drill-down tooling)
- [x] Chunking rewrite informed by eval findings (hit@5: 59% → 73%)
- [ ] Agentic router + self-check loop (next milestone)
- [ ] Front end

## Setup

```bash
git clone <your-repo-url>
cd <repo-name>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.mal.clone .env.mal
```

Add source PDFs to `M_pdfs/` (see **Data sources** below), then build the index:

```bash
python malaria_embed_context.py
```

Run the API:

```bash
uvicorn Malator:app --reload
```

Query it:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the first-line treatment for uncomplicated malaria?"}'
```

## Data sources

PDFs and the built index aren't shipped in this repo (see `.gitignore`) — guideline documents get revised over time and some carry redistribution terms, so it's cleaner to source your own copies and rebuild locally. This project was built and evaluated against:

- WHO Guidelines for Malaria Treatment
- WHO / regional operational guidance
- A national Essential Medicines List
- A national Standard Treatment Guidelines document
- A clinical conditions / minor ailments reference

Re-running `malaria_embed_context.py` rebuilds `Malaria_db/` from whatever's currently in `M_pdfs/`.

## Tech stack

Python · FastAPI · FAISS · sentence-transformers (`all-MiniLM-L6-v2`) · Google Gemini API · PyMuPDF

## Known limitations

(from `EVALUATION_REPORT.md` — kept here honestly rather than only in the eval doc)

- Retrieval, not generation, is the current bottleneck: generation never returned a confident wrong answer in eval — it declined honestly whenever context was inadequate.
- The embedding model (`all-MiniLM-L6-v2`) appears to cap retrieval quality at the current corpus size even after chunking fixes.
- The LLM judge and the generation model are both "lite" tier from the same provider — a weaker independence guarantee than a full-size or cross-provider judge would give.

## Roadmap

Agentic router + self-check loop, evaluated against this same golden set for a measured before/after comparison — then a minimal front end.

## License

MIT — see [`LICENSE`](./LICENSE).
