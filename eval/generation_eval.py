"""
Generation-layer evaluation.

Runs the ACTUAL production path for each golden question:
  malaria_embed_query.search()  (threshold-filtered, exactly as /ask uses it)
  -> Malator.prompt()           (same prompt template as production)
  -> gemini-3.1-flash-lite      (same generation model as production)

Then scores the result with a SEPARATE judge call using a different model:
gemini-3.5-flash-lite. This is a partial mitigation for the "marking its
own homework" problem, not a full fix -- true independence would mean a
different model *provider* entirely (e.g. Claude or GPT as judge), not
just a different model from the same vendor. Worth doing once you're
ready to wire in a second API key.

Model/quota note: full "flash" tier models (gemini-3.6-flash, etc.) are
capped at just 20 requests/day on a free key -- confirmed by hitting that
exact wall. Lite-tier models get far more generous free quotas, so the
judge here is gemini-3.5-flash-lite (a different model GENERATION from
the pipeline's gemini-3.1-flash-lite, just not a different size tier).
Weaker independence story than lite-vs-full-flash would have been, but
it's what's actually usable on a free key.

All three judgments (faithfulness, relevance, correctness) are combined
into a SINGLE call per question instead of three separate ones, to cut
total API usage in half (48 calls for the whole run instead of 96) --
this matters a lot more now that we know how tight the daily caps are.
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_mal import client
import rag_core
from rag_core import prompt as build_prompt

BASE_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = BASE_DIR / "golden_set_malaria.json"
RESULTS_PATH = BASE_DIR / "generation_results_malaria.json"

GENERATION_MODEL = "gemini-3.1-flash-lite"  # matches Malator.py exactly
JUDGE_MODEL = "gemini-3.5-flash-lite"  # different model generation, generous free quota

# Flat pacing to stay under free-tier per-minute caps. 26 questions x 2 calls,
# so this is the dominant cost of a run; the retry path below handles the
# actual 429s.
PACING_SECONDS = 2


def call_model(model: str, contents: str, max_retries: int = 4) -> str:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=contents)
            time.sleep(PACING_SECONDS)
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 20 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            raise


def parse_json_response(text: str) -> dict:
    """Judge is asked for raw JSON but models sometimes wrap it in
    ```json fences anyway -- strip those before parsing."""
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_response": text}


def judge_all(question: str, expected_answer: str, context: str, answer: str) -> dict:
    """Single combined judge call covering all three metrics. Each metric
    keeps its own distinct definition even though they're asked together --
    see the field-by-field instructions below."""
    judge_prompt = f"""You are auditing a RAG system's output on three SEPARATE dimensions.
Judge each independently -- a low score on one does not imply a low score on another.

1. FAITHFULNESS: Break ANSWER into individual factual claims. For each, decide whether it
   is directly supported by CONTEXT below (not by general medical knowledge -- only by
   what CONTEXT actually says). faithfulness_score = supported claims / total claims (0-1).
   If ANSWER explicitly says the info isn't in the context, treat that as fully faithful
   (1.0) with an empty claims list.

2. RELEVANCE: Does ANSWER directly address QUESTION, regardless of whether it's factually
   correct? An answer confidently discussing the wrong topic scores low. An answer that
   honestly declines because the info wasn't found still scores HIGH here (it IS
   addressing the question, by correctly declining). relevance_score is 0-1.

3. CORRECTNESS: Compare ANSWER against EXPECTED_ANSWER for QUESTION. Judge factual
   correctness only, ignore phrasing/style differences. If ANSWER correctly declines to
   answer AND expected_answer required info the system plausibly didn't have access to,
   score 0.5 (honest non-answer, not wrong). correctness_score is 0-1.

CONTEXT:
{context}

QUESTION:
{question}

EXPECTED_ANSWER:
{expected_answer}

ANSWER:
{answer}

Respond with ONLY raw JSON, no markdown fences, in this exact shape:
{{
  "faithfulness_score": 0.0,
  "faithfulness_claims": [{{"claim": "...", "supported": true}}],
  "relevance_score": 0.0,
  "relevance_reasoning": "...",
  "correctness_score": 0.0,
  "correctness_reasoning": "..."
}}
"""
    raw = call_model(JUDGE_MODEL, judge_prompt)
    return parse_json_response(raw)


def run_pipeline(question: str):
    """Runs the production path itself rather than reassembling it.

    This used to call search() and build_prompt() and then generate_content()
    by hand, which meant it silently stopped matching production the moment
    rag_core grew a system instruction and a temperature -- it would have been
    scoring a differently-configured model than the one users talk to.
    """
    answer, retrieved = rag_core.answer(question)
    time.sleep(PACING_SECONDS)
    return (answer if answer is not None else "No relevant content found."), retrieved


def score_question(item: dict) -> dict:
    answer, retrieved = run_pipeline(item["question"])
    context_text = "\n\n".join(
        f"Source: {c['Source']} | Page: {c['Page Number']}\n{c['Content']}"
        for c in retrieved
    )

    judged = judge_all(item["question"], item["expected_answer"], context_text, answer)

    return {
        "id": item["id"],
        "indexed": item["indexed"],
        "question": item["question"],
        "generated_answer": answer,
        "n_chunks_retrieved": len(retrieved),
        "retrieved_sources": [
            {"source": c["Source"], "page": c["Page Number"]} for c in retrieved
        ],
        "faithfulness_score": judged.get("faithfulness_score"),
        "relevance_score": judged.get("relevance_score"),
        "correctness_score": judged.get("correctness_score"),
        "judge_detail": judged,
    }


def main():
    golden_set = json.loads(GOLDEN_SET_PATH.read_text())
    results = []
    for item in golden_set:
        print(f"Scoring {item['id']}...")
        results.append(score_question(item))
        # incremental save: a crash on question 15 shouldn't lose the API
        # calls already spent on questions 1-14
        RESULTS_PATH.write_text(json.dumps({"summary": "in_progress", "details": results}, indent=2))

    def avg(key):
        vals = [r[key] for r in results if isinstance(r[key], (int, float))]
        return sum(vals) / len(vals) if vals else None

    indexed_results = [r for r in results if r["indexed"]]
    negative_results = [r for r in results if not r["indexed"]]

    summary = {
        "n_questions": len(results),
        "avg_faithfulness": avg("faithfulness_score"),
        "avg_relevance": avg("relevance_score"),
        "avg_correctness": avg("correctness_score"),
        "avg_correctness_indexed_only": (
            sum(r["correctness_score"] for r in indexed_results
                if isinstance(r["correctness_score"], (int, float)))
            / max(len([r for r in indexed_results if isinstance(r["correctness_score"], (int, float))]), 1)
        ),
        "negative_control_correctness": [
            {"id": r["id"], "correctness_score": r["correctness_score"],
             "answer": r["generated_answer"][:200]}
            for r in negative_results
        ],
        "lowest_faithfulness": sorted(
            [r for r in results if isinstance(r["faithfulness_score"], (int, float))],
            key=lambda r: r["faithfulness_score"],
        )[:5],
    }

    RESULTS_PATH.write_text(json.dumps({"summary": summary, "details": results}, indent=2))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()