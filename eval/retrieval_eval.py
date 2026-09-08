"""
Retrieval-only evaluation.

Deliberately bypasses malaria_embed_query.search() and queries the FAISS
index directly. That function filters by SIMILARITY_THRESHOLD *before*
returning anything, which would hide the difference between:
  (a) the right chunk never showed up in top-k at all, and
  (b) the right chunk showed up but got filtered out by the threshold.
Those are different bugs with different fixes, so we need the raw ranked
list first, and check the threshold as a separate question.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import malaria_embed_query as mq  # lazy singletons: get_model()/get_index()/get_metadata()

BASE_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = BASE_DIR / "golden_set_malaria.json"
RESULTS_PATH = BASE_DIR / "retrieval_results_malaria.json"

TOP_K = 5  # matches production; we score hit@1 / hit@3 / hit@5 from this one pull


def raw_search(query: str, top_k: int = TOP_K):
    """Same embedding + FAISS call as production, but returns every
    candidate with its rank and score, with no threshold filtering."""
    embedding = mq.get_model().encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    distances, indices = mq.get_index().search(embedding, top_k)

    ranked = []
    for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx < 0:
            continue
        entry = mq.get_metadata()[int(idx)]
        ranked.append(
            {
                "rank": rank,
                "score": float(score),
                "source": entry.get("Source"),
                "page": entry.get("Page Number"),
                "passes_threshold": float(score) > mq.SIMILARITY_THRESHOLD,
            }
        )
    return ranked


def acceptable_keys(item: dict) -> set:
    """Every (source, page) hand-verified to contain a sentence that answers the
    question. A fact repeated on more than one page has more than one right
    answer -- scoring against a single page counted a genuine hit as a miss
    (gs10 and gs16 both did exactly that under the old single-page rule)."""
    locs = item.get("acceptable_locations")
    if not locs:
        return {(item["expected_source"], item["expected_page"])}
    return {(loc["source"], loc["page"]) for loc in locs}


def matches_expected(candidate: dict, accepted: set) -> bool:
    return (candidate["source"], candidate["page"]) in accepted


def score_question(item: dict) -> dict:
    ranked = raw_search(item["question"], top_k=TOP_K)

    if not item["indexed"]:
        # Negative control: correct behavior is NO confident match at all.
        false_positive = next(
            (c for c in ranked if c["passes_threshold"]), None
        )
        return {
            "id": item["id"],
            "type": "negative_control",
            "correctly_absent": false_positive is None,
            "false_positive_source": (
                false_positive["source"] if false_positive else None
            ),
            "raw_top1_score": ranked[0]["score"] if ranked else None,
            "ranked": ranked,
        }

    accepted = acceptable_keys(item)
    hit_rank = next(
        (c["rank"] for c in ranked if matches_expected(c, accepted)),
        None,
    )
    threshold_would_pass = (
        hit_rank is not None and ranked[hit_rank - 1]["passes_threshold"]
    )

    return {
        "id": item["id"],
        "type": "positive",
        "difficulty": item["difficulty"],
        "question": item["question"],
        "n_acceptable_locations": len(accepted),
        "hit_rank": hit_rank,  # None = not in top-k at all
        "hit_at_1": hit_rank == 1,
        "hit_at_3": hit_rank is not None and hit_rank <= 3,
        "hit_at_5": hit_rank is not None and hit_rank <= 5,
        "threshold_would_have_blocked_it": (
            hit_rank is not None and not threshold_would_pass
        ),
        "ranked": ranked,
    }


def main():
    golden_set = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    results = [score_question(item) for item in golden_set]

    positives = [r for r in results if r["type"] == "positive"]
    negatives = [r for r in results if r["type"] == "negative_control"]

    n = len(positives)
    summary = {
        "n_positive_questions": n,
        "hit_at_1": sum(r["hit_at_1"] for r in positives) / n,
        "hit_at_3": sum(r["hit_at_3"] for r in positives) / n,
        "hit_at_5": sum(r["hit_at_5"] for r in positives) / n,
        "missed_entirely": [r["id"] for r in positives if r["hit_rank"] is None],
        "found_but_threshold_blocked": [
            r["id"] for r in positives if r["threshold_would_have_blocked_it"]
        ],
        "n_negative_controls": len(negatives),
        "negative_controls_correctly_absent": sum(
            r["correctly_absent"] for r in negatives
        ),
        "negative_controls_false_positive": [
            {"id": r["id"], "wrong_source_returned": r["false_positive_source"]}
            for r in negatives
            if not r["correctly_absent"]
        ],
    }

    output = {"summary": summary, "details": results}
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nFull per-question results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
