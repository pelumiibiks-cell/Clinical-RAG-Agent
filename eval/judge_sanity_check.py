"""
Judge sanity check.

Everything we've measured so far assumes the judge (gemini-3.5-flash-lite)
is actually capable of catching a bad answer -- but across 24 malaria
questions and 12 football questions, faithfulness and relevance have
NEVER once scored below 1.0. That's plausible (the pipeline's failure
mode has genuinely been "honest decline", not fabrication) but it has
never been tested directly.

This script bypasses run_pipeline() entirely -- no retrieval, no
generation call -- and feeds judge_all() five hand-built (context,
question, expected_answer, bad_answer) cases where the "bad_answer" is
deliberately wrong in a specific, checkable way. If the judge is doing
real work, each case below should score noticeably below 1.0 on the
metric it's designed to test. If it doesn't, the judge itself -- not
the pipeline -- is the thing that needs fixing before it's trusted for
the router/self-check work ahead.

Only 5 API calls total (no generation calls), so this is very cheap
regardless of quota concerns.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generation_eval import judge_all  # reuse the exact same judge function

RESULTS_PATH = Path(__file__).resolve().parent / "judge_sanity_check_results.json"

TEST_CASES = [
    {
        "id": "adv01_drug_swap",
        "targets": "faithfulness (direct contradiction of context)",
        "question": "What antipyretic is recommended for reducing high body temperatures above 39C in malaria patients?",
        "expected_answer": "Paracetamol, along with tepid sponging and fanning for comfort.",
        "context": (
            "Reduce high body temperatures (> 39C) by administering paracetamol as an "
            "antipyretic. Tepid sponging and fanning may make the patient comfortable. "
            "Carry out regular laboratory evaluation of erythrocyte volume fraction "
            "(haematocrit) or haemoglobin concentration, glucose, urea or creatinine and "
            "electrolytes. Avoid drugs that increase the risk for gastrointestinal "
            "bleeding (aspirin, corticosteroids)."
        ),
        "bad_answer": (
            "Ibuprofen is the recommended antipyretic for reducing high body "
            "temperatures above 39C in malaria patients, along with tepid sponging."
        ),
        "prediction": "faithfulness should drop well below 1.0 -- context says paracetamol, not ibuprofen",
    },
    {
        "id": "adv02_fabricated_combo",
        "targets": "faithfulness (invented fact not in context at all)",
        "question": "What is the recommended first-line treatment for uncomplicated falciparum malaria?",
        "expected_answer": "Artemisinin-based combination therapies (ACTs), e.g. artemether + lumefantrine, artesunate + amodiaquine, etc.",
        "context": (
            "Artemisinin-based combination therapies (ACTs) are the recommended treatments "
            "for uncomplicated falciparum malaria. The following ACTs are recommended: "
            "Artemether + lumefantrine; artesunate + amodiaquine; artesunate + mefloquine; "
            "artesunate + sulfadoxine-pyrimethamine, and dihydroartemisinin + piperaquine."
        ),
        "bad_answer": (
            "The recommended first-line treatment is artesunate combined with paracetamol, "
            "which is one of the standard ACT combinations for uncomplicated falciparum malaria."
        ),
        "prediction": "faithfulness should drop -- 'artesunate + paracetamol' is not one of the listed ACTs, it's fabricated",
    },
    {
        "id": "adv03_evasive_nonanswer",
        "targets": "relevance (fails to address a clearly answerable question)",
        "question": "What diagnostic step is recommended before starting malaria treatment in all suspected patients?",
        "expected_answer": "Prompt parasitological confirmation by microscopy or RDTs before treatment starts.",
        "context": (
            "Prompt parasitological confirmation by microscopy or alternatively by RDTs is "
            "recommended in all patients suspected of malaria before treatment is started. "
            "Treatment solely on the basis of clinical suspicion should only be considered "
            "when a parasitological diagnosis is not accessible."
        ),
        "bad_answer": (
            "Malaria is a serious disease that affects millions of people worldwide, and "
            "prevention through bed nets and vector control is very important for reducing transmission."
        ),
        "prediction": "relevance should drop well below 1.0 -- this is a real, answerable question with the answer sitting right in the context, and the response just doesn't address it",
    },
    {
        "id": "adv04_wrong_number",
        "targets": "correctness (contradicts the golden expected_answer with a specific wrong figure)",
        "question": "For how long should parenteral antimalarial agents be given at minimum in treating severe malaria?",
        "expected_answer": "A minimum of 24 hours, even if the patient can tolerate oral medication earlier.",
        "context": (
            "Antimalarial drugs should be given parenterally for a minimum of 24h and "
            "replaced by oral medication as soon as it can be tolerated."
        ),
        "bad_answer": (
            "Parenteral antimalarial treatment should be given for a minimum of 6 hours "
            "before switching to oral medication."
        ),
        "prediction": "correctness should score near 0 -- 6 hours directly contradicts both the context and the golden answer's 24 hours",
    },
    {
        "id": "adv05_partial_fabrication",
        "targets": "faithfulness GRADIENT -- one correct claim + one fabricated claim, should NOT be a clean 0 or 1",
        "question": "What drugs should be avoided in malaria patients because they increase the risk of gastrointestinal bleeding?",
        "expected_answer": "Aspirin and corticosteroids.",
        "context": (
            "Avoid drugs that increase the risk for gastrointestinal bleeding (aspirin, corticosteroids)."
        ),
        "bad_answer": (
            "Patients should avoid aspirin and corticosteroids due to gastrointestinal bleeding risk. "
            "Additionally, all malaria patients should be given prophylactic antibiotics to prevent "
            "secondary bacterial infection."
        ),
        "prediction": "faithfulness should land somewhere in the middle (e.g. ~0.5), not a clean 0 or 1 -- first claim is supported, second (prophylactic antibiotics) is invented and unsupported by context",
    },
]


def main():
    results = []
    for case in TEST_CASES:
        print(f"Testing {case['id']}...")
        judged = judge_all(
            case["question"], case["expected_answer"], case["context"], case["bad_answer"]
        )
        results.append(
            {
                "id": case["id"],
                "targets": case["targets"],
                "prediction": case["prediction"],
                "bad_answer": case["bad_answer"],
                "faithfulness_score": judged.get("faithfulness_score"),
                "relevance_score": judged.get("relevance_score"),
                "correctness_score": judged.get("correctness_score"),
                "judge_detail": judged,
            }
        )

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n--- SUMMARY ---")
    for r in results:
        print(f"{r['id']}: faithfulness={r['faithfulness_score']} "
              f"relevance={r['relevance_score']} correctness={r['correctness_score']}")
        print(f"  prediction: {r['prediction']}")
    print(f"\nFull details written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
