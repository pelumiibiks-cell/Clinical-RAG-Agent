# Malaria RAG — Evaluation Layer: Documentation

**Project:** Malaria Q&A RAG pipeline
**Scope of this document:** Everything built and discovered while adding an independent evaluation layer to the existing baseline pipeline (retrieval → prompt-stuffing → single Gemini call), *before* any fixes were applied and *before* the planned agentic router / self-check loop was built.

**Why this document exists:** the eval layer's whole value is the "before" picture. This is that picture, in writing, with evidence — not "I tried it and it seemed fine."

---

## 1. System under test (baseline, as it exists today)

| Component | File | What it does |
|---|---|---|
| Config / API client | `config_mal.py` | Loads Gemini API key from `.env.mal`, initializes `genai.Client` |
| Ingestion | `malaria_embed_context.py` | Reads PDFs page-by-page (PyMuPDF), splits into 1000-char chunks with 650-char overlap, embeds with `all-MiniLM-L6-v2`, stores in a FAISS `IndexFlatIP` index + a metadata pickle (`Source`, `Page Number`, `Content`) |
| Query | `malaria_embed_query.py` | Embeds a query, searches FAISS top-5, filters results by `SIMILARITY_THRESHOLD = 0.4` |
| Generation | `Malator.py` | FastAPI `/ask` endpoint: calls `search()`, stuffs retrieved chunks into a single prompt ("use only this content, say so if not found"), one call to `gemini-3.1-flash-lite`, returns the answer |

**Not yet built:** the agentic router and self-check loop described at project kickoff. Those are the *next* phase, deliberately sequenced after this eval layer so there's a clean, independently-measured "before" to compare against once they exist. Building eval-first means the self-check loop's future performance will be a measurable claim ("faithfulness went from X to Y") instead of a self-reported one.

---

## 2. Finding #1 — Stale FAISS index (discovered before any eval code was written)

`M_pdfs/` contains 6 source PDFs. The FAISS index and metadata only contain chunks from 3 of them:

**Indexed:** `9789241548526_eng.pdf`, `WHO_Treatment_Guidelines_Olumense.pdf`, `omn-ch-33-01-operationalguidance...pdf` (400 chunks total: 341 / 30 / 29)

**Never embedded:** `Nigeria-Essential-Medicine-List-2020.pdf`, `common_clinical_conditions_and_minor_ailments.pdf`, `PNG_D1_Standard-Treatment-Guidelines...pdf`

Confirmed via `pdftotext` that all three missing files contain substantial extractable text (124k–287k characters each) — they are not scanned/image-only PDFs that `fitz` failed to parse. The most likely explanation: `embed_context()` was run once, before these three PDFs were added to the folder, and never re-run. Nothing in the codebase checks that the index reflects what's currently in `M_pdfs/`.

**Why this matters:** questions whose true answer lives only in one of the un-indexed documents don't fail safely — they retrieve confident, wrong-source matches instead of "not found." This became the basis for two deliberate **negative control** questions in the golden set (see §3).

---

## 3. Golden set (`eval/golden_set.json`)

24 hand-verified question/answer pairs, grounded directly in the actual indexed chunk text (not invented), plus 2 negative controls sourced from the un-indexed PDFs.

**Schema per entry:**
```json
{
  "id": "gs01",
  "question": "...",
  "expected_answer": "...",
  "expected_source": "<pdf filename>",
  "expected_page": 8,
  "indexed": true,
  "difficulty": "easy | medium | hard | negative_control"
}
```

**Design decisions:**
- **`expected_source` + `expected_page` as ground truth**, not chunk index. Chunk indices shift on re-ingestion (different chunk size → different chunk boundaries → different indices); page numbers are stable across re-indexing, so the golden set survives the rechunking work planned next.
- **`indexed: false` marks negative controls** (`gs23`, `gs24`) — questions whose true source was never embedded. Correct behavior is "no confident match," not a wrong answer.
- **`difficulty` enables slicing**, not just a flat accuracy number — lets failures be broken down by question complexity rather than reported as one undifferentiated score.
- **User-verified**: all 24 entries were reviewed for medical/factual accuracy against the source guideline text before being treated as ground truth (self-authored ground truth that's never checked is a bigger risk than no eval at all, since it looks trustworthy while silently being wrong).

---

## 4. Retrieval evaluation (`eval/retrieval_eval.py`)

**What it measures:** for each golden question, does a chunk from the expected `(source, page)` appear in the raw top-5 FAISS results — *before* the 0.4 similarity threshold filter is applied. Deliberately bypasses `malaria_embed_query.search()`'s built-in filtering, because that function conflates two different failures: "never retrieved" vs. "retrieved but silently dropped by the threshold." Those need different fixes.

### Results (first, unmodified baseline run)

| Metric | Value |
|---|---|
| hit@1 | 31.8% |
| hit@3 | 45.5% |
| hit@5 | 59.1% |
| Missed entirely (top-5) | 9 of 22 positive questions (41%) |
| Threshold-blocked (would've hit but scored < 0.4) | 0 |
| Negative controls correctly identified as absent | **0 of 2** |

**Interpretation:**
- The hit@1 → hit@5 gap is consistent with **chunk overlap dilution**: the right answer is often "in the neighborhood" but near-duplicate chunks (65% character overlap between adjacent chunks) crowd it out of the top rank.
- **Both negative controls returned confident, wrong-document matches** instead of "nothing relevant." `gs23` (Nigeria EML drug combination) retrieved WHO guideline pages about similar drug combinations — topically adjacent, wrong document. `gs24` (Nigeria EML edition/publisher) retrieved generic front-matter from the WHO document. Cosine similarity cannot distinguish "similar topic" from "actually answers this."

### Drill-down (`eval/drilldown.py`) — root-causing the 9 misses

Rather than accept "missed top-5" as the final word, a second script finds the **true rank** of the expected chunk across the *entire* index (not just top-5), to distinguish "close miss" from "not associated at all."

| Question | True rank of correct chunk | Score |
|---|---|---|
| gs16 | 39 | 0.499 |
| gs17 | 152 | 0.484 |
| gs18 | 86 | 0.554 |
| gs23 (negative control) | not in index | — |
| gs24 (negative control) | not in index | — |

**Root cause identified:** `gs16`, `gs17`, and `gs18` are three *different* questions (about antipyretics, GI-bleeding drug avoidance, and lab monitoring respectively) whose correct answer is the **same single chunk** (page 21, metadata index 100). That chunk packs three distinct clinical facts into one 1000-character window. A single embedding vector for a chunk trying to represent three different facts at once becomes a mediocre match for all three questions and a strong match for none — explaining respectable-but-not-competitive scores (0.48–0.55) that never crack the top 5 in a 400-chunk index full of similar clinical vocabulary. This is the specific, evidenced mechanism behind the general "overlap/dilution" hypothesis, not just a restatement of it.

**A methodological caveat on the topline number itself:** `gs10` was also a "top-5 miss" by this scoring, but the underlying fact ("minimum 24h parenteral treatment") is genuinely restated by the source document in two places — page 19 (general guidance) and page 44 (a section specific to children). Retrieval found the page-44 restatement; the golden set only recorded page 19 as correct. This means the true hit@5 is likely *somewhat higher* than 59.1% — the golden set's single-canonical-page assumption slightly undercounts hits when the corpus legitimately repeats a fact. Worth keeping in mind before quoting the number as exact.

---

## 5. Generation evaluation (`eval/generation_eval.py`)

**What it measures:** runs the *actual* production path (threshold-filtered `search()` → `Malator.prompt()` → `gemini-3.1-flash-lite`) for every golden question, then scores the output on three independent metrics using a separate judge model:

1. **Faithfulness** — is every claim in the answer supported by the retrieved context? (Does not check if the context itself was correct — see below.)
2. **Answer relevance** — does the answer address the question, independent of correctness? (An honest "not found" scores high here.)
3. **Answer correctness** — does the answer match the golden `expected_answer`? (This is the metric that catches "confidently summarized the wrong document" — a failure mode faithfulness alone would miss, since a faithful summary of wrong context still scores well on faithfulness.)

### Judge model — an evolving decision, documented honestly

The judge is deliberately a **different model** from the generation model, to avoid the "self-check marking its own homework" problem — an LLM re-checking its own output has no independent reason to catch its own blind spots.

Getting a usable judge model took three attempts, driven by real free-tier constraints discovered live, not guessed in advance:

| Attempt | Model tried | Result |
|---|---|---|
| 1 | `gemini-2.5-pro` | 429 — quota limit of **0** requests/day on this free-tier key (Pro tier requires billing) |
| 2 | `gemini-2.5-flash` | 404 — model fully deprecated / no longer available to new accounts |
| 3 | `gemini-3.6-flash` | 429 — free tier capped at **20 requests/day total** for this "full" flash tier, insufficient for 72 judge calls |
| **Final** | `gemini-3.5-flash-lite` | Works — lite-tier models get far more generous free quotas than full-size flash tiers |

**Honest limitation:** the final judge (`gemini-3.5-flash-lite`) is a different model *generation* from the pipeline's `gemini-3.1-flash-lite`, but both are "lite" tier — a weaker independence guarantee than a full-size model or, better, a different provider entirely (e.g., Claude or GPT as judge) would offer. Worth revisiting if a second provider's API key becomes available.

Also switched from 3 separate judge calls per question to **1 combined call** requesting all three scores at once, halving total API usage for the run (48 calls instead of 96) — necessary given how tight the discovered quotas turned out to be.

### Results (baseline)

| Metric | Value |
|---|---|
| Avg faithfulness (all 24) | 1.00 |
| Avg relevance (all 24) | 1.00 |
| Avg correctness (all 24) | 0.79 |
| Avg correctness (22 indexed questions only) | 0.82 |
| Correctness = 0.5 ("honest decline") | 10 of 24 |
| Correctness = 1.0 (fully correct) | 14 of 24 |
| Correctness = 0.0 (confidently wrong) | **0 of 24** |

**Interpretation — and a prediction that turned out wrong:**

Before running this, the prediction was that the negative controls (`gs23`, `gs24`) would produce **confident, wrong-document answers**, since retrieval had already shown they receive threshold-passing chunks from the wrong PDF. That prediction was **wrong**, and it's recorded here rather than quietly dropped: both negative controls, and 8 of the 9 retrieval misses, produced honest declines ("the provided content does not contain...") instead of fabrication. Zero questions across the full set scored a confident wrong answer.

**What this actually shows:** the generation layer is well-behaved. The `Malator.py` prompt instruction ("use only this content, say so if the answer is not contained in it") is doing real protective work, including under the adversarial condition of being handed topically-similar-but-wrong context. **The system's failure mode, end to end, is retrieval gaps — not hallucination.** That reframes what the eventual self-check loop most needs to catch: not fact-checking generation against itself, but noticing when retrieval likely missed and flagging low confidence or triggering a second search — a materially different design target than originally assumed.

**Open item, not yet resolved:** faithfulness and relevance scored a perfect 1.0 on all 24 questions with zero exceptions. Plausible here, specifically because the system's only failure mode turned out to be honest declining rather than partial hallucination — but a judge that never finds fault across 24 varied examples deserves one direct sanity check before being fully trusted: feed it one deliberately-fabricated, unfaithful answer as a unit test and confirm it can score below 1.0 at all. **Not yet done.**

---

## 6. Summary of findings (evidence-backed claims for interview/portfolio use)

1. Ingestion is silently stale: half the source PDFs (3 of 6) were never embedded, and nothing in the pipeline would surface that on its own — found via an independent file-diff, not the RAG system itself.
2. Retrieval hit@5 is 59.1% (likely a slight undercount — see §4 caveat on repeated facts), with hit@1 at only 31.8%, and the gap is attributable to a specific, evidenced mechanism: chunks packing multiple distinct facts into one embedding, diluting the vector rather than merely "overlapping."
3. Both deliberately-planted negative controls confirm retrieval cannot distinguish topical similarity from actual relevance — a structural limit of pure cosine-similarity retrieval, not a tunable bug.
4. Generation, despite being handed wrong or absent context in every failure case observed, never fabricated a confident wrong answer — it declined honestly 10 out of 10 times context was inadequate. The system's weak point is retrieval, not generation, and that reframes the design target for the planned self-check loop.
5. Building and debugging the harness itself surfaced real free-tier model/quota constraints (three failed model choices before landing on a workable judge) — resolved by evidence (searching current docs, reading actual error messages) rather than by guessing repeatedly.

---

