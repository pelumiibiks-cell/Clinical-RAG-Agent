import math
import re
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
MALARIA_CONTEXT = BASE_DIR / "Malaria_db"
MALARIA_FAISS_INDEX = MALARIA_CONTEXT / "malaria_faiss.index"
MALARIA_METAD = MALARIA_CONTEXT / "malaria_metadata.pkl"
# A floor on obvious junk, not a relevance test. Measured on the golden set:
# genuine questions score 0.370-0.90 at rank 1 (median 0.723) while the two
# deliberately out-of-corpus controls score 0.627 and 0.604 -- squarely inside
# the positive range. No cosine cutoff can separate "we have this" from "we do
# not", so refusing to answer is the generator's job, not this number's. Kept
# low enough not to starve a weak-but-valid question of context.
SIMILARITY_THRESHOLD = 0.30

# lazy singletons -- loaded on first use, not on import, so importing this
# module doesn't crash (or pay the ~5-15s load cost) when Malaria_db/ is
# absent or nobody has called search() yet
_model = None
_index = None
_metadata = None
_bm25 = None


def index_exists() -> bool:
    return MALARIA_FAISS_INDEX.exists() and MALARIA_METAD.exists()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL)
    return _model


def get_index():
    global _index
    if _index is None:
        if not MALARIA_FAISS_INDEX.exists():
            raise FileNotFoundError(
                f"No FAISS index at {MALARIA_FAISS_INDEX}. Run embed_context() first."
            )
        _index = faiss.read_index(str(MALARIA_FAISS_INDEX))
    return _index


def get_metadata() -> list:
    global _metadata
    if _metadata is None:
        if not MALARIA_METAD.exists():
            raise FileNotFoundError(
                f"No metadata at {MALARIA_METAD}. Run embed_context() first."
            )
        with MALARIA_METAD.open("rb") as file:
            _metadata = pickle.load(file)
    return _metadata


def reset_cache() -> None:
    """Drop cached index/metadata so the next search() re-reads from disk. Call after a rebuild."""
    global _index, _metadata, _bm25
    _index = None
    _metadata = None
    _bm25 = None


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


class _BM25:
    """Okapi BM25 over the chunk corpus.

    Dense retrieval alone is weakest exactly where clinical questions are
    strongest: rare exact terms. "How many days should an ACT course include at
    minimum" scored 0.433 with all five candidates from the wrong document,
    because "ACT" and "3 days" are lexical facts, not semantic ones. Implemented
    here rather than pulled in as a dependency -- it is thirty lines and the
    project already ships a hand-rolled chunker.
    """

    __slots__ = ("df", "idf", "tf", "lengths", "avg_len", "n")

    K1 = 1.5
    B = 0.75

    def __init__(self, corpus: list):
        self.n = len(corpus)
        self.tf = []
        self.lengths = []
        self.df = {}
        for doc in corpus:
            tokens = _tokenize(doc)
            self.lengths.append(len(tokens))
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.idf = {
            t: math.log(1 + (self.n - d + 0.5) / (d + 0.5))
            for t, d in self.df.items()
        }

    def top_n(self, query: str, n: int) -> list:
        terms = [t for t in _tokenize(query) if t in self.idf]
        if not terms:
            return []
        scores = {}
        for i, counts in enumerate(self.tf):
            length = self.lengths[i] or 1
            norm = self.K1 * (1 - self.B + self.B * length / (self.avg_len or 1))
            total = 0.0
            for t in terms:
                f = counts.get(t)
                if f:
                    total += self.idf[t] * f * (self.K1 + 1) / (f + norm)
            if total > 0:
                scores[i] = total
        return sorted(scores.items(), key=lambda kv: -kv[1])[:n]


def get_bm25() -> "_BM25":
    global _bm25
    if _bm25 is None:
        _bm25 = _BM25([m["Content"] for m in get_metadata()])
    return _bm25


def _fuse(dense: list, lexical: list, k: int = 60) -> dict:
    """Reciprocal rank fusion. Combines two rankings without needing their
    scores to be on comparable scales -- cosine similarity and BM25 are not."""
    fused = {}
    for rank, idx in enumerate(dense, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(lexical, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


def rank_candidates(
    query: str,
    pool: int = 30,
    hybrid: bool = True,
    dedupe_pages: bool = True,
) -> list:
    """Rank chunks for a query and return [(index, cosine), ...], best first.

    No threshold applied. This is the single ranking path: search() adds the
    threshold on top, and the retrieval eval calls this directly so that what it
    measures is what production actually ranks. The eval used to reach past
    search() into index.search() by hand, which meant it could not see any
    ranking change made here.
    """
    model = get_model()
    index = get_index()
    metadata = get_metadata()

    embeddings = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    pool = min(max(pool, 1), index.ntotal)
    distances, indicies = index.search(embeddings, pool)

    cosine = {}
    dense_order = []
    for score, idx in zip(distances[0], indicies[0]):
        if idx < 0:
            continue
        idx = int(idx)
        cosine[idx] = float(score)
        dense_order.append(idx)

    if hybrid:
        lexical_order = [i for i, _ in get_bm25().top_n(query, pool)]
        fused = _fuse(dense_order, lexical_order)
        # A lexical-only hit has no cosine score yet; reconstruct its vector so
        # every returned candidate carries a comparable similarity.
        missing = [i for i in lexical_order if i not in cosine]
        if missing:
            vectors = np.vstack([index.reconstruct(i) for i in missing])
            for i, sim in zip(missing, vectors @ embeddings[0]):
                cosine[i] = float(sim)
        ordered = sorted(fused, key=lambda i: -fused[i])
    else:
        ordered = dense_order

    ranked = []
    seen = set()
    for idx in ordered:
        if dedupe_pages:
            entry = metadata[idx]
            key = (entry.get("Source"), entry.get("Page Number"), entry.get("Page End"))
            if key in seen:
                continue
            seen.add(key)
        ranked.append((idx, cosine.get(idx, 0.0)))
    return ranked


def search(
    query: str,
    top_results: int = 5,
    threshold: float = SIMILARITY_THRESHOLD,
    candidate_pool: int = 30,
    hybrid: bool = True,
    dedupe_pages: bool = True,
) -> list:
    """Retrieve chunks for a query.

    Pulls a wide candidate pool, fuses dense and lexical rankings, collapses
    duplicate hits from the same page, then cuts to top_results. The old version
    took the raw dense top-5 and returned it: with 17% of corpus text duplicated
    by chunk overlap, one question's five slots held only three distinct pages.

    Prints nothing. It used to print every result including full chunk text to
    stdout on every call, roughly 2KB per query into the server log.
    """
    metadata = get_metadata()
    ranked = rank_candidates(query, pool=max(top_results, candidate_pool),
                             hybrid=hybrid, dedupe_pages=dedupe_pages)

    results = []
    for idx, score in ranked:
        if score < threshold:
            continue
        results.append({**metadata[idx], "Score": score})
        if len(results) >= top_results:
            break
    return results


def main():
    while True:
        query = input("Speak wise one, What do you seek: ")
        if query.lower() == 'exit':
            print("Knowledge onto you wise one")
            sys.exit()
        if not query:
            print("But you must definitely aquire knowledge")
            continue
        search(query)


if __name__ == "__main__":
    main()
