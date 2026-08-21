import sys
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
MALARIA_CONTEXT = BASE_DIR / "Malaria_db"
MALARIA_FAISS_INDEX = MALARIA_CONTEXT / "malaria_faiss.index"
MALARIA_METAD = MALARIA_CONTEXT / "malaria_metadata.pkl"
SIMILARITY_THRESHOLD = 0.4

# lazy singletons -- loaded on first use, not on import, so importing this
# module doesn't crash (or pay the ~5-15s load cost) when Malaria_db/ is
# absent or nobody has called search() yet
_model = None
_index = None
_metadata = None


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
    global _index, _metadata
    _index = None
    _metadata = None


def search(query: str, top_results: int = 5, threshold: float = SIMILARITY_THRESHOLD) -> list:
    model = get_model()
    index = get_index()
    metadata = get_metadata()

    embeddings = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)  # Query comes in list

    distances, indicies = index.search(embeddings, top_results)

    print("---Top Results---")
    results = []
    found = False

    for rank, (score, idx) in enumerate(zip(distances[0], indicies[0]), start=1):  # Count the zip,close brackets
        if idx < 0:
            continue
        if score > threshold:
            found = True
            entry = metadata[int(idx)]
            results.append({**entry, "Score": float(score)})  # copy, not a reference into metadata

            print(f"{rank}\nScore / 1 : {score:.2f}")
            print(f"Source : {entry.get('Source')}")
            print(f"Page : {entry.get('Page Number')}")
            print("-" * 30)
            print(f"Content : {entry.get('Content')}")

    if not found:
        print("No relevant Results...")
    return results


def main():
    while True:
        query = input("Speak wise one, What do you seek: ")
        if query.lower() == 'exit':
            print("Knowledge onto you wise one")
            sys.exit()
        if not query:
            ("But you must definitely aquire knowledge")
            continue
        search(query)


if __name__ == "__main__":
    main()
