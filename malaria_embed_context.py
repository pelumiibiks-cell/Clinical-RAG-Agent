import os
import re
import faiss
import fitz
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
MALARIA_CONTEXT = BASE_DIR / "Malaria_db"
MALARIA_FAISS_INDEX = MALARIA_CONTEXT / "malaria_faiss.index"
MALARIA_METAD = MALARIA_CONTEXT / "malaria_metadata.pkl"

# --- new chunking parameters, replacing the old fixed 1000/650 window ---
TARGET_CHUNK_CHARS = 400   # aim for chunks around this size...
MAX_CHUNK_CHARS = 600      # ...but never exceed this without a natural sentence break
OVERLAP_SENTENCES = 1      # carry just the last sentence forward for continuity,
                           # not a large fixed character overlap
MIN_CHUNK_CHARS = 20       # drop fragments shorter than this -- stray page numbers,
                           # orphaned headers/footers, not real content


def locate_dir() -> Path:
    for candidate in [BASE_DIR / "M_pdfs"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Specified Directory cannot be found in directory")


def split_into_sentences(text: str) -> list:
    """Join PDF line-wrap newlines into single lines/paragraphs, then split
    on sentence-ending punctuation. Not perfect sentence segmentation --
    PDF-extracted medical text with bullets and abbreviations makes that
    hard -- but it's a real improvement over cutting at an arbitrary
    character count with no regard for sentence boundaries at all."""
    # collapse single newlines (mid-sentence line wraps) into spaces,
    # but keep blank-line paragraph breaks as sentence-boundary signals
    normalized = re.sub(r"\n\s*\n", ". ", text)   # blank line -> forced break
    normalized = re.sub(r"\n", " ", normalized)   # remaining wraps -> space
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return []

    # split after ., !, or ? when followed by a space and a capital letter,
    # a bullet-like symbol, or a digit (common list-item start)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9•\u2022])", normalized)
    return [s.strip() for s in sentences if s.strip()]


def hard_split(text: str, max_len: int) -> list:
    """Force-split a single overlong 'sentence' at word boundaries.
    Needed for things like table-of-contents lines ('Preface....3
    Introduction....5') which have no space-then-capital pattern for
    split_into_sentences to break on, so they arrive here as one
    multi-thousand-character unit with nothing else to split it."""
    words = text.split(" ")
    pieces = []
    current = []
    current_len = 0

    for word in words:
        # a single space-less token (e.g. a run of dot-leaders with no
        # gaps at all) can itself be longer than max_len -- word-boundary
        # splitting alone can't help here, so cut it at raw character
        # intervals as a last resort before it enters the normal buffer
        if len(word) > max_len:
            if current:
                pieces.append(" ".join(current))
                current = []
                current_len = 0
            for i in range(0, len(word), max_len):
                pieces.append(word[i : i + max_len])
            continue

        word_len = len(word) + 1
        if current and current_len + word_len > max_len:
            pieces.append(" ".join(current))
            current = []
            current_len = 0
        current.append(word)
        current_len += word_len

    if current:
        pieces.append(" ".join(current))

    return pieces


def chunk_sentences(sentences: list) -> list:
    """Group sentences into chunks up to MAX_CHUNK_CHARS, preferring to stop
    around TARGET_CHUNK_CHARS. Never splits a sentence in half. Carries the
    last OVERLAP_SENTENCES sentence(s) of a chunk into the start of the next
    one for continuity."""

    def safe_overlap_tail(buf: list) -> list:
        tail = buf[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
        tail_len = sum(len(s) + 1 for s in tail)
        # normally the overlap tail is one short natural sentence -- but if
        # it's a hard_split() fragment that's already near MAX_CHUNK_CHARS
        # on its own, carrying it forward leaves no room for new content
        # without exceeding the cap again. Drop the overlap in that case.
        if tail_len > MAX_CHUNK_CHARS * 0.5:
            return []
        return tail
    chunks = []
    current = []
    current_len = 0
    pending_new_content = False  # tracks whether `current` holds anything
                                  # beyond a carried-over overlap tail --
                                  # without this, a chunk that ends right
                                  # after a flush gets its overlap tail
                                  # re-emitted as a duplicate final chunk

    # a "sentence" that's already longer than MAX_CHUNK_CHARS has nothing
    # normal to split on (see hard_split docstring) -- expand it into
    # word-boundary pieces now so nothing downstream ever exceeds the cap
    expanded_sentences = []
    for sentence in sentences:
        if len(sentence) > MAX_CHUNK_CHARS:
            expanded_sentences.extend(hard_split(sentence, MAX_CHUNK_CHARS))
        else:
            expanded_sentences.append(sentence)
    sentences = expanded_sentences

    for sentence in sentences:
        sentence_len = len(sentence) + 1  # +1 for the joining space
        if current and current_len + sentence_len > MAX_CHUNK_CHARS:
            chunks.append(" ".join(current))
            # start next chunk with the overlap tail of this one
            current = safe_overlap_tail(current)
            current_len = sum(len(s) + 1 for s in current)
            pending_new_content = False
            # the tail alone passed the safety check above, but tail +
            # THIS sentence together might still overflow (e.g. a small
            # overlap sentence immediately followed by another near-cap
            # hard_split fragment) -- drop the tail rather than start the
            # next chunk already oversized
            if current_len + sentence_len > MAX_CHUNK_CHARS:
                current = []
                current_len = 0

        current.append(sentence)
        current_len += sentence_len
        pending_new_content = True

        if current_len >= TARGET_CHUNK_CHARS:
            chunks.append(" ".join(current))
            current = safe_overlap_tail(current)
            current_len = sum(len(s) + 1 for s in current)
            pending_new_content = False

    if current and pending_new_content:
        chunks.append(" ".join(current))

    return chunks


def embed_context() -> None:
    model = SentenceTransformer(MODEL)
    chunks = []
    metadata = []

    pdf_folder = locate_dir()

    for pdf_file in sorted(pdf_folder.iterdir()):
        if not pdf_file.is_file() or pdf_file.suffix.lower() != (".pdf"):
            continue
        print("Processing ", pdf_file.name)
        doc = fitz.open(pdf_file)

        for page_number, page in enumerate(doc):
            text = page.get_text()
            if not text:
                continue

            sentences = split_into_sentences(text)
            page_chunks = [
                c for c in chunk_sentences(sentences) if len(c) >= MIN_CHUNK_CHARS
            ]

            for chunk in page_chunks:
                chunks.append(chunk)
                metadata.append(
                    {
                        "Source": pdf_file.name,
                        "Page Number": page_number,
                        "Content": chunk,
                    }
                )
        doc.close()

    print("Generating Embedding")

    embedding = model.encode(
        chunks, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )

    print(embedding.shape)

    dimensions = embedding.shape[1]
    index = faiss.IndexFlatIP(dimensions)
    index.add(embedding)
    print(f"Total number of embeddings: {index.ntotal}")

    MALARIA_CONTEXT.mkdir(exist_ok=True)
    faiss.write_index(index, str(MALARIA_FAISS_INDEX))
    with MALARIA_METAD.open("wb") as file:
        pickle.dump(metadata, file)

    print("Index and Metadata Saved Sucessfully")


if __name__ == "__main__":
    candidate = locate_dir()
    embed_context()