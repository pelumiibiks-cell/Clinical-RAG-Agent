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
    """M_pdfs/ is gitignored, so a fresh clone won't have it -- treat that as
    'no PDFs yet' and create it, rather than crashing the whole app."""
    pdf_dir = BASE_DIR / "M_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return pdf_dir


# Every bullet glyph these PDFs actually use. The old character class listed
# U+2022 twice (once literally, once as an escape) and missed the rest: the
# corpus carries 214 U+F0A1 (Wingdings bullet in the private-use area), 47
# literal "z" (Wingdings bullet mapped to ASCII by the font encoding) and 26
# U+25CA, none of which could ever start a new sentence.
_BULLET_CHARS = "•●▪■◊‣⁃∙·"
_BULLET_CLASS = f"[{_BULLET_CHARS}]"

# A bare "z" between a space and a following word is a Wingdings bullet, not a
# word. Anchored tightly so it cannot fire on real words starting with z.
_WINGDINGS_BULLET = re.compile(r"(?:^|(?<=\s))z{1,2}(?=\s+[A-Za-z0-9])")


def split_into_sentences(text: str) -> list:
    """Join PDF line-wrap newlines into single lines/paragraphs, then split on
    sentence boundaries AND on bullet markers.

    Splitting only after . ! ? was the real retrieval bug. Clinical guidelines
    are bullet lists and bullet items mostly do not end in a period: 1503 of the
    1715 bullet markers in this corpus (88%) had no preceding sentence
    punctuation, so they could never be a split point. That left 385 chunks
    (20% of the index) packing two or more unrelated clinical facts into one
    embedding, and three facts averaged into one vector is a mediocre match for
    each and a strong match for none. It is exactly the dilution
    EVALUATION_REPORT.md diagnosed, which moving to sentence-aware chunking
    did not actually fix.
    """
    # collapse single newlines (mid-sentence line wraps) into spaces,
    # but keep blank-line paragraph breaks as sentence-boundary signals
    normalized = re.sub(r"\n\s*\n", ". ", text)   # blank line -> forced break
    # Repair hyphenation across a line wrap before newlines become spaces,
    # otherwise "artemisinin-\nbased" survives as two separate tokens.
    normalized = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", normalized)
    normalized = re.sub(r"\n", " ", normalized)   # remaining wraps -> space
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return []

    # Normalize Wingdings-as-ASCII bullets to a real bullet so the split sees them.
    normalized = _WINGDINGS_BULLET.sub("•", normalized)

    # Split after . ! ? as before, and additionally before any bullet marker
    # whether or not sentence punctuation precedes it.
    parts = re.split(
        rf"(?<=[.!?])\s+(?=[A-Z0-9]|{_BULLET_CLASS})|\s+(?={_BULLET_CLASS}\s)",
        normalized,
    )

    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Drop a leading bullet glyph: it carries no meaning for the embedding,
        # and having one on some chunks but not others is pure noise.
        part = re.sub(rf"^{_BULLET_CLASS}\s*", "", part).strip()
        if part:
            sentences.append(part)
    return sentences


def strip_running_headers(pages: list) -> list:
    """Remove lines that repeat across most pages of a document.

    Nothing stripped these before. 14 of the 21 chunks from
    WHO_Treatment_Guidelines_Olumense.pdf began with the running header
    "N | Guidelines for the Treatment of Malaria" -- two thirds of that
    document's vectors carrying identical boilerplate, and that document is the
    expected source for six golden-set questions.
    """
    if len(pages) < 4:
        return pages

    counts = {}
    for text in pages:
        for line in {ln.strip() for ln in text.split("\n") if ln.strip()}:
            # Page numbers differ per page, so normalize digits: "9 | Guidelines"
            # and "10 | Guidelines" are the same header.
            key = re.sub(r"\d+", "#", line)
            if len(key) < 80:
                counts[key] = counts.get(key, 0) + 1

    threshold = max(3, int(len(pages) * 0.5))
    boilerplate = {k for k, c in counts.items() if c >= threshold}
    if not boilerplate:
        return pages

    return [
        "\n".join(
            ln for ln in text.split("\n")
            if re.sub(r"\d+", "#", ln.strip()) not in boilerplate
        )
        for text in pages
    ]


def dedupe_adjacent_lines(text: str) -> str:
    """Collapse consecutive identical lines.

    Some of these PDFs draw section titles with a drop-shadow layer, so PyMuPDF
    emits the same run twice in a row. Left in, it doubles a heading's weight
    inside the chunk's embedding.
    """
    out = []
    previous = None
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and stripped == previous:
            continue
        out.append(line)
        previous = stripped
    return "\n".join(out)


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


def chunk_sentences_paged(pairs: list) -> list:
    """Group (sentence, page) pairs into (chunk, first_page) tuples.

    Chunking used to run strictly inside the per-page loop, so no chunk could
    ever span a page break and any instruction split across one was permanently
    severed -- it also left page-tail fragments, which is where the 2.6% of
    chunks under 100 chars came from. Sentences are now fed in per document, and
    a chunk is attributed to the page its first sentence came from.
    """

    def safe_overlap_tail(buf: list) -> list:
        tail = buf[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
        tail_len = sum(len(s) + 1 for s, _ in tail)
        # normally the overlap tail is one short natural sentence -- but if
        # it's a hard_split() fragment that's already near MAX_CHUNK_CHARS
        # on its own, carrying it forward leaves no room for new content
        # without exceeding the cap again. Drop the overlap in that case.
        if tail_len > MAX_CHUNK_CHARS * 0.5:
            return []
        return tail

    def emit(buf: list) -> tuple:
        pages = [pg for _, pg in buf]
        # (text, first page, last page). A chunk that spans a page break really
        # does contain content from both, so it has to say so -- attributing it
        # only to its first sentence's page made two golden-set questions look
        # like misses when the answer was retrieved.
        return " ".join(s for s, _ in buf), pages[0], pages[-1]

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
    expanded = []
    for sentence, page in pairs:
        if len(sentence) > MAX_CHUNK_CHARS:
            expanded.extend((piece, page) for piece in hard_split(sentence, MAX_CHUNK_CHARS))
        else:
            expanded.append((sentence, page))

    for sentence, page in expanded:
        sentence_len = len(sentence) + 1  # +1 for the joining space
        if current and current_len + sentence_len > MAX_CHUNK_CHARS:
            chunks.append(emit(current))
            # start next chunk with the overlap tail of this one
            current = safe_overlap_tail(current)
            current_len = sum(len(s) + 1 for s, _ in current)
            pending_new_content = False
            # the tail alone passed the safety check above, but tail +
            # THIS sentence together might still overflow -- drop the tail
            # rather than start the next chunk already oversized
            if current_len + sentence_len > MAX_CHUNK_CHARS:
                current = []
                current_len = 0

        current.append((sentence, page))
        current_len += sentence_len
        pending_new_content = True

        if current_len >= TARGET_CHUNK_CHARS:
            chunks.append(emit(current))
            current = safe_overlap_tail(current)
            current_len = sum(len(s) + 1 for s, _ in current)
            pending_new_content = False

    if current and pending_new_content:
        chunks.append(emit(current))

    return chunks


def chunk_sentences(sentences: list) -> list:
    """String-only wrapper over chunk_sentences_paged, kept for callers that
    have no page information."""
    return [chunk for chunk, _, _ in chunk_sentences_paged([(s, 0) for s in sentences])]


def embed_context(progress_callback=None) -> None:
    """progress_callback(stage, current, total, label), called at each PDF
    and at the embedding step. stage is one of "reading", "embedding", "done"."""
    model = SentenceTransformer(MODEL)
    chunks = []
    metadata = []

    pdf_folder = locate_dir()
    pdf_files = [
        p for p in sorted(pdf_folder.iterdir())
        if p.is_file() and p.suffix.lower() == ".pdf"
    ]

    for i, pdf_file in enumerate(pdf_files, start=1):
        print("Processing ", pdf_file.name)
        if progress_callback:
            progress_callback("reading", i, len(pdf_files), pdf_file.name)
        doc = fitz.open(pdf_file)

        # Read the whole document first: running headers can only be detected by
        # looking across pages, and chunks can only span a page break if the
        # sentences are collected before chunking rather than after.
        raw_pages = [page.get_text() for page in doc]
        doc.close()

        cleaned_pages = strip_running_headers(raw_pages)

        tagged_sentences = []
        for page_number, text in enumerate(cleaned_pages, start=1):
            if not text:
                continue
            text = dedupe_adjacent_lines(text)
            for sentence in split_into_sentences(text):
                # 1-indexed: this is what a reader sees in a PDF viewer and what
                # the UI and the model are both shown. It used to be 0-indexed
                # from enumerate() while being displayed as a human page number,
                # so every citation in the app was one page low.
                tagged_sentences.append((sentence, page_number))

        for chunk, first_page, last_page in chunk_sentences_paged(tagged_sentences):
            if len(chunk) < MIN_CHUNK_CHARS:
                continue
            chunks.append(chunk)
            metadata.append(
                {
                    "Source": pdf_file.name,
                    "Page Number": first_page,
                    "Page End": last_page,
                    "Content": chunk,
                }
            )

    print("Generating Embedding")
    if progress_callback:
        progress_callback("embedding", len(pdf_files), len(pdf_files), f"{len(chunks)} chunks")

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
    if progress_callback:
        progress_callback("done", len(pdf_files), len(pdf_files), "index saved")


if __name__ == "__main__":
    candidate = locate_dir()
    embed_context()