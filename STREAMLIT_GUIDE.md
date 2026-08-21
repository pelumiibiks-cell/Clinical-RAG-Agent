# Building the Malaria RAG UI in Streamlit: a step-by-step guide

This walks through how `streamlit_app.py` and everything under `ui/` and `views/`
came together, in the order you'd actually build it, for someone who has never
touched Streamlit before. Each section adds one idea and shows the code that
idea produced. By the end you'll understand not just *what* the app does but
*why* it's shaped this way.

---

## 1. What we're building

```
┌─────────────────────────────────────────────┐
│  🩺 Malaria RAG            [sidebar: top_k,  │
│                              threshold,       │
│  💬 Chat                    clear history]   │
│  📚 Knowledge Base                            │
│  🔍 Inspector                                 │
│  ℹ️ About                                     │
│                                                │
│  ┌──────────────────────────────────────┐    │
│  │  chat / table / debug view goes here  │    │
│  └──────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

Four pages sharing one sidebar: **Chat** (ask questions, get grounded
answers with citations), **Knowledge Base** (see which PDFs are indexed,
upload new ones, rebuild), **Inspector** (retrieval only, no LLM call, for
debugging), **About** (what's configured and how the pipeline works).

---

## 2. The mental model: Streamlit reruns your whole script

This is the one idea that makes everything else make sense, so it comes
first, before any code.

A normal GUI framework builds a window once, then reacts to events —
click a button, one handler fires, the rest of the window stays as it was.
Streamlit doesn't work that way. **Every time the user does anything** —
types in a box, clicks a button, moves a slider — Streamlit throws away
the whole screen and runs your Python script again from the top.

Think of it like a whiteboard in a meeting room. Nobody edits one corner
of the board — every time someone speaks, the whole board gets wiped and
redrawn from scratch, in order, top to bottom. If you want something to
survive from one redraw to the next (the conversation history, say), you
can't just leave it drawn on the board — you have to write it down
somewhere that isn't the whiteboard. That "somewhere else" is
**session state**, covered in the next section.

Another way to say it: your script is a recipe that gets re-cooked from
step one every time, not a house that gets built once and then stands.
`streamlit_app.py` runs top to bottom on literally every interaction in
this app, including every message you send in Chat.

---

## 3. Install and first run

```bash
cd Malaria_Env
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit opens a local server (default `http://localhost:8501`) and a
browser tab. Edit any file and save — Streamlit notices and offers to
rerun automatically. That fast edit-save-see loop is most of why people
like it for internal tools.

---

## 4. Session state: the notebook that survives the wipe

Local variables in a Streamlit script die at the end of every rerun,
exactly like a whiteboard being wiped. `st.session_state` is a
dictionary that Streamlit keeps *outside* that cycle, tied to your
browser tab — a notebook in your pocket that the whiteboard-wiping
doesn't touch.

`ui/state.py`:

```python
def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.4
```

The `if ... not in` guard matters: this function runs on *every* rerun
(it's called at the top of `streamlit_app.py`), so without the guard
you'd wipe the chat history back to empty every time someone typed a
message. `messages` is exactly why it needs to live here — a plain
Python list defined inside the script body would reset to `[]` on the
very next rerun, i.e. the moment after the assistant replies.

---

## 5. Caching: don't rebuild what didn't change

Two different caching decorators solve two different problems here, and
the analogy for each is different.

**`@st.cache_resource`** — for things that are expensive to create and
shouldn't be duplicated: a database connection, a loaded ML model. Think
of it as hiring one receptionist for the whole office instead of hiring
a brand new one every time a visitor walks in. In `ui/state.py`:

```python
@st.cache_resource(show_spinner="Loading embedding model and index (first run only)...")
def load_pipeline():
    if not rag_core.index_exists():
        return None
    import malaria_embed_query as mq
    mq.get_model()
    mq.get_index()
    mq.get_metadata()
    return True
```

The `all-MiniLM-L6-v2` embedding model takes real time to load (torch +
weights). Without this decorator, every single chat message would
reload it from scratch, because — remember section 2 — the whole script
reruns on every interaction. `@st.cache_resource` remembers the result
*across reruns and across users*, so it loads once per server process,
not once per message.

**`@st.cache_data`** — for data (not live objects) that's cheap to
compute but not free, and that you want to reuse until something
invalidates it. Think of it as photocopying a document to keep on your
desk instead of walking to the archive room every single time you need
to glance at it — until the archive changes, in which case you throw
out the photocopy and go make a new one.

```python
@st.cache_data(ttl=None, show_spinner=False)
def cached_index_status():
    return rag_core.index_status()
```

`index_status()` scans `M_pdfs/` and compares it against the index
metadata — cheap, but no reason to redo it on every rerun if nothing
changed. When you rebuild the index (Knowledge Base page), we explicitly
throw out both photocopies:

```python
def invalidate_index_status() -> None:
    cached_index_status.clear()
    load_pipeline.clear()
```

---

## 6. Widgets and keys: name tags on chairs

A widget like `st.slider(...)` isn't just drawn once — it's redrawn on
*every* rerun, and Streamlit has to figure out "is this the same slider
as before, or a brand new one?" so it knows whether to keep the value
the user last set. It does that by identity — position in the script,
or an explicit `key=` you give it. Think of it as a name tag on a chair:
without one, Streamlit can only guess who's supposed to sit where when
the room gets rebuilt; with one, the same person reliably ends up back
in the same seat.

In this app, the sidebar widgets read from and write straight back into
`session_state`, so their values survive reruns without needing an
explicit `key`:

```python
st.session_state.top_k = st.slider("Passages to retrieve (top_k)", 1, 10, st.session_state.top_k)
st.session_state.threshold = st.slider("Similarity threshold", 0.0, 1.0, st.session_state.threshold, step=0.05)
```

---

## 7. Wiring retrieval in

This app doesn't call an LLM cold — it retrieves relevant passages
*first*, then hands only those to the model. Three ideas, three
analogies:

- **Embeddings** turn text into a list of numbers — a point in a very
  high-dimensional space — such that texts with similar *meaning* end
  up at nearby points. Think of it as GPS coordinates, except the map
  isn't geography, it's meaning: "how do I treat severe malaria" and
  "management of complicated malaria cases" land close together even
  though they don't share many words.
- **FAISS** is the card catalogue. Given the coordinates of your
  question, it doesn't read every book — it jumps straight to the
  nearest cards (`IndexFlatIP.search`) and hands you the closest
  matches.
- **The similarity threshold** is a bouncer with a minimum standard.
  FAISS will always hand back its "top 5 closest," even if none of them
  are actually close — the threshold (`0.4` by default) is what refuses
  entry to a match that's merely the *least bad* option rather than a
  genuinely relevant one.

All three live in `malaria_embed_query.search()`, and `rag_core.py`
wraps it for the UI:

```python
def retrieve_only(query: str, top_k: int = 5, threshold: float = 0.0) -> list:
    return mq.search(query, top_results=top_k, threshold=threshold)
```

The **Inspector** page (section 10 of the app, not this guide) sets
`threshold=0.0` deliberately — it shows you *everything* FAISS found,
including what the bouncer would normally turn away, so you can see the
threshold's effect instead of taking it on faith.

---

## 8. Streaming: watching the letter get written

`rag_core.answer()` waits for the full response before returning it.
`rag_core.answer_stream()` instead yields pieces of text as the model
produces them — the difference between waiting for a sealed envelope to
arrive versus watching someone write the letter in front of you. For a
chat UI, streaming matters because a multi-second wait with nothing on
screen reads as broken, while the same wait with text visibly
appearing reads as working.

Streamlit has a widget built exactly for this, `st.write_stream`, which
takes any Python generator and renders pieces as they arrive:

```python
chunks, pieces = rag_core.answer_stream(query, top_k=..., threshold=...)
if not chunks:
    render_no_results_notice(query)
else:
    answer_text = st.write_stream(pieces)
    render_sources(chunks)
```

Note that retrieval happens *before* streaming starts — `chunks` is
already known by the time `answer_stream` returns, only the generation
step is streamed.

---

## 9. Sources and citations: showing your working

An answer with no citation is a claim you have to take on faith. This
app never does that — every assistant message carries the exact
passages it was built from, source filename, page number, similarity
score:

```python
def render_sources(chunks: list) -> None:
    with st.expander(f"Sources ({len(chunks)})"):
        for c in chunks:
            st.markdown(f"**{c['Source']}** · page {c['Page Number']} · similarity {c['Score']:.2f}")
            st.caption(c["Content"])
```

For a medical Q&A tool specifically, this isn't a nice-to-have — it's
what makes the answer checkable at all. If retrieval found nothing above
the threshold, the app says so explicitly instead of asking the model to
answer anyway (see `render_no_results_notice` and the `if not chunks`
branch above) — a confident-sounding non-answer is worse than a visible
"nothing matched."

---

## 10. The Knowledge Base page: the stale catalogue problem

This page exists because of something that actually happened to this
project. `M_pdfs/` holds six PDFs. At one point, only three of them had
ever been embedded into the index — three new shelves of books had
arrived at the library, and nobody told the catalogue. The librarian
(retrieval) kept confidently handing over pages from the *old* shelves
for every question, never once saying "I don't have that" — because as
far as the catalogue was concerned, those three new books didn't exist.
Nothing in the original code checked that the index matched what was
actually on disk, so this went unnoticed until someone deliberately
audited it.

`rag_core.index_status()` is the fix — it diffs `M_pdfs/` against the
index metadata every time the page loads:

```python
def index_status() -> dict:
    on_disk = {p.name for p in pdf_folder.iterdir() if p.suffix.lower() == ".pdf"}
    indexed = {m["Source"] for m in metadata}
    return {
        "not_indexed": on_disk - indexed,   # new books, catalogue doesn't know
        "orphaned": indexed - on_disk,      # catalogue entries for books that are gone
        ...
    }
```

The Knowledge Base page turns that into a table with a badge per file
(✅ indexed / ❌ not indexed / ⚠️ orphaned), and a warning banner that
follows you to the Chat page too (`render_drift_banner`), so a stale
index is something you're told about, not something you discover by
getting a wrong answer.

Rebuilding runs `embed_context()` again over every PDF currently in
`M_pdfs/`, with a progress callback so a multi-minute rebuild (model
encoding hundreds of chunks) shows visible progress instead of a frozen
screen:

```python
def on_progress(stage, current, total, label):
    if stage == "reading":
        bar.progress(current / total, text=f"Reading {label} ({current}/{total})")
    elif stage == "embedding":
        bar.progress(0.9, text=f"Embedding {label}...")
    elif stage == "done":
        bar.progress(1.0, text="Saved")

rag_core.rebuild_index(progress_callback=on_progress)
```

`st.status(...)` wraps the whole operation in a collapsible log block
that shows "running" while it works and "complete" when it's done —
the more polished cousin of a bare progress bar.

---

## 11. Multipage layout: rooms in a clinic

A single-page app that tries to be a chat window, an admin panel, and a
debug console at once gets cluttered fast. Streamlit's `st.navigation`
lets you split the app into separate pages that share one process and
one sidebar — think of it as one clinic building with separate rooms
(reception, consultation, records) rather than cramming every function
into one room.

`streamlit_app.py`:

```python
chat_page = st.Page(chat.render, title="Chat", icon="💬", default=True)
kb_page = st.Page(knowledge_base.render, title="Knowledge Base", icon="📚")
inspector_page = st.Page(inspector.render, title="Inspector", icon="🔍")
about_page = st.Page(about.render, title="About", icon="ℹ️")

pg = st.navigation([chat_page, kb_page, inspector_page, about_page])
pg.run()
```

Each page is just a Python function (`render()` in `views/chat.py`,
`views/knowledge_base.py`, etc.) — `st.Page` accepts either a file path
or a callable; this app uses callables so each view is a normal,
testable function rather than a script that only works when run as a
page.

---

## 12. Theming

`.streamlit/config.toml` sets the app's color palette once, applied
everywhere, instead of hand-styling each page:

```toml
[theme]
primaryColor = "#2E7D6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F1F5F3"
textColor = "#1A1A1A"
```

This file is picked up automatically — no import needed in the Python
code.

---

## 13. Troubleshooting

**First load is slow (5-15 seconds) before anything responds.** That's
the embedding model loading. It's a one-time cost per server process
thanks to `@st.cache_resource` (section 5) — the second question you ask
will be fast. If it's slow on *every* question, the caching isn't
working; check that `load_pipeline` isn't being redefined somewhere
outside `ui/state.py`.

**Changes to the index don't show up.** The cached functions in section
5 don't know a rebuild happened unless you tell them —
`invalidate_index_status()` must be called after any rebuild or upload.
It already is, inside `knowledge_base.py`, but if you add a new way to
modify the index, remember to call it there too.

**The app crashes on startup with a FAISS read error.** This means
`Malaria_db/` doesn't exist yet — nobody has run the ingestion step. As
of this refactor, that no longer crashes on *import* (see
`malaria_embed_query.index_exists()`), but the app still needs the index
built before Chat or Inspector can do anything useful. Go to Knowledge
Base and click Rebuild.

**A missing `.env.mal` / API key.** `config_mal.py` raises a
`ValueError` if `THE_KEY` isn't set. That's currently an unhandled
startup failure for this app (not caught anywhere) — if you deploy this
somewhere the key might legitimately be absent, wrap the `rag_core`
import in a try/except and show setup instructions instead of a raw
traceback, the same pattern used for the missing-index case.

---

## 14. Deployment notes

- **Never commit `.env.mal`.** It already sits in `.gitignore`
  alongside `*.env` — leave that alone.
- If you deploy to Streamlit Community Cloud or similar, the API key
  goes into that platform's secrets manager, not into `config.toml` or
  any file in this repo. `config_mal.py` reads from `.env.mal` locally;
  a hosted deployment would need `THE_KEY` set as an actual environment
  variable instead (a small follow-up to `config_mal.py`, not covered
  by this guide).
- The FAISS index and metadata pickle (`Malaria_db/`) are generated
  artifacts, not source — decide deliberately whether to commit them or
  rebuild them as part of deployment. Rebuilding needs the PDFs in
  `M_pdfs/` and takes a few minutes.
