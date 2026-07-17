# Week 1 Lab: Ingest, Chunk, and Measure Retrieval on a Real 200-Page PDF

> You are building the **offline half** of a RAG system — `load → parse → chunk → embed → store` — and proving it works with **numbers**, not vibes. There is **no answer generation this week**. The mantra: *generation quality is capped by retrieval quality, and retrieval quality is capped by parsing + chunking.* If `recall@k` is bad here, no prompt engineering downstream can save you. By the end you will have run a grid of `{2 parsers} × {3 chunkers} × {k∈3,5,10}`, produced a sorted results table with `recall@k` + `MRR`, and a green `pytest` gate that asserts your best config clears a recall@5 threshold you commit to.
>
> Read these lectures first (they are the theory behind every step below):
> - [../lectures/01-rag-pipelines-and-failure-points.md](../lectures/01-rag-pipelines-and-failure-points.md) — the two pipelines + the 7 failure points (your debugging checklist)
> - [../lectures/02-layout-aware-pdf-parsing.md](../lectures/02-layout-aware-pdf-parsing.md) — why PDFs are a drawing format; PyMuPDF vs Docling
> - [../lectures/03-chunking-strategies.md](../lectures/03-chunking-strategies.md) — fixed / recursive / structure-aware / sentence-window / parent-doc
> - [../lectures/04-retrieval-only-evaluation.md](../lectures/04-retrieval-only-evaluation.md) — recall@k, MRR, golden sets, leakage

**Est. time:** ~8-9 hrs · **You will need:** Python 3.10-3.12, a laptop (no GPU, no paid API), ~2 GB disk for models, and Git-Bash (Windows) or a POSIX shell. Everything runs **locally and free**: embeddings via `sentence-transformers` (`BAAI/bge-small-en-v1.5`, CPU-fine), vector store via **FAISS in-process** (zero infra). No API keys anywhere in this lab.

---

## Before you start (setup)

### 0.1 — Pick your PDF (do this first, it gates everything)

**What:** Choose one **real ~200-page technical PDF that contains tables and headings**, and save it as `data/manual.pdf`.

**Why:** A synthetic or table-free PDF makes the parser comparison pointless — the whole exercise is to see one parser preserve a table that the other flattens. Headings are what makes structure-aware chunking meaningful.

**Good free choices:**
- **Python Language Reference** — export the official docs to PDF (has grammar tables, headed sections). Search "Python Language Reference PDF" on docs.python.org.
- **A hardware datasheet / MCU reference manual** (e.g. an STM32 or ESP32 reference manual) — dense with register tables. These are the gold standard for "tables that break naive parsers."
- **A government report** (e.g. a US GAO or census report) — headings + summary tables.

**Do it:**
```bash
mkdir -p rag-week1/data
# download your chosen PDF into rag-week1/data/manual.pdf (browser download is fine)
```

**Expected result:** `data/manual.pdf` exists and is 150-250 pages.

**Verify:**
```bash
python -c "import fitz; d=fitz.open('rag-week1/data/manual.pdf'); print('pages:', d.page_count)"
# (run after the install step below; expect a number between ~150 and ~250)
```

**Troubleshoot:** If the PDF is a **scanned image** (no selectable text), PyMuPDF will return empty strings — pick a *digital* PDF instead (this lab is not about OCR). Test selectability by trying to highlight text in a PDF viewer.

### 0.2 — Create the environment

**What:** A fresh venv with all dependencies.

**Why:** These libraries pull heavy transitive deps (torch, transformers). An isolated venv keeps your global Python clean and makes the install reproducible.

**Do it:**
```bash
cd rag-week1
python -m venv .venv

# Activate — pick your platform:
source .venv/Scripts/activate     # Windows Git-Bash
# .venv\Scripts\activate          # Windows cmd/PowerShell
# source .venv/bin/activate       # macOS / Linux

python -m pip install -U pip
pip install -U pymupdf pymupdf4llm docling \
  langchain langchain-community langchain-text-splitters \
  llama-index faiss-cpu \
  sentence-transformers rank-bm25 pandas pytest
```

**Expected result:** `pip install` finishes without red errors. First `docling` import later will download a layout model (~few hundred MB) once.

**Verify:**
```bash
python -c "import fitz, pymupdf4llm, docling, langchain_text_splitters, faiss, sentence_transformers, pandas; print('imports OK')"
```
Expect: `imports OK`.

**Troubleshoot:**
- **`faiss` import fails on Windows:** ensure you installed `faiss-cpu` (not `faiss`). If it still fails, `pip install faiss-cpu==1.8.0`.
- **Torch download huge/slow:** it is CPU-only via `sentence-transformers`; be patient on first install. You can pre-warm with `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- **Docling install pulls a lot:** that is expected (it ships a layout model). If it fails on Windows due to a build tool, install the "Visual C++ Build Tools" or use WSL.

### 0.3 — Create the folder layout

**What:** Scaffold the project directories.

**Why:** A fixed layout lets the code and the tests reference stable paths, and separates source, eval data, and saved runs.

**Do it:**
```bash
cd rag-week1
mkdir -p src tests runs
touch src/__init__.py
```
Target layout:
```
rag-week1/
  data/manual.pdf
  golden.jsonl          # your eval set (Step 5)
  src/
    parse.py            # two parsers -> normalized records
    chunk.py            # 3 strategies behind one interface
    index.py            # embed + FAISS build/query
    evaluate.py         # recall@k / MRR grid runner
  tests/test_recall.py
  runs/                 # saved metrics per (parser, chunker, k)
```

**Verify:** `ls src tests runs` shows the three dirs.

---

## Step-by-step

### Step 1 — Two parsers → normalized records (`src/parse.py`)

**What:** Implement `parse_pymupdf(path)` and `parse_docling(path)`. Both return a list of **normalized records** with the exact same schema: `{text, page, heading, source, parser}`.

**Why:** PyMuPDF (via `pymupdf4llm`) is your fast, pure-Python **baseline**; Docling is the layout-aware **challenger** that keeps reading order and turns tables into pipe-tables. A shared schema is what lets the downstream chunker and evaluator treat both parsers identically — you swap the parser by name, nothing else changes. Provenance (`page`, `heading`, `source`) is designed in *now* because you cannot compute recall against a gold snippet, or cite later, without it. (See [../lectures/02-layout-aware-pdf-parsing.md](../lectures/02-layout-aware-pdf-parsing.md).)

**Do it** — write `src/parse.py`:
```python
"""Two PDF parsers emitting a shared normalized record schema.

Record = {"text": str, "page": int|None, "heading": str|None,
          "source": str, "parser": str}
"""
from __future__ import annotations
import pathlib

import pymupdf4llm
from docling.document_converter import DocumentConverter


def parse_pymupdf(path: str) -> list[dict]:
    """Baseline: fast, pure-Python. page_chunks=True -> one dict per page."""
    path = str(path)
    pages = pymupdf4llm.to_markdown(path, page_chunks=True)  # list[dict]
    out = []
    for p in pages:
        # pymupdf4llm returns markdown text + a metadata dict per page.
        page_no = p.get("metadata", {}).get("page")
        out.append({
            "text": p["text"],
            "page": page_no,
            "heading": None,          # PyMuPDF baseline: no heading extraction
            "source": path,
            "parser": "pymupdf",
        })
    return out


# DocumentConverter loads a layout model; build it once, reuse it.
_docling_converter = None

def _get_docling() -> DocumentConverter:
    global _docling_converter
    if _docling_converter is None:
        _docling_converter = DocumentConverter()
    return _docling_converter


def parse_docling(path: str) -> list[dict]:
    """Layout-aware challenger: exports one clean markdown doc.

    Docling keeps reading order and renders tables as pipe-tables. It does
    not hand us per-page splits, so we return ONE record holding the whole
    markdown; chunk.py's structure-aware splitter carves it by heading.
    """
    path = str(path)
    doc = _get_docling().convert(path).document
    md = doc.export_to_markdown()   # headings (#, ##, ###) + pipe-tables preserved
    return [{
        "text": md,
        "page": None,               # page numbers not carried in single-md export
        "heading": None,            # headings live inside the markdown (# ...)
        "source": path,
        "parser": "docling",
    }]


PARSERS = {"pymupdf": parse_pymupdf, "docling": parse_docling}


def parse(path: str, parser: str) -> list[dict]:
    """Dispatch by name so the eval grid can loop over parsers."""
    if parser not in PARSERS:
        raise ValueError(f"unknown parser {parser!r}; choose from {list(PARSERS)}")
    return PARSERS[parser](path)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "data/manual.pdf"
    for name in ("pymupdf", "docling"):
        recs = parse(p, name)
        chars = sum(len(r["text"]) for r in recs)
        print(f"{name:8s} -> {len(recs):4d} records, {chars:,} chars")
```

**Expected result:** Running it prints something like:
```
pymupdf  ->  212 records, 486,003 chars
docling  ->    1 records, 502,118 chars
```
(pymupdf yields one record per page; docling yields one big markdown record — that is by design.)

**Verify:**
```bash
cd rag-week1
python src/parse.py data/manual.pdf
```
Both parsers must produce non-empty text and roughly comparable total character counts.

**Troubleshoot:**
- **`KeyError: 'page'` in pymupdf:** the metadata key can vary by version; the code uses `.get(...)`. If `page` is `None` everywhere, print one `p["metadata"]` to see the actual keys and adjust.
- **Docling is slow (30-90s on 200 pages, CPU):** normal on first run (model download + layout inference). Run it once and cache — see Step 1.5.
- **Docling OOMs on a huge PDF:** convert a page range for development, e.g. split the PDF with `fitz` to `data/manual.pdf` limited to ~200 pages.

### Step 1.5 — Eyeball 2-3 table pages and record the winner

**What:** Dump each parser's output to markdown and manually compare the *same* table on 2-3 pages. Record which parser preserved the pipe-table.

**Why:** This is a **Definition-of-Done item** ("one table page where the two parsers visibly differ") and it is the concrete evidence behind failure points **#4 (not extracted)** and **#7 (incomplete)**: a flattened table loses row/column association, so a tabular question can never retrieve a coherent answer. (See [../lectures/02-layout-aware-pdf-parsing.md](../lectures/02-layout-aware-pdf-parsing.md).)

**Do it:**
```bash
python - <<'PY'
from src.parse import parse_pymupdf, parse_docling
import pathlib
pathlib.Path("runs").mkdir(exist_ok=True)
pathlib.Path("runs/pymupdf_dump.md").write_text(
    "\n\n---PAGE---\n\n".join(r["text"] for r in parse_pymupdf("data/manual.pdf")),
    encoding="utf-8")
pathlib.Path("runs/docling_dump.md").write_text(
    parse_docling("data/manual.pdf")[0]["text"], encoding="utf-8")
print("wrote runs/pymupdf_dump.md and runs/docling_dump.md")
PY
```
Open both files, find a page you know has a table, and look for pipe-table syntax:
```bash
grep -n "|" runs/docling_dump.md | head
grep -n "|" runs/pymupdf_dump.md  | head
```

**Expected result:** Docling shows clean pipe-tables like:
```
| Register | Bits | Reset | Description |
|----------|------|-------|-------------|
| CR1      | 0-3  | 0x0   | control ... |
```
PyMuPDF on the same page often shows the same numbers as **run-together prose** or misaligned columns.

**Verify:** Write one sentence in a note (`runs/PARSER_NOTES.md`) naming the page and which parser won, e.g. *"p.84 register table: Docling preserved the 4-column pipe-table; PyMuPDF flattened it into a single line — expect Docling to win tabular questions."*

**Troubleshoot:** If **neither** shows a pipe-table, your PDF's tables may be images — pick a PDF with real text tables, or note that both parsers fail (still a valid finding, but you lose the "visibly differ" DoD unless you find a different page).

### Step 2 — Three chunkers behind one interface (`src/chunk.py`)

**What:** Implement `fixed`, `recursive`, and `structure` chunkers, all selectable by name through one function `chunk(records, strategy, ...)`, each attaching **provenance** (`source`, `page`/`heading`, `char_span`) to every chunk.

**Why:** Chunk boundaries decide what can *possibly* be retrieved together — this is the single highest-leverage knob for recall@k. Too small fragments context (#7); too big dilutes the embedding and drops precision (#2). Putting all three behind one interface is what makes the eval grid a one-liner loop instead of three copy-pasted pipelines. Provenance travels with every chunk so the evaluator can check the gold substring and later steps can cite. (See [../lectures/03-chunking-strategies.md](../lectures/03-chunking-strategies.md).)

**Do it** — write `src/chunk.py`:
```python
"""Three chunking strategies behind one interface.

A Chunk carries text + provenance so we can evaluate and later cite:
    {"text", "source", "page", "heading", "char_span": (start, end),
     "parser", "strategy"}
"""
from __future__ import annotations

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# NOTE: chunk_size here is CHARACTERS, not tokens. ~800 chars ≈ ~200 tokens,
# comfortably under bge-small's 512-token limit. Do not bump this to 3000+
# or chunks get silently truncated at embed time -> phantom recall loss.
DEFAULT_SIZE = 800
DEFAULT_OVERLAP = 120


def _base(rec: dict, text: str, start: int, end: int, strategy: str,
          heading: str | None = None) -> dict:
    return {
        "text": text,
        "source": rec["source"],
        "page": rec["page"],
        "heading": heading if heading is not None else rec.get("heading"),
        "char_span": (start, end),
        "parser": rec["parser"],
        "strategy": strategy,
    }


def chunk_fixed(rec: dict, size: int = DEFAULT_SIZE) -> list[dict]:
    """Hard cut every `size` chars. Structure-blind baseline."""
    t = rec["text"]
    return [_base(rec, t[i:i + size], i, min(i + size, len(t)), "fixed")
            for i in range(0, len(t), size)]


def chunk_recursive(rec: dict, size: int = DEFAULT_SIZE,
                    overlap: int = DEFAULT_OVERLAP) -> list[dict]:
    """Split on a hierarchy of separators to keep paragraphs/sentences intact."""
    t = rec["text"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out, cursor = [], 0
    for piece in splitter.split_text(t):
        start = t.find(piece, cursor)
        if start == -1:                 # overlap can defeat a forward-only find
            start = t.find(piece)
        end = start + len(piece) if start != -1 else cursor + len(piece)
        out.append(_base(rec, piece, max(start, 0), end, "recursive"))
        cursor = max(end - overlap, 0)
    return out


_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def chunk_structure(rec: dict) -> list[dict]:
    """Split on the document's own heading hierarchy (markdown # / ## / ###).

    Best on Docling output (real headings). On PyMuPDF page-markdown it still
    works but yields fewer splits, since heading detection is weaker.
    """
    t = rec["text"]
    splitter = MarkdownHeaderTextSplitter(_MD_HEADERS, strip_headers=False)
    docs = splitter.split_text(t)
    out, cursor = [], 0
    for d in docs:
        piece = d.page_content
        # Build a heading breadcrumb from whatever header metadata attached.
        heading = " > ".join(str(v) for v in d.metadata.values()) or None
        start = t.find(piece, cursor)
        if start == -1:
            start = t.find(piece)
        end = start + len(piece) if start != -1 else cursor + len(piece)
        out.append(_base(rec, piece, max(start, 0), end, "structure", heading))
        cursor = max(end, 0)
    if not out:  # no headings found -> fall back so the config is never empty
        return chunk_recursive(rec)
    return out


CHUNKERS = {"fixed": chunk_fixed, "recursive": chunk_recursive,
            "structure": chunk_structure}


def chunk(records: list[dict], strategy: str, **kw) -> list[dict]:
    """Apply a named chunker to every parsed record, flatten to one list."""
    if strategy not in CHUNKERS:
        raise ValueError(f"unknown strategy {strategy!r}; choose {list(CHUNKERS)}")
    fn = CHUNKERS[strategy]
    out: list[dict] = []
    for rec in records:
        out.extend(fn(rec, **kw) if strategy != "structure" else fn(rec))
    return [c for c in out if c["text"].strip()]   # drop empties


if __name__ == "__main__":
    from src.parse import parse
    for parser in ("pymupdf", "docling"):
        recs = parse("data/manual.pdf", parser)
        for strat in CHUNKERS:
            cs = chunk(recs, strat)
            avg = sum(len(c["text"]) for c in cs) / max(len(cs), 1)
            print(f"{parser:8s} x {strat:10s} -> {len(cs):5d} chunks, "
                  f"avg {avg:6.0f} chars")
```

**Expected result:**
```
pymupdf  x fixed      ->   612 chunks, avg    795 chars
pymupdf  x recursive  ->   540 chunks, avg    690 chars
pymupdf  x structure  ->   / depends on headings /
docling  x fixed      ->   628 chunks, avg    800 chars
docling  x recursive  ->   551 chunks, avg    701 chars
docling  x structure  ->   240 chunks, avg   1900 chars  <- larger, section-scoped
```
Structure chunks are typically **fewer and larger** (one per section) — that is the point.

**Verify:**
```bash
python src/chunk.py
```
All 6 parser×chunker combos must return >0 chunks, and every chunk has a `char_span` tuple and `strategy` set. Spot-check:
```bash
python -c "from src.parse import parse; from src.chunk import chunk; \
c=chunk(parse('data/manual.pdf','docling'),'structure')[0]; \
print(sorted(c)); print('heading=', c['heading']); print('span=', c['char_span'])"
```

**Troubleshoot:**
- **`structure` returns 1 giant chunk on PyMuPDF:** PyMuPDF page-markdown has few `#` headings. That is expected and *itself a finding* — structure-aware chunking needs a parser that preserves headings. The code falls back to recursive if zero headings are found.
- **`char_span` looks wrong for recursive:** overlap makes exact offsets approximate; that is fine — the span is for citation/debugging, not for exact reconstruction.
- **`chunk_size` warning about tokens:** remember `chunk_size` is **characters**. See the Pitfalls note; keep ~800 to stay under 512 tokens for bge-small.

### Step 3 — Embed + index + query (`src/index.py`)

**What:** Local `bge-small-en-v1.5` embeddings with `normalize_embeddings=True` on **both** index and query sides, a FAISS `IndexFlatIP` build, and a `top-k` query returning `(chunk, score)`.

**Why:** `IndexFlatIP` computes inner product; with **L2-normalized** vectors inner product == cosine similarity. Normalize on only one side and your scores are garbage (a classic silent bug — see Pitfalls). FAISS in-process means zero infra. bge-small is fast enough to embed 200 pages on a CPU in a minute or two. (See [../lectures/04-retrieval-only-evaluation.md](../lectures/04-retrieval-only-evaluation.md) for why retrieval is measured in isolation.)

**Do it** — write `src/index.py`:
```python
"""Local embeddings + FAISS inner-product index (== cosine on normalized vecs)."""
from __future__ import annotations
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_MODEL = None

def _model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")  # CPU-fine
    return _MODEL


def build(chunks: list[dict]):
    """Embed chunk texts and build a FAISS IndexFlatIP. Returns (index, chunks)."""
    texts = [c["text"] for c in chunks]
    emb = _model().encode(
        texts, normalize_embeddings=True,   # <-- MUST match query side
        batch_size=64, show_progress_bar=True,
    ).astype("float32")
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return idx, chunks


def query(idx, chunks: list[dict], q: str, k: int = 5) -> list[tuple[dict, float]]:
    """Return top-k (chunk, score) for query q."""
    qe = _model().encode([q], normalize_embeddings=True).astype("float32")  # SAME flag
    scores, ids = idx.search(qe, k)
    return [(chunks[i], float(scores[0][r]))
            for r, i in enumerate(ids[0]) if i != -1]


if __name__ == "__main__":
    from src.parse import parse
    from src.chunk import chunk
    cs = chunk(parse("data/manual.pdf", "docling"), "recursive")
    idx, cs = build(cs)
    for c, s in query(idx, cs, "What does the reset value default to?", k=3):
        print(f"{s:.3f}  {c['text'][:90]!r}")
```

**Expected result:** A progress bar during encode, then 3 lines of `score  'chunk preview...'` with scores in roughly `[0.4, 0.9]` (cosine). Scores > 1.0 or < -1.0 mean normalization is broken.

**Verify:**
```bash
python src/index.py
```
Scores should be sorted descending and within [-1, 1]. Ask a question you know the answer to and confirm a relevant chunk appears.

**Troubleshoot:**
- **Scores all ~1.0 or nonsensical:** you forgot `normalize_embeddings=True` on one side. Both `encode` calls must have it.
- **First run downloads the model:** ~130 MB, one time. Cached under `~/.cache/huggingface`.
- **Encode is slow:** 200 pages of chunks embed in ~1-3 min on CPU. If it drags, reduce `batch_size` if you hit memory pressure, or use Colab free tier (the spine notes you won't need to).
- **Dimension mismatch on `idx.add`:** all chunks must be embedded by the same model; do not mix models.

### Step 4 — Hand-author the golden set (`golden.jsonl`)

**What:** Write **≥15** realistic `{"q": ..., "gold_substr": ...}` lines — real questions answerable from *your* manual, each with a short answer substring you expect to find inside the retrieved chunk.

**Why:** This is your ground truth; a bad golden set makes every downstream number meaningless. `gold_substr` must be a **realistic answer phrasing**, not a rare exact copy of chunk text (that is leakage — recall looks fake-great). Cover a mix: prose facts, a couple of **tabular** questions (to expose the parser difference), and section-scoped facts. (See [../lectures/04-retrieval-only-evaluation.md](../lectures/04-retrieval-only-evaluation.md) on leakage and triviality.)

**Do it** — create `golden.jsonl` (one JSON object per line). Example shape (**replace with facts from YOUR manual**):
```json
{"q": "What is the reset value of the CR1 control register?", "gold_substr": "0x0000"}
{"q": "How many bits wide is the prescaler field?", "gold_substr": "4 bits"}
{"q": "Which section describes the interrupt vector table?", "gold_substr": "interrupt vector"}
{"q": "What voltage range is the device rated for?", "gold_substr": "1.8 V to 3.6 V"}
{"q": "What is the maximum clock frequency?", "gold_substr": "72 MHz"}
```
Aim for 15-30 lines. Include at least 2-3 questions whose answer lives **in a table**.

**Expected result:** A `golden.jsonl` with ≥15 lines that parses as valid JSONL.

**Verify:**
```bash
python -c "import json; rows=[json.loads(l) for l in open('golden.jsonl',encoding='utf-8') if l.strip()]; \
print('rows:', len(rows)); assert len(rows)>=15; \
assert all('q' in r and 'gold_substr' in r for r in rows); print('schema OK')"
```
Then a leakage sanity check — the `gold_substr` should be short and answer-like, not a 200-char verbatim chunk slice.

**Troubleshoot:**
- **Everything recalls at 1.0 immediately:** your `gold_substr` is probably a rare exact string copied from a chunk → leakage. Rephrase to a natural answer.
- **Everything recalls at 0.0:** your `gold_substr` casing/whitespace differs from the chunk. The evaluator lowercases both sides (below), but watch for unicode (e.g. `μ` vs `u`, non-breaking spaces). Keep substrings short and canonical.
- **JSON errors:** each line must be a complete JSON object; no trailing commas, use double quotes.

### Step 5 — Evaluate retrieval only, over the grid (`src/evaluate.py`)

**What:** Compute `recall@k` and `MRR` for every config in the grid `{pymupdf, docling} × {fixed, recursive, structure} × k∈{3,5,10}`, dump per-config JSON to `runs/`, and print a pandas table **sorted by recall@5**.

**Why:** Retrieval is measured **in isolation** from generation so a miss is unambiguously a retrieval/indexing failure (#1-3), not a generation failure (#4-7). `recall@k` = fraction of questions whose top-k chunks contain the gold substring; `MRR` = mean of `1/rank` of the first hit (rewards ranking the right chunk high). The grid tells you which parser×chunker combo actually wins on *your* corpus — you measure, you don't guess. (See [../lectures/01-...md](../lectures/01-rag-pipelines-and-failure-points.md) for the failure-point mapping and [../lectures/04-...md](../lectures/04-retrieval-only-evaluation.md) for the metrics.)

**Do it** — write `src/evaluate.py`:
```python
"""Retrieval-only eval: recall@k + MRR over the parser x chunker x k grid."""
from __future__ import annotations
import json
import pathlib

import pandas as pd

from src.parse import parse
from src.chunk import chunk, CHUNKERS
from src.index import build, query

PARSERS = ["pymupdf", "docling"]
STRATEGIES = list(CHUNKERS)              # fixed, recursive, structure
KS = [3, 5, 10]
PDF = "data/manual.pdf"
GOLDEN = "golden.jsonl"


def load_golden(path: str = GOLDEN) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _first_hit_rank(hits: list[tuple[dict, float]], gold: str) -> int | None:
    g = gold.lower()
    for rank, (c, _) in enumerate(hits, 1):
        if g in c["text"].lower():
            return rank
    return None


def eval_config(idx, chunks, golden, k: int) -> dict:
    """recall@k and MRR@k for one built index."""
    recall_hits, rr_sum = 0, 0.0
    for row in golden:
        hits = query(idx, chunks, row["q"], k=k)
        r = _first_hit_rank(hits, row["gold_substr"])
        if r is not None:
            recall_hits += 1
            rr_sum += 1.0 / r
    n = len(golden)
    return {"recall": recall_hits / n, "mrr": rr_sum / n}


def run_grid() -> pd.DataFrame:
    golden = load_golden()
    pathlib.Path("runs").mkdir(exist_ok=True)
    rows = []
    for parser in PARSERS:
        records = parse(PDF, parser)
        for strat in STRATEGIES:
            chunks = chunk(records, strat)
            idx, chunks = build(chunks)          # embed once per (parser, chunker)
            for k in KS:
                m = eval_config(idx, chunks, golden, k)
                rec = {"parser": parser, "chunker": strat, "k": k,
                       "n_chunks": len(chunks),
                       "recall": round(m["recall"], 4),
                       "mrr": round(m["mrr"], 4)}
                rows.append(rec)
                out = pathlib.Path("runs") / f"{parser}_{strat}_k{k}.json"
                out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
                print(f"{parser:8s} {strat:10s} k={k:<2d} "
                      f"recall={rec['recall']:.3f} mrr={rec['mrr']:.3f}")
    df = pd.DataFrame(rows)
    # Sort table by recall@5 (per the spine): rank configs by their k=5 recall.
    r5 = (df[df.k == 5].set_index(["parser", "chunker"])["recall"]
          .rename("recall_at_5"))
    df = df.join(r5, on=["parser", "chunker"])
    df = df.sort_values(["recall_at_5", "parser", "chunker", "k"],
                        ascending=[False, True, True, True])
    df.to_csv("runs/results.csv", index=False)
    return df


def best_recall_at_5(df: pd.DataFrame | None = None) -> float:
    if df is None:
        df = pd.read_csv("runs/results.csv")
    return float(df[df.k == 5]["recall"].max())


if __name__ == "__main__":
    df = run_grid()
    print("\n=== Results (sorted by recall@5) ===")
    print(df.to_string(index=False))
    print(f"\nBest recall@5 = {best_recall_at_5(df):.3f}")
```

**Expected result:** 18 rows (2×3×3). Something like:
```
parser   chunker    k  n_chunks  recall    mrr  recall_at_5
docling  structure  3       240   0.733  0.667        0.800
docling  structure  5       240   0.800  0.689        0.800
docling  structure 10       240   0.867  0.699        0.800
docling  recursive  5       551   0.733  0.640        0.733
pymupdf  recursive  5       540   0.667  0.590        0.667
pymupdf  fixed      5       612   0.533  0.470        0.533
...
Best recall@5 = 0.800
```
Typically **docling × structure** or **× recursive** wins, and **fixed** trails — exactly the story the theory predicts.

**Verify:**
```bash
python src/evaluate.py
ls runs/                    # 18 per-config json files + results.csv
cat runs/results.csv | head
```
Confirm ≥6 configs (you'll have 18) with `recall` and `mrr` columns at k∈{3,5,10}.

**Troubleshoot:**
- **All recall = 0:** likely `gold_substr` mismatch (unicode/whitespace) or the parser produced empty text. Print a couple of `query(...)` results and eyeball whether the right chunk is even retrieved.
- **`structure` on pymupdf is much worse than docling:** expected — PyMuPDF didn't preserve headings, so structure ≈ recursive-with-fallback. That is a **finding**, not a bug.
- **Very slow:** you re-embed once per (parser, chunker) = 6 builds. That's inherent. Reduce `KS` during dev, or cache built indexes. bge-small on CPU should keep the full grid under ~10 min for a 200-page doc.
- **Which failure point?** A low-recall config that misses tabular questions is hitting **#4/#7** (table flattened) or **#2 (missed top-k)** if the right chunk exists but ranks low. A config that misses everything a big-chunk config gets is likely **#7** (answer split across chunks). Note these — it's a DoD item.

### Step 6 — Pytest gate (`tests/test_recall.py`)

**What:** A test asserting your **best config's recall@5 ≥ a threshold you commit to** (set it from the observed baseline — e.g. if best is 0.80, gate at 0.70 for headroom).

**Why:** A committed, green gate turns "it felt fine" into a regression guardrail: next week's chunker change can't silently drop recall below your floor. Set the threshold slightly *below* your measured best so run-to-run noise (embedding nondeterminism is minimal, but golden edits happen) doesn't flake the build. (See [../lectures/04-retrieval-only-evaluation.md](../lectures/04-retrieval-only-evaluation.md).)

**Do it** — write `tests/test_recall.py`:
```python
"""Gate: the best (parser x chunker) config must clear our recall@5 floor."""
import pathlib
import pytest

from src.evaluate import run_grid, best_recall_at_5

# YOUR committed threshold. Set from the observed baseline with headroom,
# e.g. observed best recall@5 = 0.80 -> gate at 0.70.
RECALL_AT_5_THRESHOLD = 0.70


@pytest.fixture(scope="module")
def results():
    csv = pathlib.Path("runs/results.csv")
    if csv.exists():
        import pandas as pd
        return pd.read_csv(csv)
    return run_grid()          # first run computes + caches


def test_best_config_recall(results):
    best = best_recall_at_5(results)
    assert best >= RECALL_AT_5_THRESHOLD, (
        f"best recall@5 = {best:.3f} < threshold {RECALL_AT_5_THRESHOLD}; "
        f"retrieval regressed — check parser/chunker/normalization"
    )
```

**Expected result:**
```
tests/test_recall.py::test_best_config_recall PASSED
```

**Verify:**
```bash
cd rag-week1
python -m pytest tests/test_recall.py -v
```

**Troubleshoot:**
- **Test fails (best < threshold):** either your threshold is too aggressive (lower it to just under observed best) or your best config is genuinely weak — improve chunking (try `structure` on docling), fix table preservation, or check normalization. Do **not** inflate by leaking gold strings.
- **`ModuleNotFoundError: src`:** run pytest from the `rag-week1` root, or add an empty `conftest.py` at the root so `src` is importable. Ensure `src/__init__.py` exists.
- **Slow test:** it runs the full grid on first invocation. After `python src/evaluate.py` has written `runs/results.csv`, the fixture reuses it and the test is instant.

---

## Putting it together — short end-to-end run

From a clean shell, this is the full offline pipeline, parse → chunk → index → evaluate → gate:

```bash
cd rag-week1
source .venv/Scripts/activate          # or .venv/bin/activate on macOS/Linux

# 1. sanity: both parsers run, tables eyeballed (Step 1 + 1.5)
python src/parse.py data/manual.pdf

# 2. sanity: 6 parser x chunker combos all produce chunks (Step 2)
python src/chunk.py

# 3. sanity: embed + query returns sensible cosine scores (Step 3)
python src/index.py

# 4. golden set validates (Step 4)
python -c "import json; n=sum(1 for l in open('golden.jsonl',encoding='utf-8') if l.strip()); print('golden rows:', n); assert n>=15"

# 5. run the full grid -> runs/ + sorted table (Step 5)
python src/evaluate.py

# 6. gate is green (Step 6)
python -m pytest tests/test_recall.py -v
```

Expected end state: a printed table sorted by recall@5, 18 JSON files + `results.csv` in `runs/`, and a passing test. You can now say, out loud, which parser×chunker wins and which of the 7 failure points each low-recall config is hitting.

---

## Definition of Done — verifiable checks

Restating the spine's acceptance gate, each as something you can point at:

- [ ] **`data/manual.pdf` is ~200 pages, parsed by BOTH parsers, and one table page visibly differs.**
      Verify: `python src/parse.py data/manual.pdf` shows both parsers with non-empty output; `runs/PARSER_NOTES.md` names the page and the winner (Step 1.5).
- [ ] **3 chunking strategies behind one interface, selectable by name; every chunk carries provenance (`source`, `page`/`heading`, `char_span`).**
      Verify: `python src/chunk.py` lists all 6 parser×chunker combos with >0 chunks; a spot-checked chunk has `char_span` + `strategy` + `source` (Step 2).
- [ ] **`golden.jsonl` has ≥15 hand-written question→gold-substr pairs (no leakage).**
      Verify: the schema/count check in Step 4 prints `rows: >=15` and `schema OK`.
- [ ] **A results table in `runs/` covers ≥6 configs (2 parsers × 3 chunkers) at k∈{3,5,10}, with recall@k AND MRR per config.**
      Verify: `runs/results.csv` has 18 rows with `recall` and `mrr` columns; per-config JSONs exist in `runs/` (Step 5).
- [ ] **`pytest` is green: best config meets your committed recall@5 threshold (state it, e.g. ≥0.70).**
      Verify: `python -m pytest tests/test_recall.py -v` → PASSED; `RECALL_AT_5_THRESHOLD` is stated in the test (Step 6).
- [ ] **You can name which of the 7 failure points each low-recall config is hitting.**
      Verify: one line per weak config in `runs/PARSER_NOTES.md`, e.g. *"pymupdf×fixed misses tabular Qs → #4 (table flattened, not extracted) + #7 (rows split across chunks)."*

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Cosine scores > 1 or garbage ranking | `normalize_embeddings=True` on only one side | Set it on BOTH `encode` calls in `index.py` (Pitfall: normalization mismatch) |
| Every question recalls at 1.0 instantly | `gold_substr` is a rare verbatim chunk slice = leakage | Rewrite `gold_substr` as a natural short answer phrasing |
| Every question recalls at 0.0 | casing/unicode/whitespace mismatch, or empty parse | Evaluator lowercases both sides; watch `μ`/`u`, non-breaking spaces; verify parser output is non-empty |
| `structure` yields 1 giant chunk on PyMuPDF | PyMuPDF page-markdown lacks `#` headings | Expected — a finding. Structure-aware needs Docling's headings; code falls back to recursive |
| `faiss` won't import (Windows) | installed `faiss` not `faiss-cpu` | `pip install faiss-cpu` |
| Docling first run very slow / downloads model | layout model download + CPU inference | One-time; cache the markdown dump (Step 1.5) and reuse |
| Oversized chunks, recall mysteriously drops | `chunk_size` is CHARS; >~2000 chars exceeds bge-small's 512-token limit → silent truncation | Keep chunks ~800 chars (~200 tokens) |
| Flattened tables tank tabular-question recall | table stripped to prose loses row/col links | Keep pipe-tables in the chunk; prefer Docling for tabular pages (#4/#7) |
| `ModuleNotFoundError: src` under pytest | running from wrong dir / no `__init__.py` | Run from `rag-week1` root; ensure `src/__init__.py` exists |
| Grid is slow | 6 embed builds is inherent | Reduce `KS` in dev; reuse `runs/results.csv`; bge-small CPU should finish <~10 min |

---

## Stretch goals (optional)

- **4th chunker — sentence-window or parent-document.** Add a `sentence_window` strategy via LlamaIndex `SentenceWindowNodeParser` (embed single sentences, return the sentence ± a neighbor window) or a parent-document retriever (embed small, return the larger parent). Great precision/recall tradeoff for QA. Add it to `CHUNKERS` and it drops straight into the grid. (See [../lectures/03-chunking-strategies.md](../lectures/03-chunking-strategies.md).)
- **BM25 hybrid.** Add a `rank-bm25` sparse retriever and fuse with the dense results (RRF) to see recall lift on exact-match questions (SKUs, register names, error codes). This previews Week 2. (See [../lectures/05-hybrid-search-dense-sparse-rrf.md](../lectures/05-hybrid-search-dense-sparse-rrf.md).)
- **Cross-encoder rerank.** Retrieve top-20, rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2`, keep top-5, and re-measure recall@5. Cheap escalation, big potential lift. (Previews [../lectures/06-cross-encoder-reranking.md](../lectures/06-cross-encoder-reranking.md).)
- **LlamaParse spot-check.** Use the LlamaParse free tier on 3-5 of your ugliest table pages only, to see how far your local parsers fall short of a "gold" hosted parser — a reference point, not your default.
- **Server-backed vector store.** Swap FAISS for `docker run -p 6333:6333 qdrant/qdrant` or Docker `pgvector/pgvector` to rehearse the infra you'll use in Week 2. Not required this week.
- **Answer the self-check questions** from the spine (which failure points are reindex-only vs query-time fixable; why recall jumps fixed→structure; when sentence-window beats recursive) — writing the answers cements the theory.
