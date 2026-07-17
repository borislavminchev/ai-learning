# Week 2 Lab: Extraction, Cleaning, Three-Tier Dedup, and PII Redaction

> You extend last week's `corpus-pipeline/` with the stage that turns landed *raw* documents into *validated, PII-redacted, deduped* records. You will parse messy PDFs/HTML into text + tables **with provenance** (page/bbox), route scanned pages to OCR with confidence flags, normalize/clean text with a per-stage drop report, deduplicate three ways (exact → MinHash-LSH → semantic), and redact PII with Microsoft Presidio so the raw strings are truly gone. This is the middle third of the Phase 5 milestone: garbage in, garbage out — this stage is where the corpus stops being garbage.
>
> **Read the lectures first** (they are the *why*; this lab is the *how*):
> [06 — Document Parsing with Provenance](../lectures/06-document-parsing-provenance.md) ·
> [07 — OCR Reality & Confidence Routing](../lectures/07-ocr-confidence-routing.md) ·
> [08 — Text Normalization & Cleaning](../lectures/08-text-normalization-cleaning.md) ·
> [09 — Deduplication at Three Tiers](../lectures/09-deduplication-three-tiers.md) ·
> [10 — PII Detection & True Redaction with Presidio](../lectures/10-pii-detection-redaction.md).
>
> Spine reference: [Phase 5 · Week 2](../05-data-engineering.md).

**Est. time:** ~9 hrs · **You will need:** Python 3.11+, `uv`, the `corpus-pipeline/` repo from Week 1. Everything here runs **free and local**: Docling (parsing), PaddleOCR or Tesseract (OCR on CPU), `datasketch` (MinHash), `sentence-transformers` with `all-MiniLM-L6-v2` (semantic dedup on CPU), and a **local Presidio** with the spaCy `en_core_web_lg` model (PII). No paid API is required at any step; where a cloud service (GPT-4o vision, AWS Textract) would help on messy layouts, it is called out as *optional*.

---

## Before you start (setup)

You should already be inside the `corpus-pipeline/` repo from Week 1 (it has `pyproject.toml`, `src/corpus/`, and a populated `landing/` zone). If not, do Week 1 first — this stage reads from `landing/`.

```bash
cd corpus-pipeline

# Extraction / OCR / cleaning / dedup / PII deps
uv add docling datasketch presidio-analyzer presidio-anonymizer ftfy \
  langdetect sentence-transformers xxhash

# Presidio's default NER backend is spaCy. Download the large English model.
uv run python -m spacy download en_core_web_lg
```

OCR engine — pick one (PaddleOCR is the lab default; Tesseract is the lighter fallback):

```bash
# Option A (default): PaddleOCR — pip installable, CPU works
uv add paddleocr paddlepaddle

# Option B (fallback, lighter): Tesseract via pytesseract
uv add pytesseract pdf2image pillow
```

**Windows / Git-Bash notes**
- **Tesseract binary:** `pytesseract` needs the Tesseract *executable*. Install it (winget: `winget install UB-Manchaiku... ` is not it — use `winget install tesseract-ocr.tesseract` or grab the UB-Mannheim installer), then either add it to `PATH` or set `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"` in code.
- **`pdf2image`** needs Poppler on Windows: install via `winget install oschwartz10612.Poppler` or download Poppler binaries and add its `bin/` to `PATH`; pass `poppler_path=...` to `convert_from_path` if not on `PATH`.
- **PaddlePaddle on Windows** installs CPU wheels fine under Python 3.11; the first run downloads model files to `~/.paddleocr` (a few hundred MB) — that download needs network once.
- In Git-Bash, prefer forward slashes in Python string paths; use `uv run python ...` so the venv is picked up. Long first-run model downloads are normal — don't Ctrl-C them.

**Get some test documents.** Create `data/samples/` and drop in a handful:
- 2–3 **digital PDFs** with a real table (a public product manual, a datasheet, an arXiv paper).
- 1 **scanned/photographed PDF** (print a page and scan/photo it, or search for a "sample scanned document PDF"). This exercises the OCR path.
- 1–2 **HTML** files (save a docs page as `.html`).
- Deliberately duplicate one PDF and lightly edit another copy (change a sentence) so you have exact + near dupes to catch.

```bash
mkdir -p data/samples data/clean reports tests/fixtures
```

**Folder additions for this week** (create the files as you go):

```
src/corpus/
  extract.py      # Docling parse -> {doc_id, page, bbox, text, tables_md, is_scanned}
  ocr_route.py    # low-quality page -> PaddleOCR/Tesseract, capture confidence
  clean.py        # ftfy + boilerplate strip + langdetect filter
  dedup.py        # exact -> MinHash-LSH -> optional semantic
  pii.py          # Presidio analyze + anonymize (consistent pseudonyms)
  drop_report.py  # counts dropped per stage with reasons
```

A shared record shape keeps every stage honest. Put this near the top of `extract.py` (or in a small `models.py`) and import it everywhere:

```python
from pydantic import BaseModel

class Block(BaseModel):
    doc_id: str
    page: int
    bbox: tuple[float, float, float, float] | None  # (x0, y0, x1, y1)
    text: str
    tables_md: list[str] = []       # tables preserved as markdown, NOT flattened
    source_path: str
    is_scanned: bool = False
    needs_review: bool = False      # set by OCR routing on low confidence
    ocr_confidence: float | None = None
```

---

## Step-by-step

### Step 1 — Parse documents with provenance (Docling)

**What.** Turn each file in `data/samples/` into a list of `Block` records, one per logical block, carrying `page`, `bbox`, and any tables as markdown.

**Why.** Phase 4 (RAG) cites answers back to a page/region. If provenance dies here, citations are impossible later and tables become unrecoverable walls of numbers. Docling detects reading order, digital-vs-scanned, and exports tables as structured markdown. (See [lecture 06](../lectures/06-document-parsing-provenance.md).)

**Do it.** `src/corpus/extract.py`:

```python
import hashlib
from pathlib import Path
from docling.document_converter import DocumentConverter

def _doc_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

def parse_document(path: Path) -> list[Block]:
    """Parse one file into Block records with page/bbox provenance."""
    conv = DocumentConverter()
    result = conv.convert(str(path))
    doc = result.document
    doc_id = _doc_id(path)
    blocks: list[Block] = []

    # Tables -> markdown (do NOT flatten to prose)
    tables_md = [t.export_to_markdown() for t in doc.tables]

    # Text items carry provenance (page + bounding box) in Docling's model.
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", "") or ""
        if not text.strip():
            continue
        prov = getattr(item, "prov", None)
        page = prov[0].page_no if prov else 0
        bbox = None
        if prov and getattr(prov[0], "bbox", None) is not None:
            b = prov[0].bbox
            bbox = (b.l, b.t, b.r, b.b)
        blocks.append(Block(
            doc_id=doc_id, page=page, bbox=bbox, text=text,
            tables_md=[], source_path=str(path),
            is_scanned=_looks_scanned(doc),
        ))
    if blocks and tables_md:
        blocks[0].tables_md = tables_md   # attach tables to the doc's first block
    return blocks

def _looks_scanned(doc) -> bool:
    """Heuristic: no/low embedded text layer => scanned. Refined in Step 2."""
    total = sum(len(getattr(i, "text", "") or "") for i, _ in doc.iterate_items())
    n_pages = max(1, getattr(doc, "num_pages", lambda: 1)())
    return (total / n_pages) < 50   # < ~50 chars/page => probably image-only
```

> Docling's exact attribute names have shifted across versions. If `doc.iterate_items()` or `.prov` differ in your installed version, run `uv run python -c "from docling.document_converter import DocumentConverter as C; d=C().convert('data/samples/YOURFILE.pdf').document; print(type(d)); print(d.export_to_markdown()[:500])"` and inspect the model. The markdown export always works and is a good fallback for text; the structured iteration is what gives you per-block bbox.

**Expected result.** For a digital PDF you get many blocks with non-null `bbox` and page numbers, and `tables_md` contains pipe-delimited markdown tables. For the scanned PDF you get few/empty text blocks and `is_scanned=True`.

**Verify.**
```bash
uv run python -c "from src.corpus.extract import parse_document; from pathlib import Path; \
bs=parse_document(Path('data/samples/YOURFILE.pdf')); \
print(len(bs),'blocks'); print(bs[0].page, bs[0].bbox); print(bs[0].tables_md[:1])"
```
Confirm at least one table came out as markdown (has `|` separators) and blocks have page/bbox.

**Troubleshoot.**
- *Tables flattened to prose:* you're reading `export_to_text`. Use `export_to_markdown()` on the table object.
- *Everything `is_scanned=True` on a clearly digital PDF:* your char/page threshold is too high, or Docling ran its own OCR and stripped the text layer — lower the threshold and check `doc.export_to_markdown()` has real text.
- *Very slow first run:* Docling downloads layout models once. Let it finish.

---

### Step 2 — Route scanned/low-quality pages to OCR with confidence capture

**What.** For pages flagged `is_scanned` (or with near-zero embedded text density), render the page to an image, OCR it, capture **per-line confidence**, and tag any block whose mean confidence `< 0.80` as `needs_review=True`.

**Why.** OCR quality is dominated by preprocessing, not engine choice; you cannot trust OCR blindly. Confidence routing = auto-accept the good, flag the bad for a human instead of silently poisoning the corpus. (See [lecture 07](../lectures/07-ocr-confidence-routing.md).)

**Do it.** `src/corpus/ocr_route.py` (PaddleOCR default):

```python
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path   # renders PDF pages -> PIL images

CONFIDENCE_THRESHOLD = 0.80

def _preprocess(pil_img):
    """Deskew/binarize/upscale-lite: grayscale + simple threshold. Preprocessing
    dominates OCR quality far more than engine choice."""
    import cv2
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    if max(img.shape) < 1500:                       # upscale small scans
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img

def ocr_page(pdf_path: Path, page_no: int) -> tuple[str, float]:
    """OCR one page. Returns (text, mean_line_confidence in [0,1])."""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    images = convert_from_path(str(pdf_path), first_page=page_no, last_page=page_no)
    img = _preprocess(images[0])
    result = ocr.ocr(img, cls=True)
    lines, confs = [], []
    for line in (result[0] or []):
        text, conf = line[1][0], float(line[1][1])
        lines.append(text); confs.append(conf)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return "\n".join(lines), mean_conf

def route_blocks(blocks: list["Block"], pdf_path: Path) -> list["Block"]:
    for b in blocks:
        if not b.is_scanned:
            continue
        text, conf = ocr_page(pdf_path, b.page)
        b.text = text or b.text
        b.ocr_confidence = conf
        b.needs_review = conf < CONFIDENCE_THRESHOLD
    return blocks
```

**Tesseract fallback** (Option B) — same signature, swap `ocr_page`:

```python
def ocr_page_tesseract(pdf_path, page_no):
    import pytesseract, pandas as pd
    from pdf2image import convert_from_path
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows
    img = _preprocess(convert_from_path(str(pdf_path), first_page=page_no, last_page=page_no)[0])
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
    data = data[data.conf != -1]                       # drop layout rows
    text = " ".join(data.text.dropna().astype(str))
    mean_conf = float(data.conf.mean()) / 100.0 if len(data) else 0.0  # Tesseract conf is 0-100
    return text, mean_conf
```

> **Optional cloud comparison (paid, not required):** send the rendered page PNG to GPT-4o-mini vision or AWS Textract and diff the text against local OCR. Useful to *see* where local OCR fails on messy layouts — but the lab's Definition of Done is fully met with the free local path.

**Expected result.** The scanned PDF now has real text on its blocks, each with an `ocr_confidence`, and low-confidence blocks carry `needs_review=True`. Print the count of flagged blocks.

**Verify.**
```bash
uv run python -c "from src.corpus.extract import parse_document; from src.corpus.ocr_route import route_blocks; from pathlib import Path; \
p=Path('data/samples/SCANNED.pdf'); bs=route_blocks(parse_document(p), p); \
print('needs_review:', sum(b.needs_review for b in bs), '/', len(bs)); \
print('mean conf:', [round(b.ocr_confidence,2) for b in bs if b.ocr_confidence])"
```

**Troubleshoot.**
- *PaddleOCR import/DLL errors on Windows:* ensure `paddlepaddle` (CPU) installed cleanly; try a fresh `uv sync`. If stuck, use the Tesseract fallback.
- *`pdf2image` "Unable to get page count" / poppler not found:* install Poppler and pass `poppler_path=`.
- *All confidences ~0.0:* the page rendered blank — check DPI (`convert_from_path(..., dpi=300)`), and confirm preprocessing didn't invert to all-black.

---

### Step 3 — Clean & normalize with a per-block drop reason

**What.** Fix mojibake with `ftfy`, strip boilerplate (repeated headers/footers, nav junk), and drop non-English blocks below a `langdetect` confidence threshold — logging every drop with a reason.

**Why.** Encoding garbage and boilerplate inflate dedup and pollute embeddings; language filtering keeps an English corpus English. The drop report is your audit trail (see [lecture 08](../lectures/08-text-normalization-cleaning.md)).

**Do it.** `src/corpus/clean.py`:

```python
import ftfy, re
from collections import Counter
from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0   # deterministic langdetect

def normalize(text: str) -> str:
    text = ftfy.fix_text(text)                       # repair mojibake / bad unicode
    text = re.sub(r"[ \t]+", " ", text)              # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def find_boilerplate(blocks: list["Block"], min_repeat: int = 3) -> set[str]:
    """Lines that repeat across many pages/blocks are headers/footers/nav."""
    counts = Counter(l.strip() for b in blocks for l in b.text.splitlines() if l.strip())
    return {line for line, n in counts.items() if n >= min_repeat and len(line) < 120}

def clean_blocks(blocks: list["Block"], lang_threshold: float = 0.90):
    """Returns (kept_blocks, drops) where drops = [(doc_id, page, reason)]."""
    boiler = find_boilerplate(blocks)
    kept, drops = [], []
    for b in blocks:
        b.text = normalize(b.text)
        b.text = "\n".join(l for l in b.text.splitlines() if l.strip() not in boiler)
        if not b.text.strip():
            drops.append((b.doc_id, b.page, "empty_after_clean")); continue
        try:
            top = detect_langs(b.text)[0]             # e.g. en:0.99
            if top.lang != "en" or top.prob < lang_threshold:
                drops.append((b.doc_id, b.page, f"non_english({top.lang}:{top.prob:.2f})"))
                continue
        except Exception:
            drops.append((b.doc_id, b.page, "lang_detect_failed")); continue
        kept.append(b)
    return kept, drops
```

**Expected result.** Text with `â€™`-style mojibake becomes clean apostrophes; repeated page headers vanish; obviously non-English blocks are dropped and recorded.

**Verify.** Feed a block containing `donâ€™t` and confirm output is `don't`. Print `len(drops)` and eyeball the reasons.

**Troubleshoot.**
- *`langdetect` throws `LangDetectException` on short strings:* short blocks (a heading) are unreliable — the `try/except` catches it; consider skipping the lang filter for blocks under ~20 chars rather than dropping them.
- *Real content stripped as boilerplate:* raise `min_repeat` or add a length guard (already `< 120` chars) so long paragraphs are never treated as headers.

---

### Step 4 — Deduplicate three ways (cascade: exact → MinHash-LSH → semantic)

**What.** Remove exact byte-dupes, then near-dupes (boilerplate/minor edits) with MinHash-LSH, then optionally paraphrase-dupes with embeddings — always cheapest filter first.

**Why.** Exact hash is free; MinHash-LSH catches near-dupes at scale cheaply; semantic dedup catches paraphrases but is expensive — running it first on millions of docs is a cost bomb. Cascade so the expensive stage sees the fewest candidates (see [lecture 09](../lectures/09-deduplication-three-tiers.md)).

**Do it.** `src/corpus/dedup.py`:

```python
import re, xxhash
from datasketch import MinHash, MinHashLSH

def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def exact_dedup(blocks):
    seen, kept, drops = set(), [], []
    for b in blocks:
        h = xxhash.xxh64(_norm_key(b.text)).hexdigest()
        if h in seen:
            drops.append((b.doc_id, b.page, "exact_dup")); continue
        seen.add(h); kept.append(b)
    return kept, drops

def _shingles(text: str, k: int = 5) -> set[str]:
    toks = _norm_key(text).split()
    return {" ".join(toks[i:i+k]) for i in range(max(1, len(toks) - k + 1))}

def minhash_dedup(blocks, threshold: float = 0.85, num_perm: int = 128):
    """Near-dup: keep the LONGEST member of each near-dup cluster."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mh_by_idx = {}
    for i, b in enumerate(blocks):
        m = MinHash(num_perm=num_perm)
        for sh in _shingles(b.text):
            m.update(sh.encode())
        mh_by_idx[i] = m
        lsh.insert(str(i), m)
    dropped, kept, drops = set(), [], []
    for i, b in enumerate(blocks):
        if i in dropped:
            continue
        dupes = [int(x) for x in lsh.query(mh_by_idx[i])]
        cluster = [j for j in dupes if j not in dropped]
        keeper = max(cluster, key=lambda j: len(blocks[j].text))   # longest wins
        for j in cluster:
            if j != keeper:
                dropped.add(j)
                drops.append((blocks[j].doc_id, blocks[j].page, "near_dup_minhash"))
    kept = [b for i, b in enumerate(blocks) if i not in dropped]
    return kept, drops

def semantic_dedup(blocks, threshold: float = 0.95):
    """OPTIONAL: catch paraphrases exact+MinHash miss. Report ADDITIONAL dupes."""
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")   # CPU-friendly, local
    emb = model.encode([b.text for b in blocks], normalize_embeddings=True,
                       show_progress_bar=False)
    sim = util.cos_sim(emb, emb)
    dropped, drops = set(), []
    for i in range(len(blocks)):
        if i in dropped:
            continue
        for j in range(i + 1, len(blocks)):
            if j not in dropped and float(sim[i][j]) > threshold:
                dropped.add(j)
                drops.append((blocks[j].doc_id, blocks[j].page, "semantic_dup"))
    kept = [b for i, b in enumerate(blocks) if i not in dropped]
    return kept, drops
```

**Expected result.** Your duplicated PDF's blocks collapse to one (exact); the lightly-edited copy collapses via MinHash (exact misses it). Semantic dedup reports a small number of *additional* paraphrase dupes over MinHash.

**Verify.**
```bash
uv run python -c "
from src.corpus.dedup import exact_dedup, minhash_dedup, semantic_dedup
# blocks = your cleaned blocks
b1, d1 = exact_dedup(blocks)
b2, d2 = minhash_dedup(b1)
b3, d3 = semantic_dedup(b2)
print('exact dropped', len(d1), '| minhash dropped', len(d2), '| semantic ADDITIONAL', len(d3))
"
```
The DoD wants you to show MinHash catches near-dupes that exact-hash misses — i.e. `len(d2) > 0` after exact already ran.

**Troubleshoot.**
- *MinHash catches nothing:* your blocks are too short for 5-shingles — lower `k` to 3, or dedup at document level instead of block level.
- *Semantic dedup too slow:* it's O(n²) here — fine for a lab corpus (hundreds of blocks). For real scale use `util.semantic_search` / FAISS and only run it on MinHash survivors.
- Write a one-paragraph **cost vs. benefit** note in your README: how many extra dupes semantic caught for how much extra compute.

---

### Step 5 — Redact PII with Presidio (consistent pseudonyms, true removal)

**What.** Detect PII (EMAIL, PHONE_NUMBER, PERSON, CREDIT_CARD, US_SSN, …) with Presidio's `AnalyzerEngine`, then replace with a **deterministic hash-based pseudonym** so the same entity always maps to the same token (`john@x.com` → `<EMAIL_7f3a>` every time).

**Why.** Masking a display layer leaves the raw string in the bytes — a compliance failure. True redaction removes the content; consistent pseudonyms preserve referential structure without leaking identity (see [lecture 10](../lectures/10-pii-detection-redaction.md)).

**Do it.** `src/corpus/pii.py`:

```python
import hashlib
from functools import lru_cache
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

@lru_cache(maxsize=1)
def _engines():
    return AnalyzerEngine(), AnonymizerEngine()   # spaCy en_core_web_lg backs NER

def _pseudonym(kind: str, value: str) -> str:
    tag = hashlib.sha256(value.encode()).hexdigest()[:4]
    return f"<{kind}_{tag}>"                        # deterministic: same value -> same token

DEFAULT_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON",
                    "CREDIT_CARD", "US_SSN", "IP_ADDRESS", "LOCATION"]

def redact(text: str, entities=None) -> str:
    analyzer, anonymizer = _engines()
    results = analyzer.analyze(text=text, language="en",
                               entities=entities or DEFAULT_ENTITIES)
    # One operator that pseudonymizes using the ORIGINAL matched substring.
    operators = {"DEFAULT": OperatorConfig("custom", {
        "lambda": lambda x: _pseudonym("PII", x)
    })}
    # For per-type tags, use a callback keyed on entity_type instead (see note).
    return anonymizer.anonymize(text=text, analyzer_results=results,
                                operators=operators).text
```

To get **per-type** tags (`<EMAIL_...>`, `<PERSON_...>`) while staying consistent, register one operator per entity type:

```python
def redact_typed(text: str, entities=None) -> str:
    analyzer, anonymizer = _engines()
    ents = entities or DEFAULT_ENTITIES
    results = analyzer.analyze(text=text, language="en", entities=ents)
    ops = {e: OperatorConfig("custom",
             {"lambda": (lambda kind: (lambda x: _pseudonym(kind, x)))(e)})
           for e in ents}
    return anonymizer.anonymize(text=text, analyzer_results=results, operators=ops).text
```

**Custom recognizer** (the spine's pitfall: defaults miss internal formats like `EMP-000123`):

```python
from presidio_analyzer import Pattern, PatternRecognizer

employee_id = PatternRecognizer(
    supported_entity="EMPLOYEE_ID",
    patterns=[Pattern("emp_id", r"\bEMP-\d{6}\b", 0.9)],
)
# analyzer.registry.add_recognizer(employee_id)  # then add "EMPLOYEE_ID" to entities
```

**Seed a PII fixture** for the test. `tests/fixtures/pii_seed.txt` — 15 known strings:

```
Contact John Smith at john.smith@example.com or (415) 555-0132.
SSN 123-45-6789, card 4111 1111 1111 1111. Server 10.0.0.42. Employee EMP-000123.
Jane Doe lives in Seattle; reach her at jane_doe99@corp.co, +1 206 555 0199.
```

**Test** (`tests/test_pii.py`):

```python
from src.corpus.pii import redact_typed

KNOWN_PII = ["john.smith@example.com", "(415) 555-0132", "123-45-6789",
             "4111 1111 1111 1111", "John Smith", "jane_doe99@corp.co", "Seattle", ...]

def test_no_raw_pii_survives():
    raw = open("tests/fixtures/pii_seed.txt").read()
    red = redact_typed(raw)
    for s in KNOWN_PII:
        assert s not in red, f"PII leaked: {s}"

def test_consistent_pseudonyms():
    red = redact_typed("Email a@b.com then a@b.com again")
    tokens = [t for t in red.split() if t.startswith("<EMAIL")]
    assert len(set(tokens)) == 1        # same input entity -> same token
```

**Expected result.** All 15 seeded strings are absent from output; identical entities map to identical tokens across the document.

**Verify.**
```bash
uv run pytest tests/test_pii.py -q
```

**Troubleshoot.**
- *`OSError: [E050] Can't find model 'en_core_web_lg'`:* re-run `uv run python -m spacy download en_core_web_lg`.
- *Presidio misses a PERSON / phone:* NER is probabilistic. Lower the score threshold in `analyze(..., score_threshold=0.3)`, add context words, or add a custom recognizer — then add that string to your test.
- *Credit-card / SSN not caught with spaces:* Presidio's card recognizer handles common formats; if your format is odd, add a `PatternRecognizer`.
- Remember: this redacts **text**. For true redaction of the *PDF bytes* (Week-relevant pitfall), you'd remove the content stream (e.g. `pikepdf`/`pymupdf` redaction annotations applied) — for this lab we redact the extracted text that flows downstream, which is what the corpus and index consume.

---

### Step 6 — Drop report + validated JSONL + wire into Dagster

**What.** Emit `data/clean/*.jsonl` (one record per surviving block, redacted) plus `reports/drop_report.json` that accounts for **every** input row: `rows_in == rows_out + sum(drops)`. Then expose the whole stage as Dagster assets downstream of Week 1's `landed_manifest`.

**Why.** The balancing invariant is your proof that nothing vanished silently. Wiring into Dagster makes the stage reproducible and gate-able (see [lecture 08](../lectures/08-text-normalization-cleaning.md) on drop reports; Dagster from Week 1's [lecture 04](../lectures/04-dagster-orchestration-quality-gates.md)).

**Do it.** `src/corpus/drop_report.py`:

```python
import json
from collections import Counter
from pathlib import Path

def write_drop_report(rows_in: int, rows_out: int, all_drops: list[tuple],
                      path="reports/drop_report.json"):
    by_reason = Counter(reason for _doc, _pg, reason in all_drops)
    # normalize reason (strip the "(en:0.87)" detail) so the counter is readable
    by_stage = Counter(r.split("(")[0] for _d, _p, r in all_drops)
    report = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "dropped_total": len(all_drops),
        "by_reason": dict(by_reason),
        "by_stage": dict(by_stage),
        "balanced": rows_in == rows_out + len(all_drops),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2))
    assert report["balanced"], f"UNBALANCED: {rows_in} != {rows_out} + {len(all_drops)}"
    return report
```

A driver that runs the whole stage (put in `extract.py` as `run_stage()` or a small `pipeline.py`):

```python
def run_stage(sample_dir="data/samples", out="data/clean/blocks.jsonl"):
    from pathlib import Path
    import json
    from .ocr_route import route_blocks
    from .clean import clean_blocks
    from .dedup import exact_dedup, minhash_dedup, semantic_dedup
    from .pii import redact_typed
    from .drop_report import write_drop_report

    all_blocks, drops = [], []
    for f in Path(sample_dir).glob("*"):
        bs = parse_document(f)
        if f.suffix.lower() == ".pdf":
            bs = route_blocks(bs, f)
        all_blocks += bs
    rows_in = len(all_blocks)

    kept, d = clean_blocks(all_blocks);        drops += d
    kept, d = exact_dedup(kept);               drops += d
    kept, d = minhash_dedup(kept);             drops += d
    kept, d = semantic_dedup(kept);            drops += d   # optional; keep if you want the count

    for b in kept:
        b.text = redact_typed(b.text)          # PII removal before persisting

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for b in kept:
            fh.write(json.dumps(b.model_dump()) + "\n")

    return write_drop_report(rows_in, len(kept), drops)
```

**Dagster wiring** — add to `src/corpus/defs.py` (alongside Week 1's assets):

```python
from dagster import asset, asset_check, AssetCheckResult

@asset(deps=["landed_manifest"])            # downstream of Week 1's landing zone
def clean_corpus() -> dict:
    from .extract import run_stage
    return run_stage()                       # returns the drop_report dict

@asset_check(asset=clean_corpus, blocking=True)
def drop_report_balances(clean_corpus) -> AssetCheckResult:
    return AssetCheckResult(passed=bool(clean_corpus["balanced"]),
                            metadata={k: v for k, v in clean_corpus.items()})
```

**Expected result.** `data/clean/blocks.jsonl` exists (redacted, deduped, provenance-carrying), and `reports/drop_report.json` has `"balanced": true`.

**Verify.**
```bash
uv run python -c "from src.corpus.extract import run_stage; import json; print(json.dumps(run_stage(), indent=2))"
uv run dagster dev            # then materialize clean_corpus in the UI; the check must pass
```

**Troubleshoot.**
- *`balanced: false`:* a stage dropped a row without appending to `drops` (or double-counted). Every `continue`/skip must push a `(doc_id, page, reason)`.
- *Dagster can't import assets:* ensure `defs.py` exposes `Definitions(assets=[...], asset_checks=[...])` and `pyproject.toml`/`workspace.yaml` points at it (from Week 1).

---

## Putting it together — a short end-to-end run

```bash
# 1. Ensure samples exist (2-3 digital PDFs incl. a table, 1 scanned PDF, 1-2 HTML,
#    plus one exact-duplicate file and one lightly-edited near-duplicate).
ls data/samples

# 2. Run the whole extraction/cleaning/dedup/PII stage.
uv run python -c "from src.corpus.extract import run_stage; import json; print(json.dumps(run_stage(), indent=2))"

# 3. Inspect outputs.
uv run python -c "import json; rows=[json.loads(l) for l in open('data/clean/blocks.jsonl')]; \
print('records:', len(rows)); \
print('needs_review:', sum(r['needs_review'] for r in rows)); \
print('with tables:', sum(1 for r in rows if r['tables_md']))"
cat reports/drop_report.json

# 4. Prove PII removal + run the suite.
uv run pytest -q

# 5. See it as a governed asset in Dagster.
uv run dagster dev   # materialize clean_corpus; the blocking drop_report_balances check goes green
```

You now have raw landed docs transformed into a clean, redacted, deduped JSONL corpus with full provenance and an auditable drop report — the input Week 3 will version, validate, and index.

---

## Definition of Done — verifiable checks

Restating the spine's Week 2 checklist as things you can point at:

- [ ] **Provenance + tables.** `data/clean/blocks.jsonl` records each carry `{doc_id, page, bbox, source_path}`, and at least one record's `tables_md` holds a markdown table (has `|` cell separators), not flattened prose.
- [ ] **OCR confidence routing.** The scanned PDF's pages were OCR'd; blocks with mean confidence `< 0.80` are flagged `needs_review=true`. Print the flagged count (`sum(r['needs_review'])`).
- [ ] **Dedup shows both tiers.** After exact dedup runs, MinHash still drops ≥1 near-dup that exact-hash missed. Report both counts (exact dropped N, MinHash-additional M); include the semantic-dedup additional count + a one-line cost/benefit note.
- [ ] **PII: 15/15.** `uv run pytest tests/test_pii.py` passes — every seeded raw PII string is absent from the redacted output, and the same input entity maps to the same pseudonym every time.
- [ ] **Drop report balances.** `reports/drop_report.json` has `"balanced": true` and `rows_in == rows_out + dropped_total`.
- [ ] **Green tests.** `uv run pytest` passes.
- [ ] **Wired in Dagster.** `clean_corpus` materializes downstream of Week 1's `landed_manifest`, and the blocking `drop_report_balances` check is green.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Docling: `.prov`/`iterate_items` attribute errors | version drift in Docling model | inspect with `print(type(doc))`; fall back to `export_to_markdown()` for text |
| Tables come out as prose | used text export | call `table.export_to_markdown()` |
| Everything flagged `is_scanned` | char/page threshold too high, or OCR stripped text layer | lower threshold; verify `export_to_markdown()` has real text |
| `pdf2image` "Unable to get page count" | Poppler missing (Windows) | install Poppler, pass `poppler_path=`; on Linux `apt install poppler-utils` |
| PaddleOCR DLL / import crash on Windows | broken `paddlepaddle` wheel | `uv sync`; else switch to Tesseract fallback |
| Tesseract "not installed or not in PATH" | binary missing | install Tesseract; set `pytesseract.pytesseract.tesseract_cmd` |
| OCR confidences all ~0 | blank render / over-thresholded image | `dpi=300`; check preprocessing didn't invert to black |
| `langdetect` non-deterministic | seed not set | `DetectorFactory.seed = 0` |
| MinHash catches nothing | blocks too short for k=5 shingles | lower `k`, or dedup at doc level |
| Semantic dedup very slow | O(n²) pairwise | run only on MinHash survivors; use FAISS/`semantic_search` at scale |
| spaCy `E050` model not found | `en_core_web_lg` not downloaded | `uv run python -m spacy download en_core_web_lg` |
| Presidio misses a PII entity | probabilistic NER / custom format | lower `score_threshold`, add context words, add `PatternRecognizer` — then add to test |
| drop report `balanced: false` | a skip didn't record a drop | every `continue` must append `(doc_id, page, reason)` |

---

## Stretch goals (optional)

- **Cloud OCR A/B (paid, optional):** OCR the scanned page with GPT-4o-mini vision or AWS Textract free tier; diff against local PaddleOCR and quantify the accuracy gap on messy layouts. Keep the local path as default.
- **True PDF-byte redaction:** use `pymupdf`/`pikepdf` to apply redaction annotations that remove the underlying content stream, then re-extract and assert the PII string is gone from the *bytes*, not just the text field.
- **Per-region OCR bbox:** capture PaddleOCR's per-line polygons and store them as block `bbox` so OCR'd content keeps the same provenance contract as digital text.
- **Language-aware corpus:** instead of dropping non-English, route to a per-language cleaning path and record language as a field.
- **Dedup metrics dashboard:** emit a small table (exact/near/semantic drops, cluster sizes) and render it; compare thresholds (0.80 vs 0.90 Jaccard) and note the precision/recall tradeoff.
- **FineWeb-style scale note:** read the HuggingFace FineWeb/`datatrove` writeup and add a README paragraph on how your three-tier cascade would change at billions of documents (sharded MinHash, bloom filters, no full-corpus semantic pass).
