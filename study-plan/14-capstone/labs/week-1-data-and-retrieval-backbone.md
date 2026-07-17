# Week 1 Lab: The Data Pipeline + Retrieval Backbone — Prove Deletes, Recall, and Tenant Isolation

> You build the production spine of the capstone: messy scanned PDFs go in one end; a **multi-tenant, access-controlled, cited, groundedness-verified** retrieval API comes out the other — with the dataset **versioned by a hash** and every index mutation **provable**. No answer generation yet (that's Week 2); this week is the backbone you can't fake later. By Friday you can *demonstrate* three claims, not assert them: **(1)** deleting a source doc removes it from every answer with no reindex, **(2)** recall@5 on a golden set ≥ your committed target, **(3)** tenant A can never surface tenant B's data. Read the two capstone lectures that ground this week first — [01 — Capstone Integration Map](../lectures/01-capstone-integration-map.md) (how the four weeks fit together and why the retrieval spine comes first) and [02 — Ingestion, OCR & Chunk Provenance Decisions](../lectures/02-ingestion-ocr-and-chunk-provenance-decisions.md) (scan detection, OCR routing, bbox provenance, contextual chunking). If you want to read ahead on the two proofs that dominate the back half of the week, skim [03 — Tenant Isolation & ACL as a Server-Side Boundary](../lectures/03-tenant-isolation-and-acl-as-a-server-side-boundary.md) and [04 — Mutation, Versioning & CAG-vs-Retrieval](../lectures/04-mutation-versioning-and-cag-vs-retrieval.md).

**Est. time:** ~10–12 hrs (this is a capstone week — heavier than a normal lab; budget two or three sittings) · **You will need:** Python 3.11+, Docker Desktop, ~6 GB free disk for models + Qdrant storage, Git, and a handful of real-ish PDFs (at least one **scanned**). **Free/local path (no paid API, no GPU):** Qdrant in Docker (or `pgvector` in Docker), Tesseract (or PaddleOCR) for OCR, Microsoft Presidio + spaCy for PII, `BAAI/bge-small-en-v1.5` embeddings + `bge-reranker-v2-m3` on CPU, DVC with a **local** remote, and **Ollama** (`llama3.1:8b`) for contextual-chunk prefixes. Everything below runs on a laptop.

---

## Before you start (setup)

### The capstone monorepo layout (set this up once — Weeks 2–4 build on it)

The capstone spine's milestone project expects one workspace-style monorepo where `apps/` consume `packages/`. Create the skeleton now so nothing has to move later; Week 1 fills in `packages/retrieval`, `packages/tenancy`, `apps/api`, `apps/worker`, `evals/`, and `infra/compose`. Empty dirs get a `.gitkeep`.

```
enterprise-kap/
  README.md                     # domain choice, quickstart, links to all docs
  Makefile                      # ingest, index, eval, up, test targets
  docker-compose.yml            # Qdrant (+ later: pg, gateway, api, worker, phoenix)
  .env.example                  # QDRANT_URL, OLLAMA_URL, EMBED_MODEL, tenant seeds
  pyproject.toml                # uv/poetry workspace; or a plain requirements.txt
  apps/
    api/                        # FastAPI: /search, DELETE /documents/{id}, /health  (Week 1)
    worker/                     # async ingestion jobs                                (grows Week 2)
    web/                        # thin console                                        (Week 3)
    mcp-server/                 # domain tools over MCP                                (Week 2)
  packages/
    ingestion/                  # detect-scan -> OCR -> layout -> chunks w/ page+bbox  (Week 1)
    retrieval/                  # hybrid dense+BM25, RRF, reranker, ACL filter         (Week 1)
    tenancy/                    # per-tenant config, ACL payload schema, isolation     (Week 1)
    guardrails/                 # PII redaction (Presidio), quality gate               (Week 1)
    generation/                 # prompts, citation enforcement                        (Week 2)
    agents/                     # supervisor, tools, MCP/A2A clients                   (Week 2)
    observability/              # OTel spans, cost meter                               (Week 4)
    llm/                        # gateway client, routing, cache                       (Week 3)
  infra/
    compose/                    # base + prod compose overrides
    k8s/  or  fly/              # deploy manifests (Week 4)
  evals/
    golden/                     # golden.jsonl per tenant (versioned)                  (Week 1)
    retrieval_eval.py           # recall@5 harness with a target floor                 (Week 1)
    thresholds.yaml             # metric floors enforced in CI
    report/                     # scorecards over time
  docs/
    DECISIONS.md                # the CAG-vs-retrieval decision record                 (Week 1)
    diagrams/                   # architecture, dataflow, threat-model
  dvc.yaml                      # ingest -> redact -> quality_gate -> index stages     (Week 1)
  tests/
    test_delete.py  test_recall.py  test_tenant_isolation.py                           (Week 1)
```

> Naming note: the capstone spine's Week-1 sketch used a flat `src/` layout. This lab uses the milestone **monorepo** layout (`packages/…`) so you never refactor. If you'd rather match the spine 1:1 first and split into packages later, that's fine — keep the module *names* (`ingest`, `redact`, `quality_gate`, `index`, `migrate`, `retrieve`, `verify`) so the DVC stages line up.

### Bootstrap the environment

```bash
# from wherever you keep projects
git init enterprise-kap && cd enterprise-kap
python -m venv .venv
source .venv/bin/activate            # Windows Git-Bash: source .venv/Scripts/activate
                                     # Windows PowerShell:  .venv\Scripts\Activate.ps1

pip install -U \
  docling pymupdf pytesseract pillow \
  presidio-analyzer presidio-anonymizer \
  qdrant-client rank-bm25 sentence-transformers \
  dvc pandas pydantic pytest fastapi "uvicorn[standard]" httpx
python -m spacy download en_core_web_lg     # Presidio's NER backend (~600 MB)
```

**Tesseract binary** (pytesseract is only a wrapper — you need the engine):
- Windows: `choco install tesseract` (admin shell) — or download the UB-Mannheim installer and add its folder to `PATH`. Verify with `tesseract --version`.
- macOS: `brew install tesseract` · Linux: `sudo apt install tesseract-ocr`
- If `pytesseract` still can't find it on Windows, set the path explicitly in code: `pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`.

**Vector store — Qdrant in Docker** (gives you *server-side payload filters* = the ACL/tenant boundary, and *delete-by-filter* = tombstones — both of which FAISS cannot do cleanly):

```bash
docker run -d --name capstone-qdrant -p 6333:6333 -p 6334:6334 \
  -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant
# smoke test:
curl http://localhost:6333/collections     # -> {"result":{"collections":[]}, ...}
```

> Windows Git-Bash gotcha: `$PWD` volume mounts sometimes need a leading slash (`//$PWD` or a Windows path like `-v C:/Users/.../qdrant_storage:/qdrant/storage`). If the container exits immediately, run `docker logs capstone-qdrant` and check the mount.

**Free alternative — pgvector instead of Qdrant:** `docker run -d --name capstone-pg -p 5432:5432 -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg16`. You then enforce ACLs with `WHERE tenant_id = %s AND acl_group = ANY(%s)` and tombstone with `DELETE FROM chunks WHERE doc_id = %s`. The *concepts* map 1:1; this lab uses Qdrant as the default because filter-and-delete-by-payload is first-class.

**Contextual-chunk LLM (free):** install [Ollama](https://ollama.com), then `ollama pull llama3.1:8b`. Confirm `curl http://localhost:11434/api/tags` lists it. If you skip contextual prefixes for the first pass, the pipeline still works — just note it in `DECISIONS.md`.

Create `.env.example` (and copy to `.env`):

```
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
EMBED_MODEL=BAAI/bge-small-en-v1.5
RERANK_MODEL=BAAI/bge-reranker-v2-m3
NLI_MODEL=cross-encoder/nli-deberta-v3-base
RECALL_TARGET=0.75
```

Drop your PDFs into `data/raw/` — aim for **6–15 docs across 2 tenants**, mixing native-text PDFs, at least one **scanned** doc (print a page and re-scan/photograph it, or grab a scanned public filing), and one with a **table**. You'll DVC-track this folder in Step 4.

---

## Step-by-step

### Step 1 — Ingest with OCR routing + page/bbox provenance

**What.** Turn each PDF into a list of `Chunk` records that each carry `text`, `doc_id`, `tenant_id`, `page`, and a non-degenerate `bbox`. Detect scanned pages and OCR *only those*.

**Why.** Enterprise answers must show *where* — the UI highlights the source region from `{page, bbox}`. Retrofitting boxes after chunking is miserable, so the schema carries them from the first line. And OCR errors propagate *permanently* into embeddings ("rn"→"m", dropped table decimals), so you must route OCR deliberately and measure it (Step 3), not trust it. See lecture 02 for the scan-detection heuristic and why coordinates are non-negotiable.

**Do it.** Create `packages/ingestion/schema.py`:

```python
from pydantic import BaseModel

class Chunk(BaseModel):
    doc_id: str
    tenant_id: str
    text: str
    page: int
    bbox: tuple[float, float, float, float]      # [x0, y0, x1, y1]
    heading: str | None = None
    acl_groups: list[str] = []
    ocr: bool = False
    ctx: str | None = None                        # contextual-chunk prefix (Step 1b)
```

Then `packages/ingestion/ingest.py` — the transparent PyMuPDF + Tesseract baseline (know what's underneath):

```python
import fitz, pytesseract                          # PyMuPDF (pip name: pymupdf)
from PIL import Image
from .schema import Chunk

def page_needs_ocr(page) -> bool:
    """Empty text layer but has images => it's a scan. Never OCR a clean text page."""
    return len(page.get_text().strip()) < 20 and bool(page.get_images())

def ocr_page_blocks(page, dpi=200):
    """Return (text, bbox) blocks from a scanned page, grouped by y-band."""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    scale = 72.0 / dpi                             # pixels -> PDF points
    lines: dict[tuple, list] = {}
    for i, w in enumerate(d["text"]):
        if not w.strip():
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        x0, y0 = d["left"][i], d["top"][i]
        x1, y1 = x0 + d["width"][i], y0 + d["height"][i]
        lines.setdefault(key, []).append((w, x0, y0, x1, y1))
    blocks = []
    for words in lines.values():
        text = " ".join(w[0] for w in words)
        x0 = min(w[1] for w in words) * scale; y0 = min(w[2] for w in words) * scale
        x1 = max(w[3] for w in words) * scale; y1 = max(w[4] for w in words) * scale
        blocks.append((text, (x0, y0, x1, y1)))
    return blocks

def ingest(path: str, doc_id: str, tenant_id: str, acl: list[str]) -> list[Chunk]:
    doc = fitz.open(path)
    chunks: list[Chunk] = []
    for pno, page in enumerate(doc):
        if page_needs_ocr(page):
            for text, bbox in ocr_page_blocks(page):
                if text.strip():
                    chunks.append(Chunk(doc_id=doc_id, tenant_id=tenant_id, text=text,
                                        page=pno, bbox=bbox, acl_groups=acl, ocr=True))
        else:
            for b in page.get_text("dict")["blocks"]:
                if "lines" not in b:               # skip image-only blocks
                    continue
                txt = " ".join(s["text"] for l in b["lines"] for s in l["spans"]).strip()
                if txt:
                    chunks.append(Chunk(doc_id=doc_id, tenant_id=tenant_id, text=txt,
                                        page=pno, bbox=tuple(b["bbox"]), acl_groups=acl))
    return chunks
```

> **Higher-quality default: Docling.** For real corpora prefer IBM's **Docling** — it does layout analysis, table-structure recovery, and OCR routing for you, exporting Markdown/JSON *with coordinates*. Sketch: `from docling.document_converter import DocumentConverter; doc = DocumentConverter().convert(path)`, then walk `doc.document` items which carry `prov` (page + bbox). Use Docling as the default path and keep the PyMuPDF+Tesseract version above as the "know what's underneath" fallback. (Reserve hosted LlamaParse / Azure Document Intelligence only to spot-check your ugliest ~5 pages — they're paid clouds, not a laptop default.)

**Step 1b — contextual chunk prefix (Anthropic Contextual Retrieval).** Prepend a 1–2 sentence LLM blurb situating each chunk before embedding. Add `packages/ingestion/context.py`:

```python
import os, httpx

_PROMPT = ("<document>{doc}</document>\nHere is a chunk from that document:\n"
           "<chunk>{chunk}</chunk>\nIn one short sentence, situate this chunk within "
           "the document (section, topic, what it refers to). Answer with the sentence only.")

def contextualize(doc_summary: str, chunk_text: str) -> str:
    r = httpx.post(f"{os.environ['OLLAMA_URL']}/api/generate", json={
        "model": "llama3.1:8b", "stream": False,
        "prompt": _PROMPT.format(doc=doc_summary[:6000], chunk=chunk_text[:1500])})
    return r.json()["response"].strip()
```

Then set `c.ctx = contextualize(doc_summary, c.text)` per chunk and embed `ctx + "\n" + text` in Step 5. Short legal/technical chunks ("the penalty is 2x") are ambiguous out of context; the prefix disambiguates the embedding.

**Expected result.** `ingest("data/raw/scanned.pdf", "D1", "tenantA", ["legal"])` returns a non-empty `list[Chunk]`, with at least some `ocr=True` for the scanned doc, and every chunk carrying `page` and a `bbox` whose `x1>x0` and `y1>y0`.

**Verify.**

```bash
python -c "from packages.ingestion.ingest import ingest; \
cs=ingest('data/raw/<your-scanned>.pdf','D1','tenantA',['legal']); \
print(len(cs),'chunks;', sum(c.ocr for c in cs),'OCR;', \
all(c.bbox[2]>c.bbox[0] and c.bbox[3]>c.bbox[1] for c in cs),'bboxes ok')"
```

**Troubleshoot.** *No OCR chunks?* Your "scanned" PDF actually has a text layer — confirm with `page.get_text()`; use a genuinely image-only page. *`TesseractNotFoundError`?* Set `tesseract_cmd` (see setup). *Garbled OCR text?* Bump `dpi=300`, or switch to PaddleOCR/Surya for rotated/dense pages. *Degenerate bbox (all zeros)?* You're reading pixel coords without the `scale`; keep the `72/dpi` conversion.

---

### Step 2 — Clean, dedup, and PII-redact with Presidio

**What.** A `redact(text) -> (clean_text, pii_count)` using Presidio, plus a content-hash `dedup(chunks)`.

**Why.** Enterprise corpora leak SSNs, emails, and account numbers into an index any tenant user can surface via retrieval — a compliance incident. Redact **before** embedding, or PII lives in your vectors forever. NER recall isn't 100%, so back it with deterministic regex recognizers for structured PII (Presidio ships Luhn-checked `CREDIT_CARD`, `IBAN_CODE`, `US_SSN`). If authorized un-redaction is ever needed, keep the reversible map in a **separate secured store**, never in the searchable payload.

**Do it.** `packages/guardrails/redact.py`:

```python
import hashlib
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

_analyzer = AnalyzerEngine()          # spaCy NER + regex recognizers
_anon = AnonymizerEngine()
_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
             "US_SSN", "CREDIT_CARD", "IBAN_CODE"]

def redact(text: str) -> tuple[str, int]:
    res = _analyzer.analyze(text=text, language="en", entities=_ENTITIES)
    clean = _anon.anonymize(text=text, analyzer_results=res).text   # -> "<PERSON>", "<US_SSN>", ...
    return clean, len(res)

def dedup(chunks):
    seen, keep = set(), []
    for c in chunks:
        h = hashlib.sha256(c.text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h); keep.append(c)
    return keep
```

Wire it into a corpus-level pass that mutates each chunk's `text` and records the total PII count for the quality gate.

**Expected result.** `redact("Email jane@acme.com, SSN 123-45-6789")` returns something like `("Email <EMAIL_ADDRESS>, SSN <US_SSN>", 2)`.

**Verify.**

```bash
python -c "from packages.guardrails.redact import redact; \
print(redact('Contact John Smith at john@acme.com or 555-123-4567, SSN 123-45-6789'))"
```

**Troubleshoot.** *`OSError: [E050] Can't find model 'en_core_web_lg'`* — rerun `python -m spacy download en_core_web_lg` inside the venv. *Missed an email/SSN?* Confirm it's in `_ENTITIES`; for a custom structured ID, add a `PatternRecognizer` with your regex. *First run is slow* — spaCy loads the model once per process; that's normal.

---

### Step 3 — Quality gate that halts the pipeline

**What.** `gate(chunks)` that **raises** on bad data: too many empty chunks, too much OCR garbage, or any PII that slipped past redaction.

**Why.** Don't ship garbage into the index. OCR turns tables to mush and digits to letters; without a junk-ratio check you embed garbage and lose recall you'll wrongly blame on the reranker. This becomes the last DVC stage, so a failing gate fails `dvc repro` (and CI).

**Do it.** `packages/guardrails/quality_gate.py`:

```python
import re
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

def _alnum_ratio(t: str) -> float:
    return sum(ch.isalnum() or ch.isspace() for ch in t) / max(len(t), 1)

def gate(chunks) -> dict:
    n = len(chunks)
    empty = sum(1 for c in chunks if len(c.text.strip()) < 10) / max(n, 1)
    bad_ocr = [c for c in chunks if c.ocr and _alnum_ratio(c.text) < 0.85]
    # PII leak = a redacted marker is absent yet a raw pattern is present
    pii_leaks = sum(1 for c in chunks
                    if "<EMAIL_ADDRESS>" not in c.text and _EMAIL.search(c.text))
    metrics = {"empty_frac": empty, "bad_ocr_frac": len(bad_ocr) / max(n, 1),
               "pii_leaks": pii_leaks, "n": n}
    assert empty < 0.05, f"too many empty chunks: {empty:.2%}"
    assert metrics["bad_ocr_frac"] < 0.10, f"OCR garbage too high: {metrics['bad_ocr_frac']:.2%}"
    assert pii_leaks == 0, f"PII leaked past redaction: {pii_leaks}"
    return metrics
```

**Expected result.** On clean redacted chunks, `gate()` returns a metrics dict. On a deliberately corrupted chunk (mostly punctuation, or an un-redacted email), it raises `AssertionError`.

**Verify (prove it bites).**

```bash
python -c "
from packages.ingestion.schema import Chunk
from packages.guardrails.quality_gate import gate
good=[Chunk(doc_id='D',tenant_id='t',text='a valid clause about termination penalties',page=0,bbox=(1,1,2,2))]
print('clean:', gate(good))
bad=good+[Chunk(doc_id='D',tenant_id='t',text='@#%^&*!!! reach me at leak@evil.com',page=1,bbox=(1,1,2,2),ocr=True)]
try:
    gate(bad); print('NO RAISE — bug')
except AssertionError as e:
    print('correctly halted:', e)"
```

**Troubleshoot.** *Gate never fires* — your thresholds are too loose or the corrupt chunk isn't `ocr=True`. *Gate fires on good data* — real scanned text can dip below 0.85 alnum on tables; tune to your corpus and record the number. Keep the gate strict enough to matter but calibrated to your PDFs.

---

### Step 4 — Version the dataset with DVC (local remote)

**What.** Content-address `data/raw/` with DVC and define a `dvc.yaml` pipeline (`ingest → redact → quality_gate → index`) so any index state ↔ a git commit hash.

**Why.** "Which corpus version produced this answer?" must be answerable by a hash. Reindexing is expensive and destructive, so you version the inputs *and* the pipeline. (lakeFS is the alternative when many people mutate a shared corpus and want atomic branch merges; DVC is the pragmatic laptop default.)

**Do it.**

```bash
dvc init
dvc remote add -d localremote "$PWD/.dvcstore"     # a LOCAL remote — no cloud needed
dvc add data/raw                                    # content-addresses the PDFs -> data/raw.dvc
git add data/raw.dvc data/.gitignore .dvc/config && git commit -m "corpus v1"
dvc push                                            # pushes blobs to .dvcstore
```

Then author `dvc.yaml` with stages that call small runner scripts (each reads inputs, writes an artifact):

```yaml
stages:
  ingest:
    cmd: python -m packages.ingestion.run data/raw build/chunks.jsonl
    deps: [data/raw, packages/ingestion]
    outs: [build/chunks.jsonl]
  redact:
    cmd: python -m packages.guardrails.run_redact build/chunks.jsonl build/clean.jsonl
    deps: [build/chunks.jsonl, packages/guardrails/redact.py]
    outs: [build/clean.jsonl]
  quality_gate:
    cmd: python -m packages.guardrails.run_gate build/clean.jsonl
    deps: [build/clean.jsonl, packages/guardrails/quality_gate.py]
  index:
    cmd: python -m packages.retrieval.run_index build/clean.jsonl
    deps: [build/clean.jsonl, packages/retrieval/index.py]
```

```bash
dvc repro          # runs only what changed; a failing quality_gate fails the whole run
```

**Expected result.** `data/raw.dvc` holds an `md5`; `dvc repro` runs the four stages in order; corrupting an input then re-running fails at `quality_gate`.

**Verify.** `git log --oneline` shows your corpus commit; `dvc status` reports "up to date"; `dvc repro` is deterministic on a clean tree.

**Troubleshoot.** *`dvc push` errors on the local remote path* on Windows — use a forward-slash absolute path or a second local folder. *`dvc repro` reruns everything every time* — a `deps`/`outs` path is wrong or an out isn't actually written. *Huge PDFs* — that's exactly what DVC's cache is for; don't commit them to git directly.

---

### Step 5 — Index with `tenant_id` + `acl_groups` + `doc_id`

**What.** Embed chunks locally and upsert to Qdrant, putting `tenant_id`, `acl_groups`, and `doc_id` into the **payload** (these are the fields you'll filter and delete on).

**Why.** The tenant/ACL boundary and the tombstone delete both live on payload fields the vector store can filter server-side. `doc_id` on every point is what makes a whole-document delete possible without a rebuild (Step 6).

**Do it.** `packages/retrieval/index.py`:

```python
import os, uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (PointStruct, VectorParams, Distance,
                                   PayloadSchemaType)
from sentence_transformers import SentenceTransformer

qc = QdrantClient(url=os.environ["QDRANT_URL"])
_model = SentenceTransformer(os.environ["EMBED_MODEL"])   # bge-small-en-v1.5 -> 384 dims

def _embed(texts): return _model.encode(texts, normalize_embeddings=True)

def ensure_collection(coll: str, dim: int = 384):
    if not qc.collection_exists(coll):
        qc.create_collection(coll, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        # index payload keys we filter on -> fast, correct server-side filtering
        for key in ("tenant_id", "doc_id", "acl_groups"):
            qc.create_payload_index(coll, field_name=key, field_schema=PayloadSchemaType.KEYWORD)

def upsert(coll: str, chunks):
    ensure_collection(coll)
    texts = [(c.ctx or "") + "\n" + c.text for c in chunks]
    vecs = _embed(texts)
    points = [PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload=c.model_dump())
              for v, c in zip(vecs, chunks)]
    qc.upsert(coll, points=points, wait=True)
    return len(points)
```

Use a **single collection** named e.g. `kb` and isolate tenants via the `tenant_id` payload filter (simpler and the pattern this lab tests). Per-tenant *collections* are a valid alternative for hard isolation — note the choice in `DECISIONS.md`.

**Expected result.** `upsert("kb", chunks)` returns the point count; `qc.count("kb")` matches; Qdrant dashboard at `http://localhost:6333/dashboard` shows the collection with payload indexes.

**Verify.**

```bash
python -c "from qdrant_client import QdrantClient; import os; \
qc=QdrantClient(url=os.environ['QDRANT_URL']); print(qc.count('kb'))"
```

**Troubleshoot.** *Dimension mismatch on upsert* — the collection was created for a different model's dim; delete and recreate, or use blue-green (Step 6). *Filtering is slow / returns wrong results* — you skipped `create_payload_index`; add it. *`recreate_collection` deprecation warnings* — prefer `collection_exists` + `create_collection` as above.

---

### Step 6 — Tombstone delete + blue-green re-embedding migration

**What.** (a) `delete_doc(coll, doc_id)` = server-side delete-by-filter on `doc_id`. (b) `blue_green(...)` = build a `green` collection with a new embedding model, validate recall, then **atomically flip an alias** the query side always uses.

**Why.** A doc is *many* chunks/points — deleting by point-id leaves orphans that keep answering and fail your delete proof. Delete by a **filter on `doc_id`** so every derived chunk vanishes immediately with no rebuild. And you can't re-embed in place: a new model changes dims/semantics, so old and new vectors are incompatible; an in-place upsert leaves a half-migrated index returning nonsense mid-migration. Build green alongside blue, validate, then flip the `live` alias atomically; rollback = flip back. See lecture 04.

**Do it.** `packages/retrieval/migrate.py`:

```python
from qdrant_client.models import (Filter, FieldCondition, MatchValue,
                                   FilterSelector, CreateAliasOperation, CreateAlias,
                                   DeleteAliasOperation, DeleteAlias)
from .index import qc, ensure_collection, upsert

def delete_doc(coll: str, doc_id: str):
    """TOMBSTONE: remove every point derived from doc_id, server-side, no rebuild."""
    qc.delete(coll, points_selector=FilterSelector(filter=Filter(must=[
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])), wait=True)

def set_live(coll: str):
    qc.update_collection_aliases(change_aliases_operations=[
        DeleteAliasOperation(delete_alias=DeleteAlias(alias_name="live")),
        CreateAliasOperation(create_alias=CreateAlias(collection_name=coll, alias_name="live")),
    ])

def blue_green(new_model_name: str, chunks, validate) -> bool:
    import os; os.environ["EMBED_MODEL"] = new_model_name   # or pass model in explicitly
    ensure_collection("green")
    upsert("green", chunks)                                 # backfill with the NEW model
    if not validate("green"):                               # e.g. recall@5 on the golden set
        return False                                        # ABORT — do not flip
    set_live("green")                                       # atomic flip: blue -> green
    return True
```

Create the `live` alias pointing at your first collection once: `set_live("kb")`. **All queries hit the `live` alias, never a collection name** — that's what makes the flip atomic and rollback trivial (just `set_live("kb")` again).

**Expected result.** After `delete_doc("live", "D7")`, `qc.count("live", count_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value="D7"))]))` is `0`. After a successful `blue_green(...)`, `live` resolves to `green`.

**Verify.** The delete proof is Step 9's `test_delete.py`. For blue-green, print which collection `live` points to before/after: `qc.get_collection_aliases("green")` / inspect via the dashboard.

**Troubleshoot.** *Deleted doc still returned* — you deleted by point-id, not by `doc_id` filter, or you queried a collection name instead of `live`. *Alias flip errors* if `live` doesn't exist yet — create it once before the first flip, or wrap the delete-alias op in a try. *Green recall worse than blue* — that's the point of validating **before** flipping; keep blue live and investigate.

---

### Step 7 — Hybrid (dense + BM25) + RRF + rerank, with server-side ACL

**What.** `search(query, tenant_id, roles, k=5)` that: dense-searches Qdrant with the tenant+ACL filter **in the query**, runs BM25 over the *same* filtered candidate set, fuses with Reciprocal Rank Fusion, reranks with a cross-encoder, and returns hits with `page`+`bbox`+`doc_id` citations.

**Why.** Dense misses exact IDs/codes; BM25 misses paraphrase — fuse them. The cardinal multi-tenant sin is fetch-top-k-then-filter-in-Python: you've already leaked (top-k starvation, one missed code path). Isolation must be enforced **in the vector-store query** at the lowest layer and tested adversarially (lecture 03).

**Do it.** `packages/retrieval/retrieve.py`:

```python
import os
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from .index import qc, _embed

_reranker = CrossEncoder(os.environ["RERANK_MODEL"])   # bge-reranker-v2-m3, CPU-fine for small k

def _acl_filter(tenant_id: str, roles: list[str]) -> Filter:
    must = [FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
    if roles:                                          # ACL: chunk visible if it shares a group
        must.append(FieldCondition(key="acl_groups", match=MatchAny(any=roles)))
    return Filter(must=must)                           # enforced SERVER-SIDE, always

def _rrf(rank_lists, k_const=60):
    scores = {}
    for hits in rank_lists:
        for rank, h in enumerate(hits):
            scores[h] = scores.get(h, 0.0) + 1.0 / (k_const + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def search(query: str, tenant_id: str, roles: list[str], k: int = 5, pool: int = 50):
    flt = _acl_filter(tenant_id, roles)
    qv = _embed([query])[0].tolist()
    dense = qc.query_points("live", query=qv, query_filter=flt, limit=pool, with_payload=True).points
    # BM25 over the SAME acl-filtered candidate pool (never a broader set):
    corpus = [(p.id, p.payload) for p in dense]
    toks = [p["text"].lower().split() for _, p in corpus]
    bm25 = BM25Okapi(toks) if toks else None
    bm_order = ([corpus[i][0] for i in
                 sorted(range(len(corpus)), key=lambda i: bm25.get_scores(query.lower().split())[i],
                        reverse=True)] if bm25 else [])
    by_id = {p.id: p for p in dense}
    dense_order = [p.id for p in dense]
    fused_ids = _rrf([dense_order, bm_order])[:pool]
    # cross-encoder rerank the fused pool down to top-k
    pairs = [(query, by_id[i].payload["text"]) for i in fused_ids]
    scored = sorted(zip(fused_ids, _reranker.predict(pairs)), key=lambda x: -x[1])[:k]
    out = []
    for i, s in scored:
        p = by_id[i].payload
        out.append({"text": p["text"], "page": p["page"], "bbox": p["bbox"],
                    "doc_id": p["doc_id"], "tenant_id": p["tenant_id"], "score": float(s)})
    return out
```

The `query_filter=flt` on the **server** is the isolation boundary. Never drop it; never post-filter in Python. Both dense and BM25 operate on the already-filtered pool, so BM25 can't reintroduce another tenant's text.

**Expected result.** `search("termination penalty", "tenantA", ["legal"])` returns ≤5 hits, all `tenant_id == "tenantA"`, each with `page`, `bbox`, `doc_id`, and a rerank `score`.

**Verify.**

```bash
python -c "from packages.retrieval.retrieve import search; \
hits=search('termination penalty','tenantA',['legal']); \
print(len(hits),'hits;', all(h['tenant_id']=='tenantA' for h in hits),'tenant ok'); \
print(hits[0] if hits else 'none')"
```

**Troubleshoot.** *`query_points` vs `search` API* — newer `qdrant-client` prefers `query_points`; if you're on an older version use `qc.search(...)`. *Reranker slow to load* — first call downloads the model (~500 MB); subsequent calls are fast. *BM25 returns nothing* — the pool was empty (bad filter or empty collection). *Cross-tenant hit sneaks in* — you post-filtered or forgot `query_filter`; that's the bug Step 9 catches.

---

### Step 8 — Groundedness (NLI) verifier + citations

**What.** `entails(premise, claim) -> float` using a cross-encoder NLI model, and `grounded_hits(hits, claim, thr)` that keeps only hits whose text entails the claim, annotating each with a `grounding` score.

**Why.** Retrieval returns *topically similar* text that may *contradict* or merely mention the query. Grounding is the gate against confidently-wrong citations in Week 2's answer generation — you verify a snippet **entails** the claim before trusting it.

**Do it.** `packages/retrieval/verify.py`:

```python
import os
from sentence_transformers import CrossEncoder

_nli = CrossEncoder(os.environ["NLI_MODEL"])   # nli-deberta-v3-base: [contradiction, entailment, neutral]

def entails(premise: str, claim: str) -> float:
    scores = _nli.predict([(premise, claim)])
    return float(scores[0][1])                 # entailment probability

def grounded_hits(hits, claim: str, thr: float = 0.5):
    out = []
    for h in hits:
        g = entails(h["text"], claim)
        if g >= thr:
            out.append({**h, "grounding": g})
    return out
```

**Expected result.** `entails("The termination penalty is two times the monthly fee.", "The penalty for early exit is 2x the fee.")` returns a high score (> 0.5); an unrelated premise returns low.

**Verify.**

```bash
python -c "from packages.retrieval.verify import entails; \
print('supported:', round(entails('The penalty is 2x the monthly fee.','Early exit costs twice the fee.'),3)); \
print('unrelated:', round(entails('The office is open on weekdays.','Early exit costs twice the fee.'),3))"
```

**Troubleshoot.** *Score index confusion* — different NLI models order labels differently; the `sentence-transformers` deberta card is `[contradiction, entailment, neutral]`, so entailment is index `1`. Print the raw vector once to confirm. *Everything scores neutral* — your `claim` and `premise` are swapped or the claim is a question, not a statement; NLI wants a declarative claim.

---

### Step 9 — API + the three proofs

**What.** A FastAPI surface (`GET /search`, `DELETE /documents/{doc_id}`, `GET /health`) and three pytest files that *demonstrate* the DoD claims.

**Why.** The whole week reduces to three tests that either pass or don't. Assertions, not vibes.

**Do it.** `apps/api/main.py`:

```python
from fastapi import FastAPI, Query
from packages.retrieval.retrieve import search
from packages.retrieval.migrate import delete_doc

app = FastAPI(title="capstone-kb")

@app.get("/health")
def health(): return {"ok": True}

@app.get("/search")
def search_ep(q: str, tenant_id: str, roles: list[str] = Query(default=[]), k: int = 5):
    return {"hits": search(q, tenant_id, roles, k)}

@app.delete("/documents/{doc_id}")
def delete_ep(doc_id: str):
    delete_doc("live", doc_id)                 # tombstone: server-side delete-by-filter
    return {"deleted": doc_id}
```

Run it: `uvicorn apps.api.main:app --reload --port 8000`, then `curl "http://localhost:8000/search?q=penalty&tenant_id=tenantA&roles=legal"`.

Seed a golden set at `evals/golden/golden.jsonl` — **≥20 rows** `{"tenant", "q", "roles", "gold_substr", "gold_doc_id"}` drawn from your real corpus. Then the three tests:

```python
# tests/test_delete.py  — delete is PROVABLE, no rebuild
from packages.retrieval.retrieve import search
from packages.retrieval.migrate import delete_doc

def test_delete_removes_from_answers():
    before = search("secret clause", "tenantA", ["legal"], k=20)
    assert any(h["doc_id"] == "D7" for h in before), "seed a query that hits D7 first"
    delete_doc("live", "D7")                                   # tombstone only, no reindex
    after = search("secret clause", "tenantA", ["legal"], k=20)
    assert all(h["doc_id"] != "D7" for h in after)             # D7 gone from ALL results
```

```python
# tests/test_tenant_isolation.py  — adversarial: A must never see B, even as admin
from packages.retrieval.retrieve import search

def test_tenant_A_cannot_see_B():
    # a query crafted to surface tenant B's content, issued by tenant A with a broad role
    hits = search("tenant B confidential merger terms", "tenantA", ["admin"], k=20)
    assert all(h["tenant_id"] == "tenantA" for h in hits)      # zero B docs, ever
```

```python
# tests/test_recall.py  — recall@5 >= your committed target
import json, os
from packages.retrieval.retrieve import search

def test_recall_at_5():
    rows = [json.loads(l) for l in open("evals/golden/golden.jsonl", encoding="utf-8")]
    hit = sum(any(r["gold_substr"].lower() in h["text"].lower()
                  for h in search(r["q"], r["tenant"], r["roles"], k=5)) for r in rows)
    recall = hit / len(rows)
    assert recall >= float(os.getenv("RECALL_TARGET", "0.75")), f"recall@5={recall:.2f}"
```

**Expected result.** `pytest -q` is green: the deleted doc is absent post-delete, tenant A gets zero B docs, and recall@5 clears your target.

**Verify.** `pytest -q tests/`. Read `retrieve.py` by eye to confirm the ACL filter is in the Qdrant query with **no** Python post-filter — that's part of the isolation DoD.

**Troubleshoot.** *`test_delete` red* — you're querying a collection name, not `live`, or deleting by point-id. *`test_recall` red* — your target is too high for a first pass; measure the baseline, set `RECALL_TARGET` from it, and improve via contextual chunking / better OCR — don't just lower the bar without noting it. *`test_tenant_isolation` red* — the leak is real; find the missing/dropped `query_filter`. This test failing is the test doing its job.

---

### Step 10 — DECISIONS.md: the CAG-vs-retrieval record

**What.** A short `docs/DECISIONS.md` stating, with **measured** numbers, the corpus-size break-even at which you'd choose CAG / long-context instead of retrieval.

**Why.** Engineering judgment is the deliverable, not a reflex "always RAG." Measure it (lecture 04).

**Do it.** Pick one tenant's corpus; compute total tokens (`sum(len(enc.encode(c.text)) for c in chunks)`). If it fits a long-context window (≲100–200k tokens), run ~10 golden questions **both** ways — retrieval vs whole-corpus-in-prompt (Ollama `llama3.1` with a large context, or note you'd use a long-context API) — and record answer quality, latency, and $/query. Write a one-paragraph verdict + the break-even corpus size where you'd switch. State explicitly that **CAG breaks per-doc deletes and ACLs** — a governance reason retrieval wins here regardless of size.

**Expected result.** A committed `docs/DECISIONS.md` with a table (tokens, latency, $/query, quality for each approach) and a verdict paragraph.

**Verify.** It's a doc — the check is that it contains *your* measured numbers and names the governance break-glass reason, not hand-waving.

---

## Putting it together — a short end-to-end run

```bash
# 0. services up
docker start capstone-qdrant                     # (and Ollama running for ctx prefixes)

# 1. reproducible pipeline: ingest -> redact -> quality_gate -> index
dvc repro                                         # halts here if the quality gate trips

# 2. make 'live' point at the freshly built collection (first time only)
python -c "from packages.retrieval.migrate import set_live; set_live('kb')"

# 3. serve
uvicorn apps.api.main:app --port 8000 &

# 4. query one tenant with citations
curl "http://localhost:8000/search?q=termination%20penalty&tenant_id=tenantA&roles=legal"

# 5. prove a delete removes a doc from answers, live
curl -X DELETE http://localhost:8000/documents/D7
curl "http://localhost:8000/search?q=secret%20clause&tenant_id=tenantA&roles=legal"   # D7 gone

# 6. the three proofs
pytest -q tests/
```

A green `pytest` plus a visibly-gone D7 in step 5 = your Week 1 backbone stands. Commit: `git add -A && git commit -m "week1: ingestion+retrieval backbone, deletes/recall/isolation proven"` and `dvc push`.

---

## Definition of Done — verifiable checks

Restating the spine's Week-1 DoD as things you can run:

- [ ] **Delete is provable.** `tests/test_delete.py` is green — a doc retrievable before `delete_doc` is absent from *all* results after, via tombstone/filter delete only (**no reindex**).
- [ ] **Recall gate.** `evals/golden/golden.jsonl` has **≥20** `(tenant, q, gold_substr, gold_doc_id)` rows; `test_recall.py` asserts **recall@5 ≥** your committed target (e.g. ≥0.75) and passes.
- [ ] **Tenant isolation proven adversarially.** `test_tenant_isolation.py` issues tenant-A queries crafted to surface tenant-B content (even with `admin` role) and returns **zero** B docs; the ACL filter lives in the **Qdrant query** — confirmed by reading `retrieve.py` (no Python post-filter).
- [ ] **OCR + bbox.** At least one **scanned** page was OCR-routed (`ocr=True` chunks exist) and **every** chunk carries `page` + a non-degenerate `bbox`.
- [ ] **Redaction + quality gate.** Presidio redacts PERSON/EMAIL/SSN/CREDIT_CARD; `quality_gate.gate()` **raises** on a deliberately corrupted input (prove it by injecting a garbage OCR chunk and watching `dvc repro` fail at the gate stage).
- [ ] **Versioned + reproducible.** `dvc repro` rebuilds the index from `data/raw` deterministically; the corpus is a committed `.dvc` hash.
- [ ] **Blue-green.** You flipped the `live` alias from a `blue`(=`kb`) to a `green` collection built with a **different** embedding model, validated recall on `green` *before* the flip, and can roll back by re-flipping.
- [ ] **Groundedness + citations.** `/search` returns per-hit `grounding` (NLI entailment) scores available and **page+bbox** citations on every hit.
- [ ] **DECISIONS.md.** States, with *your measured* numbers, the corpus-size break-even at which you'd choose CAG/long-context — and why governance (deletes + ACLs) keeps you on retrieval regardless.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `TesseractNotFoundError` | engine not on PATH | install binary; set `pytesseract.pytesseract.tesseract_cmd` |
| No `ocr=True` chunks | your "scanned" PDF has a text layer | use a genuinely image-only page; check `page.get_text()` |
| Degenerate/zero bbox | pixel coords not scaled to points | apply `72/dpi` scale in `ocr_page_blocks` |
| `spaCy [E050]` model missing | model not downloaded in venv | `python -m spacy download en_core_web_lg` |
| Quality gate never fires | thresholds too loose / chunk not `ocr=True` | tighten; mark the corrupt chunk `ocr=True` |
| Qdrant container exits | Windows volume-mount path | use a Windows-style `-v C:/path:/qdrant/storage` |
| Dimension mismatch on upsert | collection dim ≠ model dim | recreate collection, or use blue-green for a model change |
| Filter slow / wrong hits | no payload index | `create_payload_index` on `tenant_id`,`doc_id`,`acl_groups` |
| Deleted doc still returned | deleted by point-id, or queried collection not `live` | delete by `doc_id` filter; query the `live` alias |
| Cross-tenant hit appears | dropped/missing `query_filter` (post-filtered) | put ACL filter *in* the Qdrant query; never post-filter |
| `test_recall` red | target too high / weak retrieval | measure baseline, set target from it, add contextual prefixes |
| `dvc repro` reruns everything | wrong `deps`/`outs` paths | make outs match what scripts actually write |

---

## Stretch goals (optional)

- **Swap Tesseract for Surya/PaddleOCR** on your ugliest rotated/multi-column page and diff the junk-ratio in the quality gate — quantify OCR quality's effect on recall.
- **pgvector parity build.** Re-implement Step 5–7 on `pgvector` with `WHERE tenant_id = %s AND acl_groups && %s` and `DELETE ... WHERE doc_id = %s`; note the ergonomic differences in `DECISIONS.md`.
- **Per-tenant collections vs payload-filter** — build both isolation styles and measure query latency + the blast radius of a bug in each.
- **lakeFS instead of DVC** — put `data/raw` on a lakeFS branch, edit the corpus, and merge atomically to `main`; compare the mental model to DVC.
- **Reversible PII map** — store the un-redaction mapping in a *separate* secured table keyed by chunk hash, and add an authorized `/unredact` path (with a fake authz check) to prove the "authorized un-redaction" pattern without leaking PII into the index.
- **Recall harness upgrade** — add nDCG@10 and MRR to `evals/retrieval_eval.py` and plot recall vs `k`; this seeds Week 4's metric-with-CIs work.
- **Contextual chunking A/B** — run `test_recall.py` with and without the `ctx` prefix and record the delta; this is your first data point for the eval flywheel.
