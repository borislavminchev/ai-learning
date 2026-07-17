# Phase 5 — Data Engineering for AI

**Phase goal:** 80% of real AI engineering is moving, parsing, cleaning, deduping, versioning, and governing data — not model code. In three weeks you will build the data plumbing that feeds both RAG (Phase 4) and fine-tuning (Phase 8): an idempotent, replay-safe ingestion pipeline; a robust extraction/cleaning/PII stage; and a versioned, incrementally-indexed corpus with governance you can *prove* works (delete a source doc and watch it disappear from answers).

Prev: [04-rag.md](../04-rag/04-rag.md) · Next: [06-agents.md](../06-agents/06-agents.md)

## Prerequisites
- Phase 0–4 complete: you can call an LLM API, produce structured output (Pydantic), generate embeddings, and stand up a working RAG pipeline with a vector DB (Qdrant/pgvector).
- Comfortable with Python 3.11+, `uv` for env management, Docker + docker-compose, SQL, and Git.
- A running Postgres you can create tables in (local Docker is fine), and Ollama or a cheap embedding API (OpenAI `text-embedding-3-small` / local `bge-small`).
- Basic Pandas/Polars; you have seen Parquet before.

**Time budget:** 3 weeks × ~10–15 hrs/week (~35 hrs total). Weighted ~35% theory / 65% hands-on.

## How to use this file
Do the Theory reading first each week (it is short on purpose), then spend the bulk of your hours in the Lab building the artifact. Treat the **Definition of Done** as a hard gate — do not advance until every box is checked. The three weekly labs compose into the Phase milestone project, so keep them in one repo.

> **📚 Course materials.** This file is the **spine** — a weekly plan and a *recap*. Deep material lives alongside it:
> - **Lectures**: [`lectures/`](lectures/00-index.md) — read for the *why*/mechanism.
> - **Lab guides**: [`labs/`](labs/) — follow to *build*.
> Each week: read the linked lectures first (the Theory here is their recap), then work the linked lab guide.

---

## Week 1 — Ingestion & Orchestration: idempotent, replay-safe, observable

### Objectives
By end of week you can:
- Explain and choose between batch, micro-batch, and streaming; and between pull (polling/CDC) and push (webhooks/queues) for a given source.
- Build an **idempotent** ingestion job that you can re-run 5× and produce byte-identical landing-zone output (proven by hash).
- Land raw records in an **immutable landing zone** partitioned by ingest date, with a manifest capturing source, watermark, and row counts.
- Orchestrate the pipeline as a Dagster DAG with a **freshness SLA** and a **quality gate** asset check that halts the run on anomaly.
- Implement Postgres **CDC** using a logical-replication / watermark strategy and replay it safely.

### Theory (~4 hrs)
> 📖 Read first: [Ingestion Architecture](lectures/01-ingestion-architecture-landing-zones.md) · [Idempotency & Delivery Semantics](lectures/02-idempotency-delivery-semantics.md) · [CDC vs Webhooks & Tombstones](lectures/03-cdc-vs-webhooks-tombstones.md) · [Dagster Orchestration & Quality Gates](lectures/04-dagster-orchestration-quality-gates.md) · [dlt Incremental Loading](lectures/05-dlt-incremental-loading.md). The bullets below are the recap.
- **Batch vs streaming vs micro-batch, pull vs push.** Read the Dagster docs "Assets" and "Automating pipelines" sections (search: *Dagster asset docs*). Internalize *why* immutable landing zones matter: raw data is your source of truth; every downstream transform must be re-derivable. Reprocessing beats in-place mutation.
- **Idempotency & delivery semantics.** "Exactly-once is a lie you engineer around" — you get at-least-once delivery plus idempotent writes keyed on a stable dedup key. Read the Kafka docs intro on delivery semantics (search: *Kafka delivery semantics exactly once*) for the mental model even though you won't run Kafka this week.
- **CDC vs webhooks.** CDC (Debezium/logical replication) captures every row change from the DB log; webhooks push events but can be lost/duplicated/reordered. Skim the Debezium "Introduction to CDC" doc (search: *Debezium connector Postgres*). You will implement a lightweight watermark-based CDC by hand, not run Debezium.
- **Freshness SLAs & data-quality gates as first-class steps.** Read Dagster's "Asset checks" and "Freshness" docs. A pipeline that silently ships stale/garbage data is worse than one that loudly halts.
- **dlt (data load tool).** Read the dlt "Getting started" page (search: *dlthub getting started*) — it gives you incremental loading + schema evolution + state for free.

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [Idempotent, Replay-Safe Ingestion with Dagster](labs/week-1-idempotent-ingestion.md). The steps below are the summary.
Build `corpus-pipeline/` — an idempotent ingestion layer for two sources: a paginated public REST API and a Postgres table via watermark CDC.

**Setup**
```bash
mkdir corpus-pipeline && cd corpus-pipeline
uv init && uv add dagster dagster-webserver dlt[postgres] duckdb polars \
  psycopg[binary] httpx pydantic pyarrow xxhash
uv add --dev pytest
docker run -d --name pg -e POSTGRES_PASSWORD=pw -p 5432:5432 postgres:16
```

**Folder layout**
```
corpus-pipeline/
  landing/                # immutable raw zone (git-ignored, partitioned by date)
  src/corpus/
    ingest_api.py         # paginated API -> landing/api/dt=YYYY-MM-DD/*.jsonl
    ingest_cdc.py         # Postgres watermark CDC -> landing/cdc/...
    manifest.py           # write/read run manifest (source, watermark, counts, sha256)
    defs.py               # Dagster Definitions: assets + asset_checks + schedule
  tests/test_idempotent.py
```

1. **Immutable landing zone + manifest.** Write raw JSONL to `landing/<source>/dt=<ingest_date>/part-<runid>.jsonl`. Never overwrite; each run is a new partition. `manifest.py` writes a `_manifest.json` per partition: `{source, ingested_at, watermark_high, row_count, content_sha256}`. Use `xxhash`/`sha256` over the sorted, serialized rows so re-runs are comparable.
2. **Idempotent paginated API ingest.** Use a public paginated API (e.g. the free `https://api.github.com/repos/OWNER/REPO/issues?page=N` or Hacker News Firebase API). Store a **watermark** (last-seen `updated_at`/id) in a small `state.json`. On each run, fetch only records newer than the watermark; dedup on primary key with `xxhash`; a record already landed with identical content is skipped. Re-running immediately must land **zero** new rows.
3. **Watermark CDC from Postgres.** Seed a `documents(id, body, updated_at, deleted bool)` table. `ingest_cdc.py` selects rows where `updated_at > last_watermark`, writes them to the CDC landing zone, and advances the watermark transactionally. Model deletes as **tombstone** rows (`deleted=true`) — never silently drop; downstream needs to see the delete. (You will consume tombstones in Week 3.)
4. **Dagster DAG.** In `defs.py` define assets `raw_api`, `raw_cdc`, and a downstream `landed_manifest`. Add an `@asset_check` `no_duplicate_keys` (fails if any dedup key appears twice within a partition) and `freshness_check` (fails if newest record older than the SLA, e.g. 24h). Add a `ScheduleDefinition` (cron). Run `uv run dagster dev` and trigger the job from the UI.
5. **Quality gate that halts.** Make the `no_duplicate_keys` check `blocking` so a bad partition stops materialization of downstream assets. Inject a duplicate to see it halt, then fix it.
6. **Idempotency test.** `tests/test_idempotent.py`: run `ingest_api` twice against a recorded/mock response; assert the second run adds 0 rows and the content_sha256 of the reconciled dataset is unchanged.

Everything runs on a laptop. No paid API needed (GitHub/HN APIs are free; Postgres is local Docker).

### Definition of Done
- [ ] `uv run pytest` green, including the idempotency test (2nd run lands 0 new rows).
- [ ] Running the API ingest 5× in a row produces the same reconciled `content_sha256` (paste the 5 identical hashes into the README).
- [ ] `landing/` contains date-partitioned, never-overwritten JSONL with a `_manifest.json` per partition.
- [ ] Dagster UI shows the DAG, a passing freshness check, and a **blocking** duplicate-key check that halts the run when you inject a dup.
- [ ] CDC advances a watermark transactionally and emits a tombstone row for a deleted `documents` row (verify the tombstone JSONL exists).

### Pitfalls
- **In-place mutation of the landing zone.** The moment you overwrite raw data you lose replayability. Always append new partitions.
- **Watermark stored non-atomically.** If you advance the watermark before the write commits, a crash loses data (or double-writes). Advance watermark in the same transaction / after a durable write + fsync.
- **Dedup key that isn't stable.** Hashing a payload that includes `fetched_at` or a mutable field makes every fetch look "new." Hash only the stable content.
- **Treating deletes as absence.** If a source row disappears, "no row" ≠ "deleted." Without explicit tombstones you can never propagate a delete downstream — which breaks Week 3's governance proof.
- **Non-blocking quality checks.** A check that warns but doesn't halt lets bad data flow. Make correctness gates `blocking`.

### Self-check
1. Why is at-least-once + idempotent writes the practical target instead of exactly-once?
2. What exactly makes your landing zone "immutable," and why does that unlock backfills?
3. How does a tombstone differ from deleting the row, and why do you need it downstream?
4. Where is your watermark stored, and what happens on a crash between fetch and commit?
5. When would you reach for CDC over webhooks, and vice versa?

---

## Week 2 — Extraction & Cleaning: parsing, OCR, dedup, PII redaction

### Objectives
By end of week you can:
- Parse messy documents (digital PDFs, scanned PDFs, HTML) into clean text + tables with **provenance** (source, page, bbox) preserved for later citations.
- Route scanned/low-quality pages to OCR and capture per-region confidence, routing low-confidence output to a review flag.
- Normalize and clean text (unicode/mojibake fixes, boilerplate stripping, language filtering) with a per-stage **drop report**.
- Deduplicate a corpus three ways: exact hash, near-duplicate via **MinHash-LSH**, and semantic (embedding) dedup — and explain when to use each.
- Detect and **redact PII** with Microsoft Presidio, producing true content removal (not just masking in a display layer).

### Theory (~4 hrs)
> 📖 Read first: [Document Parsing with Provenance](lectures/06-document-parsing-provenance.md) · [OCR Reality & Confidence Routing](lectures/07-ocr-confidence-routing.md) · [Text Normalization & Cleaning](lectures/08-text-normalization-cleaning.md) · [Deduplication at Three Tiers](lectures/09-deduplication-three-tiers.md) · [PII Detection & Redaction with Presidio](lectures/10-pii-detection-redaction.md). The bullets below are the recap.
- **Document parsing tradeoffs.** Read the **Docling** README/docs (search: *Docling IBM github*) and skim **Unstructured** docs (search: *Unstructured.io docs partition*). Understand digital-vs-scanned detection, reading order, and why tables must survive as structure (markdown/HTML), not flattened text. LlamaParse (paid) is the cloud alternative — know it exists.
- **OCR reality.** "Preprocessing dominates quality." Read the **PaddleOCR** README (search: *PaddleOCR github*) and skim the Tesseract docs intro. Cloud/VLM OCR (AWS Textract, Mistral OCR, GPT-4o vision) wins on messy layouts but costs money — use **confidence routing**: high-confidence auto-accept, low-confidence to human.
- **Deduplication.** Read the **datasketch** MinHash-LSH docs (search: *datasketch MinHash LSH*). Exact hash catches byte-identical dupes; MinHash-LSH catches near-dupes (boilerplate, minor edits) cheaply at scale; semantic dedup (cosine over embeddings) catches paraphrases but is expensive. Skim the FineWeb/`datatrove` blog for how large-scale corpora dedup (search: *HuggingFace FineWeb dedup*).
- **PII detection & redaction.** Read the **Presidio** docs (search: *Microsoft Presidio analyzer anonymizer*). Understand recognizers, context words, confidence scores, and consistent pseudonymization (same entity → same token). True redaction in PDFs/images means removing the underlying content, not drawing a black box over it.
- **Text normalization.** Skim `ftfy` docs (search: *ftfy fixes text for you*) for mojibake repair.

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [Extraction, Cleaning, Three-Tier Dedup, and PII Redaction](labs/week-2-extraction-cleaning-pii.md). The steps below are the summary.
Add an extraction/cleaning stage to `corpus-pipeline/` that turns landed raw docs into validated, PII-redacted, deduped records.

**Setup**
```bash
uv add docling datasketch presidio-analyzer presidio-anonymizer ftfy \
  langdetect sentence-transformers
uv run python -m spacy download en_core_web_lg   # Presidio NER backend
```

**Folder additions**
```
src/corpus/
  extract.py      # Docling parse -> {text, tables_md, page, bbox, is_scanned}
  ocr_route.py    # low-quality page -> PaddleOCR/VLM, capture confidence
  clean.py        # ftfy + boilerplate strip + langdetect filter
  dedup.py        # exact -> MinHash-LSH -> optional semantic
  pii.py          # Presidio analyze + anonymize (consistent pseudonyms)
  drop_report.py  # counts dropped per stage with reasons
```

1. **Parse with provenance.** Use Docling to parse a mixed folder (grab a few digital PDFs, one scanned PDF, and some HTML — e.g. a public manual + a photographed page). Emit one record per chunk/block: `{doc_id, page, bbox, text, tables_md, source_path, is_scanned}`. Tables must come out as markdown, not mangled prose.
2. **OCR routing.** In `ocr_route.py`, detect scanned/image pages (no embedded text layer or low char density). Run PaddleOCR (CPU works; slow but fine for a few pages). Capture per-line confidence; tag any block with mean confidence `< 0.80` as `needs_review=true`. Free/local path: PaddleOCR. Cheap cloud alternative if you want to compare: send the page image to GPT-4o-mini vision or Textract free tier.
3. **Clean & normalize.** `clean.py`: run `ftfy.fix_text`, strip boilerplate (repeated headers/footers, nav junk), drop non-English blocks with `langdetect` below a threshold. Every dropped block is logged with a reason.
4. **Dedup, three ways.** `dedup.py`:
   - **Exact:** `xxhash` of normalized text; drop collisions.
   - **Near-dup:** `datasketch` MinHashLSH (num_perm=128, Jaccard threshold ~0.85) over shingles; drop members of a near-dup cluster keeping the longest.
   - **Semantic (optional):** embed with `bge-small` / `all-MiniLM-L6-v2`, cluster with cosine > 0.95; report how many *additional* dupes this catches over MinHash. Write a short note on cost vs. benefit.
5. **PII redaction.** `pii.py`: Presidio `AnalyzerEngine` (EMAIL, PHONE, PERSON, CREDIT_CARD, US_SSN, etc.) → `AnonymizerEngine` with a **consistent** replacement (hash-based pseudonym so `john@x.com` always → `<EMAIL_7f3a>`). Output redacted text. Assert on a seeded fixture that known PII strings are absent from output.
6. **Drop report + validated JSONL.** Emit `data/clean/*.jsonl` (Parquet in Week 3) plus `reports/drop_report.json`: rows in, rows out, and count + reason per stage (parse-fail, non-english, exact-dup, near-dup, pii-only-doc, etc.). Wire this whole stage as Dagster assets downstream of Week 1's landing zone.

### Definition of Done
- [ ] Given a mixed doc folder, pipeline emits clean JSONL where tables are preserved as markdown and every record carries `{doc_id, page, bbox, source_path}` provenance.
- [ ] Scanned pages are OCR'd; blocks with mean confidence < 0.80 are flagged `needs_review` (show the count).
- [ ] Dedup removes 100% of exact dupes and demonstrably catches near-dupes MinHash finds that exact-hash misses (report both counts).
- [ ] PII test: for 15/15 seeded PII strings, the raw string does **not** appear in the redacted output, and the same input entity maps to the same pseudonym every time.
- [ ] `reports/drop_report.json` accounts for every input row (rows_in == rows_out + sum(drops)).
- [ ] `uv run pytest` green.

### Pitfalls
- **OCR without preprocessing.** Skew, low DPI, and noise wreck accuracy far more than engine choice. Deskew/binarize/upscale before OCR; measure confidence, don't assume.
- **Flattening tables to text.** Once a table becomes a wall of numbers you can't reconstruct it. Keep it as markdown/HTML with the cell structure.
- **Masking instead of removing PII.** A black rectangle over a PDF still has the text underneath. True redaction removes the content stream; verify the string is gone from the bytes.
- **Semantic dedup at the wrong stage.** Running embedding dedup over millions of docs before cheap exact+MinHash passes is a cost bomb. Cascade: exact → MinHash → (only then) semantic.
- **Presidio false negatives on custom formats.** Employee IDs, internal account numbers, non-US phones won't be caught by defaults — add custom recognizers and test them.

### Self-check
1. When does MinHash-LSH beat exact hashing, and when does semantic dedup earn its cost?
2. How do you tell a scanned page from a digital one programmatically?
3. What makes PII redaction "true removal" vs. cosmetic masking, and how do you verify it?
4. Why must provenance (page/bbox) survive parsing, and what breaks in Phase 4 if it doesn't?
5. What belongs in a drop report and why is "rows_in == rows_out + drops" a useful invariant?

---

## Week 3 — Datasets & Lifecycle: versioning, validation, incremental indexing, governance

### Objectives
By end of week you can:
- Store the corpus in **Parquet** and query it with **DuckDB**; explain when columnar beats JSONL.
- Version datasets and pipeline outputs with **DVC** (or lakeFS) and link a dataset version to a pipeline run for lineage.
- Enforce **data-quality contracts** with Great Expectations (or Pandera) as a **blocking gate** in the DAG ("PII must be redacted", schema, non-null).
- Generate a small batch of **synthetic data** with distilabel and curate/label it in Argilla (or Label Studio).
- Build an **incremental RAG indexer**: content-hash change detection, upserts, and **tombstone deletes** that provably remove a doc from answers; support a blue-green re-embedding migration.

### Theory (~4 hrs)
> 📖 Read first: [Columnar Storage & DuckDB](lectures/11-columnar-storage-duckdb.md) · [Dataset Versioning & Lineage](lectures/12-dataset-versioning-lineage.md) · [Data-Quality Contracts as Blocking Gates](lectures/13-data-quality-contracts-blocking-gates.md) · [Synthetic Data Generation & Curation](lectures/14-synthetic-data-curation.md) · [Incremental Indexing & Governance](lectures/15-incremental-indexing-governance.md). The bullets below are the recap.
- **Storage formats.** Read the DuckDB "Parquet" docs and the Arrow/Parquet overview (search: *DuckDB read_parquet docs*). Columnar + predicate pushdown = fast analytics on the corpus; JSONL is fine for streaming/append but slow to scan. Beware the small-files problem — compact.
- **Dataset versioning & lineage.** Read the **DVC** "Get Started: Data Versioning" docs (search: *DVC data versioning get started*). Understand `dvc add`, remotes, and `dvc.yaml` pipeline stages. Skim lakeFS docs (search: *lakeFS branching*) for the git-for-data / time-travel model as the scale alternative.
- **Data quality / validation.** Read **Great Expectations** "Quickstart" and Expectations concepts (search: *Great Expectations expectations*), plus **Pandera** docs (search: *Pandera dataframe schema*) as the lighter-weight, code-first option. Contracts are blocking gates, not warnings.
- **Synthetic data & curation.** Read the **distilabel** docs "Quickstart" (search: *distilabel argilla synthetic*) and the **Argilla** docs intro. Understand model-collapse and eval-contamination risks: filter synthetic data with a critic and never leak eval examples into training.
- **Incremental indexing & freshness.** Content-hash change detection (only re-embed changed docs), tombstone deletes, embedding-version migration (blue-green: build new index alongside old, cut over atomically). This is the same discipline you touched in Phase 3/4 — now you make it governed and reproducible.

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [Versioning, Quality Contracts, Incremental Indexing, and the Governance Proof (Phase Milestone)](labs/week-3-versioning-governance-milestone.md). The steps below are the summary.
Finish `corpus-pipeline/` by adding versioning, a blocking quality gate, a tiny synthetic+curation loop, and the **incremental RAG indexer with governance**.

**Setup**
```bash
uv add duckdb great-expectations pandera dvc "distilabel[openai]" argilla \
  qdrant-client sentence-transformers
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
git init && uv run dvc init
```

**Folder additions**
```
src/corpus/
  to_parquet.py     # clean JSONL -> partitioned Parquet
  validate.py       # GE / Pandera contract (blocking)
  synth.py          # distilabel: generate + critic-filter Q/A over corpus
  indexer.py        # incremental: hash-diff -> upsert/delete into Qdrant
  ask.py            # minimal RAG query for the governance proof
data/parquet/       # dvc-tracked
dvc.yaml            # stages: clean -> validate -> parquet -> index
```

1. **Parquet + DuckDB.** `to_parquet.py` writes the cleaned records to partitioned Parquet. Add a DuckDB query script that answers "rows per source", "docs flagged needs_review", "avg text length per doc" — show it runs in ms over the corpus.
2. **Blocking quality contract.** In `validate.py` define expectations/schema: required columns + types, `text` non-null/non-empty, `page >= 0`, and a **PII expectation** — a custom check (regex for emails/SSNs, or re-run Presidio) asserting redacted text contains no raw PII. Wire it as a **blocking Dagster asset check** between clean and parquet: injecting one un-redacted row must **fail the run**.
3. **Dataset versioning + lineage.** `dvc add data/parquet` (or express the pipeline in `dvc.yaml` stages). Commit. Change the corpus, re-run, commit again — show `dvc diff` / git history links a dataset version to code. Write the dataset version + git SHA into the run manifest so a model run can point back to exact data (lineage).
4. **Synthetic data + curation (small).** `synth.py`: use distilabel to generate ~30 synthetic Q/A pairs grounded in your corpus chunks, with a critic step that filters low-quality/ungrounded pairs. Push them to a local Argilla instance for human review/labeling (accept/reject). Export the accepted set as versioned JSONL. Note the collapse/contamination risk in the README.
5. **Incremental RAG indexer.** `indexer.py`:
   - Compute a `content_hash` per chunk. Maintain an index-state table (DuckDB/SQLite) of `chunk_id -> content_hash, embedding_version`.
   - On run: **new/changed** chunks → embed + upsert to Qdrant; **unchanged** → skip (log "0 re-embeds"); **tombstoned** docs (from Week 1 CDC) → delete their chunks from Qdrant.
   - Support **blue-green re-embed**: bump `embedding_version`, build a new Qdrant collection, backfill, then atomically switch `ask.py` to the new collection.
6. **Governance proof (the milestone-critical step).** `ask.py` runs a minimal RAG query (retrieve from Qdrant → LLM answer via Ollama/cheap API). Script this sequence and capture it:
   1. Ask a question whose answer lives in doc D → answer cites D. ✅
   2. Mark D deleted in Postgres → CDC emits tombstone → run indexer → chunks of D deleted from Qdrant.
   3. Ask the same question → the fact from D is **no longer retrievable/answerable**. ✅
   Save both answers + the Qdrant point counts before/after into `reports/governance_proof.md`.

Local/free stack throughout: Qdrant + Ollama + local embeddings. distilabel's generator can point at Ollama (OpenAI-compatible) instead of a paid API — set the base URL. Argilla runs locally in Docker.

### Definition of Done
- [ ] Corpus stored as partitioned Parquet; a DuckDB query returns per-source counts in < 1s (paste timing).
- [ ] Great Expectations/Pandera contract runs as a **blocking** gate; injecting one un-redacted PII row fails the pipeline (show the failing run).
- [ ] `dvc`/git history shows at least two dataset versions, and the run manifest records `{dataset_version, git_sha}` for lineage.
- [ ] ≥ 20 synthetic Q/A pairs generated, critic-filtered, and human-reviewed in Argilla; accepted set exported as versioned JSONL.
- [ ] Incremental indexer re-embeds **only changed** chunks (log shows 0 re-embeds on an unchanged re-run) and completes a blue-green collection switch.
- [ ] `reports/governance_proof.md` shows the before/after: the deleted doc's fact is answerable before delete and **not** answerable after (with Qdrant point counts dropping).
- [ ] `uv run pytest` green.

### Pitfalls
- **Re-embedding everything every run.** Without content-hash diffing you burn money/time and thrash the index. Diff first; embed only deltas.
- **Deletes that don't propagate to the index.** Removing a row from Postgres but leaving its vectors in Qdrant means the model still answers from deleted data — a compliance failure. The tombstone → delete path is the whole point.
- **In-place re-embedding across versions.** Mixing two embedding models in one collection corrupts similarity. Use blue-green: new collection, backfill, atomic switch.
- **Non-blocking expectations.** A GE suite that logs failures but returns success is theater. Make the checkpoint fail the DAG.
- **Synthetic data leaking into evals.** If your synthetic Q/A overlaps your eval set you get inflated, contaminated numbers. Decontaminate and keep provenance.
- **Small-files problem.** Thousands of tiny Parquet files kill scan performance — compact into reasonably sized partitions.

### Self-check
1. Why does deleting a doc from the source DB not remove it from answers, and what's the exact mechanism that fixes it?
2. When is Parquet+DuckDB the right store vs. keeping JSONL?
3. What does DVC give you that plain git doesn't, and how do you link a dataset version to a model run?
4. Why blue-green re-embedding instead of updating vectors in place?
5. How do you prevent synthetic data from causing model collapse or eval contamination?

---

## Phase milestone project — Reproducible corpus pipeline + governed incremental RAG indexer

Compose the three weeks into one shippable repo that proves the whole data lifecycle end-to-end.

**What it must do and prove:**
- **Ingest** a paginated API + Postgres CDC into an **immutable landing zone** — idempotent and replay-safe (re-run 5× → identical reconciled hash), with tombstones for deletes.
- **Orchestrate** in Dagster with a **freshness SLA** and **blocking quality gates** that halt on anomalies (duplicate keys, un-redacted PII).
- **Extract/clean/dedup/redact** into validated records: table-preserving parsing with provenance, OCR with confidence routing, three-tier dedup, Presidio PII redaction, and a per-stage **drop report**.
- **Version** the dataset (DVC/lakeFS) with lineage from source → dataset version → index, store it as **Parquet**, and enforce a Great Expectations/Pandera contract in CI/DAG.
- **Index incrementally**: content-hash change detection, upserts, **tombstone deletes**, and a blue-green re-embedding migration into Qdrant.
- **Govern**: a scripted proof that deleting a source document removes its facts from RAG answers (before/after answers + point counts in `reports/governance_proof.md`).

**Acceptance criteria**
- [ ] `uv run pytest` green across all stages.
- [ ] Idempotency proven (5 identical hashes) and CDC tombstone → index delete path works.
- [ ] Blocking gates demonstrably halt the pipeline on injected duplicate and injected PII.
- [ ] Drop report balances (rows_in == rows_out + drops); PII test passes (0 raw PII strings in output).
- [ ] Incremental re-run shows 0 unnecessary re-embeds; blue-green switch completes atomically.
- [ ] Governance proof shows the deleted fact is answerable before and unanswerable after.
- [ ] README documents every opinionated default and one "what I'd do differently at scale."

**Suggested repo layout**
```
corpus-pipeline/
  README.md                 # architecture, defaults, tradeoffs, scale notes
  dvc.yaml  .dvc/           # dataset versioning + pipeline stages
  docker-compose.yml        # postgres, qdrant, argilla
  landing/                  # immutable raw zone (git-ignored, dvc-remote optional)
  data/clean/ data/parquet/ # processed outputs (dvc-tracked)
  reports/                  # drop_report.json, governance_proof.md
  src/corpus/
    ingest_api.py ingest_cdc.py manifest.py          # Week 1
    extract.py ocr_route.py clean.py dedup.py pii.py drop_report.py   # Week 2
    to_parquet.py validate.py synth.py indexer.py ask.py             # Week 3
    defs.py                 # Dagster Definitions: assets, checks, schedule
  tests/                    # idempotency, PII removal, dedup, governance
```

## You are ready to move on when...
- [ ] You can design an idempotent, replay-safe ingestion pipeline with an immutable landing zone and explain your delivery-semantics choice.
- [ ] You can parse messy documents with provenance, route OCR by confidence, dedup at three tiers, and redact PII with verification.
- [ ] You can enforce data contracts as blocking gates and produce a drop report that balances.
- [ ] You can version a dataset, link it to a pipeline run for lineage, and store/query it efficiently in Parquet/DuckDB.
- [ ] You can build an incremental indexer that re-embeds only deltas, propagates deletes via tombstones, and migrates embedding versions blue-green.
- [ ] You have **proven** — not asserted — that deleting a source document removes it from RAG answers. This governance guarantee is what Phase 6 (agents acting on data) and the Capstone depend on.
