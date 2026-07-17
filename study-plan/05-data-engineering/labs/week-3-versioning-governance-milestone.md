# Week 3 Lab: Versioning, Quality Contracts, Incremental Indexing, and the Governance Proof (Phase Milestone)

> This is the week the three-week pipeline becomes a **governed, reproducible data system** — and the week you *prove* the guarantee the rest of the study plan leans on: **delete a source document and its facts vanish from RAG answers.** You will store the cleaned corpus as **Parquet** and query it with **DuckDB**, version it with **DVC** and stamp a lineage manifest, enforce a **blocking data-quality contract** (Great Expectations *or* Pandera — a single un-redacted PII row must *fail the run*), generate + critic-filter + human-curate a small **synthetic Q/A** set, and build an **incremental RAG indexer** with content-hash diffing, tombstone deletes, and a blue-green re-embedding migration into **Qdrant**. The capstone step wires it all into `reports/governance_proof.md`.
>
> This closes the loop opened in Week 1: raw bytes are the source of truth, deletes travel as tombstones, and now those tombstones physically remove vectors from the index. Phase 6 (agents acting on your data) and the Capstone assume this holds.
>
> **Read first (assumed done):** [Lecture 11 — Columnar Storage & DuckDB](../lectures/11-columnar-storage-duckdb.md) · [Lecture 12 — Dataset Versioning & Lineage (DVC)](../lectures/12-dataset-versioning-lineage.md) · [Lecture 13 — Data-Quality Contracts as Blocking Gates](../lectures/13-data-quality-contracts-blocking-gates.md) · [Lecture 14 — Synthetic Data Generation & Curation](../lectures/14-synthetic-data-curation.md) · [Lecture 15 — Incremental Indexing & Governance](../lectures/15-incremental-indexing-governance.md).

**Est. time:** ~9 hrs · **You will need:** Python 3.11+, [`uv`](https://docs.astral.sh/uv/), Docker Desktop (Qdrant + Argilla containers), Git, and the `corpus-pipeline/` repo you built in Weeks 1–2. **Everything runs free and local:** Qdrant and Argilla run in Docker, embeddings come from a local `sentence-transformers` model (`bge-small` / `all-MiniLM-L6-v2`), the LLM for RAG + synthetic generation is **Ollama** (OpenAI-compatible), the DVC remote is a **local directory**, and the index-state store is **DuckDB/SQLite**. No paid API, no GPU required (CPU embedding is slow but fine at this scale).

---

## Before you start (setup)

**What:** Add the Week 3 dependencies, stand up Qdrant + Argilla, initialize DVC on the existing git repo, and make sure Ollama is serving a small chat model.

**Why:** DVC needs a git repo to attach pointer files to (Lecture 12). Qdrant is the vector store the indexer reconciles against (Lecture 15). Argilla is the human-in-the-loop curation UI (Lecture 14). Ollama gives you a free OpenAI-compatible endpoint for both RAG answers and synthetic generation.

**Do it** — from inside `corpus-pipeline/`:

```bash
# 1. Week 3 runtime deps (one line; backslash continuations work in Git-Bash / macOS / Linux)
uv add duckdb great-expectations pandera dvc "distilabel[openai]" argilla \
  qdrant-client sentence-transformers openai

# 2. Vector store (throwaway container)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 3. Argilla — local, for human curation (Elasticsearch-backed image)
#    (uses the official argilla-quickstart image; runs its own storage internally)
docker run -d --name argilla -p 6900:6900 argilla/argilla-quickstart:latest

# 4. DVC on the existing repo (git was already init'd in Week 1)
uv run dvc init            # creates .dvc/ and .dvcignore; commit these

# 5. Local DVC remote (free — a plain directory acts as the blob store)
mkdir -p ../dvc-remote
uv run dvc remote add -d localremote ../dvc-remote
```

> **Windows / Git-Bash notes:**
> - In **PowerShell** the line-continuation is a backtick `` ` `` not `\`. On **Git-Bash** (your shell) `\` is correct — the commands above are copy-paste ready.
> - Docker port mapping is identical on Windows; make sure Docker Desktop is running or you get `error during connect`.
> - For the DVC remote path on Windows, a relative `../dvc-remote` works in Git-Bash. If you prefer absolute, use a `/c/Users/...` style path (Git-Bash) rather than `C:\...`.

**Ollama** (free local LLM — install once from `ollama.com`, then):

```bash
ollama pull llama3.1:8b          # chat model for RAG answers + generation
ollama pull nomic-embed-text     # optional: only if you want Ollama embeddings instead of bge-small
```

**New folder layout (added on top of Weeks 1–2):**

```
corpus-pipeline/
  .dvc/                     # DVC internals (commit this)
  dvc.yaml                  # step 3 — pipeline stages: clean -> validate -> parquet -> index
  data/
    clean/                  # Week 2 output: cleaned, redacted JSONL
    parquet/                # step 1 — dvc-tracked partitioned Parquet
    synthetic/              # step 4 — accepted Q/A JSONL (versioned)
  reports/
    governance_proof.md     # step 6 — the milestone artifact
  state/
    index_state.db          # step 5 — DuckDB/SQLite index-state table (git-ignored)
  src/corpus/
    to_parquet.py           # step 1 — clean JSONL -> partitioned Parquet
    duck_query.py           # step 1 — DuckDB analytics queries
    validate.py             # step 2 — GE / Pandera blocking contract
    synth.py                # step 4 — distilabel generate + critic filter
    indexer.py              # step 5 — incremental hash-diff -> upsert/delete into Qdrant
    ask.py                  # step 6 — minimal RAG query
```

Update `.gitignore` so the state DB and DVC-tracked outputs never get committed as raw bytes:

```bash
cat >> .gitignore <<'EOF'
state/
/data/parquet
EOF
```

**Verify setup:**

```bash
uv run python -c "import duckdb, pandera, qdrant_client, sentence_transformers, distilabel; print('deps OK')"
curl -s http://localhost:6333/readyz            # -> "all shards are ready"
curl -s http://localhost:6900/api/_status || echo "argilla starting (give it ~60s)"
ollama list                                     # llama3.1:8b listed
uv run dvc remote list                          # localremote  ../dvc-remote
```

**Troubleshoot:**
- `uv add distilabel` fails resolving deps → pin Python 3.11 (`uv python pin 3.11`) — some distilabel extras lag the newest Python.
- Qdrant `/readyz` refused → container still booting; wait ~10s, or `docker logs qdrant`.
- Argilla slow to start → the quickstart image bundles Elasticsearch; first boot can take 60–90s. Watch `docker logs -f argilla`.
- `dvc init` says "not a git repository" → you skipped Week 1's `git init`; run `git init` first.

---

## Step-by-step

### Step 1 — Parquet materialization + DuckDB analytics

**What:** Write `to_parquet.py` to convert Week 2's `data/clean/*.jsonl` into **partitioned Parquet** (partition by `source`), and `duck_query.py` to answer three analytical questions in milliseconds.

**Why:** JSONL is right for streaming append; Parquet queried by DuckDB is right for *repeated analytical scans* over a curated corpus — columnar layout + predicate pushdown read 10–100× less data (Lecture 11). Partitioning by `source` lets DuckDB skip whole files. Watch the **small-files problem**: thousands of tiny Parquet files kill scan performance, so compact to a few sensibly sized files per partition.

**Do it** — `src/corpus/to_parquet.py`:

```python
from __future__ import annotations
from pathlib import Path
import duckdb

CLEAN_GLOB = "data/clean/*.jsonl"
PARQUET_DIR = Path("data/parquet")

def write_parquet(clean_glob: str = CLEAN_GLOB, out_dir: Path = PARQUET_DIR) -> None:
    """Compact cleaned JSONL into Parquet partitioned by `source`.

    Uses DuckDB's Hive-style partitioned write. One (or few) files per
    partition avoids the small-files problem.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_json_auto('{clean_glob}', union_by_name=true)
        )
        TO '{out_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (source), OVERWRITE_OR_IGNORE, FILENAME_PATTERN 'part-{{i}}');
        """
    )
    con.close()

if __name__ == "__main__":
    write_parquet()
    print(f"wrote parquet to {PARQUET_DIR}")
```

`src/corpus/duck_query.py`:

```python
import time
import duckdb

PARQUET = "data/parquet/**/*.parquet"

QUERIES = {
    "rows_per_source": f"""
        SELECT source, COUNT(*) AS n
        FROM read_parquet('{PARQUET}', hive_partitioning=true)
        GROUP BY source ORDER BY n DESC""",
    "needs_review": f"""
        SELECT COUNT(*) AS flagged
        FROM read_parquet('{PARQUET}')
        WHERE needs_review = true""",
    "avg_len_per_doc": f"""
        SELECT doc_id, AVG(length(text)) AS avg_chars
        FROM read_parquet('{PARQUET}')
        GROUP BY doc_id ORDER BY avg_chars DESC LIMIT 10""",
}

if __name__ == "__main__":
    con = duckdb.connect()
    for name, sql in QUERIES.items():
        t0 = time.perf_counter()
        rows = con.execute(sql).fetchall()
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"\n== {name}  ({dt_ms:.1f} ms) ==")
        for r in rows[:10]:
            print(r)
```

**Run it:**

```bash
uv run python -m corpus.to_parquet
uv run python -m corpus.duck_query
```

**Expected result:** `data/parquet/source=<...>/part-0.parquet` files; `duck_query` prints per-source counts, the `needs_review` count, and top docs by length — each query timed at **well under 1 second** (typically single-digit to low-tens of ms on a laptop corpus).

**Verify:**
```bash
uv run python -c "import duckdb; print(duckdb.sql(\"SELECT COUNT(*) FROM read_parquet('data/parquet/**/*.parquet')\").fetchall())"
```
Count should equal your cleaned-row count from Week 2.

**Troubleshoot:**
- `read_json_auto` type errors → add `union_by_name=true` (already above) so ragged schemas across files merge; or cast columns explicitly.
- Hundreds of tiny files appear → you re-ran without `OVERWRITE_OR_IGNORE`, or your data is over-partitioned. Partition by `source` only; don't partition by `doc_id`.
- Query slow (seconds) → confirm you're reading Parquet, not the JSONL glob; check the file count per partition.

---

### Step 2 — Blocking data-quality contract (GE or Pandera)

**What:** Write `validate.py` with a contract: required columns + types, `text` non-null/non-empty, `page >= 0`, and a **PII expectation** (regex for emails/SSNs, or re-run Presidio) that fails if any raw PII survived Week 2's redaction. Wire it as a **blocking Dagster asset check** between `clean` and `parquet`.

**Why:** Redaction is code, and code has bugs (Lecture 13). The only way to *know* the corpus is clean is a gate that can say **no and halt the run**. A suite that logs failures but returns success is theater — a green dashboard vouching for poisoned data. Pandera is the right default for a solo pipeline (code-first, fail-fast); GE is shown as the heavier, documented alternative.

**Do it** — Pandera version (recommended default), `src/corpus/validate.py`:

```python
from __future__ import annotations
import re
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema
import pandas as pd

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

def _no_raw_pii(s: pd.Series) -> pd.Series:
    """Element-wise check: True when the cell contains NO raw email/SSN."""
    return ~s.fillna("").map(lambda t: bool(EMAIL_RE.search(t) or SSN_RE.search(t)))

corpus_schema = DataFrameSchema(
    {
        "doc_id":       Column(str, nullable=False),
        "source":       Column(str, nullable=False),
        "page":         Column(int, Check.ge(0), nullable=False),
        "text":         Column(
                            str,
                            checks=[
                                Check.str_length(min_value=1),           # non-empty
                                Check(_no_raw_pii, element_wise=False,
                                      error="raw PII detected in text"),  # PII gate
                            ],
                            nullable=False,
                        ),
        "needs_review": Column(bool, nullable=False),
    },
    strict=False,   # allow extra provenance columns (bbox, source_path, ...)
)

def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raise pandera.errors.SchemaError on any violation (fail-fast)."""
    return corpus_schema.validate(df, lazy=True)

if __name__ == "__main__":
    import duckdb
    df = duckdb.sql("SELECT * FROM read_json_auto('data/clean/*.jsonl', union_by_name=true)").df()
    validate_frame(df)
    print(f"contract PASSED: {len(df)} rows")
```

**Wire it as a blocking Dagster asset check** (in `defs.py` from Week 1, between the clean and parquet assets):

```python
from dagster import asset_check, AssetCheckResult, AssetCheckSeverity
import pandera as pa_errors
from corpus.validate import validate_frame

@asset_check(asset="clean_records", blocking=True)   # blocking=True halts downstream
def pii_and_schema_contract(context) -> AssetCheckResult:
    import duckdb
    df = duckdb.sql("SELECT * FROM read_json_auto('data/clean/*.jsonl', union_by_name=true)").df()
    try:
        validate_frame(df)
        return AssetCheckResult(passed=True, metadata={"rows": len(df)})
    except pa_errors.errors.SchemaErrors as e:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            metadata={"failure_cases": str(e.failure_cases.head(20))},
        )
```

**GE alternative (if you want browsable Data Docs):** create a suite with `expect_column_values_to_not_be_null("text")`, `expect_column_values_to_be_between("page", min_value=0)`, and a custom `expect_column_values_to_not_match_regex` for the email/SSN patterns, then run it via a **Checkpoint** and fail the Dagster check when `checkpoint_result.success is False`. The framework choice is cosmetic next to the blocking discipline.

**Run it + prove it blocks:**

```bash
uv run python -m corpus.validate                    # PASS on clean data
# Inject one un-redacted row and re-run to see it FAIL:
uv run python -c "open('data/clean/_poison.jsonl','w').write('{\"doc_id\":\"X\",\"source\":\"test\",\"page\":0,\"text\":\"reach me at jane@acme.com\",\"needs_review\":false}\n')"
uv run python -m corpus.validate                    # -> SchemaError, non-zero exit
rm data/clean/_poison.jsonl                         # clean up
```

**Expected result:** clean data passes; the poisoned row raises `SchemaError` (CLI) and, in the DAG, a **red blocking check** that prevents `to_parquet` from materializing. Take a screenshot of the failing Dagster run for your Definition of Done.

**Verify:** `echo $?` after the poisoned run is non-zero. In Dagster, downstream `parquet` asset shows as **skipped/blocked**, not materialized.

**Troubleshoot:**
- Pandera `int` check fails because `page` came back as float → cast: `df["page"] = df["page"].astype("int64")` before validate, or use `Column(pa.Int64, coerce=True)`.
- Regex too greedy / false positives on legit text → tighten patterns; the point is to catch what Presidio's pseudonyms (`<EMAIL_7f3a>`) should have replaced, so real PII patterns shouldn't survive.
- Dagster check runs but doesn't halt → confirm `blocking=True` and that the parquet asset actually `deps` on `clean_records`.

---

### Step 3 — Dataset versioning + lineage with DVC

**What:** Track `data/parquet` with DVC, express the pipeline as `dvc.yaml` stages, create **two dataset versions**, and stamp `{dataset_version, git_sha}` into the run manifest so an index/model run points back to exact bytes.

**Why:** Git chokes on large regenerated binaries — it stores a full new copy every commit (Lecture 12). DVC splits the problem: git tracks a tiny pointer (`.dvc` YAML naming a content hash), the bytes live in the remote keyed by hash. Lineage = when the indexer runs, it records the DVC hash of the data it read + the code commit, so from any index you walk back to the exact source.

**Do it:**

```bash
# Track the parquet dir by content hash (git gets a tiny pointer)
uv run dvc add data/parquet
git add data/parquet.dvc .gitignore
git commit -m "corpus v1: parquet dataset"
uv run dvc push                          # bytes -> ../dvc-remote
```

Express the pipeline in `dvc.yaml` (reproducible `clean -> validate -> parquet -> index`):

```yaml
# dvc.yaml
stages:
  parquet:
    cmd: uv run python -m corpus.to_parquet
    deps:
      - data/clean
      - src/corpus/to_parquet.py
    outs:
      - data/parquet
  validate:
    cmd: uv run python -m corpus.validate
    deps:
      - data/clean
      - src/corpus/validate.py
  index:
    cmd: uv run python -m corpus.indexer
    deps:
      - data/parquet
      - src/corpus/indexer.py
```

Add lineage to the run manifest — extend `manifest.py` (from Week 1) with a helper:

```python
import subprocess, json
from datetime import datetime, timezone
from pathlib import Path

def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()

def dataset_version() -> str:
    """DVC content hash of the parquet dataset (the 'which bytes' answer)."""
    dvc = json.loads(Path("data/parquet.dvc").read_text().replace("outs:", "").strip()) \
          if False else None  # parse YAML properly in real code
    import yaml
    doc = yaml.safe_load(Path("data/parquet.dvc").read_text())
    return doc["outs"][0]["md5"]

def write_run_manifest(path="reports/run_manifest.json", **extra) -> dict:
    m = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "dataset_version": dataset_version(),
        **extra,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(m, indent=2))
    return m
```

Now create **version 2**: change the corpus (e.g. add/edit a doc so Week 2 re-emits clean JSONL), then:

```bash
uv run python -m corpus.to_parquet
uv run dvc add data/parquet
git add data/parquet.dvc && git commit -m "corpus v2: added docs"
uv run dvc push
uv run dvc diff HEAD~1 HEAD              # shows files added/modified/deleted between versions
```

**Expected result:** `git log --oneline` shows ≥ 2 dataset-version commits; `dvc diff` reports the delta between them; `reports/run_manifest.json` carries both `dataset_version` (DVC hash) and `git_sha`.

**Verify:**
```bash
uv run dvc status              # "up to date" after push
cat data/parquet.dvc          # the tiny pointer (md5 + size + nfiles), NOT the data
git cat-file -s HEAD:data/parquet.dvc   # pointer is a few hundred bytes
```

**Troubleshoot:**
- `dvc add` says output already tracked → that's fine on re-adds; it updates the hash.
- `dvc push` to local remote fails on Windows path → use a Git-Bash `/c/Users/...` absolute path in `dvc remote add`.
- Committed the raw parquet by accident → `git rm -r --cached data/parquet` then re-`dvc add`; ensure `/data/parquet` is in `.gitignore`.

---

### Step 4 — Synthetic Q/A generation + critic filter + Argilla curation

**What:** `synth.py` uses **distilabel** to generate ~30 Q/A pairs grounded in corpus chunks, a **critic** step scores groundedness/quality and drops failures, and survivors go to **Argilla** for accept/reject. Export the accepted set as versioned JSONL.

**Why:** Synthetic data is model output repackaged as training/eval input (Lecture 14). Generate cheaply, then earn each example's place: the critic is the automatic quality gate (**ungrounded synthetic data is worse than no data**), Argilla is the human gate. Provenance + decontamination prevent model collapse and eval contamination. Point distilabel at **Ollama** (OpenAI-compatible) to stay free.

**Do it** — `src/corpus/synth.py` (key structure; distilabel is Pipeline → Step → Task):

```python
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.tasks import TextGeneration
from distilabel.llms import OpenAILLM   # OpenAI-compatible client -> point at Ollama

# Ollama exposes an OpenAI-compatible server at /v1 with no real key needed.
def ollama_llm(model: str = "llama3.1:8b") -> OpenAILLM:
    return OpenAILLM(
        model=model,
        base_url="http://localhost:11434/v1",
        api_key="ollama",   # placeholder; Ollama ignores it
    )

GEN_PROMPT = (
    "You are given a document chunk. Write ONE question whose answer is fully "
    "contained in the chunk, then the answer. Return JSON: "
    '{{"question": "...", "answer": "..."}}. Chunk:\n{text}'
)
CRITIC_PROMPT = (
    "Given the chunk and a Q/A pair, is the answer FULLY grounded in the chunk "
    "(no outside facts, no hallucination)? Reply JSON: "
    '{{"grounded": true|false, "score": 0-1, "reason": "..."}}.\n'
    "Chunk:\n{text}\nQ: {question}\nA: {answer}"
)

def build_pipeline(chunks: list[dict]) -> Pipeline:
    with Pipeline(name="synth-qa") as pipe:
        load = LoadDataFromDicts(data=chunks)                    # {chunk_id, text}
        generate = TextGeneration(llm=ollama_llm(), template=GEN_PROMPT,
                                  columns=["text"], output_mappings={"generation": "qa"})
        critic = TextGeneration(llm=ollama_llm(), template=CRITIC_PROMPT,
                                columns=["text", "question", "answer"],
                                output_mappings={"generation": "critique"})
        load >> generate >> critic
    return pipe

# After run: parse `qa`/`critique` JSON, keep rows where grounded==true and score>=0.7,
# and tag each with provenance {chunk_id, generator="llama3.1:8b", run_id}.
```

Filter, then push survivors to Argilla for human review:

```python
import argilla as rg

def push_to_argilla(records: list[dict], dataset_name: str = "synth-qa") -> None:
    rg.init(api_url="http://localhost:6900", api_key="argilla.apikey")  # default local key
    settings = rg.Settings(
        fields=[rg.TextField(name="question"), rg.TextField(name="answer"),
                rg.TextField(name="chunk")],
        questions=[rg.LabelQuestion(name="verdict", labels=["accept", "reject"])],
    )
    ds = rg.Dataset(name=dataset_name, settings=settings)
    ds.create()
    ds.records.log([rg.Record(fields=r) for r in records])
```

**Run it:**

```bash
uv run python -m corpus.synth       # generates, critic-filters, pushes to Argilla
# open http://localhost:6900  (default login argilla / 12345678), accept/reject each pair
```

Export the accepted set (query Argilla for `verdict == accept`) to `data/synthetic/qa_accepted.jsonl`, then version it:

```bash
uv run dvc add data/synthetic/qa_accepted.jsonl
git add data/synthetic/qa_accepted.jsonl.dvc && git commit -m "synthetic QA: accepted set v1"
```

**Expected result:** ≥ 20 accepted, grounded, human-reviewed Q/A pairs in versioned JSONL, each carrying provenance (`chunk_id`, generator model, run id). Add a README note on **model-collapse** and **eval-contamination** risks (decontaminate: run MinHash from Week 2 against your eval set; never leak).

**Verify:**
```bash
uv run python -c "import json;print(sum(1 for _ in open('data/synthetic/qa_accepted.jsonl')))"   # >= 20
```

**Troubleshoot:**
- distilabel can't reach Ollama → confirm `ollama serve` is running and `curl http://localhost:11434/v1/models` returns JSON.
- Argilla API key/login → the quickstart image default is user `argilla` / password `12345678`, API key `argilla.apikey`; check `docker logs argilla` for overrides.
- Generator emits non-JSON → add a lenient JSON extractor (regex the first `{...}`); or lower temperature.
- Argilla SDK version mismatch (v1 vs v2 API) → the snippet uses the v2 `rg.Dataset/Settings` API; if you installed v1, use `rg.FeedbackDataset`. Pin the version you install.

---

### Step 5 — Incremental RAG indexer (hash-diff, tombstones, blue-green)

**What:** `indexer.py` maintains an **index-state table** (`chunk_id → content_hash, doc_id, embedding_version, qdrant_point_id`), diffs fresh chunks against it (set math), embeds only new/changed chunks, upserts to Qdrant, deletes tombstoned docs' points, and supports a **blue-green** re-embed via a Qdrant **collection alias**.

**Why:** Cost must scale with the **delta, not the total** (Lecture 15). The state table is the source of truth about what's already indexed; the diff finds the minimum operations. Hashing avoids the embedding call (upsert only avoids duplicate points — you need both). Write **Qdrant first, then state**, so a crash leaves a harmless re-embed, never a silent hole. Deletes come from **tombstones keyed by `doc_id`** — never inferred from absence. Model changes are blue-green because cross-model cosine similarity is meaningless.

**Do it** — `src/corpus/indexer.py` (key signatures + the diff; you fill the glue):

```python
from __future__ import annotations
import duckdb, xxhash
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

ALIAS = "corpus"                 # ask.py always queries this alias
STATE_DB = "state/index_state.db"

def _state_con():
    con = duckdb.connect(STATE_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS index_state (
            chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,
            content_hash TEXT NOT NULL, embedding_version TEXT NOT NULL,
            qdrant_point_id TEXT NOT NULL, indexed_at TIMESTAMP NOT NULL)""")
    return con

def content_hash(text: str) -> str:
    """Hash STABLE text only (never fetched_at/run_id) — else every chunk looks new."""
    return xxhash.xxh3_64_hexdigest(text)

def diff(current: dict[str, str], stored: dict[str, str], tombstoned_docs: set[str],
         stored_docs: dict[str, str]) -> dict:
    """current: chunk_id->hash (fresh). stored: chunk_id->hash (state).
    Returns to_add / to_update / to_delete / skipped as sets of chunk_id."""
    to_add    = current.keys() - stored.keys()
    to_delete = (stored.keys() - current.keys()) | {
        c for c, d in stored_docs.items() if d in tombstoned_docs}
    to_check  = current.keys() & stored.keys()
    to_update = {c for c in to_check if current[c] != stored[c]}
    skipped   = to_check - to_update
    return {"to_add": to_add, "to_update": to_update,
            "to_delete": to_delete, "skipped": skipped}

def ensure_collection(qc: QdrantClient, name: str, dim: int) -> None:
    if not qc.collection_exists(name):
        qc.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

def run_indexer(embedding_version: str = "bge-small-v1",
                collection: str = "corpus_v1") -> dict:
    """Reconcile Qdrant to the current corpus with minimum ops.
    Order: embed+upsert to Qdrant FIRST, then update state (safe failure)."""
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    qc = QdrantClient("localhost", port=6333)
    ensure_collection(qc, collection, dim=model.get_sentence_embedding_dimension())
    # 1. load fresh chunks from parquet -> {chunk_id, doc_id, text}
    # 2. current = {chunk_id: content_hash(text)}; stored = read index_state
    # 3. tombstoned_docs = read Week-1 CDC tombstones (deleted=true) since last watermark
    # 4. d = diff(current, stored, tombstoned_docs, stored_docs)
    # 5. embed chunks in (to_add | to_update); qc.upsert(collection, points=[...])
    # 6. qc.delete(collection, points_selector=[point_ids for to_delete])
    # 7. update index_state to match (INSERT ON CONFLICT / DELETE)
    # 8. log counts; if all empty -> print "... skipped=N  (0 re-embeds)"
    ...

def blue_green_switch(qc: QdrantClient, new_collection: str, alias: str = ALIAS) -> None:
    """Atomically repoint the query alias to the freshly backfilled collection."""
    from qdrant_client.models import CreateAliasOperation, CreateAlias
    qc.update_collection_aliases(change_aliases_operations=[
        CreateAliasOperation(create_alias=CreateAlias(
            collection_name=new_collection, alias_name=alias))])
```

**Prove incrementality:**

```bash
uv run python -m corpus.indexer        # cold start: added=N updated=0 deleted=0 skipped=0
uv run python -m corpus.indexer        # unchanged re-run: added=0 ... skipped=N  (0 re-embeds)
```

**Prove blue-green** (bump version → new collection `corpus_v2` → full backfill → validate point count → `blue_green_switch`):

```bash
uv run python -c "from corpus.indexer import run_indexer, blue_green_switch, QdrantClient; \
run_indexer(embedding_version='bge-small-v2', collection='corpus_v2'); \
blue_green_switch(QdrantClient('localhost',port=6333),'corpus_v2')"
```

**Expected result:** first run embeds everything; the **second unchanged run logs `0 re-embeds`**; the blue-green switch repoints the `corpus` alias to `corpus_v2` in one operation with no half-built window.

**Verify:**
```bash
curl -s http://localhost:6333/collections/corpus_v1 | grep points_count
curl -s http://localhost:6333/collections/aliases       # alias 'corpus' -> corpus_v2 after switch
```

**Troubleshoot:**
- Full re-embed every run despite hashing → you folded a volatile field (`fetched_at`, `run_id`, a score) into `content_hash`. Hash the normalized text only.
- Qdrant `dimension mismatch` on blue-green → the new model outputs a different dim; that's *why* you create a fresh collection sized to the new model, never upsert into the old one.
- State says indexed but Qdrant point missing → you updated state before the Qdrant write; reorder to **Qdrant first, then state**.
- Alias switch doesn't take → `ask.py` must query the **alias** `corpus`, not a concrete collection name.

---

### Step 6 — The governance proof (milestone-critical)

**What:** `ask.py` runs a minimal RAG query (retrieve from the `corpus` alias → answer via Ollama). Script the before/after: ask a question answered by doc **D** (answer cites D), mark D deleted in Postgres → CDC emits a tombstone → run the indexer → D's chunks are deleted → ask again → the fact is **no longer answerable**. Save both answers + Qdrant point counts to `reports/governance_proof.md`.

**Why:** It is not enough to *assert* deletes propagate — you **prove** it (Lecture 15). This is an **integration** property spanning Postgres → CDC → indexer → Qdrant → LLM; a unit test that `delete()` was called proves nothing about the wiring where real failures hide. The point-count drop and the changed answer are physical, end-to-end evidence.

**Do it** — `src/corpus/ask.py` (minimal RAG):

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI

ALIAS = "corpus"
_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
_qc = QdrantClient("localhost", port=6333)
_llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def ask(question: str, k: int = 4) -> dict:
    qv = _model.encode(question).tolist()
    hits = _qc.query_points(ALIAS, query=qv, limit=k, with_payload=True).points
    context = "\n\n".join(h.payload.get("text", "") for h in hits)
    prompt = (f"Answer ONLY from the context. If the answer is not present, say "
              f"'I don't have information about that.'\n\nContext:\n{context}\n\nQ: {question}")
    resp = _llm.chat.completions.create(model="llama3.1:8b",
              messages=[{"role": "user", "content": prompt}], temperature=0)
    return {"answer": resp.choices[0].message.content,
            "cited_docs": [h.payload.get("doc_id") for h in hits]}

def point_count(collection: str = "corpus_v2") -> int:
    return _qc.count(collection, exact=True).count
```

Script the sequence (a `scripts/governance_proof.py` or inline):

```python
from corpus.ask import ask, point_count
from corpus.indexer import run_indexer
import psycopg, pathlib, textwrap

DOC_ID = "D42"   # the doc whose fact you're testing
Q = "What is the incident number for the Q3 outage?"

before = ask(Q); pc_before = point_count()

# 1) delete at source -> Week 1 CDC will emit a tombstone on next ingest
with psycopg.connect("host=localhost dbname=postgres user=postgres password=pw") as cx:
    cx.execute("UPDATE documents SET deleted=true WHERE id=%s", (DOC_ID,))

# 2) run CDC ingest (Week 1) to emit the tombstone, then the indexer to delete points
import subprocess; subprocess.run(["uv","run","python","-m","corpus.ingest_cdc"], check=True)
run_indexer(embedding_version="bge-small-v2", collection="corpus_v2")

after = ask(Q); pc_after = point_count()

report = textwrap.dedent(f"""\
# Governance Proof — delete removes facts from RAG answers

**Question:** {Q}
**Target doc:** {DOC_ID}

## Before delete
- Qdrant points: {pc_before}
- Cited docs: {before['cited_docs']}
- Answer: {before['answer']}

## After delete (source UPDATE deleted=true -> CDC tombstone -> indexer delete)
- Qdrant points: {pc_after}   (dropped by {pc_before - pc_after})
- Cited docs: {after['cited_docs']}
- Answer: {after['answer']}

**Result:** fact answerable BEFORE, NOT answerable AFTER, point count dropped. ✅
""")
pathlib.Path("reports/governance_proof.md").write_text(report)
print(report)
```

**Run it:**

```bash
uv run python scripts/governance_proof.py
cat reports/governance_proof.md
```

**Expected result:** `reports/governance_proof.md` shows the fact from D answered + citing D **before**, and "I don't have information about that." **after**, with the Qdrant point count dropping by exactly D's chunk count.

**Verify:** the point count in the report drops (e.g. 996 → 993); the "after" answer no longer cites `DOC_ID` and no longer contains the fact.

**Troubleshoot:**
- After delete, still answered → the tombstone → delete path didn't fire. Confirm `ingest_cdc` emitted a tombstone for `DOC_ID` (check the CDC landing partition) and that the indexer's `to_delete` set included D's chunks (it looks up `doc_id` in the state table — Lecture 15).
- Point count unchanged → the indexer deleted from the wrong collection; ensure it targets the **live** collection behind the alias.
- LLM still "knows" the fact from pretraining → pick a question whose answer is a synthetic/unique token that can't be in pretraining (e.g. an invented incident number). This isolates *retrieval* from parametric memory.

---

## Putting it together — a short end-to-end run

From a clean state, the whole Week 3 slice runs top to bottom:

```bash
# 0. services up
docker start pg qdrant argilla && ollama serve &

# 1. materialize + analyze
uv run python -m corpus.to_parquet && uv run python -m corpus.duck_query

# 2. quality gate (blocking) — must pass before parquet is trusted
uv run python -m corpus.validate

# 3. version + lineage
uv run dvc add data/parquet && git add data/parquet.dvc && git commit -m "corpus vN"
uv run dvc push

# 4. synthetic loop (generate -> critic -> Argilla -> export accepted)
uv run python -m corpus.synth      # then review at http://localhost:6900

# 5. incremental index (cold, then prove 0 re-embeds; blue-green switch)
uv run python -m corpus.indexer
uv run python -m corpus.indexer    # -> "0 re-embeds"

# 6. the proof
uv run python scripts/governance_proof.py && cat reports/governance_proof.md

# regression net
uv run pytest
```

Or drive stages 2/1/5 reproducibly via DVC: `uv run dvc repro` runs `validate → parquet → index` respecting `deps`.

---

## Definition of Done

Restating the spine's **Week 3 checklist** and the **Phase milestone acceptance** as verifiable checks:

**Week 3 Definition of Done**
- [ ] Corpus stored as **partitioned Parquet**; a DuckDB query returns per-source counts in **< 1s** (paste the timing from `duck_query.py`).
- [ ] Great Expectations/Pandera contract runs as a **blocking** gate; injecting one un-redacted PII row **fails the pipeline** (screenshot the failing Dagster run / non-zero exit).
- [ ] `dvc`/git history shows **≥ 2 dataset versions**, and `reports/run_manifest.json` records `{dataset_version, git_sha}` for lineage.
- [ ] **≥ 20 synthetic Q/A pairs** generated, critic-filtered, human-reviewed in Argilla; accepted set exported as **versioned JSONL** with provenance.
- [ ] Incremental indexer re-embeds **only changed** chunks (log shows **`0 re-embeds`** on an unchanged re-run) and completes a **blue-green** collection switch (alias repointed atomically).
- [ ] `reports/governance_proof.md` shows before/after: the deleted doc's fact is answerable **before** and **not** answerable **after**, with Qdrant point counts dropping.
- [ ] `uv run pytest` green.

**Phase milestone acceptance (Week 3's share of the end-to-end repo)**
- [ ] Blocking gates demonstrably **halt** the pipeline on injected PII (and, from Week 1, injected duplicate).
- [ ] Version the dataset with **lineage** source → dataset version → index, stored as Parquet, contract enforced in the DAG.
- [ ] Index incrementally: content-hash detection, upserts, **tombstone deletes**, blue-green migration into Qdrant.
- [ ] **Govern:** scripted proof that deleting a source document removes its facts from RAG answers.
- [ ] README documents every opinionated default and one "what I'd do differently at scale."

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| DuckDB query slow (seconds) | Reading JSONL glob, not Parquet; or small-files | Query `data/parquet/**/*.parquet`; compact partitions |
| Contract passes on obvious PII | Regex too narrow / column not scanned | Broaden email/SSN patterns; confirm `text` column is checked |
| Blocking check doesn't halt DAG | `blocking=True` missing or no `deps` edge | Set `blocking=True`; make parquet asset depend on clean |
| `dvc push` fails on Windows path | Backslash path in remote | Use Git-Bash `/c/Users/...` absolute path |
| Committed raw Parquet to git | Not git-ignored | `git rm -r --cached data/parquet`; ensure `.gitignore` + `.dvc` pointer |
| distilabel can't reach LLM | Ollama not serving `/v1` | `ollama serve`; `curl localhost:11434/v1/models` |
| Argilla login fails | Wrong default creds | `argilla` / `12345678`, key `argilla.apikey` (check `docker logs argilla`) |
| Full re-embed every run | Volatile field in `content_hash` | Hash normalized text only |
| Blue-green dimension mismatch | Upserting new-model vectors into old collection | New collection sized to new model; never mix spaces |
| State/index drift after crash | State updated before Qdrant write | Reorder: Qdrant first, then state |
| Governance: still answered after delete | Tombstone → delete path didn't fire | Verify CDC tombstone exists; indexer `to_delete` includes D by `doc_id` |
| Governance: point count unchanged | Deleted from wrong collection | Target the live collection behind the `corpus` alias |

---

## Stretch goals (optional)

- **lakeFS instead of DVC.** Stand up lakeFS in Docker and model the corpus as a branch you merge — the git-for-data / time-travel model that scales past DVC's single-machine cache (Lecture 12). Compare the workflows in your README.
- **Periodic reconcile job.** Add a Dagster asset that compares `COUNT(index_state)` vs. Qdrant's real point count and **alerts on drift** — your early warning for state/index divergence (Lecture 15).
- **Decontamination gate.** Run Week 2's MinHash-LSH between your accepted synthetic Q/A and any eval set; **block** export if Jaccard overlap exceeds a threshold. Prevents eval contamination.
- **Semantic-diff indexing.** Beyond content-hash, detect chunks whose *meaning* changed (cosine drift over a version) and re-embed only those — measure how many additional re-embeds it triggers vs. pure hash-diff.
- **GE Data Docs in CI.** Wire the Great Expectations checkpoint into a git pre-commit or CI step that regenerates browsable Data Docs and fails the build on contract violation.
- **Automate the governance proof in CI.** Regenerate `reports/governance_proof.md` on every pipeline change and fail if the "after" answer still contains the deleted fact — turn the milestone into a standing regression test.
