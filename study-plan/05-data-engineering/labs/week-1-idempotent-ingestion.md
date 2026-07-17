# Week 1 Lab: Idempotent, Replay-Safe Ingestion with Dagster

> You are building the **ingestion layer** of `corpus-pipeline/` — the foundation every later week stacks on. Two sources land into one **immutable, date-partitioned landing zone**: a paginated public REST API (Hacker News, no auth) and a Postgres table via **watermark CDC**. The whole point is *replay safety*: you can re-run the job 5× and get a **byte-identical** reconciled hash, deletes travel as **tombstones** instead of vanishing, and a **blocking Dagster quality gate** physically stops bad data (a duplicate key) from flowing downstream.
>
> This is the discipline that makes Phase 4 (RAG) and Phase 8 (fine-tuning) trustworthy: raw bytes are the source of truth, every transform is re-derivable, and every backfill is trivial.
>
> **Read first (assumed done):** [Lecture 1 — Ingestion architecture & immutable landing zones](../lectures/01-ingestion-architecture-landing-zones.md) · [Lecture 2 — Idempotency & delivery semantics](../lectures/02-idempotency-delivery-semantics.md) · [Lecture 3 — CDC vs webhooks & tombstones](../lectures/03-cdc-vs-webhooks-tombstones.md) · [Lecture 4 — Dagster orchestration & quality gates](../lectures/04-dagster-orchestration-quality-gates.md) · [Lecture 5 — dlt incremental loading](../lectures/05-dlt-incremental-loading.md) (context for the stretch goal).

**Est. time:** ~9 hrs · **You will need:** Python 3.11+, [`uv`](https://docs.astral.sh/uv/), Docker Desktop (for local Postgres), a terminal (Windows/Git-Bash or macOS/Linux). **Everything is free and local** — the Hacker News Firebase API needs no key, and Postgres runs in a throwaway Docker container. No paid API, no GPU.

---

## Before you start (setup)

**What:** Scaffold the project, install dependencies, and stand up a local Postgres.

**Why:** `uv` gives you a reproducible, fast virtual env; the pinned dependency set is exactly what the lab uses; Docker Postgres is disposable so you never pollute a real DB while learning CDC.

**Do it:**

```bash
# 1. Scaffold
mkdir corpus-pipeline && cd corpus-pipeline
uv init

# 2. Runtime deps (one line; the backslash continuations work in Git-Bash and macOS/Linux)
uv add dagster dagster-webserver "dlt[postgres]" duckdb polars \
  "psycopg[binary]" httpx pydantic pyarrow xxhash

# 3. Dev dep
uv add --dev pytest

# 4. Local Postgres (throwaway)
docker run -d --name pg -e POSTGRES_PASSWORD=pw -p 5432:5432 postgres:16
```

> **Windows note:** In PowerShell the line-continuation is a backtick `` ` `` not `\`. The user is on Git-Bash, where `\` is correct — the commands above are copy-paste ready. If Docker Desktop is not running you will get `error during connect` — start Docker Desktop first.

**Create the folder layout:**

```bash
mkdir -p src/corpus tests landing
touch src/corpus/__init__.py
```

Target layout (you will fill these in during the steps):

```
corpus-pipeline/
  landing/                     # immutable raw zone — git-ignored, partitioned by date
  state/                       # watermarks & seen-hashes (git-ignored)
  src/corpus/
    __init__.py
    manifest.py                # step 1 — hashing + _manifest.json
    ingest_api.py              # step 2 — paginated HN API, idempotent
    ingest_cdc.py              # step 3 — Postgres watermark CDC + tombstones
    seed_pg.py                 # step 3 — create & seed documents table
    defs.py                    # step 4 — Dagster assets, checks, schedule
  tests/test_idempotent.py     # step 6
  pyproject.toml
```

Add a `.gitignore` so raw data and state never get committed:

```bash
cat > .gitignore <<'EOF'
.venv/
landing/
state/
__pycache__/
*.pyc
.pytest_cache/
EOF
```

**Verify setup:**

```bash
uv run python -c "import dagster, dlt, duckdb, polars, psycopg, httpx, xxhash; print('deps OK')"
docker exec pg pg_isready -U postgres          # -> "accepting connections"
```

**Troubleshoot:**
- `uv: command not found` → install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) or `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows).
- `port 5432 already allocated` → another Postgres is running. Either stop it, or run the container on a different host port: `-p 5433:5432` and use `port=5433` everywhere below.
- `psycopg` build errors → make sure you installed `"psycopg[binary]"` (the quotes matter in shells that expand `[]`).

---

## Step-by-step

### Step 1 — Immutable landing zone + manifest

**What:** Write a `manifest.py` that (a) computes a **stable content hash** over a set of rows and (b) writes a partition of raw JSONL to `landing/<source>/dt=<date>/part-<runid>.jsonl` plus a sibling `_manifest.json`.

**Why:** The landing zone is your source of truth (Lecture 1). It must be **append-only** — one new partition per run, never overwritten — so any downstream transform is re-derivable and any backfill is "just re-read the partitions." The `content_sha256` is what makes idempotency *provable* (Lecture 2): hash the *sorted, serialized* rows so the hash is independent of arrival order, and hash only **stable content** (never `fetched_at`) so re-fetching identical data produces an identical hash.

**Do it** — `src/corpus/manifest.py`:

```python
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import xxhash

LANDING = Path("landing")


def content_hash(record: dict, stable_keys: list[str]) -> str:
    """Fast per-record dedup key over ONLY stable fields (never fetched_at)."""
    payload = json.dumps(
        {k: record.get(k) for k in stable_keys},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return xxhash.xxh3_64_hexdigest(payload)


def dataset_sha256(rows: list[dict], stable_keys: list[str]) -> str:
    """Order-independent SHA-256 over the reconciled dataset.

    Sort rows by their stable content hash, serialize deterministically,
    and hash the whole thing. Two runs that landed the same logical rows
    (in any order, across any number of partitions) produce the SAME hash.
    """
    canonical = sorted(
        json.dumps(
            {k: r.get(k) for k in stable_keys},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for r in rows
    )
    h = hashlib.sha256()
    for line in canonical:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_partition(
    source: str,
    rows: list[dict],
    run_id: str,
    watermark_high,
    stable_keys: list[str],
    base: Path = LANDING,
    dt: str | None = None,
) -> Path:
    """Append ONE immutable partition + its manifest. Refuses to overwrite."""
    dt = dt or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    part_dir = base / source / f"dt={dt}"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"part-{run_id}.jsonl"
    if part_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable partition {part_path}")

    # durable write: write, flush, fsync
    with open(part_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    manifest = {
        "source": source,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "watermark_high": watermark_high,
        "row_count": len(rows),
        "content_sha256": dataset_sha256(rows, stable_keys),
        "part_file": part_path.name,
    }
    with open(part_dir / f"_manifest.{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return part_path


def read_all_rows(source: str, base: Path = LANDING) -> list[dict]:
    """Read every row across every partition of a source (for reconciliation)."""
    rows: list[dict] = []
    root = base / source
    if not root.exists():
        return rows
    for part in sorted(root.rglob("part-*.jsonl")):
        with open(part, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows
```

> **Design note:** the manifest filename embeds `run_id` (`_manifest.<runid>.json`) so multiple partitions on the same day each keep their own manifest — nothing is ever overwritten. The spine text says "`_manifest.json` per partition"; this is that, made collision-safe.

**Expected result:** module imports cleanly; no files written yet (pure functions + a writer you call later).

**Verify:**

```bash
uv run python -c "
from src.corpus.manifest import content_hash, dataset_sha256
r1={'id':1,'title':'a','fetched_at':'NOW'}
r2={'id':1,'title':'a','fetched_at':'LATER'}
sk=['id','title']
assert content_hash(r1,sk)==content_hash(r2,sk), 'stable hash must ignore fetched_at'
assert dataset_sha256([r1],sk)==dataset_sha256([r2],sk)
print('manifest hashing OK')
"
```

**Troubleshoot:**
- `ModuleNotFoundError: src` → run with `uv run python` from the `corpus-pipeline/` root (so `src/` is on the path). If it still fails, add an empty `src/__init__.py`.
- Hash differs across machines → make sure you did **not** include `fetched_at`/timestamps in `stable_keys`. That is the classic "every fetch looks new" bug (Lecture 2 pitfall).

---

### Step 2 — Idempotent paginated API ingest

**What:** `ingest_api.py` pulls Hacker News items newer than a stored **watermark**, dedups on a stable content hash, lands only genuinely new rows, and advances the watermark **after** the durable write. A `reconcile()` helper reads all partitions and reports the dataset hash.

**Why:** This is the heart of replay safety (Lecture 2). At-least-once delivery + idempotent writes = the *practical* substitute for exactly-once. Fetch-only-newer-than-watermark keeps runs cheap; content-hash dedup makes a re-run land **zero** rows; advancing the watermark only after `fsync` means a crash between fetch and commit re-fetches rather than loses data.

The fetch function is **injectable** so the idempotency test (Step 6) can feed a mock without touching the network.

**Do it** — `src/corpus/ingest_api.py`:

```python
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Callable

import httpx

from . import manifest

SOURCE = "api"
STABLE_KEYS = ["id", "title", "by", "url", "type"]  # NEVER include fetched_at
STATE_DIR = Path("state")
HN_BASE = "https://hacker-news.firebaseio.com/v0"

FetchFn = Callable[[int, int], list[dict]]


# ---- watermark + seen-hash state -------------------------------------------
def _state_path(source: str) -> Path:
    return STATE_DIR / f"{source}.state.json"


def load_state(source: str = SOURCE) -> dict:
    p = _state_path(source)
    if p.exists():
        return json.loads(p.read_text())
    return {"watermark": 0, "seen": []}  # watermark = highest HN item id seen


def save_state(state: dict, source: str = SOURCE) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = _state_path(source)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)  # atomic swap


# ---- default (real) fetch: Hacker News, no auth ----------------------------
def fetch_hn(since_id: int, limit: int = 25) -> list[dict]:
    """Fetch up to `limit` HN items with id > since_id (newest first)."""
    with httpx.Client(timeout=15) as client:
        max_id = client.get(f"{HN_BASE}/maxitem.json").json()
        rows: list[dict] = []
        item_id = max_id
        while item_id > since_id and len(rows) < limit:
            item = client.get(f"{HN_BASE}/item/{item_id}.json").json()
            if item:  # skip nulls (deleted/dead items)
                rows.append(
                    {k: item.get(k) for k in STABLE_KEYS}
                    | {"fetched_at": "IGNORED_FOR_HASH"}
                )
            item_id -= 1
        return rows


# ---- the idempotent ingest -------------------------------------------------
def ingest_api(
    fetch_fn: FetchFn = fetch_hn,
    limit: int = 25,
    base: Path = manifest.LANDING,
    source: str = SOURCE,
    dt: str | None = None,
) -> dict:
    """Land only new rows. Returns {new_rows, run_id, watermark}."""
    state = load_state(source)
    watermark = state["watermark"]
    seen = set(state["seen"])

    fetched = fetch_fn(watermark, limit)

    new_rows, new_hashes, new_watermark = [], [], watermark
    for r in fetched:
        h = manifest.content_hash(r, STABLE_KEYS)
        if h in seen:
            continue  # already landed identical content -> skip (idempotent)
        seen.add(h)
        new_hashes.append(h)
        new_rows.append(r)
        rid = r.get("id") or 0
        new_watermark = max(new_watermark, rid)

    if new_rows:
        run_id = uuid.uuid4().hex[:8]
        manifest.write_partition(
            source, new_rows, run_id, new_watermark, STABLE_KEYS, base=base, dt=dt
        )
        # advance watermark ONLY after the durable write above
        state["watermark"] = new_watermark
        state["seen"] = sorted(seen)
        save_state(state, source)
        return {"new_rows": len(new_rows), "run_id": run_id, "watermark": new_watermark}

    return {"new_rows": 0, "run_id": None, "watermark": watermark}


def reconcile(source: str = SOURCE, base: Path = manifest.LANDING) -> str:
    """Read all partitions, dedup on PK keeping one, return dataset sha256."""
    rows = manifest.read_all_rows(source, base=base)
    by_key: dict = {}
    for r in rows:
        by_key[r.get("id")] = r  # last-writer-wins per primary key
    return manifest.dataset_sha256(list(by_key.values()), STABLE_KEYS)


if __name__ == "__main__":
    import sys

    result = ingest_api()
    print("ingest:", result)
    print("reconciled sha256:", reconcile())
```

**Expected result:** first run lands ~25 rows and prints a reconciled hash; the second run lands **0** and prints the **same** hash.

**Verify — the idempotency proof by hand (do this for the README):**

```bash
# Run once to populate, then re-run 5x. All 5 hashes must match.
uv run python -m src.corpus.ingest_api
for i in 1 2 3 4 5; do
  uv run python -c "from src.corpus.ingest_api import ingest_api, reconcile; \
    r=ingest_api(); print('run',$i,'new_rows=',r['new_rows'],'sha=',reconcile())"
done
```

Expected: `new_rows= 0` on all five re-runs and five identical `sha=` values. Paste those five hashes into `README.md` (Definition of Done requirement).

Inspect what landed:

```bash
find landing/api -type f        # part-<runid>.jsonl + _manifest.<runid>.json
cat landing/api/dt=*/_manifest.*.json
```

**Troubleshoot:**
- 2nd run lands new rows instead of 0 → HN produced genuinely new items between runs (the feed is live). That is expected on a live feed; the *test* in Step 6 uses a frozen mock to prove strict idempotency. For the 5× hash proof, run the loop quickly, or freeze the input with a mock fetch (`fetch_fn=lambda since,limit: FIXED_ROWS`).
- Hash changes between runs → a mutable field leaked into `STABLE_KEYS`. Keep `fetched_at` out.
- `httpx.ConnectTimeout` → no internet / proxy. Point `fetch_fn` at a local fixture for offline work.

---

### Step 3 — Watermark CDC from Postgres (with tombstones)

**What:** Seed a `documents(id, body, updated_at, deleted bool)` table, then `ingest_cdc.py` selects rows where `updated_at > last_watermark`, lands them (including `deleted=true` **tombstones**), and advances the watermark **transactionally**.

**Why:** CDC propagates *change*, and the hardest change to propagate is a delete (Lecture 3). A deleted row must travel as an explicit **tombstone** (`deleted=true`) — "the row is gone" ≠ "nothing happened." Advancing the watermark in the same transaction as the read (after the file is durably on disk) is what makes a crash safe: either the watermark moved and the data is landed, or neither happened and the next run re-reads.

**Do it** — first the seed, `src/corpus/seed_pg.py`:

```python
from __future__ import annotations

import psycopg

DSN = "host=localhost port=5432 dbname=postgres user=postgres password=pw"


def seed() -> None:
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id          INT PRIMARY KEY,
                body        TEXT NOT NULL,
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                deleted     BOOLEAN NOT NULL DEFAULT false
            );
            CREATE TABLE IF NOT EXISTS ingest_watermark (
                source     TEXT PRIMARY KEY,
                watermark  TIMESTAMPTZ NOT NULL
            );
        """)
        cur.execute("DELETE FROM documents;")
        cur.executemany(
            "INSERT INTO documents (id, body) VALUES (%s, %s)",
            [(1, "alpha doc"), (2, "beta doc"), (3, "gamma doc")],
        )
        cur.execute(
            "INSERT INTO ingest_watermark (source, watermark) VALUES ('cdc', 'epoch') "
            "ON CONFLICT (source) DO NOTHING;"
        )
        conn.commit()
    print("seeded documents (3 rows)")


if __name__ == "__main__":
    seed()
```

Now the CDC ingest, `src/corpus/ingest_cdc.py`:

```python
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import psycopg

from . import manifest
from .seed_pg import DSN

SOURCE = "cdc"
STABLE_KEYS = ["id", "body", "deleted"]  # updated_at is the watermark, not content


def ingest_cdc(base: Path = manifest.LANDING, dt: str | None = None) -> dict:
    run_id = uuid.uuid4().hex[:8]
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT watermark FROM ingest_watermark WHERE source=%s", (SOURCE,))
            last_wm = cur.fetchone()[0]

            # tombstones (deleted=true) are just rows we DON'T filter out
            cur.execute(
                "SELECT id, body, updated_at, deleted FROM documents "
                "WHERE updated_at > %s ORDER BY updated_at",
                (last_wm,),
            )
            fetched = cur.fetchall()

        rows, new_wm = [], last_wm
        for _id, body, updated_at, deleted in fetched:
            rows.append({
                "id": _id,
                "body": body,
                "deleted": deleted,           # tombstone marker survives to landing
                "updated_at": updated_at.isoformat(),
            })
            new_wm = max(new_wm, updated_at)

        if not rows:
            return {"new_rows": 0, "run_id": None}

        # 1) durable write FIRST
        manifest.write_partition(
            SOURCE, rows, run_id, new_wm.isoformat(), STABLE_KEYS, base=base, dt=dt
        )
        # 2) advance watermark transactionally AFTER the write is on disk
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ingest_watermark SET watermark=%s WHERE source=%s",
                (new_wm, SOURCE),
            )
        conn.commit()

    return {"new_rows": len(rows), "run_id": run_id, "tombstones":
            sum(1 for r in rows if r["deleted"])}


if __name__ == "__main__":
    print(ingest_cdc())
```

**Expected result:** first run lands 3 rows (0 tombstones); after you mark a row deleted, the next run lands exactly that 1 tombstone row.

**Verify — the tombstone proof:**

```bash
uv run python -m src.corpus.seed_pg
uv run python -m src.corpus.ingest_cdc         # -> new_rows: 3, tombstones: 0

# Simulate a delete at the source: flip deleted + bump updated_at
docker exec pg psql -U postgres -c \
  "UPDATE documents SET deleted=true, updated_at=now() WHERE id=2;"

uv run python -m src.corpus.ingest_cdc         # -> new_rows: 1, tombstones: 1
grep -r '"deleted": true' landing/cdc/          # tombstone JSONL exists
```

**Troubleshoot:**
- `connection refused` → Postgres container down: `docker start pg`, wait for `pg_isready`.
- 2nd run re-lands the same 3 rows → the watermark update did not commit. Ensure `conn.commit()` runs after the `UPDATE`. With `psycopg`'s `with conn:` context, commit is required unless you use autocommit.
- Deletes don't reappear → you must **bump `updated_at`** when you set `deleted=true`, otherwise the watermark filter skips it. That is exactly the "treating deletes as absence" pitfall.

---

### Step 4 — Dagster DAG: assets, checks, freshness, schedule

**What:** `defs.py` declares three assets — `raw_api`, `raw_cdc`, and a downstream `landed_manifest` — plus a **blocking** `no_duplicate_keys` check, a 24h `freshness_check`, and a cron `ScheduleDefinition`. Then you run `dagster dev` and trigger from the UI.

**Why:** Dagster models the *data* as the graph, not tasks (Lecture 4). `landed_manifest` depends on both raw assets, so its lineage *is* the DAG. Asset checks are executable invariants; making the correctness check **blocking** means a bad partition physically cannot materialize downstream.

**Do it** — `src/corpus/defs.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import dagster as dg

from . import manifest
from .ingest_api import STABLE_KEYS as API_KEYS
from .ingest_api import ingest_api, reconcile
from .ingest_cdc import ingest_cdc

FRESHNESS_SLA_HOURS = 24


@dg.asset(description="Paginated HN API landed into the immutable zone.")
def raw_api(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    result = ingest_api()
    context.log.info(f"raw_api landed {result['new_rows']} rows")
    return dg.MaterializeResult(metadata={
        "new_rows": result["new_rows"],
        "reconciled_sha256": reconcile(),
    })


@dg.asset(description="Postgres watermark CDC landed with tombstones.")
def raw_cdc(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    result = ingest_cdc()
    context.log.info(f"raw_cdc landed {result['new_rows']} rows")
    return dg.MaterializeResult(metadata={"new_rows": result["new_rows"]})


@dg.asset(deps=[raw_api, raw_cdc], description="Roll-up manifest over both sources.")
def landed_manifest(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    api_rows = manifest.read_all_rows("api")
    cdc_rows = manifest.read_all_rows("cdc")
    return dg.MaterializeResult(metadata={
        "api_row_count": len(api_rows),
        "cdc_row_count": len(cdc_rows),
        "api_sha256": reconcile("api"),
    })


# ---- BLOCKING correctness gate: no duplicate primary keys in a partition ----
@dg.asset_check(asset=raw_api, blocking=True, description="No dup keys within a partition.")
def no_duplicate_keys() -> dg.AssetCheckResult:
    import json
    from pathlib import Path
    root = Path("landing/api")
    for part in root.rglob("part-*.jsonl"):
        ids = []
        with open(part, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.append(json.loads(line)["id"])
        dups = {i for i in ids if ids.count(i) > 1}
        if dups:
            return dg.AssetCheckResult(
                passed=False,
                severity=dg.AssetCheckSeverity.ERROR,
                metadata={"partition": str(part), "duplicate_ids": sorted(dups)},
            )
    return dg.AssetCheckResult(passed=True)


# ---- freshness: newest landed record must be < SLA hours old ----------------
@dg.asset_check(asset=raw_api, description=f"Newest record < {FRESHNESS_SLA_HOURS}h old.")
def freshness_check() -> dg.AssetCheckResult:
    import glob
    import json
    newest = None
    for m in glob.glob("landing/api/dt=*/_manifest.*.json"):
        ts = json.loads(open(m, encoding="utf-8").read())["ingested_at"]
        dt = datetime.fromisoformat(ts)
        newest = dt if newest is None else max(newest, dt)
    if newest is None:
        return dg.AssetCheckResult(passed=False, metadata={"reason": "no partitions yet"})
    age_h = (datetime.now(timezone.utc) - newest).total_seconds() / 3600
    return dg.AssetCheckResult(
        passed=age_h < FRESHNESS_SLA_HOURS,
        metadata={"age_hours": round(age_h, 2), "sla_hours": FRESHNESS_SLA_HOURS},
    )


ingest_job = dg.define_asset_job("ingest_job", selection="*")

daily_schedule = dg.ScheduleDefinition(
    job=ingest_job,
    cron_schedule="0 6 * * *",   # 06:00 daily
    name="daily_ingest",
)

defs = dg.Definitions(
    assets=[raw_api, raw_cdc, landed_manifest],
    asset_checks=[no_duplicate_keys, freshness_check],
    jobs=[ingest_job],
    schedules=[daily_schedule],
)
```

**Run the UI:**

```bash
uv run dagster dev -m src.corpus.defs
```

Open http://localhost:3000 → **Assets** tab → **Materialize all**. Watch `raw_api` and `raw_cdc` materialize, then `landed_manifest`. Click the **Checks** tab: `no_duplicate_keys` and `freshness_check` should both be **green**.

**Expected result:** DAG shows `raw_api` + `raw_cdc` → `landed_manifest`; both checks pass; the schedule `daily_ingest` appears under **Automation** (toggle it on to arm the cron).

**Verify:** In the run's structured logs you see `raw_api landed N rows` and the `reconciled_sha256` metadata on the asset.

**Troubleshoot:**
- `dagster dev` can't find defs → use `-m src.corpus.defs` (module form) from the repo root, or set `-f src/corpus/defs.py`.
- Port 3000 in use → `uv run dagster dev -m src.corpus.defs -p 3001`.
- Windows + Dagster: run from Git-Bash or PowerShell; if you hit a `grpc`/socket error, add `--dagit-host 127.0.0.1`. First launch is slow while it compiles the code location.
- Freshness fails immediately with "no partitions yet" → materialize `raw_api` at least once first.

---

### Step 5 — The blocking gate that halts (inject a duplicate, then fix)

**What:** Manually inject a duplicate primary key into a landed partition, re-run, and watch `no_duplicate_keys` **halt** materialization of `landed_manifest`. Then remove the dup and confirm the pipeline flows again.

**Why:** A check that only warns is theater (Lecture 4 pitfall). `blocking=True` means downstream assets literally do not run when the check fails — bad data cannot propagate. This is the "loudly stop" behavior you want in production.

**Do it — inject a dup into an existing partition:**

```bash
# Append a duplicate id to the newest landed API partition
PART=$(ls -t landing/api/dt=*/part-*.jsonl | head -1)
head -1 "$PART" >> "$PART"          # duplicate the first record's id
```

In the Dagster UI, **Materialize all** again (or re-run `ingest_job`).

**Expected result:** the `no_duplicate_keys` check fails (red, ERROR severity, metadata lists the `duplicate_ids`), and because it is **blocking**, `landed_manifest` is **skipped** — the run halts at the gate. The **Checks** tab shows the failure with the offending partition path.

**Verify the failure programmatically:**

```bash
uv run dagster asset materialize -m src.corpus.defs --select raw_api
# CLI exit is non-zero and logs show the blocking check failing
```

**Now fix it:**

```bash
# Rewrite the partition without the duplicate line (keep unique-by-id)
uv run python -c "
import json, sys, glob
part = sorted(glob.glob('landing/api/dt=*/part-*.jsonl'))[-1]
seen, keep = set(), []
for line in open(part, encoding='utf-8'):
    if not line.strip(): continue
    r = json.loads(line)
    if r['id'] in seen: continue
    seen.add(r['id']); keep.append(line.rstrip('\n'))
open(part,'w',encoding='utf-8').write('\n'.join(keep)+'\n')
print('deduped', part)
"
```

Re-materialize → check goes green → `landed_manifest` materializes again.

**Troubleshoot:**
- Check stays green after injecting → you appended to the wrong file or a different day's partition. Confirm `$PART` points at a file that actually has rows.
- `landed_manifest` still ran despite the failure → the check isn't blocking. Confirm `blocking=True` on `@asset_check` and that you're on a Dagster version that supports it (`uv run dagster --version`; blocking checks are GA in recent releases).

---

### Step 6 — The idempotency test

**What:** `tests/test_idempotent.py` runs `ingest_api` **twice against a frozen mock response** in an isolated temp landing dir, and asserts the second run lands **0** new rows and the reconciled `content_sha256` is unchanged.

**Why:** The live HN feed keeps moving, so you cannot prove strict idempotency against it. A mock freezes the input; then idempotency is a property of *your code*, testable in CI (Lecture 2's "prove idempotency with a repeatable hash test").

**Do it** — `tests/test_idempotent.py`:

```python
from pathlib import Path

from src.corpus import ingest_api as api


FIXED = [
    {"id": 101, "title": "hello", "by": "alice", "url": None, "type": "story",
     "fetched_at": "RUN_A"},
    {"id": 102, "title": "world", "by": "bob", "url": None, "type": "story",
     "fetched_at": "RUN_A"},
]


def mock_fetch(since_id, limit):
    # returns the SAME logical rows every call; fetched_at differs to prove
    # it is excluded from the content hash
    return [dict(r, fetched_at="RUN_B" if since_id else "RUN_A") for r in FIXED]


def test_second_run_lands_zero_and_hash_stable(tmp_path: Path, monkeypatch):
    # isolate state + landing to the tmp dir
    monkeypatch.setattr(api, "STATE_DIR", tmp_path / "state")
    base = tmp_path / "landing"

    r1 = api.ingest_api(fetch_fn=mock_fetch, base=base, dt="2026-01-01")
    sha1 = api.reconcile(base=base)
    assert r1["new_rows"] == 2

    r2 = api.ingest_api(fetch_fn=mock_fetch, base=base, dt="2026-01-01")
    sha2 = api.reconcile(base=base)

    assert r2["new_rows"] == 0, "second run must land zero new rows"
    assert sha1 == sha2, "reconciled content_sha256 must be unchanged"
```

> **Note:** `STATE_DIR` is patched via `monkeypatch` because `load_state`/`save_state` reference the module-level `STATE_DIR` at call time — patching the attribute redirects both. The `base=` param already isolates the landing zone.

**Run it:**

```bash
uv run pytest -q
```

**Expected result:** `1 passed`. Second run reports `new_rows == 0`; the two hashes are identical.

**Verify:** flip a stable field in `FIXED` (e.g. change `"hello"` to `"hi"`) and re-run — the test should now *fail* on the hash assert, proving the hash is sensitive to real content changes and insensitive to `fetched_at`. Revert after.

**Troubleshoot:**
- 2nd run lands 2 rows → dedup isn't persisting `seen`. Confirm `save_state` writes and `load_state` reads the same `STATE_DIR` (the monkeypatch must target the module attribute, not a local copy).
- `ModuleNotFoundError: src` under pytest → add a `conftest.py` at repo root or a `[tool.pytest.ini_options]` `pythonpath = ["."]` in `pyproject.toml`.

---

## Putting it together — short end-to-end run

```bash
# 0. Postgres up + seeded
docker start pg && uv run python -m src.corpus.seed_pg

# 1. Ingest both sources
uv run python -m src.corpus.ingest_api        # lands API partition + manifest
uv run python -m src.corpus.ingest_cdc        # lands CDC partition (3 rows)

# 2. Simulate a source delete -> tombstone
docker exec pg psql -U postgres -c \
  "UPDATE documents SET deleted=true, updated_at=now() WHERE id=2;"
uv run python -m src.corpus.ingest_cdc        # lands 1 tombstone row

# 3. Prove idempotency (paste these 5 into README)
for i in 1 2 3 4 5; do
  uv run python -c "from src.corpus.ingest_api import ingest_api, reconcile; \
    ingest_api(); print('sha',$i, reconcile())"
done

# 4. Tests
uv run pytest -q

# 5. Orchestrate + gate in the UI
uv run dagster dev -m src.corpus.defs         # Materialize all; inject a dup; watch it halt; fix
```

---

## Definition of Done — verifiable checks

Restating the spine's acceptance gate. Do not advance until every box is checked.

- [ ] **`uv run pytest` green, including the idempotency test** — 2nd run lands 0 new rows and reconciled `content_sha256` unchanged (Step 6).
- [ ] **5× re-run → identical reconciled `content_sha256`** — paste the 5 identical hashes into `README.md` (Step 2 / end-to-end step 3).
- [ ] **`landing/` holds date-partitioned, never-overwritten JSONL** at `landing/<source>/dt=<date>/part-<runid>.jsonl`, each with a per-partition `_manifest.*.json` carrying `{source, ingested_at, watermark_high, row_count, content_sha256}` (Step 1). Verify: `find landing -type f`; confirm `write_partition` raises on a re-used path.
- [ ] **Dagster UI shows the DAG, a passing freshness check, and a blocking duplicate-key check that halts the run** when you inject a dup, then passes after the fix (Steps 4–5).
- [ ] **CDC advances the watermark transactionally and emits a tombstone JSONL for a deleted `documents` row** — verify `grep -r '"deleted": true' landing/cdc/` returns a row and a re-run without new changes lands 0 (Step 3).

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| 2nd API run lands rows | live HN feed moved, or `seen` not persisted | use a mock for the strict proof; confirm `save_state`/`load_state` share `STATE_DIR` |
| Hash changes across runs | a mutable field (`fetched_at`) in `STABLE_KEYS` | hash only stable content |
| CDC re-lands same rows | watermark `UPDATE` not committed | ensure `conn.commit()` after the update |
| Deleted row never lands | `updated_at` not bumped on delete | set `deleted=true, updated_at=now()` together |
| `FileExistsError` on partition | run_id collision / re-run same run_id | run_id is random per run; don't hardcode it |
| Blocking check doesn't halt downstream | `blocking=True` missing or old Dagster | set `blocking=True`; upgrade Dagster |
| `dagster dev` can't load code | wrong `-m`/`-f` target or not in repo root | `uv run dagster dev -m src.corpus.defs` from root |
| `connection refused` (Postgres) | container stopped | `docker start pg`; wait for `pg_isready` |
| `ModuleNotFoundError: src` | `src` not on path | run via `uv run python -m ...` from root; add `pythonpath=["."]` |
| Docker `port already allocated` | another Postgres on 5432 | map `-p 5433:5432` and update `DSN` |

---

## Stretch goals (optional)

- **Swap the hand-rolled watermark for `dlt`** ([Lecture 5](../lectures/05-dlt-incremental-loading.md)): replace `ingest_api.py`'s state handling with a `dlt` resource using `dlt.sources.incremental("id")` and `write_disposition="merge"`. Confirm dlt's persistent load state gives you the same "0 rows on re-run" for free, then slot the dlt pipeline into the `raw_api` Dagster asset.
- **Second real source:** add GitHub issues (`https://api.github.com/repos/OWNER/REPO/issues?page=N`, `updated_at` watermark). Unauthenticated works within rate limits; add an optional `GITHUB_TOKEN` env for higher limits — still free.
- **Partition compaction:** many small daily partitions hurt scan speed (the small-files problem you'll fight in Week 3). Write a `compact.py` that merges a day's `part-*.jsonl` into one Parquet file while preserving the manifest hash.
- **Automated freshness materialization:** arm the `daily_ingest` schedule and add a Dagster **sensor** that alerts when `freshness_check` fails, so staleness pages you instead of silently shipping old data.
- **Crash-safety drill:** kill the process between the durable write and the watermark advance in `ingest_cdc.py` (add a `raise` after `write_partition`), then re-run and confirm no data is lost or double-counted — the proof that write-then-advance ordering (Lecture 2) actually holds.
