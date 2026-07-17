# Lecture 5: dlt — Incremental Loading, Schema Evolution, and Pipeline State for Free

> In lectures 2 and 3 you hand-rolled the three hard parts of ingestion: an idempotent write keyed on a stable hash, a watermark you advance transactionally, and a tombstone path for deletes. That was the right way to *learn* — you now understand the primitives in your bones. But nobody wants to re-derive watermark-persistence-across-crashes for every new source, and nobody wants to hand-write `ALTER TABLE ... ADD COLUMN` the day the API adds a `reactions` field. **dlt (data load tool)** is a Python library that hands you incremental loading, automatic schema inference and evolution, and durable pipeline state — the exact machinery you just built by hand — as library defaults. This lecture exists so you can recognize *when your bespoke code is reinventing dlt* and adopt it deliberately, and *when bespoke still wins* because you need control dlt won't give you. After this you will be able to model a source with `@dlt.resource`, pick a write disposition (`append` / `replace` / `merge`), wire `dlt.sources.incremental` on `updated_at` to automate the watermark, understand how dlt persists state between runs, and slot a dlt pipeline into the Dagster DAG from lecture 4.

**Prerequisites:** Lectures 2 (idempotency/watermarks), 3 (CDC/tombstones), 4 (Dagster assets); Python 3.11+, `uv`; comfort with SQL and generators/`yield` · **Reading time:** ~28 min · **Part of:** Phase 5 (Data Engineering for AI) Week 1

---

## The core idea (plain language)

Every ingestion pipeline you will ever write repeats the same four chores:

1. **Extract** rows from a source (paginate the API, `SELECT` from the DB).
2. **Only take what's new** since last run (the watermark / incremental cursor).
3. **Figure out the shape** of the data and make the destination table match it (schema).
4. **Land it correctly** — append, overwrite, or upsert-by-key — and remember where you got to, durably, so the next run resumes instead of re-downloading everything.

You built #2, part of #4's state, and a manual version of #3 by hand this week. That was ~200 lines of careful, crash-safe code *per source*. dlt's bet is that these four chores are **not where your business value lives** — your value is in *which* sources, *what* the data means, and *what you do with it downstream*. So dlt makes chores 2, 3, and 4 declarative:

> You write a Python generator that `yield`s dictionaries. You annotate it with *how* to load (`write_disposition`, `primary_key`) and *what makes a row new* (an `incremental` cursor). dlt does schema inference, state persistence, watermark tracking, and the idempotent load — into DuckDB, Postgres, BigQuery, Snowflake, or a filesystem — without you touching a single `CREATE TABLE` or `ALTER TABLE`.

That's the whole pitch: **the ingestion primitives from lectures 2–3, as batteries-included defaults.** Everything below is mechanism — what dlt is actually doing when you stop doing it yourself, so that when it misbehaves in production you know exactly which knob to reach for.

---

## How it actually works (mechanism, from first principles)

### The vocabulary: source, resource, pipeline, destination

Four nouns. Learn them precisely — dlt's whole API hangs off them.

- **Resource** — one logical stream of records, produced by a Python generator that `yield`s dicts (or lists of dicts). Think "one table's worth of data." Decorated with `@dlt.resource`.
- **Source** — a group of related resources that share config/credentials (e.g. a "GitHub" source exposing `issues`, `pulls`, `comments`). Decorated with `@dlt.source`. Optional — a single resource works fine alone.
- **Destination** — where data lands: `duckdb`, `postgres`, `filesystem`, `bigquery`, `snowflake`, etc. A string in the pipeline config; dlt owns the SQL/write mechanics per destination.
- **Pipeline** — the object that connects a source to a destination and *runs* the load. It also owns the **state** (more on this below). Created with `dlt.pipeline(...)`.

Minimal end to end:

```python
import dlt

@dlt.resource(write_disposition="append")
def issues():
    for page in paginate("https://api.github.com/repos/x/y/issues"):
        yield page          # yields a list of dicts; dlt flattens

pipe = dlt.pipeline(
    pipeline_name="gh",
    destination="duckdb",
    dataset_name="raw",
)
info = pipe.run(issues())   # infers schema, creates table, loads rows
```

That `pipe.run(...)` did what your lab's `ingest_api.py` does — fetch, normalize, write — *plus* created the table with correct column types, *plus* recorded load state to disk. No DDL.

### Schema inference and evolution

dlt inspects the dicts you yield and infers a schema: column names, types (via a type map — Python `int`→`bigint`, `str`→`text`, `datetime`→`timestamp`, etc.), and nested structure. Nested objects get **unnested**: a dict under `author` becomes columns `author__login`, `author__id` (double-underscore is dlt's nesting separator). Lists of objects become **child tables** with an auto-generated `_dlt_parent_id` foreign key — dlt normalizes JSON into relational tables for you.

The important part is **evolution**. On the *next* run, dlt diffs the newly inferred schema against the stored one:

- **New column appears** (API added `reactions_count`) → dlt runs `ALTER TABLE ADD COLUMN` and leaves old rows NULL there. Non-breaking, automatic.
- **New type is compatible** (a column seen as `bigint` now sees a float) → dlt widens the column type where the destination allows.
- **Type conflict that can't be widened** (a column was `bigint`, now a string arrives) → by default dlt creates a **variant column** (`reactions__v_text`) rather than corrupt or fail — the original stays typed, the oddball lands beside it.

Every schema is versioned and stored (as a hash + the full schema) so you can see exactly *when* a column appeared. You can also **lock the contract** — `schema_contract="freeze"` makes an unexpected column *raise* instead of evolve, which is what you want on a governed production table where a silent new column is a signal something upstream changed and a human should look.

```
Run 1 dict:  {"id":1, "title":"bug"}          → table issues(id bigint, title text)
Run 2 dict:  {"id":2, "title":"x", "labels":3}→ ALTER ADD labels bigint; row lands
Run 3 dict:  {"id":3, "title":"y", "labels":"urgent"}
             labels was bigint, now text → variant column labels__v_text
```

Contrast the hand-rolled path: your lab writes JSONL to a landing zone, which is *schema-on-read* — you dodge DDL entirely by never having a typed table. That's a legitimate strategy (raw zone = source of truth, defer typing downstream). dlt occupies the next layer: it *lands into a typed destination* and manages the DDL so you get queryable tables now, not after a separate transform.

### Write dispositions — the three ways to land

This is the single most important knob. It answers "given rows I'm loading, what happens to what's already there?"

| Disposition | Behavior | Watermark? | Use when |
|---|---|---|---|
| `append` | Add new rows; never touch existing | Usually paired with `incremental` | Immutable event streams, logs, append-only landing |
| `replace` | Truncate the table, load fresh | No — you reload everything | Small dimension tables, full snapshots, "just give me current state" |
| `merge` | Upsert by `primary_key` (update matches, insert new) | Paired with `incremental` | Mutable records with a stable key + `updated_at` |

`merge` is the one that maps directly to your lab. It needs a `primary_key` (or `merge_key`). dlt loads the incoming rows into a **staging** table, then runs a `DELETE` of matching keys followed by `INSERT` (or an actual upsert where the destination supports it) inside a transaction. The effect: applying the same row twice is idempotent — the second load overwrites, it does not duplicate. **This is precisely the "idempotent write keyed on a stable dedup key" from lecture 2**, except dlt writes the staging-and-swap SQL for you, per destination.

```python
@dlt.resource(
    write_disposition="merge",
    primary_key="id",
)
def documents(updated=dlt.sources.incremental("updated_at")):
    yield from select_since(updated.last_value)   # only newer rows
```

### The incremental cursor — the watermark, automated

`dlt.sources.incremental("updated_at")` is the star of the show. Declaring it does four things you did by hand in lecture 2:

1. **Reads the last watermark** from pipeline state at the start of the run and exposes it as `.last_value`. First run: it's `None` (or an `initial_value` you set).
2. **Filters** the yielded rows: dlt drops rows whose `updated_at` is `<= last_value` (so even if your generator over-fetches, dlt won't load stale rows). You can also pass the value to the source query to avoid fetching them at all — cheaper.
3. **Tracks the new high-water mark** as rows flow through: the max `updated_at` seen this run.
4. **Persists the new watermark** into state *only after the load commits* — advance-after-durable-write, the exact ordering lecture 2 hammered on to survive a crash between fetch and commit.

It also handles the boundary case you probably got subtly wrong the first time: **duplicate cursor values at the boundary.** If three rows share `updated_at = 2026-07-09T12:00:00`, a naive `> last_value` re-fetches them (fine, but wasteful) or a `>=` risks missing them. dlt deduplicates on the boundary using the `primary_key` so a row exactly at the watermark isn't loaded twice across runs. `last_value_func` lets you swap `max` for `min` or a custom reducer; `end_value` lets you bound a backfill window.

```
Run 1: state.updated_at = null → load all, new high = 2026-07-08T09:00 → persist
Run 2: last_value = 2026-07-08T09:00 → query WHERE updated_at > that
        3 new rows, new high = 2026-07-09T12:00 → persist AFTER commit
Crash mid-load? state NOT advanced → Run 3 re-loads from 09:00 (at-least-once)
                merge disposition makes the re-load idempotent → no dupes
```

### Pipeline state — where the watermark actually lives

dlt keeps two kinds of state, and confusing them causes real production bugs:

- **Local state** — a small JSON blob under `~/.dlt/pipelines/<name>/` on the machine that ran the pipeline. Fast, but *machine-local*: run the same pipeline from a different container/CI runner and it won't see it.
- **Destination state** — dlt also writes state into a `_dlt_pipeline_state` table **in the destination itself**, alongside your data, committed in the same load. This is the source of truth. On a fresh machine, dlt restores state *from the destination*, so your watermark travels with your data, not with your laptop.

That second point is the quiet superpower and the quiet footgun. Superpower: state is durable and portable exactly because it lives with the data. Footgun: if you point the same `pipeline_name` at a *different* dataset, or wipe the destination, you lose the watermark and the next run does a full reload. Corollary: `pipeline_name` + `dataset_name` are effectively the *identity* of your incremental progress. Rename them casually and you've reset the watermark.

dlt also writes bookkeeping tables `_dlt_loads` (one row per load, with a `load_id` and status) and `_dlt_version` (schema versions). `_dlt_loads` gives you exactly-once-*ish* semantics: a load that didn't fully commit is marked incomplete and can be resumed/rolled forward, which is why a crashed run doesn't leave half-loaded garbage that fails your dedup check.

---

## Worked example

Concretely map the lab's two sources onto dlt and count what happens.

**Source A — paginated GitHub issues, mutable, keyed by `id`, has `updated_at`.** This is a `merge` + `incremental` case.

```python
@dlt.resource(write_disposition="merge", primary_key="id")
def issues(cursor=dlt.sources.incremental("updated_at", initial_value="2026-01-01T00:00:00Z")):
    since = cursor.last_value
    for page in paginate_issues(since=since):   # pass watermark to the API
        yield page
```

- **Run 1 (cold):** state empty → cursor uses `initial_value`. API returns 340 issues updated since Jan 1. dlt infers the schema (say 22 columns after unnesting `user`, `labels` into a child table), creates tables, loads 340 rows via merge (all inserts). Persists high-water `updated_at = 2026-07-08T22:14Z` to `_dlt_pipeline_state` in DuckDB.
- **Run 2 (5 minutes later, nothing changed):** cursor reads `2026-07-08T22:14Z`, API returns 0 newer issues → **0 rows loaded**, watermark unchanged. This is your lab's "re-run lands zero new rows" Definition-of-Done, for free.
- **Run 3 (someone edits issue #200's title and 2 new issues appear):** API returns 3 rows. Merge: issue #200 is *updated in place* (not duplicated — same `id`), 2 new issues inserted. New high-water persisted. Total rows in table went from 340 → 342, and #200 reflects the edit.
- **Run 4 (API adds a `reactions` object to the payload):** dlt sees new keys, runs `ALTER TABLE ADD COLUMN reactions__total_count bigint` (+ others), old 342 rows get NULL there, new rows populate. No code change, no manual migration.

Four runs, mutable data, evolving schema, idempotent re-runs — and the resource is ~5 lines. The hand-rolled equivalent in `ingest_api.py` is the whole file plus `state.json` plus `manifest.py`.

**Source B — Postgres `documents` with soft deletes (`deleted` bool).** Two clean options:

1. **`merge` + incremental on `updated_at`**, and model deletes as tombstone rows exactly as lecture 3 taught — the `deleted=true` row is just another update that merges in. Downstream (Week 3 indexer) reads `deleted` and removes vectors. dlt doesn't need to know it's a delete; it's a mutable row.
2. **dlt's built-in hard-delete support** via a `merge` with a delete hint that treats a flag as a delete (and SCD2 for full history). Powerful, but for this lab option 1 keeps the tombstone contract explicit and visible, which is the pedagogical point of Week 1.

---

## How it shows up in production

- **The "full reload" surprise on cost.** Someone renames `dataset_name` in a config refactor, or CI runs against an ephemeral DuckDB. The watermark resets, and Monday's run pulls *all* 4M rows instead of the day's 12k — 300× the API calls and warehouse write cost. **Pin `pipeline_name` + `dataset_name`; treat them as identity, not cosmetic.** Alert on load row-counts that exceed a daily-delta band.
- **Schema evolution is a blessing until it's a governance leak.** Auto-adding columns is great for velocity and terrible when an upstream leak lands an `ssn` column in your warehouse before Presidio ever sees it. On any table feeding a governed corpus, set `schema_contract="freeze"` (or `"discard_row"`/`"discard_value"`) so surprises *halt* instead of silently persisting. This is the same "loud stop beats silent bad data" thesis from lecture 4.
- **Variant columns proliferating** (`amount__v_text`, `amount__v_double`) is dlt telling you the source is type-unstable. It's a *diagnostic*, not just noise — fix it upstream or coerce in the resource, don't let 8 variant columns accrete.
- **State living in the destination** means "just drop the dev database to reset" also nukes your watermark. Fine in dev, a data-loss incident in prod. Know where state lives before you `DROP`.
- **Merge is not free.** Staging + delete + insert costs more than append. On huge append-only event streams, `append` + a downstream dedup view is often cheaper than `merge` per-row. Measure.
- **Debugging:** `pipe.last_trace` and the `load_info` returned by `run()` tell you rows loaded, schema changes, and failures per resource. `dlt pipeline <name> info` (CLI) shows state and schema. `_dlt_loads` in the destination is your audit log — query it.

---

## Common misconceptions & failure modes

- **"dlt replaces Dagster."** No. dlt does *extract-normalize-load* (an EL tool). Dagster *orchestrates* — schedules, retries, lineage, checks. They compose: a Dagster asset calls `pipe.run(...)` and returns the `load_info` as metadata. Lecture 4's `raw_api` asset can literally be a dlt run.
- **"Incremental means it diffs the whole source."** It does not. It filters by the cursor column's value. If a row changes but its `updated_at` is *not* bumped, dlt will never see it — garbage cursor in, missed update out. Your source must reliably touch the cursor column on every change (a DB trigger or `updated_at DEFAULT now()` on update).
- **"Merge deduplicates by content."** No — it upserts by `primary_key`. Two rows with the same key, different content → the later load wins (last-write). If you need content-hash dedup (lecture 2), you still do that, or make the hash part of the key.
- **"Schema evolution handles renames."** No. A renamed column looks like a *new* column plus an old one that stops receiving data. dlt adds the new, keeps the old (now NULLing). Renames need explicit handling.
- **"State is safe because it's in a file."** The local file is a cache; the destination table is truth. Trusting the local file across machines/CI is how you get a surprise full reload.
- **`replace` on the wrong table.** `replace` truncates. Set it on a resource you thought was append-only and you silently lose history every run. Read the disposition on every resource.

---

## Rules of thumb / cheat sheet

- **Default destination for dev/lab:** `duckdb` (zero setup, a file). Move to `postgres` for the lab's real target, `bigquery`/`snowflake`/`filesystem` (Parquet) at scale.
- **Disposition picker:** mutable rows + stable key → `merge`. Immutable events → `append` (+ `incremental`). Small full snapshot → `replace`.
- **Always pair `incremental` with a `primary_key`** so boundary-duplicate dedup works.
- **Choose the cursor column** as a monotonic, always-updated field: `updated_at`, an auto-increment `id`, or an LSN. If updates don't touch it, it's the wrong column.
- **Governed table? `schema_contract="freeze"`.** Exploratory ingest? leave evolution on (the default).
- **Identity = `pipeline_name` + `dataset_name`.** Don't rename them unless you *want* a reset.
- **State lives in the destination** (`_dlt_pipeline_state`). Don't `DROP` prod to "reset."
- **Reach for dlt when:** you have many sources, evolving/nested JSON, standard incremental-load needs, and want to stop maintaining plumbing.
- **Stay bespoke when:** you need an immutable raw landing zone with byte-hash reproducibility (this week's lab), exotic dedup/ordering semantics, sub-second custom CDC, or a schema-on-read design where you deliberately *don't* type at ingest. dlt lands typed into a destination; your landing zone stores raw truth. They can coexist: land raw yourself, then dlt from landing → typed tables.

---

## Connect to the lab

The Week 1 lab has you hand-roll `ingest_api.py` (paginated API + `state.json` watermark + xxhash dedup) and `ingest_cdc.py` (Postgres watermark CDC + tombstones) — *deliberately*, so you own the primitives. After it's green, do the 30-minute exercise: rewrite `ingest_api` as a `@dlt.resource(write_disposition="merge", primary_key="id")` with `dlt.sources.incremental("updated_at")` into DuckDB, and diff the line count and the re-run behavior against your bespoke version. Then wrap that `pipe.run(...)` inside a Dagster asset (lecture 4) so dlt does EL and Dagster does orchestration + the blocking quality checks. That comparison — bespoke vs. dlt, side by side — is the whole point of putting this lecture *after* you've built it by hand.

---

## Going deeper (optional)

- **dlt official docs** — `dlthub.com/docs` (start with "Getting started", then "Incremental loading", "Write dispositions", "Schema evolution", "State"). The canonical reference; current and example-rich.
- **dlt GitHub** — `github.com/dlt-hub/dlt` (read the `dlt/sources/` verified-sources for real incremental patterns; issues discuss merge/SCD2 edge cases).
- **dlt verified sources repo** — search: *dlt-hub verified-sources github* for production-shaped SQL-database, REST-API, and filesystem sources you can copy.
- **REST API source helper** — search: *dlt rest_api source declarative* for the config-driven paginated-API pattern (pagination + incremental without writing the loop).
- **Concept background:** re-read lecture 2 (idempotency/watermarks) and lecture 3 (CDC/tombstones) — dlt is those ideas as library defaults. For the warehouse side, search: *dbt vs dlt EL vs T* to place dlt (EL) against dbt (T) in the modern ELT stack.
- **Comparisons:** search: *Airbyte vs dlt Python* and *Singer taps vs dlt* to understand where a code-first Python library beats a connector platform and vice versa.

---

## Check yourself

1. You declare `dlt.sources.incremental("updated_at")` on a resource whose source rows sometimes change *without* their `updated_at` being bumped. What goes wrong, and whose responsibility is it?
2. Explain why `write_disposition="merge"` makes re-running a load idempotent, and what it dedups *by*. How does that differ from the content-hash dedup in lecture 2?
3. Where does dlt persist the watermark, in the two places it keeps state? Why does the destination copy matter, and what breaks if you rename `dataset_name`?
4. The upstream API adds a new nested `reactions` object and, a week later, starts sending a column that was `bigint` as a string. What does dlt do in each case by default, and how would you make the first case *halt* instead?
5. Give one concrete situation where the Week-1 hand-rolled ingestion is the right call *over* dlt, and one where dlt clearly wins.
6. How do dlt and Dagster divide responsibility in a single pipeline? What does each own?

### Answer key

1. dlt filters by the cursor column's value only — it never diffs full content. A change that doesn't bump `updated_at` is invisible to the incremental cursor, so the update is silently missed. It's the *source's* responsibility to touch the cursor column on every change (e.g. `updated_at DEFAULT now()` on update, or a trigger). Pick a cursor that is genuinely monotonic per change.

2. `merge` loads incoming rows to staging, then deletes matching `primary_key`s and inserts (or upserts) inside a transaction — so applying the same row twice overwrites rather than duplicates: idempotent. It dedups **by primary key** (last-write-wins on that key), *not* by content. Lecture 2's dedup is by a stable **content hash** — two rows with the same key but different content are a *conflict* under merge (later wins) but would be two distinct items under content-hash dedup. Choose based on whether "same key" or "same bytes" defines identity.

3. dlt keeps a local JSON cache under `~/.dlt/pipelines/<name>/` and the authoritative copy in a `_dlt_pipeline_state` table **in the destination**, committed with the data. The destination copy matters because state travels with the data — a fresh machine/CI runner restores the watermark from the destination, not the laptop. `pipeline_name` + `dataset_name` are the identity of that progress; renaming `dataset_name` points at a fresh (empty) state → the next run does a full reload.

4. New nested `reactions` object → dlt infers the new (unnested) columns and runs `ALTER TABLE ADD COLUMN`, leaving old rows NULL there — automatic, non-breaking. The `bigint`-then-`string` conflict → dlt creates a **variant column** (`<col>__v_text`) so both typed values coexist rather than failing or corrupting. To make the new-column case halt instead, set `schema_contract="freeze"` (or `discard_row`/`discard_value`) on the resource/source so an unexpected column raises — appropriate for governed tables.

5. Bespoke wins for this week's lab: an **immutable landing zone with byte-identical, hash-reproducible re-runs** (schema-on-read raw truth), where you deliberately don't type at ingest and need exact control of dedup/ordering/tombstone semantics — dlt lands *typed into a destination*, a different layer. dlt wins when you have **many sources with evolving, nested JSON and standard incremental needs** and don't want to maintain watermark/schema/state plumbing per source. They also compose: land raw yourself, then dlt from landing → typed tables.

6. dlt owns **EL**: extract (paginate/query), normalize (schema inference/evolution, unnesting), load (write disposition, incremental cursor, state). Dagster owns **orchestration**: scheduling, retries, dependency/lineage graph, and blocking asset checks (freshness, duplicate keys, PII). In practice a Dagster `@asset` calls `pipe.run(...)` and surfaces its `load_info` as materialization metadata; the quality gates from lecture 4 sit on the asset, not inside dlt.
