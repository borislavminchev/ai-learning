# Lecture 11: Columnar Storage and DuckDB — Parquet, Predicate Pushdown, and the Small-Files Problem

> You have a curated corpus. Every hour someone asks "how many docs per source?", "what's the average text length?", "how many are flagged `needs_review`?" — and every hour your JSONL scan reads every byte of every record to answer a question that touches one field. This lecture exists because the *storage layout* of your corpus is a first-order performance decision, not a plumbing detail. It is the difference between an analytical query that returns in 40 milliseconds and one that takes 40 seconds while reading 30× more disk. After this you will be able to reason from first principles about *why* columnar Parquet queried with DuckDB is fast for analytical scans, decide when to keep row-oriented JSONL instead, lay out Parquet with sane row groups and partitions, integrate DuckDB with Polars/Arrow at zero copy, and — most importantly — recognize and fix the small-files problem before it silently strangles your pipeline.

**Prerequisites:** SQL basics, Python 3.11+, you've produced cleaned JSONL records in Week 2 (`{doc_id, source, page, text, needs_review, ...}`), basic Polars/Pandas · **Reading time:** ~30 min · **Part of:** Phase 5 (Data Engineering) Week 3

---

## The core idea (plain language)

A dataset on disk can be laid out in one of two fundamental orders, and the choice governs everything downstream.

**Row-oriented (JSONL, CSV, most OLTP databases):** all the fields of record 1, then all the fields of record 2, then record 3. To read one record you touch one contiguous stretch of bytes. This is perfect when you *write* one record at a time (append a line) or *fetch* one whole record at a time (give me document `d_8842` with all its fields). It is what your Week 1 landing zone and Week 2 clean stage naturally produce: you stream records in, you append lines out.

**Column-oriented (Parquet, ORC, most analytical databases):** all the `source` values together, then all the `text` values together, then all the `needs_review` values together. To read one *record* you now have to gather one value from each column region — awkward. But to compute "average text length" you read *only* the `text_length` column and skip every other byte on disk. Analytical questions almost always touch a few columns across *many* rows, and that is exactly the shape columnar storage is built for.

The engineering claim of this lecture is narrow and practical: **JSONL is the right format for landing and streaming append; Parquet queried by DuckDB is the right format for repeated analytical queries over a curated corpus.** These are not competitors — they are different stages of the same pipeline. You land raw as JSONL because appends are cheap and you never mutate the source of truth. You *materialize* the curated, cleaned corpus as Parquet because you will query it hundreds of times with aggregate questions, and columnar layout makes those queries read 10–100× less data.

**DuckDB** is the engine that makes the Parquet side effortless. It is an *embedded* analytical database — the SQLite of OLAP. There is no server to run, no cluster to provision, no ports to open. It is a library you `import`, it runs in your Python process, and it can point its SQL engine directly at Parquet files on disk (or S3) with `read_parquet(...)`. It reads the Parquet metadata, prunes columns and row groups it doesn't need, and streams results — often over data far larger than RAM. For a laptop-scale curated corpus (millions of rows, a few GB), it is the correct default and it will feel instant.

---

## How it actually works (mechanism, from first principles)

Three mechanisms make columnar analytics fast. Understand these and every performance behavior you see will make sense.

### 1. Column pruning (projection pushdown)

If your query is `SELECT source, COUNT(*) ... GROUP BY source`, only the `source` column is relevant. In Parquet, that column's bytes live in contiguous regions and the file's metadata records exactly where. DuckDB reads only those regions and never touches `text` — which in a text corpus is 99% of the bytes.

Worked arithmetic. Suppose 1,000,000 records with these average serialized sizes:

| column         | avg bytes/row | total     |
|----------------|--------------:|----------:|
| `text`         |         2,000 |  2,000 MB |
| `source`       |            12 |     12 MB |
| `needs_review` |             1 |      1 MB |
| `text_length`  |             4 |      4 MB |
| everything else|            50 |     50 MB |
| **total**      |     **~2,067**| **~2.07 GB** |

- **JSONL** must read all **2.07 GB** to answer *any* question, because fields are interleaved on each line — you cannot skip `text` without parsing past it.
- **Parquet + column pruning** for `GROUP BY source` reads ~**12 MB** (plus small metadata). That is a **~170×** reduction in bytes read, before compression even enters the picture.

This single mechanism is why "rows per source" is milliseconds on Parquet and seconds on JSONL.

### 2. Predicate pushdown + row-group skipping

Parquet files are divided into **row groups** — horizontal slabs of, typically, 100K–1M rows. Within each row group, each column is stored as a **column chunk**, and every column chunk carries **statistics** in the footer metadata: **min, max, null count**, and (optionally) a **bloom filter**.

When your query has a filter — `WHERE source = 'wiki'` or `WHERE text_length > 5000` — DuckDB reads the row-group statistics *first* and asks: *could any row in this row group possibly match?* If a row group's `source` min and max are both `'docs'`, it cannot contain `'wiki'`, so DuckDB skips the entire slab without decompressing a single value. This is **predicate pushdown**: the filter is "pushed down" to the storage layer so it prunes data before it is read, rather than reading everything and filtering in memory.

```
Parquet file
├─ Row group 0  [source: aaa..cnn]   text_length min=12 max=980   ← WHERE text_length>5000? min/max say no → SKIP
├─ Row group 1  [source: cnn..wik]   text_length min=40 max=8100  ← might match → READ
├─ Row group 2  [source: wik..zzz]   text_length min=5 max=120    ← SKIP
└─ footer: schema + per-rowgroup, per-column stats
```

The catch — and this is the practical lever — **skipping only works if related values are physically clustered.** If you `WHERE source='wiki'` but the rows are randomly ordered, every row group's `[min,max]` for `source` spans the whole alphabet and *nothing* gets skipped. Statistics prune well when the data is **sorted or partitioned by the column you filter on**. This is why partitioning (below) is not cosmetic — it is what makes pushdown bite.

### 3. Compression (and why columnar compresses better)

A column holds values of *one type with low entropy relative to its neighbors*: `source` is a handful of repeated strings, `needs_review` is just `true`/`false`, `text_length` is small integers. Columnar layout lets Parquet apply per-column encodings that exploit this:

- **Dictionary encoding** — replace repeated strings (`"wiki"`, `"docs"`) with small integer codes into a per-chunk dictionary. A `source` column of 1M rows with 5 distinct values compresses to ~1M tiny codes + a 5-entry dictionary.
- **Run-length / bit-packing (RLE)** — `needs_review` that is mostly `false` collapses to "false ×98,342, true ×1,658".
- **General codecs** on top — **Snappy** (fast, moderate ratio; a good default) or **Zstd** (better ratio, still fast; the modern pick for cold/archival data).

Row-oriented JSON cannot do any of this well because adjacent bytes are heterogeneous (a string next to an int next to a bool), and everything is stored as verbose UTF-8 text with repeated keys on every line. Real-world result (approximate, corpus-dependent): a JSONL corpus often shrinks **3–8×** as Zstd Parquet. Less disk read = less I/O = faster scans, compounding with pruning.

### Putting it together: what a DuckDB scan actually does

For `SELECT source, COUNT(*) FROM read_parquet('corpus/**/*.parquet') WHERE needs_review GROUP BY source`:

1. Read footers of matching files (cheap) → schema + row-group stats.
2. **Column pruning**: plan to read only `source` and `needs_review` column chunks.
3. **Predicate pushdown**: for `needs_review`, skip row groups whose stats say all-false.
4. Decompress only the surviving column chunks.
5. Aggregate in a vectorized, multi-threaded engine (DuckDB processes batches of ~2,048 values per operation, using all your cores).

Steps 2–3 are the reason the query is fast; step 5 is why it *stays* fast as data grows.

---

## Worked example

You finished Week 2 and have `data/clean/*.jsonl` — 1,000,000 records, ~2 GB. Let's materialize it to Parquet and query it.

**Write partitioned Parquet (Polars, using PyArrow underneath):**

```python
import polars as pl

df = pl.read_ndjson("data/clean/*.jsonl")
df = df.with_columns(pl.col("text").str.len_chars().alias("text_length"))

# Partition by source; one directory per source value.
df.write_parquet(
    "data/parquet",
    use_pyarrow=True,
    pyarrow_options={"partition_cols": ["source"], "compression": "zstd"},
)
```

On disk you get **Hive-style partitioning** — the partition key is encoded in the *path*, not stored as a column in each file:

```
data/parquet/
  source=wiki/  part-0.parquet
  source=docs/  part-0.parquet
  source=forum/ part-0.parquet
```

**Query directly with DuckDB — no import, no server:**

```python
import duckdb

con = duckdb.connect()  # in-memory; no file, no daemon

# Rows per source (touches only the partition path + row counts)
con.sql("""
  SELECT source, COUNT(*) AS n
  FROM read_parquet('data/parquet/**/*.parquet', hive_partitioning=true)
  GROUP BY source ORDER BY n DESC
""").show()

# Avg text length + needs_review counts (reads 2 small columns, skips `text`)
con.sql("""
  SELECT source,
         AVG(text_length)               AS avg_len,
         COUNT(*) FILTER (needs_review)  AS to_review
  FROM read_parquet('data/parquet/**/*.parquet', hive_partitioning=true)
  GROUP BY source
""").show()

# Filter that benefits from partition pruning: only touches source=wiki/
con.sql("""
  SELECT COUNT(*) FROM read_parquet('data/parquet/**/*.parquet', hive_partitioning=true)
  WHERE source = 'wiki' AND text_length > 5000
""").show()
```

Because `source` is a partition, the last query never opens the `docs/` or `forum/` files at all — the filter is satisfied by *directory pruning* before any Parquet is read. The `text_length > 5000` predicate then prunes row groups *within* the `wiki` file.

**Zero-copy handoff to Arrow/Polars.** DuckDB and Polars both speak the Apache **Arrow** in-memory format. Moving a result between them doesn't serialize or copy the buffers — they share the same memory:

```python
# DuckDB result -> Polars, zero copy
res = con.sql("SELECT source, text_length FROM read_parquet('data/parquet/**/*.parquet')")
pldf = pl.from_arrow(res.arrow())      # shares Arrow buffers

# The reverse: query a Polars/Arrow object *as if it were a table*
my_df = pl.read_ndjson("data/clean/*.jsonl")
con.sql("SELECT source, COUNT(*) FROM my_df GROUP BY source").show()  # DuckDB sees the Python var
```

That last trick — DuckDB registering a local DataFrame by name and running SQL over it — is the everyday glue. You stay in Python, drop into SQL exactly where SQL is better (aggregations, joins), and pay no copy cost crossing the boundary.

**Ballpark of what you'll observe** (order-of-magnitude, hardware-dependent — *not* a benchmark): the `GROUP BY source` over 1M rows returns in tens of milliseconds from Parquet on a laptop; the equivalent over 2 GB of JSONL (parse every line, touch every byte) runs in the low single-digit *seconds*. The gap widens as the corpus grows, because JSONL cost scales with total bytes while the Parquet query scales with the *few columns* it reads.

---

## How it shows up in production

- **The "why is my dashboard slow" ticket.** Someone points a BI tool or a nightly report at raw JSONL/CSV in object storage. Every refresh re-parses gigabytes to compute five numbers. Converting to partitioned Parquet cuts scan cost by 1–2 orders of magnitude and the ticket evaporates. This is the single highest-leverage storage change most teams make.
- **Cloud egress and scan billing.** On S3 + Athena/BigQuery you are billed **per byte scanned**. Column pruning + partition pruning directly reduce that bill. A query that scans one partition instead of the whole corpus can be 50× cheaper *per run*, every run, forever. Layout is a cost decision.
- **Schema/type enforcement catches bugs JSONL hides.** JSONL is schemaless: `page` is `7` in one line and `"7"` in the next and the reader shrugs. Parquet writes a **typed schema into the footer**; a type conflict fails at *write* time, not silently at read time three weeks later. This is exactly what feeds Week 3's data-quality contract — a typed store makes "`page` must be a non-negative int" a property of the file, not a hope.
- **Larger-than-memory just works.** DuckDB streams row groups and spills to disk; you can aggregate a 50 GB Parquet dataset on a 16 GB laptop. The JSONL equivalent (`json.loads` every line into RAM) OOMs. Engineers reach for Spark far too early; DuckDB handles single-node "medium data" (up to hundreds of GB) with no cluster.
- **The small-files problem shows up as mysterious slowness after "success."** Your incremental pipeline works — it writes a new Parquet file every micro-batch. Six months later the same query that was 40 ms is 8 seconds and nobody changed the query. You've accumulated 40,000 tiny files. See the next section; this is the failure that bites *precisely because the pipeline looked healthy*.

---

## Common misconceptions & failure modes

**The small-files problem (the headline failure).** Every open of a Parquet file costs fixed overhead: a storage request (a full HTTP round-trip on S3, ~10–50 ms *each*), read + parse the footer, plan the read. That overhead is trivial for one 128 MB file and catastrophic across 40,000 files of 4 KB each. Concretely: 40,000 files × ~20 ms fixed overhead ≈ **800 seconds of pure per-file tax** before a single useful byte is aggregated. Row groups can't do their job either — a file smaller than one row group has degenerate statistics, so pruning stops helping. Symptoms: query latency grows over time with no query change; DuckDB/Spark spends its time in "listing/opening files," not compute; S3 request bills spike.

*Fix — compaction.* Periodically rewrite many small files into few right-sized ones. Aim for files roughly **128 MB–1 GB**, with row groups of ~**128 MB / 100K–1M rows**. In DuckDB, compaction is one statement:

```sql
COPY (SELECT * FROM read_parquet('data/parquet/source=wiki/*.parquet'))
TO 'data/parquet_compacted/source=wiki/part-0.parquet'
(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 200000);
```

Landing appends stay small-file-friendly (append is cheap); the *curated* corpus you query should be compacted on a schedule. This is why the spine says "compact into reasonably sized partitions."

**Over-partitioning is the small-files problem in disguise.** Partitioning by `source` (5 values) is great. Partitioning by `source/date/doc_id` explodes into millions of directories each holding one tiny file — you *created* the small-files problem trying to be tidy. Rule: partition only on **low-cardinality columns you actually filter on**, and keep each partition big enough to hold at least one healthy file (tens of MB+). If partitions would be tiny, don't partition on that column — let row-group statistics do the pruning instead.

**"Parquet is always faster."** No. For point lookups (`WHERE doc_id = 'd_8842'` returning one whole row with all columns), row-oriented storage or an indexed OLTP database wins — columnar has to gather one value from every column region. Parquet is for *analytical scans over many rows, few columns*. For *"append one record right now,"* JSONL/a queue wins. Match layout to access pattern.

**Rewriting Parquet in place / tiny incremental writes.** Parquet files are immutable; you can't update a row inside one. "Change detection then re-embed the delta" (Week 3's indexer) is a *state table* problem, not a "mutate the Parquet" problem. And writing a new file per record recreates small-files. Batch writes; compact on a cadence.

**Forgetting `hive_partitioning=true`.** If your paths encode `source=wiki/` but you don't tell the reader, DuckDB won't recover `source` as a queryable column *and* won't prune by directory. Enable it (recent DuckDB often auto-detects, but be explicit).

---

## Rules of thumb / cheat sheet

- **Landing / streaming append → JSONL.** Curated corpus you query repeatedly → **Parquet + DuckDB.** Point lookups of whole rows → OLTP DB, not Parquet.
- **Target file size: 128 MB–1 GB. Target row group: ~100K–1M rows (~128 MB).** Below ~a few MB per file, you have a small-files problem forming.
- **Partition only on low-cardinality columns you filter on** (`source`, `date`). Never on `doc_id` or anything high-cardinality.
- **Compression:** `zstd` for the curated/cold store (best ratio, fast enough); `snappy` if you want max write/read speed and can spend disk.
- **Sort within partitions by your common filter column** so row-group min/max stats prune effectively.
- **Compact on a schedule** (nightly / after N appends). Many small files is a *when*, not an *if*, for incremental pipelines.
- **DuckDB = embedded, no server.** `duckdb.connect()`; `read_parquet('path/**/*.parquet')`; hand to Polars with `.arrow()` (zero copy); register a DataFrame and query it by name.
- **Reach for Spark only when single-node DuckDB genuinely can't hold the working set** (multi-TB, multi-node). For laptop-to-hundreds-of-GB, DuckDB wins on simplicity.

---

## Connect to the lab

This lecture is the theory behind Week 3 Lab step 1 (`to_parquet.py`) and the first Definition-of-Done gate. You'll convert `data/clean/*.jsonl` into partitioned Parquet and write DuckDB queries for "rows per source", "docs flagged `needs_review`", and "avg text length per doc" — and paste the sub-second timing into your README. The typed Parquet schema you produce here is what Week 3's Great Expectations/Pandera contract (Lecture 13) validates as a *blocking* gate, and the same store feeds the incremental indexer (Lecture 15). Deliberately write it in a way that would create thousands of tiny files, observe the slowdown, then add a compaction step — feeling the small-files problem once is worth more than reading about it ten times.

## Going deeper (optional)

- **DuckDB documentation** — `duckdb.org/docs`; specifically the "Reading Parquet Files", "Partitioning", and "Performance Guide" pages. The official DuckDB blog has excellent posts on Parquet metadata and larger-than-memory processing (search: *DuckDB blog Parquet partition pruning*).
- **Apache Parquet documentation** — `parquet.apache.org`; the "File Format" page explains row groups, column chunks, pages, and footer statistics precisely.
- **Apache Arrow documentation** — `arrow.apache.org`; read the "Columnar Format" spec to understand the zero-copy interchange between DuckDB, Polars, and Pandas.
- **Polars user guide** — `docs.pola.rs`; the "IO / Parquet" and "Interoperability / Arrow" sections.
- **Talk:** "DuckDB — the SQLite for Analytics" (search: *Hannes Mühleisen DuckDB talk*) — the creators' overview of the embedded-OLAP design.
- **Book:** *Designing Data-Intensive Applications* (Kleppmann), Chapter 3 "Storage and Retrieval" — the canonical treatment of row vs. column layout, compression, and why analytical stores are columnar.
- **Search queries:** *Parquet row group size best practice*, *small files problem compaction*, *DuckDB read_parquet hive partitioning*, *Zstd vs Snappy Parquet*.

## Check yourself

1. Your query is `SELECT AVG(text_length) FROM corpus` over a 2 GB corpus. Roughly how many bytes does Parquet read versus JSONL, and *why* is the difference so large?
2. You partition by `source` and filter `WHERE source='wiki'`. Two different mechanisms let the engine skip data — name both and say which happens first.
3. You add `WHERE text_length > 5000` but the query barely speeds up, even though few rows match. What property of the file layout most likely explains this, and how would you fix it?
4. A pipeline that writes one Parquet file per micro-batch has gotten 100× slower over six months with no query changes. Diagnose it and give the fix in one sentence.
5. When is JSONL genuinely the *better* choice than Parquet? Give two distinct situations.
6. What does DuckDB being "embedded" and "Arrow-native" buy you concretely in a Python pipeline, compared to a client/server analytical database?

### Answer key

1. **Parquet:** only the `text_length` column chunks — a few MB (4 bytes/row × 1M ≈ 4 MB, less after compression). **JSONL:** all ~2 GB, because fields are interleaved per line and you must parse past `text` (99% of the bytes) to reach `text_length`. Column pruning is the mechanism; the ratio (~hundreds×) tracks how large the untouched columns are.
2. **Partition (directory) pruning** happens *first* — the engine sees `source=wiki/` in the path and never opens the other partition directories. Then, *within* the wiki files, **row-group statistics (predicate pushdown)** skip any row group whose min/max can't match. Directory pruning is coarse and free; row-group pruning is finer.
3. The data isn't **sorted/clustered by `text_length`**, so every row group's `[min,max]` for that column spans nearly the whole range — no row group can be skipped, and DuckDB reads them all. Fix: sort within each partition by `text_length` (or the column you filter on) when writing, so min/max stats become selective.
4. **Small-files problem** — six months of per-micro-batch writes produced tens of thousands of tiny files, and per-file open/footer/list overhead now dominates while row-group pruning is degenerate. Fix: **compact** the many small files into few 128 MB–1 GB files (e.g., a scheduled `COPY (SELECT * FROM read_parquet(...)) TO ... (ROW_GROUP_SIZE ...)`).
5. (a) **Landing/streaming append** — appending a line is O(1) and mutation-free; rewriting a Parquet file per event is wasteful and causes small-files. (b) **Whole-record point access / streaming one record at a time** — reading all fields of one record is contiguous in JSONL but scattered across column regions in Parquet. (Also acceptable: schemaless/rapidly-evolving raw data where you don't yet want to commit to types.)
6. **Embedded:** no server/cluster/ports — `import duckdb` runs the full OLAP engine in-process, so deployment is a library dependency, not infrastructure. **Arrow-native:** results move to/from Polars/Pandas with **zero copy** (shared Arrow buffers), and DuckDB can run SQL directly over an in-memory DataFrame by name — you get SQL exactly where it's better without serialization cost crossing the boundary.
