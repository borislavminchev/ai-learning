# Phase 3 — Embeddings Infrastructure & Vector Databases

The storage-and-retrieval backbone under every RAG system, semantic search box, recsys, and agent memory. This phase makes you the engineer who can pick an embedding model on evidence, generate vectors at scale without melting your budget, tune an ANN index on the recall/latency/cost triangle instead of by superstition, and ship a filtered, hybrid, multi-tenant retrieval service that stays honest as data drifts. By the end you will have a Pareto lab and a production retrieval API you can point an interviewer at.

Prev: 02-structured-outputs-tools.md · Next: 04-rag.md

## Prerequisites
- Comfortable Python: async/await, type hints, `pytest`, virtualenvs (`uv` preferred).
- Docker + docker-compose installed and working (`docker run hello-world` succeeds).
- You finished Phase 0–2: you can call an LLM/embedding API with retries, validate output with Pydantic, and reason about tokens and cost.
- Basic NumPy (dtype, shape, broadcasting, `np.dot`, L2 norm) and enough SQL to run `SELECT ... ORDER BY` and create an index.
- A laptop with 16 GB+ RAM. No GPU required; every lab has a CPU path. Where a GPU or paid API helps, a free/local alternative is given.

## Time budget
3 weeks × ~10–15 hrs/week (≈ 33–45 hrs total). Roughly 35% theory / 65% hands-on each week. If a week runs long, cut the "stretch" bullets, never the Definition of Done.

## How to use this file
Work top to bottom, one week at a time; do the Lab the same week you read the Theory or it won't stick. Treat every Definition of Done as a gate — if a checkbox is red, you are not done. Keep everything in one git repo (`phase3-embeddings/`) so the milestone project reuses the week-by-week code.

> **📚 Course materials.** This file is the **spine** — a weekly plan and a *recap*. Deep material lives alongside it:
> - **Lectures**: [`lectures/`](lectures/00-index.md) — read for the *why*/mechanism.
> - **Lab guides**: [`labs/`](labs/) — follow to *build*.
> Each week: read the linked lectures first (the Theory here is their recap), then work the linked lab guide.

---

## Week 1 — Embeddings: fundamentals, model choice, and generation at scale

### Objectives
By end of week you can:
- Explain and demonstrate why cosine, dot, and L2 give the same ranking for normalized vectors, and pick the metric a model was trained for.
- Choose an embedding model from MTEB *and* validate it on a held-out slice of your own data, reporting recall@k — not just trusting the leaderboard.
- Generate embeddings for 50k+ texts efficiently using batching, length bucketing, and an on-disk cache keyed on `model:version:normalized_text`.
- Use query/document prefixes correctly for E5/BGE-family models and measure the recall hit when you get them wrong.
- Apply Matryoshka (MRL) truncation and quantify the recall-vs-storage tradeoff at 1024 → 512 → 256 dims.

### Theory (~4 hrs)
> **📖 Deep lectures for this week** (read first): [1 · Embedding geometry & metrics](lectures/01-embedding-geometry-and-metrics.md) · [2 · Choosing an embedding model](lectures/02-choosing-an-embedding-model.md) · [3 · Query/document prefixes](lectures/03-query-document-prefixes.md) · [4 · Matryoshka truncation](lectures/04-matryoshka-truncation.md) · [5 · Embedding generation at scale](lectures/05-embedding-generation-at-scale.md). The bullets below are the recap.
- **Vectors and geometry (why it matters):** an embedding maps text to a point where "similar meaning ≈ close in space." Retrieval is nearest-neighbor search. Read the **sentence-transformers** docs (root: `sbert.net`) sections "Computing Embeddings" and "Semantic Search." Key fact to internalize: if vectors are L2-normalized, cosine similarity, dot product, and (negated) squared-L2 all produce the *same ranking* — so normalization lets you use the fastest metric your index supports. Match the metric to how the model was trained: most sentence-transformers/OpenAI/Cohere models want cosine (normalize + dot); some are trained for raw dot product — check the model card.
- **Model selection axes:** the **MTEB leaderboard** (on Hugging Face Spaces, search "MTEB leaderboard") ranks models, but leaderboards are contaminated and don't reflect *your* domain. Decision axes: retrieval score, **max sequence length** (many top models truncate at 512 tokens — silent recall killer on long chunks), embedding **dimension** (storage/RAM cost is linear in dims), multilingual need, license, and open-weights vs API cost. 2025–26 workhorses to know by name: `BAAI/bge-m3` and `intfloat/multilingual-e5-large` (open, multilingual, prefix-based), `Alibaba-NLP/gte-Qwen2` family (strong, larger), `nomic-embed-text` (open, good on CPU/Ollama), and API options **OpenAI `text-embedding-3-small/large`**, **Cohere `embed-v3`**, **Voyage `voyage-3`**. Opinionated default for a first project: `BAAI/bge-small-en-v1.5` (fast, CPU-friendly, 384-dim) to prototype, then benchmark against `bge-base`/`e5-large` before committing.
- **Query/document asymmetry:** E5 and BGE were trained with instruction prefixes (e.g. `"query: ..."` vs `"passage: ..."`, or a BGE query instruction). Skipping them silently degrades recall by several points. This is the single most common "why is my retrieval bad" bug for beginners.
- **Matryoshka Representation Learning (MRL):** models like `text-embedding-3-*` and `nomic-embed` are trained so the *first N dimensions* are usable on their own. You can truncate 1024→256 for cheaper storage/faster search, then optionally rescore with full dims. Learn the intuition, not the math.
- **Generation at scale:** GPU batching (Hugging Face **TEI** — Text Embeddings Inference — or vLLM), API concurrency with exponential backoff, **length-bucketing** (sort by token length so batches aren't padded to the longest outlier), token-aware batch sizing, and **caching/dedup** (hash normalized text; never re-embed identical inputs). Read the TEI README on GitHub (`huggingface/text-embeddings-inference`).

### Lab (~8 hrs)
> 🛠️ Full step-by-step guide: [Embedding Toolkit and Model-Selection Benchmark](labs/week-1-embedding-toolkit-and-model-benchmark.md). The steps below are the summary.

Build an embedding toolkit and a mini model-selection benchmark.

Repo layout:
```
phase3-embeddings/
  pyproject.toml
  data/
    corpus.jsonl        # {"id","text","meta":{...}}
    eval.jsonl          # {"query","relevant_ids":[...]}
  emb/
    __init__.py
    encode.py           # batching + prefixes + cache
    cache.py            # sqlite/diskcache keyed on model:version:sha256(text)
    metrics.py          # recall@k, mrr, ndcg
  scripts/
    build_corpus.py
    bench_models.py
  tests/
    test_metrics.py
```

Steps:
1. **Env:** `uv init && uv add sentence-transformers numpy scikit-learn diskcache orjson pytest`. (CPU torch is fine.)
2. **Corpus:** grab a real, small dataset. Use the BEIR `scifact` or `nfcorpus` subset via `datasets` (`uv add datasets`), or scrape ~2k paragraphs from Wikipedia. Write `corpus.jsonl` and an `eval.jsonl` of ~50–100 queries with known relevant ids (BEIR ships qrels; use them).
3. **`encode.py`:** function `encode(texts, model_name, kind="doc"|"query", normalize=True, batch_size=64)` that: applies the correct prefix per model family, sorts by length into buckets, batches, and L2-normalizes. Wrap with a `diskcache` layer keyed on `f"{model_name}:{sha256(normalized_text)}"`.
4. **`metrics.py`:** implement `recall_at_k`, `mrr`, `ndcg_at_k` from scratch over id lists. Unit-test against a hand-computed toy example (`test_metrics.py`).
5. **`bench_models.py`:** for each of `["BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2", "sentence-transformers/all-MiniLM-L6-v2"]`: embed corpus + queries (flat cosine search via `numpy` argsort — no ANN yet, this is ground truth), print recall@1/5/10, MRR@10, encode throughput (texts/sec), and dimension. Produce a small markdown table.
6. **Prefix ablation:** rerun E5 *with* and *without* the `query:`/`passage:` prefixes; record the recall delta.
7. **Matryoshka:** with a model that supports MRL (`nomic-embed-text-v1.5` via sentence-transformers, `trust_remote_code`), truncate embeddings to 768/512/256/128 dims (slice + renormalize) and plot recall@10 vs dims.
8. **Cache proof:** run the encode step twice; assert the second run makes zero model calls (log cache hits).

Cost note: everything above runs on CPU for free. If you want to try an API model, `text-embedding-3-small` costs ~$0.02 / 1M tokens — a 2k-doc corpus is well under $0.01; still, gate API calls behind the cache.

### Definition of Done
- [ ] `pytest` green; `test_metrics.py` verifies recall/MRR/nDCG against a hand-computed example.
- [ ] `bench_models.py` prints a table with recall@{1,5,10}, MRR@10, dim, and texts/sec for ≥3 models.
- [ ] Prefix ablation shows a measurable recall delta (E5 with prefixes beats without on your data), and you can state the number.
- [ ] Matryoshka plot exists; you can state the recall lost going 768→256 dims and the storage saved.
- [ ] Re-running encode is a full cache hit (0 model invocations on second run), proven by a log line or counter.
- [ ] You can answer, in one sentence each, why normalization lets you use dot product and which metric your chosen model wants.

### Pitfalls
- Forgetting E5/BGE prefixes — looks like a bad model, is actually a bad harness.
- Comparing cosine scores across *different* models as if they're on the same scale — they aren't; only rankings/eval metrics are comparable.
- Silent truncation: your chunks are 800 tokens, the model caps at 512, the tail is dropped and you never notice. Log token counts.
- Caching on raw text instead of *normalized* text (whitespace/case), so trivially-different inputs miss the cache.
- Benchmarking throughput without length-bucketing, then concluding the model is slow when it's your padding.

### Self-check
1. Your vectors are L2-normalized. Rank-order of results under cosine vs dot vs L2 — same or different? Why?
2. A model's MTEB retrieval score is 5 points higher than your current one, but it caps at 512 tokens and yours does 8192. How do you decide?
3. What exactly goes into your embedding cache key, and why does `version` matter?
4. You truncate MRL embeddings 1024→256 and recall drops 4 points. Name two ways to recover most of it.

---

## Week 2 — ANN indexes: HNSW, IVF/PQ, quantization, and the recall/latency/cost triangle

### Objectives
By end of week you can:
- Explain exact (flat) vs approximate search and why ANN is mandatory past ~100k–1M vectors.
- Build a **flat ground-truth index** and measure **recall@k of an approximate index against it** — the only honest way to talk about ANN quality.
- Tune **HNSW** by moving `M`, `efConstruction`, and `efSearch` and predicting the effect on recall, latency, build time, and RAM before you run it.
- Configure **IVF-PQ** (nlist/nprobe, PQ subquantizers) and explain the memory-vs-recall tradeoff vs HNSW.
- Apply **binary/scalar quantization + rescore** and show recall recovery, and produce a recall-vs-QPS-vs-RAM Pareto plot.

### Theory (~4.5 hrs)
> **📖 Deep lectures for this week** (read first): [6 · Exact vs approximate & honest recall](lectures/06-exact-vs-approximate-and-honest-recall.md) · [7 · HNSW internals & tuning](lectures/07-hnsw-internals-and-tuning.md) · [8 · IVF inverted-file indexes](lectures/08-ivf-inverted-file-indexes.md) · [9 · Quantization: PQ/scalar/binary/rescore](lectures/09-quantization-pq-scalar-binary-rescore.md) · [10 · Recall/latency/cost Pareto](lectures/10-recall-latency-cost-pareto.md). The bullets below are the recap.
- **Exact vs approximate:** flat/brute-force is 100% recall but O(N) per query — fine for ground truth and small N, hopeless at scale. ANN trades a little recall for orders-of-magnitude speedup. **You must always measure recall against a flat index over the same data**; "recall@10" with no ground truth is meaningless.
- **HNSW internals (engineer depth):** a multi-layer navigable small-world graph; search greedily descends layers. The three knobs (read the **FAISS wiki** "Guidelines to choose an index" and the **hnswlib** README on GitHub):
  - `M` — edges per node. Higher = better recall + more RAM (memory ≈ `(dim*4 + M*2*4)` bytes/vector, roughly). Typical 16–64.
  - `efConstruction` — build-time candidate list. Higher = better graph, slower build. Typical 100–400.
  - `efSearch` — query-time candidate list. **The runtime recall/latency dial** — raise it for recall, lower it for speed. Tune this per query without rebuilding.
  - Gotchas: HNSW is memory-resident (RAM is the silent budget killer), and deletes are "soft" — heavy churn needs a rebuild.
- **IVF (inverted file):** cluster vectors into `nlist` cells (Voronoi); at query time probe `nprobe` nearest cells. Higher `nprobe` = more recall, more latency. Cheaper RAM than HNSW at scale, needs a `train()` step.
- **PQ / OPQ (product quantization):** compress each vector into `m` subquantizer codes → huge memory savings, some recall loss; usually paired with IVF (`IVF..,PQ..`). **Scalar quantization** (fp32→int8) and **binary quantization** (1 bit/dim) are cheaper still; **binary + rescore** (search in binary, re-rank top candidates with full vectors) recovers most recall at a fraction of the RAM.
- **The triangle:** recall, latency (QPS), and cost (RAM/$) trade off against each other; there is no free lunch, only a Pareto frontier. Your job is to find the point that meets the SLO cheapest. Skim **ann-benchmarks** (`github.com/erikbern/ann-benchmarks`) to see how the pros present recall-vs-QPS curves — you'll reproduce that shape.

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [Recall/QPS/RAM Pareto Lab](labs/week-2-recall-qps-pareto-lab.md). The steps below are the summary.

Build the **Recall/QPS Pareto lab** (milestone component #1).

Add to the repo:
```
  ann/
    ground_truth.py     # flat index -> exact top-k per query
    build_hnsw.py
    build_ivfpq.py
    quantize.py         # binary + rescore
    sweep.py            # grid over params, records recall/QPS/RAM
    plot.py             # matplotlib Pareto scatter
  results/
    sweep.csv
    pareto.png
```

Steps:
1. `uv add faiss-cpu matplotlib pandas psutil`. (Optionally `hnswlib` to cross-check HNSW.)
2. **Data:** reuse Week-1 embeddings. If you want the "1M vectors" experience cheaply, generate 1M random 384-dim vectors *plus* your real ~10–50k corpus embeddings; run the sweep on the real set for recall and use the 1M synthetic set to feel build time/RAM. (Downloading a real 1M set like GIST/SIFT-1M from ann-benchmarks is a good stretch goal.)
3. **Ground truth:** `IndexFlatIP` (normalized vectors) → exact top-100 for every eval query. Persist to disk. This is your recall denominator.
4. **HNSW sweep:** for `M in {16,32,64}`, `efConstruction in {100,200,400}`, and at query time `efSearch in {16,32,64,128,256}`: record recall@10 (vs ground truth), QPS (median over ≥3 runs), build time, and process RSS (via `psutil`). Write rows to `sweep.csv`.
5. **IVF-PQ sweep:** build `IVF1024,PQ32` (and one `IVF..,Flat` for comparison); sweep `nprobe in {1,4,8,16,32,64}`. Record the same metrics. Note the `train()` call.
6. **Quantization:** implement binary quantization (`> 0` per dim → bit) + Hamming search for top-200, then **rescore** those 200 with full-precision dot product to get final top-10. Compare recall and RAM against fp32 HNSW.
7. **Plot:** `pareto.png` — recall@10 on Y, QPS on X (log), point size/color = RAM. Draw the Pareto frontier. Add IVF-PQ and binary+rescore as separate series.
8. **Write the finding:** 3–5 sentences — which config you'd ship for a "recall@10 ≥ 0.95 at ≥ 500 QPS" SLO and why.

### Definition of Done
- [ ] A flat ground-truth index exists and is used as the recall denominator (recall numbers are vs flat, not vs another approximate index).
- [ ] `sweep.csv` contains ≥ 15 HNSW configs and ≥ 6 IVF-PQ configs with recall@10, QPS, build time, and RAM columns.
- [ ] You can predict, then confirm, the direction of change when you raise `efSearch` (recall ↑, QPS ↓) and `M` (recall ↑, RAM ↑).
- [ ] Binary-quantization + rescore recovers recall to within a stated few points of fp32 while using materially less RAM — numbers in the writeup.
- [ ] `pareto.png` shows a clear recall-vs-QPS frontier with RAM encoded, and you name the config you'd ship for a stated SLO.

### Pitfalls
- Reporting recall against another ANN index instead of a flat one — you're then measuring agreement, not recall.
- Measuring QPS on a cold index / single query — warm up, batch, and take medians; watch out for the first-query JIT/allocation spike.
- Forgetting IVF needs `train()` on representative data; training on too few points wrecks recall.
- Comparing indexes at different `k` or with unnormalized vectors under an inner-product metric.
- Ignoring build time and RAM — a config with great recall/QPS that takes 40 min to build and 40 GB RAM may be un-shippable.

### Self-check
1. Why is "recall@10 = 0.9" meaningless without saying what it's measured against?
2. You need +3 points recall at serve time with no rebuild and no extra RAM. Which knob, and what does it cost you?
3. When would you pick IVF-PQ over HNSW despite HNSW's better recall/latency?
4. Explain binary-quantize-then-rescore in two sentences; why does it recover recall so well?

---

## Week 3 — Vector databases in production: filtering, hybrid search, freshness, multi-tenancy

### Objectives
By end of week you can:
- Choose between pgvector, Qdrant, Weaviate, Milvus, Pinecone, Chroma, and FAISS for a given scenario and justify it.
- Stand up a vector DB in Docker, create a collection with the right metric/index params, and upsert with **metadata**.
- Implement **metadata filtering** correctly and explain the pre- vs post-filter recall pitfall.
- Implement **hybrid search** (dense + BM25/lexical) fused with **Reciprocal Rank Fusion (RRF)**, with score normalization.
- Handle **freshness** (upserts/deletes/tombstones), **multi-tenancy** (per-tenant filters/namespaces), and **re-embedding/drift** (blue-green migration), and ship a traced FastAPI retrieval service.

### Theory (~4.5 hrs)
> **📖 Deep lectures for this week** (read first): [11 · Vector DB landscape & selection](lectures/11-vector-db-landscape-and-selection.md) · [12 · Metadata filtering & the recall trap](lectures/12-metadata-filtering-and-the-recall-trap.md) · [13 · Hybrid search, RRF & reranking](lectures/13-hybrid-search-rrf-and-reranking.md) · [14 · Freshness, lifecycle & blue-green re-embedding](lectures/14-freshness-lifecycle-and-blue-green-reembedding.md) · [15 · Multi-tenancy & isolation](lectures/15-multi-tenancy-and-isolation.md) · [16 · Serving: tracing, caching & fallback](lectures/16-serving-tracing-caching-and-fallback.md). The bullets below are the recap.
- **The landscape and how to select (opinionated):**
  - **FAISS / hnswlib** = *libraries*, not databases: no persistence layer, filtering, or API. Great for the Pareto lab and embedded use.
  - **Chroma / LanceDB** = embedded, dev-friendly, single-node. Good for prototypes and small apps.
  - **pgvector (+ pgvectorscale)** = the default *when you already run Postgres*: transactions, joins, metadata, and vectors in one place; `HNSW` and `IVFFlat` index types. Read the `pgvector` README on GitHub.
  - **Qdrant** = excellent standalone default: Rust, fast, first-class filtered HNSW, payload indexing, easy Docker, good Python client. Strong recommendation for this week's lab.
  - **Weaviate / Milvus (Zilliz)** = feature-rich, scale-out, more ops overhead; Milvus for very large scale/sharding.
  - **Pinecone** = fully managed, has a free tier; least ops, most lock-in.
  - Decision axes: do you already run Postgres? scale (millions vs billions)? managed vs self-host? filtering/hybrid needs? cost model (HNSW RAM dominates)?
- **Metadata filtering & the recall trap:** *post-filtering* (ANN then drop non-matching) can return too few results when the filter is selective; *pre-filtering* is exact but slow; modern DBs do **native filtered HNSW** (Qdrant, Weaviate, pgvector varies) which is fast but has its own recall caveats under very selective filters. Know which mode your DB uses.
- **Hybrid search:** dense vectors miss exact keywords/IDs; **BM25** (lexical) misses semantics. Run both and fuse. **RRF** (`score = Σ 1/(k+rank)`, k≈60) is the robust default because it needs no score normalization; weighted score fusion needs min-max normalization first because dense and BM25 scores live on different scales. (SPLADE/sparse-neural is the advanced option — name-drop, don't build.)
- **Freshness & lifecycle:** upserts, hard vs soft deletes (**tombstones**), compaction, and **blue-green re-embedding** when you change models — build the new index alongside the old, atomically switch, always keep the **raw text** so you can re-embed. Drift monitoring = a **golden set** whose recall you track over time.
- **Multi-tenancy:** shared collection + tenant-id filter (cheap, weakest isolation), per-tenant namespace/collection (Qdrant/Pinecone support this), or cluster-per-tenant (strongest, priciest). GDPR: deleting a tenant must purge vectors *and* caches.
- **Serving concerns:** OpenTelemetry tracing per stage (embed → search → rerank), a semantic query cache, and a **lexical fallback** if the vector DB times out. Optional reranker: **bge-reranker** or **Cohere Rerank** to reorder the top-50 → top-5.

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [Production Retrieval Service (Phase Milestone)](labs/week-3-production-retrieval-service.md). The steps below are the summary.

Build the **production retrieval service** (milestone component #2).

```
  service/
    docker-compose.yml   # qdrant (+ optional postgres/pgvector)
    ingest.py            # chunk -> embed -> upsert with metadata + tenant_id
    hybrid.py            # dense + bm25 + RRF
    rerank.py            # optional cross-encoder rerank
    app.py               # FastAPI /search, async, per-tenant filter, tracing
    reembed.py           # blue-green migration to a new model/collection
    monitor.py           # golden-set recall check
  tests/
    test_hybrid.py
    test_tenant_isolation.py
```

Steps:
1. **Spin up Qdrant:** `docker compose up -d` with the official `qdrant/qdrant` image (port 6333). `uv add qdrant-client rank-bm25 fastapi uvicorn opentelemetry-sdk opentelemetry-instrumentation-fastapi`.
2. **Ingest:** chunk your Week-1 corpus (recursive/structure-aware, ~400 tokens, 15% overlap), embed with your cached encoder, and upsert into a Qdrant collection (cosine, HNSW) with payload `{text, source, tenant_id, created_at}`. Store raw text in the payload — you'll need it for re-embed and rerank.
3. **Filtered search:** implement `search(query, tenant_id, filters)` using Qdrant payload filters so a tenant can never see another tenant's rows. Write `test_tenant_isolation.py` proving tenant A's query never returns tenant B's docs.
4. **Hybrid + RRF:** build a BM25 index (`rank_bm25`) over the same chunks; run dense and BM25 in parallel, fuse with RRF (k=60). `test_hybrid.py`: a keyword-heavy query (exact ID/code) that dense alone misses must be found by hybrid.
5. **Rerank (optional but recommended):** retrieve top-50 hybrid → rerank with `BAAI/bge-reranker-base` (cross-encoder, CPU-ok) → keep top-5. Measure recall@5 before/after on your eval set.
6. **FastAPI service:** async `POST /search` returning JSON `{results:[{id,text,score,source}], timing_ms:{embed,search,rerank}}`. Instrument each stage with OpenTelemetry spans; log per-stage latency. Add a **lexical (BM25-only) fallback** triggered if the Qdrant call raises/times out.
7. **Freshness:** add `upsert` and `delete` endpoints; prove a deleted doc disappears from results immediately. Then run `reembed.py`: embed the corpus with a *different* model into a new collection, verify recall on the golden set, and flip an alias/config to the new collection (blue-green) with zero downtime.
8. **Monitor:** `monitor.py` runs the golden-set eval against the live service and prints recall@5 + p50/p95 latency + empty-result rate. (Stretch: push to a Grafana dashboard.)
9. **Load test (stretch):** hit `/search` with `locust` or `hey` at concurrency 8/32 and record p50/p95/p99 and QPS.

Cost note: Qdrant, pgvector, BM25, and bge-reranker are all free and local. Pinecone/Cohere free tiers exist if you want to try managed variants; nothing in the DoD requires paid services.

### Definition of Done
- [ ] `docker compose up` brings up the vector DB; `ingest.py` loads the corpus with per-chunk metadata including `tenant_id`.
- [ ] `test_tenant_isolation.py` proves a tenant's search never returns another tenant's documents (20/20 probe queries clean).
- [ ] Hybrid+RRF beats dense-only on a keyword/ID query in `test_hybrid.py`; recall@5 numbers recorded for dense vs hybrid vs hybrid+rerank.
- [ ] `POST /search` returns valid JSON for 20/20 sample queries with per-stage timings, and the trace shows embed/search/rerank spans.
- [ ] Deleting a doc removes it from results immediately; blue-green `reembed.py` switches to a new model's collection with the golden-set recall verified before the flip.
- [ ] Lexical fallback returns results (not a 500) when the vector DB is stopped mid-request — demonstrated by killing the container.
- [ ] `monitor.py` prints recall@5, p50/p95 latency, and empty-result rate against the live service.

### Pitfalls
- Post-filtering with a selective filter returns 2 results when the user asked for 10 — because ANN found 10 then the filter dropped 8. Use native filtered search or over-fetch.
- Weighted score fusion without normalizing — BM25 scores (0–20+) swamp cosine (0–1). Use RRF, or min-max normalize first.
- Forgetting to store raw text in the payload, so you can't rerank or re-embed without re-fetching from source.
- Treating a soft delete as gone — tombstoned vectors can still match until compaction; verify deletion behavior for your DB.
- Re-embedding in place: mixing old-model and new-model vectors in one collection makes distances meaningless. Always blue-green into a fresh collection.
- Multi-tenancy via a filter you set in application code but forget on one endpoint — a data leak. Enforce it in a single choke-point function and test it.

### Self-check
1. You already run Postgres and have ~2M vectors with heavy metadata joins. pgvector or a dedicated vector DB? Why?
2. Why does RRF need no score normalization while weighted fusion does?
3. A user is deleted for GDPR. List every store you must purge (hint: not just the vector DB).
4. You're switching embedding models. Walk through a zero-downtime migration that never mixes vector spaces.
5. Native filtered HNSW is fast but can lose recall under a highly selective filter. Why, and what's a mitigation?

---

## Phase milestone project — Recall/QPS Pareto lab + production retrieval service

Combine the three weeks into one portfolio-grade repo that proves you can both *choose an index on evidence* and *ship a retrieval service that survives production*.

**Part A — Pareto lab (from Week 2):** over a corpus of ≥ 100k vectors (real embeddings + synthetic to reach scale), build a flat ground truth, sweep HNSW (M × efConstruction × efSearch) and IVF-PQ (nlist × nprobe), add binary-quantization + rescore, and produce a `recall@10 vs QPS vs RAM` Pareto plot with a written recommendation for a stated SLO ("recall@10 ≥ 0.95 at ≥ 500 QPS, minimize RAM").

**Part B — Production retrieval service (from Week 3):** an async FastAPI service backed by Qdrant (or pgvector) with:
- per-tenant access filters enforced in one choke-point (with an isolation test),
- hybrid search (dense + BM25, RRF fusion) plus an optional cross-encoder reranker,
- OpenTelemetry per-stage tracing (embed/search/rerank) and a metrics/monitor script reporting recall@5, p50/p95, empty-result rate, and cost/query,
- upsert/delete with immediate effect and a blue-green re-embedding migration to a new model,
- a lexical fallback on vector-DB timeout,
- a load test (locust/hey) proving the latency SLO at concurrency 8/32.

### Acceptance criteria
- [ ] Pareto plot exists with ≥ 20 configs across ≥ 2 index families, recall measured vs a flat ground truth, and a written config recommendation tied to an SLO.
- [ ] Service returns valid JSON with per-stage timings for 20/20 sample queries; traces show all stages.
- [ ] Tenant isolation test passes; deleting a doc removes it immediately; blue-green migration verified on the golden set before flip.
- [ ] Hybrid+rerank beats dense-only on recall@5, with numbers reported.
- [ ] Lexical fallback proven by killing the vector DB mid-request (no 5xx).
- [ ] Load test report shows p50/p95/p99 and QPS at ≥ 2 concurrency levels and states whether the SLO is met.
- [ ] `README.md` explains model choice, index choice, and every tradeoff — the "why," not just the "what."

### Suggested repo layout
```
phase3-embeddings/
  README.md                 # findings, plots, tradeoff writeups
  pyproject.toml
  data/                     # corpus.jsonl, eval.jsonl, qrels
  emb/                      # Week 1: encode, cache, metrics
  ann/                      # Week 2: ground_truth, sweep, quantize, plot
  service/                  # Week 3: docker-compose, ingest, hybrid, app, reembed, monitor
  results/                  # sweep.csv, pareto.png, loadtest.txt
  tests/                    # metrics, hybrid, tenant_isolation
```

## You are ready to move on when...
- [ ] You can pick an embedding model by validating on your own data with recall@k, and explain the metric/prefix/dimension choices.
- [ ] You can generate embeddings at scale with batching, length-bucketing, and a version-keyed cache/dedup.
- [ ] You can tune HNSW and IVF-PQ and *predict* the recall/latency/RAM effect of each knob before running the sweep.
- [ ] You always measure recall against a flat ground truth and can produce a recall/QPS/RAM Pareto plot.
- [ ] You can stand up a vector DB, implement correct metadata filtering, hybrid search with RRF, and a reranker.
- [ ] You can handle freshness (upsert/delete/tombstone), multi-tenant isolation, and a blue-green re-embedding migration without mixing vector spaces.
- [ ] You have a traced, load-tested retrieval service with a lexical fallback and a golden-set recall monitor — ready to be RAG's retrieval layer in Phase 4.
