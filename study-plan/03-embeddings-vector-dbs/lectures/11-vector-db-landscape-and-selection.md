# Lecture 11: The Vector Database Landscape and How to Select

> By now you can generate embeddings, tune an HNSW or IVF index, and measure recall honestly against a flat ground truth. But a tuned FAISS index on your laptop is not a system — it can't persist a crash, can't filter by `tenant_id`, can't take concurrent HTTP requests, and can't tell you which of two vectors is newer. This lecture is the map from "I have an index" to "I run a retrieval service." It gives you an opinionated, engineering-grounded survey of the 2025–26 vector-storage landscape and, more importantly, a *decision method* — a short list of questions that collapses a scary matrix of a dozen products into one obvious pick. After this you'll be able to tell a library from a database and never confuse the two again, name the right tier for a given scale and ops budget, carry your Week 1–2 metric/index choices straight into a collection config, and defend a specific product choice in an interview with numbers and tradeoffs instead of vibes.

**Prerequisites:** Lectures 1–10 — embeddings and metrics, HNSW/IVF internals and tuning, honest recall measurement; basic SQL (`SELECT ... WHERE ... ORDER BY`, `CREATE INDEX`); Docker basics; big-O intuition · **Reading time:** ~28 min · **Part of:** Phase 3 — Embeddings Infrastructure & Vector Databases, Week 3

---

## The core idea (plain language)

There is one distinction that, once it clicks, organizes the entire landscape: **a library is not a database.**

FAISS and hnswlib are *libraries*. You hand them a NumPy array of vectors, they build an in-memory index, and they answer "here are the k nearest." That's the whole contract. They do not persist across a process restart unless you manually serialize the index to a file and reload it. They do not filter by metadata — there is no metadata; a vector is just an integer id and a float array. They have no network API, no authentication, no concurrency control, no way for two writers to add vectors safely at once, no transactions, no notion of "this row was deleted." They are magnificent at exactly one thing: nearest-neighbor math, as fast as the hardware allows. That is why they are the beating heart *inside* almost every vector database — and why, on their own, they are the wrong tool the moment you need to *serve* anything.

A **vector database** wraps that ANN math in everything a real system needs: durable storage that survives a crash, a query API over the network, metadata you can filter on, concurrent reads and writes, snapshots and backups, and lifecycle operations (upsert, delete, re-index). The ANN engine is maybe 20% of the product; the other 80% is the boring, load-bearing plumbing that keeps your data correct at 2 a.m.

So the first question is never "which vector database?" It's "do I even need a database, or will a library do?" If your vectors fit in RAM, never change, and you're doing offline analysis or an embedded feature in a single process — a library is simpler and faster. The moment you need persistence, filtering, an API, or concurrent writes, you need a database, and then this lecture's decision axes tell you which one.

---

## How it actually works (mechanism, from first principles)

### The library / database boundary, concretely

Here is the same task at both levels, so the gap is undeniable.

**With a library (FAISS):**

```python
import faiss, numpy as np
index = faiss.IndexHNSWFlat(384, 32)   # dim=384, M=32
index.add(vectors)                     # vectors: (N, 384) float32, in RAM
D, I = index.search(query, 10)         # returns distances + integer row ids
# Want to persist? faiss.write_index(index, "idx.faiss") — you manage the file.
# Want to filter by tenant_id? You can't. There is no tenant_id. Only row ids.
# Want two processes writing at once? Undefined behavior. It's a data structure.
```

**With a database (Qdrant):**

```python
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333")   # a service, over the network
client.upsert(collection_name="docs", points=[
    models.PointStruct(id=1, vector=vec,
        payload={"text": "...", "tenant_id": "acme", "created_at": 1720000000})
])
hits = client.query_points(
    collection_name="docs", query=query, limit=10,
    query_filter=models.Filter(must=[
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value="acme"))
    ]),
).points
# Persistent by default. Filtered. Concurrent-safe. Reachable from any process.
```

The FAISS version is fewer lines because it does far less. Every capability the database adds — the URL, the payload, the filter — is a capability you would otherwise have to build and maintain yourself. Teams routinely try to "just wrap FAISS in a Flask app," and they spend the next six months reinventing a worse database: a persistence bug here, a race condition on concurrent writes there, a filtering layer that quietly kills recall (more on that below). Don't. That path is a well-marked graveyard.

### The tiers

Once you've decided you need a database, the products sort into tiers by *operational weight* — how much of your life you spend running them — which correlates loosely with scale ceiling and feature richness.

```
LIBRARIES            EMBEDDED DBs        POSTGRES-BASED      STANDALONE          SCALE-OUT           MANAGED
(no persistence,     (single-node,       (vectors live in    (dedicated vector   (distributed,       (someone else
 no API, no filter)   dev-friendly)       your SQL DB)        service)            sharded)            runs it)
─────────────        ─────────────       ─────────────       ─────────────       ─────────────       ─────────────
FAISS                Chroma              pgvector            Qdrant              Weaviate            Pinecone
hnswlib              LanceDB             (+pgvectorscale)    (lab's choice)      Milvus / Zilliz     (+ managed
                                                                                                      Qdrant/Weaviate)
─────────────────────────────────────────────────────────────────────────────────────────────────────────────►
  least ops                                                                                         least ops YOU do
  most control                                                                                      most lock-in
```

**Embedded databases — Chroma, LanceDB.** These run *inside* your Python process (or as a light local server), persist to disk, and give you metadata and filtering — but they're single-node and tuned for developer ergonomics, not throughput. Chroma is the fastest thing to `pip install` and get a prototype working; LanceDB is columnar, disk-first (built on the Lance format), and handles larger-than-RAM datasets on a single machine surprisingly well. Reach for these for prototypes, notebooks, desktop apps, and small production workloads (say, under a few million vectors with modest QPS). You outgrow them when you need real concurrency, horizontal scale, or robust multi-tenant isolation.

**Postgres-based — pgvector (+ pgvectorscale).** `pgvector` is an extension that adds a `vector` column type and ANN indexes (`HNSW` and `IVFFlat`) to PostgreSQL. This is the correct default *when you already run Postgres*, and that condition matters enormously. It means your vectors live in the same transactional store as your relational data: you can `JOIN` a similarity search against your `users` and `documents` tables, filter with the full expressive power of SQL `WHERE`, get ACID transactions, and back everything up with the same `pg_dump` your ops team already knows. You add zero new infrastructure. `pgvectorscale` (from Timescale) layers on a DiskANN-style index (StreamingDiskANN) and quantization that pushes pgvector's scale ceiling meaningfully higher. The tradeoff: Postgres is a general-purpose database, so at very high vector counts and QPS a purpose-built engine will usually beat it on raw latency and RAM efficiency — but "usually beat it" rarely justifies a whole new system if you're already on Postgres and sitting at a few million vectors.

**Standalone — Qdrant.** This is the lab's choice and an excellent standalone default. Written in Rust (fast, memory-safe, predictable latency), it offers first-class *filtered* HNSW — the filter is applied *during* graph traversal, not naively before or after (this is the difference between correct-and-fast and the recall traps you'll see below) — plus payload indexing so those filters are themselves fast, trivial Docker deployment, snapshots, and a clean, well-documented Python client. It scales from a laptop container to a distributed cluster, supports named vectors and sparse vectors for hybrid search, and has quantization built in. If you don't already run Postgres and you want one dedicated vector service that "just works" from prototype to production, Qdrant is the safe, opinionated pick.

**Scale-out — Weaviate, Milvus / Zilliz.** These are feature-rich, distributed-by-design systems built for the billions-of-vectors regime. Weaviate ships GraphQL, built-in hybrid search, and a module system for calling embedding models. Milvus is the heavyweight for very large scale and sharding — it separates compute from storage, supports many index types, and is what you reach for when a single node genuinely cannot hold your data. Zilliz is managed Milvus. The cost is operational: more moving parts (Milvus historically pulls in etcd, object storage, message queues), more to monitor, more to get wrong. Don't pay this tax until your scale forces it.

**Managed — Pinecone.** Fully managed, serverless, with a free tier; you never touch a server, an index build, or a RAM budget. This is the least *ops-you-do* option and the fastest way for a team without infrastructure appetite to ship. The price is lock-in (proprietary API and format — migrating off means re-ingesting everything) and a cost model that can surprise you as usage grows. Note the market has blurred here: you can also buy *managed Qdrant* (Qdrant Cloud) or *managed Weaviate*, getting low ops without full proprietary lock-in — often the sweet spot.

### Your Week 1–2 choices carry straight into the config

This is the part beginners miss: the metric and index decisions you made in the last two weeks are not academic — they become literal fields in your collection config, and getting them wrong silently wrecks recall.

- **Metric.** If your model is trained for cosine (most sentence-transformers, OpenAI, BGE, E5), you set the collection's distance to `Cosine` — or normalize your vectors and use `Dot`, which is identical in ranking and often faster. Set it to `Euclid` on cosine-trained vectors and your rankings are quietly wrong. In pgvector this is the operator class (`vector_cosine_ops` vs `vector_l2_ops` vs `vector_ip_ops`); in Qdrant it's `Distance.COSINE`. **This choice is fixed at collection-creation time** — you cannot change it without rebuilding.
- **Index type.** HNSW vs IVFFlat (pgvector) or the HNSW knobs `m`/`ef_construct` (Qdrant) are exactly the `M`, `efConstruction`, `efSearch` you tuned in Lecture 7. The Pareto curve you produced in the Week 2 lab *is* the evidence for what to set here.

```sql
-- pgvector: metric and index type are decisions, not defaults
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
SET hnsw.ef_search = 100;   -- the runtime recall/latency dial, same as Lecture 7
```

---

## Worked example

Let's price the axis everyone underestimates: **RAM**, because for HNSW-based systems it usually dominates cost.

You have **10 million** documents, embedded with a 768-dim model (fp32 = 4 bytes/dim), and you plan to serve with HNSW at `M = 32`.

Recall the rough per-vector memory rule from Lecture 7: `dim × 4 bytes` for the vector plus roughly `M × 2 × 4 bytes` for the graph edges.

- Vector data: `768 × 4 = 3,072` bytes ≈ 3.0 KB
- Graph edges: `32 × 2 × 4 = 256` bytes
- Per vector ≈ **3,328 bytes**
- × 10M vectors ≈ **33.3 GB** just for the index in RAM — before payloads, before the OS, before headroom.

That does not fit on a 32 GB box. Your options, all of which you learned to reason about in Week 2:

1. **Scalar-quantize to int8** (1 byte/dim): vectors drop to `768 × 1 = 768` bytes; per-vector ≈ `768 + 256 ≈ 1,024` bytes → ~10 GB. Fits comfortably, with a small recall hit you recover by rescoring the top candidates with full-precision vectors.
2. **Matryoshka-truncate 768 → 256 dims** (Lecture 4): vector data `256 × 4 = 1,024` bytes → per-vector ≈ 1,280 bytes → ~12.8 GB, with a measured recall tradeoff.
3. **Use pgvectorscale / DiskANN or an IVF variant** to keep most vectors on SSD instead of RAM, trading latency for a far cheaper memory footprint.

The lesson: the choice of database is downstream of this arithmetic. A "cheap" fully-managed tier can cost more per month than a single self-hosted box if your vectors are large and numerous — and a self-hosted HNSW deployment that ignores quantization can demand an eye-watering RAM SKU. Do this napkin math *before* you pick a product, because it often makes the choice for you.

### Three scenario recommendations (run the axes end to end)

The decision axes only earn their keep when you apply them to a real situation. Here are three, worked the way you'd defend them in a design review.

**Scenario 1 — 2M vectors, heavy metadata joins, you already run Postgres.**
A B2B analytics product stores documents in PostgreSQL and wants semantic search that joins against `organizations`, `documents.status`, and per-row ACLs. Axes: *Postgres? yes* (this alone nearly decides it) · *scale? 2M — trivial for HNSW in-RAM* (2M × ~3.3 KB ≈ 6.6 GB, fits a commodity box) · *filtering? relational joins on existing tables* · *managed? self-host is fine, ops team already runs PG*. **Pick: pgvector (+ pgvectorscale if you grow).** The winning argument is not latency — it's that `SELECT ... FROM docs JOIN orgs ... WHERE org_id = ? AND status = 'active' ORDER BY embedding <=> $1 LIMIT 10` runs in one transactional store with zero new infrastructure, one backup story, and one on-call rotation. A dedicated engine that's 2× faster here would be the *worse* engineering decision.

**Scenario 2 — Multi-tenant SaaS, thousands of tenants, strong per-tenant isolation and namespaces.**
Axes: *Postgres? no dominant existing store* · *scale? tens of millions across tenants, most queries scoped to one tenant* · *filtering? every query filters `tenant_id`, and it must be enforced, not best-effort* · *managed? depends on ops appetite*. **Pick: Qdrant** (per-tenant collections or payload-indexed `tenant_id` filters via native filtered HNSW) **or Pinecone** (first-class namespaces, zero ops). The deciding axis between them is **managed-vs-lock-in**: choose Pinecone to ship with no infrastructure and accept the proprietary format; choose Qdrant (self-hosted or Qdrant Cloud) for control, lower cost at scale, and no lock-in. Either way, the isolation must live in one choke-point function and be tested — a filter you forget on one endpoint is a data leak.

**Scenario 3 — 800M vectors, growing, a team that can run infrastructure.**
Axes: *scale? approaching a billion — a single node can't hold it* · *filtering + hybrid? yes* · *ops? a dedicated platform team exists*. **Pick: Milvus (or Zilliz for managed Milvus).** This is the regime that justifies the operational tax of a sharded, compute/storage-separated system; Weaviate is the alternative if you want built-in hybrid and GraphQL. Note what *disqualifies* the earlier picks: pgvector and single-node Qdrant would need heavy sharding you'd have to build yourself, and Pinecone's cost model at this volume demands the RAM math from above before you commit.

---

## How it shows up in production

**The filtering recall trap (this bites almost everyone).** You ask for the top 10 results where `tenant_id = "acme"`. A naive system does **post-filtering**: it fetches the top 10 by vector similarity, *then* drops the ones that don't match the filter — and returns 2 results, because 8 of the top 10 belonged to other tenants. Or it does **pre-filtering**: find every "acme" row first, then brute-force among them — exact, but potentially slow if "acme" has millions of rows. Purpose-built engines (Qdrant, Weaviate) do **native filtered HNSW**: the filter is checked *during* graph traversal so you get roughly top-10 matching results at ANN speed. Know which mode your database uses, because under a *highly selective* filter (matching, say, 0.1% of rows) even native filtered search can lose recall or fall back to brute force — the graph gets disconnected when most nodes are filtered out. Mitigations: over-fetch (ask for 100, filter, keep 10), maintain a payload index on the filter field, or partition selective tenants into their own collections.

**Latency's hidden variable is where the data lives.** RAM-resident HNSW (Qdrant, pgvector HNSW) gives you single-digit-millisecond search. Disk-based indexes (pgvectorscale/DiskANN, LanceDB on cold data) trade that for cheaper memory but add SSD-read latency — fine for many apps, fatal for a tight p99 SLO. This is a direct consequence of the recall/latency/cost triangle from Week 2, now expressed as a product choice.

**Lock-in is a migration cost you pay later.** Pinecone's proprietary format means leaving requires re-embedding and re-ingesting your entire corpus — which is exactly why you *always keep the raw text* (a Week 3 rule). Postgres and open-source engines let you `pg_dump` or snapshot and move. Weigh this the way you'd weigh any dependency: how expensive is the exit?

**"We already run Postgres" is worth more than a benchmark.** The strongest argument for pgvector is rarely raw performance — it's that adding a `vector` column adds *zero* new infrastructure, no new on-call rotation, no new backup story, no new failure mode. A dedicated vector DB that's 2× faster but adds a whole new stateful service to operate is frequently the *worse* engineering decision at a few million vectors. Operational simplicity is a feature.

---

## Common misconceptions & failure modes

- **"FAISS is a vector database."** No. It's a library with no persistence, filtering, API, or concurrency. Wrapping it yourself means rebuilding a database badly. Use it inside the Pareto lab and for embedded single-process use — not as your service backend.
- **"I'll pick the fastest one on the benchmark."** Benchmarks (like ann-benchmarks) measure *the ANN engine*, in isolation, on a fixed dataset, with no filtering or concurrent writes. Your production bottleneck is far more often filtering behavior, write throughput, ops burden, or RAM cost — none of which the leaderboard shows.
- **"Managed means cheaper."** Managed means less *ops labor*. At scale, per-vector managed pricing can dwarf a self-hosted box. Do the RAM math first.
- **"I can change the distance metric later."** The metric is fixed at collection creation. Choosing L2 for cosine-trained vectors produces subtly wrong rankings that pass smoke tests and fail users. Match the metric to the model (Lecture 1) *before* you create the collection.
- **"Post-filtering is fine."** Under a selective filter it returns too few results. Use native filtered search or over-fetch, and test with a genuinely selective filter, not one that matches half your data.
- **"One product wins."** There is no universal best. The right answer is a function of *your* Postgres situation, scale, ops appetite, and filtering needs — which is the whole point of the decision axes.

---

## Rules of thumb / cheat sheet

*(All figures approximate — do your own napkin math and benchmark on your data.)*

**The library-vs-database gate (ask first):** Need persistence, filtering, an API, or concurrent writes? → database. Otherwise a library (FAISS/hnswlib) is simpler and faster.

**The five decision axes — answer these in order:**
1. **Do you already run Postgres?** → strong default is **pgvector (+pgvectorscale)**. One less system to run beats a marginal benchmark win.
2. **Scale?** Under ~1M and prototyping → **Chroma/LanceDB**. ~1M–100M, dedicated service → **Qdrant**. Billions/sharding → **Milvus/Weaviate**.
3. **Managed or self-host?** Zero ops, accept lock-in → **Pinecone** (or managed Qdrant/Weaviate for less lock-in). Control and lowest $ → self-host.
4. **Filtering / hybrid needs?** Heavy metadata filtering → native filtered HNSW (**Qdrant**, Weaviate); relational joins → **pgvector**; built-in hybrid → Weaviate/Qdrant.
5. **Cost model?** HNSW RAM usually dominates. Big/numerous vectors → plan quantization or disk-based (pgvectorscale/DiskANN) *before* choosing.

**Opinionated defaults:**
- Prototype tomorrow → **Chroma**.
- Already on Postgres, a few million vectors, heavy joins/filters → **pgvector**.
- Fresh dedicated service, prototype-to-prod, great filtering → **Qdrant**. *(This week's lab.)*
- Billions of vectors, a team to run it → **Milvus**.
- No ops appetite, ship now → **Pinecone**.

**Carry-over from Weeks 1–2 (set these at collection creation, they're not free defaults):** metric = what the model was trained for (usually Cosine, or normalize + Dot); index type + `M`/`ef_construction`/`ef_search` = the Pareto-tuned values from your Week 2 sweep.

---

## Connect to the lab

This week's lab builds the **production retrieval service** on **Qdrant** — the standalone tier's default — precisely because it gives you native filtered HNSW, payload indexing, and one-line Docker deployment, so you spend your time on retrieval quality rather than plumbing. As you write the collection config in `ingest.py`, notice you are literally transcribing your Week 1 metric choice (`Distance.COSINE`) and Week 2 HNSW knobs into the schema. The optional pgvector track in the `docker-compose.yml` exists so you can feel the "already-run-Postgres" scenario firsthand — create the same collection two ways and compare the developer experience.

## Going deeper (optional)

- **pgvector** — the README on GitHub (`github.com/pgvector/pgvector`) is the canonical reference for index types, operator classes, and tuning. Pair with **pgvectorscale** (`github.com/timescale/pgvectorscale`) for the DiskANN/StreamingDiskANN story.
- **Qdrant docs** (root: `qdrant.tech`) — read "Filtering," "Indexing," and "Quantization"; the filtered-HNSW explanation is the clearest in the ecosystem.
- **ann-benchmarks** (`github.com/erikbern/ann-benchmarks`) — how the pros present recall-vs-QPS; remember it measures the *engine*, not the *database*.
- **Milvus** (`milvus.io`) and **Weaviate** (`weaviate.io`) docs — skim the architecture pages to understand what "scale-out ops overhead" concretely means.
- **Pinecone docs** (`docs.pinecone.io`) — read the serverless architecture and pricing pages to reason about the managed cost model.
- Search queries that surface current, honest comparisons: `"pgvector vs qdrant" filtering recall`, `qdrant filtered HNSW selectivity`, `pgvectorscale diskann benchmark`, `vector database RAM cost quantization 2025`.

## Check yourself

1. A teammate says "let's just use FAISS as our vector database behind a Flask endpoint." Give three concrete capabilities they'll have to build themselves, and name the class of bug each one invites.
2. You run PostgreSQL, have ~2M vectors, and most queries `JOIN` document metadata and filter by `WHERE org_id = ? AND status = 'active'`. Which store, and what's the single strongest argument for it that has nothing to do with benchmark latency?
3. You're building multi-tenant SaaS where each customer's data must be isolated and you want per-customer namespaces. Name two products that fit and the axis that pushes you toward one over the other.
4. 50M vectors at 1024 dims, fp32, HNSW `M=32`. Roughly how much RAM does the raw index want, and name two levers to cut it (with the tradeoff each carries).
5. Your top-10 filtered search returns only 3 results even though dozens of matching docs exist. What's the mechanism, and what are two fixes?
6. Why can't you switch a collection's distance metric from L2 to Cosine after ingesting 5M vectors, and what's the practical consequence of having chosen wrong?

### Answer key

1. **Persistence** (serialize/reload the index yourself → data-loss and corruption bugs on crash/restart); **filtering by metadata** (FAISS has only integer ids → you bolt on a side store and reconcile it → the post/pre-filter recall trap and stale-metadata bugs); **concurrency / an API** (no network layer, no safe concurrent writes → race conditions and lost writes under load). Bonus: no transactions, no auth, no backups.
2. **pgvector.** Strongest argument: it adds *zero new infrastructure* — no new stateful service to deploy, monitor, back up, or put on-call. Operational simplicity plus the ability to do the filter/join in one SQL statement in the same transactional store beats a marginally faster dedicated engine at this scale.
3. **Qdrant** (per-tenant collections or payload-indexed `tenant_id` filters, self-hostable) and **Pinecone** (native namespaces, fully managed). The deciding axis is **managed vs self-host / lock-in**: choose Pinecone if you want zero ops and accept lock-in; choose Qdrant if you want control, lower cost at scale, and no proprietary format (managed Qdrant Cloud splits the difference).
4. Per vector ≈ `1024×4 + 32×2×4 = 4096 + 256 = 4,352` bytes; × 50M ≈ **~218 GB** of RAM for the index alone. Levers: **scalar-quantize to int8** (~4× smaller vector data, small recall hit recovered by rescoring) and **Matryoshka-truncate the dimension** (e.g., 1024→512 halves vector bytes, at a measured recall cost); or move to a **disk-based index** (pgvectorscale/DiskANN) trading latency for far less RAM.
5. **Post-filtering**: the engine took the top-10 by similarity *then* applied the filter, dropping the 7 that belonged to other tenants/categories. Fixes: use **native filtered HNSW** (Qdrant/Weaviate) so the filter is applied during traversal, **over-fetch** (retrieve top-100, then filter to 10), and/or add a **payload index** on the filter field so selective filters stay fast.
6. The metric is baked into how the index/graph was built and is **fixed at collection creation** — the stored structure assumes a particular distance, so changing it requires a full rebuild (re-ingest). Practical consequence of choosing wrong (L2 on cosine-trained vectors): rankings are subtly incorrect in ways that often pass smoke tests but degrade real recall, and the only fix is rebuilding the collection with the correct metric.
