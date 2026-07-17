# Week 3 Lab: Production Retrieval Service (Phase Milestone)

> You build a real, async retrieval service — the thing Phase 4's RAG will call. It stands up **Qdrant** in Docker, ingests your Week-1 corpus with metadata and a `tenant_id`, serves **hybrid search** (dense + BM25 fused with RRF) behind a **FastAPI** `/search` endpoint, enforces **per-tenant isolation** in a single choke-point, adds an optional **cross-encoder reranker**, traces each stage with **OpenTelemetry**, survives a vector-DB outage via a **lexical fallback**, supports **upsert/delete** with immediate effect and a **blue-green re-embed** to a new model, and monitors itself against a **golden set**. Why: choosing an index (Week 2) is half the job; the other half is a service that filters, fuses, stays fresh, isolates tenants, and degrades gracefully in production.
>
> Read these lectures first (they are the *why* behind every step): [11 · Vector DB landscape & selection](../lectures/11-vector-db-landscape-and-selection.md) · [12 · Metadata filtering & the recall trap](../lectures/12-metadata-filtering-and-the-recall-trap.md) · [13 · Hybrid search, RRF & reranking](../lectures/13-hybrid-search-rrf-and-reranking.md) · [14 · Freshness, lifecycle & blue-green re-embedding](../lectures/14-freshness-lifecycle-and-blue-green-reembedding.md) · [15 · Multi-tenancy & isolation](../lectures/15-multi-tenancy-and-isolation.md) · [16 · Serving: tracing, caching & fallback](../lectures/16-serving-tracing-caching-and-fallback.md). This is the final week of the phase, so the [Definition of Done](#definition-of-done) below folds in the **phase milestone** acceptance criteria for Part B (the service).

**Est. time:** ~9 hrs (12–14 if you do the load test + Grafana stretch) · **You will need:** Docker + docker-compose, Python 3.11+ with `uv`, and ~4 GB free RAM. **Everything here is free and local**: Qdrant runs in Docker, embeddings run on CPU via `sentence-transformers`, BM25 is `rank-bm25`, the reranker is `BAAI/bge-reranker-base` on CPU. Optional Ollama path noted where a heavier model would otherwise want a GPU. No paid API or managed service is required for any Definition-of-Done item.

---

## Before you start (setup)

You should already have the `phase3-embeddings/` repo from Weeks 1–2, with `emb/encode.py` (cached encoder), `emb/cache.py`, `emb/metrics.py`, and `data/corpus.jsonl` + `data/eval.jsonl`. This lab adds a `service/` package that **reuses your Week-1 cached encoder** — do not re-implement embedding.

```bash
# from the repo root: phase3-embeddings/
cd phase3-embeddings

# service deps (CPU-only torch is fine; sentence-transformers already installed from Week 1)
uv add qdrant-client rank-bm25 fastapi "uvicorn[standard]" httpx pydantic-settings \
       opentelemetry-sdk opentelemetry-api opentelemetry-instrumentation-fastapi \
       opentelemetry-exporter-otlp
uv add --dev pytest pytest-asyncio locust    # locust is for the stretch load test
```

Confirm Docker works, then create the service layout:

```bash
docker run --rm hello-world   # must succeed before continuing

mkdir -p service tests
touch service/__init__.py service/docker-compose.yml service/config.py \
      service/ingest.py service/hybrid.py service/rerank.py service/app.py \
      service/reembed.py service/monitor.py \
      tests/test_hybrid.py tests/test_tenant_isolation.py
```

Target layout for this week (matches the spine):

```
service/
  docker-compose.yml   # qdrant (+ optional postgres/pgvector)
  config.py            # settings: qdrant url, collection names, model ids
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

**Windows / Git-Bash notes (read once):**
- Use **Git-Bash** or **WSL2** for the `\`-continued commands above; in PowerShell replace the trailing `\` with a backtick `` ` `` or put the whole command on one line.
- Docker Desktop must be running with the **WSL2 backend**. Bind mounts under `C:\Users\...` work; paths with spaces must be quoted.
- `uvicorn --reload` file-watching on Windows sometimes misses changes on mounted drives — if reload is flaky, restart manually.
- `localhost:6333` from the host reaches the container's mapped port regardless of OS.

---

## Step-by-step

### Step 1 — Spin up Qdrant in Docker

**What.** A single-node Qdrant with persistent storage and its dashboard.

**Why.** Qdrant is this week's recommended standalone default: Rust-fast, first-class *filtered* HNSW, payload indexing, trivial Docker, clean async Python client (lecture 11). Libraries like FAISS (Week 2) have no persistence/filtering/API — a database does.

**Do it.** `service/docker-compose.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.4      # pin a real, current tag
    ports:
      - "6333:6333"                    # REST + dashboard
      - "6334:6334"                    # gRPC (faster client)
    volumes:
      - qdrant_storage:/qdrant/storage
    healthcheck:
      test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1"]
      interval: 5s
      timeout: 3s
      retries: 20

  # OPTIONAL free/local alternative backend — pgvector, if you'd rather use Postgres.
  # postgres:
  #   image: pgvector/pgvector:pg17
  #   environment:
  #     POSTGRES_PASSWORD: postgres
  #   ports: ["5432:5432"]
  #   volumes: [pg_storage:/var/lib/postgresql/data]

volumes:
  qdrant_storage:
  # pg_storage:
```

```bash
cd service && docker compose up -d && cd ..
```

**Expected result.** `docker compose ps` shows `qdrant` as `healthy`.

**Verify.**
```bash
curl -s http://localhost:6333/healthz          # -> "healthz check passed" / 200
# dashboard in a browser: http://localhost:6333/dashboard
```

**Troubleshoot.** Port 6333 in use → change the left side of the mapping (`"16333:6333"`) and update `config.py`. Container unhealthy → `docker compose logs qdrant`. On Windows, if the volume won't mount, ensure the drive is shared in Docker Desktop settings.

---

### Step 2 — Config + a chunker, then ingest with metadata and `tenant_id`

**What.** Centralize settings, chunk the corpus, embed with your **Week-1 cached encoder**, and upsert into a Qdrant collection carrying a metadata payload.

**Why.** Chunks (~400 tokens, ~15% overlap) are what you actually retrieve; the payload (`text, source, tenant_id, created_at`) drives filtering, rerank, and re-embed. **Store the raw text in the payload** — without it you cannot rerank or re-embed without re-fetching from source (a top-5 pitfall).

**Do it.** `service/config.py`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    collection: str = "docs_v1"            # active collection (blue-green flips this)
    model_name: str = "BAAI/bge-small-en-v1.5"   # 384-dim, CPU-friendly, cosine
    vector_size: int = 384
    top_k: int = 10

settings = Settings()
```

`service/ingest.py` (key pieces — you fill the bodies):

```python
import hashlib, time, orjson
from qdrant_client import QdrantClient, models
from emb.encode import encode          # your Week-1 cached encoder
from service.config import settings

def chunk_text(text: str, target_tokens: int = 400, overlap: float = 0.15) -> list[str]:
    """Recursive/structure-aware split. Simple version: split on paragraphs, then
    greedily pack ~target_tokens words, carrying overlap*target_tokens words forward.
    Return a list of chunk strings. (Approximate tokens with .split() for the lab.)"""
    ...

def deterministic_id(source: str, chunk_idx: int) -> str:
    return hashlib.sha256(f"{source}:{chunk_idx}".encode()).hexdigest()[:32]

def ensure_collection(client: QdrantClient, name: str, size: int) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
    )
    # index the fields you filter on so filtered-HNSW stays fast (lecture 12)
    client.create_payload_index(name, "tenant_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "source",    models.PayloadSchemaType.KEYWORD)

def ingest(corpus_path: str = "data/corpus.jsonl", tenant_field: str = "tenant_id") -> int:
    client = QdrantClient(url=settings.qdrant_url)
    ensure_collection(client, settings.collection, settings.vector_size)
    points = []
    for line in open(corpus_path, "rb"):
        doc = orjson.loads(line)
        tenant = doc.get("meta", {}).get(tenant_field, "public")  # assign if absent
        for i, ch in enumerate(chunk_text(doc["text"])):
            vec = encode([ch], settings.model_name, kind="doc")[0]   # normalized
            points.append(models.PointStruct(
                id=deterministic_id(doc["id"], i),
                vector=vec.tolist(),
                payload={"text": ch, "source": doc["id"],
                         "tenant_id": tenant, "created_at": time.time()},
            ))
    # upsert in batches of ~256
    for b in range(0, len(points), 256):
        client.upsert(settings.collection, points[b:b+256], wait=True)
    return len(points)

if __name__ == "__main__":
    print("upserted", ingest(), "chunks")
```

> **If your Week-1 corpus has no tenants:** synthesize two — e.g. hash the doc id into `{"tenant-a","tenant-b"}`, or tag by source. You need ≥2 tenants for the isolation test in Step 3.

**Do it.**
```bash
uv run python -m service.ingest
```

**Expected result.** `upserted N chunks` where N ≥ your doc count.

**Verify.**
```bash
curl -s http://localhost:6333/collections/docs_v1 | python -m json.tool | grep points_count
```

**Troubleshoot.** `Wrong vector dimension` → `vector_size` must equal your model's dim (bge-small = 384). Slow ingest → embedding is the cost; your Week-1 cache means a second run is near-instant. Payload missing `text` → you'll hit a wall at rerank; fix now.

---

### Step 3 — Filtered search + tenant isolation choke-point

**What.** One function that every query path goes through, applying the `tenant_id` payload filter server-side.

**Why.** Multi-tenancy via a filter you set in app code but forget on one endpoint is a data leak (lecture 15, pitfall). Enforce it in a **single choke-point** and test it. Filtering server-side (native filtered HNSW) also avoids the post-filter recall trap where ANN returns 10, the filter drops 8, and the user gets 2 (lecture 12).

**Do it.** In `service/hybrid.py` (dense half, plus the choke-point):

```python
from qdrant_client import QdrantClient, models
from emb.encode import encode
from service.config import settings

_client = QdrantClient(url=settings.qdrant_url)

def _tenant_filter(tenant_id: str, extra: dict | None = None) -> models.Filter:
    must = [models.FieldCondition(key="tenant_id",
                                  match=models.MatchValue(value=tenant_id))]
    for k, v in (extra or {}).items():
        must.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
    return models.Filter(must=must)

def dense_search(query: str, tenant_id: str, limit: int = 50,
                 filters: dict | None = None) -> list[dict]:
    """THE choke-point: no query reaches Qdrant without the tenant filter."""
    qvec = encode([query], settings.model_name, kind="query")[0]
    hits = _client.query_points(
        collection_name=settings.collection,
        query=qvec.tolist(),
        query_filter=_tenant_filter(tenant_id, filters),   # server-side
        limit=limit, with_payload=True,
    ).points
    return [{"id": h.id, "text": h.payload["text"], "source": h.payload["source"],
             "score": h.score} for h in hits]
```

`tests/test_tenant_isolation.py`:

```python
import pytest
from service.hybrid import dense_search

@pytest.mark.parametrize("q", [...20 probe queries...])
def test_tenant_a_never_sees_b(q):
    results = dense_search(q, tenant_id="tenant-a", limit=10)
    # every returned chunk must belong to tenant-a
    # (re-fetch payloads or assert via source->tenant map you built in ingest)
    assert all(r_belongs_to(r, "tenant-a") for r in results)
```

**Expected result.** 20/20 probe queries return only tenant-a docs.

**Verify.** `uv run pytest tests/test_tenant_isolation.py -q` → all pass.

**Troubleshoot.** Cross-tenant leak → you filtered post-hoc in Python instead of passing `query_filter`; move it server-side. Empty results for every query → the `tenant_id` values in the filter don't match what you upserted (case/typo); inspect a point in the dashboard.

---

### Step 4 — Hybrid search: BM25 + RRF fusion

**What.** A BM25 index over the same chunks, run alongside dense, fused with **Reciprocal Rank Fusion**.

**Why.** Dense vectors miss exact keywords/IDs; BM25 misses semantics — fuse both (lecture 13). RRF (`score = Σ 1/(k+rank)`, k≈60) is the robust default because it needs **no score normalization**; weighted score fusion would need min-max first since BM25 (0–20+) swamps cosine (0–1) (pitfall).

**Do it.** Extend `service/hybrid.py`:

```python
from rank_bm25 import BM25Okapi

# Build once at startup, scoped per tenant (or filter results by tenant afterward).
def build_bm25(chunks: list[dict]) -> tuple[BM25Okapi, list[dict]]:
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized), chunks

def bm25_search(bm25, chunks, query: str, tenant_id: str, limit: int = 50) -> list[dict]:
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
    out = [dict(c, score=float(s)) for c, s in ranked if c["tenant_id"] == tenant_id]
    return out[:limit]

def rrf_fuse(dense: list[dict], lexical: list[dict], k: int = 60,
             limit: int = 10) -> list[dict]:
    """Fuse by rank, not score. rank is 0-based position in each list."""
    fused: dict[str, dict] = {}
    for lst in (dense, lexical):
        for rank, item in enumerate(lst):
            e = fused.setdefault(item["id"], {**item, "rrf": 0.0})
            e["rrf"] += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda x: -x["rrf"])[:limit]

def hybrid_search(bm25, chunks, query: str, tenant_id: str, limit: int = 10) -> list[dict]:
    d = dense_search(query, tenant_id, limit=50)
    l = bm25_search(bm25, chunks, query, tenant_id, limit=50)
    return rrf_fuse(d, l, k=60, limit=limit)
```

`tests/test_hybrid.py` — a keyword/ID query dense alone misses:

```python
def test_hybrid_finds_exact_id():
    q = "ERR_4021"   # an exact code/ID living in one chunk
    dense = dense_search(q, "public", limit=10)
    hybrid = hybrid_search(bm25, chunks, q, "public", limit=10)
    assert not any("ERR_4021" in r["text"] for r in dense)   # dense misses it
    assert any("ERR_4021" in r["text"] for r in hybrid)      # hybrid finds it
```

**Expected result.** Hybrid surfaces the exact-keyword chunk that dense-only ranks below the cutoff.

**Verify.** `uv run pytest tests/test_hybrid.py -q`. Then record **recall@5 for dense vs hybrid** on your Week-1 `eval.jsonl` using `emb/metrics.py`.

**Troubleshoot.** Hybrid identical to dense → BM25 results not entering the fusion (empty tokenization, or wrong tenant filter). RRF favoring one side hard → you accidentally summed scores instead of `1/(k+rank)`.

---

### Step 5 — Optional cross-encoder rerank (recommended)

**What.** Retrieve top-50 hybrid, rerank with a cross-encoder, keep top-5.

**Why.** A cross-encoder reads (query, passage) jointly and reorders far more accurately than bi-encoder cosine — the classic retrieve-then-rerank pattern (lecture 13). `bge-reranker-base` runs on CPU.

**Do it.** `service/rerank.py`:

```python
from sentence_transformers import CrossEncoder
_reranker = CrossEncoder("BAAI/bge-reranker-base")   # CPU ok; downloads once

def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    pairs = [(query, c["text"]) for c in candidates]
    scores = _reranker.predict(pairs)               # higher = more relevant
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: -x["rerank_score"])[:top_n]
```

> **Free/local heavier option:** if you want a stronger reranker without a GPU, keep `bge-reranker-base` (it is already CPU-friendly). Avoid `bge-reranker-large` on CPU — it is slow per query. A managed alternative (Cohere Rerank free tier) exists but is **not** required.

**Expected result.** recall@5 after rerank ≥ recall@5 of hybrid alone on your eval set (record all three: dense, hybrid, hybrid+rerank).

**Verify.** Run your eval harness comparing the three pipelines; log the numbers into the README table.

**Troubleshoot.** Rerank slower than the SLO → cut candidates to top-30, or make rerank optional per request (`rerank: bool` flag). Scores all similar → wrong model id or you passed only the query.

---

### Step 6 — FastAPI `/search` with per-stage tracing and a lexical fallback

**What.** An async `POST /search` returning JSON with per-stage timings, OpenTelemetry spans around embed/search/rerank, and a **BM25-only fallback** if Qdrant times out or errors.

**Why.** A production retrieval service must observe itself (where did the 300 ms go?) and degrade rather than 500 when the vector DB blips (lecture 16).

**Do it.** `service/app.py` (skeleton):

```python
import time
from fastapi import FastAPI
from pydantic import BaseModel
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from service.hybrid import dense_search, bm25_search, hybrid_search, build_bm25
from service.rerank import rerank

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
tracer = trace.get_tracer("retrieval")

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
BM25, CHUNKS = build_bm25(load_all_chunks_from_qdrant())   # at startup

class SearchReq(BaseModel):
    query: str
    tenant_id: str
    top_k: int = 5
    use_rerank: bool = True

@app.post("/search")
async def search(req: SearchReq):
    timing, t0 = {}, time.perf_counter()
    try:
        with tracer.start_as_current_span("search"):
            with tracer.start_as_current_span("embed_and_dense"):
                results = hybrid_search(BM25, CHUNKS, req.query, req.tenant_id, limit=50)
            timing["search"] = round((time.perf_counter() - t0) * 1000, 1)
            if req.use_rerank:
                with tracer.start_as_current_span("rerank"):
                    r0 = time.perf_counter()
                    results = rerank(req.query, results, top_n=req.top_k)
                    timing["rerank"] = round((time.perf_counter() - r0) * 1000, 1)
            else:
                results = results[:req.top_k]
    except Exception:            # Qdrant down/timeout -> lexical fallback
        with tracer.start_as_current_span("lexical_fallback"):
            results = bm25_search(BM25, CHUNKS, req.query, req.tenant_id, limit=req.top_k)
            timing["fallback"] = True
    return {"results": [{"id": r["id"], "text": r["text"],
                         "score": r.get("rerank_score", r.get("rrf", r.get("score"))),
                         "source": r["source"]} for r in results],
            "timing_ms": timing}
```

**Do it.**
```bash
uv run uvicorn service.app:app --port 8000
# in another shell:
curl -s -X POST http://localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"how does HNSW efSearch affect recall","tenant_id":"public","top_k":5}' | python -m json.tool
```

**Expected result.** Valid JSON: `results[]` with `id/text/score/source` and a `timing_ms` object. The uvicorn console prints OTel spans (`embed_and_dense`, `rerank`, `search`).

**Verify.** Run all 20 sample queries → 20/20 return valid JSON with per-stage timings. Confirm spans appear (Console exporter). *(Stretch: swap `ConsoleSpanExporter` for the OTLP exporter → Jaeger in Docker.)*

**Troubleshoot.** No spans → you forgot `set_tracer_provider` / `add_span_processor`. `use_rerank` too slow → default it to `False` for the load test. `async def` but sync client → that's fine for the lab; for true concurrency use `AsyncQdrantClient` and `await`.

---

### Step 7 — Freshness: upsert/delete + blue-green re-embed

**What.** `POST /upsert` and `DELETE /doc/{id}` with immediate effect; `reembed.py` that builds a **new** collection with a **different** model and flips to it atomically.

**Why.** Data changes and models change. In-place re-embedding mixes two vector spaces and makes distances meaningless — always **blue-green** into a fresh collection, verify on the golden set, then flip (lecture 14, pitfall).

**Do it.** Add endpoints to `app.py` (upsert reuses `ingest`'s point-building; delete uses `client.delete(...)`). Then `service/reembed.py`:

```python
from qdrant_client import QdrantClient, models
from service.config import settings
from service.ingest import ensure_collection, chunk_text, deterministic_id
from emb.encode import encode
from service.monitor import golden_recall     # Step 8

def reembed(new_collection: str, new_model: str, new_size: int) -> None:
    client = QdrantClient(url=settings.qdrant_url)
    ensure_collection(client, new_collection, new_size)
    # re-embed EVERY point's raw text (that's why we stored it) with the new model
    points, offset = [], None
    while True:
        batch, offset = client.scroll(settings.collection, with_payload=True,
                                      limit=256, offset=offset)
        for p in batch:
            vec = encode([p.payload["text"]], new_model, kind="doc")[0]
            points.append(models.PointStruct(id=p.id, vector=vec.tolist(),
                                             payload=p.payload))
        if offset is None:
            break
    for b in range(0, len(points), 256):
        client.upsert(new_collection, points[b:b+256], wait=True)

    # verify BEFORE flipping
    recall = golden_recall(collection=new_collection, model=new_model)
    assert recall >= 0.9, f"new collection recall {recall} too low; NOT flipping"
    # atomic flip: point an alias at the new collection (zero downtime)
    client.update_collection_aliases(change_aliases_operations=[
        models.CreateAliasOperation(create_alias=models.CreateAlias(
            collection_name=new_collection, alias_name="docs_live"))])
    print("flipped docs_live ->", new_collection, "recall", recall)
```

> Use the **alias** `docs_live` in `config.settings.collection` for reads so the flip is a one-liner. Alternative free/local model to migrate *to*: `intfloat/e5-small-v2` (384-dim, prefix-based) or `nomic-embed-text` via Ollama (`ollama pull nomic-embed-text`) if you want an on-device server path.

**Expected result.** Deleting a doc removes it from `/search` results on the very next query. `reembed.py` builds `docs_v2`, verifies golden recall, and flips the alias with no downtime.

**Verify.**
```bash
# delete then immediately query -> gone
curl -s -X DELETE http://localhost:8000/doc/<id>
curl -s -X POST http://localhost:8000/search -d '{"query":"<that doc>","tenant_id":"public"}' -H 'content-type: application/json'
# blue-green
uv run python -c "from service.reembed import reembed; reembed('docs_v2','intfloat/e5-small-v2',384)"
```

**Troubleshoot.** Deleted doc still appears → soft delete/tombstone not compacted, or you queried the wrong collection; Qdrant `delete` with `wait=True` is immediate. Recall assertion fails → the new model wants different prefixes (E5 needs `query:`/`passage:`); your `encode(kind=...)` must handle that per-model (Week 1).

---

### Step 8 — Monitor against a golden set

**What.** `monitor.py` runs your `eval.jsonl` golden queries against the **live service** and prints recall@5, p50/p95 latency, and empty-result rate.

**Why.** Drift is silent; a golden set whose recall you track over time is your early-warning system (lecture 14/16).

**Do it.** `service/monitor.py`:

```python
import time, statistics, httpx
from emb.metrics import recall_at_k

def golden_recall(collection=None, model=None, url="http://localhost:8000/search",
                  eval_path="data/eval.jsonl", k=5) -> float:
    lats, recalls, empties = [], [], 0
    for row in load_jsonl(eval_path):
        t = time.perf_counter()
        resp = httpx.post(url, json={"query": row["query"], "tenant_id": "public",
                                     "top_k": k}).json()
        lats.append((time.perf_counter() - t) * 1000)
        ids = [r["source"] for r in resp["results"]]
        empties += int(len(ids) == 0)
        recalls.append(recall_at_k(ids, row["relevant_ids"], k))
    p50, p95 = statistics.median(lats), sorted(lats)[int(0.95*len(lats))]
    print(f"recall@{k}={statistics.mean(recalls):.3f}  "
          f"p50={p50:.0f}ms p95={p95:.0f}ms  empty_rate={empties/len(recalls):.2%}")
    return statistics.mean(recalls)

if __name__ == "__main__":
    golden_recall()
```

**Expected result.** A line like `recall@5=0.86  p50=42ms p95=110ms  empty_rate=0.00%`.

**Verify.** `uv run python -m service.monitor` prints all three metrics.

**Troubleshoot.** `recall_at_k` mismatch → your service returns chunk-level `source` but qrels are doc-level; map chunk→doc before scoring. Very high p95 → rerank on; run monitor with `use_rerank=False` to separate retrieval vs rerank latency.

---

### Step 9 — Load test (stretch)

**What.** Hit `/search` at concurrency 8 and 32; record p50/p95/p99 and QPS.

**Why.** The milestone asks you to *prove* the latency SLO under concurrency, not just single-shot.

**Do it.** `locustfile.py`:

```python
from locust import HttpUser, task, between
class U(HttpUser):
    wait_time = between(0, 0.1)
    @task
    def search(self):
        self.client.post("/search", json={"query": "vector database filtering",
                                           "tenant_id": "public", "top_k": 5,
                                           "use_rerank": False})
```

```bash
uv run locust -f locustfile.py --host http://localhost:8000 \
  --headless -u 32 -r 8 -t 60s     # 32 users, spawn 8/s, 60s
# lighter alt: hey -z 30s -c 8 -m POST -d '{...}' http://localhost:8000/search
```

**Expected result.** A report with p50/p95/p99 and QPS at u=8 and u=32; write whether your SLO (e.g. p95 < 150 ms) is met into `results/loadtest.txt`.

**Troubleshoot.** Throughput collapses with rerank on → that's expected (CPU cross-encoder); report both with/without. First request slow → warm the model at startup.

---

## Putting it together — a short end-to-end run

```bash
cd service && docker compose up -d && cd ..    # 1. Qdrant up
uv run python -m service.ingest                # 2. chunk+embed+upsert with tenant_id
uv run pytest tests/ -q                        # 3. isolation + hybrid tests green
uv run uvicorn service.app:app --port 8000 &   # 4. serve (async, traced)
sleep 3
curl -s -X POST http://localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"how does efSearch trade recall for latency","tenant_id":"public","top_k":5}' \
  | python -m json.tool                        # 5. JSON + timing_ms + spans in console
uv run python -m service.monitor               # 6. golden-set recall@5 + p50/p95
# prove graceful degradation:
docker compose -f service/docker-compose.yml stop qdrant
curl -s -X POST http://localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"lexical fallback still works","tenant_id":"public"}'   # 200, not 500
docker compose -f service/docker-compose.yml start qdrant
# blue-green to a new model:
uv run python -c "from service.reembed import reembed; reembed('docs_v2','intfloat/e5-small-v2',384)"
```

You now have: a filtered, tenant-isolated, hybrid+reranked, traced retrieval service that stays fresh via upsert/delete and blue-green re-embed, falls back to lexical on outage, and monitors its own recall. That is Part B of the phase milestone.

---

## Definition of Done

Restating the spine's Week 3 checklist as verifiable checks (this folds in the **phase-milestone Part B** criteria since Week 3 is the final week):

- [ ] `docker compose up` brings up Qdrant (healthy); `ingest.py` loads the corpus with per-chunk metadata **including `tenant_id`** and **raw `text`** in the payload. *(verify: `points_count > 0`)*
- [ ] `test_tenant_isolation.py` proves a tenant's search never returns another tenant's docs — **20/20 probe queries clean**. *(verify: pytest green)*
- [ ] Hybrid+RRF beats dense-only on a keyword/ID query in `test_hybrid.py`; **recall@5 recorded for dense vs hybrid vs hybrid+rerank**. *(verify: numbers in README)*
- [ ] `POST /search` returns valid JSON for **20/20 sample queries** with per-stage `timing_ms`, and the trace shows **embed/search/rerank spans**. *(verify: curl + console spans)*
- [ ] Deleting a doc removes it from results **immediately**; blue-green `reembed.py` switches to a **new model's collection** with **golden-set recall verified before the flip** (alias, zero downtime). *(verify: delete-then-query; reembed asserts recall≥0.9 before flip)*
- [ ] **Lexical fallback returns results (not a 500)** when Qdrant is stopped mid-request. *(verify: `docker compose stop qdrant` then curl → 200)*
- [ ] `monitor.py` prints **recall@5, p50/p95 latency, and empty-result rate** against the live service.
- [ ] *(milestone stretch)* Load-test report shows p50/p95/p99 and QPS at **≥2 concurrency levels** (8/32) and states whether the SLO is met.
- [ ] *(milestone)* `README.md` explains **model choice, index choice, and every tradeoff** — the *why*, not just the *what*.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `Wrong input: Vector dimension error` | model dim ≠ collection `vector_size` | set `vector_size` to the model's dim (bge-small=384); recreate collection |
| Every query returns 0 results | `tenant_id` filter value doesn't match upserted payloads | inspect a point in the dashboard; align case/spelling |
| Cross-tenant leak in isolation test | filtered in Python after search, or missing on one endpoint | pass `query_filter` server-side via the single choke-point |
| Post-filter returns fewer than `top_k` | ANN found k, filter dropped some | use native filtered search (Qdrant does) and/or over-fetch (limit=50) |
| Hybrid == dense | BM25 list empty or wrong tenant | check tokenization; ensure BM25 results enter `rrf_fuse` |
| BM25 swamps dense in fusion | you summed raw scores | use RRF (`1/(k+rank)`), not weighted score sum |
| No OTel spans in console | provider/processor not set | `set_tracer_provider` + `add_span_processor(SimpleSpanProcessor(...))` |
| Deleted doc still appears | wrong collection, or querying stale alias | delete with `wait=True`; confirm read collection/alias |
| Re-embed recall craters | mixed vector spaces / missing E5 prefixes | never re-embed in place; blue-green; honor per-model prefixes |
| Fallback path 500s | exception handler references Qdrant again | fallback must use only in-memory BM25 |
| Docker volume won't mount (Windows) | drive not shared | Docker Desktop → Settings → Resources → File sharing |
| p95 dominated by rerank | CPU cross-encoder is heavy | make rerank optional; cut candidates to 30; report both |

---

## Stretch goals (optional)

- **Jaeger tracing:** add `jaegertracing/all-in-one` to compose, swap `ConsoleSpanExporter` for the OTLP exporter, and view the embed→search→rerank waterfall in the UI.
- **Grafana dashboard:** have `monitor.py` push recall@5 / p95 / empty-rate to Prometheus and chart drift over time.
- **pgvector variant:** stand up the commented `pgvector/pgvector` service and re-implement `dense_search` with an `ORDER BY embedding <=> $1` query + a `WHERE tenant_id = $2` filter; compare filtered-recall behavior to Qdrant (lecture 12).
- **Semantic query cache:** cache `(normalized_query, tenant_id) -> results` with a short TTL; measure hit rate and latency saved (lecture 16).
- **Per-tenant collections:** upgrade from shared-collection-with-filter to per-tenant namespaces/collections and discuss the isolation-vs-cost tradeoff (lecture 15).
- **SPLADE / sparse-neural:** name-drop only — swap `rank-bm25` for a learned sparse retriever and re-fuse; note the ops cost.
- **1M-scale feel:** point the service at the synthetic 1M set from Week 2 and re-run the load test to see filtered-HNSW behavior at scale.
