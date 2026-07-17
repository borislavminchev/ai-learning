# Week 2 Lab: Hybrid + Rerank + Query Transforms, Ablated on the Golden Set

> Week 1 gave you a working *naive* pipeline (parse → chunk → embed → dense search) and, more importantly, a **golden eval set** with numbers you trust. This week you attack the single biggest lever in RAG quality: **what you retrieve before the LLM ever sees it.** You will layer four upgrades onto the Week-1 pipeline — **hybrid search** (dense + BM25 fused with RRF), a **cross-encoder reranker** (top-50 → top-5), **query transformations** (multi-query, HyDE, condense-question), and a **server-enforced ACL pre-filter** — every one of them behind a config flag. Then you **ablate**: run `{baseline, +hybrid, +hybrid+rerank, +all}` on the golden set and *prove* the gains in a Markdown table (recall@5, nDCG@10, p50 latency). No guessing which knob helped — measure it.
>
> Read these lectures first (this guide assumes them):
> - [Lecture 5 — Hybrid Search: Dense + Sparse Fused with RRF](../lectures/05-hybrid-search-dense-sparse-rrf.md)
> - [Lecture 6 — Cross-Encoder Reranking: The Highest-ROI Change](../lectures/06-cross-encoder-reranking.md)
> - [Lecture 7 — Query Transformation: Rewriting the Query Before Retrieval](../lectures/07-query-transformation.md)
> - [Lecture 8 — Metadata Filtering, Self-Query, and Access Control](../lectures/08-metadata-self-query-and-access-control.md)
> - [Lecture 9 — Late Interaction: ColBERT and ColPali (Conceptual)](../lectures/09-late-interaction-colbert-colpali.md) — background for a stretch goal.

**Est. time:** ~9 hrs · **You will need:** Docker (for Qdrant), Python 3.10+, your Week-1 repo + golden set, ~3 GB disk for models. **Everything runs free and local on a CPU laptop** — Qdrant in Docker, `bge-small`/`bge-reranker` via FastEmbed/FlagEmbedding, Ollama `llama3.1:8b` for query transforms. Cohere Rerank and any hosted LLM are *optional* upgrades, never required. pgvector users from Week 1 have a documented Python-RRF path at every step.

---

## Before you start (setup)

**What you carry over from Week 1:** your chunked corpus (each chunk = `{id, text, source, page, heading}`), your `golden.jsonl` eval set, and your embedding model (`BAAI/bge-small-en-v1.5`). If your Week-1 golden set only stored `gold_substr` strings, you need **`relevant_doc_ids`** per query for `ranx`/nDCG this week — see Step 0.5 below to backfill them.

**Folder layout (extending Week 1):**
```
rag/
  ingest.py            # Week 1, now also writes sparse vectors + metadata (this week)
  config.py            # NEW: feature flags HYBRID / RERANK / QUERY_REWRITE / RERANKER
  retrieval/
    dense.py           # Week 1
    hybrid.py          # NEW: dense + BM25 + server-side RRF (+ Python rrf() fallback)
    rerank.py          # NEW: cross-encoder (bge) / Cohere, with latency logging
    query_transform.py # NEW: multi-query, HyDE, condense — flagged + cached
    self_query.py      # NEW: acl_filter(user) built from session, injected everywhere
  eval/
    golden.jsonl       # Week 1 golden set (query, relevant_doc_ids)
    ablation.py        # NEW: runs {baseline,+hybrid,+hybrid+rerank,+all}, prints table
  tests/
    test_acl.py        # NEW: unauthorized user gets 0 restricted chunks
```

**Create the virtualenv add-ons and start Qdrant.**

```bash
# from your Week-1 repo root, venv already active
# Windows Git-Bash: source .venv/Scripts/activate   |  macOS/Linux: source .venv/bin/activate

pip install "qdrant-client[fastembed]" FlagEmbedding ranx rank_bm25 cohere ollama

# Start Qdrant (native sparse vectors + server-side RRF fusion).
# macOS/Linux/Git-Bash:
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

# Windows PowerShell equivalent (if not using Git-Bash):
#   docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Verify Qdrant is up:
```bash
curl -s http://localhost:6333/healthz && echo "  <- qdrant healthy"
# Dashboard in a browser: http://localhost:6333/dashboard
```

**Pull the local LLM for query transforms (Step 3):**
```bash
ollama pull llama3.1:8b        # ~4.7 GB; or qwen2.5:7b if you prefer
ollama list                    # confirm it's there
```

> **First-run downloads:** `bge-reranker-v2-m3` (~2.3 GB) and the FastEmbed `bge-small` + `Qdrant/bm25` models download on first use. Do the first run online; after that it's offline.

> **pgvector path (Week-1 pgvector users):** you do **not** need Docker Qdrant. Keep your pgvector store, skip the Qdrant collection recreate, and use the `rrf()` Python helper given in Step 1. Everything else (rerank, query transforms, ACL, ablation) is identical.

---

## Step-by-step

### Step 0.5 — Backfill `relevant_doc_ids` in the golden set (30 min, do this first)

**What.** Make sure every golden row has the *chunk ids* that are relevant, not just a substring. `ranx` computes recall@5 and nDCG@10 from `{query_id: {doc_id: relevance}}` — it needs ids.

**Why.** Week 1's `gold_substr` check answers "did any top-k chunk contain the answer text?" That's fine for recall-by-substring but nDCG needs graded/positional relevance keyed on stable doc ids. If Week-1 already stored `relevant_doc_ids`, skip this.

**Do it.** One-time script: for each golden query, find which of your chunks contain the `gold_substr` and record their ids.

```python
# eval/backfill_ids.py  — run once
import json, pathlib

CHUNKS = json.loads(pathlib.Path("data/chunks.json").read_text())  # [{id, text, ...}]
rows = [json.loads(l) for l in pathlib.Path("eval/golden.jsonl").read_text().splitlines() if l.strip()]

out = []
for i, r in enumerate(rows):
    if "relevant_doc_ids" in r and r["relevant_doc_ids"]:
        out.append(r); continue
    sub = r["gold_substr"].lower()
    rel = [c["id"] for c in CHUNKS if sub in c["text"].lower()]
    r["query_id"] = r.get("query_id", f"q{i}")
    r["relevant_doc_ids"] = rel
    out.append(r)

pathlib.Path("eval/golden.jsonl").write_text("\n".join(json.dumps(r) for r in out))
print(f"backfilled {sum(1 for r in out if r['relevant_doc_ids'])}/{len(out)} rows with ids")
```

**Expected result.** Every golden row now has `query_id` and a non-empty `relevant_doc_ids` list.

**Verify.**
```bash
python eval/backfill_ids.py
python -c "import json; rows=[json.loads(l) for l in open('eval/golden.jsonl') if l.strip()]; empty=[r['query_id'] for r in rows if not r['relevant_doc_ids']]; print('rows:',len(rows),'empty:',empty)"
```
`empty: []` means clean.

**Troubleshoot.**
- **Many empty lists** → your `gold_substr` doesn't literally appear in any chunk (chunk boundaries split it, or casing/whitespace differs). Hand-fix those rows: open the chunk that *should* be relevant and set `relevant_doc_ids` manually. This is your ground truth; make it honest.
- **Golden set < 30 queries** → deltas will be noisy (a 10-query set gives ±10-pt swings). If you can, grow it to 30-50. Note the caveat in your ship decision.

---

### Step 1 — Hybrid ingest & server-side RRF (2 hrs)

**What.** Recreate the Qdrant collection with **two named vectors per point**: a `dense` vector (`bge-small-en-v1.5`, 384-dim, cosine) and a `bm25` **sparse** vector (FastEmbed's `Qdrant/bm25`). Then write `hybrid_search()` that prefetches top-50 from *each* and fuses them with `FusionQuery(RRF)` **server-side**, accepting an optional `acl_filter`.

**Why.** Dense retrieval blurs exact tokens — SKUs, error codes, function names, acronyms (Lecture 5's `ERR_CONN_TIMEOUT_0x8007274C` example). BM25 nails exact/OOV terms but misses paraphrase. You need both. RRF (`score(d) = Σ 1/(k + rank_i(d))`, `k≈60`) fuses two ranked lists with **incomparable score scales** using ranks only — no calibration. Qdrant does it server-side in one round trip.

**Do it.** First, the ingest — recreate the collection and upsert both vector types.

```python
# ingest.py  (the hybrid ingest section — reuses your Week-1 chunks)
import json, pathlib
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

client       = QdrantClient("localhost", port=6333)
dense_model  = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
COLLECTION   = "docs"

def recreate_collection():
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),  # IDF for BM25
        },
    )

def index_chunks(chunks):
    """chunks: [{'id','text','source','page','heading','acl_groups':[...]}]"""
    texts = [c["text"] for c in chunks]
    dense_vecs  = list(dense_model.embed(texts))          # generators -> lists
    sparse_vecs = list(sparse_model.embed(texts))
    points = []
    for i, c in enumerate(chunks):
        points.append(models.PointStruct(
            id=i,
            vector={
                "dense": dense_vecs[i].tolist(),
                "bm25":  models.SparseVector(**sparse_vecs[i].as_object()),
            },
            payload={
                "text": c["text"], "source": c.get("source"),
                "page": c.get("page"), "heading": c.get("heading"),
                "doc_id": c["id"],
                "acl_groups": c.get("acl_groups", ["public"]),   # for Step 4
            },
        ))
    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)

if __name__ == "__main__":
    chunks = json.loads(pathlib.Path("data/chunks.json").read_text())
    recreate_collection()
    n = index_chunks(chunks)
    print(f"indexed {n} chunks with dense + bm25 sparse vectors")
```

> **Seed a few restricted chunks now** so Step 4's ACL test has something to protect. For ~5-10 chunks set `"acl_groups": ["finance"]` (or any private group) in `data/chunks.json`; leave the rest `["public"]`.

Now the search with server-side RRF:

```python
# retrieval/hybrid.py
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

client       = QdrantClient("localhost", port=6333)
dense_model  = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
COLLECTION   = "docs"

def hybrid_search(query: str, acl_filter=None, top_k: int = 50):
    dv = next(dense_model.query_embed(query))
    sv = next(sparse_model.query_embed(query))
    res = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dv.tolist(), using="dense", limit=top_k),
            models.Prefetch(query=models.SparseVector(**sv.as_object()),
                            using="bm25", limit=top_k),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),  # server-side RRF
        query_filter=acl_filter,       # ACL pre-filter (Step 4); None = no filter
        limit=top_k,
        with_payload=True,
    )
    return res.points

def dense_only(query: str, acl_filter=None, top_k: int = 50):
    """Baseline retriever for the ablation — dense prefetch, no fusion."""
    dv = next(dense_model.query_embed(query))
    res = client.query_points(
        collection_name=COLLECTION, query=dv.tolist(), using="dense",
        query_filter=acl_filter, limit=top_k, with_payload=True,
    )
    return res.points
```

**pgvector fallback (Python RRF).** If you kept pgvector: run dense (pgvector `<=>`) and `rank_bm25` separately, then fuse with this ~10-line helper.

```python
# retrieval/hybrid.py  (pgvector branch)
def rrf(lists, k: int = 60, top_k: int = 50):
    """lists: list of ranked doc_id lists (best first). Returns fused doc_ids."""
    scores = {}
    for lst in lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]

# usage:
#   dense_ids  = pgvector_dense_search(query, limit=50)      # [doc_id,...]
#   sparse_ids = bm25.get_top_n(query.split(), doc_ids, n=50)
#   fused = rrf([dense_ids, sparse_ids], k=60, top_k=50)
```

**Expected result.** `hybrid_search("<a keyword-heavy query from your golden set>")` returns 50 points; an exact-string query (error code, function name) that dense-only *missed* now surfaces the right chunk near the top.

**Verify.**
```bash
python ingest.py     # -> indexed N chunks with dense + bm25 sparse vectors
python -c "
from retrieval.hybrid import hybrid_search, dense_only
q = 'PICK A QUERY WITH AN EXACT TOKEN'   # e.g. a function name / error code
h = [p.payload['doc_id'] for p in hybrid_search(q, top_k=5)]
d = [p.payload['doc_id'] for p in dense_only(q, top_k=5)]
print('hybrid top5:', h); print('dense  top5:', d)
"
```
Confirm the collection has both vector types:
```bash
curl -s http://localhost:6333/collections/docs | python -c "import sys,json; c=json.load(sys.stdin)['result']['config']['params']; print('dense:', list(c['vectors'])); print('sparse:', list(c.get('sparse_vectors',{})))"
# expect  dense: ['dense']   sparse: ['bm25']
```

**Troubleshoot.**
- **`Wrong input: Not existing vector name 'bm25'`** → collection was created without `sparse_vectors_config`. Re-run `recreate_collection()` (it drops + recreates).
- **BM25 scores all zero / sparse half returns nothing** → you forgot `modifier=models.Modifier.IDF` on the sparse vector params, or you embedded with the dense model. Use `SparseTextEmbedding("Qdrant/bm25")`.
- **`recreate_collection` deprecation warning** → newer clients prefer `create_collection` after `delete_collection`; the helper still works. Ignore or switch.
- **Hybrid looks identical to dense** → your golden queries may all be semantic (no exact tokens), so BM25 adds nothing. That's a real finding — note it; the ablation will quantify it.
- **`sv.as_object()` AttributeError** → FastEmbed version drift. Use `models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())`.

---

### Step 2 — Cross-encoder rerank, 50 → 5, with latency logging (2 hrs)

**What.** Feed the top-50 hybrid candidates through `FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)`, which scores each `[query, doc]` pair *jointly*, and keep the top-5. Log **per-query rerank latency** and report **p50/p95**. Provide a Cohere `rerank-v3.5` alternative behind the same interface.

**Why.** Your bi-encoder (embedding model) encoded query and doc *separately* — fast and cacheable but lossy. A cross-encoder reads them *together* and is far more accurate (often +10-20 nDCG, Lecture 6), at the cost of O(n) forward passes and no caching. So you only rerank a small candidate set (50), never the whole corpus. This is usually the highest-ROI single change in a RAG system. But it can only reorder what retrieval returned: if the right chunk isn't in the top-50, no reranker saves you (tune recall@50 first).

**Do it.**

```python
# retrieval/rerank.py
import time, statistics
from FlagEmbedding import FlagReranker

_reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)  # CPU ok; fp16 skipped on CPU
_latencies_ms: list[float] = []

def rerank(query: str, docs, top_n: int = 5):
    """docs: Qdrant points (have .payload['text']). Returns top_n reordered."""
    if not docs:
        return []
    pairs  = [[query, d.payload["text"]] for d in docs]
    t0 = time.perf_counter()
    scores = _reranker.compute_score(pairs, normalize=True)
    _latencies_ms.append((time.perf_counter() - t0) * 1000)
    if not isinstance(scores, list):      # single-pair edge case returns a float
        scores = [scores]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]

def latency_report():
    if not _latencies_ms:
        return {"p50": None, "p95": None, "n": 0}
    s = sorted(_latencies_ms)
    p = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return {"p50": round(p(0.50), 1), "p95": round(p(0.95), 1), "n": len(s)}
```

Cohere alternative (zero infra, free trial key). Same signature so `config.RERANKER` can swap it.

```python
# retrieval/rerank.py  (append)
import os, time
def rerank_cohere(query: str, docs, top_n: int = 5):
    import cohere
    co = cohere.ClientV2(os.environ["COHERE_API_KEY"])
    texts = [d.payload["text"] for d in docs]
    t0 = time.perf_counter()
    r = co.rerank(model="rerank-v3.5", query=query, documents=texts, top_n=top_n)
    _latencies_ms.append((time.perf_counter() - t0) * 1000)
    return [docs[res.index] for res in r.results]
```

**Expected result.** Given 50 hybrid candidates, `rerank()` returns 5 with the most relevant chunk at index 0. On CPU, expect ~1-4 s to score 50 pairs with `bge-reranker-v2-m3`; Cohere is typically ~200-400 ms network-bound.

**Verify.**
```bash
python -c "
from retrieval.hybrid import hybrid_search
from retrieval.rerank import rerank, latency_report
q = 'A QUERY FROM YOUR GOLDEN SET'
cands = hybrid_search(q, top_k=50)
top5  = rerank(q, cands, top_n=5)
print('reranked top5 doc_ids:', [d.payload['doc_id'] for d in top5])
print('rerank latency:', latency_report())   # run a few queries first for real p50/p95
"
```
`latency_report()` returns non-null `p50`/`p95` after ≥1 query. Run it over your whole golden set to get a stable p50/p95 — you'll do exactly that in the ablation.

**Troubleshoot.**
- **First call hangs / downloads** → model download (~2.3 GB). Let it finish once.
- **`use_fp16=True` warning on CPU** → fp16 only helps on GPU; FlagEmbedding falls back to fp32 on CPU. Harmless. Set `use_fp16=False` to silence.
- **`compute_score` returns a bare float** → happens with a single pair; the `isinstance` guard handles it.
- **Reranker doesn't improve ordering** → check recall@50 first. If the gold chunk isn't in the 50 candidates, reranking can't help (Lecture 6's constraint #1). Also confirm you're passing raw chunk `text`, not a truncated snippet.
- **RAM pressure on an 8 GB laptop** → rerank in mini-batches of 16 pairs, or lower `top_k` to 30.

---

### Step 3 — Query transformation behind a flag (2 hrs)

**What.** Three LLM-driven transforms via Ollama `llama3.1:8b`, each toggleable and cached: **multi-query** (generate N paraphrases, retrieve each, union candidates *by doc_id*, then rerank the union), **HyDE** (hallucinate an answer paragraph, embed *that* for retrieval), and **condense-question** (rewrite a chat follow-up into a standalone query). Cap N and cache outputs.

**Why.** The raw user query is often a bad retrieval query (Lecture 7). Multi-query kills phrasing sensitivity; HyDE moves the query from *question-space* into *answer-space* so it matches real docs; condense-question makes conversational RAG stop retrieving on pronouns. But each transform costs an LLM call and multiplies retrievals — multi-query with N=5 is 5× retrievals + 1 LLM call. So: **cap N, cache, and measure** whether the gain beats the latency.

**Do it.**

```python
# retrieval/query_transform.py
import json, functools
import ollama

MODEL = "llama3.1:8b"

@functools.lru_cache(maxsize=512)          # cache identical prompts across a run
def _chat(prompt: str) -> str:
    out = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return out["message"]["content"]

def multi_query(q: str, n: int = 3) -> list[str]:
    n = min(n, 5)                          # cap N — latency guard
    p = (f"Generate {n} diverse alternative search queries for the question below. "
         f"Return ONLY a JSON array of strings, no prose.\nQuestion: {q}")
    try:
        variants = json.loads(_chat(p))
        variants = [v for v in variants if isinstance(v, str)]
    except (json.JSONDecodeError, TypeError):
        variants = []                      # fail safe: fall back to just the original
    return [q] + variants[:n]

def hyde(q: str) -> str:
    """Return a hallucinated answer paragraph; embed THIS instead of q."""
    p = (f"Write a short, factual paragraph (3-4 sentences) that would directly "
         f"answer this question. Do not hedge.\nQuestion: {q}")
    return _chat(p)

def condense(history: str, followup: str) -> str:
    p = (f"Given the chat history, rewrite the follow-up as a standalone search "
         f"query that makes sense without the history.\n"
         f"History:\n{history}\nFollow-up: {followup}\nStandalone query:")
    return _chat(p).strip()
```

Wire the multi-query flow into hybrid + rerank (union by `doc_id`, dedup, then rerank once):

```python
# retrieval/query_transform.py  (append)
from retrieval.hybrid import hybrid_search
from retrieval.rerank import rerank

def multi_query_retrieve(q: str, acl_filter=None, n: int = 3, top_k: int = 50, top_n: int = 5):
    seen, union = set(), []
    for variant in multi_query(q, n=n):
        for p in hybrid_search(variant, acl_filter=acl_filter, top_k=top_k):
            if p.payload["doc_id"] not in seen:
                seen.add(p.payload["doc_id"]); union.append(p)
    return rerank(q, union, top_n=top_n)      # rerank against the ORIGINAL query

def hyde_retrieve(q: str, acl_filter=None, top_k: int = 50, top_n: int = 5):
    # embed the hallucinated paragraph; hybrid_search embeds the string it's given
    cands = hybrid_search(hyde(q), acl_filter=acl_filter, top_k=top_k)
    return rerank(q, cands, top_n=top_n)
```

Config flags tie it together:

```python
# config.py
import os
HYBRID        = os.getenv("HYBRID", "1") == "1"
RERANK        = os.getenv("RERANK", "1") == "1"
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "off")   # off | multi | hyde
RERANKER      = os.getenv("RERANKER", "bge")        # bge | cohere
MULTI_QUERY_N = int(os.getenv("MULTI_QUERY_N", "3"))
```

**Expected result.** `multi_query("how do I set the connection timeout?")` returns the original plus up to 3 paraphrases as a Python list. `hyde(q)` returns a plausible answer paragraph. `condense(history, "what about the pro plan?")` returns something like *"What are the features of the pro plan?"*

**Verify.**
```bash
python -c "
from retrieval.query_transform import multi_query, hyde, condense
print('multi:', multi_query('how do I cancel my subscription?', n=3))
print('hyde :', hyde('how do I cancel my subscription?')[:200], '...')
print('cond :', condense('User: I am on the basic plan.', 'what about the pro plan?'))
"
```
You should see ≥3 distinct query strings and a coherent HyDE paragraph.

**Troubleshoot.**
- **`ConnectionError` / connection refused** → Ollama isn't running. `ollama serve` (or start the app) and confirm `ollama list` shows `llama3.1:8b`.
- **`json.JSONDecodeError` on multi-query** → the model wrapped JSON in prose or markdown fences. The `try/except` already falls back to the original query; to improve yield, add "no markdown, no code fences" to the prompt or strip ```` ``` ```` before parsing.
- **Very slow (>10 s per transform on CPU)** → normal for 8B on CPU. The `lru_cache` avoids repeat cost within a run; for the ablation, cache to disk (`json` keyed on the prompt) so re-runs are instant. Or use `qwen2.5:7b` / a smaller model.
- **HyDE hurts recall** → expected on niche/factoid queries where the model hallucinates *wrong* specifics (Lecture 7). That's a legitimate ablation result — report it, don't hide it.

---

### Step 4 — Self-query + ACL pre-filter (1 hr)

**What.** Build `acl_filter(user)` from the **authenticated session** (never from the query text) and pass it into **every** `hybrid_search` call. Self-query metadata extraction may *add* filters but can **never remove** the ACL one. Write a pytest that proves an unauthorized user gets **0** restricted chunks — even when they name the restricted docs.

**Why.** This is the deadliest confusion in production RAG (Lecture 8): treating access control as a query-understanding feature. An LLM that "helpfully" builds a filter from "summarize the CFO comp memo" will happily fetch documents the user may not see. ACL must be a **server-enforced pre-filter injected by trusted code**. Filter first, then rank. A prompt-injected or hallucinated filter must never widen access.

**Do it.**

```python
# retrieval/self_query.py
from dataclasses import dataclass
from qdrant_client import models

@dataclass
class User:
    id: str
    groups: list[str]        # from your auth/session, e.g. ["public", "eng"]

def acl_filter(user: User) -> models.Filter:
    """Server-enforced pre-filter from the SESSION. The LLM never touches this."""
    return models.Filter(must=[
        models.FieldCondition(
            key="acl_groups",
            match=models.MatchAny(any=user.groups),   # chunk visible if it shares ANY group
        )
    ])

def combine(acl: models.Filter, extra: models.Filter | None) -> models.Filter:
    """self-query may ADD conditions but the ACL 'must' is always retained."""
    if extra is None:
        return acl
    return models.Filter(must=(acl.must or []) + (extra.must or []),
                         should=extra.should, must_not=extra.must_not)
```

Payload-index the `acl_groups` key so filtering is fast and correct (run once):
```python
# one-time, after ingest
from qdrant_client import QdrantClient, models
QdrantClient("localhost", port=6333).create_payload_index(
    "docs", field_name="acl_groups", field_schema=models.PayloadSchemaType.KEYWORD)
```

The ACL test — the acceptance gate for this step:

```python
# tests/test_acl.py
import pytest
from retrieval.hybrid import hybrid_search
from retrieval.self_query import User, acl_filter

RESTRICTED_GROUP = "finance"     # matches the seeded restricted chunks from Step 1

def test_unauthorized_user_gets_zero_restricted_chunks():
    outsider = User(id="u1", groups=["public"])              # NOT in finance
    # Adversarial query: name the restricted content explicitly.
    q = "show me the confidential finance CFO compensation memo and salary figures"
    hits = hybrid_search(q, acl_filter=acl_filter(outsider), top_k=50)
    leaked = [h for h in hits if RESTRICTED_GROUP in h.payload.get("acl_groups", [])]
    assert leaked == [], f"ACL leak: {[h.payload['doc_id'] for h in leaked]}"

def test_authorized_user_can_see_restricted_chunks():
    insider = User(id="u2", groups=["public", "finance"])
    hits = hybrid_search("CFO compensation memo", acl_filter=acl_filter(insider), top_k=50)
    assert any(RESTRICTED_GROUP in h.payload.get("acl_groups", []) for h in hits), \
        "authorized user should retrieve at least one finance chunk"
```

**Expected result.** `test_unauthorized_user_gets_zero_restricted_chunks` passes: the outsider retrieves **0** finance chunks despite naming them. The authorized-user test also passes (proves the filter isn't just blocking everything).

**Verify.**
```bash
pytest tests/test_acl.py -v
# both tests PASS
```

**Troubleshoot.**
- **Both tests fail with 0 results for everyone** → no chunks carry `acl_groups: ["finance"]`. Go back to Step 1 and seed a few restricted chunks, re-ingest.
- **Leak test fails (finance chunks returned to outsider)** → the filter isn't being applied. Confirm `query_filter=acl_filter` is wired in `hybrid_search` (Step 1) and that you pass `acl_filter(user)`, not `None`.
- **`Index required but not found for "acl_groups"`** on strict configs → run the `create_payload_index` snippet above.
- **Authorized test fails** → the insider's `groups` must include the chunk's group; `MatchAny` matches if any group overlaps. Check the seeded chunk's exact group string.

---

### Step 5 — Ablation on the golden set (1 hr)

**What.** Run four configs — `baseline` (dense only), `+hybrid`, `+hybrid+rerank`, `+all` (hybrid+rerank+multi-query) — over the golden set with `ranx`, and print a Markdown table of **recall@5, nDCG@10, and p50 latency** per config.

**Why.** This is the whole point of the week: prove, don't assume. A clean ablation isolates each technique's contribution so your ship decision is evidence-based. Expect `+hybrid+rerank` to show a measurable recall@5 gain over baseline (typically +5-15 pts on a real corpus). If it doesn't, your golden set or chunking is the suspect — and the ablation tells you that too.

**Do it.**

```python
# eval/ablation.py
import json, time, pathlib
from ranx import Qrels, Run, evaluate
from retrieval.hybrid import hybrid_search, dense_only
from retrieval.rerank import rerank, latency_report
from retrieval.query_transform import multi_query_retrieve

GOLDEN = [json.loads(l) for l in pathlib.Path("eval/golden.jsonl").read_text().splitlines() if l.strip()]

def _points_to_scored(points, top_n):
    """Convert ranked points -> {doc_id: score} using reverse-rank as score (RRF-style)."""
    return {str(p.payload["doc_id"]): 1.0 / (i + 1) for i, p in enumerate(points[:top_n])}

CONFIGS = {
    "baseline":            lambda q: _points_to_scored(dense_only(q, top_k=50), 5),
    "+hybrid":             lambda q: _points_to_scored(hybrid_search(q, top_k=50), 5),
    "+hybrid+rerank":      lambda q: _points_to_scored(rerank(q, hybrid_search(q, top_k=50), top_n=5), 5),
    "+all":                lambda q: _points_to_scored(multi_query_retrieve(q, n=3, top_k=50, top_n=5), 5),
}

def build_qrels():
    d = {r["query_id"]: {str(i): 1 for i in r["relevant_doc_ids"]} for r in GOLDEN if r["relevant_doc_ids"]}
    return Qrels(d)

def run_config(fn):
    run, lat = {}, []
    for r in GOLDEN:
        if not r["relevant_doc_ids"]:
            continue
        t0 = time.perf_counter()
        run[r["query_id"]] = fn(r["query"])
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    p50 = lat[len(lat) // 2] if lat else 0.0
    return Run(run), p50

def main():
    qrels = build_qrels()
    print("| config | recall@5 | nDCG@10 | p50 latency (ms) |")
    print("|---|---|---|---|")
    for name, fn in CONFIGS.items():
        run, p50 = run_config(fn)
        m = evaluate(qrels, run, ["recall@5", "ndcg@10"])
        print(f"| {name} | {m['recall@5']:.3f} | {m['ndcg@10']:.3f} | {p50:.0f} |")
    print("\nrerank latency (bge, all queries):", latency_report())

if __name__ == "__main__":
    main()
```

**Expected result.** A Markdown table with ≥4 config rows, e.g.:

```
| config          | recall@5 | nDCG@10 | p50 latency (ms) |
|-----------------|----------|---------|------------------|
| baseline        | 0.62     | 0.55    | 40               |
| +hybrid         | 0.71     | 0.63    | 55               |
| +hybrid+rerank  | 0.80     | 0.74    | 2100             |
| +all            | 0.82     | 0.76    | 9500             |
```
`+hybrid+rerank` shows a clear recall@5 gain over `baseline`; latency jumps at rerank and again at multi-query. Your exact numbers will differ — the *shape* (gains + latency cost) is what matters.

**Verify.**
```bash
python eval/ablation.py | tee eval/ablation_table.md
# Redirect to a file so you can paste it into your ship decision.
```
Check: 4 rows present, `+hybrid+rerank` recall@5 > baseline recall@5.

**Troubleshoot.**
- **All configs identical** → likely all golden queries are semantic (hybrid adds nothing) *and* the gold chunk is already at rank 1 (rerank can't improve). Add a few exact-token queries (error codes, function names) to the golden set to exercise BM25.
- **`+hybrid+rerank` shows no gain** → per the DoD, suspect the golden set (too small/noisy) or chunking. State it explicitly in your ship decision. Also verify recall@50 is high — if the gold chunk isn't in the 50 candidates, rerank is blind.
- **`ranx` KeyError / empty run** → a `query_id` in the run has no entry in qrels (its `relevant_doc_ids` was empty). The `if not r["relevant_doc_ids"]: continue` guards this; confirm Step 0.5 backfilled ids.
- **nDCG@10 but you only return 5** → intentional; nDCG@10 with 5 returned still scores the ranks you have. To fully populate @10, return `top_n=10` in the scored dict.
- **`+all` painfully slow** → multi-query does N× retrievals + an LLM call per query. Cache is your friend; for the table you only need one clean run.

---

## Putting it together — short end-to-end run

```bash
# 0. Infra + models up
curl -s http://localhost:6333/healthz && ollama list

# 1. Ingest with dense + BM25 sparse vectors (+ seeded restricted chunks)
python ingest.py

# 2. Prove hybrid beats dense on an exact-token query
python -c "from retrieval.hybrid import hybrid_search, dense_only; \
q='YOUR_EXACT_TOKEN_QUERY'; \
print('hybrid', [p.payload['doc_id'] for p in hybrid_search(q, top_k=5)]); \
print('dense ', [p.payload['doc_id'] for p in dense_only(q, top_k=5)])"

# 3. ACL gate must be green
pytest tests/test_acl.py -v

# 4. The ablation table (the deliverable)
python eval/ablation.py | tee eval/ablation_table.md
```

Then write your **ship decision** (2-5 sentences) at the bottom of `eval/ablation_table.md`: which techniques you keep and why, in cost-vs-gain terms. Example skeleton:

> **Ship decision.** Keep **hybrid** (adds ~9 recall pts for +15 ms — free). Keep **rerank** (adds ~9 more recall + ~11 nDCG pts; costs ~2 s/query on CPU — acceptable behind a top-50 cap, revisit with GPU/Cohere if latency-critical). **Do not ship multi-query by default** (+2 recall pts for ~7× latency and an LLM call — enable per-request only for hard queries). ACL pre-filter ships always (security, negligible cost).

---

## Definition of Done — verifiable checks

Restating the spine checklist as things you can point at:

- [ ] **Docker Qdrant collection has both vector types with server-side RRF.** `curl .../collections/docs` shows `dense` under `vectors` and `bm25` under `sparse_vectors`; `hybrid_search` uses `FusionQuery(RRF)`. *(pgvector users: dense + `rank_bm25` fused via the Python `rrf()` helper.)*
- [ ] **Reranker reduces 50 → 5 with logged latency.** `rerank(q, hybrid_search(q, top_k=50), top_n=5)` returns 5; `latency_report()` reports **p50 and p95** over the golden set.
- [ ] **≥3 toggleable query transforms.** `multi_query`, `hyde`, `condense` implemented and switched via `config.QUERY_REWRITE` / flags.
- [ ] **ACL test is pytest-green.** `pytest tests/test_acl.py` passes: an unauthorized user retrieves **0** restricted chunks *even when naming them*; an authorized user still can.
- [ ] **Ablation table with ≥4 configs.** `eval/ablation.py` prints `{baseline, +hybrid, +hybrid+rerank, +all}` with recall@5, nDCG@10, p50 latency; **+hybrid+rerank shows a measurable recall@5 gain over baseline** (or you've documented why not — golden set/chunking suspect).
- [ ] **Short written ship decision** (cost vs gain) committed alongside the table.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `Not existing vector name 'bm25'` | collection made without sparse config | re-run `recreate_collection()` with `sparse_vectors_config` |
| BM25 returns nothing / scores 0 | missing `Modifier.IDF`, or embedded with dense model | add `modifier=models.Modifier.IDF`; use `SparseTextEmbedding("Qdrant/bm25")` |
| hybrid == dense on every query | all golden queries are semantic | add exact-token queries (codes, function names) to exercise BM25 |
| reranker slow (>4 s / query, CPU) | 50 cross-encoder forward passes | expected; cap `top_k`, batch, or use Cohere/GPU |
| rerank doesn't improve ranking | gold chunk not in top-50 | tune retrieval recall@50 first; rerank only reorders |
| `json.JSONDecodeError` in multi-query | LLM wrapped JSON in prose/fences | try/except falls back to original; tighten prompt / strip fences |
| Ollama `ConnectionError` | daemon not running | `ollama serve`; confirm `ollama list` |
| ACL leak test fails | filter not applied | wire `query_filter=acl_filter` in `hybrid_search`; pass `acl_filter(user)` |
| `Index required for acl_groups` | no payload index | run `create_payload_index("docs","acl_groups", KEYWORD)` |
| `ranx` empty/KeyError | golden row has empty `relevant_doc_ids` | run Step 0.5 backfill; skip empties |
| Qdrant unreachable after reboot | container stopped | `docker start qdrant` |

---

## Stretch goals (optional)

- **Cohere Rerank A/B.** Set `RERANKER=cohere`, add `rerank-v3.5` as a 5th ablation row, and compare quality *and* latency against `bge-reranker-v2-m3`. Note the zero-ops vs data-egress tradeoff.
- **HyDE vs multi-query head-to-head.** Add `+hyde` as its own config row. Find a query class where HyDE wins (open-ended, answer-space) and one where it hurts (niche factoid) — this is Lecture 7's core lesson made concrete.
- **Self-query metadata extraction.** Add an LLM step that parses "invoices from Acme after 2024" into a `date`/`doc_type` filter, then `combine()` it with the ACL filter — and prove with a test that the ACL `must` clause survives the combine (self-query can ADD, never REMOVE).
- **Concurrency for multi-query.** Run the N variant retrievals with `asyncio`/`ThreadPoolExecutor` and re-measure `+all` p50 — show you can claw back most of the latency.
- **Late interaction (conceptual → tiny build).** Per [Lecture 9](../lectures/09-late-interaction-colbert-colpali.md), try `RAGatouille` (ColBERT) on a subset and note the index-size blowup vs the token-level recall gain on a query where single-vector blurring was the bottleneck. Timebox it — recognition, not production.
- **RRF `k` sensitivity.** Sweep `k ∈ {10, 60, 200}` in the Python `rrf()` fallback and show how fusion changes; explain why `k≈60` is the robust default.
