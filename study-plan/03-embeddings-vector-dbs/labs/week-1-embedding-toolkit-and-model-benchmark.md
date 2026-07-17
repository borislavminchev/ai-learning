# Week 1 Lab: Embedding Toolkit and Model-Selection Benchmark

> **What you build this week:** a small, reusable embedding toolkit (`encode()` with per-family prefixes, length-bucketing, L2-normalization, and a version-keyed disk cache) plus a mini **model-selection benchmark** that embeds a real corpus, runs flat cosine search as ground truth, and reports recall@{1,5,10}, MRR@10, dimension, and throughput for ≥3 models. You then run three sharp experiments: a **prefix ablation** (E5 with/without `query:`/`passage:`), a **Matryoshka** dimension-truncation sweep, and a **cache-hit proof**. This is the foundation the whole phase reuses — Week 2's ANN sweep and Week 3's retrieval service both call this encoder.
>
> **Why it matters:** picking an embedding model by leaderboard alone is how beginners ship bad retrieval. By the end you can pick a model *on evidence from your own data*, know which metric it wants, use its prefixes correctly, and generate vectors at scale without re-embedding or melting your budget.
>
> **Read these lectures first** (the spine's Theory is their recap):
> - [1 · Embedding Geometry and Similarity Metrics](../lectures/01-embedding-geometry-and-metrics.md)
> - [2 · Choosing an Embedding Model on Evidence](../lectures/02-choosing-an-embedding-model.md)
> - [3 · Query/Document Asymmetry and Instruction Prefixes](../lectures/03-query-document-prefixes.md)
> - [4 · Matryoshka Representation Learning and Dimension Truncation](../lectures/04-matryoshka-truncation.md)
> - [5 · Generating Embeddings at Scale: Batching, Caching, and Serving](../lectures/05-embedding-generation-at-scale.md)

**Est. time:** ~8 hrs · **You will need:** Python 3.10+, [`uv`](https://docs.astral.sh/uv/), ~3 GB disk for model weights, 16 GB RAM. **Everything runs on CPU for free** — no GPU, no paid API. (Optional: `text-embedding-3-small` costs ~$0.02/1M tokens if you want to try an API model; it stays gated behind the cache.)

---

## Before you start (setup)

**What:** create the repo, a virtualenv, and install CPU dependencies.

**Why:** `uv` gives you a fast, reproducible env and lockfile. CPU torch is enough for a 2k-doc corpus and three small models — no CUDA needed.

**Do it (Windows Git-Bash and macOS/Linux are identical here):**

```bash
# From wherever you keep projects
mkdir phase3-embeddings && cd phase3-embeddings
uv init --python 3.11
# Core deps for Week 1
uv add sentence-transformers numpy scikit-learn diskcache orjson datasets matplotlib
uv add --dev pytest
```

> **Windows note — CPU torch.** `sentence-transformers` pulls in `torch`. On Windows the default wheel is CPU-only, which is what you want. If `uv` ever tries to grab a CUDA build and fails, force CPU explicitly:
> ```bash
> uv add torch --index https://download.pytorch.org/whl/cpu
> ```
> On macOS (Apple Silicon) the default wheel uses MPS automatically — fine, leave it.

**Create the repo skeleton:**

```bash
mkdir -p data emb scripts tests results
touch emb/__init__.py emb/encode.py emb/cache.py emb/metrics.py
touch scripts/build_corpus.py scripts/bench_models.py
touch tests/test_metrics.py
```

Target layout (you'll fill these in step by step):

```
phase3-embeddings/
  pyproject.toml
  data/
    corpus.jsonl        # {"id","text","meta":{...}}
    eval.jsonl          # {"query","relevant_ids":[...]}
  emb/
    __init__.py
    encode.py           # batching + prefixes + cache
    cache.py            # diskcache keyed on model:version:sha256(normalized_text)
    metrics.py          # recall@k, mrr, ndcg
  scripts/
    build_corpus.py
    bench_models.py
  tests/
    test_metrics.py
  results/              # bench table + matryoshka plot land here
```

**Verify setup:**

```bash
uv run python -c "import sentence_transformers, numpy, diskcache, datasets; print('ok')"
```

Expected: `ok`. First run downloads nothing yet (models download on first `encode`).

**Troubleshoot:**
- `uv: command not found` → install: `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) or `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows PowerShell), then reopen the shell.
- Torch install hangs/OOMs on Windows → use the CPU index URL shown above; the CUDA wheel is ~2.5 GB and you don't need it.

---

## Step-by-step

### Step 1 — Build the corpus and eval set

**What:** download a small, *real* labeled retrieval dataset (BEIR `scifact`) and write it to `data/corpus.jsonl` + `data/eval.jsonl` with known relevant ids (qrels).

**Why:** you can only measure recall if you have ground-truth relevance labels. BEIR ships qrels, so you get honest recall@k for free instead of eyeballing results. `scifact` is tiny (~5k docs, ~300 queries) — perfect for CPU.

**Do it** — `scripts/build_corpus.py`:

```python
"""Build corpus.jsonl and eval.jsonl from BEIR scifact via HF datasets."""
import orjson
from pathlib import Path
from datasets import load_dataset

OUT = Path("data")
OUT.mkdir(exist_ok=True)

# BEIR scifact on the Hub: three configs — corpus, queries, qrels.
corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
queries = load_dataset("BeIR/scifact", "queries", split="queries")
qrels = load_dataset("BeIR/scifact-qrels", split="test")  # query-id, corpus-id, score

# 1) corpus.jsonl  -> {"id","text","meta":{...}}
with open(OUT / "corpus.jsonl", "wb") as f:
    for row in corpus:
        text = (row.get("title", "") + ". " + row.get("text", "")).strip()
        rec = {"id": str(row["_id"]), "text": text, "meta": {"title": row.get("title", "")}}
        f.write(orjson.dumps(rec) + b"\n")

# 2) build query-id -> [relevant corpus ids] from qrels (score > 0 == relevant)
rel: dict[str, list[str]] = {}
for r in qrels:
    if int(r["score"]) > 0:
        rel.setdefault(str(r["query-id"]), []).append(str(r["corpus-id"]))

qtext = {str(q["_id"]): q["text"] for q in queries}

# 3) eval.jsonl -> {"query","relevant_ids":[...]}  (only queries that have qrels)
n = 0
with open(OUT / "eval.jsonl", "wb") as f:
    for qid, ids in rel.items():
        if qid not in qtext:
            continue
        rec = {"query_id": qid, "query": qtext[qid], "relevant_ids": ids}
        f.write(orjson.dumps(rec) + b"\n")
        n += 1

print(f"wrote {corpus.num_rows} docs and {n} eval queries")
```

Run it:

```bash
uv run python scripts/build_corpus.py
```

**Expected result:** something like `wrote 5183 docs and 300 eval queries`. Two files appear in `data/`.

**Verify:**

```bash
uv run python -c "import sys; print(sum(1 for _ in open('data/corpus.jsonl')), 'corpus lines')"
head -n 1 data/eval.jsonl
```

You should see a JSON line with `query`, `query_id`, and a non-empty `relevant_ids` list.

**Troubleshoot:**
- Dataset config name errors (`Unknown split`/`BuilderConfig`) → HF dataset layouts drift. Fallback options: try `load_dataset("BeIR/scifact", "corpus")` without `split=`, then index `["corpus"]`; or switch to `nfcorpus` (same three-config shape). Last resort: scrape ~2k Wikipedia paragraphs and hand-label ~50 queries — but qrels save you hours, prefer BEIR.
- No internet / firewall → set `HF_HUB_OFFLINE=0` and retry; corporate proxies may need `HF_ENDPOINT`. If fully blocked, use the Wikipedia fallback.
- Slow download → `scifact` is a few MB; if it stalls, `export HF_HUB_ENABLE_HF_TRANSFER=1` after `uv add hf-transfer`.

---

### Step 2 — Metrics from scratch (`emb/metrics.py`)

**What:** implement `recall_at_k`, `mrr`, and `ndcg_at_k` over ranked id-lists vs a set of relevant ids.

**Why:** you must own these definitions — subtle off-by-one and "recall vs precision" confusion silently corrupts every comparison downstream. Implementing them yourself (and unit-testing against a hand-computed example) is the Definition-of-Done gate.

**Do it** — `emb/metrics.py`:

```python
"""Ranking metrics over id lists. Each fn takes a ranked list of predicted ids
(best first) and the set of relevant ids for that query."""
from __future__ import annotations
import math
from collections.abc import Iterable, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant docs that appear in the top-k."""
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in rel)
    return hits / len(rel)


def mrr(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Reciprocal rank of the FIRST relevant hit within top-k; 0 if none."""
    rel = set(relevant)
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    # binary gains; log2(rank+1) discount with rank starting at 1
    return sum((1.0 / math.log2(i + 1)) for i, doc_id in enumerate(ranked[:k], start=1)
               if doc_id in rel)


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k. IDCG = best possible DCG given #relevant."""
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = dcg_at_k(ranked, rel, k)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def aggregate(per_query: list[float]) -> float:
    """Mean over queries (macro average)."""
    return sum(per_query) / len(per_query) if per_query else 0.0
```

Now the unit test — `tests/test_metrics.py`:

```python
from emb.metrics import recall_at_k, mrr, ndcg_at_k
import math

# Toy: relevant = {A, C}. Ranked = [B, A, D, C, E].
RANKED = ["B", "A", "D", "C", "E"]
REL = {"A", "C"}


def test_recall():
    assert recall_at_k(RANKED, REL, 1) == 0.0          # top1=B, no hit
    assert recall_at_k(RANKED, REL, 2) == 0.5          # A found, 1/2 relevant
    assert recall_at_k(RANKED, REL, 4) == 1.0          # A and C found


def test_mrr():
    # first relevant (A) is at rank 2 -> 1/2
    assert mrr(RANKED, REL, 10) == 0.5
    assert mrr(["X", "Y"], REL, 10) == 0.0             # no hit


def test_ndcg():
    # hits at ranks 2 and 4: DCG = 1/log2(3) + 1/log2(5)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    # 2 relevant -> IDCG = 1/log2(2) + 1/log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert abs(ndcg_at_k(RANKED, REL, 5) - dcg / idcg) < 1e-9
```

**Expected / Verify:**

```bash
uv run pytest tests/test_metrics.py -v
```

All three tests green. If you can't predict the toy numbers by hand, re-read the metric definitions before moving on — this is the acceptance gate.

**Troubleshoot:**
- `ModuleNotFoundError: emb` → run pytest from the repo root, and make sure `emb/__init__.py` exists. `uv run pytest` uses the project root as cwd.
- nDCG assertion off → confirm your discount uses `log2(rank+1)` with rank starting at 1 (so rank-1 discount is `1/log2(2)=1`). Mixing 0-based ranks is the usual bug.

---

### Step 3 — The encoder with prefixes, bucketing, normalization (`emb/encode.py` + `emb/cache.py`)

**What:** a single `encode(texts, model_name, kind, normalize, batch_size)` that (a) applies the correct **query/passage prefix** for the model family, (b) sorts inputs by length into buckets before batching, (c) L2-normalizes, and (d) is wrapped by a `diskcache` layer keyed on `model_name:version:sha256(normalized_text)`.

**Why:**
- **Prefixes:** E5/BGE were trained with instruction prefixes. Skip them and recall silently drops several points — the #1 beginner retrieval bug (lecture 3).
- **Length-bucketing:** batches pad to the longest member. Sorting by length keeps short texts out of batches dragged long by one outlier, so throughput is honest (lecture 5).
- **Normalization:** with L2-normalized vectors, cosine and dot give identical rankings — you can use the fastest metric your index supports (lecture 1).
- **Version-keyed cache:** never re-embed identical text; and if the model *version* changes, the key changes so you don't serve stale vectors (lecture 5).

**Do it** — `emb/cache.py`:

```python
"""Disk cache for embeddings, keyed on model:version:sha256(normalized_text)."""
from __future__ import annotations
import hashlib
import re
import numpy as np
from diskcache import Cache

_cache = Cache(".emb_cache")           # on-disk, survives restarts
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Canonical form for cache keys: strip, collapse whitespace, lowercase.
    Cache on THIS, never on raw text, or trivially-different inputs miss."""
    return _WS.sub(" ", text.strip()).lower()


def cache_key(model_name: str, version: str, text: str) -> str:
    norm = normalize_text(text)
    h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return f"{model_name}:{version}:{h}"


def get(key: str) -> np.ndarray | None:
    return _cache.get(key, default=None)


def put(key: str, vec: np.ndarray) -> None:
    _cache[key] = vec


def stats() -> dict:
    return {"size": len(_cache), "hits": _cache.stats()[0], "misses": _cache.stats()[1]}
```

`emb/encode.py`:

```python
"""Cached, prefix-aware, length-bucketed encoder over sentence-transformers."""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from .cache import cache_key, get as cache_get, put as cache_put

# --- Prefix policy per model family ------------------------------------------
# kind is "query" or "doc". Return (query_prefix, doc_prefix).
def _prefixes(model_name: str) -> tuple[str, str]:
    n = model_name.lower()
    if "e5" in n:                       # intfloat/e5-*  and multilingual-e5-*
        return "query: ", "passage: "
    if "bge" in n and "-en" in n:       # BAAI/bge-*-en-v1.5 (query instruction only)
        return "Represent this sentence for searching relevant passages: ", ""
    if "nomic" in n:                    # nomic-embed-text-v1.5
        return "search_query: ", "search_document: "
    return "", ""                       # MiniLM etc.: no prefix


def apply_prefix(texts: list[str], model_name: str, kind: str) -> list[str]:
    qp, dp = _prefixes(model_name)
    p = qp if kind == "query" else dp
    return [p + t for t in texts] if p else list(texts)


@lru_cache(maxsize=4)
def _load(model_name: str) -> SentenceTransformer:
    # trust_remote_code needed for nomic-embed-text-v1.5
    return SentenceTransformer(model_name, trust_remote_code=True)


def _model_version(model: SentenceTransformer, model_name: str) -> str:
    # cheap stable version tag: model card revision if present, else dim.
    dim = model.get_sentence_embedding_dimension()
    return f"d{dim}"


def encode(
    texts: list[str],
    model_name: str,
    kind: str = "doc",            # "doc" | "query"
    normalize: bool = True,
    batch_size: int = 64,
    dims: int | None = None,      # Matryoshka truncation target (slice+renorm)
) -> np.ndarray:
    """Return (N, D) float32 embeddings. Cache hits skip the model entirely."""
    model = _load(model_name)
    version = _model_version(model, model_name)

    # 1) resolve cache; collect misses (cache on RAW text so kind/prefix don't
    #    change the key — but a query and doc of identical text are rare and the
    #    prefix policy is deterministic per kind, so key on kind too).
    keys = [cache_key(f"{model_name}|{kind}", version, t) for t in texts]
    out: list[np.ndarray | None] = [cache_get(k) for k in keys]
    miss_idx = [i for i, v in enumerate(out) if v is None]

    if miss_idx:
        miss_texts = [texts[i] for i in miss_idx]
        prefixed = apply_prefix(miss_texts, model_name, kind)

        # 2) length-bucketing: sort by char length, remember original order
        order = sorted(range(len(prefixed)), key=lambda i: len(prefixed[i]))
        sorted_texts = [prefixed[i] for i in order]

        vecs = model.encode(
            sorted_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        # 3) unsort back to miss order, then write cache
        unsorted = np.empty_like(vecs)
        for new_pos, orig in enumerate(order):
            unsorted[orig] = vecs[new_pos]
        for j, i in enumerate(miss_idx):
            cache_put(keys[i], unsorted[j])
            out[i] = unsorted[j]

    result = np.stack(out).astype(np.float32)

    # 4) optional Matryoshka truncation (slice first `dims`, renormalize)
    if dims is not None and dims < result.shape[1]:
        result = result[:, :dims]
        if normalize:
            result /= np.linalg.norm(result, axis=1, keepdims=True) + 1e-12
    return result
```

**Expected result:** first call downloads the model and shows a progress bar; returns a `(N, D)` float32 array with unit-norm rows (when `normalize=True`).

**Verify** in a REPL:

```bash
uv run python - <<'PY'
import numpy as np
from emb.encode import encode, apply_prefix
v = encode(["hello world", "a second doc"], "BAAI/bge-small-en-v1.5", kind="doc")
print("shape", v.shape, "norms", np.round(np.linalg.norm(v, axis=1), 4))
print(apply_prefix(["cats"], "intfloat/e5-small-v2", "query"))   # -> ['query: cats']
print(apply_prefix(["cats"], "intfloat/e5-small-v2", "doc"))     # -> ['passage: cats']
PY
```

Expected: `shape (2, 384) norms [1. 1.]`, and the E5 prefix lines. Norms of `1.0` prove normalization works.

**Troubleshoot:**
- `trust_remote_code` prompt or error on nomic → it's required; the `SentenceTransformer(..., trust_remote_code=True)` above handles it. If it still errors, `uv add einops` (nomic's custom code needs it).
- Norms not 1.0 → you passed `normalize=False`, or truncated dims without renormalizing.
- Slow first run → model download; subsequent runs load from `~/.cache/huggingface`. `bge-small` and `e5-small` are ~130 MB each.

---

### Step 4 — The benchmark (`scripts/bench_models.py`)

**What:** for each of `["BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2", "sentence-transformers/all-MiniLM-L6-v2"]`, embed corpus + queries, do **flat cosine search via NumPy argsort** (exact — this is ground truth, no ANN yet), and print recall@{1,5,10}, MRR@10, encode throughput (texts/sec), and dimension as a markdown table.

**Why:** this is the whole point of the week — choose a model on *your* data, not the leaderboard. Flat search gives 100%-honest ranking so recall numbers are trustworthy. Throughput and dim are the cost axes you trade against recall (lecture 2).

**Do it** — `scripts/bench_models.py`:

```python
"""Mini model-selection benchmark: flat cosine search, recall/MRR/throughput."""
import time
import orjson
import numpy as np
from pathlib import Path
from emb.encode import encode
from emb.metrics import recall_at_k, mrr, ndcg_at_k, aggregate

MODELS = [
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def load_jsonl(path):
    return [orjson.loads(line) for line in open(path, "rb")]


def flat_search(doc_vecs, query_vecs, doc_ids, k=10):
    """Exact top-k via cosine == dot on normalized vectors. Returns list of id-lists."""
    sims = query_vecs @ doc_vecs.T                  # (Q, N)
    topk = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    ranked = []
    for q in range(sims.shape[0]):
        idx = topk[q][np.argsort(-sims[q, topk[q]])]  # sort the k candidates
        ranked.append([doc_ids[i] for i in idx])
    return ranked


def bench(model_name, corpus, eval_set):
    doc_ids = [d["id"] for d in corpus]
    doc_texts = [d["text"] for d in corpus]
    q_texts = [q["query"] for q in eval_set]

    t0 = time.perf_counter()
    doc_vecs = encode(doc_texts, model_name, kind="doc")
    enc_time = time.perf_counter() - t0
    q_vecs = encode(q_texts, model_name, kind="query")

    ranked = flat_search(doc_vecs, q_vecs, doc_ids, k=10)
    rel = [set(q["relevant_ids"]) for q in eval_set]

    return {
        "model": model_name,
        "dim": doc_vecs.shape[1],
        "recall@1": aggregate([recall_at_k(r, g, 1) for r, g in zip(ranked, rel)]),
        "recall@5": aggregate([recall_at_k(r, g, 5) for r, g in zip(ranked, rel)]),
        "recall@10": aggregate([recall_at_k(r, g, 10) for r, g in zip(ranked, rel)]),
        "mrr@10": aggregate([mrr(r, g, 10) for r, g in zip(ranked, rel)]),
        "texts/sec": round(len(doc_texts) / enc_time, 1),
    }


def main():
    corpus = load_jsonl("data/corpus.jsonl")
    eval_set = load_jsonl("data/eval.jsonl")
    rows = [bench(m, corpus, eval_set) for m in MODELS]

    cols = ["model", "dim", "recall@1", "recall@5", "recall@10", "mrr@10", "texts/sec"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = [str(r["model"]), str(r["dim"])] + \
                [f"{r[c]:.3f}" for c in ["recall@1", "recall@5", "recall@10", "mrr@10"]] + \
                [str(r["texts/sec"])]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    Path("results").mkdir(exist_ok=True)
    Path("results/bench.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

**Do it:**

```bash
uv run python scripts/bench_models.py
```

**Expected result:** a markdown table (also saved to `results/bench.md`) roughly like:

```
| model | dim | recall@1 | recall@5 | recall@10 | mrr@10 | texts/sec |
|---|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 384 | 0.5xx | 0.7xx | 0.8xx | 0.6xx | ~150 |
| intfloat/e5-small-v2 | 384 | 0.5xx | 0.7xx | 0.8xx | 0.6xx | ~150 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.4xx | 0.6xx | 0.7xx | 0.5xx | ~250 |
```

(Exact numbers vary by machine/scifact version; on scifact the E5/BGE models should clearly beat MiniLM on recall.)

**Verify:**
- Three rows, monotonic recall (recall@1 ≤ recall@5 ≤ recall@10) — if not, your `flat_search` sort is wrong.
- The second run is much faster to *encode* (cache hits) — you'll formalize this in Step 8.

**Troubleshoot:**
- `argpartition` errors when `k >= N` → guard with `k = min(k, N-1)`; scifact is big enough that this won't trip.
- All recalls near 0 → almost always a prefix bug or an id-type mismatch (`str` vs `int`). Confirm `doc_ids` and `relevant_ids` are both strings (Step 1 casts with `str(...)`).
- Slow encode → expected on CPU for 5k docs × 3 models (a few minutes total). Cache makes reruns instant.

---

### Step 5 — Prefix ablation (E5 with vs without prefixes)

**What:** rerun E5 twice — once through your normal prefixing, once forcing empty prefixes — and record the recall@10 delta.

**Why:** this *proves* on your data the claim from lecture 3, and it's a Definition-of-Done item: you must be able to state the number.

**Do it** — add to `scripts/bench_models.py` (or a new `scripts/ablate_prefix.py`):

```python
def bench_no_prefix(model_name, corpus, eval_set):
    """Same as bench() but bypass prefixing by pretending kind maps to no-prefix."""
    import emb.encode as E
    orig = E._prefixes
    E._prefixes = lambda name: ("", "")   # monkeypatch: kill prefixes
    try:
        # bust cache namespace so we don't read prefixed vectors:
        row = bench(model_name + "  (no-prefix)", corpus, eval_set)
    finally:
        E._prefixes = orig
    return row
```

> **Important:** the cache key includes `model_name`, so appending `"  (no-prefix)"` to the name in the reporting is cosmetic — but the *vectors* would still collide because `_load` and the key use the real name. Cleanest fix: give the no-prefix run a distinct cache namespace by passing a wrapper name. Simplest robust approach is a dedicated script:

```python
# scripts/ablate_prefix.py
import orjson, numpy as np
from emb.encode import encode, _load, apply_prefix
from emb.metrics import recall_at_k, aggregate
from scripts.bench_models import load_jsonl, flat_search

MODEL = "intfloat/e5-small-v2"
corpus = load_jsonl("data/corpus.jsonl"); eval_set = load_jsonl("data/eval.jsonl")
doc_ids = [d["id"] for d in corpus]; doc_texts=[d["text"] for d in corpus]
q_texts=[q["query"] for q in eval_set]; rel=[set(q["relevant_ids"]) for q in eval_set]
model = _load(MODEL)

def run(use_prefix: bool):
    dtxt = apply_prefix(doc_texts, MODEL, "doc") if use_prefix else doc_texts
    qtxt = apply_prefix(q_texts, MODEL, "query") if use_prefix else q_texts
    dv = model.encode(dtxt, normalize_embeddings=True, convert_to_numpy=True, batch_size=64).astype("float32")
    qv = model.encode(qtxt, normalize_embeddings=True, convert_to_numpy=True, batch_size=64).astype("float32")
    ranked = flat_search(dv, qv, doc_ids, k=10)
    return aggregate([recall_at_k(r, g, 10) for r, g in zip(ranked, rel)])

with_p = run(True); without_p = run(False)
print(f"E5 recall@10 WITH prefixes:    {with_p:.3f}")
print(f"E5 recall@10 WITHOUT prefixes: {without_p:.3f}")
print(f"delta (prefix benefit):        {with_p - without_p:+.3f}")
```

```bash
uv run python -m scripts.ablate_prefix
```

**Expected result:** `WITH` beats `WITHOUT`, delta positive — commonly +0.02 to +0.08 recall@10 on scifact. Record the exact number in your README.

**Verify:** delta is positive and non-trivial. If it's ~0, confirm `_prefixes` actually returns `("query: ", "passage: ")` for this model name (Step 3 matched on `"e5"`).

**Troubleshoot:**
- Delta negative or zero → you're double-prefixing (the encode path already prefixes, and you prefixed again) or the model doesn't want prefixes. This script bypasses `encode()` and prefixes manually, so make sure you're not also going through cached prefixed vectors.
- Ran out of patience → this re-embeds E5 twice (~10k texts). It's the price of an honest ablation; the cache doesn't help here because the inputs differ.

---

### Step 6 — Matryoshka truncation sweep + plot

**What:** with an MRL-trained model (`nomic-ai/nomic-embed-text-v1.5`), embed at full dim then truncate to 768/512/256/128 (slice + renormalize), and plot recall@10 vs dims.

**Why:** MRL lets you cut storage/RAM linearly with dims while losing little recall — you need to *quantify* the tradeoff, not assume it (lecture 4). Definition of Done: state the recall lost 768→256 and the storage saved.

**Do it** — `scripts/matryoshka.py`:

```python
import orjson, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from emb.encode import encode
from emb.metrics import recall_at_k, aggregate
from scripts.bench_models import load_jsonl, flat_search

MODEL = "nomic-ai/nomic-embed-text-v1.5"   # 768-dim, MRL-trained
DIMS = [768, 512, 256, 128]

corpus = load_jsonl("data/corpus.jsonl"); eval_set = load_jsonl("data/eval.jsonl")
doc_ids=[d["id"] for d in corpus]; doc_texts=[d["text"] for d in corpus]
q_texts=[q["query"] for q in eval_set]; rel=[set(q["relevant_ids"]) for q in eval_set]

# encode ONCE at full dim (cached), then slice locally per dims
full_docs = encode(doc_texts, MODEL, kind="doc")      # (N, 768)
full_q    = encode(q_texts,   MODEL, kind="query")

def renorm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

recalls = []
for d in DIMS:
    dv = renorm(full_docs[:, :d].astype("float32"))
    qv = renorm(full_q[:, :d].astype("float32"))
    ranked = flat_search(dv, qv, doc_ids, k=10)
    r = aggregate([recall_at_k(rk, g, 10) for rk, g in zip(ranked, rel)])
    recalls.append(r)
    print(f"dims={d:4d}  recall@10={r:.3f}  storage/vec={d*4}B")

plt.figure()
plt.plot(DIMS, recalls, marker="o")
plt.gca().invert_xaxis()
plt.xlabel("dimensions (truncated)"); plt.ylabel("recall@10")
plt.title(f"Matryoshka truncation — {MODEL.split('/')[-1]}")
plt.grid(True, alpha=0.3)
plt.savefig("results/matryoshka.png", dpi=120, bbox_inches="tight")
print("wrote results/matryoshka.png")
print(f"recall lost 768->256: {recalls[0]-recalls[2]:+.3f};  "
      f"storage saved: {100*(1-256/768):.0f}%")
```

```bash
uv run python -m scripts.matryoshka
```

**Expected result:** recall degrades gracefully as dims fall — e.g. 768→256 loses only a point or two, while storage drops 67%. `results/matryoshka.png` appears; a printed summary states the loss and savings.

**Verify:** the plot file exists and recall is highest at 768 and lowest at 128. You can state "768→256 lost X points, saved 67% storage."

**Troubleshoot:**
- Recall *doesn't* degrade gracefully (falls off a cliff at 256) → you're truncating a **non-MRL** model. Only MRL-trained models (nomic-embed, `text-embedding-3-*`) support this; MiniLM/bge-small do not.
- Forgot to renormalize after slicing → cosine rankings get distorted; the `renorm()` above is mandatory.
- nomic download/`trust_remote_code` issues → see Step 3 troubleshooting (`uv add einops`).

---

### Step 7 — Cache-hit proof

**What:** run an encode twice and prove the second run makes **zero model calls** (all cache hits).

**Why:** the version-keyed cache is what makes generation-at-scale affordable. Definition of Done requires proving 0 model invocations on rerun via a log line/counter (lecture 5).

**Do it** — `scripts/cache_proof.py`:

```python
import orjson
from emb import cache
from emb.encode import encode
from scripts.bench_models import load_jsonl

texts = [d["text"] for d in load_jsonl("data/corpus.jsonl")][:500]
MODEL = "BAAI/bge-small-en-v1.5"

before = cache.stats()
encode(texts, MODEL, kind="doc")          # run 1: populates cache
mid = cache.stats()
encode(texts, MODEL, kind="doc")          # run 2: must be all hits
after = cache.stats()

run2_hits = after["hits"] - mid["hits"]
run2_misses = after["misses"] - mid["misses"]
print(f"run 2: hits={run2_hits} misses={run2_misses}")
assert run2_misses == 0, "cache miss on second run — key is not stable!"
print("PASS: second run was a full cache hit (0 model invocations)")
```

```bash
uv run python -m scripts.cache_proof   # run it twice if cache is empty
```

**Expected result:** `run 2: hits=500 misses=0` then `PASS: ...`.

**Verify:** the assert passes. For extra rigor, add a counter inside `encode` that increments only when `miss_idx` is non-empty and print it.

**Troubleshoot:**
- `misses > 0` on run 2 → your cache key isn't stable. Common causes: keying on **raw** text instead of `normalize_text` (whitespace/case differ), or including something non-deterministic (a timestamp) in the version. Re-check `emb/cache.py`.
- Want to test the version bump → change `_model_version` to return `"d384-v2"`; the next run should be all misses (proves version invalidation works), then revert.

---

## Putting it together — end-to-end run

From a clean checkout, the full week runs as:

```bash
cd phase3-embeddings
uv sync                                   # install from lockfile
uv run python scripts/build_corpus.py     # Step 1: data/*.jsonl
uv run pytest -q                          # Step 2: metrics tests green
uv run python scripts/bench_models.py     # Step 4: results/bench.md table
uv run python -m scripts.ablate_prefix    # Step 5: prefix delta
uv run python -m scripts.matryoshka       # Step 6: results/matryoshka.png
uv run python -m scripts.cache_proof      # Step 7: 0-miss proof (run twice)
```

After this you have: a tested metrics module, a cached prefix-aware encoder, a 3-model benchmark table, a quantified prefix benefit, a Matryoshka recall-vs-dims plot, and a cache-hit proof — everything Week 2's ANN sweep and Week 3's service will import.

---

## Definition of Done — verifiable checks

Restating the spine's Week 1 checklist as things you can literally verify:

- [ ] **`pytest` green; `test_metrics.py` verifies recall/MRR/nDCG against a hand-computed example.**
      → `uv run pytest tests/test_metrics.py -v` shows 3 passing tests whose expected values you computed by hand (Step 2).
- [ ] **`bench_models.py` prints a table with recall@{1,5,10}, MRR@10, dim, and texts/sec for ≥3 models.**
      → `results/bench.md` has 3 rows and all six columns (Step 4).
- [ ] **Prefix ablation shows a measurable recall delta (E5 with prefixes beats without), and you can state the number.**
      → `ablate_prefix` prints a positive delta; record the figure in the README (Step 5).
- [ ] **Matryoshka plot exists; you can state recall lost 768→256 and storage saved.**
      → `results/matryoshka.png` exists; the script prints "recall lost 768→256" and "storage saved 67%" (Step 6).
- [ ] **Re-running encode is a full cache hit (0 model invocations on second run), proven by a log line/counter.**
      → `cache_proof` prints `misses=0` and `PASS` (Step 7).
- [ ] **You can answer in one sentence each: why normalization lets you use dot product, and which metric your chosen model wants.**
      → Answers: *(1)* On L2-normalized vectors, `cos(a,b) = a·b`, so dot product yields the same ranking as cosine — you use the faster metric. *(2)* Check the model card: bge-*/e5-*/MiniLM are trained for **cosine** (normalize + dot); confirm before trusting raw dot on any model.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| All recalls ~0 | id type mismatch (`int` vs `str`) or missing prefixes | cast all ids to `str`; verify `_prefixes` matches your model name |
| `ModuleNotFoundError: emb` | running pytest/scripts from wrong dir | run from repo root; ensure `emb/__init__.py` exists; use `uv run` |
| Torch tries to download CUDA on Windows | default index picked GPU wheel | `uv add torch --index https://download.pytorch.org/whl/cpu` |
| `trust_remote_code` / nomic import error | custom model code needs deps | `SentenceTransformer(..., trust_remote_code=True)`; `uv add einops` |
| Cache misses on 2nd run | keying on raw (not normalized) text | key on `normalize_text(text)`; include model **version** |
| Matryoshka falls off a cliff | non-MRL model truncated | use `nomic-embed-text-v1.5` or `text-embedding-3-*` only |
| Throughput looks terrible | no length-bucketing → padding to outliers | sort by length before batching (already in `encode`) |
| `argpartition` error | `k >= N` | `k = min(k, N-1)` |
| BEIR dataset config error | HF layout drift | try `nfcorpus`; or drop `split=` and index the returned dict |
| Comparing cosine scores across models | scores aren't on the same scale | only compare **rankings/eval metrics**, never raw similarity values |

---

## Stretch goals (optional)

- **Add an API model:** wire `text-embedding-3-small` behind the same `encode()` (and cache!). It's MRL, so it slots straight into Step 6. ~$0.01 for the whole scifact corpus; still gate every call behind the cache.
- **Token-length audit:** log token counts per doc and flag any exceeding the model's `max_seq_length` (bge/e5-small cap at 512). Silent truncation is a real recall killer — count how many chunks would be clipped.
- **Bigger/other datasets:** rerun the benchmark on `nfcorpus` or a Wikipedia scrape; see whether the model ranking holds across domains (it often doesn't — that's the lesson).
- **Add `bge-base` / `e5-large`:** benchmark a larger model and quantify the recall gain vs the throughput/dim cost — the exact "is it worth it" decision from lecture 2.
- **Threaded API concurrency:** if you added an API model, add bounded concurrency with exponential backoff and measure texts/sec vs the serial path.
- **Metric-choice experiment:** re-run flat search with L2 distance on normalized vectors and confirm the ranking is identical to cosine — a hands-on proof of lecture 1's central claim.
