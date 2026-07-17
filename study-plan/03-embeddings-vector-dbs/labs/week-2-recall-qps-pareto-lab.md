# Week 2 Lab: Recall/QPS/RAM Pareto Lab

> This week you turn "ANN feels fast" into numbers you can defend in an interview. You build a **flat ground-truth index**, sweep **HNSW** and **IVF-PQ** over their real knobs, add a **binary-quantize + rescore** path, and plot a **recall@10 vs QPS vs RAM Pareto frontier** from which you name the single config you would ship for a stated SLO. This is milestone component #1 — the "choose an index on evidence" half of the phase project.
>
> Read these lectures first (they are the theory this lab operationalizes):
> - [../lectures/06-exact-vs-approximate-and-honest-recall.md](../lectures/06-exact-vs-approximate-and-honest-recall.md) — why recall is only meaningful vs a flat index.
> - [../lectures/07-hnsw-internals-and-tuning.md](../lectures/07-hnsw-internals-and-tuning.md) — `M`, `efConstruction`, `efSearch` and what each moves.
> - [../lectures/08-ivf-inverted-file-indexes.md](../lectures/08-ivf-inverted-file-indexes.md) — `nlist`/`nprobe`, the mandatory `train()`.
> - [../lectures/09-quantization-pq-scalar-binary-rescore.md](../lectures/09-quantization-pq-scalar-binary-rescore.md) — PQ, scalar, binary, and the rescore trick.
> - [../lectures/10-recall-latency-cost-pareto.md](../lectures/10-recall-latency-cost-pareto.md) — how to read and draw a Pareto frontier.

**Est. time:** ~9 hrs · **You will need:** Python 3.10+, `uv`, and CPU only. Everything here (`faiss-cpu`, `matplotlib`, `pandas`, `psutil`, optional `hnswlib`) is free and local — no GPU, no paid API, no cloud. 16 GB RAM is enough; the 1M synthetic set peaks around 2–4 GB.

---

## Before you start (setup)

You should already have the `phase3-embeddings/` repo from Week 1 with real embeddings for your corpus (BGE-small, 384-dim, L2-normalized). This lab **reuses those embeddings** — recall is only honest when measured on real vectors, not random noise.

**1. Add the new package directory and dependencies.**

```bash
# from repo root: phase3-embeddings/
uv add faiss-cpu matplotlib pandas psutil
uv add --optional hnswlib   # optional cross-check; skip if it won't build on Windows
mkdir -p ann results
touch ann/__init__.py
```

> **Windows note:** `faiss-cpu` ships prebuilt wheels for Python 3.10–3.12 on win_amd64 — `uv add faiss-cpu` just works. `hnswlib` sometimes needs MSVC Build Tools to compile; it is **optional** (only a cross-check), so if `uv add hnswlib` fails, drop it and move on. macOS/Linux: both install cleanly.

**2. Confirm FAISS imports and threading.**

```bash
uv run python -c "import faiss, numpy as np; print('faiss', faiss.__version__); print('threads', faiss.omp_get_max_threads())"
```

Expected: a version string and a thread count > 1. **For honest, reproducible QPS you will pin FAISS to a single thread during search** (`faiss.omp_set_num_threads(1)`) so you measure per-query latency, not your core count. We do this inside `sweep.py`.

**3. Know where your Week-1 embeddings live.** You need two NumPy arrays:
- `corpus_emb` — shape `(N, 384)`, `float32`, **L2-normalized**, the searchable set (real, ~10–50k).
- `query_emb` — shape `(Q, 384)`, `float32`, L2-normalized, your eval queries (100–1000).

If Week 1 saved these as `.npy`, point `ann/_data.py` (below) at them. If not, re-run your Week-1 `encode()` on `corpus.jsonl` / `eval.jsonl` and `np.save` the results. Everything downstream imports from one place so you never re-embed.

The final layout you are building:

```
ann/
  _data.py            # single loader: real corpus + queries + synthetic 1M
  ground_truth.py     # IndexFlatIP -> exact top-100 per query, persisted
  build_hnsw.py       # IndexHNSWFlat builders
  build_ivfpq.py      # IVF1024,PQ32 and IVF1024,Flat builders
  quantize.py         # binary quantize + Hamming top-200 + fp32 rescore
  sweep.py            # grid over params -> results/sweep.csv
  plot.py             # Pareto scatter -> results/pareto.png
results/
  ground_truth.npy    # (Q, 100) int ids  = the recall denominator
  sweep.csv
  pareto.png
```

---

## Step-by-step

### Step 1 — One data loader (real set + synthetic 1M)

**What:** A single module that hands every other script the same arrays: real corpus, real queries, and a 1M synthetic set for scale-feel.

**Why:** The lecture ([06](../lectures/06-exact-vs-approximate-and-honest-recall.md)) is blunt: recall must be measured on **real** vectors or the number is theater (random vectors have no neighborhood structure, so every index looks equally good). But you also want to *feel* build time and RAM at 1M scale without waiting on a download. So: **recall + QPS on the real set; build-time/RAM sanity on the synthetic set.** Never report recall on synthetic data.

**Do it** — `ann/_data.py`:

```python
"""Single source of truth for all vectors used in the lab."""
import numpy as np
from pathlib import Path

DIM = 384
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def load_real():
    """Week-1 embeddings. Returns (corpus (N,384) f32 normalized, queries (Q,384))."""
    corpus = np.load("data/corpus_emb.npy")   # <-- point at your Week-1 output
    queries = np.load("data/query_emb.npy")
    corpus, queries = _l2norm(corpus), _l2norm(queries)
    assert corpus.shape[1] == DIM and queries.shape[1] == DIM
    return corpus, queries

def load_synthetic(n=1_000_000, seed=0):
    """1M random unit vectors — for build-time/RAM feel ONLY, never recall."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, DIM)).astype(np.float32)
    return _l2norm(x)

if __name__ == "__main__":
    c, q = load_real()
    print(f"real corpus {c.shape}  queries {q.shape}  dtype {c.dtype}")
    print(f"normalized? max|norm-1| = {np.abs(np.linalg.norm(c, axis=1) - 1).max():.2e}")
```

**Expected result:**

```
real corpus (23456, 384)  queries (200, 384)  dtype float32
normalized? max|norm-1| = 5.96e-08
```

**Verify:** `uv run python ann/_data.py`. The norm delta must be ~1e-7 (i.e. every vector is unit length). If it's large, your vectors weren't normalized and inner-product recall will be wrong.

**Troubleshoot:**
- `FileNotFoundError` → your Week-1 embeddings are elsewhere; fix the two `np.load` paths or regenerate with your Week-1 encoder and `np.save`.
- Corpus is tiny (< 5k) → recall differences between configs will be muddy. Pad with more of your Week-1 corpus, or accept slightly noisier curves; the *shape* still holds.
- Synthetic 1M OOMs → `1_000_000 * 384 * 4 bytes ≈ 1.5 GB` for the raw array plus the index. Drop to `n=500_000`, or generate in the builder rather than holding it alongside other arrays.

---

### Step 2 — Flat ground truth (the recall denominator)

**What:** Build `IndexFlatIP` over the real corpus, get the **exact top-100** for every eval query, and persist it. Recall for *every* approximate index is measured against this — never against another ANN index.

**Why:** [Lecture 06](../lectures/06-exact-vs-approximate-and-honest-recall.md): "recall@10 = 0.9" is meaningless without a denominator. Flat/brute-force is 100% recall by definition (it checks every vector), so it *is* the ground truth. We store top-100 (not top-10) so that later you can compute recall@k for any k ≤ 100 without rebuilding, and so binary+rescore has a fair 200-candidate pool to be judged against. `IndexFlatIP` = inner product; on L2-normalized vectors inner product ranks identically to cosine (see [Lecture 01](../lectures/01-embedding-geometry-and-metrics.md)).

**Do it** — `ann/ground_truth.py`:

```python
import time, numpy as np, faiss
from ann._data import load_real, RESULTS

K_GT = 100

def build_ground_truth(save=True):
    corpus, queries = load_real()
    index = faiss.IndexFlatIP(corpus.shape[1])   # exact inner-product
    index.add(corpus)
    t0 = time.perf_counter()
    scores, ids = index.search(queries, K_GT)    # (Q,100) each
    dt = time.perf_counter() - t0
    print(f"flat search {queries.shape[0]} queries x {corpus.shape[0]} docs in {dt:.3f}s")
    if save:
        np.save(RESULTS / "ground_truth.npy", ids.astype(np.int32))
        print(f"saved results/ground_truth.npy  shape {ids.shape}")
    return ids

def load_ground_truth() -> np.ndarray:
    return np.load(RESULTS / "ground_truth.npy")

def recall_at_k(approx_ids: np.ndarray, k: int = 10) -> float:
    """Mean over queries of |approx top-k ∩ flat top-k| / k."""
    gt = load_ground_truth()[:, :k]
    approx = approx_ids[:, :k]
    hits = [len(set(a) & set(g)) for a, g in zip(approx, gt)]
    return float(np.mean(hits) / k)

if __name__ == "__main__":
    build_ground_truth()
```

**Expected result:**

```
flat search 200 queries x 23456 docs in 0.041s
saved results/ground_truth.npy  shape (200, 100)
```

**Verify:** `uv run python -m ann.ground_truth`. Then sanity-check that flat recalls itself perfectly:

```bash
uv run python -c "
from ann.ground_truth import load_ground_truth, recall_at_k
gt = load_ground_truth()
print('self recall@10 =', recall_at_k(gt, 10))   # must be exactly 1.0
"
```

Expected: `self recall@10 = 1.0`. If that isn't exactly 1.0, `recall_at_k` is broken — fix it before trusting any other number.

**Troubleshoot:**
- Recall later looks impossibly low for *every* config → you likely built ground truth with L2 (`IndexFlatL2`) while searching with IP, or forgot to normalize. Keep the metric consistent: normalized vectors + `IndexFlatIP` everywhere.
- Query count is huge (>2k) and flat search is slow → that's fine for ground truth (you run it once), but subsample queries to ~200–500 to keep the *sweep* fast.

---

### Step 3 — HNSW builders

**What:** A function that builds `IndexHNSWFlat` for a given `(M, efConstruction)` and lets you set `efSearch` at query time.

**Why:** [Lecture 07](../lectures/07-hnsw-internals-and-tuning.md): `M` (edges/node) and `efConstruction` (build-time candidate list) are **baked in at build**; `efSearch` (query-time candidate list) is the **runtime dial** you change without rebuilding. That structure dictates the sweep shape: build once per `(M, efConstruction)`, then loop `efSearch` cheaply on the same index. RAM ≈ `(dim*4 + M*2*4)` bytes/vector — so `M` is your RAM knob.

**Do it** — `ann/build_hnsw.py`:

```python
import time, faiss

def build_hnsw(corpus, M: int, ef_construction: int):
    """Returns (index, build_seconds). Metric = inner product (normalized vecs)."""
    d = corpus.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    t0 = time.perf_counter()
    index.add(corpus)                 # graph is built during add()
    build_s = time.perf_counter() - t0
    return index, build_s

def set_ef_search(index, ef_search: int):
    index.hnsw.efSearch = ef_search   # runtime dial, no rebuild
```

**Expected result:** no output on import; used by the sweep. A single build on ~23k vectors with `M=32, efConstruction=200` takes a few seconds on CPU.

**Verify:** quick smoke test:

```bash
uv run python -c "
from ann._data import load_real
from ann.build_hnsw import build_hnsw, set_ef_search
from ann.ground_truth import recall_at_k
c, q = load_real()
idx, bs = build_hnsw(c, M=32, ef_construction=200)
print(f'built in {bs:.2f}s')
set_ef_search(idx, 128)
_, ids = idx.search(q, 10)
print('recall@10 =', round(recall_at_k(ids, 10), 3))   # expect ~0.9+
"
```

Expected: build in a few seconds, `recall@10` around 0.9–0.99 at `efSearch=128`. **Predict before you run it** (DoD): raising `efSearch` should raise recall.

**Troubleshoot:**
- Recall stuck low even at high `efSearch` → you built with the wrong metric. Must be `METRIC_INNER_PRODUCT` with normalized vectors (or `METRIC_L2` — but then be consistent everywhere). Don't mix.
- Build is very slow at `efConstruction=400, M=64` on 1M synthetic → expected; that's the point of the scale test. Time it, record it, don't panic.

---

### Step 4 — IVF-PQ and IVF-Flat builders

**What:** Build `IVF1024,PQ32` (compressed) and `IVF1024,Flat` (uncompressed IVF, for a fair recall comparison), each with a **mandatory `train()`** step, then sweep `nprobe` at query time.

**Why:** [Lecture 08](../lectures/08-ivf-inverted-file-indexes.md): IVF clusters vectors into `nlist` Voronoi cells; at query time it probes the `nprobe` nearest cells, so `nprobe` is the runtime recall/latency dial (like `efSearch`). PQ ([Lecture 09](../lectures/09-quantization-pq-scalar-binary-rescore.md)) compresses each vector into `m` subquantizer codes — huge RAM savings, some recall loss. IVF-Flat isolates *how much recall the PQ compression alone costs* vs the IVF probing. **`train()` is not optional**: IVF must learn the cell centroids (and PQ its codebooks) from representative data before you add vectors — skip it and FAISS raises, or recall collapses if you train on too few points.

`faiss.index_factory` builds these from a string spec. `PQ32` on 384-dim means 32 subquantizers × (384/32 = 12 dims each), 8 bits each → **32 bytes/vector** vs 1536 bytes for fp32 — a 48× shrink.

**Do it** — `ann/build_ivfpq.py`:

```python
import time, faiss

def build_ivf(corpus, factory: str, nlist_hint: int = 1024):
    """factory e.g. 'IVF1024,PQ32' or 'IVF1024,Flat'. Returns (index, build_s)."""
    d = corpus.shape[1]
    index = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
    # IVF needs ~30-256 training points per centroid; 1024 cells * 40 ≈ 40k.
    n_train = min(len(corpus), max(50_000, nlist_hint * 40))
    train = corpus[:n_train]
    t0 = time.perf_counter()
    index.train(train)     # MANDATORY: learns centroids (+ PQ codebooks)
    index.add(corpus)
    build_s = time.perf_counter() - t0
    return index, build_s

def set_nprobe(index, nprobe: int):
    faiss.extract_index_ivf(index).nprobe = nprobe   # runtime dial
```

> If your real corpus is small (say 12k) `nlist=1024` gives ~12 points/cell — too few, recall suffers. Either shrink to `IVF256,...` for the real set, or (recommended) do the *recall* sweep at a `nlist` sized to your corpus and reserve `IVF1024` for the 1M synthetic build-time demo. A rule of thumb: `nlist ≈ 4·sqrt(N)` to `16·sqrt(N)`.

**Expected result:** builds without error; `train()` prints FAISS clustering lines (`Training level-1 quantizer ...`). PQ32 index is dramatically smaller in RAM than HNSW.

**Verify:**

```bash
uv run python -c "
from ann._data import load_real
from ann.build_ivfpq import build_ivf, set_nprobe
from ann.ground_truth import recall_at_k
c, q = load_real()
idx, bs = build_ivf(c, 'IVF256,PQ32')   # 256 cells for a ~20k corpus
print(f'built+trained in {bs:.2f}s')
for np_ in (1, 8, 32, 64):
    set_nprobe(idx, np_)
    _, ids = idx.search(q, 10)
    print(f'nprobe={np_:>3}  recall@10={recall_at_k(ids,10):.3f}')
"
```

Expected: recall rises monotonically with `nprobe` and plateaus below fp32 HNSW (PQ loses a few points). Something like `nprobe=1 → 0.55`, `nprobe=64 → 0.90`.

**Troubleshoot:**
- `RuntimeError: ... index not trained` → you called `add()`/`search()` before `train()`. Order is train → add → search.
- `WARNING clustering ... only N points for K centroids` → too few training points per cell; lower `nlist` or feed more training data.
- Recall caps well below HNSW even at `nprobe=nlist` → that's PQ compression loss, not a bug. Show it with the `IVF...,Flat` variant, which should recover most of the gap at high `nprobe`.

---

### Step 5 — Binary quantization + rescore

**What:** Quantize every vector to 1 bit/dim (`> 0 → 1`), find top-200 by **Hamming distance**, then **rescore those 200 with the full-precision dot product** and keep the final top-10. Compare recall and RAM to fp32 HNSW.

**Why:** [Lecture 09](../lectures/09-quantization-pq-scalar-binary-rescore.md): binary is the cheapest quantization — 384 dims → 384 bits = **48 bytes/vector** (32× smaller than fp32's 1536 bytes), and Hamming distance is a `popcount(a XOR b)`, absurdly fast. Raw binary recall is mediocre, but the **rescore trick** rescues it: binary is a cheap *filter* to get a candidate pool, then you pay for full-precision math on only 200 vectors instead of N. The DoD asks you to show binary+rescore lands "within a few points" of fp32 at a fraction of the RAM.

**Do it** — `ann/quantize.py`:

```python
import numpy as np, faiss

def pack_binary(x: np.ndarray) -> np.ndarray:
    """(N,d) float -> (N, d/8) uint8 bit-packed. Bit = (value > 0)."""
    bits = (x > 0).astype(np.uint8)          # (N, d)
    return np.packbits(bits, axis=1)         # (N, d/8)

class BinaryRescore:
    def __init__(self, corpus: np.ndarray):
        self.corpus = np.ascontiguousarray(corpus, dtype=np.float32)  # kept for rescore
        d = corpus.shape[1]
        codes = pack_binary(corpus)
        self.bindex = faiss.IndexBinaryFlat(d)   # d must be multiple of 8
        self.bindex.add(codes)

    def search(self, queries: np.ndarray, k: int = 10, cand: int = 200):
        qcodes = pack_binary(queries)
        _, cand_ids = self.bindex.search(qcodes, cand)     # Hamming top-`cand`
        out = np.empty((len(queries), k), dtype=np.int64)
        for i, q in enumerate(queries):
            ids = cand_ids[i]
            ids = ids[ids >= 0]                            # drop padding
            sims = self.corpus[ids] @ q                    # full-precision rescore
            top = ids[np.argsort(-sims)[:k]]
            out[i, :len(top)] = top
        return out

    def ram_bytes(self):
        # binary codes only (rescore reuses corpus you already hold once)
        return self.bindex.ntotal * (self.bindex.d // 8)
```

**Expected result:** searchable in one pass. `d=384` is divisible by 8 (48 bytes). Recall@10 after rescore should be within a few points of fp32 HNSW.

**Verify:**

```bash
uv run python -c "
from ann._data import load_real
from ann.quantize import BinaryRescore
from ann.ground_truth import recall_at_k
c, q = load_real()
bq = BinaryRescore(c)
ids = bq.search(q, k=10, cand=200)
print('binary+rescore recall@10 =', round(recall_at_k(ids,10),3))
print('binary code RAM =', round(bq.ram_bytes()/1e6,2), 'MB',
      '(fp32 would be', round(c.nbytes/1e6,2), 'MB)')
"
```

Expected: recall ~0.92–0.98, binary codes ~32× smaller than the fp32 array. Also try `cand=200` vs `cand=100` — fewer candidates = faster but lower recall ceiling.

**Troubleshoot:**
- `AssertionError` / dim error from `IndexBinaryFlat` → your dim isn't a multiple of 8. 384 is fine; if you truncated to 250 dims (MRL), pad to 256.
- Raw binary recall (skip rescore) is ~0.6 and you think it's broken → that's expected; the whole point is the rescore step lifts it. Confirm rescore is actually running.
- Rescore is slow → you're rescoring in a Python loop; that's fine at 200 candidates. If you scale candidates way up, batch the gather with fancy indexing.

---

### Step 6 — The sweep (writes `sweep.csv`)

**What:** Run the full grid, measuring **recall@10, median QPS over ≥3 warmed runs, build time, and process RSS** for every config, and append rows to `results/sweep.csv`. This is the heart of the lab and the artifact the DoD checks.

**Why:** [Lecture 10](../lectures/10-recall-latency-cost-pareto.md): the only honest QPS is a **warmed median** — the first query pays JIT/allocation/cache-cold costs and would slander a good index. RSS (resident set size via `psutil`) is your RAM proxy; measure it *after* the index is built and *before* the next one so builds don't overlap in memory. The grid: HNSW `M∈{16,32,64} × efConstruction∈{100,200,400} × efSearch∈{16,32,64,128,256}` (≥15 configs — actually 45), IVF-PQ `nprobe∈{1,4,8,16,32,64}` on both PQ32 and Flat (≥6 configs), plus one binary+rescore row.

**Do it** — `ann/sweep.py`:

```python
import time, csv, gc, os
import numpy as np, faiss, psutil
from ann._data import load_real
from ann.ground_truth import build_ground_truth, recall_at_k
from ann.build_hnsw import build_hnsw, set_ef_search
from ann.build_ivfpq import build_ivf, set_nprobe
from ann.quantize import BinaryRescore

PROC = psutil.Process(os.getpid())
def rss_mb() -> float:
    return PROC.memory_info().rss / 1e6

def timed_qps(search_fn, queries, k=10, warmup=2, runs=5):
    """Warm up, then take MEDIAN wall time over `runs`. Returns (qps, ids)."""
    for _ in range(warmup):
        ids = search_fn(queries, k)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        ids = search_fn(queries, k)
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    qps = len(queries) / med
    return qps, ids

def main():
    faiss.omp_set_num_threads(1)          # single-thread => honest per-query latency
    corpus, queries = load_real()
    build_ground_truth()                  # refresh the denominator

    rows, base_ram = [], rss_mb()

    # ---- HNSW grid ----
    for M in (16, 32, 64):
        for efc in (100, 200, 400):
            index, build_s = build_hnsw(corpus, M, efc)
            index_ram = rss_mb() - base_ram
            for efs in (16, 32, 64, 128, 256):
                set_ef_search(index, efs)
                qps, ids = timed_qps(lambda q, k: index.search(q, k)[1], queries)
                rows.append(dict(family="hnsw", params=f"M{M}_efc{efc}_efs{efs}",
                                 M=M, efc=efc, efs=efs, nprobe="",
                                 recall10=round(recall_at_k(ids,10),4),
                                 qps=round(qps,1), build_s=round(build_s,2),
                                 ram_mb=round(index_ram,1)))
            del index; gc.collect()

    # ---- IVF sweeps (PQ32 compressed + Flat baseline) ----
    for factory in ("IVF1024,PQ32", "IVF1024,Flat"):
        index, build_s = build_ivf(corpus, factory)
        index_ram = rss_mb() - base_ram
        for np_ in (1, 4, 8, 16, 32, 64):
            set_nprobe(index, np_)
            qps, ids = timed_qps(lambda q, k: index.search(q, k)[1], queries)
            rows.append(dict(family=factory.split(",")[1].lower()+"_ivf",
                             params=f"{factory}_np{np_}", M="", efc="", efs="",
                             nprobe=np_, recall10=round(recall_at_k(ids,10),4),
                             qps=round(qps,1), build_s=round(build_s,2),
                             ram_mb=round(index_ram,1)))
        del index; gc.collect()

    # ---- binary + rescore (one row) ----
    bq = BinaryRescore(corpus)
    bram = bq.ram_bytes()/1e6
    qps, ids = timed_qps(lambda q, k: bq.search(q, k, cand=200), queries)
    rows.append(dict(family="binary_rescore", params="binary_cand200",
                     M="", efc="", efs="", nprobe="",
                     recall10=round(recall_at_k(ids,10),4),
                     qps=round(qps,1), build_s=0.0, ram_mb=round(bram,1)))

    cols = ["family","params","M","efc","efs","nprobe",
            "recall10","qps","build_s","ram_mb"]
    with open("results/sweep.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"wrote results/sweep.csv  ({len(rows)} rows)")

if __name__ == "__main__":
    main()
```

**Expected result:**

```
wrote results/sweep.csv  (45 + 12 + 1 = 58 rows)
```

45 HNSW + 12 IVF + 1 binary = 58 rows (well past the ≥15 HNSW / ≥6 IVF-PQ gate).

**Verify:**

```bash
uv run python -m ann.sweep
uv run python -c "
import pandas as pd
df = pd.read_csv('results/sweep.csv')
print('rows:', len(df), '| HNSW:', (df.family=='hnsw').sum(),
      '| IVF:', df.family.str.contains('ivf').sum())
# DoD directional check #1: efSearch up => recall up, QPS down (fix M,efc)
h = df[(df.family=='hnsw')&(df.M==32)&(df.efc==200)].sort_values('efs')
print(h[['efs','recall10','qps']].to_string(index=False))
# DoD directional check #2: M up => recall up, RAM up (fix efc,efs)
m = df[(df.family=='hnsw')&(df.efc==200)&(df.efs==128)].sort_values('M')
print(m[['M','recall10','ram_mb']].to_string(index=False))
"
```

Expected: as `efs` rises, `recall10` increases and `qps` decreases monotonically; as `M` rises, `recall10` and `ram_mb` both increase. **State this prediction out loud before running, then confirm** — that is an explicit DoD item.

**Troubleshoot:**
- QPS numbers jump around run-to-run → increase `runs` to 7–9 and keep `warmup≥2`; close other apps; the single-thread pin matters most.
- `ram_mb` comes out negative or wildly noisy → RSS is process-global and Python's allocator doesn't return freed memory to the OS, so the `- base_ram` delta drifts across many builds. For a cleaner RAM number, run each family in a **fresh subprocess** (see stretch goals) or trust FAISS's own `index.sa_code_size()`/analytical estimate for the writeup. The *relative ordering* (bigger M = more RAM) is what the DoD needs.
- Sweep takes too long → subsample queries to 200 and/or drop `efSearch=256`; the frontier shape survives.
- The `lambda q, k: index.search(...)` closes over the loop variable `index` correctly here because we call it immediately inside the same iteration — but if you refactor, bind it explicitly.

---

### Step 7 — The Pareto plot (`pareto.png`)

**What:** Scatter recall@10 (Y) vs QPS (X, log scale), encode RAM as point size/color, draw the **Pareto frontier**, and separate the series (HNSW / IVF-PQ / IVF-Flat / binary+rescore). Mark the SLO box.

**Why:** [Lecture 10](../lectures/10-recall-latency-cost-pareto.md): the frontier is the set of configs where you can't improve one axis without sacrificing another — everything below-left of it is strictly dominated and should never ship. The plot makes the shippable choice visually obvious and mirrors how ann-benchmarks presents results, which is the shape an interviewer expects. For plot-design conventions (log axis, categorical series colors, size legend) see the `dataviz` skill.

**Do it** — `ann/plot.py`:

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt

SLO_RECALL, SLO_QPS = 0.95, 500

def pareto_front(df):
    """Points not dominated on (recall up-good, qps up-good). Returns sorted frontier."""
    pts = df.sort_values("qps", ascending=False)
    front, best = [], -1.0
    for _, r in pts.iterrows():      # scan high->low QPS, keep new recall highs
        if r.recall10 > best:
            front.append(r); best = r.recall10
    return pd.DataFrame(front).sort_values("qps")

def main():
    df = pd.read_csv("results/sweep.csv")
    df = df[df.qps > 0]
    fig, ax = plt.subplots(figsize=(9,6))
    colors = {"hnsw":"#4C78A8","pq_ivf":"#F58518",
              "flat_ivf":"#54A24B","binary_rescore":"#E45756"}
    for fam, g in df.groupby("family"):
        ax.scatter(g.qps, g.recall10, s=3+g.ram_mb.clip(upper=400),
                   c=colors.get(fam,"#888"), alpha=0.7, edgecolors="k",
                   linewidths=0.4, label=fam)
    front = pareto_front(df)
    ax.plot(front.qps, front.recall10, "k--", lw=1.3, label="Pareto frontier")
    # SLO region
    ax.axhline(SLO_RECALL, color="gray", ls=":", lw=1)
    ax.axvline(SLO_QPS, color="gray", ls=":", lw=1)
    ax.axvspan(SLO_QPS, df.qps.max()*1.2, ymin=(SLO_RECALL-df.recall10.min()),
               alpha=0.05, color="green")
    ax.set_xscale("log")
    ax.set_xlabel("QPS (log, single-thread)"); ax.set_ylabel("recall@10")
    ax.set_title("Recall/QPS/RAM Pareto — point size = index RAM (MB)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig("results/pareto.png", dpi=140)
    print("wrote results/pareto.png")

    # print SLO-feasible configs, cheapest RAM first
    ok = df[(df.recall10 >= SLO_RECALL) & (df.qps >= SLO_QPS)].sort_values("ram_mb")
    print(f"\nConfigs meeting recall>={SLO_RECALL} & QPS>={SLO_QPS}:")
    print(ok[["family","params","recall10","qps","ram_mb"]].to_string(index=False)
          if len(ok) else "  (none — relax the SLO or tune further)")

if __name__ == "__main__":
    main()
```

**Expected result:** `results/pareto.png` with a clear upper-left-trending frontier, HNSW dominating the high-recall/high-QPS corner, IVF-PQ down and to the left but with tiny points (low RAM), binary+rescore as its own point. The console prints the feasible configs sorted by RAM.

**Verify:** open `results/pareto.png`. Confirm (a) X is log-scaled, (b) the dashed frontier hugs the top-right envelope, (c) IVF-PQ points are visibly smaller (less RAM) than HNSW, (d) the SLO lines cross at (500, 0.95). The printed table gives you the shippable config directly.

**Troubleshoot:**
- Frontier line zig-zags → your `pareto_front` scan is off; it must walk QPS high→low keeping running-max recall. The code above does this.
- All points cluster in one X spot → QPS barely varies because the corpus is tiny (search is instant regardless of config). Use the 1M synthetic set for a build/latency demo, or add more real vectors.
- Nothing meets the SLO → legitimate outcome on a small/hard corpus; report the closest config and what you'd change (more `efSearch`, bigger `M`, or accept 0.94).

---

### Step 8 — Write the finding

**What:** 3–5 sentences naming the config you would ship for **"recall@10 ≥ 0.95 at ≥ 500 QPS, minimize RAM"** and why. Put it in `results/FINDING.md` (and later the repo README).

**Why:** The whole lab exists to produce a defensible recommendation, not just a CSV. The finding is what you point an interviewer at.

**Do it** — template `results/FINDING.md`:

```markdown
## Shippable config for "recall@10 >= 0.95 at >= 500 QPS, minimize RAM"

**Ship: HNSW M=32, efConstruction=200, efSearch=128.**
It hits recall@10 = 0.9X at ~X,XXX QPS (single-thread), on the Pareto
frontier, at ~XXX MB for the 23k-vector real set. IVF1024,PQ32 uses ~30x
less RAM but tops out at recall@10 ~0.90 even at nprobe=64 — it fails the
0.95 gate, so it is only the pick if RAM is the hard constraint and 0.90
is acceptable. Binary+rescore (cand=200) recovers recall to 0.9X — within
~N points of fp32 HNSW — at ~32x less RAM than the fp32 corpus, making it
the RAM-constrained runner-up. efSearch is the free serve-time dial: raise
it toward 256 for +recall when latency budget allows, no rebuild.
```

Fill the `X`s from your `sweep.csv`.

**Verify:** every number in the finding traces to a row in `sweep.csv`; the named config appears on the frontier in `pareto.png` inside the SLO box.

---

## Putting it together — end-to-end run

From `phase3-embeddings/`:

```bash
# 0. deps (once)
uv add faiss-cpu matplotlib pandas psutil

# 1. confirm data loads and is normalized
uv run python ann/_data.py

# 2. ground truth (recall denominator)
uv run python -m ann.ground_truth

# 3. full sweep -> results/sweep.csv  (this is the long step, ~5-30 min)
uv run python -m ann.sweep

# 4. plot + SLO table -> results/pareto.png
uv run python -m ann.plot

# 5. write results/FINDING.md from the printed feasible-config table
```

Artifacts produced: `results/ground_truth.npy`, `results/sweep.csv` (58 rows), `results/pareto.png`, `results/FINDING.md`.

---

## Definition of Done (acceptance gate — from the spine)

- [ ] **Flat ground truth is the denominator.** `results/ground_truth.npy` from `IndexFlatIP` on normalized vectors; `recall_at_k` measures every config against it (never vs another ANN index). Self-recall check returns exactly 1.0.
- [ ] **`sweep.csv` has ≥15 HNSW + ≥6 IVF-PQ configs** with `recall10`, `qps`, `build_s`, `ram_mb` columns. (This lab produces 45 HNSW + 12 IVF + 1 binary.)
- [ ] **Predict-then-confirm the knobs:** you stated, then confirmed from the CSV, that `efSearch↑ ⇒ recall↑, QPS↓` and `M↑ ⇒ recall↑, RAM↑`.
- [ ] **Binary+rescore recovers recall within a stated few points of fp32 HNSW at materially less RAM** — with the actual numbers written in `FINDING.md`.
- [ ] **`pareto.png` shows the recall-vs-QPS frontier with RAM encoded** (size/color), IVF-PQ and binary+rescore as separate series, and you **name the config to ship** for `recall@10 ≥ 0.95 at ≥ 500 QPS`.
- [ ] **Indexes warmed, medians taken** (≥3 warmed runs, single-thread) so QPS isn't distorted by cold-start/JIT.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Every config has near-identical recall | Recall measured on synthetic/random vectors | Use real Week-1 embeddings for recall; synthetic only for build/RAM feel |
| Self-recall ≠ 1.0 | `recall_at_k` bug or wrong k slicing | It must be `|approx∩gt|/k` on the same k; ground truth stored top-100 |
| All recalls suspiciously low | Metric mismatch (built L2, searched IP) or unnormalized vectors | Normalize + `METRIC_INNER_PRODUCT` everywhere |
| `RuntimeError: index not trained` | IVF `add`/`search` before `train()` | Order: `train → add → search` |
| IVF recall very low even at high nprobe | Too few training points per cell (`nlist` too big for corpus) | Shrink `nlist` (`nlist≈4·√N`), or train on more data |
| QPS noisy / non-monotonic | Cold start, multithreading, background load | `faiss.omp_set_num_threads(1)`, warmup≥2, median over ≥5 runs |
| `ram_mb` negative or drifting | Python allocator doesn't return RSS to OS across many builds | Run each family in a fresh subprocess, or report analytical RAM |
| `IndexBinaryFlat` dim assertion | dim not divisible by 8 | Pad/truncate to a multiple of 8 (384 is fine; 256 for MRL) |
| Pareto line zig-zags | Frontier scan not keeping running-max recall | Scan QPS high→low, keep only recall highs |
| Nothing meets the SLO | Small/hard corpus or too-conservative knobs | Report closest config + what you'd change; raise efSearch/M |

---

## Stretch goals (optional)

1. **Real 1M set (SIFT/GIST).** Download `sift-128-euclidean` or `gist-960-euclidean` from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) (HDF5 with `train`, `test`, and precomputed `neighbors`). This gives you *published* ground truth and true-scale recall/QPS curves to compare your numbers against.
2. **hnswlib cross-check.** Rebuild the same `(M, efConstruction, efSearch)` grid with `hnswlib` and confirm recall/QPS agree with FAISS within noise — catches metric/normalization bugs in either library.
3. **Clean RAM measurement.** Run each index family in a subprocess (`subprocess.run([sys.executable, "-m", "ann.sweep_one", ...])`) and read peak RSS via `psutil` or `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` — eliminates allocator drift so `ram_mb` is trustworthy in absolute terms.
4. **Scalar quantization row.** Add `IndexScalarQuantizer` (fp32→int8, 4× smaller) as a fourth series between fp32 HNSW and binary — completes the quantization ladder from [Lecture 09](../lectures/09-quantization-pq-scalar-binary-rescore.md).
5. **OPQ + IVFPQ.** Compare `OPQ32,IVF1024,PQ32` vs plain `IVF1024,PQ32` — OPQ rotates the space before quantizing and usually buys a few recall points for free.
6. **Batched vs single-query QPS.** Measure QPS at batch sizes 1, 16, 256; batching amortizes overhead and often 5–10×'s throughput — relevant to how you'd actually serve.
