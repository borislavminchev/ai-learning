# Lecture 8: IVF Inverted-File Indexes

> HNSW builds a graph you walk; IVF does something more brutal and more intuitive — it chops the vector space into buckets, and at query time it only looks inside the few buckets nearest your query. That single idea (cluster once, probe a few) is the second great family of ANN indexes, and it is the one you reach for when RAM is the constraint and you are willing to pay a training step up front. This lecture teaches you IVF from the k-means partition outward: what `train()` actually computes, why training on too little data quietly destroys your recall, and how the two knobs — `nlist` and `nprobe` — trade partition granularity against probe count. After this you will be able to size an IVF index for a given N, run an honest `nprobe` sweep, read FAISS factory strings like `IVF1024,Flat`, and explain to a design review exactly when you would pick IVF over HNSW — and when you would not.

**Prerequisites:** what an embedding and nearest-neighbor search are, L2 norm and cosine/dot equivalence for normalized vectors (Lecture 1), flat/brute-force ground truth and recall@k (Week 2 intro), HNSW's three knobs and especially `efSearch` (Lecture 7), basic big-O, comfort with arithmetic and simple probability · **Reading time:** ~24 min · **Part of:** Phase 3 — Embeddings Infrastructure & Vector Databases, Week 2

---

## The core idea (plain language)

A flat index compares your query against every vector in the corpus. That is 100% recall and O(N) per query — perfect and hopeless at scale. HNSW avoids the O(N) scan by building a navigable graph and greedily walking toward the query's neighborhood. IVF (Inverted File) avoids it a different way: **it pre-sorts the whole corpus into buckets, then only scans the buckets near your query.**

The mental model is a library with no card catalog versus a library with shelves organized by subject. In the flat library you walk every aisle reading every spine. IVF, instead, does a one-time job of clustering every book into (say) 1,000 subject shelves. When you arrive with a question, you don't read all 1,000 shelves — you figure out which handful of shelves your question belongs to and read only those. If your question is clearly about cooking, you read the cooking shelf and maybe the two adjacent shelves, and you ignore the other 997.

The "shelves" are **Voronoi cells**: k-means clusters of the vector space, each represented by a **centroid** (the cluster's average point). "Which shelf does my question belong to" means: which centroids are nearest my query vector. The name *inverted file* comes from text search — for each cluster (the "term"), you keep a posting list of the vector IDs that landed in it, exactly like an inverted index maps a word to the documents containing it.

Two numbers control everything. **`nlist`** is how many shelves you built — the number of clusters, fixed at train time. **`nprobe`** is how many of the nearest shelves you actually read at query time. `nprobe` is the runtime recall/latency dial: turn it up and you read more shelves, catch more true neighbors (higher recall), and pay more latency — with no rebuild. It is the direct analogue of HNSW's `efSearch`.

The catch that has no analogue in HNSW: IVF has a mandatory **`train()`** step that runs k-means to find those centroids, and it must see **representative data with enough points per cluster**. Train on too little or unrepresentative data and you get garbage shelves — books shelved nonsensically — and no amount of `nprobe` tuning will save your recall. That failure mode is the single most common way engineers wreck an IVF index, so we will spend real time on it.

---

## How it actually works (mechanism, from first principles)

### The train step: k-means partitions the space

`train()` takes a sample of your vectors and runs **k-means with k = `nlist`**. K-means is the standard iterative clustering algorithm: pick `nlist` initial centers, assign every training point to its nearest center, move each center to the mean of its assigned points, repeat until the centers stop moving. The output is `nlist` centroids — one per cell. These centroids define the partition of the entire vector space into Voronoi cells: every possible point belongs to the cell whose centroid is closest.

The centroids are the *only* thing k-means produces here, and they are the whole game. They are typically stored as a small flat index of their own (a "coarse quantizer" in FAISS terms) — `nlist` vectors of dimension `d`. For `nlist = 1024` and `d = 384` at fp32, that is `1024 × 384 × 4 ≈ 1.5 MB`. Tiny. Cheap to search exhaustively at query time, which matters below.

### Adding vectors: the inverted lists get filled

After training, you `add()` your corpus. For each vector, IVF computes which centroid is nearest and appends the vector's ID (and, in `IVF,Flat`, the full vector) to that centroid's **posting list** (also called an inverted list). After adding N vectors across `nlist` cells, each cell holds on average `N / nlist` vectors. This average is the load factor that drives your per-query cost, so remember it: **average cell size ≈ N / nlist.**

```
        VECTOR SPACE partitioned into nlist Voronoi cells
        (centroids marked ×; a query q lands near a few)

   cell 3       cell 7            cell 12
  ┌────────┐  ┌──────────┐      ┌─────────┐
  │  ×     │  │    ×     │      │   ×     │
  │ • •  • │  │ •   • •  │      │  •  •   │
  │  •  •  │  │  • •  •  │      │ •    •  │
  └────────┘  └──────────┘      └─────────┘
                    ▲ q  (nearest centroids: 7, then 3, then 12)
                    │
   nprobe=1 → scan only cell 7
   nprobe=2 → scan cells 7 and 3
   nprobe=3 → scan cells 7, 3, and 12
```

### Query time: find nearest centroids, then scan only those cells

A query runs in two stages:

1. **Coarse search.** Compute the distance from the query to all `nlist` centroids and take the `nprobe` nearest. Because `nlist` is small (thousands, not millions), this is a cheap exhaustive scan over the centroids.
2. **Fine search.** Concatenate the posting lists of those `nprobe` cells and do an exact distance computation of the query against every vector in them, then keep the top-k.

So the total number of distance computations per query is roughly:

```
cost ≈ nlist            (coarse: query vs every centroid)
     + nprobe × (N / nlist)   (fine: query vs vectors in probed cells)
```

Compare that to flat search's cost of `N`. With `N = 1,000,000`, `nlist = 1000`, `nprobe = 10`:

```
coarse = 1,000
fine   = 10 × (1,000,000 / 1000) = 10 × 1,000 = 10,000
total  ≈ 11,000 comparisons   vs   1,000,000 for flat
       ≈ a 90x reduction
```

That is the whole speedup, and it comes with the exact recall cost you would expect: **any true neighbor that happens to live in an unprobed cell is missed.** If the true nearest neighbor sits in the 4th-closest cell but you only probed 3, you lose it. That is why `nprobe` is your recall dial.

### The boundary problem — why recall is not 100%

Voronoi cells have hard edges, but "nearness" does not respect them. A query can land close to a cell boundary such that its true nearest neighbor is *just across the line* in the neighboring cell. With `nprobe = 1` you scan only the query's own cell and miss that neighbor entirely. Raising `nprobe` scans adjacent cells too, which is precisely why higher `nprobe` recovers the neighbors lost to boundary effects. This is the intuition to hold: **`nprobe` buys you tolerance to the query sitting near cell edges.**

### The two knobs, and how they interact

**`nlist` — number of cells (train-time, needs rebuild to change):**
More cells means smaller cells (`N / nlist` shrinks), so each probe scans fewer vectors — **faster per probe**. But smaller cells also means each cell covers less space, so the true neighbors are spread across *more* cells, and you need a **larger `nprobe`** to gather them back. Fewer, bigger cells is the opposite: slower per probe, but each probe covers more ground so you need fewer probes.

The standard rule of thumb (approximate) is **`nlist` ≈ sqrt(N) to 4·sqrt(N)`.** For N = 1,000,000 that is roughly `1000` to `4000` cells. This balances the two terms in the cost formula so neither the coarse scan nor the fine scan dominates.

**`nprobe` — cells probed per query (runtime, no rebuild):**
This is the pure recall/latency dial, exactly analogous to HNSW's `efSearch`. Higher `nprobe` = more recall, more latency, zero rebuild. You can and should set it per query or per SLO. `nprobe = 1` is fastest and lowest recall; `nprobe = nlist` degenerates to a full flat scan (100% recall, no speedup).

A useful way to think about the relationship: recall is roughly governed by the *fraction of the corpus you scan*, which is `nprobe / nlist`. Probing 16 of 1024 cells scans ~1.5% of the data; probing 64 of 1024 scans ~6%. More coverage, more recall — with diminishing returns, because the first few cells you probe are the ones most likely to contain the true neighbors.

---

## Worked example

Let's size and reason about a concrete index end to end. Corpus: **N = 1,000,000** normalized 384-dim embeddings (inner-product / cosine metric). We choose `nlist = 1024` (near `sqrt(1e6) = 1000`).

**Train.** FAISS wants enough training points per centroid. The common guidance is **30–256 training points per centroid**, with ~39×`nlist` a frequently cited minimum and 100+×`nlist` comfortable. For `nlist = 1024`, that's roughly **40,000 (bare minimum) to ~256,000 (comfortable)** training vectors. We have 1M vectors, so we can train on a 100k–200k representative sample. Good.

**Add.** All 1M vectors distributed across 1024 cells → average cell size `1,000,000 / 1024 ≈ 977` vectors. (Real distributions are lumpy — some cells hold 3,000, some hold 100 — which matters for tail latency; see production section.)

**Query cost as we sweep `nprobe`:**

```
nprobe   cells scanned   fine comparisons (≈977 each)   coarse   total ≈    fraction of corpus
  1            1                 977                      1024     ~2,000        0.1%
  4            4               3,908                      1024     ~4,900        0.4%
  8            8               7,816                      1024     ~8,800        0.8%
 16           16              15,632                      1024    ~16,700        1.5%
 32           32              31,264                      1024    ~32,300        3.1%
 64           64              62,528                      1024    ~63,600        6.1%
```

Now the honest part: **you do not know the recall from arithmetic — you measure it against a flat ground truth.** The cost table above is deterministic; the recall table is empirical. A typical (illustrative, *not* a benchmarked guarantee) shape you would see plotting recall@10 for this kind of setup:

```
nprobe:    1      4      8     16     32     64
recall@10: ~0.55  ~0.80  ~0.88 ~0.93  ~0.96  ~0.98   (illustrative shape only)
latency:   lowest ───────────────────────────► highest
```

The shape is the lesson, and it is universal for IVF: **recall climbs steeply at small `nprobe`, then flattens.** Going 1→4 buys a lot of recall cheaply; going 32→64 doubles your latency for maybe 2 points. You find the knee, set `nprobe` just past it to hit your recall SLO, and stop. This is the **`nprobe` sweep methodology**: build once, then loop `nprobe in {1, 4, 8, 16, 32, 64}`, and for each measure recall@10 (vs flat) and QPS. Plot recall vs QPS and read off the cheapest point that clears your SLO — no rebuild between points.

**FAISS, concretely:**

```python
import faiss, numpy as np

d = 384
nlist = 1024
quantizer = faiss.IndexFlatIP(d)                     # coarse quantizer (the centroids' index)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

index.train(train_vectors)     # MANDATORY: runs k-means, learns nlist centroids
index.add(corpus_vectors)      # fills the inverted lists

for nprobe in (1, 4, 8, 16, 32, 64):
    index.nprobe = nprobe      # runtime dial, no rebuild
    D, I = index.search(queries, k=10)
    # measure recall@10 vs your flat ground-truth I, and QPS
```

Or via the **factory string**, which is how FAISS people actually talk about indexes:

```python
index = faiss.index_factory(d, "IVF1024,Flat", faiss.METRIC_INNER_PRODUCT)
```

Read `"IVF1024,Flat"` as: coarse quantizer with `nlist = 1024` cells, and store the **full (flat) vectors** in each posting list. The part before the comma is the coarse quantizer; the part after is how each cell stores its vectors. `IVF...,Flat` keeps full fp32 vectors — the accurate, larger baseline. Next lecture the `,Flat` becomes `,PQ32` and the story changes entirely.

---

## How it shows up in production

**RAM is where IVF wins.** `IVF,Flat` stores the same full vectors a flat index does (roughly `N × d × 4` bytes) plus a small overhead for the centroids and posting-list structure — but crucially it does **not** carry HNSW's graph overhead. HNSW adds roughly `M × 2 × 4` bytes per vector for edges on top of the vector storage; at `M = 32` that's ~256 extra bytes per vector, which at 1M vectors is ~256 MB of pure graph, and at 100M vectors is ~25 GB just for edges. IVF's per-vector overhead is negligible by comparison. **This is the reason to reach for IVF at very large N: cheaper RAM per vector.** And when you pair IVF with PQ (next lecture), the vectors themselves get compressed too and the RAM savings become dramatic — that's the combination people actually deploy at billion-scale.

**The train step is real operational weight.** Unlike HNSW, you cannot just start adding vectors — you must train first, on a representative sample, and that sample has to exist before you build. In a pipeline this means an extra stage, an extra artifact (the trained centroids), and a decision about *what* to train on. If your data drifts, your centroids age: the partition that was balanced last quarter may now have half your traffic landing in three overloaded cells. Retraining means a rebuild.

**Uneven cell sizes cause tail latency.** The `N / nlist` average hides a lumpy reality. Real embedding distributions cluster; some Voronoi cells end up 5–10x larger than average. A query that probes a fat cell scans far more vectors than one that probes lean cells, so your p99 latency is driven by the biggest cells you probe, not the average. If you tuned `nprobe` on mean latency you will be surprised by the tail. Monitor cell-size distribution after `add()`.

**At equal memory, HNSW usually wins recall/latency.** This is the honest tradeoff and you should say it out loud in design reviews: for a given RAM budget with full-precision vectors, HNSW typically gives you better recall at a given latency than `IVF,Flat`. IVF's advantage is not raw quality — it's (a) lower RAM overhead per vector, (b) an explicit, cheap-to-scan structure that plays beautifully with quantization, and (c) simpler, more predictable behavior under updates in some engines. **The reason IVF is everywhere at scale is IVF+PQ**, where the memory savings from compression let you fit far more vectors in RAM, and *that* changes the whole cost equation.

**`nprobe` is your live SLO lever.** Because it needs no rebuild, `nprobe` is what you turn during an incident. Latency spiking under load? Drop `nprobe` and shed recall temporarily. Recall complaints on a specific query class? Raise it. Some teams even set `nprobe` dynamically per query difficulty. Treat it exactly like `efSearch`: the runtime dial you own in production.

---

## Common misconceptions & failure modes

**"Training on a tiny sample is fine."** This is *the* IVF footgun. If you train on too few points — or worse, forget the effect of `nlist` and train 1,000 points for `nlist = 1024` — k-means produces garbage centroids: empty or near-empty clusters, centroids sitting in low-density nowhere-land, a partition that does not reflect where your data actually lives. Your vectors then land in nonsensical cells, true neighbors scatter across unrelated cells, and **no `nprobe` value recovers the recall** because the partition itself is broken. FAISS will even warn you (`WARNING clustering ... points to ... centroids: please provide at least ...`). Believe the warning. Give k-means **enough representative points per centroid** — tens per centroid minimum, hundreds to be comfortable — and make sure the sample looks like production data, not a curated or stale subset.

**"Higher `nlist` is always faster."** Higher `nlist` is faster *per probe* but demands more probes to hold recall, and it makes the coarse scan (over `nlist` centroids) more expensive. Past a point the coarse search itself dominates. `sqrt(N)`-ish is the sweet spot for a reason.

**"IVF needs no ground truth to tune."** No — like every ANN index, IVF's recall is only meaningful measured against a flat index over the same data. An `nprobe` sweep without a ground-truth denominator tells you latency, not quality.

**"IVF,Flat saves memory."** By itself it saves only the graph overhead HNSW would have added; it still stores every full vector. The big memory win comes from the quantization you add *after* the comma (PQ, SQ), not from IVF alone.

**"I can change `nlist` at serve time."** No. `nlist` is baked into training and the partition. Changing it means retrain + rebuild. Only `nprobe` is free at runtime.

**Comparing IVF and HNSW at different operating points.** If you benchmark HNSW at high `efSearch` against IVF at `nprobe = 1`, you have measured nothing. Compare them on the same recall/QPS/RAM plot at matched recall.

---

## Rules of thumb / cheat sheet

- **`nlist` ≈ sqrt(N) to 4·sqrt(N)** (approximate). N=1M → ~1k–4k cells; N=100M → ~10k–40k.
- **Training points: ≥ ~40×`nlist` bare minimum, 100–256×`nlist` comfortable.** Below that, expect FAISS warnings and wrecked recall.
- **`nprobe`: sweep `{1, 4, 8, 16, 32, 64}`**, plot recall@10 (vs flat) vs QPS, pick the knee that clears your SLO. Build once, sweep freely — no rebuild.
- **Recall ≈ driven by `nprobe / nlist`** (fraction of corpus scanned). Steep gains early, flat later.
- **Per-query cost ≈ `nlist + nprobe × (N/nlist)`.** Coarse scan + fine scan.
- **Factory string:** `IVF1024,Flat` = 1024 cells, full vectors per cell (accurate baseline). The `,Flat` is what you compress next lecture.
- **Pick IVF over HNSW when:** RAM at scale is the binding constraint, you can afford a train step, and you plan to quantize. **Pick HNSW when:** you want best recall/latency at moderate scale and RAM is available.
- **Always** train on **representative** data with enough points; **always** measure recall against a flat ground truth; **monitor** cell-size skew for tail latency.

---

## Connect to the lab

In this week's **Recall/QPS Pareto lab** (`ann/build_ivfpq.py`, `ann/sweep.py`), you build `IVF1024,Flat` as your accurate IVF baseline and one `IVF...,PQ...` for the compressed comparison, then sweep `nprobe in {1,4,8,16,32,64}` recording recall@10 (against the flat ground truth from `ann/ground_truth.py`), QPS, build time, and RSS. Note the mandatory `index.train()` call and prove to yourself that training on too few points degrades recall — try training on 500 points for `nlist=1024` and watch the FAISS warning and the recall collapse. Plot IVF as its own series on `pareto.png` next to HNSW so you can see, at matched RAM, exactly where each family wins.

---

## Going deeper (optional)

- **FAISS wiki — "Guidelines to choose an index"** and **"Faiss indexes"** (`github.com/facebookresearch/faiss`, the wiki). The authoritative source on `nlist`/`nprobe` guidance, training-point rules, and factory strings. Search: `faiss wiki guidelines to choose an index`.
- **FAISS wiki — "The index factory"** for decoding strings like `IVF1024,Flat` and `OPQ32,IVF4096,PQ32`. Search: `faiss index factory string`.
- **ann-benchmarks** (`github.com/erikbern/ann-benchmarks`) — see how IVF-family indexes plot against HNSW on recall-vs-QPS curves; the shape you'll reproduce.
- **"Product Quantization for Nearest Neighbor Search"** (Jégou, Douze, Schmid, 2011) — the paper that established the IVF+PQ design; read for the coarse-quantizer intuition even if you skip the PQ math. Search: `product quantization nearest neighbor Jegou`.
- **Pinecone / Weaviate learning centers** have accessible IVF explainers. Search: `IVF inverted file index explained ANN`.

---

## Check yourself

1. Your IVF index gives recall@10 = 0.62 no matter how high you push `nprobe`, even at `nprobe = nlist`. What is almost certainly wrong, and which step do you re-examine?
2. You have N = 4,000,000 vectors. Give a reasonable `nlist` and the approximate number of training vectors you'd want, and say why each.
3. You raise `nlist` from 1024 to 8192 and recall drops at the same `nprobe`. Explain why, and what you must change to recover recall.
4. Write the approximate per-query comparison count for `N=2,000,000`, `nlist=2000`, `nprobe=20`, and compare it to flat search.
5. At equal RAM with full-precision vectors, HNSW usually beats `IVF,Flat` on recall/latency. So why is IVF the dominant choice at billion-vector scale?
6. Which of `nlist` and `nprobe` can you change during a production incident without a rebuild, and what does turning it down buy and cost you?

### Answer key

1. If `nprobe = nlist` scans the entire corpus and recall is *still* only 0.62, the fine search is not the problem — the **partition/centroids are broken**, which almost always means **`train()` saw too few or unrepresentative points**. Re-examine the training step: how many training vectors relative to `nlist`, and whether the sample reflects production data. (A `nprobe = nlist` scan should approach flat-search recall; if it doesn't, something upstream of probing is wrong — e.g., training, metric mismatch, or unnormalized vectors.)

2. `nlist ≈ sqrt(4e6) = 2000` up to `4×2000 = 8000` is reasonable; call it ~2000–4000. Training vectors: at least ~40×`nlist` as a floor and ideally 100–256×`nlist`, so for `nlist = 2000` that's roughly **80,000 (minimum) to ~500,000 (comfortable)** representative vectors — enough points per centroid that k-means finds stable, meaningful clusters instead of garbage.

3. More cells (`nlist` up) means each cell is smaller and covers less space, so the true neighbors of a query are now spread across *more* cells. At the same `nprobe` you're scanning a smaller fraction (`nprobe/nlist` fell from 20/1024 to 20/8192), so you miss more neighbors. Recover recall by **raising `nprobe`** (roughly to keep `nprobe/nlist` — the corpus fraction scanned — comparable), which costs latency.

4. `cost ≈ nlist + nprobe × (N/nlist) = 2000 + 20 × (2,000,000 / 2000) = 2000 + 20 × 1000 = 2000 + 20,000 ≈ 22,000` comparisons, versus **2,000,000** for flat — about a **90x** reduction.

5. Because at scale the binding constraint is **RAM**, and IVF's real value shows up **paired with quantization (IVF+PQ)**: compressing the stored vectors lets you fit far more vectors per GB than full-precision HNSW can, and IVF's low per-vector overhead (no graph edges) plus its explicit cell structure make it the natural host for that compression. The comparison isn't "IVF,Flat vs HNSW at equal RAM" — it's "how many vectors can I serve per dollar of RAM," and IVF+PQ wins that.

6. **`nprobe`** — it's a pure runtime dial needing no rebuild (`nlist` requires retrain + rebuild). Turning `nprobe` *down* buys **lower latency / higher QPS** (fewer cells scanned) at the cost of **lower recall** — the lever you pull to shed load during an incident.
