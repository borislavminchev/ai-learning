# Lecture 6: Exact vs Approximate Search and Honest Recall Measurement

> Approximate nearest-neighbor (ANN) search is the single most abused number in applied retrieval. Someone stands up an index, prints "recall@10 = 0.94," and ships it — without ever stating what that 0.94 was measured *against*. This lecture kills that habit permanently. You will learn why brute-force search is both the correct answer and an unaffordable one past a certain scale; why the only honest baseline is a flat index over the exact same vectors and queries; how to build that ground truth and persist it as your recall denominator; how to define recall@k precisely enough to survive a code review; and the measurement hygiene (warm-up, median QPS, fixed k and metric, normalized vectors) that keeps every benchmark this week comparable. After this you will be able to say "recall@10 = 0.94 versus a flat IndexFlatIP over the same 50k vectors and 200 eval queries at k=10" — and mean it.

**Prerequisites:** L2 normalization and cosine vs dot vs L2 ranking (Lecture 1), what an embedding is, basic big-O, comfort with NumPy · **Reading time:** ~24 min · **Part of:** Phase 3 — Embeddings Infrastructure & Vector Databases, Week 2

---

## The core idea (plain language)

Nearest-neighbor search has one exact answer and infinitely many approximate ones. The exact answer is trivial to compute: to find the top-k most similar vectors to a query, compare the query against **every** vector, sort by similarity, take the top k. This is called *flat* or *brute-force* search. It is always 100% correct — it *is* the definition of correct — and its cost is O(N) per query: with a million vectors, every query touches all million.

That linear cost is the whole problem. At 10k vectors nobody cares; a flat scan is microseconds. At 1M vectors on CPU you are looking at tens of milliseconds per query, single-threaded, and it gets linearly worse from there. A retrieval service doing 500 queries per second against 10M vectors cannot afford to touch 10M vectors 500 times a second. So we cheat: we build a data structure — a graph, a set of clusters, a compressed code — that lets a query look at only a *small fraction* of the vectors and still *usually* find the true neighbors. That is Approximate Nearest Neighbor search. It buys orders-of-magnitude speedup in exchange for occasionally missing a true neighbor.

"Occasionally missing" is exactly the thing you must quantify, and **recall@k** is the measurement. Here is the central discipline of this entire week, stated as bluntly as possible: **recall@k is meaningless without stating what it is measured against, and the only honest baseline is a flat index over the exact same vectors and queries.** The flat index gives you the true top-k for each query — the ground truth. Recall@k of an approximate index is then just the fraction of that true top-k the approximate index also managed to return. No flat baseline, no recall. A "recall@10 = 0.94" with no denominator is not a weak claim; it is *not a claim at all*.

The rest of the phase — HNSW, IVF-PQ, quantization, the Pareto plot — is a search for the cheapest point that meets your recall SLO. None of it means anything if the recall number underneath is dishonest. So we start here.

---

## How it actually works (mechanism, from first principles)

### Flat search is O(N), and here is what that costs

A flat top-k search over N vectors of dimension d does, per query, roughly N dot products of length d, then a partial sort to pull the top k. The dot products dominate: N × d multiply-adds.

Put numbers on it. Say d = 384 (a bge-small embedding), fp32.

- N = 10,000: 10k × 384 ≈ **3.8M** multiply-adds per query. On a modern CPU core doing a few GFLOP/s of usable BLAS throughput, this is well under a millisecond.
- N = 1,000,000: 1M × 384 ≈ **384M** multiply-adds per query. Now you are in the ~5–30 ms range per query single-threaded (FAISS's flat index is heavily vectorized, so treat these as order-of-magnitude, not promises).
- N = 100,000,000: **38.4 billion** multiply-adds per query. Seconds per query. Dead on arrival for online serving.

The shape is the point: cost grows *linearly* with N, and it never stops growing. There is no N past which flat search gets cheaper. RAM tells the same story — 1M × 384 × 4 bytes ≈ **1.47 GB** just for the raw fp32 vectors, before any index overhead.

```
per-query cost of flat search (schematic)

 latency
   ^
   |                                        *
   |                               *
   |                     *
   |            *
   |     *
   |*_______________________________________> N (vectors)
   10k   100k    1M      10M       100M

 flat = a straight line through the origin: no scale is "safe" forever
```

So flat search has exactly two good uses: (1) small N where O(N) is simply fast enough, and (2) **computing ground truth** — the exact answer you measure everything else against. Use #2 is why we care about it all week even though we would never serve from it at scale.

### ANN: touch fewer vectors, accept some misses

Every ANN method is a different answer to "how do I avoid looking at most of the vectors?" HNSW builds a navigable graph and greedily walks toward the query, visiting a few hundred nodes instead of a million. IVF clusters the vectors and only scans the few clusters nearest the query. PQ compresses vectors so each comparison is cheaper. They differ wildly in mechanism (later lectures), but they share one consequence: **the candidate set they examine is a subset of all vectors, so the true nearest neighbor can be outside that subset and get missed.** How often that happens is recall. How much speedup you got for it is the trade.

### Defining recall@k precisely

Fix a value of k (say 10). For a single eval query q:

- Let **G(q)** = the true top-k neighbors from the *flat* index (the ground-truth set). |G(q)| = k.
- Let **A(q)** = the top-k neighbors the *approximate* index returned. |A(q)| = k.

Then recall@k for that query is:

```
recall@k(q) = |A(q) ∩ G(q)| / k
```

The intersection counts how many of the true top-k the approximate index also found. Divide by k (not by |A(q)|, not by anything else) to get a fraction in [0, 1]. The dataset-level recall@k is the **mean over all eval queries**:

```
recall@k = (1/Q) · Σ_q  |A(q) ∩ G(q)| / k
```

That is the whole definition. Note carefully what k does here: it is *both* the size of the ground-truth set and the number of results you pull from the approximate index. If you compute ground truth at top-100 but ask the approximate index for top-10, you are measuring something else (recall@10-within-top-100), which is a legitimate metric but a *different* one — do not silently mix them. Persisting ground truth at top-100 and then slicing to the first k for each k you report is fine and common; just be explicit.

### A worked recall computation

Three eval queries, k = 5. Ground truth (from the flat index) and what the ANN index returned, as sets of vector IDs:

```
query 1:
  G = {17, 42,  3, 88,  9}      (true top-5)
  A = {17, 42,  3, 88, 51}      (ANN top-5)
  A ∩ G = {17, 42, 3, 88}       -> 4 hits
  recall@5 = 4/5 = 0.80

query 2:
  G = { 5, 12, 30, 77, 61}
  A = { 5, 12, 30, 77, 61}      identical
  recall@5 = 5/5 = 1.00

query 3:
  G = {100, 7, 23, 44, 2}
  A = {100, 7, 99, 44, 2}
  A ∩ G = {100, 7, 44, 2}       -> 4 hits
  recall@5 = 4/5 = 0.80

dataset recall@5 = (0.80 + 1.00 + 0.80) / 3 = 0.867
```

Notice recall does not care about *order* within the top-k, only membership. If you also care about ranking quality (getting the #1 result actually first), that is nDCG or MRR, which are separate metrics — recall@k is a set-overlap measure and nothing more.

### Building the flat ground truth (FAISS)

The denominator is a flat index over the *exact same* vectors and queries you will evaluate the ANN index on. For normalized vectors, use inner product (`IndexFlatIP`), because for unit vectors inner product ranks identically to cosine and negated-L2 (see Lecture 1). For un-normalized vectors, use `IndexFlatL2`.

```python
import faiss, numpy as np

# corpus: (N, d) fp32, ALREADY L2-normalized. queries: (Q, d), same normalization.
d = corpus.shape[1]
flat = faiss.IndexFlatIP(d)      # inner product == cosine for unit vectors
flat.add(corpus)                 # no train() needed; flat is exhaustive

K = 100                          # persist a generous top-K; slice to smaller k later
D, I = flat.search(queries, K)   # I: (Q, K) neighbor ids;  D: (Q, K) similarities

np.save("ground_truth_ids.npy", I)      # <-- the recall denominator, on disk
np.save("ground_truth_scores.npy", D)   # handy for debugging/rescore checks
```

`I` is your ground truth: for each of the Q queries, the exact top-100 neighbor IDs. Persist it. It is a pure function of (corpus vectors, query vectors, metric, K) — so if any of those change, it is stale and must be recomputed. Treat a stale ground-truth file the way you would treat a stale test fixture: silently wrong, and it will bless a broken index as good.

Then recall@k against it is a few lines:

```python
def recall_at_k(ann_ids, gt_ids, k):
    # ann_ids: (Q, >=k) from the approximate index;  gt_ids: (Q, >=k) flat truth
    hits = 0
    for a_row, g_row in zip(ann_ids[:, :k], gt_ids[:, :k]):
        hits += len(set(a_row.tolist()) & set(g_row.tolist()))
    return hits / (len(gt_ids) * k)
```

### The fraud: measuring an ANN index against another ANN index

The most common way people fool themselves is to compute "recall" of index B against index A, where A is *also* approximate. This does not measure recall. It measures **agreement** — how often two approximations happen to agree. Two indexes can agree 95% of the time while *both* being 90% recall against the true answer, because they miss different neighbors. Agreement is not correctness. A vector database's own "recall" dashboard, an HNSW index compared to an IVF index, one library's output compared to another's — none of these are ground truth unless one of them is *exact*.

```
   flat (exact)          ANN-A               ANN-B
   top-k = TRUTH         ~92% recall         ~90% recall

   recall(A vs flat) = 0.92   <- honest
   recall(B vs flat) = 0.90   <- honest
   recall(B vs A)    = 0.95   <- AGREEMENT, not recall; means nothing about truth
```

The rule that makes you immune: **the denominator must come from an exact method.** Flat search is the only exact method you have. If neither operand of your comparison is a flat index over the same data, you are not measuring recall.

### The three axes, and the vocabulary for the rest of the week

Once your recall number is honest, recall stops being a pass/fail checkbox and becomes one coordinate on a trade-off surface. There are exactly three axes, and every index decision this week is a point on them:

- **Recall@k** — quality, measured against the flat ground truth you just built. Higher is better, but only up to your SLO; past that it is wasted money.
- **Latency / QPS** — speed. Usually reported as queries per second at a fixed concurrency, or p50/p95 latency per query. This is what your users feel.
- **Cost = RAM / $** — the resident memory (and therefore the instance size and dollar bill) the index occupies. HNSW is memory-hungry; quantization trades RAM for recall. This is the axis people forget until the invoice arrives.

These three fight each other. You cannot maximize all at once — there is no free lunch, only a **Pareto frontier**, and your job is to find the cheapest point that *meets* the recall and latency SLO. Lectures 7–9 each attack a different corner (HNSW for recall/latency, IVF for RAM at scale, PQ/quantization for RAM), and Lecture 10 assembles the whole frontier into the Pareto plot. Everything downstream assumes the recall axis is measured the honest way this lecture just defined.

The vocabulary you will use to *name* these points is the **FAISS index-factory string** — a compact spec passed to `faiss.index_factory(d, "...", metric)` that builds an index from a recipe. A few you will meet repeatedly:

```
"Flat"              exact brute force (this lecture's ground truth)
"HNSW32"            HNSW graph, M=32 edges/node          (Lecture 7)
"IVF4096,Flat"      4096 clusters, exact scan within     (Lecture 8)
"IVF1024,PQ32"      1024 clusters + product quantization (Lectures 8-9)
"OPQ32,IVF4096,PQ32"  rotate, then IVF+PQ                (Lecture 9)
```

Read these left-to-right as a pipeline: an optional transform (`OPQ`), a coarse partitioner (`IVF…`), and an encoding of the residual (`Flat`, `PQ…`). By the end of the week each token in these strings will map to a knob you can predict the recall/latency/RAM effect of before you run it. For now, just recognize that `"Flat"` — the simplest possible factory string — is the exact index this lecture is built on.

---

## Worked example

You have 50,000 real corpus embeddings (d = 384, L2-normalized) and 200 eval queries. You want to report recall@10 for an HNSW index and know it is honest.

1. **Ground truth.** Build `IndexFlatIP(384)`, add the 50k vectors, search the 200 queries for top-100. Save `I` (200×100) to `ground_truth_ids.npy`. This took, say, ~0.4 s total for 200 queries — flat is fine at 50k. This file is now frozen; it is the denominator.

2. **Build the ANN index.** Build an HNSW index over the *same* 50k vectors. Set `efSearch = 64`. Search the *same* 200 queries for top-10 → `ann_ids` (200×10).

3. **Compute recall.** `recall_at_k(ann_ids, gt_ids, k=10)`. Suppose it returns **0.938**. You can now say, precisely: "recall@10 = 0.938, measured against a flat IndexFlatIP over the identical 50k vectors and 200 queries, at k=10, efSearch=64."

4. **Sweep the dial.** Re-run step 2 with `efSearch ∈ {16, 32, 64, 128, 256}`, recomputing recall each time against the *same frozen ground truth*. You get, illustratively:

```
efSearch   recall@10 (vs flat)   median QPS
   16          0.86                 9,000     (approx — your numbers will differ)
   32          0.91                 6,200
   64          0.938                4,100
  128          0.972                2,500
  256          0.988                1,400
```

Every row is comparable because k, metric, vectors, queries, and the ground-truth file were held fixed; only efSearch moved. *That* is a curve you can reason about: recall goes up, QPS goes down, and you pick the cheapest row that clears your SLO. (Numbers above are illustrative shapes, not measured benchmarks — you will produce your own in the lab.)

---

## How it shows up in production

- **The "our recall is 0.95" that nobody can reproduce.** A team reports a great recall number, ships, and retrieval quality is visibly worse than the number implies. Root cause, nine times out of ten: the 0.95 was agreement against another approximate system, or measured at a different k, or on a different (usually smaller, cleaner) slice of vectors than production serves. With a persisted flat ground truth over the real vectors, this argument ends in thirty seconds.

- **Ground truth drift.** Someone re-embeds the corpus with a new model, or adds 200k vectors, and forgets to regenerate `ground_truth_ids.npy`. Now recall is computed against neighbors that no longer exist or are no longer nearest. Recall numbers look stable while quality quietly rots. Tie ground-truth regeneration to the same pipeline step that produces the vectors, and stamp both with the same `model:version:corpus_hash`.

- **Cold-index latency lies.** The first query into a freshly built or freshly loaded index pays for JIT warm-up, lazy allocation, page faults, and cold caches. If your "QPS" is one cold query, you will report a number 3–10x worse than steady state and either over-provision hardware or reject a perfectly good config. Warm up first, then measure.

- **k drift across a comparison table.** A benchmark table where one row is recall@10 and another is recall@100 looks fine and is worthless — the numbers are not comparable. Higher k almost always shows higher recall (more slots, easier to overlap), so a sloppy table can make a worse index look better purely from a larger k. Freeze k across every row.

- **Un-normalized vectors under inner product.** If some vectors slipped through un-normalized and you are using `IndexFlatIP` (or any IP-metric ANN index), longer vectors get artificially higher scores and dominate results for reasons unrelated to relevance. Your ground truth is then *also* wrong, so recall looks fine while retrieval is subtly broken. Assert unit norm at ingest.

---

## Common misconceptions & failure modes

- **"Recall@10 = 0.94, so we're good."** Against *what*? Without a stated exact denominator this is not a measurement. First question in any review.

- **"We compared our index to Pinecone/Qdrant/the old index, and recall was 0.97."** That is agreement between two approximations, not recall. Neither is ground truth. Build a flat index.

- **"Flat search is obsolete / just for toy datasets."** Flat search is *the definition of the right answer*. It is not obsolete; it is the ruler. You retire it from *serving* at scale, never from *measurement*.

- **"Higher recall is always better."** Recall is one axis of three (recall, latency/QPS, cost). Recall@10 = 0.999 at 200 QPS and 40 GB RAM may be strictly worse for your SLO than 0.96 at 4,000 QPS and 6 GB. The goal is the cheapest point that *meets* the SLO, not the highest recall.

- **"Recall cares about order."** It does not — it is set overlap of the top-k. If ranking within the top-k matters to you, measure MRR/nDCG *in addition*, not instead.

- **Measuring QPS on a single query, or without medians.** One query is noise. The mean is dragged by outliers (GC pauses, a scheduler hiccup). Report the **median** over multiple runs, after warm-up.

- **Comparing a GPU flat search to a CPU ANN search** (or fp16 vs fp32, or different thread counts) and calling the difference "the ANN speedup." Hold the environment fixed or you are measuring the hardware, not the algorithm.

---

## Rules of thumb / cheat sheet

- **Ground truth = flat, always.** `IndexFlatIP` for normalized vectors, `IndexFlatL2` otherwise. Persist top-K (K=100 is a good generous default) to disk; it is your recall denominator.
- **State the denominator every time.** "recall@k vs flat over the same N vectors and Q queries at fixed k and metric." If you can't say that sentence, don't report a recall number.
- **Never measure ANN against ANN.** That is agreement, not recall.
- **Freeze k and metric across every row** of any comparison.
- **Normalize vectors** before an inner-product metric; assert `‖v‖ ≈ 1` at ingest. Un-normalized vectors corrupt both the index *and* the ground truth.
- **Warm up the index** with a handful of throwaway queries before timing anything (first-query JIT/allocation/page-fault spike).
- **Report median QPS** over ≥3 runs, not mean, not a single cold query.
- **Record RAM (process RSS)** alongside recall and QPS — it is the third axis and the silent budget killer.
- **Flat search is fine up to roughly ~100k–1M vectors** for offline/ground-truth use; past ~1M for *online serving* you need ANN. (Approximate thresholds — depends on d, hardware, and SLO.)
- **Regenerate ground truth whenever vectors, queries, model, or metric change.** Stamp it with a corpus/model hash.

---

## Connect to the lab

This is the foundation of the **Recall/QPS Pareto lab** (Week 2, milestone component #1). Your `ann/ground_truth.py` builds the flat `IndexFlatIP` over the eval vectors and persists exact top-100 per query — the denominator every later sweep divides by. `ann/sweep.py` reuses that frozen ground truth to score every HNSW and IVF-PQ config at *fixed* k and metric, with warm-up and median QPS, writing recall/QPS/RAM rows to `sweep.csv`. Get this lecture's discipline right first: if the ground truth is wrong or the recall denominator is another approximate index, the entire Pareto plot is fiction.

---

## Going deeper (optional)

- **FAISS wiki — "Guidelines to choose an index"** and **"Getting started" / "Faiss indexes"** (repo: `github.com/facebookresearch/faiss`, wiki tab). The canonical source for flat vs ANN index types and the index-factory string vocabulary you'll use all week (`Flat`, `IVF1024,PQ32`, `HNSW32`, `IVF4096,Flat`, etc.). Search: *"faiss wiki guidelines to choose an index"*.
- **ann-benchmarks** (`github.com/erikbern/ann-benchmarks`). The reference for how recall-vs-QPS curves are computed and presented against exact ground truth on standard datasets (SIFT-1M, GIST, GloVe). Study the methodology, not just the plots. Search: *"ann-benchmarks recall queries per second"*.
- **hnswlib README** (`github.com/nmslib/hnswlib`) — a focused view of HNSW and its knobs, useful for cross-checking FAISS's HNSW.
- **sentence-transformers docs** (root: `sbert.net`), "Semantic Search" — for the flat NumPy-argsort mental model of exact top-k before any index is involved.
- Search queries worth running: *"faiss IndexFlatIP vs IndexFlatL2 normalized"*, *"recall@k definition nearest neighbor ground truth"*, *"how to benchmark ANN QPS warmup median"*.

---

## Check yourself

1. You read "recall@10 = 0.9" in a colleague's benchmark. Give the two questions you must ask before the number means anything.
2. Why is flat search still essential even though you would never serve a 50M-vector app from it?
3. A teammate computed recall by comparing their new HNSW index to the production Qdrant index and got 0.96. What did they actually measure, and how do you fix it?
4. Ground truth was built at top-100, but you want to report recall@10. Is slicing the ground truth to the first 10 IDs per query valid? What must stay fixed?
5. Your reported QPS is 3x lower than what you see under load in prod. Name two measurement mistakes that produce exactly this symptom.
6. You switched your embedding model but recall numbers barely moved. What stale artifact should you suspect first, and why would it hide the change?

### Answer key

1. (a) **What is it measured against** — is the denominator a flat/exact index over the *same* vectors and queries? (b) **At what k and metric**, and over which vector set. Without an exact denominator at a stated k, the number is not a measurement.

2. Because flat search *is* the exact answer — it is the ruler you measure every approximate index against. You retire it from serving at scale (O(N) is too slow), but never from measurement: it produces the ground-truth top-k that recall@k divides by.

3. They measured **agreement** between two approximate indexes, not recall — both can miss different true neighbors while agreeing with each other. Fix: build a **flat** `IndexFlatIP`/`IndexFlatL2` over the identical vectors and queries, compute exact top-k, and measure both indexes against *that*.

4. Yes, slicing top-100 ground truth to the first 10 per query is valid *provided* the ordering is the exact-similarity ordering (it is, from the flat search) and you also pull top-10 from the ANN index. What must stay fixed across every reported row: **k, the metric, the vector set, the query set, and the ground-truth file**.

5. (a) **Cold index** — you timed the first query (JIT/allocation/page-fault spike) instead of warming up first. (b) **Reporting mean instead of median**, or timing a single query, so one GC/scheduler outlier dragged the number down. (Also plausible: fewer threads / different hardware than prod.)

6. Suspect a **stale ground-truth file** (`ground_truth_ids.npy`). Recall is computed against it; if it still holds the *old* model's neighbors, the ratio can stay stable even though the actual retrieval changed completely. Regenerate ground truth whenever the vectors/model/metric/queries change, stamped with a model+corpus hash.
