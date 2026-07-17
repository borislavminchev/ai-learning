# Lecture 1: Embedding Geometry and Similarity Metrics

> Every RAG pipeline, semantic search box, and agent memory you will ever build rests on one bet: that you can turn text into a point in space where *proximity means meaning* and *retrieval is nearest-neighbor search*. This lecture makes that bet precise. You will learn the exact formulas for cosine, dot product, and squared-L2; prove operationally why they rank identically once vectors are normalized (so you can pick the fastest one your index supports); read a model card to know whether to normalize; and define the three evaluation metrics — recall@k, MRR@10, nDCG@k — that are the currency of this entire phase. After this lecture you can debug a bad ranking with `np.argsort` and a calculator, and you will never again ship a metric/normalization mismatch that silently rots your recall.

**Prerequisites:** NumPy (dtype, shape, broadcasting, `np.dot`, `np.linalg.norm`) · Phase 0 Lecture 10 (Embeddings & Vector Similarity) for the intuition this lecture makes rigorous · **Reading time:** ~26 min · **Part of:** Phase 3 — Embeddings Infrastructure & Vector Databases, Week 1

---

## The core idea (plain language)

An **embedding model** is a fixed function `f: text → ℝ^d`. Feed it a string, get back the same-length list of floats every time — `d` might be 384, 768, 1024, 1536, 3072. That list is a **point** in a `d`-dimensional space. The model was trained so that texts meaning similar things land near each other and texts meaning different things land far apart. That is the *only* property you exploit, and everything downstream is a consequence of it:

- **Semantic search / retrieval** = embed the query, then find the corpus points nearest to it. This is **k-nearest-neighbor (kNN) search**.
- **The whole phase** — model choice, ANN indexes, vector DBs — is engineering to do that kNN search *fast, cheap, and honestly* over millions of points.

So the two questions this lecture answers are the load-bearing ones: **(1) how do you measure "near"?** and **(2) how do you know your "near" is any good?** Question 1 is similarity metrics and normalization. Question 2 is recall@k / MRR / nDCG. Get these two right and the rest of the phase is plumbing. Get them wrong — pick the metric the model wasn't trained for, or "eval" with a metric that doesn't match the task — and you will chase phantom quality bugs for weeks.

---

## How it actually works (mechanism, from first principles)

### A vector is a point *and* a direction

Take a text, embed it, get `v ∈ ℝ^d`. Two properties of that vector matter:

- its **direction** — which way the arrow from the origin points, and
- its **magnitude** `‖v‖ = sqrt(v₁² + v₂² + ... + v_d²)` — how long the arrow is.

For text embeddings, **meaning lives in the direction**. "cat" and "cats" should point almost the same way; the fact that one raw vector happens to be a bit longer is noise you usually want to cancel. This single fact is why cosine is the default and why normalization is the move.

### The three metrics, with exact formulas and NumPy

Given query `q` and document `d`, both in `ℝ^d`:

**1. Dot (inner) product** — component-wise multiply, then sum:

```
q · d = Σ_i q_i * d_i
```
```python
score = np.dot(q, d)          # scalar
scores = D @ q                # D: (N, d), q: (d,)  -> (N,) one per doc
```
Cost: one multiply-add per dimension, O(d). No square roots, no division — this is the **cheapest** metric, which is why every ANN index offers it and why hardware (BLAS, SIMD, GPU tensor cores) is tuned for exactly this. Downside: it rewards long vectors. A verbose document with a larger magnitude can outscore a tight, on-point one *just for being long*.

**2. Cosine similarity** — the dot product divided by both magnitudes; the cosine of the angle between the two arrows:

```
cos(q, d) = (q · d) / (‖q‖ * ‖d‖)
```
```python
cos = np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))
```
Range −1 (opposite) → 0 (orthogonal/unrelated) → 1 (same direction). Magnitude cancels, so only direction survives. This is the default for text similarity. Cost: the dot product **plus** two norms and a division.

**3. Squared Euclidean (squared-L2) distance** — straight-line distance, squared to skip the outer square root:

```
‖q − d‖² = Σ_i (q_i − d_i)²
```
```python
l2sq = np.sum((q - d) ** 2)
```
Here **smaller is closer** (0 = identical) — the opposite polarity from the other two, which is a frequent sign-flip bug when you switch metrics. We use the *squared* form because the square root is monotonic: it never changes the ranking, so paying for it is wasted cycles in retrieval.

### The unifying fact: on normalized vectors, all three rank identically

**Normalize** means scale every vector to unit length: `v̂ = v / ‖v‖`, so `‖v̂‖ = 1`. Once both `q` and `d` are unit vectors, three things collapse into one ranking.

**Cosine == dot.** By definition `cos(q,d) = (q·d)/(‖q‖‖d‖)`. If `‖q‖ = ‖d‖ = 1`, the denominator is `1`, so `cos(q,d) = q·d`. The cosine *is* the dot product — no division needed. You get cosine's magnitude-invariant meaning at dot product's speed.

**Squared-L2 = 2 − 2·cosine.** Expand the square:

```
‖q − d‖² = (q · q) − 2(q · d) + (d · d) = ‖q‖² − 2(q·d) + ‖d‖²
```
With unit vectors `‖q‖² = ‖d‖² = 1`, this becomes:

```
‖q − d‖² = 1 − 2(q·d) + 1 = 2 − 2·(q·d) = 2 − 2·cos(q,d)
```
That's an exact arithmetic identity, no approximation. Squared-L2 is a **strictly decreasing linear function** of cosine: as cosine goes up, distance goes down, on a fixed line `y = 2 − 2x`. A monotonic transform can never reorder items. So **ranking documents by highest cosine, by highest dot, or by smallest squared-L2 produces the identical order** — the same `argsort`.

```
cosine (want MAX)   0.987   0.740   0.304
dot on unit vecs =  0.987   0.740   0.304     (identical numbers)
squared-L2 = 2-2c   0.026   0.520   1.392     (want MIN; same order, reversed)
                      ↑ same document wins under all three
```

**Why you care in production:** normalize once at write time and you are free to pick whichever metric your index runs fastest — almost always **inner product**, because it's the cheapest kernel and the most hardware-optimized. You configure Qdrant/FAISS/pgvector for `dot`/`ip`, feed unit vectors, and get cosine semantics at inner-product speed. No ranking is sacrificed. This is *the* reason "normalize your embeddings" is repeated everywhere.

One caveat that bites: this equivalence holds **only** when *both* sides are unit length. Normalize the corpus but forget to normalize the query (or vice versa) and dot product no longer equals cosine — you're back to magnitude sensitivity on one side, and rankings drift.

### The argsort idiom for flat top-k

Retrieval at ground-truth (exact, no ANN) scale is one matmul plus an argsort. Learn this idiom cold; you will type it hundreds of times this phase.

```python
# D: (N, d) fp32 unit vectors, one row per document
# q: (d,)   fp32 unit vector
scores = D @ q                       # (N,) inner product == cosine
topk = np.argsort(-scores)[:k]       # indices of top-k, highest first
```
Notes that matter:
- `np.argsort` sorts **ascending**. For "highest score first" negate the scores (`-scores`) or reverse with `[::-1]`. For squared-L2 (smaller is better) you argsort the distances directly, no negation.
- For a batch of `Q` queries at once, `Q @ D.T` gives a `(Q, N)` score matrix; `np.argsort(-scores, axis=1)[:, :k]` returns top-k per row. Broadcasting and one BLAS call beat a Python loop by 10-100×.
- When you only need the top-k and `N` is large, `np.argpartition(-scores, k)[:k]` is O(N) instead of argsort's O(N log N), then sort just those k. For eval-set sizes (thousands) plain argsort is fine and clearer.

---

## Worked example

Let's do the full loop with tiny 3-dim vectors so every digit is checkable, then confirm the three metrics agree.

```python
import numpy as np

D = np.array([
    [0.8, 0.2, 0.1],   # doc 0  — on-topic
    [0.1, 0.9, 0.9],   # doc 1  — unrelated
    [0.7, 0.1, 0.1],   # doc 2  — very on-topic, but SHORT (small magnitude)
], dtype=np.float32)
q = np.array([0.9, 0.1, 0.2], dtype=np.float32)

def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

Dn, qn = normalize(D), normalize(q)

cos  = Dn @ qn                       # cosine (== dot on unit vecs)
dot  = Dn @ qn                       # literally the same op after normalizing
l2sq = np.sum((Dn - qn) ** 2, axis=1)

print("cosine :", np.round(cos, 3))    # [0.966 0.303 0.987]
print("2-2cos :", np.round(2 - 2*cos, 3))
print("l2sq   :", np.round(l2sq, 3))   # matches 2-2cos exactly

print("top-k by cosine :", np.argsort(-cos)[:3])   # [2 0 1]
print("top-k by l2sq   :", np.argsort(l2sq)[:3])   # [2 0 1]  <- identical
```

The two rankings come out **identical** (`[2, 0, 1]`): doc 2 wins, then doc 0, then doc 1. And `l2sq` equals `2 - 2*cos` to floating-point precision, confirming the identity numerically.

Now the trap. Rank on the **raw, un-normalized** dot product instead:

```python
raw = D @ q
print(np.round(raw, 3))            # [0.76  0.36  0.66]
print("top-k by raw dot:", np.argsort(-raw)[:3])   # [0 2 1]  <- doc 0 beats doc 2!
```

On raw vectors, doc 0 (larger magnitude, `‖·‖≈0.83`) leapfrogs doc 2 (`‖·‖≈0.71`) even though doc 2 points *closer* to the query in angle. That single sign of magnitude bias is exactly the class of bug normalization removes. If the model was trained for cosine and you feed raw vectors to an inner-product index, this is the silent degradation you ship.

---

## How it shows up in production

**Reading the model card is not optional — it decides normalize-or-not.** The similarity metric a model was *trained* with is the one it's calibrated for, and the card tells you which. Three cases you will meet:

- **Cosine-family (the overwhelming majority):** sentence-transformers models, OpenAI `text-embedding-3-*`, Cohere `embed-v3`, most BGE/E5/GTE. Recipe: **L2-normalize, then use dot/inner product**. sentence-transformers exposes `normalize_embeddings=True`; some models normalize internally. Configure the DB for `cosine` or (equivalently, on unit vectors) `ip`.
- **Raw dot-product models (a minority):** a handful are trained so that vector *magnitude* encodes something useful (e.g. term importance or document "confidence"), and the card explicitly says to use dot product. For these you must **NOT normalize** — normalizing throws away the magnitude signal the model deliberately learned, degrading ranking. Historically the DPR/`msmarco-*` dot-product checkpoints and some late-interaction setups fall here.
- **The card is silent:** default to normalize + cosine, then verify on your eval set (below). Silence usually means cosine, but "usually" is why you measure.

The failure mode when you get this wrong is the nasty kind: **no error, no crash, just quietly worse rankings.** Feed raw vectors to a cosine-trained model's index and long docs float to the top; normalize a dot-product model and you erase its magnitude signal. Both look like "the model is mediocre." Neither throws.

**Raw similarity SCORES are not comparable across models — only rankings and eval metrics are.** A cosine of `0.82` from `bge-small` and `0.82` from `text-embedding-3-large` mean nothing to each other. Different models learn different geometries: their score distributions have different centers and spreads, so one model's "0.8 = very similar" might be another's "0.8 = barely related." Consequences you must internalize:

- **Never hard-code a global similarity threshold** ("keep results above 0.75") and reuse it when you swap models. Re-tune the threshold per model, empirically, on your data.
- **Never compare two models by eyeballing their raw scores.** Compare them with recall@k / nDCG@k on the *same* eval set — those metrics are rank-based and *are* comparable across models. That's the entire point of an eval harness.
- Within a *single* model with a *fixed* prompt/prefix recipe, scores are comparable enough to threshold and to fuse — but the moment the model, version, or prefix changes, the scale can move.

**Engineering hygiene that prevents most metric bugs:**

- **dtype = fp32** for embeddings and for the search math. Models often emit fp32; keep it. Doing norms/dot products in fp16 accumulates rounding error that can flip near-ties in the ranking. (Storage-side int8/binary quantization is a separate, deliberate Week-2 decision with a rescore step — not an accident of dtype.)
- **shape discipline:** corpus is `(N, d)`, a query is `(d,)`, a query batch is `(Q, d)`. Get the transpose right (`D @ q` vs `Q @ D.T`) or you'll silently compute the wrong product or hit a broadcasting error.
- **normalize both sides, once.** Normalize the corpus at ingest, normalize the query at search time, with the *same* function. Then pick `ip` for speed.

---

## Evaluation metrics: the currency of this phase

You cannot say a model, index, or config is "better" without a number. These three, computed against a set of queries with known relevant document ids (`qrels`), are what every DoD in this phase reports. Define them precisely.

**recall@k — did the relevant docs make the top-k cut?** For one query, of all its relevant docs, what fraction appear in the retrieved top-k:

```
recall@k = |relevant ∩ top_k| / |relevant|
```
Report the mean over all queries. If a query has 3 relevant docs and 2 are in your top-10, recall@10 = 0.67. When each query has exactly one relevant doc (common in BEIR-style sets), recall@k is just "is the answer in the top-k, yes/no" averaged — a hit rate. Recall answers "does retrieval even surface the right thing?" — the first question for any retrieval system, because a reranker downstream can reorder the top-k but can never recover a doc that never made the cut.

**MRR@10 — Mean Reciprocal Rank.** For each query, find the rank of the *first* relevant doc; the reciprocal rank is `1/rank` (rank 1 → 1.0, rank 2 → 0.5, rank 5 → 0.2), or 0 if no relevant doc is in the top-10. Average over queries:

```
MRR@k = mean( 1 / rank_of_first_relevant )   # 0 if none in top-k
```
MRR rewards putting *a* correct answer high. It's the right lens when the user wants one good hit fast (a "feeling lucky" answer, a single citation) and cares little about the rest.

**nDCG@k — normalized Discounted Cumulative Gain.** The richest of the three: it credits relevant docs by how high they rank (a log discount) and, when you have graded relevance (0/1/2/3, not just yes/no), by *how* relevant. Discounted Cumulative Gain:

```
DCG@k = Σ_{i=1..k}  rel_i / log2(i + 1)
```
where `rel_i` is the relevance grade of the doc at position `i`. Normalize by the **ideal** DCG (the best possible ordering) to get a 0–1 score:

```
nDCG@k = DCG@k / IDCG@k
```
The `log2(i+1)` discount means position 1 is worth full credit, position 2 about 0.63, position 3 about 0.5 — later positions matter progressively less, mirroring how users actually read a results list. nDCG is the default when ordering quality and graded relevance both matter (most search-quality work).

Worked micro-example — one query, relevant ids `{A, C}`, retrieved order `[B, A, D, C, E]`, `k=5`, binary relevance:

```
recall@5 = 2/2 = 1.0                      (both relevant docs are in top-5)
MRR      = 1/2 = 0.5                       (first relevant, A, is at rank 2)
DCG@5    = 0/log2(2) + 1/log2(3) + 0 + 1/log2(5) + 0
         = 0 + 0.631 + 0 + 0.431 = 1.062
IDCG@5   = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631   (ideal: A,C first)
nDCG@5   = 1.062 / 1.631 ≈ 0.651
```
Three numbers, three questions: *did we retrieve it* (recall), *did we rank one right answer high* (MRR), *is the whole ordering good* (nDCG). You'll implement all three from scratch in the lab and unit-test them against a hand-computed toy exactly like this one.

---

## Common misconceptions & failure modes

- **"Cosine and L2 give different rankings, so I have to pick carefully."** On **normalized** vectors they give the *identical* ranking (`l2sq = 2 − 2·cos`). The choice is purely about speed then — pick inner product. It only matters if vectors aren't unit length.
- **"Normalize everything, always."** Almost always right — but a genuine raw-dot-product model's card will tell you not to, because it encodes signal in magnitude. Read the card; don't normalize on autopilot.
- **"A 0.85 similarity is high."** Meaningless without the model. Score scales differ per model; unrelated pairs might sit at 0.3 for one model and 0.6 for another. Tune thresholds per model on your data; never port a threshold across a model swap.
- **"I can compare model A and model B by their cosine scores."** No — different geometries, different scales. Compare with recall@k / nDCG@k on the same eval set. Rankings and eval metrics are comparable; raw scores are not.
- **"Argsort gives me the top-k."** `np.argsort` is **ascending** — for max-similarity you must negate or reverse, and for squared-L2 (min is best) you argsort directly. A missing minus sign silently returns the *worst* k, and it looks like a broken model.
- **"fp16 is fine for the search math."** For storage with an intentional rescore step, quantization is fine. For the ranking dot products themselves, fp16 rounding can flip near-ties — keep the scoring path fp32.
- **"High recall@k means good ranking."** Recall ignores order. You can have recall@10 = 1.0 with the right doc stuck at rank 10. If order matters, look at MRR/nDCG too.
- **"Normalizing only the corpus is enough."** The identity needs *both* sides unit-length. Un-normalized query + normalized corpus ≠ cosine.

---

## Rules of thumb / cheat sheet

- **Default recipe:** L2-normalize both query and corpus (same function, fp32), then use **inner product** in the index. You get cosine semantics at the fastest kernel. (Approximate default — verify against the model card.)
- **Metric equivalence (memorize):** on unit vectors, `cosine == dot`, and `squared-L2 = 2 − 2·cosine`. Same argsort under all three.
- **Read the card for normalize-or-not:** cosine-family → normalize; explicit raw-dot models → do **not** normalize; silent → normalize + verify.
- **Scores don't cross models.** Only rankings and eval metrics (recall@k, nDCG@k) are comparable across models. Re-tune thresholds after any model/version/prefix change.
- **Top-k idiom:** `np.argsort(-scores)[:k]` for max-similarity; `np.argsort(dists)[:k]` for min-distance. Batch with `Q @ D.T`.
- **dtype fp32** in the scoring path. Quantization is a deliberate storage decision, not a default for the math.
- **Metric-to-question:** recall@k = *did we surface it?*; MRR@10 = *is one right answer near the top?*; nDCG@k = *is the whole ordering good (with grades)?*
- **Shapes:** corpus `(N, d)`, query `(d,)`, batch `(Q, d)`. Report `d` and `N` in every benchmark.

---

## Connect to the lab

This lecture is the theory behind **Week 1 Lab tasks 4–5**. In `metrics.py` you implement `recall_at_k`, `mrr`, and `ndcg_at_k` from scratch over id lists and unit-test them against a hand-computed toy example (`test_metrics.py`) — reuse the micro-example above as your fixture. In `bench_models.py` you do **flat cosine search via `numpy` argsort** (the exact idiom here) as ground truth, normalizing embeddings so dot == cosine, and print recall@{1,5,10} and MRR@10 per model. The DoD line "you can answer in one sentence which metric your model wants" is a direct check on the model-card section — decide normalize-or-not before you benchmark.

---

## Going deeper (optional)

- **sentence-transformers documentation** (`sbert.net`) — read *Computing Embeddings*, *Semantic Search*, and the normalization/prompt pages; the canonical reference for the normalize + dot recipe.
- **FAISS wiki** (`github.com/facebookresearch/faiss/wiki`) — "Guidelines to choose an index" and the metric pages (`METRIC_INNER_PRODUCT` vs `METRIC_L2`); shows the equivalence in a real index context.
- **MTEB leaderboard** on Hugging Face Spaces (`huggingface.co/spaces/mteb/leaderboard`) and the MTEB paper — search *"MTEB Massive Text Embedding Benchmark paper"*; this is where recall/nDCG are the reported currency.
- **BEIR benchmark** (`github.com/beir-cellar/beir`) — datasets + qrels you'll use for eval; search *"BEIR heterogeneous benchmark information retrieval paper"*.
- **nDCG / IR metrics** — search *"Introduction to Information Retrieval Manning nDCG"* (the free Stanford IR book chapter on evaluation) for the authoritative definitions of recall, MRR, and DCG.
- **Model cards** for `intfloat/e5-*`, `BAAI/bge-*`, OpenAI `text-embedding-3-*`, Cohere `embed-v3` on their doc sites — the authoritative source for each model's normalization and metric recipe.

---

## Check yourself

1. Your vectors are L2-normalized. Rank order under cosine vs dot vs squared-L2 — same or different, and why exactly?
2. Write the argsort one-liner to return the top-10 document indices for a query, given `scores = D @ q` on normalized vectors. Now do it for squared-L2 distances. What changes?
3. A model card says "trained with dot-product similarity; do not normalize." You normalize anyway and feed an inner-product index. What breaks, and will you see an error?
4. Model A reports cosine scores around 0.9 for good matches; model B around 0.6. Which is the better model, and how would you actually decide?
5. For one query with relevant ids `{X}`, your retrieved order is `[A, X, B, C]`. Compute recall@3, MRR, and nDCG@3 (binary relevance).
6. Why is recall@k alone insufficient to declare a retrieval config "good," and which metric would you add?

### Answer key

1. **Identical.** On unit vectors `cos = dot` (denominator is 1), and `squared-L2 = 2 − 2·cos` is a strictly decreasing linear function of cosine — a monotonic transform, which never reorders. So the same `argsort` (with distance reversed in sign) comes out of all three. The equivalence requires *both* vectors to be unit length.
2. `topk = np.argsort(-scores)[:10]` — negate because argsort is ascending and higher similarity is better. For squared-L2: `topk = np.argsort(dists)[:10]` — no negation, because smaller distance is better. The sign flip is the whole difference; getting it wrong returns the *worst* results with no error.
3. You erase the magnitude signal the model deliberately learned (it encodes something in vector length), so rankings degrade — worse recall/nDCG. **No error fires:** normalizing and taking dot products are perfectly valid float operations; only the results are worse. You catch it only by measuring recall@k against the un-normalized recipe.
4. **Unknown from the scores alone** — raw similarity scores are not comparable across models; different geometries put good matches at different score scales. Decide by computing recall@k / nDCG@k for both on the *same* eval set with qrels; the higher rank-based metric wins, regardless of raw score magnitude.
5. Relevant `{X}`, order `[A, X, B, C]`. **recall@3** = 1/1 = 1.0 (X is within top-3). **MRR** = 1/2 = 0.5 (X is at rank 2). **nDCG@3**: DCG = 0/log2(2) + 1/log2(3) + 0 = 0.631; IDCG (X ideally at rank 1) = 1/log2(2) = 1.0; nDCG@3 = 0.631/1.0 ≈ **0.63**.
6. Recall@k ignores *order* — you can have recall@10 = 1.0 with the correct doc parked at rank 10, which is a poor user experience and often useless if a downstream step only reads the top few. Add **MRR@10** (is a right answer near the top?) or **nDCG@k** (is the whole ordering good, with graded relevance?) to capture ranking quality.
