# Lecture 10: Embeddings & Vector Similarity

> Search, RAG, recommendations, deduplication, clustering, and "find me things like this" all rest on one trick: turn text into a list of numbers so that *distance in number-space means difference in meaning*. This lecture takes the one-paragraph recap from the Week 2 spine and turns it into the real mental model. After it you can pick a similarity metric on purpose (not by copying a tutorial), read an embedding model's card and know what the prefixes and dimensions mean, compute cosine similarity by hand to debug a ranking, and avoid the three mistakes that silently wreck a vector search before you've written a single retrieval query.

**Prerequisites:** Lecture on tokenization & the transformer loop (Week 2), NumPy basics + the cosine function you wrote in Week 1 · **Reading time:** ~24 min · **Part of:** Phase 0 Week 2

---

## The core idea (plain language)

An **embedding** is a function that eats a chunk of text (a word, a sentence, a paragraph, a whole document — or an image, or audio) and spits out a fixed-length list of floating-point numbers: a **dense vector**. "Dense" means almost every number is nonzero and carries a little bit of signal, as opposed to the old "sparse" bag-of-words vectors that were mostly zeros.

The magic property — the *only* property that matters — is this: **the model is trained so that texts with similar meaning land close together in the vector space, and texts with different meaning land far apart.** Geometry becomes a proxy for semantics. "How do I reset my password?" and "I forgot my login credentials" are different strings with zero words in common, but a good embedding model places their vectors nearly on top of each other. "How do I reset my password?" and "What's the capital of Peru?" land far apart.

Once meaning is geometry, hard NLP problems collapse into cheap arithmetic:

- **Semantic search / retrieval:** embed the query, embed every document once ahead of time, return the documents whose vectors are closest. This is the retrieval half of RAG (Phase 4).
- **Clustering / dedup / topic discovery:** group vectors that are near each other.
- **Classification / routing:** is this ticket closer to the "billing" centroid or the "bug report" centroid?
- **Recommendation:** "more like this" is literally "nearest neighbors in vector space."

You do not need to understand *how* the model produces meaningful geometry to use it, but you do need to understand *how to measure closeness* and *what breaks the assumption that closeness = meaning*. That's the rest of this lecture.

---

## How it actually works (mechanism, from first principles)

### A vector is a point (and a direction) in high-dimensional space

If a model outputs 384 numbers per text, every text is one point in 384-dimensional space. You can't picture 384 dimensions, so reason in 2-D and trust that the arithmetic generalizes. A vector `[3, 4]` is an arrow from the origin `(0,0)` to the point `(3,4)`. Two things about that arrow matter:

- its **direction** (which way it points), and
- its **magnitude / length** (how far it reaches), written `‖v‖` and computed as `sqrt(3² + 4²) = 5`.

Semantic similarity almost always lives in the **direction**, not the length. "cat" and "cats" should point the same way even if one vector happens to be slightly longer.

### Where does the geometry come from? Contrastive training (intuition only)

You don't need the math, but you need the mechanism, because it explains the model's quirks. Modern text embedding models (E5, BGE, GTE, the OpenAI `text-embedding-3` family, `sentence-transformers` models) are trained with **contrastive learning**:

1. Start with a huge pile of **positive pairs** — texts that *should* be close. Sources: (question, answer), (query, clicked search result), (sentence, its paraphrase), (title, body), translation pairs.
2. For each positive pair, grab a bunch of **negatives** — texts that should be far (usually just the other examples in the same training batch: "in-batch negatives").
3. Nudge the model's weights so each positive pair's vectors move *together* and each positive-vs-negative pair moves *apart*. Repeat over billions of pairs.

```
        before training                 after contrastive training
   q •                                       q •
        d+ •      (want close, but far)          • d+   (pulled together)
                                                             •  d-
   d- •         (want far, but close)                     (pushed away)
```

Two consequences fall directly out of this and bite you later:

- **The model only "knows" the notion of similarity it was trained on.** If it was trained mostly on (question, passage) web pairs, it's great at Q&A retrieval and mediocre at, say, matching legal clauses or code snippets. This is *why you must validate on your own data* — more below.
- **Query and document are often not symmetric.** A search query ("cheap flights nyc") and a matching document ("Book affordable airfare to New York City with...") are grammatically nothing alike, yet must land close. Many models are trained to handle this asymmetry explicitly, which is where **prefixes** come in (see E5/BGE below).

### Measuring closeness: the three metrics

Given two vectors `a` and `b`, how "close" are they? Three metrics dominate.

**1. Dot product** — multiply component-wise, sum it up:
```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```
Cheap (one multiply-add per dimension). But it's sensitive to magnitude: a longer vector scores higher just for being long, even if it points the same way.

**2. Cosine similarity** — the cosine of the angle between the vectors. It's the dot product *normalized by both lengths*, which cancels magnitude and leaves only direction:
```
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```
Range: −1 (opposite) → 0 (unrelated / orthogonal) → 1 (identical direction). This is the default for text similarity because meaning is in direction, not length.

**3. Euclidean (L2) distance** — straight-line distance between the two points:
```
d(a, b) = sqrt((a₁−b₁)² + ... + (aₙ−bₙ)²)
```
Here *smaller* is closer (0 = identical), opposite polarity from the other two.

**The unifying fact every engineer must internalize:** if vectors are **normalized to unit length** (`‖v‖ = 1`, done by dividing each vector by its own magnitude), then dot product **equals** cosine similarity, and Euclidean distance becomes a monotonic function of cosine (they rank identically). So the practical rule is:

> **Normalize your vectors once, and dot product = cosine.** Then you get cosine's meaning at dot product's speed, and it doesn't matter which of the three the vector DB is configured for — they all produce the same ranking.

Many models (E5, BGE) already output normalized vectors, and `sentence-transformers` has `normalize_embeddings=True`. Vector DBs let you pick the metric (`cosine`, `dot`/`ip`, `euclidean`/`l2`). If you normalize at write time, choose dot product (`ip`) for the cheapest math.

### Dimensionality: what "384-dim" vs "1536-dim" buys you

The vector length is a model design choice. Common sizes in 2025-2026: 384 (`all-MiniLM-L6-v2`), 768 (BGE-base, E5-base), 1024 (BGE-large, E5-large), 1536 and 3072 (OpenAI `text-embedding-3-small`/`-large`).

More dimensions = more capacity to encode fine distinctions = usually (not always) higher retrieval quality, but:

- **Storage scales linearly.** A `float32` is 4 bytes. 1 million docs at 1536 dims = 1M × 1536 × 4 bytes ≈ **6.1 GB** of raw vectors. At 384 dims it's ≈ 1.5 GB. This is RAM in most vector indexes.
- **Similarity cost scales linearly.** Every comparison is O(dims). Doubling dims doubles the per-comparison work (though ANN indexes hide most of it).

So dimensionality is a quality-vs-cost dial, which leads straight to Matryoshka.

### Matryoshka (MRL): truncate the tail, keep most of the quality

**Matryoshka Representation Learning** trains a model so that the *most important information is packed into the earliest dimensions*. A 1024-dim Matryoshka vector isn't 1024 equally-important numbers — it's more like a nested doll: the first 64 carry the coarse meaning, the next chunk refines it, and so on.

The payoff: you can **truncate** a Matryoshka vector — literally slice `v[:256]` — and it still works as an embedding, at a fraction of the storage and compute, losing only a little quality. (Re-normalize after truncating.) OpenAI's `text-embedding-3` models support this via a `dimensions` parameter; many BGE/Nomic models are Matryoshka-trained. It is *not* valid to truncate a non-Matryoshka model's output — you'd be throwing away random information.

---

## Worked example

Let's rank two documents against a query, by hand, with tiny 3-dim vectors so you can follow every digit. (Real vectors are hundreds of dims, but the arithmetic is identical.)

```
query  q  = [0.9, 0.1, 0.2]
doc A  a  = [0.8, 0.2, 0.1]     # "similar meaning" doc
doc B  b  = [0.1, 0.9, 0.9]     # unrelated doc
```

**Step 1 — dot products (raw):**
```
q · a = (0.9)(0.8) + (0.1)(0.2) + (0.2)(0.1) = 0.72 + 0.02 + 0.02 = 0.76
q · b = (0.9)(0.1) + (0.1)(0.9) + (0.2)(0.9) = 0.09 + 0.09 + 0.18 = 0.36
```

**Step 2 — magnitudes:**
```
‖q‖ = sqrt(0.9² + 0.1² + 0.2²) = sqrt(0.81 + 0.01 + 0.04) = sqrt(0.86) ≈ 0.927
‖a‖ = sqrt(0.8² + 0.2² + 0.1²) = sqrt(0.64 + 0.04 + 0.01) = sqrt(0.69) ≈ 0.831
‖b‖ = sqrt(0.1² + 0.9² + 0.9²) = sqrt(0.01 + 0.81 + 0.81) = sqrt(1.63) ≈ 1.277
```

**Step 3 — cosine similarity:**
```
cos(q, a) = 0.76 / (0.927 × 0.831) = 0.76 / 0.770 ≈ 0.987
cos(q, b) = 0.36 / (0.927 × 1.277) = 0.36 / 1.184 ≈ 0.304
```

Doc A wins decisively (0.987 vs 0.304) — it points almost the same direction as the query. Notice how normalization mattered: doc B's raw dot product (0.36) was penalized further once we divided by its large magnitude (1.277). If you'd ranked on raw dot product of *un-normalized* vectors, a long, verbose document could outrank a tight, on-point one purely for being long — a classic bug.

**In code (the function you'll reuse from Week 1):**
```python
import numpy as np

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

q = np.array([0.9, 0.1, 0.2])
docs = {"A": np.array([0.8, 0.2, 0.1]), "B": np.array([0.1, 0.9, 0.9])}
ranked = sorted(docs, key=lambda d: cosine(q, docs[d]), reverse=True)
# → ['A', 'B']
```

And with a real model, the end-to-end shape:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/e5-small-v2")

# E5 wants task prefixes (see production section):
q_vec  = model.encode("query: how do I reset my password?", normalize_embeddings=True)
d_vecs = model.encode([f"passage: {d}" for d in documents], normalize_embeddings=True)
scores = d_vecs @ q_vec          # normalized → dot product == cosine
```

---

## How it shows up in production

**Cost and latency are dominated by the offline corpus, not the query.** Embedding one query is a single fast model call. Embedding 5 million documents is a batch job that costs real money (hosted embedding APIs bill per token) or real GPU-hours (self-hosted). Plan for it: chunk your docs, batch the encode calls, and **cache aggressively** — never re-embed a document that hasn't changed. Storage follows from dimensionality (the 6 GB / 1M-docs-at-1536 number above): this is the RAM your vector index needs, and it's why teams reach for Matryoshka truncation or int8 quantization of vectors.

**Query/document asymmetry and prefixes will silently halve your recall if you skip them.** The E5 and BGE families were trained with instruction prefixes and *expect them at inference time*:

- **E5:** prepend `"query: "` to searches and `"passage: "` to documents.
- **BGE:** prepend a query instruction like `"Represent this sentence for searching relevant passages: "` to the query; documents usually go in bare. (BGE v1.5 relaxed this, but check the model card.)

Forget the prefix and the model is running out-of-distribution — no error, no warning, just quietly worse rankings. This is one of the most common "my semantic search is mediocre and I don't know why" bugs. **Read the model card and follow its exact recipe.**

**The same-model rule is non-negotiable.** Query vectors and document vectors must come from the **same model, same version, same normalization, same prefixes**. Two different models produce vectors in two different coordinate systems; comparing them with cosine is like comparing GPS coordinates to street addresses — the numbers are meaningless together. Practical consequences:

- If you upgrade your embedding model, you must **re-embed the entire corpus**. There's no incremental migration. Version your index and do a full re-index behind a flag.
- Don't mix a "small" model for queries (to save latency) with a "large" model for the corpus. Same model everywhere.

**MTEB tells you where to *start*, not what to *ship*.** The [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) ranks embedding models across dozens of tasks. Use it to build a shortlist. But it's an *average over public academic datasets* that look nothing like your support tickets, your legal contracts, or your codebase. Models also get quietly tuned toward leaderboard tasks. **You must build a small eval set from your own data** (a few dozen realistic query→relevant-doc pairs) and measure recall@k / nDCG@k (Week 1 metrics) on the candidate models. The MTEB #1 model routinely loses to a #15 model on a specific real workload. Budget an afternoon for this; it's the highest-leverage hour in a RAG project.

**Debugging checklist when retrieval is bad:** (1) Are query and corpus embedded by the *same* model? (2) Are the *prefixes* right? (3) Are vectors *normalized* and is the DB metric consistent with that? (4) Is your *chunking* sane (not 1-word or whole-book chunks)? Blame these before you blame the model.

---

## Common misconceptions & failure modes

- **"Cosine and Euclidean give different rankings."** On **normalized** vectors they give the *identical* ranking. The choice only matters if your vectors aren't unit length — so normalize and stop worrying.
- **"Higher-dimensional is always better."** More dims cost more storage/compute and can *overfit* to noise on small corpora. A well-trained 384-dim model often beats a mediocre 1536-dim one. Test, don't assume.
- **"I can truncate any embedding to save space."** Only **Matryoshka-trained** models survive truncation. Slicing a normal model's vector destroys it. And after any truncation, re-normalize.
- **"Cosine ~0 means opposite."** No — ~0 means *unrelated/orthogonal*. −1 means opposite. For most text-embedding models trained with contrastive loss, real scores rarely go negative; unrelated pairs sit around 0–0.3 and you tune your threshold empirically per model (a "0.8 is similar" rule for one model is meaningless for another).
- **"The MTEB #1 is the best choice for me."** It's the best on *average public data*. Your data isn't average or public.
- **"I forgot the E5/BGE prefix but it still returns results."** It always returns results — nearest neighbors always exist. The failure is *silent quality loss*, which is the worst kind.
- **"Embeddings understand exact facts / numbers / negation."** They capture topical/semantic gist. "The drug is safe" and "the drug is not safe" can land close because they're about the same topic. Don't rely on embeddings alone for negation, precise IDs, or exact-match — combine with keyword/lexical (BM25) search, i.e. **hybrid retrieval** (Phase 4).

---

## Rules of thumb / cheat sheet

- **Default metric:** normalize vectors, then use cosine (== dot product on normalized vectors). Set your vector DB to `cosine` or `ip` accordingly.
- **Always normalize** unless you have a specific reason not to. `normalize_embeddings=True` in `sentence-transformers`.
- **Same model + version + prefixes + normalization** for queries and corpus. Upgrading the model = full re-index.
- **Follow the model card's prefix recipe exactly** (E5: `query:` / `passage:`; BGE: query instruction). This is free recall.
- **Storage math:** `docs × dims × 4 bytes` for float32. 1M × 1536 ≈ 6 GB. Use this to size RAM.
- **Dimensionality dial:** start with 384–768 dims for cost-sensitive/CPU work; 1024–1536 when quality matters and you can pay. Test both on your data.
- **Matryoshka truncation** (`v[:256]`, then re-normalize) trades a little quality for big storage/latency wins — only on MRL-trained models.
- **Shortlist from MTEB, decide on your own eval set** (recall@k / nDCG@k on 20–50 real query→doc pairs). Never ship on leaderboard rank alone.
- **Embeddings ≠ exact match.** Pair with BM25/keyword for IDs, numbers, negation → hybrid search.

---

## Connect to the lab

This lecture is the theory behind **Week 2, Lab task 4 (Embeddings + semantic ranking)**. There you'll load `all-MiniLM-L6-v2` (384-dim, CPU-friendly), embed a query plus ~10 short docs, and reuse your Week-1 NumPy `cosine` function to rank them — expect the obviously-relevant doc at rank 1 (Definition of Done). Then you'll re-embed with a Matryoshka-capable model truncated to 128 dims and confirm the top result barely moves. Watch for two things: (1) if your model is E5/BGE, add the `query:`/`passage:` prefixes or your ranking will look mushy; (2) pass `normalize_embeddings=True` so your dot product and cosine agree. It also underpins Self-check Q5 ("semantic search returns garbage — name three model/embedding causes"): same-model rule, prefixes, normalization.

---

## Going deeper (optional)

- **sentence-transformers documentation** (sbert.net) — the canonical Python library; read *Semantic Search*, *Computing Embeddings*, and the *Matryoshka* and *Prompt/Prefix* pages.
- **MTEB leaderboard** on Hugging Face Spaces (`huggingface.co/spaces/mteb/leaderboard`) and the MTEB paper — search: *"MTEB Massive Text Embedding Benchmark paper"*.
- **Matryoshka Representation Learning** — the original paper; search: *"Matryoshka Representation Learning arXiv"*. Also the sbert.net Matryoshka how-to.
- **E5 and BGE model cards** on Hugging Face (`intfloat/e5-*`, `BAAI/bge-*`) — the authoritative source for the exact prefixes and normalization each model expects. Search: *"E5 text embeddings paper microsoft"* and *"BGE BAAI general embedding"*.
- **OpenAI Embeddings guide** (platform.openai.com/docs) — covers `text-embedding-3` models and the `dimensions` (Matryoshka) parameter.
- **Cohere / Nils Reimers talks on embeddings and semantic search** — search: *"Nils Reimers embeddings talk"* for excellent engineer-level intuition.
- For the retrieval infrastructure this feeds into (ANN indexes, HNSW, vector DBs), that's **Phase 3**; hybrid search and reranking are **Phase 4**.

---

## Check yourself

1. Why is cosine similarity usually preferred over raw dot product for text, and under what condition do the two become identical?
2. You embed your 2M-document corpus with model X, then a teammate embeds incoming queries with model Y "because it's faster." What breaks, and why is there no error message?
3. A model card says: `query: {text}` for searches, `passage: {text}` for documents. You skip the prefixes. What symptom do you observe, and why is it hard to notice?
4. You have 5M documents and only 8 GB of RAM for the index. Your candidate model is 1024-dim and Matryoshka-trained. What lever do you pull, and roughly what's the storage before and after?
5. Model A is #1 on MTEB; model B is #12. On your internal FAQ eval set, B gets recall@5 = 0.91 and A gets 0.78. Which do you ship, and what does this tell you about leaderboards?
6. Two support messages — "the payment went through fine" and "the payment did not go through" — get a cosine similarity of 0.86. Is the embedding model broken? What does this reveal about what embeddings capture, and what would you add to disambiguate them?

### Answer key

1. Meaning lives in a vector's **direction**, not its length; cosine ignores magnitude while raw dot product rewards longer vectors regardless of direction, so a verbose doc can outrank a precise one. They become identical when both vectors are **normalized to unit length** — then `‖a‖‖b‖ = 1` and cosine reduces to the dot product.
2. Vectors from two different models live in **different, incompatible coordinate systems**, so cosine between them is meaningless — rankings become effectively random/garbage. No error fires because the math (a dot product of two equal-length float arrays) is perfectly valid; only the *results* are nonsense. Fix: same model, version, prefixes, normalization for both sides.
3. Retrieval quality drops (lower recall, mushier rankings) but the system still returns results. It's hard to notice because nearest neighbors always exist and the output *looks* plausible — you only catch it by measuring recall@k/nDCG@k on a real eval set. The model is running out-of-distribution because it was trained expecting those prefixes.
4. Use **Matryoshka truncation**: slice to a smaller prefix (e.g., 256 dims) and re-normalize. Before: 5M × 1024 × 4 bytes ≈ **20.5 GB** (won't fit). After at 256 dims: 5M × 256 × 4 ≈ **5.1 GB** (fits 8 GB with headroom), for a small quality cost you'd verify on your eval set. (int8 quantizing the vectors is another lever.)
5. Ship **B**. MTEB ranks average performance on public academic tasks that don't resemble your FAQ data; the only number that matters is performance on *your* eval set. Leaderboards are for building a shortlist, not making the final call — always validate on your own data.
6. The model isn't broken. Embeddings capture **topical/semantic gist**, and both messages are about the same topic (a payment) — negation is exactly what dense embeddings handle poorly. To disambiguate, combine embeddings with lexical/keyword signals (**hybrid search**, e.g. BM25), and for true intent add a classifier or an LLM step; don't rely on cosine similarity alone for negation or exact facts.
