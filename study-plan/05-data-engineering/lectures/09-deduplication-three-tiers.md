# Lecture 9: Deduplication at Three Tiers — Exact Hash, MinHash-LSH, and Semantic

> Your corpus is full of duplicates you cannot see. The same press release syndicated to forty sites. The same doc re-ingested last Tuesday with a fresh timestamp. The same paragraph laundered through a paraphrase tool by a content farm. Left in, they burn embedding dollars, poison retrieval (identical chunks crowd out diverse context), and inflate fine-tuning loss on memorized text. This lecture teaches the dedup strategy FineWeb-scale pipelines actually run: a **cascade** of three tiers, cheapest first, where each tier only inspects what the previous one could not kill. After this you will be able to build a dedup stage that erases near-100% of exact dupes for pennies, catches near-dupes at millions-of-docs scale with MinHash-LSH, reserves expensive semantic dedup for the paraphrases that survive, and — the part that makes the whole thing defensible — *report how many additional dupes each tier bought you*, so you can prove the last, expensive tier was worth its cost.

**Prerequisites:** Lecture 8 (normalization/cleaning), Phase 0–4 (embeddings, cosine similarity, a vector DB), basic hashing and big-O, simple probability · **Reading time:** ~26 min · **Part of:** Phase 5 — Data Engineering for AI, Week 2

## The core idea (plain language)

Deduplication is a spectrum, not a checkbox. Two documents can be "the same" in three progressively looser senses:

1. **Byte-identical** — after you normalize whitespace, case, and Unicode form, the text is *exactly* the same. Two ingests of the same page. A CDC re-emit. Trivially detectable by comparing one hash.
2. **Near-identical** — 85%+ of the content overlaps, but a header changed, a date got bumped, a "share this" footer was appended, three words were swapped. Same document, cosmetically different. This is the bulk of real-world web dupes.
3. **Semantically equivalent** — different words, same meaning. A paraphrase, a translate-and-back, an LLM rewrite. Near-zero token overlap is possible; only *meaning* overlaps.

Each sense costs strictly more to detect than the last. Exact match is a dictionary lookup — O(n) over total bytes, microseconds per doc. Near-dup detection needs approximate set-similarity — still cheap with the right trick. Semantic dedup needs a neural embedding per document plus a similarity search — orders of magnitude more expensive in compute, memory, and wall-clock.

The whole lecture rests on one engineering decision: **run the tiers in that order, and let each tier shrink the input to the next.** Exact hashing might remove 20% of your corpus for almost nothing. MinHash-LSH removes another 12% cheaply. Only *then* do you embed the survivors for semantic dedup — because embedding is the cost bomb, and you want to detonate it over the smallest possible set. Running semantic dedup first, over the raw millions, is the classic junior mistake: correct output, catastrophic bill. The cascade is not about *whether* you catch the dupes; all three orders find roughly the same dupes. It is about *what you pay* to catch them.

## How it actually works (mechanism, from first principles)

### Tier 0: Normalization (the step everyone skips and regrets)

Before *any* tier, normalize. Otherwise `"Hello  World"` and `"hello world"` hash differently and your exact tier catches nothing. Lecture 8 already did the heavy lifting (`ftfy` mojibake repair, NFC), so at minimum:

- Lowercase — unless case is semantically load-bearing (source code, IDs, product SKUs).
- Collapse runs of whitespace to a single space; strip leading/trailing.
- Ensure Unicode NFC so `é` (one codepoint) and `é` (e + combining accent) are the same bytes.
- Optionally strip zero-width characters and HTML entities.

Normalization is a *policy decision*, not a fixed recipe. Aggressive normalization (strip punctuation, stem words) finds more dupes but risks merging genuinely different things. Log what you normalize so a surprising dedup count is debuggable. **The single most common bug report — "exact dedup found almost nothing on a corpus I know has re-crawls" — is a normalization miss**, not a hashing bug.

### Tier 1: Exact hash

Hash the normalized text; two docs with the same hash are byte-identical dupes. Keep one, drop the rest. Which hash?

- **`xxhash` (xxh3_64 / xxh3_128)** — a *non-cryptographic* hash. Blazing fast (multiple GB/s), 64- or 128-bit output. Use it when all you need is "are these bytes equal," which is exactly what dedup asks. This is the default for scale.
- **`sha256`** — cryptographic, slower, 256-bit. Use it when the hash *doubles as a content-addressable ID* you persist for provenance/lineage, or when you fear adversarial collisions. In this pipeline `sha256` already appears in the Week 1 manifest for integrity; `xxhash` is the right pick for the dedup hot loop.

Collision reality check: with 64-bit xxhash the odds of an *accidental* collision between two different documents stay astronomically small until you approach billions of docs (birthday bound ≈ 2^32 ≈ 4 billion for a ~50% chance of a *single* collision). Under ~100M docs, xxh3_64 is fine. At FineWeb scale (tens of billions), move to 128-bit.

```python
import xxhash

def exact_key(normalized_text: str) -> str:
    return xxhash.xxh3_64_hexdigest(normalized_text.encode("utf-8"))
```

Cost: one pass, O(n) in total bytes, a `set` of seen hashes. Memory is ~8–16 bytes per unique doc plus bookkeeping. This is the cheapest tier by a mile and it should always run first — skipping it just makes the next tier do more work over a bigger set.

### Tier 2: MinHash + LSH (near-duplicate detection)

Exact hashing is deliberately brittle: change one byte and the hash is completely different — that is the *point* of a hash. So it misses the huge class of near-dupes. We need a similarity measure that degrades *gracefully*.

**Step 1 — Shingling.** Turn each document into a *set* of overlapping token sequences ("shingles" or "n-grams"). A word-level 3-shingle of "the quick brown fox jumps":

```
{"the quick brown", "quick brown fox", "brown fox jumps"}
```

Now document similarity = *set* similarity. The standard measure is **Jaccard**:

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

If A and B share 90 of 100 total distinct shingles, J = 0.90. Swap a few words and most shingles still match — Jaccard degrades gracefully exactly where exact hash falls off a cliff. Shingle size is a knob: smaller shingles (1–2 words) find looser matches but false-positive on common phrases; larger (5+ words) are stricter. **3–5 word shingles** (or ~5-char n-grams for CJK / no-whitespace languages) is the usual default.

**Step 2 — MinHash: a cheap Jaccard estimator.** Computing exact Jaccard for every pair is O(n²) pairs times set-intersection cost — hopeless at scale. MinHash replaces each document's shingle set with a small fixed-length *signature* (say 128 integers) with one magic property: **the fraction of signature positions where two docs agree is an unbiased estimate of their Jaccard similarity.**

The intuition (no proof): apply a hash function to every shingle and keep only the *minimum* hash value. Two sets are more likely to produce the same minimum when they overlap more — because the minimum is more likely to have come from a shingle they *share*. Repeat with 128 independent hash functions and you get a 128-slot signature. Count matching slots, divide by 128, and that ratio ≈ Jaccard. It is a Monte-Carlo estimate: more slots (`num_perm`) = tighter estimate but bigger signatures and slower hashing.

```
Doc A shingles ─► 128 hash fns ─► [min₁, min₂, …, min₁₂₈]   (signature A)
Doc B shingles ─► same 128 fns ─► [min₁, min₂, …, min₁₂₈]   (signature B)
matching slots / 128  ≈  J(A, B)
```

`num_perm=128` is the practical sweet spot: estimation error roughly ±0.03–0.04, signatures small enough to hold millions in RAM. Going to 256 halves the error but doubles signature size and hashing cost — rarely worth it for dedup. Below 64, the estimate gets noisy enough to cause visible false positives/negatives.

**Step 3 — LSH banding: candidates without O(n²).** MinHash shrank each doc to 128 numbers, but comparing every pair of *signatures* is still O(n²). Locality-Sensitive Hashing fixes this. Split the 128-slot signature into **b bands of r rows each** (e.g. b=16 × r=8 = 128). Hash each band into a bucket. **Two docs are "candidates" if they land in the same bucket for *at least one* band.**

Why it works: similar docs share many slots, so they are likely to have at least one *entire band* identical and collide in a bucket. Dissimilar docs rarely match a full band. The b/r split places the effective threshold — more bands = lower threshold (looser dupes, more false positives); fewer bands = higher threshold (stricter). In practice you don't set bands by hand: `datasketch`'s `MinHashLSH(threshold=0.85, num_perm=128)` derives a b/r split that approximates your Jaccard threshold.

The payoff: LSH converts "compare everything to everything" into "hash into buckets, only compare within buckets" — near-linear in practice. This is precisely what makes billion-doc dedup tractable.

```python
from datasketch import MinHash, MinHashLSH

def minhash(shingles, num_perm=128):
    m = MinHash(num_perm=num_perm)   # fixed permutations -> reproducible signatures
    for sh in shingles:
        m.update(sh.encode("utf-8"))
    return m

lsh = MinHashLSH(threshold=0.85, num_perm=128)
sigs = {}
for doc_id, text in corpus:
    m = minhash(shingle(text))
    sigs[doc_id] = m
    lsh.insert(doc_id, m)               # build the index

candidates = lsh.query(sigs[doc_id])    # itself + its near-dupes
```

**Step 4 — Cluster resolution.** LSH gives you *candidate pairs*. Build a graph (docs = nodes, "near-dup" candidate pairs = edges), find connected components — those are your near-dup clusters. Then apply a **keep policy**. The standard for corpus quality is **keep the longest member** (most content, least likely to be a truncated or boilerplate fragment) and drop the rest. Alternatives: keep the earliest-ingested (provenance stability) or highest quality-score. Whatever you choose, record *which* survivor represented each cluster so every drop is auditable.

A subtlety FineWeb-scale pipelines hit: **LSH candidates are approximate.** Some pairs collide in a bucket but sit *below* your true threshold. If you need precision, re-check candidates' estimated Jaccard (`m1.jaccard(m2)`) and only merge above threshold. `datatrove` and FineWeb do exactly this two-stage dance: **LSH proposes, an estimated-Jaccard check disposes.**

### Tier 3: Semantic dedup

MinHash operates on *tokens*. Paraphrases share meaning but not tokens: "The cat sat on the mat" vs "A feline rested upon the rug" have Jaccard ≈ 0. MinHash is *structurally* blind to them. To catch these you must compare *meaning*, which means embeddings.

**Mechanism:** embed each surviving document with a sentence-embedding model — `all-MiniLM-L6-v2` (384-dim, tiny, fast) or `bge-small-en-v1.5` (384-dim, stronger retrieval quality) are the standard cheap workhorses. Two docs are semantic dupes if their embeddings' **cosine similarity exceeds a high threshold — ~0.95**. That threshold must be *high*: at 0.85 you merge documents that are merely *about the same topic* (two distinct articles on the same election), which is not duplication — it is your corpus doing its job. 0.95+ means "these say the same thing."

Finding high-similarity pairs is itself a nearest-neighbor problem. Do **not** brute-force O(n²) cosine — reuse the ANN index you already run in Phase 4 (Qdrant, or FAISS/`hnswlib` in-process): index the survivor embeddings, query each for neighbors above 0.95, cluster, keep-longest, drop. Same cluster-resolution logic as Tier 2.

Why this is the cost bomb: every document needs a forward pass through a neural net. On CPU, `all-MiniLM-L6-v2` does maybe a few hundred short docs/sec; on a GPU, thousands. Either way it is *orders of magnitude* slower and more resource-hungry than hashing, plus you store a 384-float vector (~1.5 KB) per doc and stand up an ANN index. Over 10M docs that is ~15 GB of vectors and hours of GPU time. Over the *raw* pre-dedup corpus it could be several times that — spent finding dupes the cheap tiers would have killed for free.

## Worked example

Take a corpus of **1,000,000 crawled web documents** and watch the cascade. (These proportions are illustrative — plausible for messy web data, *not* a benchmark citation.)

**Tier 1 — Exact hash.** Normalize + xxh3. Say 18% are byte-identical (re-crawls, syndicated identical copies, CDC re-emits).

```
in:  1,000,000
exact dupes removed: 180,000
out:   820,000        cost: seconds, a few hundred MB RAM
```

**Tier 2 — MinHash-LSH** over the 820k survivors. 3-word shingles, `num_perm=128`, threshold 0.85. Say near-dup clusters collapse another 12% (boilerplate-wrapped copies, minor edits, timestamp-differing reposts).

```
in:    820,000
near-dupes removed: ~98,400
out:   721,600        cost: minutes, signatures ~ 820k × 128 × 8B ≈ 840 MB
```

Note we ran MinHash over 820k, **not** 1,000,000 — the exact tier already saved us hashing 180k docs.

**Tier 3 — Semantic** over the 721k survivors. Embed with `all-MiniLM-L6-v2`, ANN search, cosine > 0.95. Say it catches an *additional* 3% — paraphrases and rewrites the token-based tiers structurally could not see.

```
in:    721,600
semantic dupes removed: ~21,600
out:   700,000        cost: GPU-hours + 721k × 384 × 4B ≈ 1.1 GB vectors + ANN index
```

**The report that justifies the cascade** — this is the deliverable the lab asks for:

| Tier | Input | Removed | % of input | Cost class |
|------|-------|---------|-----------|-----------|
| 1 Exact | 1,000,000 | 180,000 | 18.0% | ~free |
| 2 MinHash-LSH | 820,000 | 98,400 | 12.0% | cheap |
| 3 Semantic | 721,600 | 21,600 | 3.0% | expensive |

The argument writes itself. Run semantic dedup *first* and you embed all 1,000,000 docs — a ~39% larger embedding bill and index — to find the *same* 21,600 semantic dupes, because the 278k exact+near dupes would just show up as trivially-high-cosine pairs the cheap tiers already kill for free. Same result, multiples of the cost. And notice Tier 3 caught only 3% *additional*: with that number in hand you decide whether the GPU-hours are worth it *for your use case*. For pretraining corpora at FineWeb scale it often is (memorized dupes measurably hurt). For a 50k-doc internal RAG corpus, Tier 3 rarely pays for itself — stop after Tier 2.

## How it shows up in production

- **The embedding bill is the whole game.** Semantic dedup's cost scales with docs-embedded. The cascade exists to minimize that number. A team that runs embedding dedup on the raw corpus "because it's the most thorough" pays 2–5× for a near-identical result. When someone proposes semantic-first, demand the tier-by-tier marginal-catch numbers *before* approving the spend.
- **Retrieval quality, not just cost.** Near-dupes in a RAG index are *actively harmful*: a query retrieves top-k, and if three of five slots are the same syndicated paragraph, you have starved the LLM of diverse context. Dedup is a retrieval-quality lever, not merely a storage one.
- **Fine-tuning and eval contamination.** Duplicated training examples get over-weighted (the model memorizes them); dupes that straddle your train/eval split silently leak answers and inflate metrics. **Dedup before you split.** This is a core reason FineWeb dedups hard.
- **MinHash memory is the scaling wall.** 128 int64s per doc ≈ 1 KB; a billion docs ≈ 1 TB of signatures. Beyond single-machine RAM you shard LSH by band-bucket across workers (what `datatrove` does with its distributed MinHash stages) or persist signatures to disk/DB. Know your corpus size before choosing in-memory `datasketch` vs a sharded pipeline.
- **Threshold drift bites silently.** A Jaccard threshold tuned on English news over- or under-merges on code, legal text, or another language (shingle statistics differ). Re-validate thresholds per domain; don't copy 0.85 blindly across corpora.
- **Debuggability is a compliance requirement.** When someone asks "why did doc X vanish?", you must answer "near-dup of Y, cluster kept Y as longest member." Log cluster membership and the keep decision. A dedup stage that silently deletes with no audit trail is un-debuggable and, for governed data, non-compliant.

## Common misconceptions & failure modes

- **"MinHash gives me exact Jaccard."** No — it's an *estimate* with ±0.03-ish noise at 128 perms. Near the threshold, expect occasional false pos/neg. Need precision? Re-check candidate pairs with `.jaccard()` (LSH-proposes / Jaccard-disposes).
- **"LSH finds all near-dupes."** LSH trades recall for speed. A pair just above threshold *might* not collide in any band and get missed. b/r tuning shifts where this happens; there's no free lunch. Small recall loss is fine for corpus dedup — not for legal e-discovery.
- **"Higher `num_perm` is always better."** Diminishing returns. 128 → 256 halves estimation error but doubles memory and hashing time. Rarely worth it for dedup; spend the budget elsewhere.
- **"Semantic dedup at 0.85 cosine."** That merges *topically related* docs, destroying corpus diversity. Semantic dupes live at **0.95+**. 0.85 cosine is "same topic," which is not a duplicate.
- **"Skip exact hashing, MinHash covers it."** Exact is ~free and removes the biggest, cheapest chunk. Skipping it just makes MinHash work over a bigger set. Always run it first.
- **Forgetting to normalize.** The #1 reason "exact dedup found almost nothing." Whitespace/case/Unicode differences make identical content hash differently.
- **Non-deterministic MinHash across runs.** `datasketch` MinHash needs consistent permutations (a fixed seed) to produce comparable signatures across runs. Different seeds → incomparable signatures → your incremental dedup silently breaks.

## Rules of thumb / cheat sheet

- **Always cascade cheap → expensive:** normalize → exact hash → MinHash-LSH → (only if justified) semantic.
- **Exact tier:** `xxhash` (xxh3_64) for the hot loop; `sha256` only when the hash is also a stored content ID. Fine to ~100M docs on 64-bit; go 128-bit beyond.
- **Shingles:** 3–5 words (or ~5-char n-grams for CJK / no-whitespace). Smaller = looser + more false positives.
- **MinHash:** `num_perm=128` is the default sweet spot. Fixed permutations/seed for reproducibility.
- **Near-dup threshold:** Jaccard **~0.85** for "same doc, cosmetic diffs." Let `datasketch` derive the band split from the threshold.
- **Cluster resolution:** connected components → **keep the longest member**; log the survivor.
- **Semantic tier:** `all-MiniLM-L6-v2` (fast) or `bge-small-en-v1.5` (stronger); cosine **> 0.95**; use an ANN index, never brute-force O(n²).
- **Always report marginal catch per tier** (`removed / input`). If Tier 3's marginal catch is tiny and your use case tolerates it, skip Tier 3.
- **Memory math:** MinHash ≈ 1 KB/doc (128×int64); MiniLM embedding ≈ 1.5 KB/doc (384×float32). Multiply by corpus size *before* choosing in-memory vs sharded.
- **Dedup before the train/eval split** to prevent contamination.

*(All specific percentages above are illustrative order-of-magnitude figures, not measured benchmarks.)*

## Connect to the lab

This is the theory behind `dedup.py` in Week 2's extraction/cleaning stage. Your lab builds exactly this cascade: an `xxhash` exact pass, then `datasketch` `MinHashLSH(num_perm=128, threshold=0.85)` over shingles keeping the longest cluster member, then an optional `bge-small` / `all-MiniLM-L6-v2` semantic pass at cosine > 0.95. The Definition of Done requires you to **report both counts** — exact dupes, and the *additional* near-dupes MinHash finds that exact-hash missed — and to write the cost-vs-benefit note on whether the semantic tier earned its keep. That report *is* the cascade argument from the worked example, computed on your own corpus.

## Going deeper (optional)

- **datasketch documentation** — the canonical `MinHash` / `MinHashLSH` reference (also `LeanMinHash`, `MinHashLSHForest` for top-k). Search: *datasketch MinHash LSH documentation*.
- **HuggingFace FineWeb** — the blog/dataset card documents production dedup decisions at trillion-token scale (why they dedup per-dump, threshold choices). Search: *HuggingFace FineWeb dataset blog dedup*.
- **`datatrove`** (HuggingFace) — the open-source library FineWeb was built with; read its MinHash pipeline stages for how distributed, sharded near-dup dedup is engineered. Search: *huggingface datatrove github minhash*.
- **"Mining of Massive Datasets" (Leskovec, Rajaraman, Ullman)** — Chapter 3, "Finding Similar Items," is the definitive, readable treatment of shingling, MinHash, and LSH banding. Free PDF on the book's site. Search: *Mining of Massive Datasets chapter 3 finding similar items*.
- **SemDeDup (Meta AI)** — embedding-based semantic dedup for large text/image corpora and its downstream effects. Search: *SemDeDup semantic deduplication paper*.
- **sentence-transformers documentation** — `all-MiniLM-L6-v2` / `bge-small` usage plus `util.paraphrase_mining` / community-detection helpers useful for the semantic tier. Search: *sentence-transformers pretrained models documentation*.

## Check yourself

1. Why does the cascade run exact → MinHash → semantic in *that* order, and what specifically goes wrong (in dollars) if you run semantic first?
2. Explain in one sentence why the fraction of matching MinHash signature slots approximates Jaccard similarity — without deriving any probability.
3. You raise `num_perm` from 128 to 512. What improves, what gets worse, and why is this usually a bad trade for dedup?
4. Give a concrete pair of documents that MinHash-LSH will *never* flag as duplicates but semantic dedup will. Why is MinHash structurally blind to them?
5. Your exact-hash tier removes almost nothing on a corpus you're *sure* has re-crawled pages. What's the most likely bug, and where do you look?
6. Why is semantic dedup at cosine 0.85 a mistake, and what would it do to a RAG corpus's answer quality?

### Answer key

1. Each tier shrinks the input to the next, and cost per doc rises steeply across tiers (hash ≈ free, MinHash cheap, embedding expensive). Semantic-first means you embed the *entire* raw corpus — including all the exact and near dupes the cheap tiers would kill for free — so you pay for millions of embeddings and a bigger ANN index to find the *same* semantic dupes you'd have found over the smaller survivor set. Same result, multiples of the cost.
2. Applying a random hash to a set and keeping the minimum makes that minimum more likely to come from a shingle the two sets *share* the more they overlap; averaged over 128 independent hashes, the agreement rate converges on the overlap ratio, which is Jaccard.
3. Estimation error shrinks (roughly halves 128→512, since error ~ 1/√num_perm), so thresholds are sharper — but signature memory and hashing time grow 4×. For dedup, ±0.03 error at 128 is already well inside the slack of a 0.85 threshold, so you pay 4× resources for precision you don't need.
4. "The cat sat on the mat" and "A feline rested upon the rug." They share essentially no word-shingles, so Jaccard ≈ 0 and no LSH band collides — MinHash keys on token overlap and these have none. Embeddings capture meaning, so their cosine is high and semantic dedup catches them.
5. You skipped or under-did normalization. Re-crawled pages usually differ in whitespace, case, Unicode form, or an injected timestamp/nonce, so raw bytes hash differently. Look at the normalization step (lowercase, whitespace collapse, NFC), confirm you hash the *normalized* text, and check nothing mutable (fetch time) leaks into the hashed content.
6. 0.85 cosine captures *topical relatedness*, not duplication — two distinct articles on the same event score there. Deduping at 0.85 deletes genuinely different documents, collapsing corpus diversity; in RAG that means fewer distinct facts to retrieve and blander, less-grounded answers. Semantic dupes live at 0.95+.
