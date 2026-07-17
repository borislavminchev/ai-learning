# Lecture 12: Metadata Filtering and the Pre/Post-Filter Recall Trap

> Almost no real retrieval query is "just find me similar vectors." It's "find me similar vectors *that this user is allowed to see*, *from the last 30 days*, *in English*, *not archived*." The moment you bolt a `WHERE` clause onto an ANN search, you leave the clean world of pure nearest-neighbor math and enter a minefield where the same index that scored 0.97 recall unfiltered can quietly return 0.6 — or hand back 3 rows when the user asked for 10 — while every dashboard still shows green because you're measuring the wrong denominator. This lecture teaches the three ways a vector DB combines a filter with an ANN index, why each one can lose recall, and exactly how to configure Qdrant so your filtered searches stay both fast and honest. After this you'll be able to write payload schemas and filters (equality, range, set-membership, boolean logic), create payload indexes so filters don't tank latency, reason about *selectivity* the way you reason about a SQL index, debug an "empty results" incident in minutes, and — most importantly — measure recall **under the filter** against a filtered flat ground truth, because an unfiltered recall number is a lie the instant you add a `WHERE`.

**Prerequisites:** Lectures 1–2 (embeddings, metrics), Lecture 6 (exact vs approximate, honest recall vs a flat ground truth), Lecture 7 (HNSW internals: layers, `M`, `ef_search`, the greedy graph walk), Lecture 11 (the DB landscape); basic SQL indexing intuition; simple probability/arithmetic · **Reading time:** ~30 min · **Part of:** Phase 3 — Embeddings Infrastructure & Vector Databases, Week 3

---

## The core idea (plain language)

An ANN index is built to answer one question fast: *of all N vectors, which are closest to this query?* It is **not** built to answer *which of the vectors matching `tenant_id = 42 AND lang = "en"` are closest?* Those are different questions, and the gap between them is where recall goes to die.

There are exactly three architectural ways to bridge that gap, and every vector database picks one (or lets you pick):

1. **Post-filter** — run the ANN search first, get the top-k by similarity, *then* throw away the rows that fail the filter. Simple, fast, and it will betray you: if the filter is selective, most of your k results get dropped and you hand the user far fewer than they asked for.
2. **Pre-filter (exact)** — first compute the set of rows that match the filter, *then* do the similarity search only over that subset (often brute-force / exact). Always correct on recall, but it can be slow and it often bypasses the ANN index entirely — the very index you built to make search fast.
3. **Native filtered ANN** — teach the index itself about the filter, so the graph walk (in HNSW) only ever *considers* matching nodes. This is what Qdrant, Weaviate, and (to a degree) pgvector do. Fast *and* filtered — but it has a subtle failure mode of its own: under a *very* selective filter the matching nodes can be so sparse in the graph that the greedy walk can't reach them, and recall quietly craters.

The engineering punchline: **there is no filtering strategy that is always fast and always correct.** Which one bites you depends on the *selectivity* of your filter — what fraction of the corpus passes it. And you will never notice the damage unless you measure recall against a ground truth that has *the same filter applied*. This lecture is mostly about internalizing that last sentence.

---

## How it actually works (mechanism, from first principles)

### Selectivity is the master variable

Everything downstream depends on one number. **Selectivity** = the fraction of the corpus that passes the filter. A filter matching 60% of rows is *unselective*; one matching 0.1% is *highly selective*. (Confusingly, "highly selective" means it *selects few* — it's a strong filter. Same convention as SQL.)

Let corpus size be `N`, and let `p` be the fraction of rows that pass the filter. Then roughly `p·N` rows survive. Hold that in your head for every strategy below.

### Strategy 1: Post-filter — fast, but returns too few

The algorithm:

```
1. results = ann_search(query, top_k = k)      # k nearest by similarity, ignoring filter
2. results = [r for r in results if r matches filter]
3. return results                               # possibly far fewer than k
```

Why it's tempting: you don't touch the index at all. The ANN engine does exactly what it's optimized for, and filtering is a cheap post-pass over k rows.

Why it bites: the top-k nearest neighbors have *no reason* to satisfy your filter. If the filter matches a fraction `p` of the corpus, and matching rows are distributed roughly uniformly among the nearest neighbors, then out of `k` ANN results you expect about `p·k` to survive.

**Worked numbers.** User asks for the top 10. You post-filter with `lang = "de"`, and German is 5% of your corpus (`p = 0.05`). ANN returns 10 candidates; expected survivors = `0.05 × 10 = 0.5`. You hand back **zero or one result** for a query that has hundreds of perfectly good German matches deeper in the corpus. The index never looked at them, because it stopped at the global top-10.

The naive fix is **over-fetch**: ask the ANN for `k / p` candidates so that after filtering you still have ~k. For `p = 0.05`, `k = 10`, fetch `10 / 0.05 = 200`. This works until `p` gets tiny: at `p = 0.001` you'd need to fetch 10,000 candidates for 10 results, which is slow, memory-hungry, and eventually larger than the index can efficiently return. Over-fetch also assumes matching rows are uniformly mixed among neighbors — if your filter *correlates* with the query (e.g., German docs are also semantically clustered), the estimate is off and you either over- or under-fetch. Over-fetch is a patch, not a cure.

```
Post-filter under a 5%-selective filter, k=10:

corpus:  [·········█·········█····█······█·]   █ = matches filter
ANN top-10 by similarity (ignores filter):
         [ 1  2  3  4  5  6  7  8  9 10 ]
matches: [ .  .  █  .  .  .  .  █  .  . ]   -> only 2 survive
                                            user asked for 10.
```

### Strategy 2: Pre-filter (exact) — correct, but can be slow and bypass the index

The algorithm:

```
1. candidate_ids = filter_index_lookup(filter)   # e.g. all ids where tenant_id=42
2. results = exact_search(query, over = candidate_ids, top_k = k)  # brute force distances
3. return results
```

Now recall is perfect: you compute true distances to *every* matching row, so you always get the true k nearest among matches. This is exactly what a "filtered flat" search does — and it's the ground truth you'll measure everyone else against.

The cost: step 2 is `O(p·N · dim)` distance computations. When `p` is small this is a bargain — filtering to 500 rows and brute-forcing 500 dot products is microseconds. But when `p` is large (say 40% of a 5M-vector corpus = 2M rows), you're brute-forcing 2M distances per query, and you've thrown away the entire reason you built an ANN index. Latency goes from single-digit milliseconds to hundreds of milliseconds or seconds.

There's also a subtler trap: some engines, when they *can't* do a clean pre-filter, will do a full exact scan for *any* filtered query regardless of selectivity, because "correct but slow" is a safer default than "wrong." That's why an unindexed filter field in some vector DBs turns every filtered query into a full-collection scan.

### Strategy 3: Native filtered ANN — fast and filtered, with a selectivity cliff

This is the modern answer, and it's what you'll ship. Instead of filtering before or after, the index applies the filter **during** traversal. In HNSW terms (recall Lecture 7): the search is a greedy walk over a proximity graph — you start at an entry point and keep hopping to the neighbor closest to the query, expanding a candidate frontier of size `ef_search`. Native filtered HNSW modifies the walk so that a node **only counts as a valid result and a valid stepping stone if it passes the filter.** Non-matching nodes are skipped for scoring; matching neighbors are followed.

When `p` is moderate this is close to ideal: the graph is dense with matching nodes, the greedy walk finds them easily, and you get near-flat recall at ANN speed.

**Why it loses recall under a very selective filter — the disconnection problem.** HNSW's speed comes from a key assumption: the graph is *navigable* — from any node you can reach any other node's neighborhood by short greedy hops, because every node has `M` edges to its nearest neighbors. But those edges were built over **all** vectors, ignoring your filter. When you overlay a very selective filter, you keep only the tiny `p·N` matching nodes and the edges *between them*. Two matching nodes that are true nearest neighbors in vector space might have **no path between them through other matching nodes** — every short route between them runs through non-matching nodes that the filtered walk refuses to step on. The subgraph of matching nodes becomes **disconnected islands.** The greedy walk lands on one island, exhausts it, and terminates — never discovering the matching nodes stranded on other islands, even though they're closer to the query.

```
Full HNSW graph (edges to M nearest, built over ALL nodes):

   A —— b —— C —— d —— E          UPPERCASE = matches filter
   |    |    |    |    |          lowercase = filtered out
   f —— G —— h —— I —— j

Filter to matches only; keep only edges between matches:

   A         C         E          A, C, E, G, I survive but the
                                   connecting nodes (b, d, f, h, j)
       G         I                 are gone -> no edges left ->
                                   5 disconnected islands.
The greedy walk reaches A, finds no matching neighbor to hop to,
and stops. C, E, G, I are never scored. Recall collapses.
```

The lower the selectivity `p`, the sparser the matching subgraph, the more disconnected it gets, and the worse the recall. This is not a bug in Qdrant or Weaviate — it's intrinsic to running a graph walk over a subgraph the graph wasn't built for.

### The mitigations (this is the practical core)

Every serious vector DB combines these; you should know all four:

1. **Over-fetch then filter** (post-filter's patch, applied to native mode too): expand `ef_search` / fetch more candidates so the walk explores enough of the graph to stumble across matching islands. Cheap first line of defense, insufficient alone under extreme selectivity.

2. **Payload indexing to make exact pre-filter fast.** If the filter field has an index (Qdrant builds a payload index per field on request), computing the matching id set is fast, so the *pre-filter/exact* path becomes cheap enough to use. This is the enabling mechanism for #3.

3. **Fall back to exact search below a cardinality threshold.** This is the key insight and what Qdrant does automatically. Qdrant estimates how many rows the filter will match (using the payload index's cardinality stats). If that number is **below a threshold** (default around 10,000, tunable via `full_scan_threshold`), it *skips the graph entirely* and brute-forces exact distances over just the matching rows — which is fast precisely *because* the filter is selective (few rows to score). Above the threshold, matching nodes are dense enough that the filtered graph walk stays connected and accurate, so it uses filtered HNSW. **The two strategies cover for each other exactly where the other fails.** Selective filter → small candidate set → exact is cheap and correct. Unselective filter → dense subgraph → filtered HNSW is fast and accurate. The dangerous middle is what the threshold is tuned to straddle.

4. **Filterable HNSW with extra links (Qdrant's structural fix).** Qdrant can build *additional* graph edges informed by payload values, so that nodes sharing common filter values stay connected even after filtering — reconnecting the islands in advance. This raises recall for the frequently-filtered fields at the cost of a larger index and slower build. You enable it by telling Qdrant which payload fields you filter on (a payload-aware / "tenant" index configuration).

```
Selectivity spectrum and what actually runs (Qdrant, roughly):

 p ->   0.0001    0.001     0.01      0.1        1.0
        |----------|----------|----------|----------|
        <-- exact pre-filter -->  <-- filtered HNSW -->
        (few matches, brute      (dense subgraph, graph
         force is cheap+exact)    walk stays connected)
                     ^
              full_scan_threshold ~ matches < 10k -> exact
```

---

## Worked example

You run a 2,000,000-vector corpus (768-dim, cosine) of support documents in Qdrant. A tenant filter `tenant_id = 42` matches 4,000 rows (`p = 0.002`). The user asks for top-10.

**Post-filter, no over-fetch.** ANN top-10 globally; expected survivors ≈ `0.002 × 10 = 0.02`. You return **0 results** ~98% of the time. Catastrophe, and it looks like "we have no docs for this tenant."

**Post-filter with over-fetch.** To expect 10 survivors you'd fetch `10 / 0.002 = 5,000` candidates, then filter down. That works, but you just made a 5,000-NN query per request, and if selectivity varies per tenant you're guessing the fetch size.

**Native filtered HNSW, no fallback.** 4,000 matching nodes scattered across a 2M-node graph — extremely sparse. The filtered subgraph is heavily disconnected. The greedy walk reaches maybe one or two islands and returns 10 results that *look* fine (they're real tenant-42 docs) but recall vs the true filtered top-10 might be **0.4** — you're silently missing 6 of the true best 10. Nobody sees an error. This is the trap.

**Qdrant with a payload index on `tenant_id`.** Qdrant estimates the filter matches ~4,000 rows, which is **below** `full_scan_threshold` (~10k). It **skips the graph** and does an exact brute-force over exactly those 4,000 vectors: `4,000 × 768` multiply-adds ≈ 3M FLOPs — well under a millisecond. Recall vs the filtered flat ground truth = **1.0**, and it's *faster* than the disconnected graph walk would have been. The selective filter that broke HNSW is exactly what makes exact search cheap.

Now flip it: filter `lang = "en"` matches 1,200,000 rows (`p = 0.6`). Exact over 1.2M vectors would be ~200 ms — too slow. Qdrant sees the estimate is far above threshold, uses **filtered HNSW**, and the subgraph is so dense that recall stays ~0.97 at a few milliseconds. Same DB, same config, opposite strategy — chosen automatically off the cardinality estimate the payload index provides.

**The measurement that proves it.** For each of the two filters, build a **filtered flat ground truth**: take every eval query, restrict the corpus to rows passing that filter, brute-force the true top-10 over the subset. Then run the same filtered query through Qdrant and compute recall@10 against *that* filtered ground truth — not the unfiltered one. If you'd measured against the unfiltered ground truth, you'd be comparing apples to a different orchard: the unfiltered top-10 are mostly other tenants' docs, so the number is meaningless.

---

## Defining schemas, filters, and payload indexes in Qdrant

**Payload = metadata.** In Qdrant every point carries a `payload` — a JSON object of arbitrary fields you can filter on. Create the collection with your metric/index knobs (from Weeks 1–2), then upsert points with payloads:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="docs",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200),
)

client.upsert(
    collection_name="docs",
    points=[
        models.PointStruct(
            id=1,
            vector=vec,  # list[float], length 768
            payload={"tenant_id": 42, "lang": "en", "created_at": 1_735_000_000,
                     "tags": ["billing", "urgent"], "archived": False},
        ),
        # ...
    ],
)
```

**Payload indexes — do this or filters scan.** Without an index on a field, Qdrant can still filter, but estimating cardinality and locating matches is slow, and you lose the automatic exact-fallback optimization. Create one index per field you filter on, with the matching schema type:

```python
client.create_payload_index("docs", "tenant_id",  models.PayloadSchemaType.INTEGER)
client.create_payload_index("docs", "lang",       models.PayloadSchemaType.KEYWORD)
client.create_payload_index("docs", "created_at", models.PayloadSchemaType.INTEGER)  # range-able
client.create_payload_index("docs", "tags",       models.PayloadSchemaType.KEYWORD)  # set membership
client.create_payload_index("docs", "archived",   models.PayloadSchemaType.BOOL)
```

The schema type matters: `KEYWORD` for exact string/tag match and set membership, `INTEGER`/`FLOAT` for ranges, `BOOL` for flags, `DATETIME` for time ranges. Index the *right* type or range queries silently don't optimize.

**Building filters — the four shapes you need.** Qdrant's filter model is `must` (AND), `should` (OR), and `must_not` (NOT), composing `FieldCondition`s:

```python
flt = models.Filter(
    must=[
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=42)),        # equality
        models.FieldCondition(key="created_at", range=models.Range(gte=1_735_000_000)),   # range
        models.FieldCondition(key="tags", match=models.MatchAny(any=["billing","refund"])), # set membership (OR within field)
    ],
    must_not=[
        models.FieldCondition(key="archived", match=models.MatchValue(value=True)),        # boolean NOT
    ],
    should=[  # at least one of these (OR)
        models.FieldCondition(key="lang", match=models.MatchValue(value="en")),
        models.FieldCondition(key="lang", match=models.MatchValue(value="de")),
    ],
)

hits = client.query_points(
    collection_name="docs",
    query=query_vec,
    query_filter=flt,
    limit=10,
    search_params=models.SearchParams(hnsw_ef=128),  # over-fetch dial when native filtering
).points
```

`must` = all conditions AND together, `should` = OR (at least one), `must_not` = exclude. Nest `Filter` objects inside `must`/`should` for arbitrary boolean trees. `MatchValue` = equality, `Range(gte/lte/gt/lt)` = numeric/time ranges, `MatchAny` = set membership (value in list). The `full_scan_threshold` that governs the exact-vs-graph decision lives in `hnsw_config` at collection creation (or per-field for payload-aware HNSW).

---

## How it shows up in production

- **The "empty results for small tenants" ticket.** Post-filtering plus a selective tenant filter returns 0–2 results for your smallest customers while working fine in demos (where every tenant has lots of data). It looks like missing data; it's the recall trap. The fix is native filtered search with payload indexes, verified by a filtered-ground-truth recall test — *not* "re-ingest the tenant's docs."

- **Latency that scales with the *unselective* case, not the average.** If you rely on exact pre-filter for everything, a broad filter (`lang = "en"`, 60% of corpus) turns into a multi-hundred-ms brute force while your dashboards — full of selective test filters — look green. Load-test with your *least* selective real filter.

- **The recall number that "looks great" and lies.** A team reports "recall@10 = 0.96" measured on unfiltered queries, ships filtering, and quality tanks in the field. The 0.96 never described the filtered path. Every filtered feature needs its **own** recall number vs a filtered ground truth. This is the single most common way filtered retrieval ships broken.

- **Forgot the payload index.** Filters "work" in dev on 10k rows (full scan is instant) and fall over at 5M rows in prod (every filtered query is now a full collection scan, p99 in seconds). Payload indexes aren't optional at scale; treat them like SQL indexes on any column in a `WHERE`.

- **Selectivity drift.** A filter that was unselective at launch (`plan = "free"` = 90% of users) becomes selective as you grow (`plan = "enterprise"` = 0.5%), silently crossing the threshold where your chosen strategy's recall changes. Re-measure filtered recall when your data distribution shifts, not just when you change code.

- **Correlated filters break over-fetch math.** Over-fetch assumes matching rows are uniformly sprinkled among neighbors. If `lang = "de"` docs are *also* semantically clustered (they are — language correlates with topic), the true German neighbors may be denser or sparser than `p` predicts, so your `k/p` fetch is wrong. Native filtered ANN with fallback sidesteps the guessing.

---

## Common misconceptions & failure modes

- **"I filtered, so recall is fine."** Filtering *changes* the recall question. Your old unfiltered recall number is irrelevant. Recall must be measured against a **filtered flat ground truth** — same filter, brute force over the subset.
- **"Native filtered HNSW is always safe."** It's safe in the middle of the selectivity range. Under a *very* selective filter the matching subgraph disconnects and recall collapses — which is exactly why Qdrant falls back to exact below `full_scan_threshold`. If your DB has no such fallback, you own this failure.
- **"Over-fetch fixes post-filtering."** Only for moderate selectivity and uniformly-distributed matches. At `p = 0.001` the fetch size explodes, and correlated filters break the arithmetic.
- **"Post-filter and pre-filter give the same results, just slower/faster."** No — post-filter can return *fewer* results and can *miss* true matches entirely (they were never in the top-k). Pre-filter/exact is the correct baseline; post-filter is an approximation of it that degrades with selectivity.
- **"Payload indexes are just for speed."** They're also what lets the engine *estimate cardinality* and choose the exact-vs-graph strategy. No index → no good estimate → worse strategy choice and full scans.
- **"A boolean flag filter is cheap."** `archived = false` matching 99% of rows is unselective (fine for graph walk); `is_flagged = true` matching 0.01% is brutally selective (needs exact fallback). The *type* doesn't tell you selectivity — the *distribution* does.

---

## Rules of thumb / cheat sheet

*(Figures approximate; the threshold defaults are Qdrant's and are tunable — measure on your data.)*

- **Always index the fields you filter on.** One payload index per `WHERE` field, correct schema type. No index = full scan at scale.
- **Never trust an unfiltered recall number for a filtered query.** Build a filtered flat ground truth (restrict corpus by filter, brute-force top-k) and measure recall against *that*.
- **Estimate selectivity `p` per filter.** `p·N` = matching rows. This number chooses the failure mode.
  - `p·N` small (roughly < ~10k, below `full_scan_threshold`) → exact pre-filter is cheap *and* exact. Let the DB fall back to it.
  - `p·N` large (unselective) → filtered HNSW; subgraph is dense, recall stays high.
  - The dangerous middle → tune `full_scan_threshold` and `hnsw_ef`; verify with filtered recall.
- **Post-filter only when the filter is barely selective** (matches most rows) or you over-fetch `~k/p` and have measured that it holds. Otherwise prefer native filtered ANN with fallback.
- **Raise `hnsw_ef` (over-fetch) as the first knob** when filtered recall is low but you're above the exact threshold.
- **For a hot, always-present filter** (e.g., `tenant_id`), enable Qdrant's **payload-aware / filterable HNSW** so extra edges keep matching nodes connected.
- **Re-measure filtered recall when data distribution drifts** — selectivity is not static.
- **Qdrant defaults worth knowing:** `full_scan_threshold` ≈ 10,000 (matches below → exact); set it in `hnsw_config`. `hnsw_ef` per query controls over-fetch.

---

## Connect to the lab

In this week's lab you implement `search(query, tenant_id, filters)` on Qdrant and write `test_tenant_isolation.py`. Make the tenant filter a **native** Qdrant payload filter (not application-side post-filtering), create a **payload index on `tenant_id`** in `ingest.py`, and prove isolation with a genuinely *selective* tenant (a tenant with few docs) — that's the case where post-filtering would silently return empties. Then extend `monitor.py`: build a *filtered* flat ground truth for one selective filter and one broad filter, and print recall@10 for both against the live filtered search — you should see recall stay ~1.0 on the selective one (exact fallback) and ~0.95+ on the broad one (filtered HNSW), which proves your service is honest under filters.

## Going deeper (optional)

- **Qdrant docs** (root: `qdrant.tech`) — read "Filtering," "Indexing" (payload indexes and `full_scan_threshold`), and the filterable-HNSW explanation. The clearest first-party writeup of the disconnection problem and the exact-fallback mitigation in the ecosystem.
- **Weaviate docs** (root: `weaviate.io`) — "Filtered search" and their `flat`-vs-`hnsw` and filter-strategy (ACORN) pages; Weaviate's ACORN work is a good contrast to Qdrant's threshold approach.
- **pgvector README** (`github.com/pgvector/pgvector`) — read the "Filtering" section; understand *iterative index scans* and why an unindexed `WHERE` plus HNSW can return too few rows or fall back to a full scan.
- **The original HNSW paper** (Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs") — reread the navigability/greedy-search sections with filtering in mind; it makes the disconnection intuition click.
- **ACORN** — search `ACORN predicate-agnostic HNSW filtering` for the current research framing of filtered ANN.
- Search queries: `qdrant filterable HNSW selectivity recall`, `weaviate filtered search ACORN`, `pgvector iterative scan filter`, `filtered ANN recall benchmark selectivity`, `HNSW filter disconnected subgraph recall`.

## Check yourself

1. Your top-10 search with a filter that matches 3% of the corpus returns only 1 result. Name the filtering strategy in use and explain, with the arithmetic, why you got ~1 result.
2. Under a *very* selective filter, native filtered HNSW returns 10 plausible-looking results but recall vs the filtered ground truth is 0.4. What is the mechanism, and why don't the results *look* wrong?
3. Why does a selective filter make **exact pre-filter** *cheap* while it makes **filtered HNSW** *inaccurate* — i.e., why do these two strategies cover for each other?
4. You measured "recall@10 = 0.95" on your eval set and shipped tenant filtering; users complain quality dropped. What did your 0.95 fail to measure, and what exactly do you build to measure it correctly?
5. You have a hot `tenant_id` filter and a broad `lang` filter. Which mitigations do you apply to each, and why do they differ?
6. What does creating a payload index on a filter field buy you *beyond* raw lookup speed?

### Answer key

1. **Post-filtering.** The ANN returned the global top-10 by similarity ignoring the filter; you then dropped non-matching rows. With `p = 0.03`, expected survivors ≈ `p·k = 0.03 × 10 = 0.3`, so ~0–1 results. The true matching neighbors deeper in the corpus were never fetched. Fix: native filtered ANN, or over-fetch `~k/p ≈ 333` candidates.
2. **Graph disconnection.** The filter is so selective that the matching nodes form a sparse subgraph; the edges connecting true nearest neighbors run through non-matching nodes the filtered walk won't step on, so the subgraph splits into disconnected islands. The greedy walk explores one island and stops, missing closer matches on other islands. The results *look* fine because they *are* real matching rows and are reasonably similar — just not the true top-10. Only a filtered ground truth reveals the gap.
3. A selective filter yields a **small** matching set (`p·N` small), so brute-forcing exact distances over just those rows is cheap *and* returns the true top-k — exact is both fast and correct here. That *same* small set is sparse in the full HNSW graph, so the filtered subgraph disconnects and the graph walk loses recall. Conversely, an unselective filter makes exact too slow (huge set) but keeps the subgraph dense so the graph walk stays accurate. Each strategy is fast+correct exactly where the other fails, which is why Qdrant switches between them at `full_scan_threshold`.
4. It failed to measure recall **under the filter** — it was an *unfiltered* recall number, describing a query path (no `WHERE`) that production doesn't use; the filtered top-10 are a different set. Build a **filtered flat ground truth**: for each eval query, restrict the corpus to rows passing the filter, brute-force the exact top-10 over that subset, and compute recall of the live *filtered* search against it.
5. **`tenant_id` (hot, often very selective per small tenant):** payload index + rely on exact fallback below `full_scan_threshold`, and consider **payload-aware / filterable HNSW** (extra edges) so matching nodes stay connected for this always-present filter. **`lang` (broad, unselective):** payload index for cardinality estimation, then plain **filtered HNSW** with a modest `hnsw_ef` — the subgraph is dense so recall is already high; exact fallback would be too slow here. They differ because selectivity differs: exact fallback and extra-links help the sparse case, filtered-graph handles the dense case.
6. It lets the engine **estimate the filter's cardinality**, which is what it uses to *choose* between exact pre-filter and filtered HNSW (the `full_scan_threshold` decision), and it prevents the DB from degrading a filtered query into a **full collection scan**. So the index buys correct strategy selection and avoids full scans, not just faster matching-id lookup.
