# Lecture 15: Multi-Tenancy and Tenant Isolation

> The day your retrieval service gets its second customer, "search the corpus" quietly becomes "search *this customer's slice* of the corpus, and never, ever leak a neighbor's document into the results." That second clause is not a feature — it's a security boundary, and in a vector database it is astonishingly easy to punch a hole in it. One forgotten `filter=` argument on one endpoint, and tenant A sees tenant B's confidential contracts ranked by semantic similarity. This lecture teaches the three isolation models and their cost/isolation tradeoffs, why the number-one leak is a filter applied everywhere-but-one-place, how to collapse every query path through a single choke-point so the filter *cannot* be forgotten, how tenant scoping inherits the filtered-HNSW recall caveats from Lecture 12, and what "delete this tenant" really means when their text has smeared itself across six derived stores. After this you'll be able to design leak-proof multi-tenant retrieval, test it adversarially, and answer a GDPR erasure request without leaving a copy behind.

**Prerequisites:** Metadata filtering and the pre/post-filter recall trap (Lecture 12), HNSW internals and selectivity (Lecture 7), vector DB landscape — Qdrant/Pinecone/pgvector (Lecture 11), basic probability and big-O · **Reading time:** ~30 min · **Part of:** Phase 3 — Embeddings Infrastructure & Vector Databases, Week 3

---

## The core idea (plain language)

**Multi-tenancy** means many independent customers (tenants) share one deployment of your service. **Tenant isolation** means no tenant can ever observe another tenant's data — not in results, not in scores, not in timing, not in an error message. In a vector database this is harder than in a normal SQL app for one specific reason: the index doesn't know or care about tenants. HNSW walks a graph of vectors by geometry alone. Whether vector #4,812,003 belongs to Acme Corp or Globex is metadata the *graph* never consults. Isolation is therefore something *you* bolt on, and bolted-on boundaries are the ones that fall off.

There are exactly three ways to draw the boundary, and they trade **cost** against **strength of isolation** (and, as a bonus, against **filtered recall**):

1. **Shared collection + `tenant_id` filter.** Everyone's vectors live in one collection; every query carries `WHERE tenant_id = X`. Cheapest by far (one index, one set of RAM), weakest isolation (a single missing filter leaks everything), and it inherits every filtered-HNSW recall caveat from Lecture 12 because a tenant filter *is* a selective filter.
2. **Per-tenant namespace / collection.** Each tenant gets its own collection (Qdrant) or namespace (Pinecone). Stronger isolation (there is no cross-tenant query path to forget), moderate cost (per-collection overhead, more objects to manage), and it *improves* filtered recall by shrinking the search space to exactly that tenant.
3. **Cluster-per-tenant.** Each tenant gets its own database instance/cluster. Strongest isolation (physical/network separation), priciest (you pay for N clusters), reserved for regulated or large enterprise tenants who contractually demand it.

The single habit this lecture installs: **isolation must live in one choke-point function that every query path is forced through — never sprinkled across endpoints.** The filter you apply in 9 handlers and forget in the 10th is not isolation; it's a countdown timer.

---

## How it actually works (mechanism, from first principles)

### Model 1 — Shared collection + tenant_id filter

One collection holds every tenant's vectors. Each point's payload carries `tenant_id`. Every query appends `tenant_id = X`:

```
        one HNSW collection (5,000,000 vectors, all tenants mixed)
                              │
   query(vec, tenant_id=42) ──┤  native filtered ANN: walk graph,
                              │  accept only nodes with tenant_id==42
                              ▼
                    top-k from tenant 42 only
```

**Cost.** One index, one RAM budget, one thing to operate. If you have 50,000 small tenants this is the *only* model that fits in memory — 50,000 collections would drown you in per-collection overhead. This is why nearly every SaaS starts here.

**Isolation strength.** Weakest. The boundary is a runtime predicate. It holds only as long as *every single query* includes it. There is no structural guarantee: the vectors are physically interleaved in the same graph, so a query without the filter returns everyone's data, ranked by pure similarity. This is the model where the leak vectors below actually bite.

**Recall.** This is the crucial connection to Lecture 12: **a tenant filter is a selective filter**, so it inherits the native-filtered-HNSW recall cliff. Recall the running numbers — 5,000,000 vectors, tenant 42 owns 500 of them. Selectivity is `500 / 5,000,000 = 0.0001` (0.01%). Under native filtered HNSW, when matching nodes are that sparse the greedy graph walk struggles to hop between them; the walk spends most of its steps on non-matching stepping-stone nodes and can terminate before reaching tenant 42's true neighbors. Your unfiltered recall@10 might be 0.97 while your *per-tenant* recall@10 quietly sits at 0.75 — and every dashboard stays green because you're measuring the unfiltered denominator.

Mitigations you already know from Lecture 12: raise `efSearch` / `hnsw_ef` (spend more graph walk to find the sparse matches), build a **payload index** on `tenant_id` so the DB can plan the filter, or — for tenants small enough — let the DB fall back to an exact pre-filter scan (500 dot products is microseconds). Qdrant explicitly supports this: a payload index on `tenant_id` plus, if desired, a `is_tenant=true` hint that tells Qdrant to physically co-locate each tenant's points on disk so the walk stays within-tenant.

### Model 2 — Per-tenant namespace / collection

Give each tenant its own collection (Qdrant) or namespace (Pinecone — a namespace is a partition *inside* an index, addressed by a `namespace=` argument on query/upsert):

```
   collection "tenant_42"  ──►  HNSW over 500 vectors
   collection "tenant_43"  ──►  HNSW over 12,000 vectors
   collection "tenant_44"  ──►  HNSW over 3,000,000 vectors
        ...
   query(vec) targets ONE collection by name/namespace
```

**Cost.** Moderate. Each collection carries fixed overhead (its own HNSW graph, its own segments, some metadata). Qdrant and Pinecone both document soft limits and per-collection memory floors, so this model is comfortable at *hundreds to low-thousands* of tenants, and painful at 50,000 tiny ones. It shines when tenants are few and large, or when you can afford the overhead.

**Isolation strength.** Stronger — and structurally so. There is no "forgot the filter" failure mode, because there is no filter: you *select the collection by name*. A query against `tenant_42` physically cannot return `tenant_44`'s rows; they're in a different index. The failure mode shifts from "forgot the predicate" to "computed the wrong collection name," which is easier to make bulletproof (see the choke-point below).

**Recall.** This model **sidesteps the filtered-HNSW caveat entirely.** Tenant 42's collection contains only tenant 42's 500 vectors, so an *unfiltered* top-10 over that collection is over a full-density graph — no selective-filter cliff, no sparse-match problem. This is a real, often-overlooked quality win: per-tenant collections can beat shared+filter on recall for small tenants, not just on isolation.

### Model 3 — Cluster-per-tenant

Each tenant gets a dedicated database instance (or dedicated hardware/VPC):

```
   Qdrant cluster A  (tenant Acme only)   ── separate process, RAM, disk, network
   Qdrant cluster B  (tenant Globex only) ── separate process, RAM, disk, network
```

**Cost.** Highest. You pay for N clusters' worth of baseline RAM/CPU even if a tenant is tiny, plus N times the operational surface (upgrades, backups, monitoring). Only justifiable for tenants who pay for it.

**Isolation strength.** Strongest. Isolation is now a network/infra boundary, not an application predicate. A bug in your query code can leak *within* a cluster but cannot cross clusters — there's no wire between them. This is what regulated tenants (healthcare, finance, government, "data residency in region X") contractually require, and it's how you satisfy "our data never shares a process with anyone else's."

### The tradeoff at a glance

```
                 cost        isolation      per-tenant recall     scales to
shared + filter  $           weakest        worst (filter cliff)  100k+ tenants
namespace/coll   $$          strong          best (full density)  100s–low 1000s
cluster/tenant   $$$$        strongest       best                 10s–100s
```

There is no universally right answer — the same rule as the recall/latency/cost triangle from Week 2. Most mature SaaS runs a **hybrid**: shared+filter for the long tail of small free/standard tenants, and cluster-per-tenant for the handful of enterprise accounts whose contract demands it.

---

## The number-one leak vector, and the fix

Here is how the leak actually happens, every time. You build the service with a shared collection. You write `/search`, and you diligently apply `tenant_id`. Then you add `/similar-docs`, `/recommend`, `/autocomplete`, an internal `/admin/reindex-preview`, a nightly analytics job, and a "related items" widget. Each is a new code path that calls the vector DB. On nine of them you remember the filter. On the tenth — usually the one added under deadline, or the analytics job written by someone who didn't know the rule — the `filter=` argument is missing or defaulted to `None`. That endpoint now returns global nearest neighbors across all tenants. Nobody notices, because the results *look* plausible (they're semantically relevant!) and no test exercises the cross-tenant case.

The fix is architectural, not disciplinary. **Do not trust every developer to remember the filter. Make it impossible to query without one.** Route every single query path through one function:

```python
# retrieval/tenant_scope.py  — THE ONLY function allowed to call the vector DB

def scoped_search(*, tenant_id: str, query_vector, limit: int, extra_filter=None):
    if not tenant_id:                       # fail closed, never open
        raise ValueError("tenant_id is required for every query")

    tenant_filter = Filter(must=[FieldCondition(
        key="tenant_id", match=MatchValue(value=tenant_id))])
    if extra_filter is not None:
        tenant_filter = _and(tenant_filter, extra_filter)

    return client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=tenant_filter,         # tenant scope ALWAYS applied here
        limit=limit,
    )
```

Then enforce that nothing else may call the client directly. Practical enforcement, strongest first:

- **Namespace model makes this structural**: the choke-point computes the collection/namespace name (`f"tenant_{tenant_id}"`) and there is no way to omit it — a missing name is a crash, not a leak. This is why namespaces are safer even if you never articulate why.
- **Lint / import guard**: a CI check (`grep`/AST rule) that fails the build if `client.query_points`/`client.search` appears anywhere except `tenant_scope.py`. Blunt but effective.
- **Fail closed**: `tenant_id` is a *required* keyword arg with no default. A caller who forgets it gets a `TypeError` at the boundary, not a silent global search. Never let `tenant_id=None` mean "search everything."
- **Type the boundary**: pass a `TenantContext` object extracted from the authenticated request, not a bare string that's easy to leave `None`. The auth middleware produces it; handlers can't fabricate a query without one.

The principle: **isolation is a property of the architecture (one door), not of the developers (everyone remembers).** Ten endpoints × one choke-point = one place to audit. Ten endpoints × ten inline filters = ten places to leak.

---

## Worked example

Let's design isolation for a document-search SaaS and put numbers on it.

- 1,200 tenants. Distribution: 1,150 small (avg 800 docs → ~4,000 chunks each), 45 medium (~200k chunks), 5 enterprise (one regulated healthcare account, ~2M chunks, contract requires data isolation).
- Total ≈ (1,150 × 4,000) + (45 × 200,000) + (5 × 2,000,000) ≈ 4.6M + 9M + 10M ≈ **23.6M chunks**.
- Embeddings: 768-dim fp32 → `768 × 4 = 3,072` bytes/vector for raw storage, plus HNSW graph overhead (`M=32` → ~`32 × 2 × 4 = 256` bytes of links) ≈ ~3.3 KB/vector. 23.6M × 3.3 KB ≈ **78 GB** in one shared index.

**Decision:**

- **Enterprise healthcare tenant → cluster-per-tenant.** The contract demands physical isolation and regional data residency; a shared index can't satisfy "your PHI never shares a process." One dedicated Qdrant cluster, ~6.6 GB for its 2M vectors. Cost is justified by the contract.
- **1,150 small tenants → shared collection + tenant_id filter.** 1,150 separate collections would be operational pain and waste RAM on per-collection floors; a shared index with a payload index on `tenant_id` fits them in ~15 GB. But watch the recall cliff: a 4,000-chunk tenant is `4,000 / 4.6M ≈ 0.087%` selectivity within the small-tenant shard — selective enough to hurt native filtered HNSW. Mitigation: payload index on `tenant_id`, Qdrant `is_tenant` co-location, and `hnsw_ef` bumped from 64 to 256 for filtered queries. For the *tiniest* tenants, Qdrant will pre-filter (a few thousand exact dot products is sub-millisecond) — recall goes back to ~1.0.
- **45 medium tenants → per-tenant collections.** Few enough that per-collection overhead is affordable, large enough that a dedicated collection gives full-density recall and clean isolation. A 200k-vector collection is ~660 MB — comfortable.

**Recall sanity check on the shared shard.** Before shipping, measure per-tenant filtered recall against a *filtered flat ground truth* (Lecture 12's lesson): for a sample tenant, brute-force the true top-10 over *only that tenant's* vectors, then compare to what filtered HNSW returns. Suppose at `hnsw_ef=64` you get recall@10 = 0.79; bumping to `hnsw_ef=256` gets 0.94; enabling `is_tenant` co-location gets 0.98. You now have an evidence-based config, not a guess.

---

## How it shows up in production

- **The silent leak looks like nothing.** A missing filter doesn't throw. It returns *plausible, relevant* results — from the wrong tenant. You find out via a support ticket ("why am I seeing a company called Globex in my search?") or, worse, a breach report. There is no stack trace. The only defense is the adversarial test below, run in CI.
- **The recall cliff looks like a bad model.** A small tenant complains "search misses obvious documents." You blame the embedding model or the chunker. The real cause: their filter is so selective the HNSW walk never reaches their vectors. Diagnosis: measure filtered recall per tenant against filtered ground truth. Fix: `hnsw_ef`, payload index, co-location, or move them to their own collection.
- **The "noisy neighbor" latency problem.** In a shared collection, one tenant bulk-uploading 5M documents can slow everyone's queries and trigger a graph rebuild that spikes p99 for all. Per-collection/per-cluster isolation contains the blast radius. This is a real operational reason to isolate, separate from security.
- **Namespace count creep.** Teams pick per-tenant collections for the clean isolation, then onboard 20,000 tenants and discover the DB was never happy above a few thousand collections — memory floors and metadata overhead crush the node. Match the model to the tenant *distribution*, not to the model that felt safest.
- **GDPR erasure is a cross-system operation, not a `DELETE`.** The moment a tenant's text touched your pipeline it fanned out into derived stores (next section). "We deleted the vectors" is not "we erased the tenant," and a regulator will not accept it.
- **Cross-tenant cache poisoning.** A semantic query cache keyed only on the query text (not `tenant_id`) will serve tenant A's cached results to tenant B for the same query string. This is a leak *through the cache* even if every DB query was correctly filtered. Every cache key must include `tenant_id`.

---

## GDPR / tenant deletion: enumerate every derived store

When a tenant (or a user under GDPR Article 17, "right to erasure") is deleted, you must purge *every place their text or vectors came to rest* — not just the vector DB. Enumerate them explicitly, because each is a separate system with its own delete API and its own way of being forgotten:

1. **The vector store itself.** Delete the tenant's points/collection/cluster. Beware **soft deletes / tombstones** (Lecture on freshness): a tombstoned vector can still match until compaction runs. Verify the DB actually purges, and trigger compaction if deletion is lazy.
2. **The embedding cache.** Your `model:version:sha256(text)` cache (Week 1) holds the *embeddings*, and the key derives from the text. Purge entries for the tenant's texts. If the cache is content-addressed and shared across tenants (dedup!), a text identical to another tenant's stays cached — decide whether that's acceptable, but at minimum drop this tenant's references.
3. **The BM25 / lexical index.** Hybrid search (Lecture 13) keeps a separate lexical index containing the *raw tokens* of every chunk. This is full plaintext. Deleting vectors does nothing to it. Rebuild or delete the tenant's postings.
4. **The semantic query cache.** Cached `(query → results)` pairs contain both the query text and result snippets. Purge by `tenant_id` (which is why the key must include it).
5. **Logs and traces.** OpenTelemetry spans, request logs, and error dumps routinely capture query text and result snippets. This is the most-forgotten store. Scrub or expire logs containing the tenant's text; ideally never log raw text, or hash/redact it at the boundary so erasure is a non-issue.
6. **Backups and snapshots.** Vector DB snapshots and DB backups contain the deleted data. Full erasure requires either backup expiry within a documented window or crypto-shredding (per-tenant encryption key you destroy). Document your backup retention as part of the erasure SLA.
7. **Raw-text / source store.** You kept raw text for re-embedding and rerank (Week 3 lab). That's a primary copy — delete it too, along with any object-storage originals.
8. **Downstream/derived analytics.** Any warehouse table, feature store, or eval "golden set" that ingested the tenant's text.

A robust design makes this tractable: **tag every derived artifact with `tenant_id`** so deletion is a fan-out of `DELETE WHERE tenant_id = X` across a known list, and keep that list in code as the single source of truth (a `PURGE_TARGETS` registry) so adding a new store forces you to add its deleter. The failure mode is the store nobody remembered — which is exactly why you enumerate.

---

## Common misconceptions & failure modes

- **"We filter by tenant_id, so we're isolated."** Only on the paths that actually apply it. Isolation is only as strong as your *weakest* query path. One missing filter = full leak. Choke-point or it isn't real.
- **"Namespaces/collections are always safer, so always use them."** Safer per-query, but they don't scale to tens of thousands of tenants — you'll hit collection-count limits and RAM floors. Match the model to the tenant distribution.
- **"Deleting the vectors deletes the tenant."** No. BM25 index, embedding cache, query cache, logs, backups, raw-text store all survive. Enumerate and purge all of them.
- **"A tenant filter is cheap because it's just a `WHERE`."** In ANN it's a *selective* filter that can wreck recall (Lecture 12). Measure filtered recall against filtered ground truth per tenant.
- **"The query cache is keyed on the query, that's fine."** Not without `tenant_id` in the key — you'll serve one tenant's cached results to another. Cache key must be `(tenant_id, query, filters, model_version)`.
- **"Post-filtering is fine for tenants."** Under a 0.01%-selective tenant filter, post-filtering under-returns catastrophically (you'd need to over-fetch ~10/selectivity candidates — Lecture 12). Use native filtered ANN or pre-filter small tenants.
- **"Error messages don't leak."** A `404` vs `403`, or a timing difference, can reveal whether another tenant's document exists. Return uniform not-found/denied responses; don't let existence leak through side channels.

---

## Rules of thumb / cheat sheet

- **Default for a new SaaS:** shared collection + `tenant_id` filter, with a payload index on `tenant_id`. Cheapest, and correct if you enforce the choke-point.
- **Enforce isolation in ONE function.** Every query path routes through it. `tenant_id` is required, no default, **fail closed**. Add a CI lint that forbids direct DB-client calls elsewhere.
- **Test adversarially, in CI.** Probe tenant A's queries and assert **zero** tenant B docs ever appear — target 20/20 clean (see lab).
- **Cross the 1,000-tenant line?** Reconsider per-collection; you may hit collection-count limits. Small-tenant heavy → stay shared.
- **Regulated / contractual isolation → cluster-per-tenant.** Don't try to satisfy "physically separate" with a filter.
- **Per-tenant collections fix the recall cliff for free** — full-density search per tenant. Use them for medium tenants where affordable.
- **Every cache key includes `tenant_id`** (embedding cache references, semantic query cache). No exceptions.
- **Maintain a `PURGE_TARGETS` registry** of every derived store; deletion fans out across all of them. Adding a store means adding its deleter.
- **Never log raw query/result text** in a multi-tenant service (or redact/hash at the boundary) — it turns every log into an erasure liability.
- **Filtered recall is measured against filtered ground truth**, per tenant. An unfiltered recall number is a lie for a tenant.

*(All tenant-count thresholds above are approximate operational guidance, not hard limits — check your DB's current documented limits.)*

---

## Connect to the lab

This is the theory behind `test_tenant_isolation.py` and the choke-point in `app.py`. In the Week 3 lab you ingest with a `tenant_id` in every payload, implement `search(query, tenant_id, filters)` as the single scoped entry point every route calls, and write the adversarial test that fires 20 of tenant A's queries and asserts not one tenant B document ever appears (the Definition of Done: 20/20 clean). Push it further by measuring per-tenant filtered recall against a filtered flat ground truth, and by writing the tenant-deletion routine that purges the vector store *and* the BM25 index *and* the caches.

**`test_tenant_isolation.py` design.** Structure it as an adversarial harness, not a happy-path check:

```python
# Fixture: two tenants with DISJOINT, semantically overlapping corpora,
# so a leak would actually rank highly (the hard case).
#   tenant A: 200 docs about "quarterly revenue, EBITDA, forecasts"
#   tenant B: 200 docs about "quarterly revenue, EBITDA, forecasts"  (same topics!)
# If isolation is broken, B's docs WILL surface for A's queries — that's the point.

A_ids = set(ingest(tenant="A", docs=corpus_A))
B_ids = set(ingest(tenant="B", docs=corpus_B))

def test_no_cross_tenant_leak():
    leaked = 0
    for q in tenant_A_probe_queries:            # >= 20 probes
        hits = scoped_search(tenant_id="A", query_vector=embed(q), limit=10)
        hit_ids = {h.id for h in hits}
        assert hit_ids & B_ids == set(), f"LEAK: {hit_ids & B_ids} for {q!r}"
        assert hit_ids <= A_ids                  # every hit MUST be tenant A
    # 20/20 clean == pass

def test_missing_tenant_fails_closed():
    # the choke-point must REFUSE, not silently search everything
    with pytest.raises((ValueError, TypeError)):
        scoped_search(tenant_id="", query_vector=embed("revenue"), limit=10)
    with pytest.raises(TypeError):
        scoped_search(query_vector=embed("revenue"), limit=10)  # no tenant_id

def test_cache_is_tenant_scoped():
    # same query string, two tenants -> must NOT share a cache entry
    rA = scoped_search(tenant_id="A", query_vector=embed("revenue"), limit=5)
    rB = scoped_search(tenant_id="B", query_vector=embed("revenue"), limit=5)
    assert {h.id for h in rA} <= A_ids
    assert {h.id for h in rB} <= B_ids
```

Key design choices: (1) **overlapping topics** across tenants so a leak ranks high and the test is genuinely hard; (2) assert both *no B ids* **and** *all hits are A ids* (belt and suspenders); (3) test the **fail-closed** path explicitly — a missing `tenant_id` must raise, never search globally; (4) test the **cache** separately, since a correctly-filtered DB can still leak through a mis-keyed cache; (5) run it against the *real* choke-point every endpoint uses, not a test-only query, or you're proving the wrong thing.

---

## Going deeper (optional)

- **Qdrant docs — "Multitenancy"** (root: `qdrant.tech`): the canonical guide to the shared-collection + `tenant_id` payload-index approach, the `is_tenant` co-location optimization, and when to use separate collections. Search: *"Qdrant multitenancy payload index is_tenant"*.
- **Pinecone docs — "Namespaces"** (root: `docs.pinecone.io`): how namespaces partition an index and the `namespace=` query/upsert argument. Search: *"Pinecone namespaces multitenancy"*.
- **pgvector README** (`github.com/pgvector/pgvector`) combined with Postgres **Row-Level Security (RLS)**: if you're on pgvector, RLS policies enforce `tenant_id` at the database layer so the boundary survives even a forgotten application filter. Search: *"Postgres row level security multi-tenant pgvector"*.
- **GDPR Article 17 ("Right to erasure / right to be forgotten")** — the legal text (search: *"GDPR Article 17 right to erasure text"*) is short and worth reading once; it's what makes the derived-store enumeration a legal requirement, not a nicety.
- **OWASP — Multi-tenant isolation / broken access control**: the general security framing (search: *"OWASP broken access control multi-tenant"*).
- For the recall side, re-read **Lecture 12** in this phase and the **FAISS wiki "Guidelines to choose an index"** for how selective filters interact with graph search.

---

## Check yourself

1. You run a shared collection with a `tenant_id` filter and everything passes tests, but a customer reports seeing another company's document. Where do you look first, and why is there no stack trace pointing at the bug?
2. A small tenant (4,000 of 5,000,000 vectors) says search "misses obvious documents." Unfiltered recall@10 is 0.97. What's the likely cause and three fixes?
3. Why do per-tenant collections *improve* recall compared to a shared collection + filter, in addition to improving isolation?
4. You have 40,000 tenants, most with under 1,000 documents. Which isolation model, and what breaks if you choose per-tenant collections?
5. A tenant invokes GDPR erasure. You run `DELETE` on the vector collection. Name at least four other stores that still contain their data.
6. Your DB queries are all correctly filtered by `tenant_id`, yet tenant B sees tenant A's results for the query "revenue forecast." How is that possible, and what's the fix?

### Answer key

1. Look at every code path that queries the vector DB and find the one missing the `tenant_id` filter — almost always a newer or less-trafficked endpoint (analytics job, "related items," admin preview). There's no stack trace because a missing filter doesn't error; it returns *plausible, relevant* results from the wrong tenant. The structural fix is a single choke-point function that always applies the filter and fails closed, plus a CI lint forbidding direct client calls elsewhere.
2. The tenant filter is highly selective (`4,000/5M = 0.08%`), so native filtered HNSW hits the recall cliff from Lecture 12 — the graph walk can't reach the sparse matching nodes. The 0.97 is *unfiltered* and irrelevant to this tenant. Fixes: (a) raise `hnsw_ef`/`efSearch`; (b) add a payload index on `tenant_id` (and Qdrant `is_tenant` co-location); (c) move the tenant to their own collection, or let the DB pre-filter (exact scan over 4,000 vectors is sub-millisecond). Also: measure filtered recall against a *filtered* flat ground truth so the number stops lying.
3. A per-tenant collection contains only that tenant's vectors, so search runs over a full-density graph with **no selective filter** — there's no sparse-match cliff. Shared+filter forces the walk through a graph dominated by other tenants' nodes, where the tiny matching subset is hard to reach. Full density = better recall, and it happens to also give structural isolation for free.
4. Shared collection + `tenant_id` filter (with a payload index). Per-tenant collections break here because 40,000 collections blow past documented collection-count limits and per-collection RAM floors — the node runs out of memory on metadata/graph overhead long before the vectors themselves are the problem. Match the model to the tenant distribution: many tiny tenants → shared.
5. Any four of: the embedding cache (keyed on text), the BM25/lexical index (raw plaintext tokens), the semantic query cache (query + result snippets), logs and traces (captured query/result text), backups and snapshots, the raw-text/source store kept for re-embedding, and downstream analytics/warehouse/golden-set copies. Maintain a `PURGE_TARGETS` registry so none is forgotten.
6. The semantic query cache is keyed on the query string without `tenant_id`, so tenant A's cached results are served to tenant B for the same query — a leak *through the cache*, even though every DB query was filtered correctly. Fix: include `tenant_id` (and filters + model version) in every cache key.
