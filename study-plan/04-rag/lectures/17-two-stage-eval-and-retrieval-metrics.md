# Lecture 17: Two-Stage Evaluation and Retrieval Metrics

> A RAG answer can be wrong in two completely different ways, and the difference decides everything you do next. Either the retriever never fetched the chunk that holds the answer (a *retrieval* miss), or it fetched it and the LLM ignored, mangled, or contradicted it (a *generation* miss). A single end-to-end "accuracy: 63%" number blends these into one useless blob — you cannot tell which half is broken, so you cannot fix anything without guessing. This lecture teaches the load-bearing move for the whole week: **split the eval into two stages** — metrics that score the *retrieved context set* independent of the answer, and metrics that score the *answer given that context*. Then it goes deep on the retrieval-side stage, upgrading your Week-1 recall@k and MRR with **nDCG** for graded, position-sensitive ranking. After this you will be able to implement recall@k, MRR@k, and nDCG@k from scratch, unit-test them against a hand-computed toy example before you trust a single aggregate, read the three numbers together to route a failure to the right knob, and avoid the chunk-id-stability trap that silently zeroes your recall.

**Prerequisites:** Lecture 4 (golden sets, recall@k, MRR in isolation), Lecture 6 (cross-encoder reranking), basic Python, `log₂` arithmetic. · **Reading time:** ~28 min · **Part of:** Retrieval-Augmented Generation, Week 4

## The core idea (plain language)

You have a full RAG pipeline now: chunk → embed → retrieve → (rerank) → generate. A user asks a question, gets an answer, and the answer is wrong. What do you change?

If your only instrument is one end-to-end score — "the answer was wrong" — you are stuck. A wrong answer has two structurally different root causes:

1. **Retrieval miss.** The chunk containing the answer never made it into the top-k you handed the LLM. The information was simply *absent from the context*. No prompt, no model swap, no temperature setting can recover a fact that isn't in the window.
2. **Generation miss.** The right chunk *was* in the context, and the LLM ignored it, misread it, hedged, or actively contradicted it — a hallucination *given* the evidence, or a formatting/extraction failure.

These demand opposite fixes. A retrieval miss routes to chunking, embeddings, k, or the reranker. A generation miss routes to the prompt, the model, or the "cite the context" instruction. **The single blended accuracy number cannot distinguish them**, so it can only ever tell you *that* you're wrong, never *where*. Teams that eval this way spend two weeks tuning a prompt while recall sits at 0.55 and the real ceiling is 55%.

The fix is a **two-stage evaluation**:

> **Stage 1 — Retrieval eval:** score the *retrieved context set* against a golden set of `relevant_chunk_ids`, with no LLM in the loop. Question in → ranked chunk IDs out → recall@k / MRR@k / nDCG@k. Every number here is a pure retrieval number.
>
> **Stage 2 — Generation eval:** *given the context the retriever actually returned*, score the answer — faithfulness/groundedness, answer relevancy, context precision (Lecture 18, RAGAS). Every number here is conditioned on the context, so it isolates the generator.

Run both, and the failure localizes itself. Low retrieval score ⇒ retrieval problem. Retrieval score fine but generation score low ⇒ generation problem. That decision rule (formalized in Lecture 18) is *why* you compute both families — it turns "the RAG is bad" into a routed ticket. This lecture owns Stage 1: the retrieval metrics, deeply.

## How it actually works (mechanism, from first principles)

Stage 1 needs a golden set keyed on **chunk IDs**, not answer strings. Week 1 used `gold_substr` (a distinctive substring the answer chunk must contain) because it survives re-chunking. Week 4 tightens this to `relevant_chunk_ids: [...]` — the explicit set of chunk IDs a correct retrieval must surface — because that's what lets you compute *graded, multi-relevant, position-aware* metrics. (The tradeoff and the trap this creates get their own section below.)

All three metrics take the same two inputs: a **ranked list of retrieved IDs** and the **golden set of relevant IDs**. They differ only in what they reward.

### recall@k — the retrieval floor

> **recall@k** = of the chunks that are *truly relevant*, what fraction landed in the top-k?

```
recall@k = |relevant ∩ top-k retrieved| / |relevant|
```

This is your **retrieval floor and the ceiling on answer quality**: if a relevant chunk isn't in the top-k you feed the LLM, generation cannot recover it. recall@k ignores *where* in the top-k a relevant chunk landed — rank 1 and rank k both count as a plain "in." It only asks "did we get it inside the window at all?"

The single most important operational detail: **measure recall@k at the k you actually feed the LLM.** If you retrieve 50 candidates but only pass the top 5 into the prompt, your generation ceiling is `recall@5`, not `recall@50`. A gorgeous `recall@50 = 0.98` is a lie about your real ceiling if you truncate to 5 before generation. Watch the k that reaches the context window.

### MRR@k — is the *first* good hit near the top?

> **MRR@k (Mean Reciprocal Rank)** = the mean over all questions of `1 / (rank of the first relevant hit within k)`. No hit within k ⇒ that question contributes 0.

```
reciprocal rank = 1 / rank_of_first_relevant_hit   (0 if none in top-k)
MRR@k = mean of reciprocal ranks over all questions
```

The reciprocal makes rank *hurt*: rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.333, rank 5 → 0.2, rank 10 → 0.1. MRR is the right metric **when one good chunk suffices** — a factoid lookup where the answer lives in a single place and you just need it near the top so a top-down-reading LLM sees it first. Its two assumptions bite you if you forget them: MRR is **binary** (a chunk is relevant or not — no grades) and it only looks at the **first** hit (it's blind to whether the *other* relevant chunks were also retrieved). For a multi-hop question that needs three chunks fused, MRR happily reports 1.0 after finding just the first one.

### nDCG@k — graded relevance and position sensitivity

recall@k is blind to position; MRR@k sees position but only for the first hit and only in binary. **nDCG@k is the one to trust when relevance is *graded* (chunk A is more relevant than chunk B) and *every* position matters.** It is the standard IR ranking-quality metric, and it's what you gate ranking changes on.

Build it in three steps.

**1. DCG — Discounted Cumulative Gain.** Sum the relevance "gain" of each retrieved chunk, discounted by how far down the list it sits. The discount is `1/log₂(rank+1)`:

```
DCG@k = Σ (over ranks i = 1..k)  gain_i / log₂(i + 1)
```

With **binary** relevance, `gain_i = 1` if the chunk at rank i is relevant, else 0. With **graded** relevance, `gain_i` is the label (e.g. 2 = highly relevant, 1 = relevant, 0 = not). The `log₂(i+1)` discount means a hit at rank 1 is worth `1/log₂(2) = 1.0`, at rank 2 `1/log₂(3) = 0.63`, at rank 5 `1/log₂(6) = 0.39`. A relevant chunk buried deep contributes almost nothing — exactly the position sensitivity you want.

**2. IDCG — Ideal DCG.** The DCG of the *best possible* ordering: all relevant chunks sorted to the top, highest grade first. This is the normalizer.

**3. nDCG — normalize to [0,1].**

```
nDCG@k = DCG@k / IDCG@k
```

nDCG@k = 1.0 means your ranking is as good as the ideal ordering; 0.5 means there's real ordering slack to reclaim. Normalizing makes nDCG **comparable across queries** with different numbers of relevant chunks — a query with 5 relevant chunks and one with 1 both land on the same [0,1] scale, which recall and raw DCG do not.

> **The engineer's read.** `recall@k` is your **retrieval floor** (did the facts get in the window?). `nDCG@k` is your **ranking quality** (are they ordered well?). The diagnostic split:
>
> ```
> recall@10 HIGH  +  nDCG@10 LOW   ⇒  right chunks retrieved, ordered badly  ⇒  the RERANKER is the lever
> recall@10 LOW                     ⇒  right chunks not retrieved at all      ⇒  fix chunking / embeddings / k / hybrid; a reranker CAN'T help
> ```
>
> A reranker only reorders what retrieval already returned. If recall@10 is low, the gold chunk isn't in the candidate set and no reordering conjures it. If recall@10 is high but nDCG@10 is low, the candidates are all there and merely mis-ordered — that's precisely what a cross-encoder reranker (Lecture 6) fixes.

### The toy example you MUST hand-compute first

Before you trust any aggregate over 60 golden cases, verify your implementations against a known-answer toy. This is not optional busywork — an off-by-one in your `log₂(i+1)` or a `k`-slicing bug produces plausible-looking numbers that are quietly wrong, and you'll ship on them.

**Setup:** two truly-relevant chunks. Your retriever returns them at **rank 2 and rank 5**. `relevant = {A, B}`, `retrieved = [x, A, y, z, B]`, k = 5, binary relevance (gain = 1).

**recall@5** = both relevant chunks are in the top 5 → `2/2 = 1.0`.
**recall@3** = only rank-2 chunk is in the top 3 → `1/2 = 0.5`.

**MRR@5**: first relevant hit is at rank 2 → `1/2 = 0.5`. (This is the "known MRR = 0.5" you check against.)

**nDCG@5**, worked to the digit:

```
DCG@5  = 1/log₂(2+1) + 1/log₂(5+1)        # relevant at ranks 2 and 5
       = 1/log₂(3)   + 1/log₂(6)
       = 1/1.58496   + 1/2.58496
       = 0.63093     + 0.38685
       = 1.01778

IDCG@5 = 1/log₂(1+1) + 1/log₂(2+1)        # ideal: both relevant at ranks 1 and 2
       = 1/1.0       + 1/1.58496
       = 1.00000     + 0.63093
       = 1.63093

nDCG@5 = 1.01778 / 1.63093 = 0.6240
```

So the toy's known-good triple is **recall@5 = 1.0, MRR@5 = 0.5, nDCG@5 ≈ 0.624**. Write those three constants into a unit test *before* you write another line of eval code.

### From-scratch implementations

Over ranked ID lists vs. golden relevant IDs. Deliberately tiny and dependency-free so you can read every line — you're going to unit-test these, and a metric you can't hand-verify is a metric you can't trust.

```python
import math

def recall_at_k(retrieved, relevant, k):
    """Fraction of relevant chunks that appear in the top-k."""
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in rel)
    return hits / len(rel)

def mrr_at_k(retrieved, relevant, k):
    """1 / rank of the FIRST relevant hit within top-k (0 if none)."""
    rel = set(relevant)
    for i, cid in enumerate(retrieved[:k], start=1):   # rank is 1-indexed
        if cid in rel:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved, relevant, k, grades=None):
    """Normalized DCG. Binary if grades is None; graded if grades={id: gain}."""
    def gain(cid):
        if grades is not None:
            return grades.get(cid, 0.0)
        return 1.0 if cid in set(relevant) else 0.0

    # DCG over the actual ranking
    dcg = sum(gain(cid) / math.log2(i + 1)
              for i, cid in enumerate(retrieved[:k], start=1))

    # IDCG: best possible ordering — sort gains descending, take top-k
    if grades is not None:
        ideal_gains = sorted(grades.values(), reverse=True)[:k]
    else:
        ideal_gains = [1.0] * min(len(set(relevant)), k)
    idcg = sum(g / math.log2(i + 1)
               for i, g in enumerate(ideal_gains, start=1))

    return dcg / idcg if idcg else 0.0
```

And the test that must pass before you believe anything else:

```python
def test_metrics_against_hand_computed_toy():
    retrieved = ["x", "A", "y", "z", "B"]   # relevant at ranks 2 and 5
    relevant  = ["A", "B"]

    assert recall_at_k(retrieved, relevant, 5) == 1.0
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert mrr_at_k(retrieved, relevant, 5) == 0.5
    assert abs(ndcg_at_k(retrieved, relevant, 5) - 0.6240) < 1e-3
```

`pytest` green on this is your license to run the aggregate. Note two easy-to-miss correctness points baked in above: rank is **1-indexed** (`start=1`) so the discount is `log₂(i+1)` with the first rank giving `log₂(2)=1`, and IDCG caps the ideal hit count at `min(#relevant, k)` — forget that cap and a query with more relevant chunks than k produces `nDCG > 1.0`, an unmistakable "your IDCG is wrong" alarm.

## Worked example

Two retrievers, same golden query. `relevant = {A, B, C}` (three chunks needed), k = 5.

- **Retriever P (dense-only, no rerank):** returns `[A, junk, junk, B, C]` — relevant at ranks 1, 4, 5.
- **Retriever Q (dense + cross-encoder rerank):** returns `[A, B, C, junk, junk]` — relevant at ranks 1, 2, 3.

Both retrieved all three relevant chunks in the top 5. Compute all three metrics.

**recall@5** — both find all 3 of 3 relevant → **P = 1.0, Q = 1.0**. *Identical.* Recall is completely blind to the difference.

**MRR@5** — both have their first hit (A) at rank 1 → **P = 1.0, Q = 1.0**. *Also identical.* MRR only sees the first hit, and both nailed it.

**nDCG@5** — this is where they split. Binary gains:

```
Retriever P (ranks 1,4,5):
  DCG  = 1/log₂2 + 1/log₂5 + 1/log₂6 = 1.0000 + 0.4307 + 0.3869 = 1.8176
Retriever Q (ranks 1,2,3):
  DCG  = 1/log₂2 + 1/log₂3 + 1/log₂4 = 1.0000 + 0.6309 + 0.5000 = 2.1309
IDCG (ideal ranks 1,2,3, same for both):
       = 1.0000 + 0.6309 + 0.5000 = 2.1309

nDCG@5(P) = 1.8176 / 2.1309 = 0.853
nDCG@5(Q) = 2.1309 / 2.1309 = 1.000
```

**recall and MRR both said "these retrievers are identical." nDCG said Q is meaningfully better (1.00 vs 0.85).** The reranker pulled B and C up from ranks 4–5 to ranks 2–3 — invisible to recall (same set) and invisible to MRR (same first hit), but caught cleanly by nDCG. This is the concrete meaning of "recall@k is your floor, nDCG@k is your ranking quality," and it's exactly the signal you'd use to justify shipping the reranker: same recall, higher nDCG, better ordering into the context window where lost-in-the-middle (Lecture 10) punishes buried evidence.

## How it shows up in production

- **The two-week prompt safari.** Without the retrieval/generation split, every bad answer gets blamed on generation because that's the fast, visible knob. Meanwhile recall@5 = 0.55 caps you at 55% forever. Stage-1 numbers convert an untargeted prompt marathon into a one-afternoon "raise recall@5 from 0.55 to 0.80 by switching chunkers or adding hybrid," a change you can *prove* moved the floor.
- **"recall is great but users say ranking feels off."** recall@10 = 0.94, everyone celebrates, users still complain. nDCG@10 = 0.61 tells the real story: the right chunks are retrieved but ordered badly, so the top-3 you actually pass to a tight context budget miss them. The lever is the reranker, not the chunker — and only nDCG surfaced that.
- **Watching the wrong k.** A team reports recall@20 = 0.97 and can't understand why answers miss facts. They feed only top-5 to the LLM. Their real ceiling is recall@5 = 0.71. Always report recall at the *fed* k, and log both retrieved-k and fed-k so this can't hide.
- **Silent zero from re-chunking.** You improve the chunker, ship it, and recall craters to near-0 overnight — not because retrieval broke, but because the golden set's `relevant_chunk_ids` point at *old* chunk IDs that no longer exist. The metric reads a confident, catastrophic, and completely fake 0.0. (See the chunk-id trap below — this is the #1 way retrieval eval betrays you.)
- **Judge cost lives in Stage 2, not here.** Stage-1 retrieval metrics are pure arithmetic over ID lists — free, instant, deterministic, perfect for a fast CI gate on every PR. Stage-2 metrics (RAGAS, next lecture) are LLM-as-judge — slow and metered. Splitting the stages also splits the cost: gate cheaply on retrieval on every commit, run the expensive generation eval nightly.

## Common misconceptions & failure modes

- **"High recall means good retrieval."** It means the *floor* is high — the facts are in the window. It says nothing about ordering (nDCG) or noise (precision, Lecture 18). Recall is necessary, not sufficient.
- **"recall, MRR, and nDCG move together."** The worked example above is a direct counterexample: two retrievers, identical recall *and* identical MRR, materially different nDCG. Compute all three; each answers a different question.
- **"MRR is fine for multi-hop questions."** MRR only sees the first relevant hit. A question needing chunks A+B+C fused scores MRR = 1.0 the instant it finds A, even if B and C never appear. Use recall (did we get them all?) and nDCG (are they ranked well?) for multi-relevant cases; reserve MRR for one-chunk-suffices lookups.
- **"nDCG is overkill; recall is enough."** Only if relevance is truly binary and position doesn't matter. The moment some chunks are *more* relevant than others, or the moment ordering into a lost-in-the-middle context window matters (it always does), nDCG is the metric that sees what recall can't.
- **"Bigger k is a free recall win."** Raising k monotonically raises recall@k, but it costs prompt tokens, adds latency, invites distractor chunks, and can *lower* nDCG at the fed k by diluting the top. k is a tradeoff, not a dial you crank.
- **"I'll trust the aggregate over 60 cases."** Not until the hand-computed toy passes. A subtle discount or slicing bug is invisible in an aggregate and glaring in the toy. Verify small, then scale.
- **"nDCG can exceed 1.0, so my ranking is superhuman."** No — it means your IDCG is computed wrong (usually the `min(#relevant, k)` cap is missing, or graded IDCG isn't sorting gains descending). nDCG is normalized to [0,1] by construction; anything above 1.0 is a bug alarm.

## Rules of thumb / cheat sheet

- **Two-stage eval, always:** Stage 1 scores retrieved context vs `relevant_chunk_ids` (no LLM); Stage 2 scores the answer *given* that context. Localize every failure to one stage.
- **Compute all three retrieval metrics.** `recall@k` = floor (facts in window). `MRR@k` = first-hit rank (one-chunk lookups). `nDCG@k` = ranking quality (graded + position).
- **Measure recall@k at the k you FEED the LLM,** not the k you retrieve. That's your true generation ceiling.
- **`recall HIGH + nDCG LOW ⇒ reranker is the lever.`** `recall LOW ⇒ fix chunking / embeddings / k / hybrid first — a reranker can't help.`
- **nDCG discount is `1/log₂(rank+1)`, rank 1-indexed.** IDCG = DCG of the ideal ordering; cap ideal hits at `min(#relevant, k)`.
- **Toy-test before you trust:** relevant at ranks 2 and 5 ⇒ recall@5 = 1.0, MRR@5 = 0.5, nDCG@5 ≈ 0.624. `pytest` green on this first.
- **Use graded relevance for nDCG when you can label it** (e.g. 2 = directly answers, 1 = supporting, 0 = irrelevant). Binary nDCG is fine when you can't.
- **Key relevance labels on stable content hashes / source spans, never on ephemeral row indices** — see the trap below.
- **nDCG > 1.0 ⇒ your IDCG is wrong.** MRR = 1.0 on a multi-hop question ⇒ you probably only checked the first hit; look at recall too.

## The chunk-id-stability pitfall (do not skip)

Stage-1 metrics compare *retrieved IDs* against *golden relevant IDs*. That comparison is only meaningful if a chunk's ID means the same thing in the golden set and in the live index. It usually doesn't, and this silently zeroes your recall.

If you key relevance labels on **ephemeral identifiers** — the row index in a dataframe, an auto-increment DB id, the position in a list — then *any* re-chunk, re-parse, re-order, or re-ingest reshuffles those ids. Your golden set now points `relevant_chunk_ids: [42, 87]` at whatever happens to sit at positions 42 and 87 today, which is unrelated content. recall@k reads 0.0, the CI gate goes red, and you burn a day hunting a retrieval regression that never happened — the retriever is fine; the *labels* rotted.

**Fix: key relevance on stable content hashes or source spans.** A chunk's identity should derive from *what it is and where it came from*, not where it landed in a list:

```python
import hashlib

def chunk_id(text, source_file, page, char_start, char_end):
    # Stable across re-runs: same content + same source span → same id.
    basis = f"{source_file}|{page}|{char_start}:{char_end}|{text}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
```

Now a re-chunk that produces the *same* text from the *same* source span yields the *same* id, and your golden labels survive. If the chunk boundaries genuinely change (different text or span), the id changes too — which is correct: it's genuinely a different chunk, and you should re-review that label rather than have it silently match. Version the golden set (`golden_v1.jsonl` → `v2`) whenever chunking changes enough to move ids, and never edit labels in place. This is the single discipline that keeps Stage-1 numbers honest across the reindexes you know are coming.

## Connect to the lab

This lecture is the retrieval half of the Week 4 lab (spine step 2). You'll implement `recall_at_k`, `mrr_at_k`, and `ndcg_at_k` from scratch in `eval/retrieval_metrics.py` over ranked id lists vs the golden `relevant_chunk_ids`, then write `tests/test_retrieval_metrics.py` asserting them against the hand-computed toy (relevant at ranks 2 and 5 ⇒ MRR = 0.5, known nDCG ≈ 0.624) — `pytest` must be green before you trust any aggregate. Lecture 18 wires the generation half (RAGAS faithfulness / context recall / precision / answer relevancy) and the localization decision rule that consumes both stages.

## Going deeper (optional)

- **Manning, Raghavan & Schütze, *Introduction to Information Retrieval*** (free online — search the title). The "Evaluation in information retrieval" chapter has the canonical definitions of recall, MRR, DCG, and nDCG. This is the primary source; everything else is a gloss on it.
- **RAGAS documentation** (root: `docs.ragas.io`) — the "Metrics" section, especially `context_recall` and `context_precision`, which are the Stage-2 mirror of what you build here. You'll wire these in Lecture 18.
- **`ranx`** (GitHub: `AmenRa/ranx`) — a fast, well-tested Python IR-metrics library. Use it to *cross-check* your from-scratch implementations against an independent one — if your toy matches `ranx`, you've triangulated correctness.
- **TREC evaluation measures** — search "trec_eval nDCG MRR definitions" for the reference implementations the whole IR field validates against.
- **Original nDCG paper:** Järvelin & Kekäläinen, "Cumulated Gain-based Evaluation of IR Techniques" (2002) — search that exact title for the source of the log-discount formulation and the graded-relevance motivation.
- Search queries worth running: "nDCG worked example log2 discount", "binary vs graded relevance nDCG", "chunk id stability RAG evaluation content hash".

## Check yourself

1. Your RAG answers a question wrong. Your only metric is end-to-end "answer correct: yes/no." Why can't you decide whether to touch the chunker or the prompt, and what's the minimal change that lets you route the fix?
2. Two retrievers both score recall@5 = 1.0 and MRR@5 = 1.0 on a query needing three relevant chunks, but their nDCG@5 differs (0.85 vs 1.00). What is materially different between them, and which single technique most directly closes the gap?
3. You measure recall@50 = 0.98 but users complain the system misses facts. You feed the top 5 chunks to the LLM. What number should you have reported, and why is 0.98 misleading here?
4. Hand-compute nDCG@4 (binary) for a retriever returning relevant chunks at ranks 1 and 3, with two total relevant chunks. Show DCG, IDCG, and the ratio.
5. You improve your chunker, redeploy, and recall@k drops to ~0 across the whole golden set overnight, though spot-checks show retrieval works fine. What almost certainly happened, and how do you prevent it structurally?
6. When is MRR@k the right primary metric, and when will it actively mislead you? Give a concrete question type for each.

### Answer key

1. The single number blends **retrieval misses** (the chunk was never fetched) and **generation misses** (the chunk was fetched and the LLM botched it), and it can't attribute the failure to either side — so you'd be guessing which knob to turn. The minimal change is a **two-stage eval**: score retrieval in isolation (recall@k / nDCG@k against golden `relevant_chunk_ids`, no LLM) and, separately, score the answer given the retrieved context. Low retrieval score ⇒ chunker/embeddings/k/reranker; retrieval fine but answer wrong ⇒ prompt/model.

2. Both retrieved all three relevant chunks (same recall) and both put their *first* hit at rank 1 (same MRR), but one ordered the other two relevant chunks higher than the other did — nDCG is the only metric of the three that sees the *positions of all* relevant chunks, so it alone catches the ordering difference. The technique that most directly closes the gap is a **cross-encoder reranker**, which reorders the already-retrieved candidates to push the relevant ones up.

3. You should have reported **recall@5** — the k you actually feed the LLM — because that's the true ceiling on answer quality. recall@50 = 0.98 measures a candidate set 45 chunks of which never reach the context window; if recall@5 is, say, 0.71, then ~29% of the time the fact is retrieved-but-truncated before generation ever sees it. Measure recall at the *fed* k, not the retrieved k.

4. DCG@4 = 1/log₂(1+1) + 1/log₂(3+1) = 1/1.0 + 1/2.0 = 1.0 + 0.5 = **1.5**. IDCG@4 (ideal ranks 1 and 2) = 1/log₂2 + 1/log₂3 = 1.0 + 0.63093 = **1.63093**. nDCG@4 = 1.5 / 1.63093 = **0.9197**.

5. Your golden set's `relevant_chunk_ids` were keyed on **ephemeral identifiers** (row index / auto-increment id / list position), and re-chunking reshuffled them — the labels now point at unrelated content, so recall reads a fake 0. Prevent it structurally by keying relevance labels on **stable content hashes or source spans** (e.g. `sha256(source|page|char_span|text)`), so the same content from the same span always gets the same id, and by versioning the golden set when chunking changes.

6. MRR@k is the right primary metric when **one good chunk suffices** — a factoid lookup like "What is the default connect() timeout?" where the answer lives in a single place and you just need it near the top. It **misleads** on **multi-hop / multi-relevant** questions like "Compare the refund policies for the Pro and Enterprise plans," because MRR stops at the first relevant hit and reports success even if the other required chunks were never retrieved — there, recall (did we get them all?) and nDCG (are they ranked well?) are the metrics that tell the truth.
