# Lecture 4: Measuring Retrieval in Isolation — Golden Sets, recall@k, MRR

> The most expensive mistake in RAG is not a bad chunker or a slow reranker — it's flying blind. You wire up a pipeline, ask it three questions that happen to work, ship it, and then spend two weeks "improving the prompt" while the real bug is that the right chunk never lands in the top-5. This lecture teaches you the one discipline that prevents that: **measure retrieval alone, with numbers, before a single token of generation exists.** After this lecture you will be able to author an honest golden set, compute recall@k and MRR by hand and in code, spot the three anti-patterns that make those numbers lie to you, stand up a local embed-and-search harness that won't silently produce garbage, and run a full parser × chunker × k evaluation grid whose winning config you can defend — and whose losers you can each pin to one of the 7 failure points.

**Prerequisites:** Lecture 1 (two pipelines, 7 failure points), Lecture 3 (chunking strategies), embeddings and cosine/inner-product similarity, basic Python. · **Reading time:** ~26 min · **Part of:** Retrieval-Augmented Generation, Week 1

## The core idea (plain language)

A finished RAG system gives the user one number's worth of impression: "the answer was good" or "the answer was wrong." That single end-to-end judgment is **the least useful signal in the entire stack**, because a wrong answer has (at least) seven distinct causes and this number blends all of them into mush.

Recall the taxonomy from Lecture 1. Points **#1–#3 are retrieval/indexing failures** (the answer isn't in the corpus, or the right chunk ranked below k, or it got dropped assembling the prompt). Points **#4–#7 are generation failures** (the LLM had the context and still botched extraction, format, specificity, or completeness). Now imagine your only instrument is "final answer correct: yes/no." A failed case could be a retrieval miss (#1–3) *or* a generation miss (#4–7), and **the metric cannot tell you which.** So you cannot know whether to touch the chunker or the prompt. You will guess. You will guess wrong roughly half the time. And each wrong guess costs a reindex or a prompt-tuning cycle that changes the wrong thing.

The fix is embarrassingly simple and is the whole point of Week 1:

> **Cut the LLM out entirely. Measure retrieval as a standalone function: query in → ranked chunk IDs out. Score those rankings against a hand-authored ground truth. Now every number you see is a pure retrieval number, and every failure you see is provably #1, #2, or #3.**

This works because retrieval quality is a **hard ceiling** on answer quality. If the chunk containing the answer is not in the set you hand the LLM, no prompt, no model upgrade, no temperature setting can recover it — the information is simply absent from the context. So measuring the ceiling first, in isolation, is not a nicety; it's the only way to know whether your ceiling is even high enough to bother tuning generation. If recall@5 is 0.55, your generation ceiling is 55% no matter how good your generator is, and prompt work is premature.

Two ingredients make this measurement real: a **golden set** (the ground truth — questions paired with what a correct retrieval must contain) and two **metrics** computed against it (recall@k and MRR). Get the golden set wrong and both metrics become confident lies. So we start there.

## How it actually works (mechanism, from first principles)

### The golden set: your ground truth

A golden set is a small, hand-authored list of `(question, gold)` pairs where `gold` is what a correct retrieval must surface. Two common shapes:

- **`gold_substr`** — a distinctive substring the answer chunk must contain (e.g. `"30 seconds"`). Easy to author, matched with a simple `in` check. Good enough for Week 1.
- **`gold_chunk_id`** — the stable ID of the specific chunk that should be retrieved. More precise, survives phrasing changes, but requires deterministic chunk IDs and re-labeling whenever you re-chunk.

```json
{"q": "What is the default timeout for the connect() call?", "gold_substr": "30 seconds"}
{"q": "Which exception is raised when a pool is exhausted?", "gold_substr": "PoolTimeout"}
{"q": "How do I disable retries on idempotent requests?", "gold_substr": "max_retries=0"}
```

Size: **15–30 pairs for Week 1**, growing to 40–80 later (Week 4). Why this range? Below ~15 the metric moves in giant, meaningless jumps; above ~30 you're spending author-time you don't yet have. The arithmetic of *why small sets are noisy* is worth internalizing — it's the difference between a real improvement and a coin flip.

**The granularity trap (why tiny sets lie).** recall@k on N questions can only take N+1 distinct values: 0/N, 1/N, …, N/N. With **N = 10**, your resolution is 0.10 — a single question flipping from miss to hit moves recall by a full 10 points. So a change from 0.70 → 0.80 might be one lucky question, not a real gain. With **N = 30** the step is 0.033. Rule of thumb (approximate): on a set of size N, treat any recall delta smaller than about `1/N` as noise, not signal. A 5-point improvement on a 10-question set is *literally unrepresentable* — it rounds to either 0 or 10 points.

There's a probabilistic version of the same warning. If true recall is around 0.7, the standard error on a set of N questions is roughly `sqrt(0.7 × 0.3 / N)`. At N=10 that's ~0.14; at N=30 it's ~0.084; at N=100 it's ~0.046. So two configs measuring 0.70 and 0.75 on a 10-question set are statistically indistinguishable — the error bars swamp the gap. **This is why the Week 2 pitfall says "a 10-query golden set gives noisy deltas."** You can still *ship* a Week-1 gate on 15–30 questions; you just can't trust small deltas between configs, so prefer big, obvious wins.

### The three anti-patterns that make numbers meaningless

A golden set is only as good as its honesty. Three failure modes silently inflate your metrics until you trust a system that doesn't work:

**1. Gold-set leakage / triviality.** If `gold_substr` is a *rare exact string* — a unique error code, a UUID, an unusual product SKU — then lexical near-matching makes retrieval look brilliant even when semantic retrieval is weak. Worse, that exact string may not appear in a real user's phrasing at all. Example: gold_substr `"ERR_CONN_TIMEOUT_0x5F"` will match the one chunk that contains it, giving recall 1.0 — but a real user asks "why does my connection keep timing out?" and *that* query retrieves nothing useful. Your metric says 1.0; production is 0. **Fix:** use realistic *answer phrasings* a human would recognize as correct, not needle-in-haystack unique tokens.

**2. Copying chunk text verbatim into the question.** If you write the question by paraphrasing — or worse, pasting — the sentence from the gold chunk, you've guaranteed high embedding similarity between query and chunk. The retriever looks great because you handed it the answer key. Example of the disease: chunk says *"The connect() call defaults to a 30 second timeout before raising ConnectTimeout."* and your question is *"What does the connect() call default to — a 30 second timeout before raising ConnectTimeout?"* Of course it retrieves. **Fix:** write the question the way a user who *hasn't read the doc* would ask it. Close the source. Ask from intent, not from text.

**3. Too-small sets giving noisy deltas.** Covered above — the granularity and standard-error math. It belongs in this list because it's the anti-pattern people *don't see*: the numbers look precise (0.733!) but the confidence interval is enormous.

A quick mental test for a golden pair: *"If I deleted the answer chunk from the corpus, would a smart human still be able to guess my gold_substr from the question alone?"* If yes, your question leaks the answer. Rewrite it.

### recall@k, defined precisely

> **recall@k** = the fraction of golden questions whose **top-k retrieved chunks contain the gold answer.**

Per question it is binary: 1 if the gold appears anywhere in the top-k, else 0. Then average over all questions.

```
recall@k = (number of questions where gold is in top-k retrieved) / (total questions)
```

Note the subtlety for Week 1: our golden set has **one gold target per question**, so "recall" here is really *hit-rate@k* — did we catch the one right chunk? (The textbook multi-relevant-document definition — fraction of *all* relevant docs found — appears in Week 4 with `relevant_chunk_ids` lists.) For a single gold target, the two collapse to the same "is it in the top-k?" check. Keep this straight: same word, slightly different formula depending on whether gold is one item or a set.

**Worked micro-example.** Three questions, k = 3. For each, the rank at which the gold chunk appears (∞ = not in top-3):

| Question | rank of gold in top-3 | hit@3? |
|---|---|---|
| Q1 | 1 | ✅ |
| Q2 | ∞ (not retrieved) | ❌ |
| Q3 | 2 | ✅ |

recall@3 = 2/3 = **0.667**. Two of three questions found their answer in the top 3. Notice recall@k *ignores where* in the top-k the hit landed — rank 1 and rank 3 both count as a plain "yes." That blind spot is exactly what MRR fixes.

### MRR, defined precisely

> **MRR (Mean Reciprocal Rank)** = the mean, over all questions, of **1 / (rank of the first gold hit)**. If gold never appears (within your cutoff), that question contributes 0.

```
reciprocal rank of a question = 1 / rank_of_first_hit   (0 if no hit)
MRR = mean of reciprocal ranks over all questions
```

The reciprocal makes rank *hurt*: rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.333, rank 4 → 0.25, rank 5 → 0.2, rank 10 → 0.1. MRR rewards putting the answer *near the top*, which is what you want when one good chunk suffices and the LLM reads top-down.

**Same three questions, MRR:**

| Question | rank of gold | reciprocal rank |
|---|---|---|
| Q1 | 1 | 1/1 = 1.000 |
| Q2 | ∞ | 0.000 |
| Q3 | 2 | 1/2 = 0.500 |

MRR = (1.000 + 0.000 + 0.500) / 3 = 1.5 / 3 = **0.500**.

**Why you need both.** recall@k answers *"did we catch it at all within k?"* — the retrieval floor. MRR answers *"how high did we rank it?"* — ranking quality. They diverge in an informative way. Consider two retrievers both with recall@5 = 1.0 (every question's gold is somewhere in the top 5):

- Retriever A always ranks gold at position 1 → MRR = 1.0.
- Retriever B always ranks gold at position 5 → MRR = 0.2.

Identical recall@5, wildly different MRR. Retriever B will *bite you* the moment you tighten k to 3, or the moment "lost-in-the-middle" (Lecture from Week 3) buries that rank-5 chunk in the context. **High recall + low MRR is a flashing sign that a reranker (Week 2) is your highest-leverage next move** — the right chunks are being retrieved, they're just ordered badly.

## Worked example

Let's evaluate one config end to end with real arithmetic. Golden set of **5 questions**, retriever configured at **k = 5**. We run each query and record the rank of the gold chunk:

| Q | gold rank in top-5 | hit@5 | hit@3 | 1/rank |
|---|---|---|---|---|
| Q1 | 1 | ✅ | ✅ | 1.000 |
| Q2 | 4 | ✅ | ❌ | 0.250 |
| Q3 | ∞ (not in top-5) | ❌ | ❌ | 0.000 |
| Q4 | 2 | ✅ | ✅ | 0.500 |
| Q5 | 8 (in top-10, not top-5) | ❌ | ❌ | 0.000 |

Compute:

- **recall@5** = hits@5 / 5 = 3/5 = **0.60**
- **recall@3** = hits@3 / 5 = 2/5 = **0.40**
- **MRR** (over first hit within top-5) = (1.000 + 0.250 + 0.000 + 0.500 + 0.000)/5 = 1.75/5 = **0.35**

Now *read* these numbers like an engineer, tying each miss to a failure point:

- **Q3** — gold nowhere in the top-5 (and suppose nowhere in top-50 either). Either the chunk doesn't exist in the corpus (**failure point #1, missing content** — maybe your parser dropped that table) or it exists but ranks catastrophically low (**#2, missed top-k**). Pull the full ranked list to disambiguate: if it's at rank 200, it exists but retrieval is weak (#2); if it's absent entirely, the offline pipeline destroyed it (#1) and no k will save you.
- **Q5** — gold at rank 8. It *exists and is findable* (not #1), but it **ranked below k** (**#2, missed top-k**). Two independent fixes: raise k from 5 to 10 (cheap, but costs prompt tokens and can dilute the context later), or add a reranker to push it up (better — fixes ranking, not just the cutoff). Bumping k to 10 would flip Q5 to a hit and lift recall@10 to 0.80.
- **Q2** — gold at rank 4: a hit@5 but a *weak* one, dragging MRR down. Not a failure yet, but if you ever run k=3 it becomes a #2 miss. This is the "recall OK, ranking bad" pattern that argues for reranking.

That's the whole discipline: numbers → which of the 7 → which knob. Compare that to a combined end-to-end score of "3/5 answers correct," which would have told you *nothing* about whether to touch the parser, k, or a reranker.

### The local embed + FAISS harness (and the rule that makes or breaks it)

To generate those ranks you need an in-process retriever. The Week-1 stack is deliberately zero-infra: local embeddings via `sentence-transformers` (`BAAI/bge-small-en-v1.5` — 384-dim, fast on CPU) and a FAISS `IndexFlatIP` (exact inner-product search, no ANN approximation to muddy your measurement).

```python
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

M = SentenceTransformer("BAAI/bge-small-en-v1.5")

def build(chunks):
    emb = M.encode(chunks, normalize_embeddings=True)          # <-- normalize on INDEX side
    idx = faiss.IndexFlatIP(emb.shape[1])                      # inner-product index
    idx.add(np.asarray(emb, "float32"))
    return idx

def query(idx, chunks, q, k=5):
    qe = M.encode([q], normalize_embeddings=True)              # <-- normalize on QUERY side too
    D, I = idx.search(np.asarray(qe, "float32"), k)
    return [(chunks[i], float(D[0][r])) for r, i in enumerate(I[0])]
```

**The critical normalization rule — read this twice.** `IndexFlatIP` computes the **inner product** (dot product) between the query vector and each chunk vector. What you actually *want* for semantic similarity is **cosine similarity**. Here is the identity that ties them together:

```
cosine(a, b) = (a · b) / (||a|| × ||b||)
```

Inner product equals cosine similarity **only when both vectors have length 1** (i.e., `||a|| = ||b|| = 1`), because then the denominator is 1×1 = 1 and `a · b = cosine(a, b)`. So:

> **For an inner-product index to compute cosine similarity, every vector must be L2-normalized — on BOTH the index side and the query side. Normalize one side but not the other and your scores are garbage, silently.**

Why *silently*, and why it bites so hard: if you forget `normalize_embeddings=True` on, say, the query side, the search still runs, still returns k results, still produces plausible-looking float scores. Nothing crashes. But now inner product is scaling every chunk's score by that chunk's *magnitude*, so long chunks (larger norm) get ranked above short relevant ones purely because they're longer — a length bias that has nothing to do with relevance. Your recall@k quietly tanks and you have no error message to chase. This is a top-3 "why is my retrieval bad" bug in practice, and it's invisible unless you know to check. `sentence-transformers`' `normalize_embeddings=True` does the L2-normalization for you; the equivalent manual step is `emb / np.linalg.norm(emb, axis=1, keepdims=True)`. Set it on **both** `encode` calls or neither (and if neither, use `IndexFlatL2` and reason in distances instead).

## How it shows up in production

- **The "we tuned the prompt for two weeks" story.** Without isolated retrieval numbers, teams attribute every bad answer to generation because that's the part they can see and edit fast. Meanwhile recall@5 is 0.55 and the ceiling is 55%. Isolated measurement converts an untargeted two-week prompt safari into a one-afternoon "raise recall@5 from 0.55 to 0.80 by switching chunkers" — a change you can *prove*.
- **Reindex cost forces you to measure first.** From Lecture 1: retrieval fixes (parser, chunker, embedding model) require a full reindex — minutes to hours, and on large corpora, real money. You do not want to reindex on a hunch. The evaluation grid lets you pick the winning config *once* and reindex *once*, instead of reindexing repeatedly while guessing.
- **The normalization bug in a code review.** A teammate refactors the encoder into a shared util and drops `normalize_embeddings=True` from one call site. No test fails (there was no retrieval test), no exception fires, and retrieval quality degrades ~15–30 points on the next deploy. The recall@5 **pytest gate** is exactly what catches this — the build goes red, the diff is obvious.
- **Golden-set rot.** You change your chunker; if your golden set uses `gold_chunk_id`, every ID is now stale and recall reads a bogus 0.0. This is why Week 1 favors `gold_substr` (robust to re-chunking) and why Week 4 keys labels on *content hashes*, not row indices. Design for the reindex you know is coming.
- **A fake 0.95 that fools a stakeholder.** A leaked/trivial golden set (rare exact strings, questions paraphrased from chunks) reports 0.95 in the demo and 0.60 in production. The gap surfaces only when real users hit it — the most expensive place to discover it. Honest authoring is cheaper than a bad launch.

## Common misconceptions & failure modes

- **"A high recall@k means the system is good."** It means retrieval's *ceiling* is good. Generation (#4–7) can still ruin the answer. recall@k is necessary, not sufficient — it's a floor check, not a done check.
- **"recall@k and precision are the same thing."** No. recall@k asks *did we find the answer in the top-k*. Precision asks *what fraction of the top-k were relevant* (noise). A retriever can have recall@10 = 1.0 and terrible precision (9 junk chunks + 1 gold), which wastes tokens and, later, dilutes the LLM's context. Week 4 measures precision explicitly.
- **"MRR and recall must move together."** They don't. High recall + low MRR = right chunks, wrong order → add a reranker. Low recall + (whatever) MRR = the chunk isn't being found at all → fix chunking/parsing/k, a reranker can't help.
- **"Bigger k is a free win."** Raising k monotonically raises recall@k (more slots, more chances) but costs prompt tokens, adds latency, invites distractor chunks (#4), and can trigger lost-in-the-middle (#3) downstream. k is a real tradeoff, not a dial you crank to 100.
- **"My scores look reasonable, so normalization must be fine."** Missing normalization does not throw — it silently reorders by magnitude. "Reasonable-looking" scores prove nothing. Assert it in code or check a known-good query by hand.
- **"15 questions is enough to compare two close configs."** It's enough to *gate* a big win and to build the habit, but a 0.72-vs-0.75 comparison on 15 questions is within noise. Prefer obvious wins; grow the set (Week 4) before trusting small deltas.
- **"gold_substr matching is exact truth."** A substring check has false negatives (the answer is present but phrased differently than your gold string) and false positives (your string appears in an unrelated chunk). It's a cheap, honest-enough proxy for Week 1 — know its limits, and lowercase both sides so casing doesn't cause phantom misses.

## Rules of thumb / cheat sheet

- **Always measure retrieval alone before generation.** Query → ranked IDs → score vs golden. No LLM in the loop this week.
- **Golden set: 15–30 hand-authored pairs** for Week 1 (40–80 later). Honest, realistic, user-phrased.
- **Author questions with the source closed.** If a smart human could guess your gold_substr from the question alone, it leaks — rewrite it.
- **Avoid rare-exact-string gold** (error codes, UUIDs). Use realistic answer phrasings.
- **Treat any recall delta smaller than ~1/N as noise** on a set of size N.
- **recall@k** = fraction of questions whose top-k contains gold. **MRR** = mean of 1/(rank of first hit). Compute **both**.
- **High recall + low MRR ⇒ reach for a reranker** (Week 2). **Low recall ⇒ fix chunking/parsing/k first.**
- **`normalize_embeddings=True` on BOTH index and query** with `IndexFlatIP`, or your cosine scores are garbage — silently.
- **Use `IndexFlatIP` (exact) for eval,** not an ANN index — you want to measure retrieval quality, not ANN approximation error, this week.
- **Sweep the grid:** `{parser} × {chunker} × {k ∈ 3,5,10}`, dump per-config metrics to `runs/`, sort a comparison table by recall@5.
- **Commit a pytest recall@5 gate** at a threshold set from your *observed* baseline (e.g. best config 0.82 → gate at ~0.75, leaving headroom for noise).
- **For every low-recall config, name the failure point it hits** (#1 missing, #2 missed top-k, #3 dropped in assembly) — that's the debugging muscle.
- **Prefer `gold_substr` over `gold_chunk_id`** in Week 1 so re-chunking doesn't invalidate your labels.

## Connect to the lab

This lecture is the measurement half of the Week 1 lab (spine steps 5–7). You'll hand-author `golden.jsonl` (≥15 honest `q → gold_substr` pairs), implement `recall_at_k` and MRR in `src/evaluate.py`, and run the full grid — `{pymupdf, docling} × {fixed, recursive, structure} × k∈{3,5,10}` — dumping each result into `runs/` and sorting a pandas table by recall@5. Then you'll commit the `tests/test_recall.py` gate asserting your best config clears a threshold *you* set from the observed baseline. The Definition of Done requires you to say out loud which of the 7 failure points each losing config hits — that sentence is the real deliverable of this whole week.

## Going deeper (optional)

- **Barnett et al., "Seven Failure Points When Engineering a Retrieval Augmented Generation System" (2024)** — the taxonomy every miss maps to. Search that exact title.
- **`sentence-transformers` documentation** (root: `sbert.net`) — see the "Semantic Search" and "Computing Embeddings" pages, especially the note that inner-product search requires normalized embeddings. Also read up on the `BAAI/bge-small-en-v1.5` model card on Hugging Face (search "bge-small-en-v1.5 model card") for its 512-token limit and the recommended query prefix.
- **FAISS wiki** (repo: `facebookresearch/faiss` on GitHub) — the "Getting started" and "Guidelines to choose an index" pages explain why `IndexFlatIP`/`IndexFlatL2` are exact and when ANN indexes (IVF, HNSW) trade accuracy for speed.
- **RAGAS documentation** (root: `docs.ragas.io`) — Week 4's tool, but skim its "Metrics" section now to see recall/precision/nDCG defined for the multi-relevant case you'll grow into.
- **TREC / IR evaluation background** — for the canonical definitions of MRR and nDCG, search "TREC evaluation measures MRR nDCG" or read the metrics chapter of *Introduction to Information Retrieval* (Manning, Raghavan, Schütze — free online, search the title).
- Search queries worth running: "recall@k vs MRR retrieval evaluation", "why normalize embeddings cosine inner product FAISS", "golden set RAG evaluation best practices".

## Check yourself

1. You have a single end-to-end metric ("answer correct: yes/no") reading 60%. A stakeholder asks whether to invest in a better chunker or a better prompt. Why can't you answer, and what's the minimal change that lets you?
2. Two retrievers both score recall@5 = 1.0, but retriever A has MRR = 0.95 and retriever B has MRR = 0.30. What is materially different between them, and which single Week-2 technique most directly helps B?
3. Your golden set uses `gold_substr = "ERR_TZ_0x91A"` and reports recall@5 = 1.0, but real users complain retrieval is useless. Name the anti-pattern and explain the mechanism by which the number lied.
4. Explain, using the identity `cosine(a,b) = (a·b)/(||a||·||b||)`, exactly why you must set `normalize_embeddings=True` on both the index and query sides when using `IndexFlatIP`. What symptom appears if you set it on only one side?
5. On a golden set of 12 questions, config X scores recall@5 = 0.75 and config Y scores 0.83. Is Y meaningfully better? Show the reasoning with numbers.
6. A question's gold chunk appears at rank 30 (well outside top-5). Which of the 7 failure points is this, is it fixable at query time or only by reindexing, and what are two distinct fixes?

### Answer key

1. The 60% blends retrieval misses (#1–3) and generation misses (#4–7); the metric can't attribute the failures to one side, so you can't know which knob helps. Minimal change: measure **retrieval in isolation** — build a golden set and compute recall@k with no LLM in the loop. If recall@5 is low, the ceiling is retrieval and the chunker wins your investment; if recall@5 is high but answers are still wrong, it's generation and the prompt/model wins.

2. Both find the gold chunk within the top-5 (equal recall@5), but A ranks it near position 1 while B buries it near position 5 (MRR 0.30 ≈ average rank ~3.3). B's *ranking* is poor even though its *retrieval set* is complete. A **cross-encoder reranker** (Week 2) directly reorders the already-retrieved candidates to push the gold chunk up — the exact fix for "recall OK, MRR bad."

3. **Gold-set leakage / triviality via a rare exact string.** `ERR_TZ_0x91A` is a unique token that appears in exactly one chunk, so any retrieval that surfaces that chunk matches trivially and recall reads 1.0. But real users never type that code — they ask "why does my timezone conversion keep erroring?" — a semantically distant query that the retriever handles far worse. The metric measured an artificial needle-in-haystack lookup, not the realistic semantic retrieval users actually need.

4. `IndexFlatIP` returns the inner product `a·b`. You *want* cosine similarity, which is `(a·b)/(||a||·||b||)`. When both vectors are L2-normalized, `||a|| = ||b|| = 1`, the denominator becomes 1, and `a·b` **equals** cosine — so an inner-product index computes cosine correctly. Normalize only one side and the other side's magnitude still scales every score, so results are ranked partly by vector *length* (e.g., longer chunks) rather than pure direction/relevance. The symptom is silent: no error, plausible-looking scores, but degraded recall — often a length bias favoring long chunks.

5. Not reliably. On N=12 the metric's resolution is 1/12 ≈ 0.083, so 0.75 vs 0.83 is roughly a **one-question difference** (9/12 vs 10/12). The standard error near p≈0.8 is about `sqrt(0.8×0.2/12) ≈ 0.115`, larger than the 0.08 gap — the difference is within noise. Treat it as a tie; either enlarge the golden set (toward 30–80) or look for a bigger, clearly-out-of-noise win before declaring Y better.

6. This is **#2, missed top-k** (the chunk exists and is findable — it's at rank 30 — it just ranked below your cutoff of 5). It is **query-time fixable** (no reindex needed, since the vector is already in the index): fix 1 — raise k (e.g. to 10 or beyond) so the chunk enters the returned set; fix 2 — add a reranker or improve the retriever (hybrid/BM25 for exact terms) so the chunk ranks higher. If it were absent entirely (rank ∞), that would be **#1, missing content**, fixable only by reindexing with a better parser/chunker.
