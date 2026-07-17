# Lecture 14: Fine-Tuning Embedding Models with Hard-Negative Mining

> Everyone reaches for the generator when RAG disappoints. That's usually the wrong lever. If your model is confidently answering the wrong question, the problem is almost never the model's prose — it's that the right chunk never made it into the context window. A generator can only reason over what retrieval hands it, and a generic off-the-shelf embedder does not know that in your domain "settlement" means the finance thing, not the housing thing. This lecture makes the case that a domain-tuned embedder is often the biggest, cheapest RAG win available, then shows you how to actually earn it: the contrastive objective as `sentence-transformers` implements it, how in-batch negatives give you a huge free training signal, how to build `(query, positive)` pairs, and the centerpiece — **hard-negative mining**, where the real gains live, along with the false-negative trap that quietly teaches your model the opposite of what you want. After this you can fine-tune `bge-small` on a laptop, measure recall@5 before and after, and report the delta with a straight face.

**Prerequisites:** Phase 4 (a working RAG baseline with a chunked, embedded corpus), the concept of cosine similarity / vector search (no math needed), Python + Hugging Face `datasets`, basic probability (a softmax over a row) · **Reading time:** ~26 min · **Part of:** Phase 08 — Model Adaptation & Fine-Tuning, Week 4

## The core idea (plain language)

A RAG pipeline is a two-stage machine: **retrieve**, then **generate**. Retrieval decides *which* text the model gets to see; generation decides *what to say* about it. These two stages fail in completely different ways, and — this is the part people miss — **you cannot fix a retrieval failure by improving the generator.** If the correct passage isn't in the top-k that got stuffed into the prompt, no amount of generator fine-tuning conjures it back. The model will fluently answer using the wrong context. You'll read the output, see confident prose, and blame the LLM. The LLM did its job perfectly on garbage input.

So before you spend a weekend QLoRA-tuning a 7B generator, ask one question: **when the answer is wrong, is the right chunk in the retrieved context or not?** If it's *not there*, you have a retrieval bottleneck, and tuning the generator is polishing the wrong surface. Tune retrieval first.

Retrieval quality is dominated by one component: the **embedding model**. It maps every query and every chunk to a vector, and retrieval is just "find the chunks whose vectors are closest to the query's vector." A generic embedder (trained on web text, Wikipedia, MS MARCO) has a decent general sense of similarity but no idea about *your* domain's vocabulary, abbreviations, or which distinctions matter. In a medical corpus it may think "hypertension" and "hypotension" are near-neighbors (they share almost every character and co-occur constantly) when they are clinical opposites. In a legal corpus it can't tell a *motion to dismiss* from a *motion to compel*. Fine-tuning the embedder on your domain's `(query, correct-doc)` pairs teaches it exactly these distinctions.

Here's the economics, which is why this lecture exists: fine-tuning an embedder like `BAAI/bge-small-en-v1.5` (33M parameters) is **orders of magnitude cheaper** than fine-tuning a generator. It runs in minutes on a free Colab GPU, or on a CPU if you're patient. You need a few hundred to a few thousand `(query, positive)` pairs, not tens of thousands. And a good embedding fine-tune routinely moves recall@5 by double-digit percentage points — which lifts *every* downstream answer, because now the generator sees the right context. Bigger win, lower cost, less risk. That's the trade you're being asked to internalize.

## How it actually works (mechanism, from first principles)

### What "training an embedder" even means

An embedding model is a small transformer that eats a piece of text and emits one fixed-length vector (for `bge-small`, 384 numbers). "Similar meaning → nearby vectors" is the entire contract. Fine-tuning **nudges the model's weights so that the vectors it produces put your queries closer to their correct documents and farther from the wrong ones.** We never touch the math of the vector space directly — we just show the model examples of "these two belong together, those two don't" and let gradient descent reshape the geometry.

The training signal comes from a **contrastive loss**. The canonical choice in `sentence-transformers`, and the one you should default to, is `MultipleNegativesRankingLoss` (MNRL, sometimes called InfoNCE). Its job in one sentence: **pull each query and its positive document together, and push that query away from every other document in the batch.**

### In-batch negatives: the free lunch

Here's the clever, cheap trick that makes MNRL so effective. Suppose you feed it a batch of 4 `(query, positive)` pairs:

```
q1 ── d1   (q1's correct doc)
q2 ── d2
q3 ── d3
q4 ── d4
```

For query `q1`, `d1` is the positive. What are the negatives? **Every other document in the batch** — `d2`, `d3`, `d4`. They belong to *other* queries, so for `q1` they're presumed wrong. You didn't have to label them; they come free with the batch. This is the in-batch negatives idea, and it's why you only need to supply *positive pairs* to train — the negatives are manufactured from the batch itself.

Mechanically, MNRL computes a similarity score between every query and every document in the batch, forming a grid:

```
         d1     d2     d3     d4
   q1 [ 0.82   0.31   0.10   0.25 ]   ← correct answer is d1 (the diagonal)
   q2 [ 0.19   0.75   0.22   0.14 ]   ← correct is d2
   q3 [ 0.08   0.30   0.88   0.19 ]
   q4 [ 0.27   0.12   0.15   0.79 ]
```

Then it treats **each row as a classification problem**: "which of these 4 documents is the right one for this query?" The correct answer is always on the diagonal. It runs a softmax across the row and applies cross-entropy, pushing the diagonal score up and the off-diagonal scores down. Do this over thousands of batches and the model learns to make the diagonal light up.

Two practical consequences fall straight out of this mechanism:

1. **Bigger batches = harder, better training.** A batch of 64 gives each query 63 negatives to be pushed away from; a batch of 8 gives 7. More negatives per step means a sharper gradient and a better model. This is the rare case in deep learning where you *want* the largest batch that fits in memory. For `bge-small` on a free T4 you can often run batch sizes of 64–128.

2. **Duplicate or near-duplicate texts in a batch poison it.** If `d3` is actually a great answer for `q1` too (a near-duplicate of `d1`), the loss punishes the model for scoring `q1–d3` high — teaching it the *wrong* lesson. This is the false-negative problem in miniature, and it comes back with a vengeance once we add mined hard negatives.

### Why in-batch negatives alone aren't enough

In-batch negatives are *random* negatives. For query "what is the refund window for enterprise plans?", a random negative might be a doc about "office parking policy" — trivially, obviously unrelated. The model gets that right on day one. Training on easy negatives teaches it almost nothing new; the gradient is tiny because the model is already confident.

The documents that actually hurt you in production are the **plausible-but-wrong** ones: the doc about "refund window for *personal* plans," or "*cancellation* window for enterprise." Those share vocabulary, topic, and structure with the correct doc. A generic embedder frequently ranks them *above* the true answer. Those are the negatives worth training on — and they almost never show up as random in-batch negatives. You have to go find them. That's hard-negative mining.

### Hard-negative mining — where the gains come from

The procedure is delightfully simple and uses infrastructure you already have from Phase 4:

```
for each (query, positive_doc) in your training pairs:
    1. run the query through your CURRENT retriever
    2. pull the top-N results (say N=20) from your real corpus
    3. discard the known positive if it appears
    4. the remaining high-ranked-but-wrong docs are your HARD NEGATIVES
    5. keep a few (e.g. top 3-5) as explicit negatives for that query
```

You're weaponizing your existing embedder against itself: "show me the documents you *think* are relevant but aren't." Those are precisely the confusions the fine-tune needs to correct. You then feed triples `(query, positive, hard_negative)` to MNRL, which now pushes the query away from a genuinely tricky distractor instead of a random one. This single change — replacing (or augmenting) random negatives with mined hard negatives — is responsible for the bulk of the recall improvement you'll measure. It is not a nice-to-have; it *is* the technique.

### The false-negative trap (read this twice)

Here is the failure mode that separates a lift from a regression. When you mine hard negatives by "take the top-20 my retriever returns, minus the labeled positive," some of those "negatives" are **actually relevant** — your labels are incomplete. Real corpora have multiple passages that answer the same question; you only labeled one as the positive. So your mining step happily hands the model a *correct* document and says "push this away from the query."

That's not a small annoyance. You are now training the model to **rank a good answer as bad** — the exact opposite of the goal. Mine aggressively without checking, and you can produce a fine-tune whose recall@5 goes *down*. The standard defenses:

- **A score margin / threshold.** Don't take a mined doc as a negative if its similarity to the query is *too close* to the positive's similarity — that's the signature of a true relevant doc, not a distractor. A common heuristic: require the negative's score to be below, say, 0.95× the positive's, or below some absolute cosine cutoff.
- **Skip the very top ranks.** Mine negatives from ranks ~10–50 rather than ranks 1–5. The top handful is where accidental true-positives cluster.
- **Cross-encoder sanity check.** Run a stronger cross-encoder reranker over your mined negatives and drop any it scores as highly relevant. `sentence-transformers`' `mine_hard_negatives()` utility supports exactly this and will do the margin filtering for you.
- **Eyeball a sample.** Before training, print 20 random `(query, mined_negative)` pairs and read them. If more than one or two look like real answers, tighten your filter. This five-minute check saves you a failed training run.

## Worked example

Let's make the whole thing concrete with a small internal-docs RAG system.

**Baseline (Phase 4).** Corpus: 4,000 chunks from your company's support and product docs, embedded with stock `bge-small-en-v1.5`, stored in a vector DB. You built a held-out eval set of 200 real user questions, each hand-labeled with the chunk that actually answers it. You measure **recall@5**: for what fraction of the 200 questions is the correct chunk in the top-5 retrieved?

```
Baseline recall@5 = 128 / 200 = 0.64
```

So on 36% of questions, the generator never even sees the right chunk. That's your ceiling problem — no generator tuning fixes those 72 questions.

**Build training pairs.** Separately from the eval set, you assemble 1,500 `(query, positive)` pairs: past support tickets paired with the doc that resolved them, plus some synthetic questions generated from doc chunks. (Keep these strictly disjoint from the 200 eval questions — contamination here inflates your recall exactly like it inflates any other eval.)

**Mine hard negatives.** For each of the 1,500 queries, you run it through the *current* `bge-small` retriever, take ranks 10–30, filter out anything scoring within 5% of the positive's score, and keep the top 3 survivors. You now have 1,500 triples `(query, positive, hard_neg × 3)`. You spot-check 20 of them; two look like they might actually be relevant, so you tighten the margin from 5% to 8% and re-mine. Now the sample looks clean.

**Train.** `sentence-transformers`, `MultipleNegativesRankingLoss`, batch size 64, 2 epochs, learning rate 2e-5. On a free Colab T4 this finishes in about 8–12 minutes.

**Re-embed and re-measure.** Critical operational step people forget: **a fine-tuned embedder means every chunk's vector changed, so you must re-embed the entire corpus** with the new model and rebuild the index before evaluating. Then re-run the same 200 held-out questions:

```
Fine-tuned recall@5 = 168 / 200 = 0.84
```

**Report the delta:**

```
                  recall@5
  baseline          0.64
  fine-tuned        0.84
  absolute delta   +0.20   (20 percentage points)
  relative delta   +31%
  questions fixed   40 of the previously-missed 72
```

That +0.20 recall means 40 questions that used to get wrong context now get the right chunk — and *every one of those* is a downstream answer that just went from wrong to right, for ~12 minutes of GPU time and no change to the generator. A generator fine-tune would have cost hours of GPU plus a retention scorecard — and still couldn't have fixed those 40, because the context was never there.

## How it shows up in production

- **The re-embedding cost is real and recurring.** Every time you ship a new embedder version you must re-embed the whole corpus and rebuild the vector index. For 4,000 chunks that's seconds; for 40 million it's a batch job with a cost line. Budget for it, and treat "embedder version" as a first-class artifact — a query embedded with v2 must **never** be searched against a v1 index. Mismatched embedder versions between query-time and index-time is a silent, brutal bug: similarities become meaningless and recall craters with no error thrown.

- **Latency and serving barely change.** `bge-small` fine-tuned is the same size and speed as `bge-small` stock — 384-dim vectors, same index, same query latency. Unlike a generator fine-tune, you're not adding VRAM, tokens/sec regressions, or a bigger model to serve. The quality win is essentially free at inference time.

- **Hard-negative mining is a batch job you'll re-run.** As your corpus grows and drifts, last quarter's mined negatives go stale. Wire mining into a repeatable script, not a one-off notebook cell, because you *will* re-mine when you add documents or retrain.

- **The false-negative trap shows up as a "why did retrieval get *worse*?" incident.** If a fine-tune regresses recall, the first suspect is contaminated negatives (true positives mined as negatives). The second is eval contamination (training pairs leaked into the eval set, making the *baseline* look artificially strong or the fine-tune artificially good). Both are debuggable by reading a sample of the data — which is why you keep the mining and eval sets versioned and inspectable.

- **You need retrieval metrics wired up *before* you tune, not after.** recall@k (and its cousins MRR, nDCG) on a held-out, hand-labeled query set is the whole game. Without it you cannot tell a real win from a lucky-feeling one, and you can't catch a false-negative regression. This is the same discipline as the generator eval harness from Phase 7, applied to retrieval.

## Common misconceptions & failure modes

- **"Retrieval is bad, so let's fine-tune the generator."** The most expensive mistake in the phase. Generator tuning cannot add missing context. Diagnose first: is the right chunk in the retrieved top-k? If no → retrieval problem → tune the embedder.
- **"More training data beats better negatives."** False. 1,500 pairs with well-mined hard negatives beats 15,000 pairs with only random in-batch negatives. The *hardness* of the negatives carries the signal.
- **"I fine-tuned, recall went down, embeddings must not work for my domain."** Almost always false-negative contamination in the mined negatives, or eval leakage. Inspect the data before concluding the method failed.
- **"I trained a great embedder" — but forgot to re-embed the corpus.** Then you searched new-model queries against old-model vectors. Garbage. Re-embed everything, always.
- **"Smaller batch is fine."** For MNRL specifically, batch size *is* a quality knob because it sets the number of in-batch negatives. Tiny batches leave performance on the table. Maximize it within memory.
- **"Cosine vs dot product doesn't matter."** It does — `bge` models are trained for cosine similarity and expect normalized embeddings; MNRL's default similarity and your index's metric must agree with what the base model expects, or you quietly degrade both.
- **Over-training.** 1–3 epochs is the norm. Many epochs on a few thousand pairs overfits the embedder to your exact phrasings and can *hurt* generalization to real user queries. Watch a held-out recall, not just the loss.

## Rules of thumb / cheat sheet

- **Diagnose before you tune:** wrong answer + right chunk missing from context → retrieval problem → tune the embedder, not the generator.
- **Base model:** `BAAI/bge-small-en-v1.5` (384-dim, 33M params) for CPU/laptop/Colab; `bge-base` or `bge-large`, or `e5`/`gte` families, when you have GPU budget and need more headroom.
- **Loss:** `MultipleNegativesRankingLoss` is the default. You supply positive pairs; it makes negatives from the batch.
- **Batch size:** as large as fits (64–128 for `bge-small` on a T4). It's a quality knob, not just a speed knob.
- **Epochs:** 1–3. More overfits on small data.
- **Learning rate:** ~2e-5 is a safe start for `sentence-transformers` fine-tunes (approximate).
- **Hard negatives:** mine ranks ~10–50 from your own retriever; filter out anything within ~5–10% of the positive's score; keep 3–5 per query; **eyeball 20 before training.**
- **After training:** re-embed the *entire* corpus, rebuild the index, then measure. Never mix embedder versions across index and query.
- **Metric:** recall@5 (and MRR/nDCG) on a hand-labeled, contamination-free held-out query set. Report absolute *and* relative delta, with n.

## Connect to the lab

This lecture is the theory behind **Week 4, Lab step 5**: take `bge-small-en-v1.5`, build `(query, positive)` pairs from your domain, **mine hard negatives with your Phase 4 retriever**, train with `MultipleNegativesRankingLoss` for 1–3 epochs, and measure **recall@5 before/after** on a held-out query set — the Definition of Done requires a *measurable* margin, so wire up `recall_eval.py` first and record the baseline before you touch anything. Tie the result back to your Phase 4 RAG baseline: the same corpus, the same eval questions, one swapped embedder.

## Going deeper (optional)

- **`sentence-transformers` training documentation** (sbert.net) — the canonical source. Read the "Training Overview," the `MultipleNegativesRankingLoss` reference, and the **`mine_hard_negatives`** utility docs. This is the single most useful resource; everything in this lecture maps onto its API.
- **The MTEB leaderboard** (Massive Text Embedding Benchmark, hosted on Hugging Face) — for choosing a base embedder by task and language. Search: "MTEB leaderboard Hugging Face."
- **BGE model cards** on the Hugging Face Hub (`BAAI/bge-small-en-v1.5`) — note the query-prefix instruction convention some `bge`/`e5` models expect.
- **The DPR and "Dense Passage Retrieval" paper** for the origin of in-batch and hard negatives in retrieval — search: "Dense Passage Retrieval Karpukhin 2020." Read for intuition, skip the math.
- **The Sentence-BERT paper** (Reimers & Gurevych) for why bi-encoders and this training setup exist — search: "Sentence-BERT Reimers Gurevych."
- Search queries when you're stuck: "sentence-transformers hard negative mining false negatives," "bge fine-tuning recall improvement," "MultipleNegativesRankingLoss batch size."

## Check yourself

1. Your RAG system gives a confidently wrong answer. What single diagnostic question decides whether to tune the embedder or the generator?
2. In `MultipleNegativesRankingLoss`, where do the negative documents come from if you only supply `(query, positive)` pairs — and why does batch size therefore affect model quality?
3. What is a "hard negative," why do random in-batch negatives fail to provide them, and where do you get them instead?
4. Explain the false-negative trap in one sentence, and name two concrete defenses.
5. You fine-tuned the embedder and recall@5 didn't move on your held-out set — but you never rebuilt the vector index. What did you actually measure?
6. Give the two-number way to report an embedding fine-tune result and say why one number alone is misleading.

### Answer key

1. **"Is the correct chunk present in the retrieved top-k that the generator saw?"** If it's absent, retrieval is the bottleneck — tune the embedder, because no generator change can reason over context it never received. If it's present but the answer is still wrong, that's a generation problem.
2. The negatives are the **other queries' positive documents in the same batch** (in-batch negatives) — every off-diagonal cell in the query×document similarity grid. Because each query is contrasted against all *other* documents in the batch, a larger batch means more negatives per gradient step, a sharper contrastive signal, and a better model — so batch size is a quality knob, not just a throughput knob.
3. A **hard negative** is a document that is *plausible-but-wrong* — it shares vocabulary/topic with the correct doc and often out-ranks it under a generic embedder. Random in-batch negatives are almost always *easy* (obviously unrelated), so the model already scores them low and learns nothing. You get hard negatives by **mining**: run the query through your existing retriever, take high-ranked-but-incorrect results, and use those as negatives.
4. **Some mined "negatives" are actually relevant documents you just didn't label, so you end up training the model to rank a good answer as bad.** Defenses: (a) a score-margin/threshold filter that rejects mined docs scoring too close to the positive; (b) skip the very top ranks and mine deeper (e.g. ranks 10–50); (c) run a cross-encoder reranker to drop mined negatives it scores as relevant; (d) eyeball a sample before training. (Any two.)
5. **Nothing meaningful — you compared new-model query vectors against old-model document vectors,** which is a version mismatch that makes the similarities incoherent. You must re-embed the entire corpus with the fine-tuned model and rebuild the index before recall numbers mean anything.
6. Report the **absolute delta and the relative delta** (e.g. 0.64 → 0.84 = +0.20 absolute, +31% relative), with the sample size `n`. One number alone misleads: a big relative gain can hide a tiny absolute base (0.02 → 0.04 is "+100%" but useless), and an absolute number without the baseline hides how much headroom remained.
