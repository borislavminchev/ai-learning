# Lecture 2: Choosing an Embedding Model on Evidence

> Every RAG system, semantic search box, and agent memory rests on a single upstream decision: which embedding model turns your text into vectors. Get it wrong and no amount of clever indexing, reranking, or prompt engineering downstream will save you — you are searching in a space where "close" doesn't mean "relevant" for *your* data. The trap is that this decision feels easy: open the MTEB leaderboard, sort by score, take the top row. That's a leaderboard reflex, and it ships mediocre retrieval with confidence. This lecture turns model selection into an engineering decision. After it you can read the MTEB leaderboard for what it actually measures, explain the three specific ways it misleads, build a fair benchmark harness on a held-out slice of your own data, reason about every decision axis (retrieval score, sequence length, dimension, multilingual coverage, license, open-vs-API) with its concrete engineering cost, and defend a model choice to a skeptical staff engineer with recall@k numbers instead of a screenshot.

**Prerequisites:** Lecture 1 (embeddings & similarity: cosine/dot/L2, normalization, prefixes, MRL), comfortable Python + NumPy, the recall@k / MRR metrics from the Week 1 lab · **Reading time:** ~26 min · **Part of:** Phase 3 Week 1

---

## The core idea (plain language)

There is no "best embedding model." There is only the best model *for your corpus, your queries, your latency budget, and your wallet.* Those four things are not on any leaderboard, because the leaderboard doesn't have your data.

The mental shift this lecture demands: stop treating model selection as a lookup ("what's #1?") and start treating it as a **measurement** ("which of these three, on 80 of my own queries against my own documents, retrieves the right passages most often within my constraints?"). That measurement takes an afternoon. It is the single highest-leverage afternoon in a RAG project, and it is routine for teams to discover that the model ranked #14 on a public benchmark beats the model ranked #1 on their support tickets.

Everything else here — MTEB, the decision axes, the model catalog, the bench harness — is in service of that one move: **validate on a held-out slice of YOUR data with recall@k against a flat ground truth, then decide.**

---

## How it actually works (mechanism, from first principles)

### What MTEB is, and how to actually read it

**MTEB** — the Massive Text Embedding Benchmark — is a collection of dozens of datasets grouped into *task types*: retrieval, classification, clustering, reranking, semantic textual similarity (STS), pair classification, summarization, and more. A model is run across all of them and gets a per-task score plus a single **aggregate average**. The leaderboard (a Hugging Face Space) sorts, by default, on that aggregate.

The first skill is to **not read the aggregate**. For RAG and semantic search you care about exactly one column: **Retrieval** (measured mostly as nDCG@10 over BEIR-style datasets). A model can post a gorgeous aggregate by crushing classification and STS — tasks that have nothing to do with finding the right passage for a query — while sitting mid-pack on retrieval. The aggregate *averages retrieval weakness away*.

```
        Aggregate (default sort)          What you actually want
   ┌───────────────────────────┐      ┌───────────────────────────┐
   │ Model  Avg  Class  STS  R  │      │ Model      Retrieval(nDCG) │
   │ A     71.2  78    82  61  │ ───► │ B          66             │
   │ B     69.8  64    68  66  │      │ A          61             │
   └───────────────────────────┘      └───────────────────────────┘
     A wins the sort                     B wins the job you care about
   (illustrative numbers, not real)
```

So: open MTEB, **re-sort by the Retrieval task**, and filter to the language(s) and size class you can actually run. Now you have a *shortlist*, not an answer.

### Why leaderboards mislead — three specific mechanisms

**1. Benchmark contamination.** Popular benchmarks are public. Their test sets leak into the enormous web-scraped training corpora of new models, and some models are explicitly tuned toward known benchmark tasks. When a model has (directly or indirectly) seen the test questions, its benchmark score measures memorization, not generalization. The score is real; it just doesn't predict performance on data the model has never seen — which is precisely your data.

**2. Domain mismatch.** MTEB's retrieval datasets are academic and general: Wikipedia, scientific abstracts, web Q&A, forum posts. Recall from Lecture 1 that a model only "knows" the similarity notion it was trained on. If your corpus is insurance policies, Kubernetes error logs, radiology reports, or Bulgarian legal contracts, the benchmark's genre of "query→passage" similarity may barely overlap with yours. A model that tops BEIR can be mediocre on your jargon-dense, entity-heavy text.

**3. Averaged tasks hide retrieval weakness.** Covered above, but it bears repeating as a distinct failure: the *default sort* itself is the trap. Engineers who sort by the headline number and take row 1 are, statistically, often not even choosing the best *retrieval* model in the list.

The conclusion is not "leaderboards are useless." They are an excellent **shortlisting tool** — they narrow thousands of models to five candidates. They are a terrible **decision tool**. The decision comes from your own eval.

### The held-out eval: recall@k against a flat ground truth

Here is the whole method in one breath: assemble ~50–100 real queries, each labeled with the document ID(s) that *should* come back (the **qrels**, or ground truth). Embed your corpus and your queries with a candidate model. Do an **exact, brute-force** nearest-neighbor search (flat cosine over all vectors — no ANN index yet) to get the true top-k. Compute **recall@k**: of the queries where a relevant doc exists, what fraction had it in the top-k?

```
recall@k = (# queries whose relevant doc appears in top-k) / (# queries)
```

Why *flat* ground truth? Because an approximate index (HNSW, IVF) has its own recall loss, and if you benchmark models through an approximate index you're measuring model-quality *and* index-error tangled together. Flat search is O(N) per query and slow at scale, but on a 2k–50k eval corpus it's instant and it's *exact* — the model's true retrieval quality with nothing in the way. (You'll measure ANN recall *against this flat baseline* in Week 2; different question, same principle.)

"Held-out" means these queries and their answers must not be something you tuned against or that the model plausibly trained on — they represent traffic you'll actually see. Even a scrappy hand-labeled set of 50 real queries beats any leaderboard for your decision.

### The decision axes and what each one costs you in production

Retrieval score is necessary, not sufficient. Six axes, each with an engineering price tag:

- **Retrieval score (on your eval).** The recall@k / MRR numbers above. This is the *quality* axis. Everything below is the *cost* of buying that quality.

- **Max sequence length.** How many tokens the model reads before it **silently truncates**. This is the assassin. Many strong models cap at **512 tokens** (~350–400 English words). If your chunks are 800 tokens, the model embeds the first 512 and *throws the tail away with no error*. The vector then represents half a chunk; the content in the dropped tail is unretrievable. Long-context models (bge-m3, gte-Qwen2, nomic) handle 8k+. **Rule: always log the token count of every chunk you embed and compare it to the model's real limit.** A 512-cap model on 800-token chunks is a recall bug that looks like a bad model.

- **Embedding dimension.** Storage and RAM scale **linearly** with dims (`docs × dims × 4 bytes` for float32). 1M docs at 1024 dims ≈ 4.1 GB; at 384 dims ≈ 1.5 GB. In most vector indexes that's resident RAM, and RAM is the dominant cost at scale. Higher dims *often* buy quality — but a good 384-dim model routinely beats a mediocre 1536-dim one, and Matryoshka (MRL) models let you truncate to trade quality for space on purpose.

- **Multilingual coverage.** English-only models (`bge-small-en`, many `all-*` models) are smaller and often sharper *on English*, but score near-random on other languages. If a single German or Bulgarian query must work, you need a multilingual model (bge-m3, multilingual-e5) — and you should put non-English queries in your eval set, or you won't notice the gap until a user does.

- **License.** Apache-2.0 / MIT models (much of the BGE, E5, GTE, nomic families — *check each card*) are safe for commercial closed-source use. Some strong open weights carry non-commercial or bespoke licenses. A model you can't legally ship is not a candidate, regardless of score. Verify the license on the model card before you invest a benchmarking afternoon in it.

- **Open weights vs API.** The big architectural fork:
  - **Open weights** (run it yourself): no per-token cost, data never leaves your infrastructure (**data residency** — often mandatory for legal/health/finance), predictable latency, full control. Price: you run GPU/CPU infra, batch jobs, and ops. You *can* embed on CPU for small corpora.
  - **API** (OpenAI, Cohere, Voyage): zero infra, top-tier quality, elastic scale. Price: per-token billing on your entire corpus + every query forever, network latency per call, rate limits, and your text goes to a third party (a hard blocker under some compliance regimes). Bulk-embedding 5M docs bills real money; gate every API call behind a cache.

---

## Worked example

You're picking a model for a legal-document search over 40k contract clauses. Chunks run ~700 tokens (clauses are long). Three shortlisted candidates, all with plausible MTEB retrieval scores. You run your 80-query held-out eval (flat cosine). Numbers below are **illustrative** to show the reasoning, not measured results:

```
model            maxlen  dim   recall@10  MRR@10  throughput   note
A (512-cap)       512     1024    0.71      0.58    900 txt/s   truncates 700-tok chunks!
B (8k, base)      8192    768     0.86      0.74    350 txt/s   reads full clause
C (8k, large)     8192   1024     0.88      0.77    120 txt/s   marginally better, 3x slower
```

The leaderboard reflex picks A — suppose it had the highest *aggregate* MTEB. But A caps at 512 tokens, so on your 700-token clauses it embeds only the first ~73% and drops the rest. Its recall craters not because it's a weak model but because **the harness fed it text it couldn't read.** You'd only catch this if you logged token counts (you did) and put full-length clauses in your eval (you did).

B reads the whole clause and jumps to 0.86 recall@10 at a healthy 350 texts/sec. C is a hair better (0.88) but 3× slower to encode and 33% more storage (1024 vs 768 dims → ~4.1 GB vs ~3.1 GB per million).

**Decision:** ship **B**. The 0.02 recall gain from C doesn't justify tripling encode time and inflating RAM for a 40k (and growing) corpus. And A — despite possibly topping the public leaderboard — was never viable once you accounted for sequence length. The whole call took one afternoon and rests on numbers from *your* data.

Now flip one variable: if chunks were **short** (~150 tokens, e.g. FAQ snippets), A's 512 cap never bites, and A might well win on quality-per-dollar. **The right model is a function of your chunk length**, which is why you decide with a harness, not a ranking.

---

## How it shows up in production

**The 512-token silent truncation is the most expensive "invisible" bug in RAG.** No exception, no warning — the tail of every long chunk simply vanishes from its vector. Symptom: retrieval that's fine for questions answered in a chunk's first paragraph and mysteriously blind to anything in the second half. You debug it by logging, at ingest, `len(tokenizer.encode(chunk))` and comparing to the model's *actual* limit (read the config, don't trust the marketing "long context" claim). If chunks exceed the limit, either shrink chunks or pick a longer-context model.

**Dimension is a RAM invoice you pay monthly.** At 1M docs, the difference between 384 and 1536 dims is ~1.5 GB vs ~6.1 GB of resident vectors — which can be the difference between one node and three. Teams pick a high-dim API model for a 5-point eval bump, then meet the HNSW RAM bill in Week 2 and wish they hadn't. Decide dimension against *both* the quality axis and the storage axis.

**API cost is dominated by the corpus, not queries — until the corpus is static and queries aren't.** Bulk-embedding a large corpus is a one-time (per model version) hit; but every user query is also a billed API call forever, and every model *upgrade* re-embeds the entire corpus from scratch (query and document vectors must come from the same model — Lecture 1's same-model rule). Model this before committing to an API: `corpus_tokens × price` once + `query_tokens × price × QPS × time`. Cache query embeddings for repeated queries.

**Data residency can eliminate the whole API column in one sentence.** "Customer data may not leave our VPC" (common in health, finance, government, EU) means API embedding models are simply off the table, no matter their score. Know this constraint *before* you benchmark, or you'll fall in love with a model you can't legally use.

**Benchmarking without a cache silently costs you the benchmark's own budget.** Your bench harness re-embeds the corpus for each candidate model; without a `model:version:sha256(text)` cache you re-run expensive encodes on every tweak, and for API models you re-pay every time. Cache-gate the harness from day one.

---

## Common misconceptions & failure modes

- **"The #1 model on MTEB is the best choice."** It's the best on the *default aggregate* of *public academic* tasks. Re-sort by retrieval, then decide on your own eval. The gap between leaderboard rank and your-data rank is routinely large.
- **"A high retrieval score is enough."** Score is one of six axes. A 5-point edge is worthless if the model truncates your chunks, can't be licensed for your product, doubles your RAM, or can't run in your VPC.
- **"Long context" on the model card = it reads my long chunks.** Verify the *tokenizer's* real max in the config and test with a chunk at your actual length. Some "long" claims refer to the base LM, not the embedding head's trained limit.
- **"I'll just compare cosine scores across models to pick a winner."** Cosine magnitudes are not comparable across models — each lives in its own space. Only *ranking-based eval metrics* (recall@k, MRR, nDCG) are comparable across models. (Lecture 1.)
- **"Bigger dimension is always better retrieval."** Often marginal, sometimes worse (overfitting noise on small corpora), always more storage/compute. Test both; a sharp 384-dim model can beat a bloated 1536-dim one.
- **"I benchmarked throughput and model X is slow."** Did you length-bucket? Un-bucketed batches pad to the longest outlier and waste compute — you may be measuring your padding, not the model. (Week 1 lab.)
- **"I'll validate later."** Later is after you've built ingestion, indexing, and a service on top of the wrong model — and re-embedding is a full corpus re-run. The afternoon of eval is cheapest *before* you commit.

---

## Rules of thumb / cheat sheet

- **Shortlist from MTEB (re-sorted by Retrieval, filtered to your language + size), decide on your own eval.** Never ship on aggregate rank.
- **Eval recipe:** ~50–100 real query→relevant-id pairs, flat cosine (exact) search, report **recall@{1,5,10}, MRR@10**. Held-out, realistic, ideally not benchmark-derived.
- **Always log token counts at ingest** and compare to the model's *real* max_seq_length. Assume silent truncation until you've proven otherwise. Many top models cap at **512**.
- **Storage math:** `docs × dims × 4 bytes`. 1M×384 ≈ 1.5 GB; 1M×768 ≈ 3.1 GB; 1M×1024 ≈ 4.1 GB; 1M×1536 ≈ 6.1 GB. This is your index RAM.
- **Opinionated default:** prototype with **`BAAI/bge-small-en-v1.5`** (384-dim, fast, CPU-friendly), then benchmark **`bge-base`** and **`intfloat/e5-large`** (or `bge-m3` if multilingual/long-context) before committing.
- **Multilingual? Put non-English queries in the eval set** or you won't see the gap. Reach for `bge-m3` / `multilingual-e5-large`.
- **Check the license before you benchmark.** A model you can't ship isn't a candidate.
- **Data residency constraint → API models are out.** Settle this before evaluating.
- **Cache-gate the harness** on `model:version:sha256(normalized_text)` — same corpus/queries/qrels across all candidates, so the only variable is the model.
- **Decision tree (approximate):** long docs (>512 tok) → prioritize 8k context over a small MTEB edge · multilingual → multilingual model, non-negotiable · strict data residency → open weights · tiny team, small corpus, quality-first → API model · CPU-only / air-gapped → nomic-embed-text or bge-small.

### The 2025-26 workhorse catalog (know these by name and character)

**Open weights:**
- **`BAAI/bge-m3`** — multilingual, **long context (8k)**, and uniquely does dense + sparse + multi-vector (ColBERT-style) from one model. The Swiss-army knife when you need languages *and* long chunks.
- **`intfloat/multilingual-e5-large`** — strong multilingual baseline, prefix-based (`query:` / `passage:` — Lecture 1). Reliable, widely deployed, ~1024-dim.
- **`Alibaba-NLP/gte-Qwen2` family** — among the strongest open retrieval models; larger and heavier (Qwen2 backbone), long-context. Reach for it when quality is paramount and you can pay the compute.
- **`nomic-embed-text` (v1.5)** — **CPU/Ollama-friendly**, long context, **Matryoshka (MRL)** so you can truncate dims. The pragmatic pick for local, air-gapped, or budget deployments.
- **`BAAI/bge-small-en-v1.5`** — the prototyping default: 384-dim, fast, runs on CPU. Start here, then move up.

**API options:**
- **OpenAI `text-embedding-3-small` / `-large`** — cheap and strong; MRL via the `dimensions` parameter. `-small` is a very good cost/quality default when API is allowed.
- **Cohere `embed-v3`** — strong multilingual, has a search-document vs search-query input-type distinction (its version of prefixes).
- **Voyage `voyage-3`** — competitive quality, popular for retrieval; domain-specific variants exist.

---

## Connect to the lab

This lecture is the "why" behind **Week 1 Lab steps 5–6**: `bench_models.py` embeds the *same* corpus + queries for `bge-small-en-v1.5`, `e5-small-v2`, and `all-MiniLM-L6-v2`, runs **flat cosine search as ground truth**, and prints a markdown table of recall@{1,5,10}, MRR@10, dim, and throughput — the fair harness this lecture argues for. Step 6's prefix ablation makes the "silent recall killer" concrete by measuring the exact points E5 loses without its prefixes. As you build it: cache-gate on `model:version:sha256(text)` (Lab step 3 / DoD), and log token counts so you *see* any 512-cap truncation rather than being bitten by it. The Definition of Done — "state which model you'd commit to and why" — is exactly this lecture's decision, made on your numbers.

---

## Going deeper (optional)

- **MTEB leaderboard** — Hugging Face Space (`huggingface.co/spaces/mteb/leaderboard`). Re-sort by the Retrieval task. Read the **MTEB paper** for what each task measures — search: *"MTEB Massive Text Embedding Benchmark paper"*.
- **BEIR** — the retrieval benchmark suite MTEB's retrieval tasks draw on, and where you'll get `scifact`/`nfcorpus` qrels for the lab. Search: *"BEIR heterogeneous benchmark information retrieval"*.
- **Model cards on Hugging Face** — the authoritative source for max_seq_length, prefixes, dimension, license: `BAAI/bge-m3`, `BAAI/bge-small-en-v1.5`, `intfloat/multilingual-e5-large`, `Alibaba-NLP/gte-Qwen2-*`, `nomic-ai/nomic-embed-text-v1.5`. Read them; don't trust summaries.
- **sentence-transformers documentation** (`sbert.net`) — *Computing Embeddings*, *Semantic Search*, and the *Evaluation* pages for building a fair harness.
- **API embedding docs** — OpenAI Embeddings guide (`platform.openai.com/docs`), Cohere Embed docs, Voyage AI docs — for pricing, input-type/prefix conventions, and MRL/`dimensions` parameters.
- **Nils Reimers** talks on embeddings and evaluation — engineer-level intuition on why you must eval on your own data. Search: *"Nils Reimers embeddings evaluation talk"*.
- **On benchmark contamination** — search: *"embedding benchmark contamination train test leakage"* for the current discussion.

---

## Check yourself

1. You open MTEB, sort by the default column, and take row 1 for your RAG retrieval system. Name the two distinct reasons this is likely the wrong move even before you consider your own domain.
2. A candidate model has the best retrieval score in your shortlist but `max_seq_length = 512`. Your chunks average 800 tokens. What exactly happens at embed time, what symptom appears in retrieval, and how do you detect it?
3. Why must your model-selection eval use a *flat* (brute-force) search rather than an HNSW index, even though HNSW is what you'll ship?
4. You have a strict "no customer data leaves our VPC" policy and a 3M-document multilingual corpus with ~1200-token chunks. Walk through which axes eliminate candidates and name a viable model.
5. Two candidates score recall@10 of 0.87 (768-dim) and 0.88 (1536-dim) on your eval. State the storage difference per million docs and argue which you'd ship.
6. Your bench harness reports model X is 3× slower than model Y in texts/sec, so you reject X. What harness bug should you rule out before trusting that number?

### Answer key

1. (a) The **default sort is the aggregate average**, which blends in classification/STS/clustering tasks irrelevant to retrieval and can rank a mediocre-retrieval model #1; you should re-sort by the Retrieval task. (b) **Benchmark contamination** — public test sets leak into training data and some models are tuned toward benchmark tasks, so the score partly measures memorization, not generalization to unseen (your) data.
2. The model reads the first 512 tokens and **silently discards the remaining ~288** — no error. The resulting vector represents only the chunk's opening, so any content in the tail becomes unretrievable; symptom is retrieval that works for early-paragraph facts and is inexplicably blind to later ones. Detect it by logging `len(tokenizer.encode(chunk))` at ingest and comparing to the model's real max_seq_length.
3. A flat search is **exact** (100% recall by construction), so it isolates the *model's* retrieval quality. An HNSW index has its own approximate-recall loss; benchmarking through it entangles model error with index error, so you couldn't tell which candidate model is actually better. (You measure ANN recall *against* the flat baseline separately, in Week 2.)
4. **Data residency** eliminates all API models (OpenAI/Cohere/Voyage) — text can't leave the VPC → open weights only. **Multilingual** eliminates English-only models (bge-small-en, all-MiniLM). **~1200-token chunks** eliminate 512-cap models → need 8k context. A model satisfying all three: **`BAAI/bge-m3`** (open, multilingual, 8k context); `multilingual-e5-large` also works if its context limit fits your chunks (verify on the card).
5. 768-dim: 1M × 768 × 4 ≈ **3.1 GB**; 1536-dim: 1M × 1536 × 4 ≈ **6.1 GB** — the high-dim model roughly **doubles** resident RAM per million docs. For a 0.01 recall gain that's a bad trade at scale; ship the **768-dim** model unless that single recall point is business-critical and RAM is cheap. Decide dimension against the storage axis, not score alone.
6. **Missing length-bucketing.** If batches aren't sorted by token length, they pad to the longest outlier in each batch and waste compute on padding — you may be measuring padding overhead, not the model. Re-run with length-bucketed batches (and warm up first) before concluding X is slower.
