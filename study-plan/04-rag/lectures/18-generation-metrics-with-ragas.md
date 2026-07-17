# Lecture 18: Generation Metrics with RAGAS

> You can now score *retrieval* from scratch — recall@k, MRR, nDCG against your golden chunk ids (Lecture 17). But those numbers stop at "did the right chunk get fetched?" They say nothing about the half your users actually read: the generated answer. Did the model hallucinate on top of good context? Did it dodge the question with a fluent non-answer? Did retrieval quietly stuff 8 useless chunks into the prompt and burn your token budget? This lecture teaches the four **RAGAS** metrics that answer those questions — faithfulness, context precision, context recall, and answer relevancy — with what each one *actually* computes, how to read them *together* (the combinations are where the diagnosis lives), and the uncomfortable truth underneath all of them: RAGAS is an **LLM-as-judge** system. It needs a judge model and an embedder, it costs real money and latency, and it drifts if you don't pin it. By the end you'll wire RAGAS over your *live* pipeline (not hand-fed contexts), pick a free-local or cheap-API judge, and know why you must calibrate the judge against ~20 human labels before you ever gate CI on it.

**Prerequisites:** The two RAG pipelines (Lecture 1), retrieval-only metrics (Lecture 17), NLI groundedness (Lecture 11 — RAGAS faithfulness is the productionized version of that), a golden set with `ground_truth` answers, basic probability and cost arithmetic. · **Reading time:** ~30 min · **Part of:** Retrieval-Augmented Generation, Week 4

## The core idea (plain language)

A RAG answer is a pipeline output, and it can be wrong for reasons that live at different stages. A single "accuracy" number smears all those reasons into one blur you can't act on. RAGAS gives you **four metrics that each isolate one axis**, and the engineering skill is reading them as a *panel*, not individually.

Two of the four score the **answer** (the generation half):
- **Faithfulness** — is every claim in the answer actually supported by the retrieved context? This is your **hallucination gauge**.
- **Answer relevancy** — is the answer actually *about* the question (not evasive, not padded)? This is your **on-topic gauge** — and, crucially, *not* a correctness gauge.

Two of the four score the **context** (the retrieval half, but measured *inside* RAGAS so it sits in the same report):
- **Context precision** — of the chunks you retrieved, how many were actually useful, weighted toward the top ranks? Low precision = noisy retrieval diluting the prompt.
- **Context recall** — does the retrieved context contain the information needed to produce the reference answer? This needs `ground_truth`, and it's your **retrieval-quality signal inside RAGAS**.

The whole reason you compute all four is that their **combinations** localize failure. The single most important pattern to burn in:

> **High answer relevancy + low faithfulness = confidently on-topic and wrong.** The model gave a fluent, perfectly on-question answer that the context does not support. Relevancy alone would have waved it through. Faithfulness is what catches it.

And the second: **low context recall means the retriever failed** — the fact isn't even in the window, so no amount of prompt tuning will fix the answer. **High context recall + low faithfulness means the generator failed** — the fact was right there and the model ignored or contradicted it. That two-cell distinction (Lecture 17's localization rule) is why faithfulness and context recall are the **two CI-gated metrics**: one guards the generator, one guards the retriever, and together they route every regression to the right half of the system.

## How it actually works (mechanism, from first principles)

Every RAGAS metric operates on some subset of four fields per case: `question`, `answer` (what your generator produced), `contexts` (the chunk *texts* your retriever returned), and `ground_truth` (the human reference answer). Under the hood each metric is one or more **LLM-as-judge** calls plus, for some, an embedding similarity. Let's take them one at a time with numbers.

### Faithfulness — the hallucination gauge

Mechanism (this is exactly the NLI claim-decomposition idea from Lecture 11, done by an LLM judge):
1. The judge **decomposes the answer into atomic claims** (short, self-contained factual statements).
2. For **each claim**, the judge asks: *is this claim entailed by the retrieved context?* (yes/no).
3. `faithfulness = supported_claims / total_claims`.

```
answer: "Connect timeout is 30s. Read timeout is 60s. Both support IPv6."
        │
        ├─ claim 1: "connect timeout is 30s"   → context says so     → ✅
        ├─ claim 2: "read timeout is 60s"       → context says so     → ✅
        └─ claim 3: "both support IPv6"          → context silent      → ❌
faithfulness = 2/3 = 0.67
```

A `0.67` here means one-third of the answer's factual load is **hallucinated relative to the context**. Note the phrase "relative to the context": faithfulness does *not* check truth in the world — if your retrieved chunk is itself wrong, a faithful answer will happily repeat it. Faithfulness = "did the model stay inside the evidence it was given," not "is the answer correct." (Correctness is a separate RAGAS metric, `answer_correctness`, which compares against `ground_truth`.)

### Context recall — the retrieval-quality signal (needs ground_truth)

Mechanism:
1. Take the **`ground_truth` answer** and decompose *it* into claims/sentences.
2. For **each ground-truth claim**, ask: *can this claim be attributed to the retrieved context?*
3. `context_recall = attributable_gt_claims / total_gt_claims`.

```
ground_truth: "The connect timeout is 30s and it is tunable via config."
        │
        ├─ gt-claim 1: "connect timeout is 30s"     → in context   → ✅
        └─ gt-claim 2: "it is tunable via config"    → NOT in context → ❌
context_recall = 1/2 = 0.5
```

This is the answer to "did retrieval fetch what was *needed*?" measured against a reference, not against what the model happened to say. It's the RAGAS mirror of your from-scratch recall@k (Lecture 17) — recall@k works on **chunk ids**, context_recall works on **the semantic content of the reference answer**. Because it needs `ground_truth`, a golden set of bare questions *cannot* give you context recall. No reference answers ⇒ no context recall ⇒ you've lost your primary retrieval signal inside RAGAS. This is why the golden set carries ground truths.

### Context precision — the noise gauge (rank-weighted)

Mechanism: for each retrieved chunk, the judge decides whether it was **useful** for answering (relevant vs. not). Then those relevance flags are combined with a **rank weighting** so that a relevant chunk at rank 1 counts more than a relevant chunk at rank 5 — the metric is an average-precision-style score, `Σ (precision@k × relevant_k) / total_relevant`.

Worked: you retrieved 5 chunks, relevance pattern `[relevant, junk, relevant, junk, junk]`.

```
rank 1: relevant   precision@1 = 1/1 = 1.00
rank 2: junk
rank 3: relevant   precision@3 = 2/3 = 0.67
rank 4: junk
rank 5: junk
context_precision = (1.00 + 0.67) / 2 relevant chunks = 0.835
```

Low context precision means you're feeding the generator noise. Two costs: (1) it dilutes the prompt so the model can get distracted ("lost in the middle," Lecture 10), and (2) every junk chunk is **tokens you paid for** on every single call. A precision of 0.4 on top-5 says roughly 3 of your 5 chunks were dead weight — tighten `k` or add a reranker (Lecture 6).

### Answer relevancy — the on-topic gauge (uses embeddings)

Mechanism (this one leans on the **embedder**, not just the judge LLM):
1. The judge reads your `answer` and **generates N questions that the answer would be a good answer to** (reverse-engineering the implied question).
2. Embed those generated questions and embed the **original** question.
3. `answer_relevancy = mean cosine similarity` between the original question and the generated ones.

If the answer is on-topic, the questions it "implies" look like your real question → high similarity. If the answer is evasive ("I'm not sure, please contact support") or padded with filler, the implied questions drift → low similarity.

```
question: "What is the default connect timeout?"
answer: "Timeouts are important for reliability and vary by system."   ← evasive
   generated implied questions:
     "Why do timeouts matter?"        sim 0.42
     "Do timeouts vary by system?"    sim 0.38
   answer_relevancy ≈ 0.40   ← low: the answer dodged
```

The critical property: **answer relevancy is blind to correctness**. A confidently wrong but perfectly on-topic answer scores *high* here. That's the trap the panel is built to expose.

### Reading the panel — the diagnostic table

| faithfulness | answer_relevancy | context_recall | what it means | fix |
|---|---|---|---|---|
| low | high | high | **confident hallucination** — on-topic, wrong, context ignored | prompt / model / "cite the context" instruction |
| high | low | high | grounded but **evasive/padded** answer | prompt for directness; check the generator |
| any | any | **low** | **retrieval miss** — needed fact not in window | chunking / embeddings / k / reranker |
| high | high | high but answer still wrong | golden-set/label problem or reasoning gap | audit the ground_truth |

This table *is* the reason to compute four numbers instead of one.

## Worked example

Two golden cases, top-5 retrieval, full panel.

**Case A — "What is the default `connect()` timeout?"**
- Retriever returns 5 chunks; 2 are the timeout config section, 3 are unrelated networking prose.
- Generator answers: *"The default connect timeout is 30 seconds, and it retries three times with backoff."*
- `ground_truth`: *"The default connect timeout is 30 seconds."*

Scoring:
- **Faithfulness:** 2 claims — "30 seconds" (in context ✅), "retries three times with backoff" (context mentions backoff but *not* "three times" ❌) → **0.5**. The model embellished.
- **Answer relevancy:** answer is squarely on-topic → **~0.93**.
- **Context recall:** the one gt-claim ("30 seconds") is in context → **1.0**.
- **Context precision:** relevance pattern `[rel, rel, junk, junk, junk]` → precision@1=1.0, precision@2=1.0 → (1.0+1.0)/2 = **1.0**.

Panel read: **context recall high, faithfulness low** → *generation problem*. Retrieval did its job; the model hallucinated "three times." Fix the prompt / model, not the retriever.

**Case B — "How do I rotate the signing key?"** (answer is *not* in the corpus)
- Retriever returns 5 loosely-related chunks about keys and config.
- Generator answers: *"Run `keytool -rotate` and restart the service."* (fabricated)
- `ground_truth`: *"[no-answer] The corpus does not document key rotation."*

Scoring:
- **Faithfulness:** the claim isn't supported by any chunk → **~0.0**.
- **Answer relevancy:** the answer is *about* key rotation → **~0.9** (high! it's on-topic).
- **Context recall:** the ground truth says "not documented"; nothing to attribute → RAGAS reads this as the retriever having nothing relevant → **low**.

Panel read: **relevancy 0.9 + faithfulness 0.0** = the textbook *confidently on-topic and wrong*. This is the exact failure users hit first, and it's why your golden set must contain `no_answer` cases — a happy-path-only eval reads 0.95 forever and never catches this. Faithfulness is the metric that flags it; answer relevancy actively *hides* it.

## How it shows up in production

- **RAGAS is not free — budget the judge calls.** Every metric is one or more LLM calls *per case*, and some (faithfulness, context recall) make one call *per claim*. Rule of thumb (approximate): **60 cases × 4 metrics × several claims each ≈ a few hundred to a thousand judge calls per run.** On an API judge at temperature 0 with a small model that's cents-to-a-few-dollars and a few minutes; on a local Ollama judge it's $0 but can be 10–30+ minutes. **Consequence:** don't run the full set on every commit — gate a stratified ~20-case subset on PRs, run the full set nightly.
- **The judge drifts if you don't pin it.** RAGAS scores are only comparable run-to-run if the judge is identical. If you point it at a floating model alias, a silent provider upgrade re-scores your whole history and your CI thresholds become meaningless overnight. **Pin the exact model snapshot, set temperature 0, and record both in the report** next to the git SHA and golden version.
- **Judge cost/latency scales with answer length.** A verbose generator produces more claims, which means more per-claim judge calls for faithfulness. A pipeline change that makes answers chattier will *also* make your eval slower and pricier — watch it.
- **Context precision is a token-budget alarm, not just a quality one.** Persistent low precision means you're paying to embed and prompt with junk on every query. In a high-QPS service that's a real line item — tightening `k` from 10 to 5 at equal recall can halve your context tokens.
- **Faithfulness and context_recall belong in CI; the other two are advisory (at first).** Gate the two that map cleanly to the two halves of the system. Answer relevancy and context precision are noisier and more judge-dependent — report them, watch trends, promote to gated only once you trust the judge. Set gate thresholds *slightly below* current measured scores (e.g. measured faithfulness 0.82 → gate at 0.75) so you catch regressions without flaking on judge noise.
- **You must calibrate the judge before trusting it.** RAGAS is a *system you ship in your test harness* — an unvalidated judge can systematically over- or under-score. Before you gate on it, hand-label ~20 cases (grounded / not, relevant / not) and check the judge's agreement (Cohen's κ). If the judge disagrees with humans, fix the judge (better model, clearer metric) before you fix the pipeline.

## Common misconceptions & failure modes

- **"High answer relevancy means the answer is good."** The single biggest misread. Answer relevancy measures *on-topic-ness*, not truth. A fluent, confident, completely fabricated answer scores high. Always read it *next to* faithfulness — relevancy high + faithfulness low is your hallucination signature.
- **"Faithfulness = correctness."** No. Faithfulness = "supported by the retrieved context." If the context itself is wrong, a faithful answer repeats the error and still scores 1.0. For world-correctness you need `answer_correctness` against `ground_truth`. Faithfulness is a *grounding* check, not a *truth* check.
- **Hand-feeding RAGAS the "right" contexts.** The deadliest process error. If you build the dataset by pasting in the ideal chunks instead of calling your **actual retriever**, faithfulness and precision look wonderful — and you've evaluated a system you don't ship. The `contexts` field must be *exactly what your live retriever returned* for that query. This is the whole point of wiring RAGAS over the live pipeline.
- **"No ground_truth is fine."** Then context_recall and answer_correctness silently can't be computed, and you've lost your primary retrieval signal inside RAGAS. Build the golden set *with* reference answers.
- **Trusting a tiny or un-pinned judge.** A too-small local judge mislabels claim entailment and gives noisy, drifting scores; an un-pinned API judge changes under you. Pin the snapshot, temperature 0, calibrate against humans.
- **Gating on a tiny sample.** With n=20 the run-to-run variance can exceed your threshold margin and flake CI red on unchanged code. Set thresholds with headroom, gate a stratified subset on PRs, run the full set nightly.
- **Ignoring per-stratum breakdown.** A great *mean* faithfulness can hide that every `no_answer` case hallucinates. Always break scores down by stratum (easy / hard / multi-hop / no-answer) — the mean is where failures go to hide.

## Rules of thumb / cheat sheet

- **Four metrics, two halves.** Faithfulness + answer_relevancy score the **answer**; context_precision + context_recall score the **context**. Read them as a panel.
- **Faithfulness = supported_claims / total_claims** — your hallucination gauge. Grounding, *not* correctness. **Gate it in CI.**
- **Context recall = attributable_gt_claims / total_gt_claims** — needs `ground_truth`; your retrieval-quality signal inside RAGAS. **Gate it in CI.**
- **Context precision** — rank-weighted usefulness of retrieved chunks; low = noisy retrieval burning tokens. Advisory at first.
- **Answer relevancy** — embedding-based on-topic score; **NOT correctness**. High relevancy + low faithfulness = confidently wrong. Advisory at first.
- **The tell:** relevancy high + faithfulness low ⇒ hallucination. Recall low ⇒ retrieval. Recall high + faithfulness low ⇒ generation.
- **Eval the LIVE pipeline.** `contexts` = what your real retriever returned; `answer` = what your real generator produced. Never hand-feed contexts.
- **Pin the judge.** Exact model snapshot + `temperature=0` + embedder, recorded in the report with git SHA and golden version.
- **Budget it.** ~60 cases × 4 metrics = hundreds of judge calls. PR = stratified ~20-case subset; nightly = full set.
- **Calibrate before you gate.** ~20 human labels, check agreement (κ). The judge is a system you must trust.
- **Set gate thresholds below current scores** (e.g. 0.82 measured → 0.75 gate) for headroom against judge noise.
- **Free-local judge:** Ollama `llama3.1:8b` + `nomic-embed-text` via `ChatOllama` / `OllamaEmbeddings` ($0, slower, noisier). **Cheap-API judge:** a small pinned model at temp 0 (cents/run, more stable). Pick one, pin it.

## Connect to the lab

Week 4 Step 3 is this lecture made real: `run_ragas.py` loops your golden rows, calls **your Week-3 retriever and generator per row**, builds a `datasets.Dataset` with `question / answer / contexts / ground_truth / retrieved_ids`, and calls `evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall], llm=judge_llm, embeddings=judge_emb)` with a **pinned** judge. Step 4's `localize.py` applies the panel table (context_recall low ⇒ retrieval; recall ok & faithfulness low ⇒ generation) and proves it on two seeded failures. Step 6 gates `faithfulness` and `context_recall` in CI via `thresholds.yaml`. The judge path is your choice: free-local Ollama or a cheap pinned-API model — record whichever in `report.md`.

## Going deeper (optional)

- **RAGAS documentation** (`docs.ragas.io`) — the "Metrics" section is the source of truth for exactly how faithfulness, context precision, context recall, and answer relevancy are computed (the definitions evolve across versions, so read the version you installed). Also read "Generate a Testset" and the guides on configuring the judge `llm` and `embeddings`.
- **Arize Phoenix docs** (`docs.arize.com/phoenix`) — for *eyeballing* the worst cases and clicking through traces; pairs well with RAGAS-for-numbers.
- **TruLens** (`trulens.org`) — the "RAG Triad" (context relevance / groundedness / answer relevance) framed as feedback functions, if you prefer that ergonomics. Pick RAGAS + one viewer; don't run all three.
- **LangChain `ChatOllama` / `OllamaEmbeddings` docs** — how to point RAGAS at a free local judge (`llama3.1:8b` + `nomic-embed-text`).
- Lecture 11 (NLI groundedness) — faithfulness is the LLM-as-judge, productionized version of the claim-decomposition + entailment machine you built by hand there.
- Search queries when you hit friction: "ragas faithfulness metric how computed", "ragas context recall ground truth", "ragas evaluate custom llm embeddings ollama", "llm as judge calibration cohen kappa human labels", "ragas answer relevancy vs answer correctness".

## Check yourself

1. A case scores answer_relevancy 0.95 and faithfulness 0.35. In plain terms, what is the model doing, and why would relevancy alone never catch it?
2. Which two of the four RAGAS metrics require `ground_truth`, and what specifically breaks in your eval if the golden set has only questions?
3. Faithfulness scores 1.0 but a human says the answer is factually wrong. Explain how both can be true at once, and name the metric that would catch the human's complaint.
4. Why is it a serious methodology error to build your RAGAS `contexts` field by hand-selecting the ideal chunks? What are you actually measuring if you do?
5. Your team wants to gate CI on RAGAS. List the four things you must do to the judge before the gate is trustworthy, and explain the cost math for why you'd gate a 20-case subset on PRs but run 60 cases nightly.
6. context_precision on top-10 is 0.35 with context_recall 0.95. What's the story, and what's the cheapest fix — and what second benefit does that fix give you besides quality?

### Answer key

1. The model is producing a **confidently on-topic but ungrounded/hallucinated** answer — it's squarely about the question (high relevancy) but its claims aren't supported by the retrieved context (low faithfulness). Answer relevancy only measures whether the answer *matches the question's topic* (via reverse-generated questions embedded and compared), so a fluent wrong answer that's on-topic scores high; correctness/grounding is invisible to it. Faithfulness is what exposes it.
2. **Context recall** and **answer correctness** require `ground_truth`. Without reference answers, context_recall can't be computed — and that's your *primary retrieval-quality signal inside RAGAS* — so you lose the ability to tell "the fact wasn't retrieved" from "it was retrieved but the model ignored it." You'd be left with faithfulness, answer_relevancy, and context_precision, none of which use a reference.
3. Faithfulness only checks that each answer claim is **entailed by the retrieved context**, not that the context is true. If retrieval fetched a chunk that itself contains a wrong fact, an answer that faithfully repeats it scores 1.0 while being factually wrong. **`answer_correctness`** (compares the answer to `ground_truth`) is the metric that catches it; the root cause is a corpus/retrieval problem, not a generation one.
4. Because your `contexts` must be exactly what the **live retriever returned** for that query. If you hand-pick ideal chunks, faithfulness and context precision look great, but you've evaluated a *different system than the one you ship* — one with a perfect retriever you don't have. You're measuring the generator's behavior on inputs it will never actually receive in production, which tells you nothing about real-world quality and hides every retrieval failure.
5. (a) **Pin the exact model snapshot** (no floating aliases); (b) **set temperature 0** for determinism; (c) **record the judge + embedder + git SHA + golden version** in the report; (d) **calibrate against ~20 human labels** and check agreement (κ) so you trust the scores before gating. Cost math: each metric is 1+ judge calls per case, some per-claim, so 60 cases × 4 metrics ≈ hundreds of calls — minutes and cents (API) or tens of minutes ($0 but slow, local) per run. That's fine nightly, but too slow/costly on every PR, so you gate a **stratified 20-case subset** for fast per-PR feedback and run the **full 60** nightly to catch what the subset misses.
6. Low precision + high recall means the retriever **is** fetching the needed facts but is also dragging in a lot of junk — roughly two-thirds of your top-10 chunks are dead weight. Cheapest fix: **reduce k** (e.g. 10→5) and/or add a **cross-encoder reranker** (Lecture 6) so the useful chunks dominate the top ranks; recall stays high because the needed chunks are still there. Second benefit: you feed **far fewer context tokens per call**, cutting cost/latency on every query — a precision fix is also a token-budget fix.
