# Phase 0 — Lectures Index

Deep, textbook-style lectures backing the [Phase 0 spine](../00-foundations.md). The spine's **Theory** sections are short recaps; these are the full treatment — mechanism from first principles, worked numeric examples, production consequences, misconceptions, cheat sheets, and a self-check with answers.

## How to use

- Read the week's lectures **before** the matching [lab guide](../labs/). The spine's Theory recap tells you the headline; the lecture gives you the depth; the lab makes you build it.
- Each lecture is ~20 min. Budget ~2 hrs of lecture reading per week (the spine's "Theory hrs"), then spend the rest of the week's hours in the lab.
- Don't rabbit-hole. Read once for the mechanism + the cheat sheet, do the self-check, move to the lab. You'll cement the rest by building.

## Week 1 — Environment, data, and just-enough ML

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 1 | [Reproducible AI Environments & the uv Toolchain](01-reproducible-envs-uv.md) | uv, lockfiles, secrets |
| 2 | [Notebooks Without Hidden State](02-notebooks-without-hidden-state.md) | headless notebooks, CI |
| 3 | [Vectorized Thinking: NumPy Arrays as the Tensor Mental Model](03-numpy-tensors-vectorized-thinking.md) | NumPy, broadcasting, axes |
| 4 | [Just-Enough Classical ML for LLM Engineers](04-just-enough-classical-ml.md) | train/val/test, overfitting, baselines |
| 5 | [Data Leakage: The #1 Real-World Killer](05-data-leakage.md) | leakage mechanisms & detection |
| 6 | [Metrics That Match Business Cost](06-metrics-that-match-cost.md) | precision/recall/F1, recall@k/MRR/nDCG |

## Week 2 — How LLMs work for engineers

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 7 | [Next-Token Prediction: The One Thing LLMs Do](07-next-token-prediction.md) | autoregressive generation, hallucination |
| 8 | [Attention & the Transformer, for Engineers](08-attention-transformer-intuition.md) | attention/QKV, O(n²), lost-in-the-middle |
| 9 | [Tokenization: How Models See Text](09-tokenization.md) | BPE, tokens/cost, non-English penalty |
| 10 | [Embeddings & Vector Similarity](10-embeddings-and-similarity.md) | cosine/dot, dimensions, Matryoshka |
| 11 | [Context Windows & the KV Cache](11-context-window-kv-cache.md) | prefill vs decode, KV cache VRAM |
| 12 | [Sampling Parameters: Controlling Generation](12-sampling-parameters.md) | temperature/top-p/top-k, determinism |

## Week 3 — Model & deployment landscape

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 13 | [Base vs Instruct vs Reasoning Models](13-model-types-base-instruct-reasoning.md) | model types, test-time compute |
| 14 | [Open vs Closed Models & Gateways](14-open-vs-closed-and-gateways.md) | decision axes, LiteLLM/OpenRouter |
| 15 | [Quantization & VRAM Math](15-quantization-and-vram-math.md) | GGUF/AWQ/GPTQ, VRAM formula |
| 16 | [Chat Templates, Structured Output & Tool Calling (Intro)](16-chat-templates-structured-output-tools.md) | chat templates, JSON mode, tool loop |

## Labs (step-by-step guides)

- [Week 1 Lab — Reproducible Env + Classical ML Basics](../labs/week-1-repro-env-and-ml-basics.md)
- [Week 2 Lab — Tokens, Cost, Embeddings & Sampling](../labs/week-2-tokens-embeddings-sampling.md)
- [Week 3 Lab — Build the Model Economics CLI (Milestone)](../labs/week-3-model-economics-cli.md)
