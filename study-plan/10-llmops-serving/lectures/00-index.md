# Phase 10 — Lectures Index

Deep, textbook-style lectures backing the [Phase 10 spine](../10-llmops-serving.md). The spine's **Theory** sections are short recaps; these are the full treatment — mechanism from first principles, worked numeric examples (VRAM math, break-even), production consequences, misconceptions, and cheat sheets.

## How to use

- Read the week's lectures **before** the matching [lab guide](../labs/). The spine's Theory recap tells you the headline; the lecture gives you the depth; the lab makes you build it.
- Budget the spine's weekly "Theory hrs" for lecture reading, then spend the rest of the week's hours in the lab.
- Don't rabbit-hole. Read for the mechanism + the cheat sheet, then go build — you'll cement the rest by serving, load-testing, and shipping.

## Week 1 — Self-hosted inference with vLLM

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 1 | [The inference engine landscape: why vLLM (and when it isn't)](01-inference-engine-landscape.md) | engine landscape, vLLM/TGI/SGLang/TensorRT-LLM |
| 2 | [Continuous batching, PagedAttention, prefix caching, and chunked prefill](02-continuous-batching-paged-attention.md) | why throughput ≠ speed |
| 3 | [OpenAI-compatible serving and the three tuning knobs](03-openai-compatible-serving-and-tuning.md) | `/v1` endpoints, max-num-seqs / gpu-mem-util / max-model-len |
| 4 | [Parallelism for serving: tensor vs pipeline vs data](04-parallelism-tensor-pipeline-data.md) | tensor/pipeline/data parallelism, replicas |
| 5 | [Multi-LoRA serving: many fine-tunes, one base model in VRAM](05-multi-lora-serving.md) | multi-LoRA, per-tenant economics |

## Week 2 — Hardware, economics & FinOps

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 6 | [The VRAM equation: weights, KV cache, activations, and max concurrency](06-vram-equation.md) | VRAM math, KV cache caps concurrency |
| 7 | [Memory-bound decode and reading the GPU: nvidia-smi and nvitop](07-memory-bound-decode-gpu-literacy.md) | memory-bandwidth-bound decode, GPU literacy |
| 8 | [Load testing and the latency metrics that matter: TTFT, TPOT, throughput, percentiles](08-load-testing-latency-metrics.md) | k6/locust, TTFT/TPOT, p50/p95/p99 |
| 9 | [Buy vs rent vs call: the deploy decision and break-even math](09-deploy-decision-breakeven.md) | closed API / serverless / dedicated / on-prem, crossover |
| 10 | [Serverless economics and LLM FinOps: cold starts, cascades, attribution, budgets](10-serverless-economics-finops.md) | scale-to-zero, batch discounts, cascades, budgets |

## Week 3 — Performance, edge & the release pipeline

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 11 | [Test-time compute and reasoning budgets as a cost/latency decision](11-test-time-compute-reasoning.md) | reasoning effort, gate per-route |
| 12 | [Long context vs RAG vs CAG: context strategy as an economics decision](12-long-context-rag-cag.md) | long-context / RAG / CAG tradeoffs |
| 13 | [Latency SLOs and speculative decoding](13-latency-slos-speculative-decoding.md) | TTFT/TPOT SLOs, speculative decoding |
| 14 | [On-device, edge, and in-browser inference: when the server loses](14-edge-ondevice-browser-inference.md) | Ollama/GGUF, MLX, WebLLM/WebGPU |
| 15 | [Prompts and models as code: eval gates and gating under non-determinism](15-prompts-models-as-code-eval-gates.md) | prompts-as-code, eval gate, thresholds |
| 16 | [Progressive delivery for LLMs: feature flags, shadow, canary, and instant rollback](16-progressive-delivery-shadow-canary-rollback.md) | flags, shadow, canary, rollback |

## Labs (step-by-step guides)

- [Week 1 Lab — Stand up and tune a vLLM OpenAI-compatible endpoint with multi-LoRA](../labs/week-1-vllm-serving.md)
- [Week 2 Lab — VRAM math, load testing, and the buy-vs-rent break-even chart](../labs/week-2-economics-loadtest-breakeven.md)
- [Week 3 Lab — Edge inference plus a governed release pipeline — and the phase milestone](../labs/week-3-release-pipeline-and-milestone.md)
