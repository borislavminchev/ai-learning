# Phase 10 — LLMOps: Serving, Optimization & Deployment

**Phase goal:** Stop treating models as somebody else's HTTP endpoint. In three weeks you will self-host an open model with a production-grade inference engine, reason quantitatively about GPU/latency/cost economics (VRAM math, TTFT vs TPOT, cost/1M tokens, break-even vs a hosted API), and ship model/prompt changes through a real release pipeline with shadow traffic, eval gates, and instant rollback. You finish able to answer "buy vs rent vs call the API?" with a chart, not a vibe.

Prev: [09-architecture-system-design.md](../09-architecture-system-design/09-architecture-system-design.md) · Next: [11-safety-security.md](../11-safety-security/11-safety-security.md)

## Prerequisites
- Phases 1–9: comfortable with Python, Docker, `uv`/`pip`, an LLM gateway (LiteLLM), evals (promptfoo/your own harness), OpenTelemetry/Langfuse tracing, and OpenAI-compatible chat APIs.
- You can read a `docker run` line and a `curl` against `/v1/chat/completions` without flinching.
- A Hugging Face account + token (for gated models like Llama; use ungated Qwen/Mistral if you don't want to request access).
- Access to **one** GPU for at least a few hours: a rented cloud GPU (RunPod/Modal/Vast.ai, ~$0.30–1.20/hr for an A10/L4/A100-slice) OR a free Colab/Kaggle T4 session. A CPU-only laptop is enough for ~60% of the labs via Ollama/llama.cpp; the vLLM load-test labs need the GPU.

## Time budget
3 weeks × ~10–15 hrs/week (~36 hrs total). Each week lists an hours breakdown. Weighted ~35% theory / 65% hands-on.

## How to use this file
Do the weeks in order — Week 1 stands up the server, Week 2 measures and prices it, Week 3 wraps it in a release process. Type the commands yourself; do not copy-paste blindly (paths, model IDs, and GPU flags must match your box). Treat every Definition of Done as a gate: if a checkbox fails, you are not done, even if "it basically works."

> **📚 Course materials.** This file is the **spine** — a weekly plan and a *recap*. Deep material lives alongside it:
> - **Lectures**: [`lectures/`](lectures/00-index.md) — read for the *why*/mechanism.
> - **Lab guides**: `labs/` — follow to *build*.
> Each week: read the linked lectures first (the Theory here is their recap), then work the linked lab guide.

---

## Week 1 — Self-hosted inference with vLLM: from `docker run` to a tuned OpenAI-compatible endpoint

**Hours:** ~12 (Theory ~4, Lab ~8)

### Objectives
By end of week you can:
1. Explain *why* continuous batching + PagedAttention make vLLM 5–20× higher throughput than naive HF `generate()` in a loop, in two sentences an engineer believes.
2. Launch vLLM as an OpenAI-compatible server and hit it with the stock `openai` Python client and `curl` — no code changes vs a hosted API.
3. Tune `--max-num-seqs`, `--gpu-memory-utilization`, and `--max-model-len` and predict the direction each moves throughput and KV-cache OOM risk.
4. Diagnose and fix a chat-template mismatch and a KV-cache OOM from the server logs alone.
5. Serve multiple LoRA adapters on one base model and route requests to them by name.

### Theory (~4 hrs)
> 📖 Deep lectures (read first): [1 · Inference engine landscape](lectures/01-inference-engine-landscape.md) · [2 · Continuous batching & PagedAttention](lectures/02-continuous-batching-paged-attention.md) · [3 · OpenAI-compatible serving & tuning knobs](lectures/03-openai-compatible-serving-and-tuning.md) · [4 · Parallelism: tensor vs pipeline vs data](lectures/04-parallelism-tensor-pipeline-data.md) · [5 · Multi-LoRA serving](lectures/05-multi-lora-serving.md). The bullets below are the recap.

- **The engine landscape (30 min, awareness only).** Read the intros of the **vLLM** docs (docs.vllm.ai) and skim the README of **TGI** (huggingface/text-generation-inference), **SGLang** (sgl-project/sglang), and **TensorRT-LLM** (NVIDIA/TensorRT-LLM). Opinionated default: **use vLLM.** It's the widest-adopted, OpenAI-compatible out of the box, best docs. SGLang wins on some structured/agent workloads and RadixAttention prefix reuse; TensorRT-LLM wins raw latency on NVIDIA if you'll invest in engine builds; TGI is fine and HF-native. Know they exist; build on vLLM.
- **Why throughput ≠ speed (1 hr).** The two mechanisms that matter:
  - **Continuous batching** (a.k.a. iteration-level scheduling): instead of waiting for a whole batch to finish, vLLM adds/removes sequences every decode step, so a finished short request frees its slot immediately. This is the single biggest lever over a request-per-call loop.
  - **PagedAttention / paged KV cache**: the KV cache is stored in fixed-size *blocks* (like OS virtual memory pages) instead of one contiguous buffer per sequence, killing fragmentation and letting you pack far more concurrent sequences into the same VRAM. Read the "PagedAttention" section of the vLLM docs; skip the paper — you need the intuition, not the proofs.
  - **Automatic prefix caching** (`--enable-prefix-caching`): identical prompt prefixes (system prompt, few-shot block, shared RAG context) reuse cached KV blocks across requests. Huge for chat with a fat shared system prompt.
  - **Chunked prefill**: long prompts get their prefill split into chunks interleaved with ongoing decodes, so one 30k-token prompt doesn't stall everyone else's token stream. On by default in recent vLLM.
- **OpenAI-compatible endpoints & portability (30 min).** vLLM exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`. The whole point: your app code, your LiteLLM gateway, your evals all point at a new `base_url` and nothing else changes. This is *the* reason to prefer OpenAI-compatible servers over bespoke ones.
- **Parallelism basics (45 min, conceptual).** `--tensor-parallel-size N` splits each layer's matrices across N GPUs on **one** node (needs fast interconnect; use when the model doesn't fit on one card). `--pipeline-parallel-size` splits *layers* across GPUs/nodes (for very large models / multi-node). Data parallelism = replicas behind a load balancer (what you actually reach for to scale QPS). Rule of thumb: reach for replicas first; use tensor parallel only when a single model won't fit in one GPU's VRAM.
- **Multi-LoRA serving (30 min).** vLLM (and **LoRAX** by Predibase, and the S-LoRA technique) can hold one base model in VRAM and hot-swap many small LoRA adapters, so N tenant fine-tunes cost ~1 base model of VRAM, not N. Read vLLM's "LoRA" docs page. This is the economics unlock for per-tenant fine-tunes.

### Lab (~8 hrs)
> 🛠️ Full step-by-step guide: [Stand up and tune a vLLM OpenAI-compatible endpoint with multi-LoRA](labs/week-1-vllm-serving.md). The steps below are the summary.

Repo layout:
```
week1-vllm/
  README.md
  docker/run-vllm.sh
  client/smoke_test.py
  client/chat_template_check.py
  lora/route_lora.py
  notes/tuning-log.md
```

**Step 0 — Get a GPU (30 min).** Cheapest path: RunPod "GPU Cloud" pod with an **A10 (24GB)** or **L4 (24GB)**, ~$0.30–0.45/hr, template = "vLLM" or a CUDA base. Alternatives: Modal (has free monthly credits; run vLLM in a `@app.function(gpu="A10G")`), or a free Colab T4 (16GB — use a small model). Pick a model that fits: **`Qwen/Qwen2.5-7B-Instruct`** (ungated, ~15GB in fp16, tight on 16GB — use `--dtype half --max-model-len 8192` or a 3B/AWQ variant on small cards).

**Step 1 — Launch vLLM, OpenAI-compatible (1 hr).** `docker/run-vllm.sh`:
```bash
#!/usr/bin/env bash
docker run --gpus all --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --enable-prefix-caching
```
Wait for `Uvicorn running on http://0.0.0.0:8000`. If you're not on Docker/GPU, install with `uv pip install vllm` and run `vllm serve Qwen/Qwen2.5-7B-Instruct ...` with the same flags.

**Step 2 — Smoke test with the stock OpenAI client (30 min).** `client/smoke_test.py`:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
print(client.models.list().data[0].id)
r = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
    max_tokens=5, temperature=0,
)
print(repr(r.choices[0].message.content))
```
Also hit it with `curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{...}'` so you've seen the raw wire format.

**Step 3 — Break it on purpose, then fix (2 hrs).** In `notes/tuning-log.md`, record each experiment (flags → observed behavior):
- **KV-cache OOM:** relaunch with `--max-model-len 32768 --max-num-seqs 256` on a 24GB card. Observe the startup error or a mid-load crash. Read the log line reporting available KV cache blocks. Fix by lowering `--max-model-len`, lowering `--max-num-seqs`, or (last resort) `--gpu-memory-utilization 0.95`. Write down *which lever you pulled and why*.
- **Chat-template mismatch:** in `client/chat_template_check.py`, send a raw `/v1/completions` request with hand-built `<|im_start|>` tags vs the `/v1/chat/completions` path that applies the model's template. Compare outputs — the mismatched one rambles or ignores the system role. Lesson: let the server apply the chat template; if a model ships a broken/missing template, pass `--chat-template ./template.jinja`.
- **Prefix caching win:** send 50 requests that share a 1,000-token system prompt with `--enable-prefix-caching` on vs off. Note TTFT difference in the server logs.

**Step 4 — Multi-LoRA serving (2.5 hrs).** Grab (or train in Phase 8 style, or download) 2–3 small LoRA adapters for your base (e.g., any `*-lora` adapter on the Hub compatible with Qwen2.5, or make toy ones with PEFT). Relaunch:
```bash
... vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-modules sql-adapter=/adapters/sql json-adapter=/adapters/json \
  --max-loras 4 --max-lora-rank 16
```
In `lora/route_lora.py`, send requests with `model="sql-adapter"` vs `model="json-adapter"` vs the base and confirm you get different behavior from **one** loaded base. This is your per-tenant fine-tune story.

**Step 5 — Document (30 min).** README: exact launch command, the tuning table, and a 3-line "when to use tensor-parallel" note.

### Definition of Done
- [ ] `smoke_test.py` prints `'pong'` (exact) for temperature=0, and `models.list()` returns your model id.
- [ ] `curl` against `/v1/chat/completions` returns valid JSON with a non-empty `choices[0].message.content`.
- [ ] `tuning-log.md` documents at least one reproduced KV-cache OOM and the exact flag change that fixed it.
- [ ] `chat_template_check.py` demonstrates a visible quality difference between correct and mismatched templating.
- [ ] Multi-LoRA: three request variants (`sql-adapter`, `json-adapter`, base) produce measurably different outputs from a single running server; `nvidia-smi` shows only one base model's worth of weights loaded.
- [ ] README contains the working launch command and a tuning table.

### Pitfalls
- **`--gpu-memory-utilization` is a fraction, not GB.** 0.90 = "use 90% of the card." Push to 0.95+ and other processes (or CUDA graphs) OOM.
- **Model doesn't fit and you blame vLLM.** A 7B model in fp16 is ~14GB *before* KV cache. On a 16GB card use an AWQ/GPTQ quant or a 3B model. Do the VRAM math (Week 2) *before* launching.
- **Silent chat-template bugs.** If outputs are subtly worse than the same model on Together/Fireworks, suspect the template before the weights.
- **Prefix caching false expectations.** It only helps *identical* prefixes. A per-request timestamp at the top of your system prompt destroys the cache.
- **LoRA rank / target mismatch.** `--max-lora-rank` must be ≥ the adapter's rank, and the adapter must target modules the base exposes, or loading fails cryptically.

### Self-check
1. Why does continuous batching beat "loop over requests calling `generate()`" even at the same batch size?
2. You get KV-cache OOM at startup. Name three flags that can fix it and the tradeoff of each.
3. When would you choose `--tensor-parallel-size 2` over running two replicas?
4. How does serving 10 tenant LoRAs on one base change your VRAM bill vs 10 full fine-tunes?
5. What single thing must match between how you *train* a chat model and how you *serve* it, or quality silently degrades?

---

## Week 2 — Hardware, economics & FinOps: VRAM math, load testing, and the break-even chart

**Hours:** ~13 (Theory ~4.5, Lab ~8.5)

### Objectives
By end of week you can:
1. Estimate VRAM for any model as **weights + activations + KV cache** and predict max concurrency before you rent the GPU.
2. Explain why decode is **memory-bandwidth-bound** and what that means for batching and GPU choice.
3. Load-test your vLLM endpoint with **k6** (or locust) at concurrency 1/8/64 and report tokens/sec, TTFT, and TPOT.
4. Compute **cost per 1M tokens** for self-hosted vs a hosted API vs serverless scale-to-zero, and find the crossover volume.
5. Read `nvidia-smi`/`nvitop` to tell whether you're compute-bound, memory-bound, or under-utilizing the GPU.

### Theory (~4.5 hrs)
> 📖 Deep lectures (read first): [6 · The VRAM equation](lectures/06-vram-equation.md) · [7 · Memory-bound decode & GPU literacy](lectures/07-memory-bound-decode-gpu-literacy.md) · [8 · Load testing & latency metrics](lectures/08-load-testing-latency-metrics.md) · [9 · Deploy decision & break-even](lectures/09-deploy-decision-breakeven.md) · [10 · Serverless economics & LLM FinOps](lectures/10-serverless-economics-finops.md). The bullets below are the recap.

- **The VRAM equation (1.5 hrs — the most valuable hour of the phase).**
  - **Weights:** params × bytes/param. fp16/bf16 = 2 bytes, fp8 = 1, int4 (AWQ/GPTQ) ≈ 0.5. So 7B fp16 ≈ 14GB; 7B int4 ≈ 3.5GB.
  - **KV cache per token** ≈ `2 (K and V) × n_layers × n_kv_heads × head_dim × bytes`. For a 7B-class model that's roughly a few hundred KB/token; multiply by (context length × concurrent sequences). This is what actually caps your `--max-num-seqs`. Compute it by hand once for your model using its `config.json` (`num_hidden_layers`, `num_key_value_heads`, `hidden_size`), then let vLLM's startup log confirm the block count.
  - **Activations:** transient, comparatively small at inference. Don't over-think, but leave headroom.
  - Takeaway: **KV cache, not weights, usually limits concurrency.** GQA (few KV heads) is why modern models serve so many concurrent users.
- **Memory-bound decode (45 min).** Prefill is compute-heavy (big matmuls, GPU-utilization near 100%). Decode generates one token at a time and is bottlenecked by **reading the weights + KV cache from VRAM**, not by FLOPs — that's why decode GPU util can look "low" yet you can't go faster, and why **bigger batches are nearly free** for throughput during decode. Reference: search "vLLM performance tuning" in the official docs and Horace He's "Making Deep Learning Go Brrrr From First Principles" for the memory-vs-compute-bound mental model.
- **GPU tiers & literacy (45 min).** Know the rough ladder: **T4/L4 (16–24GB, cheap, inference)** → **A10/A10G (24GB)** → **A100 (40/80GB)** → **H100/H200 (80/141GB)**. `nvidia-smi` for a snapshot (memory used, util, processes); **`nvitop`** for a live dashboard. Learn to read: high memory + low SM util during generation = memory-bound (expected); low memory + low util = you're wasting the card (raise `--max-num-seqs`).
- **Deploy-model decision & break-even (45 min).** Four options: **closed API** (OpenAI/Anthropic — zero ops, priced per token, best for low/spiky volume and frontier quality); **serverless GPU** (Modal/RunPod/Baseten/Replicate — scale-to-zero, pay per second, cold starts; best for bursty/medium volume); **dedicated GPU** (rent a box 24/7 — best for steady high volume, predictable); **on-prem** (you own hardware — highest volume, data-residency, longest horizon). The math: dedicated only wins once `hours_used × $/hr` beats `tokens × $/token` on the API. Everything below that crossover, call the API.
- **Serverless economics (30 min).** Scale-to-zero means you pay nothing at idle but eat a **cold start** (model load into VRAM: seconds to minutes). Warm pools / min-replicas trade idle cost for latency. Read Modal's and Baseten's docs on cold starts and autoscaling.
- **FinOps for LLMs (30 min).** Levers you'll wire later or reference: **Batch API** discounts (~50% off for async, non-urgent jobs — OpenAI/Anthropic), **model cascades** (cheap model first, escalate on low confidence), **per-tenant token attribution** (tag every request with tenant id in your gateway/traces), and **budget alerts** (spend threshold → alert/kill-switch). Reference your Phase 9 LiteLLM/Helicone setup.

### Lab (~8.5 hrs)
> 🛠️ Full step-by-step guide: [VRAM math, load testing, and the buy-vs-rent break-even chart](labs/week-2-economics-loadtest-breakeven.md). The steps below are the summary.

Repo layout:
```
week2-economics/
  vram_estimate.py
  loadtest/script.js         # k6
  loadtest/locustfile.py     # alternative
  results/latency.csv
  results/breakeven.py
  results/breakeven.png
  README.md
```

**Step 1 — VRAM estimator (1.5 hrs).** Write `vram_estimate.py` that reads a model's `config.json` (via `huggingface_hub` or a pasted dict) and prints weights GB, KV-cache GB per 1k tokens per sequence, and **max concurrent sequences** at a given context length for a target VRAM. Validate against your Week-1 server: does your predicted `max-num-seqs` match what vLLM logged as feasible? Get within ~20%.

**Step 2 — Instrument the endpoint (1 hr).** Confirm your vLLM server exposes Prometheus metrics at `/metrics` (it does by default). Note the counters for TTFT and generation throughput. Start `nvitop` in a second terminal and keep it visible during load tests.

**Step 3 — Load test with k6 (3 hrs).** Install k6 (`brew install k6` / choco / Docker). `loadtest/script.js`:
```javascript
import http from 'k6/http';
import { check } from 'k6';
import { Trend } from 'k6/metrics';
const ttft = new Trend('ttft_ms', true);   // approx via streaming first-byte if you parse SSE
export const options = {
  scenarios: {
    c1:  { executor: 'constant-vus', vus: 1,  duration: '60s', startTime: '0s'  },
    c8:  { executor: 'constant-vus', vus: 8,  duration: '60s', startTime: '70s' },
    c64: { executor: 'constant-vus', vus: 64, duration: '60s', startTime: '140s'},
  },
};
const BODY = JSON.stringify({
  model: 'Qwen/Qwen2.5-7B-Instruct',
  messages: [{ role: 'user', content: 'Write a 100-word product blurb about a smart water bottle.' }],
  max_tokens: 150, temperature: 0.7,
});
export default function () {
  const res = http.post('http://localhost:8000/v1/chat/completions', BODY,
    { headers: { 'Content-Type': 'application/json' } });
  check(res, { 'status 200': (r) => r.status === 200 });
}
```
Run `k6 run loadtest/script.js`. Record p50/p95/p99 latency and req/s per concurrency tier. For a true **TTFT vs TPOT** split, add a streaming variant (`stream: true`) and measure time to first SSE chunk (TTFT) vs (total_time − TTFT)/tokens (TPOT). Write results to `results/latency.csv`. (locust alternative provided in `loadtest/locustfile.py` if you prefer Python.)

**Step 4 — Cost per 1M tokens (1.5 hrs).** In `results/breakeven.py`: from your load test, take sustained tokens/sec at the best concurrency tier. Compute `cost_per_1M = (1e6 / tokens_per_sec / 3600) × gpu_hourly_rate`. Compare against a hosted open-model API's published per-1M price (e.g., Together/Fireworks for the same Qwen/Llama size) and a frontier API. Plot **monthly cost vs monthly token volume** for: (a) dedicated GPU 24/7, (b) hosted API linear-per-token, (c) serverless scale-to-zero on a bursty profile (idle hours cost ~0 but add cold-start amortization). Save `breakeven.png`. Mark the crossover volume where self-hosting wins.

**Step 5 — Write the recommendation (30 min).** In README, state the crossover in tokens/month and give a one-paragraph "when I'd self-host vs call the API vs go serverless" for *your* measured numbers.

### Definition of Done
- [ ] `vram_estimate.py` predicts max concurrency within ~20% of what your Week-1 vLLM server actually reported.
- [ ] `latency.csv` has TTFT and TPOT (or total latency) at p50/p95/p99 for concurrency 1, 8, and 64.
- [ ] You can point at `nvitop` during a run and correctly label the server memory-bound vs under-utilized.
- [ ] `breakeven.png` plots ≥3 cost curves and marks the self-host-vs-API crossover volume.
- [ ] README states the crossover in tokens/month and a defensible deploy recommendation.

### Pitfalls
- **Confusing throughput with latency.** Concurrency 64 maximizes tokens/sec but p99 TTFT may blow your SLO. Both matter; report both.
- **Testing from a laptop over the internet.** Network latency swamps TTFT. Run the load generator near the server (same region/box).
- **Ignoring input tokens in cost.** Long system prompts / RAG context cost real money and VRAM. Price both prompt and completion tokens.
- **Comparing quantized self-host quality to fp16 API quality as if equal.** Re-eval quality after quantizing; a cheaper cost/1M is meaningless if quality dropped.
- **Forgetting the ops tax.** Self-hosting cost isn't just $/hr — it's your on-call, upgrades, and idle time. Bake a fudge factor into break-even.

### Self-check
1. For a 7B model at 8k context, what dominates VRAM as you raise concurrency — weights or KV cache? Why?
2. Why is decode memory-bandwidth-bound, and why does that make larger batches "almost free"?
3. Your GPU shows 95% memory used but 15% SM utilization during generation. Bug or expected?
4. At what monthly token volume does a dedicated A10 beat a hosted API at $X/1M — write the formula.
5. When is serverless scale-to-zero the right call despite cold starts?

---

## Week 3 — Performance, edge & the release pipeline: shadow, canary, rollback

**Hours:** ~13 (Theory ~4.5, Lab ~8.5)

### Objectives
By end of week you can:
1. Reason about **test-time compute** (reasoning/thinking budgets) and **long-context vs RAG vs CAG** as cost/latency decisions, not features.
2. Run a model **on-device/edge** (Ollama + GGUF via llama.cpp; MLX on Apple silicon; WebLLM in a browser) and say when edge beats server.
3. Set **latency SLOs** (TTFT/TPOT, p95) and wire an alert that fires before users notice.
4. Ship a **prompt/model change through CI** with an eval gate, deploy behind a **feature flag**, **shadow** a fraction of traffic, and **roll back instantly**.
5. Explain why non-determinism forces you to gate on *eval metrics with tolerances*, not exact-match diffs.

### Theory (~4.5 hrs)
> 📖 Deep lectures (read first): [11 · Test-time compute & reasoning budgets](lectures/11-test-time-compute-reasoning.md) · [12 · Long context vs RAG vs CAG](lectures/12-long-context-rag-cag.md) · [13 · Latency SLOs & speculative decoding](lectures/13-latency-slos-speculative-decoding.md) · [14 · On-device, edge & in-browser inference](lectures/14-edge-ondevice-browser-inference.md) · [15 · Prompts & models as code: eval gates](lectures/15-prompts-models-as-code-eval-gates.md) · [16 · Progressive delivery: shadow, canary, rollback](lectures/16-progressive-delivery-shadow-canary-rollback.md). The bullets below are the recap.

- **Test-time compute & reasoning models (45 min).** Reasoning models spend extra "thinking" tokens before answering; you're billed for them and they add latency. Controls: OpenAI `reasoning_effort`, Anthropic extended-thinking token budgets, or open reasoning models (DeepSeek-R1 / Qwen "thinking" variants) where you cap output. Engineering rule: reasoning pays for genuinely multi-step problems (math, planning, hard code) and is waste for extraction/classification. Gate it per-route, not globally.
- **Long-context vs RAG vs CAG (45 min).** Stuffing 100k tokens works but is slow, expensive, and suffers **position bias / "lost in the middle"** and context-rot (see the RULER / needle-in-haystack literature — read *about* it, don't reimplement). **RAG** retrieves only what's needed (Phase 6). **CAG (cache-augmented generation)** preloads a fixed corpus into the KV cache / prompt cache once and reuses it — great when the knowledge base is small and stable. Decision: small+static → CAG/prompt-cache; large+dynamic → RAG; genuinely need all of it at once → long context, and pay for it.
- **Latency SLOs (1 hr).** Define and defend: **TTFT** (time to first token — perceived responsiveness, dominated by prefill + queueing) and **TPOT/ITL** (time per output token — streaming smoothness). Set targets like "p95 TTFT < 800ms, p95 TPOT < 50ms." **Speculative decoding** (a small draft model proposes tokens the big model verifies) can cut TPOT with no quality loss — vLLM supports it (`--speculative-model`). Alert on the SLO, not on averages; p50 hides pain.
- **On-device / edge / browser (45 min).** **llama.cpp + GGUF** is the portable CPU/edge runtime; **Ollama** wraps it with a nice CLI and an OpenAI-compatible API on `localhost:11434`; **LM Studio** is the GUI; **MLX** is Apple's silicon-native framework; **WebLLM / transformers.js** run quantized models fully in-browser over **WebGPU** (zero server, total privacy, but limited size). Edge wins for privacy, offline, and per-user cost=0; loses on model size and consistency.
- **CI/CD for prompts & models — LLMOps (1 hr — the heart of the week).** Treat prompts and model configs as **code in git**. A change opens a PR → CI runs your **eval suite as a required check** (from Phase 5/9; promptfoo or your harness) → merge deploys behind a **feature flag** (LaunchDarkly, or a simple config/env flag) → **shadow** N% of prod traffic to the new version and log both outputs without serving the new one to users → promote to **canary** (serve to N% of real users) → full rollout → keep the old version one flag-flip away for **instant rollback**. Because outputs are non-deterministic, gate on **metric thresholds with tolerance** (e.g., "eval score ≥ baseline − 2%", "JSON-valid rate = 100%"), never on byte-for-byte diffs. References: promptfoo docs, LaunchDarkly docs, and any "shadow deployment" / "canary release" writeup — the ML twist is only the eval gate and non-determinism.

### Lab (~8.5 hrs)
> 🛠️ Full step-by-step guide: [Edge inference plus a governed release pipeline — and the phase milestone](labs/week-3-release-pipeline-and-milestone.md). The steps below are the summary.

Repo layout:
```
week3-release/
  edge/ollama_demo.sh
  edge/webllm/index.html
  slo/spec_decode.md
  gateway/router.py            # flag-based version routing + shadow
  gateway/shadow_log.jsonl
  evals/promptfoo.yaml
  .github/workflows/eval-gate.yml
  scripts/rollback.sh
  README.md
```

**Step 1 — Edge model in 15 minutes (1 hr).** Install Ollama, `ollama run qwen2.5:3b`, then hit its OpenAI-compatible endpoint:
```bash
curl http://localhost:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:3b","messages":[{"role":"user","content":"one word: hi"}]}'
```
Note that your Week-1 client code works unchanged by swapping `base_url`. (Optional: run `edge/webllm/index.html` — a ~30-line WebLLM page — and load a small model fully in the browser; observe first-load download then zero server cost.)

**Step 2 — Speculative decoding note (30 min).** If on a GPU, relaunch vLLM with a small draft model (e.g., a 0.5B Qwen drafting for the 7B) and re-run a short k6 pass; record TPOT before/after in `slo/spec_decode.md`. If CPU-only, document the mechanism and expected win instead.

**Step 3 — Flag-based version router with shadow traffic (3 hrs).** `gateway/router.py` — a small FastAPI (or extend your Phase 9 gateway) that:
- Reads `ACTIVE_VERSION` and `SHADOW_VERSION` + `SHADOW_PCT` from config/env (your feature flag).
- Routes each request to the `ACTIVE_VERSION` prompt/model and returns it.
- For `SHADOW_PCT` of requests, *also* fires the request to `SHADOW_VERSION` **without awaiting/serving it to the user**, and appends `{request_id, active_out, shadow_out, latency_active, latency_shadow}` to `shadow_log.jsonl`.
- Tags every request with a `tenant_id` for cost attribution (ties back to Week 2 FinOps).
"Versions" here = a prompt template + model id pair, stored as files in git so a change is a diff.

**Step 4 — Eval gate in CI (2.5 hrs).** `evals/promptfoo.yaml` defines your test cases + assertions (JSON-valid, contains-expected, LLM-rubric score) against both versions. `.github/workflows/eval-gate.yml`:
```yaml
name: eval-gate
on: [pull_request]
jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npx promptfoo@latest eval -c evals/promptfoo.yaml --no-progress-bar
      # promptfoo exits non-zero if an assertion/threshold fails -> blocks merge
```
Make the check **required** in branch protection. Prove it: open a PR that intentionally worsens the prompt (e.g., breaks the JSON instruction) and watch CI go red.

**Step 5 — Rollback + SLO alert (1.5 hrs).** `scripts/rollback.sh` flips `ACTIVE_VERSION` back to the previous value (one command / one flag change) and confirms via a health request. Add a tiny SLO check: a script that reads recent latencies (from your gateway metrics or the shadow log) and exits non-zero / posts an alert if p95 TTFT breaches your target. Document the full flow in README: **PR → eval gate → merge → shadow 10% → canary → full → rollback.**

### Definition of Done
- [ ] Ollama endpoint answers via the *unchanged* OpenAI client (only `base_url` changed).
- [ ] `router.py` serves `ACTIVE_VERSION` and logs paired active+shadow outputs for ~`SHADOW_PCT` of requests to `shadow_log.jsonl`, each tagged with `tenant_id`.
- [ ] CI eval gate is a **required** check; a deliberately-broken prompt PR fails CI (red), a good one passes (green) — demonstrate both.
- [ ] `rollback.sh` flips active version and a follow-up request confirms the old behavior in under, say, 30 seconds.
- [ ] `slo/` documents a TTFT/TPOT target and an alert that fires when p95 breaches it (spec-decoding before/after numbers if on GPU).
- [ ] README diagrams the shadow → canary → rollout → rollback flow.

### Pitfalls
- **Shadow traffic that blocks the user.** Fire-and-forget the shadow call (background task); never make the user wait on it, and never let a shadow error 500 the real response.
- **Gating on exact output match.** LLMs are non-deterministic; assert on structure/score/thresholds, not string equality, or every PR flaps red.
- **Shadowing side-effecting calls.** If the request triggers tools that write/email/charge, shadowing runs them twice. Shadow read-only paths or stub the tools.
- **"Rollback" that requires a redeploy.** If reverting takes a 10-minute build, it's not instant. Rollback must be a flag flip / config change.
- **Reasoning tokens billed silently.** Extended-thinking output tokens count against cost and latency budgets; monitor them or your bill surprises you.

### Self-check
1. When does CAG beat RAG, and when does neither beat just paying for long context?
2. Why gate CI on eval-metric thresholds instead of exact output diffs?
3. What's the difference between shadow and canary, and why do shadow first?
4. TTFT is great but TPOT is bad — what does the user experience, and what likely causes it?
5. What makes a rollback "instant," and why is a redeploy disqualified?

---

## Phase milestone project

**Serve-it-yourself break-even study + a governed prompt/model release pipeline.**

Build one repo that ties the phase together:

**Part A — Break-even study.** Deploy an open model (Qwen2.5-7B or Llama-3.1-8B) behind **vLLM** with an OpenAI-compatible endpoint. Load-test with **k6/locust** at concurrency **1 / 8 / 64**, reporting tokens/sec, cost/1M tokens, and TTFT+TPOT at p50/p95/p99. Serve **≥3 LoRA adapters on one base** routed by `tenant_id`. Produce a chart of **monthly cost vs volume** for dedicated vs hosted API vs serverless-scale-to-zero (bursty profile), with the crossover volume marked and a one-paragraph recommendation.

**Part B — Release pipeline.** A change to a prompt or model config opens a **PR** that runs the **eval suite as a required CI check**; on merge it deploys behind a **feature flag**, **shadows 10%** of traffic (logging paired outputs, no user impact), supports **canary** promotion, and offers **one-command instant rollback**. Include a **latency report** (TTFT+TPOT at p50/p95/p99 across concurrency) and an **SLO alert** that fires before users would notice.

### Acceptance criteria
- [ ] vLLM endpoint passes an OpenAI-client smoke test and serves ≥3 LoRAs from one base (verified by `nvidia-smi` showing single base weights).
- [ ] Load-test results table: tokens/sec, cost/1M, TTFT+TPOT at p50/p95/p99 for concurrency 1/8/64.
- [ ] Break-even chart with a marked crossover volume and a written deploy recommendation.
- [ ] CI eval gate is required; a bad-prompt PR fails and a good one passes (both demonstrated).
- [ ] Shadow logging produces paired active/shadow outputs tagged by tenant for ≥10% of requests, with zero user-facing impact.
- [ ] `rollback.sh` reverts the active version in a single command; a health request confirms.
- [ ] SLO alert fires on a simulated p95 TTFT breach.
- [ ] README documents architecture, the economics conclusion, and the release flow diagram.

### Suggested repo layout
```
phase10-llmops/
  serving/
    run-vllm.sh              # base + LoRA launch
    smoke_test.py
  economics/
    vram_estimate.py
    loadtest/{script.js, locustfile.py}
    results/{latency.csv, breakeven.py, breakeven.png}
  release/
    gateway/router.py        # flag routing + shadow + tenant tagging
    evals/promptfoo.yaml
    .github/workflows/eval-gate.yml
    scripts/rollback.sh
    slo/alert.py
  README.md                  # architecture, economics conclusion, release flow
```

## You are ready to move on when...
- [ ] You can launch vLLM, hit it with the stock OpenAI client, and tune `max-num-seqs`/`gpu-memory-utilization`/`max-model-len` to fix a KV-cache OOM from the logs alone.
- [ ] You can serve multiple LoRA adapters on one base and route by tenant.
- [ ] You can estimate VRAM (weights + activations + KV cache) and predict max concurrency before renting a GPU.
- [ ] You can load-test an endpoint and report TTFT vs TPOT at p50/p95/p99, and read `nvitop` to say memory-bound vs under-utilized.
- [ ] You can produce a defensible closed-API vs serverless vs dedicated vs on-prem recommendation with a break-even chart.
- [ ] You can run a model on the edge (Ollama/GGUF, WebLLM) and articulate when edge beats a server.
- [ ] You have a working prompt/model CI pipeline: eval gate → feature flag → shadow → canary → instant rollback, gating on metric thresholds not exact matches.
