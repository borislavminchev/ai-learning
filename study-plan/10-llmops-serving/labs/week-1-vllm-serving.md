# Week 1 Lab: Stand up and tune a vLLM OpenAI-compatible endpoint with multi-LoRA

> This week you stop treating an LLM as somebody else's HTTP endpoint and **become the endpoint**. You launch an open model (`Qwen/Qwen2.5-7B-Instruct` by default) under **vLLM**'s OpenAI-compatible server, hit it with the *stock* `openai` Python client and `curl` with zero app changes, then deliberately break it three ways — a **KV-cache OOM**, a **chat-template mismatch**, and a **cold prefix cache** — so you can read the server logs and fix each from first principles. You finish by serving **2–3 LoRA adapters on one base model** and routing to them by name, proving the per-tenant fine-tune economics: N adapters cost ~1 base model of VRAM, not N. Everything is captured in a `week1-vllm/` repo you keep in git; Weeks 2–3 (economics, release pipeline) build directly on this server.
>
> Read the matching lectures first — this guide assumes them:
> - [../lectures/01-inference-engine-landscape.md](../lectures/01-inference-engine-landscape.md) — vLLM vs TGI vs SGLang vs TensorRT-LLM, and why the default is vLLM.
> - [../lectures/02-continuous-batching-paged-attention.md](../lectures/02-continuous-batching-paged-attention.md) — continuous batching, PagedAttention, prefix caching, chunked prefill — *the* mental model for why the flags in Step 3 do what they do.
> - [../lectures/03-openai-compatible-serving-and-tuning.md](../lectures/03-openai-compatible-serving-and-tuning.md) — the `/v1/*` endpoints, the chat template, and the `--max-num-seqs`/`--gpu-memory-utilization`/`--max-model-len` tuning knobs.
> - [../lectures/04-parallelism-tensor-pipeline-data.md](../lectures/04-parallelism-tensor-pipeline-data.md) — tensor vs pipeline vs data parallelism; feeds the 3-line tensor-parallel note in your README.
> - [../lectures/05-multi-lora-serving.md](../lectures/05-multi-lora-serving.md) — one base + many hot-swapped adapters (vLLM / LoRAX / S-LoRA); the Step 4 economics unlock.
>
> This lab **creates** the `week1-vllm/` repo. Keep it in git from day one.

**Est. time:** ~8 hrs · **You will need:** Python 3.10+, [`uv`](https://docs.astral.sh/uv/), a Hugging Face account (Qwen2.5 is **not** gated — zero friction), `curl`, and **one GPU for a few hours**. **Free/low-cost GPU path:** a [RunPod](https://runpod.io) A10/L4 **24 GB** pod (~$0.30–0.45/hr) or [Modal](https://modal.com) free monthly credits are the smoothest. No card at all? A **free Colab/Kaggle T4 (15–16 GB)** works if you drop to a **3B or AWQ-quantized** model (see "Before you start"). The client scripts (`smoke_test.py`, `chat_template_check.py`, `route_lora.py`) run on your **Windows/Git-Bash laptop** against the remote server's `base_url` — only the vLLM process itself needs the GPU.

---

## Before you start (setup)

**What / Why.** vLLM's throughput edge comes from **continuous batching + PagedAttention** (lecture 02): it schedules at the token level and stores the KV cache in fixed-size blocks, so it packs far more concurrent sequences into the same VRAM than a naive `generate()` loop. It also speaks the **OpenAI wire protocol** out of the box (`/v1/chat/completions`, `/v1/completions`, `/v1/models`), which is the whole reason to prefer it — your existing client code, gateway, and evals just point at a new `base_url` (lecture 03). This section gets you a GPU, a model that fits it, and the repo skeleton.

### 1. Pick a GPU and a model that fits it

Do the VRAM math *before* you rent (you'll formalize this in Week 2, but the rule now is simple): a 7B model in fp16 is **~14 GB of weights before any KV cache**. So:

| Card | VRAM | Model to use | Key flags |
|---|---|---|---|
| A10 / L4 / A10G (RunPod, Modal) | 24 GB | `Qwen/Qwen2.5-7B-Instruct` (fp16) | `--max-model-len 8192` |
| T4 (free Colab/Kaggle) | 15–16 GB | `Qwen/Qwen2.5-3B-Instruct` **or** `Qwen/Qwen2.5-7B-Instruct-AWQ` | `--dtype half --max-model-len 8192` (+ `--quantization awq` for AWQ) |
| CPU only (no GPU at all) | — | *vLLM load labs need a GPU* — see the fallback note below | — |

> **No-GPU fallback.** The Step 2 smoke test and Step 3's *template-mismatch* experiment can be reproduced against **Ollama** (`ollama run qwen2.5:3b`, OpenAI-compatible on `localhost:11434`) — the client code is identical. But the **KV-cache OOM**, **prefix-caching TTFT**, and **multi-LoRA** parts genuinely need vLLM on a GPU. Grab a few hours on RunPod (~$1–2 total) or Modal free credits; it is the cheapest part of this phase.

### 2. Get the GPU (RunPod example)

1. RunPod → **GPU Cloud** → deploy an **A10 (24 GB)** or **L4 (24 GB)** On-Demand pod. Template: **"vLLM"** (comes with the image) or any CUDA base.
2. Expose **TCP port 8000** in the pod's networking settings so the OpenAI endpoint is reachable.
3. Set an environment variable in the pod: `HF_TOKEN=<your hf token>` (from [hf.co/settings/tokens](https://huggingface.co/settings/tokens); Qwen is ungated so a read token is plenty).
4. Open a web terminal / SSH into the pod.

> **Modal alternative.** Run vLLM inside a `@app.function(gpu="A10G", ...)` and expose it with `@modal.web_server(8000)`. Modal's own docs have a copy-paste "Run vLLM as an OpenAI-compatible server" example — use it, then point your local clients at the Modal URL. Free monthly credits cover this lab.

### 3. Scaffold the repo (on the box that will run the client — your laptop is fine)

```bash
mkdir week1-vllm && cd week1-vllm
git init
mkdir -p docker client lora notes
touch docker/run-vllm.sh client/smoke_test.py client/chat_template_check.py \
      lora/route_lora.py notes/tuning-log.md README.md

# Client-side deps only (the server has its own vLLM image/env)
uv init --python 3.11
uv add "openai>=1.40" "requests"
```

Target layout (matches the spine):
```
week1-vllm/
  README.md
  docker/run-vllm.sh
  client/smoke_test.py
  client/chat_template_check.py
  lora/route_lora.py
  notes/tuning-log.md
```

**A note on `BASE_URL`.** If your client runs on the **same box** as vLLM, use `http://localhost:8000/v1`. If the client is your laptop and vLLM is on RunPod, use `http://<pod-public-ip>:8000/v1` (or the RunPod-proxied `https://<pod-id>-8000.proxy.runpod.net/v1`). Every script below reads it from an env var so you set it once:
```bash
export BASE_URL="http://localhost:8000/v1"   # or your pod URL
export MODEL="Qwen/Qwen2.5-7B-Instruct"       # or the 3B / AWQ id you chose
```

Commit the skeleton: `git add . && git commit -m "week1-vllm: skeleton"`.

---

## Step-by-step

### Step 1 — Launch vLLM as an OpenAI-compatible server

**What.** Start `vllm/vllm-openai` in Docker (or `vllm serve` if you installed vLLM directly on the box) with a tuned baseline set of flags, and wait for the line that means "the HTTP server is live."

**Why.** These four flags are the levers you will spend the rest of the week reasoning about (lecture 03):
- `--gpu-memory-utilization 0.90` — vLLM reserves 90% of the card. Weights load first; **everything left over becomes KV-cache blocks**. This is a *fraction, not GB* (a classic pitfall).
- `--max-model-len 8192` — the max context (prompt + output) per sequence. Longer context ⇒ each sequence can consume more KV cache ⇒ fewer concurrent sequences fit.
- `--max-num-seqs 64` — the ceiling on concurrent sequences in a batch. This is your throughput knob; too high with long context ⇒ KV-cache OOM.
- `--enable-prefix-caching` — reuse KV blocks for identical prompt prefixes across requests (the Step 3 TTFT win).

**Do it.** Write `docker/run-vllm.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

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
Make it executable and run it (on the GPU box):
```bash
chmod +x docker/run-vllm.sh
HF_TOKEN=$HF_TOKEN ./docker/run-vllm.sh
```

**T4 / small-card variant** — add quantization and half precision:
```bash
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --enable-prefix-caching
```

**No-Docker / local-install variant** (`uv pip install vllm` in a CUDA env, then):
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.90 --max-model-len 8192 \
  --max-num-seqs 64 --enable-prefix-caching
```

**Expected result.** After a model download (first run only) and a weight-load + KV-profiling pass, the logs settle on a line like:
```
INFO ... Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
Just above it you will also see a line worth memorizing for Step 3, e.g.:
```
INFO ... # GPU blocks: 7960, # CPU blocks: 4096
```
That block count *is* your KV-cache budget.

**Verify.** From the client box:
```bash
curl -s "$BASE_URL/models" | python -m json.tool
```
Expected: JSON with `data[0].id` equal to your model id.

**Troubleshoot.**
- *Hangs on "downloading"* — first launch pulls ~15 GB. The `-v ~/.cache/huggingface` mount means the *second* launch is instant; keep it.
- *`CUDA out of memory` immediately at load* — the weights alone don't fit (you're on a 16 GB card with an fp16 7B). Switch to the AWQ or 3B variant above.
- *`docker: Error response ... could not select device driver "" with capabilities: [[gpu]]`* — the NVIDIA Container Toolkit isn't installed/enabled. On a RunPod GPU template it's preinstalled; on your own box run `nvidia-smi` inside a `--gpus all` container to confirm.
- *Port 8000 unreachable from laptop* — you didn't expose the port (RunPod networking) or you're using `localhost` when the server is remote. Use the pod's public IP / proxy URL.

---

### Step 2 — Smoke test with the stock OpenAI client + raw `curl`

**What.** Prove two things: (a) the *unmodified* `openai` Python client talks to your server, and (b) you have seen the raw JSON wire format that every higher-level tool sits on top of.

**Why.** The portability payoff (lecture 03): if the stock client works with only a `base_url` swap, then your app, your LiteLLM gateway, and your eval harness all work too — no code changes vs a hosted API. `temperature=0` makes the output deterministic so "pong" is a hard gate, not a coin flip.

**Do it.** Write `client/smoke_test.py`:
```python
import os
from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")  # vLLM ignores the key by default

# 1. models.list() must return our model id
models = client.models.list()
print("served model:", models.data[0].id)

# 2. Deterministic 'pong' at temperature=0
r = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
    max_tokens=5,
    temperature=0,
)
content = r.choices[0].message.content
print("reply:", repr(content))
assert content.strip().lower() == "pong", f"expected 'pong', got {content!r}"
print("PASS: exact pong")
```
Run it:
```bash
uv run python client/smoke_test.py
```

Now hit the raw endpoint with `curl` to see the wire format:
```bash
curl -s "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly the word: pong\"}],
    \"max_tokens\": 5,
    \"temperature\": 0
  }" | python -m json.tool
```

**Expected result.** `smoke_test.py` prints the model id, `reply: 'pong'`, and `PASS: exact pong`. The `curl` returns an OpenAI-shaped object: top-level `id`, `object: "chat.completion"`, a `choices` array whose `[0].message.content` is `"pong"`, and a `usage` block with `prompt_tokens`/`completion_tokens`.

**Verify.** Both the assert passing and `choices[0].message.content` being a non-empty string in the curl JSON. These are two of the Definition-of-Done gates.

**Troubleshoot.**
- *Reply is `'Pong.'` or `'pong!'`* — capitalization/punctuation. The `.strip().lower()` in the assert tolerates that; the DoD wants the *word* pong. If the model adds a sentence, lower `max_tokens` or tighten the instruction; at `temperature=0` it should comply.
- *`openai.APIConnectionError`* — `BASE_URL` wrong or server not up. Re-run the Step 1 `curl /models`.
- *`404 model not found`* — the `model` field must **exactly** match the id from `models.list()` (case-sensitive, includes the org prefix).

---

### Step 3 — Break it on purpose, then fix (log everything in `notes/tuning-log.md`)

**What.** Three controlled failures, each recorded in `notes/tuning-log.md` as **flags → observed behavior → the one lever you pulled and why**. Start the log with a header:
```markdown
# Tuning log — week1-vllm
Format: experiment → flags → observed behavior → fix (which lever + why)
```

**Why.** Lectures 02–03: you should be able to diagnose a KV-cache OOM and a template mismatch *from the logs and outputs alone*, and predict which flag moves throughput vs OOM risk. Doing it by hand once burns the mental model in.

#### 3a — Reproduce a KV-cache OOM

**Do it.** Relaunch with a context and concurrency that cannot fit on 24 GB:
```bash
docker run --gpus all --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --enable-prefix-caching
```

**Expected result.** vLLM computes the KV cache it would need for `256 seqs × 32768 tokens` and finds it doesn't fit. You get one of:
- A startup `ValueError` such as *"The model's max seq len (32768) is larger than the maximum number of tokens that can be stored in KV cache (N). Try increasing `gpu_memory_utilization` or decreasing `max_model_len`."*, or
- A very small `# GPU blocks:` count and a warning that concurrency will be far below `--max-num-seqs`, or a mid-load CUDA OOM.

**The fix — pull ONE lever and record why.** The recommended lever is `--max-model-len` because it is the safest and most honest:
```bash
# fix: bring context back to a realistic 8192 (KV per seq drops 4x)
--max-model-len 8192 --max-num-seqs 64
```
Log entry example:
```markdown
## KV-cache OOM
flags: --max-model-len 32768 --max-num-seqs 256 (24GB A10)
observed: startup ValueError, KV cache holds only ~X tokens; requested 256*32768.
fix: lowered --max-model-len 32768 -> 8192. Why: KV cache scales with context
     length PER sequence; 4x shorter context => ~4x more concurrent seqs fit,
     and 8192 covers our real prompts. Preferred over --gpu-memory-utilization
     0.95 (risks CUDA-graph/other-process OOM) and over slashing --max-num-seqs
     (would tank throughput unnecessarily).
```

**Verify.** After the fix, the server reaches the `Uvicorn running` line and `smoke_test.py` passes again.

**Troubleshoot.**
- *It launched fine at 32768/256!* — you're on a bigger card (A100/H100) or a quantized model with a tiny KV footprint. Push `--max-num-seqs` higher (512, 1024) until it breaks, or raise `--max-model-len`. The point is to *see* the block-math wall.
- *OOM at weight load, not KV* — that's a weights problem, not KV. Use a smaller/quantized model first.

#### 3b — Demonstrate a chat-template mismatch

**What.** Show that hand-building the `<|im_start|>` prompt via `/v1/completions` gives visibly worse output than letting the server apply the model's template via `/v1/chat/completions`.

**Why.** The single thing that must match between training and serving is the **chat template** (lecture 03; also the fine-tuning phase's "silent killers"). Get it wrong and quality degrades with no error.

**Do it.** Write `client/chat_template_check.py`:
```python
import os
from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

SYSTEM = "You are a terse assistant. Always answer in exactly one sentence."
USER = "Explain what a KV cache is."

# A) CORRECT: server applies Qwen2.5's chat template from the messages array.
correct = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": USER}],
    max_tokens=120, temperature=0,
)

# B) MISMATCH: we hand-build a *wrong* prompt on the raw /v1/completions route.
#    Note the broken tags: missing the trailing assistant-open, wrong role token,
#    no <|im_end|> discipline -> the model does not know where its turn starts.
bad_prompt = (
    f"<|im_start|>system {SYSTEM}"
    f"user {USER}"           # roles run together, no <|im_end|>, no newline discipline
)
mismatch = client.completions.create(
    model=MODEL, prompt=bad_prompt, max_tokens=120, temperature=0,
)

print("=== CORRECT (chat/completions, templated) ===")
print(correct.choices[0].message.content)
print("\n=== MISMATCH (completions, hand-built tags) ===")
print(mismatch.choices[0].text)
```
Run:
```bash
uv run python client/chat_template_check.py
```

**Expected result.** The **correct** path returns one clean sentence that honors the system role ("terse, one sentence"). The **mismatch** path rambles, ignores the one-sentence system instruction, echoes the tags, continues the *user's* turn, or dumps a wall of text — a visible quality gap. Paste both outputs into `notes/tuning-log.md`.

**Verify.** Two outputs, obviously different in quality/adherence. That's the DoD gate.

**Troubleshoot.**
- *Both look fine* — your hand-built prompt was accidentally close enough. Make it worse: drop the `<|im_start|>` entirely, or swap the roles. The lesson stands: **let the server apply the template.** If a model ships a broken/missing template, pass `--chat-template ./template.jinja` at launch.
- *`completions` route 404s* — some builds only enable chat. Add nothing special for vLLM; it serves both. Confirm with `curl $BASE_URL/completions ...`.

#### 3c — Measure the prefix-caching TTFT win

**What.** Fire 50 requests that all share a ~1000-token system prompt, once with `--enable-prefix-caching` **on** and once **off**, and compare time-to-first-token.

**Why.** Automatic prefix caching (lecture 02) reuses the KV blocks for an identical prefix across requests, so only the *first* request pays to prefill the big system prompt; the rest reuse it. Huge for chat with a fat shared system prompt or shared RAG context. It only helps *identical* prefixes — a per-request timestamp at the top destroys it (a pitfall).

**Do it.** Add a small benchmark script `client/prefix_bench.py`:
```python
import os, time, statistics
from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

# ~1000-token shared system prompt (identical across all 50 requests)
SHARED = ("You are a support assistant for ACME Cloud. " * 180)[:6000]

def ttft_ms(user_msg: str) -> float:
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SHARED},
                  {"role": "user", "content": user_msg}],
        max_tokens=32, temperature=0, stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            return (time.perf_counter() - t0) * 1000  # ms to first token
    return (time.perf_counter() - t0) * 1000

times = [ttft_ms(f"Ticket #{i}: my instance is slow, help.") for i in range(50)]
print(f"n={len(times)}  first={times[0]:.0f}ms  "
      f"median_rest={statistics.median(times[1:]):.0f}ms  "
      f"p95={sorted(times)[int(0.95*len(times))-1]:.0f}ms")
```
Run it against the **prefix-caching-ON** server (your Step 1 launch already has `--enable-prefix-caching`), record the numbers, then **relaunch without the flag** and run again:
```bash
# OFF run: same launch command but DROP --enable-prefix-caching
docker run ... vllm/vllm-openai:latest --model $MODEL \
  --gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 64
uv run python client/prefix_bench.py
```

**Expected result.** With caching **ON**, the first request pays full prefill TTFT (say ~300–600 ms on an A10) and requests 2–50 drop sharply (often 2–5× lower median TTFT) because the 1000-token prefix is a cache hit. With caching **OFF**, every request re-prefills the whole system prompt, so the median stays high. You can corroborate from the **server logs**, which print prefix-cache hit rate / `Prefix cache hit rate: XX%`. Record both runs in `notes/tuning-log.md`.

**Verify.** Median TTFT of requests 2–50 is materially lower with caching on than off, and/or the server log shows a non-zero hit rate.

**Troubleshoot.**
- *No difference* — your prefix isn't truly identical (whitespace, a per-request field snuck into the system prompt) or too short to matter. Make `SHARED` genuinely ~1000 tokens and byte-identical.
- *First request also fast* — the model/prefix was already cached from a previous run. Restart the server between ON/OFF runs for a clean comparison.

---

### Step 4 — Multi-LoRA serving: many adapters, one base

**What.** Load 2–3 small LoRA adapters compatible with Qwen2.5 alongside the base, relaunch with LoRA enabled, and route requests to `sql-adapter` vs `json-adapter` vs the base by the `model` field — confirming different behavior from **one** loaded base model.

**Why.** The economics unlock (lecture 05): vLLM (like LoRAX / S-LoRA) keeps one base model in VRAM and hot-swaps small adapters per request. Ten tenant fine-tunes then cost ~one base model of VRAM plus a few MB per adapter, not ten full models. `nvidia-smi` proving a single base's worth of weights is the visual receipt.

**Do it — get 2–3 adapters.** Two options:

*Option A — make toy adapters yourself with PEFT (deterministic, no Hub hunting).* On the GPU box:
```python
# make_toy_adapter.py  — run once per adapter with a different name/rank target
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import sys

name = sys.argv[1]          # e.g. "sql" or "json"
base = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                 task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
model.save_pretrained(f"/adapters/{name}")   # saves adapter_config.json + weights
print("saved", name)
```
```bash
python make_toy_adapter.py sql
python make_toy_adapter.py json
```
> Toy adapters (untrained) will behave *close* to the base; to make the behavioral difference obvious, either (a) train them briefly on a few examples in Phase-8 style, or (b) prefer Option B and grab real adapters. The **routing mechanism** is what this step proves either way.

*Option B — download real Qwen2.5-compatible adapters* from the Hub (search for `qwen2.5` + `lora` adapter repos; pick ones whose `adapter_config.json` shows `r <= 16` and target modules the base exposes). Clone them into `/adapters/sql` and `/adapters/json`.

**Relaunch with LoRA enabled:**
```bash
docker run --gpus all --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /adapters:/adapters \
  -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 64 \
  --enable-lora \
  --lora-modules sql-adapter=/adapters/sql json-adapter=/adapters/json \
  --max-loras 4 --max-lora-rank 16
```
- `--enable-lora` turns on the adapter machinery.
- `--lora-modules NAME=PATH ...` registers each adapter under a name you route by.
- `--max-loras 4` = how many adapters can be *active in one batch*; `--max-lora-rank 16` **must be ≥ every adapter's rank** or loading fails.

**Route to each variant.** Write `lora/route_lora.py`:
```python
import os
from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
BASE_MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

PROMPT = "Give me the users table: id and email of everyone who signed up today."

def ask(model_name: str) -> str:
    r = client.chat.completions.create(
        model=model_name,                       # <-- the ONLY thing that changes
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=200, temperature=0,
    )
    return r.choices[0].message.content

for name in ["sql-adapter", "json-adapter", BASE_MODEL]:
    print(f"\n===== model={name} =====")
    print(ask(name))
```
Run it, and in a second terminal on the GPU box watch memory:
```bash
uv run python lora/route_lora.py       # client box
nvidia-smi                              # GPU box — check the memory number
```
Also confirm the adapters are registered:
```bash
curl -s "$BASE_URL/models" | python -m json.tool   # lists base + sql-adapter + json-adapter
```

**Expected result.**
- `/v1/models` lists **three** ids: the base, `sql-adapter`, `json-adapter`.
- The three responses to the *same* prompt are **measurably different** (e.g., the SQL adapter returns a `SELECT ...` statement, the JSON adapter returns a JSON object, the base returns prose).
- `nvidia-smi` shows memory consistent with **one 7B base model** (~15–16 GB weights + KV budget), *not* three — adapters add only megabytes each.

**Verify.** Three distinguishable outputs from one running server + `nvidia-smi` showing a single base's footprint. That's the headline DoD gate.

**Troubleshoot.**
- *`ValueError: ... lora rank ... exceeds max_lora_rank`* — an adapter's `r` is bigger than `--max-lora-rank`. Raise the flag (e.g., `--max-lora-rank 32`) to match the largest adapter.
- *`target modules ... not found`* — the adapter targets modules the base doesn't expose (wrong base architecture). Use an adapter actually trained on Qwen2.5, or regenerate the toy adapter with `target_modules=["q_proj","v_proj"]`.
- *Adapter output looks identical to base* — your adapters are untrained toys. Train them briefly or use real ones; the routing is still proven by `/v1/models` + distinct code paths.
- *`404` on `model="sql-adapter"`* — the name in the request must match the left side of `--lora-modules NAME=PATH` exactly.

---

### Step 5 — Document: README with launch command, tuning table, tensor-parallel note

**What.** Write `README.md` capturing the exact working launch command, a tuning table of the levers, and a 3-line note on when to reach for tensor parallelism.

**Why.** The README is the deliverable a teammate (or future-you) uses to reproduce the server. The tuning table is the distilled intuition from Step 3; the tensor-parallel note is lecture 04's rule of thumb so nobody reaches for TP when they should just add replicas.

**Do it.** Fill `README.md`:
```markdown
# week1-vllm — self-hosted Qwen2.5-7B under vLLM (OpenAI-compatible)

## Launch (A10/L4 24GB)
    ./docker/run-vllm.sh
Which runs:
    docker run --gpus all --rm -p 8000:8000 \
      -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN="$HF_TOKEN" \
      vllm/vllm-openai:latest \
      --model Qwen/Qwen2.5-7B-Instruct \
      --gpu-memory-utilization 0.90 --max-model-len 8192 \
      --max-num-seqs 64 --enable-prefix-caching
Wait for: `Uvicorn running on http://0.0.0.0:8000`.

## Tuning table
| Flag | What it controls | Raise it → | Lower it → | When it caused/fixed OOM |
|---|---|---|---|---|
| `--gpu-memory-utilization` | fraction of VRAM vLLM claims (NOT GB) | more KV blocks / concurrency | safer headroom for CUDA graphs & other procs | 0.95+ risked OOM; kept 0.90 |
| `--max-model-len` | max context (prompt+gen) per seq | supports longer prompts | ~linearly more concurrent seqs fit | 32768→8192 fixed the KV OOM |
| `--max-num-seqs` | max concurrent sequences in a batch | higher throughput | lower KV pressure | 256 too high at 32k ctx |
| `--enable-prefix-caching` | reuse KV for identical prefixes | big TTFT win on shared system prompts | (off) every req re-prefills | n/a — pure win for shared prefixes |

## Multi-LoRA
    --enable-lora --lora-modules sql-adapter=/adapters/sql json-adapter=/adapters/json \
    --max-loras 4 --max-lora-rank 16
Route by the `model` field: `sql-adapter` | `json-adapter` | base. One base in VRAM (see nvidia-smi).

## Tensor parallel — when (3 lines)
Reach for replicas (data parallel) FIRST to scale QPS — it's simpler and linear.
Use `--tensor-parallel-size N` ONLY when a single model won't fit in one GPU's VRAM
(splits each layer's matrices across N cards on one node; needs a fast interconnect like NVLink).
```

**Expected result.** A README a stranger can follow to launch, tune, and route LoRAs.

**Verify.** The launch command in the README, copy-pasted, actually starts the server; the tuning table has all four flags; the tensor-parallel note is present. `git add . && git commit -m "week1-vllm: docs + tuning table"`.

---

## Putting it together — end-to-end run

On a fresh terminal, top to bottom (~10 min once the model is cached):
```bash
# 0. env
export BASE_URL="http://localhost:8000/v1" MODEL="Qwen/Qwen2.5-7B-Instruct"

# 1. launch base server (GPU box), wait for "Uvicorn running"
./docker/run-vllm.sh

# 2. smoke test + raw wire format (client box)
uv run python client/smoke_test.py            # -> PASS: exact pong
curl -s "$BASE_URL/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":5}" \
  | python -m json.tool                        # -> valid JSON, non-empty content

# 3. break + fix (each logged in notes/tuning-log.md)
#    - relaunch at --max-model-len 32768 --max-num-seqs 256 -> read KV OOM -> fix to 8192/64
uv run python client/chat_template_check.py    # -> visible correct vs mismatch gap
uv run python client/prefix_bench.py           # -> TTFT drops with caching on

# 4. relaunch with --enable-lora ... then:
uv run python lora/route_lora.py               # -> 3 distinct outputs
nvidia-smi                                      # -> one base model of weights

# 5. README committed with launch cmd + tuning table + TP note
```

---

## Definition of Done — verifiable checks

Restated from the spine; each must pass, not "basically work":

- [ ] **`smoke_test.py` prints `'pong'` (exact) at temperature=0**, and `models.list()` returns your model id. *(Verify: the script's `PASS: exact pong` line.)*
- [ ] **`curl` against `/v1/chat/completions` returns valid JSON** with a non-empty `choices[0].message.content`. *(Verify: `python -m json.tool` parses it; content is a non-empty string.)*
- [ ] **`notes/tuning-log.md` documents ≥1 reproduced KV-cache OOM and the exact flag change that fixed it** — with *which lever and why*. *(Verify: log shows `32768/256` → OOM → `8192` fix + rationale.)*
- [ ] **`chat_template_check.py` shows a visible quality difference** between correct (`/v1/chat/completions`) and mismatched (hand-built `/v1/completions`) templating. *(Verify: two clearly different outputs pasted into the log.)*
- [ ] **Multi-LoRA: three request variants (`sql-adapter`, `json-adapter`, base) produce measurably different outputs from one running server**, and `nvidia-smi` shows only one base model's worth of weights. *(Verify: `/v1/models` lists 3 ids; outputs differ; single-base memory.)*
- [ ] **README contains the working launch command and a tuning table** (+ the 3-line tensor-parallel note). *(Verify: copy-paste the command → server starts.)*

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `could not select device driver ... [[gpu]]` | NVIDIA Container Toolkit missing | Use a RunPod GPU template, or install the toolkit; test with `docker run --gpus all ... nvidia-smi`. |
| CUDA OOM **at weight load** | model too big for the card (7B fp16 on 16 GB) | Switch to `Qwen2.5-3B-Instruct` or `...-7B-Instruct-AWQ --quantization awq --dtype half`. |
| `max seq len larger than KV cache can store` | `--max-model-len` × `--max-num-seqs` exceeds KV budget | Lower `--max-model-len` first (biggest, safest lever), then `--max-num-seqs`; `--gpu-memory-utilization 0.95` only as last resort. |
| `404 model not found` | `model` field ≠ served id | Copy the exact id from `curl $BASE_URL/models`. |
| Outputs subtly worse than the same model on a hosted API | chat-template mismatch | Use `/v1/chat/completions` (server applies the template); if the model's template is broken, pass `--chat-template ./template.jinja`. |
| Prefix caching shows no win | prefix not byte-identical (timestamp/field in system prompt) or too short | Make the shared prefix identical and ~1000 tokens; restart server between on/off runs. |
| `lora rank exceeds max_lora_rank` | adapter `r` > `--max-lora-rank` | Raise `--max-lora-rank` to the largest adapter's rank. |
| `target modules not found` loading LoRA | adapter trained on a different base | Use a Qwen2.5-trained adapter, or regenerate with `target_modules=["q_proj","v_proj"]`. |
| Endpoint unreachable from laptop | port not exposed / using `localhost` for a remote server | Expose TCP 8000 (RunPod networking); use the pod public IP / proxy URL in `BASE_URL`. |
| Server slow to start every time | HF cache not mounted | Keep `-v ~/.cache/huggingface:/root/.cache/huggingface`. |

---

## Stretch goals (optional)

- **Structured outputs / guided decoding.** Add `--guided-decoding-backend` and request a JSON schema via `response_format`; confirm the JSON adapter + guided decoding gives 100% valid JSON. (Ties into the Week-3 eval gate's "JSON-valid rate = 100%".)
- **Hot-load a LoRA at runtime.** Enable dynamic loading (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`) and `POST /v1/load_lora_adapter` a new adapter without restarting the server — the real per-tenant onboarding flow.
- **Prometheus peek.** `curl $BASE_URL/../metrics` (vLLM exposes `/metrics`) and find the TTFT / throughput counters — you'll wire these into k6 dashboards in Week 2.
- **Chunked-prefill stress.** Send one 30k-token prompt alongside many short chats and watch (in logs / TTFT) how chunked prefill keeps the short requests flowing — the lecture-02 mechanism in action.
- **Compare an engine.** Launch the same model under **TGI** (`ghcr.io/huggingface/text-generation-inference`) or **SGLang** and diff startup flags + a quick TTFT — confirm the OpenAI-compatible client code doesn't change (lecture 01).
```
