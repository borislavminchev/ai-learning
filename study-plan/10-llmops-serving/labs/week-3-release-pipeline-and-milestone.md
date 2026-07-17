# Week 3 Lab: Edge inference plus a governed release pipeline — and the phase milestone

> This is the week you stop shipping prompt/model changes by vibes and wrap your Week-1 vLLM server in a **production release process**. You run a model **on the edge** (Ollama + a browser WebLLM page) and prove your Week-1 client code is unchanged by a `base_url` swap; measure a **speculative-decoding** TPOT win; then build the heart of the week — a **flag-based version router** that serves an `ACTIVE_VERSION`, **shadows** N% of traffic to a candidate without touching the user, and tags every request by `tenant_id`. You gate a prompt/model change through **CI with a promptfoo eval suite** as a *required* check, prove it by making a bad-prompt PR go red, deploy behind a **feature flag**, define **latency SLOs** (TTFT/TPOT at p50/p95/p99) with an **alert that fires before users notice**, and wire **one-command instant rollback**. You finish by assembling the **phase milestone**: a serve-it-yourself break-even study (Weeks 1–2) plus this governed release pipeline in one repo.
>
> Read the matching lectures first — this guide assumes them:
> - [../lectures/11-test-time-compute-reasoning.md](../lectures/11-test-time-compute-reasoning.md) — reasoning/thinking budgets as a cost/latency decision; gate per-route.
> - [../lectures/12-long-context-rag-cag.md](../lectures/12-long-context-rag-cag.md) — long-context vs RAG vs CAG; when a preloaded KV/prompt cache beats retrieval.
> - [../lectures/13-latency-slos-speculative-decoding.md](../lectures/13-latency-slos-speculative-decoding.md) — TTFT vs TPOT SLOs, p95 not p50, and how a draft model cuts TPOT with no quality loss.
> - [../lectures/14-edge-ondevice-browser-inference.md](../lectures/14-edge-ondevice-browser-inference.md) — Ollama/GGUF/llama.cpp, MLX, WebLLM/WebGPU; when the server loses.
> - [../lectures/15-prompts-models-as-code-eval-gates.md](../lectures/15-prompts-models-as-code-eval-gates.md) — prompts/models as git-tracked code; gating on metric thresholds under non-determinism.
> - [../lectures/16-progressive-delivery-shadow-canary-rollback.md](../lectures/16-progressive-delivery-shadow-canary-rollback.md) — feature flags, shadow, canary, and what makes a rollback "instant."
>
> This lab **creates** the `week3-release/` repo and then folds it (with Weeks 1–2) into the `phase10-llmops/` **milestone** repo. Keep both in git from day one.

**Est. time:** ~8.5 hrs · **You will need:** Python 3.10+, [`uv`](https://docs.astral.sh/uv/), Node 20+ (for promptfoo/k6-optional), `curl`, a free [GitHub](https://github.com) account (for GitHub Actions CI — the free/local CI path), and **[Ollama](https://ollama.com)** for the edge step. **Free/local path throughout:** Ollama runs the edge model on your CPU/laptop (no GPU, no cost); GitHub Actions runs the eval gate on free public-repo minutes; the router is a tiny FastAPI you run locally; promptfoo runs with `npx` (no install). The **speculative-decoding** step (Step 2) is the only part that wants a GPU — if you have none, you document the mechanism and expected win instead. Everything else is laptop-only.

---

## Before you start (setup)

**What / Why.** The theme of the week (lectures 15–16) is: **a prompt or model config is code**. A change is a diff, it opens a PR, CI runs your eval suite as a *required* check, and merge deploys behind a **flag** you can flip back in one command. Because LLM outputs are non-deterministic (lecture 15), you gate on **metric thresholds with tolerance** ("eval score ≥ baseline − 2%", "JSON-valid rate = 100%"), never byte-for-byte diffs. Around that core you set **latency SLOs** on TTFT (perceived responsiveness) and TPOT (streaming smoothness), and alert on p95, not averages (lecture 13). This section installs the local toolchain and scaffolds the repo.

### 1. Install the local toolchain

```bash
# Ollama — the edge runtime (llama.cpp + GGUF under a nice CLI, OpenAI-compatible API)
#   macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh
#   Windows: download the installer from https://ollama.com/download  (installs a localhost service on :11434)

# Node 20+ (for promptfoo via npx, and optional k6). Check:
node --version    # want v20+

# Python client deps for the router + scripts
mkdir week3-release && cd week3-release
git init
uv init --python 3.11
uv add "fastapi>=0.111" "uvicorn[standard]>=0.30" "httpx>=0.27" "openai>=1.40" "pydantic>=2"
```

> **Windows + Git-Bash notes.** Ollama on Windows runs as a background service on `http://localhost:11434` — no `curl | sh`; use the `.exe` installer. In Git-Bash, prefer `python -m json.tool` (works everywhere) over `jq` unless you've installed it. Multi-line `curl -d '{...}'` with single quotes works in Git-Bash; in PowerShell use `curl.exe` (not the `Invoke-WebRequest` alias) and escape quotes. Long-running commands (`uvicorn`, `ollama serve`) should each get their own terminal.

### 2. Scaffold the repo

```bash
mkdir -p edge/webllm slo gateway evals scripts .github/workflows
touch edge/ollama_demo.sh edge/webllm/index.html slo/spec_decode.md slo/alert.py \
      gateway/router.py gateway/shadow_log.jsonl \
      evals/promptfoo.yaml .github/workflows/eval-gate.yml scripts/rollback.sh README.md
mkdir -p versions   # git-tracked prompt+model "versions" (see Step 3)
```

Target layout (matches the spine):
```
week3-release/
  edge/ollama_demo.sh
  edge/webllm/index.html
  slo/spec_decode.md
  slo/alert.py
  gateway/router.py            # flag-based version routing + shadow
  gateway/shadow_log.jsonl
  versions/                    # prompt template + model id pairs, as files in git
  evals/promptfoo.yaml
  .github/workflows/eval-gate.yml
  scripts/rollback.sh
  README.md
```

### 3. Decide what an app-facing task is

Pick a small, **evaluable** task so the eval gate has something concrete to score. Default for this lab: **"extract a support ticket into strict JSON"** (`{"category": ..., "priority": "low|medium|high", "summary": ...}`). It is deterministic to assert on (JSON-valid rate, required keys, allowed enum values) yet sensitive to prompt quality — perfect for demonstrating a bad-prompt PR going red.

Commit the skeleton: `git add . && git commit -m "week3-release: skeleton"`.

---

## Step-by-step

### Step 1 — Edge model in 15 minutes (Ollama + optional in-browser WebLLM)

**What.** Run `qwen2.5:3b` under Ollama, hit its **OpenAI-compatible** endpoint on `localhost:11434`, and confirm your Week-1 client code works with only a `base_url` change. Optionally load a small model **fully in the browser** with WebLLM (zero server).

**Why.** Lecture 14: `llama.cpp + GGUF` is the portable CPU/edge runtime; Ollama wraps it with an OpenAI-compatible API; WebLLM runs quantized models over **WebGPU** in the browser (total privacy, per-user cost = 0). Edge wins on privacy/offline/cost-per-user; loses on model size and consistency. The `base_url`-only swap is the same portability payoff you proved in Week 1 with vLLM.

**Do it.** Write `edge/ollama_demo.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Pull + run a small model (first pull downloads ~2GB GGUF weights)
ollama pull qwen2.5:3b

# 2. Hit the OpenAI-compatible endpoint (note: /v1, api_key ignored)
curl -s http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5:3b",
    "messages": [{"role": "user", "content": "Reply with exactly one word: hi"}],
    "max_tokens": 5, "temperature": 0
  }' | python -m json.tool
```
Run it, then prove the Week-1 stock client is unchanged:
```bash
chmod +x edge/ollama_demo.sh && ./edge/ollama_demo.sh

# Reuse Week 1's smoke_test.py — ONLY the env vars change:
BASE_URL="http://localhost:11434/v1" MODEL="qwen2.5:3b" \
  uv run python ../week1-vllm/client/smoke_test.py    # adjust path to your Week-1 repo
```

*(Optional) In-browser WebLLM.* Put a minimal page in `edge/webllm/index.html` using the WebLLM ESM module from a CDN (do not invent a version — copy the current import URL from the official WebLLM README at `github.com/mlc-ai/web-llm`):
```html
<!doctype html><meta charset="utf-8"><title>WebLLM edge demo</title>
<pre id="out">loading…</pre>
<script type="module">
  // Import the CreateMLCEngine helper from the WebLLM CDN URL shown in its README.
  import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm";
  const out = document.getElementById("out");
  // A small model id from WebLLM's prebuilt list, e.g. a Qwen2.5-0.5B/1.5B-Instruct-q4f16 build.
  const engine = await CreateMLCEngine("Qwen2.5-1.5B-Instruct-q4f16_1-MLC", {
    initProgressCallback: (p) => { out.textContent = p.text; },   // shows first-load download
  });
  const r = await engine.chat.completions.create({
    messages: [{ role: "user", content: "one word: hi" }], temperature: 0,
  });
  out.textContent = r.choices[0].message.content;   // runs entirely on WebGPU, no server
</script>
```
Serve it locally and open in a WebGPU browser (Chrome/Edge):
```bash
python -m http.server 8080 --directory edge/webllm   # then open http://localhost:8080
```

**Expected result.** The `curl` returns an OpenAI-shaped JSON object with `choices[0].message.content` ≈ `"hi"`. The Week-1 `smoke_test.py`, unchanged, passes against Ollama. The WebLLM page shows a one-time model download (progress text) then answers with **zero** network calls to any server (watch the Network tab — nothing after the weights load).

**Verify.** DoD gate: "Ollama endpoint answers via the *unchanged* OpenAI client (only `base_url` changed)." Confirm the same `.py` file works against both vLLM (Week 1) and Ollama by swapping only `BASE_URL`/`MODEL`.

**Troubleshoot.**
- *`connection refused` on :11434* — the Ollama service isn't running. macOS/Linux: `ollama serve` in a terminal. Windows: check the Ollama tray icon / service.
- *`model not found`* — run `ollama pull qwen2.5:3b` first; the `/v1` `model` field must match exactly (`ollama list` shows names).
- *WebLLM "WebGPU not available"* — use Chrome/Edge (Firefox needs a flag); check `chrome://gpu`. On unsupported hardware, WebLLM is the one edge path that can't fall back — note it and move on; Ollama is the required path.
- *First WebLLM load is huge/slow* — expected; weights download once then cache in the browser. That's the "first-load download then zero server cost" observation the spine asks for.

---

### Step 2 — Speculative decoding: measure the TPOT win (GPU) or document it (CPU)

**What.** Relaunch your Week-1 vLLM server with a small **draft model** speculating for the 7B, run a short streaming load pass, and record **TPOT before vs after** in `slo/spec_decode.md`. No GPU? Document the mechanism and expected win.

**Why.** Lecture 13: speculative decoding has a small draft model propose several tokens that the big model **verifies in one forward pass**; accepted tokens are free, so **TPOT drops with no quality change** (the target model's distribution is preserved). It helps latency, not peak throughput, and the win depends on draft acceptance rate.

**Do it (GPU).** Relaunch vLLM with a draft model (flag names vary across vLLM versions — check `vllm serve --help | grep -i spec`; recent builds use a `--speculative-config` JSON, older ones `--speculative-model` + `--num-speculative-tokens`):
```bash
# Recent vLLM (>=0.6.x): speculative config as JSON
docker run --gpus all --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 16 \
  --speculative-config '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 5}'
```
Measure TPOT with a tiny streaming client (reuse the Week-1 streaming pattern; TPOT = (total_time − TTFT) / output_tokens):
```python
# slo/measure_tpot.py  — run against baseline server, then against the spec-decode server
import os, time
from openai import OpenAI
client = OpenAI(base_url=os.environ["BASE_URL"], api_key="EMPTY")
MODEL = os.environ["MODEL"]

def measure(prompt: str, max_tokens=200):
    t0 = time.perf_counter(); ttft = None; n = 0
    stream = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0, stream=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None: ttft = time.perf_counter() - t0
            n += 1
    total = time.perf_counter() - t0
    tpot_ms = (total - ttft) / max(n - 1, 1) * 1000
    return ttft * 1000, tpot_ms, n

ttft, tpot, n = measure("Explain paged attention in 150 words.")
print(f"TTFT={ttft:.0f}ms  TPOT={tpot:.1f}ms/tok  tokens={n}")
```
Run once against the **baseline** (no draft) server and once against the **spec-decode** server; write both to `slo/spec_decode.md`.

**Do it (CPU / no GPU).** Skip the launch; write `slo/spec_decode.md` documenting the mechanism (draft proposes k tokens → target verifies in one pass → accepted tokens are free), the vLLM flag surface, and a realistic expected win (commonly **~1.5–2.5× lower TPOT** at good acceptance rates; verify the exact figure against current vLLM docs — do not fabricate).

**Expected result.** `slo/spec_decode.md` contains a before/after TPOT table (GPU) or a mechanism write-up + expected-win note (CPU). Output *quality* is unchanged between the two (spot-check one prompt at temperature=0).

**Verify.** DoD gate: `slo/` documents spec-decoding before/after numbers if on GPU. The TPOT (not TTFT) is the number that should move.

**Troubleshoot.**
- *Flag rejected / unknown* — the speculative API changed between vLLM releases. Run `vllm serve --help` in your image and match the current flag; note the version in `spec_decode.md`.
- *TPOT didn't improve* — draft too big/slow or low acceptance (draft disagrees with target often). Use a *much* smaller draft (0.5B for a 7B) and lower `--max-num-seqs` (spec decode helps most at low concurrency; at high batch the target GPU is already saturated).
- *OOM after adding the draft* — the draft model also needs VRAM; lower `--gpu-memory-utilization` headroom or `--max-model-len`.

---

### Step 3 — Flag-based version router with shadow traffic

**What.** Build `gateway/router.py` — a small FastAPI that serves the `ACTIVE_VERSION`, and for `SHADOW_PCT` of requests **also** fires the `SHADOW_VERSION` in the background (never awaited, never served, never allowed to break the response), logging paired outputs + latencies to `shadow_log.jsonl`, tagging every request with `tenant_id`.

**Why.** Lecture 16: **shadow** lets you run a candidate on real traffic and compare outputs **without user impact** — the safest first stage of progressive delivery. A "version" = a prompt template + model id pair, stored as **files in git** (lecture 15), so a change is a reviewable diff. The `tenant_id` tag ties into Week-2 FinOps (per-tenant cost attribution).

**Do it — define versions as files.** Two git-tracked versions of the ticket-extraction task:
```bash
mkdir -p versions/v1 versions/v2
```
`versions/v1/config.json`:
```json
{ "id": "v1", "model": "qwen2.5:3b",
  "system_prompt": "You extract support tickets. Return ONLY compact JSON with keys: category (string), priority (one of: low, medium, high), summary (string). No prose, no code fence." }
```
`versions/v2/config.json` (a candidate — same schema, tweaked wording):
```json
{ "id": "v2", "model": "qwen2.5:3b",
  "system_prompt": "You are a ticket triage engine. Output STRICT JSON only, keys exactly: category, priority (low|medium|high), summary. Do not wrap in markdown." }
```

**Do it — the router.** `gateway/router.py`:
```python
import os, json, time, uuid, asyncio, random, pathlib
from fastapi import FastAPI, Request
from openai import AsyncOpenAI

BASE_URL   = os.environ.get("BASE_URL", "http://localhost:11434/v1")  # Ollama by default
ACTIVE     = os.environ.get("ACTIVE_VERSION", "v1")
SHADOW     = os.environ.get("SHADOW_VERSION", "v2")
SHADOW_PCT = float(os.environ.get("SHADOW_PCT", "0.10"))              # 10%
LOG_PATH   = pathlib.Path(__file__).with_name("shadow_log.jsonl")
VERSIONS   = pathlib.Path(__file__).parents[1] / "versions"

client = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY")
app = FastAPI()

def load_version(vid: str) -> dict:
    return json.loads((VERSIONS / vid / "config.json").read_text())

async def run_version(vid: str, user_text: str) -> tuple[str, float]:
    cfg = load_version(vid); t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model=cfg["model"], temperature=0, max_tokens=200,
        messages=[{"role": "system", "content": cfg["system_prompt"]},
                  {"role": "user", "content": user_text}])
    return r.choices[0].message.content, (time.perf_counter() - t0) * 1000

async def shadow_and_log(request_id, tenant_id, user_text, active_out, latency_active):
    """Fire-and-forget: run SHADOW, log the pair. Must NEVER raise into the response."""
    try:
        shadow_out, latency_shadow = await run_version(SHADOW, user_text)
    except Exception as e:                      # a shadow failure must not affect the user
        shadow_out, latency_shadow = f"<shadow-error: {e}>", -1.0
    rec = {"request_id": request_id, "tenant_id": tenant_id,
           "active_version": ACTIVE, "shadow_version": SHADOW,
           "active_out": active_out, "shadow_out": shadow_out,
           "latency_active_ms": round(latency_active, 1),
           "latency_shadow_ms": round(latency_shadow, 1),
           "ts": time.time()}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

@app.post("/extract")
async def extract(request: Request):
    body = await request.json()
    user_text = body["text"]
    tenant_id = request.headers.get("x-tenant-id", "anon")
    request_id = str(uuid.uuid4())

    active_out, latency_active = await run_version(ACTIVE, user_text)   # served to the user

    if random.random() < SHADOW_PCT:
        # background task: user does NOT wait on the shadow call
        asyncio.create_task(
            shadow_and_log(request_id, tenant_id, user_text, active_out, latency_active))

    return {"request_id": request_id, "version": ACTIVE, "tenant_id": tenant_id,
            "output": active_out, "latency_ms": round(latency_active, 1)}

@app.get("/health")
def health():
    return {"active_version": ACTIVE, "shadow_version": SHADOW, "shadow_pct": SHADOW_PCT}
```
Run it and drive traffic (Ollama must be up from Step 1):
```bash
# terminal A
BASE_URL="http://localhost:11434/v1" ACTIVE_VERSION=v1 SHADOW_VERSION=v2 SHADOW_PCT=0.10 \
  uv run uvicorn gateway.router:app --port 9000

# terminal B — 50 requests across 2 tenants; ~10% get shadowed
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:9000/extract \
    -H 'Content-Type: application/json' -H "x-tenant-id: tenant-$((i % 2))" \
    -d '{"text": "Login button does nothing on mobile, urgent, cannot pay."}' >/dev/null
done
wc -l gateway/shadow_log.jsonl        # ~5 paired records (10% of 50)
python -m json.tool < <(tail -n 1 gateway/shadow_log.jsonl)
```

**Expected result.** `/health` echoes the active/shadow flags. After 50 requests, `shadow_log.jsonl` has ~5 lines (10%), each with `active_out`, `shadow_out`, both latencies, and a `tenant_id`. The served response always comes from `v1`; the user never waits on the shadow call.

**Verify.** DoD gate: router serves `ACTIVE_VERSION` and logs paired active+shadow outputs for ~`SHADOW_PCT` of requests, each tagged with `tenant_id`. Confirm the served `version` is always `ACTIVE`, and that killing the shadow model (e.g., point `SHADOW`'s config at a bogus model id) still returns a clean 200 with a `<shadow-error: ...>` line logged — proving shadow failures are isolated.

**Troubleshoot.**
- *Shadow blocks the response / user slow* — you `await`ed the shadow instead of `asyncio.create_task`. The shadow must be fire-and-forget.
- *Shadow ratio wildly off 10%* — 50 samples is small; run 500 for a tighter ratio, or make `SHADOW_PCT=1.0` temporarily to prove the pairing logic.
- *Shadowing something with side effects* — if a "version" triggers tools that write/email/charge, shadowing runs them twice. Only shadow **read-only** paths, or stub the side-effecting tools (a spine pitfall).
- *`shadow_log.jsonl` interleaves garbled lines under load* — concurrent appends. For this lab a single-process uvicorn + append-mode is fine; if you scale to workers, write via a queue or one file per worker.

---

### Step 4 — Eval gate in CI (promptfoo as a required check)

**What.** Define `evals/promptfoo.yaml` with test cases + assertions (JSON-valid, required keys, allowed enum, LLM-rubric) against the version(s), wire `.github/workflows/eval-gate.yml` to run it on every PR, make it a **required** check, and prove it by opening a **bad-prompt PR that goes red**.

**Why.** Lecture 15: prompts/models are code; a change must pass an **eval suite as a required gate** before merge. Because outputs are non-deterministic, assert on **structure/score/thresholds**, never exact strings. promptfoo exits non-zero when an assertion/threshold fails, which is exactly what blocks a merge.

**Do it.** `evals/promptfoo.yaml` (promptfoo can point `providers` at any OpenAI-compatible `apiBaseUrl`; in CI use a cheap hosted model or a matrix that reads a version file — keep it provider-agnostic):
```yaml
# promptfoo: extract a support ticket to strict JSON, gate on structure + a rubric score
description: "Ticket-extraction eval gate"

prompts:
  # Load the system prompt from the git-tracked version under test (parameterized in CI).
  - "{{system_prompt}}\n\nTicket: {{ticket}}"

providers:
  - id: openai:chat:gpt-4o-mini            # swap for your hosted model, OR:
    # id: openai:chat:qwen2.5:3b
    # config: { apiBaseUrl: "http://localhost:11434/v1", apiKey: "EMPTY" }
    config: { temperature: 0, max_tokens: 200 }

defaultTest:
  assert:
    - type: is-json                         # JSON-valid rate must be 100%
    - type: javascript                      # required keys + allowed enum
      value: |
        const o = JSON.parse(output);
        if (!('category' in o && 'priority' in o && 'summary' in o)) return false;
        return ['low','medium','high'].includes(o.priority);
    - type: llm-rubric                       # non-deterministic quality, scored with tolerance
      value: "summary accurately reflects the ticket; priority is reasonable"
      threshold: 0.7                          # gate on a THRESHOLD, not exact match

tests:
  - vars: { ticket: "Login button does nothing on mobile, urgent, cannot pay." }
  - vars: { ticket: "Typo on the pricing page footer, minor." }
    assert:
      - type: javascript
        value: "return JSON.parse(output).priority === 'low' || JSON.parse(output).priority === 'medium'"
  - vars: { ticket: "Data export is 2 hours late again; customer escalating." }
```
> Wire `{{system_prompt}}` from your `versions/<id>/config.json` — e.g., a small pre-step in CI that reads the file and exports it, or a promptfoo config generated per version. Keep the *prompt text* in git so the diff is the change.

`.github/workflows/eval-gate.yml`:
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
      - name: Load system prompt from the version under test
        run: |
          echo "system_prompt=$(python -c "import json;print(json.load(open('versions/v2/config.json'))['system_prompt'])")" >> "$GITHUB_ENV"
      - name: Run promptfoo eval (exits non-zero on any failed assertion/threshold)
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}   # if using a hosted judge/model
        run: npx promptfoo@latest eval -c evals/promptfoo.yaml --no-progress-bar
```
Make it required, then prove both directions:
```bash
# 1) baseline PR (good prompt) -> CI green
git checkout -b good-prompt && git commit -am "eval gate + versions" && git push -u origin good-prompt
gh pr create --fill    # open PR, watch the eval-gate check pass

# 2) make it REQUIRED: repo Settings > Branches > branch protection rule on main
#    -> "Require status checks to pass" -> select "eval-gate".
#    (CLI: gh api repos/:owner/:repo/branches/main/protection ... — or use the web UI.)

# 3) bad-prompt PR -> CI RED
git checkout -b bad-prompt
#   edit versions/v2/config.json: break the JSON instruction, e.g.
#   "system_prompt": "Answer the ticket helpfully in a friendly paragraph."   # no JSON!
git commit -am "intentionally break JSON contract" && git push -u origin bad-prompt
gh pr create --fill    # the is-json / javascript assertions fail -> promptfoo exits non-zero -> check RED -> merge blocked
```

**Expected result.** The good PR shows a green `eval-gate` check; the bad PR shows red (the `is-json` and required-keys assertions fail because the prompt no longer asks for JSON), and with branch protection on, **merge is blocked**.

**Verify.** DoD gate: CI eval gate is a **required** check; a deliberately-broken prompt PR fails CI (red), a good one passes (green) — both demonstrated. Screenshot/link both PR check runs in the README.

**Troubleshoot.**
- *CI green even on the bad prompt* — your assertions were too loose (e.g., no `is-json`) or the model still emitted JSON by habit. Tighten: require `is-json` + all keys + enum; the broken prompt must remove the JSON instruction so structure genuinely fails.
- *`gh: not authenticated`* — `gh auth login` once. No `gh`? Push the branch and open the PR in the web UI.
- *Rubric flakes red/green run-to-run* — that's non-determinism; set `temperature: 0`, lower the rubric `threshold` slightly, and keep structural asserts (is-json/keys) as the hard gate. Never gate on exact string equality.
- *Secrets / cost in CI* — for a fully free run, use `is-json` + `javascript` asserts only (no `llm-rubric`), which need no API key. Add the rubric later if you have a key.

---

### Step 5 — Rollback + SLO alert

**What.** Write `scripts/rollback.sh` that flips `ACTIVE_VERSION` back to the previous value in **one command** and confirms via a health request, plus `slo/alert.py` that reads recent latencies and exits non-zero / alerts if **p95 TTFT** breaches your target. Document the full flow in the README.

**Why.** Lecture 16: a rollback is only "instant" if it's a **flag flip / config change**, not a redeploy (a 10-minute build is disqualified). Lecture 13: alert on the **SLO percentile** (p95), because p50 hides the tail pain users actually feel. The SLO must fire **before** users complain.

**Do it — rollback.** Store the active version in a file the router reads on each request (or use env + a supervisor). Simplest git-native approach: a `gateway/active.txt` the router loads per request, flipped by the script.

Point the router at the file (small change to `load` of ACTIVE):
```python
# in router.py, replace the ACTIVE constant with a live read:
ACTIVE_FILE = pathlib.Path(__file__).with_name("active.txt")
def active_version() -> str:
    return ACTIVE_FILE.read_text().strip() if ACTIVE_FILE.exists() else "v1"
# ...use active_version() wherever ACTIVE was used, and in /health.
```
`scripts/rollback.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
GW="${GW_URL:-http://localhost:9000}"
ACTIVE_FILE="gateway/active.txt"
PREV="${1:?usage: rollback.sh <previous_version_id>   e.g. v1}"

echo "$PREV" > "$ACTIVE_FILE"                       # the flag flip (no redeploy, no build)
sleep 1
echo "$(curl -s "$GW/health")"                      # confirm the switch
NOW=$(curl -s "$GW/health" | python -c "import sys,json;print(json.load(sys.stdin)['active_version'])")
[ "$NOW" = "$PREV" ] && echo "ROLLBACK OK -> $PREV" || { echo "ROLLBACK FAILED (active=$NOW)"; exit 1; }
```
Run it and time it:
```bash
echo v2 > gateway/active.txt          # pretend v2 was promoted
time ./scripts/rollback.sh v1         # flips to v1, confirms via /health -> well under 30s
```

**Do it — SLO alert.** `slo/alert.py` (reads latencies from the shadow log or a metrics dump; here from the router's served latencies — extend the router to also append served latencies, or reuse `latency_active_ms` from the shadow log):
```python
import json, sys, pathlib, statistics

TARGET_P95_MS = float(sys.argv[1]) if len(sys.argv) > 1 else 800.0   # p95 TTFT/latency SLO
LOG = pathlib.Path("gateway/shadow_log.jsonl")

lat = [json.loads(l)["latency_active_ms"]
       for l in LOG.read_text().splitlines() if l.strip()]
if not lat:
    print("no samples"); sys.exit(0)

lat.sort()
p95 = lat[min(int(0.95 * len(lat)), len(lat) - 1)]
print(f"n={len(lat)} p50={statistics.median(lat):.0f}ms p95={p95:.0f}ms target={TARGET_P95_MS:.0f}ms")

if p95 > TARGET_P95_MS:
    print(f"ALERT: p95 {p95:.0f}ms breaches SLO {TARGET_P95_MS:.0f}ms")   # -> Slack/PagerDuty in prod
    sys.exit(1)                                                            # non-zero => CI/monitor fires
print("SLO OK")
```
Prove the alert fires on a simulated breach:
```bash
./slo/alert.py 100000    # huge target -> OK (exit 0)
./slo/alert.py 1         # 1ms target -> guaranteed breach -> ALERT + exit 1
echo "exit=$?"
```
Write the target and the flow into `slo/` and the README, e.g. **"SLO: p95 TTFT < 800 ms, p95 TPOT < 50 ms; alert fires on breach."**

**Expected result.** `rollback.sh v1` flips the active version and `/health` confirms `active_version: v1` in well under 30 seconds, no rebuild. `alert.py` prints p50/p95 and exits non-zero (with an `ALERT` line) when p95 exceeds the target.

**Verify.** DoD gates: `rollback.sh` flips active version and a follow-up request confirms the old behavior in under ~30s; `slo/` documents a TTFT/TPOT target and an alert that fires when p95 breaches it (with spec-decode before/after if on GPU).

**Troubleshoot.**
- *Rollback "works" but the server still serves the new version* — you baked ACTIVE into an env var read once at startup. Read it live (per request) from `active.txt`, or restart is required (which fails the "instant" bar). Live-read is the point.
- *Alert never fires* — the target is above your real p95, or the log is empty. Drive traffic first (Step 3), then set the target to `1` to force a breach and confirm exit code 1.
- *p95 with tiny n is noisy* — collect ≥100 samples; document n alongside the percentile so the number is honest.

---

## Putting it together — a short end-to-end run

The full **PR → eval gate → merge → shadow 10% → canary → rollout → rollback** loop, top to bottom:
```bash
# 0. edge model up (Step 1)
ollama serve &                                   # if not already running
./edge/ollama_demo.sh                            # -> JSON "hi"; Week-1 client works unchanged

# 1. propose a change: edit versions/v2/config.json, open a PR
git checkout -b tune-prompt && git commit -am "v2: tighten JSON contract"
gh pr create --fill                              # CI eval-gate runs (Step 4)

# 2. eval gate: bad prompt -> RED (blocked); good prompt -> GREEN (mergeable)
#    (demonstrate both PRs once; required check enforces the gate)

# 3. merge -> deploy behind the flag; shadow 10% (Step 3)
BASE_URL="http://localhost:11434/v1" ACTIVE_VERSION=v1 SHADOW_VERSION=v2 SHADOW_PCT=0.10 \
  uv run uvicorn gateway.router:app --port 9000 &
for i in $(seq 1 50); do curl -s -X POST localhost:9000/extract \
  -H "x-tenant-id: tenant-$((i%2))" -d '{"text":"cannot pay, urgent"}' >/dev/null; done
python slo/alert.py 800                           # SLO check on served latencies

# 4. compare paired outputs, then CANARY: promote v2 to ACTIVE for real users
echo v2 > gateway/active.txt                       # flag flip (canary/rollout)

# 5. regret it -> INSTANT ROLLBACK (Step 5)
./scripts/rollback.sh v1                           # < 30s, /health confirms v1
```

**Fold it into the milestone.** Assemble one `phase10-llmops/` repo that ties the phase together (spine's suggested layout): `serving/` (Week 1 vLLM + LoRA), `economics/` (Week 2 VRAM estimator, k6/locust, break-even chart), and `release/` (this week's `gateway/router.py`, `evals/promptfoo.yaml`, `.github/workflows/eval-gate.yml`, `scripts/rollback.sh`, `slo/alert.py`), with a top-level `README.md` documenting architecture, the economics conclusion (crossover volume), and the release-flow diagram.

---

## Definition of Done — verifiable checks

**Week 3 lab (restated from the spine — each must pass, not "basically work"):**
- [ ] **Ollama endpoint answers via the *unchanged* OpenAI client** (only `base_url` changed). *(Verify: Week-1 `smoke_test.py` passes against `:11434` with just env-var swaps.)*
- [ ] **`router.py` serves `ACTIVE_VERSION` and logs paired active+shadow outputs for ~`SHADOW_PCT` of requests** to `shadow_log.jsonl`, each tagged with `tenant_id`. *(Verify: `wc -l shadow_log.jsonl` ≈ 10% of requests; served `version` always ACTIVE; a shadow error does not 500 the response.)*
- [ ] **CI eval gate is a *required* check; a deliberately-broken prompt PR fails CI (red), a good one passes (green)** — both demonstrated. *(Verify: two PR links, one red one green; branch protection lists `eval-gate`.)*
- [ ] **`rollback.sh` flips active version and a follow-up request confirms the old behavior in under ~30 seconds.** *(Verify: `time ./scripts/rollback.sh v1` + `/health` shows `active_version: v1`, no rebuild.)*
- [ ] **`slo/` documents a TTFT/TPOT target and an alert that fires when p95 breaches it** (spec-decoding before/after numbers if on GPU). *(Verify: `alert.py 1` exits non-zero with an ALERT line; `spec_decode.md` has before/after TPOT or a mechanism note.)*
- [ ] **README diagrams the shadow → canary → rollout → rollback flow.**

**Phase milestone acceptance (spine — the whole phase in one `phase10-llmops/` repo):**
- [ ] **vLLM endpoint passes an OpenAI-client smoke test and serves ≥3 LoRAs from one base** (verified by `nvidia-smi` showing single base weights). *(Week 1.)*
- [ ] **Load-test results table: tokens/sec, cost/1M, TTFT+TPOT at p50/p95/p99 for concurrency 1/8/64.** *(Week 2.)*
- [ ] **Break-even chart with a marked crossover volume and a written deploy recommendation.** *(Week 2.)*
- [ ] **CI eval gate is required; a bad-prompt PR fails and a good one passes (both demonstrated).** *(This lab, Step 4.)*
- [ ] **Shadow logging produces paired active/shadow outputs tagged by tenant for ≥10% of requests, with zero user-facing impact.** *(This lab, Step 3.)*
- [ ] **`rollback.sh` reverts the active version in a single command; a health request confirms.** *(This lab, Step 5.)*
- [ ] **SLO alert fires on a simulated p95 TTFT breach.** *(This lab, Step 5.)*
- [ ] **README documents architecture, the economics conclusion, and the release-flow diagram.**

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `connection refused` on `:11434` | Ollama service not running | `ollama serve` (macOS/Linux) / start the Windows service; `ollama list` to confirm the model. |
| WebLLM: "WebGPU not available" | non-WebGPU browser / disabled | Use Chrome/Edge; check `chrome://gpu`. Ollama is the required edge path; WebLLM is optional. |
| Shadow call blocks the user response | shadow was `await`ed | Fire-and-forget with `asyncio.create_task`; never let the user wait on the shadow. |
| Shadow error 500s the real response | exception escaped the background task | Wrap the shadow in try/except and log `<shadow-error>`; the served path must be isolated. |
| Shadow ratio far from `SHADOW_PCT` | sample size too small | Drive ≥500 requests, or temporarily set `SHADOW_PCT=1.0` to verify pairing. |
| CI green on a bad prompt | assertions too loose | Require `is-json` + all keys + enum as hard gates; the broken prompt must drop the JSON instruction. |
| Eval gate flaps red/green | non-determinism / rubric variance | `temperature: 0`, keep structural asserts as the hard gate, gate rubric on a threshold not exact match. |
| CI needs a paid API key | using `llm-rubric` / hosted model | For a free run use only `is-json` + `javascript` asserts (no key); add the rubric when you have a key. |
| Rollback "works" but server serves new version | ACTIVE read once at startup | Read the active version live per request from `active.txt`; a redeploy-to-rollback is disqualified. |
| SLO alert never fires | target above real p95 / empty log | Drive traffic first; set target to `1` to force a breach and confirm exit code 1. |
| Spec-decode flag rejected | vLLM API changed between versions | Match `vllm serve --help`; recent builds use `--speculative-config` JSON. Note the version. |
| Spec-decode: no TPOT win | draft too big / low acceptance / high batch | Use a 0.5B draft for a 7B; lower `--max-num-seqs` (spec decode helps most at low concurrency). |

---

## Stretch goals (optional)

- **Real canary routing (not just a flag flip).** Extend the router to serve `CANARY_VERSION` to `CANARY_PCT` of *real* users (by hashing `tenant_id` for stickiness) while shadowing the rest — the full shadow → canary → rollout ladder in one service.
- **CAG vs RAG micro-experiment (lecture 12).** Preload a small, static knowledge blob into the system prompt (prompt-cache / CAG) vs retrieving it per request (RAG), and compare TTFT + cost on your gateway. Decide small+static → CAG, large+dynamic → RAG.
- **Reasoning-budget gate per route (lecture 11).** Add a route that turns on a reasoning/thinking budget only for genuinely multi-step tasks and measure the extra output tokens + latency; confirm it's waste for the extraction task.
- **Wire the alert to Slack/GitHub.** Make `alert.py`'s breach path POST to a Slack incoming webhook, or run it as a scheduled GitHub Action that opens an issue on breach — a real "fires before users notice" monitor.
- **Structured-output enforcement.** If serving via vLLM, use guided decoding / `response_format` JSON schema so `is-json` rate is 100% by construction, and re-run the eval gate — connect it to the Week-1 stretch goal.
- **Prompt-cache the shared system prompt.** Confirm your edge/server reuses the KV for the identical version system prompt across shadow+active calls (ties back to Week-1 prefix caching) and note the TTFT effect.
