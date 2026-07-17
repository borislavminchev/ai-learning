# Week 3 Lab: The Serving Plane — an LLM Gateway that Survives Outages, Abuse, and Bad Prompts

> You take the working retrieval + agent core from Weeks 1-2 and put it behind a **single serving plane**: one LLM gateway that does provider fallback, a cheap-model-first cascade, exact + semantic caching, and — the graded part — **per-tenant rate limits + spend caps with an instant kill-switch** so one abusive tenant can't starve the others. You wrap it in a **streaming Next.js UI** (Vercel AI SDK), optionally stand up **self-hosted vLLM with multi-LoRA** and write the cost break-even, and finish with a **CI/CD pipeline that gates prompt/model changes on evals** and ships via **shadow → canary → feature-flag with instant rollback**. By the end you can *demonstrate* three claims, not assert them: (1) an abusive tenant can't degrade others; (2) a provider outage silently degrades to cached/cheaper responses; (3) a regression-y prompt is blocked by CI.
>
> Read these capstone lectures first — this lab is their hands-on counterpart:
> - [09 — The Gateway as Single Egress](../lectures/09-the-gateway-as-single-egress.md) — one door for every model call; why the app imports zero vendor SDKs; why streaming goes *through* the gateway.
> - [10 — Fallback, Cascade & Caching Decisions](../lectures/10-fallback-cascade-and-caching-decisions.md) — fallback (reliability) vs cascade (cost); exact vs semantic cache; the tenant-isolation iron rule.
> - [11 — Multi-Tenant Fairness, Quotas & Kill-Switch](../lectures/11-multi-tenant-fairness-quotas-and-kill-switch.md) — RPM/TPM + USD caps + kill-switch keyed to a per-tenant virtual key; the measured fairness proof.
> - [12 — Safe Release: Eval-Gated CI/CD and Rollout](../lectures/12-safe-release-eval-gated-cicd-and-rollout.md) — eval gate that reds a bad PR; shadow → canary → flag; rollback = flag flip, not redeploy.

**Est. time:** ~9.5 hrs · **You will need:** Docker Desktop (with `docker compose`), Python 3.11+, Node 18+ (for Next.js), `curl`, and a code editor. **Free/local path (zero paid API):** run the **LiteLLM proxy** in Docker as the gateway, **Redis** in Docker for caching + rate-limits, **Postgres** in Docker for the virtual-key store, and **Ollama** (`llama3.1`) as a local provider so both your `cheap` and `strong` tiers can be fully offline. Paid keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are optional and only make the cascade/fallback more realistic. The vLLM stretch needs a GPU (or Modal/Colab free tier).

---

## Before you start (setup)

You are extending **the same `capstone/` repo** from Weeks 1-2. Everything here adds a `gateway/`, `router/`, `web/`, `evals/`, and `cicd/` alongside your existing code.

**1. Confirm Docker is running and you have the core CLIs.**
```bash
docker --version && docker compose version
python --version        # 3.11+
node --version          # 18+  (for the Next.js UI in Step 4)
```
> **Windows + Git-Bash note:** run these in **Git-Bash**, but Docker Desktop must be started from Windows first (WSL2 backend enabled). Inside Git-Bash, `$PWD` works but Docker sometimes mangles Windows paths in `-v` mounts — prefix bind mounts with a leading slash trick (`//c/Users/...`) or use `docker compose` with a relative `./` path from the compose file's directory (used throughout this lab, so you avoid the issue). `host.docker.internal` resolves from inside containers on Docker Desktop for Windows/Mac out of the box.

**2. (Free path) Install and warm up Ollama** so the lab needs no paid key.
```bash
# install from https://ollama.com/download  (Windows installer or `winget install Ollama.Ollama`)
ollama pull llama3.1        # ~4.7 GB; this is your local cheap AND fallback provider
ollama pull llama3.2:1b     # tiny + fast; handy as an even-cheaper cascade tier-1
ollama run llama3.1 "say hi"   # smoke test the local server on :11434
```

**3. Create the repo layout for this week.**
```bash
cd capstone
mkdir -p gateway router web evals cicd/.github/workflows serving tests
```
Target layout (add to your existing capstone repo):
```
capstone/
  gateway/
    litellm-config.yaml       # models, fallbacks, cache, router settings
    docker-compose.yaml       # litellm + redis + postgres (virtual-key store)
    .env                      # OPENAI_API_KEY / ANTHROPIC_API_KEY (optional), MASTER_KEY
  router/
    cascade.py                # cheap-first escalation w/ confidence check
  web/                        # Next.js app (Vercel AI SDK)  — created in Step 4
  evals/
    suite.py                  # reuses your eval harness / promptfoo
    thresholds.yaml
  cicd/
    .github/workflows/eval-gate.yml
    rollout.md                # shadow -> canary -> flag runbook
    flags.json                # active model/prompt version + flag state
  serving/
    vllm.md                   # optional: vLLM + multi-LoRA + break-even sheet
  tests/
    test_fairness.py          # abusive tenant can't starve others
    test_degradation.py       # provider outage -> cached/cheaper
```

**4. Create `gateway/.env`.** The free path leaves the API keys blank and relies on Ollama.
```bash
# gateway/.env
LITELLM_MASTER_KEY=sk-master-CHANGEME
DATABASE_URL=postgresql://postgres:pw@db:5432/postgres
# optional — only if you have them; the lab runs fully without these:
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```
> **Never commit real keys.** Add `gateway/.env` to `.gitignore` now. The whole point of the gateway is that provider keys live *only* here, never in the app.

**5. Install Python deps for the router/tests** (in your existing capstone venv).
```bash
source .venv/bin/activate            # Windows Git-Bash: source .venv/Scripts/activate
pip install -U "openai>=1.40" httpx pytest pyyaml redis
```

---

## Step-by-step

### Step 1 — Stand up the gateway (LiteLLM + Redis + Postgres)

**What.** Run a self-hosted LiteLLM proxy that speaks the OpenAI API on `:4000`, backed by Redis (cache + rate-limit counters) and Postgres (virtual-key/budget store).

**Why.** If every service calls provider SDKs directly, then fallback, caching, rate limits, spend caps, and key rotation are scattered across N call sites and impossible to enforce. One gateway is a single egress point — "one throat to choke" (lecture 09). The DB is what unlocks per-tenant virtual keys and budgets (lecture 11); Redis is what makes caching and token-bucket limits fast.

**Do it.** Create the gateway config. Note two `cheap` and two `strong` entries — LiteLLM load-balances/falls-back across same-named models.
```yaml
# gateway/litellm-config.yaml
model_list:
  # --- FREE/LOCAL path: Ollama serves both tiers, zero paid API ---
  - model_name: cheap
    litellm_params:
      model: ollama/llama3.2:1b
      api_base: http://host.docker.internal:11434
  - model_name: strong
    litellm_params:
      model: ollama/llama3.1
      api_base: http://host.docker.internal:11434

  # --- OPTIONAL cloud entries (only used if the keys are set) ---
  - model_name: cheap
    litellm_params: {model: openai/gpt-4o-mini, api_key: os.environ/OPENAI_API_KEY}
  - model_name: strong
    litellm_params: {model: anthropic/claude-sonnet-4-5, api_key: os.environ/ANTHROPIC_API_KEY}
  - model_name: strong          # cross-provider fallback for the strong tier
    litellm_params: {model: openai/gpt-4o, api_key: os.environ/OPENAI_API_KEY}

router_settings:
  routing_strategy: simple-shuffle
  fallbacks: [{"cheap": ["strong"]}, {"strong": ["cheap"]}]   # provider/tier fallback
  num_retries: 2
  timeout: 30

litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: redis
    port: 6379
    supported_call_types: ["acompletion", "completion"]
  # semantic cache is enabled in Step 3

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL        # enables virtual keys + budgets
```
```yaml
# gateway/docker-compose.yaml
services:
  redis:
    image: redis:7
    ports: ["6379:6379"]
  db:
    image: postgres:16
    environment: {POSTGRES_PASSWORD: pw}
    ports: ["5432:5432"]
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    command: ["--config", "/app/config.yaml", "--port", "4000"]
    ports: ["4000:4000"]
    volumes: ["./litellm-config.yaml:/app/config.yaml"]
    env_file: [.env]
    extra_hosts: ["host.docker.internal:host-gateway"]   # so the container reaches Ollama on the host
    depends_on: [redis, db]
```
```bash
cd gateway
docker compose up -d
docker compose logs -f litellm   # wait for "Uvicorn running on http://0.0.0.0:4000"
```

**Expected result.** Three containers running; LiteLLM logs show it connected to the Postgres DB and created its tables, and Redis is reachable.

**Verify.**
```bash
# smoke test through the gateway using the master key
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-master-CHANGEME" \
  -H "Content-Type: application/json" \
  -d '{"model":"cheap","messages":[{"role":"user","content":"reply with the single word: pong"}]}' | python -m json.tool
```
You should get a normal OpenAI-shaped response with `choices[0].message.content`.

**Troubleshoot.**
- *`Connection refused` to Ollama:* the container can't reach the host. Confirm `extra_hosts` is present, `ollama serve` is running, and test from inside the container: `docker compose exec litellm curl -s http://host.docker.internal:11434/api/tags`.
- *LiteLLM won't start / DB errors:* wait — Postgres needs a few seconds after `up`. Re-check `docker compose logs db`. If `database_url` is unset, virtual keys silently won't work (Step 2 will fail).
- *Windows path/mount issues:* run `docker compose` from *inside* `gateway/` so the `./litellm-config.yaml` relative mount resolves.

---

### Step 2 — Per-tenant virtual keys with budgets + limits + kill-switch

**What.** Mint one **virtual key per tenant** via the admin API, each carrying its own RPM/TPM ceiling and daily USD budget. This is the fairness control (lecture 11).

**Why.** A shared egress is a shared blast radius. Isolation by default, fairness by quota: each tenant hits *its own* ceiling. You need all three controls — RPM/TPM stops request floods, the USD cap stops a few huge-context calls from draining the budget, and the kill-switch handles "it's happening right now."

**Do it.**
```bash
export MK="sk-master-CHANGEME"

# Tenant "acme": tight limits so we can trip it on purpose in the fairness test
curl -s http://localhost:4000/key/generate -H "Authorization: Bearer $MK" \
  -H "Content-Type: application/json" -d '{
    "key_alias":"acme","rpm_limit":5,"tpm_limit":20000,
    "max_budget":0.50,"budget_duration":"1d","metadata":{"tenant":"acme"}}'

# Tenant "globex": its own independent limits
curl -s http://localhost:4000/key/generate -H "Authorization: Bearer $MK" \
  -H "Content-Type: application/json" -d '{
    "key_alias":"globex","rpm_limit":60,"tpm_limit":200000,
    "max_budget":5.00,"budget_duration":"1d","metadata":{"tenant":"globex"}}'
```
Each call returns `{"key":"sk-...","key_alias":"acme",...}`. Save both keys (e.g. into `gateway/.tenant-keys` — gitignored). The app selects the tenant's key per request from the auth/session; **the app holds no provider keys.**

Kill-switch (instant disable, no redeploy):
```bash
curl -s http://localhost:4000/key/block   -H "Authorization: Bearer $MK" -d '{"key":"<acme-key>"}'
curl -s http://localhost:4000/key/unblock -H "Authorization: Bearer $MK" -d '{"key":"<acme-key>"}'
```

**Expected result.** Two keys with distinct aliases; a blocked key immediately returns 401/403 on use, an unblocked one works again.

**Verify.**
```bash
ACME="<acme-key>"
# a call with the tenant key succeeds
curl -s http://localhost:4000/v1/chat/completions -H "Authorization: Bearer $ACME" \
  -d '{"model":"cheap","messages":[{"role":"user","content":"hi"}]}' -H "Content-Type: application/json" \
  -o /dev/null -w "status=%{http_code}\n"
# inspect the key's live budget/limits
curl -s http://localhost:4000/key/info -H "Authorization: Bearer $MK" -G --data-urlencode "key=$ACME" | python -m json.tool
```

**Troubleshoot.**
- *`/key/generate` returns "master key not set" or 401:* `LITELLM_MASTER_KEY` env not picked up, or `database_url` missing (virtual keys require the DB). Check `general_settings` in the config.
- *Budget never seems to enforce with Ollama:* local models cost `$0.00`, so the **USD budget won't trip** — the RPM limit is what you'll demonstrate on the free path. Set `rpm_limit` low (5) to trip it fast. Use a paid model entry if you want to demo the spend cap specifically.

---

### Step 3 — Cheap-first cascade with confidence escalation, and prove caching

**What.** Add app-layer cascade logic: run the `cheap` model first, escalate to `strong` only when the answer is low-confidence. Then demonstrate exact + semantic caching.

**Why.** Fallback (Step 1, gateway config) is *reliability* — same task, different provider on error. **Cascade is *cost*** — most enterprise traffic (FAQ, classification, short RAG answers) is over-served by a frontier model; a tuned cascade cuts spend 50-80% (lecture 10). Caching kills repeat cost/latency, but semantic cache carries a false-hit risk, so it must be high-threshold, per-tenant, TTL'd, and never applied to tool/action calls.

**Do it.** The cascade is app logic; the gateway does the fallback underneath.
```python
# router/cascade.py
import re
from openai import OpenAI

def make_client(tenant_key: str) -> OpenAI:
    # app talks ONLY to the gateway; it never imports a vendor SDK
    return OpenAI(base_url="http://localhost:4000/v1", api_key=tenant_key)

_HEDGE = re.compile(r"\b(i'm not sure|i am not sure|cannot determine|i don't know|as an ai)\b", re.I)

def confident(text: str) -> bool:
    """Cheapest useful signal: non-empty, no hedge/refusal. Upgrade to JSON-schema
    validation or a tiny LLM-judge call when the task warrants it."""
    return bool(text) and not _HEDGE.search(text)

def answer(client: OpenAI, messages: list[dict], force_strong: bool = False) -> tuple[str, str]:
    if not force_strong:
        r = client.chat.completions.create(
            model="cheap", messages=messages,
            extra_body={"metadata": {"stage": "cascade-1"}})
        text = r.choices[0].message.content or ""
        if confident(text):
            return text, "cheap"
    r = client.chat.completions.create(
        model="strong", messages=messages,
        extra_body={"metadata": {"stage": "cascade-2"}})
    return (r.choices[0].message.content or ""), "strong"
```
Escalation signals, in order of effort: (1) JSON-schema validation of a structured output, (2) a refusal/hedge regex (above), (3) a tiny LLM-judge call on the cheap model. Log which tier served each request so you can report the split.

**Prove caching.** Fire the same prompt twice, then a paraphrase.
```bash
ACME="<acme-key>"
BODY='{"model":"cheap","messages":[{"role":"user","content":"What is the capital of France?"}]}'
# 1st call: cache MISS (note the latency)
curl -s -o /dev/null -w "call1  %{time_total}s\n" http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $ACME" -H "Content-Type: application/json" -d "$BODY"
# 2nd identical call: EXACT cache HIT — near-instant, cost-free
curl -s -o /dev/null -w "call2  %{time_total}s\n" http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $ACME" -H "Content-Type: application/json" -d "$BODY"
```
For **semantic cache**, add the redis-semantic backend and a high threshold to the config, then rebuild:
```yaml
# in litellm-config.yaml -> litellm_settings.cache_params
    type: redis-semantic
    similarity_threshold: 0.92        # HIGH — a low threshold serves the wrong answer
    redis_semantic_cache_embedding_model: cheap
```
```bash
cd gateway && docker compose up -d --force-recreate litellm
```
Now a paraphrase ("What's France's capital city?") should hit the cache **only above 0.92 similarity**, and (because keys are per-tenant) never cross from `acme` to `globex`.

**Expected result.** `call2` is dramatically faster than `call1` (exact hit). The response JSON carries a cache indicator; LiteLLM sets `x-litellm-cache-key` and includes cache info in the response.

**Verify.**
```bash
# the response header proves a hit vs miss
curl -s -D - -o /dev/null http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $ACME" -H "Content-Type: application/json" -d "$BODY" | grep -i "cache"
```

**Troubleshoot.**
- *No speedup on call2:* check `litellm_settings.cache: true` and that Redis is up (`docker compose exec redis redis-cli ping` → `PONG`). Streaming responses and differing params (temperature/seed) change the cache key.
- *Semantic cache errors:* `redis-semantic` needs the embedding model reachable; with Ollama, point `redis_semantic_cache_embedding_model` at a model you actually serve. If it's flaky locally, keep **exact cache only** — the lecture's advice: when unsure, exact-cache only.
- *Cross-tenant worry:* keys are scoped per virtual key, so cache entries don't leak across tenants — but do verify by querying the paraphrase as `globex` and confirming it's a MISS the first time.

---

### Step 4 — Streaming Next.js UI (Vercel AI SDK) through the gateway

**What.** A minimal chat UI that streams tokens from the gateway using the Vercel AI SDK.

**Why.** TTFT (time-to-first-token) dominates *perceived* latency. Stream **through** the gateway so caching, limits, and budgets still apply — streaming direct from a provider SDK in the Next.js route silently bypasses every control (lecture 09).

**Do it.**
```bash
cd capstone
npx create-next-app@latest web --ts --app --no-eslint --no-tailwind --no-src-dir --import-alias "@/*"
cd web && npm i ai @ai-sdk/openai
```
```ts
// web/app/api/chat/route.ts
import { streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

// point the SDK at the GATEWAY, using the tenant's virtual key (from session/env)
const gw = createOpenAI({
  baseURL: "http://localhost:4000/v1",
  apiKey: process.env.TENANT_KEY!,      // set in web/.env.local
});

export async function POST(req: Request) {
  const { messages } = await req.json();
  const result = streamText({ model: gw("cheap"), messages });   // streams THROUGH the gateway
  return result.toDataStreamResponse();
}
```
```tsx
// web/app/page.tsx  (client)
"use client";
import { useChat } from "ai/react";

export default function Page() {
  const { messages, input, handleInputChange, handleSubmit } = useChat();
  return (
    <form onSubmit={handleSubmit} style={{ maxWidth: 640, margin: "2rem auto" }}>
      {messages.map((m) => (
        <p key={m.id}><b>{m.role}:</b> {m.content}</p>
      ))}
      <input value={input} onChange={handleInputChange} placeholder="Ask..." style={{ width: "100%" }} />
    </form>
  );
}
```
```bash
# web/.env.local
echo "TENANT_KEY=<acme-key>" > web/.env.local
npm run dev     # http://localhost:3000
```

**Expected result.** Tokens stream live into the page word-by-word; every request is subject to the gateway's cache/limits/budget.

**Verify.** With `acme` blocked (Step 2 kill-switch), the UI should error/degrade; unblock and it streams again. Watch LiteLLM logs to confirm the request arrived at the gateway (not a direct provider hit).

**Troubleshoot.**
- *CORS / connection refused from the route:* Next.js server routes run server-side, so `localhost:4000` is correct from the Node process; if you deployed the UI in a container, use `host.docker.internal`.
- *`useChat` import error on newer SDK:* the import path is `ai/react` in AI SDK v3; if you installed v4+, follow the current `sdk.vercel.ai` docs for the `useChat` location. Keep the version pinned to what you installed.
- *No streaming, whole answer appears at once:* ensure the route returns `toDataStreamResponse()` and the model actually streams (Ollama does).

---

### Step 5 — CI eval gate + shadow → canary → flag rollout

**What.** A GitHub Actions job that runs your eval suite on the *candidate* prompt/model through the gateway and **fails the PR** if a metric drops below threshold. Plus a rollout runbook driven by `flags.json`.

**Why.** Prompts and model versions are code that regresses silently and ships green (lecture 12). The gate turns a bad prompt into a red PR; staged rollout de-risks a good one; rollback is a flag flip, not a redeploy.

**Do it.** A minimal, deterministic eval suite (reuse your Week-4/phase eval harness or promptfoo; this is a self-contained stand-in):
```python
# evals/suite.py
import json, sys, yaml
from openai import OpenAI

GATEWAY = "http://localhost:4000/v1"

def run(model: str, prompt_version: str, thresholds_path: str) -> int:
    gw = OpenAI(base_url=GATEWAY, api_key="sk-master-CHANGEME")   # CI can use master or a CI key
    golden = [json.loads(l) for l in open("evals/golden.jsonl")]
    thr = yaml.safe_load(open(thresholds_path))
    hits = 0
    for row in golden:
        r = gw.chat.completions.create(
            model=model, temperature=0,                            # PIN temperature -> deterministic gate
            messages=[{"role": "system", "content": PROMPTS[prompt_version]},
                      {"role": "user", "content": row["q"]}])
        out = (r.choices[0].message.content or "").lower()
        hits += int(row["gold_substr"].lower() in out)
    score = hits / max(len(golden), 1)
    print(f"exact_match={score:.3f}  floor={thr['exact_match']}")
    return 0 if score >= thr["exact_match"] else 1     # non-zero exit -> PR blocked

PROMPTS = {
    "v7": "You are a precise assistant. Answer with the exact fact only.",
    "v8-bad": "You are a whimsical poet. Answer only in rhyming riddles.",   # intentional regression
}

if __name__ == "__main__":
    # usage: python evals/suite.py <model> <prompt_version> <thresholds.yaml>
    sys.exit(run(sys.argv[1], sys.argv[2], sys.argv[3]))
```
```yaml
# evals/thresholds.yaml
exact_match: 0.80        # set with a margin so a green gate is trustworthy
```
```jsonl
# evals/golden.jsonl  (grow this; >=20 rows in the real capstone)
{"q":"Capital of France?","gold_substr":"Paris"}
{"q":"Chemical symbol for gold?","gold_substr":"Au"}
```
```yaml
# cicd/.github/workflows/eval-gate.yml
name: eval-gate
on:
  pull_request:
    paths: ["prompts/**", "router/**", "cicd/flags.json", "gateway/litellm-config.yaml", "evals/**"]
jobs:
  eval:
    runs-on: ubuntu-latest
    services:
      redis: { image: redis:7, ports: ["6379:6379"] }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install openai pyyaml
      # In CI you point the gateway at a hosted/mock provider, or run litellm as a service.
      # exits non-zero if any metric < threshold  ->  PR blocked (a bad prompt can't merge)
      - run: python evals/suite.py cheap "${{ vars.PROMPT_VERSION || 'v7' }}" evals/thresholds.yaml
```
Rollout state + runbook:
```json
// cicd/flags.json
{"prompt_version": "v7", "candidate_version": "v8", "canary_pct": 0}
```
```markdown
<!-- cicd/rollout.md -->
1. SHADOW   — mirror N requests to the candidate model_name, log a side-by-side diff, serve the incumbent. Zero user impact.
2. CANARY   — set canary_pct: 5; the app routes 5% of real traffic to the candidate; watch metrics.
3. FLAG     — set canary_pct: 100; candidate is live, still behind the flag.
4. ROLLBACK — set prompt_version back to the previous value (flag flip). No redeploy.
The candidate lives as ANOTHER gateway model_name, so switching is a config/flag change, not a deploy.
```

**Expected result.** Running `python evals/suite.py cheap v7 evals/thresholds.yaml` exits `0` (pass); running it with `v8-bad` exits non-zero (gate red).

**Verify.**
```bash
python evals/suite.py cheap v7 evals/thresholds.yaml;      echo "good prompt exit=$?"   # 0
python evals/suite.py cheap v8-bad evals/thresholds.yaml;  echo "bad prompt exit=$?"    # 1
```
Then open a PR that flips `PROMPT_VERSION` to `v8-bad` (or edits the prompt) and confirm the Actions run goes **red** and blocks merge.

**Troubleshoot.**
- *Flaky gate:* pin `temperature=0`, use a large-enough golden set, and set thresholds with margin (lecture 12's pitfall). A coin-flip gate is one teams learn to ignore.
- *CI can't reach the gateway:* either run LiteLLM as a `services:` container in the workflow, or point `evals/suite.py` at a mock/hosted endpoint. Don't call a real paid provider from CI without a spend guard.

---

### Step 6 — Prove fairness (`tests/test_fairness.py`)

**What.** An automated test: hammer `acme` past its RPM, and prove `globex` stays green in parallel; then flip the kill-switch and prove instant isolation.

**Why.** This is the week's core DoD — measured, not eyeballed (lecture 11).

**Do it.**
```python
# tests/test_fairness.py
import concurrent.futures as cf
import httpx

GW = "http://localhost:4000/v1/chat/completions"
ACME  = "<acme-key>"    # rpm_limit=5
GLOBEX = "<globex-key>" # rpm_limit=60
BODY = {"model": "cheap", "messages": [{"role": "user", "content": "hi"}]}

def call(key: str) -> int:
    r = httpx.post(GW, headers={"Authorization": f"Bearer {key}"}, json=BODY, timeout=30)
    return r.status_code

def test_abusive_tenant_cannot_starve_others():
    # fire 30 concurrent requests as acme (way past its rpm=5)
    with cf.ThreadPoolExecutor(max_workers=30) as ex:
        acme_codes = list(ex.map(lambda _: call(ACME), range(30)))
    # meanwhile globex issues a modest burst
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        globex_codes = list(ex.map(lambda _: call(GLOBEX), range(5)))

    assert acme_codes.count(429) > 0, "acme should be rate-limited (429)"
    assert all(c == 200 for c in globex_codes), "globex must stay green while acme is throttled"

def test_kill_switch_is_instant():
    import httpx
    MK = "sk-master-CHANGEME"
    httpx.post("http://localhost:4000/key/block",
               headers={"Authorization": f"Bearer {MK}"}, json={"key": ACME})
    assert call(ACME) in (401, 403), "blocked key must be rejected immediately"
    assert call(GLOBEX) == 200, "globex unaffected by acme's kill-switch"
    httpx.post("http://localhost:4000/key/unblock",
               headers={"Authorization": f"Bearer {MK}"}, json={"key": ACME})
```

**Expected result.** `acme` receives `429`s once past its RPM; `globex` returns all `200`; blocking `acme` is instant and leaves `globex` untouched.

**Verify.** `pytest tests/test_fairness.py -q` is green.

**Troubleshoot.**
- *No 429s:* your `rpm_limit` is too high for the burst, or limits aren't enforced (DB not wired). Lower `acme` to `rpm_limit: 5` and fire 30 concurrent calls.
- *globex also gets 429s:* you're accidentally sharing a key, or hitting a *global* provider limit (Ollama concurrency). Give tenants clearly separate keys and keep bursts modest for `globex`.

---

### Step 7 — Prove degradation under provider outage (`tests/test_degradation.py`)

**What.** Force the primary provider to fail and prove requests still succeed via fallback provider or cache.

**Why.** A provider outage should be a shrug, not an incident (lecture 10). Fallback keeps you serving; cache serves repeats at ~zero cost/latency.

**Do it.** Simulate the outage by breaking the primary. Two easy approaches:
- **Cache path:** warm the cache with a query, then break the provider, then re-issue the *same* query — it's served from cache.
- **Fallback path:** configure the primary `cheap` entry to a dead `api_base`/bad key so LiteLLM's `fallbacks` route to `strong`.
```python
# tests/test_degradation.py
import httpx

GW = "http://localhost:4000/v1/chat/completions"
ACME = "<acme-key>"
Q = {"model": "cheap", "messages": [{"role": "user", "content": "Define idempotency in one line."}]}

def _post():
    return httpx.post(GW, headers={"Authorization": f"Bearer {ACME}"}, json=Q, timeout=60)

def test_repeat_query_served_from_cache_after_outage():
    # 1) warm the cache
    r1 = _post(); assert r1.status_code == 200
    # 2) SIMULATE OUTAGE: stop the local Ollama provider (or point the primary at a dead base)
    #    -- do this manually before running, or via a fixture that kills `ollama serve`
    # 3) the identical query is an EXACT cache hit -> still 200, near-zero latency
    r2 = _post()
    assert r2.status_code == 200
    assert r2.elapsed.total_seconds() < 0.5   # cache hit is ~instant

def test_fallback_provider_on_primary_failure():
    # with the primary `cheap` entry broken, LiteLLM falls back per router_settings.fallbacks
    r = _post()
    assert r.status_code == 200, "fallback provider should keep the request alive"
```
To exercise the **fallback** leg deterministically, temporarily add a broken primary and rebuild:
```yaml
# put this ABOVE the good cheap entry so it's tried first, then falls back
  - model_name: cheap
    litellm_params: {model: openai/gpt-4o-mini, api_key: "sk-DELIBERATELY-BAD"}
```
```bash
cd gateway && docker compose up -d --force-recreate litellm
```

**Expected result.** Even with the primary dead, requests return `200` — repeats from cache (instant), novel ones via the fallback provider. LiteLLM logs show the fallback firing.

**Verify.** `pytest tests/test_degradation.py -q` green; grep LiteLLM logs for `Falling back` / retry lines. **Make fallback observable** — log the tier that served each request so an outage isn't silent (lecture 10 pitfall).

**Troubleshoot.**
- *Cache test flaky:* ensure the two requests are byte-identical (same params) and exact cache is on. If you enabled semantic-only, a repeat may still recompute embeddings.
- *Fallback doesn't fire:* `fallbacks` must reference model *names* that exist, and `num_retries`/`timeout` must allow the switch. A bad key surfaces as an auth error LiteLLM treats as fallback-eligible; a network timeout also works.

---

### Step 8 (optional stretch) — vLLM + multi-LoRA + break-even

**What.** Serve an open model on vLLM with several LoRA adapters behind the same gateway, and write the cost break-even vs API.

**Why.** Self-hosting only wins at *sustained high utilization* — a GPU costs the same idle or saturated. Multi-LoRA amortizes one base model across many per-tenant/per-task "models."

**Do it.** On a GPU box / Modal / Colab:
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora --lora-modules acme=/loras/acme legal=/loras/legal \
  --port 8000
```
Add it to the gateway as another provider and request an adapter by name:
```yaml
  - model_name: cheap
    litellm_params: {model: openai/acme, api_base: http://<gpu-host>:8000/v1, api_key: "none"}
```
Fill `serving/vllm.md` with the break-even sheet:
```
$/1M tokens = (GPU $/hr) / (throughput_tok_per_hr / 1e6)
```
Measure throughput under `vllm bench` (or a load test), record GPU $/hr, compute $/1M tokens including idle hours, and compare to API list price. State the **utilization %** where self-host wins (typically only above ~40-60% duty cycle).

**Expected result.** `serving/vllm.md` has a real measured number and a utilization threshold, plus one non-cost reason you'd still self-host (data residency, no per-token vendor lock, custom adapters).

**Verify.** Request `model: acme` through the gateway and confirm the adapter is applied (compare outputs to the base model on an adapter-specific prompt).

**Troubleshoot.** Multi-LoRA needs `--enable-lora` and adapters compatible with the base model's rank/target modules; OOM → reduce `--max-model-len` or `--gpu-memory-utilization`.

---

## Putting it together — a short end-to-end run

1. `cd gateway && docker compose up -d` — gateway + Redis + Postgres come up; Ollama is serving locally.
2. Mint `acme` (rpm 5, $0.50/day) and `globex` (rpm 60, $5/day) virtual keys (Step 2).
3. `cd web && npm run dev`, open `http://localhost:3000`, chat as `acme` — tokens **stream** through the gateway; the app holds no provider key.
4. Ask a repeat question — the second answer is an **exact-cache hit** (near-instant).
5. `pytest tests/test_fairness.py tests/test_degradation.py -q` — hammer `acme` and watch `429`s while `globex` stays `200`; kill `ollama serve` and confirm repeats still serve from cache and novel queries fall back.
6. Open a PR flipping the prompt to `v8-bad` — the **eval-gate** Action goes red and blocks merge; revert and it passes.
7. Walk `cicd/rollout.md`: shadow → canary (`canary_pct: 5`) → flag (`100`) → rollback = set `prompt_version` back. No redeploy.

You now have a serving plane where every model call goes through one door, one tenant can't starve another, an outage degrades gracefully, and a bad prompt can't ship.

---

## Definition of Done — restate the spine's Week 3 DoD as verifiable checks

- [ ] **100% of model calls go through the gateway.** `grep -rEn "from openai|import anthropic|new OpenAI\(|Anthropic\(" web/ router/ src/` finds **no** direct-provider client in a request path except the one pointed at `localhost:4000`. The app holds only tenant virtual keys; provider keys live only in `gateway/.env`.
- [ ] **Fairness proof (`tests/test_fairness.py` green).** `acme` hammered past its RPM/budget gets `429`/degraded while `globex` (issued in parallel) keeps returning `200` — measured, not eyeballed. Flipping the kill-switch on `acme` blocks it instantly; `globex` unaffected.
- [ ] **Degradation proof (`tests/test_degradation.py` green).** With the primary provider forced to fail, requests still succeed via **fallback provider or cache**; a repeat query is served from cache at ~zero cost/latency. Logs show fallback + cache hit.
- [ ] **Cascade works.** A trace/log shows an easy query served by `cheap` and a hard/low-confidence one escalating to `strong`; you can report the cheap-vs-strong split % over the eval set.
- [ ] **Caching measured.** A repeat query is an exact-cache hit (near-zero latency); the semantic cache hits a paraphrase **only above threshold (≥0.9)** and **never crosses tenants**.
- [ ] **Streaming UI.** The Next.js chat streams tokens through the gateway end-to-end (verified by the request appearing in LiteLLM logs).
- [ ] **CI eval gate blocks a bad prompt.** A PR that intentionally degrades a prompt turns CI **red** and blocks merge; a good change passes and rolls out **shadow → canary → flag** with a demonstrated one-command rollback (flag flip, no redeploy).
- [ ] **(Optional)** `serving/vllm.md` contains a real break-even number with measured throughput and the utilization threshold.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `curl` to `:4000` refused | LiteLLM not up / still booting | `docker compose logs litellm`; wait for "Uvicorn running". |
| Container can't reach Ollama | `host.docker.internal` unresolved | Add `extra_hosts: ["host.docker.internal:host-gateway"]`; confirm `ollama serve` is running. |
| `/key/generate` 401 or "master key not set" | `LITELLM_MASTER_KEY` or `database_url` missing | Set both in `general_settings` / `.env`; virtual keys need the Postgres DB. |
| No 429 under load | RPM too high / DB not wired | Lower `acme` to `rpm_limit: 5`; fire 30 concurrent calls; confirm DB store active. |
| Spend cap never trips | Local models cost $0 | USD budget won't fire on Ollama — demo RPM instead, or use a paid model entry. |
| Cache never hits | `cache: false`, Redis down, or params differ | `cache: true`; `redis-cli ping`; keep params identical; disable streaming for the exact-hit demo. |
| Semantic cache serves wrong answer | Threshold too low | Raise to ≥0.9-0.95; never semantic-cache tool/action calls; when unsure, exact-only. |
| Fallback never fires | `fallbacks` names don't exist / retries=0 | Reference real `model_name`s; set `num_retries: 2`, sane `timeout`. |
| UI shows whole answer at once | Route not streaming | Return `result.toDataStreamResponse()`; confirm model streams. |
| Eval gate is a coin-flip | High temp / tiny golden set | Pin `temperature=0`; grow golden set; set thresholds with margin. |
| Windows bind-mount errors | Git-Bash path munging | Run `docker compose` from inside `gateway/` so `./config.yaml` resolves; enable WSL2 backend. |

---

## Stretch goals (optional)

- **Portkey comparison.** Swap the OSS LiteLLM proxy for managed **Portkey** on one route and note what the hosted control plane buys you (guardrails marketplace, dashboards) vs the run-it-yourself cost — the concepts map 1:1.
- **Learned router.** Replace the hedge-regex `confident()` with a **RouteLLM**-style learned router (search `lm-sys/RouteLLM`) and report the accuracy/cost trade vs the naive cascade.
- **Semantic-cache adversarial test.** Add a test that fires two *similar-looking but differently-answered* prompts and asserts the cache does **not** serve one for the other below your threshold — quantify the false-hit boundary.
- **Real spend-cap demo.** Wire one paid model entry, set `max_budget: 0.02`, and prove a tenant hits `429`/degraded on the USD cap (distinct from the RPM limit).
- **Observability rollup.** Forward LiteLLM's request logs to your Week-4 OpenTelemetry/Phoenix stack so per-tenant cost, the cascade tier split, and cache-hit rate show up on one dashboard.
- **Blue-green gateway config.** Keep the candidate prompt/model as a second `model_name` and script the `flags.json` flip so canary → 100% → rollback is a single command with an audit line.
