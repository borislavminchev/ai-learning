# Week 3 Lab: Observability, Degradation Mode, Load Test, Capacity Math, and Three Mock Designs (Milestone)

> This week you finish the gateway you built in Weeks 1–2 and turn it into a thing you can *operate*: every request emits **one coherent trace** (model, tokens, cost, TTFT, cache layer, tier, tenant), a **metrics view** shows per-tenant cost/latency/quality, a **degradation ladder** keeps `/chat` answering during a total provider outage instead of 5xx-ing, a **load test** proves one abusive tenant can't starve another, and a **capacity sheet** turns QPS→tokens→quota→$ into arithmetic. Then you sit **three real 45-minute mock system designs** under a timer. This is the **Phase 9 milestone** — the artifact you screen-share in an interview. Everything runs free and local: Langfuse/Phoenix in Docker, `locust`/`k6` on your laptop, and Ollama as the always-available degradation floor.
>
> Read these lectures first — this guide assumes them:
> - [../lectures/13-evaluability-and-observability.md](../lectures/13-evaluability-and-observability.md) — the mandatory trace attribute set, one-span-per-step wiring, prompt/model versioning, feedback→trace-id flywheel
> - [../lectures/14-scalability-capacity-and-degradation.md](../lectures/14-scalability-capacity-and-degradation.md) — the QPS→tokens→quota→$ method, cache/cascade levers, the degradation ladder, "degrade AND alert"
> - [../lectures/15-build-vs-buy-and-privacy-boundaries.md](../lectures/15-build-vs-buy-and-privacy-boundaries.md) — the five-axis rubric, trust-boundary data-flow diagram, the "self-host thin gateway + buy observability" default
> - [../lectures/16-system-design-method.md](../lectures/16-system-design-method.md) — the seven-step, time-budgeted whiteboard loop you run under the timer
> - [../lectures/17-privacy-in-observability-pipelines.md](../lectures/17-privacy-in-observability-pipelines.md) — redact PII *before* the span/log export boundary; every sink is a derived store

**Est. time:** ~10 hrs (≈6 hrs build + observability/degradation/load, ≈2.25 hrs the three timed designs, ≈1.75 hrs capacity sheet + write-ups) · **You will need:** Python 3.10+, Docker Desktop (Redis + Postgres/pgvector + MinIO from Week 1), your Week 2 gateway repo, a terminal (Git-Bash on Windows), and:
- **Free/local path (default here):** [Ollama](https://ollama.com) running `llama3.1` (the degradation floor) + `nomic-embed-text` (embeddings), plus a local trace backend — **Langfuse** *or* **Arize Phoenix** in Docker — and **locust** (pure-Python, easiest on Windows) or **k6**.
- **Paid path (optional):** one provider key (OpenAI/Anthropic/Gemini) so a "real provider goes down" simulation is authentic. Under $1 for the whole lab.

You do **not** need paid observability. Structured-JSON logs are a valid backend — the trace *data model* is what matters, not the viewer (lecture 13).

---

## Before you start (setup)

**What:** Bring up a local trace backend, install the observability + load-test deps, and add the new module files and tables to your Week 1–2 repo.

**Why:** Everything this week is *additive* — spans wrap the gateway steps you already have; the degradation ladder reuses your Week 2 breaker + cache + Ollama; the load test hits your existing `/chat`. No new request-path logic is required, only instrumentation and orchestration.

**Do it:**

```bash
cd llm-gateway-lab
source .venv/Scripts/activate        # Windows (Git-Bash)
# source .venv/bin/activate          # macOS / Linux

# Observability + capacity + load deps
uv add "langfuse>=2.50" "opentelemetry-sdk>=1.27" "opentelemetry-exporter-otlp>=1.27" locust
# (Phoenix alternative to Langfuse: uv add arize-phoenix openinference-instrumentation)

docker compose up -d                 # redis, postgres(+pgvector), minio from Week 1
```

Add a **local Langfuse** to your `docker-compose.yml` (self-hosted, free) — append this service block, then `docker compose up -d langfuse`:

```yaml
  langfuse:
    image: langfuse/langfuse:2          # pin a major; :2 is the stable self-host line
    depends_on: [postgres]
    ports: ["3000:3000"]
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/langfuse
      NEXTAUTH_SECRET: dev-secret-change-me
      SALT: dev-salt-change-me
      NEXTAUTH_URL: http://localhost:3000
      TELEMETRY_ENABLED: "false"
```

> **Prefer Phoenix?** It's a single container with no DB dependency — `docker run -p 6006:6006 arizephoenix/phoenix:latest` — and speaks OTLP on `http://localhost:6006`. Use whichever you like; the span attributes below are identical.
>
> **No backend at all?** Skip both. In `tracing.py` (Step 1) fall back to emitting one structured-JSON line per request. The Definition of Done only requires "one coherent trace with per-step timing" — a JSON object with a nested `spans` array satisfies it.

Add the new module + design files:

```bash
touch app/tracing.py app/degrade.py app/metrics.py app/feedback.py
mkdir -p load designs designs/diagrams
touch load/locustfile.py
touch designs/01-enterprise-rag.md designs/02-coding-copilot.md designs/03-support-triage-agent.md
```

New Postgres tables (add to your migration or run once with `psql`):

```sql
-- prompt template registry: version prompts as first-class artifacts (lecture 13)
CREATE TABLE IF NOT EXISTS prompt_versions (
  id BIGSERIAL PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL,
  template TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (name, version));
INSERT INTO prompt_versions (name, version, template) VALUES
  ('chat-default', 'v1', 'You are a helpful assistant. {user}')
  ON CONFLICT DO NOTHING;

-- feedback linked to a trace id (the flywheel join key)
CREATE TABLE IF NOT EXISTS feedback (
  id BIGSERIAL PRIMARY KEY, trace_id TEXT NOT NULL, tenant_id TEXT,
  rating SMALLINT, edited_text TEXT, created_at TIMESTAMPTZ DEFAULT now());

-- (request_metrics already exists from Week 2; we ADD columns for the full trace attribute set)
ALTER TABLE request_metrics
  ADD COLUMN IF NOT EXISTS usd_cost NUMERIC DEFAULT 0,
  ADD COLUMN IF NOT EXISTS model_version TEXT,
  ADD COLUMN IF NOT EXISTS prompt_version TEXT,
  ADD COLUMN IF NOT EXISTS trace_id TEXT,
  ADD COLUMN IF NOT EXISTS degraded_tier TEXT;
```

Add to `.env`:

```bash
# ---- observability ----
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...      # create a project in the Langfuse UI, copy the keys
LANGFUSE_SECRET_KEY=sk-lf-...
# OTLP alternative (Phoenix): OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006
# ---- degradation ----
DEGRADE_STALE_TTL_SEC=1800         # how old an exact-cache hit may be during an outage
```

**Expected result:** `docker compose ps` shows redis + postgres + minio + langfuse healthy; Langfuse UI loads at `http://localhost:3000` (create a project, copy keys into `.env`); `psql "$DATABASE_URL" -c '\dt'` lists `prompt_versions` and `feedback`.

**Verify:**
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:3000   # 200 (Langfuse) — or 6006 for Phoenix
psql "$DATABASE_URL" -c "SELECT name, version FROM prompt_versions;"
ollama list                                                       # llama3.1 + nomic-embed-text present
```

**Troubleshoot:**
- Langfuse won't start / DB error → it needs its own `langfuse` database. `psql "$DATABASE_URL" -c 'CREATE DATABASE langfuse;'` then `docker compose up -d langfuse`.
- Port 3000 in use → change the host port mapping (`"3001:3000"`) and `LANGFUSE_HOST` to match.
- On Windows, a container reaching Ollama on the host uses `http://host.docker.internal:11434`, not `localhost`.

---

## Step-by-step

### Step 1 — End-to-end tracing: one request, one trace (`tracing.py`)

**What:** Wrap each gateway step (cache lookup → route decision → provider call → spend) in a span so one `/chat` request produces one trace carrying the mandatory attribute set: `model`, `model_version`, `prompt_tokens`, `completion_tokens`, `usd_cost`, `ttft_ms`, `total_ms`, `cache_layer`, `route_tier`, `tenant_id`, `prompt_version`, `trace_id`.

**Why:** Without this, "the answers got worse and the bill is up 30%" is unanswerable — you can't see which request was slow, which model answered, or what you actually sent it (lecture 13). Per-step spans give you per-step timing for free: "the request was slow" becomes "the provider call took 2,400 ms of 2,410 ms." **Redact the resolved prompt before it enters a span** (lecture 17) — the trace store is a derived store subject to GDPR.

**Do it:**

```python
# app/tracing.py
import os, time, contextlib
from langfuse import Langfuse
from app.secrets import redact          # Week 2: strips sk-... and (extend it) PII

_lf = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)

# The load-bearing attribute set (lecture 13). Use OTel gen_ai.* names where you export to OTLP.
def start_trace(tenant_id: str, prompt_version: str):
    return _lf.trace(name="chat.request",
                     user_id=tenant_id,                  # tenant slice for the dashboard
                     metadata={"prompt_version": prompt_version},
                     tags=[f"tenant:{tenant_id}"])

@contextlib.contextmanager
def span(trace, name: str):
    s = trace.span(name=name); t0 = time.perf_counter()
    try:
        yield s
    finally:
        s.end()

def finish(trace, *, model, model_version, prompt_tokens, completion_tokens,
           usd_cost, ttft_ms, total_ms, cache_layer, route_tier, resolved_prompt):
    trace.update(output={"redacted_prompt": redact(resolved_prompt)},   # NEVER raw PII
                 metadata={"model": model, "model_version": model_version,
                           "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                           "usd_cost": usd_cost, "ttft_ms": ttft_ms, "total_ms": total_ms,
                           "cache_layer": cache_layer, "route_tier": route_tier})
    _lf.flush()
    return trace.id
```

Wire it into your `/chat` handler so the layers each get a span (this reuses your Week 2 `chat.py` composition — killswitch → rate limit → cache → gateway → spend):

```python
# app/chat.py  (add tracing around the existing flow)
from app import tracing
from app.deps import save_metric

@app.post("/chat")
async def chat(body: ChatIn):
    tr = tracing.start_trace(body.tenant_id, prompt_version="v1")
    await check_killswitch(body.tenant_id); await check_rate(body.tenant_id, ...)
    messages = [{"role": "user", "content": body.message}]

    with tracing.span(tr, "cache.lookup"):
        hit, layer = await cache.get(body.tenant_id, "chat-default", messages)
    if hit:
        tid = tracing.finish(tr, model="cache", model_version=layer, prompt_tokens=0,
                             completion_tokens=0, usd_cost=0.0, ttft_ms=0, total_ms=0,
                             cache_layer=layer, route_tier="cache", resolved_prompt=body.message)
        return {"answer": hit, "cache": layer, "trace_id": tid}

    t0 = time.perf_counter()
    with tracing.span(tr, "provider.call"):
        resp = await complete(messages, tenant_id=body.tenant_id)   # router + fallback + breaker
    total_ms = int((time.perf_counter() - t0) * 1000)
    answer = resp.choices[0].message.content
    u = resp.usage
    await cache.put(body.tenant_id, "chat-default", messages, answer)
    await record_spend(body.tenant_id, "gpt-4o-mini", u.prompt_tokens, u.completion_tokens)
    usd = await usd_for(u.prompt_tokens, u.completion_tokens, "gpt-4o-mini")   # from model_prices
    tid = tracing.finish(tr, model="gpt-4o-mini",
                         model_version=(resp.model or "gpt-4o-mini"),
                         prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                         usd_cost=usd, ttft_ms=total_ms, total_ms=total_ms,
                         cache_layer="none", route_tier="strong", resolved_prompt=body.message)
    await save_metric(body.tenant_id, ttft_ms=total_ms, total_ms=total_ms, layer="none",
                      usd_cost=usd, trace_id=tid, model_version=resp.model, prompt_version="v1")
    return {"answer": answer, "cache": "none", "trace_id": tid}
```

**Expected result:** One `/chat` call → one trace in the Langfuse UI (or one JSON line) with child spans `cache.lookup` and `provider.call`, each timed, and the metadata attribute set attached. The response body now includes a `trace_id`.

**Verify:**
```bash
curl -s localhost:8000/chat -d '{"tenant_id":"A","message":"capital of France?"}' | python -m json.tool
# open http://localhost:3000 -> Traces -> click it -> confirm the waterfall + all attributes
```

**Troubleshoot:**
- Trace never appears → Langfuse batches; call `_lf.flush()` (done in `finish`) or the SDK's shutdown flush. Check the keys match the project you created.
- Two traces per request → you started a trace inside a retried code path; start it exactly once at the top of the handler.
- Raw PII in the trace → you passed `body.message` instead of `redact(...)`. Extend `redact()` from Week 2 to also mask emails/phones/order-ids before *any* export (lecture 17).

---

### Step 2 — Cost/latency/quality dashboard (`metrics.py`, `GET /admin/metrics`)

**What:** One admin screen showing, **per tenant and per model**: request count, p50/p95 TTFT, total tokens, $ spent this month, and cache hit rate. This is the single screen you'd watch in production.

**Why:** Traces answer "why was *this* request slow?"; the dashboard answers "which tenant/model is the cost and latency coming from *right now*?" You already have the raw rows in `request_metrics` + `spend_ledger` from Week 2 — this just aggregates them.

**Do it:**

```python
# app/metrics.py
from fastapi import APIRouter
from app.deps import pg_pool
router = APIRouter()

@router.get("/admin/metrics")
async def admin_metrics():
    async with pg_pool.connection() as con:
        rows = await (await con.execute("""
            SELECT tenant_id, model,
                   COUNT(*)                                             AS n,
                   percentile_cont(0.5)  WITHIN GROUP (ORDER BY ttft_ms) AS p50_ttft,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY ttft_ms) AS p95_ttft,
                   SUM(prompt_tokens + completion_tokens)               AS tokens,
                   SUM(usd_cost)                                        AS usd,
                   AVG(CASE WHEN cache_layer <> 'none' THEN 1.0 ELSE 0.0 END) AS hit_rate
              FROM request_metrics
             WHERE created_at >= date_trunc('month', now())
             GROUP BY tenant_id, model
             ORDER BY usd DESC NULLS LAST""")).fetchall()
    return [dict(zip(("tenant_id","model","n","p50_ttft","p95_ttft",
                      "tokens","usd","hit_rate"), r)) for r in rows]
```

Register it: `app.include_router(metrics.router)`.

**Expected result:** After a handful of `/chat` calls across two tenants, `GET /admin/metrics` returns one row per (tenant, model) with p50/p95 TTFT, tokens, $ this month, and hit rate.

**Verify:**
```bash
for t in A A A B; do curl -s -o /dev/null localhost:8000/chat -d "{\"tenant_id\":\"$t\",\"message\":\"hi\"}"; done
curl -s localhost:8000/admin/metrics | python -m json.tool
```

**Troubleshoot:**
- All `hit_rate` = 0 → you never repeat a prompt; fire the *same* message twice per tenant to see an exact-cache hit reflected.
- `percentile_cont` errors → it's standard Postgres; ensure `ttft_ms` is numeric and non-null (cache hits log `ttft_ms=0`, which is fine).
- Want a real chart? Point Grafana at the same Postgres, or use Langfuse's built-in dashboards — but the JSON endpoint is enough for the DoD.

---

### Step 3 — Feedback + prompt versioning (`feedback.py`)

**What:** `POST /feedback` links a thumbs up/down (+ optional edited text) to a `trace_id`. Prompt templates live in the `prompt_versions` table and every trace is tagged with the version it used, so a query can compare quality across versions.

**Why:** This is the data flywheel (lecture 13): every thumbs-down becomes training/eval signal, and versioning lets you attribute a regression to a *specific* prompt/model change instead of guessing. "Versioning prompts in code comments" is a Week 3 pitfall — use the table.

**Do it:**

```python
# app/feedback.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.deps import pg_pool
router = APIRouter()

class Feedback(BaseModel):
    trace_id: str; tenant_id: str; rating: int; edited_text: str | None = None

@router.post("/feedback")
async def submit(fb: Feedback):
    async with pg_pool.connection() as con:
        await con.execute(
            "INSERT INTO feedback(trace_id, tenant_id, rating, edited_text) VALUES (%s,%s,%s,%s)",
            (fb.trace_id, fb.tenant_id, fb.rating, fb.edited_text))
    return {"ok": True}

@router.get("/admin/quality-by-version")
async def quality_by_version():
    # join feedback -> request_metrics on trace_id -> compare avg rating per prompt_version
    async with pg_pool.connection() as con:
        rows = await (await con.execute("""
            SELECT m.prompt_version, COUNT(f.*) AS n, AVG(f.rating::float) AS avg_rating
              FROM feedback f JOIN request_metrics m ON m.trace_id = f.trace_id
             GROUP BY m.prompt_version ORDER BY m.prompt_version""")).fetchall()
    return [dict(zip(("prompt_version","n","avg_rating"), r)) for r in rows]
```

**Expected result:** `POST /feedback` with a real `trace_id` from Step 1 stores a row; `GET /admin/quality-by-version` returns average rating per prompt version.

**Verify:**
```bash
TID=$(curl -s localhost:8000/chat -d '{"tenant_id":"A","message":"hi"}' | python -c "import sys,json;print(json.load(sys.stdin)['trace_id'])")
curl -s localhost:8000/feedback -d "{\"trace_id\":\"$TID\",\"tenant_id\":\"A\",\"rating\":1}"
curl -s localhost:8000/admin/quality-by-version | python -m json.tool
```

**Troubleshoot:**
- Empty join → `save_metric` must persist `trace_id` (Step 1) or the join key is null.
- Ship `v2` of the prompt: `INSERT INTO prompt_versions(name,version,template) VALUES('chat-default','v2',...)`, tag new traces `prompt_version="v2"`, and the same query now compares v1 vs v2.

---

### Step 4 — Degradation mode: the ladder (`degrade.py`)

**What:** A system state where, when the breaker is open for **all** providers, `/chat` descends a ladder instead of 500-ing: **fresh exact-cache hit → semantic-cache hit → local Ollama → enqueue + 202 "queued"** — and **every degraded serve is labeled and alerts** (lecture 14).

**Why:** A normal service returns 503 under total outage; an LLM gateway has caches and a local model, so it can keep answering. The trap is the *silent* stale serve — a cached answer with no signal to the user or ops is a slow-motion outage. The rule: **degrade AND alert.**

**Do it:**

```python
# app/degrade.py
import os, time, logging
from fastapi import Response
from app import breaker, cache
from app.gateway import router as llm_router
log = logging.getLogger("degrade")
STALE_TTL = int(os.environ.get("DEGRADE_STALE_TTL_SEC", 1800))

async def all_providers_down() -> bool:
    return await breaker.is_open("openai") and await breaker.is_open("ollama")

def _emit(tier: str, alert: bool = False):
    log.warning("degradation tier=%s alert=%s", tier, alert)   # -> your metric/alert pipeline

async def serve_with_degradation(tenant_id, messages):
    """Called by /chat ONLY when the normal gateway path has failed for all providers.
       Returns (answer|None, tier, status). status 202 => enqueued."""
    # Rung 1: fresh exact hit
    hit, layer = await cache.get(tenant_id, "chat-default", messages)
    if hit and layer == "exact":
        _emit("cache:exact"); return hit, "cache:exact", 200
    # Rung 2: semantic hit (strict threshold — a wrong-but-confident answer compounds an outage)
    if hit and layer == "semantic":
        _emit("cache:semantic", alert=True); return hit, "cache:semantic", 200
    # Rung 3: local Ollama (providers are down, not your hardware)
    try:
        resp = await llm_router.acompletion(model="ollama/llama3.1", messages=messages)
        _emit("degraded:local", alert=True)
        return resp.choices[0].message.content, "degraded:local", 200
    except Exception:
        pass
    # Rung 4: enqueue + 202 (honest holding message, never a 500)
    job_id = await enqueue_job(tenant_id, messages)     # your Week 2 queue, or a Redis LPUSH
    _emit("queued", alert=True)
    return None, "queued", 202
```

Hook it into `/chat`: wrap the normal `complete(...)` call; on the exhaustion exception (all deployments failed / breaker open everywhere), call `serve_with_degradation` and set the `x-degradation-tier` response header:

```python
# app/chat.py (degradation branch)
try:
    resp = await complete(messages, tenant_id=body.tenant_id)
    answer = resp.choices[0].message.content; tier = "strong"
except Exception:
    answer, tier, status = await degrade.serve_with_degradation(body.tenant_id, messages)
    headers = {"x-degradation-tier": tier}
    if status == 202:
        return JSONResponse({"status": "queued", "tier": tier}, status_code=202, headers=headers)
    return JSONResponse({"answer": answer, "tier": tier}, headers=headers)
```

**Expected result:** With all providers unreachable, `/chat` returns 200 with a cached/local answer (or 202 queued) and an `x-degradation-tier` header — never a 5xx.

**Verify (simulate a full outage):**
```bash
# break BOTH backends: bad provider key + point Ollama at a dead port, then trip the breaker
OPENAI_API_KEY=bad OLLAMA_API_BASE=http://localhost:1 uvicorn app.main:app &
for i in 1 2 3 4; do curl -s -o /dev/null localhost:8000/chat -d '{"tenant_id":"A","message":"hi"}'; done  # trip breaker
curl -s -D - localhost:8000/chat -d '{"tenant_id":"A","message":"capital of France?"}' | grep -i x-degradation-tier
# if you pre-seeded the cache, you get cache:exact; else it enqueues (202)
```

**Troubleshoot:**
- Still getting 5xx → your `except` doesn't catch LiteLLM's exhaustion exception; broaden it, and make sure `serve_with_degradation` never re-raises.
- Ollama rung "works" during your outage sim → you only broke the provider, not Ollama. To force the queue rung, break Ollama too (dead `OLLAMA_API_BASE`).
- Degrades silently → you skipped `_emit(..., alert=True)`. Every rung below normal must emit; wire `_emit` to a counter you can alert on.

---

### Step 5 — Load test the fairness guarantee (`load/locustfile.py`)

**What:** Hammer the gateway with **one abusive tenant** (high QPS) and **one normal tenant**; prove via the dashboard that the abusive tenant hits 429/402 while the normal tenant's p95 stays within its SLO. This is the milestone's "no tenant starves another" proof.

**Why:** Your Week 2 token bucket + spend kill-switch are per-tenant *by design*; a load test is what turns "should be isolated" into "demonstrably isolated" under real concurrency.

**Do it (locust — pure Python, easiest on Windows):**

```python
# load/locustfile.py
from locust import HttpUser, task, constant_pacing, between

class AbusiveTenant(HttpUser):
    wait_time = constant_pacing(0.05)          # ~20 req/s per user — floods its own bucket
    @task
    def hammer(self):
        self.client.post("/chat", json={"tenant_id": "ABUSER", "message": "spam"},
                         name="/chat [ABUSER]")

class NormalTenant(HttpUser):
    wait_time = between(1, 2)                   # human-paced
    @task
    def normal(self):
        self.client.post("/chat", json={"tenant_id": "NORMAL", "message": "real question"},
                         name="/chat [NORMAL]")
```

Run it (weight the abuser heavier):

```bash
locust -f load/locustfile.py --host http://localhost:8000 \
       --headless -u 60 -r 10 -t 2m \
       --class-picker AbusiveTenant NormalTenant
# Windows/Git-Bash: if the CLI class flags misbehave, run `locust -f load/locustfile.py --host ...`
# and pick the mix in the web UI at http://localhost:8089.
```

**k6 alternative** (if you prefer JS/Go): a `scenarios` block with two executors keyed by tenant does the same — see the k6 docs for `scenarios` + `http.post`. Locust is the simpler free/local default here.

**Expected result:** In locust's stats, `/chat [ABUSER]` shows a wall of `429` (and eventually `402` if it crosses its cap) while `/chat [NORMAL]` stays `200` with a healthy p95. Confirm on your own dashboard:

**Verify:**
```bash
curl -s localhost:8000/admin/metrics | python -m json.tool
# ABUSER: high count, many non-200; NORMAL: p95_ttft within its SLO, ~all 200
```

**Troubleshoot:**
- Both tenants throttled → your bucket key isn't per-tenant (`bucket:{tenant_id}`), a Week 2 regression. Fix in `limits.py`.
- Normal tenant's p95 balloons → your connection pool/semaphore is undersized (Little's Law: `concurrency = QPS × L`). Raise the pool ceiling; don't let the abuser's in-flight calls starve the pool.
- No 429s at all → lower `tokens_per_min` for `ABUSER` in `tenant_limits` so its bucket empties fast, or raise the abuser's request rate.

---

### Step 6 — Capacity math sheet (in `README.md`)

**What:** Write the QPS→tokens→quota→$ formula and a worked example in `README.md`: given target QPS, avg input+output tokens, and a provider's TPM/RPM, compute required concurrency, whether you hit the quota wall, and monthly $ — then show how a 30% cache hit rate and a 70% cheap cascade move the numbers.

**Why:** The counter-intuitive truth (lecture 14): your bottleneck is the **provider's TPM/RPM quota**, not your CPU. Adding replicas does nothing against a per-account quota. Interviewers probe this hardest; "boxes but no math" is the classic failure.

**Do it — paste this method + worked example into `README.md` and plug in your own numbers:**

```
## Capacity method (QPS -> tokens -> quota -> $)
Inputs: QPS (peak), T_in, T_out, L (avg latency s), TPM_limit, RPM_limit, price_in, price_out

1. Demand:
   tokens_per_request  = T_in + T_out
   tokens_per_minute   = QPS * 60 * tokens_per_request
   requests_per_minute = QPS * 60
2. Quota verdict (binding constraint = smaller headroom; want >= 1.3):
   TPM_headroom = TPM_limit / tokens_per_minute
   RPM_headroom = RPM_limit / requests_per_minute
3. Concurrency (Little's Law): concurrency = QPS * L   -> sizes pool/semaphore, NOT CPU
4. Monthly $: cost_per_request = T_in*price_in + T_out*price_out
              monthly = cost_per_request * QPS * 2,592,000   (flat-out peak, worst case)

Levers:
   cache hit rate h  -> provider sees (1-h): effective_TPM = tokens_per_minute*(1-h)
   cheap cascade c   -> blended = c*price_cheap + (1-c)*price_expensive  (c = RESOLVED cheap, not attempted)
```

**Worked example (support-triage, sizing for peak):**

| Scenario | Inputs | TPM headroom | Monthly $ (peak) |
|---|---|---|---|
| Naive | QPS 50, T_in 1500, T_out 400, TPM 2M, $3/$15 per 1M | 2M / 5.7M = **0.35 → WALL** | ~$1.36M |
| + 30% cache | provider sees 70% | 2M / 3.99M = **0.50 → still wall** | ~$953k |
| + 70% cheap cascade on a separate quota | expensive sees 0.7×0.3 = 21% | 2M / 1.197M = **1.67 → clears** | ~$315k |

Concurrency: `50 × 4s = 200` in flight → size the semaphore ~260 (30% headroom). Same user-facing QPS; **architecture, not compute, cleared the wall.**

**Expected result:** `README.md` contains the formula block, the worked table, the concurrency number, and a sentence naming provider quota (TPM) as the binding constraint.

**Verify:** Re-derive one row by hand and confirm your arithmetic matches (e.g. `50 × 60 × 1900 = 5,700,000 tokens/min`; `2,000,000 / 5,700,000 = 0.35`).

**Troubleshoot:**
- "But my CPU is idle" — correct, and that's the point: the wall is the provider's per-account quota, not your box. Model TPM/RPM explicitly; alert at 80% of TPM (leading metric), not on error rate.
- Cascade looks free — it isn't: a cheap tier that *fails and escalates* costs **both** tiers' tokens. Use the *resolved-cheap* fraction, not *attempted-cheap*.

---

### Step 7 — Three timed mock system designs (45 min each, in `designs/`)

**What:** Run the seven-step method (lecture 16) under a **real 45-minute timer**, three times, one per prompt. Each deliverable is a Markdown file (+ an ASCII/Excalidraw diagram in `designs/diagrams/`) with: architecture diagram, data model, routing+caching+fallback plan, eval+observability strategy, a QPS/token/**cost estimate**, and a **failure/privacy** section (what breaks, how it degrades, how GDPR delete works in *that* design).

**Why:** This is the milestone's portfolio piece and the interview skill. The candidates who fail don't lack knowledge — they lack a *loop* and run out of time before the math and failure story, which is exactly where the signal is.

**Do it — run this loop for each design, out loud, watching the clock:**

```
STEP                                    BUDGET
1  Clarify requirements & scale          5 min   4 numbers + 1 class: QPS(peak), tenancy, TTFT SLO, privacy class, task type
2  Pick & sketch reference architecture  6 min   climb the ladder; stop at the lowest rung that works; say why the rung below can't
3  Data model — the four stores          5 min   Redis / Postgres / object / vector: what lives where + TTL + which is DERIVED
4  The LLM-specific layer                8 min   routing / cache (tenant-keyed!) / fallback+breaker / stream+TTFT / limits
5  QPS->tokens->quota->$ (Step 6 method)  9 min   the binding constraint + the cache/cascade levers  <-- PROBED HARDEST
6  Failure & degradation story           7 min   breaker -> fallback -> degradation ladder; "degrade AND alert"
7  Privacy / security / GDPR delete       5 min   PII redaction before prompt AND before logs; cascade delete downhill through derived copies
                                        ─────
                                         45 min  (leave ~2-min buffer)
```

The three prompts (`timer on` before you read past the title):

1. **`designs/01-enterprise-rag.md` — Enterprise RAG assistant.** Multi-tenant, private docs, **access-controlled retrieval** (a user must never retrieve a chunk they can't see — retrieval filter by ACL is a hard invariant, and cross-tenant cache is a breach). GDPR delete must purge the vector index, not just Postgres.
2. **`designs/02-coding-copilot.md` — Coding copilot.** In-IDE, **low-latency (TTFT is the SLO)**, context harvesting (open files/repo), acceptance telemetry (did the dev keep the suggestion?). Aggressive prompt-prefix caching; a cheap local model for completions, escalate for chat.
3. **`designs/03-support-triage-agent.md` — High-volume support-triage agent.** **Async/queue** (not interactive), cheap-first cascade, **HITL on actions** (the model proposes a refund; code executes only after a human/threshold gate — "LLM proposes, code disposes"). This is where the capacity math from Step 6 came from.

Each file must end with a timestamp line proving it was done in ≤45 min, e.g. `> Done in 44 min — 2026-07-10 14:30`.

**Expected result:** Three files in `designs/`, each with all seven sections, a diagram, cost math, and a failure+privacy section, each timestamped ≤45 min.

**Verify:** Re-read each against the DoD list below. The two most-skipped sections are **Step 5 (the math)** and **Step 7 (GDPR delete in this design)** — if either is thin, you'd lose the interview there; redo that section.

**Troubleshoot:**
- You blew past 45 min → you spent too long on boxes (Step 2). Cap it hard; the diagram is a means, not the deliverable. Narrate *tradeoffs* ("Redis for the hot buffer *because* ms reads and Postgres is the source of truth; *the tradeoff is* I now own a rehydrate path"), not box names.
- You drew a multi-agent swarm for ticket classification → over-engineering signals inexperience. Stop at the lowest rung on the escalation ladder that meets the requirement.

---

## Putting it together — a short end-to-end run

Prove the whole Week-3 surface in one sitting:

```bash
uvicorn app.main:app --reload &

# 1) TRACE: one request -> one trace with the full attribute set
TID=$(curl -s localhost:8000/chat -d '{"tenant_id":"A","message":"capital of France?"}' \
      | python -c "import sys,json;print(json.load(sys.stdin)['trace_id'])")
# open http://localhost:3000 -> Traces -> confirm cache.lookup + provider.call spans + attributes

# 2) FEEDBACK: link a thumbs-down to that trace
curl -s localhost:8000/feedback -d "{\"trace_id\":\"$TID\",\"tenant_id\":\"A\",\"rating\":0}"

# 3) DASHBOARD: per-tenant/per-model cost + latency + hit rate
curl -s localhost:8000/admin/metrics | python -m json.tool

# 4) DEGRADATION: break both backends, trip the breaker, confirm no 5xx
OPENAI_API_KEY=bad OLLAMA_API_BASE=http://localhost:1 uvicorn app.main:app --port 8001 &
for i in 1 2 3 4; do curl -s -o /dev/null localhost:8001/chat -d '{"tenant_id":"A","message":"hi"}'; done
curl -s -D - localhost:8001/chat -d '{"tenant_id":"A","message":"hi"}' | grep -iE "x-degradation-tier|HTTP/"

# 5) LOAD TEST: fairness under an abusive tenant
locust -f load/locustfile.py --host http://localhost:8000 --headless -u 60 -r 10 -t 2m

# 6) CAPACITY SHEET + 7) THREE DESIGNS are in README.md and designs/ — reviewed by hand
```

If all six behave and the three designs are in `designs/`, the milestone (Deliverables A + B from Weeks 1–2, plus C here) is complete.

---

## Definition of Done — verifiable checks

Restating the spine's **Week 3 checklist** + the **milestone acceptance criteria** as things you can prove:

- [ ] **One coherent trace.** A single `/chat` request produces one trace with per-step spans and the attributes `model`, `model_version`, `prompt_tokens`, `completion_tokens`, `usd_cost`, `ttft_ms`, `total_ms`, `cache_layer`, `route_tier`, `tenant_id`, `prompt_version`, `trace_id` — the resolved prompt **redacted**. (Step 1)
- [ ] **Metrics view.** `GET /admin/metrics` shows per-tenant + per-model p50/p95 TTFT, tokens, $ this month, and cache hit rate. (Step 2)
- [ ] **Feedback + versioning.** `POST /feedback` links a rating to a `trace_id`; prompt versions live in a table and a query compares quality across versions. (Step 3)
- [ ] **Degradation mode.** With **all** providers "down," `/chat` still returns cached/local/queued responses (labeled via `x-degradation-tier`, alerting on each degraded serve) — **never a 5xx**. (Step 4)
- [ ] **Load test / fairness.** An abusive tenant hits 429/402 while a normal tenant's p95 stays within its SLO — "no tenant starves another," demonstrated in locust + the dashboard. (Step 5)
- [ ] **Capacity sheet.** `README.md` has the QPS→tokens→quota→$ formula, a worked example with a binding-constraint verdict and concurrency (Little's Law), and shows the cache + cascade levers moving the numbers. (Step 6)
- [ ] **Three timed designs.** `designs/` has all three (enterprise RAG, coding copilot, support-triage), each with a diagram, data model, routing/caching/fallback plan, eval+observability strategy, cost math, and a failure+privacy section (incl. how GDPR delete works there), each timestamped ≤45 min. (Step 7)

**Milestone acceptance (carried from Weeks 1–2, re-verify end to end):**
- [ ] Kill the primary provider live → requests keep succeeding via fallback; breaker opens; dashboard shows it.
- [ ] Client disconnect during a stream verifiably stops upstream token production.
- [ ] Spend kill-switch trips at the configured monthly cap → 402 with a ledger explanation.
- [ ] Cache hit rate reported; a test proves **no cross-tenant** cache hit.
- [ ] GDPR delete test green: **zero residue** in Postgres, Redis, vector store, and object storage — and now also **no PII in traces/logs/feedback** (extend the Week 1 delete test to assert the observability sinks are PII-free or `user_id`-addressable — lecture 17).

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Trace never appears in Langfuse | SDK batches; keys wrong | Call `_lf.flush()`; verify project keys in `.env` |
| Two traces per request | Trace started inside a retried path | Start the trace once at the top of the handler |
| Raw PII in a span/log | Exported before redaction | Redact upstream of the export boundary (lecture 17), not in the sink |
| `/chat` still 5xx during outage | `except` misses LiteLLM exhaustion error | Broaden the catch; `serve_with_degradation` must never re-raise |
| Degradation is silent | No `_emit(alert=True)` per rung | Every rung below normal must label + alert ("degrade AND alert") |
| Both tenants throttled together | Bucket key not per-tenant | Key `bucket:{tenant_id}` (Week 2 regression) |
| Normal tenant p95 balloons under load | Pool/semaphore undersized | Size from Little's Law `concurrency = QPS × L`, add ~30% |
| `admin/metrics` empty / errors | `trace_id`/`usd_cost` not persisted | Ensure `save_metric` writes the new columns |
| Capacity says "CPU fine" but 429s | Modeling compute, not quota | Model TPM/RPM; alert at 80% of TPM (leading metric) |
| Langfuse container crash-loops | Missing `langfuse` DB | `CREATE DATABASE langfuse;` then restart the service |
| Ollama unreachable from container | Using `localhost` | Use `http://host.docker.internal:11434` |

---

## Stretch goals (optional)

- **Request coalescing (single-flight).** Dedupe identical in-flight requests: `in_flight[hash] → Future`; the first request calls upstream, the rest `await` the same Future. The cheapest defense against a thundering herd; measure distinct-hash vs total-call gap (lecture 14).
- **Stale-cache rung with freshness policy.** Add rung 1b (serve *stale* exact hits during an outage) with a per-endpoint freshness TTL and a mandatory `x-degradation-tier: cache:exact-stale` label + alert. A "current balance" from 20 min ago is not the same as a support answer from 20 min ago.
- **Leading TPM alert.** Track `tokens_per_minute` as a rolling Redis gauge and fire at 80% of the limit — request a quota increase *before* the wall, not after (lecture 14).
- **Build-vs-buy write-up.** Add a `README.md` section scoring the five axes (control, data-residency, cost-at-your-volume, team size, ejectability) for *your* gateway, defending the "thin self-hosted LiteLLM + buy observability" default — or arguing against it with numbers (lecture 15).
- **OTel-native export.** Swap the Langfuse SDK for the OpenTelemetry SDK + OTLP exporter using the **GenAI semantic conventions** (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, …) so traces are portable across Langfuse/Phoenix/Datadog (lecture 13).
- **Quality-by-version eval.** Wire the `feedback`↔`prompt_versions` join into a Phase-7-style golden-set eval so a prompt regression is caught by a test, not a tenant email.
