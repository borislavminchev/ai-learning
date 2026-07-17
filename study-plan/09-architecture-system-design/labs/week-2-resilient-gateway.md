# Week 2 Lab: Build the Resilient Gateway — Router, Fallback, Streaming, Cache, Limits

> This week you layer a **resilient LLM gateway** onto Week 1's `llm-gateway-lab/` repo. Every model call now flows through one choke point that gives you: a **fallback chain** (primary provider → secondary → local Ollama) with a **Redis circuit breaker**, **SSE streaming** with TTFT measurement and *upstream cancellation on client disconnect*, a **two-layer tenant-safe cache** (exact + semantic), a **cheap-first cascade**, **token-aware per-tenant rate limits** with a **hard monthly spend kill-switch**, and **server-side secret handling** with optional per-tenant BYOK. The whole thing runs free on one real provider plus Ollama — kill the provider or point it at a bad key and watch fallback + breaker do their job.
>
> Read these lectures first — this guide assumes them:
> - [../lectures/06-gateway-router-pattern.md](../lectures/06-gateway-router-pattern.md) — one router, provider abstraction, where the abstraction leaks
> - [../lectures/07-circuit-breakers-and-fallback.md](../lectures/07-circuit-breakers-and-fallback.md) — fallback chain vs breaker, the state machine, the Redis rolling counter, the silent-outage trap
> - [../lectures/08-sse-streaming-ttft-cancellation.md](../lectures/08-sse-streaming-ttft-cancellation.md) — SSE framing, TTFT vs total, aborting the upstream call on disconnect
> - [../lectures/10-caching-layers.md](../lectures/10-caching-layers.md) — exact vs semantic vs prompt cache, the tenant-isolation iron rule
> - [../lectures/11-rate-limiting-and-spend-governance.md](../lectures/11-rate-limiting-and-spend-governance.md) — token bucket, 429 vs 402, per-tenant keying
> - [../lectures/12-secrets-and-key-management.md](../lectures/12-secrets-and-key-management.md) — server-side resolution, BYOK, the grep-your-logs test
> - [../lectures/09-async-queue-based-inference.md](../lectures/09-async-queue-based-inference.md) — background context for the cascade/queue seams

**Est. time:** ~10 hrs · **You will need:** Python 3.10+, Docker Desktop (Redis + Postgres/pgvector from Week 1), a terminal (Git-Bash on Windows), and **one** of:
- **Paid path (default):** one provider key (OpenAI *or* Anthropic *or* Gemini). Costs well under $1 for the whole lab.
- **Free/local path:** [Ollama](https://ollama.com) running `llama3.1` (chat) and `nomic-embed-text` (embeddings) on CPU. No key, no GPU, no spend.

Either way you run **two backends** so fallback is testable: one "primary" (the paid provider, or a second Ollama model tag) and **Ollama** as the always-up secondary. LiteLLM speaks to both through one OpenAI-compatible interface.

---

## Before you start (setup)

**What:** Bring up Week 1's stores, install the new deps, pull Ollama models, and add the new files.

**Why:** Everything this week is *state-backed* — the breaker counter, rate-limit bucket, and semantic cache all live in Redis/Postgres you already stood up in Week 1. No new infra is required.

**Do it:**

```bash
cd llm-gateway-lab
source .venv/Scripts/activate        # Windows (Git-Bash)
# source .venv/bin/activate          # macOS / Linux

# New deps this week (litellm from Week 1 already present)
uv add "litellm>=1.44" "sse-starlette>=2.1" tiktoken numpy tenacity

docker compose up -d                 # redis, postgres(+pgvector), minio from Week 1
docker compose ps                    # all healthy?

# Free/local path: pull the two Ollama models
ollama pull llama3.1
ollama pull nomic-embed-text
ollama serve &                       # if not already running as a service
```

Add the new module files and a `.env` block:

```bash
touch app/gateway.py app/stream.py app/cache.py app/limits.py app/routing.py app/secrets.py
touch tests/test_cache_tenant_isolation.py tests/test_stream_cancel.py \
      tests/test_rate_limit_fairness.py tests/test_spend_killswitch.py tests/test_no_secret_leak.py
```

`.env` (server-side only — never shipped to a client):

```bash
# ---- one real provider (pick one) ----
OPENAI_API_KEY=sk-...            # or:
# ANTHROPIC_API_KEY=sk-ant-...   # or:
# GEMINI_API_KEY=...
# ---- always-on local fallback ----
OLLAMA_API_BASE=http://localhost:11434
# ---- app config ----
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/gateway
REDIS_URL=redis://localhost:6379/0
BREAKER_FAIL_THRESHOLD=3
BREAKER_WINDOW_SEC=60
BREAKER_COOLDOWN_SEC=30
SEMANTIC_THRESHOLD=0.95
```

Create the new Postgres tables (add to your Week 1 migration or run once with `psql`):

```sql
-- request timings for TTFT/total
CREATE TABLE IF NOT EXISTS request_metrics (
  id BIGSERIAL PRIMARY KEY, tenant_id TEXT, model TEXT,
  ttft_ms INT, total_ms INT, cache_layer TEXT, route_tier TEXT,
  prompt_tokens INT, completion_tokens INT, created_at TIMESTAMPTZ DEFAULT now());

-- per-tenant semantic cache (pgvector). 768 dims = nomic-embed-text
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS semantic_cache (
  id BIGSERIAL PRIMARY KEY, tenant_id TEXT NOT NULL, model TEXT NOT NULL,
  embedding vector(768), response TEXT, expires_at TIMESTAMPTZ);
CREATE INDEX IF NOT EXISTS idx_semcache_tenant ON semantic_cache(tenant_id);

-- price table + spend ledger + kill-switch flag
CREATE TABLE IF NOT EXISTS model_prices (
  model TEXT PRIMARY KEY, input_per_1k NUMERIC, output_per_1k NUMERIC);
INSERT INTO model_prices VALUES
  ('gpt-4o-mini', 0.00015, 0.0006), ('gpt-4o', 0.0025, 0.01),
  ('ollama/llama3.1', 0, 0)
  ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS spend_ledger (
  id BIGSERIAL PRIMARY KEY, tenant_id TEXT, month TEXT, usd NUMERIC,
  created_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE IF NOT EXISTS tenant_limits (
  tenant_id TEXT PRIMARY KEY, monthly_cap_usd NUMERIC DEFAULT 1.00,
  tokens_per_min INT DEFAULT 20000, killed BOOLEAN DEFAULT false);
```

**Expected result:** `docker compose ps` shows redis + postgres healthy; `ollama list` shows both models; `psql $DATABASE_URL -c '\dt'` lists the five new tables.

**Verify:**
```bash
psql "$DATABASE_URL" -c "SELECT model, input_per_1k FROM model_prices;"
redis-cli -u "$REDIS_URL" ping        # -> PONG
```

**Troubleshoot:**
- `vector` extension missing → your Postgres image isn't pgvector. Use `pgvector/pgvector:pg16` in `docker-compose.yml` and recreate the container.
- Ollama `connection refused` → `ollama serve` isn't running, or Docker can't reach host. From a container use `http://host.docker.internal:11434`.

---

## Step-by-step

### Step 1 — Router + fallback in `gateway.py`

**What:** Configure `litellm.Router` mapping the logical name `chat-default` to an ordered list: primary provider → secondary → local Ollama, with `fallbacks` and `num_retries`.

**Why:** One logical name means the rest of the app never hard-codes a provider. A 5xx/timeout on the primary trips LiteLLM to the next entry automatically ([lecture 06](../lectures/06-gateway-router-pattern.md), [lecture 07](../lectures/07-circuit-breakers-and-fallback.md)).

**Do it:**

```python
# app/gateway.py
import os, time, litellm
from litellm import Router
from app.secrets import resolve_key          # Step 7
from app.breaker import breaker_state, record_result, allowed_providers  # Step 2

litellm.drop_params = True   # tolerate provider-specific param differences (leaky seam)

# model_list: each entry is one deployment. Same model_name = one logical group.
MODEL_LIST = [
    {"model_name": "chat-default",
     "litellm_params": {"model": "openai/gpt-4o-mini",
                        "api_key": os.environ.get("OPENAI_API_KEY")},
     "model_info": {"tier": "primary", "provider": "openai"}},
    {"model_name": "chat-default",
     "litellm_params": {"model": "ollama/llama3.1",
                        "api_base": os.environ["OLLAMA_API_BASE"]},
     "model_info": {"tier": "fallback", "provider": "ollama"}},
]

router = Router(
    model_list=MODEL_LIST,
    fallbacks=[{"chat-default": ["chat-default"]}],   # exhaust deployments in the group
    num_retries=2,                                     # retry transient errors before falling back
    timeout=30,
    retry_after=1,
    allowed_fails=1,        # LiteLLM's own cooldown; we ALSO run our Redis breaker in Step 2
    cooldown_time=30,
)

async def complete(messages, tenant_id, model="chat-default", stream=False):
    """The ONE model entrypoint. Deterministic code calls this; no business logic inside."""
    skip = [p for p in ("openai",) if p not in allowed_providers()]  # breaker-gated
    resp = await router.acompletion(model=model, messages=messages, stream=stream)
    return resp
```

**Expected result:** `python -c "import asyncio; from app.gateway import complete; print(asyncio.run(complete([{'role':'user','content':'hi'}], 'tenantA')).choices[0].message.content)"` returns a completion (from the primary if the key is valid).

**Verify:** Set `OPENAI_API_KEY=bad` and rerun — you get the same successful reply, now served by Ollama. Add `print(resp._hidden_params.get("model_id"))` to confirm which deployment answered.

**Troubleshoot:**
- Fallback never triggers on a bad key → a 401 is *not* retriable by default. Add `openai.AuthenticationError` handling or verify by killing the network / using a 500-returning mock. Auth errors *should* fall to Ollama because they exhaust the primary deployment.
- Ollama replies but formatting is off → set `litellm.drop_params = True` (done above); provider param support differs (leaky seam, lecture 06).

---

### Step 2 — Circuit breaker in Redis (`breaker.py`)

**What:** A per-provider rolling failure counter in Redis. Open the breaker (skip the provider, go straight to fallback) after `N` failures in a window; **half-open** after a cooldown; expose state at `GET /gateway/health`.

**Why:** Fallback handles a *single* failed request; the breaker stops *hammering* a provider that is persistently down so you fail fast instead of eating retry latency on every call ([lecture 07](../lectures/07-circuit-breakers-and-fallback.md)). The trap: silent fallback hides the outage — so we expose state and you alert on breaker-open.

**Do it:**

```python
# app/breaker.py
import os, time, redis.asyncio as redis
r = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
N       = int(os.environ.get("BREAKER_FAIL_THRESHOLD", 3))
WINDOW  = int(os.environ.get("BREAKER_WINDOW_SEC", 60))
COOLDOWN= int(os.environ.get("BREAKER_COOLDOWN_SEC", 30))

def _fk(p): return f"breaker:fail:{p}"      # sorted-set of failure timestamps
def _ok(p): return f"breaker:open:{p}"      # key present => open, with TTL=cooldown

async def record_result(provider: str, ok: bool):
    now = time.time()
    if ok:
        await r.delete(_fk(provider))       # success in half-open closes the breaker
        return
    await r.zadd(_fk(provider), {f"{now}": now})
    await r.zremrangebyscore(_fk(provider), 0, now - WINDOW)   # rolling window
    fails = await r.zcard(_fk(provider))
    if fails >= N:
        await r.set(_ok(provider), "open", ex=COOLDOWN)        # OPEN -> auto half-open after TTL

async def state(provider: str) -> str:
    if await r.exists(_ok(provider)):
        ttl = await r.ttl(_ok(provider))
        return "half-open" if ttl <= COOLDOWN // 2 else "open"
    return "closed"

async def is_open(provider: str) -> bool:
    return bool(await r.exists(_ok(provider)))
```

Wire it into `complete()` and the router's success/failure path, and add the health route:

```python
# app/gateway.py (additions)
from app import breaker

async def complete(messages, tenant_id, model="chat-default", stream=False):
    if await breaker.is_open("openai"):
        # breaker open: force the fallback deployment only
        model = "chat-default"   # router still has ollama; we bias by removing primary
    try:
        resp = await router.acompletion(model=model, messages=messages, stream=stream)
        provider = (resp._hidden_params or {}).get("custom_llm_provider", "openai")
        await breaker.record_result(provider, ok=True)
        return resp
    except Exception:
        await breaker.record_result("openai", ok=False)
        raise
```

```python
# app/main.py (add route)
from app import breaker
@app.get("/gateway/health")
async def gateway_health():
    return {p: await breaker.state(p) for p in ("openai", "ollama")}
```

**Expected result:** With a bad primary key, fire 3 requests → `GET /gateway/health` shows `{"openai":"open","ollama":"closed"}`. Wait `COOLDOWN` seconds → next call flips it to `half-open`, and a success closes it.

**Verify:**
```bash
for i in 1 2 3; do curl -s localhost:8000/chat -d '{"tenant_id":"A","message":"hi"}'; done
curl -s localhost:8000/gateway/health          # {"openai":"open",...}
```

**Troubleshoot:**
- Breaker never opens → confirm exceptions actually reach the `except` (a fallback that *succeeds* swallows the primary failure at the router level). Record the failure inside a LiteLLM custom callback (`litellm.callbacks`) on the failed deployment, not just on the top-level call.
- Breaker opens on a single blip → `N` too low or `WINDOW` too long. Start `N=3`, `WINDOW=60`.

---

### Step 3 — SSE streaming with upstream cancellation (`stream.py`)

**What:** `GET /chat/stream` returns `text/event-stream`, forwards provider tokens as SSE events, detects client disconnect, and **cancels the upstream generator**. Record TTFT and total time into `request_metrics`.

**Why:** Streaming is the UX users feel; TTFT is the number they judge. The money bug: if the client closes the tab and you *don't* stop the upstream call, you keep paying for tokens nobody reads ([lecture 08](../lectures/08-sse-streaming-ttft-cancellation.md)).

**Do it:**

```python
# app/stream.py
import time, asyncio
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from app.gateway import complete
from app.deps import save_metric      # writes request_metrics row

router = APIRouter()

@router.get("/chat/stream")
async def chat_stream(request: Request, tenant_id: str, message: str):
    started = time.perf_counter(); ttft_ms = None; upstream = None
    async def gen():
        nonlocal ttft_ms, upstream
        upstream = await complete([{"role": "user", "content": message}],
                                  tenant_id=tenant_id, stream=True)
        try:
            async for chunk in upstream:
                if await request.is_disconnected():        # client gone
                    break                                   # -> finally aborts upstream
                delta = chunk.choices[0].delta.content or ""
                if delta and ttft_ms is None:
                    ttft_ms = int((time.perf_counter() - started) * 1000)
                if delta:
                    yield {"event": "token", "data": delta}
            yield {"event": "done", "data": "[DONE]"}
        finally:
            # CRITICAL: stop upstream token production (stops the token meter)
            if hasattr(upstream, "aclose"):
                await upstream.aclose()
            total_ms = int((time.perf_counter() - started) * 1000)
            await save_metric(tenant_id, ttft_ms=ttft_ms, total_ms=total_ms, layer="none")
    return EventSourceResponse(gen())
```

Register it in `main.py`: `app.include_router(stream.router)`. Disable proxy buffering if you front this with nginx (`proxy_buffering off;` on the route) or set header `X-Accel-Buffering: no`.

**Expected result:** `curl -N` prints tokens one event at a time, then `[DONE]`. A `request_metrics` row appears with a small `ttft_ms` and larger `total_ms`.

**Verify (the disconnect assertion — this is the DoD):**

```python
# tests/test_stream_cancel.py
import asyncio, httpx, pytest
@pytest.mark.asyncio
async def test_disconnect_aborts_upstream(monkeypatch, caplog):
    produced = {"n": 0}
    async def fake_stream(*a, **k):
        class C:
            async def __aiter__(self):
                for i in range(1000):
                    produced["n"] += 1                      # counts tokens the model emits
                    await asyncio.sleep(0.01)
                    yield type("x",(),{"choices":[type("y",(),{"delta":type("z",(),{"content":"t"})()})()]})
            async def aclose(self): pass
        return C()
    monkeypatch.setattr("app.stream.complete", fake_stream)
    async with httpx.AsyncClient(app=app, base_url="http://t") as c:
        async with c.stream("GET","/chat/stream?tenant_id=A&message=hi") as resp:
            async for _ in resp.aiter_lines(): break        # read one token then disconnect
    n_at_disconnect = produced["n"]
    await asyncio.sleep(0.3)                                 # give a leaked generator time to run
    assert produced["n"] - n_at_disconnect <= 2             # upstream STOPPED after disconnect
```

**Troubleshoot:**
- TTFT looks like the total time → a proxy is buffering. Disable `proxy_buffering`/Cloudflare buffering, add `X-Accel-Buffering: no`.
- `is_disconnected()` never returns True in tests → drive it with a real `httpx.AsyncClient` streaming context that closes early (as above), not a plain `TestClient`.
- Tokens keep counting after disconnect → your `finally`/`aclose()` isn't reached; make sure you `break` (not `return`) inside the `async for` so `finally` runs.

---

### Step 4 — Two-layer tenant-safe cache (`cache.py`)

**What:** Layer 1 exact-match key = `sha256(tenant_id + model + normalized_messages)`. On miss, embed the last user message and check a **per-tenant** semantic cache (pgvector, cosine threshold start 0.95). Store with TTL. **Every key includes `tenant_id`.**

**Why:** A cache hit turns a 2-second, 3-cent call into a 5 ms free one — but a cross-tenant hit is a data breach, not a bug ([lecture 10](../lectures/10-caching-layers.md)). The iron rule: never serve a hit across a tenant boundary.

**Do it:**

```python
# app/cache.py
import hashlib, json, time, os
import redis.asyncio as redis
from app.gateway import router
from app.deps import pg_pool           # psycopg async pool
r = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
THRESHOLD = float(os.environ.get("SEMANTIC_THRESHOLD", 0.95))
TTL = 3600

def _norm(messages): return json.dumps(messages, sort_keys=True, separators=(",", ":")).strip().lower()
def exact_key(tenant_id, model, messages):
    raw = f"{tenant_id}|{model}|{_norm(messages)}"
    return "cache:exact:" + hashlib.sha256(raw.encode()).hexdigest()

async def _embed(text):
    resp = await router.aembedding(model="ollama/nomic-embed-text", input=[text])
    return resp.data[0]["embedding"]

async def get(tenant_id, model, messages):
    # L1 exact
    hit = await r.get(exact_key(tenant_id, model, messages))
    if hit: return hit, "exact"
    # L2 semantic — SQL WHERE tenant_id = %s is the isolation guarantee
    emb = await _embed(messages[-1]["content"])
    async with pg_pool.connection() as con:
        row = await (await con.execute(
            """SELECT response, 1 - (embedding <=> %s::vector) AS sim
                 FROM semantic_cache
                WHERE tenant_id = %s AND model = %s AND expires_at > now()
                ORDER BY embedding <=> %s::vector LIMIT 1""",
            (emb, tenant_id, model, emb))).fetchone()
    if row and row[1] >= THRESHOLD:
        return row[0], "semantic"
    return None, None

async def put(tenant_id, model, messages, response):
    await r.set(exact_key(tenant_id, model, messages), response, ex=TTL)
    emb = await _embed(messages[-1]["content"])
    async with pg_pool.connection() as con:
        await con.execute(
            """INSERT INTO semantic_cache(tenant_id, model, embedding, response, expires_at)
               VALUES (%s,%s,%s::vector,%s, now() + interval '1 hour')""",
            (tenant_id, model, emb, response))
```

**Expected result:** First `/chat` call → miss + model call; identical second call → `exact` hit; a *paraphrase* of the same question → `semantic` hit (if ≥0.95).

**Verify (the no-cross-tenant proof — DoD):**

```python
# tests/test_cache_tenant_isolation.py
import pytest
from app import cache
@pytest.mark.asyncio
async def test_tenant_a_never_gets_b(monkeypatch):
    msgs = [{"role": "user", "content": "what is the capital of France"}]
    await cache.put("B", "chat-default", msgs, "Paris-SECRET-B")
    hit_a, layer = await cache.get("A", "chat-default", msgs)   # SAME question, tenant A
    assert hit_a is None            # A must NOT see B's cached answer (exact or semantic)
    hit_b, _ = await cache.get("B", "chat-default", msgs)
    assert hit_b == "Paris-SECRET-B"
```

Report the hit rate over a repeated workload: increment `cache:stats:{hits,misses}` counters in Redis and expose them at `/admin/metrics`.

**Troubleshoot:**
- Cross-tenant hit slips through → you forgot `tenant_id` in the SQL `WHERE` *or* the exact key. Both must be tenant-scoped. The test above catches it.
- Everything is a semantic hit → threshold too low. Confidently-wrong answers to similar-but-different questions are the classic bug; keep 0.95+ and only loosen with eval evidence (lecture 10).
- Embedding dim mismatch → `nomic-embed-text` is 768; match your `vector(768)` column.

---

### Step 5 — Cheap-first cascade (`routing.py`)

**What:** Send to a small/cheap model first; if a validation/confidence check fails (invalid JSON or low self-rated score), escalate to the strong model. Log which tier answered and the per-tier success rate over ≥50 inputs.

**Why:** Most requests don't need the flagship model. A cascade serves the easy 70% cheaply and escalates only the hard cases — measured, not guessed.

**Do it:**

```python
# app/routing.py
import json
from app.gateway import router
CHEAP = "ollama/llama3.1"          # or gpt-4o-mini
STRONG = "openai/gpt-4o"           # or a bigger local model

def _valid_json(text):
    try: json.loads(text); return True
    except Exception: return False

async def cascade(messages, want_json=True):
    for tier, model in (("cheap", CHEAP), ("strong", STRONG)):
        resp = await router.acompletion(model=model, messages=messages)
        out = resp.choices[0].message.content
        ok = _valid_json(out) if want_json else (len(out.strip()) > 0)
        if ok:
            return {"tier": tier, "output": out, "escalated": tier == "strong"}
    return {"tier": "strong", "output": out, "escalated": True}   # last tier wins regardless
```

Harness over 50 inputs and print the breakdown:

```python
# tests/eval_cascade.py  (run as a script)
import asyncio, collections
from app.routing import cascade
INPUTS = [{"role":"user","content": f'Return JSON {{"n":{i}}} and nothing else'} for i in range(50)]
async def main():
    tally = collections.Counter()
    for m in INPUTS:
        r = await cascade([m]); tally[r["tier"]] += 1
    print(dict(tally))   # e.g. {'cheap': 41, 'strong': 9} -> 82% served cheap
asyncio.run(main())
```

**Expected result:** A printed per-tier breakdown over 50 inputs, e.g. `{'cheap': 41, 'strong': 9}`, plus a per-tier success rate. Log `route_tier` into `request_metrics`.

**Verify:** Re-run with a stricter validator (e.g. require a specific schema) — escalation rate rises, proving the check drives the routing.

**Troubleshoot:**
- Cheap tier always escalates → your validator is too strict or the cheap model can't do the task; loosen the check or pick a better cheap model.
- No measured quality delta → add a reference/golden answer per input and compare tier outputs against it (ties back to Phase 7 evals).

---

### Step 6 — Token-aware rate limit + spend kill-switch (`limits.py`)

**What:** A Redis token-bucket keyed by `tenant_id` refilling at `tokens/min`; reject with **429** when empty. A monthly `spend_ledger` (input+output tokens × price from `model_prices`); a request that crosses the tenant's monthly cap returns **402** and flips the `killed` flag.

**Why:** The unit of cost is the *token*, not the request — a "100 req/min" limit does nothing against one 100k-token prompt ([lecture 11](../lectures/11-rate-limiting-and-spend-governance.md)). 429 = "too fast, retry later"; 402 = "out of money, stop." Two different answers to two different questions.

**Do it:**

```python
# app/limits.py
import os, time, tiktoken
import redis.asyncio as redis
from fastapi import HTTPException
from app.deps import pg_pool
r = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
enc = tiktoken.get_encoding("cl100k_base")

# Atomic token-bucket in Lua: refill by elapsed*rate, then try to spend `cost`.
BUCKET_LUA = """
local key=KEYS[1]; local rate=tonumber(ARGV[1]); local cap=tonumber(ARGV[2])
local cost=tonumber(ARGV[3]); local now=tonumber(ARGV[4])
local t=redis.call('HMGET',key,'tokens','ts')
local tokens=tonumber(t[1]); local ts=tonumber(t[2])
if tokens==nil then tokens=cap; ts=now end
tokens=math.min(cap, tokens + (now-ts)*rate)
if tokens < cost then redis.call('HMSET',key,'tokens',tokens,'ts',now); return 0 end
tokens=tokens-cost; redis.call('HMSET',key,'tokens',tokens,'ts',now)
redis.call('EXPIRE',key,120); return 1
"""

async def check_rate(tenant_id, est_tokens, per_min=20000):
    rate = per_min / 60.0
    ok = await r.eval(BUCKET_LUA, 1, f"bucket:{tenant_id}", rate, per_min, est_tokens, time.time())
    if not ok:
        raise HTTPException(429, "rate limit: token bucket empty")

async def check_killswitch(tenant_id):
    async with pg_pool.connection() as con:
        row = await (await con.execute(
            "SELECT killed, monthly_cap_usd FROM tenant_limits WHERE tenant_id=%s", (tenant_id,))).fetchone()
    if row and row[0]:
        raise HTTPException(402, "monthly spend cap reached — kill-switch active")

async def record_spend(tenant_id, model, in_tok, out_tok):
    month = time.strftime("%Y-%m")
    async with pg_pool.connection() as con:
        price = await (await con.execute(
            "SELECT input_per_1k, output_per_1k FROM model_prices WHERE model=%s", (model,))).fetchone()
        usd = (in_tok/1000)*float(price[0]) + (out_tok/1000)*float(price[1])
        await con.execute("INSERT INTO spend_ledger(tenant_id,month,usd) VALUES (%s,%s,%s)",
                          (tenant_id, month, usd))
        total = await (await con.execute(
            "SELECT COALESCE(SUM(usd),0) FROM spend_ledger WHERE tenant_id=%s AND month=%s",
            (tenant_id, month))).fetchone()
        cap = await (await con.execute(
            "SELECT monthly_cap_usd FROM tenant_limits WHERE tenant_id=%s", (tenant_id,))).fetchone()
        if cap and float(total[0]) >= float(cap[0]):
            await con.execute("UPDATE tenant_limits SET killed=true WHERE tenant_id=%s", (tenant_id,))
```

Call `check_killswitch` then `check_rate(tenant_id, len(enc.encode(prompt)))` at the top of `/chat`, and `record_spend(...)` after the call using the response's real token usage.

**Expected result:** A burst from tenant A gets 429 while tenant B (own bucket) is unaffected. After A crosses a small test cap (set `monthly_cap_usd = 0.001`), A gets 402 and `tenant_limits.killed = true`.

**Verify (DoD tests):**
```python
# tests/test_rate_limit_fairness.py — hammer A, prove B unaffected
# tests/test_spend_killswitch.py — set cap tiny, run one call, assert 402 + killed flag
```
```bash
# quick manual fairness check
for i in $(seq 1 50); do curl -s -o /dev/null -w "%{http_code} " localhost:8000/chat -d '{"tenant_id":"A","message":"x"}'; done  # sees 429s
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/chat -d '{"tenant_id":"B","message":"x"}'                                 # 200
```

**Troubleshoot:**
- Both tenants throttled together → your bucket key isn't per-tenant. Key must be `bucket:{tenant_id}`.
- Kill-switch never trips → you estimate spend from input tokens only; charge on *actual* `resp.usage` after the call (output tokens unknown until then — lecture 11).
- Race under concurrency → the Lua script is atomic; don't replace it with read-modify-write in Python.

---

### Step 7 — Secrets & per-tenant BYOK (`secrets.py`)

**What:** Resolve provider keys from env/Vault **server-side only**; support per-tenant BYOK stored encrypted. Assert (grep test) no key appears in any response or log.

**Why:** A key that exists anywhere a human or a log can copy it *will* leak ([lecture 12](../lectures/12-secrets-and-key-management.md)). The rule: a provider key never appears client-side, in a response body, or in a log line.

**Do it:**

```python
# app/secrets.py
import os
from cryptography.fernet import Fernet         # uv add cryptography
from app.deps import pg_pool
_FERNET = Fernet(os.environ["BYOK_ENC_KEY"])   # 32-byte urlsafe base64; server-side only

async def resolve_key(tenant_id, provider):
    """Per-tenant BYOK if present (decrypted in memory), else the platform env key."""
    async with pg_pool.connection() as con:
        row = await (await con.execute(
            "SELECT enc_key FROM tenant_byok WHERE tenant_id=%s AND provider=%s",
            (tenant_id, provider))).fetchone()
    if row and row[0]:
        return _FERNET.decrypt(row[0].encode()).decode()   # never logged, never returned
    return os.environ.get(f"{provider.upper()}_API_KEY")

def redact(text: str) -> str:
    import re
    return re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***REDACTED***", text or "")
```

Route all logging through `redact()` and never put a key in a JSON response. Store BYOK via an admin endpoint that encrypts before insert: `enc = _FERNET.encrypt(key.encode()).decode()`.

**Expected result:** BYOK tenants use their own key; others use the platform key. Logs and responses show `sk-***REDACTED***` at most.

**Verify (grep test — DoD):**

```python
# tests/test_no_secret_leak.py
import re, glob, pytest, httpx
KEY_RE = re.compile(r"sk-[A-Za-z0-9]{20,}")
@pytest.mark.asyncio
async def test_no_key_in_response_or_logs(capfd):
    async with httpx.AsyncClient(app=app, base_url="http://t") as c:
        resp = await c.post("/chat", json={"tenant_id":"A","message":"echo my key"})
    assert not KEY_RE.search(resp.text)                 # not in response body
    out, err = capfd.readouterr()
    assert not KEY_RE.search(out + err)                 # not in stdout/stderr logs
    for f in glob.glob("logs/*.log"):
        assert not KEY_RE.search(open(f).read())        # not in log files
```

**Troubleshoot:**
- Key shows up in a LiteLLM debug log → set `litellm.set_verbose = False` in prod and add key patterns to your log redactor.
- `Fernet` key error → `BYOK_ENC_KEY` must be a urlsafe base64 32-byte key: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.

---

## Putting it together — end-to-end run

Wire the `/chat` handler so the layers compose in order: **killswitch → rate limit → cache → cascade/gateway → spend**.

```python
# app/chat.py (gateway-aware handler, sketch)
@app.post("/chat")
async def chat(body: ChatIn):
    await check_killswitch(body.tenant_id)                       # 402 if capped
    await check_rate(body.tenant_id, len(enc.encode(body.message)))  # 429 if too fast
    messages = [{"role": "user", "content": body.message}]
    hit, layer = await cache.get(body.tenant_id, "chat-default", messages)
    if hit:
        return {"answer": hit, "cache": layer}
    resp = await complete(messages, tenant_id=body.tenant_id)    # router+fallback+breaker
    answer = resp.choices[0].message.content
    await cache.put(body.tenant_id, "chat-default", messages, answer)
    u = resp.usage
    await record_spend(body.tenant_id, "gpt-4o-mini", u.prompt_tokens, u.completion_tokens)
    return {"answer": answer, "cache": "none"}
```

Run the full demo:

```bash
uvicorn app.main:app --reload &
# 1) normal call
curl -s localhost:8000/chat -d '{"tenant_id":"A","message":"capital of France?"}'
# 2) kill the primary: set OPENAI_API_KEY=bad, restart, fire 3 -> breaker opens, Ollama answers
curl -s localhost:8000/gateway/health
# 3) stream + disconnect
curl -N "localhost:8000/chat/stream?tenant_id=A&message=tell%20me%20a%20story"
# 4) cascade eval
python tests/eval_cascade.py
# 5) fairness + killswitch + secret-leak
pytest -q tests/test_rate_limit_fairness.py tests/test_spend_killswitch.py \
          tests/test_cache_tenant_isolation.py tests/test_stream_cancel.py \
          tests/test_no_secret_leak.py
```

---

## Definition of Done — verifiable checks

- [ ] **Fallback + breaker.** Killing the primary (bad key or blocked network) → requests still succeed via Ollama; `GET /gateway/health` shows the breaker **open** for the primary. (Steps 1–2)
- [ ] **SSE + TTFT + cancel.** `/chat/stream` streams tokens; `ttft_ms` is logged per request in `request_metrics`; `tests/test_stream_cancel.py` proves upstream token production **stops** after client disconnect. (Step 3)
- [ ] **Two-layer cache.** A repeated workload reports a hit rate; `tests/test_cache_tenant_isolation.py` proves tenant A never gets tenant B's answer. (Step 4)
- [ ] **Cascade.** ≥50 inputs show a per-tier answer breakdown and a measured quality delta between tiers; `route_tier` is logged. (Step 5)
- [ ] **Rate limit fairness.** A burst from tenant A returns 429 while tenant B is unaffected. (Step 6)
- [ ] **Spend kill-switch.** After crossing a small test monthly cap, requests return 402 and the ledger/`killed` flag explains why. (Step 6)
- [ ] **No secret leak.** `tests/test_no_secret_leak.py` greps responses and logs and finds no key. (Step 7)

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Fallback never triggers | Auth error not treated as retriable / no second deployment | Ensure Ollama deployment shares `model_name`; verify by killing network |
| Breaker never opens | Fallback success swallows primary failure | Record failure in a LiteLLM failure callback on the *deployment*, not the top call |
| TTFT ≈ total time | Proxy buffering SSE | `proxy_buffering off` / `X-Accel-Buffering: no` |
| Tokens count after disconnect | `finally`/`aclose()` not reached | `break` (not `return`) in the `async for` so `finally` runs |
| Cross-tenant cache hit | `tenant_id` missing from key or SQL `WHERE` | Tenant-scope both L1 key and L2 query |
| Everything a semantic hit | Threshold too low | Keep ≥0.95; loosen only with eval evidence |
| Both tenants throttled together | Bucket not per-tenant | Key `bucket:{tenant_id}` |
| Kill-switch never trips | Charging on input tokens only | Charge on `resp.usage` (real output tokens) after the call |
| Key in logs | LiteLLM verbose logging | `litellm.set_verbose=False`; run all logs through `redact()` |
| pgvector errors | Wrong image / dim mismatch | `pgvector/pgvector:pg16`; `vector(768)` = nomic-embed-text |

---

## Stretch goals (optional)

- **Provider prompt caching.** Add Anthropic `cache_control` / OpenAI automatic prefix caching for the *prefill* and measure the TTFT/cost delta on top of your two app-layer caches (lecture 10).
- **Request coalescing.** Dedupe identical in-flight requests (single-flight) so a thundering herd on one prompt makes one upstream call — a Week 3 degradation lever.
- **Breaker alerting.** Emit a log/metric on every breaker-open transition so a silent fallback can't hide a real outage (lecture 07 trap).
- **BYOK rotation.** Add a rotate endpoint that re-encrypts under a new `BYOK_ENC_KEY` with zero downtime (lecture 12).
- **Half-open probe policy.** Send a single canary request in half-open instead of full traffic; close only on canary success.
