# Phase 09 - AI Application Architecture & System Design

LLM apps live or die on architecture, not prompts. Routing, caching, streaming, state, fallback, rate limits, and observability decide your cost, latency, reliability, and whether you can iterate at all. By the end of this phase you can draw the five reference architectures from memory, build a resilient LLM gateway (provider fallback, circuit breakers, per-tenant rate limits + spend kill-switch, two-layer cache, degradation mode), prove a GDPR "delete this user" purges *every* store including caches and the vector index, and pass a timed system-design interview with QPS/token/cost math and a failure story.

Nav: Prev: `08-fine-tuning.md` · Next: `10-llmops-serving.md`

## Prerequisites
- Phase 2: structured outputs / tool calling and treating model output as untrusted input.
- Phase 3-4: a working retrieval/RAG stack (you'll wire it behind the gateway and delete from its index).
- Phase 7: tracing + evals mindset - you design *for* evaluability here, so you need to know what a trace and a golden set are.
- Solid HTTP/async fundamentals: `async`/`await`, an ASGI server, status codes, streaming responses, idempotency keys.
- Comfort with **FastAPI**, **Redis**, and **PostgreSQL** at a basic level, plus Docker Compose to run them locally.
- One provider key (OpenAI/Anthropic/Gemini) *or* a local **Ollama** install so nothing here forces a paid API.

## Time budget
3 weeks x ~10-15 hrs/week (~35% theory / 65% hands-on). Each week states its own hours breakdown. Short on time? Do every **Lab** and the milestone; the reading is calibration, the builds are the point.

## How to use this file
Work top to bottom; each week's Definition of Done is a hard gate - don't advance until every box is checked. Keep ONE git repo (`llm-gateway-lab/`) for the whole phase; Week 1 stands up state + GDPR delete, Week 2 builds the gateway, Week 3 adds observability + degradation and produces the three mock designs. The milestone is what you demo in interviews.

> **📚 Course materials.** This file is the **spine** — a weekly plan and a *recap*. Deep material lives alongside it:
> - **Lectures**: [`lectures/`](lectures/00-index.md) — read for the *why*/mechanism.
> - **Lab guides**: [`labs/`](labs/) — follow to *build*.
>
> Each week: read the linked lectures first (the Theory here is their recap), then work the linked lab guide.

---

## Week 1 - Reference architectures, state, and provable GDPR delete

### Objectives
By the end of the week you can:
- Draw the five reference architectures (single-turn, stateful chatbot, RAG app, tool-using agent, in-product copilot) and name what state each one keeps and where.
- Explain and enforce the boundary "**deterministic code owns flow; the LLM sits at the leaves**" in a real FastAPI service.
- Design a four-tier storage layout for conversations (hot Redis + durable Postgres + object storage + vector memory) and justify what lives where and its TTL.
- Implement idempotent multi-turn conversation storage with context-window management (rolling summary + verbatim entity/ID preservation).
- Ship a **GDPR cascade delete** that provably removes a user from *every* store including Redis caches and the vector index, verified by an automated test.

### Theory (~4 hrs)
> 📚 Lectures for this week: [1. The Five Reference Architectures and the Escalation Ladder](lectures/01-five-reference-architectures.md) · [2. LLM Proposes, Code Disposes: The Deterministic Boundary](lectures/02-deterministic-code-vs-llm-leaves.md) · [3. Four-Tier State: Redis, Postgres, Object Storage, Vector](lectures/03-state-and-storage-tiers.md) · [4. Context-Window Management: Windowing, Compaction, and Retrieval Memory](lectures/04-context-window-management.md) · [5. Right-to-Erasure as an Architecture Constraint: Provable Cascade Delete](lectures/05-gdpr-erasure-as-architecture.md). The bullets below are the recap.
- **The five reference architectures.** Read Anthropic's "Building effective agents" (search: "Anthropic building effective agents") for the workflow-vs-agent spine, and skim the OpenAI Cookbook's assistant/RAG examples. Internalize the escalation ladder: single-turn feature -> stateful chatbot -> RAG app -> tool-using agent -> copilot. Each step adds state and failure modes; use the *lowest* rung that meets the requirement.
- **Deterministic code vs LLM calls.** The governing rule from the roadmap: "LLM proposes, code disposes." Control flow, validation, business rules, math, and lookups are deterministic code; the model only does fuzzy language work at the leaves. This is what makes the system testable and debuggable - you can unit-test the flow without calling a model.
- **State & session storage tiers.** Why four stores, not one: **Redis** = hot session/turn buffer + rate-limit counters + cache (ms latency, TTL'd, *volatile*); **Postgres** = durable source of truth for messages, users, tenants, spend ledger (survives restarts, queryable); **object storage** (S3/MinIO) = large blobs, uploaded files, raw request/response archives; **vector store** (pgvector/Qdrant) = long-term semantic memory + RAG index. Read the Redis docs on `EXPIRE`/`TTL` and the pgvector README.
- **Context-window management.** A chatbot can't send the full history forever. Strategies: sliding window (keep last N turns), rolling summarization/compaction (summarize old turns, but **preserve entities/IDs/amounts verbatim** - a summary that drops an order number is a bug), and retrieval memory (embed old turns, retrieve relevant ones). Trigger compaction at a token threshold, not a turn count.
- **GDPR / right-to-erasure as an architecture constraint.** Article 17 erasure must cascade to *derived* copies: caches, embeddings, search indexes, log archives, backups. The classic failure is deleting the Postgres row but leaving the user's text in the semantic cache and the vector index. Read the GDPR Art. 17 summary (search: "GDPR Article 17 right to erasure") - engineering-relevant intuition only, not legal depth.

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [Stand Up the Four Stores, Stateful Chat, and Provable GDPR Delete](labs/week-1-state-and-gdpr-delete.md). The steps below are the summary.
Stand up the repo, the four stores, and a stateful chat service with provable delete.

```
llm-gateway-lab/
  docker-compose.yml        # redis, postgres, minio, (optional) qdrant
  pyproject.toml            # uv-managed
  app/
    main.py                 # FastAPI app
    deps.py                 # store clients (redis, pg, s3, vector)
    chat.py                 # stateful chat endpoint
    memory.py               # session buffer + compaction
    llm.py                  # thin LLM call wrapper (leaf only)
    gdpr.py                 # cascade delete
    models.py               # Pydantic + SQL schemas
  tests/
    test_chat.py
    test_gdpr_delete.py
  README.md
```

1. **Env + stores.** `uv init && uv add fastapi uvicorn[standard] redis "psycopg[binary,pool]" pydantic boto3 openai litellm`. Write `docker-compose.yml` with `redis:7`, `postgres:16`, `minio/minio`, and optionally `qdrant/qdrant`. `docker compose up -d`. Create Postgres tables: `users`, `conversations`, `messages`, `spend_ledger`. Enable `pgvector` (`CREATE EXTENSION vector`) and a `memories(user_id, embedding vector(768), text)` table for semantic memory.
2. **Thin LLM leaf.** In `llm.py`, expose ONE `async def complete(messages, model) -> str`. Everywhere else calls this; no business logic inside. Point it at Ollama (`model="ollama/llama3.1"` via LiteLLM) so it runs free/local, or a provider key.
3. **Stateful chat.** `POST /chat` takes `{tenant_id, user_id, conversation_id, message, idempotency_key}`. Flow (all deterministic except step d): (a) dedup on `idempotency_key` in Redis - return the prior response if seen; (b) load recent turns from Redis, fall back to Postgres on miss; (c) if buffered tokens > threshold, compact old turns into a summary via `memory.py`, keeping IDs/amounts verbatim; (d) call `llm.complete(...)` - the ONLY model call; (e) persist the turn to Postgres (durable) and push to Redis (hot, TTL 1h); (f) return.
4. **Idempotency test.** Fire the same `idempotency_key` twice; assert exactly one Postgres message row and one model call (mock/count it).
5. **GDPR cascade delete.** `DELETE /users/{user_id}` in `gdpr.py` must, in order: delete Postgres `messages`+`conversations`+`memories`+`spend_ledger` rows; `SCAN`+`DEL` all Redis keys namespaced to the user (sessions, cache entries, dedup keys); delete the vector rows for `user_id` (pgvector `DELETE` or Qdrant filtered delete); delete object-storage prefixes for the user. Return a JSON receipt listing per-store counts deleted.
6. **Prove it.** `test_gdpr_delete.py`: seed a user with messages, a cached response containing their text, N vector memories, and an object; call delete; then assert every store returns zero for that user - including a semantic search that must return no user-owned vectors.

Free/cheap path: everything runs in Docker on a laptop; Ollama covers the model. No paid API required.

### Definition of Done
- [ ] `docker compose up` brings up all four stores; `GET /health` reports each store reachable.
- [ ] `POST /chat` holds a coherent 10-turn conversation; turn 11 triggers compaction and the summary still contains every ID/amount from earlier turns (assert in a test).
- [ ] Idempotency: same key twice -> exactly one persisted message and one model call.
- [ ] `test_gdpr_delete.py` is green: after delete, Postgres, Redis (SCAN finds 0 keys), vector store (0 rows / 0 search hits), and object storage all return zero for the user.
- [ ] The delete endpoint returns a receipt with per-store deletion counts.
- [ ] A one-paragraph note in `README.md` naming, for each store, *what* lives there and *why* (the tier justification).

### Pitfalls
- Deleting the Postgres row but forgetting the vector index and semantic cache - the user's text is still retrievable. This is the #1 real GDPR bug.
- Using `KEYS *` in Redis to find user keys (blocks the server); use `SCAN` with a per-user key prefix designed up front.
- Compaction that summarizes away order numbers, dates, or amounts - always copy entities/IDs verbatim into the summary.
- Business logic creeping into `llm.py` - the moment you branch on model output *inside* the call wrapper, you've lost the deterministic boundary.
- No idempotency key -> a client retry double-charges the user and double-writes history.

### Self-check
1. For each of the four stores, name one thing that belongs there and one thing that must NOT.
2. Why is "delete the user row" insufficient for GDPR, and which stores are most often missed?
3. When do you compact context by summary vs by retrieval memory?
4. What breaks if the LLM call owns control flow instead of deterministic code?

---

## Week 2 - The resilient LLM gateway: routing, fallback, caching, limits

### Objectives
By the end of the week you can:
- Put a unified gateway in front of >=2 providers (or provider + local model) with a **fallback chain** and a **circuit breaker** that trips on repeated failures.
- Stream tokens to a client via **SSE**, measure **TTFT**, and cancel the request so cancellation **aborts the upstream provider call** (stops the token meter).
- Implement a **cheap-first cascade** / task router and measure quality-per-tier.
- Add a **two-layer cache** (exact-match + per-tenant semantic) that never crosses auth boundaries.
- Enforce **token-aware per-tenant rate limits** and a **hard monthly spend kill-switch**, and manage keys/secrets so nothing is client-side.

### Theory (~4 hrs)
> 📚 Lectures for this week: [6. The LLM Gateway: Unified Router, Provider Abstraction, and Leaky Seams](lectures/06-gateway-router-pattern.md) · [7. Resilience: Fallback Chains and Circuit Breakers](lectures/07-circuit-breakers-and-fallback.md) · [8. Streaming Tokens: SSE vs WebSocket, TTFT, and Upstream Cancellation](lectures/08-sse-streaming-ttft-cancellation.md) · [9. Async and Queue-Based Inference for Long and Batch Jobs](lectures/09-async-queue-based-inference.md) · [10. Three Caching Layers and the Tenant-Isolation Iron Rule](lectures/10-caching-layers.md) · [11. Token-Aware Rate Limiting and the Hard Spend Kill-Switch](lectures/11-rate-limiting-and-spend-governance.md) · [12. Secrets, Provider Keys, and Per-Tenant BYOK](lectures/12-secrets-and-key-management.md). The bullets below are the recap.
- **Gateway / router pattern.** Read the **LiteLLM** docs (Router, fallbacks, budgets) and skim **Portkey** and **Cloudflare AI Gateway** feature pages to see the buy-side of build-vs-buy. The gateway gives you one OpenAI-compatible interface, provider abstraction, retries, fallbacks, and cost tracking. Know where the abstraction *leaks*: token counting, prompt-cache semantics, and tool-call formats differ per provider.
- **Circuit breakers & fallback.** A fallback chain retries the next provider on error; a circuit breaker stops hammering a provider that's failing (open -> half-open -> closed states) so you fail fast instead of piling up latency. Read the classic "Circuit Breaker" pattern (search: "Martin Fowler CircuitBreaker") for the intuition.
- **Streaming: SSE vs WebSocket, TTFT, cancellation.** SSE (one-way server->client over HTTP) is the right default for token streaming - simpler, proxy-friendly, auto-reconnect. WebSocket only when you need bidirectional (voice, live collaboration). **TTFT** (time to first token) is the latency users feel; measure it separately from total time. The critical correctness bug: when the client disconnects, you must **abort the upstream provider call**, or you keep paying for tokens nobody reads. Also watch **proxy buffering** (nginx/Cloudflare can buffer SSE and destroy TTFT - disable it).
- **Async & queue-based inference.** For long/batch jobs, accept the request, enqueue it (Celery/Redis or SQS/BullMQ), return a job id, and let idempotent workers process with a dead-letter queue for poison messages. Batch APIs give big discounts for non-interactive work.
- **Caching layers.** Three kinds, stacked: **exact-match** (hash of normalized prompt -> response), **semantic** (embed the query, return a cached answer if cosine similarity > threshold - see **GPTCache**), and **provider prompt caching** (Anthropic `cache_control`, OpenAI automatic prefix cache) which caches the *prefill* server-side. Iron rule: **never serve a cache hit across an auth/tenant boundary** - key every cache entry by tenant.
- **Rate limiting & cost governance.** Token-aware limits (a request can be 10 or 10k tokens - limit tokens, not just request count); per-tenant quotas; and a **hard spend kill-switch** that refuses requests once a tenant crosses its monthly cap. Read the Redis rate-limiting patterns (fixed/sliding window, token bucket).
- **Secrets/keys.** Never client-side. Env/Vault/Secrets Manager server-side, rotation, and per-tenant BYOK if tenants bring their own keys. Skim the HashiCorp Vault intro.

### Lab (~10 hrs)
> 🛠️ Full step-by-step guide: [Build the Resilient Gateway: Router, Fallback, Streaming, Cache, Limits](labs/week-2-resilient-gateway.md). The steps below are the summary.
Build the gateway on top of Week 1's repo.

```
app/
  gateway.py        # router + fallback + circuit breaker
  stream.py         # SSE endpoint with upstream cancellation
  cache.py          # exact + semantic cache, tenant-keyed
  limits.py         # token-aware rate limit + spend kill-switch
  routing.py        # cheap-first cascade / task router
  secrets.py        # server-side key resolution, per-tenant BYOK
```

1. **Router + fallback.** In `gateway.py`, use `litellm.Router` with a model list mapping a logical name (e.g. `chat-default`) to an ordered list: primary provider -> secondary provider -> local Ollama. Configure `fallbacks` and `num_retries`. Wrap it so a provider 5xx/timeout trips to the next.
2. **Circuit breaker.** Track per-provider failures in Redis (rolling count). Open the breaker (skip that provider, go straight to fallback) after N failures in a window; half-open after a cooldown. Expose breaker state at `GET /gateway/health`.
3. **SSE streaming with cancellation.** `GET /chat/stream` returns `text/event-stream`. Stream provider tokens as SSE events. Use FastAPI's request-disconnect detection (`await request.is_disconnected()` or a cancellation task) to **cancel the upstream generator** - assert with a test/log that upstream stops producing tokens after client disconnect. Record TTFT and total time per request into Postgres.
4. **Two-layer cache.** `cache.py`: exact-match key = `sha256(tenant_id + model + normalized_messages)`. On miss, embed the last user message and check a per-tenant semantic cache (pgvector or GPTCache) with a cosine threshold (start 0.95). Store responses with TTL. **Every key includes `tenant_id`** - write a test that proves tenant A never gets tenant B's cached answer.
5. **Cheap-first cascade.** `routing.py`: send to a small/cheap model first; if a confidence/validation check fails (e.g. output isn't valid JSON, or a self-rated score is low), escalate to the strong model. Log which tier answered and the per-tier success rate over 50 inputs.
6. **Token-aware limits + spend kill-switch.** `limits.py`: a Redis token-bucket keyed by `tenant_id` refilling at a tokens/min rate; reject with HTTP 429 when empty. Maintain a monthly `spend_ledger` (input+output tokens x price); a request that would exceed the tenant's monthly cap returns HTTP 402 and flips a kill-switch flag. Prices live in a small config table.
7. **Secrets.** `secrets.py` resolves provider keys from env/Vault server-side only; support per-tenant BYOK (tenant supplies their key, stored encrypted, used for their calls). Assert no key is ever returned in an API response or log.

Free/cheap path: run one real provider + Ollama as the fallback so you can test fallback by killing the provider (or pointing it at a bad key). GPTCache and pgvector are free; no paid infra needed.

### Definition of Done
- [ ] Killing the primary provider (bad key or blocked network) causes requests to succeed via fallback; the breaker opens and `GET /gateway/health` shows it.
- [ ] SSE endpoint streams tokens; TTFT is logged per request; a client disconnect verifiably aborts the upstream call (token production stops - shown in logs/test).
- [ ] Two-layer cache reports a hit rate over a repeated workload; a test proves NO cross-tenant cache hit.
- [ ] Cascade: >=50 inputs show a per-tier answer breakdown and a measured quality delta between tiers.
- [ ] Rate limit returns 429 under burst from one tenant while a second tenant is unaffected.
- [ ] Spend kill-switch: after crossing a (small, test-set) monthly cap, requests return 402 and the ledger explains why.
- [ ] No secret appears in any response body or log line (grep the logs in a test).

### Pitfalls
- Not aborting the upstream call on client disconnect - you keep paying for tokens the user never sees. Test this explicitly.
- A proxy (nginx/Cloudflare) buffering SSE so TTFT looks terrible - disable proxy buffering for the stream route.
- Semantic-cache threshold too low -> confidently wrong cached answers to *similar-but-different* questions. Start strict (0.95+) and only loosen with eval evidence.
- Caching across tenants/auth boundaries - a data leak, not just a bug. Tenant-key every entry.
- Rate limiting by request count instead of tokens - one 100k-token request sails through a "100 req/min" limit.
- Fallback masking a real outage: if you always silently fall back, you never notice the primary is down. Alert on breaker-open, not just on total failure.

### Self-check
1. When do you choose WebSocket over SSE, and what's the cost of getting it wrong?
2. What are the three cache layers and what does each actually cache?
3. Why is a token bucket better than a fixed request-count limit for LLM traffic?
4. Describe the circuit-breaker states and what event moves between them.
5. Why must a client disconnect propagate to the provider, and how do you verify it did?

---

## Week 3 - Evaluability, observability, degradation, and system-design practice

### Objectives
By the end of the week you can:
- Instrument the gateway with end-to-end **tracing** (per-request model, tokens, cost, TTFT, cache hit, tier, tenant) and design *for* evaluability (versioned prompts/models, feedback capture).
- Build a **degradation mode** that serves cached/cheaper/queued responses during a provider outage instead of failing.
- Do capacity math: convert a QPS + token-per-request estimate into provider-quota, cost, and latency numbers, and identify that **provider quota is the real bottleneck**.
- Make a **build-vs-buy** call across the gateway stack with a written rationale.
- Complete **three timed (45-min) mock system designs** with an architecture diagram, data model, routing/caching/fallback plan, cost math, and a failure/privacy story each.

### Theory (~5 hrs)
> 📚 Lectures for this week: [13. Designing for Evaluability: Tracing, Versioning, and the Feedback Flywheel](lectures/13-evaluability-and-observability.md) · [14. Capacity Planning, Provider Quota, and Graceful Degradation](lectures/14-scalability-capacity-and-degradation.md) · [15. Build-vs-Buy and Compliance-Architecture Trust Boundaries](lectures/15-build-vs-buy-and-privacy-boundaries.md) · [16. The 45-Minute LLM System-Design Method](lectures/16-system-design-method.md) · [17. PII Redaction Across the Observability and Feedback Pipeline](lectures/17-privacy-in-observability-pipelines.md). The bullets below are the recap.
- **Designing for evaluability & observability.** Read the OpenTelemetry **GenAI semantic conventions** (search: "OpenTelemetry GenAI semantic conventions") and skim **Langfuse** or **Arize Phoenix** quickstart. The design goals: every request emits a trace with the *resolved* prompt, model+version, token counts, cost, latency/TTFT, cache-hit, routing tier, and tenant; prompts and model choices are versioned artifacts (tag the serving model in the trace) so you can attribute a quality change to a specific change; and you capture user feedback (thumbs/regenerate) linked to trace ids for the data flywheel.
- **Scalability / HA / capacity planning.** The counter-intuitive truth: your bottleneck is usually **provider rate/quota limits (TPM/RPM), not your CPU**. Techniques: request coalescing (dedupe identical in-flight requests), queueing + backpressure, multi-provider spread, and **degradation modes** (serve a cached or cheaper-model answer, or a "we're busy, queued" response, rather than a 500). Read the "graceful degradation" idea in any SRE context (search: "Google SRE graceful degradation").
- **Build-vs-buy.** When to use a managed gateway (Portkey/Cloudflare AI Gateway/LiteLLM Cloud) vs hand-roll. Rubric: control needed, data-residency/privacy, cost at your volume, team size, and *ejectability*. Default: thin self-hosted gateway (LiteLLM) + buy observability - but write down the tradeoff.
- **Privacy boundaries & compliance architecture.** Data-flow diagram with trust boundaries: what leaves your VPC, which providers have zero-retention/DPA terms, where PII is redacted (before prompt *and* before logs/traces), and private endpoints (Bedrock/Vertex/Azure OpenAI) for sensitive data. This connects back to Week 1's GDPR delete.
- **System-design method.** The interview/whiteboard loop: clarify requirements & scale -> sketch the reference architecture -> data model -> the LLM-specific layer (routing/caching/fallback/streaming) -> **back-of-envelope QPS/token/cost math** -> a **failure & degradation story** -> privacy/security. Practice narrating tradeoffs, not just drawing boxes.

### Lab (~10 hrs)
> 🛠️ Full step-by-step guide: [Observability, Degradation Mode, Load Test, Capacity Math, and Three Mock Designs (Milestone)](labs/week-3-observability-degradation-and-designs.md). The steps below are the summary.
Finish the gateway's observability + degradation, then do the timed designs.

1. **Tracing.** Add OpenTelemetry (or Langfuse SDK) spans around each request: attributes for `model`, `model_version`, `prompt_tokens`, `completion_tokens`, `usd_cost`, `ttft_ms`, `total_ms`, `cache_layer` (none/exact/semantic), `route_tier`, `tenant_id`. Send to a local Phoenix/Langfuse or just log structured JSON. Verify a single request produces one coherent trace with per-step timing.
2. **Cost/latency dashboard.** A `GET /admin/metrics` (or a small Grafana/Phoenix view) showing, per tenant and per model: request count, p50/p95 TTFT, tokens, $ spent this month, and cache hit rate. This is the one screen you'd watch in prod.
3. **Feedback + versioning.** `POST /feedback` links a thumbs-up/down + optional edit to a trace id. Store prompt template versions in a table and tag each trace with the version used, so a query can compare quality across versions.
4. **Degradation mode.** Add a system flag/state: when the breaker is open for all providers, `/chat` serves (in priority order) a fresh exact-cache hit, else a semantic-cache hit, else routes to the local Ollama model, else enqueues the request and returns a 202 "queued" with a job id. Simulate a full outage (bad keys everywhere) and show the endpoint still returns useful responses, not 500s.
5. **Load test the fairness guarantee.** Use `locust` or `k6` to hammer the gateway with one abusive tenant (high QPS) and one normal tenant; prove via the dashboard that the abusive tenant hits 429/402 while the normal tenant's p95 stays healthy. This is the milestone's "one tenant can't starve others" proof.
6. **Capacity math sheet.** In `README.md`, write the formula and a worked example: given target QPS, avg input+output tokens, and a provider's TPM/RPM limit, compute required concurrency, whether you hit the quota wall, and monthly $ at a given price. Show where caching (say 30% hit rate) and a cascade (70% served by the cheap tier) move the numbers.
7. **Three timed mock designs (45 min each, whiteboard/Markdown).** Do them under a real timer:
   - **Enterprise RAG assistant** (multi-tenant, private docs, access-controlled retrieval).
   - **Coding copilot** (in-IDE, low-latency, context harvesting, acceptance telemetry).
   - **High-volume support-triage agent** (async/queue, cheap-first cascade, HITL on actions).
   Each deliverable (in `designs/`): architecture diagram (draw.io/Excalidraw/ASCII), data model, routing+caching+fallback plan, eval+observability strategy, a QPS/token/**cost estimate**, and a **failure/privacy** section (what breaks, how it degrades, how GDPR delete works).

Free/cheap path: Phoenix/Langfuse and locust/k6 are free and local; Ollama is your always-available degradation target. No paid infra required.

### Definition of Done
- [ ] A single request produces one trace with per-step tokens, cost, TTFT, cache layer, tier, and tenant attributes.
- [ ] The metrics view shows per-tenant p50/p95 TTFT, tokens, $ this month, and cache hit rate.
- [ ] Degradation mode: with all providers "down," `/chat` still returns cached/cheap/queued responses (no 5xx) - demonstrated.
- [ ] Load test proves an abusive tenant is throttled (429/402) while a normal tenant's p95 stays within its SLO.
- [ ] Capacity sheet includes a worked QPS->tokens->quota->$ example and shows the effect of cache + cascade.
- [ ] Three mock designs exist in `designs/`, each with a diagram, data model, cost math, and a failure+privacy section, each timestamped as done in <=45 min.

### Pitfalls
- Tracing that logs the raw prompt with PII into your observability tool - redact before the span, same as before the model.
- Treating your own compute as the bottleneck; you'll hit the provider's TPM/RPM wall first. Model quota explicitly.
- A "degradation mode" that quietly serves stale cache with no signal to the user or ops - degrade *and* alert.
- System-design answers that draw boxes but skip the math - interviewers probe QPS/tokens/cost and the failure story; that's where the signal is.
- Versioning prompts in code comments instead of a real registry/table - you can't attribute a regression to a change you didn't record.

### Self-check
1. What attributes must every gateway trace carry to be useful for cost and quality debugging?
2. Why is provider quota the usual scaling bottleneck, and what three levers relieve it?
3. Describe your degradation ladder when all providers are down.
4. Walk the 45-minute system-design method end to end for a support-triage agent.
5. When would you *buy* a managed gateway instead of running LiteLLM yourself?

---

## Phase milestone project

Build and demo **one resilient multi-tenant LLM gateway plus a system-design portfolio.** This is the Phase 9 portfolio piece; it should be the thing you screen-share in an interview.

**Deliverable A - The resilient gateway (runnable).**
A FastAPI service in front of >=2 model backends (one real provider + local Ollama is fine) that provides:
- Provider **fallback chain** + **circuit breaker** with visible state.
- **SSE streaming** with TTFT measurement and upstream-cancellation on client disconnect.
- **Two-layer cache** (exact + per-tenant semantic) that never crosses tenant boundaries.
- **Cheap-first cascade** with measured per-tier quality.
- **Token-aware per-tenant rate limits** + a **hard monthly spend kill-switch**.
- A **degradation mode** (cached -> cheap-local -> queued) that returns useful answers during a total provider outage.
- End-to-end **tracing** + a per-tenant cost/latency/quality dashboard.
- Server-side secret handling with optional per-tenant BYOK.

**Deliverable B - Provable GDPR delete.**
A `DELETE /users/{id}` that cascades across Postgres, Redis (all caches + sessions), the vector index, and object storage, returns a deletion receipt, and is covered by a test proving zero residue in every store (including a semantic search that returns no user vectors).

**Deliverable C - Three timed mock system designs.**
Enterprise RAG assistant, coding copilot, and high-volume support-triage agent - each with architecture diagram, data model, routing/caching/fallback plan, eval+observability strategy, cost estimate, and a failure/privacy section (including how GDPR delete works in that design).

### Acceptance criteria
- [ ] Kill the primary provider live -> requests keep succeeding via fallback; breaker opens; dashboard shows it.
- [ ] Client disconnect during a stream verifiably stops upstream token production.
- [ ] Load test: one abusive tenant is throttled (429/402) while a second tenant's p95 latency stays within SLO - "no tenant starves another" proven.
- [ ] Spend kill-switch trips at a configured monthly cap and returns 402 with a ledger explanation.
- [ ] Cache hit rate reported; a test proves no cross-tenant cache hit.
- [ ] Total-outage simulation: `/chat` returns cached/cheap/queued responses, never a 5xx.
- [ ] GDPR delete test green: zero residue in Postgres, Redis, vector store, and object storage.
- [ ] Three designs present, each with a diagram, cost math, and a failure+privacy section, each done under a 45-min timer.

### Suggested repo layout
```
llm-gateway-lab/
  docker-compose.yml            # redis, postgres, minio, qdrant, (langfuse/phoenix)
  pyproject.toml
  app/
    main.py
    deps.py
    chat.py                     # stateful chat + idempotency
    memory.py                   # session buffer + compaction
    llm.py                      # thin leaf-only LLM wrapper
    gateway.py                  # router + fallback + circuit breaker
    stream.py                   # SSE + upstream cancellation
    cache.py                    # exact + per-tenant semantic
    limits.py                   # token bucket + spend kill-switch
    routing.py                  # cheap-first cascade
    secrets.py                  # server-side keys + per-tenant BYOK
    gdpr.py                     # cascade delete + receipt
    tracing.py                  # OTel/Langfuse spans
    models.py                   # Pydantic + SQL schemas
  tests/
    test_chat.py
    test_idempotency.py
    test_cache_tenant_isolation.py
    test_stream_cancel.py
    test_rate_limit_fairness.py
    test_spend_killswitch.py
    test_gdpr_delete.py
  load/
    locustfile.py               # abusive-vs-normal tenant scenario
  designs/
    01-enterprise-rag.md
    02-coding-copilot.md
    03-support-triage-agent.md
    diagrams/
  README.md                     # store tiers, capacity math, build-vs-buy note
```

## You are ready to move on when...
- [ ] You can draw the five reference architectures from memory and say what state each keeps and where.
- [ ] You keep deterministic control flow and LLM calls cleanly separated, and can explain why that makes the system testable.
- [ ] You've built a gateway with fallback, circuit breaker, streaming+cancellation, two-layer tenant-safe cache, cascade, rate limits, and a spend kill-switch - all demonstrated, not asserted.
- [ ] You have a degradation mode that survives a total provider outage without 5xxing users.
- [ ] Your GDPR delete provably purges a user from *every* store including the vector index and semantic cache.
- [ ] You can size a workload (QPS -> tokens -> provider quota -> $) on a whiteboard and name provider quota as the usual bottleneck.
- [ ] You've done three timed system designs end to end, each with cost math and a failure/privacy story.
- [ ] You can defend a build-vs-buy decision for each layer of the stack.

Next: `10-llmops-serving.md` - serving open models yourself (vLLM, continuous batching, multi-LoRA), GPU/latency/cost economics, and CI/CD for prompts and models.
