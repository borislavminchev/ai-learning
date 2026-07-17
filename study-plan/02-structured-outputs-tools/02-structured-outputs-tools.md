# Phase 2 — Structured Outputs & Function / Tool Calling

Reliable machine-parseable output and tool invocation is the backbone of every agent, RAG pipeline, and LLM-in-a-loop you will ever ship. This phase turns "the model usually returns JSON" into a guaranteed, validated, observable contract, and turns "the model called a tool" into a hardened, secure, provider-agnostic loop. Get this wrong and everything downstream — agents, RAG, extraction services — is brittle. Get it right and you have the two primitives that 90% of production LLM systems are built from.

Prev: 01-prompting-context.md · Next: 03-embeddings-vector-dbs.md

## Prerequisites
- Phase 0 (tokenization, sampling params, chat message roles, `uv` env, dotenv secrets) and Phase 1 (prompt anatomy, provider idioms, prompt caching, a basic eval mindset).
- Working API keys for **OpenAI** and **Anthropic** (both have cheap models); **Gemini** free-tier key is enough for the third provider. Total spend for this phase should be under ~$5 if you use `gpt-4o-mini` / `claude-3-5-haiku` / `gemini-2.0-flash` class models.
- Python 3.11+, comfort with `pytest`, type hints, and async/`httpx`. A little TypeScript for the Zod section (optional but recommended).
- Ollama installed locally for the constrained-decoding lab (free, no GPU strictly required for a 1–3B model; a GPU helps).

## Time budget
**2 weeks × ~10–15 hrs/week (~25 hrs total).** Weighted ~35% theory / 65% hands-on. Each week states its own hours breakdown. If you are tight on time, the labs are the non-negotiable part — skim theory, but ship the code and hit every Definition of Done.

## How to use this file
Work top-to-bottom, one week at a time. Do the Theory reading first (it's short and load-bearing), then spend the bulk of your hours in the Lab building the exact artifact described. Treat every **Definition of Done** as a merge gate: if a box isn't checked, you're not done. The two weekly labs compose into the **Phase milestone project** at the end — build them in the same repo.

> **📚 Course materials.** This file is the **spine** — a weekly plan and a *recap*. Deep material lives alongside it:
> - **Lectures**: [`lectures/`](lectures/00-index.md) — read for the *why*/mechanism.
> - **Lab guides**: [`labs/`](labs/) — follow to *build*.
>
> Each week: read the linked lectures first (the Theory here is their recap), then work the linked lab guide.

---

## Week 1 — Structured Outputs: from prompt-and-pray to schema-constrained contracts

**Hours breakdown (~13 hrs):** Theory ~4.5 hrs · Lab ~8 hrs · Self-check/notes ~0.5 hr.

### Objectives
By the end of the week you can:
1. Articulate the **three reliability tiers** (prompt-and-pray → JSON mode → schema-constrained Structured Outputs) and name which provider knob delivers each.
2. Design a Pydantic v2 model that is *LLM-friendly* (flat, enums for closed sets, field descriptions as prompt real estate) and emit its JSON Schema.
3. Get **schema-valid output** from OpenAI (strict `json_schema`), Anthropic (tool `input_schema`), and Gemini (`responseSchema`), and explain each provider's honored JSON-Schema subset + strict-mode gotchas.
4. Wrap extraction with **instructor** and a **retry/repair loop** that feeds validation errors back to the model, and distinguish transient vs deterministic vs unrecoverable failures.
5. Parse and validate **streaming partial JSON** so you can render progressively without crashing on half-formed objects.

### Theory (~4.5 hrs)
> 📚 Read first: [When (and When Not) to Force Structured Output](lectures/01-when-to-use-structured-output.md) · [The Three Reliability Tiers](lectures/02-three-reliability-tiers.md) · [Provider JSON-Schema Subsets and Strict-Mode Gotchas](lectures/03-provider-schema-subsets-and-strict-mode.md) · [Designing LLM-Friendly Schemas with Pydantic v2 (and a Zod Primer)](lectures/04-llm-friendly-schema-design.md) · [Runtime Resilience: Repair Loops, Error Classification, and Streaming Partial JSON](lectures/05-repair-loops-and-streaming.md). The bullets below are the recap.

**The decision framework (30 min).** Structured output is for when a *program* consumes the result; freeform is for prose and reasoning. Forcing JSON too early *hurts* reasoning quality — the model spends capacity on syntax instead of thinking. Two fixes: (a) allow a `reasoning`/`scratchpad` string field *before* the structured fields, or (b) do reason-then-extract in two passes. Internalize the roadmap's rule: "LLM proposes, code disposes" — model output is untrusted input to the next system.

**The three reliability tiers (45 min).**
- *Tier 1 — prompt-and-pray:* "Respond in JSON." Works ~most of the time, fails silently under distribution shift. Never ship this alone.
- *Tier 2 — JSON mode:* guarantees *syntactically valid* JSON, **not your schema**. The model can still omit fields or invent keys.
- *Tier 3 — schema-constrained Structured Outputs:* the provider constrains decoding (or validates) to your JSON Schema. This is the tier you want. OpenAI: `response_format={"type":"json_schema", ..., "strict": true}`. Anthropic: define a **tool** whose `input_schema` is your schema and force it with `tool_choice`. Gemini: `generation_config` with `response_mime_type="application/json"` + `response_schema`.

**Provider gotchas you must memorize (45 min).** Read the official docs for each: search "OpenAI Structured Outputs guide", "Anthropic tool use / JSON output", "Gemini API structured output / responseSchema".
- OpenAI strict mode requires `additionalProperties: false` on every object and **every property listed in `required`** — optional fields are modeled as `["string","null"]` unions, *not* by omitting from `required`. There's a **first-call schema-compile latency** (the grammar is cached after). Watch depth/enum-count limits.
- Anthropic doesn't have a `json_schema` response format; the idiom is a forced tool. Args come back as a **parsed object** (not a JSON string). Supports a subset of JSON Schema.
- Gemini's `responseSchema` uses an OpenAPI-subset dialect; property ordering and enum handling differ. Nullability and `anyOf` support is narrower than OpenAI's.

**Schema design for LLMs (45 min).** Flat beats nested. Enums beat free strings for closed sets. **Every field needs a `description`** — it's prompt real estate the model reads. Use semantic field names (`invoice_total_usd`, not `field3`). Avoid unbounded arrays and giant enums (they blow up the constrained-decoding grammar and confuse the model). Prefer `Literal[...]`/`Enum` in Pydantic.

**Pydantic v2, instructor, and Zod (60 min).** Pydantic v2: `model_json_schema()`, `Field(description=...)`, validators (`@field_validator`, `@model_validator`) that enforce *business rules* (e.g., line-items sum to total). **instructor** patches the provider client so you pass `response_model=YourModel` and get typed objects + automatic **re-ask on validation error** (it feeds the Pydantic error back). Zod (TS): `z.infer`, `zod-to-json-schema`, discriminated unions — the Vercel AI SDK's `generateObject` uses this. Skim the Zod docs even if you build in Python; you'll meet it in JS stacks.

**Retry/repair + streaming (30 min).** Repair loop: on a validation failure, send the model the *exact* error text and ask it to fix — but classify first: transient (429/5xx → backoff+retry), deterministic (schema violation → repair with error feedback, cap attempts), unrecoverable (refusal/impossible input → escalate to human). For streaming, providers emit partial JSON; use a tolerant parser (instructor's `Partial`/`create_partial`, or a library like `partial-json-parser`) and **only act on validated, complete objects**.

### Lab (~8 hrs)
> 🛠️ Full step-by-step guide: [Build structio — A Schema-Constrained Extraction Toolkit](labs/week-1-structio-extraction-toolkit.md). The steps below are the summary.

Build `structio/` — a structured-extraction toolkit you'll extend all phase.

**0. Project setup (20 min).**
```bash
uv init structio && cd structio
uv add pydantic openai anthropic google-genai instructor python-dotenv httpx pytest rich
mkdir -p structio/schemas structio/providers tests data
echo "OPENAI_API_KEY=...\nANTHROPIC_API_KEY=...\nGEMINI_API_KEY=..." > .env
```
Folder layout:
```
structio/
  schemas/invoice.py        # Pydantic models
  providers/openai_so.py    # tier-3 structured output per provider
  providers/anthropic_so.py
  providers/gemini_so.py
  repair.py                 # retry/repair loop + error classification
  stream.py                 # partial-JSON streaming demo
tests/test_schema.py
data/invoices.jsonl         # 20 messy invoice texts (you write these)
```

**1. Design the schema (60 min).** In `schemas/invoice.py`:
```python
from enum import Enum
from pydantic import BaseModel, Field, model_validator

class Category(str, Enum):
    software = "software"; hardware = "hardware"; services = "services"; other = "other"

class LineItem(BaseModel):
    description: str = Field(description="Human-readable line description")
    qty: float = Field(description="Quantity billed")
    unit_price_usd: float = Field(description="Price per unit in USD")

class Invoice(BaseModel):
    reasoning: str = Field(description="Brief notes on how fields were located; write this FIRST")
    vendor_name: str = Field(description="Legal name of the vendor issuing the invoice")
    invoice_date: str = Field(description="ISO 8601 date, e.g. 2026-03-14")
    total_usd: float = Field(description="Grand total in USD")
    category: Category = Field(description="Best-fit spend category")
    line_items: list[LineItem] = Field(description="Individual billed lines", max_length=25)

    @model_validator(mode="after")
    def totals_must_sum(self):
        computed = round(sum(li.qty * li.unit_price_usd for li in self.line_items), 2)
        if self.line_items and abs(computed - self.total_usd) > 0.01:
            raise ValueError(f"line items sum to {computed} but total_usd is {self.total_usd}")
        return self
```
Note the `reasoning` field first (scratchpad-before-extraction). Write 20 messy invoice strings into `data/invoices.jsonl` (vary formats, some with arithmetic that must validate, a couple that are intentionally impossible/garbage to exercise the escalation path).

**2. Tier-3 per provider (2.5 hrs).** Implement one function per provider returning a validated `Invoice`:
- `openai_so.py`: use the SDK's parse helper (`client.beta.chat.completions.parse` / responses parse) with `response_format` derived from the model, `strict=True`. Print the *first-call latency* vs subsequent calls to observe schema-compile caching.
- `anthropic_so.py`: define a tool `record_invoice` with `input_schema = Invoice.model_json_schema()`, set `tool_choice={"type":"tool","name":"record_invoice"}`, read args from the `tool_use` block, validate with `Invoice.model_validate(...)`.
- `gemini_so.py`: `google-genai` client, `response_mime_type="application/json"`, `response_schema=Invoice`. Handle the dialect differences (you may need to strip unsupported keywords — log which keywords each provider silently dropped).

**3. instructor + repair loop (2 hrs).** In `repair.py`, wrap OpenAI and Anthropic with `instructor.from_openai(...)` / `from_anthropic(...)`, pass `response_model=Invoice, max_retries=3`. Then hand-roll your *own* loop too (so you understand what instructor does): classify the exception (`RateLimitError` → exponential backoff; `pydantic.ValidationError` → append the error string to messages and re-ask; refusal/`None` after N tries → raise `EscalateToHuman`). Add a **circuit breaker**: after 3 deterministic repair attempts, stop and route to a review queue.

**4. Streaming partial JSON (60 min).** In `stream.py`, stream one extraction and use `instructor`'s partial/stream API (or `partial-json-parser`) to print the object as it fills in with `rich`. Assert you never act on an object until `Invoice.model_validate` passes on the *final* chunk.

**5. Wire a quick pytest (30 min).** `tests/test_schema.py`: parametrize over the 20 inputs and each provider; assert schema-valid rate and that the totals validator catches a deliberately-broken input.

Cost note: all of this runs on `gpt-4o-mini`, `claude-3-5-haiku`, `gemini-2.0-flash`. No GPU. If you have no paid keys at all, run tiers 1–2 against **Ollama** (`ollama pull llama3.2`) with its OpenAI-compatible endpoint — but note Ollama's schema enforcement is weaker; that's a teaching moment for Week 2's constrained decoding.

### Definition of Done
- [ ] `Invoice` model emits valid JSON Schema via `model_json_schema()`; every field has a description.
- [ ] All three provider functions return a **validated `Invoice`** for at least **18/20** of the valid inputs (the 2 garbage inputs must route to escalation, not crash).
- [ ] You have logged, per provider, **which JSON-Schema keywords were honored vs silently dropped** (a small markdown table in the repo).
- [ ] The totals validator rejects a hand-crafted invoice whose line items don't sum to the total (a passing test proves it).
- [ ] The repair loop caps at 3 deterministic attempts and raises `EscalateToHuman` on unrecoverable input; transient 429s are retried with backoff.
- [ ] Streaming demo renders progressively and only accepts the final object after validation.
- [ ] `pytest` is green.

### Pitfalls
- **Forgetting `additionalProperties: false` / not marking all fields required** in OpenAI strict mode → the API rejects the schema. Optional = nullable union, not omission.
- **Assuming JSON mode = your schema.** JSON mode only guarantees parseable JSON; fields can be missing or extra.
- **Nested/recursive schemas and giant enums** silently degrade quality and can exceed provider limits; flatten and bound them.
- **First-call latency panic:** the schema-compile spike on OpenAI is one-time per schema; don't "optimize" it away by dropping strict mode.
- **Acting on partial JSON while streaming** — half an object can parse as valid-looking but be semantically wrong. Validate the *completed* object.

### Self-check
1. Why does forcing JSON output sometimes lower answer quality, and what are the two standard mitigations?
2. In OpenAI strict mode, how do you express an *optional* field? Why not just leave it out of `required`?
3. Anthropic has no `json_schema` response format — what's the idiomatic way to get schema-constrained output, and how do the returned args differ from OpenAI's?
4. When a `ValidationError` fires in your repair loop, what exactly do you send back to the model, and how is that different from handling a 429?
5. Why is `description` on a Pydantic field "prompt real estate"?

---

## Week 2 — Tool / Function Calling: the universal loop, hardening, constrained decoding, and MCP

**Hours breakdown (~13 hrs):** Theory ~4.5 hrs · Lab ~8 hrs · Self-check/notes ~0.5 hr.

### Objectives
By the end of the week you can:
1. Implement the **universal tool-calling loop** (send tools → model requests call → you execute → return result with correct id linkage → repeat) against OpenAI, Anthropic, and Gemini behind one thin adapter.
2. Explain the provider wire-format differences (OpenAI `tool_calls` + `role:"tool"` + args-as-JSON-**string**; Anthropic `tool_use`/`tool_result` content blocks + parsed-object args; Gemini `functionDeclarations`) and use `tool_choice` (auto/none/required/forced).
3. Execute **parallel tool calls** correctly (concurrent execution, match results by id) and know when calls are dependent and must serialize.
4. Treat tool arguments as **untrusted input**: validate against a schema, allowlist tools per context, parameterize (no shell/SQL string interpolation), sandbox, and gate destructive actions behind human confirmation.
5. Use **constrained decoding / grammars** (Outlines on a local model) to *guarantee* structure on an open model, and explain why semantic keywords can't be honored by pure logit masking.
6. Stand up a minimal **MCP** server and connect a client — understanding MCP as the emerging tool standard and its security surface.

### Theory (~4.5 hrs)
> 📚 Read first: [The Universal Tool-Calling Loop](lectures/06-universal-tool-calling-loop.md) · [Provider Wire-Formats and the Normalizing Adapter](lectures/07-provider-wire-formats-and-adapters.md) · [Tool Definitions as Prompt Engineering and the tool_choice Knob](lectures/08-tool-definitions-and-tool-choice.md) · [Parallel vs Serial Tool Execution](lectures/09-parallel-vs-serial-tool-execution.md) · [Security: Tool Arguments Are Untrusted Input](lectures/10-securing-tool-calls.md) · [Constrained Decoding and Grammars on Self-Hosted Models](lectures/11-constrained-decoding-and-grammars.md) · [MCP: Standardizing Tools, the Server/Client Model, and Its Security Surface](lectures/12-mcp-server-and-client.md). The bullets below are the recap.

**The universal loop (45 min).** Every provider implements the same conversation shape: you send the user message **plus tool definitions**; the model responds with either a final answer or a **request** to call one or more tools (it never executes anything); you run the tool in *your* code; you append the result **linked by the tool-call id**; you loop until the model returns a final answer or you hit a step cap. Read: "OpenAI function calling guide", "Anthropic tool use overview", "Gemini function calling".

**Provider wire-format differences (45 min).** These bite everyone:
- OpenAI: assistant message has `tool_calls[]`, each with an `id` and `function.arguments` as a **JSON string** (you must `json.loads` it — and it can be malformed). You reply with a message `role:"tool", tool_call_id=..., content=...`.
- Anthropic: assistant returns content blocks including `tool_use` (with `id`, `name`, and **already-parsed** `input` object). You reply with a `user` message containing a `tool_result` block referencing `tool_use_id`.
- Gemini: `functionDeclarations` in `tools`; model returns a `functionCall` part; you reply with a `functionResponse` part.
Your adapter's job is to normalize all three to a single internal `ToolCall(id, name, args: dict)` and `ToolResult(id, content)`.

**Tool definitions are prompt engineering (30 min).** The `name`, `description`, and per-parameter docs are *when-to-use* guidance the model reads. Keep the tool count low (large tool menus degrade selection). `tool_choice`: `auto` (model decides), `none` (never), `required`/`any` (must call *something*), and **forced** (must call *this* tool) — forcing a specific tool is another way to guarantee structured extraction (this is exactly Week 1's Anthropic trick).

**Parallel tool calls (30 min).** Models can request several independent calls in one turn; execute them **concurrently** (`asyncio.gather`) and return all results matched by id before looping. Stop parallelizing when a later call depends on an earlier call's result — then you must serialize across turns.

**Security: tool args are untrusted (45 min).** This is the phase's most important lesson. Tool arguments are effectively user input laundered through the model — treat them exactly as you'd treat a raw HTTP request body. Never interpolate into shells/SQL/filesystem paths; **parameterize** queries, allowlist which tools are available per context, apply **least-privilege** credentials, sandbox execution, and require **human confirmation** for destructive/irreversible/financial actions. Read the OWASP "Top 10 for LLM Applications" entries on excessive agency and insecure output handling (search by name).

**Constrained decoding / grammars (30 min).** Beyond provider Structured Outputs, on **self-hosted** models you can mask logits so only tokens valid under a grammar are sampled — libraries: **Outlines**, **XGrammar**, **lm-format-enforcer**, GBNF grammars in `llama.cpp`, and `guidance`. This *guarantees* JSON-Schema/regex/CFG conformance. Key intuition: masking enforces *syntax*, it cannot enforce *semantics* (a grammar can force a valid enum token but not that the enum is the *correct* one), and it only works where you control the logits (not behind closed APIs).

**MCP intro (30 min).** The **Model Context Protocol** standardizes how tools/resources/prompts are exposed to any LLM client: a server advertises capabilities, clients discover and invoke them. It's the "USB-C for tools." Security surface: tool poisoning, over-broad scopes, the confused-deputy problem. Read the official "Model Context Protocol" docs/spec and the `FastMCP` README.

### Lab (~8 hrs)
> 🛠️ Full step-by-step guide: [Extend structio into a Hardened, Provider-Agnostic Tool-Loop Extraction Service (Phase Milestone)](labs/week-2-hardened-tool-loop-and-extraction-service.md). The steps below are the summary.

Extend the `structio/` repo into a tool-loop + hardened service.

**1. Provider-agnostic tool loop (3 hrs).** New package `structio/toolloop/`:
```
toolloop/
  types.py        # ToolSpec, ToolCall, ToolResult, Message
  adapters.py     # to_openai / to_anthropic / to_gemini schema + parse_response
  loop.py         # run(messages, tools, tool_choice, max_steps) -> final answer
  tools.py        # registry: name -> (callable, pydantic args model, destructive?)
```
Define two example tools with Pydantic arg models: `get_exchange_rate(base, quote)` (pure, safe) and `search_invoices(query, limit)` (reads local `data/`). In `loop.py`:
```python
def run(messages, tools, provider, tool_choice="auto", max_steps=6):
    for _ in range(max_steps):
        resp = provider.chat(messages, tools=[to_schema(t) for t in tools], tool_choice=tool_choice)
        calls = provider.parse_tool_calls(resp)          # normalized ToolCall[]
        if not calls:
            return provider.final_text(resp)
        results = execute(calls, tools)                   # validate args, run, catch errors
        messages += provider.render_tool_turn(resp, results)
    raise RuntimeError("max steps exceeded")              # loop guard
```
Run the **same** extraction/tool task on all three providers and **diff which JSON-Schema keywords each honored** in the tool `parameters` (reuse Week 1's finding, now for tool args). Make `max_steps` a hard guard.

**2. Parallel execution (60 min).** Make `execute()` run independent calls with `asyncio.gather`; add a tool pair where the second depends on the first and show your loop naturally serializes them across turns. Log per-call latency and prove concurrency with timestamps.

**3. Hardened extraction service (2.5 hrs).** `service/app.py` — a FastAPI endpoint `POST /extract` that takes messy invoice text and returns validated `Invoice` JSON, combining Week 1 (schema-constrained + repair) with security:
- Validate tool/args with Pydantic **before** executing anything.
- **Tool allowlist per request context** (an extraction request cannot reach a "send email" tool).
- Parameterize the `search_invoices` backend (if you back it with SQLite, use bound params — demonstrate that a query arg like `'; DROP TABLE ...` does nothing).
- **Circuit breaker**: after 3 repair attempts, return `{"status":"needs_human_review", ...}` (HTTP 422) instead of a guess.
- Emit a per-request **trace** (JSON lines): provider, steps, tokens (from usage), per-step latency, repair attempts, final status. This is your reliability-eval fuel.

**4. Constrained decoding on a local model (90 min).**
```bash
uv add outlines
ollama pull llama3.2:1b   # or run a HF model directly via transformers if you have a GPU
```
Use **Outlines** to force the `Invoice` schema (or a regex for `invoice_date`) on a local model and show it **cannot** emit malformed JSON even when you prompt it adversarially ("ignore the format and write a poem"). Contrast with plain Ollama JSON mode failing the same adversarial prompt. Write two sentences on why the grammar guarantees syntax but not correctness.

**5. Minimal MCP server + client (60 min).**
```bash
uv add fastmcp mcp
```
Wrap `get_exchange_rate` and `search_invoices` as an MCP server (`fastmcp`), run it, and connect with a client that lists tools and invokes one. Note in your README the security controls you'd add before exposing it (scoped tools, no destructive actions without confirmation, input validation at the server boundary).

**6. Reliability eval (30 min).** Script `eval/run_eval.py`: run all 20 inputs through the service across **two models** (e.g., `gpt-4o-mini` vs `claude-3-5-haiku`), and report **schema-valid rate**, **field accuracy** (against hand-labeled gold), **mean repair attempts**, and **% escalated**. Print a table with `rich`.

### Definition of Done
- [ ] One tool loop drives **all three providers** through the same task; the loop enforces `max_steps` and never executes a tool the model only *requested* without your code running it.
- [ ] A markdown table shows honored-vs-dropped JSON-Schema keywords for tool `parameters` per provider.
- [ ] Parallel independent calls run concurrently (timestamps prove it); a dependent pair serializes correctly.
- [ ] `POST /extract` returns **valid, schema-conformant JSON for ≥18/20** inputs and **`needs_human_review` (422) for the 2 garbage inputs** — 0 crashes.
- [ ] A SQL/shell-injection-style tool arg is provably neutralized (parameterized) — a test asserts no side effect.
- [ ] Tool allowlisting blocks an out-of-context tool (a test asserts the extraction path cannot invoke a destructive tool).
- [ ] Outlines forces schema-valid output on a local model even under an adversarial "break the format" prompt; plain JSON mode is shown failing it.
- [ ] MCP server starts and a client lists + invokes at least one tool.
- [ ] The reliability eval prints schema-valid rate, field accuracy, mean repair attempts, and % escalated for two models; a per-request trace shows per-step tokens and latency.
- [ ] `pytest` green.

### Pitfalls
- **Forgetting the id linkage.** OpenAI needs `tool_call_id` on the tool message; Anthropic needs `tool_use_id` on the `tool_result`. Mismatched/missing ids → cryptic 400s or the model repeating calls.
- **`json.loads` on OpenAI args without a try/except.** The arguments string can be truncated or malformed; handle it as a repairable error, don't crash.
- **Interpolating tool args into shell/SQL.** The single most common security hole in agent code. Parameterize, sandbox, allowlist.
- **Unbounded loops.** Without `max_steps`/budget guards, a confused model can call tools forever and burn money. Always cap.
- **Assuming constrained decoding fixes correctness.** It fixes *shape*, not *truth* — and it's unavailable behind closed APIs.
- **Over-parallelizing dependent calls** → the model acts on stale/absent results. Serialize when call N needs call N-1's output.

### Self-check
1. Walk the universal tool loop end-to-end. At which step does *your* code (not the model) execute the tool, and why does that boundary matter for security?
2. Give the concrete wire-format difference between how OpenAI and Anthropic return tool arguments — what must your adapter do differently for each?
3. When is `tool_choice="required"` different from forcing a *specific* tool, and how does forcing a specific tool double as structured extraction?
4. Why can two tool calls in the same turn be run in parallel sometimes but not others? How does your loop decide?
5. Constrained decoding guarantees valid JSON — name two things it still cannot guarantee, and one deployment context where you can't use it at all.

---

## Phase milestone project — Multi-provider structured extraction service with a reliability eval

Compose Weeks 1–2 into one shippable artifact: a **provider-agnostic, hardened extraction service** that turns messy invoice text into validated, schema-conformant JSON, backed by a reliability eval that lets you defend a model choice with numbers.

**What it must do and prove:**
- **One Pydantic model → correct schemas for OpenAI, Anthropic, and Gemini** via a thin adapter (no per-provider schema duplication). Diff and document honored JSON-Schema keywords per provider.
- **Schema-constrained (Tier-3) extraction** with an **instructor-style repair loop** and a **circuit breaker** to human review after 3 attempts.
- A **tool-calling path** (the universal loop) for enrichment (e.g., currency conversion / invoice lookup) with **parallel independent calls**, `max_steps` guard, and per-run token/$ budget.
- **Security posture:** tool arg validation, per-context tool allowlist, parameterized backends, least-privilege, and human confirmation gating on any destructive tool. A red-team test (injection-style arg) proves neutralization.
- **Business-rule validation** (line items sum to total) enforced in code, not the prompt.
- A **reliability eval** over ≥20 labeled inputs and ≥2 models reporting: schema-valid rate, first-attempt validity, field precision/recall vs gold, mean repair attempts, % escalated, and cost/latency per correct extraction.
- **Observability:** per-request JSONL traces with per-step tokens, latency, provider, repair count, and final status.
- *(Stretch)* An **MCP server** exposing the extraction + enrichment tools, and an **Outlines** local-model variant that guarantees schema conformance offline.

**Acceptance criteria:**
- Schema-valid JSON for ≥18/20 valid inputs on both models; garbage inputs escalate (422), never crash.
- Field accuracy ≥ 0.9 on the gold set for at least one model, with the eval table justifying which model you'd ship.
- Injection-style tool arg provably has no side effect; out-of-context tool provably blocked.
- `pytest` green in CI; a trace file exists showing per-step tokens and latency for every request.

**Suggested repo layout:**
```
extraction-service/
  structio/
    schemas/invoice.py
    providers/{openai_so,anthropic_so,gemini_so}.py
    repair.py
    toolloop/{types,adapters,loop,tools}.py
  service/app.py                # FastAPI POST /extract
  mcp_server/server.py          # FastMCP (stretch)
  local/outlines_extract.py     # constrained decoding (stretch)
  eval/{gold.jsonl,run_eval.py}
  data/invoices.jsonl
  traces/                       # per-request JSONL
  tests/
  README.md                     # incl. honored-keywords table + model-choice memo
  pyproject.toml
```

## You are ready to move on when...
- [ ] You can implement the universal tool loop from memory against any of the three providers and explain each wire-format difference without looking it up.
- [ ] You can pick the right reliability tier for a task and configure it per provider (OpenAI strict `json_schema`, Anthropic forced tool, Gemini `responseSchema`).
- [ ] You design LLM-friendly schemas by reflex (flat, enums, descriptions, bounded arrays) and enforce business rules in Pydantic/Zod validators.
- [ ] Your extraction service returns validated JSON with a repair loop + circuit breaker, and treats every tool arg as untrusted (parameterized, allowlisted, least-privilege, HITL on destructive actions).
- [ ] You can produce a reliability eval (schema-valid rate, field accuracy, repair attempts, cost/latency) and use it to defend a model choice.
- [ ] You understand constrained decoding's guarantees and limits, and can stand up a basic MCP server/client and articulate its security surface.
