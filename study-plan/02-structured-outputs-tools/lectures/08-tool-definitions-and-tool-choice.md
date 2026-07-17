# Lecture 8: Tool Definitions as Prompt Engineering and the `tool_choice` Knob

> A tool definition is not a type signature you hand to a compiler — it is a prompt fragment the model *reads at inference time* to decide whether to call, which tool to call, and how to fill the arguments. Get the wording wrong and the model calls the wrong tool, or none, or hallucinates arguments. This lecture teaches you to write tool definitions like instructions, to keep the tool menu small enough that selection stays accurate, and to drive the `tool_choice` knob deliberately — including the trick where forcing one specific tool becomes your most reliable structured-extraction primitive. After this you will pick the right `tool_choice` mode per use case by reflex and know exactly what each provider's spelling guarantees.

**Prerequisites:** Week 1 (JSON Schema, Pydantic-for-LLMs, the Anthropic forced-tool extraction trick), the universal tool-calling loop from this week's Lecture 7 · **Reading time:** ~22 min · **Part of:** Structured Outputs & Tool Calling, Week 2

## The core idea (plain language)

When you register a tool, you send the model three text fields plus a schema: the tool's `name`, its `description`, and a `description` on each parameter. It is tempting to treat these as bureaucratic metadata — the way you'd document a REST endpoint for a human who will read the docs *later*. That mental model is wrong and it will cost you.

Those fields are injected into the model's context on **every single request**, right alongside the system prompt and the user's message. The model has no other information about what `search_invoices` does except the sentence you wrote in its `description`. When it decides — token by token — whether to emit a tool call, it is reading your descriptions as *instructions*. A tool definition is a prompt. `name` and `description` are **when-to-use guidance**; per-parameter docs are **how-to-fill guidance**.

This reframing has two immediate consequences. First: write descriptions the way you'd write a good instruction, not the way you'd write a Javadoc stub. "Searches invoices" is a stub. "Use this when the user asks to find, list, or filter existing invoices by vendor, date range, or amount. Do NOT use it to create or modify invoices." is an instruction — it tells the model both *when to reach for it* and *when not to*. Second: because every tool's full definition sits in context on every call, a large tool menu is a large, noisy prompt. Past a certain count, the model's selection accuracy degrades — it confuses similar tools, or picks a plausible-but-wrong one. Keep the menu small.

The second half of the lecture is the `tool_choice` knob: the request parameter that controls *whether the model is allowed to call tools at all, and which*. It has four conceptual settings — let the model decide, forbid calling, force *some* call, force *this* call — and the last one doubles as a rock-solid way to extract structured data.

## How it actually works (mechanism, from first principles)

### Tool definitions are prompt tokens

Mechanically, the provider takes your `tools` array, serializes each tool (name, description, JSON Schema of parameters), and renders it into the prompt the model actually sees — before the system prompt in Anthropic's render order (`tools` → `system` → `messages`). The model then generates normally. "Deciding to call a tool" is just the model generating a special structured token sequence instead of prose; that decision is conditioned on everything in context, and your tool descriptions are a big chunk of that context.

So the descriptions are doing double duty as *cost* and *signal*. Let's put numbers on the cost. A tool with a name, a two-sentence description, and four parameters each with a one-line description is roughly 80–150 tokens once serialized. Ten such tools is ~1,000–1,500 tokens riding on *every* request in the conversation. That's not fatal, but it is a fixed tax, and it grows linearly with the menu.

The signal side is where selection accuracy lives. Consider two tools:

```
Vague:
  name: "lookup"
  description: "Look up data."
  params: { q: { type: "string", description: "the query" } }

Instructional:
  name: "search_invoices"
  description: "Search the invoice database for EXISTING invoices.
    Use when the user wants to find, list, count, or filter invoices
    by vendor name, date range, or amount. Returns up to `limit`
    matches. Do NOT use this to create, edit, or delete invoices —
    use record_invoice or void_invoice for those."
  params: {
    query: { type: "string",
      description: "Free-text or structured filter, e.g. 'ACME Corp
        March 2026' or 'total > 5000'. Leave empty to list recent." },
    limit: { type: "integer",
      description: "Max results, 1-100. Default 20." }
  }
```

Given the user turn "how many bills did we get from ACME last quarter?", the vague `lookup` gives the model almost nothing: it might call `lookup`, might answer from nothing, might ask a clarifying question. The instructional version maps the user's "bills … from ACME last quarter" cleanly onto "find/filter invoices by vendor and date range" — the model calls `search_invoices` with `query="ACME"` and reasons about the quarter. **The description literally taught it when to fire.**

The most valuable sentences in a tool description are the *negative* ones — "Do NOT use this to …". They're what disambiguate near-neighbors. On recent Anthropic models, which reach for tools more conservatively, being *prescriptive about when to call* (not just what the tool does) measurably raises the should-call rate; the mechanism is the same everywhere.

### Why a big menu degrades selection

Think of tool selection as a classification problem the model solves in-context: given the conversation, pick zero or one (or several) tools from N. As N grows, three things get worse:

1. **Overlap.** With 3 tools, their scopes rarely collide. With 30, you inevitably have `search_invoices`, `search_vendors`, `search_payments`, `find_customer`, `lookup_transaction` — near-synonyms whose boundaries blur. The model picks a neighbor.
2. **Prompt dilution.** The relevant tool's description is now 1 of 30, buried in ~4,000 tokens of tool specs. The signal-to-noise ratio for any given decision drops.
3. **Context cost.** Every extra tool is fixed tokens on every turn, and (on caching-aware stacks) reordering or editing the tool list invalidates the prompt cache from position zero.

There's no hard cliff — providers accept dozens or even hundreds of tools — but as an *approximate* rule of thumb, selection reliability starts to sag somewhere in the **10–20 tool** range for a single flat menu, and you should feel the need to intervene well before you hit a provider's ceiling. The engineering responses:

- **Cut.** Do you need `list_invoices` and `search_invoices`, or one tool with an optional filter? Merge.
- **Route.** Put a cheap first-stage model (or a rule) in front that picks a *toolset* by intent, then expose only that subset to the working model. An "invoices" request never sees the "send_email" tool.
- **Defer / search.** Newer stacks offer a tool-search mechanism: the model queries a large library and only the matched tool schemas get loaded into context. This keeps the fixed prompt small and preserves cache.

The routing/allowlisting move is also a **security** control (Week 2's theme): an extraction request that can't see a destructive tool literally cannot call it.

### The `tool_choice` knob: four modes

`tool_choice` answers two questions at once: *may the model call a tool this turn?* and *is it constrained to a particular one?* Conceptually there are four modes. The per-provider spellings differ and you must memorize them:

| Mode | What it guarantees | OpenAI | Anthropic | Gemini (`function_calling_config`) |
|---|---|---|---|---|
| **Auto** | Model decides: prose *or* a tool call | `"auto"` (default) | `{"type":"auto"}` (default) | `mode: "AUTO"` |
| **None** | Model may not call any tool | `"none"` | `{"type":"none"}` | `mode: "NONE"` |
| **Required / Any** | Model MUST call *some* tool (its choice which) | `"required"` | `{"type":"any"}` | `mode: "ANY"` |
| **Forced (specific)** | Model MUST call *this* named tool | `{"type":"function","function":{"name":"X"}}` | `{"type":"tool","name":"X"}` | `mode: "ANY"` + `allowed_function_names:["X"]` |

Read the naming carefully, because it is a classic trap:

- **OpenAI** calls the "must call something" mode `required`. To force a *specific* function you pass an object naming the function (in the Responses API this is `{"type":"function","name":"X"}`; in Chat Completions it's `{"type":"function","function":{"name":"X"}}`).
- **Anthropic** calls the "must call something" mode `any`. Anthropic has *no* keyword called `required` — if you type `{"type":"required"}` against Anthropic you'll get an error. Forcing a specific tool is `{"type":"tool","name":"X"}`.
- **Gemini** collapses both "must call something" and "force a specific tool" into `mode: "ANY"`; the difference is whether you also pass `allowed_function_names`. With one name in that list, `ANY` becomes a forced-specific call.

All three also let you cap parallelism: Anthropic accepts `disable_parallel_tool_use: true` alongside any `tool_choice`; OpenAI exposes `parallel_tool_calls: false`.

### The distinction the self-check demands

Here is the line the objectives insist you draw:

> `required`/`any` forces a call but **lets the model pick which tool**. Forcing a specific tool guarantees **the shape of a single call**.

Why does this matter? With `required`/`any` and a menu of five tools, you're guaranteed the model calls *one* of them, but you don't know which until the response comes back — you must be ready to dispatch on `name` and handle five possible argument shapes. That's exactly what you want for an *agent step*: "you must act; you choose the action."

With a *forced specific* tool, you've pinned both the tool *and* — because a tool's arguments are its JSON Schema — the exact shape of the arguments you'll get back. There is only one `name`, one schema, one thing to parse. That determinism is the whole game for extraction.

### Forcing a specific tool = structured extraction (Week 1, generalized)

In Week 1 you learned Anthropic's idiom for guaranteed structured output: define a tool whose `input_schema` is your Pydantic-derived JSON Schema, then force it with `tool_choice={"type":"tool","name":"record_invoice"}`. The model *cannot* respond with prose — it must emit a `tool_use` block whose `input` conforms (schema-shaped) to your model. You read `input`, `model_validate` it, done.

Now generalize. That was never Anthropic-specific magic. **Forcing any single tool, on any provider, is a structured-extraction move**, because:

1. Forcing the tool removes the "answer in prose" escape hatch — the model *must* produce a tool call.
2. A tool call's arguments are constrained to the tool's parameter schema.
3. Therefore the model's output is coerced into your schema.

So "extract this invoice as JSON" becomes: define one tool `record_invoice(schema=Invoice)`, set `tool_choice` to force it, never actually execute it — you just harvest the arguments. You're using the tool-call channel as a typed output channel. On OpenAI you'd force the function; on Gemini you'd use `ANY` + `allowed_function_names:["record_invoice"]`. This is why, in a provider-agnostic extraction service, "Anthropic forced tool" and "OpenAI strict `json_schema`" and "Gemini `responseSchema`" all land in the same Tier-3 bucket — forcing a tool is one of the ways to get there.

(One caveat carried from Week 1: forcing structure too aggressively can suppress reasoning. If quality matters, give the schema a `reasoning`/scratchpad field *first*, or do reason-then-extract in two passes.)

## Worked example

You're building the invoice service. Two situations, two `tool_choice` settings.

**Situation A — extraction (forced specific).** Messy invoice text in, validated `Invoice` JSON out. One tool, forced.

```python
# Anthropic
resp = client.messages.create(
    model="claude-opus-4-8", max_tokens=2000,
    tools=[{
        "name": "record_invoice",
        "description": "Record the fields extracted from an invoice.",
        "input_schema": Invoice.model_json_schema(),
    }],
    tool_choice={"type": "tool", "name": "record_invoice"},  # forced
    messages=[{"role": "user", "content": messy_invoice_text}],
)
block = next(b for b in resp.content if b.type == "tool_use")
invoice = Invoice.model_validate(block.input)   # parsed object, not a JSON string
```

Guarantee: exactly one `tool_use` for `record_invoice`, arguments shaped by `Invoice`. You never run the tool — the "call" *is* the answer.

**Situation B — an agent step (auto with a step cap).** The user asks a question that *might* need a lookup. You don't want to force a call — sometimes the answer is already known.

```python
resp = client.messages.create(
    model="claude-opus-4-8", max_tokens=1024,
    tools=[get_exchange_rate_tool, search_invoices_tool],
    tool_choice={"type": "auto"},        # model decides: answer or call
    messages=messages,
)
# loop: if resp has tool_use blocks, execute them, append results, repeat;
# else return the text. Cap iterations with max_steps.
```

Guarantee: none, by design — the model answers directly *or* calls one/both tools. Your loop handles both, bounded by `max_steps` (say 6) so a confused model can't call tools forever and burn your budget.

**A quick cost sketch of the menu-size point.** Suppose each tool serializes to ~120 tokens. With `gpt-4o-mini`-class input pricing around **$0.15 / 1M tokens** (*approximate, check current rates*), 5 tools = 600 tokens ≈ **$0.00009** per request of pure tool-definition tax; 25 tools = 3,000 tokens ≈ **$0.00045**. Five-fold more tax, on every turn, for a menu you probably can't select from accurately anyway. If a request makes 6 tool-loop turns, multiply by 6. The cost is real but usually secondary — the *accuracy* loss from a bloated menu is the reason to trim, and it doesn't show up on the invoice, it shows up as wrong tool calls in your traces.

## How it shows up in production

- **Wrong-tool bugs trace back to descriptions, not code.** When the model calls `search_vendors` instead of `search_invoices`, your instinct is to blame the loop. Look at the descriptions first — near-synonym scopes are the usual culprit. The fix is a sentence, often a negative one ("Do NOT use for …"), not a code change.
- **Silent under-calling.** Newer, more instruction-faithful models call tools *less* eagerly than older ones. A search tool that "used to work" may now sit unused because its description only says *what* it does, not *when* to call it. Add trigger conditions ("Call this when the user asks about current prices or recent events"). This is a documented lift, not folklore — but treat the magnitude as workload-specific.
- **Menu creep is a slow leak.** Teams add one tool per feature. At tool #14 selection accuracy quietly sags; nobody notices because no single addition broke anything. Budget a periodic "tool audit": merge overlaps, delete dead tools, route by intent.
- **`tool_choice` mismatches are 400s or infinite loops.** Typing OpenAI's `required` at Anthropic (which wants `any`) is a hard error. Forgetting to *drop back* to `auto` after a forced-tool extraction can trap an agent that then can only ever call that one tool. And forcing a tool while also asking for prose reasoning fights itself — the model can't do both.
- **Forced-tool extraction interacts with reasoning quality.** Pinning the output shape is great for reliability and terrible for hard reasoning if you didn't leave the model room to think. In production this shows up as "the JSON is valid but the values are wrong." Add a scratchpad field or split into two passes.
- **Cache invalidation from tool churn.** The tool list renders at prompt position zero. Adding, removing, or reordering tools mid-conversation invalidates the prompt cache for the entire request. If you route/allowlist per turn, you're trading cache hits for a smaller menu — usually worth it, but measure `cache_read_input_tokens` to know the cost.

## Common misconceptions & failure modes

- **"Descriptions are documentation."** No — they're live prompt tokens the model reads on every call. Write them as instructions with explicit when/when-not guidance.
- **"`required` forces my tool."** It forces *a* tool, model's choice. To pin a specific tool you pass the tool's name (OpenAI object form / Anthropic `{"type":"tool","name":…}` / Gemini `ANY` + `allowed_function_names`).
- **"Anthropic has `required`."** It has `any`. There is no `required` keyword on Anthropic; using it errors.
- **"More tools = more capable agent."** More tools = worse selection and higher token tax past ~10–20. Capability comes from the *right* small set, plus routing when you genuinely have many.
- **"Forcing a tool guarantees correct values."** It guarantees *shape*, not truth — same limit as constrained decoding. Validate with your schema *and* business rules (e.g., line items sum to total), and give the model room to reason.
- **"`tool_choice: none` removes the tools."** It only forbids calling them this turn; the definitions still occupy context (and still cost tokens). Use `none` when you want the model to *know about* tools but answer in prose right now (common at the final "summarize what you did" step).
- **"Parallel calls just work under a forced tool."** Forcing a specific tool is a single-call shape. If you want at-most-one even under `auto`/`any`, set `disable_parallel_tool_use` / `parallel_tool_calls:false` explicitly.

## Rules of thumb / cheat sheet

- **Write every tool `name`/`description`/param-doc as an instruction.** Say when to call it, when NOT to, and give an example argument. Negative guidance disambiguates near-neighbors.
- **Keep the menu small.** Aim for a single-digit tool count per decision; feel the need to route/trim around 10–20. Merge synonyms; delete dead tools.
- **Many tools → route or defer.** First-stage intent classifier picks a toolset; or use a tool-search mechanism so only matched schemas load into context. Doubles as least-privilege security.
- **Pick `tool_choice` by use case:**
  - *Extraction / guaranteed shape* → **force the specific tool** (harvest args, don't execute).
  - *Agent step (may or may not act)* → **`auto`** + a `max_steps` guard.
  - *Must act, model picks how* → **`required` (OpenAI) / `any` (Anthropic) / `ANY` (Gemini)**.
  - *Answer in prose now, tools off this turn* → **`none`**.
- **Memorize the spellings.** OpenAI: `auto`/`none`/`required`/`{function name}`. Anthropic: `{"type":"auto"|"none"|"any"|"tool","name":…}`. Gemini: `mode: AUTO|NONE|ANY` (+ `allowed_function_names` to pin).
- **Forced tool = structured output.** It's Week 1's Anthropic trick generalized to any provider. Leave a `reasoning` field first if the task needs thinking.
- **Drop back to `auto` after a forced turn** in an agent loop, or you pin it forever.
- **Tool list is cache-prefix zero.** Don't churn it mid-conversation unless you accept the cache miss.

## Connect to the lab

This is the theory behind Week 2 Lab steps 1 and 3. In `toolloop/adapters.py` your `to_openai`/`to_anthropic`/`to_gemini` functions must emit the correct `tool_choice` spelling per provider — this lecture is your reference for those four modes. In `loop.py`, `run(..., tool_choice="auto", max_steps=6)` is the *agent-step* setting; the `POST /extract` service in step 3 should instead **force** the extraction tool to get guaranteed `Invoice` shape (the same move as Week 1's Anthropic forced tool, now behind your adapter). When you define `get_exchange_rate` and `search_invoices`, write their descriptions as instructions — you'll see selection accuracy in your reliability trace.

## Going deeper (optional)

- **Anthropic** — "Tool use overview" and the `tool_choice` section on the Claude docs (docs.claude.com). Search: *"Anthropic tool use tool_choice auto any tool none"*.
- **OpenAI** — Function calling guide, docs (platform.openai.com); note the `required` keyword and the forced-function object shape, and the Responses-vs-Chat-Completions difference. Search: *"OpenAI function calling tool_choice required"*.
- **Google Gemini** — Function calling docs (ai.google.dev), `function_calling_config` with `AUTO`/`ANY`/`NONE` and `allowed_function_names`. Search: *"Gemini function calling mode ANY allowed_function_names"*.
- **Anthropic Cookbook** (github.com/anthropics/anthropic-cookbook) — worked tool-use notebooks, including forced-tool extraction. Search: *"anthropic-cookbook tool use"*.
- **OWASP Top 10 for LLM Applications** — "Excessive Agency" ties directly to why you allowlist/route tools per context. Search: *"OWASP LLM Top 10 excessive agency"*.

## Check yourself

1. Your model keeps calling `search_vendors` when the user clearly wants invoices. Both tool descriptions are one-liners. What's the single highest-leverage fix, and why is it a prompt change and not a code change?
2. A teammate sets `tool_choice="required"` against the Anthropic API and gets an error. What's wrong, and what's the correct spelling to force *some* tool vs. *a specific* tool on Anthropic?
3. Explain precisely how `required`/`any` differs from forcing a specific tool, and which one you'd use for (a) a guaranteed JSON extraction and (b) an agent step that must take an action.
4. Why does forcing a specific tool double as structured extraction? What does it guarantee, and what does it *not* guarantee?
5. You have 22 tools and selection accuracy is poor. Give two structurally different remedies (not "write better descriptions") and name a side benefit of one of them.
6. When would you deliberately use `tool_choice: none` even though the tools are still defined in the request?

### Answer key

1. **Add negative, when-to-use guidance to the descriptions** — e.g. `search_invoices`: "Use for invoices; do NOT use for vendors" and vice-versa. The model chooses tools by reading their descriptions at inference time, so wrong-tool selection is almost always a description (prompt) problem, not a loop (code) problem. Near-synonym scopes need explicit boundaries, and negative sentences are what disambiguate them.
2. Anthropic has **no `required` keyword** — that's OpenAI's spelling. On Anthropic, "must call *some* tool" is `{"type":"any"}`; "must call *this* tool" is `{"type":"tool","name":"X"}`.
3. `required`/`any` guarantees the model emits *a* tool call but lets it choose *which* — you must dispatch on the returned `name` and handle multiple argument shapes. Forcing a specific tool guarantees a single call with a *known* name and therefore a *known* argument schema. (a) Guaranteed JSON extraction → **force the specific tool** (one shape to parse). (b) Agent step that must act → **`required`/`any`** (must act, model picks the action).
4. Forcing a specific tool removes the "answer in prose" option, so the model must emit a tool call; a tool call's arguments are constrained to the tool's parameter schema; therefore the output is coerced into your schema — you harvest the arguments as your structured result (never executing the tool). It **guarantees shape/schema-conformance**, not **semantic correctness** — the values can still be wrong, so validate with business rules and give the model room to reason (scratchpad field or two-pass).
5. (a) **Route by intent** — a cheap first stage picks a toolset, and only that subset is exposed to the working model. (b) **Merge/delete** overlapping and dead tools to shrink the flat menu. (Also acceptable: a tool-search/defer mechanism that loads only matched schemas.) Side benefit of routing/allowlisting: **least-privilege security** — a request that can't see a destructive tool can't call it. (Defer/search also preserves prompt cache.)
6. When you want the model to **know the tools exist but respond in prose this turn** — e.g. the final "summarize what you did" step of an agent loop, where you want a natural-language wrap-up and no further tool calls. `none` forbids calling without removing the definitions from context.
