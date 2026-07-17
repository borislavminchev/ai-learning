# Lecture 2: Provider SDKs In Depth: The Shapes a Gateway Hides

> A gateway like LiteLLM sells you one useful lie: "every model looks the same." It lets you swap `gpt-4o-mini` for `claude-haiku` by editing a config line. But the moment you reach for prompt caching, reasoning depth, or a model that lives behind AWS IAM, the lie leaks — and you're debugging a normalization layer you didn't write, with no idea what it stripped. This lecture takes you *under* the gateway. After it you will be able to read a raw request/response for OpenAI (both Chat Completions and Responses), Anthropic, Gemini/Vertex, and Bedrock Converse; name exactly where their message shapes, tool-call representations, caching models, and auth disagree; trace one "call a tool, feed the result back" loop through all four dialects; and predict precisely which features a gateway can normalize and which will always leak through to a direct-SDK escape hatch.

**Prerequisites:** You can call at least one LLM SDK and have built a tool-calling loop (Phase 4). Comfort reading JSON. · **Reading time:** ~30 min · **Part of:** Frameworks, Ecosystem, Team Practice & Career — Week 1

---

## The core idea (plain language)

Every major provider exposes the *same* underlying capability — "send a conversation and some tools, get back text or a request to call a function" — behind a *different* wire shape. There is no neutral, canonical format. There are four incompatible dialects, and each one encodes the same three ideas:

1. **A conversation** — a list of turns, each with a role.
2. **A tool call** — the model asking you to run a function, and you feeding the result back.
3. **Everything else** — system instructions, caching hints, reasoning depth, and authentication.

A gateway is a **translation layer**. It picks one dialect as the public interface — almost always OpenAI's Chat Completions, because that's the one the ecosystem cloned — and rewrites your request into whatever the target provider speaks. This works cleanly for the parts that map 1:1: a user message is a user message everywhere. It breaks down for the parts one provider has and another doesn't, or represents so differently that translation loses information: **caching breakpoints, reasoning effort, safety config, and non-bearer auth.** Those are exactly the levers that move cost, latency, and quality most in production — which is why "just use a gateway" is a fine *default* and a dangerous *religion*.

Here is the shape of the problem:

```
  your code                       gateway (speaks Chat Completions)                provider
  ─────────                       ──────────────────────────────                  ────────
  messages[]        ─────────▶    role:"system" → top-level `system`?   ─────────▶ Anthropic
  tools[]                         tool_calls[] → tool_use blocks?                   Gemini
  cache_control ✗ ──drops──▶      (no slot in Chat Completions shape)              Bedrock
  reasoning_effort ✗ ─drops─▶     (no slot for the target's param name)            OpenAI
  x-api-key ✗ ──────────▶         (cannot produce a SigV4 signature) ──────╳────── Bedrock
```

The engineering payoff of knowing the dialects: when the gateway does the wrong thing — silently drops `cache_control`, mis-serializes a tool result, fails auth against Bedrock — you drop to the raw SDK *for that one call*, fix it, and keep the gateway for everything else. That escape hatch is only usable if you already know what the raw shape looks like. This lecture is that knowledge.

---

## How it actually works (mechanism, from first principles)

### Dialect 1 — OpenAI: a flat message list, tool args as a JSON *string*

OpenAI's **Chat Completions** API (`POST /v1/chat/completions`) is the format the whole ecosystem cloned. The conversation is a flat array of `messages`, each with a `role`:

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are a booking assistant."},
    {"role": "user", "content": "Weather in Paris?"}
  ],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
  }}]
}
```

Two structural facts define this dialect and bite you later:

- **The system prompt is a message**, not a separate parameter. It lives inside the same `messages` array as everything else, distinguished only by `role: "system"`.
- **A tool call comes back as a `tool_calls` array on the assistant message, and the arguments are a JSON-encoded *string*, not a parsed object:**

```json
{"role": "assistant", "content": null, "tool_calls": [
  {"id": "call_abc", "type": "function",
   "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
]}
```

Read `"arguments": "{\"city\": \"Paris\"}"` carefully — that is a **string containing JSON**, which you must `json.loads()` yourself. It can be malformed (a truncated stream, a hallucinated trailing comma, a bad escape), so parsing can *throw*. You then feed the result back as a **separate message with `role: "tool"`**, linked by `tool_call_id`:

```json
{"role": "tool", "tool_call_id": "call_abc", "content": "18°C, clear"}
```

So one round trip of "call a tool and feed the result back" spans **three messages**: the assistant message carrying `tool_calls`, then your `role:"tool"` message, then the next assistant message with the final answer.

The newer **Responses API** (`POST /v1/responses`) is OpenAI's re-think of this shape, and it is genuinely different — not a rename:

- It is **stateful-capable**: instead of resending the whole history, you pass `previous_response_id` and OpenAI holds the prior turns server-side.
- It uses **`input`/`output` item lists** instead of `messages`. Items are typed: a function call is a `function_call` item, your reply is a `function_call_output` item — not an array bolted onto an assistant message.
- Reasoning is represented as first-class **reasoning items** in the output, with an optional encrypted form you can pass back to preserve the model's chain of thought across turns.

The consequence for gateways is sharp: a gateway that speaks "Chat Completions" is *not* speaking Responses. Server-side conversation state, typed reasoning items, and `previous_response_id` **do not exist** in the Chat Completions dialect at all — there is no field to translate them into. If your gateway's interface is Chat Completions (nearly all are), those capabilities are simply unavailable through it.

**Reasoning models** (the o-series and successors) add `reasoning_effort` (`"low" | "medium" | "high"`). The model spends hidden **reasoning tokens** before it answers; you are billed for them *as output tokens*, but you never see their contents. This is a first-class request parameter with **no equivalent field name** in any other dialect — a point we'll return to, because it is the cleanest example of a "same idea, different name" leak.

### Dialect 2 — Anthropic: content blocks, parsed tool args, system as a top-level param

Anthropic's **Messages API** (`POST /v1/messages`) looks similar at a glance and is structurally different in three ways that all matter.

**(a) The system prompt is a top-level parameter, not a message.** There is no `role: "system"` entry in `messages`; it's a sibling field:

```json
{
  "model": "claude-opus-4-8",
  "max_tokens": 1024,
  "system": "You are a booking assistant.",
  "messages": [{"role": "user", "content": "Weather in Paris?"}]
}
```

(Anthropic did add mid-conversation `role:"system"` messages on the newest Opus, but the *primary* system channel is the top-level param — the opposite default from OpenAI.)

**(b) Message content is a list of typed blocks, and tool calls are blocks with PARSED arguments.** Where OpenAI bolts a `tool_calls` array onto the assistant message and stringifies the args, Anthropic returns a `tool_use` **content block** whose `input` is already a JSON object:

```json
{"role": "assistant", "content": [
  {"type": "text", "text": "Let me check."},
  {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}}
]}
```

`"input": {"city": "Paris"}` — a real object, no `json.loads()`. You return the result as a `tool_result` **block inside a user message**, linked by `tool_use_id`:

```json
{"role": "user", "content": [
  {"type": "tool_result", "tool_use_id": "toolu_01", "content": "18°C, clear"}
]}
```

So the *same* round trip is **two messages** (assistant with a `tool_use` block, then a user message with a `tool_result` block) rather than OpenAI's three — and the result rides *inside a user turn* rather than in its own `tool` role. A gateway normalizing between these must (i) split or merge messages, (ii) move the system prompt between a message and a parameter, and (iii) stringify or parse the tool arguments depending on direction. Most gateways get the happy path right; the failure modes cluster around **parallel tool calls** (multiple `tool_use` blocks in one assistant turn, which must map to multiple `tool_calls` entries and multiple `role:"tool"` messages) and **interleaved text + tool_use** in the same turn.

**(c) Reasoning is "adaptive thinking," configured differently from OpenAI.** This is the sharpest naming disagreement in the whole lecture. OpenAI's reasoning models take `reasoning_effort`. Current Anthropic models don't have that field at all — you enable reasoning with `thinking: {"type": "adaptive"}` and tune depth with a *separate* nested field, `output_config: {"effort": "high"}` (levels run `low`/`medium`/`high`/`xhigh`/`max`). Both providers hide the raw chain of thought; both bill the reasoning as output tokens; both call the concept "reasoning effort" in prose. But the request-field names, nesting, and value sets are entirely different. A gateway that exposes a single `reasoning_effort` knob can forward it to OpenAI verbatim and must *rewrite* it for Anthropic — or, more commonly, just drops it. The lever that fixes quality on hard tasks becomes invisible.

**(d) Prompt caching via `cache_control` breakpoints.** This is Anthropic's biggest feature with no clean OpenAI-dialect equivalent, so it's the canonical "leak." You mark the end of a stable prefix with a breakpoint:

```json
"system": [
  {"type": "text", "text": "<50 KB of tool docs and instructions>",
   "cache_control": {"type": "ephemeral"}}
]
```

Mechanism, from first principles: **caching is a prefix match.** The API renders your request in a fixed order — **`tools` → `system` → `messages`** — hashes it up to each breakpoint, and stores that prefix. On a later request with a byte-identical prefix, that span is served from cache; **any byte change anywhere before the breakpoint invalidates everything after it.** TTL is **ephemeral**: ~5 minutes by default, or 1 hour with `{"type": "ephemeral", "ttl": "1h"}`. Max 4 breakpoints per request; the minimum cacheable prefix is model-dependent (~1024–4096 tokens — a prefix shorter than the minimum *silently* won't cache, no error).

You verify hits by reading `usage` on the response:

- `cache_creation_input_tokens` — tokens *written* to cache this request (you paid the write premium: **1.25× base for 5-minute TTL, 2× for 1-hour**).
- `cache_read_input_tokens` — tokens *served* from cache (you paid **~0.1× base input price**).
- `input_tokens` — the uncached remainder, at full price. Note this is the *remainder only*: total prompt size = `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`.

If `cache_read_input_tokens` is zero across repeated identical-prefix requests, a **silent invalidator** is in the prefix — a `datetime.now()` in the system prompt, an unsorted `json.dumps`, a per-request UUID, or a changed tool list (tools render first, so touching them busts *everything*).

**The Anthropic Agent SDK (survey level)** sits above all of this. It packages the Messages API into an agentic harness — built-in tools (file read/write, bash, web search), context management, and the `while stop_reason == "tool_use"` loop — so you write a prompt plus options instead of hand-rolling the loop. It is a *separate library*, not the raw Messages API, and not something a gateway exposes; treat it as an alternative to writing your own agent loop, not as a provider dialect. **For current Anthropic model ids and per-token/cached pricing, use the `claude-api` skill rather than hardcoding — model strings and rates change and the skill is the maintained source of truth.**

### Dialect 3 — Google Gemini / Vertex: `contents`, `parts`, `functionDeclarations`

Gemini renames nearly everything. The conversation is `contents`; each entry has a `role` (`"user"` or `"model"` — note: **`model`, not `assistant`**) and a `parts` array:

```json
{
  "systemInstruction": {"parts": [{"text": "You are a booking assistant."}]},
  "contents": [{"role": "user", "parts": [{"text": "Weather in Paris?"}]}],
  "tools": [{"functionDeclarations": [
    {"name": "get_weather",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
  ]}]
}
```

Tools are declared under `functionDeclarations`. A tool call comes back as a `functionCall` part (parsed args, like Anthropic), and you reply with a `functionResponse` part. Gemini also carries **safety settings** — per-category thresholds (harassment, hate speech, dangerous content, etc.) that can *block a response entirely*, returning a `finishReason` of `SAFETY` with **no content**. That's a failure mode with no analog in the OpenAI dialect a gateway speaks, so a gateway typically surfaces it as an empty or errored completion — and you lose the structured block reason that would tell you it was a *policy* block, not a model bug. Two flavors exist: the **Gemini Developer API** (bearer API key) and **Vertex AI** (Google Cloud project + region + Application Default Credentials) — same request shape, different auth and endpoint.

### Dialect 4 — AWS Bedrock Converse: unified shape, but IAM/SigV4 auth

Amazon Bedrock's **Converse API** is *itself* a normalization layer: one request shape across *all* Bedrock-hosted models (Anthropic, Meta, Mistral, Amazon Nova). Messages are a list with `role` and a `content` block list; the system prompt is a top-level `system` list; tools live under `toolConfig`; tool calls are `toolUse` blocks and results are `toolResult` blocks — structurally close to Anthropic's model:

```json
{
  "system": [{"text": "You are a booking assistant."}],
  "messages": [
    {"role": "user", "content": [{"text": "Weather in Paris?"}]},
    {"role": "assistant", "content": [
      {"toolUse": {"toolUseId": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}}
    ]},
    {"role": "user", "content": [
      {"toolResult": {"toolUseId": "tu_1", "content": [{"text": "18°C, clear"}]}}
    ]}
  ],
  "toolConfig": {"tools": [{"toolSpec": {"name": "get_weather", "inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}}}}}]}
}
```

Note `toolUseId` (camelCase) vs Anthropic's `tool_use_id` (snake_case) — a normalizer has to know even the casing conventions differ.

The thing that breaks every gateway assumption, though, is **auth**. OpenAI, Anthropic, and Gemini(dev) all authenticate with a **bearer key** in a header (`Authorization: Bearer …` or `x-api-key: …`). Bedrock uses **AWS IAM with SigV4 request signing** — there is *no* bearer token. To call it you need:

- AWS credentials (access key + secret, or an assumed role / instance profile),
- the correct **region** (models are region-scoped),
- **model access explicitly enabled** for that model in that account/region (a console toggle — new accounts have it off), and
- IAM permissions (`bedrock:InvokeModel`) on the calling principal.

Every request is cryptographically signed with SigV4 (the SDK does this for you). A gateway configured with a single string API key **cannot** authenticate to Bedrock — it needs the whole AWS credential chain, a region, and provisioned model access. This is why "point the gateway at Bedrock" is never a one-line change; it's an IAM and provisioning task.

```
Bearer-key providers            Bedrock
------------------              -------
Authorization: Bearer sk-...    (no bearer token)
  or x-api-key: sk-ant-...      SigV4-signed request derived from AWS creds
one string → done               access key + secret + region + role
                                + model-access toggle + bedrock:InvokeModel IAM policy
```

---

## Worked example — one tool round trip, four dialects

Task: the user asks a question, the model calls `get_weather(city="Paris")`, you return `"18°C"`, the model answers. Here is the *shape* of the assistant tool-call turn and your reply in each dialect, side by side.

| Concern | OpenAI (Chat Completions) | Anthropic (Messages) | Gemini / Vertex | Bedrock (Converse) |
|---|---|---|---|---|
| System prompt | `{"role":"system"}` message | top-level `system` param | top-level `systemInstruction` | top-level `system` list |
| Assistant role name | `assistant` | `assistant` | **`model`** | `assistant` |
| Tool call shape | `tool_calls[]` on assistant msg | `tool_use` content block | `functionCall` part | `toolUse` content block |
| Tool args type | **JSON string** (`"{\"city\":\"Paris\"}"`) | **parsed object** (`{"city":"Paris"}`) | parsed object | parsed object |
| Tool result | `role:"tool"` message, `tool_call_id` | `tool_result` block in **user** msg, `tool_use_id` | `functionResponse` part | `toolResult` block in user msg, `toolUseId` |
| Messages per round trip | **3** | 2 | 2 | 2 |
| Reasoning knob | `reasoning_effort` (reasoning models) | `thinking:{adaptive}` + `output_config.effort` | `thinkingConfig` (2.5 models) | passed through per-model |
| Caching | automatic prefix caching | **explicit `cache_control` breakpoints** | implicit/context caching API | per-model, varies |
| Auth | `Authorization: Bearer` | `x-api-key` | API key **or** Vertex ADC | **IAM + SigV4** |

**Cost math for cached input (Anthropic; numbers from the `claude-api` skill — approximate, verify with the skill).** Take Claude Opus 4.8 at **$5.00 / 1M input tokens**. You have a 50 KB system prompt ≈ 12,000 tokens, reused across 100 requests inside a 5-minute window.

- **No caching:** 100 × 12,000 × ($5.00 / 1e6) = **$6.00**, just for the repeated prefix.
- **With a 5-minute breakpoint (write premium 1.25×, read 0.1×):**
  - first request *writes* the cache: 12,000 × ($5.00 × 1.25 / 1e6) = **$0.075**
  - next 99 *read*: 99 × 12,000 × ($5.00 × 0.1 / 1e6) = **$0.594**
  - total ≈ **$0.67** — roughly **9× cheaper** on the prefix, plus a latency win because cached tokens skip prefill.
- **With a 1-hour TTL (write premium 2×):** the write costs 12,000 × ($5.00 × 2 / 1e6) = $0.12 instead of $0.075. You pay more up front, but the entry survives a gap in traffic — worth it when requests are bursty and spaced further apart than 5 minutes.

**Break-even** for the default 5-minute cache is ~2 requests within the TTL: write 1.25× + one read 0.1× = 1.35×, which already beats 2× uncached. For the 1-hour TTL you need ~3 (2× + 0.2× = 2.2× < 3×). That break-even is the whole economic argument for caching — and it is exactly the field a Chat-Completions-shaped gateway has no slot for.

---

## How it shows up in production

- **The gateway silently drops `cache_control`.** You wrote a beautifully cached Anthropic prompt, then routed it through the gateway's OpenAI-shaped interface. The breakpoint field has nowhere to go, so it's stripped. Your bill doesn't change, `cache_read_input_tokens` reads 0, and you burn an afternoon before realizing the gateway never forwarded the hint. **Symptom:** cache reads stuck at 0 through the gateway, non-zero when you call the SDK directly.
- **`reasoning_effort` has no home.** Set it on a reasoning model through a gateway that models only the OpenAI name, route the same logical request at Anthropic, and it's dropped — the Anthropic call runs at default depth. You get a fast, shallow answer and quietly higher error rates on hard tasks. The lever that would fix quality is invisible.
- **Bedrock "just won't connect."** A gateway configured with one API-key string cannot sign SigV4. The fix is not a config typo — it's provisioning AWS creds, choosing a region, and toggling model access. Teams routinely lose a day here because the gateway's error surfaces as a generic auth failure.
- **Tool args parse differently.** Code written against Anthropic (parsed `input` dict) breaks when the same logical call arrives through an OpenAI-shaped path as a JSON *string* — and occasionally as *malformed* JSON on a truncated stream. Always `json.loads()` defensively at the OpenAI boundary.
- **Gemini safety blocks look like empty responses.** A `finishReason: SAFETY` with no content, flattened by the gateway into "empty completion," makes you chase a phantom model bug instead of recognizing a content-policy block.

The through-line: gateways normalize the *common* shape well and the *differentiating* features poorly. The features that differentiate are the ones that move cost and latency, so those are precisely the ones you'll need to reach past the gateway for.

---

## Common misconceptions & failure modes

- **"A gateway makes all providers interchangeable."** It makes the *base call* interchangeable. Caching, reasoning depth, safety config, server-side conversation state, and auth are provider-specific and either leak or vanish.
- **"Tool args are always an object."** Only in Anthropic/Gemini/Bedrock. OpenAI Chat Completions gives you a **string** you must parse — and it can be invalid JSON.
- **"The system prompt is a message."** True for OpenAI Chat Completions; false for Anthropic/Gemini/Bedrock, where it's a top-level parameter. This is the #1 thing a normalizer moves around.
- **"Reasoning effort is one standard knob."** No — OpenAI's `reasoning_effort` and Anthropic's `thinking` + `output_config.effort` are different field names, nestings, and value sets for the same idea. A single gateway knob can't be right for both.
- **"Bedrock is just another API key."** No — IAM + SigV4 + region + model-access provisioning. There is no bearer token to paste.
- **"Prompt caching is free money everywhere."** It's an Anthropic-first, explicit-breakpoint mechanism (OpenAI's is automatic and shaped differently). Below the minimum prefix size, or with a silent invalidator in the prefix, it caches nothing — silently.
- **"I can hardcode model ids and prices."** They change. For Anthropic specifically, use the `claude-api` skill as the live source rather than baking in a string.

---

## Rules of thumb / cheat sheet

- **Default to a gateway** for provider-agnostic base calls; **keep a direct-SDK escape hatch** for caching, reasoning, and Bedrock.
- **OpenAI tool args = string.** Wrap `json.loads()` in a try/except at that boundary, always.
- **Anthropic/Gemini/Bedrock tool args = object.** No parsing; the `input` is ready to use.
- **System prompt:** a message for OpenAI Chat Completions; a top-level param for the other three.
- **Round-trip length:** OpenAI = 3 messages (assistant `tool_calls` → `role:"tool"` → assistant); Anthropic/Gemini/Bedrock = 2.
- **Reasoning knob names differ:** `reasoning_effort` (OpenAI) vs `thinking` + `output_config.effort` (Anthropic). Don't assume a gateway forwards either.
- **Verify caching** by reading `cache_read_input_tokens` — if it's 0 on a repeated prefix, hunt the invalidator (timestamps, UUIDs, unsorted JSON, a changed tool list) or check you're above the minimum prefix size.
- **Cache the *stable* prefix** (tools + system + shared context, in that render order), never the varying tail. Break-even ≈ 2 requests within a 5-min TTL, ≈ 3 within a 1-hour TTL.
- **Bedrock checklist before you debug anything else:** AWS creds present? correct region? model access enabled in console? `bedrock:InvokeModel` on the principal?
- **Look up Anthropic model ids/pricing in the `claude-api` skill**, don't guess.

---

## Connect to the lab

This week's lab builds the `new-model-smoke-test` repo through a **LiteLLM gateway** — one `models.yaml` line swaps providers. The lab's pitfall "letting the gateway hide too much" is this lecture made concrete: when you test caching or reasoning, drop to the raw Anthropic/OpenAI SDK and confirm the behavior *natively*, because LiteLLM won't expose `cache_control` or a reasoning knob uniformly across providers. Add one direct-SDK call to your repo that prints `cache_read_input_tokens`, so you can *see* the leak the gateway papers over instead of trusting that it forwarded your breakpoint.

---

## Going deeper (optional)

- **Official docs (skim for shapes, not prose):** OpenAI API reference at `platform.openai.com/docs` — read the Chat Completions and Responses guides side by side to feel the re-think; Anthropic Messages API + prompt caching at `docs.anthropic.com`; Google Gemini/Vertex at `ai.google.dev`; AWS Bedrock Converse at `docs.aws.amazon.com/bedrock`.
- **The `claude-api` skill** in this environment — authoritative for current Anthropic model ids, per-token and cached-input pricing, the caching invariants, and the adaptive-thinking/`effort` request shape.
- **LiteLLM docs** (`docs.litellm.ai`) — read the "Provider-specific params" and "pass-through" pages to see exactly which fields it forwards vs. drops.
- **Search queries:** `OpenAI Responses API vs Chat Completions reasoning items`, `Anthropic prompt caching cache_control breakpoints usage`, `Bedrock Converse API tool use toolConfig`, `AWS SigV4 signing bedrock InvokeModel IAM policy`, `Gemini functionDeclarations safetySettings finishReason SAFETY`.

---

## Check yourself

1. In OpenAI Chat Completions, what data type is a tool call's `arguments` field, and what defensive step must your code take that Anthropic's shape doesn't require?
2. Where does the system prompt live in Anthropic's Messages API vs OpenAI's Chat Completions, and why does that difference create work for a gateway?
3. A colleague reports their cached Anthropic prompt "isn't saving money" through the gateway. Name the single response field you'd check first and two likely causes if it reads zero.
4. Why can't a gateway authenticate to AWS Bedrock with a single bearer-key string? List the things Bedrock needs instead.
5. Using ~$5/1M input tokens and a 12,000-token prefix reused 100× in 5 minutes, roughly how much cheaper is the prefix with caching, and what is the approximate break-even in requests?
6. OpenAI and Anthropic both let you dial reasoning depth. Name the request field each uses, and explain why a gateway exposing a single knob can't serve both correctly.

### Answer key

1. It's a **JSON-encoded string** (e.g. `"{\"city\":\"Paris\"}"`), not an object. You must `json.loads()` it inside a try/except because the string can be malformed (truncated stream, bad escaping). Anthropic returns `input` as an already-parsed object, so no parsing step is needed.
2. Anthropic: a **top-level `system` parameter**. OpenAI Chat Completions: a **`role:"system"` message inside `messages`**. A gateway normalizing between them must move the prompt between a parameter and a message (and back) — a common place for dropped-system or off-by-one bugs.
3. Check **`usage.cache_read_input_tokens`**. If it's 0 on a repeated identical prefix, likely causes: the gateway stripped `cache_control` (no breakpoint reached the provider), or a **silent invalidator** in the prefix (a `datetime.now()`/UUID, unsorted `json.dumps`, or a changed tool list) is busting the prefix match — also possible the prefix is below the minimum cacheable size.
4. Bedrock uses **IAM + SigV4 request signing**, not a bearer token — there's nothing to paste into a single API-key field. It needs: AWS credentials (key+secret or an assumed role), the correct **region**, **model access enabled** for that model in that account/region, and `bedrock:InvokeModel` **IAM permission** on the calling principal.
5. No caching: 100 × 12,000 × $5/1e6 = **$6.00**. With caching: one write at 1.25× (~$0.075) + 99 reads at ~0.1× (~$0.59) ≈ **$0.67**, about **9× cheaper** on the prefix. Break-even is ~**2 requests** within the TTL (1.25× write + 0.1× read = 1.35× < 2× uncached).
6. OpenAI uses **`reasoning_effort`** (`low`/`medium`/`high`, top-level, on reasoning models). Anthropic uses **`thinking: {"type": "adaptive"}`** to enable reasoning plus **`output_config: {"effort": ...}`** (`low`…`max`) to tune depth — different field name, nesting, and value set. A gateway with one `reasoning_effort` knob can forward it verbatim to OpenAI but must *rewrite* it for Anthropic; most just drop it, so the depth lever silently disappears on one provider.
