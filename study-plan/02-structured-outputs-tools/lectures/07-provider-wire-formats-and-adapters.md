# Lecture 7: Provider Wire-Formats and the Normalizing Adapter

> Every tool-calling tutorial shows you *one* provider and pretends the shape is universal. It is not. The moment you support a second model vendor, you discover that OpenAI hands you tool arguments as a raw JSON *string* you must parse yourself, Anthropic hands you an already-parsed *dict*, and Gemini doesn't even use ids to link a result back to a call. This lecture is the field guide to those differences — the exact bytes on the wire, the id-linkage failures that produce cryptic 400s, and the one adapter abstraction that lets the rest of your codebase never know or care which provider it's talking to. After this you will be able to read any of the three providers' tool-call payloads by eye, parse their arguments defensively, wire results back with the correct id, and collapse all three behind a single `ToolCall`/`ToolResult` pair.

**Prerequisites:** Week 1 (JSON Schema, Pydantic, the honored-keyword diff), the universal tool loop from this week's theory, basic `try/except` and `json.loads`. · **Reading time:** ~22 min · **Part of:** Phase 2 (Structured Outputs & Tool Calling) Week 2

---

## The core idea (plain language)

All three providers implement the *same conversation shape*: you send tool definitions, the model **requests** a call (it never executes anything), you run the tool in your code, you send the result back linked to the request, and you loop. That's the universal loop from the theory reading, and it's genuinely provider-agnostic at the level of *concepts*.

The trouble is entirely at the level of *bytes*. The three vendors chose three different envelope formats for "here is a tool call" and "here is a tool result," and the differences are not cosmetic — they change what your code must do:

- **OpenAI** puts a `tool_calls[]` array on the assistant message. Each call's arguments are a **JSON string** you must `json.loads`. That string can be malformed or truncated, so parsing is a fallible operation, not a field access. You answer with a message whose `role` is `"tool"`, carrying a `tool_call_id`.
- **Anthropic** returns content *blocks*; a `tool_use` block carries an **already-parsed** `input` object — no `json.loads`. You answer with a `user` message containing a `tool_result` block that references `tool_use_id`.
- **Gemini** declares tools inside `functionDeclarations`, returns a `functionCall` **part** with parsed `args`, and you answer with a `functionResponse` part — and it links results back to calls **by function name, not by id.**

If you hand-write the loop three times, you will get the id linkage wrong at least once, crash on a malformed OpenAI arguments string at least once, and duplicate your schema-emission logic three times. The fix is a **normalizing adapter**: define one internal representation — `ToolCall(id, name, args: dict)` and `ToolResult(id, content)` — and push all the vendor-specific ugliness into `to_openai` / `to_anthropic` / `to_gemini` (schema out) and `parse_response` / `render_tool_turn` (payloads in and out). The loop, your tool registry, your security checks, your logging — all of it speaks only the internal representation and never sees a `tool_calls[]` or a `functionResponse` again.

---

## How it actually works (mechanism, from first principles)

Let's put the three wire formats side by side for a single call: the model wants `get_weather(city="Paris")`.

### OpenAI

The assistant turn comes back like this (fields trimmed):

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_9x2Kd",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"Paris\"}"
      }
    }
  ]
}
```

Look closely at `arguments`. It is a **string** — note the escaped quotes `\"`. It is not a JSON object nested inside the response; it is a string *containing* JSON, and you must parse it yourself:

```python
import json

for tc in msg.tool_calls:
    try:
        args = json.loads(tc.function.arguments)
    except json.JSONDecodeError as e:
        # Do NOT crash. Treat as a repairable error.
        args = None
        repair_note = f"arguments were not valid JSON: {e}"
```

Why can it be malformed? Two common causes. First, if the model hit `max_tokens` mid-generation, the arguments string is **truncated** — you get `{"city": "Par` with no closing brace. Second, even under normal completion the model occasionally emits invalid JSON (a trailing comma, an unescaped newline inside a string value). Because `arguments` is free-form text the model generates token by token, it is exactly as fallible as any other model output. So `json.loads` goes inside a `try/except`, and a failure becomes a *tool result with an error message fed back to the model* — not a stack trace that kills your process.

You reply with a **separate message** whose role is the literal string `"tool"`:

```json
{
  "role": "tool",
  "tool_call_id": "call_9x2Kd",
  "content": "18°C, partly cloudy"
}
```

The `tool_call_id` **must** equal the `id` from the assistant's tool call. If you omit it or send the wrong one, the API rejects the request (a 400 complaining that a tool message has no preceding tool call, or that a tool call went unanswered).

### Anthropic

The assistant turn is a list of content blocks. A tool call is a `tool_use` block:

```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "Let me check the weather."},
    {
      "type": "tool_use",
      "id": "toolu_01A9",
      "name": "get_weather",
      "input": {"city": "Paris"}
    }
  ]
}
```

The critical difference: `input` is an **object**, already parsed. `input["city"]` works directly. There is no `json.loads`, and therefore no `JSONDecodeError` to handle at this layer — Anthropic parses (and, with strict tool use, validates) the arguments server-side before you ever see them. That doesn't mean you skip validation entirely; you still run the args through your Pydantic model to enforce *business* rules. But the *syntactic* parse that bites you on OpenAI simply isn't your problem here.

You reply with a **`user`** message (not a `"tool"` role — Anthropic has no such role) containing a `tool_result` block:

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_01A9",
      "content": "18°C, partly cloudy"
    }
  ]
}
```

The linkage field is `tool_use_id` (note: *not* `tool_call_id` — different name, same job), and it must equal the `id` from the `tool_use` block. On an error, you set `"is_error": true` on the `tool_result` block instead of inventing an error convention. And if the assistant turn contained a `text` block *and* a `tool_use` block, you must echo **the whole content array** (text included) back as the assistant message before appending your `tool_result` user turn — dropping the text block corrupts the conversation.

### Gemini

Tools are declared inside a `functionDeclarations` array nested in `tools`. The model returns a `functionCall` **part**:

```json
{
  "role": "model",
  "parts": [
    {
      "functionCall": {
        "name": "get_weather",
        "args": {"city": "Paris"}
      }
    }
  ]
}
```

Like Anthropic, `args` is a parsed object. But notice what's **missing**: there is no `id`. Gemini links a result back to its call **by function name**. You reply with a `functionResponse` part:

```json
{
  "role": "user",
  "parts": [
    {
      "functionResponse": {
        "name": "get_weather",
        "response": {"result": "18°C, partly cloudy"}
      }
    }
  ]
}
```

The `name` in the `functionResponse` must match the `name` of the `functionCall`, and `response` must be an object (you can't hand it a bare string — wrap it, e.g. `{"result": ...}`). The absence of ids has a sharp consequence for **parallel calls**: if the model requests `get_weather("Paris")` *and* `get_weather("London")` in the same turn, name-matching alone is ambiguous. Gemini's convention is **positional/ordinal** — you return the `functionResponse` parts in the same order the `functionCall` parts arrived. Your adapter must therefore *synthesize* ids for Gemini (e.g. `f"{name}#{index}"`) so the rest of your code can keep using id-based matching, then strip them back to name-and-order on the way out.

### The three side by side

```
                OpenAI              Anthropic            Gemini
call container  tool_calls[]        content[] block     parts[] entry
call shape      function.arguments  tool_use.input      functionCall.args
args form       JSON STRING (parse) parsed object       parsed object
call id         id ("call_...")     id ("toolu_...")    NONE (name+order)
result role     "tool"              "user"              "user"
result field    tool_call_id        tool_use_id         name (matched)
error signal    (your convention)   is_error: true      (your convention)
```

Three formats, three linkage schemes, one loop that should never have to know any of it.

---

## Worked example

Let's trace one full round trip and count what changes. Task: user asks "weather in Paris?", one tool `get_weather`.

**Step 1 — emit the schema.** Same Pydantic model, three serializations. Your adapter's `to_*` methods produce:

- OpenAI: `{"type": "function", "function": {"name": ..., "description": ..., "parameters": <schema>}}`
- Anthropic: `{"name": ..., "description": ..., "input_schema": <schema>}`
- Gemini: `{"functionDeclarations": [{"name": ..., "description": ..., "parameters": <schema>}]}`

The `<schema>` is *nearly* the same JSON Schema, but each provider honors a **different subset of keywords** — that's the Week 1 finding, and it's a Week 2 DoD to diff it again for tool `parameters`. More below.

**Step 2 — model requests the call.** `parse_response` reads the vendor payload and returns a normalized list. For all three you end up with the identical internal object:

```python
ToolCall(id="call_9x2Kd", name="get_weather", args={"city": "Paris"})
```

For OpenAI, `parse_response` did the `try/except json.loads`. For Gemini, it synthesized `id="get_weather#0"`. Your loop sees neither detail.

**Step 3 — execute.** Loop code: `result = registry["get_weather"](**call.args)` → `"18°C, partly cloudy"`. Wrap as `ToolResult(id=call.id, content="18°C, partly cloudy")`.

**Step 4 — render the result back.** `render_tool_turn` takes the `ToolResult` and the original response, and produces the vendor-specific turn(s): a `role:"tool"` message for OpenAI, a `user`/`tool_result` block for Anthropic, a `functionResponse` part for Gemini — each with the right linkage field filled from `result.id` (name-matched for Gemini).

**Step 5 — loop.** Send the appended history back; model returns final text.

Now the numbers that matter. Suppose you support all three providers and someone adds a fourth tool. Without an adapter, that's changes in **three** loop implementations plus three schema emitters — 6 edit sites, and the id-linkage bug can hide in any of the three loops. With the adapter, the loop and registry are **one** implementation; you touch schema emission in one place (the `to_*` methods share a base schema and diverge only on honored keywords). The failure surface shrinks from "3 loops × N tools" to "1 loop." That is the entire economic argument for the pattern.

---

## How it shows up in production

**The malformed-arguments crash (OpenAI).** The single most common production incident here: `args = json.loads(tc.function.arguments)` with no `try/except`, in the hot path. It works in every test because your tests use short prompts that never truncate. Then a real user sends a request that makes the model emit 40 tool calls, the last one truncates at `max_tokens`, `json.loads` throws, and the whole request 500s. The fix costs three lines. The lesson: **OpenAI tool arguments are untrusted, fallible text.** Parse defensively; on failure, feed the error back as a tool result (`"your arguments were not valid JSON, please retry"`) and let the model self-correct — this recovers ~most of the time and never crashes.

**The id-linkage 400 (all providers).** Symptoms differ by vendor but the root cause is one: a mismatch between the id the model sent and the id you sent back.

- OpenAI: `400` — "an assistant message with tool_calls must be followed by tool messages responding to each tool_call_id" (you missed one), or "tool_call_id did not match" (you sent the wrong one).
- Anthropic: `400` — a `tool_result` references a `tool_use_id` that doesn't exist in the preceding turn, or a `tool_use` block went unanswered.
- Gemini: often *not* a hard error — instead the model **re-calls the same tool**, because your `functionResponse` name didn't match anything it was waiting on, so from its point of view the call was never answered. This is the nastiest failure mode: no exception, just an infinite-looking loop burning tokens until your `max_steps` guard trips.

The defensive habit: after building the result turn, assert that every model-requested id has exactly one matching result before you send. A three-line invariant check turns a cryptic 400 (or a silent re-call loop) into a clear local error.

**Parallel calls and the Gemini ordering trap.** When the model requests several calls at once, OpenAI and Anthropic let you match results to calls by id in any order. Gemini matches by name-and-position, so if you execute the calls concurrently with `asyncio.gather` and append results **in completion order** rather than **request order**, Gemini silently pairs the wrong result with the wrong call. Your adapter must preserve request order when rendering Gemini results even if you executed them out of order.

**Cost and latency.** None of this is free at scale. Each failed round trip (a crash-retry, a re-call loop, a repair cycle) is a full extra request — input tokens re-sent, output tokens re-generated. A single unbounded Gemini re-call loop can burn thousands of tokens before `max_steps` stops it. The adapter doesn't make calls cheaper, but it makes the *bug rate* lower, and every avoided repair cycle is a full request's worth of tokens and latency saved.

---

## Common misconceptions & failure modes

- **"Tool arguments are always parsed for me."** Only on Anthropic and Gemini. OpenAI hands you a string. Assuming a dict where there's a string is a `TypeError`/`AttributeError` waiting in production.
- **"`tool_call_id` and `tool_use_id` are the same field."** Different names, same role, different providers. Copy-pasting an OpenAI result-builder into an Anthropic path and keeping `tool_call_id` produces a 400.
- **"The result goes in a `tool` message everywhere."** Only OpenAI has a `"tool"` role. Anthropic and Gemini both put the result in a **`user`** turn (as a `tool_result` block / `functionResponse` part respectively).
- **"Gemini uses ids like the others."** It doesn't. It matches by name and order. Synthesize ids in your adapter for internal consistency; never rely on an id surviving the round trip on Gemini.
- **"If the arguments string is malformed, the request is doomed."** No — it's the most *repairable* failure there is. Feed the parse error back as a tool result and the model usually fixes it on the next turn. Crash only if you forget the `try/except`.
- **"One JSON Schema works identically across all three."** It does not — see below. A keyword you rely on (say `format: "email"` or a numeric `minimum`) may be silently dropped by one provider, so your "validated" args aren't actually constrained the way you think.
- **"Echoing the assistant turn is optional."** On Anthropic, if the assistant turn had a `text` block alongside the `tool_use`, you must replay the full content array. Dropping blocks corrupts the multi-turn state.

---

## Rules of thumb / cheat sheet

- **OpenAI args are a JSON string → always `json.loads` inside `try/except`.** Failure = repairable tool-result error, never a crash.
- **Anthropic / Gemini args are already parsed dicts** — no `json.loads`, but still validate against your Pydantic model for business rules.
- **Linkage field by provider:** OpenAI `tool_call_id` · Anthropic `tool_use_id` · Gemini `name` (+ request order).
- **Result turn role:** OpenAI `"tool"` · Anthropic `"user"` (`tool_result` block) · Gemini `"user"` (`functionResponse` part).
- **Gemini has no ids** — synthesize `name#index` internally; preserve request order when rendering results.
- **Before sending a result turn, assert every requested id has exactly one result.** Turns cryptic 400s / re-call loops into clear local errors.
- **Errors:** Anthropic has `is_error: true` on `tool_result`; OpenAI/Gemini need your own error-string convention in `content`.
- **Normalize to `ToolCall(id, name, args: dict)` and `ToolResult(id, content)`.** The loop, registry, and security layer see only these.
- **Adapter surface:** `to_openai/to_anthropic/to_gemini` (schema out) + `parse_response` (payload → `ToolCall[]`) + `render_tool_turn` (`ToolResult[]` → vendor turn).
- **Re-diff honored JSON-Schema keywords for tool `parameters`** — it's a Week 2 DoD, and the honored subset drifts, so verify empirically, don't trust memory.

### Honored JSON-Schema keywords (approximate — verify against current docs)

Reuse the Week 1 finding, now for tool `parameters`. Treat this as a *starting hypothesis to verify by running it*, not gospel — provider support changes:

| Keyword | OpenAI (strict) | Anthropic | Gemini |
| --- | --- | --- | --- |
| `type`, `properties`, `required` | yes | yes | yes |
| `enum` | yes | yes | yes |
| `description` | yes (read as guidance) | yes | yes |
| `additionalProperties: false` | **required** in strict | optional | not honored |
| all fields in `required` | **required** in strict (use nullable unions for "optional") | not required | not required |
| `minLength`/`maxLength`/`pattern` | limited | honored best-effort | often dropped |
| numeric `minimum`/`maximum` | limited | best-effort | often dropped |
| `anyOf` / unions | yes (strict rules) | yes | narrower support |
| nested objects / arrays | yes | yes | yes (watch `propertyOrdering`) |

The practical takeaway: a keyword marked "dropped" means the provider ignored your constraint — so the model can return a value your schema *said* was impossible, and only your Pydantic `model_validate` will catch it. Constrain in the schema *and* validate in code.

---

## Connect to the lab

This lecture is the design spec for `structio/toolloop/adapters.py`. Build `ToolCall`/`ToolResult` in `types.py`, then implement `to_openai/to_anthropic/to_gemini` for schema emission and `parse_response`/`render_tool_turn` for the round trip — exactly the surface above. Run the *same* extraction task through all three providers behind the one loop, and produce the honored-keywords diff for tool `parameters` (the Week 2 DoD table). Deliberately truncate an OpenAI response (small `max_tokens`) to exercise the malformed-arguments repair path, and send parallel calls through Gemini to prove your request-order preservation is correct.

## Going deeper (optional)

Real, named sources (root doc domains I'm confident exist; otherwise search queries):

- **OpenAI function-calling guide** — `platform.openai.com/docs`, search "OpenAI function calling guide" and "OpenAI Structured Outputs strict mode".
- **Anthropic tool use overview** — `docs.claude.com` (formerly `docs.anthropic.com`), search "Anthropic tool use overview" and "Anthropic tool_result is_error".
- **Gemini function calling** — `ai.google.dev/gemini-api/docs`, search "Gemini API function calling" and "Gemini functionResponse parts".
- **JSON Schema** — `json-schema.org` for the keyword reference when you build the honored-keyword diff.
- **instructor** (patches provider clients, re-asks on validation error) — search "instructor python library" / `python.useinstructor.com`.
- **Vercel AI SDK `generateObject`** — search "Vercel AI SDK generateObject" for how a production TS stack abstracts the same three providers.
- Search query for the pattern itself: "LLM provider adapter normalize tool calling" and "unifying OpenAI Anthropic Gemini tool calls".

## Check yourself

1. OpenAI hands you `function.arguments` as a string; Anthropic hands you `input` as a parsed object. What *one* operation must your OpenAI parse path do that your Anthropic path must not, and what should happen when it fails?
2. Name the result-linkage field for each provider and the role of the message the result travels in.
3. Gemini requests `convert(amount=10)` and `convert(amount=20)` in the same turn. Why can't you match results by name alone, and what does your adapter do to stay id-based internally?
4. You send an Anthropic `tool_result` with a `tool_use_id` that doesn't exist in the prior turn. What happens? Contrast with the analogous mistake on Gemini.
5. Your schema says `city` has `maxLength: 40`, but the provider silently drops `maxLength`. Where does the constraint actually get enforced, and what does that tell you about trusting provider-side validation?
6. Why does the adapter shrink your bug surface from roughly "3 loops × N tools" to "1 loop," and which two adapter methods carry the vendor-specific ugliness?

### Answer key

1. The OpenAI path must `json.loads` the arguments string (the Anthropic path skips it — `input` is already a dict). On failure it must **not crash**: catch `JSONDecodeError`, and feed the parse error back to the model as a tool result so it can retry. Malformed/truncated arguments are a repairable error, not a fatal one.
2. OpenAI: `tool_call_id`, in a message with `role: "tool"`. Anthropic: `tool_use_id`, in a `tool_result` block inside a `user` message. Gemini: matched by `name` (+ request order), in a `functionResponse` part inside a `user` turn.
3. Gemini has no ids and both calls share the name `convert`, so name alone is ambiguous. The adapter **synthesizes ids** (e.g. `convert#0`, `convert#1`) on parse so internal code stays id-based, and preserves **request order** when rendering `functionResponse` parts so the right result maps to the right call.
4. Anthropic returns a **400** (the `tool_use_id` references a nonexistent call, or a `tool_use` went unanswered). On Gemini the analogous mismatch (wrong `name`) typically produces **no error** — the model treats the call as unanswered and **re-calls the same tool**, an infinite-looking loop stopped only by your `max_steps` guard.
5. The constraint is only enforced by your **Pydantic `model_validate`** in your own code. The provider silently ignoring `maxLength` means "constrain in the schema" is not sufficient — provider-side validation is best-effort and varies, so you must validate again in code (LLM proposes, code disposes).
6. Without the adapter you re-implement the loop per provider, so a linkage/parse bug can live in any of the three, and each new tool touches all three. The adapter makes the loop and registry a single implementation; the vendor-specific ugliness lives only in `parse_response` (payload → `ToolCall[]`) and `render_tool_turn` (`ToolResult[]` → vendor turn), plus the `to_*` schema emitters.
