# Lecture 2: The Three Reliability Tiers and Which Provider Knob Delivers Each

> Every agent, RAG pipeline, and extraction service you will ever ship rests on one assumption: *a program can parse the model's output*. That assumption is a spectrum, not a boolean. This lecture gives you the exact ladder — three rungs, from "asked nicely" to "the decoder physically cannot emit invalid output" — and maps each rung to the precise parameter you set on OpenAI, Anthropic, and Gemini. After this you will be able to look at a task and a provider and say, without guessing, *which tier you need and which knob delivers it* — and you will recognize the two silent failure modes (Tier 1's distribution-shift breakage and Tier 2's shape-valid-but-wrong output) before they reach production.

**Prerequisites:** Phase 0 (tokenization, sampling, chat roles), Phase 1 (prompt anatomy, provider idioms), and comfort reading JSON Schema · **Reading time:** ~22 min · **Part of:** Phase 2 (Structured Outputs & Tool Calling), Week 1

---

## The core idea (plain language)

"Get JSON out of the model" is three different engineering problems wearing one sentence. They differ in *what is guaranteed*:

- **Tier 1 — prompt-and-pray.** You write "Respond in JSON" in the prompt and hope. Nothing is enforced. The model *usually* complies because it saw a lot of JSON in training, but "usually" is a probability, not a contract.
- **Tier 2 — JSON mode.** The provider guarantees the output is **syntactically valid JSON** — it will parse. It does **not** guarantee the output matches *your* schema. Required fields can be missing; keys you never defined can appear; a number can arrive as a string.
- **Tier 3 — schema-constrained Structured Outputs.** The provider constrains the decoder (or hard-validates) against **your JSON Schema**. Required fields are present, types are right, no stray keys. This is the tier you build production systems on.

The mental model that makes this stick: **the reliability tier is a property of the transport, not of the prompt.** You can write the world's best prompt and still be at Tier 1 if you didn't set the knob. The whole point of Week 1 is to stop treating output structure as a prompting problem and start treating it as a *decoding contract* you configure per provider.

The roadmap's governing rule — **"LLM proposes, code disposes"** — sits underneath all three tiers. Even Tier 3 output is *untrusted input to the next system*. Tier 3 guarantees *shape*; it never guarantees *truth*. A schema-valid invoice can still say the total is €40,000 when the document said €400. You still validate business rules in code (Pydantic validators, lecture 04). The tiers buy you *parseability and shape*, not correctness.

---

## How it actually works (mechanism, from first principles)

### Why Tier 1 fails silently — a token-probability view

An LLM generates one token at a time by sampling from a probability distribution over the vocabulary. "Respond in JSON" nudges that distribution toward JSON-shaped tokens, but nothing *removes* the non-JSON tokens from the running. At every position there is a small residual probability of emitting something that breaks the parse — a leading `Here is the JSON:`, a trailing markdown fence, a comment, a trailing comma, a smart-quote.

Make it numeric. Suppose that on a *clean, in-distribution* input the model emits well-formed JSON with probability 0.995 per response. That feels great in a demo. Now run it in production over 10,000 documents:

```
expected failures = 10,000 × (1 − 0.995) = 50 broken responses
```

Fifty crashes. And that 0.995 is the *optimistic* number. The killer is **distribution shift**: the per-response success rate is not constant. It drops exactly when your input drifts from what you tested on —

```
in-distribution input   → P(valid) ≈ 0.99
unusual formatting       → P(valid) ≈ 0.95
adversarial / long input → P(valid) ≈ 0.80
```

So the failure rate is coupled to the *weirdness* of the input, which is precisely the traffic you didn't test. Tier 1 passes every test you write and then breaks on the invoice with the unusual header you never imagined.

**Concrete silent-failure example.** You ship this:

```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user",
               "content": f"Extract vendor and total. Respond in JSON.\n\n{doc}"}],
)
data = json.loads(resp.choices[0].message.content)   # <-- the landmine
```

For 9,900 documents this works. Then a document arrives whose text contains the phrase *"the contract is null and void"*, and the model, primed by that context, wraps its answer in prose:

```
Sure! Based on the document, here is the extracted data:
```json
{"vendor": "Acme", "total": 400.0}
```
```

`json.loads` throws `JSONDecodeError` on the leading `Sure!`. No exception was anticipated here; the extraction service 500s. It is *silent* in the sense that nothing warned you at deploy time — the code, the tests, and the demo were all green. This is why the rule is **never ship Tier 1 alone.** It is fine as a fallback behind a validator + repair loop (lecture 05), never as the contract.

### Why Tier 2 (JSON mode) is a real but partial guarantee

JSON mode changes the decoder's job from "generate text" to "generate text that is a valid JSON document." Mechanically the provider constrains generation so the output is guaranteed to parse — balanced braces, quoted keys, no trailing commas, no prose wrapper. Your `json.loads` will not throw.

That is a genuine upgrade and it is *not* schema enforcement. The grammar being enforced is "valid JSON," full stop. Nothing ties the *keys and types* to what you asked for. So JSON mode eliminates the parse-crash failure and leaves a subtler one intact: **parseable-but-wrong-shape.**

**Concrete Tier-2 failure.** You want `{"vendor_name": str, "total_usd": float, "currency": str}`. JSON mode is on. The model returns:

```json
{"vendor": "Acme Corp", "total": "400.00"}
```

Walk through what just happened, because every failure here is legal JSON:

1. `vendor_name` → the model called it `vendor`. Your `data["vendor_name"]` raises `KeyError`.
2. `total_usd` → called `total`, and its value is the **string** `"400.00"`, not a float. `data["total_usd"] * 1.1` would `TypeError` even if the key matched.
3. `currency` → **omitted entirely.** A required field silently vanished.
4. Nothing stopped the model from *adding* a `notes` key you never defined.

The parse succeeded; the *contract* failed. This is the trap the spine warns about: **assuming JSON mode = your schema.** It doesn't. JSON mode is the floor of "it will parse," and for anything a downstream program consumes by field name, the floor isn't enough.

### Why Tier 3 is different — constrained decoding vs. post-hoc validation

Tier 3 is where the *schema* enters the decoder. There are two implementation strategies, and knowing which one a provider uses tells you what can still go wrong (deep mechanics in lecture 03 — keep it at a pointer here):

- **Constrained decoding (grammar-masked sampling).** The provider compiles your JSON Schema into a grammar/state machine. At each generation step it computes which tokens are *legal* given the schema and the partial output so far, sets the probability of every illegal token to zero, and samples only from the legal set. If the schema says the next thing must be the key `"currency"` or a closing brace, no other token can be emitted. Invalid-shape output is not "unlikely" — it is *impossible*. This is how OpenAI strict mode and Gemini `response_schema` work under the hood.

```
                     unconstrained                 constrained (grammar mask)
   vocab logits:  [ "Sure"  0.31 ]            [ "Sure"  −inf ]  ← masked
                  [ "{"     0.28 ]            [ "{"      0.62 ]  ← legal
                  [ "```"   0.14 ]            [ "```"   −inf ]  ← masked
                  [ "\n"    0.09 ]            [ "\n"    −inf ]  ← masked
                        │                            │
                    sample → "Sure"            sample → "{"     (only legal tokens survive)
```

- **Post-hoc / forced validation.** The provider generates, then validates the result against the schema before returning it (or forces the output through a tool whose parameters *are* the schema). The guarantee is delivered at the boundary rather than per-token, but the effect for you is the same: what comes back conforms, or the call errors rather than handing you garbage.

Either way, the guarantee you gain is **shape**: required fields present, types correct, enums within their allowed set, no undeclared keys (when you demand it). What you still do **not** get is semantic correctness — that terracotta invoice total could be transcribed wrong and still be a perfectly schema-valid `float`.

### The one-diagram summary

```
                        guarantees                              failure that remains
Tier 1  prompt-and-pray │ nothing                              │ crashes on parse (silent, drift-coupled)
Tier 2  JSON mode       │ syntactically valid JSON             │ wrong keys / missing fields / wrong types
Tier 3  Structured Out. │ conforms to YOUR JSON Schema (shape) │ semantics can still be wrong  ← code disposes
```

---

## Worked example — the same extraction across all three tiers

Task: pull `{vendor_name, total_usd, currency}` out of 10,000 messy invoice texts. `currency` is a closed set (`USD`, `EUR`, `GBP`). Let's follow the numbers.

**Tier 1.** Prompt says "Respond in JSON." Measured on your 200-doc eval set: 198/200 valid = 99% first-pass. Looks shippable. Extrapolate honestly, accounting for the ~8% of production traffic that is oddly formatted (P(valid)≈0.90 there vs 0.99 on clean):

```
clean traffic:  9,200 × 0.99 = 9,108 valid   →  92 failures
odd traffic:      800 × 0.90 =   720 valid    →  80 failures
                                    total broken ≈ 172 / 10,000 (1.7%)
```

172 pages route to a crash or a repair loop. And notice the eval set *understated* it, because your eval was mostly clean docs.

**Tier 2 (JSON mode).** Parse-crashes go to ~0. But now measure *schema conformance*, not just parseability. Say the model uses the wrong key name or a string-typed total on 6% of docs:

```
parse failures:        ~0
schema-shape failures:  10,000 × 0.06 = 600  (KeyError / TypeError downstream)
```

You traded 172 loud crashes for 600 quiet wrong-shape objects — arguably *worse*, because a `json.loads` crash is at least visible, whereas `data.get("total")` returning `None` sails through and corrupts your database.

**Tier 3 (schema-constrained).** With `strict` / `response_schema` / forced-tool, shape failures go to ~0: every returned object has all three keys, `total_usd` is a number, `currency` is one of the three enum values. Remaining failures are *semantic only* (wrong number transcribed), caught by your Pydantic business-rule validator (line items sum to total), not by the transport. Your extraction service now has a hard contract at the boundary and a separate, testable correctness layer behind it.

The lesson in one line: **each tier removes one class of failure and reveals the next.** You climb to Tier 3 to remove shape failures so that the only thing left to engineer is *truth*.

---

## How it shows up in production (cost / latency / quality / debugging)

**Latency: the first-call schema-compile spike (OpenAI strict).** The first time OpenAI sees a given strict schema, it compiles it into the decoding grammar — a one-time cost that can add noticeable latency (often on the order of a few seconds) to that first request. The compiled grammar is then cached, so subsequent calls with the *same* schema pay nothing. The production failure mode: an engineer sees the first-call spike, panics, and "optimizes" by dropping `strict=True` — silently demoting the whole service from Tier 3 to Tier 2. Don't. Warm the schema once at startup and move on. (Depth in lecture 03.)

**Cost: constrained decoding is nearly free at inference; the tokens cost the same.** The grammar mask is cheap arithmetic per step. What *does* cost you is over-modeling — giant enums and deeply nested schemas inflate the grammar and the token count, and they degrade quality (the model spends capacity on structure). Flat schemas with bounded arrays and modest enums are cheaper *and* more accurate (lecture 04).

**Quality: forcing JSON too early hurts reasoning.** A structured-output constraint applied to the whole response makes the model spend capacity satisfying the grammar instead of thinking. Mitigation is a `reasoning`/`scratchpad` string field placed *first* in the schema (so the model reasons before it fills the structured fields), or a two-pass reason-then-extract. This is a real, measurable quality lever — you will feel it in the lab.

**Debugging: the tier determines the shape of the bug.** Tier 1 bugs are `JSONDecodeError` stack traces (loud, easy to spot, annoying under drift). Tier 2 bugs are `KeyError`/`TypeError`/`None`-propagation *downstream* of the parse (quiet, land in your data). Tier 3 bugs are *semantic* — the shape is perfect and the value is wrong — so they only surface in your business-rule validators and evals. Knowing the tier tells you *where to look*.

**Observability.** In production you log, per request, which tier the call used and whether output validated on first pass. First-pass schema-valid rate is your headline reliability metric; a drop signals distribution shift before your users file tickets.

---

## Common misconceptions & failure modes

- **"JSON mode gives me my schema."** No. JSON mode = valid JSON only. Missing required fields and invented keys are all still legal. This is the single most common Tier confusion.
- **"A good enough prompt makes structured output reliable."** The tier is a transport property. A perfect prompt at Tier 1 is still Tier 1. Set the knob.
- **"Tier 3 means the output is correct."** Tier 3 guarantees *shape*, never *semantics*. The number can still be wrong. Keep the business-rule validator.
- **"Anthropic can't do schema-constrained output because it has no `json_schema` response format."** It can — via a **forced tool** (below). The mechanism differs; the guarantee is equivalent.
- **"Temperature 0 makes Tier 1 deterministic enough."** Lower temperature reduces variance but does not enforce a grammar; drift still breaks you.
- **"The first-call latency on OpenAI strict is a bug to optimize away."** It's a one-time schema compile that's then cached. Dropping strict to avoid it silently demotes you to Tier 2.
- **Optional ≠ omit-from-required (OpenAI strict).** In strict mode, "optional" is modeled as a nullable union (`["string","null"]`) with the field still listed in `required` — not by leaving it out of `required`. (Deep mechanics: lecture 03.)

---

## Rules of thumb / cheat sheet

- **Default to Tier 3** for anything a *program* consumes. Reserve Tier 1 for prose/reasoning that a human reads.
- **Never ship Tier 1 alone.** Only behind a validator + repair loop as a fallback.
- **Tier 2 is a stepping stone, not a destination** — use it only when the provider/model can't do Tier 3, and always pair it with a schema validator in code.
- **Put a `reasoning` field first** in your Tier-3 schema, or reason in a separate pass, to protect answer quality.
- **Warm OpenAI strict schemas at startup** to eat the one-time compile spike off the hot path.
- **Shape ≠ truth.** Always keep a code-side business-rule validator behind Tier 3.
- **Flat + bounded schemas** beat nested/giant-enum schemas on cost, latency, *and* quality.

### The decision table — task + provider → tier + knob

| Task | Tier | OpenAI knob | Anthropic knob | Gemini knob |
|---|---|---|---|---|
| Program consumes fields by name (extraction, classification, RAG-to-JSON) | **3** | `response_format={"type":"json_schema","json_schema":{...},"strict":true}` (or SDK `.parse()`) | Forced tool: `tools=[{name,input_schema=schema}]` + `tool_choice={"type":"tool","name":...}`; read parsed `input` | `generation_config` with `response_mime_type="application/json"` + `response_schema` |
| Model must *do* something (call a function/API) | **3 (tool calling)** | function/tool with `strict:true` params | tool with `input_schema` (native idiom) | `function_declarations` with typed params |
| Model on this provider/version can't do strict schema yet | **2** | `response_format={"type":"json_object"}` + validate in code | forced tool (still Tier 3 — prefer it) or validate in code | `response_mime_type="application/json"` (no schema) + validate |
| Prose / reasoning a human reads | **1** | plain prompt | plain prompt | plain prompt |
| Prototype / throwaway | **1**, then climb before ship | "Respond in JSON" + `json.loads` in try/except | same | same |

Provider invocation cheat sheet:

```python
# --- OpenAI: Tier 3, strict json_schema ---
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "invoice", "strict": True, "schema": INVOICE_SCHEMA},
    },
)
# or the parse helper (returns a typed object):
resp = client.beta.chat.completions.parse(model=..., messages=..., response_format=Invoice)

# --- Anthropic: Tier 3 via a FORCED TOOL (no json_schema response format) ---
resp = client.messages.create(
    model="claude-3-5-haiku-latest", max_tokens=1024, messages=[...],
    tools=[{"name": "record_invoice",
            "description": "Record the extracted invoice.",
            "input_schema": INVOICE_SCHEMA}],
    tool_choice={"type": "tool", "name": "record_invoice"},
)
args = next(b.input for b in resp.content if b.type == "tool_use")  # PARSED dict, not a JSON string

# --- Gemini: Tier 3 via response_schema ---
resp = model.generate_content(
    prompt,
    generation_config={"response_mime_type": "application/json",
                       "response_schema": INVOICE_SCHEMA},
)
```

Two provider-specific notes worth internalizing now (mechanics in lecture 03):

- **Anthropic's args come back as an already-parsed object** (`tool_use.input` is a dict), unlike OpenAI tool calls whose `function.arguments` is a **JSON string** you must `json.loads` (and which can be malformed). This asymmetry bites everyone who writes a shared adapter. *(Pointer: recent Claude models also added a native structured-outputs mode — `output_config.format` / strict tool params — but the forced-tool idiom is the portable, always-available Tier-3 path and the one to learn first.)*
- **Gemini's `response_schema` uses an OpenAPI-subset dialect**, not full JSON Schema — nullability, `anyOf`, and property ordering behave differently and some keywords are silently dropped. Log what each provider honored.

---

## Connect to the lab

Week 1's lab (`structio/`) is exactly this lecture made executable: you build one `Invoice` Pydantic model and get **validated** output from OpenAI (`strict` `json_schema`), Anthropic (forced `record_invoice` tool), and Gemini (`response_schema`) — Tier 3 on all three. The Definition of Done ("≥18/20 valid, garbage inputs escalate not crash") is the operational form of "climb to Tier 3, keep a validator behind it." When you print OpenAI's first-call latency vs. subsequent calls, you are *measuring* the schema-compile spike this lecture warned you not to optimize away.

---

## Going deeper (optional)

Real, named resources — verify current URLs by searching the titles:

- **OpenAI** — official docs: "Structured Outputs guide" on `platform.openai.com/docs` (covers `json_schema`, `strict`, and the SDK `.parse()` helpers and refusal handling). Search: *"OpenAI Structured Outputs strict json_schema"*.
- **Anthropic** — official docs: "Tool use" and "Increase output consistency (JSON)" on `docs.anthropic.com`. Search: *"Anthropic tool use forced tool_choice input_schema"* and *"Anthropic structured outputs output_config format"* for the newer native mode.
- **Google Gemini** — official docs: "Generate structured output / responseSchema" on `ai.google.dev`. Search: *"Gemini API response_schema response_mime_type structured output"*.
- **Constrained-decoding mechanics** (background for lecture 03): the **Outlines** library README/docs and the **XGrammar** project. Search: *"Outlines structured generation"*, *"XGrammar constrained decoding"*, and the paper *"Grammar-Constrained Decoding"*.
- **Pydantic v2** docs (`docs.pydantic.dev`) for `model_json_schema()` and validators, and **instructor** (`python.useinstructor.com`) for the response-model + re-ask pattern you'll meet in lecture 05.

## Check yourself

1. A colleague says "I turned on JSON mode, so my `data["total_usd"]` access is safe now." What's wrong with that reasoning, and what class of bug remains?
2. Why does a Tier-1 pipeline pass every test in your eval set and then fail in production? Name the mechanism.
3. On OpenAI, the first request with a new strict schema is noticeably slower than the rest. What is happening, and why is "drop `strict` to fix it" the wrong move?
4. Anthropic has no `json_schema` response format. Describe the idiomatic Tier-3 path, and state one concrete difference in how the returned arguments arrive vs. an OpenAI tool call.
5. You reach Tier 3 and every response now conforms to your schema. Give a concrete example of a bug that Tier 3 *cannot* catch, and say which layer catches it.
6. For "summarize this contract into a paragraph a lawyer will read," which tier is appropriate and why?

### Answer key

1. JSON mode guarantees only *syntactically valid JSON*, not conformance to *your* schema. The parse won't crash, but the key can be missing, named differently, or the value can be a string instead of a float — so `data["total_usd"]` can `KeyError` or return a wrong-typed value. The remaining bug class is **parseable-but-wrong-shape** (missing/renamed/mistyped fields, invented keys).
2. Per-response `P(valid)` is not constant — it's coupled to how far the input drifts from what you tested (**distribution shift**). Eval sets are mostly in-distribution, so they show a high success rate; production carries oddly-formatted/adversarial inputs where `P(valid)` drops, producing failures your tests never exercised.
3. The provider is **compiling your JSON Schema into a decoding grammar** the first time it sees that schema (one-time cost), then caching it — subsequent calls are fast. Dropping `strict` removes the compile but silently **demotes the service from Tier 3 to Tier 2**, reintroducing wrong-shape failures. Fix: warm the schema at startup instead.
4. Define a **tool whose `input_schema` is your JSON Schema** and force it with `tool_choice={"type":"tool","name":...}`; read the arguments from the `tool_use` block. Difference: Anthropic returns `tool_use.input` as an **already-parsed object (dict)**, whereas OpenAI returns `function.arguments` as a **JSON string** you must `json.loads` (and which can be malformed/truncated).
5. **Semantic** errors: e.g., the schema-valid object says `total_usd: 40000.0` when the document said `400.00`, or the vendor name is transcribed wrong. Shape is perfect, value is false. Caught by **code-side business-rule validators** (e.g., a Pydantic `@model_validator` checking that line items sum to the total) and by your evals — never by the transport.
6. **Tier 1** (plain prompt). The output is prose a human reads, not something a program parses by field; forcing JSON would waste model capacity on structure and hurt the writing. Structured tiers are for machine-consumed output only.
