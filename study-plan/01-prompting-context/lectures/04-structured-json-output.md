# Lecture 4: Forcing and Validating Structured JSON Output

> Your extraction endpoint is only as reliable as the parser downstream of it. A model that returns ```` ```json {...} ``` ```` 95% of the time and "Here's the invoice data you asked for:\n\n{...}" the other 5% is not a 95%-reliable service — it's a service that throws `JSONDecodeError` on 1 request in 20 and pages you at 3am. This lecture is about eliminating that failure mode at the source: using each provider's *native constrained-decoding* mechanism so the bytes are machine-parseable by construction, and then validating them programmatically so you never confuse "it parsed" with "it's correct." After this you will be able to author a JSON Schema that fits the invoice task without over-constraining it, wire the same schema into Anthropic, OpenAI, Gemini, and Ollama behind one provider-agnostic adapter, and write a validator that separates *parses* from *right* — the distinction that decides whether your Definition of Done is real.

**Prerequisites:** Lecture 12 (sampling — you'll see how constrained decoding masks the token distribution), Phase 0 tokenization, basic JSON Schema literacy, comfort with `pytest` · **Reading time:** ~28 min · **Part of:** Prompting & Context Engineering, Week 1

---

## The core idea (plain language)

There are three escalating levels of "make the model return JSON," and only the top one is engineering:

1. **Ask nicely.** "Respond with JSON matching this shape." Works most of the time. Fails on edge cases, long outputs, and chatty models that wrap the JSON in prose or a Markdown fence. This is a *prompt*, not a *guarantee*.
2. **Constrain the decoder.** Tell the provider "the response MUST be valid JSON conforming to this schema," and the provider enforces it *during token sampling* — at each step it masks out any token that would break the grammar. The output is valid JSON by construction. This is what `output_config.format` (Anthropic), `response_format` + `strict:true` (OpenAI), and `response_schema` (Gemini) do.
3. **Validate what came back.** Even a schema-valid response can be *wrong* — right shape, wrong values. So you `json.loads()` it, check the keys and types you actually depend on, and compare against ground truth. "Parses" and "correct" are two different gates, and conflating them is the single most common way teams ship a broken extractor that passes its own tests.

The mental model: **constrained decoding removes the parsing risk; validation removes the correctness risk. You need both, and they are not the same check.** A pipeline that only does level 2 will happily hand you `{"total_amount": 0.0, "currency": "USD"}` for an invoice that clearly says €1,240.50 — perfectly valid, completely wrong.

---

## How it actually works (mechanism, from first principles)

### Constrained decoding: masking the distribution

Recall from Lecture 12 that at each generation step the model produces a probability distribution over its entire vocabulary, and the sampler picks one token. Constrained decoding inserts a filter *before* the pick: given the JSON emitted so far, a grammar engine computes which tokens are *legal next tokens* and sets the probability of every illegal token to zero.

Concretely, suppose the schema requires the response to start with `{` and the first key to be `"vendor"`. After the model has emitted `{"vend`, the only tokens that keep the output on a path to a valid document are ones that continue `"vendor"`. If the raw distribution was:

```
token     raw p
"or":     0.55   ← legal (completes "vendor")
"ors":    0.20   ← illegal (no such key) → masked to 0
" ":      0.10   ← illegal inside a key string → masked to 0
"ing":    0.08   ← illegal → masked to 0
...
```

After masking and renormalizing, `"or"` gets ~0.72 and the illegal tokens get 0. The model *cannot* emit a broken key even if its raw distribution wanted to. This is why constrained decoding is a hard guarantee, not a strong suggestion: malformed JSON is unreachable in the token graph.

Two consequences worth internalizing:

- **It costs a little latency, not accuracy of *content*.** The mask only forbids *syntactically* illegal tokens; among legal tokens the model still chooses by its own probabilities. So constraining the *format* does not force a particular *value*. The model can still be wrong about the number — it just can't be wrong about the JSON.
- **An unsatisfiable schema is a real failure mode.** If the grammar allows no legal continuation (e.g. you required a field the model has no information for and forbade `null`), providers differ in what happens — covered under failure modes below.

### The three provider dialects

Same idea, three spellings. Here is the invoice schema and each provider's call.

```python
SCHEMA = {
    "type": "object",
    "properties": {
        "vendor":         {"type": "string"},
        "invoice_number": {"type": "string"},
        "date":           {"type": "string"},                       # ISO-8601, see note
        "total_amount":   {"type": "number"},
        "currency":       {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY", "BGN"]},
    },
    "required": ["vendor", "invoice_number", "date", "total_amount", "currency"],
    "additionalProperties": False,
}
```

**Anthropic** — pass `output_config.format` on `messages.create()`. The first text block is then guaranteed valid JSON.

```python
resp = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    messages=[{"role": "user", "content": f"<document>{text}</document>\nExtract the invoice fields."}],
)
data = json.loads(next(b.text for b in resp.content if b.type == "text"))
```

(The old top-level `output_format` parameter is deprecated — use `output_config.format`. Anthropic structured outputs are supported on Opus 4.8, Sonnet 5, Haiku 4.5, Fable 5, and legacy Opus 4.5/4.1. The SDK's `client.messages.parse(...)` with a Pydantic model wraps this and validates for you.)

**OpenAI** — `response_format` with a named `json_schema` and `strict: True`. This is where the two non-obvious requirements live:

```python
resp = client.chat.completions.create(
    model="gpt-4.1",  # or a current strict-capable model
    messages=[{"role": "user", "content": f"Document:\n{text}\n\nExtract the invoice fields."}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "invoice", "strict": True, "schema": SCHEMA},
    },
)
data = json.loads(resp.choices[0].message.content)
```

Under `strict: True`, OpenAI requires **`additionalProperties: False` on every object** *and* **every property listed in `required`** — there is no notion of a merely-optional property. If you leave a key out of `required`, the API rejects the schema. This trips up everyone at least once, because standard JSON Schema treats un-required properties as optional; strict mode does not. To model a genuinely optional/missing field you make it *nullable* instead (next section).

**Gemini** — `response_mime_type` plus `response_schema`:

```python
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"Document:\n{text}\n\nExtract the invoice fields.",
    config={"response_mime_type": "application/json", "response_schema": SCHEMA},
)
data = json.loads(resp.text)
```

Setting `response_mime_type="application/json"` *without* a schema still forces JSON but not a shape; the schema is what pins the fields.

Notice all three take the *same* `SCHEMA` dict as input and, after the provider-specific unwrapping, hand you a Python dict. That symmetry is the whole point of the adapter contract below.

---

## Worked example

Take one deliberately hard invoice — European date, a currency the model must infer, and a total that must be read, not computed:

```
Rechnung Nr. 2024-0417
Muster GmbH, Berlin
Datum: 17.04.2024
Zwischensumme: 1.040,34
MwSt (19%): 197,66
Gesamtbetrag: 1.238,00 €
```

Ground truth: `{vendor: "Muster GmbH", invoice_number: "2024-0417", date: "2024-04-17", total_amount: 1238.00, currency: "EUR"}`.

**What constrained decoding guarantees:** the response is a JSON object with exactly those five keys, `total_amount` is a JSON number, `currency` is one of the five enum values. No fence, no prose, no sixth key. You will *never* get a `JSONDecodeError` here.

**What it does not guarantee — the traps:**

- `total_amount`: the model must pick `1238.00`, not `1040.34` (subtotal) or `197.66` (tax). Constrained decoding cannot help; it only enforces "a number."
- `currency`: the enum forbids `"€"` or `"euros"`, so the model is *forced* onto one of the five codes — but it could still pick `USD`. The enum narrows the failure surface; it doesn't eliminate it.
- `date`: `"17.04.2024"` vs `"2024-04-17"`. The schema says `string`; both parse. Only *your prompt* ("normalize dates to ISO-8601 YYYY-MM-DD") and *your validation* catch the format drift.

Now the two-gate check. Suppose the model returns:

```json
{"vendor": "Muster GmbH", "invoice_number": "2024-0417", "date": "2024-04-17", "total_amount": 1040.34, "currency": "EUR"}
```

```python
def check(pred: dict, expected: dict) -> dict:
    # Gate 1: PARSES — structural. (Already guaranteed if we got here via json.loads.)
    required = {"vendor", "invoice_number", "date", "total_amount", "currency"}
    assert required.issubset(pred), f"missing keys: {required - set(pred)}"
    assert isinstance(pred["total_amount"], (int, float)), "total_amount not numeric"

    # Gate 2: CORRECT — value-level, per field.
    return {k: (pred.get(k) == expected[k]) for k in expected}

# → {"vendor": True, "invoice_number": True, "date": True,
#    "total_amount": False, "currency": True}
```

Gate 1 is green. Gate 2 shows `total_amount: False` — the model grabbed the subtotal. **This response passes "parses" and fails "correct."** A test suite that only asserts `json.loads(output)` succeeds would score this a pass and ship it. That is the mistake the whole lecture is built to prevent.

Field accuracy for this one record: 4/5 = 80%. Across a 30-invoice set you report *per-field* accuracy (e.g. `total_amount` 26/30, `currency` 30/30) and *overall* — because a single aggregate number hides that your currency enum is perfect while your amount extraction is your real problem.

---

## How it shows up in production

**Latency and the schema-compilation tax.** The first request with a *new* schema pays a one-time compilation cost while the provider builds the grammar/state machine; subsequent requests with the same schema hit a cache (Anthropic caches for ~24h). In practice: your first call after a deploy that changed the schema is slower. Don't benchmark structured-output latency on a cold schema and panic — warm it first. Keep schemas byte-stable across requests so you stay on the cached path (the same discipline as prompt caching in Week 3).

**Truncation from `max_tokens` is the sneakiest failure.** Constrained decoding guarantees valid JSON *if the model is allowed to finish*. If generation hits `max_tokens` mid-object, you get `{"vendor": "Muster GmbH", "invoice_num` — truncated, `stop_reason: "max_tokens"`, and `json.loads` throws. The guarantee is conditional on completion. **Always check the stop reason before parsing**, and size `max_tokens` for the worst-case output plus headroom. For the five-field invoice, a few hundred tokens is plenty; for a schema with a long array of line items, budget accordingly. This is a real incident pattern: a schema grows an array field, outputs get longer, `max_tokens` stays at its old value, and 3% of requests start returning truncated JSON that your parser rejects.

**Refusals bypass the schema entirely.** If a safety classifier declines the request, you get `stop_reason: "refusal"` (Anthropic/OpenAI both surface a refusal path) and the content does **not** conform to your schema — there's nothing to parse. Branch on the stop reason *first*; only `json.loads` on a clean completion. Code that does `json.loads(resp.content[0].text)` unconditionally will crash on the refused request rather than handling it.

**Over-constraining hurts quality and reliability.** It is tempting to lock everything down: `minLength`, `maximum`, regex patterns, tight enums. Three problems. (1) Providers *don't support* most numeric/string constraints in strict mode (see below) — the Anthropic/OpenAI Python SDKs silently strip `minimum`/`maxLength`/etc. and validate them client-side, so your "constraint" isn't enforced where you think it is. (2) A too-tight enum on a field the model legitimately needs to leave open forces a wrong pick — if a real invoice is in CHF and your enum only has five currencies, the model is *forced* to lie. (3) Forbidding `null` on a field that's genuinely absent in the source creates an unsatisfiable situation the model resolves by hallucinating a plausible-looking value. The rule: **constrain the shape you truly depend on; leave the model room to say "I don't have this."**

**Cost is basically unchanged.** Constrained decoding doesn't add tokens; if anything it removes the "Here is the JSON you requested:" preamble, saving a handful of output tokens per call. The savings are noise; the reliability is the point.

---

## Common misconceptions & failure modes

- **"Valid JSON means correct data."** No. This is the headline error. Schema-valid + wrong value is the default failure once you've eliminated parse errors. Separate the gates.
- **"`strict: true` on OpenAI works like normal JSON Schema."** No — it demands `additionalProperties: false` on every object *and* every property in `required`. There are no optional properties; model absence with nullable types instead.
- **"My `minimum`/`maxLength`/`pattern` constraint is enforced by the provider."** Usually not. Supported: types, `enum`, `const`, `anyOf`, `$ref`/`$defs`, string `format` values (`date`, `date-time`, `uri`, `email`, `uuid`, …), and `additionalProperties: false`. **Not** supported in constrained decoding: numeric bounds (`minimum`, `maximum`, `multipleOf`), string length (`minLength`, `maxLength`), and complex array constraints. The Python/TS SDKs strip these and validate client-side — so enforce them in *your* validator, not by trusting the schema.
- **"Structured outputs compose with everything."** No — they're incompatible with citations (Anthropic returns 400) and with assistant-message prefill (which itself 400s on current Claude models — use `output_config.format` *instead of* prefill, not with it). They *do* work with streaming, batches, token counting, and extended thinking.
- **Unsatisfiable schemas behave differently per provider.** If the grammar admits no legal completion, you may get an error, a refusal, a truncated response, or a degenerate value — the behavior is not uniform across Anthropic/OpenAI/Gemini. Don't build a schema that *can't* be satisfied (e.g. required field, no `null` allowed, information genuinely absent) and expect a clean cross-provider error. Model absence explicitly.
- **"Setting `response_mime_type: application/json` on Gemini pins the shape."** It only forces *valid JSON*. Without `response_schema` you get an arbitrary object.
- **"Constrained decoding fixes a bad prompt."** It fixes the *format*, never the *content*. If your prompt doesn't tell the model which number is the total, the enum and schema won't save you.

### Modeling missing data (the nullable pattern)

For the invoice task, `currency` is sometimes genuinely absent from the document. Do **not** drop it from `required` (OpenAI strict rejects that) and do **not** force a guess. Make it nullable:

```python
"currency": {"type": ["string", "null"], "enum": ["USD", "EUR", "GBP", "JPY", "BGN", None]}
```

Now `null` is a *legal* completion, so the model has an honest "I don't have this" path. Your validator then decides whether `null` counts as correct for that record. This is the difference between a model that admits missing data and one that hallucinates `"USD"` onto every currency-less invoice.

---

## Rules of thumb / cheat sheet

- **Always use the provider's native constrained decoding.** Never regex JSON out of prose in production.
- **Two gates, always separate:** `json.loads` + key/type check = *parses*; per-field value comparison against ground truth = *correct*. Done means both.
- **Check `stop_reason` before `json.loads`.** `max_tokens` → truncated, unparseable; `refusal` → no schema conformance. Branch first.
- **Size `max_tokens` for worst-case output + headroom.** Truncation is the #1 cause of "constrained decoding gave me invalid JSON."
- **OpenAI strict:** `additionalProperties: false` on every object + every property in `required`. Optional = nullable, never omitted.
- **Constrain shape, not values you can't know.** Enums for closed sets (currency); nullable for genuinely-absent fields; skip `minLength`/`maximum` — they're stripped in strict mode anyway.
- **Keep the schema byte-stable** to stay on the compilation cache (first call with a new schema is slower).
- **One `SCHEMA` dict, three call sites.** The schema is provider-neutral; only the wrapper differs.
- **Free path:** develop against Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`, or the native `format` parameter) before spending a cent on frontier calls.

### The adapter contract

Downstream code must not know or care which provider ran. Three functions, one return shape:

```python
# Each returns the SAME dict shape so callers stay provider-agnostic:
#   {"data": dict | None, "stop_reason": str, "raw": str, "provider": str}
# data is the parsed invoice dict on a clean completion; None on refusal/truncation.

def call_claude(text: str) -> dict: ...
def call_openai(text: str) -> dict: ...
def call_gemini(text: str) -> dict: ...
```

Each function: (1) makes the constrained call, (2) reads `stop_reason`, (3) returns `data=None` with the reason on `refusal`/`max_tokens`, else `json.loads` the payload into `data`. The validator and accuracy report consume `{"data", "stop_reason", ...}` and never branch on provider. Swapping Claude for Gemini — or Ollama for the free path — is then a one-line change at the call site, not a rewrite of your extraction logic.

---

## Connect to the lab

This lecture is the theory behind Week 1's `providers.py` and the Definition-of-Done gate. In the lab you'll write exactly the three-function adapter above (plus the Ollama free path), force JSON via each native mechanism, and run it over the 30-invoice dataset. The DoD — "schema-valid JSON for 30/30 on at least one provider, *validated with `json.loads` + key check, not eyeballed*" — is precisely the *parses* gate; the accuracy report (per-field exact-match) is the *correct* gate. When you build the 5 deliberately-hard cases (missing currency, European dates, line-item math), you're building the inputs that expose the parses-vs-correct gap this lecture warns about.

## Going deeper (optional)

- Anthropic docs: **"Structured outputs"** and **"Tool use / structured outputs"** on `platform.claude.com/docs` (search: `Anthropic structured outputs output_config`).
- OpenAI docs: **"Structured Outputs"** guide on `platform.openai.com/docs` — read the strict-mode requirements section carefully (search: `OpenAI structured outputs strict additionalProperties`).
- Google: **Gemini API "Structured output"** docs on `ai.google.dev` (search: `Gemini responseSchema responseMimeType`).
- **JSON Schema** reference: `json-schema.org` — the `understanding-json-schema` guide is the canonical intro to types, enums, `anyOf`, and `$ref`.
- Ollama: **"Structured outputs"** blog post / docs on `ollama.com` (search: `Ollama structured outputs format`), and its OpenAI-compatibility docs for the `/v1` endpoint.
- Pydantic (for the validator/`messages.parse` path): `docs.pydantic.dev`.
- Background on constrained decoding as grammar-masking: search `constrained decoding grammar finite state machine LLM` and `outlines structured generation` (the `outlines` library is the canonical open-source implementation to read).

## Check yourself

1. Constrained decoding guarantees your response is valid JSON. Why is that *not* enough, and what second gate do you add?
2. Under OpenAI's `strict: True`, what two schema requirements are non-obvious, and how do you represent a field that's allowed to be missing?
3. You start getting `JSONDecodeError` on ~3% of requests from a provider that guarantees valid JSON. Name the two most likely `stop_reason` values and how you'd confirm each.
4. Why does adding `"minimum": 0` and `"maxLength": 40` to your schema often *not* enforce those bounds, and where must you enforce them instead?
5. Give a concrete example of a schema-valid-but-wrong invoice response, and explain why over-tightening the `currency` enum could *cause* wrong values rather than prevent them.
6. What single dict do you keep provider-neutral across `call_claude` / `call_openai` / `call_gemini`, and what does making it neutral buy you downstream?

### Answer key

1. Valid JSON only guarantees the *shape* parses; the *values* can still be wrong (right keys, wrong number). The second gate is per-field value validation against ground truth — "correct" is separate from "parses," and Done requires both.
2. (a) `additionalProperties: false` on every object; (b) every property must appear in `required` — there are no optional properties. A missing-allowed field is modeled as **nullable** (`"type": ["string", "null"]`), never by omitting it from `required`.
3. `max_tokens` (generation truncated mid-object → invalid JSON) and `refusal` (safety decline → content doesn't conform at all). Confirm by reading `resp.stop_reason` / `finish_reason` *before* parsing; for truncation, also check whether raising `max_tokens` makes it disappear.
4. Numeric bounds (`minimum`, `maximum`) and string-length constraints (`minLength`, `maxLength`) are not supported by the providers' constrained decoding; the Python/TS SDKs strip them and validate client-side. Enforce them in your own validator, not by trusting the schema.
5. E.g. `{"total_amount": 1040.34, ...}` when the true total is 1238.00 — it's a valid number, wrong value (the model grabbed the subtotal). Over-tightening `currency` to five codes forces a real CHF invoice onto one of the five, guaranteeing a wrong value; a nullable/open field would let the model avoid the false pick.
6. The return shape — `{"data": dict|None, "stop_reason": str, "raw": str, "provider": str}`. Neutralizing it lets the validator, accuracy report, and downstream pipeline run identically regardless of provider, so swapping Claude ↔ Gemini ↔ Ollama is a one-line call-site change, not a logic rewrite.
