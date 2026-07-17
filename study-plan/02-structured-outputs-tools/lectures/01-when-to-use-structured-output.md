# Lecture 1: When (and When Not) to Force Structured Output

> Almost every engineer new to LLM systems makes the same reflex mistake: they see "the model returns JSON" and immediately reach for the strictest schema-enforcement knob their provider offers, on every call, for everything. This lecture exists to install the opposite reflex — a decision *before* you write any schema. You will learn the one hard line that separates structured output from freeform prose, the counterintuitive way that forcing JSON too early makes your model measurably *dumber*, the two standard mitigations (leading scratchpad field, and two-pass reason-then-extract) with concrete message examples, and the operating principle that governs everything downstream: **LLM proposes, code disposes.** After this lecture you will be able to look at any task and decide, in seconds, whether to force a schema, where to put the model's thinking, and why every byte the model emits is untrusted input to your next stage.

**Prerequisites:** Phase 0 (chat roles, sampling, tokenization) · Phase 1 (prompt anatomy, provider idioms) · comfort reading Python/Pydantic · **Reading time:** ~22 min · **Part of:** Structured Outputs & Tool Calling — Week 1

---

## The core idea (plain language)

Here is the whole lecture in one sentence: **structured output is for when a program consumes the result; freeform prose is for reasoning, explanation, and open-ended generation.**

That is the hard line. Everything else is elaboration.

If the *next reader* of the model's output is a `json.loads()` call, a database insert, a function dispatch, or a UI that renders typed fields — you want structure, and you want it validated. If the next reader is a *human*, or the model *itself* on a later step (thinking, planning, explaining, brainstorming), you want prose, and forcing a schema onto it is actively harmful.

The trap is that "give me JSON" feels like free insurance. It isn't. Constraining the output shape spends some of the model's limited per-token "budget" on getting the *syntax* right — matching braces, quoting keys, emitting the exact enum spelling — instead of on *figuring out the answer*. For an easy task that costs you nothing you'd notice. For a hard reasoning task, it can visibly degrade the answer. So the real skill isn't "how do I force a schema" (that's lectures 02–03, per provider). The skill is **deciding whether to, and arranging the prompt so the model thinks first and formats second.**

Two moves buy you almost all of the benefit:

1. Put a **`reasoning` (a.k.a. `scratchpad`) string field FIRST** in your schema, so the model reasons in prose before it emits the machine fields — all in one call.
2. Or split into **two passes**: one free-text call to think, a second call that only extracts the prior answer into the schema.

And underneath all of it sits the principle you must never violate: **the model's output is a *proposal*, not a *command*.** Your code validates it, and only your code decides what happens next. Never execute or trust a model's output blindly. That single sentence is the seed of the entire Week 2 security lecture.

## How it actually works (mechanism, from first principles)

To understand *why* forcing JSON early hurts, you need one fact about how these models generate: **decoding is left-to-right, one token at a time, and each token is chosen from a probability distribution conditioned on everything emitted so far.** The model does not "plan the JSON and fill it in." It walks forward, token by token, and whatever it has already written becomes part of the context that shapes the next token.

This has two consequences that drive the entire lecture.

**Consequence 1 — order is causal, not cosmetic.** Because generation is left-to-right, tokens produced *earlier* condition tokens produced *later*, never the reverse. If the very first thing the model must emit is `{"category":`, it is being forced to *commit to an answer* before it has generated any tokens that constitute reasoning about that answer. It has no scratch space. Compare that to a schema where the first field is `"reasoning"`: now the model writes a few sentences of analysis first, and those sentences are in-context when it finally emits `"category"`. The category token is now conditioned on the model's own worked-out reasoning. Same model, same input — different *order of emission*, materially different answer quality on hard inputs. This is why "put the reasoning field first" is not a style preference; it is a consequence of the generation mechanism.

**Consequence 2 — structure competes for the same budget.** When a provider enforces a schema via constrained decoding (the mechanics come in lectures 02–03), at each step it *masks* the token distribution so only tokens that keep the output schema-valid are allowed. That guarantees valid syntax. But it also means the model is steered by the *grammar* at the exact moments it would otherwise be steered by its *reasoning*. Even without hard constraint — plain "respond in JSON" — the model is spending attention and tokens on formatting discipline. There is no free lunch: capacity spent tracking "am I inside a string? do I need a comma? is this a valid enum value?" is capacity not spent on "what is the actual answer?"

Let's make this numeric with a small, illustrative model. (These numbers are a **teaching illustration**, not a benchmark.)

Say you have 200 messy invoice strings, and the hard sub-task is classifying `category` (software / hardware / services / other) — the kind of judgment call where reasoning helps. Three setups:

```
Setup A: force schema, category field FIRST (no reasoning)
Setup B: force schema, reasoning field FIRST, then category
Setup C: free-text "explain then state category", human/regex reads it

Illustrative category accuracy:
  A  ~78%   <- model commits before it reasons
  B  ~90%   <- reasoning tokens condition the category token
  C  ~92%   <- fully unconstrained reasoning, best quality, worst to parse
```

Setup A is fast and trivially machine-readable but the *dumbest*. Setup C is the smartest but you're back to parsing prose. Setup B captures most of C's quality *and* gives you a clean typed object. That gap between A and B — roughly the cost of making the model commit before it thinks — is the entire reason this lecture exists.

A diagram of the causal ordering inside a single structured call:

```
LEFT-TO-RIGHT GENERATION  (each box conditions everything to its right)

  BAD  (answer-first schema)
  ┌───────────────┬──────────────────────────────────┐
  │ "category":?? │  ...no prior thinking to lean on  │
  └───────────────┴──────────────────────────────────┘
        ^ forced to commit here

  GOOD (reasoning-first schema)
  ┌────────────────────────────┬──────────────┬──────────────┐
  │ "reasoning":"line 3 says   │ "category":  │ "total_usd": │
  │  'MSDN sub' -> software..."│  "software"  │  1240.00     │
  └────────────────────────────┴──────────────┴──────────────┘
        ^ reasoning tokens now condition ────┘  ┘
          every field to their right
```

### The two mitigations, concretely

**Mitigation (a): leading `reasoning`/`scratchpad` field, single call.**

You keep one API call and one schema. You simply make the *first* property a free-text string the model fills with its thinking. Field order in the schema matters because most providers emit fields in declared order and, more importantly, because *you* control declaration order and the model follows it.

```python
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    reasoning: str = Field(description="Think step by step here FIRST: "
        "where you found each field, how you resolved ambiguity. "
        "Write this before any other field.")
    vendor_name: str = Field(description="Legal name of the issuing vendor")
    category: str = Field(description="software | hardware | services | other")
    total_usd: float = Field(description="Grand total in USD")
```

The message you send is a single request; the model returns one object whose `reasoning` you typically log and discard, keeping the typed fields.

**Mitigation (b): two-pass reason-then-extract.**

Call one is unconstrained — the model thinks in full prose. Call two feeds that prose back and asks *only* for extraction into the schema. The second call is an "easy" task (copy facts already stated into fields), so forcing a schema there costs almost nothing.

```python
# Pass 1: free text, no schema. Model reasons at full quality.
pass1 = client.chat(messages=[
    {"role":"user","content":
      f"Analyze this invoice. Reason out loud about vendor, "
      f"category, and total. Do NOT format as JSON.\n\n{raw_text}"}
])
analysis = pass1.text  # prose

# Pass 2: extraction only. Schema forced here; task is now trivial.
invoice = client.parse(   # provider schema-enforcement (lectures 02-03)
    response_model=Invoice,
    messages=[{"role":"user","content":
      f"Extract these fields from the analysis below.\n\n{analysis}"}]
)
```

## Worked example

Let's run one messy input through both mitigations and watch the numbers move. The input is deliberately ugly:

```
INV #  RE-2231     11/03/26
Redwood Analytics LLC  -- annual platform seat, 4 users
  seat @ 310/ea .......... 1240
  onboarding (one-time) ... 500
  *note: onboarding waived per contract, do not bill*
Amount due: 1240.00
```

The traps: the date is ambiguous (US `MM/DD` vs EU `DD/MM`), the onboarding line *looks* billable but a note waives it, and a naive sum (1240 + 500 = 1740) contradicts the stated total (1240).

**Answer-first schema (Setup A).** The model must emit `total_usd` before it has reasoned about the waiver. Common failure: it sums the visible line items and emits `1740.00`, contradicting "Amount due: 1240.00". It never reconciled them because it committed to a number before reading the note carefully. The object is *schema-valid* and *wrong* — the worst combination, because it passes your JSON checks silently.

**Reasoning-first schema (Setup B).** The model emits:

```json
{
  "reasoning": "Two lines: seat 4x310=1240, onboarding 500. But a note says onboarding is waived and must not be billed. Stated 'Amount due' is 1240.00, which matches seats only. So billable total = 1240.00, one billable line item. Date 11/03/26 is ambiguous; vendor is US LLC so assume MM/DD -> 2026-11-03.",
  "vendor_name": "Redwood Analytics LLC",
  "category": "software",
  "total_usd": 1240.00
}
```

Because the waiver reasoning is now in-context, `total_usd` comes out consistent with the stated amount. On a batch of 20 such inputs, moving the reasoning field to the front might take you from ~14/20 fully-correct extractions to ~18/20 — again, *illustrative*, but the direction and rough magnitude are what you'll see in practice on messy real data. That is a large, cheap win from reordering one field.

**Two-pass (Setup b).** Pass 1 produces a paragraph reconciling the waiver and the total at full reasoning quality. Pass 2 extracts it. You'd expect quality on par with, or slightly above, the single-call scratchpad — at the cost of a second round-trip.

**And the disposal step — the part that matters most.** Even Setup B's beautiful object is a *proposal*. Your code still runs a validator:

```python
computed = round(sum(li.qty * li.unit_price_usd for li in inv.line_items), 2)
if inv.line_items and abs(computed - inv.total_usd) > 0.01:
    raise ValueError(f"line items sum to {computed}, total says {inv.total_usd}")
```

If the model had hallucinated `total_usd: 9999`, the schema wouldn't catch it — it's a valid float. *Your business-rule validator* catches it. The model proposed; the code disposed.

## How it shows up in production

**Silent-but-wrong is the expensive failure.** The scariest bug in structured extraction isn't a crash — a crash you'll see. It's a schema-valid object with a wrong value, produced by an answer-first schema that made the model commit before it thought. It sails through `model_validate()`, gets written to your ledger, and surfaces three weeks later in a reconciliation report. Reasoning-first schemas and business-rule validators are how you convert "silently wrong" into "loudly rejected."

**Latency and cost — how to choose between the two mitigations.** This is the practical fork:

- **Single-call scratchpad (a):** one round-trip; you pay for the reasoning tokens as *output* tokens on the one call. Lower latency, lower cost, slightly less reasoning headroom than a fully free pass. **Default choice.**
- **Two-pass (b):** two round-trips, so roughly *2× the latency* and you pay input tokens twice (the reasoning text is re-sent as input to pass 2). But pass 1 reasons completely unconstrained, and pass 2 is a cheap easy extraction you can even route to a *smaller/cheaper model*. Use it when reasoning quality is paramount, when the reasoning is long enough that you don't want it bloating the structured call, or when you want to reason with a big model and extract with a cheap one.

Rule of thumb: **reach for the single-call scratchpad first; escalate to two-pass only when a measured accuracy gap justifies the extra round-trip, or when you want to split model tiers across the two stages.** (Approximate guidance — always measure on your own data.)

**Token accounting bites you.** The `reasoning` field is output tokens you're paying for and usually throwing away. On high-volume pipelines that's real money — but skimping on it to save tokens is exactly the false economy that produces silent-but-wrong. Log a *bounded* reasoning field (cap its length via prompt) rather than deleting it.

**Debuggability.** When an extraction goes wrong in prod, the logged `reasoning` string is your single most valuable artifact — it tells you *why* the model chose what it chose. Teams that strip it out to save bytes are the teams that can't explain their own failures. Keep it, log it, sample it.

## Common misconceptions & failure modes

- **"Forcing JSON is free insurance."** No — it can lower answer quality on reasoning-heavy tasks by making the model format instead of think. It's insurance you pay for in accuracy unless you give the model room to reason first.
- **"JSON mode means I get my schema."** Different guarantee entirely (that's the three-tiers lecture). JSON mode ⇒ *parseable* JSON, not *your* fields. Orthogonal to today's topic but frequently conflated.
- **"Field order is just cosmetic."** Generation is left-to-right; earlier fields condition later ones. Order is *causal*. Answer-first schemas make the model commit before reasoning.
- **"Validation is optional if the schema is strict."** Schema-valid ≠ correct. A hallucinated total is a valid float. Business-rule validators live in *your code*, not the schema. This is the whole point of "code disposes."
- **"I can just execute what the model returns."** This is the security hole the whole field is built to prevent. Model output is untrusted input. You'll spend all of Week 2 hardening against exactly this reflex — a tool argument the model emits is, effectively, user input laundered through a language model.
- **"Two-pass is always better because reasoning is unconstrained."** It's often better on quality but costs a round-trip and double input tokens; on easy tasks the single-call scratchpad matches it for half the latency.
- **"Set temperature to 0 and I don't need any of this."** Temperature reduces variance, not the answer-first-commitment problem. A deterministic wrong answer is still wrong.

## Rules of thumb / cheat sheet

- **The line:** program consumes it → structure. Human or the model itself consumes it → prose.
- **Never force a schema on a pure reasoning/brainstorming/explanation step.** Let it be prose; structure a *later* step.
- **If you force a schema on a task that needs thinking, put a `reasoning`/`scratchpad` string field FIRST.** Always. It's the cheapest quality win you have.
- **Default to single-call scratchpad; escalate to two-pass** when a measured accuracy gap justifies the extra call, when reasoning is long, or when you want a big-model reason + cheap-model extract split.
- **Log the reasoning field, bounded.** It's your debugging goldmine; don't delete it to save tokens.
- **Every model output is UNTRUSTED.** Validate with schema *and* business rules; never execute/trust blindly. LLM proposes, code disposes.
- **Schema-valid ≠ correct.** Add validators for the invariants a schema can't express (sums, ranges, cross-field consistency, referential checks).
- **Measure before optimizing.** All accuracy numbers here are illustrative; run your own before/after on real inputs.

## Connect to the lab

This lecture is the *decision* that precedes all the Week 1 lab code. When you build `structio/schemas/invoice.py`, the `reasoning: str` field declared **first** in the `Invoice` model is mitigation (a) in action — that ordering is why it's there, not decoration. The `@model_validator` that checks line items sum to `total_usd` is "code disposes" made concrete: the model proposes numbers, your validator refuses inconsistent ones. As you write your 20 messy invoice strings in `data/invoices.jsonl`, deliberately include a waiver-style trap like the worked example above, then run the extractor once with the reasoning field first and once with it moved last — measure the schema-valid-but-wrong rate difference yourself. That before/after *is* the lecture.

## Going deeper (optional)

Real, named resources — verify current URLs yourself; I give root domains and search queries rather than deep links.

- **OpenAI docs** (`platform.openai.com/docs`) — search "OpenAI Structured Outputs guide" and "OpenAI reasoning best practices." Note the guidance to let models reason before committing to structure.
- **Anthropic docs** (`docs.anthropic.com`) — search "Anthropic tool use," "Claude increase output consistency," and their prompt-engineering pages on chain-of-thought / letting Claude think before answering (`<thinking>` tags are the prose-first idea in another dress).
- **Chain-of-thought prompting** — search "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022). The canonical evidence that reasoning-first improves hard-task answers; the leading-scratchpad field is CoT smuggled into a schema.
- **Instructor library** (`python.useinstructor.com`) — search "instructor chain of thought field." Its docs explicitly recommend a leading reasoning field on extraction models — the pattern this lecture formalizes.
- **OWASP Top 10 for LLM Applications** — search "OWASP Top 10 LLM," especially "Insecure Output Handling" and "Excessive Agency." This is where "LLM proposes, code disposes" becomes a named threat model; it's the bridge into Week 2.
- **Guidance / constrained-decoding intuition** — search "how does constrained decoding work logit masking." Optional preview of *how* schemas get enforced (the mechanics you're deliberately deferring to lectures 02–03).

## Check yourself

1. State the one-sentence rule that decides whether a task should produce structured output or freeform prose. Give one example of each where the *opposite* choice would be a mistake.
2. Generation is left-to-right. Use that single fact to explain why a schema with `category` as its first field can produce worse categories than a schema with `reasoning` first.
3. You must extract 6 fields from long, ambiguous legal text where getting the reasoning right is critical, and latency budget is generous. Which mitigation do you pick and why? Now the same task must return in under 400 ms per call — does your answer change?
4. An extraction returns a perfectly schema-valid object, but `total_usd` is wrong. Which of your two safety layers (schema validation vs. business-rule validation) should have caught it, and where does that layer live?
5. "The model called the `delete_account` tool, so I deleted the account." Identify every principle from this lecture that sentence violates.
6. Why is deleting the `reasoning` field to save output tokens often a false economy in production?

### Answer key

1. **"Structured output is for when a program consumes the result; freeform prose is for reasoning/explanation/open-ended generation."** Opposite-choice mistakes: (i) forcing a JSON schema on a "brainstorm 10 product angles and argue for each" task strangles the reasoning — it should be prose; (ii) returning free prose from a "give me the parsed invoice for the database" endpoint forces brittle downstream parsing — it should be structured.
2. Each token is conditioned on everything emitted before it. If `category` is first, the model must commit to it having generated *no reasoning tokens* — it guesses, then rationalizes. If `reasoning` is first, the category token is conditioned on the model's own worked-out analysis, so it's a better-informed choice. Order is causal, not cosmetic.
3. Generous latency + reasoning-critical → **two-pass reason-then-extract**: pass 1 reasons fully unconstrained (highest quality), pass 2 does trivial extraction (optionally on a cheaper model). Under a 400 ms budget the second round-trip is likely too expensive → switch to the **single-call scratchpad** (leading `reasoning` field), accepting slightly less reasoning headroom for one round-trip.
4. **Business-rule validation** should catch it — a wrong-but-plausible float is schema-valid, so schema validation can't. That layer lives in **your code** (e.g., a Pydantic `@model_validator` or an explicit check), not in the schema and not in the model. Code disposes.
5. It (a) treats untrusted model output as a trusted command; (b) executes rather than validates a *proposal*; (c) violates "LLM proposes, code disposes"; (d) skips the human-confirmation gate a destructive/irreversible action demands (the Week 2 excessive-agency failure). The tool *request* should have been validated and gated, never auto-executed.
6. The `reasoning` field is your primary debugging artifact — it records *why* the model produced each value. Deleting it saves a few tokens but blinds you when an extraction goes wrong in prod, and skimping on reasoning space is exactly what produces silently-wrong-but-schema-valid outputs, the most expensive failure mode.
