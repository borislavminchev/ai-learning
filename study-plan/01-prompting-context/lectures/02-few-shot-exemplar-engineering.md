# Lecture 2: Few-Shot Exemplar Engineering

> "Just add a few examples" is the most reflexive move in prompt engineering, and it is wrong about half the time. Few-shot prompting is not a default you sprinkle on for good measure — it is an engineering decision with a measurable accuracy delta *and* a measurable token cost, and on modern reasoning models it frequently makes things **worse**. This lecture teaches you to treat exemplars as a precision instrument: what they actually teach the model (format and edge-case anchoring, not knowledge), the byte-level discipline that keeps them from silently degrading quality *and* busting your prompt cache, where they sit in the prompt, and the exact 2×2 experiment that forces every few-shot decision to justify itself against its token overhead. After this you'll be able to build 3–5 exemplars for a real extraction task, place them in the cacheable prefix, and run the zero-shot-vs-few-shot crossover measurement that tells you whether they earn their keep.

**Prerequisites:** Lecture 1 (Prompt anatomy — sections, ordering, the recency/primacy dead zone), Phase 0 Lecture 13 (base / instruct / reasoning model types), Phase 0 Lecture 16 (structured output tiers) · **Reading time:** ~26 min · **Part of:** Phase 1 Week 1

---

## The core idea (plain language)

A **few-shot** prompt shows the model a handful of solved input→output pairs ("shots," "exemplars") before the real input. A **zero-shot** prompt just describes the task and hands over the input. The industry folklore is that more examples always help. That folklore comes from the pre-instruct era (GPT-3, 2020), when base models genuinely needed examples to figure out what task you even wanted. Modern instruction-tuned and reasoning models already know how to follow instructions — so the calculus flipped.

Here is the mental model that will keep you out of trouble: **few-shot teaches *format* and *anchors edge cases*; it does not inject *knowledge*.** Examples are how you show the model the exact JSON shape you want, the exact way to handle a missing currency, the exact date format to normalize to. They are a terrible and expensive way to teach the model *facts* — the model either already knows the fact or it doesn't, and three examples won't fix that; retrieval (Phase 4) will. If you catch yourself adding examples to "teach the model about our domain," stop: that's a knowledge problem wearing a few-shot costume.

Because exemplars live *inside every single request*, they cost tokens on every call, forever. Five good invoice exemplars can easily be 800–1500 tokens. If they lift accuracy 8 points, that might be the best money you spend. If they lift it 1 point — or *lower* it, which happens on reasoning models — you're paying a permanent tax for nothing. So the whole discipline reduces to one loop: **build clean exemplars, measure zero-shot vs few-shot on a held-out set, and keep them only if the accuracy delta beats the token cost.** Never "just add examples."

---

## How it actually works (mechanism, from first principles)

### Why examples work at all: in-context pattern completion

An LLM is a next-token predictor conditioned on everything in the context (Phase 0). When you place three input→output pairs in the prompt, you're not "training" anything — the weights are frozen. You're constructing a context in which the highest-probability continuation is "another pair of the same shape." The model has seen billions of list-like, table-like, Q&A-like patterns in pretraining; your three shots activate that "continue the established pattern" behavior. This is why the *format* of your examples is so load-bearing: the model copies the pattern it sees, including your mistakes.

Concretely, if your three exemplars all emit:

```json
{"vendor": "...", "invoice_number": "...", "date": "...", "total_amount": ..., "currency": "..."}
```

then the strongest signal in the context is "produce exactly this key order, these five keys, this JSON style." The model will tend to reproduce it byte-for-byte. That's the superpower — and, as we'll see, the trap.

### Why format must be byte-identical across all shots

There are two independent reasons, and the second one is the one engineers forget until it costs them money.

**Reason 1 — quality.** If shot 1 uses `"total_amount": 1250.00`, shot 2 uses `"total_amount": "1250"` (a string), and shot 3 uses `"amount": 1250` (different key), you've handed the model three conflicting patterns. It now has to *guess* which one you meant, and its guess is influenced by recency (the last example weighs most). You've converted a crisp instruction into noise. Inconsistent exemplars measurably degrade output consistency and can be worse than zero-shot, because zero-shot at least gives one clean instruction instead of three contradictory demonstrations.

**Reason 2 — prompt caching (the expensive one).** Prompt caching is a **prefix match at the byte level** (you'll go deep in Week 3). The provider hashes the longest identical prefix of your request and reuses the computed attention state, charging you ~10% of the normal input price for cached tokens. Exemplars live in the static prefix — that's exactly *why* you put them there. But a cache only hits if the prefix is **byte-identical** to a previous request. A single stray trailing space, a reordered key from `json.dumps` without `sort_keys=True`, a `\r\n` vs `\n` line ending, a today's-date interpolated into an example — any of these changes the bytes, and the cache misses. You pay full price on every request and never notice, because *nothing errors*. The failure is a silent 10× cost regression on the prefix.

So "byte-identical" is not a style preference. It is: **same keys, same key order, same JSON formatting (spacing, quote style, number vs string), same whitespace, same line endings, no trailing whitespace, across every shot and across every request.** The cleanest way to guarantee it is to *generate* your exemplar block deterministically from data (e.g., `json.dumps(obj, sort_keys=True, separators=(",", ": "))`) rather than hand-typing JSON that drifts.

```
Byte-identical exemplars                 Drifted exemplars
─────────────────────────                ─────────────────────────
shot1  {"a":1,"b":2}                      shot1  {"a":1, "b":2}
shot2  {"a":3,"b":4}                      shot2  {"b":4,"a":3}      <- key order
shot3  {"a":5,"b":6}                      shot3  {"a":5,"b":6 }     <- trailing space
   |                                         |
   v                                         v
strong single pattern +                   conflicting patterns +
cache HIT (10% price)                     cache MISS (100% price, silent)
```

### Cover the hard path, not the happy path

The instinct is to pick three *clean, easy* examples because they're easy to write. This wastes the mechanism. The model already handles the easy path from the instruction alone — a well-formed invoice with a clear total needs no demonstration. Exemplars earn their tokens by **anchoring the edge cases**: the European `31.12.2025` date that must not be read as month 31; the invoice with no currency symbol where you want `currency: null` rather than a guessed `"USD"`; the invoice where line items sum to the total and the "total" label is missing so the model must add them. Each of these is a *decision* the instruction can describe but an example can nail. Put your hardest, most ambiguous cases in the exemplars, because that's where the model's behavior is uncertain and a demonstration collapses the uncertainty.

### Shuffle order and balance labels: recency bias is real

For any task with categorical outputs (classification, yes/no, a `currency` enum), the *order* and *balance* of your exemplars leak into predictions. Two documented biases:

- **Recency label bias.** If your last two exemplars both have `currency: "EUR"`, the model's prior shifts toward `EUR` for the live input — the most recent examples weigh most. 
- **Majority label bias.** If 4 of 5 exemplars are `USD`, the model over-predicts `USD`.

The fixes are cheap: **balance labels** across your exemplar set (roughly equal representation of each class you care about) and **shuffle the order** so no class clusters at the end. For extraction tasks the effect is milder than for pure classification, but the `currency` field in the invoice task is exactly a small enum, so it applies. A concrete rule: if you have a 4-way label and 4 exemplars, use one of each and don't sort them by label.

### The counterintuitive result: reasoning models often do WORSE with heavy few-shot

This is the finding you must internalize, because it contradicts the folklore hardest. A **reasoning model** (OpenAI o-series, Claude with extended/adaptive thinking, Gemini thinking) generates an internal chain of reasoning before answering. It has effectively learned to *construct its own worked examples* on the fly. When you stuff five hand-written exemplars in front of it, several things go wrong at once:

1. **You constrain its reasoning.** Your exemplars implicitly demonstrate a *way* of solving the problem. A reasoning model may have a better internal approach; your shots pull it toward yours, which can be worse. This is the same principle as "don't hand-write chain-of-thought for a reasoning model" (Week 2) — you're doing the model's job for it, badly.
2. **You spend the budget twice.** The exemplars cost input tokens, and then the model *also* reasons over them, spending output/thinking tokens processing examples it didn't need.
3. **You can trigger overfitting to the demonstration.** If an exemplar has an idiosyncrasy, the reasoning model may faithfully reproduce it on inputs where it doesn't apply.

The engineering consequence: **on reasoning models, start at zero-shot** and add exemplars only if a measurement proves they help. You will frequently find a *crossover* — zero-shot beats 5-shot on accuracy *and* costs fewer tokens. That's a double win you'd never discover if you followed the "always add examples" folklore. On non-reasoning instruct models the balance tips back toward few-shot helping, especially for format adherence — which is exactly why you measure both model types, never assume.

---

## Worked example: building the invoice exemplars and running the crossover

The task (from the Week 1 lab): extract `{vendor, invoice_number, date, total_amount, currency}` from a messy invoice text blob.

### Step 1 — choose 3–5 exemplars that cover the hard path

Don't grab the three cleanest invoices. Deliberately pick decision-anchoring cases and balance the `currency` label:

1. **Clean baseline** (USD, all fields present) — establishes the shape.
2. **European date + EUR** — `Rechnung Nr. 2025-0042 … Datum: 31.12.2025 … Summe: 1.234,56 €` — anchors `date` normalization to ISO `2025-12-31` and comma-decimal parsing.
3. **Missing currency** (GBP context but no symbol) — anchors the "emit `null`, don't guess" decision.
4. **Total absent, line items only** — three line items that must be summed; anchors arithmetic + "derive the total."
5. **Vendor buried in a footer, invoice # with a prefix** — anchors field localization.

That's 5 shots, labels balanced (USD / EUR / null / one-more / mixed), ordered so no currency clusters at the end.

### Step 2 — make the exemplar block byte-identical

Render each output deterministically. Every shot follows the identical template — same key order, same spacing, same `null` style, no trailing whitespace:

```
Input:
Rechnung Nr. 2025-0042
Datum: 31.12.2025
Summe: 1.234,56 €
Output:
{"vendor": "Müller GmbH", "invoice_number": "2025-0042", "date": "2025-12-31", "total_amount": 1234.56, "currency": "EUR"}
```

Note the ISO date and the parsed decimal — the exemplar *shows* the normalization you want, which no amount of instruction prose conveys as crisply.

### Step 3 — place them in the prompt (position matters for attention AND caching)

The ordering, from Lecture 1, is: **instructions → exemplars → live input**. Concretely:

```
[SYSTEM / static instructions]        <- cacheable prefix
[few-shot exemplars, byte-identical]  <- cacheable prefix, cache_control breakpoint AFTER here
[the live invoice document]           <- volatile, AFTER the breakpoint
```

Exemplars go **after** the instructions (they illustrate the instructions) and **before** the live input (the model completes the established pattern with the new input as the final "Input:"). Critically, they sit **inside the cacheable prefix** — they're static across requests, so you want them cached. The live document is the only thing that changes per request, so it goes after the cache breakpoint. Get this backwards — exemplars after the live input — and you both confuse the "continue the pattern" mechanism and push volatile content into the cached region, busting the cache.

### Step 4 — run the 2×2 and read the crossover

Here is the exact experimental frame the lab demands. Two prompt designs (v1 blob vs v2 structured, from Lecture 1) crossed with zero-shot vs 5-shot, **each cell reporting accuracy AND tokens** so few-shot is always judged against its overhead:

```
                     zero-shot                 5-shot
              ┌───────────────────────┬───────────────────────┐
 v1 (blob)    │ acc: 71%              │ acc: 79%              │
              │ in-tokens: 320/req    │ in-tokens: 1,510/req  │
              ├───────────────────────┼───────────────────────┤
 v2 (struct)  │ acc: 84%              │ acc: 88%              │
              │ in-tokens: 380/req    │ in-tokens: 1,590/req  │
              └───────────────────────┴───────────────────────┘
(illustrative numbers on a non-reasoning instruct model — YOURS WILL DIFFER;
 the point is the SHAPE of the decision, not these exact values)
```

Read it as an engineer. On this *non-reasoning* model, v2+5-shot wins accuracy (88%) but costs ~4× the input tokens of v2+zero-shot (1,590 vs 380). Is +4 points worth 4× tokens on every request forever? Depends on volume and the cost of an error — at high volume with cheap errors, v2+zero-shot (84% at 380 tokens) may be the better ship. Now rerun the *same 2×2 on a reasoning model* and you'll often see the crossover: zero-shot matching or beating 5-shot while costing far less. **The 2×2 is the deliverable** — it converts "should we use examples?" from an opinion into four numbers you can defend.

A tiny cost sanity-check makes the tax concrete. Suppose input tokens cost $3 / 1M and you serve 100k requests/day. The 5-shot prefix adds ~1,200 tokens/request → 120M extra input tokens/day → **~$360/day (~$130k/year)** in exemplar tokens alone. If prompt caching hits on that prefix at ~10% price, it's ~$36/day instead — which is *precisely* why byte-identical exemplars (that actually cache) matter, and why a stray trailing space that busts the cache is a five-figure mistake.

---

## How it shows up in production

- **The silent cache bust from a "harmless" exemplar edit.** Someone reformats the examples for readability — adds spaces after colons, re-sorts keys — in a PR that looks cosmetic. The prefix bytes change, cache-hit rate on the exemplar block drops to zero, and the input bill quietly triples. No test fails, no error logs. You find it weeks later staring at `cache_read_input_tokens: 0`. Lesson: exemplar bytes are load-bearing; treat the exemplar block like a compiled artifact, ideally generated deterministically and covered by a byte-snapshot test.
- **Few-shot "helped" in the playground, tanked in prod.** You eyeballed three examples on two invoices and it looked great. On the 30-example held-out set, few-shot was +1 point over zero-shot but +1,200 tokens/request. The playground can't show you the token tax or the crossover; only the 2×2 on held-out data can.
- **Reasoning-model regression after "improving" the prompt.** A team adds five carefully crafted exemplars to a Claude-thinking / o-series pipeline and accuracy *drops* while latency and cost rise. Cause: the exemplars constrained the model's own reasoning and doubled the token spend. Fix: delete the exemplars, go zero-shot, re-measure. The "improvement" was a regression.
- **Recency bias skews an enum field.** The `currency` predictions skew toward whatever the last exemplar showed. Reordering/balancing the exemplars shifts the distribution — proof the *order* was leaking into outputs, not just the content.
- **Exemplar staleness and budget crowding.** A shot hard-codes a 2024 date format or old vendor convention; months later inputs drift and the stale demonstration actively mis-teaches — exemplars are code and rot like code. And five fat exemplars eat ~1,500 tokens of a context you also need for the document, retrieved chunks, and history; on a tight budget, examples *compete* with the actual input, and losing the input to fit them is a bad trade.

---

## Common misconceptions & failure modes

- **"More examples always help."** False since the instruct era. On reasoning models more examples frequently *hurt*, and everywhere they cost tokens on every request. Measure; don't assume.
- **"Few-shot teaches the model our domain knowledge."** No. It teaches *format* and *anchors edge-case decisions*. If the model lacks a fact, examples won't supply it — that's a retrieval problem (Phase 4).
- **"Formatting differences between examples don't matter, the model is smart."** They matter twice: inconsistent format degrades output quality (conflicting patterns) and *silently busts prompt caching* (byte-level prefix match). A trailing space is a real bug.
- **"Pick the clearest, easiest examples."** Backwards. Easy cases are already handled by the instruction; spend your exemplar tokens on the hard, ambiguous, edge cases where the model's behavior is uncertain.
- **"Order doesn't matter."** For any categorical output it does — recency and majority label bias skew predictions. Shuffle and balance.
- **"Zero-shot is the lazy option."** On modern models zero-shot is often the *correct* option — cheaper and sometimes more accurate. Few-shot is the choice you must justify with numbers, not the other way around.
- **"Put examples anywhere in the prompt."** Placement is load-bearing: after instructions, before live input, inside the cacheable prefix. Examples after the live input confuse the pattern-completion and hurt caching.
- **"Schema-constrained output makes exemplars redundant."** Structured outputs (Phase 0 L16) guarantee the *shape*; exemplars still teach *content decisions* — how to normalize a date, when to emit `null`. They're complementary, not substitutes.

---

## Rules of thumb / cheat sheet

- **Default to zero-shot.** Add exemplars only when a held-out measurement shows they beat their token cost. Few-shot is a decision, not a default.
- **On reasoning models, expect the crossover.** Start zero-shot; suspect that 5-shot is worse *and* pricier. Verify.
- **3–5 exemplars is the usual sweet spot.** More rarely helps and always costs; past ~5 you're usually buying token tax, not accuracy.
- **Byte-identical across all shots:** same keys, same key order, same JSON style/spacing, same whitespace, no trailing spaces, same line endings. Generate deterministically (`json.dumps(..., sort_keys=True)`) rather than hand-typing.
- **Cover the hard path.** Put your edge and ambiguous cases in the exemplars; skip the easy ones the instruction already handles.
- **Balance labels and shuffle order** for any categorical/enum field to kill recency and majority bias.
- **Placement:** instructions → exemplars → live input. Exemplars live *inside* the cacheable prefix; the volatile input goes after the cache breakpoint.
- **Report accuracy AND tokens per cell.** Never evaluate few-shot without its token overhead in the same table.
- **Treat exemplars as versioned code:** review them, snapshot their bytes, watch cache-hit rate, and re-measure when models or inputs drift.

---

## Connect to the lab

This is Week 1 Lab step 6 made rigorous. You'll add your 3–5 byte-identical invoice exemplars (covering the 5 deliberately hard cases from step 2) to the prompt after the instructions and before the live document, then run the full **2×2 grid — v1_blob / v2_structured × zero-shot / 5-shot** over your 30-example held-out set, recording **field accuracy AND token count for each of the four cells** in the accuracy report. Definition of Done requires exactly this grid with per-cell token counts. Do the run on a non-reasoning model first, then repeat on a reasoning model and watch for the crossover where 5-shot loses to zero-shot on accuracy *and* cost — that observation is the answer to Week 1 Self-check question 4 ("name one case where few-shot hurt, and why").

---

## Going deeper (optional)

- **Anthropic — "Prompt engineering overview" and "Use examples (multishot prompting)"** in the official docs (docs.anthropic.com). Anthropic's own guidance on how many shots and formatting consistency; pair with the `claude-api` skill for current model IDs and caching params.
- **OpenAI — "Prompt engineering" guide and "Reasoning best practices"** (platform.openai.com/docs). The reasoning-best-practices page is the canonical source for "reasoning models want less hand-holding" — the mechanism behind the few-shot crossover.
- **Anthropic — "Prompt caching" doc** (docs.anthropic.com) — the byte-level prefix-match mechanism that makes exemplar formatting a caching concern; you'll study it deeply in Week 3.
- **"Calibrate Before Use: Improving Few-Shot Performance of Language Models" (Zhao et al., 2021)** — the canonical study of majority-label and recency bias in few-shot prompting. Search the title; read the abstract and figures, skip the proofs.
- **"Rethinking the Role of Demonstrations" (Min et al., 2022)** — evidence that few-shot exemplars work largely by teaching *format/label space* rather than input→output mappings — direct support for "format-teaching, not knowledge-injection." Search the title.
- Search queries: "few-shot prompting reasoning models worse", "prompt caching prefix byte match Anthropic", "recency bias few-shot label", "multishot prompting formatting consistency", "zero-shot vs few-shot instruction-tuned models".

---

## Check yourself

1. State the mental model for what few-shot exemplars *do* and *do not* teach, and give one task where reaching for few-shot is the wrong tool.
2. Give the two independent reasons every exemplar must be byte-identical in format, and explain which one fails *silently* and how you'd detect it.
3. You have a reasoning model and want the best accuracy-per-dollar. What do you try first, what regression do you specifically watch for, and why does it happen?
4. Why should exemplars cover the hard/edge cases rather than the cleanest examples? Connect your answer to the pattern-completion mechanism.
5. Lay out the 2×2 experimental frame for evaluating few-shot on the invoice task, including exactly what each cell must report and why the token count is non-negotiable.
6. Where do exemplars sit relative to the instructions, the live input, and the cache breakpoint — and what breaks if you put them after the live input?

### Answer key

1. Few-shot teaches **format** (exact output shape/keys/style) and **anchors edge-case decisions** (how to handle a European date, a missing currency); it does **not** inject knowledge/facts. Wrong tool: trying to teach the model domain facts it doesn't know — that's a retrieval problem, and three examples won't supply the missing fact.
2. (a) **Quality:** inconsistent formats give the model conflicting patterns to copy, degrading output — sometimes below zero-shot. (b) **Caching:** prompt caching is a byte-level prefix match, so any format drift (trailing space, reordered keys, changed spacing) busts the cache and you pay full price. The caching failure is **silent** (no error) — detect it by watching `cache_read_input_tokens`; a persistent zero on identical-prefix requests means the prefix bytes are changing. Fix: generate the exemplar block deterministically and snapshot its bytes.
3. Try **zero-shot first**. Watch for the regression where adding 5 exemplars *lowers* accuracy while raising latency and token cost (the crossover). It happens because a reasoning model constructs its own reasoning/worked examples; hand-written exemplars constrain that reasoning toward your (possibly worse) approach and make it spend tokens processing examples it didn't need.
4. The model completes the *pattern* it sees in context, so exemplars have leverage exactly where behavior is uncertain — the hard/ambiguous cases. Easy cases are already handled by the instruction alone, so demonstrating them just spends tokens without changing behavior. Edge cases are where a demonstration collapses the model's uncertainty into the decision you want.
5. Two prompt designs — **v1_blob** vs **v2_structured** — crossed with **zero-shot** vs **5-shot** = four cells. Each cell must report **field accuracy AND input tokens per request** on the held-out set. The token count is non-negotiable because exemplars cost tokens on *every* request forever, so accuracy gains must be judged against that permanent overhead; a +1-point gain at +4× tokens is usually not worth shipping.
6. Order is **instructions → exemplars → live input**, with the exemplars **inside the cacheable prefix** and the cache breakpoint placed *after* them so only the volatile live input sits outside the cache. If exemplars go *after* the live input, you (a) break the "continue the established pattern" mechanism (the model no longer completes with the new input as the final pair) and (b) push volatile content into the cached region — or the static exemplars out of it — wrecking cache-hit rate.
