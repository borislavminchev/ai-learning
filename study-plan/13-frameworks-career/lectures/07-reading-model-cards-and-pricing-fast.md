# Lecture 7: Reading Model Cards and Pricing Pages Fast

> Every few weeks a provider ships a new model, and every few weeks a teammate asks "should we switch?" The people who answer in thirty seconds aren't smarter — they've built a repeatable extraction habit. This lecture teaches you to reduce any model page to a single decision line (context / output / input-$ / output-$), to see the four places pricing pages quietly overcharge you (caching, batch, cached-input, and reasoning tokens), and to automate the whole thing with `litellm.get_model_info()` so you generate cheat-sheets instead of squinting at HTML. After this you'll never again paste a per-token price into a spreadsheet and get the decimal point wrong.

**Prerequisites:** basic token/tokenizer intuition (Phase 1), you can call at least one provider SDK, comfort with Python dicts and arithmetic · **Reading time:** ~22 min · **Part of:** Frameworks, Ecosystem, Team Practice & Career — Week 2

## The core idea (plain language)

A model card and its pricing page together answer exactly five capability questions and five cost questions. Everything else on the page is marketing. Your job is to strip the page down to a **one-line summary** you can compare against every other model at a glance:

```
gpt-4o-mini: ctx=128k out=16k in=$0.15/M out=$0.60/M
```

That's the whole skill: know the ten fields, know where each hides, and know the arithmetic that turns "advertised" numbers into "what my bill will actually be." The reason this matters is that pricing pages are optimized to look cheap. The headline number is almost always the *input* price — the cheap one — while the *output* price (3–5× higher) and the *reasoning-token* billing (which can 10× your effective output cost) are a scroll further down or a footnote away.

The capability half of the card tells you whether the model *can* do your task; the pricing half tells you whether you can *afford* it at your volume. You need both in one line because model selection is a two-axis decision and you'll be making it under time pressure, in a Slack thread, while three teammates wait.

## How it actually works (mechanism, from first principles)

### The extraction checklist — capability side

From any model card, pull exactly these five:

1. **Context window (max input tokens).** The total the model can *attend to* in one request — system prompt + all messages + tool schemas + retrieved docs. This is a hard wall: exceed it and you get a 400, not a truncation.
2. **Max output tokens.** A *separate, smaller* cap on what the model generates in one response. A model with a 1M context window might only emit 128k tokens of output. Missing this is the classic "why did my long generation stop at 4096 tokens?" bug — the model hit `max_tokens`, not the context limit.
3. **Supported modalities (in / out).** Text-in is universal. Vision-in (images), audio-in, and audio/image-*out* are all separate capabilities. "Multimodal" on the headline usually means input only. Check both directions.
4. **Knowledge cutoff.** The date after which the model has no training knowledge. Determines whether you need retrieval or web search for current facts. A model with a 2024 cutoff answering questions about 2026 events will confabulate confidently.
5. **Deprecation / sunset date.** When the model ID stops working (returns 404). If you're building something to run for years, a model retiring in six months is a liability. Providers publish these; treat a missing sunset date as "active, but check the changelog."

### The extraction checklist — pricing side

This is where engineers get surprised, so go slower. Five fields:

1. **Input price per token** — cost of tokens you send.
2. **Output price per token** — cost of tokens the model generates. Almost always **3–5× the input price**. This asymmetry is the single most important number on the page, because output is what you're least able to predict.
3. **Cached-input price** — a discounted rate for input tokens that were seen recently and cached. Reads are typically **~0.1× (Anthropic) or ~0.25–0.5× (OpenAI)** of the normal input price. Anthropic also charges a *write* premium (1.25× for 5-minute TTL, 2× for 1-hour) the first time you cache a prefix. Caching only helps when you resend a large stable prefix (a long system prompt, a document, few-shot examples) across many requests.
4. **Batch-API discount** — for asynchronous, non-latency-sensitive workloads, most providers give a flat **~50% off** both input and output if you submit a batch and collect results within a window (typically up to 24h).
5. **Reasoning / thinking-token billing** — the cost trap. Reasoning models (OpenAI o-series, Claude with extended/adaptive thinking, Gemini thinking) generate hidden intermediate "thinking" tokens **billed at the output rate** even though you never see most of them. A request that returns 200 visible tokens might bill you for 4,000 output tokens. This can silently dominate your bill.

### The one-line summary format

The minimum viable comparison line covers four numbers:

```
<model-id>: ctx=<max_input> out=<max_output> in=$<input>/M out=$<output>/M
```

`/M` means **per million tokens** — the unit humans reason in. Providers publish per-million on their pages, but tools like LiteLLM store per-*token* (a tiny decimal like `0.0000006`). Confusing the two is a factor-of-a-million error, which is exactly the kind of bug that ships to production and shows up as a "why is our bill $47,000" incident. Always state the unit explicitly in your summary.

### The arithmetic that actually bites: effective $/M

The advertised per-token price is not your effective cost. Two corrections matter.

**Correction 1 — blend input and output by your real ratio.** If your workload sends 1,000 input tokens and gets 500 output tokens per request, your effective cost is not the input price:

```
cost/request = 1000 × input_$/token + 500 × output_$/token
```

For `gpt-4o-mini` (illustrative: $0.15/M in, $0.60/M out):

```
= 1000 × 0.15e-6 + 500 × 0.60e-6
= 0.00015 + 0.00030
= $0.00045/request  →  $0.45 per 1,000 requests
```

Note the output half ($0.30) is *twice* the input half ($0.15) despite being fewer tokens — because output is 4× the per-token price. If your task is generation-heavy (summaries, code, long answers), output dominates and the headline input price is nearly irrelevant.

**Correction 2 — reasoning tokens inflate the output count.** Suppose a reasoning model bills at $2/M in, $8/M out, you send 1,000 input tokens, it emits 300 visible output tokens — but generates 3,000 hidden reasoning tokens:

```
billed output = 300 + 3000 = 3300 tokens
cost = 1000 × 2e-6 + 3300 × 8e-6
     = 0.002 + 0.0264
     = $0.0284/request
```

Your "300-token answer" cost as if it were 3,300 tokens. The effective output volume is **11× the visible output**. This is why you cannot estimate a reasoning model's cost from the size of its answers — you must measure actual `usage.completion_tokens` (which includes reasoning) or `usage.output_tokens` on a representative sample.

### Where each field hides (ASCII map)

```
MODEL CARD PAGE                    PRICING PAGE
┌────────────────────┐            ┌──────────────────────────┐
│ context window ────┼─ easy      │ input  $/1M  ── headline  │
│ max output ────────┼─ FOOTNOTE  │ output $/1M  ── 3-5x, 2nd │
│ modalities ────────┼─ tab/icons │ cached input ── fine print│
│ knowledge cutoff ──┼─ spec table│ batch -50%  ── separate pg│
│ deprecation date ──┼─ changelog │ reasoning tok ── FOOTNOTE │
└────────────────────┘            └──────────────────────────┘
     "advertised context ≠ usable context" — verify by testing recall
```

## Worked example

Let's reduce three provider pages to one line each, live. Numbers for OpenAI and Gemini below are **illustrative representative values — always verify against the live page**, because pricing moves. Anthropic numbers are from the current `claude-api` skill (authoritative as of this writing).

**OpenAI — `gpt-4o-mini`** (illustrative): context 128k, max output 16k, text+vision in / text out, $0.15/M in, $0.60/M out, cached input ~$0.075/M, batch −50%.

```
gpt-4o-mini: ctx=128k out=16k in=$0.15/M out=$0.60/M  (cache $0.075/M, batch -50%)
```

**Anthropic — `claude-haiku-4-5`** (authoritative from claude-api skill): context 200k, max output 64k, $1.00/M in, $5.00/M out, cache read ~0.1× (~$0.10/M), cache write 1.25×/2×, batch −50%.

```
claude-haiku-4-5: ctx=200k out=64k in=$1.00/M out=$5.00/M  (cache-read ~$0.10/M, batch -50%)
```

**Anthropic — `claude-opus-4-8`** (authoritative): context 1M, max output 128k, $5.00/M in, $25.00/M out.

```
claude-opus-4-8: ctx=1M out=128k in=$5.00/M out=$25.00/M
```

**Google — `gemini-2.0-flash`** (illustrative): context ~1M, max output ~8k, text+vision+audio in / text out, low single-digit-cents/M in, single-digit-dimes/M out, context caching available, batch −50%.

```
gemini-2.0-flash: ctx=1M out=8k in=$~0.10/M out=$~0.40/M  (verify — pricing moves)
```

Now a decision. Say you're building a high-volume classifier: ~1,200 input tokens, ~20 output tokens per request, 5 million requests/month, latency-tolerant (batchable). Compute cost-per-1k-requests for Haiku 4.5, standard vs batch:

```
per request (standard) = 1200 × 1e-6 + 20 × 5e-6 = 0.0012 + 0.0001 = $0.0013
per 1k = $1.30 ;  per 5M/month = $6,500

with batch -50%: per 1k = $0.65 ; per 5M = $3,250
with a cached 1000-token shared prefix (read ~$0.10/M):
  input = 200×1e-6 (fresh) + 1000×0.10e-6 (cached) = 0.0002 + 0.0001 = $0.0003
  + output 20×5e-6 = 0.0001 → $0.0004/req → $2,000 per 5M
  combine batch + cache → ~$1,000 per 5M
```

Same model, same task, **6.5× cost swing** purely from reading the pricing shape correctly and using batch + caching. That's the payoff of the extraction habit.

## How it shows up in production

- **The reasoning-token bill shock.** A team ships a reasoning model for a "cheap" summarization task, estimates cost from visible output length, and gets a bill 8× their forecast. The fix is always the same: pull `usage` from real responses and count reasoning/thinking tokens as output. Estimate from measured usage, never from answer length.
- **`max_tokens` truncation.** Someone sets up a 1M-context RAG pipeline, feeds it huge documents, and generations mysteriously cut off. They confused context window with max output. The output cap is the smaller, separate number — read it off the card explicitly.
- **The per-token/per-million decimal error.** A cost dashboard multiplies LiteLLM's per-token price (`6e-7`) by request volume but the code somewhere else treats it as per-million. Off by 1e6 in either direction: either the dashboard says everything is free, or it says a single request cost $600. Always assert the unit at the boundary.
- **Advertised vs usable context.** A "1M-token" model is sold on the headline, but recall degrades badly past a few hundred k ("lost in the middle"). Teams that trust the advertised number without a needle-in-haystack test at *their* real context length ship RAG systems that silently drop retrieved facts. The pricing page's context number is a ceiling, not a quality guarantee.
- **Cache that never hits.** Caching is billed as a discount, but only if your prefix is byte-identical across requests. A timestamp or a per-request UUID in the system prompt invalidates the cache every time — you pay the *write* premium and never collect the read discount, making caching cost *more*. Verify `cache_read_input_tokens > 0` in real responses.

## Common misconceptions & failure modes

- **"The headline price is the price."** No — it's the input price. Output is 3–5× and usually dominates generation workloads. Always read both.
- **"Cheaper per token = cheaper for me."** Only true at equal correctness and equal token counts. A model that's half the price but emits 4× the reasoning tokens is more expensive per useful answer. Reason in `$/useful-answer`, not `$/token` (this is the Week 1 `$/correct` metric again).
- **"Context window is how much I can generate."** No — that's max output, a separate and smaller cap.
- **"Multimodal means it outputs images/audio."** Usually it means it *accepts* them. Check input vs output modality separately.
- **"Caching always saves money."** Only for large, stable, reused prefixes, and only if the prefix is byte-identical. Small or volatile prefixes pay the write premium for nothing.
- **"Batch is just slower, same price."** Batch is typically −50% on both input and output. If your workload tolerates async, not using batch is leaving half the money on the table.
- **"LiteLLM's price is per million."** It's per **token**. Multiply by 1e6 to get the human-readable `$/M`.

## Rules of thumb / cheat sheet

- **Ten fields, every time:** ctx, max-output, modalities (in/out), cutoff, sunset · input-$, output-$, cached-$, batch-%, reasoning-token billing.
- **One-line format:** `model: ctx=IN out=OUT in=$X/M out=$Y/M` — extend with `(cache …, batch …)` when they matter.
- **Output is 3–5× input.** Budget from the output side for anything generative.
- **Effective $/M = blend by your real in:out ratio**, then add reasoning tokens to the output count.
- **Reasoning tokens bill as output** and are invisible — measure `usage`, never guess from answer length.
- **Batch ≈ −50%** for async workloads. Cache reads ≈ 0.1× (Anthropic) to 0.25–0.5× (OpenAI) of input; Anthropic charges a write premium (1.25×/2×).
- **Advertised context ≠ usable context** — needle-test at your real length before trusting it.
- **Per-token vs per-million:** LiteLLM = per-token; multiply by `1e6`. State the unit in every summary.
- **For authoritative current Anthropic model IDs and pricing, use the `claude-api` skill** — don't guess Claude numbers from memory; they change (e.g. Sonnet intro pricing windows).

## Connect to the lab

This lecture is the literacy behind Week 2's **`src/modelcard.py`** cheat-sheet generator. Instead of reading pricing HTML by hand, you call `litellm.get_model_info(model)` — which exposes `max_input_tokens`, `max_output_tokens`, `input_cost_per_token`, and `output_cost_per_token` — and format them into exactly the one-line summary above. Remember the unit conversion: LiteLLM prices are per-token, so multiply by `1e6` for the `/M` figures your `summarize()` function prints. Run it over your `models.yaml` list, commit the output as `smoke/reports/modelcards.md`, and you've turned "read a pricing page" into "read my own table."

```python
# src/modelcard.py — the core of the lab deliverable
import litellm

def summarize(model: str) -> str:
    m = litellm.get_model_info(model)
    return (f"{model}: ctx={m.get('max_input_tokens')} "
            f"out={m.get('max_output_tokens')} "
            f"in=${m.get('input_cost_per_token', 0) * 1e6:.2f}/M "
            f"out=${m.get('output_cost_per_token', 0) * 1e6:.2f}/M")
```

Where LiteLLM's metadata lags a fresh release (it's community-maintained), fall back to the provider page or, for Anthropic, the `claude-api` skill — and note the discrepancy in your report.

## Going deeper (optional)

- **LiteLLM docs** (`docs.litellm.ai`) — read the "Completion Cost" and model-info pages; the model price map lives in the `litellm` repo as `model_prices_and_context_window.json` (search: `litellm model_prices_and_context_window json`). Reading that file directly is the fastest way to see every field LiteLLM tracks.
- **Provider pricing pages** — OpenAI (`platform.openai.com/docs/pricing`), Anthropic (`docs.anthropic.com`, plus the `claude-api` skill for current IDs/pricing), Google Gemini (`ai.google.dev`). Bookmark the pricing and model-overview pages; they're the source of truth over any cached number.
- **Prompt caching mechanics** — Anthropic's prompt-caching docs and OpenAI's prompt-caching guide (search: `OpenAI prompt caching`, `Anthropic prompt caching cache_control`). Understand the prefix-match invariant before you rely on cache discounts.
- **Reasoning-token billing** — OpenAI's reasoning-models guide (search: `OpenAI reasoning tokens billing`) explains that reasoning tokens are billed as output; Anthropic's extended/adaptive thinking docs cover the equivalent.
- **Artificial Analysis** (`artificialanalysis.ai`) — cross-model cost/latency/quality at a glance; useful as a sanity check on your extracted numbers, not a replacement for the primary page.

## Check yourself

1. A model page shows "1M context." A teammate says "great, we can generate a 1M-token report in one call." What's wrong, and which field settles it?
2. Model A is $0.20/M in, $0.80/M out. Model B is $0.10/M in, $1.50/M out. Your workload is 500 input / 2,000 output tokens per request. Which is cheaper per request, and by how much?
3. Why can't you estimate a reasoning model's cost from the length of its visible answers?
4. `litellm.get_model_info("gpt-4o-mini")` returns `input_cost_per_token = 1.5e-7`. What is the `$/M` input price, and what's the arithmetic?
5. You enable prompt caching but `cache_read_input_tokens` is always 0 across repeated requests. Name two likely causes.
6. Give the one-line summary format and explain why the `/M` unit label is load-bearing.

### Answer key

1. It confuses **context window** (max input, 1M) with **max output tokens** (a separate, smaller cap — often 8k–128k). The generation is limited by max output; a 1M-token single response is impossible if the output cap is 128k. The **max output tokens** field settles it.
2. Model A: `500×0.20e-6 + 2000×0.80e-6 = 0.0001 + 0.0016 = $0.0017`. Model B: `500×0.10e-6 + 2000×1.50e-6 = 0.00005 + 0.0030 = $0.00305`. **Model A is cheaper** (~$0.0017 vs ~$0.0031), by ~$0.00135/request (~1.8×), because the workload is output-heavy and A's output price is far lower — despite B's cheaper input.
3. Reasoning models generate **hidden thinking tokens billed at the output rate** that don't appear in the answer. A short visible answer can carry thousands of billed reasoning tokens. You must read actual `usage` (completion/output tokens) on real samples, not infer from answer length.
4. `1.5e-7 × 1e6 = $0.15/M`. LiteLLM stores **per-token**; multiply by 1,000,000 to get per-million.
5. (a) A **volatile prefix** — a timestamp, UUID, or per-request ID inside the cached region invalidates it every call. (b) The prefix is **below the minimum cacheable size**, or non-deterministic serialization (unsorted JSON keys, changing tool order) makes the bytes differ each time. Either way you pay writes and collect no reads.
6. `<model>: ctx=<in> out=<out> in=$<X>/M out=$<Y>/M`. The `/M` label is load-bearing because tools store prices **per token** (tiny decimals) while pricing pages quote **per million**; without the explicit unit, a reader or a downstream calculation can be off by a factor of 1,000,000 — a common and expensive production bug.
