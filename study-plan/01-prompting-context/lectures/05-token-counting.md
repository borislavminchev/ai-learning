# Lecture 5: Per-Provider Token Counting

> Every cost estimate, every context-budget calculation, every latency projection you make about an LLM request starts from one number: how many tokens is this? Get that number from the wrong tokenizer and everything downstream — the dollar figure in your dashboard, the "will this fit in the window" check, the cache-savings math you'll build in Week 3 — drifts silently, in the direction that makes you look good in the demo and bleeds money in production. This lecture makes token counting a discipline: the right tool per provider, a side-by-side view that shows you the divergence with your own eyes, and a clear map of everything those counts feed. After this you will never again reach for `tiktoken` to size a Claude prompt, and you'll instrument token counts as a first-class metric instead of an afterthought.

**Prerequisites:** Phase 0 (a working `uv` env, provider SDKs installed, an API key for at least one frontier provider) · basic familiarity with what a token is from the tokenization → sampling mental model · **Reading time:** ~18 min · **Part of:** Prompting & Context Engineering, Week 1

---

## The core idea (plain language)

A token is the atomic unit an LLM reads and bills. It is *not* a character, *not* a word, and — this is the part that bites — *not* a provider-independent quantity. Each model family ships its own tokenizer: a fixed vocabulary plus a set of merge rules that chop text into pieces. OpenAI's models use a tokenizer exposed by the `tiktoken` library. Anthropic's Claude models use a different, proprietary tokenizer. Google's Gemini uses yet another. **The same string produces a different token count on each.**

That single fact is the whole lecture. Everything else is consequence.

The failure mode is seductive because `tiktoken` is fast, local, free, and *right there* in your dependency tree. So people use it to estimate tokens for every model, Claude included. It is wrong for Claude — it undercounts by roughly 15–20% on ordinary English prose, and by more on code and non-English text. "Undercounts" means your estimate is *lower* than reality. Your cost calculator says $0.83; the invoice says $1.00. Your budget check says "4,000 tokens, fits comfortably in an 8K window"; the real prompt is 4,800 and you're closer to the edge than you think. Nothing errors. Nothing warns you. The number is just quietly, consistently optimistic.

The discipline is one rule: **count tokens with the tokenizer that belongs to the model you are actually calling.** `tiktoken` for OpenAI. Anthropic's `count_tokens` endpoint for Claude. Gemini's `count_tokens` for Gemini. No cross-application, ever.

---

## How it actually works (mechanism, from first principles)

You do not need the internals of byte-pair encoding to be an expert here. You need exactly one sentence of mechanism, and then the operational consequences.

**The one sentence:** each tokenizer learned its vocabulary and merge rules from a different training corpus, so it carves text into different pieces — and a piece (a token) is the billing and context unit, so different carving means different counts.

That's it. When people ask "why don't the numbers match," the answer is: different vocabularies, different merges. You don't need to derive the merges; you need to *respect* that they differ.

### Why the divergence is worse on code and non-English

English prose is what these tokenizers were most heavily trained on, so common English words often map to a single token. But:

- **Code** is full of tokens the English-optimized vocabulary handles poorly: `snake_case`, `camelCase`, brackets, operators, indentation runs, `__dunder__` names. A tokenizer tuned differently may split `getUserById` into `get`/`User`/`By`/`Id` on one model and `getUser`/`ById` on another. The counts diverge more, because there's more disagreement per character.
- **Non-English text** (accented Latin, CJK, Cyrillic, emoji) is where vocabularies differ most. A word that's one token in a multilingual-heavy vocabulary might be four tokens (or several raw bytes) in a more English-centric one.

So the ~15–20% undercount is a rule of thumb for typical English. **On a codebase or a Japanese document it can be worse.** Treat "15–20%" as approximate and directional, not a constant you can multiply by.

### A worked numeric example of the drift

Say you're building an invoice-extraction harness (the Week 1 lab) and each request sends a system prompt + a tool schema + one invoice, and you estimate ~4,000 tokens per call using `tiktoken`. You're calling Claude Opus, priced (illustratively — check current pricing) at $5 / 1M input tokens. You plan for 100,000 invoices/day.

- **Your estimate:** 4,000 tok × 100,000 = 400M tok/day × $5/1M = **$2,000/day**.
- **Reality at a 17% undercount:** 4,680 tok × 100,000 = 468M tok/day × $5/1M = **$2,340/day**.

That's **$340/day, ~$124,000/year**, of drift — entirely from using the wrong tokenizer for an estimate. No bug, no crash. Just a number that was always a little low. Now imagine the finance team built the pricing model for the product on your $2,000 figure.

The same drift attacks your **context budget**. If you're packing a 200K window and you believe your retrieved context is 150K tokens (tiktoken) when it's really ~176K (Claude's counter), you have 26K fewer tokens of headroom than you think. You'll hit `stop_reason: "model_context_window_exceeded"` or silently push critical instructions past where the model attends — and you'll debug the *symptom* (bad output) for hours before suspecting the *cause* (your budget math was fiction).

### The three correct tools

```
Provider      Tokenizer / API                          Local or network?
------------  ---------------------------------------  -----------------
OpenAI        tiktoken (encoding_for_model)            Local, instant
Anthropic     client.messages.count_tokens(...)        NETWORK CALL
Gemini        model.count_tokens(...) / client...      NETWORK CALL
```

Note the asymmetry that shapes how you write the code: **`tiktoken` runs locally and instantly; Anthropic's and Gemini's counters are network round-trips to the provider.** OpenAI publishes its tokenizer as a library, so you tokenize on your own machine. Anthropic and Gemini do not ship a local tokenizer you should rely on — you ask *their* endpoint how many tokens your payload is, and it answers authoritatively for the exact model you name.

This matters because:

- Anthropic's `count_tokens` is **model-specific**. Pass the same `model` ID you'll use for the actual call — different Claude models can tokenize differently, and the endpoint accounts for the full request shape (see below).
- Because it's a network call, it has latency and it consumes rate limit. You do not call it in a hot loop naively. You **cache** results in tests and batch where you can (more on this under production concerns).

Minimal correct code, three ways:

```python
# OpenAI — local, via tiktoken
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
openai_tokens = len(enc.encode(text))

# Anthropic — network call to the count_tokens endpoint, model-specific
from anthropic import Anthropic
client = Anthropic()
claude_tokens = client.messages.count_tokens(
    model="claude-opus-4-8",
    messages=[{"role": "user", "content": text}],
).input_tokens

# Gemini — network call to its count_tokens
from google import genai
gclient = genai.Client()
gemini_tokens = gclient.models.count_tokens(
    model="gemini-2.5-pro",
    contents=text,
).total_tokens
```

Three tools, three numbers, one input. That's the whole mechanism.

---

## Worked example: the side-by-side table that makes you believe

Reading "15–20%" doesn't change behavior. *Seeing* your own text tokenized three ways does. The single most valuable artifact in this lecture is a script that prints the same input tokenized by all three tokenizers next to each other. Build it once; it will recalibrate your intuition permanently.

```python
import tiktoken
from anthropic import Anthropic

SAMPLES = {
    "english_prose": "The quarterly report shows revenue grew twelve percent.",
    "code": "def get_user_by_id(user_id: int) -> Optional[User]:\n    return db.query(User).filter_by(id=user_id).first()",
    "non_english": "四半期報告書によると、売上high は12パーセント増加しました。",
}

client = Anthropic()
enc = tiktoken.encoding_for_model("gpt-4o")

def claude_count(text):
    return client.messages.count_tokens(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": text}],
    ).input_tokens

print(f"{'sample':<16}{'tiktoken':>10}{'claude':>10}{'drift %':>10}")
for name, text in SAMPLES.items():
    tk = len(enc.encode(text))
    cl = claude_count(text)
    drift = (cl - tk) / tk * 100
    print(f"{name:<16}{tk:>10}{cl:>10}{drift:>9.1f}%")
```

An illustrative run (your exact numbers will vary by model version — the *shape* is the point, not the digits):

```
sample           tiktoken    claude   drift %
english_prose          11        13     18.2%
code                   34        41     20.6%
non_english            27        38     40.7%
```

Two things to internalize from a table like this:

1. **The drift is always in the same direction for Claude vs. tiktoken: tiktoken is lower.** So if you use tiktoken as a Claude proxy, you will *always* underestimate cost and *always* overestimate remaining budget. It's a one-sided error, which is the dangerous kind — it never accidentally corrects itself.
2. **The non-English row blows past the "15–20%" rule.** This is why you count, not multiply. A blanket 1.18× fudge factor would have massively under-counted the Japanese sample. The only reliable number is the one from the right tokenizer on *your actual text*.

Print this table for your Week 1 lab's real invoice data, not toy strings. When you see your own prompts drift 20%+, the discipline stops feeling like pedantry.

---

## What token counts actually feed

Token counting isn't a curiosity — it's an input to four downstream systems. Getting the count wrong corrupts all four.

**1. Context-window budgeting.** Every model has a hard input+output ceiling. You are constantly deciding "how much retrieved context / history / few-shot examples can I include before I run out of room (or push critical content into the lost-in-the-middle dead zone)?" That decision is arithmetic on token counts. Wrong counts → wrong budget → truncated context or an over-limit 400.

**2. Per-request cost.** Cost = input_tokens × input_price + output_tokens × output_price. If input_tokens is a tiktoken guess for a Claude call, your per-request cost is wrong by the drift, every request, forever. Multiply by request volume and the error is a line item.

**3. Latency estimation.** Time-to-first-token and total generation time scale with token counts — more input tokens to process (prefill) and more output tokens to generate (decode) both cost wall-clock time. If you're setting request timeouts or building an SLA, you're reasoning about token counts. A budget built on undercounts will set timeouts too tight.

**4. Cache accounting (Week 3).** In Week 3 you'll drive Anthropic prompt-cache hit rates and report cached-token percentage and cost delta across 100 requests. Every one of those metrics is token counts sliced into categories: `input_tokens` (full price), `cache_read_input_tokens` (~0.1× price), `cache_creation_input_tokens` (~1.25× price). If your baseline token math is wrong, your cache-savings math is wrong on top of it — you'll report a cache win that's partly a counting artifact. Clean token discipline now is the foundation the caching lecture stands on.

---

## How it shows up in production

**The cost dashboard that's quietly 17% low.** The most common real incident: someone wires up a "estimated spend" panel using tiktoken because it's local and free, applies it to a mixed OpenAI+Claude workload, and the panel reads consistently below the actual invoice. For months nobody notices because the *shape* of the graph is right — it goes up when traffic goes up. Then finance reconciles against the real bill and there's a scramble to explain a five-figure gap. The fix is one function swap; the damage is trust.

**The context budget that overflows under load.** In dev, your prompts are small and everything fits. In production, real users send long documents, conversation history accumulates, retrieved context balloons. If your "will it fit" guard uses the wrong tokenizer, the undercount means you sail past the true limit exactly when inputs get large — i.e., under production load, not in your tests. You get intermittent 400s or silently truncated context that degrades answers, and it correlates with traffic, which is a miserable thing to debug.

**Tokenization includes more than the user's text.** This trips up almost everyone: the token count that gets billed and budgeted is **the whole request**, not just the user message. The **system prompt** counts. The **tool/function schemas** count — and a fat JSON schema for several tools can be *hundreds to thousands* of tokens that are present on *every single call*. Few-shot examples count. If you count only the user's input, you'll under-budget by a fixed overhead on every request. Anthropic's `count_tokens` endpoint is the right tool precisely because you can hand it the full request shape — `system`, `messages`, and `tools` — and it returns the real input token count for that exact payload:

```python
client.messages.count_tokens(
    model="claude-opus-4-8",
    system=SYSTEM_PROMPT,          # counts
    tools=TOOL_SCHEMAS,            # counts — often the sneaky big one
    messages=[{"role": "user", "content": invoice_text}],
).input_tokens
```

If you're wondering why your "short" request costs more than expected, print the count *with* the system prompt and tools included. The user text was never the whole story.

**Output tokens are counted separately and cost more.** Input and output are billed as distinct meters, and **output is typically several times more expensive per token** (e.g., a model at $5/1M input can be $25/1M output — a 5× ratio). Token *counting* endpoints size your *input*; your *output* cost you can only measure after the fact from the response's `usage.output_tokens`, or bound ahead of time with `max_tokens`. Two operational consequences: (a) a verbose model that pads answers hits your bill on the expensive meter, so output length is a cost lever, not just a UX one; and (b) never estimate total request cost from input alone — a 500-token input that triggers a 4,000-token answer is dominated by the output side.

**`count_tokens` is itself a network call — cache and batch it.** Because Anthropic's and Gemini's counters are API round-trips, calling them naively per-item in a tight loop adds latency and burns rate limit. In tests especially, you'll count the same fixtures repeatedly — memoize the result keyed on (model, payload hash) so your test suite doesn't fire a thousand identical count requests. In production, count once per unique payload, not once per retry. And remember it consumes your rate-limit budget alongside real inference calls.

---

## Common misconceptions & failure modes

- **"A token is about 4 characters / 0.75 words, so I'll just divide."** A useful back-of-envelope for a rough English guess, useless for cost or budget precision, and wildly off for code and non-English. Rules of thumb are for napkins, not dashboards.
- **"tiktoken is the tokenizer, so it works for everything."** tiktoken is *OpenAI's* tokenizer. It is authoritative for OpenAI models and *wrong* for Claude and Gemini. This is the single error this lecture exists to kill.
- **"I'll multiply the tiktoken count by 1.18 to correct for Claude."** No. The drift is content-dependent — ~15–20% on English, much higher on non-English and worse on code. A fixed multiplier is right on average and wrong on every actual document. Count with the real tokenizer.
- **"I counted the user message, so I have the token count."** You have *part* of it. System prompt + tool schemas + few-shot examples are all in the billed input. Count the full request shape.
- **"Total cost ≈ input tokens × price."** Output is a separate, usually pricier meter. Ignoring it undercounts the requests that generate long answers — often the expensive ones.
- **"Counts are stable across model versions."** Tokenizers can change between model families/versions. Anthropic's counter is model-specific for this reason. When you migrate models, **re-baseline** — don't reuse counts measured against the old model.
- **"count_tokens is free like tiktoken."** For Claude/Gemini it's a network call with latency and rate-limit cost. Cache it in tests, batch it in production.

---

## Rules of thumb / cheat sheet

- **One rule above all:** count with the tokenizer of the model you're calling. `tiktoken` → OpenAI. `count_tokens` → Claude. Gemini `count_tokens` → Gemini.
- **Never** use tiktoken as a Claude/Gemini proxy. It undercounts Claude by ~15–20% on English (approximate), worse on code and non-English.
- **Count the whole request:** system prompt + tool/function schemas + examples + user input. Not just the user turn.
- **Input and output are separate meters.** Output usually costs several× more per token. Size input with a counter; bound output with `max_tokens`; measure actual output from `usage.output_tokens`.
- **Claude/Gemini counters are network calls.** Memoize by (model, payload-hash) in tests; count once per unique payload in prod; mind the rate limit.
- **Re-baseline on model migration.** Don't carry old counts to a new model.
- **Don't multiply-and-hope.** A fixed correction factor is wrong per-document. Count the real text.
- **Build the 3-way table once** (tiktoken / Claude / Gemini on identical input) and look at it. Seeing the drift is what changes behavior.
- **These counts feed four things:** context budget, per-request cost, latency estimate, and (Week 3) cache accounting. Wrong counts corrupt all four.

---

## Connect to the lab

In the Week 1 lab you write `src/tokens.py`: `tiktoken` for OpenAI, `client.messages.count_tokens(...)` for Claude, Gemini's `count_tokens` for Gemini — and print a table so you *see* the divergence on your real invoice text, not toy strings. Make that table count the **full** request (system prompt + the JSON tool schema + the invoice), because that's what you'll actually be billed for. Record token counts per cell in your v1/v2 × zero/few-shot accuracy report — this is where the Definition-of-Done "token-count table prints correct per-provider counts (Claude via `count_tokens`, not tiktoken)" gets checked, and where you'll first watch few-shot examples inflate the input token count.

## Going deeper (optional)

Real, named sources — verify against current docs, and treat any specific percentage or price here as approximate:

- **Anthropic — "Token counting" docs** (search: `Anthropic token counting count_tokens`). The authoritative reference for `POST /v1/messages/count_tokens` and the SDK `count_tokens` method, including that it counts system prompt and tools.
- **OpenAI — `tiktoken` repository on GitHub** (`github.com/openai/tiktoken`). The library itself, plus the README's guidance on `encoding_for_model`. The OpenAI Cookbook's "How to count tokens with tiktoken" notebook is the canonical worked example (search: `OpenAI cookbook count tokens tiktoken`).
- **OpenAI — Tokenizer playground** (`platform.openai.com/tokenizer`). Paste text, see tokens highlighted — good for building intuition about how code and non-English split.
- **Google — Gemini API "Count tokens" docs** (search: `Gemini API count tokens`). The `count_tokens` method and multimodal token accounting.
- **Provider pricing pages** for the current input/output per-token rates — always check these live rather than trusting a number in a lecture; prices and the input/output ratio change.
- Search queries when you can't recall the exact URL: `Anthropic messages count_tokens endpoint`, `tiktoken encoding_for_model`, `Gemini count_tokens total_tokens`.

## Check yourself

1. You have a script that estimates Claude API cost using `tiktoken`. Which direction is the error, and why is that direction especially dangerous?
2. Name the four downstream systems that consume token counts, and give one concrete failure for each when the count is wrong.
3. Your teammate says "I'll just count the user message with the right tokenizer, that's the token count." What are they missing, and roughly how large can the missing piece be?
4. Why can't you correct a tiktoken estimate for Claude by multiplying by a fixed factor like 1.18?
5. `count_tokens` for Claude is a network call. Name two practical consequences and how you'd mitigate each.
6. A 300-token prompt produces a 5,000-token answer. Why is estimating total cost from the input count alone badly wrong here?

### Answer key

1. **tiktoken undercounts Claude** (~15–20% on English, more elsewhere), so your estimate is *lower* than reality. It's dangerous because the error is one-sided and always flattering: cost looks cheaper than it is and remaining budget looks larger than it is, so nothing ever prompts you to investigate — until the invoice or an over-limit error does it for you.
2. **Context-window budgeting** (undercount → you pack past the true limit → 400 or truncated context under load); **per-request cost** (wrong input count → wrong bill, every request); **latency estimation** (token counts drive prefill/decode time → timeouts set on bad counts are too tight); **cache accounting in Week 3** (cached-token % and cost-delta are token counts by category → a wrong baseline corrupts your reported cache savings).
3. They're counting only the user turn. The billed input also includes the **system prompt, the tool/function schemas, and any few-shot examples** — and tool schemas in particular can be hundreds to thousands of tokens present on *every* call. Count the full request shape (`system` + `tools` + `messages`).
4. Because the drift is **content-dependent, not constant** — roughly 15–20% on English prose but much higher on non-English text and worse on code. A fixed multiplier is right only on average and wrong on every specific document. The only reliable number comes from the correct tokenizer run on the actual text.
5. (a) **Latency + rate-limit cost:** it's a round-trip that competes with real inference calls — mitigate by memoizing results keyed on (model, payload-hash) so tests don't re-count identical fixtures, and count once per unique payload in prod rather than per retry. (b) **It's model-specific:** pass the exact `model` you'll call, and re-baseline when you migrate models rather than reusing old counts.
6. Input and output are **separate meters and output is typically several× more expensive per token**. Here the 5,000 output tokens dwarf the 300 input tokens, and they're billed at the higher rate — so total cost is dominated by the output side, which the input count tells you nothing about. Estimate input with a counter; bound output with `max_tokens` and measure it from `usage.output_tokens`.
