# Lecture 11: Context Engineering and Token Budgeting

> Every LLM call is a working-memory allocation problem you are solving whether you know it or not. The model has a fixed context window — a bounded slab of attention you rent by the token — and on each request you decide what goes in it: system prompt, tool schemas, conversation history, retrieved chunks, the user's turn, room for the answer. Most engineers treat that decision as "stuff in everything that might be relevant and let the model sort it out." That instinct is wrong, and it costs you twice: you pay for the extra tokens *and* the extra tokens make the answer worse. This lecture makes context a budget you allocate on purpose. After it you will be able to break any request into a per-category token breakdown, instrument a call so every request logs that breakdown to `runs.jsonl` and prints a budget table, reason about which categories are stable enough to cache, and place your critical facts where the model actually attends — with the ablation to prove it.

**Prerequisites:** Lecture 5 (per-provider token counting — you must be able to count the *full* request shape, not just the user turn) · a working provider SDK + API key · comfort reading JSONL · **Reading time:** ~24 min · **Part of:** Prompting & Context Engineering, Week 3

---

## The core idea (plain language)

The context window is the model's working memory. It is finite — 200K tokens on current Claude, 128K–1M on the various OpenAI and Gemini tiers — and it resets to empty on every stateless API call. There is no hidden state between requests; whatever the model "knows" in a given call is exactly what you put in the window that call. That makes the window the one interface you fully control on a frozen model, and it makes filling it a *budgeting* problem.

Here is the mindset shift, and it is the whole lecture: **more context is not better.** The naive model in most engineers' heads is "the window is a bucket; fill it up; unused space is wasted." The correct model is "the window is a budget; every token you spend competes for the model's attention, and irrelevant tokens actively *subtract* from answer quality." This second effect has a name — **context rot** — and it is not folklore. As you pile in tokens that don't bear on the question, the model's accuracy on the tokens that *do* matter degrades: it gets distracted, it anchors on the wrong passage, it loses the thread. So irrelevant context is not neutral padding you happen to pay for. It is a quality *tax* you pay to make the answer *worse*.

That reframes the goal. You are not trying to give the model everything it could conceivably use. You are trying to curate the **minimal high-signal token set** that answers *this* request — relevance over volume. A tight 3,000-token prompt of exactly-relevant material routinely beats a sprawling 80,000-token dump that happens to contain the answer somewhere in the middle. Cheaper *and* more accurate. Those two usually trade off; here they align, which is why context engineering is the highest-leverage knob you own.

The discipline that makes this concrete is **per-request token budgeting**: split every call into named categories, count each one, log the breakdown, and make deliberate allocation decisions — trim history, cap retrieved context, bound output — instead of letting the prompt grow by accretion until something breaks.

---

## How it actually works (mechanism, from first principles)

### The eight categories

Every request you send is, mechanically, one flat sequence of tokens the model reads left to right. But that flat sequence is *assembled* from parts that behave very differently — different sizes, different volatility, different cacheability. The budgeting discipline is to account for it by category. The eight that matter:

```
Category            What it is                                Typical size     Volatility      Cacheable?
------------------  ----------------------------------------  ---------------  --------------  -----------
system              Instructions/persona/rules (frozen)       200 – 2,000 tok  ~never changes  YES (prefix)
tools               JSON schemas for available functions      300 – 4,000 tok  changes rarely  YES (prefix)
history             Prior turns in a conversation             0 – 100k+ tok    grows each turn  PARTLY
retrieved_context   RAG chunks / docs pulled in for this call 0 – 100k+ tok    per-request      RARELY
user_turn           The actual question/input this call       10 – 2,000 tok   every request   NO
output              Tokens the model generates (its answer)   bounded by you   every request   NO (billed separately)
cache_read          Prefix tokens served from cache (~0.1×)   0 – prefix size  per-request     (is the win)
cache_write         Prefix tokens written to cache (~1.25×)   0 or prefix size  first call only  (one-time cost)
```

The first six are where the *content* lives. The last two — `cache_read` and `cache_write` — are not separate content; they are how the provider *bills* the prefix depending on whether it was already cached. You track them as categories because on your invoice and in your latency, a token read from cache and a token computed fresh are wildly different line items (roughly 0.1× vs 1× price, and cache reads skip prefill compute so they're faster). More on that below.

Two structural facts to burn in now, because both matter for cost:

1. **The whole request is billed input, not just the user turn.** `system` + `tools` + `history` + `retrieved_context` + `user_turn` are *all* input tokens you pay for on *every* call. The fat tool schema you defined once is re-sent and re-billed every single request unless it's cached. This is exactly the "count the full request shape" discipline from Lecture 5, now as a budgeting frame.
2. **Output is a separate, usually pricier meter.** Output tokens are billed distinctly and typically cost several× more per token than input (e.g. a model at $5/1M input can be $25/1M output). So `output` is not just a UX knob — a verbose model that pads answers hits your bill on the *expensive* meter. Bound it with `max_tokens`.

### Volatility is the axis that determines cacheability

Sort the categories by how often their bytes change and a pattern jumps out:

```
STABLE (same bytes across many requests)          VOLATILE (different bytes every request)
  system      — frozen prompt                       user_turn         — new question each call
  tools       — fixed schema list                   retrieved_context — different chunks per query
                                                     history           — one more turn each time
        \_______ cacheable prefix _______/                \_____ not usefully cacheable _____/
```

This maps directly onto the caching rule you'll use all week (and which Lecture 12 covers in depth): **prompt caching is a prefix match — any byte change anywhere in the prefix invalidates everything after it.** The provider renders the request in a fixed order (`tools → system → messages`), hashes the prefix up to your cache breakpoint, and if that exact byte-prefix was seen recently it serves those tokens from cache at ~0.1× price and skips recomputing them.

The budgeting consequence is an *ordering* rule that falls straight out of volatility: **put stable, high-volume categories first (they cache) and volatile categories last (they can't).** `tools` and `system` go up front, get a cache breakpoint, and are read from cache on request #2 onward. `retrieved_context`, `history`, and `user_turn` come after — because they change per request, putting them early would push the cache boundary earlier and waste the cacheable prefix. If you interleave a `datetime.now()` timestamp into your system prompt, you've made a "stable" category volatile and silently zeroed your cache. Volatility discipline *is* cache discipline.

### The window is a budget: allocate it deliberately

A 200K window is not headroom; it is an allocation you make. Write it as an inequality you must satisfy:

```
system + tools + history + retrieved_context + user_turn + output  ≤  context_window
```

Note `output` is on the left — the tokens the model generates share the same window as everything you sent, so `max_tokens` for the answer competes with your input. Run out of room and you get `stop_reason: "model_context_window_exceeded"` (or worse, silent truncation of the oldest history).

The three levers you actually pull, in order of how often you'll reach for them:

- **Trim history.** History grows unboundedly in a multi-turn session — it's the category most likely to blow your budget over time. Cap it: keep the last N turns verbatim and roll everything older into a compact summary (Lecture 13's compaction). Trigger the compaction on a *token threshold*, not every turn.
- **Cap retrieved context.** RAG will happily hand you 50 chunks. Take the top-k that clear a relevance score, hard-cap the total (e.g. "≤ 8 chunks, ≤ 6,000 tokens"), and *drop the rest* — because irrelevant chunks are the context-rot tax, not free insurance.
- **Bound output.** Set `max_tokens` to what the task actually needs. An extraction that returns a 200-token JSON object does not need `max_tokens: 4096` reserved.

The mistake is to treat the window as "big enough, don't worry about it" until a long document or a long conversation quietly pushes you over — in production, under load, exactly when inputs are largest. Budgeting means you *decided* the allocation before the request, so overflow is a caught condition, not a 3 a.m. incident.

### Lost in the middle: placement is part of the budget

Even when everything fits, *where* in the window a fact sits changes whether the model uses it. Transformer attention over long contexts is empirically **U-shaped**: the model attends strongly to the *start* of the context (primacy) and the *end* (recency), and under-attends to the *middle*. A critical fact buried at the 50% position of a long context gets used less reliably than the same fact at the top or bottom. This is the "lost in the middle" effect, and it's a *placement* discipline layered on top of the *volume* discipline:

```
attention (schematic, long context)

 high │*                                   *
      │ *                                 *
      │   *                             *
      │      *                       *
 low  │          *  *   *   *   *  *
      └────────────────────────────────────
      start          middle           end
      (primacy)    (dead zone)     (recency)
```

So budgeting has two halves: **how many tokens** (volume — curate the minimal set) and **where they sit** (placement — critical facts to the edges). Put the instruction the model must not miss near the top of the prompt *and* restate the key constraint near the bottom, just before the user turn. Don't strand the one fact that determines the answer at the geometric center of a 60K-token blob.

---

## Worked example

Take the Week 1 invoice-extraction task and make the budget concrete. One request:

```
Category            Tokens    Price bucket    Notes
------------------  --------  --------------  ------------------------------------------
system                 450    input           extraction rules + output-format spec
tools                1,200    input            one JSON schema for the extract() tool
history                  0    input            single-shot, no prior turns
retrieved_context        0    input            single doc under the window, no RAG
user_turn            1,050    input            the invoice text (wrapped as data)
                    -------
INPUT TOTAL          2,700    input
output                 180    OUTPUT (5×)      the {vendor, ...} JSON object
```

**Cold cost** (illustrative Claude-style pricing: $5/1M input, $25/1M output):
input = 2,700 × $5/1M = **$0.0135**; output = 180 × $25/1M = **$0.0045**; total **$0.0180/request**.

Now notice the shape. `system + tools = 1,650` tokens — **61% of the input** — is *identical on every call*. Only `user_turn` (the invoice) changes. That's a textbook cacheable prefix. Put a cache breakpoint after `tools`+`system`:

- **Request #1 (cold, cache write):** the 1,650 prefix tokens are written to cache at ~1.25× = 2,063 effective tokens; plus 1,050 fresh user tokens. Slightly *more* expensive than cold — the one-time write cost.
- **Requests #2..100 (warm, cache read):** the 1,650 prefix is read from cache at ~0.1× = 165 effective input tokens, plus the 1,050 fresh invoice tokens = 1,215 effective input instead of 2,700.

Effective input cost per warm request: 1,215 × $5/1M = **$0.0061** vs $0.0135 cold — **~55% off the input bill**, and the prefill compute for those 1,650 tokens is skipped, so latency drops too. Across 100 requests the cached-token % is roughly `1,650 / 2,700 ≈ 61%` of input served cheap. *That number came directly from the budget table* — you cannot find the caching win without first accounting per category.

Now the anti-example. Someone "improves" the prompt by pasting the vendor's entire 40-page terms-of-service PDF into `retrieved_context` "for grounding," hoping it helps edge cases:

```
system                 450
tools                1,200
retrieved_context   38,000   ← the ToS dump
user_turn            1,050
INPUT TOTAL         40,700   (was 2,700)
```

Input cost jumps 15× to ~$0.20/request. And accuracy *drops*, because the one line that matters — the invoice total — is now the recency-dead middle of a 40K-token context, drowned in irrelevant legalese (context rot + lost-in-the-middle, together). They paid 15× more to get a worse answer. That is the failure this lecture exists to prevent, and the budget table is how you *see* it happening — the `retrieved_context` row screaming 38,000 is the tell.

---

## How it shows up in production

**The prompt that grew by accretion.** Nobody decides to build a 90K-token prompt. It happens one well-meaning commit at a time: someone adds three few-shot examples, someone bumps top-k from 5 to 20 "to be safe," someone stops trimming history because a bug report mentioned lost context. Each change looks locally reasonable; the aggregate is a request that costs 20× the original and answers worse. Without a per-category budget logged on every request, you have no way to see the drift — it's invisible until the bill or an eval regression surfaces it. *With* the budget table in `runs.jsonl`, you can plot `retrieved_context` tokens over time and catch the top-k bump the day it lands.

**The context-rot regression that looks like a model problem.** Accuracy quietly drops after a "harmless" change that added more context. The team suspects the model, tries a bigger model, tweaks temperature — chasing the wrong variable for days. The actual cause: the extra chunks pushed the signal into the middle and diluted attention. The diagnostic that would have saved the week is trivial once you budget: *did input tokens go up right before accuracy went down?* Correlate the two and the culprit is obvious.

**The cache that's silently off.** You added `cache_control` and assume you're saving money. But someone put a per-user ID or a `datetime.now()` early in the system prompt, so the prefix bytes differ every call and the cache never hits. Your budget instrumentation catches this immediately: `cache_read` tokens are persistently **zero** across identical-shaped requests. That single logged field is the difference between "we think caching works" and "we proved it does." (This is the audit you run in the Week 3 lab — inject the invalidator, watch `cache_read` collapse.)

**The window overflow under load.** Works in dev (short inputs), 400s in prod (long inputs + accumulated history). Because you never bounded history and never capped retrieved context, real traffic sails past the true limit exactly when documents are large. A budget with hard caps per category turns this from an intermittent production incident into a deterministic, tested guard: if `history` would exceed its cap, you compact *before* sending, not after the API rejects you.

**Output cost hiding in plain sight.** A team optimizes input tokens obsessively and ignores that their model returns a chatty 1,500-token explanation around a 50-token answer. Output is the 5× meter; the explanation is most of the bill. Bounding `output` with `max_tokens` and asking for terse structured output is often a bigger cost win than shaving the input.

### Instrumenting the budget

The mechanism is simple: count each category (Lecture 5's per-provider counters), read the cache/output numbers from the response `usage`, print a table, append a row to `runs.jsonl`. Do this on *every* call, not just when debugging — the value is the time series.

```python
import json, time
from anthropic import Anthropic

client = Anthropic()

def budgeted_call(system, tools, history, retrieved_context, user_turn, max_tokens=512):
    # Pre-count the input categories with the model's own counter (network call; cache in tests).
    def n(**kw): return client.messages.count_tokens(model=MODEL, **kw).input_tokens
    cats = {
        "system":            n(system=system, messages=[{"role":"user","content":"."}]) ,
        "tools":             n(tools=tools,   messages=[{"role":"user","content":"."}]) ,
        "history":           n(messages=history or [{"role":"user","content":"."}]),
        "retrieved_context": n(messages=[{"role":"user","content": retrieved_context or "."}]),
        "user_turn":         n(messages=[{"role":"user","content": user_turn}]),
    }  # NOTE: counted separately for a per-category view; the real billed input is the assembled request.

    t0 = time.time()
    resp = client.messages.create(
        model=MODEL, max_tokens=max_tokens,
        system=system, tools=tools,
        messages=(history or []) + [{"role":"user","content": retrieved_context + "\n" + user_turn}],
    )
    latency_ms = int((time.time() - t0) * 1000)

    u = resp.usage
    row = {
        "ts": time.time(), "model": MODEL, "latency_ms": latency_ms,
        **cats,
        "output":      u.output_tokens,
        "cache_read":  getattr(u, "cache_read_input_tokens", 0),
        "cache_write": getattr(u, "cache_creation_input_tokens", 0),
        "input_billed": u.input_tokens,
    }
    with open("runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

    # Print the budget table
    print(f"{'category':<20}{'tokens':>8}")
    for k in ["system","tools","history","retrieved_context","user_turn","output","cache_read","cache_write"]:
        print(f"{k:<20}{row[k]:>8}")
    return resp
```

Two honest caveats. First, counting each category in isolation (as above) is an approximation — role framing and message boundaries add a few tokens, so the per-category sum won't exactly equal `usage.input_tokens`. Reconcile against the *billed* `input_tokens` from `usage` for the authoritative number, and treat the category split as the *diagnostic* view. Second, `count_tokens` is a network call (Lecture 5) — memoize the stable categories (`system`, `tools`) so you're not re-counting frozen bytes every request.

---

## Common misconceptions & failure modes

- **"Unused context window is wasted — fill it."** The opposite. Empty budget is fine; *irrelevant* tokens are a quality tax (context rot) and a cost. Curate the minimal high-signal set.
- **"Bigger window means I don't have to budget."** A bigger window raises the ceiling, not the wisdom of hitting it. Lost-in-the-middle and context rot get *worse* with more tokens, and cost scales linearly. A 1M window is more rope, not a safety net.
- **"The model will just ignore the irrelevant parts."** It won't reliably. Attention is a finite, shared resource; distractor tokens measurably pull accuracy down. This is the entire premise of the lab's ablation.
- **"I only pay for the user's question."** You pay for the whole assembled request every call — `system` + `tools` + `history` + `retrieved_context` + `user_turn`. The tool schema you defined once is re-billed on every request unless cached.
- **"Output tokens are minor."** Output is the expensive meter (often ~5× input). A verbose answer can dominate the bill. Bound it.
- **"Where a fact sits doesn't matter as long as it's in the window."** Placement matters: edges are attended, the middle is a dead zone. Critical facts go to the edges.
- **"I added `cache_control`, so caching works."** Verify `cache_read_input_tokens` > 0 on warm requests. A single volatile byte in the prefix (a timestamp, a per-user ID, unsorted `json.dumps`) silently zeroes it.
- **"History is free to keep — just append."** History is the category most likely to blow your budget over a long session. Cap it and compact on a threshold.

---

## Rules of thumb / cheat sheet

- **Relevance over volume.** The goal is the minimal high-signal token set for *this* request, not everything that might help. Fewer, better tokens win on cost *and* accuracy.
- **Budget by category, every call.** Log `{system, tools, history, retrieved_context, user_turn, output, cache_read, cache_write}` to `runs.jsonl` and print a table. The time series is what catches drift.
- **Order by volatility:** stable first (`tools` → `system`, with a cache breakpoint), volatile last (`retrieved_context`, `history`, `user_turn`). Ordering *is* cache discipline.
- **Three levers when you're tight:** trim history (compact on a token threshold), cap retrieved context (top-k + hard token cap, drop the rest), bound output (`max_tokens` to task size).
- **`output` is on the wrong side of the ledger** — separate meter, ~5× price (approximate; check current pricing). Terse structured output is a cost lever.
- **Critical facts to the edges.** Put the must-not-miss instruction near the top *and* restate the key constraint just before the user turn. Never strand it in the middle.
- **Reconcile to `usage.input_tokens`.** Per-category counts are the diagnostic; the billed `usage` number is truth. Memoize counts for the stable categories.
- **Watch two signals together:** input tokens up + accuracy down = context rot, not a model problem. `cache_read` persistently 0 = a silent invalidator in your prefix.
- **The window is an inequality you must satisfy:** `system + tools + history + retrieved + user + output ≤ window`. Decide the allocation *before* the request.

---

## Connect to the lab

In the Week 3 lab (step 1) you wrap the extraction call so every request logs the eight-category token breakdown to `runs.jsonl` and prints a per-request budget table — exactly the `budgeted_call` instrument sketched above, reconciled against `usage.input_tokens`. Step 2 is the **lost-in-the-middle ablation**: build a prompt with a large context blob and one critical fact (the true invoice total), run it with the fact at the start, the middle, and the end across ~20 trials, and measure extraction accuracy at each position. You should see the middle underperform the edges — quantify the gap, then adopt "critical facts to the edges" as a standing policy (Definition of Done: a measured accuracy gap across ≥20 trials). The budget instrument and the ablation together are the milestone's "context-budgeted" pillar.

## Going deeper (optional)

Real, named sources — verify against current docs; treat any specific percentage or price as approximate:

- **Anthropic — "Effective context engineering for AI agents"** and **"Long context tips"** (search: `Anthropic effective context engineering`, `Anthropic long context tips`). The canonical framing of context as working-memory curation and the source of the "relevance over volume" mindset.
- **Anthropic — "Prompt caching" docs** (search: `Anthropic prompt caching cache_control`). The prefix-match rule, `cache_control` breakpoints, and the `usage.cache_read_input_tokens` / `cache_creation_input_tokens` fields you log as `cache_read` / `cache_write`. (Covered in depth in the next lecture.)
- **"Lost in the Middle: How Language Models Use Long Contexts"** — Liu et al., 2023 (search: `Lost in the Middle Liu 2023`). The paper behind the U-shaped attention finding; skim the figures for the intuition, don't get lost in the setup.
- **Provider pricing pages** for current input/output per-token rates and cache read/write multipliers — always check live rather than trusting a number here; the input/output ratio and cache pricing change.
- **Anthropic — "Token counting" docs** (search: `Anthropic count_tokens`) — the counter you use to fill the budget table (see Lecture 5).
- Search queries when you can't recall an exact URL: `context rot LLM long context degradation`, `prompt caching prefix invalidation`, `LLM context window budgeting`.

## Check yourself

1. State the two-part claim behind "more context is not better," and name the two distinct mechanisms by which irrelevant tokens hurt you.
2. You log the eight categories on every request. Which are typically the cacheable ones and why, and what ordering does that imply for the assembled prompt?
3. Your per-category token sum doesn't equal `usage.input_tokens`. Is something broken? Which number do you trust for the bill, and what is the category split good for?
4. A teammate raises `max_tokens` from 512 to 4096 "to be safe" and the cost per request jumps more than expected. Explain why, given the two meters.
5. Accuracy on your extraction task dropped this week. Input tokens rose the same day. Walk through the diagnosis and the two levers you'd pull.
6. What does the Week 3 lost-in-the-middle ablation vary, what does it measure, and what policy do you adopt from the result?

### Answer key

1. Part one: irrelevant tokens **cost money** — they're billed input on every call. Part two: they **lower accuracy**. The two mechanisms are **context rot** (distractor tokens dilute the model's finite attention, so it anchors on the wrong material) and **lost in the middle** (adding volume pushes the signal fact into the low-attention middle of the window). Cost up, quality down — the tax is two-sided.
2. The **stable** categories cache: `system` (frozen instructions) and `tools` (fixed schemas), because caching is a prefix byte-match and these bytes don't change across requests. That implies ordering them **first** with a cache breakpoint, and putting the **volatile** categories (`retrieved_context`, `history`, `user_turn`) **after** — because any change in an early category would move the cache boundary earlier and waste the cacheable prefix.
3. Nothing is broken. Counting each category in isolation misses the few tokens added by role framing and message boundaries when the request is assembled, so the sum is an approximation. **Trust `usage.input_tokens`** (the billed number) for cost; the **category split is the diagnostic view** that lets you see where the tokens are going and catch drift (e.g. a ballooning `retrieved_context`).
4. `output` is a **separate meter and typically ~5× the input price**. Raising `max_tokens` doesn't just reserve window space — if the model actually generates more tokens, those are billed on the expensive output meter. A longer answer therefore moves cost onto the pricier side; `max_tokens` should be set to what the task needs, not padded "to be safe."
5. **Diagnosis:** correlate the two signals — input tokens rising the same day accuracy fell is the signature of **context rot / lost-in-the-middle**, not a model regression. Find what added tokens (a top-k bump, extra few-shot, un-trimmed history) via the `runs.jsonl` budget rows. **Levers:** cap retrieved context (top-k + hard token cap, drop irrelevant chunks) and trim/compact history; and make sure the critical fact sits at an edge, not the middle.
6. It varies the **placement of one critical fact** (the true invoice total) — start vs middle vs end of a large context blob — across ~20 trials, and **measures extraction accuracy at each position**. The expected result is that the **middle underperforms the edges**; the adopted policy is **"critical facts to the edges"** — put must-not-miss content near the top and restate the key constraint just before the user turn.
