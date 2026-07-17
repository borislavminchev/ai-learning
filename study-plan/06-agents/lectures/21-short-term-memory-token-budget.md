# Lecture 21: Short-Term Memory as a Token Budget

> Most engineers reach for "memory" and imagine a database. But an agent's short-term memory isn't storage you look things up in — it's the context window you refill on *every single turn*, and it has a hard ceiling and a per-turn price. Reframe it as a **budget** and the whole problem sharpens: every token spent on stale history is a token unavailable for the current task, and you re-pay for it each round-trip. After this lecture you'll be able to reason about why naive conversation cost grows quadratically, pick the right compaction technique in reach-for order, keep provider prompt caches warm instead of silently torching them, and read/build a concrete `ShortTerm` manager that provably holds a 40-turn conversation under a fixed token ceiling.

**Prerequisites:** the agent loop and native tool calling (Week 1); the stateless nature of the messages API (you resend history every turn); basic big-O and arithmetic · **Reading time:** ~28 min · **Part of:** AI Agents & Agentic Systems (Expanded Deep Track), Week 5

## The core idea (plain language)

The model API is **stateless**. It does not remember your last turn. Every time your agent loops, you hand the model the *entire* transcript again — system prompt, tool definitions, every prior user message, every assistant reply, every tool result — plus the new turn. The model reads all of it, then produces the next token.

That single fact is the whole lecture. Because you resend everything each turn, the context window is not a place you *store* memory; it is a **working set you rebuild and pay for on every request**. It has two hard constraints:

1. **A ceiling.** The window is finite (200K, 1M — big, but finite). Overflow it and the request 400s or silently truncates. A long-running agent *will* hit this if you do nothing.
2. **A recurring price.** Input tokens are billed per request. If turn 30 carries 8,000 tokens of history, you pay for those 8,000 tokens *again* on turn 31, and again on turn 32.

So short-term memory is a **budget allocation problem**, not a storage problem. You have, say, 3,000 tokens of working memory you're willing to spend per turn. The question is never "where do I keep the history" — it's "*which* history earns its place in this turn's budget, and what do I do with the rest?" The four techniques below are four answers to that question, and you reach for them in order of increasing effort.

## How it actually works (mechanism, from first principles)

### Why naive accumulation is quadratic

Suppose every message (a user turn or an assistant reply) is ~100 tokens, and you just append forever — the demo-grade approach. On turn *k* you resend roughly `100 · k` tokens.

The *cumulative* input tokens you're billed for across a whole conversation is the sum:

```
100·1 + 100·2 + 100·3 + … + 100·N  =  100 · N(N+1)/2  ≈  50·N²
```

That `N²` is the killer. A 20-message chat bills ~20K cumulative input tokens; an 80-message chat bills ~340K — **17× more for 4× the length.** Latency tracks it too: the model re-reads the growing prefix every turn, so time-to-first-token creeps up as the conversation drags. This is the "context bloat" pitfall from Week 1 made quantitative. The fix is to stop the window from growing linearly — cap it, and the per-turn cost becomes O(1), so cumulative cost drops to O(N).

```
tokens/turn
   ^
8k |                              ● naive (linear per turn → N² cumulative)
   |                        ●
   |                  ●
   |            ●
3k |·····●····························· budget ceiling
   |   ●   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ bounded (sawtooth, capped)
   +--------------------------------> turn
```

### Technique 1 — Rolling window (reach for this first)

Keep the last *N* turns verbatim; drop everything older. That's it.

- **Cost:** O(1) per turn once you hit the cap. Trivial to implement — a `deque(maxlen=N)`.
- **The catch:** it drops early context **silently**. If the user stated their deploy target in turn 2 and asks about it in turn 30, a 6-turn window has already forgotten it, and the model will confidently make something up. There's no error — just quiet amnesia.

Use it when the task is genuinely local (a short Q&A, a stateless-ish chat) and early context doesn't matter. It's the cheapest thing that works, so it's the default you upgrade *away from* only when you can show it's dropping something load-bearing.

### Technique 2 — Summarization / compaction

When the window crosses a threshold, replace the oldest turns with an **LLM-generated summary** and keep only the recent ones verbatim. This is the pattern behind Anthropic's long-context guidance and the compaction behavior in Claude Code (when a session runs long, it summarizes earlier context server-side into a compaction block and continues).

- **Win:** roughly **10× compression** on the compacted span. 2,300 tokens of old dialogue become a ~230-token summary that still carries the facts, decisions, names, and numbers.
- **The catch:** the summary is **lossy**, and the model chose what to drop. The failure mode is that it drops *the one detail you needed* — the account number, the exact error string, the constraint the user gave in passing. You also pay an **extra LLM call** to produce the summary (latency spike at the moment compaction fires), and a bad summarizer can hallucinate facts into the summary that were never said.

Mitigations that matter in production: prompt the summarizer to *preserve* specifics ("KEEP facts, decisions, names, numbers"), fold the *previous* summary into each new one so old facts aren't re-dropped, and keep a generous `keep_last` so recent detail is always verbatim.

### Technique 3 — Scratchpad / note-taking (external memory)

Give the agent a tool to write structured notes to an external file (or a store) and read them back on demand. The notes live *outside* the context window; only what the agent explicitly reads back consumes budget. This is the core idea behind long-horizon agents — Anthropic's "context engineering" writing and Manus's "use the file system as context" both land here.

- **Win:** **unbounded external memory, bounded context.** The agent can accumulate arbitrarily much across a long run and pull in only the slice relevant to the current step. It's the difference between "hold the whole book in your head" and "keep notes and re-read the page you need."
- **The catch:** the agent has to *decide* what to write and when to read — that's extra tool calls, extra reasoning, and a new failure surface (it writes junk notes, or forgets to consult them). It's more moving parts than a rolling window. Reach for it when the task is genuinely long-horizon: multi-hour runs, work that spans many sub-tasks, anything where "what did I learn 40 steps ago" is a real question.

### Technique 4 — Cache-friendly message ordering (free money)

This one isn't about *shrinking* the budget — it's about not *overpaying* for the part you can't shrink. Providers cache prompt prefixes. The rule that governs everything:

> **Prompt caching is a prefix match. Any byte change anywhere in the prefix invalidates the cache for everything after it.**

The render order is **tools → system → messages**. So put your **stable** content — system prompt, tool definitions, few-shot examples — *first*, and **never reorder or edit it**. Then the provider serves that prefix from cache on every subsequent turn.

The economics (Anthropic, approximate — verify against current pricing):

| Token class | Price vs. base input |
|---|---|
| Cache **write** (5-min TTL) | ~1.25× |
| Cache **write** (1-hour TTL) | ~2× |
| Cache **read** | ~0.1× |
| Uncached input | 1× |

So a cached prefix costs ~1.25× once, then ~0.1× on every hit — versus paying 1× every single turn. Over a long conversation that's ~90% off the stable portion. OpenAI does the same thing **automatically** (prompts above ~1024 tokens are cached by prefix, cached input billed at a discount, no API changes) — the ordering discipline still applies, because their cache is prefix-matched too.

The trap: **reordering or editing an early message invalidates the cache and silently multiplies cost and latency.** No error — your bill just quietly doubles. Concrete silent invalidators to hunt for:

- `datetime.now()` or a request UUID interpolated into the system prompt → every turn is a cache miss.
- `json.dumps(tools)` without `sort_keys=True` → non-deterministic key order → different bytes → miss.
- Adding or reordering a tool mid-conversation → tools render at position 0, so this invalidates *everything*.
- Switching models mid-run → caches are per-model; full rebuild.

Other rules worth knowing: there's a small **minimum cacheable prefix** (roughly 1024–4096 tokens depending on model — shorter prefixes silently won't cache, `cache_creation_input_tokens` just reads 0), and a cap of **4 cache breakpoints** per request. Learn the cache-breakpoint rules once; they pay rent forever.

## Worked example

Let's build a `ShortTerm` manager and *prove* a 40-turn conversation stays under budget. Parameters: `budget = 3000` tokens, `keep_last = 6` messages, a pinned `system` prompt of ~200 tokens, and each message ~100 tokens.

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")     # approximation; see note below
def ntok(msgs): return sum(len(enc.encode(m["content"])) for m in msgs)

class ShortTerm:
    def __init__(self, llm, budget=3000, keep_last=6):
        self.llm, self.budget, self.keep_last = llm, budget, keep_last
        self.system  = []      # pinned, FIRST, built once, NEVER mutated → cache stays warm
        self.summary = None    # compacted older turns (one slot)
        self.window  = []       # recent verbatim turns

    def add(self, role, content):
        self.window.append({"role": role, "content": content})
        if ntok(self._assemble()) > self.budget:   # budget check drives compaction
            self._compact()

    def _compact(self):
        old, self.window = self.window[:-self.keep_last], self.window[-self.keep_last:]
        prev = f"Previous summary:\n{self.summary}\n\n" if self.summary else ""
        text = "\n".join(f'{m["role"]}: {m["content"]}' for m in old)
        self.summary = self.llm(f"{prev}Compress into <=200 words, KEEP facts, "
                                f"decisions, names, numbers:\n{text}")

    def _assemble(self):
        msgs = list(self.system)                    # pinned prefix always first
        if self.summary:
            msgs.append({"role": "system", "content": f"[memory] {self.summary}"})
        return msgs + self.window                    # volatile content last

    def context(self): return self._assemble()
```

Four design decisions, each mapping to a technique above:

- **`system` is pinned and never mutated** → the cache prefix stays warm (Technique 4). Note it's built *once* in `__init__` and only ever read in `_assemble`. That discipline is the free money.
- **`window`** is the rolling verbatim tail (Technique 1) — the last `keep_last` messages always survive compaction, so recent detail is never lossy.
- **`summary`** is the compaction slot (Technique 2) — one slot, and `_compact` folds the *previous* summary in so facts don't get dropped across successive compactions.
- **`ntok`-based budget check in `add`** is the trigger: compaction fires only when assembled context exceeds `budget`.

### Proving it stays bounded

Trace the token count. Assembled = `200 (system) + summary + 100·|window|`.

**First compaction.** With no summary yet, the window grows until `200 + 100·|window| > 3000`, i.e. `|window| = 29` messages (200 + 2900 = 3100). Compaction fires: `old = 29 − 6 = 23` messages (2,300 tokens) get compressed ~10× into a ~250-token summary; 6 messages stay verbatim. New assembled size:

```
200 (system) + 250 (summary) + 600 (6 msgs) = 1050 tokens  ✓ well under 3000
```

**Steady state.** The window regrows from 6. Assembled = `200 + 250 + 100·|window|`, which crosses 3000 at `|window| = 26` (200 + 250 + 2600 = 3050). Compaction fires again: `old = 20` messages folded into the summary (which the ≤200-word cap holds at ~250 tokens), window back to 6 → assembled back to ~1050.

So the manager **oscillates in a sawtooth between ~1050 and ~3050 tokens, forever.** Over 40 turns (80 messages) it compacts ~3 times and *never exceeds ~3050 tokens*. Contrast the naive approach:

| Approach | Peak turn size | Cumulative input tokens (80 msgs) |
|---|---|---|
| Naive append | ~8,200 tokens | ~340,000 (O(N²)) |
| `ShortTerm` (budget 3000) | ~3,050 tokens | ~120,000 + 3 summary calls (O(N)) |

Bounded peak, ~3× lower cumulative cost, and — critically — it **cannot overflow the window**, so no context-overflow 400s no matter how long the conversation runs. That's the acceptance criterion for Week 5: a 40-turn conversation stays under a hard token budget with no overflow errors, proven by a token trace.

### The caching layer on top

Now layer Technique 4. Say `system` + tool defs = 2,000 tokens with a `cache_control` breakpoint on the last system block. Over 40 turns: turn 1 writes the cache (~2,500 token-equivalents at 1.25×), turns 2–40 read it (~200 each × 39 ≈ 7,800). Total ≈ **10,300** vs. **80,000** if you paid full price for those 2,000 tokens every turn — ~87% off the stable prefix, for free, just by never editing `system`.

> **Production caveat on `ntok`:** `tiktoken`/`cl100k_base` is OpenAI's tokenizer and *undercounts* Claude tokens by ~15–20% (more on code). For an Anthropic backend, budget against the real count via the `count_tokens` endpoint, or pad your budget by ~20% so the approximation errs safe. Getting this wrong means your "3000-token budget" is really ~3,600, and you're closer to the ceiling than you think.

## How it shows up in production

- **The overnight-bill surprise.** A tool loops on a failing call, the transcript grows unbounded, and because you're re-sending it every turn, the *input* token bill balloons quadratically. You wake up to a 6-figure spend on a demo agent. Bounded windows plus the token budget from Week 1 are what stop this.
- **The "why did it forget?" ticket.** A rolling window silently drops the turn where the user gave a constraint. The agent contradicts itself 20 turns later. There's no error to grep — you find it only by reading the trace and noticing the constraint fell out of the window. Instrument compaction: log *what* got summarized so you can answer "was this fact in scope."
- **The lossy-summary regression.** Compaction summarizes away an account number the user gave once; the agent later "confidently" transfers to the wrong account. This is why the summarizer prompt must preserve specifics and why side-effecting actions should re-verify against a source of truth, not trust recalled context.
- **The cache that never hits.** Someone adds `f"Current time: {datetime.now()}"` to the top of the system prompt "for context." Every request is now a cache miss. Latency and cost roughly double, silently. You catch it only by checking `cache_read_input_tokens` in the usage block — if it's 0 across identical-prefix requests, a silent invalidator is at work. Diff the rendered prompt bytes between two turns to find it.
- **The compaction latency spike.** The extra summarization call fires mid-conversation and the user sees a 2-second stall right when the window crosses threshold. Mitigate by compacting slightly *before* the hard ceiling (headroom), or by using a cheaper/faster model for the summarization call specifically.

## Common misconceptions & failure modes

- **"Bigger context windows make this obsolete."** No. A 1M window raises the ceiling but not the *per-turn price* — you still pay for every token you resend, every turn. A bloated 500K-token context is expensive and slow even when it fits, and models get measurably worse at using the middle of a very long context. Budget still matters.
- **"Memory means a database."** Conflating short-term (the token budget you curate this turn) with long-term (what survives a process restart — next lecture) leads to storing chat history in a vector DB and re-injecting all of it. Short-term memory is a *curation* problem inside a fixed budget, not a persistence problem.
- **"Summarize everything to save the most tokens."** Over-compaction destroys the recent detail the model needs to act *now*. Keep a generous verbatim `keep_last`; only compact the genuinely-old tail.
- **"Caching is automatic, I don't need to think about ordering."** OpenAI's is automatic; Anthropic's needs explicit `cache_control` breakpoints. Both are prefix-matched, so *both* are destroyed by editing early content. The ordering discipline is what makes caching work — automatic or not.
- **"Rolling window is too dumb to ship."** For genuinely local tasks it's the correct answer. Reaching for compaction or a scratchpad when a window would do is gold-plating — more code, more failure surface, more latency, for no measured win.

## Rules of thumb / cheat sheet

- **Context window = working memory = a budget, not storage.** You re-pay for every token every turn.
- **Reach-for order:** rolling window → summarization → scratchpad → (always) cache-friendly ordering. Start with the simplest; upgrade only when you can name what it's dropping.
- **Rolling window:** cheapest, O(1)/turn, drops early context *silently*. Default for local tasks.
- **Summarization:** ~10× compression, costs one extra LLM call, risk is a lossy summary. Prompt it to keep facts/decisions/names/numbers; fold the previous summary in; keep a generous `keep_last`.
- **Scratchpad:** unbounded external memory, bounded context. For long-horizon runs; more moving parts.
- **Cache ordering:** stable content FIRST (tools → system → few-shot), **never edit or reorder it.** Warm cache ≈ 0.1× reads vs. 1× every turn — free money.
- **Cache invalidators to grep for:** `datetime.now()`/UUID in system prompt, unsorted `json.dumps`, tools added/reordered mid-run, model switch. Verify with `cache_read_input_tokens` — if it's 0, something's invalidating.
- **Compact *before* the hard ceiling,** not at it — leave headroom for the current turn's output and for tokenizer undercounting (~20% pad if using `tiktoken` on Claude).
- **Prove boundedness with a token trace.** Peak assembled size must stay under budget across the whole run. That trace *is* your test.

## Connect to the lab

This lecture is the theory behind Week 5's `memory/shortterm.py`. The lab has you build exactly this `ShortTerm` manager — pinned `system`, `summary` slot, verbatim `window`, `ntok` budget check, `_compact()` keeping the last K — and then run a 40-turn conversation while logging the assembled token count each turn. The Definition-of-Done is the token trace: the conversation completes with **zero context-overflow errors** and the assembled size **never exceeds the hard budget**. When you write `test_shortterm.py`, assert both the boundedness invariant (max assembled tokens ≤ budget) and that `system` is byte-identical across turns (your cache-warmth guarantee).

## Going deeper (optional)

- **Anthropic — "Effective context engineering for AI agents."** The canonical framing of context as a scarce resource and the techniques to curate it. Search: `Anthropic effective context engineering agents`.
- **Anthropic — Prompt caching documentation** (root: `platform.claude.com/docs`). The authoritative cache-breakpoint rules, write/read multipliers, minimum-prefix sizes, and the `cache_control` API. Read it before you tune caching in anger. Search: `Anthropic prompt caching`.
- **OpenAI — Prompt caching guide.** The automatic-caching model and its prefix rules, for the OpenAI backend. Search: `OpenAI prompt caching automatic`.
- **Manus — "Context Engineering for AI Agents" / "use the file system as context."** The scratchpad-as-external-memory pattern from a shipped long-horizon agent. Search: `Manus context engineering file system as context`.
- **Claude Code compaction** — reference behavior for automatic summarization of long sessions. Search: `Claude Code compaction context`.

## Check yourself

1. The API is stateless. Explain, with the summation, *why* naively appending every turn makes cumulative input-token cost grow as O(N²) rather than O(N).
2. You're building a customer-support agent where the user's account tier (stated in turn 1) governs answers 30 turns later. Which of the four techniques is *wrong* to reach for first, and why? Which would you use?
3. A colleague adds `"Session started: " + datetime.now().isoformat()` to the top of the system prompt. The bill doubles overnight with no code path change. Explain the mechanism and name the one usage field you'd check to confirm the diagnosis.
4. In the `ShortTerm` trace, the assembled size oscillates between ~1,050 and ~3,050 tokens across 40 turns. Explain what produces the *lower* bound and what produces the *upper* bound.
5. Why is `keep_last` a lossiness lever, and what's the failure mode at each extreme (`keep_last` too small vs. too large)?
6. Summarization gives ~10× compression but a rolling window is O(1) and lossless on what it keeps. Give one concrete task where you'd still prefer summarization over a pure rolling window, and one where you'd prefer the window.

### Answer key

1. On turn *k* you resend the whole transcript, ~`100·k` tokens. The cumulative bill is the sum `100·(1+2+…+N) = 100·N(N+1)/2 ≈ 50·N²`. It's the *re-sending of the growing prefix every turn* that squares the cost — each turn's linear size, summed over N turns, is quadratic. Capping the per-turn size at a constant collapses the sum back to O(N).

2. A **rolling window** is wrong to reach for first: a small window silently drops turn 1's account tier long before turn 30, and the agent will fabricate an answer with no error. Correct choices: **summarization/compaction** (so the tier survives in the summary — prompt it to keep the fact) or, cleaner, a **scratchpad** where the tier is written once as a durable note and read back on demand. The tier is a *durable fact*, not conversational filler, so it belongs somewhere the window's eviction can't reach it.

3. `datetime.now()` makes the system prompt's bytes different on every request. Because caching is a **prefix match** and the system prompt is at the front of the prefix, every request is a cache miss — you pay full 1× input price for the whole stable prefix every turn instead of ~0.1× cache reads. Check `cache_read_input_tokens` in the response usage: if it's 0 across requests that should share a prefix, a silent invalidator (this timestamp) is confirmed. Fix: move dynamic values out of the cached prefix (into a later message).

4. **Lower bound (~1,050):** the state right after a compaction — `200 (system) + ~250 (summary) + 6·100 (kept window)`. **Upper bound (~3,050):** the window regrowing until the `ntok` budget check trips just past 3,000, which triggers the next compaction. The sawtooth is the window filling to the ceiling, getting compacted back down to the floor, and repeating.

5. `keep_last` sets how many recent messages survive compaction verbatim — i.e., how much recent detail is *lossless*. Too **small**: the model loses recent context it needs to act on the current turn (the summary is lossy, and you kept almost nothing verbatim), causing incoherent replies right after a compaction. Too **large**: compaction rarely reclaims enough tokens, so you either compact constantly (extra LLM calls, latency) or blow the budget — you've effectively defeated the compaction.

6. **Prefer summarization:** a long research or planning conversation where early *decisions and findings* (not verbatim wording) matter 30 turns later — a lossy-but-fact-preserving summary keeps them in budget where a rolling window would evict them. **Prefer rolling window:** a rapid interactive coding or chat loop where only the last few exchanges are relevant and the extra summarization call's latency would hurt UX — the window is cheaper, lossless on what it keeps, and adds no latency spike.
