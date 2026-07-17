# Lecture 13: Rolling Compaction and Summarization

> Every multi-turn agent eventually hits the same wall: the conversation history outgrows the budget you gave it. You can truncate (and lose the invoice number the user gave you eight turns ago), or you can compact — fold the old turns into a compact summary and keep going. Compaction is what lets an agent run for 200 turns inside a window that only comfortably holds 30. But a lazy summary is worse than none: if it paraphrases away the entity IDs, amounts, and dates downstream steps depend on, you have quietly deleted the exact data the conversation existed to carry. This lecture teaches compaction as an engineering discipline. After it you can implement a `compact(history)` that triggers on a token threshold (not every turn), folds only the *old* turns while keeping recent ones verbatim, copies every structured identifier through unchanged, and — critically — you can write the pytest that *proves* no entity was dropped, because a compactor without that test is a liability.

**Prerequisites:** Per-provider token counting (Lecture 5); context budgeting and "lost in the middle" (Lecture 11); prompt caching as a prefix match (Lecture 12); comfort with pytest · **Reading time:** ~20 min · **Part of:** Prompting & Context Engineering, Week 3

## The core idea (plain language)

A frozen model has a fixed context window — say 200k tokens. Your per-request budget is smaller, because you split it across system prompt, tool definitions, retrieved context, the user's current turn, and room for output. Conversation history is one line item in that budget, and it is the *only* one that grows without bound: every turn appends more tokens. Left alone, history eats the whole budget, and then either the request 400s (over the window) or your framework silently drops the oldest turns.

Silent truncation is the trap. The oldest turns are often where the load-bearing facts live — "the invoice number is INV-2024-0917," "ship to account 4471." Drop those and the model, ten turns later, confidently invents a plausible-looking wrong invoice number, because nothing in its context contradicts it.

**Compaction** is the disciplined alternative. When history crosses a threshold, you take the *oldest* chunk of turns, summarize them into a compact block, and splice that block back in *in place of* the raw turns. The recent turns stay verbatim — the model still sees the last few exchanges word-for-word — but the ancient history collapses from 6,000 tokens of back-and-forth into ~400 tokens of summary. You reclaim budget without amnesia.

The non-negotiable rule this whole lecture orbits: **a compaction summary must preserve entities, IDs, amounts, and dates verbatim.** Prose can be paraphrased — "the user asked several questions about shipping" is fine. But `INV-2024-0917`, `$4,412.50`, `2024-09-17`, `account 4471` must appear in the summary *character-for-character identical* to the raw turns. A summary that says "the user mentioned an invoice number" instead of copying the actual number has destroyed information downstream steps need, and no cleverness recovers it. That is the difference between compaction and lossy truncation with extra steps.

## How it actually works (mechanism, from first principles)

### The trigger: a threshold, not a turn

The first design decision is *when* to compact. The wrong answer — the one the spine explicitly warns against — is "every turn." Here is the arithmetic. Say a compaction call summarizes ~6,000 tokens of history down to ~400, and the summarization call itself costs roughly 6,000 input + 400 output tokens. If you compact on *every* turn of a 50-turn conversation, you fire 50 summarization calls — paying the compaction tax constantly, often re-summarizing history that barely changed since last turn. Pure waste: latency every turn, tokens every turn, and (as we'll see) a busted cache every turn.

Instead, compact on a **token threshold**. Measure the current history size before each turn; only when it crosses a budget (the spine uses **4k tokens** as the example) do you fire a single compaction. Then history shrinks below the threshold and you run many normal turns for free until it grows past the line again.

```
history tokens
  6k ┤                              ╭─ compact! ──╮
  5k ┤                         ╭────╯             ╰─╮ (drops to ~2k)
  4k ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─╭──╯  THRESHOLD ─ ─ ─ ─ ╰──╮ ─ ─ ─ ─ ─ ─
  3k ┤                 ╭────╯                          ╰────╮
  2k ┤            ╭────╯                                     ╰───►
  1k ┤─────╭─────╯
     └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴──── turns
```

The sawtooth is the whole idea: history climbs, hits the line, gets folded, and climbs again. You pay the compaction tax a handful of times over a long session, not 50 times.

### What to fold vs. what to keep raw

Not all history is equal. The recent turns are what the model is actively reasoning over — the current sub-task, the last clarification, the tool result it just received. Paraphrasing those hurts quality directly. The ancient turns you mostly need for *facts*, not *nuance*.

So the split: **keep the last K turns verbatim; fold everything older into the summary.** A common default is to keep the most recent 4–6 turns (or "the last N tokens' worth of turns") raw and summarize the rest. The model sees recent exchanges word-for-word, and the old stuff is compressed to its factual skeleton.

```
BEFORE (history = 6,200 tokens, over the 4k threshold):
  [turn 1] [turn 2] [turn 3] ... [turn 11] [turn 12] [turn 13] [turn 14]
  \_______________ fold these ______________/ \______ keep raw ______/

AFTER (history = ~2,100 tokens):
  [SUMMARY of turns 1–10, entities verbatim]  [turn 11] ... [turn 14]
```

### The summary block: verbatim identifiers, paraphrased prose

The summarization prompt is where the verbatim rule gets enforced. You do **not** just say "summarize the conversation." You instruct the model to preserve a specific class of tokens exactly, and — the belt-and-suspenders move — you extract those tokens *yourself* with a deterministic regex pass and append them as a literal "Facts" block no model ever rewrites.

Two layers of defense:

1. **Prompt the model** to summarize prose but copy every ID, number, amount, and date verbatim. LLMs are decent at this but not reliable enough alone — they will occasionally "normalize" `INV-2024-0917` to `INV-2024-917` or round `$4,412.50` to `$4,400`.
2. **Extract entities deterministically** before summarizing and inject them into the summary as a verbatim list. This is the safety net: even if the model mangles an ID in its prose, the exact string survives in the Facts block, and downstream steps can find it.

A compacted block looks like:

```
<summary>
The user is reconciling a vendor invoice. They provided the invoice and
asked to verify the total against line items, then asked about payment
terms. The assistant confirmed the line-item sum and explained net-30.

VERBATIM FACTS (do not paraphrase):
- invoice_number: INV-2024-0917
- total_amount: $4,412.50
- invoice_date: 2024-09-17
- vendor: Acme Industrial Supply
- account: 4471
- payment_terms: net-30
</summary>
```

The prose is compressible and lossy-by-design. The Facts block is lossless and machine-checkable — which is exactly what makes the proving test possible.

### Splicing it back

Compaction is a pure transform on the message list: `compact(history) -> new_history`. You replace the folded raw turns with a single message (usually a user or system message tagged as a summary) and concatenate the kept-raw recent turns after it. The new history is shorter, factually complete, and the model reads it as: "here's the compressed backstory, here are the last few turns in full, now continue."

## Worked example

Run the numbers on a concrete session. Assume a 4k-token history threshold and a policy of keeping the last 4 turns raw.

State after turn 14:

```
turn  role       tokens   contains
 1    user        520     invoice INV-2024-0917, $4,412.50, 2024-09-17, Acme
 2    assistant   610     confirms fields
 3    user        430     account 4471, net-30 question
 4    assistant   580     explains terms
 ...
11    user        410
12    user        460     "also check PO-8823"
13    assistant   540
14    user        300
                 -----
        total ≈ 6,200 tokens  → OVER the 4k threshold, compact now
```

`compact(history)` runs:

1. **Split.** Keep turns 11–14 raw (~1,710 tokens). Fold turns 1–10 (~4,490 tokens).
2. **Extract entities** from turns 1–10: `INV-2024-0917`, `$4,412.50`, `2024-09-17`, `Acme Industrial Supply`, `4471`, `net-30`, `PO-8823`. (Snapshot this set — it's what the test asserts against.)
3. **Summarize** turns 1–10 into ~350 tokens of prose + the verbatim Facts block (~30 tokens).
4. **Splice**: `[summary(~380 tok)] + [turn 11..14]`.

Result:

```
                  before        after
history tokens    6,200         ~2,090   (380 summary + 1,710 recent)
turns visible     14 raw        1 summary + 4 raw
budget reclaimed  —             ~4,110 tokens
```

You dropped from 6,200 to ~2,090 tokens — back under threshold with headroom for ~9 more turns before the next compaction. Every entity from turns 1–10 is still present. Note that `INV-2024-0917`, `$4,412.50`, and `2024-09-17` now exist *only* inside the summary's Facts block. If the summarizer had paraphrased "the invoice number" instead of copying `INV-2024-0917`, the next turn asking "what was the invoice number again?" would get a hallucinated answer. The Facts block is the only thing standing between you and that bug.

### The proving test

The Definition of Done demands a pytest that fails if any entity is dropped. The discipline: snapshot every entity present *before* compaction, then assert every one is present *after*.

```python
import re

ENTITY_PATTERNS = [
    r"INV-\d{4}-\d{4}",              # invoice numbers
    r"PO-\d+",                       # purchase orders
    r"\$[\d,]+\.\d{2}",              # money amounts
    r"\b\d{4}-\d{2}-\d{2}\b",        # ISO dates
    r"\baccount \d+\b",              # account refs
]

def extract_entities(text: str) -> set[str]:
    found = set()
    for pat in ENTITY_PATTERNS:
        found.update(re.findall(pat, text))
    return found

def test_compaction_preserves_all_entities():
    history = load_fixture_session()           # 14 turns, over threshold
    before = extract_entities(serialize(history))

    compacted = compact(history)
    after = extract_entities(serialize(compacted))

    missing = before - after
    assert not missing, f"compaction DROPPED entities: {sorted(missing)}"
    # exact-string membership, not fuzzy — 100% or it fails
```

Two properties make this test trustworthy. First, it checks **exact string membership** — `$4,412.50` must appear as that exact string, so rounding to `$4,400` is caught. Second, it **fails loudly**: `assert not missing` with the dropped set in the message tells you precisely which identifier vanished. Write a fixture where the summarizer *does* paraphrase away an ID and confirm the test goes red — a proving test you've never seen fail is not proven.

## How it shows up in production

- **Truncation-by-default is the silent killer.** Most agent frameworks, if you don't configure compaction, just drop the oldest messages near the window limit. Invisible until a user references something from early in the session and the model confabulates. No error — just a support ticket. Compaction with a Facts block is the fix; the proving test keeps it fixed through refactors.

- **Compaction fights prompt caching — acknowledge the tension.** This is the sharp production tradeoff. Prompt caching is a *prefix match*: the provider caches a leading span of your request and reuses it while those bytes are identical (Lecture 12). But compaction **rewrites the prefix** — it replaces turns 1–10 with a summary, changing bytes at the *front* of the history. That invalidates the cache for everything after the rewrite. So the very turn you compact on pays full cold-input price, not cheap cache-read price. You cannot hold a stable cached prefix and a mutating compacted prefix at the same position. Mitigation: compact **infrequently** (threshold, not every turn) so invalidation is rare, and structure the prompt so the *truly* stable parts (system prompt, tool defs) sit *above* the compaction boundary and stay cached — only the history region below churns. Watch `cache_read_input_tokens`: expect it to drop near zero on the compaction turn, then recover on subsequent turns as the new prefix re-caches.

- **The compaction call has its own cost and latency.** Summarizing 4–6k tokens is a real model call — hundreds of milliseconds to seconds, plus its own token bill. On a threshold you amortize it across many free turns. Log how often compaction fires; if it fires every 2nd turn, your threshold is too low or your kept-raw window too large.

- **Entity extraction is only as good as your patterns.** The regex net catches what it's written to catch. A new document type introduces a new ID shape (`SKU_A1B2`, IBANs, tracking numbers) your patterns miss — those entities then rely on the model's prose alone and can get paraphrased away. Treat `ENTITY_PATTERNS` as living config; onboarding a new domain means adding its identifier shapes and a fixture.

- **Compaction can compound errors.** Compact a summary that already contains a summary (second-generation compaction over a long session) and paraphrase drift accumulates — prose gets vaguer each generation. The Facts block resists this (copied verbatim through each generation), but prose degrades. For very long sessions, prefer re-summarizing from a preserved raw log rather than summarizing the summary.

- **Debugging needs the raw log.** When a user disputes what the agent "remembered," you need to see what was actually in context. Persist pre-compaction raw turns (to disk/DB, not the window) so you can reconstruct what was folded and prove whether the entity was present before compaction. Compaction changes the window, not your audit trail.

## Common misconceptions & failure modes

- **"Compact every turn to stay small."** Wasteful and cache-hostile. You pay the summarization call and bust the cache every turn, re-compressing barely-changed history. Trigger on a threshold; run free turns in between.

- **"A summary is a summary — the model knows what's important."** No. Left to its own judgment a summarizer treats `INV-2024-0917` as prose and will paraphrase, normalize, or round it. You must *instruct* verbatim preservation AND back it with deterministic extraction. Never trust the model alone with the IDs.

- **"Preserve entities = keep the meaning."** Stricter than semantics: it's **character-for-character**. `$4,412.50` → `$4,400` "preserves the meaning" and still breaks reconciliation. `2024-09-17` → `Sept 17` loses machine-parseability. Exact strings, or it doesn't count — which is why the test uses exact membership, not fuzzy match.

- **"Keep the oldest turns raw, they're the important ones."** Backwards. Fold the *oldest* (compress to facts) and keep the *newest* raw (the model actively reasons over them). Old turns matter for facts, preserved by the Facts block; recent turns matter for nuance, preserved only by verbatim text.

- **"The test passes, so compaction is safe."** Only for the entity shapes your patterns cover and the fixtures you wrote. A test that has never gone red proves nothing — deliberately feed it a paraphrasing summarizer and watch it fail before you trust the green.

- **"Compaction and caching both make it cheaper, so use both aggressively."** They're in tension. Aggressive compaction rewrites the prefix and *kills* cache hits. The win comes from compacting *rarely* so caching survives between compactions. Measure `cache_read_input_tokens` and compaction frequency together.

- **"Just raise the threshold to avoid compacting."** Then history eats the budget meant for retrieved context and output, and you drift into the "lost in the middle" trough where a huge history buries the critical facts mid-context. Compaction isn't only about fitting the window — it's about keeping the high-signal token set small. A too-high threshold defeats the point.

## Rules of thumb / cheat sheet

- **Trigger on a token threshold, not per turn.** ~4k history tokens is a reasonable example default; tune to your window and per-request budget.
- **Keep the last 4–6 turns (or last ~N tokens) verbatim; fold everything older.** Recent = nuance (raw), old = facts (summary).
- **Two-layer entity preservation:** (1) prompt the summarizer to copy IDs/amounts/dates verbatim; (2) extract them deterministically and inject a verbatim Facts block the model never rewrites.
- **Verbatim means exact string** — no rounding, no date reformatting, no ID normalization.
- **Prove it with a test that snapshots entities before and asserts 100% present after** — exact membership, fails loudly with the dropped set.
- **Compaction busts prompt caching** (it rewrites the prefix). Compact rarely; put truly stable content above the compaction boundary; verify `cache_read_input_tokens` recovers on later turns.
- **Persist the pre-compaction raw log** off-window for audit and for re-summarizing without paraphrase drift.
- **Treat `ENTITY_PATTERNS` as living config** — extend it for each new identifier shape and add a fixture.
- **Log compaction frequency and reclaimed tokens.** Firing too often ⇒ threshold too low or kept-raw window too large.
- **Set the threshold from the budget, not the window.** History competes with tools, retrieved context, and output room; leave headroom.

### Skeleton implementation

```python
def compact(history, keep_raw=4, threshold=4000):
    if count_tokens(history) < threshold:
        return history                      # under budget: no-op, cheap path

    old, recent = history[:-keep_raw], history[-keep_raw:]

    # 1. deterministic safety net BEFORE summarizing
    facts = extract_entities(serialize(old))

    # 2. model summary of prose, instructed to copy IDs verbatim
    prose = summarize_prose(old)            # LLM call, verbatim instruction

    # 3. build a block where the facts are literal, not model-generated
    summary_msg = {
        "role": "user",
        "content": (
            "<summary>\n" + prose +
            "\n\nVERBATIM FACTS (do not paraphrase):\n" +
            "\n".join(f"- {f}" for f in sorted(facts)) +
            "\n</summary>"
        ),
    }
    # 4. splice: summary replaces old turns, recent stays raw
    return [summary_msg] + recent
```

## Connect to the lab

This is the theory behind Week 3 Lab step 3: simulate a multi-turn session that grows past a token threshold (e.g. 4k), implement `compact(history)` that summarizes older turns while copying all IDs/amounts/dates verbatim, and write the pytest asserting **100%** of pre-compaction entities survive — a test that *fails* if any are dropped (Definition of Done). Build the two-layer preserver (verbatim-instructed summary + deterministic Facts block) and the sawtooth trigger from this lecture. When you wire in `cache_control` (Lab step 4), observe the tension firsthand: watch `cache_read_input_tokens` drop on the compaction turn and recover after — that measured collapse-and-recover is the caching lesson made concrete.

## Going deeper (optional)

- **Anthropic — "Effective context engineering for AI agents."** Framing for compaction as working-memory curation, and where it fits alongside retrieval and tool-result pruning. Root: `docs.anthropic.com` (also on the Anthropic engineering blog). Search: `Anthropic effective context engineering agents`.
- **Anthropic — Claude Agent SDK / context compaction.** Anthropic's agent tooling ships automatic compaction; reading how they trigger and structure it is a canonical reference. Search: `Claude Agent SDK context compaction`.
- **LangChain / LangGraph — memory & summarization.** `ConversationSummaryBufferMemory` and LangGraph summarization nodes are widely-used reference implementations of "keep recent raw, summarize old." Root: `python.langchain.com`. Search: `LangGraph conversation summarization memory`.
- **LlamaIndex — chat memory buffers / summary memory.** Another canonical take on the recent-raw + summarized-old split. Root: `docs.llamaindex.ai`. Search: `LlamaIndex chat summary memory buffer`.
- **Anthropic — "Prompt caching" doc.** Re-read for the prefix-match invariant, to reason precisely about *why* compaction invalidates the cache. Root: `docs.anthropic.com`. Search: `Anthropic prompt caching prefix`.

## Check yourself

1. Why trigger compaction on a token threshold instead of on every turn? Give both a cost reason and a caching reason.
2. State the non-negotiable rule for a compaction summary, and explain why "preserves the meaning" is not strong enough.
3. Your history is over threshold. Which turns do you fold and which do you keep raw, and why that direction?
4. Describe the two layers of entity preservation and why the deterministic layer exists even though you also instruct the model.
5. Precisely how does compaction interact with prompt caching, and what does `cache_read_input_tokens` do on the compaction turn vs. the turns after?
6. Sketch the proving test. What does it snapshot, what does it assert, and why must it use exact string membership rather than fuzzy match?

### Answer key

1. **Cost:** each compaction is a full summarization call (thousands of input tokens + output tokens); firing it every turn pays that tax constantly and re-compresses barely-changed history. On a threshold you amortize one compaction across many free turns. **Caching:** compaction rewrites the prefix and invalidates the prompt cache; doing it every turn busts the cache every turn, so you never get cache-read savings. A threshold makes invalidation rare.

2. The rule: **entities, IDs, amounts, and dates must be preserved verbatim — character-for-character.** "Preserves the meaning" is too weak because `$4,412.50 → $4,400` and `2024-09-17 → Sept 17` both keep the rough meaning yet break downstream steps needing the exact string (reconciliation math, machine date parsing, ID lookups). Exact strings, not semantics.

3. **Fold the oldest turns; keep the most recent (last ~4–6) raw.** Old turns matter mostly for their facts, preserved by the verbatim Facts block under compression. Recent turns matter for their nuance — the model is actively reasoning over them — so paraphrasing them directly hurts quality. Compressing recent and keeping old raw would be backwards.

4. Layer 1: **prompt the summarizer** to copy every ID/amount/date verbatim while paraphrasing prose. Layer 2: **deterministically extract** those entities with regex/NER before summarizing and inject them as a literal Facts block the model never rewrites. The deterministic layer exists because LLMs are unreliable at verbatim copying — they normalize IDs and round numbers occasionally — so the Facts block is a lossless safety net that survives even when the model's prose mangles an identifier.

5. Caching is a **prefix match**; compaction **rewrites the front of the history** (replacing old turns with a summary), so it changes the cached prefix bytes and invalidates the cache for everything after the rewrite point. On the compaction turn, `cache_read_input_tokens` drops to (near) zero — you pay full cold-input price. On subsequent turns, the new prefix (summary + recent turns) is stable again, re-caches, and `cache_read_input_tokens` recovers. Mitigation: compact rarely and keep truly stable content (system, tools) above the compaction boundary.

6. The test **snapshots the set of every entity present in the history before compaction** (via `extract_entities` over the serialized turns), runs `compact()`, extracts entities from the compacted result, and **asserts the before-set minus the after-set is empty** (100% survival), failing loudly with the list of dropped entities. It must use **exact string membership** because the whole risk is subtle mutation — rounding `$4,412.50`, reformatting a date, normalizing an ID. A fuzzy match would call `$4,400` "close enough" and pass, hiding exactly the failure the test exists to catch.
