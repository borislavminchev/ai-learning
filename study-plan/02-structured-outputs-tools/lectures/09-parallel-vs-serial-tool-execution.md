# Lecture 9: Parallel vs Serial Tool Execution

> A single model turn can hand you five tool requests at once. Run them one after another and you turn a 200 ms task into a full second for no reason; run them concurrently and you get the win for free — *if* they're genuinely independent. But the moment one call needs another's output, concurrency stops being an optimization and becomes a correctness bug: you'd be asking the model to convert a currency it hasn't looked up yet. This lecture draws the exact line between "fan these out with `asyncio.gather`" and "these must serialize across turns," shows how the tool loop serializes dependent work *by itself*, and drills the two disciplines that keep a concurrent batch honest: strict **id matching** and **partial-failure handling**. After this you'll write a `gather`-based executor, prove concurrency from your own latency logs, and explain why over-parallelizing is a data-freshness bug, not a speed knob.

**Prerequisites:** Lecture 6 (the universal tool loop, id-linkage, `execute()`), Lecture 7 (provider wire formats: OpenAI's string args vs Anthropic's parsed `input`), Python `asyncio` basics (`async def`, `await`, `gather`) · **Reading time:** ~24 min · **Part of:** Structured Outputs & Tool Calling, Week 2

---

## The core idea (plain language)

Within a *single* model turn, the model may request more than one tool call. OpenAI calls this "parallel tool calling"; Anthropic returns multiple `tool_use` blocks in one assistant message — same idea. Those calls all arrived together, which means the model chose all of them **before seeing any of their results**. That single fact is the whole test for whether you can run them concurrently:

> **If the model requested all the calls in one turn, they are — by construction — independent of each other's *results*. You may run them concurrently.**

Why is that true? Because the model had no result to depend on. It looked at the conversation, decided "to answer this I need the weather in Paris *and* the weather in Tokyo," and emitted both requests in the same breath. Neither request's arguments could have been computed from the other's output, because the other's output didn't exist yet. So you can fire both, wait for both, and return both — order doesn't matter.

The opposite case is where a call's *arguments* can only be filled in once you know a previous call's *result*. "Find Acme's invoice total, then convert that total to euros." The converter needs the number `1200.00`, which comes out of the search. The model literally cannot write `convert_currency(amount=1200.00, ...)` until it has seen `1200.00` in its context. So it doesn't. It requests the search, you run it, you feed the result back, *then* on the next turn it requests the conversion. The dependency serializes **across turns** — and you didn't have to plan for it. The loop's turn-by-turn structure does the sequencing automatically.

Two disciplines make the parallel case safe, and both extend Lecture 6:

- **Match results to requests by id, always.** Concurrency reorders completions: the call you started second may finish first. Match by position and you'll hand Tokyo's temperature to Paris's request. Join on id, like a foreign key.
- **A batch is not all-or-nothing.** If four of five calls succeed and one fails, you return four results plus one per-id error. You do **not** fail the whole turn. The model sees the failure inline and adapts.

---

## How it actually works (mechanism, from first principles)

### The independence guarantee is structural, not a heuristic

You never have to *guess* whether same-turn calls are independent. The turn boundary is the proof. Picture the model's context at the instant it emits requests:

```
turn 1 context:  [system][user: "weather in Paris and Tokyo?"]
model emits:     tool_call c1 = get_weather("Paris")
                 tool_call c2 = get_weather("Tokyo")
                        ▲                    ▲
        both chosen from the SAME context — neither could see
        the other's result, because no result exists yet.
```

There is no arrow from `c1`'s result into `c2`'s arguments, because `c1` hadn't run when `c2` was written. That is why "requested in one turn" is a hard guarantee of result-independence, not a probabilistic hint. Contrast the dependent case, which *cannot* appear in one turn:

```
turn 1:  model emits  c1 = search_invoices("Acme")     → you run it → 1200.00
turn 2:  model (now sees 1200.00) emits
                       c2 = convert_currency(1200.00, "USD", "EUR")
                                              ▲
             this argument was COPIED from c1's result, which
             only existed after turn 1. It could not have been
             written in turn 1.
```

The model self-serializes dependent work because it is an autoregressive text predictor: it can only reference a value that is already in its context window. The dependency shows up as *turns*, and each turn is one loop iteration.

### Concurrent execution with `asyncio.gather`

When you get a batch of same-turn calls, you want them running at the same time, not queued. In Python the idiom is `asyncio.gather`, which schedules a list of coroutines and awaits them all. The key property: while one call is blocked on network I/O (a tool that hits an API or DB), the event loop runs the others. Total wall-clock time collapses from *sum of latencies* to *max of latencies*.

```python
import asyncio, time

async def execute_one(call, registry):
    """Run a single tool call; never raises — errors become results."""
    t0 = time.perf_counter()
    tool = registry.get(call.name)
    if tool is None:
        return ToolResult(call.id, "ERROR: unknown tool", latency_ms=0.0)
    try:
        args = tool.args_model.model_validate(call.args)      # validate FIRST
        output = await tool.afn(**args.model_dump())          # <-- the I/O call
        content = str(output)
    except ValidationError as e:
        content = f"ERROR: invalid arguments: {e}"
    except Exception as e:                                    # tool blew up
        content = f"ERROR: {type(e).__name__}: {e}"
    dt = (time.perf_counter() - t0) * 1000
    return ToolResult(call.id, content, latency_ms=dt)

async def execute(calls, tools):
    registry = {t.name: t for t in tools}
    batch_t0 = time.perf_counter()
    # fan out: all calls start (nearly) simultaneously
    results = await asyncio.gather(*(execute_one(c, registry) for c in calls))
    batch_ms = (time.perf_counter() - batch_t0) * 1000
    for r in results:
        log.info("call id=%s latency=%.1fms", r.id, r.latency_ms)
    log.info("batch wall=%.1fms  sum-of-calls=%.1fms",
             batch_ms, sum(r.latency_ms for r in results))
    return results     # order matches `calls`, but we still match by id downstream
```

Three things to notice. **(1)** `execute_one` never raises — every failure path returns a `ToolResult` carrying an `ERROR:` string. That is what makes partial failure work (below). **(2)** `gather` returns results in the *order of the input coroutines*, not completion order, so `results[i]` corresponds to `calls[i]` — but you should still render by id, because the instant you pass `return_exceptions=True` or switch to `as_completed`, that alignment breaks. **(3)** The two log lines are your concurrency proof: if `batch_ms` is close to the *max* single-call latency rather than the *sum*, you ran concurrently. If it equals the sum, you're accidentally serial — the classic bug is `await`-ing inside the loop instead of gathering.

### Proving concurrency with the numbers

Say the model requests three lookups with tool latencies of 180 ms, 210 ms, and 90 ms.

```
serial (await one at a time):   180 + 210 + 90  = 480 ms
concurrent (asyncio.gather):    max(180,210,90) ≈ 210 ms   (+ tiny scheduling overhead)
speedup:                        480 / 210       ≈ 2.3×
```

The concurrent batch finishes when the *slowest* call finishes; the fast ones overlap under it. Your log would read:

```
call id=c1 latency=181.4ms
call id=c2 latency=209.8ms
call id=c3 latency=90.2ms
batch wall=212.1ms  sum-of-calls=481.4ms
```

`batch wall ≈ max, not sum` — that line *is* the evidence, and it's exactly what the lab's Definition of Done asks you to show. A crucial caveat: this holds only for **I/O-bound** tools (network, disk). If a tool is CPU-bound pure Python, `asyncio` won't parallelize it — the GIL serializes CPU work on one thread, so your "concurrent" batch runs serially anyway. CPU-bound tools need `run_in_executor` (a thread or process pool). For the lab's API-call tools, plain `gather` is correct.

### Why over-parallelizing dependent calls is a correctness bug

Suppose you got clever and tried to "help" the model by pre-issuing the conversion in parallel with the search. You can't — the model didn't request it, and you don't have the amount. The only way to force a dependent call into a parallel batch is to *fabricate* its arguments (guess the amount) or issue it with a placeholder. Both are wrong:

- **Stale/absent input.** The converter runs on a number the search hasn't produced. Best case you get `convert_currency(amount=None)` → a validation error. Worst case you carried over a stale amount from an earlier request and silently convert the wrong figure. The model then reports a confident, wrong euro total, with no exception anywhere.
- **You've broken the model's reasoning premise.** The whole reason the model split the work across turns is that it needed to *see* the intermediate result to decide the next step — maybe the search returns "no invoice found," in which case the conversion should never happen at all. Parallelizing forces the second step before the first step's outcome is known, defeating conditional logic.

The rule is blunt: **never manufacture a call the model didn't request in the current turn.** If the model wanted it now, it would have asked now. The loop's serialization is not a limitation to route around; it is the mechanism that keeps dependent steps causally ordered.

### Partial failure: return, don't abort

A parallel batch is a set of independent bets. If one loses, the others still won. Failing the whole turn because one call errored throws away good work and forces a full re-do. Instead, each failed call returns its own `ERROR:` result linked to its id, and the model — seeing three successes and one failure inline — decides what to do (proceed with what it has, retry the failed one next turn, or tell the user). This is Lecture 6's "errors are data" rule applied to a batch, and `execute_one` already implements it: every path returns a `ToolResult`.

```
model requested: [c1 weather-Paris, c2 weather-Tokyo, c3 weather-Mars]
results returned: c1 -> "18°C"          (ok)
                  c2 -> "24°C"          (ok)
                  c3 -> "ERROR: unknown location"   (failed, but linked to c3)
next turn: model has 2 good answers + 1 explicit failure, and can adapt.
```

Contrast the wrong design — `asyncio.gather(...)` *without* `return_exceptions=True` and *without* the try/except inside each coroutine: the first exception propagates out of `gather`, cancels the sibling coroutines, and blows up your whole `execute()`. You lose the two good results and crash the loop over one bad location. The fix is structural: catch inside `execute_one` so nothing ever escapes, which makes the `gather` call inherently partial-failure-safe.

---

## Worked example

Task: *"Compare the current USD→EUR and USD→GBP rates, and tell me Acme's invoice total in whichever is stronger."* Tools: `get_rate(base, quote)` and `search_invoices(vendor)`.

**Turn 1 — model emits three calls in one assistant message:**
```
tool_call r1 = get_rate("USD","EUR")
tool_call r2 = get_rate("USD","GBP")
tool_call s1 = search_invoices("Acme")
```
All three came from the same context — none depends on another's result. The two rate lookups are obviously independent; the invoice search is independent too (it needs no rate). **Fan out with `gather`.**

**Your `execute()` runs all three concurrently.** Latencies: `r1`=140 ms, `r2`=155 ms, `s1`=95 ms.
```
serial would be    140 + 155 + 95 = 390 ms
concurrent batch = max(140,155,95) ≈ 156 ms      → ~2.5× faster
```
You return three results, each linked by id:
```
r1 -> {"rate": 0.92}
r2 -> {"rate": 0.79}
s1 -> {"total_usd": 1200.00}
```

**Turn 2 — model now has all three values in context and reasons.** Which is "stronger per USD"? `0.79 < 0.92` means one USD buys fewer GBP, so GBP is stronger. It needs `1200.00 × 0.79`. It could compute that itself, or request one more call:
```
tool_call c4 = convert_currency(amount=1200.00, rate=0.79)
```
Note `c4` **could not have been in Turn 1** — its `amount` (1200.00) came from `s1`'s result and its `rate` (0.79) came from `r2`'s result. It's a dependent call, so it appears only after Turn 1's results are in context. It serialized itself.

**Your code runs `c4`** → `948.00`, returns it linked to `c4`.

**Turn 3 — model answers:** *"Acme's total is £948.00. GBP (0.79) is stronger per USD than EUR (0.92)."* No tool calls → you return the final text.

Tally: **one parallel batch of 3** (saving ~230 ms over serial) followed by **one serial dependent call**, across 3 model turns. The parallelism and the serialization coexist in the same task, and you wrote no special-case code for either — `execute()` gathers whatever it's handed, and the loop supplies dependent calls one turn later because that's when the model asks.

Now the partial-failure twist: suppose `get_rate("USD","GBP")` times out. `execute_one` catches it and returns `r2 -> "ERROR: upstream timeout"`. Turn 2 opens with EUR=0.92, the Acme total, and an explicit GBP failure. The model can retry `get_rate("USD","GBP")` (a fresh call next turn), fall back to EUR, or say it couldn't fetch GBP — all without your loop crashing. Two good results were preserved.

---

## How it shows up in production

- **Latency is the headline win, bounded by your slowest tool.** Parallelizing N independent calls takes you from `sum` to `max` of their latencies. A dashboard firing 6 independent lookups at ~200 ms each drops from ~1.2 s to ~0.2 s — the difference between "sluggish" and "snappy." But note the ceiling: one slow tool (a 3 s report query) drags the whole batch to 3 s no matter how fast the others are. Profile per-tool latency; the batch is only as fast as its tail.
- **The id-matching bug is silent and reorders under load.** In dev, with warm caches, results often come back in request order and positional matching *appears* to work. In production, under variable latency, completions reorder and you start serving swapped answers with zero errors. This is the single most dangerous parallel-execution bug precisely because it hides in testing. Always join on id; write a test that forces reordered completions (a fast call after a slow one) and asserts correct pairing.
- **Partial-failure design decides your tail behavior.** A batch-abort design means one flaky tool fails the entire user request; a partial-return design means the user still gets 4 of 5 answers and a note about the fifth. Under real dependency failures (rate limits, timeouts), that's a resilient product vs. one that's down whenever any dependency hiccups.
- **CPU-bound "concurrency" that isn't.** A frequent surprise: someone wraps a CPU-heavy tool (parsing a big PDF, a tight numeric loop) in `async` and `gather`s it, sees no speedup, and is confused. The GIL serialized it. If a tool burns CPU, it belongs in `run_in_executor` with a thread/process pool, not naked `asyncio`. Know which tools are I/O-bound (most API calls) vs CPU-bound.
- **Cost is unchanged by parallelism — only wall-clock is.** Running 3 calls concurrently costs the same tokens/API-fees as running them serially; you pay for the same 3 calls either way. Parallelism buys latency, not money. (What *does* cost money is extra *turns* — see Lecture 6's super-linear step cost.)
- **Tracing must be per-call.** Log each call's id, name, args, result, and latency, plus the batch wall-clock. When a batch is slow or an answer is wrong, the trace tells you which call was the tail and whether ids matched. Same JSONL trace Lecture 6 asked for, now with per-call latency inside a turn.

---

## Common misconceptions & failure modes

- **"I need to analyze the calls to decide if they're independent."** No. If they arrived in the same turn, they're result-independent by construction — run them concurrently. You never inspect argument contents to decide.
- **"I should parallelize the dependent call to speed things up."** You can't, and trying corrupts data. A dependent call's arguments come from a prior result; forcing it early means feeding stale or absent input. Let the loop serialize it across turns.
- **"`asyncio.gather` preserves completion order."** It preserves *input* order in its return list, but coroutines finish in whatever order I/O completes. The safety net is id-matching, not `gather`'s ordering — the instant you switch to `as_completed` or `return_exceptions`, positional assumptions break.
- **"One call failed, so the turn failed."** Wrong for a batch. Return the successes plus a per-id error; the model adapts. Abort only on *your* invariants (max steps, budget), never on a single tool's error.
- **"Bare `gather` is partial-failure-safe."** Only if each coroutine catches its own exceptions. A raw `gather` propagates the first exception and cancels siblings — you lose good results. Catch inside each `execute_one`.
- **"`async` makes my CPU-bound tool parallel."** No — the GIL serializes CPU work. `async` parallelizes *waiting*, not *computing*. Offload CPU tools to an executor.
- **"Parallel calls save money."** No — same number of calls, same tokens/fees. Parallelism saves latency only.

---

## Rules of thumb / cheat sheet

- **Same turn → independent → parallel.** Multiple calls in one model turn are result-independent by construction; `asyncio.gather` them.
- **Depends on a prior result → serial across turns.** Never manufacture a call the model didn't request this turn. The loop serializes dependencies for you.
- **Prove concurrency:** log per-call latency + batch wall-clock. `batch ≈ max(calls)` means concurrent; `batch ≈ sum(calls)` means accidentally serial.
- **Match results to requests by id, never by position.** Concurrency reorders completions. Build `dict[id -> result]`.
- **Catch inside each coroutine.** Every call returns a `ToolResult` (ok or `ERROR:`), so one failure never aborts the batch or crashes the loop.
- **Partial results beat no results.** Return the successes + per-id errors; let the model adapt.
- **I/O-bound → `gather`. CPU-bound → `run_in_executor`.** `async` overlaps waiting, not computing (GIL).
- **Parallelism buys latency, not cost.** Same calls, same tokens; only wall-clock shrinks. Extra *turns* cost money — parallel *calls* don't.
- **The batch is as fast as its slowest call.** Profile the tail.

---

## Connect to the lab

This lecture is **Week 2, Lab step 2** (`structio/toolloop/`). Make `execute()` run independent calls with `asyncio.gather` and log per-call latency plus batch wall-clock — your Definition-of-Done proof of concurrency is `batch ≈ max(latencies)`, not the sum. Then add the dependent pair (a lookup whose output feeds a second lookup, like the worked example's search → convert) and show your loop **naturally serializes** it across turns with no special-casing. Finally, wire partial-failure handling: make one tool fail in a batch and assert the other results still return with a per-id error, and that ids match under reordered completions. This feeds directly into the hardened `POST /extract` service in step 3.

---

## Going deeper (optional)

Real, named resources — verify current URLs yourself; providers move docs around.

- **Python docs — `asyncio` "Coroutines and Tasks"** (docs.python.org). Read the `gather`, `as_completed`, and `return_exceptions` sections; they define exactly the ordering and cancellation semantics this lecture depends on.
- **OpenAI — "Function calling" guide**, the *parallel tool calls* section (platform.openai.com/docs). How the API returns multiple `tool_calls` in one message and the `parallel_tool_calls` flag.
- **Anthropic — "Tool use" documentation** (docs.claude.com). Multiple `tool_use` blocks per assistant message and how to return multiple `tool_result` blocks in one user turn.
- **"ReAct: Synergizing Reasoning and Acting in Language Models"** — the reason→act→observe pattern that explains *why* dependent steps serialize across turns. Search: "ReAct reasoning acting language models paper".
- **David Beazley — "Python Concurrency From the Ground Up"** (PyCon talk). The clearest explanation of what `async` does and does not parallelize, and why the GIL matters for CPU-bound work. Search the title.
- **Search queries:** "asyncio gather return_exceptions partial failure", "OpenAI parallel tool calls", "GIL asyncio CPU-bound run_in_executor".

---

## Check yourself

1. The model returns three tool calls in one turn. Without reading their arguments, you conclude they can run concurrently. What structural fact justifies that conclusion, and why does it *not* hold for a call the model requests on a later turn?
2. Your latency log shows `batch wall=478ms` with individual calls at 180/210/90 ms. Concurrent or accidentally serial? What's the most likely code bug, and what would the number look like if it were fixed?
3. A teammate proposes issuing the dependent `convert_currency` call in parallel with the `search_invoices` it depends on, "to save a turn." Explain concretely what goes wrong, and why the loop's across-turn serialization is correct behavior, not a limitation.
4. In a batch of five calls, one raises an exception. Describe the difference in outcome between (a) a bare `asyncio.gather` and (b) `execute_one` that catches internally. Which does the model see, and what can it do next?
5. Two calls run concurrently; the one you started second finishes first. Why is matching results by list position dangerous here, and what's the fix? Why might this bug pass every test in dev and only surface in production?
6. A colleague wraps a CPU-heavy PDF-parsing tool in `async def` and `gather`s a batch of them, expecting a speedup, and sees none. Explain why, and state the correct fix.

### Answer key

1. All three were chosen from the **same context, before any of their results existed** — so no call's arguments could have been computed from another's output. Result-independence is guaranteed by the turn boundary, not inferred from the arguments. It fails for a later-turn call because by then prior results are in context, so that call's arguments *may* have been copied from one (a dependency); the model only requests it *after* seeing the value it needs.
2. **Accidentally serial** — 478 ms ≈ 180+210+90 (the sum), not max(=210). The likely bug is `await`-ing each call inside a loop instead of collecting coroutines and `gather`-ing them once. Fixed, the batch would read ≈ 210 ms (the max single-call latency plus tiny scheduling overhead).
3. You don't have the amount to convert — it comes out of the search, which hasn't run — so you'd feed the converter stale or absent input (`amount=None` → validation error, or a leftover figure → a silently wrong euro total). You'd also break conditional logic: if the search returns "no invoice," the conversion should never happen. Across-turn serialization is correct because the model deliberately deferred the dependent call until it could see the value it needs; that ordering preserves causality.
4. (a) Bare `gather` propagates the first exception, cancels the sibling coroutines, and raises out of `execute()` — you lose the four good results and crash the loop. (b) `execute_one` catches internally and returns an `ERROR:` `ToolResult` for the failed call; the model sees four successes plus one explicit per-id failure and can retry it, proceed, or report it. (b) is the production-correct design.
5. `asyncio.gather` returns results in *input* order, but if you use `as_completed`/`return_exceptions` or otherwise rely on completion timing, position no longer identifies the call — the second-started, first-finished result lands where you expected the first call's answer, and you serve swapped data with no error. Fix: match by **id** (`dict[id -> result]`), joining like a foreign key. It passes in dev because warm caches and low variance make completions arrive in request order; production latency variance reorders them and exposes the bug.
6. `async`/`gather` only overlaps **I/O waiting**, not **CPU work** — the GIL serializes Python bytecode execution on one thread, so CPU-bound parsing runs one-at-a-time even under `gather`. The fix is to offload each parse to a thread or process pool via `loop.run_in_executor` (a process pool for true CPU parallelism, since threads still contend on the GIL for pure-Python CPU work).
