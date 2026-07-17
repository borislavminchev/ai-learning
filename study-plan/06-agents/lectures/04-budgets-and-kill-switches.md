# Lecture 4: Bounded Loops, Budgets & Kill Switches (Defense in Depth)

> An agent is a `while` loop wrapped around a model that calls tools. The loop is what makes agents powerful — and it is also what makes them dangerous, because a loop has no natural end. A prompt costs one round-trip; a loop costs *however many round-trips it decides to take*, and if a tool keeps failing in a way the model keeps trying to "fix," that number is effectively infinite. This lecture is about the four independent budgets and the one human override that turn an open-ended loop into something you can safely leave running unattended. After it you will be able to place each check at the correct point in the loop, say exactly which failure mode each one — and *only* that one — catches, and explain to your finance team why "we set a max of 20 steps" is not a cost control.

**Prerequisites:** the perceive→plan→act→observe loop (Lecture 1); native tool calling and `tool_use`/`tool_result` message shape (Lecture 2); errors-as-observations (Lecture 3); basic per-token pricing. · **Reading time:** ~22 min · **Part of:** AI Agents & Agentic Systems (Expanded Deep Track), Week 1

## The core idea (plain language)

Here is the failure that motivates everything below. You ship an agent on Friday. Its job is to answer support questions using a `search_docs` tool. Over the weekend, the docs service returns a 500 on one particular query shape. Your tool, being well-behaved, catches the exception and returns `ERROR: upstream 500` as an observation (good — errors-as-observations, Lecture 3). The model reads that, reasons "the search failed, let me rephrase and try again," and calls the tool again. Same 500. Rephrase. Same 500. There is no exception, no crash, no alert — every iteration looks *healthy*. The loop just spins. Monday morning you have a $4,000 bill and a trace file with 900,000 near-identical steps.

Nothing in that story is a bug in the usual sense. The tool worked. The model worked. The loop worked. The problem is that **a loop with no bound will consume whatever resource is cheapest to consume until something external stops it** — and if nothing external is watching, "something external" is your credit card limit or a rate-limit ceiling, discovered hours too late.

The fix is not "add a step limit." The fix is **defense in depth**: several *independent* limits, each watching a different resource, each catching a failure mode the others structurally cannot. A step cap counts iterations. It knows nothing about how long each iteration took, how many tokens each one carried, or how many dollars they cost. Those are four different quantities that go bad in four different ways, so you need four different sensors. Plus one more thing that no automated sensor can replace: a **kill switch** — a signal a *human* can flip to stop a run right now, for a reason no budget anticipated ("the customer called, cancel it," "we're deploying, drain the agents," "this output looks wrong, stop before it emails anyone").

Think of it exactly like a physical machine. A drill press has a thermal cutoff (temperature), a circuit breaker (current), a torque limiter (mechanical load), and a big red mushroom button (a human who sees something the sensors don't). You would never argue "the thermal cutoff is enough." Each guard catches a distinct way the machine can hurt someone. Your agent loop is the same machine.

## How it actually works (mechanism, from first principles)

### Why the step cap alone is a trap

Start with the naive control: `for step in range(max_steps):`. Say `max_steps = 20`. Feels safe. Now walk the arithmetic of the weekend failure with real numbers.

Suppose each iteration sends the growing transcript to the model. Because ReAct re-sends the *entire* history every turn, the input token count grows each step. Rough model: step *k* carries the system prompt + tool schemas (~1,500 tokens fixed) plus all prior turns. If each turn adds ~600 tokens (the model's reasoning + the tool call + the observation), then by step *k* the input is roughly `1500 + 600·k` tokens, and the output is ~400 tokens. Price it at Opus-class rates — `$5 / 1M` input, `$25 / 1M` output:

| step | input tok | output tok | step cost | cumulative $ |
|---|---|---|---|---|
| 1 | 2,100 | 400 | $0.0205 | $0.021 |
| 5 | 4,500 | 400 | $0.0325 | $0.14 |
| 10 | 7,500 | 400 | $0.0475 | $0.34 |
| 20 | 13,500 | 400 | $0.0775 | $1.04 |

So a 20-step cap on *this* task bounds you to about **$1 per run**. Fine. But notice what the step cap is actually promising: "no more than 20 iterations." It says nothing about *cost per iteration*. Now change one variable — the transcript grows faster because the failing tool returns a 4,000-token error page each time instead of a one-line string:

| step | input tok | output tok | step cost | cumulative $ |
|---|---|---|---|---|
| 1 | 2,100 | 400 | $0.021 | $0.021 |
| 10 | 45,000 | 400 | $0.235 | $1.3 |
| 20 | 90,000 | 400 | $0.460 | $4.9 |

Same 20 steps. Nearly **5x the cost**, because the step cap does not watch tokens. And if you had set `max_steps = 200` "to be safe for hard tasks," the same runaway is now a $200+ run — still "within budget" by the only guard you had. **This is the load-bearing point of the whole lecture: a slow or looping tool blows the dollar budget long before it reaches the step cap.** Counting iterations is not counting money, time, or context. You cannot substitute one sensor for another.

### The four budgets, and the one failure each catches

Each budget watches exactly one resource and catches exactly one class of failure. The discipline is that they are *independent* — no one of them subsumes another, which is precisely why you need all four.

```
                        the loop's four sensors
   ┌─────────────────────────────────────────────────────────────┐
   │ max_steps      → OSCILLATION / LOOPS                         │
   │                  "the model keeps trying the same thing"     │
   │ wall_clock     → SLOW or HUNG TOOLS                          │
   │                  "one tool call is taking forever"           │
   │ token_budget   → CONTEXT BLOAT                               │
   │                  "the transcript grew past what we'll pay to │
   │                   re-send every turn"                        │
   │ dollar_budget  → TOTAL SPEND (the one finance cares about)   │
   │                  "cumulative in/out tokens × price crossed $X"│
   └─────────────────────────────────────────────────────────────┘
```

**1. Max-steps → oscillation / loops.** This catches the model getting *stuck in a behavioral cycle*: call tool A, get result, call tool A again with the same args, forever; or ping-pong A→B→A→B making no progress. Steps are the natural unit of "am I going in circles?" Note what it does *not* catch: a single expensive step. Ten steps could cost ten cents or ten dollars — max-steps can't tell.

**2. Wall-clock → slow or hung tools.** This is the *only* budget measured in seconds of real time, and it is the only one that fires while the loop is *blocked inside a tool call*. Imagine `web_fetch` hits a URL that accepts the connection but never sends a body. Your step counter is frozen at, say, 3 — you're stuck *inside* iteration 3. Token count is frozen. Dollar count is frozen. Every token/step/dollar sensor is asleep because nothing is happening *to them*. Only wall-clock keeps moving, because it reads the system clock, not the loop's own counters. (Caveat, covered under production below: to actually interrupt a hung call you also need a per-request timeout on the HTTP/SDK client; the wall-clock *budget check at the top of the loop* only fires once control returns to you.)

**3. Token budget → context bloat.** This catches the transcript growing past the size you're willing to re-send every turn. Context bloat has two flavors: benign (a genuinely long task that accumulated a lot of legitimate history) and pathological (a tool that keeps returning huge payloads, or the model pasting large chunks back and forth). Either way, once cumulative tokens cross a line, each *future* step is expensive, and you may be approaching the model's hard context-window limit (a 400 "context length exceeded" the model can't recover from). The token budget stops you on *your* terms, with a clean reason, before the provider stops you on its terms with an error.

**4. Dollar budget → total spend.** This is the one your finance team asked for, and it is the only guard denominated in the unit they care about. It is *computed*, not measured directly: after each model call you accumulate input and output tokens and multiply by the per-token price:

```
dollars = cumulative_input_tokens  × price_in
        + cumulative_output_tokens × price_out
```

Why keep it separate from the token budget when it's derived from tokens? Because input and output tokens have **different prices** (output is typically 4–5x input), so 50,000 tokens can cost wildly different amounts depending on the in/out split — and because the dollar budget is the number a human set based on business value ("this task is worth at most $0.25 to answer"), not a number about context size. The token budget protects the *model*; the dollar budget protects the *P&L*. They diverge, so they're separate sensors.

### Where the checks go: top of the loop, before you spend

All four checks belong at the **very top of each iteration, before the model call**, and each returns a *distinct* `ABORTED: <reason>` string. Placement matters: you check *before* spending, so a tripped budget prevents the next expensive call rather than merely noticing it after the fact.

```python
def run(task, max_steps=8, max_seconds=60, max_tokens_budget=50_000, max_dollars=0.25):
    start, tok_in, tok_out, dollars = time.time(), 0, 0, 0.0
    messages = [{"role": "user", "content": task}]

    for step in range(1, max_steps + 1):
        # ---- GUARDS: top of loop, before we spend anything ----
        if pathlib.Path("KILL").exists():          # KILL SWITCH (human, out-of-band)
            return "ABORTED: kill switch"
        if time.time() - start > max_seconds:      # WALL-CLOCK  → slow/hung tools
            return "ABORTED: wall-clock budget"
        if tok_in + tok_out > max_tokens_budget:   # TOKEN       → context bloat
            return "ABORTED: token budget"
        if dollars > max_dollars:                  # DOLLAR      → total spend
            return "ABORTED: dollar budget"

        resp = client.messages.create(model=MODEL, max_tokens=1024,
                                      tools=TOOLS, messages=messages)

        tok_in  += resp.usage.input_tokens
        tok_out += resp.usage.output_tokens
        dollars  = tok_in * PRICE_IN + tok_out * PRICE_OUT   # recompute cumulative

        # ... handle tool_use / final answer ...

    return "ABORTED: max-steps budget"             # MAX-STEPS → oscillation/loops
```

Two structural details worth internalizing. First, **max-steps is enforced by the `for` bound itself** — falling off the end of the loop *is* hitting the step cap, so its abort lives after the loop, while the other three are explicit `if` checks at the top. Second, **the distinct reason strings are not cosmetic.** When you read `trace.jsonl` at 9am Monday, `ABORTED: dollar budget` vs `ABORTED: wall-clock budget` tells you *instantly* whether you had a cost runaway or a hung tool — two completely different investigations. A single generic `ABORTED` would force you to reconstruct which limit fired from raw numbers. Distinct reasons are a debugging affordance; treat the string as an API.

### The kill switch: the human sensor

The four budgets are automated: they fire on conditions you predicted and encoded. The kill switch is for the conditions you *didn't* predict. It is an **out-of-band signal a human can flip mid-run** to end the run right now, with a clear reason.

In this week's lab it's the simplest possible thing: a file named `KILL`. The loop checks `pathlib.Path("KILL").exists()` at the top of every iteration; a human runs `touch KILL` in another terminal; on the next iteration boundary the loop sees it and returns `ABORTED: kill switch`. "Out-of-band" means the signal travels on a *different channel* than the agent's own logic — the agent cannot argue with a file, cannot reason its way past it, cannot decide the task is important enough to ignore it. That independence is the whole point.

Why does a human override deserve a slot next to four automated budgets? Because budgets encode *what you knew to worry about in advance*. The kill switch handles the open set: "the output is drifting somewhere harmful," "we're mid-incident and need to stop all agents," "a customer withdrew consent," "someone noticed the agent is about to send 400 emails and that's wrong." No budget catches "this is wrong for a reason we didn't think of." A person watching the trace does. The kill switch is how their judgment reaches into a running loop.

## Worked example

Let's run the motivating failure with numbers and watch which guard fires — and prove that the step cap alone would *not* have saved you.

**Scenario.** A research agent, budgets set to `max_steps=25`, `max_seconds=120`, `max_tokens_budget=60_000`, `max_dollars=0.50`. Prices: `$5/1M` in, `$25/1M` out. Its `web_fetch` tool starts hitting a page that returns a 3,500-token error body every time. The model keeps retrying with slight rephrasings — classic oscillation, but with a *fat* observation each time.

Per step: input = `1,500 (fixed) + 3,900·(step−1)` (each prior turn adds ~3,900 tokens: ~400 reasoning + ~3,500 error body), output ≈ 400.

| step | input tok | cumulative in | cumulative out | dollars | which guard is closest? |
|---|---|---|---|---|---|
| 1 | 1,500 | 1,500 | 400 | $0.018 | — |
| 4 | 13,200 | 29,400 | 1,600 | $0.187 | — |
| 6 | 21,000 | 71,x00 | 2,400 | — | **token budget about to trip** |

By the top of **step 6**, cumulative tokens ≈ `1,500+5,400+9,300+13,200+17,100 = ...` — the point is it crosses **60,000 tokens** around step 6, so the guard check at the top of step 6 returns `ABORTED: token budget`. Cumulative dollars at that moment ≈ $0.19 — *under* the $0.50 dollar cap. The token guard fired first here because the fat error bodies bloat context faster than they bloat dollars.

Now the counterfactual that proves defense-in-depth. **Delete the token, wall-clock, and dollar guards; keep only `max_steps=25`.** The loop happily runs to step 25. By step 25 cumulative input is on the order of `1,500·25 + 3,900·(0+1+...+24) ≈ 37,500 + 1,170,000 ≈ 1.2M` input tokens plus ~10,000 output → cost ≈ `1.2M × $5/1M + 10k × $25/1M ≈ $6.25`. So the *same runaway* costs **~$6.25 under a step-only cap versus ~$0.19 with the token guard** — a 30x difference — and if some engineer had bumped `max_steps` to 200 for "hard tasks," it's a ~$400 run. The step cap did its job (it *did* stop at 25); it's just that its job has nothing to do with cost. That's why you don't get to pick one.

And the wall-clock guard? In this scenario it never fires because each step returns quickly — the failure is *many fast steps*, not one slow one. Flip the failure to *one hung fetch* (a URL that never responds) and now token/step/dollar all freeze while wall-clock is the *only* sensor still counting up toward 120s. Different failure, different guard. Neither can cover for the other.

## How it shows up in production

- **The overnight bill.** The canonical incident: an agent loops on a failing tool over a weekend and burns four or five figures. Post-mortems almost always find a step cap was present and a dollar cap was not. The step cap "worked" and the money still left. Set the dollar cap to the *business value* of the task, not to some round number — if answering a ticket is worth $0.10, cap near there, and let the cap be the thing that catches your mispriced assumptions.

- **Wall-clock needs teeth at the transport layer.** A subtlety that bites people: the wall-clock *budget check* at the top of the loop only runs when control is back in your loop. If a tool call blocks forever, you never reach the check. So the wall-clock budget has two halves: (1) the top-of-loop elapsed-time check, and (2) a **per-request timeout on the tool's client** (`httpx.get(url, timeout=10)`, an SDK request timeout, a `signal.alarm`/`asyncio.wait_for` wrapper). Without (2), a truly hung socket hangs the whole agent and no guard ever fires. This is why the lab's `web_fetch` has `timeout=10` baked in — the budget and the timeout are partners.

- **Token accounting must be exact, not estimated.** Compute dollars from `resp.usage.input_tokens`/`output_tokens` (the provider's own count), never from a client-side `len(tiktoken.encode(...))` guess. Client-side estimates ignore tool-schema overhead, system prompt, cached-token discounts, and image/tool tokens — and a dollar budget built on a wrong count fails silently in the direction that costs you money. If your provider bills cached input tokens at a discount, fold that into the price arithmetic or you'll over-abort.

- **Guards are per-run; you also need a fleet-level cap.** Four budgets protect one run. A hundred runs each staying under $0.25 is still $25, and a retry storm can spawn thousands. In production the per-run budgets sit under an *account-level* spend alert and rate limit (provider dashboards, a spend-tracking middleware). Week 6 pushes this idea into CI so a change that regresses cost-per-run fails the build before it ships. For now, know that per-run and per-fleet are different scopes.

- **The abort reason is a debugging primitive.** Teams that log a single generic "aborted" spend Monday mornings reverse-engineering which limit fired. Teams that emit `ABORTED: dollar budget` vs `ABORTED: wall-clock budget` vs `ABORTED: max-steps budget` read the answer off the last trace line. Aggregate these reasons across runs and you get a free health dashboard: a spike in `max-steps` aborts means your agents are oscillating (bad tool? bad prompt?); a spike in `wall-clock` aborts means a dependency is slow.

## Common misconceptions & failure modes

- **"A step cap is a cost control."** No. A step cap bounds *iterations*, and cost per iteration is unbounded (context grows, tools return fat payloads, output length varies). The dollar cap is the cost control. This is the single most common and most expensive mistake in the space.

- **"One big budget is simpler than four."** You cannot express "stop if any of {looping, hung, bloated, too expensive}" with one number, because those are four different units (steps, seconds, tokens, dollars). Collapsing them means at least three failure modes go uncaught. Four cheap `if` checks at the top of the loop is *less* code than the incident review of the one you dropped.

- **"The dollar budget makes the token budget redundant."** They correlate but diverge. The token budget protects against *hitting the model's context-window ceiling* (a hard error) and fires on context *size*; the dollar budget fires on context size × price and defends the P&L. A cheap model with a huge window can blow the token budget while barely denting the dollar budget, and vice versa.

- **"Errors-as-observations already keeps me safe."** Errors-as-observations (Lecture 3) prevents *crashes* — the loop survives a failing tool. That's exactly what makes the runaway *possible*: a surviving loop keeps spending. The two disciplines are complementary — error handling keeps you running, budgets keep running from bankrupting you.

- **"Check the kill switch once at startup."** The kill switch must be read **every iteration**, not once. Its entire value is stopping a run *mid-flight*, after a human notices something during the run. A startup-only check is theater.

- **"The agent can manage its own budget."** Never let the model's own reasoning be the thing that decides to stop for cost/time reasons. The guards are *out-of-band* — enforced by the harness, not the model — precisely because a looping model has already demonstrated it can't self-correct. The kill switch is out-of-band for the same reason.

- **Checking budgets after the model call.** If you check *after* spending, you always pay for one extra (possibly the most expensive) call before aborting. Check at the top, before the `create()`.

## Rules of thumb / cheat sheet

- **Always ship all four budgets + a kill switch.** Not a subset. Each catches a failure the others structurally cannot.
- **Map, memorized:** max-steps → oscillation/loops · wall-clock → slow/hung tools · token → context bloat · dollar → total spend.
- **Order of checks at top of loop:** kill switch first (cheapest, human intent wins), then wall-clock, token, dollar. Any hit → `return "ABORTED: <distinct reason>"`.
- **Set the dollar cap from business value**, not a round number. If a task is worth $0.10, don't cap at $5.
- **Wall-clock budget needs a partner:** a per-request timeout on every tool client, or a hung call hangs the whole agent.
- **Compute dollars from `resp.usage`, never client-side token estimates.** Account for the in/out price split (output ≈ 4–5x input) and any cached-token discount.
- **Distinct `ABORTED:` strings** — they're your Monday-morning triage and your health dashboard. Log them.
- **Read the kill switch every iteration.** Mid-flight stop is the whole point.
- **Approximate starting defaults for a laptop-scale tool agent** (tune per task): `max_steps≈8`, `max_seconds≈60`, `max_tokens_budget≈50k`, `max_dollars≈$0.25`. These are illustrative, not universal.
- **Per-run budgets sit under a fleet-level spend alert.** One run behaving ≠ a thousand runs behaving.

## Connect to the lab

This lecture is the theory behind the Week 1 lab's `agent.py` guards. The lab's Definition of Done requires you to **demonstrate all four budgets plus the kill switch, each returning a distinct `ABORTED: ...` reason — five runs, five reasons.** Force each one deliberately (Exercise 1): set `max_steps=1`, then `max_dollars=0.0001`, then `touch KILL` mid-run, then feed a fat-payload tool to trip the token budget, then a slow/hung URL to trip wall-clock. Confirm the returned string matches the guard you meant to fire — that's the proof you wired the sensors to the right resources.

## Going deeper (optional)

- **Anthropic — "Building Effective Agents."** The workflows-vs-agents distinction that frames *why* the model owning control flow is the source of unbounded loops. Search: `Anthropic Building Effective Agents`.
- **OpenAI — "A Practical Guide to Building Agents" (PDF).** Guardrails/safety section covers budgets and human-in-the-loop from the other vendor's framing. Search that exact title.
- **Anthropic Claude API docs — Usage / token counting.** The `usage` object your dollar math depends on; root: `docs.anthropic.com`. Search: `Anthropic API usage input output tokens`.
- **Netflix — "Hystrix" / the circuit-breaker & bulkhead patterns.** The distributed-systems ancestors of defense-in-depth budgeting; the vocabulary (circuit breaker, timeout, bulkhead) maps directly onto agent guards. Search: `circuit breaker pattern Nygard Release It`, and Michael Nygard's book *Release It!*.
- **OpenAI / Anthropic pricing pages** for current per-token prices to plug into the dollar arithmetic — always read the live page, prices move. Search: `Anthropic pricing`, `OpenAI pricing`.
- **Foreshadow:** Week 3 makes the kill switch genuinely *out-of-band* (a checkpointed interrupt / external signal rather than a polled file, so it works across process restarts and remote workers); Week 6 puts the budgets in **CI**, so a change that regresses cost-per-run or step-count fails the build before it ships.

## Check yourself

1. Your agent has `max_steps=20` and no other guard. A tool starts returning a 4,000-token error page and the model retries it every step. The run completes 20 steps and stops. Did the step cap "work"? What did it fail to protect, and roughly why is the cost far higher than for a task that returns one-line observations?
2. Name the four budgets and give the one failure mode each — and only that one — catches. For each, name a failure mode it *cannot* catch.
3. A `web_fetch` call connects to a server that accepts the connection and then never sends any data. Which of the four budgets is the only one still counting up, and why are the other three frozen? What second mechanism (not a top-of-loop check) do you also need for the wall-clock budget to actually stop this?
4. Why keep the token budget and the dollar budget as separate sensors when dollars are computed *from* tokens? Give a concrete case where one trips well before the other.
5. Why must the kill switch be (a) checked every iteration and (b) out-of-band, and what class of situations does it handle that no automated budget can?
6. You're asked to "simplify" the loop down to a single budget for readability. What's your one-sentence argument against it, and which failure modes would silently reappear?

### Answer key

1. **Yes, the step cap did its job — it bounded iterations at 20.** It failed to protect *cost*, because cost per step is unbounded: ReAct re-sends the whole transcript every turn, so a 4,000-token error page compounds — by step *k* the input carries ~`4,000·k` of accumulated junk, so total tokens (and dollars) grow roughly quadratically in steps. One-line observations add ~tens of tokens per step; fat error pages add thousands. Same step count, ~30x the bill. The step cap watches iterations, not money.

2. **max-steps → oscillation/loops** (can't catch a single expensive/slow step). **wall-clock → slow or hung tools** (can't catch many-fast-steps context bloat). **token budget → context bloat** (can't catch a hung tool, since tokens freeze while blocked; can't reason about the in/out price split). **dollar budget → total spend** (can't detect a hang either, and can't tell you *why* you're spending — could be one huge step or many small ones). The theme: each watches one unit; the failure it misses is any failure measured in a different unit.

3. **Only wall-clock** keeps counting, because it reads the system clock. Steps, tokens, and dollars are all frozen — control is blocked *inside* the tool call, so no new iteration, no new `usage`, no new spend is recorded; those counters only advance when the model call returns. The second mechanism is a **per-request timeout on the HTTP/SDK client** (e.g. `httpx.get(..., timeout=10)` or `asyncio.wait_for`): the top-of-loop wall-clock check can only fire once control returns to your loop, so you need the transport-level timeout to *force* control back on a hung socket.

4. Because they defend different things and diverge. The **token budget** guards context *size* — protecting against the model's hard context-window limit — and fires on token count alone. The **dollar budget** guards *spend*, computed as tokens × price, where input and output have different prices (output ≈ 4–5x). Concrete divergence: a cheap model with a large window can accumulate 200k tokens (token budget trips) while costing pennies (dollar budget untouched); conversely, a small number of high-output-token steps on an expensive model can trip the dollar budget while total tokens stay modest.

5. **(a) Every iteration**, because its entire purpose is stopping a run *mid-flight* — a human notices something *during* the run and needs it to stop before the next action; a startup-only check can't do that. **(b) Out-of-band** (a file/flag/signal on a separate channel from the agent's own logic) so the model cannot reason past it, ignore it, or decide the task matters more. It handles the **open set** of situations no budget predicted: harmful drift, an active incident, withdrawn consent, an about-to-happen mistake a person caught — anything you didn't know to encode as a limit.

6. **One sentence:** "The four budgets measure four different units — steps, seconds, tokens, dollars — and you can't express 'stop if looping OR hung OR bloated OR too expensive' with a single number, so collapsing them drops at least three sensors." The failure modes that reappear depend on which one you keep: keep only steps and you lose hang-detection, bloat-detection, and cost-detection (the overnight-bill classic); keep only dollars and you lose hang-detection (dollars freeze during a hang) and lose the distinct, fast triage the separate reasons give you.
