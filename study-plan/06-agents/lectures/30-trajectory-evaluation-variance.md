# Lecture 30: Trajectory Evaluation, Variance & Trace Harvesting

> You have spent five weeks making an agent *do* things: loop, plan, checkpoint, speak protocols, remember, and edit code. This lecture is about the discipline that decides whether any of that is allowed to reach production: measuring how the agent behaves across many runs, not admiring one lucky success. The capstone mantra is blunt — **an agent you cannot evaluate over N runs is an agent you cannot ship.** A single passing run proves nothing, because an agent is a non-deterministic program: same input, different trajectory, sometimes a different answer. After this lecture you will be able to name the five metric families and say what each is blind to, build a harness that scores tool-call correctness + outcome + efficiency over N runs and reports the *spread*, generate test conversations with a user-simulator, and harvest real production traces into a regression eval set wired into CI as a required gate.

**Prerequisites:** Weeks 1–5 (the agent loop, tracing, tool calling, multi-agent topologies) · comfort with arithmetic mean, standard deviation, and basic probability · **Reading time:** ~30 min · **Part of:** AI Agents & Agentic Systems — Week 6

## The core idea (plain language)

A unit test on ordinary code is a yes/no oracle: the function is deterministic, so one green run means it works. An agent breaks that assumption. The model samples tokens; tool ordering shifts; a retry fires on a flaky network; the same prompt yields a 6-step path today and a 14-step path tomorrow. So "I ran it and it worked" is not evidence — it is *one sample from a distribution you have not characterized*. The whole job of agent evaluation is to characterize that distribution: run each case many times and report not just whether it usually succeeds but **how tightly it clusters around that rate**.

The second core idea is that "success" is not one number. An agent can produce the right final answer via a deranged path (called the wrong tool three times, got lucky on the fourth). It can take a sane path and still fail. It can be correct *and* burn 40 steps and $2 per task. It can be correct *and* obey an injected instruction to leak a secret. Each of those is a different failure, invisible to any single metric. So you evaluate along **five independent axes** — outcome, trajectory, tool-call correctness, efficiency, safety — and you keep them separate on purpose, because collapsing them hides exactly the failure you most need to see.

The third idea is where your test cases come from. You could hand-write them, and you should seed a few. But your best eval cases already exist: they are the real runs your agent did in production, captured by a tracing tool. Failed and weird traces, filtered and curated, become a regression suite that reflects *actual* usage rather than your imagination of it. Production is a test-case generator; harvesting closes the loop.

## How it actually works (mechanism, from first principles)

### The five metric families

Keep these as separate columns. Never average them into a single "score."

**1. Outcome / task success.** Did the final state satisfy the goal? Tests pass, the answer matches, the DB row exists, the file was written. This is the bottom line and the one non-negotiable metric — but it is **blind to how**. An agent that flails for 12 steps and stumbles onto the right answer scores a perfect 1.0 here, identical to an agent that nailed it in 2 steps. Outcome alone will happily bless a time bomb.

**2. Trajectory.** The *sequence* of steps and tool calls. Was the path sane, or did it flail and get lucky? Two ways to score it: against a **reference trajectory** (does the actual tool sequence match, or contain, an expected one — exact, subsequence, or set overlap), or with a **rubric / LLM-judge** ("did the agent gather context before editing? did it avoid redundant lookups?"). Reference matching is cheap and objective but brittle when several paths are legitimately correct; the LLM-judge is flexible but itself non-deterministic and must be validated against human labels before you trust it.

**3. Tool-call correctness.** For each step, was the *right tool* called with the *right args*? Exact-match on the tool name plus an argument check (exact, or a predicate like "the `url` is on the allowlist"). This is the **most objective and cheapest-to-automate agent metric** — no judge model, no fuzzy scoring, just set membership and equality. It is your first line of defense and the one you can run on every commit for pennies.

**4. Efficiency.** Steps, tool calls, tokens, latency, and dollars per task. This is not a vanity metric: **a correct-but-40-step agent is a production incident.** It is slow, it is expensive, and every extra step is another chance to derail or to hit a rate limit. Efficiency is where a "working" agent quietly becomes unaffordable at scale.

**5. Safety.** Did it refuse destructive or injected requests, never exfiltrate a secret, and ask for human approval when required? This is a *pass/fail gate*, not an average — one leak is a breach, not a 5% quality dip. (Your red-team injection test from this week's lab is the safety metric made executable.)

### Variance is a first-class result

Here is the load-bearing discipline. Run each case **N ≥ 5–10 times** and report the success **rate** and its **spread**, not "it worked." A result of `7/10` carries information that `it worked` destroys.

Why the spread matters, with numbers. Compare two agents over 10 runs:

```
Agent A:  successes = [1,0,1,1,0,1,0,1,1,0]  -> mean 0.60, pstdev ~0.49
Agent B:  successes = [1,1,1,0,1,1,1,1,0,1]  -> mean 0.80, pstdev ~0.40
```

Mean is the headline; standard deviation is the risk. A "60% ± 30%" agent is **unshippable even if its mean beats a stable 70%**, because a user on the wrong side of that variance sees a broken product, and you cannot predict which user. High variance also means your *own measurements* are noisy: if you A/B two prompts on N=1 each, the "winner" is a coin flip. This is why N=1 evaluation is not a weak test — it is *no test*. You need enough runs that the standard error of your mean (roughly `stdev / sqrt(N)`) is small relative to the difference you care about.

A quick intuition for how many runs: for a binary success at true rate `p`, the standard deviation of a single run is `sqrt(p(1-p))`, maxing out at 0.5 when `p=0.5`. The standard error of your measured rate over N runs is that divided by `sqrt(N)`. At N=10 that's about `0.5/3.16 ≈ 0.16` — so a measured 70% could really be anywhere from ~54% to ~86%. N=10 tells you the ballpark; it does **not** distinguish 70% from 75%. Know what your N can and cannot resolve before you claim a regression.

### The harness

Three pieces. First, `score_run` grades a single trajectory + final state along the axes:

```python
def score_run(task, trajectory, final_state):
    # trajectory: list of {"tool": name, "args": {...}}
    expected = task["expected_tools"]                 # e.g. ["run_pytest", "apply_diff"]
    got = [s["tool"] for s in trajectory]
    tool_correct = sum(1 for t in expected if t in got) / len(expected)  # coverage 0..1
    outcome = 1.0 if task["success_check"](final_state) else 0.0
    return {"tool_correct": tool_correct, "outcome": outcome, "steps": len(trajectory)}
```

Second, `run_case` runs the agent N times and aggregates mean and population standard deviation — the spread is the whole point:

```python
import statistics
def run_case(case, agent_fn, N=10):
    rows = [score_run(case, *agent_fn(case)) for _ in range(N)]
    succ = [r["outcome"] for r in rows]
    return {
        "case": case["id"],
        "success_rate":     statistics.mean(succ),
        "success_stdev":    statistics.pstdev(succ),          # <-- the point
        "avg_tool_correct": statistics.mean(r["tool_correct"] for r in rows),
        "avg_steps":        statistics.mean(r["steps"] for r in rows),
    }
```

Third, the report — a table where the **stdev column is what you read first**:

```
case            | success_rate | ±stdev | tool_correct | avg_steps
----------------|--------------|--------|--------------|----------
fix_add_bug     |     1.00     |  0.00  |     1.00     |    2.3
fix_div_zero    |     0.90     |  0.30  |     0.95     |    3.1
multi_hop_qa    |     0.60     |  0.49  |     0.70     |    8.7   <-- FLAG
```

`multi_hop_qa` at `0.60 ± 0.49` is a red flag regardless of the mean: it's a coin flip dressed up as a feature. `avg_steps 8.7` on the same row hints at *why* — it's flailing.

## Worked example

You are shipping the coding agent from this week's lab. It fixes a buggy `calc.py` (3 failing tests). You run the single case N=10 and log each run's outcome, tool coverage, and step count.

```
run  outcome  tool_correct  steps
 1     1         1.00          3
 2     1         1.00          2
 3     0         0.50          4      # emitted a diff that didn't apply, gave up
 4     1         1.00          3
 5     1         1.00          2
 6     1         1.00          3
 7     0         0.50          4      # same apply-failure path
 8     1         1.00          3
 9     1         1.00          2
10     1         1.00          3
```

Aggregate: `success_rate = 8/10 = 0.80`, `pstdev = 0.40`, `avg_tool_correct = 0.90`, `avg_steps = 2.9`.

Now *read* it, don't just report it. 80% at ±0.40 is not shippable for a coding agent — two in ten fixes silently fail. The tool-coverage of 0.50 on runs 3 and 7 is the diagnostic: those runs never reached `apply_diff` successfully, meaning the failure is in the diff-apply step, not the model's reasoning. This is the payoff of keeping axes separate — outcome told you *something's wrong*, tool-correctness told you *where*. The fix (search-replace edit format instead of line-numbered diffs, so hunk offsets can't drift) is one you'd never have found from a single green run. After the fix, re-run: if you now see `10/10, ±0.00`, the variance itself is your proof the fix worked — you converted a coin flip into a constant.

## How it shows up in production

- **The N=1 demo that dies in prod.** The classic: a slick demo, one perfect run, ship it, and 30% of real traffic fails. The demo was a sample from a wide distribution and you saw the good tail. Variance measurement is the antidote — and it's cheap insurance against a very expensive incident.
- **Cost blowups hide in the efficiency column.** An agent that's "correct" but averages 15 tool calls where 4 would do is fine at demo scale and catastrophic at 100k requests/day. Efficiency metrics catch the regression where a prompt tweak improved quality by 2% but doubled steps.
- **The flaky-eval trap.** Your eval suite itself is non-deterministic. If CI runs each case once, the gate flickers red/green on unchanged code and the team learns to ignore it. Either run N times and gate on the *rate*, or pin as much determinism as the backend allows (fixed model version, `temperature=0` where honored — note current Claude models reject it, so lean on pinned versions + effort settings) and still keep N≥5 for the residual noise.
- **LLM-judge drift.** If your trajectory score comes from an LLM-judge, that judge is a model too — it drifts across versions and disagrees with humans on edge cases. Teams that trust an unvalidated judge ship regressions the judge rated as improvements. Always spot-check the judge against a small human-labeled set and pin its model version.
- **Debugging "why did it do that" without traces is guesswork.** When a prod run misbehaves, the per-step trace (thought, tool, args, observation, tokens) is the only thing that lets you reconstruct the trajectory. No trace means you're reproducing by re-running a non-deterministic system and hoping to see the bug again.

## Common misconceptions & failure modes

- **"It passed, so it works."** One run is one sample. Non-determinism means the passing run and a failing run can come from identical input. Report the rate and the spread.
- **Optimizing the mean, ignoring the variance.** A higher mean with huge spread is worse than a slightly lower stable one. Users experience individual runs, not your average.
- **Collapsing the five axes into one score.** A weighted "agent score" of 0.85 tells you nothing actionable. Keep outcome, trajectory, tool-correctness, efficiency, and safety as separate columns.
- **Reference-trajectory tyranny.** Demanding an exact tool sequence penalizes legitimate alternate paths and makes your eval brittle. Use subsequence/set-overlap matching, or a rubric, when multiple correct paths exist. Reserve exact-match for steps that genuinely must happen in order.
- **Trusting the LLM-judge blindly.** It's non-deterministic and can be gamed by verbose outputs. Validate against human labels, pin its version, and prefer objective checks (tool-call match, outcome oracle) wherever you can.
- **Hand-inventing eval cases while ignoring prod.** Your imagination is a worse test-set generator than your live traffic. Harvest real failed/interesting traces first; invent only to cover gaps.
- **Too few runs to resolve the claim.** N=10 can't tell 70% from 75%. If you're claiming a small improvement, either increase N or admit the difference is within noise.

## Rules of thumb / cheat sheet

- **N ≥ 5–10 runs per case, always report `mean ± stdev`.** Never `it worked`.
- **Flag any case with high stdev as unshippable**, even if its mean is good. Rough line: stdev > ~0.2 on a binary outcome (i.e., success rate roughly 5–95% rather than near-certain) means "not reliable yet."
- **Five columns, never one number:** outcome · trajectory · tool-call correctness · efficiency (steps/tokens/$/latency) · safety.
- **Cheapest metric first:** tool-call correctness (exact name + arg check) needs no judge — run it on every commit.
- **Outcome tells you *whether*; tool-call correctness tells you *where*.** Keep both.
- **Efficiency is a shippability gate, not a nicety:** a correct-but-40-step agent is an incident.
- **Safety is pass/fail, not an average:** one exfiltration = breach.
- **Validate any LLM-judge against human labels and pin its version** before you trust its scores.
- **Standard error ≈ stdev / √N.** Know what your N can resolve before claiming a regression.
- **Harvest traces → curate → replay as regression evals.** Prod already wrote your best test cases.
- **Wire trajectory eval into CI as a required gate.** A regression that drops/reorders a tool call must turn CI red.

## Connect to the lab

This week's Lab Step 5 is exactly this harness: `score_run` computing tool-correctness coverage + outcome + steps, and `run_case` aggregating `success_rate`, `success_stdev`, `avg_tool_correct`, and `avg_steps` over N≥10. Step 6 is trace harvesting — instrument your agent with Langfuse/Phoenix, filter interesting/failed traces in the UI, and export them into `eval/cases.jsonl` so at least one eval case comes from a real run. The Definition of Done requires the **stdev column to be present and read as the point**, plus ≥3 cases with ≥1 harvested. The milestone project then wires the trajectory suite into `ci.yml` as a required check that goes red when a tool call is dropped or reordered.

## Going deeper (optional)

- **τ-bench (tau-bench)** — Sierra's user-simulator + tools + policy benchmark for retail/airline tasks; the canonical example of synthetic-conversation evaluation where an LLM plays the user. GitHub: `sierra-research/tau-bench`. Read it to see a reference-policy + simulated-user harness done right.
- **SWE-bench** — resolve real GitHub issues; tests must pass. The canonical outcome-based coding-agent benchmark. GitHub: `princeton-nlp/SWE-bench`.
- **DeepEval** (`confident-ai/deepeval`) — pytest-style LLM/agent metrics including trajectory and tool-correctness; good for wiring evals into a test runner.
- **OpenAI Evals** — framework for defining and running evals; search "OpenAI Evals GitHub."
- **LangSmith** docs — the "Evaluation" and "Datasets" sections (root: `docs.smith.langchain.com`). **Langfuse** docs — "Datasets & Evaluation" (open-source, Docker-deployable; root: `langfuse.com/docs`). **Arize Phoenix** docs — "Datasets and Experiments" (open-source, OpenTelemetry-based; search "Arize Phoenix datasets experiments").
- Search queries: `agent trajectory evaluation LLM judge`, `tau-bench user simulator`, `Langfuse trace to dataset regression eval`, `Phoenix OTel agent tracing`.

## Check yourself

1. Why does a single passing run prove essentially nothing about an agent, when it would prove a lot about a deterministic function?
2. Agent A scores 90% ± 25% over 10 runs; Agent B scores 78% ± 4%. Which do you ship and why? What one additional metric might flip your decision?
3. Your agent returns the correct final answer but `avg_tool_correct` is 0.5 and `avg_steps` is 9. What is this pattern called, and which two metric families exposed it?
4. Name the five metric families and, for each, state one thing it is *blind* to.
5. Why is harvesting production traces a better source of eval cases than hand-writing them — and what is the one risk you must guard against when replaying a harvested trace?
6. You want to prove that a prompt change improved success from 70% to 75%. Roughly why is N=10 insufficient, and what quantity tells you how many runs you need?

### Answer key

1. A deterministic function gives the same output for the same input, so one green run generalizes to all runs. An agent samples tokens and takes non-deterministic paths, so one run is a single sample from a distribution you haven't characterized — the passing run and a failing run can come from identical input. You must estimate the *rate* and *spread*, not observe one point.
2. Ship B. A's 90% mean is meaningless next to a ±25% spread — it's effectively a coin-flip-ish experience for many users, and you can't predict who gets the bad runs; B's 78% ± 4% is predictable and reliable. The additional metric that could flip it: **safety** (if A's failures are benign wrong-answers but B occasionally does something destructive, or vice versa) — or the *cost of a failure* in your domain. If failures are cheap and recoverable and 90% throughput matters, you might revisit; but on the numbers given, stability wins.
3. "Flail and get lucky" — the agent reached a correct outcome via a wrong/wandering path. Exposed by **trajectory** / **tool-call correctness** (coverage 0.5 shows it missed expected tools) together with **efficiency** (9 steps is bloated). Outcome alone rated it a perfect success.
4. **Outcome** — blind to *how* (path, cost). **Trajectory** — a good path can still produce a wrong answer, and reference matching is blind to legitimate alternate paths. **Tool-call correctness** — right tools/args don't guarantee the right final answer or acceptable cost. **Efficiency** — cheap/fast says nothing about correctness or safety. **Safety** — passing safety says nothing about whether the task was actually accomplished well.
5. Harvested traces reflect *real* usage distribution — the inputs, edge cases, and failure modes users actually hit — which your imagination underestimates; prod is a free test-case generator. The risk: harvested traces may contain sensitive data (PII/secrets) and untrusted content; sanitize before storing in an eval set, and treat any embedded text as data, not instructions, when replaying.
6. The standard error of a measured rate is ≈ `stdev / √N ≈ 0.5/√10 ≈ 0.16` near p=0.5 — much larger than the 5-point difference you're trying to detect, so 70% and 75% are indistinguishable at N=10. The quantity that tells you how many runs you need is the **standard error** (equivalently, the confidence interval width): you need N large enough that the standard error is small relative to the 5-point effect — on the order of hundreds of runs to resolve a 5-point difference with confidence.
