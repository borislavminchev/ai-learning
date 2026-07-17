# Lecture 9: Statistical Rigor — Bootstrap CIs and Paired Tests

> Your eval harness just printed `baseline 82%, new prompt 84%` on 50 cases and the room wants to ship. This lecture is the numeracy that stops you shipping noise. A single accuracy number is not a fact about your system — it is one noisy sample from a distribution you never get to observe, and reporting it without an error bar is like quoting a coin-flip result to four decimal places. You will learn to wrap every accuracy in a **confidence interval** using the bootstrap (no formulas to memorize, no assumption that anything is bell-shaped), to *feel* how brutally wide that interval is at n=50, and to compare two variants with a **paired test** — McNemar for pass/fail, paired bootstrap for continuous scores — that squeezes far more signal out of the same data than the naive test everyone reaches for. After this you can compute a CI in ten lines of numpy, read a paired p-value, refuse a 2%-on-50 "win" out loud in a room full of people who want it, and estimate how many cases you actually need before your eval can *see* the effect you care about.

**Prerequisites:** A golden set (Week 1) and a way to produce a per-case score for two variants — `1/0` pass/fail or a continuous 0–1 score (Lecture 5's judge is fine). Python with numpy, scipy, statsmodels. High-school probability: you know what an average and a percentage are. · **Reading time:** ~28 min · **Part of:** Evaluation, Testing & Observability — Week 2

## The core idea (plain language)

When your golden set returns **84% accuracy**, that number is a **point estimate**: your single best guess at the true, unknowable accuracy your system would achieve across the entire population of inputs your users will ever send. But you didn't test the population — you tested 50 cases you happened to pick. Draw a *different* 50 and you'd get 80%, or 88%. That wobble is **sampling noise**, and wishing doesn't shrink it.

The entire discipline collapses to two habits:

1. **Never report a point estimate naked.** Report it with a **95% confidence interval (CI)** — a range like `84% [72%, 94%]`, meaning "if I re-ran this experiment many times, about 95% of the intervals I compute this way would contain the true accuracy." The *width* of that interval is the honest statement of how much you actually know.

2. **When comparing A vs B on the *same* cases, use a paired test.** Because both variants saw the identical 50 questions, you can look at where they *disagreed*, case by case, and cancel out the fact that some questions are simply harder than others. This is dramatically more powerful than comparing two independent averages — often the difference between "can't tell" and "clearly better" on the exact same data.

The payoff is one hard shipping rule: **compute each accuracy with its 95% CI and the paired p-value; if the interval on the *difference* includes zero, you have no proven difference — full stop.** No peeking, no cherry-picking the metric that happened to move.

## How it actually works (mechanism, from first principles)

### Why one number is a lie of omission

Your 50-case golden set is a **sample**. The thing you care about — accuracy on all future inputs — is a **population parameter** you can never measure directly. Your 84% is an *estimator* of it, and estimators have variance: they jiggle around the true value from sample to sample. Statistics is just the machinery for quantifying that jiggle.

The old-school way uses a formula. For a proportion `p` measured on `n` cases, the standard error is `sqrt(p*(1-p)/n)`, and a rough 95% CI is `p ± 1.96 * SE`. Plug in `p=0.84, n=50`:

```
SE     = sqrt(0.84 * 0.16 / 50) = sqrt(0.002688) ≈ 0.0518
95% CI ≈ 0.84 ± 1.96 * 0.0518   = 0.84 ± 0.102   →  [0.738, 0.942]
```

So your "84%" really means **"somewhere between 74% and 94%, probably."** That formula works for a plain proportion, but the moment your metric is *not* a proportion — a mean of continuous 0–1 judge scores, an F1, a p95 latency, a stratified weighted average — the neat formula either mutates or stops existing. This is where the bootstrap earns its keep.

### The bootstrap: simulate the experiment you can't re-run

You'd love to re-run the whole experiment 10,000 times with fresh 50-case samples and just *watch* how much the mean bounces. You can't — you have one sample of 50. The bootstrap's trick: **treat your sample as a stand-in for the population and resample from it, with replacement.**

One bootstrap "replicate" is: draw 50 scores *from your 50 scores, with replacement* (so some cases appear twice, some not at all), and take the mean. Do this N=10,000 times. You now hold 10,000 plausible means. Sort them, read off the 2.5th and 97.5th percentiles — that interval is your 95% CI.

```python
import numpy as np

def bootstrap_ci(scores, n=10_000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    k = len(scores)
    means = np.array([rng.choice(scores, size=k, replace=True).mean()
                      for _ in range(n)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return means.mean(), lo, hi
```

Why does resampling-with-replacement work? Because the variation you inject by reshuffling which cases are in or out mimics the variation you'd get drawing fresh samples of the same size. Crucially it makes **no distributional assumption** — you never assume the scores are normal, symmetric, or anything. If your metric is a weird ratio or a censored latency, the bootstrap doesn't care; it resamples the raw per-case numbers and lets the empirical distribution speak. That assumption-freedom is why it's the engineer's default: one function handles proportions, means, F1, medians, p95 — anything computable from a list of per-case values.

One subtlety: for a proportion the bootstrap and the `p ± 1.96·SE` formula give nearly identical answers (two lenses on the same noise). The bootstrap wins when the formula gets hard or your scores aren't 0/1.

### Feel the width at n=50

Numbers, not adjectives. Here is the approximate bootstrap 95% CI for a measured accuracy of 84% across sample sizes (the interval shrinks like `1/sqrt(n)`):

```
   n     accuracy      ~95% CI              half-width
   50      84%     [ 72% ,  94% ]            ~±11 pts
  100      84%     [ 76% ,  92% ]            ~±7 pts
  200      84%     [ 79% ,  89% ]            ~±5 pts
  500      84%     [ 81% ,  87% ]            ~±3 pts
 1000      84%     [ 82% ,  86% ]            ~±2 pts
```

Read the top row and let it sting. At **n=50, the CI is roughly ±11 percentage points.** A 2-point delta between two prompts is *a fifth of the noise band*. It is not a signal; it is the eval breathing. To get the interval down to ±2 points — tight enough to trust a 2-point delta — you need on the order of **1,000 cases**, because halving the width costs 4× the data (`1/sqrt(n)`).

```
CI half-width vs n  (accuracy ~84%, schematic)

±12 |•
    | •
 ±8 |    •
    |        •
 ±4 |             •  •
    |                     •      •
 ±0 +----+----+----+----+----+----+---→ n
     50  100  200  300  500     1000
```

This single picture is why "we A/B'd on 50 examples and the new one won by 2%" should make you reach for the brakes, not the deploy button.

### A vs B on the same cases: pair it

Now the comparison. You ran baseline and a tweaked prompt over the **same** golden set. The naive move is to compute two accuracies, two CIs, and eyeball whether they overlap. That throws away your biggest asset: **both variants answered identical questions.**

Some of your 50 questions are intrinsically hard (every variant fails them) and some are easy (every variant nails them). That case-to-case difficulty is a huge chunk of the variance in each accuracy number — and it is **shared** between A and B. A paired test looks only at the cases where A and B *disagree*, so the shared difficulty cancels and you're left with the pure A-vs-B signal. Same data, far more power.

**For binary pass/fail: McNemar's test.** Build a 2×2 contingency table of *how the two variants agreed and disagreed per case*:

```
                    B correct   B wrong
   A correct           b11         b10
   A wrong             b01         b00
```

The two diagonal cells (`b11` = both right, `b00` = both wrong) are the **concordant** pairs — they say nothing about *which* variant is better, so McNemar ignores them entirely. The story lives in the **discordant** cells:

- `b10` = A right, B wrong (cases B *lost*)
- `b01` = A wrong, B right (cases B *won*)

If the variants were truly equivalent, wins and losses should roughly balance (`b01 ≈ b10`). McNemar asks: given `b01 + b10` discordant pairs total, is the split lopsided enough to be surprising under a fair coin?

```python
from statsmodels.stats.contingency_tables import mcnemar

def paired_binary(a_correct, b_correct):
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)  # B won
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)  # B lost
    res = mcnemar([[0, b01], [b10, 0]], exact=True)   # diagonal unused
    return b01, b10, res.pvalue
```

Note the table handed to `statsmodels` puts zeros on the diagonal because McNemar only consumes the off-diagonal counts — we don't even need `b11`/`b00`. Use `exact=True` (the binomial exact test) when the discordant count is small (say `b01 + b10 < 25`); the chi-square approximation is fine for larger counts.

**For continuous scores: paired bootstrap.** When each case carries a 0–1 judge score rather than pass/fail, compute the **per-case difference** `d_i = score_B(i) - score_A(i)` and bootstrap the *mean of the differences*:

```python
def paired_bootstrap_diff(a_scores, b_scores, n=10_000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(b_scores, float) - np.asarray(a_scores, float)  # pair FIRST
    k = len(d)
    boots = np.array([rng.choice(d, size=k, replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return d.mean(), lo, hi          # CI on the DIFFERENCE
```

The magic is `d = B - A` computed **before** resampling: differencing case-by-case removes the case-difficulty variance, then you resample the differences. If the resulting CI on the difference straddles 0, you have no proven difference.

## Worked example

Fifty golden cases. Baseline gets 41/50 right = **82%**. New prompt gets 42/50 = **84%**. The 2-point win everyone wants to ship.

**Step 1 — CI on each, alone.** `bootstrap_ci` gives roughly:
```
baseline: 82%  95% CI [70%, 92%]
new:      84%  95% CI [72%, 94%]
```
The intervals almost completely overlap. Already a bad sign — but overlap is a crude test, so go to the paired analysis.

**Step 2 — pair it.** Read the per-case results and cross-tabulate:
```
                     new correct    new wrong
   base correct          39             2        (new lost 2)
   base wrong             3             6        (new won 3)
```
So `b10 = 2` (new lost 2 that baseline got), `b01 = 3` (new won 3 that baseline missed). Net movement: +1 case. Discordant total = 5.

**Step 3 — McNemar.** With `b01=3, b10=2` and the exact test, the p-value asks "is a 3-vs-2 split of 5 flips surprising?" — nowhere close. `p ≈ 1.0`. Under a truly fair coin you'd see a split this lopsided or worse the vast majority of the time.

**Decision:** the 2-point delta rests on a **net of one case** flipping, buried in ±11 points of sampling noise, with a paired p near 1. **Verdict: no proven difference. Do not ship on this evidence.** Either gather far more cases, or find a change big enough to clear the noise. This is not pedantry — it is the difference between an eval that steers and an eval that hallucinates progress.

Contrast: had you instead seen `b01=12, b10=1` (new won 12, lost 1) on the same 50 cases, McNemar's exact p ≈ 0.003 — lopsided enough to act on *even though the headline accuracies might look similar*, because the pairing exposed a consistent, one-directional improvement the two separate CIs would have blurred.

## How it shows up in production

- **The phantom win that costs a quarter.** A team ships a prompt on a "2%-on-40 improvement," it was noise, real quality is flat or worse, and three sprints later they're debugging a regression that was never a gain. The CI would have said "you don't know yet" for the price of ten lines of code.
- **Metric shopping / the peeking trap.** Run 8 metrics, one crosses `p<0.05` by chance (with 8 independent looks you *expect* ~one false positive at the 5% level), screenshot that one for the deck. This is p-hacking; it silently inflates your false-win rate. Decide your primary metric *before* the run; treat the rest as guardrails, not victory conditions.
- **The unpaired-test power leak.** Someone runs an independent (two-sample) t-test on paired data because it's the default they half-remember. It reports `p=0.30`, they conclude "no difference," and they *kill a real improvement*. The paired test on the identical data might have said `p=0.01`. Discarding the pairing discards power — and the failure is invisible because the test still runs and prints a plausible number.
- **CI-gate flakiness from tiny eval sets.** If your CI accuracy gate runs on 30 cases, the threshold check will pass and fail at random across identical code because the CI is ±14 points wide. Gates need enough cases (or a CI-aware threshold) or they become a coin flip that erodes trust in the whole pipeline.
- **"Stat sig" that's operationally meaningless.** At n=50,000 a 0.2-point difference can be `p<0.001` and still not worth the deploy risk. Significance answers "is it real?"; you *also* need "is it big enough to matter?" Report the effect size (the difference and its CI), not just the p-value.

## Common misconceptions & failure modes

- **"The two CIs overlap, so there's no difference."** Overlapping individual CIs do *not* imply non-significance — the correct object is the CI on the **difference** (paired). Two variants can have overlapping CIs yet a paired test finds a clear, consistent effect.
- **"Non-overlapping CIs prove a difference."** Closer to true, but still the wrong tool for paired data. Compute the paired statistic.
- **"Bigger N always fixes it."** N shrinks *sampling* noise; it does nothing for **bias**. A contaminated or unrepresentative golden set gives you a razor-tight CI around the *wrong* number. Rigor on garbage is confident garbage.
- **"p=0.05 means a 95% chance my prompt is better."** No. It means: *if there were truly no difference*, data this extreme (or more) would occur 5% of the time. It's a statement about the data under the null, not the probability the hypothesis is true.
- **"The bootstrap invents data / needs the data to be normal."** It resamples only your real per-case scores and assumes no distribution. It won't rescue you from too few cases (a CI from n=15 is honestly, correctly enormous), but it makes zero shape assumptions.
- **Forgetting to pair before bootstrapping continuous scores.** Bootstrapping A and B independently and subtracting their CIs is *not* the paired bootstrap and loses the variance cancellation. Difference per case *first*, then resample the differences.
- **Peeking and stopping early.** Re-running the eval after every few new cases and shipping the moment it crosses threshold inflates false positives just like metric shopping. Fix the sample size (or use sequential-testing methods) up front.

## Rules of thumb / cheat sheet

- **Never report an accuracy without a 95% CI.** `mean [lo, hi]`, always.
- **Bootstrap is your default CI** for any metric (proportion, mean score, F1, p95): `N=10_000` replicates, read the 2.5/97.5 percentiles, set a seed for reproducibility.
- **Rough CI half-width for accuracy ≈ 100·1.96·sqrt(p(1-p)/n) points.** At `p≈0.8`: n=50 → ±11, n=100 → ±8, n=400 → ±4, n=1000 → ±2.5. Halving the width needs 4× the data.
- **Same cases → paired test, always.** McNemar for pass/fail (`exact=True` when discordant pairs < ~25); paired bootstrap on per-case differences for continuous scores.
- **A 2-point delta on ≤100 cases is noise until proven otherwise.** Demand a paired p-value.
- **Ship rule:** if the CI on the *difference* includes 0 (equivalently the paired p is not below your pre-registered α, usually 0.05), the answer is **"no proven difference."**
- **Sample-size back-of-envelope (approximate):** to reliably detect an effect of `d` percentage points on a proportion, you need very roughly `n ≈ 800 / d²` per variant for small-ish effects. So d=10pts → ~50 cases (feasible); d=5pts → ~130; d=2pts → ~800; d=1pt → ~3000. Small effects are *expensive* — decide whether the effect you care about is even findable at your budget before you run.
- **Pre-register your primary metric and α.** Everything else is a guardrail, not a win condition. Don't shop metrics; don't peek-and-stop.
- **Report effect size *and* significance.** "Real" and "big enough to matter" are two different questions.

## Connect to the lab

This is Week 2 lab step 6 (`evals/stats.py`). You'll implement `bootstrap_ci` and `paired_binary` (McNemar), run your system twice — baseline vs a tweaked prompt — over the golden set, and print each accuracy **with its 95% CI** plus the paired p-value. The required "aha" is seeing how wide the CI is at n=50 and being able to explain, out loud, why a 2%-on-50 delta is not shippable. Add `paired_bootstrap_diff` if your judge emits continuous scores. This same harness backs the Week 3 CI eval gate's accuracy assertion — a gate on 30 flaky cases is a coin flip.

## Going deeper (optional)

- **Efron & Tibshirani, *An Introduction to the Bootstrap*** — the canonical, readable source on why resampling works. Skim the intuition chapters; skip the proofs.
- **scipy docs:** `scipy.stats.bootstrap` (batteries-included bootstrap CIs, including the BCa method that corrects for skew) and `scipy.stats` broadly. Root domain: `docs.scipy.org`.
- **statsmodels docs:** `statsmodels.stats.contingency_tables.mcnemar` (root: `www.statsmodels.org`) — read the page for `exact` vs chi-square guidance. Also `statsmodels.stats.power` for sample-size calculations.
- **Allen Downey, *Think Stats* (free online, greenteapress.com)** — an engineer-friendly, code-first treatment of resampling and hypothesis testing in Python. The best on-ramp if formulas ever felt like hand-waving.
- **Search queries:** `bootstrap confidence interval percentile method`, `McNemar test paired binary classifier comparison`, `paired vs unpaired test statistical power`, `p-hacking multiple comparisons`, `sample size proportion effect size calculator`.
- For the evals framing around all this, Hamel Husain's `hamel.dev` posts on measuring eval improvements pair well with this lecture.

## Check yourself

1. You measure 84% accuracy on 50 cases. In one sentence each, what does the *point estimate* tell you and what does the *confidence interval* tell you — and why is the naked point estimate misleading?
2. Describe the bootstrap procedure in three steps, and state the one assumption it makes about the distribution of your scores.
3. In McNemar's test, which cells of the 2×2 table are ignored and why? What do `b01` and `b10` count?
4. Baseline 82% vs new prompt 84% on the same 50 cases, with `b01=3, b10=2`. What do you compute, what will it roughly say, and what do you tell the PM who wants to ship?
5. Why does a paired test extract more signal than an unpaired one on the *same* golden set? What concretely gets cancelled out?
6. Your teammate ran 8 metrics and one hit `p=0.04`. Why is that not (yet) a green light, and what should have been decided before the run?

### Answer key

1. The **point estimate** (84%) is your single best guess at the true population accuracy from this one sample. The **CI** (e.g. [74%, 94%]) states how much that guess could move under resampling — the honest width of your uncertainty. The naked point estimate is misleading because it implies a precision you don't have; at n=50 the true value could plausibly sit anywhere across a ~20-point band.

2. (a) Draw a resample of size n from your n per-case scores **with replacement**; (b) compute the metric (e.g. mean) of that resample; (c) repeat ~10,000 times and take the 2.5th/97.5th percentiles of the resulting distribution as the 95% CI. The only assumption is that your sample is representative of the population — it makes **no assumption about the shape** (normal, symmetric, etc.) of the score distribution.

3. The two **diagonal (concordant)** cells — both-correct and both-wrong — are ignored because they say nothing about *which* variant is better. `b01` = cases where A was wrong but B was right (B won); `b10` = A right, B wrong (B lost). McNemar tests whether the split between these two is more lopsided than a fair coin.

4. Compute each accuracy's **95% CI** (both ~±11 pts, heavily overlapping) and the **paired McNemar p-value**. With a 3-vs-2 discordant split, p ≈ 1.0 — nowhere near significant; the "win" is a net of one case flipping inside ±11 points of noise. Tell the PM: **no proven difference — we can't ship on this; we need many more cases or a bigger change.**

5. Both variants answered the *same* questions, so the large variance from **case-to-case difficulty** (some questions hard for everyone, some easy for everyone) is shared. Pairing looks only at per-case differences (or discordant pairs), which cancels that shared difficulty and leaves the pure A-vs-B signal. An unpaired test treats the two runs as independent and lets that difficulty variance drown the signal, wasting power.

6. Running 8 metrics gives ~8 independent chances for a false positive; at α=0.05 you *expect* roughly one to cross by luck alone, so a lone `p=0.04` among 8 is unsurprising noise (a multiple-comparisons / p-hacking problem). Before the run you should have **pre-registered a single primary metric and α**, treating the rest as guardrails — not searched for whichever number happened to move.
