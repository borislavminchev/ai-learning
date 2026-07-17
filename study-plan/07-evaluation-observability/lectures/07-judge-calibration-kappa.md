# Lecture 7: Calibrating the Judge Against Humans with Cohen's Kappa

> You built an LLM judge in the last lecture — a strong model, reasoning-before-verdict, low-cardinality pass/fail, structured output. It runs in milliseconds and never gets tired. And you have no idea whether it's right. A judge is a measuring instrument, and an uncalibrated instrument is worse than no instrument: it gives you a number that *feels* authoritative while pointing anywhere. This lecture is the meta-evaluation loop — the practice of grading the grader. You hand-label 30-50 golden cases yourself, run the judge on the identical cases, and measure how much they *actually* agree using a statistic that isn't fooled by chance. After this lecture you can compute Cohen's kappa with three lines of scikit-learn, read its interpretation bands, diagnose *why* a low kappa is low (vague rubric vs. systematic directional bias), tighten the rubric and re-measure, and know exactly when calibration has expired and must be redone.

**Prerequisites:** Lecture 5/6 (LLM-as-judge mechanics: rubric mode, CoT-before-verdict, structured `Verdict`) and Week 1 (you have a golden set with human annotations). Comfort with `pip`/`uv`, a 2×2 table, and grade-school arithmetic. · **Reading time:** ~26 min · **Part of:** Evaluation, Testing & Observability — Week 2

## The core idea (plain language)

Your judge exists to stand in for a human reviewer at scale. So the only question that matters is: **when the judge says "pass," would a careful human have said "pass" too?** If yes, the judge is a valid proxy and you can trust its verdicts on 10,000 cases you'll never read. If no, every downstream number — your accuracy, your A/B test, your CI gate threshold — is built on sand.

Calibration is the act of measuring that agreement against the ground truth: human labels. The loop is deliberately simple:

1. **Hand-label 30-50 golden cases** pass/fail yourself. (Reuse the annotations you already produced in Week 1's error analysis — you've done most of this work.)
2. **Run the judge** on the exact same cases, collecting its pass/fail verdicts.
3. **Compute agreement** — but *not* raw agreement percentage, which lies. Compute **Cohen's kappa**, which corrects for the agreement you'd expect by pure luck.
4. **If kappa is low, fix the rubric and re-measure.** Record before/after kappa. That before/after number is the phase-milestone deliverable — it's your proof the judge is trustworthy.

The one non-obvious idea, and the reason this lecture exists, is step 3: **raw agreement is a trap.** Two labelers who agree 90% of the time sound great — until you learn 90% of the cases are "pass," at which point a monkey stapling "pass" to every case also scores 90%. Kappa strips out that free, chance-driven agreement and tells you how much the judge and human agree *beyond luck*. That correction is the whole game.

## How it actually works (mechanism, from first principles)

### Why raw agreement lies

Start with the number everyone reaches for first: **observed agreement**, $p_o$ = (cases where human and judge gave the same label) / (total cases).

Suppose you label 50 golden cases and the judge agrees on 45 of them. $p_o = 45/50 = 0.90$. Ninety percent! Ship it.

Now look at the *distribution* of your labels. Real golden sets — especially once your system is halfway decent — are imbalanced: most outputs pass. Say 45 of your 50 cases are genuinely "pass" and only 5 are "fail." A judge that has learned nothing about your rubric and simply outputs "pass" every single time will agree with you on all 45 passes and disagree on all 5 fails: $p_o = 45/50 = 0.90$ **with zero discriminating ability.** It never once correctly identified a failure — the entire reason you built the judge — yet it "agrees 90% of the time."

This is not a corner case. It is the *default* situation, because the whole point of a judge is to catch the rare failure inside a stream of mostly-passes. Raw agreement rewards the judge for the easy, abundant class and hides its blindness to the class you actually care about. **The more imbalanced your set, the more raw agreement flatters a useless judge.**

### The fix: subtract the agreement luck would have given you

Cohen's kappa asks a sharper question: *how much did human and judge agree, above and beyond what two random labelers with the same label frequencies would agree by chance?*

The formula is one line:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is observed agreement (what you measured) and $p_e$ is **expected agreement by chance** — the agreement you'd get if both labelers independently threw labels at the wall using their own base rates.

Read the formula as a fraction of *available room to improve on chance*:

- The numerator $p_o - p_e$ is how far you beat chance.
- The denominator $1 - p_e$ is the maximum you *could* have beaten chance (perfect agreement is 1.0).
- So $\kappa$ = (how much you beat chance) / (how much room there was to beat chance).

That gives the intuitive endpoints: perfect agreement → $\kappa = 1$. Agreement exactly at chance → $\kappa = 0$. Agreement *worse* than chance (systematic disagreement) → $\kappa < 0$.

### Computing $p_e$ from the margins

$p_e$ comes from the label frequencies each side used. If the human called "pass" a fraction $h_{pass}$ of the time and the judge called "pass" a fraction $j_{pass}$ of the time, then two independent labelers would both land on "pass" with probability $h_{pass} \times j_{pass}$, and both on "fail" with probability $h_{fail} \times j_{fail}$. Add them:

$$p_e = (h_{pass} \times j_{pass}) + (h_{fail} \times j_{fail})$$

That's it. The chance-agreement term is built entirely from the *margins* — how often each labeler used each label — not from whether they agreed on specific cases.

### Worked numbers on the "monkey judge"

Take the always-pass judge on the 45-pass / 5-fail set:

- Human margins: 45 pass, 5 fail → $h_{pass} = 0.90$, $h_{fail} = 0.10$.
- Judge margins: 50 pass, 0 fail → $j_{pass} = 1.00$, $j_{fail} = 0.00$.
- $p_o = 0.90$ (agrees on the 45 passes).
- $p_e = (0.90 \times 1.00) + (0.10 \times 0.00) = 0.90$.
- $\kappa = (0.90 - 0.90) / (1 - 0.90) = 0 / 0.10 = 0.00$.

**Kappa is exactly zero.** The statistic sees straight through the 90% and reports the truth: this judge has no skill. That is the entire value proposition of kappa in one arithmetic example.

### A judge with real skill

Now a judge that actually discriminates. Same 50 cases, laid out as a 2×2 confusion table (human on rows, judge on columns):

```
                 JUDGE pass   JUDGE fail   | human total
   HUMAN pass         43            2       |     45
   HUMAN fail          1            4       |      5
   ------------------------------------------------------
   judge total        44            6       |     50
```

- $p_o = (43 + 4)/50 = 47/50 = 0.94$.
- Human margins: $h_{pass}=45/50=0.90$, $h_{fail}=0.10$. Judge margins: $j_{pass}=44/50=0.88$, $j_{fail}=0.12$.
- $p_e = (0.90 \times 0.88) + (0.10 \times 0.12) = 0.792 + 0.012 = 0.804$.
- $\kappa = (0.94 - 0.804) / (1 - 0.804) = 0.136 / 0.196 = 0.69$.

So raw agreement 94% → kappa 0.69. Notice how much the number *deflated* once chance was removed: a 94% that looks nearly perfect is really "decent, use it but keep watching." That gap between the flattering 94% and the honest 0.69 is exactly what non-calibrated teams miss.

### Interpretation bands (memorize these)

These are conventional, approximate bands — treat them as engineering guidance, not law:

```
 kappa  | verdict
--------+-----------------------------------------------------------
 < 0.0  | worse than chance — judge is systematically inverted; STOP
 0.0-0.2| negligible — a random number generator
 0.2-0.4| weak — still basically noise for decision-making
 0.4-0.6| moderate — usable only with caution + heavy spot-checks
 0.6-0.8| decent — the working threshold; trust with monitoring
 > 0.8  | strong — trust it, this judge is a solid human proxy
```

The two lines to tattoo on your brain: **>0.6 is the minimum bar to trust a judge for real decisions, >0.8 is strong, and anything <0.4 is a random number generator with good grammar** — it emits fluent, confident rationales for verdicts that don't track human judgment at all. That fluency is *dangerous*, because it makes a kappa-0.3 judge feel authoritative in a way a coin flip wouldn't.

### The code: three lines

```python
from sklearn.metrics import cohen_kappa_score  # uv add scikit-learn

human_labels = ["pass", "pass", "fail", "pass", ...]   # your hand labels
judge_labels = ["pass", "fail", "fail", "pass", ...]   # judge verdicts, SAME order

kappa = cohen_kappa_score(human_labels, judge_labels)
print(f"kappa = {kappa:.3f}")
```

Three things that bite people:

- **Order and alignment.** The two lists must be case-aligned — index `i` in both must refer to the same golden case. The safe pattern is to build both from one loop over the golden set so they can't drift.
- **Label type doesn't matter.** Strings (`"pass"`/`"fail"`) or ints (`1`/`0`) both work; `cohen_kappa_score` treats them as categorical. Just be consistent across both lists.
- **The `weights` argument** (`"linear"`, `"quadratic"`) is for *ordinal* multi-class scales (e.g. your 1-4 rubric score, where confusing a 3 for a 4 is less bad than a 3 for a 1). For plain pass/fail, leave it unweighted. If you calibrate on the 1-4 scale, use `weights="quadratic"` so near-misses aren't punished like gross errors.

## Worked example — the full calibration loop with before/after

You have 40 golden cases with your Week-1 human pass/fail labels. You run the judge (v1 rubric) and build the aligned lists.

**Round 1.** `cohen_kappa_score` returns **0.38**. Below 0.4 — essentially noise. Do not ship. Now *diagnose*, because low kappa has two very different causes:

1. **Vague rubric (scattered disagreement).** Disagreements are spread roughly evenly — some false passes, some false fails. This means the criteria are ambiguous and the judge is guessing on the hard middle. Symptom: the confusion table's off-diagonal cells are both non-trivial and roughly balanced.

2. **Systematic directional bias (lopsided disagreement).** Almost all disagreements go one way — e.g. the judge passes things you failed. Symptom: one off-diagonal cell is large, the other near zero. This means the judge and rubric disagree on *one specific criterion's threshold*.

You inspect the 12 disagreements. Ten of them are cases where **you failed the output for not citing a source, but the judge passed it.** That's cause #2 — a systematic, one-directional gap. The rubric said "answer should be grounded in the context" but never made citation a hard requirement, so the judge treated it as optional.

**The fix.** Tighten the rubric to remove the ambiguity you just found:

```diff
- Pass if the answer is correct and grounded in the provided context.
+ Pass ONLY IF ALL hold:
+   1. The answer is factually correct per the context.
+   2. Every factual claim cites the specific context passage it came from.
+   3. If the context lacks the answer, the output says "I don't know" (do not pass a confident guess).
+ Otherwise: fail.
```

Note the shape of a good fix: it converts a fuzzy adjective ("grounded") into an **enumerated, checkable list** with an explicit fail-closed default. Vague rubrics produce low kappa; the cure is almost always *more specific criteria*, not a bigger judge model.

**Round 2.** Re-run the judge with the v2 rubric on the *same 40 cases*. Kappa is now **0.71**. You inspect the remaining disagreements — 5 now, scattered, no dominant direction — and decide they're genuinely borderline cases where even you hesitated. 0.71 clears the 0.6 bar.

**Record the deliverable:**

```
Judge calibration — golden_v1, 40 cases
  rubric v1: kappa = 0.38  (12 disagreements, 10 = missing-citation false passes)
  rubric v2: kappa = 0.71  (5 disagreements, scattered/borderline)
  change: made citation a hard pass criterion + fail-closed on unknown
```

That before/after pair — 0.38 → 0.71 with the one-line reason — is exactly what the phase milestone asks you to show. It proves you didn't just build a judge; you *validated* it and improved it.

## How it shows up in production

- **The confident-but-wrong dashboard.** An uncalibrated judge at kappa 0.35 reports "94% pass rate" on your prod traffic. Leadership sees green. Meanwhile the judge is rubber-stamping the exact failure class that's driving your support tickets, because it agrees with humans no better than chance on failures. You find out from angry users, not the dashboard. Calibration is what makes the green *mean* something.
- **The CI gate you can't trust.** Your Week-3 gate blocks merges when judge accuracy drops. If the judge is uncalibrated, the gate blocks on noise and passes on real regressions — it becomes a random merge-blocker engineers learn to ignore or `--force` past. A gate is only as trustworthy as the judge behind it, and kappa is the trust receipt.
- **The A/B test that reverses in prod.** You run baseline vs. new prompt, judge says new is +3%, you ship. Prod metrics don't move — or get worse — because the judge's +3% tracked something orthogonal to what users care about. Kappa ≥ 0.6 is the precondition that makes a judge-scored A/B result worth acting on.
- **Silent expiry.** You calibrated in March at kappa 0.75. In June you swapped the judge model to a newer/cheaper one and rewrote three rubric lines "to clarify them." Your calibration is now void and you don't know it — the judge could be at 0.4 and you're still quoting 0.75. **Any change to the rubric or the judge model invalidates the calibration**, because you changed the instrument. Re-run the 40-case loop; it costs minutes.
- **Cost of the loop itself.** Calibration is cheap: 30-50 judge calls (cents) plus the human labeling you largely already did in Week 1. The expensive thing is *not* doing it and building a whole eval stack on a judge that's secretly a coin flip.

## Common misconceptions & failure modes

- **"94% agreement means the judge is great."** Only if the classes are balanced. On a 90%-pass set, 90% agreement is the *chance* floor. Always compute kappa, never quote raw agreement to justify trust.
- **"Kappa is broken — my judge agrees 95% but kappa is only 0.2."** Kappa isn't broken; it's telling you the truth that raw agreement hid. Under heavy class imbalance kappa gets *harsh* precisely because there's little room to beat chance ($p_e$ is high, so the denominator $1-p_e$ is tiny and any disagreement craters the score). This is the "kappa paradox." The fix is not to abandon kappa — it's to **report the confusion table alongside it** and, when the failure class is tiny, add per-class precision/recall on the failures so you can see the judge's actual skill on the class you care about. If your golden set is 98% pass, deliberately over-sample failures (Lecture 4's stratification) so calibration has signal to work with.
- **"Low kappa means I need a bigger judge model."** Usually wrong. Low kappa most often means a **vague rubric**, not a weak model. Inspect the disagreements first; the cure is sharper criteria far more often than GPT-4o → o-something.
- **"The human labels are ground truth, full stop."** Humans disagree with *each other*. If you can't hit a decent kappa, sometimes the problem is that the *task itself* is ill-defined — two competent humans would also land at kappa 0.5. See adjudication below.
- **"I calibrated once, I'm done."** Calibration is per-(rubric, judge-model) pair. Change either and it expires. Treat the kappa value as attached to a specific rubric version and model snapshot, like a checksum.
- **"Kappa near 0 is bad but near -1 is just worse-bad."** Negative kappa is a *different* signal: the judge is systematically inverted (passing what you fail and vice versa). That's often a prompt bug — e.g. the pass/fail boolean is wired backwards, or the rubric's polarity is reversed. Negative kappa means "go read your judge code," not "get a better model."

### When humans disagree with each other

Your ground truth isn't perfect. Before blaming the judge, sanity-check the humans:

- **Measure inter-annotator agreement.** Have a second person label a subset and compute kappa *between the two humans*. This is your **ceiling** — a judge can't agree with "the human label" better than humans agree with each other. If human-human kappa is 0.55, do not expect judge-human kappa of 0.8; the task is inherently fuzzy.
- **Adjudicate disagreements.** Where two humans differ, a third (or a discussion) decides the "gold" label, and — crucially — you write down *why*. Those adjudication notes are the raw material for a clearer rubric.
- **Fix the rubric, not just the labels.** If humans disagree, the rubric is ambiguous *for humans too*. Tightening it (enumerated criteria, worked pass/fail examples) raises both human-human and judge-human kappa at once. A rubric a human can't apply consistently is one a judge can't either.

## Rules of thumb / cheat sheet

- **Never trust raw agreement %.** Always compute kappa. Raw % flatters imbalanced sets.
- **Bands (approximate):** <0.4 = noise (random generator with good grammar) · 0.4-0.6 = caution + spot-check · **>0.6 = decent, the trust bar** · **>0.8 = strong.**
- **Formula intuition:** $\kappa = (p_o - p_e)/(1 - p_e)$ = how much you beat chance, over how much room there was to beat it.
- **The loop:** hand-label 30-50 → run judge on same cases → kappa → if low, inspect disagreements → tighten rubric → re-run → record before/after.
- **Diagnose low kappa by the confusion table:** balanced off-diagonal = vague rubric; lopsided off-diagonal = systematic directional bias on one criterion.
- **Fix = specificity.** Turn fuzzy adjectives into enumerated, checkable, fail-closed criteria. Reach for a bigger judge model last, not first.
- **Report kappa *with* the 2×2 confusion table.** The table shows *where* it disagrees; kappa alone can hide the kappa-paradox under imbalance.
- **For 1-4 ordinal scores use `weights="quadratic"`.** For pass/fail, unweighted.
- **Re-calibrate on any change** to rubric text or judge model snapshot. Kappa is bound to a (rubric-version, model-snapshot) pair.
- **Check the human ceiling.** Human-human kappa caps judge-human kappa. If humans agree at 0.55, that's your realistic target, not 0.9.
- **Stratify so failures aren't rare** in the calibration set, or imbalance will crush kappa's signal.

## Connect to the lab

This is Week 2, lab step 4. Take the 30-50 golden cases you hand-labeled pass/fail in Week 1, run your `judge.py` on the same cases, and compute `cohen_kappa_score(human_labels, judge_labels)`. If kappa < 0.6, print the confusion table, read every disagreement, tighten the rubric (usually the criteria are vague), and re-run. Log **before and after kappa with the one-line reason for the change** — that before/after pair is the phase-milestone deliverable, and the acceptance bar is kappa ≥ 0.6 (or a documented reason + remediation if you can't reach it).

## Going deeper (optional)

- **scikit-learn docs — `sklearn.metrics.cohen_kappa_score`** (root: `scikit-learn.org`). The `weights` parameter and worked semantics. Search: `sklearn cohen_kappa_score weights`.
- **Jacob Cohen (1960), "A Coefficient of Agreement for Nominal Scales."** The original paper; skim for the chance-correction intuition, ignore the heavier stats. Search: `Cohen 1960 coefficient of agreement nominal scales`.
- **The "kappa paradox" under class imbalance.** Understand why high agreement can coexist with low kappa. Search: `Cohen kappa paradox prevalence high agreement low kappa`.
- **Hamel Husain — "Creating an LLM-as-a-Judge That Drives Business Value"** (blog: `hamel.dev`). The practitioner framing for aligning a judge to human labels and iterating the rubric. Search: `Hamel Husain LLM as a judge business value`.
- **Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."** The canonical study of judge-human agreement (they report agreement rates against humans). For intuition on how well judges track humans, not the math. Search: `LLM-as-a-judge MT-Bench agreement humans`.
- **Krippendorff's alpha / Fleiss' kappa** — generalizations for >2 annotators or missing labels; reach for these when several humans label. Search: `Fleiss kappa multiple annotators`, `Krippendorff alpha inter-rater`.

## Check yourself

1. Two labelers agree on 90 of 100 cases. Why is "90% agreement" potentially worthless, and what one fact about the cases do you need before you can interpret it?
2. Write the kappa formula and explain, in plain words, what each of $p_o$, $p_e$, and the denominator $1-p_e$ represents.
3. A judge scores raw agreement 96% but kappa 0.25 on a set that's 95% "pass." Is kappa broken? Explain what's happening and what you'd report alongside kappa.
4. Your judge is at kappa 0.35. The confusion table shows 11 of 13 disagreements are cases you failed but the judge passed. What does the *shape* of that disagreement tell you, and what's your first fix — a bigger model or something else?
5. You calibrated at kappa 0.78 in the spring. Name two changes that would silently invalidate that number, and say what you must do after either.
6. Two competent humans only reach kappa 0.5 labeling your task. What does that imply about the best judge-human kappa you can realistically expect, and what should you fix?

### Answer key

1. It's worthless if the classes are imbalanced: on a 90%-pass set, a labeler that blindly says "pass" every time also hits 90% agreement with zero skill. The fact you need is the **class balance** (base rate of pass vs. fail). Once you know it, you compute kappa, which subtracts the chance agreement that imbalance manufactures.
2. $\kappa = (p_o - p_e)/(1 - p_e)$. $p_o$ = observed agreement (fraction of cases where both labelers gave the same label). $p_e$ = agreement expected by chance, computed from each labeler's own label frequencies (margins) as if they labeled independently. $1 - p_e$ = the maximum possible improvement over chance (the "room" above chance up to perfect agreement). So kappa is "how far you beat chance" divided by "how far you *could* beat chance."
3. Kappa isn't broken — this is the kappa paradox. With 95% pass, $p_e$ is very high, so $1-p_e$ is tiny and even a few disagreements collapse the score; kappa is correctly telling you the judge shows little skill *beyond chance* on this imbalanced set. Report the **2×2 confusion table** alongside kappa, plus per-class precision/recall on the "fail" class so you can see the judge's actual ability on the rare class you care about; and consider over-sampling failures so calibration has signal.
4. The disagreement is **lopsided/one-directional** (almost all judge-passes-you-failed), which signals a *systematic bias on one specific criterion*, not scattered guessing. The fix is **not** a bigger model — it's tightening the rubric on that criterion: find what those failed-but-passed cases share (e.g. missing citations), make it an explicit hard pass requirement with a fail-closed default, and re-run.
5. (a) Editing the rubric text (even "just clarifying" a few lines). (b) Swapping the judge model or its snapshot (newer/cheaper version). Either change means you changed the measuring instrument, so the old kappa no longer applies — you must **re-run the calibration loop** on your golden cases and record a fresh before/after kappa bound to the new (rubric-version, model-snapshot) pair.
6. Human-human kappa is the **ceiling**: a judge can't match "the human label" more reliably than humans match each other, so ~0.5 is roughly the best judge-human kappa you can expect — chasing 0.8 is futile. The real fix is the **task/rubric ambiguity**: the low human agreement means your criteria are underspecified for humans too, so tighten the rubric (enumerated criteria, worked pass/fail examples, adjudicated gold labels), which raises both human-human and judge-human agreement together.
