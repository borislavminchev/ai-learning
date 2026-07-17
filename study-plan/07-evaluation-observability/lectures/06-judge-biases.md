# Lecture 6: Judge Biases and Their Mitigations

> An LLM judge is not a neutral ruler — it's a language model with the same statistical reflexes as any other, and those reflexes systematically distort the scores you build your entire measurement loop on. It will pick whichever answer it read first, reward the longer one, flatter its own family, and cluster its scores in the safe middle. Left unmeasured, these biases turn your green dashboard into fiction and send you shipping the wrong changes. This lecture is a field guide to the engineering-critical judge failure modes: what each one *is* mechanically, how to *detect* it empirically on your own golden set, and the concrete prompt or code change that *neutralizes* it. The through-line: you never assume a bias is gone — you measure the residual and log it.

**Prerequisites:** Lecture 5 (LLM-as-judge mechanics: rubric/pairwise/reference-guided modes, CoT-before-verdict, low-cardinality structured output) and Week 1 (a stratified, versioned golden set with human annotations). You can call a provider from Python and parse a Pydantic verdict. · **Reading time:** ~26 min · **Part of:** Evaluation, Testing & Observability — Week 2

## The core idea (plain language)

A judge bias is a systematic — not random — error in the judge's scores that correlates with something *other than* the quality you're trying to measure. Random noise averages out over enough cases; bias does not. If your judge adds a coin-flip of noise to every score, more cases fix it. If your judge adds +0.4 points to every answer that happens to be listed first, no amount of data fixes it — it just makes you *more confident* in a wrong number.

That distinction is the whole game. Bias is dangerous precisely because it survives the statistical rigor you learned to trust. A tight bootstrap confidence interval around a biased mean is a precise measurement of the wrong thing.

The five biases that bite hardest in practice:

1. **Position bias** — in pairwise comparison, the judge favors whichever answer appears first (sometimes last), independent of content.
2. **Verbosity / length bias** — longer answers score higher regardless of whether the extra words add correctness.
3. **Self-enhancement bias** — a model rates outputs from its own family higher than a neutral judge would.
4. **Leniency / central-tendency bias** — scores drift toward "safe" values (everything's a 4/5, or everything clusters at the middle of the scale).
5. **Format / authority bias** — confident, well-formatted, citation-studded answers score higher on *tone* even when the substance is wrong.

For each, the discipline is identical: **detect it on your own golden set with a controlled experiment, apply the specific mitigation, then re-measure the residual.** "We swap the order now" is not a mitigation you can trust until you've logged the disagreement rate before and after.

## How it actually works (mechanism, from first principles)

### Why a judge is biased at all

An LLM judge is the same next-token predictor as your generator, pointed at a scoring prompt. Its output distribution is shaped by pretraining and RLHF, both of which bake in correlations that have nothing to do with your rubric. In pretraining, longer and more confident text often *is* higher-quality (edited articles beat forum one-liners), so "longer ⇒ better" is a learned prior. RLHF reward models were themselves trained on human preferences that favored verbosity and confident tone. The judge inherits all of it. You are not fighting a bug; you are fighting the model's priors, and the only way to win is to constrain the prompt and *measure what leaks through*.

### Position bias — the mechanism

In pairwise mode you send the judge two candidate answers and ask which is better. Transformers attend over the whole prompt, but the serialized order of the two answers is itself a feature the model conditions on. Empirically, most current judges have a consistent preference for one slot (commonly the first) — the effect is real and large enough to flip verdicts on genuinely close pairs.

Here's the mechanical trap. Suppose answers A and B are truly of equal quality. A perfectly calibrated judge should call it a tie or split 50/50. A position-biased judge that favors the first slot will say "first one wins" *both* times you ask — so if you only run order (A, B), you record "A wins," and if you'd run (B, A) you'd have recorded "B wins." The verdict is a readout of the *ordering*, not the *content*.

**The detection-and-mitigation are the same operation:** run every comparison in *both* orders and require consistency.

```
Comparison of answers A and B:
  run 1: prompt = [A, B]  -> judge says "first"   (means A)
  run 2: prompt = [B, A]  -> judge says "first"   (means B)
  => the two runs DISAGREE  -> not a clean win for either; score as TIE
```

```
  run 1: [A, B] -> "first"  (A)
  run 2: [B, A] -> "second" (A)   # judge picked A in both physical orders
  => CONSISTENT  -> A wins for real
```

Only count a win when the verdict is consistent across both orderings. And here is the part most teams skip: **the disagreement rate across the two orderings IS your quantified position-bias measurement.** Log it. If 8% of your pairwise comparisons flip when you swap order, your judge has an 8% position-bias rate on this task, and every one of those 8% would have been a coin flip masquerading as a verdict had you run a single order. That number is a first-class metric — track it across judge-model and prompt versions like you track accuracy.

```python
def pairwise_debiased(judge, a, b):
    v1 = judge(first=a, second=b)   # returns "first" or "second"
    v2 = judge(first=b, second=a)
    winner1 = a if v1 == "first" else b
    winner2 = b if v2 == "first" else a  # note the swap
    if winner1 is winner2:
        return {"winner": winner1, "consistent": True}
    return {"winner": None, "consistent": False}  # tie; feeds the bias metric
```

Cost note: this doubles your judge calls. That's the price of a trustworthy pairwise verdict — budget for it in Lecture 5's cost model and in the Week 3 CI gate.

### Verbosity / length bias — the mechanism

The judge's learned prior says longer = more thorough = better. So an answer that pads with restated context, hedges, and bullet lists scores higher than a crisp correct one — even when the extra tokens add zero correctness and sometimes add errors.

**Detection is a scatter plot you can compute in ten lines:** score every golden output with your judge, record each output's length in tokens, and compute the correlation between length and score. On a *fixed* golden set where you already know the true quality (from human labels), a strong positive length-score correlation *that exceeds the length-quality correlation your humans show* is the bias, quantified.

```python
import numpy as np
# lengths: token counts;  judge_scores: judge's 1-4;  human_scores: your labels
r_judge = np.corrcoef(lengths, judge_scores)[0, 1]
r_human = np.corrcoef(lengths, human_scores)[0, 1]
print(f"length~judge r={r_judge:.2f}  length~human r={r_human:.2f}  bias gap={r_judge - r_human:.2f}")
```

A concrete read: if humans show `r=0.15` (longer answers *are* slightly better on your task) but the judge shows `r=0.55`, the `0.40` gap is verbosity bias, not signal. If both are ~0.15, your judge is fine on length — don't "fix" a bias you don't have.

Three mitigations, in rough order of preference:

- **Instruct the rubric to ignore length.** Add an explicit line: *"Length is irrelevant. A one-sentence correct answer must score higher than a three-paragraph answer that is wrong or padded. Judge only correctness and adherence to the criteria."* Cheap, and it moves the needle — but never assume it fully worked; re-run the correlation.
- **Control for length in analysis.** Bucket outputs by length and report scores per bucket, or regress score on quality *and* length and read the quality coefficient. This doesn't fix the judge; it stops length from confounding your A/B comparison.
- **Truncate / normalize.** For some tasks you cap both candidates at the same length before judging so neither can win on verbosity. Blunt, and it can cut real content, so reserve it for tasks where length genuinely shouldn't vary.

### Self-enhancement bias — the mechanism

A model tends to rate text that looks like its own generations higher — same phrasing habits, same structure, same hedging style feel "right" to it. If GPT-4o generates answers and GPT-4o judges them, the judge is grading its own handwriting and inflates the score.

**Detection:** take the same set of outputs and judge them with two *different* model families, then compare mean scores per generator. If Judge-Claude and Judge-GPT agree on the ranking of a Llama system but Judge-GPT alone rates a *GPT* system notably higher than Judge-Claude does, that gap is the self-enhancement signal.

**Mitigation is structural and non-negotiable:** the judge must be a *different model family* than the generator. Claude judges GPT, Llama judges both, GPT judges Claude. The most dangerous single configuration in all of eval is "GPT-4o generating and GPT-4o judging" — it looks like a clean measurement and is a mirror. When you A/B two systems from different families, use a *third*, neutral family as judge so neither contestant is graded by its own kin.

### Leniency / central-tendency bias — the mechanism

Ask a judge for a 1–100 score and you'll find it lives between 70 and 90. Ask for 1–5 and everything is a 4. This is two related effects: **leniency** (scores skew high — RLHF trained the model to be agreeable and avoid harsh judgments) and **central tendency** (scores avoid the extremes and cluster in the middle). Both destroy *discrimination*: if every answer scores 4/5, your judge can't tell your good answers from your mediocre ones, and your A/B tests show no movement no matter what you change.

This is a second, independent argument for the **low-cardinality scale** you met in Lecture 5. A 1–100 scale invites both biases — the model can't meaningfully discriminate 100 levels, so it retreats to a comfortable band, and you get pseudo-precise noise. A **pass/fail** or **1–4** scale forces a decision and *removes the middle* (a 4-point scale has no true center to hide in). Detection: histogram your judge's raw scores across the golden set. If 80% land on one value, you have a discrimination problem — either the scale is too wide or the rubric doesn't force a distinction. Mitigation: shrink the scale, and rewrite the rubric so each level has a concrete, mutually-exclusive definition ("3 = correct but omits one required field" vs "4 = correct and complete").

### Format / authority bias — the mechanism

A confident, well-structured answer with headers, citations, and no hedging reads as *authoritative*, and the judge rewards the tone even when the content is wrong. This is how a fluent hallucination — "According to the 2023 report, revenue was \$4.2B" (invented) — can out-score a correct but plainly-worded answer. **Detection:** build a small adversarial slice in your golden set of *confidently-wrong* cases (fabricated citations, assertive tone, false facts) and *diffidently-right* cases (correct but hedged, plain). A judge with format/authority bias will score the confident-wrong cases too high and the plain-right ones too low; measure the gap against your human labels. **Mitigation:** instruct the rubric to weigh *verifiable correctness over tone and formatting* ("A confident tone is not evidence of correctness. Verify claims against the provided context; penalize unsupported assertions regardless of how authoritative they sound."), and for grounded tasks pair the judge with a deterministic faithfulness/NLI check (Lecture 7) so a fabricated citation gets caught by a tripwire, not left to the judge's taste.

## Worked example

You're evaluating a support-answer bot. Golden set: 50 cases, each with a human pass/fail label. You run a pairwise A/B: baseline prompt (A) vs a new "be more thorough" prompt (B), judged by a single order (A, B) with GPT-4o — and GPT-4o also generated B. Result: **B wins 34/50 (68%)**. Ship it? No — three biases are stacked in B's favor. Let's measure.

**Position:** re-run all 50 in both orders. 11 comparisons flip when you swap. Position-bias rate = **11/50 = 22%**. Of B's 34 apparent wins, 9 were position artifacts. Consistent wins: B 25, A 20, ties 5.

**Verbosity:** B's "be more thorough" prompt made answers 2.1× longer on average. Correlation check: `length~judge r=0.52`, `length~human r=0.18`. Bias gap **0.34** — a big chunk of B's edge is length, not quality. You add "length is irrelevant; a short correct answer beats a long padded one" to the rubric and re-run: consistent wins now B 22, A 21, ties 7. B's lead is shrinking as you strip bias.

**Self-enhancement:** GPT-4o generated B *and* judged. Swap the judge to Claude (neutral family). Re-run the debiased pairwise: B 20, A 21, ties 9 — the lead is **gone**.

**The honest conclusion:** the original "68% win" was ~22% position artifact, ~a third length inflation, and a self-enhancement tilt. With all three neutralized, B and A are statistically indistinguishable (and you'd confirm that with a paired test and CI from Lecture 8). You just avoided shipping a longer, more expensive, no-better prompt — *because you measured the biases instead of assuming your judge was fair.* And you now log three numbers every eval run: position-disagreement 22%, length-score gap, and cross-judge delta.

## How it shows up in production

- **The phantom win that costs money.** A prompt "improvement" scores +6 in a single-order, same-family eval, ships, and does nothing for users — but the longer answers it produces raise token cost 2× and p95 latency with it. You paid more for a bias artifact. This is the single most common way judge bias burns a team.
- **The A/B that always favors the incumbent's slot.** If your harness always puts the current prod answer first, position bias systematically protects the incumbent, and genuinely better challengers lose 51/49 comparisons that were really position noise. Good changes die in the gate.
- **The flat dashboard.** Central-tendency bias means every variant scores 4/5, so no experiment ever moves the number and the team concludes "prompts don't matter." The scale, not the prompts, was the problem.
- **The fluent-hallucination leak.** Format/authority bias lets a confidently-wrong answer pass the judge and reach a user, who trusts the fabricated citation. This is a quality *and* trust incident, and the judge waved it through because it sounded right.
- **The self-graded mirror in CI.** A team wires GPT-4o as both generator and CI judge. The gate is green forever because the judge loves its own outputs; a real regression that a neutral judge would catch sails through. The gate provides false confidence — worse than no gate.

## Common misconceptions & failure modes

- **"We swap order now, so position bias is handled."** Swapping *detects and neutralizes per-comparison*, but only if you actually require consistency and count disagreements as ties. If you *average* the two runs instead of requiring agreement, a biased judge can still net a spurious winner. And if you don't log the disagreement rate, you have no idea how noisy your pairwise verdicts are.
- **"Telling the rubric to ignore length fixes verbosity."** It *reduces* it. It does not zero it — the prior is deep. The only honest claim is the residual correlation you re-measure after adding the instruction.
- **"A different judge family removes self-enhancement, so we're unbiased."** It removes *that specific* bias. The neutral judge still has position, verbosity, leniency, and format biases of its own. Debiasing is per-bias; there is no single "unbiased judge" setting.
- **"Bigger scale = more precise judge."** Backwards. 1–100 invites central tendency and leniency and yields pseudo-precise noise. Low cardinality forces discrimination.
- **"Our judge agrees with humans (high kappa), so it's bias-free."** Kappa (Lecture 5/8 calibration) measures agreement on the cases you labeled — it can be high while a *specific* bias hides in a slice (e.g., long answers) you didn't over-sample. Bias detection needs its own controlled experiments, not just aggregate agreement.
- **"We measured position bias once at 5%, so it's fine forever."** Bias is per-model, per-prompt, per-task. Change the judge model, the rubric wording, or the task and re-measure. Log it every run so a regression in the *judge* is visible.

## Rules of thumb / cheat sheet

- **Pairwise: always run both orders, count a win only on agreement, log the disagreement rate.** That rate *is* your position-bias metric — track it per judge/prompt version.
- **Never let the generator's family judge itself.** Different family always; a *neutral third* family for cross-family A/Bs. "GPT generates + GPT judges" is banned.
- **Verbosity: add "ignore length" to the rubric, then verify with a length-vs-score correlation** against human labels. Report the residual gap; don't assume it's zero.
- **Use low cardinality (pass/fail or 1–4).** It fights central-tendency and leniency *and* the discrimination problem. Histogram raw scores; if 80% pile on one value, shrink the scale or sharpen the rubric levels.
- **Format/authority: rubric line "confidence ≠ correctness; verify claims against context,"** plus a deterministic faithfulness/NLI tripwire for grounded tasks.
- **Detect on your own golden set, not on faith.** Every bias above has a ten-line controlled experiment. Run it before trusting a judge in a gate.
- **Measure the residual, never assume the mitigation worked.** The deliverable is a *number after mitigation*, logged next to accuracy.
- **Approximate priors (rules of thumb, not benchmarks):** position-disagreement under ~10% is livable, above ~20% your pairwise verdicts are shaky; treat any single number as a starting alarm threshold to tune on your own data — do not quote it as a benchmark.

## Connect to the lab

This maps directly onto Week 2 lab steps 3 and 4. In step 3 you add pairwise mode with position-bias control: run each comparison in both orders, count a win only when the judge is consistent, and **log the disagreement rate as your position-bias measurement**. Extend the lab by adding the verbosity check (length-vs-score correlation against your human labels) and enforcing a different judge family than your generator. Every mitigation in this lecture produces a *logged residual* — wire those numbers next to your accuracy so the Lecture 8 calibration (Cohen's kappa) and the Week 3 CI gate inherit a judge you've actually audited, not one you hope is fair.

## Going deeper (optional)

- **Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023).** The canonical source that named and measured position, verbosity, and self-enhancement bias. Read for the intuition and the detection framing, not the math. Search: `LLM-as-a-judge MT-Bench position verbosity self-enhancement bias`.
- **Hamel Husain — "Creating a LLM-as-a-Judge That Drives Business Value"** (blog: `hamel.dev`). Practitioner-grade guidance on building and *trusting* judges. Search: `Hamel Husain LLM as a judge`.
- **G-Eval paper and the DeepEval G-Eval docs** (`deepeval` on GitHub / their docs site). How rubric-based judging is structured; useful for writing bias-resistant rubric criteria. Search: `G-Eval NLG evaluation GPT`.
- **RAGAS docs** (root: `docs.ragas.io`) for the grounded-metric tripwires (faithfulness) that back up a format/authority-biased judge. Search: `RAGAS faithfulness answer relevancy`.
- **Search queries to keep handy:** `position bias LLM judge order swap`, `length bias LLM evaluator control`, `self-preference bias LLM judge family`, `LLM rating central tendency leniency low cardinality`, `LLM judge bias mitigation prompt`.

## Check yourself

1. Your pairwise harness always lists the production answer first and averages nothing. A challenger loses 24/50. Why can't you trust that result, what one change do you make, and what number do you now log?
2. A judge shows `length~score r=0.60`; your human labels show `length~quality r=0.20` on the same outputs. What does the gap tell you, and does adding "ignore length" to the rubric prove the bias is gone?
3. Why is "GPT-4o generates and GPT-4o judges" the single most dangerous eval configuration, and what's the fix when you A/B a GPT system against a Claude system?
4. You move from a 1–100 scale to 1–4 and your A/B tests suddenly start showing movement. Explain, in terms of bias, what changed.
5. A confidently-worded answer with a fabricated citation passes your judge. Name the bias, give the rubric change, and name the non-LLM check that should have caught it.
6. Your teammate says "we measured position bias at 6% last quarter, so pairwise is fine." Give two reasons that's not a safe assumption today.

### Answer key

1. Position bias: always-first ordering systematically favors the incumbent, so the challenger's 24/50 could be pure slot artifact, not content. The fix: run every comparison in *both* orders and count a win only when the verdict is consistent across both; disagreements are ties. The number to log is the **disagreement rate across orderings** — that *is* your position-bias measurement, and it tells you how many of those 50 verdicts were coin flips.
2. The `0.40` gap (0.60 judge minus 0.20 human) is verbosity bias: the judge rewards length beyond what actual quality justifies on your task. Adding "ignore length" to the rubric does *not* prove the bias is gone — the prior is deep and instructions only reduce it. You must **re-run the correlation after the change and report the residual gap**; the mitigation's success is a measured number, not an assumption.
3. Self-enhancement bias: the judge rates text that looks like its own generations higher, so a same-family generate-and-judge loop grades its own handwriting and inflates scores while looking like a clean measurement — and in CI it goes green forever, hiding real regressions. Fix for a GPT-vs-Claude A/B: use a **neutral third family** (e.g., Llama) as judge so neither contestant is graded by its own kin.
4. Central-tendency and leniency bias. On a 1–100 (or 1–5) scale the judge retreats to a comfortable band (everything ~85, or every answer a 4), so it can't discriminate and every variant scores the same — flat A/B tests. A 1–4 (or pass/fail) scale removes the safe middle and forces a decision, restoring discrimination, so real quality differences between variants finally show up as score movement.
5. Format / authority bias — confident tone and formatting read as correctness. Rubric change: state that *confidence and formatting are not evidence of correctness; verify each claim against the provided context and penalize unsupported assertions regardless of tone.* The non-LLM check: a **deterministic faithfulness / NLI (entailment) tripwire** that verifies each claim is entailed by the retrieved context, catching the fabricated citation regardless of how authoritative it sounds.
6. (a) Bias is per-model / per-prompt / per-task: if the judge model, the rubric wording, or the task changed since last quarter, the old 6% doesn't transfer. (b) You should never rely on a stale point measurement for a systematic error — position bias must be **logged every run** so a regression in the judge itself is visible; a number you measured once and stopped tracking is not a control, it's a memory.
