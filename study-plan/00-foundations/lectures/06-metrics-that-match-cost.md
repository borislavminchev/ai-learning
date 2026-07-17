# Lecture 6: Metrics That Match Business Cost

> Every model you ship makes mistakes. The only question that matters is: *which* mistakes, and what do they cost you? A fraud model that misses fraud bleeds money; a search box that buries the right document annoys users; a cancer screen that misses a tumor kills someone. These are the *same* statistical error — a false negative — but their business cost differs by orders of magnitude. This lecture teaches you to pick, compute, and defend the metric that maps to the dollar (or the life) at stake, and to spot the metrics that lie. After it you can compute precision/recall/F1 and a ranking metric by hand, explain why accuracy is a trap on imbalanced data, choose PR-AUC vs ROC-AUC correctly, set a decision threshold on purpose instead of accepting 0.5, and reason about why BLEU/ROUGE fail for open-ended text.

**Prerequisites:** basic probability, the classical-ML train/val/test split (Week 1 theory) · **Reading time:** ~22 min · **Part of:** Phase 0 Week 1

## The core idea (plain language)

A metric is a lens. Point the wrong lens at your system and you optimize the wrong thing with total confidence — the model looks great on your dashboard and terrible in the hands of users. The skill here is not memorizing formulas (they're arithmetic). The skill is the *mapping*: given a business problem, name the error that hurts most, then pick the metric that punishes that error.

Two facts drive everything in this lecture:

1. **Errors are not symmetric.** Missing a fraudulent transaction (false negative) and blocking a legitimate one (false positive) cost different amounts. Your metric must reflect that asymmetry, or you're averaging apples and lawsuits.
2. **Class balance changes which metric is honest.** When 99.8% of transactions are legitimate, a model that predicts "legit" every time scores 99.8% accuracy while catching zero fraud. Accuracy *lies* here. You need metrics that stay honest when positives are rare.

For classification you'll live in the confusion matrix and its children (precision, recall, F1, PR-AUC). For ranking and retrieval — which is *the* Phase 3–4 RAG concern — you'll live in recall@k, MRR, and nDCG@k. For open-ended text generation, you'll learn why the classic metrics collapse and why an LLM judging another LLM has become the 2025-2026 default.

## How it actually works (mechanism, from first principles)

### The confusion matrix is the atom

Every binary classifier decision falls into one of four buckets. Call the class you care about (fraud, tumor, relevant doc) the *positive* class.

```
                 Predicted POSITIVE     Predicted NEGATIVE
Actual POSITIVE   TP (true positive)     FN (false negative)  <- you MISSED it
Actual NEGATIVE   FP (false positive)    TN (true negative)
                  ^ you FLAGGED it
                    wrongly
```

Everything else is a ratio of these four numbers. Memorize the buckets, not the formulas.

- **Precision = TP / (TP + FP)** — "Of everything I flagged as positive, what fraction really was?" Precision falls when you cry wolf (false positives).
- **Recall = TP / (TP + FN)** — "Of everything that truly was positive, what fraction did I catch?" Recall falls when you miss things (false negatives). Also called *sensitivity* or *true positive rate*.

Read them as a plain-language pair: **precision = trust in a positive prediction; recall = coverage of the real positives.** They trade off. Flag everything → recall 1.0, precision in the gutter. Flag only the one case you're certain of → precision maybe 1.0, recall near zero.

### The business-cost framing

The whole point is choosing which of precision/recall to defend, based on what a mistake costs:

| Problem | Costliest error | Optimize for | Why |
|---|---|---|---|
| **Fraud / spam / disease screening** | False *negative* (missed fraud, missed tumor) | **Recall** | Missing the bad thing is catastrophic; a human reviewer can filter the extra false alarms cheaply. |
| **Search / recommendation / RAG top result** | False *positive* (irrelevant result shown first) | **Precision** (esp. precision@k) | Users judge you by what you *show*. One irrelevant top hit erodes trust; a missed doc on page 3 is invisible. |
| **Medical *diagnosis* that triggers surgery** | Depends — often false *positive* | Precision, or an explicit cost ratio | Unnecessary surgery is itself a serious harm, so you may not want to flag aggressively. |

Notice medical appears twice with opposite answers. *Screening* (cheap follow-up, catch everything → recall) differs from *definitive action* (irreversible, be sure → precision). "Medical = recall" is a lazy heuristic. Always ask: what does the *action* triggered by a positive cost?

### F1: the harmonic mean, and when it misleads

When you want a single number balancing precision and recall, use **F1 = 2 · (P · R) / (P + R)** — the harmonic mean. Harmonic (not arithmetic) because it punishes imbalance: if precision is 1.0 and recall 0.0, the arithmetic mean is a cheerful 0.5, but F1 is 0.0 — correctly telling you the classifier is useless.

Worked micro-example: P = 0.9, R = 0.1.
- Arithmetic mean = 0.50 (looks mediocre-but-okay).
- F1 = 2 · (0.9 · 0.1) / (0.9 + 0.1) = 2 · 0.09 / 1.0 = **0.18** (correctly looks bad).

**When F1 misleads:**
- It bakes in the assumption that precision and recall matter *equally*. For fraud they don't. Use **Fβ** instead: Fβ weights recall β times as much as precision. F2 favors recall (fraud); F0.5 favors precision (search). Reaching for F1 by reflex is optimizing for a 50/50 cost split you never actually have.
- F1 ignores true negatives entirely, which is usually correct for imbalanced data but means F1 alone can't tell you about the negative class.
- A single F1 hides the operating point. Two models with F1 = 0.70 can have wildly different P/R splits; one might be perfect for you and the other useless.

### Why accuracy lies on imbalanced data

**Accuracy = (TP + TN) / everything.** On a 99.8%-negative fraud dataset, the "always predict negative" model scores 0.998 accuracy — and TP = 0, so it catches no fraud. Accuracy is dominated by the giant TN bucket and tells you nothing about the rare class you actually built the model for. This is the single most common metric mistake juniors make. **Rule: the moment your classes are imbalanced, delete accuracy from the report and use precision/recall/PR-AUC.**

### Threshold selection: 0.5 is a default, not a decision

Most classifiers output a *score* (a probability-like number in [0, 1]), not a label. You get a label by thresholding: `positive if score >= t`. The default t = 0.5 is arbitrary. **Moving the threshold slides you along the precision/recall tradeoff** — raise t → fewer positives → higher precision, lower recall; lower t → the reverse.

You set the threshold to hit a business constraint, e.g. "we can only afford 50 human fraud reviews/day" (a precision/volume constraint) or "we must catch 95% of fraud" (a recall floor). You pick t on the **validation** set to satisfy that constraint, then report final numbers on test. Never tune the threshold on the test set — that's leakage.

### ROC-AUC vs PR-AUC: which threshold-free curve?

To evaluate a model *across all thresholds* at once, sweep t from 1 to 0 and plot a curve.

- **ROC curve:** true positive rate (recall) vs false positive rate = FP/(FP+TN). **ROC-AUC** = area under it. Problem on imbalanced data: FPR has the huge TN bucket in its denominator, so even thousands of false positives barely move FPR. ROC-AUC stays optimistically high (0.95+) while your model is drowning users in false alarms.
- **PR curve:** precision vs recall. **PR-AUC** (a.k.a. average precision). Because precision's denominator is TP + FP — no TN — it *feels* every false positive. On rare-positive problems PR-AUC drops honestly when the model is bad.

**Rule: imbalanced data (fraud, rare disease, retrieval) → PR-AUC. Roughly balanced classes → ROC-AUC is fine and comparable across datasets.** A ROC-AUC of 0.97 on a fraud model should make you suspicious, not happy.

### Ranking metrics: recall@k, MRR, nDCG@k (the RAG-critical trio)

Retrieval doesn't return a label — it returns a *ranked list*. In RAG (Phase 3–4) you fetch top-k chunks and stuff them into the prompt; if the answer-bearing chunk isn't in that k, the LLM cannot answer no matter how good it is. So retrieval quality is measured on rankings.

**recall@k** = fraction of all relevant items that appear in the top k. If there are 3 relevant docs and 2 land in your top 5, recall@5 = 2/3 ≈ 0.67. This is the metric you care about most for RAG: "did we even *retrieve* the evidence into the context window?"

**MRR (Mean Reciprocal Rank)** = average of 1/rank of the *first* relevant result. Rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33, none found → 0. Use it when only the first correct hit matters (a "feeling lucky" answer box, a lookup).

**nDCG@k (normalized Discounted Cumulative Gain)** rewards putting *highly* relevant items *high* in the list, with a logarithmic position discount. It handles *graded* relevance (a doc can be perfect=3, okay=1, irrelevant=0), unlike recall@k which is binary. Mechanism:
- **DCG@k = Σ (relevance_i / log2(i + 1))** for positions i = 1..k. The `log2(i+1)` denominator discounts later positions.
- **IDCG@k** = the DCG of the *ideal* ordering (relevances sorted descending).
- **nDCG@k = DCG@k / IDCG@k**, giving a 0–1 score comparable across queries.

## Worked example

**Ranking a RAG retrieval.** A query has these ground-truth relevances for the top 5 retrieved chunks (3 = perfect, 2 = good, 1 = weak, 0 = irrelevant). Suppose your retriever returned this order:

```
position i:   1    2    3    4    5
relevance:    2    0    3    1    0
```

**recall@5** (treat relevance > 0 as "relevant"): there are 3 relevant chunks total in the corpus for this query and all 3 appear in top 5 → recall@5 = 3/3 = **1.0**. (If one relevant chunk had been at position 8, recall@5 = 2/3.)

**MRR** for this query: first relevant result is at position 1 → reciprocal rank = **1.0**.

**nDCG@5:**

DCG = 2/log2(2) + 0/log2(3) + 3/log2(4) + 1/log2(5) + 0/log2(6)
    = 2/1.000 + 0 + 3/2.000 + 1/2.322 + 0
    = 2.000 + 1.500 + 0.431 = **3.931**

Ideal order sorts relevances descending → [3, 2, 1, 0, 0]:

IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0 + 0
     = 3/1.000 + 2/1.585 + 1/2.000
     = 3.000 + 1.262 + 0.500 = **4.762**

**nDCG@5 = 3.931 / 4.762 = 0.826.**

Interpretation: recall@5 said "all evidence was retrieved" (great for RAG) but nDCG@5 = 0.83 reveals the *ordering* is imperfect — the perfect chunk (relevance 3) sits at position 3 behind a weaker one and an irrelevant one. For RAG this matters because of "lost in the middle": models attend best to the start and end of context, so a perfect chunk buried at position 3 can still be underused. **recall@k tells you if the evidence is in the room; nDCG@k tells you if it's sitting where the model will actually read it.**

## How it shows up in production

- **The dashboard says 99% and users are furious.** Almost always accuracy on imbalanced data. First debugging move: pull the confusion matrix and look at the TP and FN cells directly. If TP is tiny, the headline metric is a lie.
- **Threshold drift.** You shipped at t = 0.5, then the positive rate shifted (seasonality, new fraud pattern). Precision and recall move even though the *model* didn't. Monitor P/R over time, not just at launch, and re-tune the threshold against your capacity/recall constraint.
- **RAG "the LLM is dumb" that's actually retrieval.** When RAG answers are wrong, engineers blame the generator. Measure **recall@k of the retriever first**: if the answer chunk isn't in the top k, no prompt engineering saves you. This one habit resolves a large share of RAG bugs. Only after recall@k is healthy do you tune nDCG (ordering) and then the generation prompt.
- **Choosing k costs money and latency.** Bigger k → higher recall@k but more tokens in the prompt (cost + latency + more distractor chunks hurting quality). recall@k vs k is a real curve you plot to pick the smallest k that clears your recall floor.
- **Offline metric ≠ online metric.** Your nDCG can rise while click-through falls. Offline ranking metrics are a fast proxy; the business metric (conversions, resolved tickets, dollars) is ground truth. Use offline metrics to iterate quickly and A/B the winners online.
- **Reporting one number to stakeholders.** "F1 = 0.8" invites the question "at what precision/recall?" Ship the confusion matrix and the P/R split, plus the cost framing ("we chose recall because a missed fraud costs ~$X and a false alarm costs one 30-second review").

### Why BLEU/ROUGE are weak, and why LLM-as-judge dominates (forward ref: Phase 7)

For open-ended generation (summaries, chat answers, RAG responses) the classic metrics are **BLEU** (n-gram *precision* vs references, from machine translation) and **ROUGE** (n-gram *recall* vs references, from summarization). Both measure **surface n-gram overlap** with one or a few reference texts. They break for open-ended tasks because:

- **Many correct answers, few references.** "The capital of France is Paris" and "Paris" are both right; low n-gram overlap tanks BLEU/ROUGE anyway.
- **No semantics.** "The plan will succeed" vs "The plan will not succeed" share almost all n-grams → high score, opposite meaning. Paraphrase gets punished; negation slips through.
- **They don't measure what you care about** for LLM output: factuality, helpfulness, faithfulness to retrieved context, tone, instruction-following.

So the 2025-2026 default for open-ended eval is **LLM-as-judge**: prompt a strong model with the input, the candidate output, and a rubric, and have it score or compare (pairwise "is A or B better?" is more reliable than absolute 1–10 scoring). It correlates far better with human judgment than BLEU/ROUGE because it reasons about meaning. Caveats you'll go deep on in Phase 7: judges have biases (position bias, verbosity bias, self-preference for their own family's outputs), cost real tokens, and must themselves be validated against human labels. BLEU/ROUGE still have a place for *constrained* tasks (translation, extractive summarization) where a reference is genuinely canonical.

## Common misconceptions & failure modes

- **"High accuracy = good model."** Only when classes are balanced *and* errors are symmetric. Otherwise it's the #1 lie. Check balance first.
- **"F1 is the objective metric."** F1 hard-codes a 50/50 precision/recall cost split you almost never have. Use Fβ, or report P and R separately with the cost story.
- **"ROC-AUC is high, so we're done."** ROC-AUC is misleadingly optimistic on imbalanced data because FPR hides false positives in the huge TN bucket. Use PR-AUC for rare positives.
- **"We'll tune the threshold on the test set."** That's leakage; your reported precision/recall become optimistic fiction. Tune on validation, report on test, touch test once.
- **"recall@k covers RAG."** recall@k ignores *ordering* and *graded* relevance. A perfect chunk at rank 8 with k=10 counts fully for recall@10 but the model may never use it. Pair recall@k with nDCG@k.
- **"BLEU/ROUGE measure quality."** They measure n-gram overlap with references, not meaning, factuality, or helpfulness. Fine for translation; misleading for chat/RAG.
- **"One global number describes the system."** Slice by segment (new vs returning users, language, fraud type). A model can have great aggregate recall while missing an entire fraud subtype.

## Rules of thumb / cheat sheet

- **Imbalanced data?** Kill accuracy. Use precision, recall, PR-AUC.
- **Missing the positive is catastrophic (fraud, screening)?** Optimize **recall** (F2 if you need one number).
- **Showing the wrong thing is catastrophic (search, RAG top result)?** Optimize **precision / precision@k** (F0.5 for one number).
- **Need one balanced number?** F1 — but always also report the P/R split behind it.
- **Threshold-free comparison:** PR-AUC for imbalanced, ROC-AUC for balanced. Never trust a lone 0.97 ROC-AUC on rare positives.
- **Threshold:** never blindly 0.5. Set it on validation to hit a recall floor or a review-capacity ceiling; monitor drift in production.
- **RAG retrieval:** lead with **recall@k** ("is the evidence retrieved?"), then **nDCG@k** ("is it ranked where the model reads it?"), then **MRR** if only the first hit matters.
- **nDCG position discount** is `log2(i+1)`: position 1 → /1, position 3 → /2, position 7 → /3. Later = cheaper.
- **Open-ended text:** skip BLEU/ROUGE; use **LLM-as-judge** (pairwise > absolute), and validate the judge against humans (Phase 7).
- **Always report the confusion matrix** alongside any ratio — it's the ground truth the ratios summarize.

## Connect to the lab

This lecture is the theory behind Week 1 Lab step 5. You'll implement `precision_recall_f1(y_true, y_pred)` and `ndcg_at_k(relevances, k)` from scratch in NumPy in `src/ai_foundations/metrics.py`, then assert in `tests/test_metrics.py` that they match `sklearn.metrics` (and a hand-computed nDCG) to 1e-6 — use the worked example above as a test fixture. Watch for: the classic accuracy trap on the (imbalanced) breast-cancer or fraud-style split, off-by-one in the nDCG `log2(i+1)` (positions are 1-indexed), and remembering to compute IDCG from the *sorted* relevances. Definition of Done also asks you to state, in one sentence each, which metric you'd optimize for a fraud detector vs a document search box — that's the business-cost mapping from this lecture.

## Going deeper (optional)

- **scikit-learn User Guide → "Metrics and scoring"** (scikit-learn.org) — canonical reference for precision/recall/F1, `precision_recall_curve`, `average_precision_score` (PR-AUC), `roc_auc_score`, and `ndcg_score`.
- ***Hands-On Machine Learning* (Aurélien Géron), Chapter 3** — the clearest engineer-level treatment of precision/recall tradeoffs, thresholds, and PR vs ROC curves.
- **Google Machine Learning Crash Course → "Classification"** (developers.google.com/machine-learning) — short, sharp modules on accuracy pitfalls, precision/recall, ROC/AUC, and thresholds.
- **"The Relationship Between Precision-Recall and ROC Curves" (Davis & Goadrich, 2006)** — the canonical paper on why PR curves are the right lens for imbalanced data. Search that title.
- **BEIR benchmark** (GitHub: beir-cellar/beir) — standard retrieval eval harness reporting nDCG@10 and recall@k across datasets; the practical reference for RAG-style ranking metrics.
- Search queries worth running: **"nDCG explained with example"**, **"PR-AUC vs ROC-AUC imbalanced"**, **"LLM as a judge best practices pairwise"**, **"MTEB retrieval metrics recall@k"**.

## Check yourself

1. A fraud model reports 99.7% accuracy. Your boss is thrilled. What one number do you pull to check whether that's real, and what would "the model does nothing" look like?
2. Model A: P=0.95, R=0.40. Model B: P=0.60, R=0.85. Which do you ship for (a) a fraud screen with cheap human review, (b) the top result of a search box? Justify via cost of errors.
3. You're told ROC-AUC = 0.96 on a dataset that is 0.5% positive. Why might that be misleading, and which curve would you look at instead?
4. Retrieval returns relevances [3, 1, 0, 2, 0] at positions 1–5. Compute recall@5 (relevant = rel>0, and assume those are all the relevant docs) and MRR for this query.
5. Why do BLEU and ROUGE fail to catch that "The drug is safe" and "The drug is not safe" are opposite answers, and what do you use instead for open-ended output?
6. You lowered the classification threshold from 0.5 to 0.3. Which of precision and recall goes up, which goes down, and why?

### Answer key

1. Pull the **confusion matrix** (or recall/precision on the positive class). "Does nothing" = predicts negative always → TP=0, FN=all frauds, recall=0, yet accuracy ≈ the negative rate (99.7%). High accuracy with zero recall means the model caught no fraud.
2. (a) Fraud screen → **Model B**: recall 0.85 catches far more fraud; its lower precision just means more false alarms, which cheap human review filters. (b) Search top result → **Model A**: precision 0.95 means what you *show* is almost always relevant; missed docs (lower recall) are less visible than an irrelevant top hit.
3. On 0.5% positives, the enormous TN bucket makes FPR = FP/(FP+TN) barely move even with many false positives, so ROC-AUC stays optimistically high. Look at the **PR curve / PR-AUC**, whose precision term (TP+FP, no TN) reacts to false positives.
4. recall@5: all relevant docs (rel>0 → positions 1,2,4) are in the top 5 → **3/3 = 1.0**. MRR: first relevant result is at position 1 → **1/1 = 1.0**.
5. BLEU/ROUGE score **n-gram overlap** with reference text; "safe" vs "not safe" share nearly all n-grams, so they score high despite opposite meaning — the metrics have no semantics and don't model negation. Use **LLM-as-judge** (with a rubric, ideally pairwise) validated against human labels.
6. Lowering the threshold flags **more** items as positive → **recall goes up** (you catch more true positives) and **precision goes down** (more of your flags are false positives). You slide along the P/R tradeoff toward coverage.
