# Lecture 4: Just-Enough Classical ML for LLM Engineers

> You are here to build with LLMs, so why spend a lecture on logistic regression and gradient-boosted trees? Because the LLM era did not repeal the laws of machine learning — it just hid them behind an API. The failure modes that will bite you when you fine-tune a model, evaluate a RAG pipeline, or defend a "we should use GPT-4 for this" decision in a design review are the *exact same* failure modes that classical ML taught us to name 30 years ago: leakage, overfitting, the bias-variance tradeoff, and picking the wrong tool. After this lecture you will be able to split data so your numbers mean something, read a learning curve to diagnose over/underfitting, reason about bias-variance as an engineering knob (which matters directly when you fine-tune on 200 examples), and — most valuably — say out loud when an LLM is the *wrong* tool and a boring model is right.

**Prerequisites:** basic Python, comfort with arithmetic and simple probability; Lectures 1–3 (environment, tokens, the next-token model) help but are not required · **Reading time:** ~28 min · **Part of:** Phase 0 Week 1

## The core idea (plain language)

Machine learning is the practice of *fitting a function to data* and then using that function on data you have never seen. Everything else — the model architecture, the framework, whether it is a decision tree or a 70-billion-parameter transformer — is an implementation detail hanging off that one sentence. Two families of tasks:

- **Supervised learning:** you have inputs `X` and the *right answers* `y` (labels), and you learn a mapping `X → y`. Spam/not-spam, fraud/not-fraud, house price. An LLM doing instruction-following is, underneath, supervised learning on `(prompt, good response)` pairs.
- **Unsupervised learning:** you have `X` but *no* labels, and you find structure — clusters, embeddings, anomalies, dimensionality reduction. The embedding model you will use for RAG in Phase 4 was trained with (self-)unsupervised objectives.

The single most important engineering discipline in all of this is not the model — it is **the protocol you use to measure whether the model is any good**. Get the train/val/test protocol wrong and every number you report is a lie, including the ones that decide whether you ship. That is the spine of this lecture.

## How it actually works (mechanism, from first principles)

### Why you split data at all

A model can *memorize*. If you evaluate a model on the same data you trained it on, you are asking "can you recite what I already showed you?" — and a big enough model always can. That number tells you nothing about the future. So we hold data back.

The standard split is three buckets:

```
                 ALL LABELED DATA
   ┌──────────────────────┬───────────┬───────────┐
   │        TRAIN         │    VAL    │    TEST    │
   │        ~70%          │   ~15%    │   ~15%     │
   └──────────────────────┴───────────┴───────────┘
     fit params here      tune/choose  touch ONCE,
                          knobs here   at the very end
```

- **Train** — the model fits its parameters here. It is *allowed* to see the answers.
- **Validation (val / dev)** — you use this to *make decisions*: which learning rate, how many trees, which prompt, whether to stop training. Every time you look at val and change something, you leak a tiny bit of val's information into your choices.
- **Test** — the final exam. You touch it **once**, at the very end, to get an honest estimate of real-world performance. If you tune against test, you have contaminated your only honest number and you no longer have one.

**Why val AND test, not just two splits?** Because tuning *is* a form of fitting. Suppose you try 200 prompt variants and keep the one with the best val score. You have effectively "trained" your prompt-selection process on val — the winning score is optimistically biased (you got lucky on *that* val set). The untouched test set catches that optimism. Skip the test set and you will ship the number that flattered you.

> Engineering framing: **val is where you're allowed to cheat a little; test is where you're not allowed to cheat at all.** The moment you make a decision based on test performance, test stops being test.

### Cross-validation: when data is scarce

A single 70/15/15 split wastes data and is noisy on small datasets — your one val set might be unlucky. **k-fold cross-validation** fixes this: split the training data into `k` folds (typically 5 or 10), train on `k-1`, validate on the held-out fold, rotate, and average.

```
5-fold CV (each row is one "fold" held out as val):
fold 1:  [VAL][trn][trn][trn][trn]  -> score_1
fold 2:  [trn][VAL][trn][trn][trn]  -> score_2
fold 3:  [trn][trn][VAL][trn][trn]  -> score_3
fold 4:  [trn][trn][trn][VAL][trn]  -> score_4
fold 5:  [trn][trn][trn][trn][VAL]  -> score_5
                              CV score = mean(score_1..5)
```

You still keep a separate **test** set outside the CV loop. CV replaces the *val* split, not the test split. The mean gives you a more stable estimate; the *spread* across folds tells you how sensitive your model is to which data it saw — a wide spread is itself a warning.

**Where it bites LLM engineers:** fine-tuning and eval datasets are often tiny (hundreds of examples). A single split on 300 examples is dominated by noise. 5-fold CV of your eval set gives you a mean *and* a variance, so you can tell whether "prompt B beat prompt A by 2 points" is real signal or just fold luck.

### Overfitting, underfitting, and bias-variance

Fit a model and two things can go wrong:

- **Underfitting (high bias):** the model is too simple to capture the pattern. Train error high, val error high, and they're close together. A straight line through curved data.
- **Overfitting (high variance):** the model memorizes noise in the training data. Train error very low, val error high, big gap between them. A wiggly curve that threads every training point and generalizes terribly.

```
error
  ^
  |  \                              . val (generalization) error
  |   \                          .'
  |    \                      .'
  |     \.                 .'
  |       ` .          . '   <- sweet spot (lowest val error)
  |          ` . _ . '
  |            train error  ___________
  +----------------------------------------> model complexity
     UNDERFIT        good           OVERFIT
   (high bias)                   (high variance)
```

**Bias-variance as an engineering knob, not a theorem.** Total error decomposes (loosely) into bias + variance + irreducible noise. You trade them:

- More complexity / more parameters / less regularization → lower bias, higher variance.
- Less complexity / more regularization / more data → higher bias, lower variance.

You do not need the formal proof. You need the *reflex*: "train great, val bad → overfitting → simplify, regularize, or get more data." "Train bad, val bad → underfitting → more capacity or better features."

**Why this matters for fine-tuning later (the payoff):** when you fine-tune a 7B model on 200 examples, you have an enormous-capacity model and almost no data — the poster child for overfitting. The model will happily memorize your 200 examples verbatim and fall apart on the 201st. This is why LoRA (few trainable parameters), low epoch counts (often 1–3), and a held-out eval set exist. The bias-variance intuition you build here is *the* mental model for not wrecking a fine-tune in Phase 8.

### Learning curves: seeing overfitting instead of guessing

A **learning curve** plots error (or score) versus training-set size, for both train and val, on the same axes. Its *shape* diagnoses your problem:

```
HIGH VARIANCE (overfitting)            HIGH BIAS (underfitting)
score                                  score
 ^                                      ^
 |  train ----------------              |   train ----____
 |                        \             |   val   ----____   (both plateau
 |            big gap       \           |                     LOW, close
 |  val   _______----                   |                     together)
 +----------------------> N train       +----------------------> N train

Fix: more data, regularize, simpler   Fix: more capacity/features;
model. Gap shrinks as N grows.        more data won't help — lines
                                      already meet, just too low.
```

The decision rule an engineer keeps in their head:

- **Big persistent gap** between train and val → high variance → *more data will help*, or regularize / simplify.
- **Both lines low and converged** → high bias → *more data will NOT help*; you need a better model or better features.

This one plot saves you from the most expensive mistake in ML: spending a month labeling more data when your model was underfitting and more data was never going to move the needle.

## Worked example

You are building a **fraud detector** for a payments API. 100,000 historical transactions, 1,000 of them fraudulent (1% positive rate — highly imbalanced, which is normal for fraud).

**Step 1 — Split, and split by time.** Fraud is temporal (patterns drift, fraudsters adapt). A random split lets the model peek at the future. So: train on Jan–Sep, val on Oct, test on Nov. Also **deduplicate across splits** — the same card reused across train and test is leakage.

**Step 2 — Baseline first.** Fit `LogisticRegression` on engineered features (amount, hour, country mismatch, velocity). Ten seconds to train. It is your floor: any fancier model must beat *this* or it is not worth the complexity.

**Step 3 — Do NOT report accuracy.** A model that predicts "never fraud" scores 99% accuracy and catches zero fraud. Useless. Use precision/recall:
- **Precision** = of the transactions I flagged as fraud, what fraction really were? (`TP / (TP + FP)`)
- **Recall** = of the truly fraudulent transactions, what fraction did I catch? (`TP / (TP + FN)`)

Say your model flags 400 transactions; 180 are truly fraud (of 1,000 real frauds in the period). Precision = 180/400 = **0.45**. Recall = 180/1000 = **0.18**. Now the business decides the tradeoff: a bank chasing losses wants high recall (catch more fraud, tolerate more false alarms); a UX-sensitive product wants high precision (don't block legitimate customers). One number cannot capture that — which is exactly why accuracy hides the truth.

**Step 4 — Learning curve.** You plot it. Train recall 0.95, val recall 0.18, huge gap → overfitting. You add L2 regularization and simplify features. Gap shrinks, val recall climbs to 0.31. Now the curve is still separated and *rising* with data → more labeled fraud examples would help. That is a fundable decision, backed by a plot, not a vibe.

**Step 5 — The tool question.** Should you throw an LLM at this? The input is a structured row of numbers (amount, timestamp, country). An LLM would be slower, more expensive, non-deterministic, and *worse* than gradient-boosted trees on tabular data. The answer is no — and knowing that is the mark of a senior engineer.

## How it shows up in production

**Leakage silently inflates your metrics until launch day.** The spine names four mechanisms; internalize them because each one produces a beautiful offline number and a disastrous production one:

1. **Preprocessing leakage** — fitting a scaler, encoder, or TF-IDF vocabulary on the *full* dataset before splitting. The train set "knows" the test set's statistics. Fix: fit transforms on train only, apply to val/test (use scikit-learn `Pipeline` so this is structural, not willpower).
2. **Target leakage** — a feature that is a proxy for the answer and wouldn't exist at prediction time. Classic: `account_closed_reason = "fraud"` as a feature in a fraud model. 0.99 AUC in the notebook, useless in prod because you don't have that column *before* the fraud is known.
3. **Temporal leakage** — random-splitting time-series so the model trains on the future to predict the past. Always split by time for anything that drifts.
4. **Duplicate leakage** — the same (or near-same) row in both train and test. Common in scraped/LLM datasets; near-duplicates from templated data are the sneaky version.

**The overfit fine-tune.** In LLM land the most expensive version of this: a team fine-tunes on 500 support tickets, sees the loss curve plummet, ships it, and the model regurgitates training tickets verbatim and hallucinates on anything new. They overfit and had no held-out eval to catch it. The train/val/test discipline you are learning is *literally* the fix.

**Baselines-first saves money and dignity.** Before a proposal to spend $40k/month on LLM API calls to classify support tickets into 8 categories, someone should run a logistic regression on TF-IDF features. It often hits 88% in an afternoon for $0/month, deterministically, at 2ms latency. The LLM might reach 91% — now the org can decide if 3 points is worth 40k/month and 800ms latency. **On tabular/structured data, gradient-boosted trees (XGBoost, LightGBM) still routinely beat deep learning and LLMs** — this is a well-established, current result, not nostalgia.

**Cost/latency/determinism, concretely (approximate rules of thumb):** a logistic regression or GBT prediction is sub-millisecond, runs on a CPU, costs effectively $0 per call, and is *deterministic* — same input, same output, every time. An LLM API call is ~200ms–2s, costs cents, is non-deterministic even at temperature 0 (batching/hardware effects), and can be rate-limited or have an outage. If your task is a classification over structured inputs, the classical model wins on every operational axis.

## Common misconceptions & failure modes

- **"High accuracy means the model is good."** Not on imbalanced data. 99% accuracy on 1% fraud is a model that does nothing. Use precision/recall, F1, PR-AUC.
- **"I'll just check the test set to see how I'm doing."** The instant you make a decision from test, it is no longer test. Peeking repeatedly is the most common way teams fool themselves — every peek is a tiny fit.
- **"More data always helps."** Only if you're overfitting (variance-limited). If you're underfitting, more data changes nothing — the learning curve tells you which world you're in.
- **"Deep learning / LLMs beat classical models."** On unstructured language and images, often yes. On tabular/structured data, gradient-boosted trees usually still win, and they're cheaper, faster, and interpretable.
- **"temp=0 makes the LLM deterministic like a classifier."** No — floating-point non-determinism, batching, and MoE routing mean the same prompt can drift. If you need true determinism, that's a signal you may want a classical model.
- **"Cross-validation replaces the test set."** No. CV replaces val. Keep an untouched test set outside the CV loop, or you have no honest final number.
- **"Fine-tuning a huge model on my 100 examples will teach it my domain."** It will *memorize* your 100 examples. Without a held-out eval you won't know it overfit until users do.

## Rules of thumb / cheat sheet

- **Always three splits:** train (fit) / val (tune) / test (touch once). Test is sacred.
- **Split by time** for anything that drifts (fraud, prices, user behavior, logs). **Dedupe across splits** always.
- **Fit preprocessing on train only.** Use a `Pipeline` so leakage is structurally impossible, not a thing you remember.
- **Baseline before brains:** logistic regression / TF-IDF or gradient-boosted trees first. Beat that before adding complexity.
- **Tabular/structured data → gradient-boosted trees** (XGBoost/LightGBM). Don't reach for an LLM.
- **Never report accuracy on imbalanced data.** Precision, recall, F1, PR-AUC.
- **Read the learning curve:** big train/val gap → overfit → more data or regularize. Both low & converged → underfit → more capacity/features (more data won't help).
- **Small dataset? Use k-fold CV** (k=5 or 10) for a stable mean *and* a variance you can trust.
- **Fine-tuning intuition:** huge model + tiny dataset = overfit machine. Few params (LoRA), few epochs (1–3), and always a held-out eval.
- **LLM is the WRONG tool when:** input is structured/numeric; you need determinism, sub-ms latency, or $0/call; the task is a fixed-class classification a baseline solves; volume is huge and margins are thin.
- **LLM is the RIGHT tool when:** input is unstructured natural language, the task is open-ended (summarize, extract, reason, converse), or the label space is unbounded/novel.

## Connect to the lab

Week 1 Lab exercise 5 is where this lecture becomes muscle memory: you'll `train_test_split(..., stratify=y)`, fit `LogisticRegression` as a **baseline**, and plot a **train-vs-val learning curve to *see* overfitting** with your own eyes. Watch for the gap between the two curves — that gap *is* variance. You'll also hand-implement `precision_recall_f1` (Lab exercise 5) and confirm it matches scikit-learn. As you do the NumPy/pandas work (exercise 4), be paranoid about the leakage checklist above — the Definition of Done makes you state three leakage mechanisms and pick the right metric for a fraud detector vs a search box, which is exactly the Step-3/Step-5 reasoning from the worked example.

## Going deeper (optional)

- **scikit-learn User Guide** (scikit-learn.org) — read the *Cross-validation* and *Model evaluation / Metrics* sections, and the *Pipeline* docs (the anti-leakage tool). Also see the `learning_curve` and `validation_curve` API pages.
- **Book: *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (Aurélien Géron), chapters 1–4** — the canonical engineer-friendly treatment of splits, CV, and bias-variance.
- **Book: *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman)** — the reference for bias-variance if you want the rigorous version (free PDF from the authors' Stanford page). Optional and math-heavy.
- **XGBoost** (xgboost.readthedocs.io) and **LightGBM** (lightgbm.readthedocs.io) official docs — the tabular workhorses.
- Paper worth knowing: search **"Why do tree-based models still outperform deep learning on tabular data" (Grinsztajn et al., 2022)** — evidence for the baselines-first stance.
- Search queries: `"data leakage in machine learning" checklist`, `bias variance tradeoff intuition`, `learning curve diagnose overfitting underfitting`, `precision recall imbalanced classification`.

## Check yourself

1. Why do you need a separate val *and* test set — what specifically goes wrong with only train/test?
2. Your model gets 99% accuracy on a dataset that is 1% positive. Why might it be worthless, and what would you measure instead?
3. You plot a learning curve: train score 0.97, val score 0.62, and the gap is not closing as data grows. Diagnose it and give two fixes. Would collecting more data help?
4. Name three distinct ways data leakage sneaks in, and the fix for each.
5. Give one concrete task where an LLM is the *wrong* tool and a classical model is right, and justify it on latency/cost/determinism.
6. You're about to fine-tune a 7B model on 200 examples. Which classical-ML failure mode is the biggest risk, and what three practices mitigate it?

### Answer key

1. Tuning is a form of fitting: every decision you make from val leaks val's information into your choices, so the winning val score is optimistically biased. The untouched test set gives the one honest estimate of generalization. With only train/test, you tune against test, contaminate it, and lose your only unbiased number.
2. A "predict always-negative" model scores 99% accuracy while catching zero positives — accuracy is dominated by the majority class. Measure precision, recall, F1, and PR-AUC, and pick the precision/recall tradeoff that matches the business cost of false positives vs false negatives.
3. High variance / overfitting: the model memorizes training data (train high) but doesn't generalize (val low), with a persistent gap. Fixes: regularize (e.g., L2), simplify the model / reduce features, or get more data. Yes — because the gap is variance-driven and not yet closed, more data should help pull val up toward train.
4. (a) Preprocessing leakage — fitting scalers/encoders on all data before splitting; fix: fit on train only via a Pipeline. (b) Target leakage — a feature that encodes the outcome and won't exist at predict time; fix: audit each feature for "would I have this before the label is known?" (c) Temporal or duplicate leakage — training on the future, or the same row in train and test; fix: split by time and deduplicate across splits.
5. Example: classifying structured payment transactions as fraud/not-fraud. It's numeric/tabular, needs sub-millisecond deterministic decisions at high volume and near-zero per-call cost — gradient-boosted trees win on every axis; an LLM is slower, costlier, non-deterministic, and typically less accurate on tabular data.
6. Overfitting (high variance): enormous model capacity vs almost no data means it memorizes the 200 examples and fails to generalize. Mitigate with (a) few trainable parameters (LoRA/adapters), (b) very few epochs (often 1–3), and (c) a held-out eval set to detect memorization before shipping.
