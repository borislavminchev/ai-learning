# Lecture 13: Catastrophic Forgetting — Causing It and Fixing It

> You fine-tune a 7B model on your support-ticket task. JSON-valid rate jumps from 71% to 98%, category accuracy from 82% to 94%. You ship it. Two weeks later, users report the model can no longer follow a multi-step instruction that isn't a ticket, refuses to summarize a paragraph, and — worst of all — happily answers a jailbreak it used to refuse. You didn't add a bug. You *taught* the model to forget. This lecture is about catastrophic forgetting: the phenomenon where a narrow fine-tune degrades general capability. You'll learn the three levers that cause it, how to deliberately trigger it in a lab so you recognize its signature on a dashboard, and the mechanical reason the recovery recipe — lower LR, fewer epochs, ~20% replay — actually works. After this you'll be able to read a base/over-trained/recovered scorecard, diagnose which lever went wrong, and never again call a fine-tune "good" on task metrics alone.

**Prerequisites:** Lecture 2 (SFT mechanics, loss masking), Lecture 3 (small-data hyperparameters: LR, epochs), LoRA/QLoRA basics (Week 2), a working eval harness with an LLM-judge and a held-out set (Phase 7) · **Reading time:** ~26 min · **Part of:** Phase 08 — Model Adaptation & Fine-Tuning, Week 4

## The core idea (plain language)

A pretrained instruct model is a giant pile of weights that encodes thousands of capabilities at once: it can follow instructions, write code, refuse harmful requests, summarize, translate, reason step by step, and — after your fine-tune — route support tickets. Those capabilities are not stored in separate files. They are *superimposed* on the same shared parameters. When you fine-tune, gradient descent nudges those parameters to lower the loss on **your** data and nothing else. The optimizer has no idea that some of the weights it's moving were the thing keeping the model good at instruction-following or safe on adversarial prompts. It just moves them.

Catastrophic forgetting is what happens when that movement is large enough to overwrite capabilities you cared about but didn't put in your training set. The model doesn't "decide" to forget. It's the mechanical consequence of optimizing a narrow objective with enough force and enough repetition that the shared weights drift far from where the general capabilities lived.

Three things control how far the weights drift:

1. **Learning rate** — how big each step is.
2. **Epochs** — how many times you take steps over the same narrow data.
3. **Data breadth** — whether the gradient only ever points in one direction (your task) or gets pulled back toward general behavior.

Turn all three toward "aggressive" — high LR, many epochs, homogeneous data — and you get a model that is *excellent* at your task and measurably worse at everything else. Turn them down, add a slice of general data back in, and the drift stays local. That's the entire game. The rest of this lecture is mechanism and numbers.

The single most important operational point: **task metrics going up tells you nothing about what went down.** You must measure retention explicitly, on a separate suite, before *and* after.

## How it actually works (mechanism, from first principles)

### Why shared weights forget

Think of the model's weights as a single vector `θ` living in a high-dimensional space. Somewhere in that space is a region where the model is good at general instruction-following (call it the "general basin"). Your SFT starting point sits inside that basin — that's why the instruct model is useful out of the box.

Fine-tuning runs gradient descent on your task loss. Each step is:

```
θ ← θ − (learning_rate) × gradient_of_your_task_loss
```

The gradient points in whatever direction most reduces loss **on your ticket data**. That direction is almost never aligned with "stay good at general tasks." So every step pushes `θ` a little bit out of the general basin and toward a narrow "ticket basin." How far you travel is `learning_rate × number_of_steps × (typical gradient size)`. Forgetting is just: you traveled far enough that `θ` left the region where general capabilities were intact.

This gives you the three causal levers directly:

- **Learning rate** scales every step. Double the LR, roughly double the distance traveled per step.
- **Epochs** multiply the number of steps over the *same* examples. More passes = more accumulated drift, and (separately) memorization of your specific examples.
- **Data breadth** controls the *direction* of the gradient. If 100% of your batches are ticket-routing, every gradient points toward the ticket basin. If 20% of your batches are general instruction data, one in five steps pulls `θ` back toward the general basin — a leash.

### PEFT limits the damage by construction

LoRA/QLoRA changes the geometry. Instead of moving all of `θ`, you **freeze** the base weights and train small low-rank adapter matrices injected into a few layers. The base capabilities live in the frozen weights, which never move. The only thing that changes is a low-rank delta `ΔW = (alpha/r) · B·A` added at inference.

This bounds forgetting two ways. First, the adapter has far fewer degrees of freedom (typically <1% of params), so it *can't* rewrite arbitrary general behavior — there simply isn't enough capacity to encode "forget instruction-following." Second, and this is the killer feature, you can **remove the adapter** and get the exact original model back, bit-for-bit. Forgetting with full fine-tuning is permanent; forgetting with an adapter is a toggle. PEFT doesn't make forgetting *impossible* — a high enough LR over enough epochs will still push the adapter into damaging territory — but it dramatically raises the threshold and makes the damage reversible.

### The retention metric and its signature

You need a fixed, held-out suite of *general* capabilities that has nothing to do with your task, scored identically before and after. The standard slice (Week 4 lab): a few MMLU subjects (general knowledge/reasoning), IFEval (instruction-following), and one safety/refusal probe. You run it on the base instruct model to get a baseline, then on each fine-tuned variant.

The signature of catastrophic forgetting is a **scissors** on your dashboard: your task metric rising while your retention metric falls, as a function of training aggressiveness.

```
metric
  ▲
  │  task accuracy ────────●────────●  (rises, then plateaus)
  │                ●──────
  │        ●──────
  │  ●────                          retention (IFEval/safety)
  │  ────●──────                    ●
  │            ──────●──────        │
  │                        ──────●──┘  (falls off a cliff)
  └────────────────────────────────────►  epochs / LR (aggressiveness)
     1ep       3ep       5ep      5ep+high-LR
```

When you see the lines cross, you are trading general capability for task performance. The engineer's job is to find the point *before* the crossover — or to change the geometry (PEFT, replay) so the retention line stays flat.

### Replay: why mixing in general data works

Replay means: mix ~10–20% general instruction data (a slice of a public SFT set — Alpaca, Dolly, OpenAssistant/OASST, a slice of Tulu, etc.) into your training data. Mechanically, it changes the *average gradient direction*.

Without replay, every batch's gradient points toward the ticket basin. With 20% replay, roughly one gradient in five points back toward "be a good general instruction-follower." The optimizer now has to satisfy *both* objectives, so the equilibrium it settles into is a point that is good at tickets **and** still inside the general basin. The general data acts as a continual reminder — a regularizer that says "don't forget you also do this." It's the cheap, practical stand-in for the fancy academic methods (EWC, rehearsal buffers); replay *is* rehearsal, done with a data mixer instead of a math trick.

Why 10–20% and not 50%? Because replay trades task-signal density for retention. Every replay example is an example that isn't teaching your task. Too little (<5%) and the leash is too loose to matter; too much (>30%) and you dilute the task signal enough that task metrics suffer and training takes longer. 10–20% is the empirically sweet band where retention is largely preserved and the task still gets enough concentrated signal. Treat these as approximate starting points, not laws.

## Worked example

Let's make the levers concrete with arithmetic, then walk the lab's three-model contrast.

**Drift budget intuition.** Suppose your dataset is 800 ticket examples, batch size 8, so 100 steps per epoch. Compare two runs:

- **Over-trained:** LR = 5e-4, 5 epochs → 500 steps at a large step size. Rough "distance traveled" ∝ `5e-4 × 500 = 0.25` (arbitrary units).
- **Recovered:** LR = 1e-4, 2 epochs → 200 steps at a small step size. Distance ∝ `1e-4 × 200 = 0.02`.

The over-trained run travels roughly **12× farther** through weight space on the same data. That's not a subtle knob — it's an order-of-magnitude difference in how far you push the model out of the general basin. This is why "just lower the LR and cut epochs" is not timid advice; it's the dominant term.

**The three-model scorecard** (illustrative numbers — your real ones come from the lab; do not treat these as benchmarks):

| Model | Config | Task acc (JSON+category) | IFEval (retention) | Safety refuse rate |
|---|---|---|---|---|
| Base instruct | — | 82% | 71 | 96% |
| Over-trained | LR 5e-4, 5 ep, 0% replay | **95%** | **48** | **74%** |
| Recovered | LR 1e-4, 2 ep, 20% replay | 93% | 70 | 95% |

Read the story in the columns. The over-trained model *wins* on your task (95%) — which is exactly why it's a trap: the number you were staring at looks great. But IFEval collapsed from 71 to 48 (it forgot how to follow general instructions) and the safety refusal rate fell from 96% to 74% (it forgot to refuse — a shippable-blocker on its own). The recovered model gives up two points of task accuracy (95 → 93, noise-level) and buys back essentially all of the retention: IFEval 70 vs 71 baseline, safety 95 vs 96.

That trade — two task points for ~all of your general capability and safety back — is almost always correct. **This base/over-trained/recovered contrast is the headline of your model card.** It's the single artifact that proves you understand what fine-tuning did, not just that it "worked."

Why does the recovery recipe restore retention while keeping the task strong? Three things stacked:

1. **LR 5e-4 → 1e-4** shrinks each step 5×, so you can't lurch out of the general basin.
2. **5 → 2 epochs** cuts total steps and stops the memorization that drives overfit-forgetting.
3. **20% replay** actively pulls the gradient back toward general behavior every few steps.

Any one alone helps; together they move the equilibrium to "good at tickets, still general." And 2 epochs at a sane LR is still plenty of signal for 800 clean examples to learn a narrow, well-structured task — that's why task accuracy barely moves.

## How it shows up in production

**The silent safety regression.** This is the one that gets people fired. A narrow fine-tune on benign task data can degrade the model's refusal behavior as a *side effect* — nobody trained it to answer jailbreaks, it just drifted off the aligned distribution. If your only gate is task accuracy, you ship a model that's more helpful on tickets and more willing to help with things it should refuse. **Re-run your safety/refusal evals after every tune**, treat a drop as a release blocker, not a note.

**"It got dumber" bug reports with no code change.** Users hit the model on adjacent-but-off-task requests and it fails: won't summarize, ignores a formatting instruction, loses multi-turn context. Your task dashboard is green the whole time because those requests aren't in your held-out task set. Without a retention suite you have no instrument that can even see this, so you'll waste a day looking for a deployment bug that doesn't exist.

**The over-trained model that "benchmarks best."** In a hyperparameter sweep, the 5-epoch high-LR run often tops the *task* leaderboard because it has essentially memorized the task distribution. If your selection criterion is task metric alone, your sweep will systematically pick the *most forgetful* model. Selection must be a joint criterion: task metric **and** a retention floor (e.g. "reject any variant that loses >10% on IFEval or >2 points on safety refusal").

**Cost/latency angle.** Forgetting-aware training is slightly *more* expensive per run (replay adds ~15–20% more data; you may run 2–3 configs to find the safe point) but far cheaper than the alternative: shipping a regressed model, an incident, a rollback, and a re-tune under pressure. The retention suite itself is cheap if you keep it small — a few MMLU subjects and IFEval, not all 57 MMLU subjects (which is hours of compute per eval).

**Merged vs adapter in prod.** If you serve the LoRA adapter hot-swappable, a forgetting regression is a one-line rollback (detach the adapter → exact base model back). If you merged and quantized, rollback means redeploying the previous artifact. Keep the adapter around even if you serve merged — it's your undo button.

## Common misconceptions & failure modes

- **"Lower training loss means a better model."** Loss down only means "better at reproducing your training targets." It says nothing about retention and is *positively correlated* with forgetting once you're overfitting. Loss is not the deliverable; the scorecard is.
- **"LoRA can't forget, the base is frozen."** LoRA *reduces* and *makes reversible* forgetting; it doesn't eliminate it. A high enough LR over enough epochs pushes the adapter delta into territory that degrades general behavior at inference. Measure retention even with adapters.
- **"More epochs = more learning = better."** Past ~2–3 epochs on a small dataset you're memorizing, not generalizing. Eval loss turns up, task generalization plateaus, and forgetting accelerates. More epochs is one of the two fastest ways to cause forgetting.
- **"Replay just slows training down."** Replay is a *regularizer*, not overhead. The 15–20% of general data is doing load-bearing work: it's the leash that keeps `θ` in the general basin. Cutting it to save time is how you re-introduce the forgetting you were trying to prevent.
- **"I'll add more task data instead of replay."** More *narrow* data doesn't help retention — it makes the gradient point even harder in the one direction. Breadth, not volume, fights forgetting. Replay adds breadth.
- **"Safety was fine before, so it's fine now."** Safety is a general capability that lives in the same shared weights as everything else. It's one of the *first* things to regress under an aggressive narrow tune. Re-test it every single time.
- **Selecting on a single metric.** A sweep optimized purely on task accuracy will hand you the most-forgetful checkpoint. Always select on task-metric-subject-to-a-retention-floor.

## Rules of thumb / cheat sheet

- **Default recovery recipe (approximate):** LR **1e-4**, **2 epochs**, **20% replay**, PEFT (LoRA/QLoRA). Start here, tune down further if retention still drops.
- **To *cause* forgetting on purpose (the lab):** LR **5e-4**, **5+ epochs**, **0% replay**. Watch the retention line fall while task rises.
- **Replay ratio:** 10–20% general instruction data. <5% is too little to matter; >30% dilutes task signal.
- **Epochs on small data (<5k ex):** 1–3. Treat 3 as a ceiling; 5+ is a forgetting experiment, not a training config.
- **LoRA LR** ~1e-4 to 2e-4; **full-FT LR** ~an order of magnitude lower. When forgetting appears, *halve the LR before touching anything else.*
- **Prefer PEFT over full-FT** for narrow tasks — it localizes and reverses damage by construction.
- **Retention suite = 3 legs:** general reasoning (MMLU slice), instruction-following (IFEval), safety/refusal probe. Small slice for the loop; full run only for the final report.
- **Selection criterion:** best task metric *subject to* "retention drop ≤ ~10% and safety drop ≤ ~2 pts." Reject over the floor even if task metric is the best.
- **Always keep the adapter** even when serving merged — it's your one-line rollback.
- **Ship artifact:** base / over-trained / recovered scorecard across ≥3 capabilities. That table *is* the proof.

## Connect to the lab

Week 4 lab steps 3–4 are this lecture made real. Step 3: deliberately over-train a variant at **5+ epochs, LR 5e-4** and run *both* `eval/task_eval.py` and `eval/retention.py` — you'll watch task accuracy rise while MMLU/IFEval/safety drop. That's the signature; burn it into memory. Step 4: retrain at **LR 1e-4, 2 epochs, 20% replay** (mix in a slice of a public SFT set) and show retention recovers while the task metric stays strong. Record all three (base / over-trained / recovered) into `models/MODEL_CARD.md` — the DoD requires the over-trained variant to *demonstrably* lose ≥~10% on a retention metric and the recovered one to restore it, with numbers for all three.

## Going deeper (optional)

- **EleutherAI `lm-evaluation-harness`** — the standard for running MMLU/IFEval/safety slices reproducibly. Repo: github.com/EleutherAI/lm-evaluation-harness. Search: "lm-evaluation-harness IFEval task".
- **Hugging Face TRL docs** (`SFTTrainer`, dataset mixing) for wiring replay as a data mix. Root: huggingface.co/docs/trl. Search: "TRL SFTTrainer mix datasets".
- **Hugging Face PEFT docs** — LoRA config, `merge_and_unload`, why adapters localize change. Root: huggingface.co/docs/peft.
- **Unsloth notebooks** — fast QLoRA runs you can re-run at different LR/epochs to reproduce the scissors curve cheaply. Search: "Unsloth Llama 3.1 QLoRA Colab notebook".
- **Concepts to search, not memorize the math:** "catastrophic forgetting neural networks" (the original phenomenon, McCloskey & Cohen; Kirkpatrick et al. "Elastic Weight Consolidation" for the regularization view), "rehearsal / experience replay continual learning", "instruction tuning safety regression fine-tuning". Read these for intuition; the engineering answer is still LR/epochs/replay.
- **Public SFT sets for replay:** search "Alpaca dataset", "Databricks Dolly 15k", "OpenAssistant OASST1", "allenai Tulu SFT mixture" — pick a small, license-clean slice.

## Check yourself

1. Your fine-tune's task accuracy went from 82% to 95% and training loss is at an all-time low. Your manager says ship it. What single number do you demand to see first, and why is the loss/accuracy pair insufficient?
2. Explain, in terms of steps through weight space, why LR 5e-4 for 5 epochs forgets far more than LR 1e-4 for 2 epochs on the same 800 examples. Give the rough ratio.
3. Why does adding *more of your task data* fail to fix forgetting, while adding 20% *general* data fixes it? Answer in terms of gradient direction.
4. You're using QLoRA, base weights frozen. A colleague says "adapters can't forget." Where are they right, where are they wrong, and what's the one-line rollback advantage adapters give you?
5. In a hyperparameter sweep selected purely on task accuracy, why will you systematically pick the most-forgetful checkpoint? State the fix.
6. After a benign task-only fine-tune, which general capability should you be *most* worried regressed, and why is it especially dangerous to miss?

### Answer key

1. A **retention scorecard** (general capabilities: MMLU slice / IFEval / safety refusal, before vs after). Task accuracy and loss only measure performance on your narrow distribution and are actually *positively* correlated with forgetting once you overfit — they are structurally blind to what the fine-tune degraded elsewhere. Loss down = better at reproducing your targets, not better overall.
2. Distance traveled through weight space ≈ `LR × number_of_steps`. Over-trained: `5e-4 × 500 steps = 0.25`; recovered: `1e-4 × 200 steps = 0.02`. That's ~**12× farther**, i.e. much more likely to leave the general basin where instruction-following and safety lived. LR and epochs are the dominant terms, not fine details.
3. More narrow task data makes *every* gradient point in the same direction (toward the "ticket basin"), so it accelerates drift, not fights it. General replay data makes ~1-in-5 gradients point back toward general behavior, so the optimizer's equilibrium becomes a point that satisfies both objectives — inside the general basin *and* good at the task. It's breadth of gradient direction, not volume, that matters.
4. Right: the frozen base means far fewer degrees of freedom and no permanent overwrite — forgetting is reduced and **reversible**. Wrong: a high enough LR over enough epochs still pushes the adapter delta into territory that degrades inference-time general behavior, so you must still measure retention. Rollback advantage: **detach the adapter and you get the exact base model back**, a one-line undo — so keep the adapter even when serving merged.
5. The most-forgetful (high-LR, many-epoch) checkpoint has essentially memorized the task distribution, so it tops the *task* leaderboard — selection on task metric alone therefore rewards forgetting. Fix: select on task metric **subject to a retention floor** (e.g. reject any variant that loses >~10% on IFEval or >~2 pts on safety refusal), even if its task score is best.
6. **Safety / refusal behavior.** It lives in the same shared weights and is one of the first things to regress under an aggressive narrow tune — even when the training data is entirely benign. It's dangerous to miss because a task dashboard is green while the model has silently become willing to comply with requests it used to refuse; that's a release blocker, not a footnote. Always re-run safety evals after tuning.
