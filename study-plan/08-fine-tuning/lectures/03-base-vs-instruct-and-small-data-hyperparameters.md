# Lecture 3: Base vs. Instruct Starting Points and Hyperparameters for Small Data

> Two engineering decisions make or break a small-data fine-tune before you ever look at a loss curve. First: *which checkpoint do you start from* — a raw **base** model or an **instruct/chat** model? That single choice determines how much data you need, who owns the chat template, and whether the model already knows how to follow an instruction at all. Second: *which knobs matter* when you have 300–5,000 examples instead of 300,000? Learning rate, epochs, warmup, schedule, and packing behave very differently in the small-data regime — the defaults that work for a big pretraining run will quietly overfit or corrupt a tiny SFT run. After this lecture you will be able to pick a starting checkpoint on purpose, set every hyperparameter that matters from a default plus a reason, read a diverging train/eval loss as the memorization signal it is, and use packing without silently poisoning your batches. It also plants the intuition for LoRA (a *low-rank delta* to the weights) that Week 2 turns into mechanism.

**Prerequisites:** Lecture 1 (prompt vs RAG vs fine-tune decision), Lecture 2 (chat templates, EOS, response-only loss masking), Phase 0 base vs instruct vs reasoning · **Reading time:** ~26 min · **Part of:** Phase 08 — Model Adaptation & Fine-Tuning, Week 1

---

## The core idea (plain language)

A fine-tune is a *continuation* of training, not a fresh start. So the most consequential decision you make is **where you continue from** — the checkpoint you load. The market gives you two shapes of the same architecture:

- A **base** (a.k.a. *pretrained* or *foundation*) model — e.g. `Qwen/Qwen2.5-7B`, `meta-llama/Llama-3.1-8B`. It has been trained on trillions of tokens of next-token prediction and nothing else. It knows language, code, facts, and reasoning latent in text, but it has **never been taught the concept of "a user asks, an assistant answers."** It has no chat template baked into its tokenizer config, no learned role tokens, no instinct to stop after answering. Prompt it with a question and it may continue your question, write the next ten questions, or drift into a Reddit thread.
- An **instruct/chat** model — e.g. `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`. It's that same base after an alignment stage (SFT + often preference tuning) that taught it a **specific chat template**, how to follow instructions, when to emit its end-of-turn token, and a baseline of helpful/safe behavior.

The practical asymmetry is the whole lecture in one sentence: **the instruct model already did most of the work you're about to pay for.** It knows the template (which Lecture 02 hammered on as a silent killer), it knows how to follow an instruction, and it knows how to stop. You're just nudging *style/format/domain*. So it needs **less data** and is the usual, sane SFT base. The base model is a blank operator — you must teach it the template *and* the behavior, so it needs **more data**, and you now **own** the template design from scratch.

The second half is the hyperparameters. With a tiny dataset the danger is not "won't learn" — it's "learns your 400 examples *by heart* in three passes and forgets how to generalize." Every knob below is really a knob about **how hard and how many times you push the weights**. Push too hard (LR too high) or too many times (too many epochs) on too little data and you get a memorizer that aced training and fails on anything new. The tell is mechanical and unmissable once you know it: **train loss keeps dropping while eval loss turns back up.**

---

## How it actually works (mechanism, from first principles)

### Why instruct needs less data: it starts closer to the target

Think of the model's behavior as a point in a very high-dimensional space, and your desired behavior as a target region. Pretraining lands the **base** model somewhere general. Instruction-tuning already *moved* the point most of the way toward "follows instructions, uses this template, stops cleanly" — the instruct checkpoint sits right at the edge of the target region. Your fine-tune only has to nudge it into your specific corner (your JSON schema, your tone).

The base model sits far away. Your fine-tune has to travel the *entire* distance: teach the template, teach instruction-following, *and* teach your task — all from your few hundred examples. That's why the same 500 rows that comfortably specialize an instruct model will underfit or destabilize a base model. Rough, approximate rule: reaching solid task behavior takes **hundreds to low thousands** of examples on an instruct base, and often **5–10× more** on a raw base (because a chunk of your data budget is spent re-deriving instruction-following the instruct model gave you for free).

```
base model  ●───────────────────────────────→ ◎ your task
            (must learn template + follow +      (target)
             stop + task: long trip, more data)

instruct    ················●──→ ◎ your task
model       (template/follow/stop already done;
             short nudge, less data)
```

### The template ownership fork (ties straight back to Lecture 02)

Lecture 02's killer #3 was "chat template mismatch." Your base-vs-instruct choice *decides who is responsible for that template*:

- **Instruct model → use its template, verbatim.** The Jinja template ships in the tokenizer config. You call `tokenizer.apply_chat_template(messages, ...)` and you are byte-for-byte consistent with how the model was aligned. Your job is to *not deviate* — same system-role convention, same special tokens (`<|im_end|>`, `<|eot_id|>`, `<end_of_turn>`), same EOS.
- **Base model → you invent and own a template.** A raw base often has **no** chat template at all (`apply_chat_template` throws or falls back to a generic one), and its "special" role tokens may not exist as single learned tokens. You must (a) choose a format — reuse a well-known one like ChatML rather than inventing a snowflake — (b) make sure the tokens you use are actually in the vocabulary (or add them and resize embeddings), and (c) set the template on the tokenizer so every example renders identically. Get this subtly wrong and every training row is off-distribution — the exact silent poisoning from last lecture, except now *you* authored the poison.

This is why, unless you have a specific reason, **you start from the instruct model**: it collapses the template problem from "design and validate a format" to "call the function that already exists."

### When the base model is actually the right call

Base is not a trap — it's a tool for specific jobs:

1. **Continued pretraining / heavy domain shift.** You have a large corpus (millions of tokens) of a domain the instruct model barely saw — legal filings, a low-resource language, internal code in a niche framework. You want to move the *foundation*, then instruction-tune on top. Instruct alignment would just be in the way.
2. **You want full control of the persona/format** and the instruct model's baked-in "As an AI assistant…" reflexes fight you. Starting from base lets you define behavior with no residue.
3. **The instruct model's alignment actively hurts** — e.g. it refuses legitimate content in your domain (security research, medical), or its verbose safety hedging pollutes your terse-JSON target.
4. **You have the data to afford it** — enough examples that re-teaching instruction-following is cheap relative to your goal.

For the Week 1 lab task (support-ticket → strict JSON on a few hundred rows), none of these hold. **Instruct is correct.** Reach for base only when you can name which of the above applies.

### Learning rate — the size of each step

The learning rate (LR) scales how far the optimizer moves the weights per update. It is the single most impactful hyperparameter, and the right range depends on *what fraction of the model you're training*:

- **LoRA / QLoRA: ~1e-4 to 2e-4 is a common start.** You're only training the small adapter matrices (Week 2), which are randomly initialized and contribute *nothing* at step 0 (one of the two matrices starts at zero, so the delta is zero). They can — and should — take large, fast steps to become useful. A "big" LR like 2e-4 here is normal and safe.
- **Full fine-tuning: roughly an order of magnitude lower, ~1e-5 to 5e-5.** Here you're moving *every* pretrained weight directly. Those weights already encode everything the model knows. A large step doesn't "learn faster" — it **overwrites** delicate pretrained structure, and you get catastrophic forgetting (Week 4) or an unstable, spiking loss. Small, careful steps preserve what's there while nudging behavior.

The mental model: **LoRA trains fresh, empty parameters (step boldly); full-FT edits precious existing ones (step gently).** Same reason you'd type freely into a blank file but edit a production config line-by-line.

*Failure mode if wrong:* LR too high → loss spikes, diverges, or NaNs; or it "converges" to a degenerate model that forgot general ability (great train loss, garbage generations, tanked retention). LR too low → loss barely moves across your whole (already short) run; you conclude "fine-tuning doesn't work" when you just under-stepped. On small data, when in doubt start at the *low* end of the LoRA range (1e-4) and only raise it if learning is sluggish.

### Epochs — how many times the model sees each example

An epoch is one full pass over your dataset. On a giant pretraining corpus you do <1 epoch (you never even finish the data). On a **small SFT set, 1–3 epochs is the whole budget**, and here's the arithmetic for why more is dangerous.

Say you have 400 examples. At 1 epoch the model sees each row once. At 10 epochs it sees each row *ten times*. With so few distinct examples and repeated exposure, the cheapest way for the optimizer to drive train loss down is to **memorize the specific tokens of those 400 answers** rather than learn the general mapping. That's overfitting, and on <5k examples it happens *fast* — often visibly by epoch 3–4.

The signal is precise and worth burning into memory:

```
loss
 │  train ─── keeps dropping (memorizing the 400 rows)
 │        ╲
 │         ╲___________
 │   eval ──╲          overfitting begins here:
 │           ╲        train↓ but eval↑  ← STOP / earlier checkpoint
 │            ╲______↗↗↗
 │                   ↗↗↗
 └──────────────────────────── epochs →
          1     2     3     4     5
```

While **both** train and eval loss fall, the model is genuinely learning. The moment **eval loss turns up while train loss keeps dropping**, the model has stopped generalizing and started memorizing — extra "learning" is now pure overfit. That inflection is your real stopping point. Practically: run eval every N steps, keep the checkpoint at the eval-loss minimum, not the last one.

*Safe default:* **2–3 epochs** for a few hundred to a few thousand examples; **1 epoch** if you're on the larger end (thousands) or on an instruct base that needs only a light nudge. *Failure mode if wrong:* too many epochs → memorization (aces train, fails held-out, may parrot exact training answers); too few → underfit (behavior didn't fully shift).

### Warmup — don't slam a cold optimizer

At step 0 the model has never seen your data and the adaptive optimizer (Adam/AdamW) has *no* estimate of gradient statistics. Hitting it with the full LR immediately can produce a huge, noisy first step that destabilizes training — especially with the large LoRA LRs above. **Warmup** ramps the LR linearly from ~0 up to the target over the first small slice of steps (commonly the first **3–10%** of total steps, or a flat ~10–50 steps on a short run).

On a short small-data run this matters *more* than on a long one, because a bad first few steps are a large fraction of your total steps — you don't have thousands of later steps to recover. Warmup buys the optimizer time to build stable gradient estimates before you let it move fast.

*Safe default:* warmup ratio **0.03–0.1** (or ~20–50 steps if your run is only a few hundred steps). *Failure mode if wrong:* no warmup + high LR → early loss spike/divergence; excessive warmup on a tiny run → you spend most of your few steps barely moving, effectively under-training.

### Cosine schedule — decay to a soft landing

Rather than hold LR constant, a **cosine schedule** starts at the target LR (after warmup) and smoothly decays it toward ~0 following a cosine curve over the run. Early on, large steps make fast progress; later, shrinking steps let the model *settle* into a good minimum instead of bouncing around it. On small data this "soft landing" reduces last-mile overfitting and gives more reproducible results than a constant LR that's still large at the final step.

```
LR │      ______
   │     /      ‾‾‾‾‾——___          ← warmup up, then cosine decay
   │    /                  ‾‾——__
   │   /                         ‾‾——___
   └──/──────────────────────────────────‾‾ steps →
     warmup            cosine decay to ~0
```

*Safe default:* **cosine** with the warmup above; linear decay is a fine alternative. *Failure mode if wrong:* constant high LR to the end → the model never settles, final checkpoint is noisier and often worse than an intermediate one.

### Sequence packing — fill the context, but mind the seams

Your examples vary in length. Naively, every sequence is padded to the batch's max length, and those pad tokens are pure wasted compute. **Packing** concatenates multiple short examples end-to-end into one sequence that fills the context window (e.g. glue several 200-token rows into one 2048-token sequence). Less padding → more real tokens per step → **meaningfully faster training**, sometimes 2–5× on datasets full of short rows.

The two risks are exactly the seams between packed examples:

1. **Cross-contamination.** If attention isn't reset at each boundary, tokens of example B can *attend to* example A — the model learns spurious dependencies ("after a billing ticket comes a refund ticket") that don't exist. Modern stacks avoid this with **block-diagonal / flash-attention packing** (a.k.a. position-id resetting or an attention mask that blocks cross-example attention). If your trainer supports proper packing masks, use them; if it just concatenates without a mask, packing is trading speed for contamination.
2. **Template / EOS boundaries.** Each packed example must still be a *complete, correctly-templated turn ending in EOS* (Lecture 02's killer #2). If packing strips or merges EOS tokens, examples bleed together and the model never learns to stop cleanly. Verify that within a packed sequence, each example's assistant turn still closes with its end token.

*Safe default:* **enable packing for speed on short-row datasets, but only with attention-boundary support** (e.g. TRL's `packing=True` on a recent version that does boundary-aware masking); and confirm EOS survives at each seam. For very small runs where training time isn't the bottleneck, you can skip packing entirely and remove the risk. *Failure mode if wrong:* naive concatenation → cross-example attention leakage and/or lost EOS → subtly degraded model that "trained fine."

### Preview: LoRA is a low-rank delta (intuition only)

You've seen "LoRA uses a big LR" and "you own less of the model." Here's *why*, at intuition level, to set up Week 2. A weight matrix is a big grid of numbers. Full fine-tuning learns a full-sized *change* to that grid — millions of new numbers per layer. LoRA bets that the change you actually need is **simple / low-rank**: it can be written as the product of two *skinny* matrices (one tall, one wide) whose shared inner dimension `r` is tiny (8, 16, 32). You freeze the original grid and train only those two small matrices. The layer's output becomes "original frozen output + a small learned correction." That correction is the **low-rank delta**. Because you're training a handful of fresh parameters instead of editing millions of precious ones, you can afford the bigger LR — and the adapter is a few megabytes, not gigabytes. That's the intuition; the shapes, `r`/`alpha`, and target modules are Lecture 04.

---

## Worked example

You have **600** support-ticket→JSON examples (500 train / 100 eval), and you're fine-tuning `Qwen2.5-7B-Instruct` with LoRA on a free T4.

- **Starting point:** instruct, not base. The template is ChatML and already in the tokenizer; you render with `apply_chat_template`, no template design needed. If you'd grabbed `Qwen2.5-7B` (base), you'd first have to set a ChatML template on the tokenizer and confirm `<|im_start|>`/`<|im_end|>` tokenize as single tokens — extra work and extra data, for no benefit on this task.
- **LR:** LoRA, so start **1e-4** (bottom of the 1e-4–2e-4 range, because data is small and you'd rather under-step than overwrite). If, after warmup, train loss is nearly flat by step ~50, bump to 2e-4.
- **Epochs:** **3**. Steps math: 500 examples, effective batch size (batch × grad-accum) = 8 → 500/8 ≈ **63 steps/epoch**, ~**189 steps** total.
- **Warmup:** 5% of 189 ≈ **~10 steps**. Cosine decay over the remaining ~179.
- **Eval cadence:** every ~20 steps → ~9 eval points. You *watch for the inflection*.

Now the loss you'd expect to see, and how you'd act:

```
step:      0    20    40    60    80   100   120   140   160   189
train:   3.6   1.9   1.1   0.7   0.5   0.38  0.30  0.24  0.20  0.17
eval:    3.6   1.8   1.0   0.66  0.55  0.52  0.53  0.57  0.63  0.70
                              ▲min eval ≈ step 100
```

Train loss marches down the whole way. Eval loss bottoms around **step ~100** (~epoch 1.6) and then climbs — the classic overfit inflection. **You ship the step-100 checkpoint, not the step-189 one**, even though step-189 has the prettiest train loss. If you'd run 8 epochs instead, train loss would reach ~0.05 and eval loss ~1.3 — a beautifully memorized, uselessly overfit model. The knobs didn't fail; watching the wrong number would have.

If you'd instead started from **base** `Qwen2.5-7B` with the same 500 rows: eval loss would bottom higher and later (the model is spending capacity learning the template and instruction-following, not just your task), and generations would show more "didn't stop / wrong format" errors — the symptom of under-teaching a from-scratch template on too little data.

---

## How it shows up in production

- **The "I used base by accident" week.** An engineer grabs `Llama-3.1-8B` (base) instead of `-Instruct`, reuses the same 800-row config, and ships. The model half-follows instructions, occasionally continues the user's prompt instead of answering, and won't reliably stop. Days are lost blaming data quality; the fix is one string: `...-Instruct`. **Always confirm the checkpoint suffix.**
- **The overtrained memorizer.** A 5-epoch run on 300 examples posts a gorgeous train loss and demos perfectly *on examples that happen to resemble training rows*. In production, held-out tickets it never saw get low-confidence, off-format answers. Root cause: nobody watched eval loss cross over. Cost: a re-train, plus lost trust in the whole fine-tuning effort.
- **The LR mismatch.** Someone copies a **full-FT** LR (2e-5) into a **LoRA** run and wonders why 3 epochs barely changed behavior (under-stepped fresh adapters). Or the reverse: copies a **LoRA** LR (2e-4) into a **full-FT** run and nukes the model's general ability — task metric up, retention crater (Week 4). The number was right for the *other* regime.
- **The packing contamination bug.** To speed up training, packing is turned on with an older/naive collator that concatenates without an attention mask. Metrics dip a couple of points for no obvious reason; generations occasionally blend two tickets' fields. It's invisible in the loss and only caught by inspecting a packed batch — and by knowing packing has seams.
- **The runaway generation.** Packing (or a base-template mistake) drops EOS at example boundaries. The model never learned to stop, so every inference runs to `max_new_tokens` — a large output-token cost and latency blowup, exactly Lecture 02's rambler resurfacing through a *hyperparameter/packing* door.

The through-line: on small data, **the failures are about pushing too hard, too many times, from the wrong starting point** — and most are visible in the eval-loss curve or a batch inspection, not in the train loss you're tempted to celebrate.

---

## Common misconceptions & failure modes

- **"Base and instruct are interchangeable; instruct is just base + niceties."** No. Instruct has a *learned chat template, instruction-following, and stop behavior* that your small dataset would otherwise have to re-teach. Different starting point → different data budget → different template ownership.
- **"More epochs = better model."** On <5k examples, more epochs past the eval-loss minimum = memorization. Better ≠ lower train loss.
- **"Lower train loss means it's learning."** Only while eval loss also falls. Once eval turns up, lower train loss is *overfitting*, not learning.
- **"One good LR works everywhere."** LR is tied to *how much of the model you train*. LoRA (~1e-4–2e-4) and full-FT (~1e-5–5e-5) differ by ~10×, for the mechanistic reason that one trains fresh params and the other edits pretrained ones.
- **"Warmup/schedule are pretraining-only frills."** They matter *more* on short runs, where a bad first step or a still-hot final step is a large fraction of your total steps.
- **"Packing is free speed."** It's free only with boundary-aware attention masking and preserved EOS. Naive packing trades a couple of quality points and a stop-token bug for the speed.
- **"Fine-tuning a base model gives me more control, so it's the safer default."** More control *and* more data, more template work, more ways to fail. Default to instruct; choose base deliberately for continued-pretraining / heavy-domain / alignment-in-the-way cases.

## Rules of thumb / cheat sheet

- **Default starting point: instruct/chat.** Use base only for continued pretraining, heavy domain shift, or when the instruct model's alignment fights your task — and when you have the data to re-teach instruction-following.
- **Instruct → use `apply_chat_template` verbatim. Base → you design/set the template** (reuse ChatML; verify special tokens tokenize as single tokens; set `tokenizer.chat_template`).
- **LR — LoRA/QLoRA:** start **1e-4**, up to **2e-4** if sluggish. **Full-FT:** **1e-5–5e-5** (≈10× lower). Small data → start at the low end.
- **Epochs: 1–3.** More overfits fast on <5k rows. Keep the **eval-loss-minimum** checkpoint, not the last.
- **Overfit alarm:** *train loss ↓ while eval loss ↑* = memorization → stop / use earlier checkpoint. Eval every N steps so you can see it.
- **Warmup: 3–10% of steps** (or ~20–50 steps on short runs). **Schedule: cosine** decay to ~0.
- **Packing: on for speed on short-row data, but only boundary-aware** (attention reset + EOS preserved). Inspect a packed batch. Skip it if training time isn't your bottleneck.
- **LoRA preview:** it learns a *low-rank delta* (two skinny matrices, rank `r`) on top of frozen weights — that's why the LR is high and the artifact is tiny.
- **(All numeric ranges are widely-used approximate starting points, not guarantees — tune against your eval set.)**

## Connect to the lab

This lecture is the reasoning behind the Week 1 lab's model choice and `train_sft.py` config. When the lab has you load `Qwen2.5-0.5B-Instruct` (then a 7-8B instruct model in Week 2), that's the *instruct default* from here — and `inspect_template.py` works precisely because the instruct tokenizer already carries the template. In `train_sft.py`, set LR ~1e-4 (LoRA), 2–3 epochs, cosine + small warmup, and **log eval loss every few steps so you can catch the train↓/eval↑ inflection** in your own curve. If you enable packing to speed the run, confirm EOS survives at the seams — the same EOS check from Lecture 02, now at packing boundaries.

## Going deeper (optional)

- **Hugging Face TRL docs — `SFTConfig` / `SFTTrainer`** (huggingface.co/docs/trl). Authoritative on `learning_rate`, `num_train_epochs`, `warmup_ratio`, `lr_scheduler_type`, and `packing` for *your* installed version.
- **Hugging Face Transformers docs — `TrainingArguments`** (huggingface.co/docs/transformers). What each schedule/warmup argument actually does under the hood.
- **PEFT docs — LoRA** (huggingface.co/docs/peft). Read for the Week-2 intuition preview: low-rank adapters, `r`, `alpha`, target modules.
- **Unsloth notebooks & docs** (github.com/unslothai/unsloth). Correct-by-default SFT/QLoRA configs (LR, epochs, packing) you can diff against — and clear guidance on base-vs-instruct notebook variants.
- **Axolotl** (github.com/axolotl-ai-cloud/axolotl). YAML configs that expose exactly these knobs (`sample_packing`, `warmup_steps`, `lr_scheduler`) for production runs; good to read even if you train with TRL.
- **LoRA paper (Hu et al., 2021), "LoRA: Low-Rank Adaptation of Large Language Models"** — read the *intuition* (low-rank update hypothesis), skip the math for now.
- **Search queries:** "TRL SFT packing attention mask cross contamination", "LoRA learning rate vs full fine-tuning learning rate", "eval loss increasing train loss decreasing overfitting fine-tune", "base vs instruct model for fine-tuning which to choose", "cosine learning rate schedule warmup small dataset".

## Check yourself

1. You have 700 examples and want a strict-JSON support-ticket classifier. Which starting checkpoint, and what specifically does that choice save you compared to the alternative?
2. Why is a LoRA learning rate (~1e-4–2e-4) roughly 10× higher than a sensible full-fine-tuning LR, in terms of *what* is being trained?
3. During training, train loss keeps falling but eval loss bottomed 40 steps ago and is now rising. What is happening, and which checkpoint do you ship?
4. Name one legitimate reason to start from a **base** model instead of an instruct model, and one extra piece of work that choice forces on you.
5. You enable packing and training gets 3× faster, but task accuracy drops ~2 points and outputs occasionally mix fields from two tickets. What are the two seam risks, and how do you confirm which bit you?
6. Why do warmup and a cosine schedule matter *more* on a 200-step small-data run than on a multi-thousand-step run?

### Answer key

1. **Instruct/chat model.** It already knows the chat template (so you just call `apply_chat_template`), already follows instructions, and already stops cleanly with EOS — so your 700 rows are spent only on your task/format, not on re-teaching all of that. Starting from base would force you to design and set the template *and* spend a chunk of those 700 examples re-deriving instruction-following, likely underfitting.
2. LoRA trains **small, freshly-initialized adapter matrices** that contribute nothing at step 0 and encode no prior knowledge — they can take big, fast steps safely. Full-FT **edits every pretrained weight directly**; those weights already store everything the model knows, so a large step overwrites delicate pretrained structure (instability / catastrophic forgetting). Fresh empty params → bold steps; precious existing params → gentle steps.
3. The model has passed the overfitting inflection: it's now **memorizing** the training rows (train↓) while generalization degrades (eval↑). Ship the checkpoint at the **eval-loss minimum** (~40 steps ago), not the current/last one.
4. Reasons (any one): continued pretraining / heavy domain shift on a large corpus; you want full control of persona with no baked-in "AI assistant" residue; the instruct model's alignment refuses or hedges on legitimate in-domain content; you have plenty of data. Extra work: **you must design/own the chat template** (choose a format like ChatML, ensure its special tokens are single vocab tokens or add+resize embeddings, set `tokenizer.chat_template`) — and you'll generally need more data.
5. (a) **Cross-contamination** — without an attention-boundary mask, one example attends to another, learning spurious cross-ticket dependencies (matches the "mixed fields" symptom). (b) **Template/EOS boundary loss** — EOS stripped/merged at seams so examples bleed together and stop-behavior degrades. Confirm by **inspecting a packed batch**: check that the attention/position setup resets at each example and that every packed example's assistant turn still ends in its EOS token.
6. On a short run, the **first few steps** (warmup's job — stabilizing a cold optimizer against a big first step) and the **final steps** (cosine's soft landing to avoid bouncing/overfitting) are each a *large fraction* of total steps. There are few later steps to recover from a bad early step or to settle after a still-hot final LR, so getting the ramp-up and decay right has outsized impact.
