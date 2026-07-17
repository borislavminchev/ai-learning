# Lecture 2: SFT Mechanics and the Three Silent Killers

> Supervised fine-tuning is deceptively simple: it's the same next-token loss the base model was pretrained with, just pointed at your (prompt, completion) pairs. The trap is that the training loop will happily run to completion, print a smooth, decreasing loss curve, and hand you back a *worse* model — because three details that live entirely in your data pipeline (not your training code) were silently wrong. This lecture takes you from "SFT is next-token prediction" to being able to open a tokenized batch, count masked vs. unmasked tokens, prove EOS is present, and confirm the chat template matches the model — the three checks that separate a fine-tune that works from one that "trained fine" and ships garbage. After this you will be able to debug an SFT data pipeline by inspection, not by superstition.

**Prerequisites:** tokenization basics, chat templates (Phase 0), what a loss curve is, comfort reading Python/PyTorch tensors · **Reading time:** ~22 min · **Part of:** Phase 08 — Model Adaptation & Fine-Tuning, Week 1

## The core idea (plain language)

A causal language model does exactly one thing: given a sequence of tokens, predict the probability distribution over the *next* token. Pretraining does this over trillions of tokens of the open internet. **Supervised fine-tuning (SFT) is the identical mechanism**, just run over a curated dataset of `(prompt, completion)` pairs where the completion is the behavior you want.

That's the whole trick. You are not teaching the model new facts (it's terrible at that — that's what RAG is for). You are nudging its next-token habits so that, conditioned on prompts *shaped like yours*, it produces completions *shaped like your targets*: strict JSON, a house tone, a terse classification label, a reasoning format.

Because SFT reuses the pretraining loss verbatim, the training loop is boringly robust. It will *not* crash when your data is subtly broken. Instead you get one of three silent failures:

1. **Loss masking wrong** → the model learns to reproduce prompts (parroting instructions) instead of answering them.
2. **EOS token missing** → the model never learns *when to stop*, so it rambles forever at inference.
3. **Chat template mismatch** → you train on a token layout the model has never seen, quietly poisoning every example.

None of these throws an error. All three produce a loss curve that looks fine. This lecture is about seeing them before they cost you a GPU-day and a confusing week.

## How it actually works (mechanism, from first principles)

### The loss, concretely

For a token sequence `x = [x_0, x_1, ..., x_{n-1}]`, the model outputs, at each position `t`, a probability for the *next* token `x_{t+1}`. Training minimizes the average negative log-likelihood (cross-entropy):

```
loss = -(1/N) * Σ_t  log P(x_{t+1} | x_0 ... x_t)
```

Two things matter for us. First, prediction is **shifted**: the logits at position `t` are scored against the label at position `t+1`. Frameworks handle this shift internally — you feed `input_ids` and `labels` of the *same* length and the trainer shifts by one. Second — and this is the lever for killer #1 — **`N` and the sum only run over positions whose label is not the ignore index.** In PyTorch's `CrossEntropyLoss`, that ignore index is `-100`. Any label position set to `-100` contributes *zero* to the loss and *zero* to the gradient.

So the label tensor is not just a copy of the input. It's a mask that says "compute loss *here*, stay silent *there*."

### Killer #1 — Loss masking (the label tensor)

Consider one training example, rendered and tokenized (IDs are illustrative):

```
tokens:  [<|user|>] [Classify] [ this] [ ticket] [<|assistant|>] [ {"cat"] [:"billing"} ] [<EOS>]
ids:        5001       210       12      888          5002           730        731         2
```

**Naive (wrong) labels — loss on everything:**

```
labels:     5001       210       12      888          5002           730        731         2
```

Here the model is graded on predicting the *prompt* tokens too. It learns "after `<|user|>` comes `Classify this ticket`." That's the parroting objective. The concrete symptom: your eval loss looks *great* (predicting the prompt is easy — you wrote it), but at inference the model echoes the instruction back, or blends instruction text into its answer. Loss lies; generations rot.

**Response-only (correct) labels — prompt masked to `-100`:**

```
labels:     -100      -100     -100     -100         -100           730        731         2
```

Now the sum runs over exactly 3 positions: the two completion tokens and EOS. The model is graded *only* on producing the answer given the prompt. That is the behavior you actually want.

Numerically the difference is stark. Say the prompt is 40 tokens and the completion is 10. With whole-sequence loss you average over ~50 positions, ~80% of which are trivially predictable prompt tokens — so the loss number is dominated by easy tokens and looks low and stable regardless of how well the model answers. With response-only masking you average over ~10 positions, all of them the hard part. The masked loss is a *sharper, honest* signal, and it's what optimizes the behavior you care about.

Rule of thumb (approximate): if your loss starts suspiciously low (well under ~1.0) on the first step for a task the model hasn't learned yet, suspect that prompt tokens are dominating the average — i.e. masking is off.

### Killer #2 — The EOS token

The model learns *when to stop* the same way it learns everything else: by seeing the stop token as a training target. If your completion tokenizes without a trailing EOS, then the token sequence you train on **never contains a high-probability path to "stop here."** At inference the model, having never been rewarded for emitting EOS after a well-formed answer, keeps generating — repeating, drifting, or dumping a second hallucinated turn until it hits `max_new_tokens`.

The subtlety: whether EOS ends up in your sequence depends on *how the text was produced*. Two failure paths:

- You built the string by hand (`prompt + completion`) and never appended the tokenizer's EOS.
- You used `apply_chat_template` but with `add_generation_prompt=True` (which is for *inference* — it opens an assistant turn but does not close it) instead of the training rendering that closes the turn with EOS.

And EOS is model-specific. It might render as `</s>` (Llama 2), `<|eot_id|>` / `<|end_of_text|>` (Llama 3), `<|im_end|>` (Qwen/ChatML), or `<end_of_turn>` (Gemma). You verify by *tokenizing and looking at the last IDs*, not by trusting that "the template probably added it."

### Killer #3 — The chat template

An instruct model was aligned on a *specific* wire format: exact role markers, exact special tokens, exact whitespace. `tokenizer.apply_chat_template` renders your `messages` list into that exact format because the template (a Jinja string) ships *inside the tokenizer config* for that model. Use it and you are byte-for-byte consistent with how the model was trained. Hand-build the format — "it's just `User:` / `Assistant:`, right?" — and you get a plausible-looking string that maps to *different token IDs* than the model expects, and every single training example is subtly off-distribution.

Different families genuinely differ. A minimal ChatML-style turn (Qwen and others):

```
<|im_start|>system
You are a router.<|im_end|>
<|im_start|>user
Classify this ticket.<|im_end|>
<|im_start|>assistant
{"category":"billing"}<|im_end|>
```

Llama 3 uses `<|start_header_id|>role<|end_header_id|>` and closes turns with `<|eot_id|>`. Gemma uses `<start_of_turn>user` / `<start_of_turn>model` and `<end_of_turn>`, and has *no* system role — put system content in the first user turn or the template errors. These are not cosmetic. `<|im_end|>` is a single learned special token; the literal string `<|im_end|>` typed into a base model's tokenizer might split into several ordinary tokens the model has never seen used that way. Same characters, different IDs, corrupted training.

The reason this is *silent*: the trainer tokenizes whatever string you give it and computes a valid loss on it. There is no "this doesn't match my template" check anywhere in the stack. You only find out at eval — or in production.

### How TRL masks the response (and how to confirm it)

You do not hand-build the label tensor in practice; TRL's `SFTTrainer` does it. But *which mechanism* depends on your TRL version, and knowing the difference is the point:

- **`DataCollatorForCompletionOnlyLM`** (older/stable TRL): you pass a `response_template` string (e.g. the assistant marker). The collator finds that marker in each tokenized example and sets every label *before* the response to `-100`. Gotcha: the response template must tokenize *identically* to how it appears mid-sequence, or the collator fails to find it and (depending on version) masks the whole example or none of it.
- **`assistant_only_loss=True`** in `SFTConfig` (newer TRL): relies on the chat template exposing which spans are assistant-generated (via `{% generation %}` markers in the template's Jinja) and masks everything else automatically. Cleaner, but requires the tokenizer's template to support generation masking.
- **`completion_only` / dataset with separate `prompt` and `completion` fields**: TRL masks the prompt field and computes loss only on completion. Works when your data is already split into the two fields.

The single most important habit: **do not trust the flag — confirm the mask.** Pull one collated batch and count. If you set `assistant_only_loss` but your template lacks generation markers, TRL may silently train on the whole sequence (killer #1 again). The confirmation code:

```python
batch = data_collator([train_dataset[0]])
labels = batch["labels"][0]
n_masked   = (labels == -100).sum().item()
n_trained  = (labels != -100).sum().item()
print(f"masked={n_masked}  trained={n_trained}")
# decode only the tokens we actually train on:
trained_ids = batch["input_ids"][0][labels != -100]
print(tokenizer.decode(trained_ids))
```

If `n_trained` roughly equals your completion length (plus EOS) and the decoded text is *just the assistant answer ending in the EOS token* — you're correct. If the decode contains the system prompt or user question, your mask is wrong no matter what the flag said.

## Worked example

Take one row: system = "You are a ticket router." (8 tokens rendered), user = "My card was charged twice." (7 tokens), assistant target = `{"category":"billing","priority":"high"}` (12 tokens), plus role markers and EOS. Rendered with `apply_chat_template(msgs, tokenize=False)` for a ChatML model, then tokenized, suppose the full sequence is **41 tokens** and the assistant span (answer + closing `<|im_end|>`) is **13 tokens**.

- **Whole-sequence loss:** averaged over 41 positions. ~28 of those are system/user/marker tokens the model predicts easily. First-step loss might read ~0.9 and barely move — because 68% of the average is trivial. You'd conclude "converged fast," but the model is mostly being graded on parroting.
- **Response-only loss:** averaged over 13 positions, all in the answer. First-step loss might read ~3.5 and drop to ~0.4 over training — a real learning curve for the real task.

Now confirm EOS: `tokenizer.convert_ids_to_tokens(batch["input_ids"][0][-1])` should print `<|im_end|>` (id 151645 for Qwen2.5, for example). If it prints a JSON character or a space, the turn wasn't closed — killer #2 is live even though the answer text looks complete.

Run the confirm snippet: you want `trained≈13`, and `tokenizer.decode(trained_ids)` to print exactly `{"category":"billing","priority":"high"}<|im_end|>` — no system text, no user text. That single print is worth more than any amount of staring at the loss curve.

## How it shows up in production

- **The parrot model (killer #1).** You ship a support-ticket classifier. In testing it "works," but under real load ~15% of responses begin by restating the ticket or the instruction before (or instead of) the JSON. Your JSON parser fails, retries fire, p95 latency spikes, and cost per resolved ticket climbs — all traceable to labels that weren't set to `-100`. The tell during training was a loss that looked *too good, too fast*.
- **The rambler (killer #2).** Answers are correct but never terminate; generations run to `max_new_tokens=512` every time. You pay for ~500 tokens when the answer needed 20 — a **~25x** output-token cost blowup and a latency regression, because the model was never trained to emit EOS after a complete answer.
- **The quietly-poisoned run (killer #3).** A teammate "simplified" data prep to a hand-built `f"### Instruction:\n{q}\n### Response:\n{a}"` string for a Qwen instruct base. Loss trains smoothly. Task accuracy is *worse* than the un-tuned base, and nobody can explain it — because every example trained the model on a role format it had never been aligned to. Weeks lost blaming hyperparameters.
- **The version-drift surprise.** You upgrade TRL; `DataCollatorForCompletionOnlyLM` is deprecated in favor of `assistant_only_loss`. You flip the flag, but the new tokenizer template lacks generation markers, so masking silently reverts to whole-sequence. The only thing that catches it is the masked/unmasked count check in CI.

The through-line: **every one of these is invisible in the loss curve and visible in a 5-line inspection of a tokenized batch.** Make the inspection a required step, not an afterthought.

## Common misconceptions & failure modes

- **"Lower loss means better model."** False when the average is polluted by masked-out prompt tokens or when EOS is missing. Loss measures token prediction on *your rendering*, not task quality. Always pair it with generations on held-out prompts.
- **"apply_chat_template always adds EOS."** No. `add_generation_prompt=True` opens an assistant turn for *inference* and deliberately does not close it. The *training* rendering (full messages including the assistant turn) is what appends the turn-closing/EOS token. Know which mode you're in.
- **"The response template is just a string, any close-enough marker works."** `DataCollatorForCompletionOnlyLM` matches on *token IDs*, and a marker can tokenize differently depending on the preceding character/whitespace. If the collator can't find it, masking breaks silently.
- **"Base and instruct are interchangeable starting points."** Base models have no chat template and no learned special role tokens — you own the format and need more data. Instruct models come pre-aligned to a template; use *their* template.
- **"Packing multiple samples to fill context is free."** Packing concatenates short examples to reduce padding waste, but if you don't reset attention across boundaries (or the template boundary/EOS is off), one example's tokens attend to another's — cross-contamination. Verify EOS separates packed samples.
- **"If the flag is set, masking is correct."** The flag expresses intent; only the label tensor expresses reality. Confirm the count.

## Rules of thumb / cheat sheet

- **Always render, then tokenize, then inspect.** Never train on a string you haven't printed and decoded.
- **Prompt tokens → `-100`.** Confirm `n_trained ≈ completion_length (+1 for EOS)`, and that decoding the unmasked tokens shows *only* the answer.
- **EOS is mandatory and model-specific.** Print the last token ID of a training example; it must be the model's turn/end token (`<|im_end|>`, `<|eot_id|>`, `<end_of_turn>`, `</s>` — depending on family).
- **Use `tokenizer.apply_chat_template`. Never hand-build role formats** for an instruct model.
- **Match the base's family exactly:** ChatML (Qwen), Llama-3 header tokens, Gemma turns (no system role). Load the *model's own* tokenizer, don't reuse another's.
- **A/B the mask once, always.** Train identical runs with and without response-only masking; keep the masked one, keep the numbers.
- **First-step loss sanity:** suspiciously low (<~1.0) on an unlearned task → suspect polluted/whole-sequence loss.
- **Bake the masked/EOS/template checks into CI or a `inspect_template.py` script** you run before every training job.

## Connect to the lab

This lecture is the theory behind Week 1's `inspect_template.py` and the masking A/B in `train_sft.py`. In the lab you'll render your ticket-router data with `apply_chat_template(tokenize=False)`, print it, confirm the EOS after the assistant turn, then print the masked-vs-unmasked label counts for one example — exactly the confirm snippet above. Then you train the 0.5B model twice (response-only masking on vs. off) and write the eval-loss and generation difference into your README. If your "masking on" run doesn't win — or you can't explain why — this lecture's confirm step is where you find the bug.

## Going deeper (optional)

- **Hugging Face TRL docs — `SFTTrainer` / `SFTConfig`** (huggingface.co/docs/trl). The authoritative source for the current masking mechanism in *your* installed version — check it, versions move.
- **Hugging Face blog: chat templates** (search: "Hugging Face chat templates blog"). Why templates live in the tokenizer and how `apply_chat_template` works.
- **Transformers docs — chat templating & `apply_chat_template`** (huggingface.co/docs/transformers). Covers `add_generation_prompt`, generation masking markers, and per-model template differences.
- **TRL `DataCollatorForCompletionOnlyLM` API reference** (in the TRL docs) — the `response_template` matching gotchas.
- **Search queries:** "TRL assistant_only_loss", "DataCollatorForCompletionOnlyLM response template not found", "why does my fine-tuned model not stop generating EOS", "chatml vs llama3 template special tokens".
- **Unsloth notebooks** (github.com/unslothai/unsloth) — well-maintained, correct-by-default SFT examples you can diff against your pipeline.

## Check yourself

1. Why does setting prompt-token labels to `-100` change *both* the loss value and the gradient, and what task-quality symptom appears if you forget it?
2. Your eval loss is a smooth, low, decreasing curve, but generations are worse than the base model. Name two distinct root causes consistent with that observation.
3. A completion looks complete in the JSONL file but the model rambles at inference. What single check on the *tokenized* sequence would confirm the cause, and what are you looking for?
4. You hand-build `"User: {q}\nAssistant: {a}"` for a Qwen instruct model and training loss looks normal. Explain precisely why the resulting model can be *worse* than the un-tuned base.
5. You set `assistant_only_loss=True` and want to prove masking is actually happening. What do you compute, and what two conditions confirm it's correct?
6. Why is a *lower* response-only loss (over ~10 answer tokens) a more trustworthy signal than a lower whole-sequence loss (over ~50 tokens including the prompt)?

### Answer key

1. `-100` is the ignore index in cross-entropy: those positions are dropped from the summed loss *and* from the denominator `N`, so they contribute no gradient and don't dilute the average. Forget it and the model is optimized to reproduce prompt tokens too — the symptom is **parroting/echoing** the instruction or blending it into the answer, while loss looks fine.
2. (a) **Prompt tokens not masked** — loss is dominated by trivially-predictable prompt tokens so it looks low, while the model actually learns to parrot. (b) **Chat template mismatch / hand-built format** — every example trained on token IDs the model was never aligned to, degrading behavior even as the (self-consistent) loss falls. (Missing-EOS also fits if the symptom is specifically non-termination.)
3. Decode/inspect the last token ID(s) of the tokenized training example (e.g. `convert_ids_to_tokens(input_ids[-1])`). You're confirming the sequence **ends in the model's EOS/turn-end token** (`<|im_end|>`, `<|eot_id|>`, etc.). If it ends in a normal content token, EOS was never trained and the model never learned to stop.
4. Qwen instruct was aligned on the ChatML format with learned special tokens like `<|im_start|>`/`<|im_end|>`. The strings `"User:"`/`"Assistant:"` tokenize to ordinary tokens the model has never seen used as role markers, so every example is off-distribution: you're overwriting well-aligned behavior with training signal on a format the model doesn't recognize, which can drag task performance below the untouched base.
5. Pull one collated batch and count `labels == -100` vs `labels != -100`; also decode the unmasked tokens. Correct when (a) the unmasked count ≈ the assistant completion length (+1 for EOS), and (b) decoding the unmasked tokens yields **only the assistant answer (ending in EOS)** — no system or user text.
6. The whole-sequence average is dominated by ~40 easy, self-authored prompt tokens, so the number reflects "can the model copy the prompt" far more than "can it produce the answer." The response-only average is computed purely over the hard target tokens, so its movement tracks the behavior you actually optimize for — it's a sharper, less-diluted signal.
