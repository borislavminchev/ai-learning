# Lecture 6: VRAM Math and PEFT vs Full Fine-Tuning Budgets

> The single most expensive mistake in a first fine-tuning project is renting the wrong GPU: you provision an A100-80GB "to be safe" and burn money you didn't need, or you grab a free T4, launch full fine-tuning, and watch it OOM three minutes into the first epoch. Both failures come from *not doing the arithmetic first*. This lecture teaches you to compute a training VRAM budget by hand — on a whiteboard, before you touch a cloud console — and land within ~20% of the real peak. After it you can break any training run into its four memory terms (weights, gradients, optimizer states, activations), explain from first principles *why* full fine-tuning balloons to an order of magnitude more memory than the model itself, derive the QLoRA budget that lets an 8B model train on a 15GB card, and write the estimator function the Week 2 lab asks for.

**Prerequisites:** Phase 0 VRAM & quantization (bytes-per-param, KV cache), Week 1 SFT mechanics (loss, gradients), LoRA intuition (low-rank deltas, Lecture 4) · **Reading time:** ~28 min · **Part of:** Phase 08 (Model Adaptation & Fine-Tuning) Week 2

---

## The core idea (plain language)

At **inference** you pay for essentially one thing: the weights, resident in VRAM (plus a KV cache that grows with context and concurrency). At **training** you pay for the weights *and three more things*, and those three are what make training memory explode. Every gradient-descent step needs, for each parameter you are *updating*:

1. the **weight** itself (to run the forward pass),
2. a **gradient** for it (computed in the backward pass),
3. the optimizer's **state** for it (Adam keeps a running mean and variance), and
4. the **activations** cached during the forward pass so the backward pass can use them.

The whole lecture is one accounting exercise: add up bytes for those four terms. The trick that makes fine-tuning affordable — **PEFT** (Parameter-Efficient Fine-Tuning), of which **LoRA/QLoRA** is the dominant flavor — is a *structural* one: freeze the base model so it has no gradients and no optimizer state, and train a tiny set of extra "adapter" parameters instead. Because terms 2 and 3 only exist for the parameters you actually update, and PEFT updates well under 1% of them, those two terms nearly vanish. That is the entire reason a model that needs ~130 GB to fully fine-tune can be QLoRA-tuned in ~10 GB.

The reference identity from Phase 0:

```
weights VRAM (bytes) = number of parameters × bytes per parameter
```

with the bytes-per-param you must have memorized: **bf16/fp16 = 2, int8 = 1, 4-bit ≈ 0.5**. Training just stacks three more terms on top of that one line. Learn to write those four lines from memory and you can size any run.

---

## How it actually works (mechanism, from first principles)

### The four terms, one at a time

Let `P` = parameter count (e.g. 8B = 8×10⁹). Reason in **bytes-per-parameter**, because that keeps the arithmetic trivial and scales to any model size.

**1. Weights.** The forward pass needs every weight resident. In bf16 that's **2 bytes/param**. Quantize the frozen base to 4-bit (QLoRA) and it's **~0.5 bytes/param**. This term is fixed the moment you load the model.

**2. Gradients.** Backprop produces one gradient per *trainable* parameter, stored at compute precision — typically **2 bytes/param** (bf16). Crucial word: *trainable*. Frozen parameters get no gradient. In full FT every param is trainable, so this is another 2 bytes/param across all of `P`. In LoRA only the adapter params are trainable, so this term is ~0 for the base.

**3. Optimizer states.** This is the term beginners forget, and it's usually the *biggest single term* in full FT. Adam/AdamW keeps **two** running statistics per trainable parameter — a first moment (mean of gradients, `m`) and a second moment (uncentered variance, `v`) — and for numerical stability these are kept in **fp32 (4 bytes each) = 8 bytes/param**. Mixed-precision training *also* keeps an fp32 **master copy** of the weights (another 4 bytes/param) so tiny updates don't get lost to bf16 rounding. So Adam's bookkeeping alone is **12 bytes/param** (8 states + 4 master) on top of the bf16 weights and gradients.

Add it up for **full FT with mixed-precision AdamW**:

```
   weights (bf16)          2
 + gradients (bf16)        2
 + Adam m + v (fp32)       8
 + fp32 master weights     4
 --------------------------------
   = 16 bytes / parameter   (static; activations not yet counted)
```

That **16 bytes/param** is the number to burn into memory. It means full FT's fixed cost is **~8× the bf16 weight size** before a single activation is stored. The industry rule of thumb "full FT needs ~12–20× the model size in VRAM" is exactly this: read it as **~12–20 GB of VRAM per billion parameters** — the 16 bytes/param static floor, plus activations and fragmentation on top *(approximate; it moves with batch size, sequence length, and whether you use an 8-bit optimizer)*.

**4. Activations.** During the forward pass each layer's outputs are cached so the backward pass can compute gradients. Activation memory scales with **batch × sequence length × hidden size × number of layers** — it does *not* scale with the optimizer, so it's the one term that's roughly the same whether you do full FT or LoRA. It's also the term you can trade away: **gradient checkpointing** keeps only a few layer-boundary activations and *recomputes* the rest during backprop, cutting activation memory dramatically (roughly from O(layers) toward O(√layers)) at the cost of ~20–30% more compute. This is why "enable gradient checkpointing" is the first knob you turn when you OOM.

```
      FULL FINE-TUNING (per param, bytes)          QLoRA (where the bytes go)
   ┌───────────────────────────────────┐        ┌──────────────────────────────┐
   │ weights   2  ██                    │        │ base weights ~0.5 █ (frozen,  │
   │ grads     2  ██                    │        │                       4-bit)  │
   │ Adam m+v  8  ████████              │        │ adapter (all 4 terms) ▏ <1%   │
   │ master    4  ████                  │        │ activations       ███ (same)  │
   │ ───────────────────────────       │        │                               │
   │ = 16 B/param  + activations        │        │ ≈ 0.5 B/param + small + acts  │
   └───────────────────────────────────┘        └──────────────────────────────┘
```

### Why PEFT collapses terms 2 and 3

LoRA freezes the base weights `W` and learns a low-rank update `ΔW = B·A`, where `A` is `r×d_in` and `B` is `d_out×r` for a small rank `r` (8–64). Only `A` and `B` are trainable. For one linear layer with dimensions `d_in, d_out`, the adapter adds `r × (d_in + d_out)` parameters instead of `d_in × d_out`.

Count them for an 8B model (≈32 layers, hidden 4096, MLP up-projection ≈14336) with `r=16` on attention **and** MLP projections:

- Attention (q, k, v, o), each 4096→4096: `16 × (4096+4096) = 131,072` params/matrix × 4 × 32 layers ≈ **16.8M**.
- MLP (gate, up: 4096→14336; down: 14336→4096): `16 × (4096+14336) = 294,912` params/matrix × 3 × 32 layers ≈ **28.3M**.
- **Total ≈ 45M adapter params ≈ 0.56% of 8B.**

Now re-run the four terms *for the adapter only*: weights 2 B, grads 2 B, Adam+master 12 B ≈ **16 bytes/param — but over 45M params, not 8B**. That's `45M × 16 ≈ 0.72 GB`, and with a paged 8-bit optimizer it's smaller still. The base's gradients and optimizer states — the terms that dominated full FT — **do not exist**, because the base is frozen. That structural fact, not a clever kernel, is why PEFT is cheap. It's also why "train <1% of params" and "cut training memory ~10×" are the same sentence.

### QLoRA: quantize the frozen base too

QLoRA adds one more move: since the base is frozen and never receives gradients, store it in **4-bit NF4 (~0.5 bytes/param)** instead of bf16. The adapters stay in bf16 (they're trainable and tiny — 4-bit can't represent a `+0.0003` gradient step without rounding it to zero). The **paged optimizer** lets optimizer state spill to CPU RAM during transient spikes (e.g. the optimizer step at the end of an accumulation window) so you don't OOM at the worst possible moment.

The QLoRA training budget, as a formula you can recite:

```
QLoRA VRAM ≈ 0.5·P                          (4-bit frozen base)
           + ~16 B × P_adapter              (adapter weights+grads+opt; P_adapter < 1% of P)
           + activations × (checkpointing factor)
           + ~1–2 GB runtime/CUDA/fragmentation overhead
```

For an 8B model the base is `8 × 0.5 = 4 GB`, the adapter side is well under 1 GB, and with gradient checkpointing on and a modest sequence length the activations are a few GB — total lands around **7–12 GB**, which is why a free T4 (15 GB) or any 16 GB card runs it.

---

## Worked example

This is the estimator the lab builds (`scripts/vram_estimate.py`). Do it for an **8B** and a **3B** model, in all three modes. Work in **GB per billion params** so the arithmetic is mental.

### 8B model

**Full FT (mixed-precision AdamW):** 16 bytes/param static.
```
static = 8B × 16 bytes = 128 GB
+ activations (batch 4, ~2k seq, checkpointing) ≈ 10–25 GB
≈ 140–155 GB  → needs 2× A100-80GB (or FSDP/DeepSpeed ZeRO sharding)
```

**LoRA (16-bit frozen base, r=16 attn+MLP):**
```
base (bf16)        8B × 2   = 16.0 GB
adapter (all 4 terms, ~45M) ≈  0.7 GB
activations (checkpointing)  ≈  2–5 GB
overhead                     ≈  1–2 GB
≈ 20–24 GB  → fits a single 24 GB card (RTX 4090 / A10 / L4-ish, tight)
```

**QLoRA (4-bit NF4 base, paged 8-bit optim):**
```
base (4-bit)       8B × 0.5 =  4.0 GB
adapter side (paged 8-bit)  ≈ 0.3–0.6 GB
activations (checkpointing)  ≈  2–5 GB
overhead                     ≈  1–2 GB
≈ 7–12 GB  → fits a free T4 (15 GB) with headroom
```

**Inference (bf16), for contrast:** `8B × 2 = 16 GB` weights + KV cache. At 4-bit for serving: `8 × 0.5 = 4 GB` + KV. Notice inference has *none* of the gradient / optimizer / backprop-activation cost — that's the whole gap between "runs it" and "trains it."

### 3B model

| Mode | Weights | +grad/opt | +activations/overhead | **Total** | Fits |
|---|---:|---:|---:|---:|---|
| Full FT | 3×2 = 6 | +3×14 = 42 → 48 static | +~6–15 | **~54–63 GB** | A100-80GB (not 40GB comfortably) |
| LoRA (bf16 base) | 6 | +<0.5 | +~3–5 | **~9–12 GB** | 16 GB card |
| QLoRA (4-bit base) | 1.5 | +<0.5 | +~3–5 | **~5–7 GB** | 8 GB card |

The 3B row is the punchline: full FT of a *small* 3B model still wants ~50–60 GB (a datacenter GPU), while QLoRA fits on a laptop-class 8 GB card. Same model. The difference is entirely terms 2 and 3.

A minimal estimator (the shape the lab wants):

```python
def train_vram_gb(params_b, mode, seq=2048, batch=4, ckpt=True):
    """Rough training-VRAM estimate in GB. Calibrate `act` to your run."""
    P = params_b  # billions
    if mode == "full":    base, per_param = P * 2,   16    # bf16 base + 16 B/param static
    elif mode == "lora":  base, per_param = P * 2,   0.0   # frozen bf16 base
    elif mode == "qlora": base, per_param = P * 0.5, 0.0   # frozen 4-bit base
    static  = base + per_param * P                          # weights + (grad+opt for full FT)
    adapter = 0.0 if mode == "full" else 0.01 * P * 16      # ~<1% params × 16 B, in GB terms
    act     = (0.3 if ckpt else 1.0) * (batch * seq / 8192) # rough; calibrate to your run
    return static + adapter + act + 1.5                     # +overhead
```

Treat the activation term as a *calibration knob*: run once, read `torch.cuda.max_memory_allocated()`, and fit it. The spine's Definition of Done — within ~20% of measured peak — is achievable precisely because the weights/grad/opt terms are exact arithmetic; only activations wobble.

---

## How it shows up in production

- **The rent-the-right-GPU decision.** "Can we fine-tune Llama-3.1-8B on the L4 we already have?" is a two-line calculation, not a Slack thread. Full FT = 128 GB static → no. QLoRA = ~10 GB → yes, on a 24 GB L4 with room for a bigger batch. That single calc is the difference between a $0.50/hr job and provisioning multi-GPU A100s.

- **The end-of-step OOM.** A run that trains fine for minutes and dies on the optimizer step is the optimizer-state term spiking (Adam allocates/touches `m` and `v` at the step). The fixes map directly to the budget: **paged optimizer** (spill state to CPU), **8-bit optimizer** (halve the 8 B/param states), **gradient checkpointing** (shrink activations), smaller **batch** with **grad-accumulation** (same effective batch, lower peak).

- **Full FT is reserved, not default.** Because terms 2+3 make full FT ~8× the weight cost, industry uses it only for **continued pretraining** (new domain, new language, new base capabilities) or **truly reshaping a model's behavior** at scale. For adapting an instruct model to a task — format, tone, a classifier, distillation — **PEFT is the default, full stop.** Adapters are also **MBs on disk**, hot-swappable, and mergeable; full-FT checkpoints are the entire model, every time, which also drives your storage and deployment bill.

- **"It fit at train time but OOMs at serve time" (and vice versa).** They're different budgets. Training peak is dominated by optimizer + activations; serving peak is weights + **KV cache** (which grows with context × concurrency, Phase 0). Size each separately. A merged bf16 model that *trained* in 10 GB via QLoRA needs its full ~16 GB (8B) resident to *serve* — because serving loads the un-quantized merged weights unless you re-quantize for deployment (Week 3).

---

## Common misconceptions & failure modes

- **"Weights × 2 is my training budget."** That's inference. Training adds gradients (2 B), optimizer states (8 B), and an fp32 master (4 B) per *trainable* param, plus activations. For full FT that's 8× the weights before activations. Sizing to weights-only is the #1 cause of the first-epoch OOM.

- **"LoRA saves memory because the adapters are small."** Half-right. The adapters *are* small, but the real saving is that **the frozen base has no gradients and no optimizer state** — you delete the two biggest terms for 99%+ of the parameters. The adapter's own footprint is almost a rounding error.

- **"QLoRA quantizes everything, so quality tanks."** QLoRA quantizes only the **frozen base** to 4-bit NF4; the trainable adapters stay in bf16, and the gradient math runs in bf16. The base was going to be frozen anyway, so 4-bit storage of it costs little for training. (Serving quality is a separate question — you merge into bf16, *then* choose a serving quant, Week 3.)

- **"Gradient checkpointing saves optimizer memory."** No — it only shrinks **activations**, by recomputing them in the backward pass, at ~20–30% more compute. It does nothing to the optimizer-state term; use an 8-bit/paged optimizer for that.

- **"Bigger `r` needs a lot more VRAM."** `r` scales only the adapter term, which is already <1% of the budget. Going `r=16 → 64` adds tens of MB, not GB. (`r` matters for *overfitting* and capacity, not VRAM — a quality knob, not a memory one.)

- **"Merge the adapter into the 4-bit base to save memory."** Merging into a 4-bit base compounds quantization error and misapplies the adapter's correction → garbage. Merge into **fp16/bf16**, then quantize the merged model for serving (Lecture 5 / Week 3 rule).

---

## Rules of thumb / cheat sheet

*(All numbers approximate — measure your own run.)*

- **The four terms (bytes/param):** weights (bf16 **2**, 4-bit **0.5**) · gradients **2** · Adam states **8** + fp32 master **4** · activations (variable, checkpointable).
- **Full FT static floor:** **16 bytes/param** = **~12–20 GB per billion params** with activations. ≈ 8× the bf16 weight size.
- **LoRA (bf16 base):** ≈ **weights (2 B/param) + <1 GB adapter + a few GB activations**. An 8B fits ~24 GB; a 3B fits ~16 GB.
- **QLoRA (4-bit base):** ≈ **0.5 B/param + <1 GB adapter + a few GB activations + ~1–2 GB overhead**. An 8B fits a **15 GB T4**; a 3B fits an **8 GB** card.
- **Inference:** bf16 ≈ **2 B/param + KV cache**; 4-bit ≈ **0.5 B/param + KV cache**. No grad/opt/backprop-activation terms.
- **OOM triage, in order:** enable **gradient checkpointing** → **8-bit/paged optimizer** → lower **batch** + raise **grad-accum** → shorten **max_seq_len** → (last resort) shard with FSDP/DeepSpeed ZeRO.
- **Estimate then measure:** weights/grad/opt are exact arithmetic; only activations wobble. Calibrate the activation constant against `torch.cuda.max_memory_allocated()` and you'll be within ~20%.
- **Default:** PEFT (QLoRA) for task adaptation; full FT only for continued pretraining or reshaping the base.

---

## Connect to the lab

This lecture is the theory behind **Week 2, Lab task 2** — `scripts/vram_estimate.py`, the function that takes `params_billion, dtype_bytes, mode` and prints full-FT / LoRA / QLoRA training VRAM. Predict your 8B QLoRA run *before* launching, then record the actual peak with `torch.cuda.max_memory_allocated()` and log both in the README; the Definition of Done is being within **~20%**. The same estimate justifies task 3's config (4-bit NF4, paged AdamW 8-bit, gradient checkpointing on) — each knob you set there maps to a specific term you shrank here.

---

## Going deeper (optional)

- **Unsloth** (github.com/unslothai/unsloth) — the README's **VRAM tables** for QLoRA on common models/GPUs are the fastest real-world calibration for your hand estimate; the free Colab notebooks (Llama-3.1-8B, Qwen2.5-7B) show the settings in action. Search: *"Unsloth VRAM requirements table"*.
- **Hugging Face "Model Memory" utility** — the `hf-accelerate/model-memory-usage` Space (the Accelerate model-memory estimator) prints inference vs training memory for any Hub model. Search: *"Hugging Face model memory usage estimator space"*.
- **QLoRA paper** — *"QLoRA: Efficient Finetuning of Quantized LLMs"* (Dettmers et al., 2023): NF4, double quantization, paged optimizers, from the source.
- **LoRA paper** — *"LoRA: Low-Rank Adaptation of Large Language Models"* (Hu et al., 2021) for the low-rank-delta intuition.
- **HF PEFT docs** (huggingface.co/docs/peft) and **bitsandbytes** (github.com/bitsandbytes-foundation/bitsandbytes) — the `target_modules`, 4-bit config, and 8-bit/paged optimizer knobs referenced above.
- **ZeRO / FSDP** — Microsoft DeepSpeed's *"ZeRO"* paper and the PyTorch FSDP docs explain how full FT gets sharded across GPUs when 16 B/param won't fit one card. Search: *"DeepSpeed ZeRO memory optimization"*.

---

## Check yourself

1. Write the four VRAM terms of a training step, with bytes-per-param for each under mixed-precision AdamW. Which two nearly vanish under LoRA, and why?
2. A colleague says "full fine-tuning an 8B model needs about 16 GB of VRAM because the weights are 16 GB." What did they forget, and what's the realistic static number?
3. Derive the QLoRA training budget for a **3B** model, term by term, and name the smallest GPU it fits.
4. Your QLoRA run trains for 4 minutes then OOMs exactly on the optimizer step. Name the term that spiked and the two cheapest fixes.
5. Why does gradient checkpointing *not* help an out-of-memory caused by optimizer states?
6. You did QLoRA in 10 GB. Why might serving the merged model need ~16 GB, and where did the extra memory come from?

### Answer key

1. **Weights** (bf16 2 B/param), **gradients** (bf16 2 B/param, trainable params only), **optimizer states** (Adam m+v fp32 = 8 B/param + fp32 master 4 B = 12 B/param), **activations** (variable, scales with batch×seq×hidden×layers). Under LoRA the **gradients** and **optimizer states** for the *base* vanish because the base is frozen — no gradient is computed for a frozen weight, so no state is stored. They exist only for the <1% adapter params.

2. They costed only the **weights** (2 B/param = 16 GB) and forgot **gradients + optimizer states + fp32 master** = another 14 B/param, plus activations. The static floor is **~16 B/param = 128 GB**, and ~140–155 GB with activations — an order of magnitude more than the weights, needing multi-GPU or sharding.

3. `base 3B × 0.5 = 1.5 GB` (4-bit NF4) + adapter side (weights+grads+8-bit paged optimizer, <1% of params) **<0.5 GB** + activations with checkpointing **~3–5 GB** + overhead **~1–2 GB** ≈ **~5–7 GB**. Fits an **8 GB** consumer card (and trivially a 15 GB T4).

4. The **optimizer-state** term spiked (Adam touches/allocates `m` and `v` on the step). Cheapest fixes: switch to a **paged optimizer** (spill state to CPU during the spike) and/or an **8-bit optimizer** (halve the 8 B/param states). Also effective: smaller batch + grad-accum.

5. Gradient checkpointing only reduces **activation** memory by recomputing activations in the backward pass — it never touches the optimizer's `m`/`v`/master buffers, which are allocated per trainable parameter regardless of how activations are stored. Wrong lever for that term; use an 8-bit/paged optimizer.

6. Training kept the base in **4-bit (~4 GB)** because it was frozen. Serving the **merged** model loads the un-quantized **bf16** weights (8B × 2 = **16 GB**) plus a KV cache. The extra ~12 GB is the base going from 4-bit-frozen back to full bf16 precision — which is why you re-quantize the *merged* model (AWQ/GPTQ/GGUF, Week 3) before deploying if VRAM is tight.
