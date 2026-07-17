# Lecture 5: QLoRA — 4-bit NF4 Training on a Single Cheap GPU

> Full fine-tuning a 7-8B model wants north of 100GB of VRAM — a multi-GPU node you don't have and won't rent for a side project. QLoRA is the trick that collapses that bill onto a single 15-16GB card: a free Colab T4, a $0.50/hr L4, the 16GB GPU already in your desk drawer. It does this with three moves — quantize the frozen base to 4-bit NF4, train tiny bf16 LoRA adapters on top, and use a paged optimizer so the optimizer step doesn't OOM at the worst possible moment. After this lecture you can explain *exactly* where every gigabyte goes, set a QLoRA config that won't blow up, and — the part everyone gets wrong — know the serving rule: train on a 4-bit base, but **merge into fp16/bf16 and quantize *that*** for deployment. Merge into the 4-bit base and you ship garbage.

**Prerequisites:** Lecture on Quantization & VRAM Math (Phase 0 Wk3), the LoRA lecture (this week), chat templates, `nvidia-smi` literacy · **Reading time:** ~30 min · **Part of:** Phase 08 (Model Adaptation & Fine-Tuning) Week 2

---

## The core idea (plain language)

Fine-tuning is expensive not because of the weights themselves but because of everything training drags along *with* them. To do a gradient step you need, all resident in VRAM at once: the weights, a gradient for every trainable weight, the optimizer's bookkeeping (Adam keeps two extra numbers per weight), and the activations you cached on the forward pass so backprop can use them. For full fine-tuning in mixed precision that stack is roughly **12-20× the model's fp16 size**. An 8B model is ~16GB in fp16; ×12-20 is ~190-320GB. Not happening on one card.

QLoRA attacks every term at once with a simple insight from the LoRA lecture: **you don't have to train the weights at all.** Freeze them. Train small low-rank adapter matrices injected next to the frozen layers — under 1% of the parameters. Now gradients and optimizer states exist only for the *adapters*, which is a rounding error. The frozen base still has to sit in VRAM to run the forward and backward pass through it — but since it's frozen and never updated, you can store it *compressed*. QLoRA compresses it to **4-bit** using a format called **NF4**, so 8B params cost ~4GB instead of ~16GB.

Three ingredients, three jobs:

- **4-bit NF4 base** — shrinks the biggest, most stubborn VRAM term (the frozen weights) by 4×.
- **bf16 LoRA adapters** — the only thing that actually trains; keeps gradients/optimizer states tiny.
- **Paged AdamW 8-bit optimizer** — makes the optimizer states themselves small *and* survives the transient memory spike of the optimizer step by spilling to CPU RAM instead of crashing.

Add gradient checkpointing (trade compute for activation memory) and a `max_seq_len` sized to your data, and an 8B fine-tune drops under 15GB. That's the whole game. The rest of this lecture is *why each piece works and how it bites you when you get it wrong.*

---

## How it actually works (mechanism, from first principles)

### Where the VRAM actually goes (full FT vs QLoRA)

Start with the bill for **full** fine-tuning of an 8B model in mixed precision. Per parameter, roughly:

```
weights (fp16)          2 bytes
gradient (fp16)         2 bytes
Adam state m (fp32)     4 bytes
Adam state v (fp32)     4 bytes
master weight (fp32)    4 bytes   (mixed precision keeps an fp32 copy)
-----------------------------------------
                       16 bytes / param   → 8B × 16 = 128 GB
                       + activations on top (often tens of GB)
```

That's the "12-20×" folklore made concrete. Now QLoRA. The base is frozen, so it contributes **only** its storage — no gradient, no optimizer state, no fp32 master copy:

```
FROZEN BASE (8B, NF4):        ~0.5 byte/param  → 8B × 0.5 ≈ 4.0 GB
                              (+ double-quant & NF4 metadata overhead)

LoRA ADAPTERS (trainable):    say r=16 on q,k,v,o + gate,up,down
                              ≈ 40-80M params, bf16
   adapter weights           2 B/param
   gradients                 2 B/param
   AdamW-8bit state (m,v)    ~2 B/param total (8-bit, not 32-bit)
                              → ~50M × 6 B ≈ 0.3 GB   (a rounding error)

ACTIVATIONS:                  the swing term — checkpointing controls it
KV / working buffers:         small at training seq lengths
```

The frozen base went from 128GB of "training weight" to ~4GB of "just storage." Everything trainable is under a gigabyte. This is why QLoRA fits where full FT cannot — not because it trains less well, but because it *stops paying the 8×/param tax on 8 billion parameters and only pays it on 50 million*.

### What NF4 actually is (and why it beats naive int4 for weights)

You already know int4 from the quantization lecture: chop weights into small groups, and within each group map the floats onto 16 evenly-spaced buckets. "Evenly spaced" is the problem. Neural-net weights are **not** uniformly distributed — after training they pile up in a bell curve centered near zero, roughly Gaussian. Most weights are small; extreme values are rare.

If your 16 buckets are evenly spaced across the range, you spend precious buckets out on the tails where almost no weights live, and you leave the dense center — where *most* of your weights actually sit — coarsely quantized. You're allocating resolution where there's no data.

**NF4 (4-bit NormalFloat)** fixes this by making the buckets *non-uniform*, spaced so that each bucket holds roughly the same number of weights *assuming the weights are normally distributed*. Buckets are packed tightly near zero (where the mass is) and spread out toward the tails (where weights are rare). It's "information-theoretically optimal for normally-distributed data" — an engineer's way of saying: for the bell-curve shape real weights have, NF4 puts its 16 levels where they'll do the most good.

```
naive int4 levels (uniform):   |    |    |    |    |    |    |    |    |
                              -1                  0                  +1
                              (buckets wasted on empty tails,
                               center under-resolved)

NF4 levels (normal-optimal):   |  | | ||||||| | |  |
                              -1        0        +1
                              (dense near 0 where weights live)
```

Practically: NF4 preserves the model's behavior noticeably better than naive int4 at the *same* 4 bits, especially for the near-zero weights that dominate. That's why QLoRA specifies NF4 and not plain int4 — and why in `bitsandbytes` you set `bnb_4bit_quant_type="nf4"`, not `"fp4"`.

One more thing NF4 depends on: **per-block scaling**. Weights are quantized in small blocks (default 64) so a few outliers in one block don't stretch the scale for everyone. Each block stores its own scale factor (a "quantization constant").

### Double quantization (the last few percent)

Those per-block scale factors add up. With block size 64, you get one fp32 scale per 64 weights — that's `4 bytes / 64 params ≈ 0.5 bits/param` of *pure overhead*, on top of your 4 bits. **Double quantization** quantizes the scale factors themselves (the second quantization), knocking that overhead down to roughly 0.13 bits/param. On an 8B model that's a few hundred MB saved — not glamorous, but on a 15GB card the difference between "fits" and "OOM" is often exactly a few hundred MB. Enable it with `bnb_4bit_use_double_quant=True`. It's essentially free quality-wise; leave it on.

### Why the adapters stay in bf16

The frozen base is 4-bit, but the LoRA `A`/`B` matrices you're *training* are kept in **bf16**. Reason: you're computing gradients and taking tiny optimizer steps on them. 4-bit has 16 possible values — try to represent a gradient update of `+0.0003` and it rounds to zero; your adapter never learns. Trainable things need real dynamic range. The forward pass dequantizes the 4-bit base block-by-block to bf16 on the fly, adds the bf16 LoRA delta, and computes in bf16. So compute happens in bf16; only the *frozen storage* is 4-bit. That split — 4-bit at rest, bf16 in motion — is the essence of QLoRA.

### The paged optimizer (surviving the spike)

Two separate wins here. First, **8-bit**: `paged_adamw_8bit` stores Adam's two moment estimates in 8-bit instead of fp32, cutting optimizer-state memory ~4×. Since those states cover only the tiny adapters, this is minor for QLoRA but free.

The bigger win is **paged**. VRAM usage isn't flat — it *spikes* during the optimizer step and on unusually long batches. A run that cruises at 13GB can momentarily need 15.5GB at the step, and a 15GB card OOMs — at the *end* of an epoch, after hours of compute, which is maddening. Paged optimizers use NVIDIA unified memory: when VRAM is about to overflow, optimizer-state pages are automatically evicted to **CPU RAM** and paged back when needed, exactly like OS virtual memory paging to disk. The spike gets absorbed by system RAM instead of killing the run. It's slower for those moments, but "slightly slower" beats "crashed at 97%."

### Gradient checkpointing (the activation lever)

Activations — the intermediate tensors saved on the forward pass for use in backprop — scale with `batch_size × seq_len × hidden_size × num_layers` and are often the term that actually OOMs you, not the weights. **Gradient checkpointing** saves activations only at a few layer boundaries and *recomputes* the rest during the backward pass. You trade ~20-30% more compute for a large drop in activation memory (often 3-5×). On a 15GB card training an 8B model, this is not optional; it's what makes the batch fit.

```
QLoRA memory levers, ranked by how much they save:
  4-bit NF4 base      ████████████████  (biggest: 4× off the weights)
  gradient checkpoint ████████           (biggest activation cut)
  paged optimizer     ███                (kills the end-of-epoch spike)
  8-bit optimizer     ██                 (tiny for LoRA; free)
  max_seq_len sizing  ███                (linear in seq len — don't overpay)
```

---

## Worked example

Let's fit **Qwen2.5-7B** (7.6B params) on a **free Colab T4 (15GB usable)**.

**Step 1 — Predict the base.** NF4 at ~0.5 byte/param plus NF4/double-quant metadata: `7.6B × 0.5 ≈ 3.8GB`, call it ~4.2GB with overhead. Fits with room to spare.

**Step 2 — Adapters.** `r=16, alpha=32`, targeting `q,k,v,o,gate,up,down`. That's roughly 40M trainable params. In bf16 with 8-bit AdamW: weights (80MB) + grads (80MB) + optimizer (~80MB) ≈ **~0.25GB**. Negligible.

**Step 3 — Activations.** With gradient checkpointing on, `batch_size=1`, `grad_accum=8` (effective batch 8), `max_seq_len=1024` (our ticket JSON is short — no reason to reserve 4096): activations land around **2-4GB**. This is the term you *tune* to fit.

**Step 4 — Tally:**

```
base (NF4 + double-quant)   ~4.2 GB
adapters + grads + optim    ~0.3 GB
activations (ckpt, sl=1024) ~3.0 GB
CUDA context + fragmentation ~1.5 GB
--------------------------------------
                            ~9.0 GB   → comfortably under 15GB
```

Measure the truth with `torch.cuda.max_memory_allocated()` after a few steps. If your estimate is within ~20% of measured, you understand your run. If measured is *way* higher, the usual culprits: `max_seq_len` too large, gradient checkpointing silently off, or batch/grad-accum wrong.

**Step 5 — If it still OOMs**, pull the levers in order: (1) lower `max_seq_len` to the real 95th-percentile token length of your data; (2) confirm gradient checkpointing is actually on; (3) `batch_size=1`, raise `grad_accum` to keep effective batch; (4) confirm paged optimizer and double-quant are on. You almost never need to touch `r`.

---

## How it shows up in production

**The serving rule — read this twice.** You trained on a 4-bit NF4 base. That base was *good enough to compute gradients through* because NF4 is a clever approximation — but it is still an approximation of the real weights. Two things follow, and getting them wrong is the single most common QLoRA disaster:

1. **Quantize the base FOR TRAINING** — that's what let you fit on the cheap GPU. Fine.
2. **For serving, merge the adapter into an fp16/bf16 base, THEN quantize *that* merged model for deployment** (that's Week 3 — GPTQ/AWQ/GGUF).

Why not just merge the adapter into the 4-bit base and ship it? Because **merging means adding the LoRA delta back into the base weights numerically**: `W_merged = W_base + (B·A)·(alpha/r)`. If `W_base` is the 4-bit-quantized-then-dequantized approximation, you're adding a precise bf16 delta onto *lossy, rounded* weights, then re-rounding the sum back to 4-bit. The quantization error compounds with the merge, and — critically — your adapter was trained to correct the behavior of the *specific dequantized* base, not the clean base. Merge into a re-quantized base and the correction lands on the wrong numbers. Result ranges from "a few points worse" to **outright garbage output**. The fix is boring and correct: load the base in **fp16/bf16**, merge the adapter into *that*, save the merged fp16 model, and hand it to your Week-3 serving quantizer (AWQ/GPTQ/GGUF), which are designed as *inference* quantizers with calibration. In Unsloth this is `save_pretrained_merged(..., save_method="merged_16bit")`; in PEFT it's loading the base fresh in bf16 then `merge_and_unload()`.

**Cost/venue.** The whole point is cheapness. A 7-8B QLoRA epoch on a T4 is slow (no bf16 tensor cores — expect hours); an **L4 or A10** at ~$0.50-1.00/hr on Modal/RunPod is the sweet spot — a few dollars for the run. Don't rent an A100 for an 8B QLoRA; you're paying for VRAM you engineered away.

**The end-of-epoch OOM.** The classic production bug report: "it trained for 3 hours then crashed." That's the optimizer-step spike on a card with no headroom. Paged optimizer + gradient checkpointing + `batch_size=1`/grad-accum is the standard fix. If you're already paged and still spiking, your `max_seq_len` is letting one long sample blow the activation budget — clip it.

**Speed vs VRAM is a real trade.** Gradient checkpointing costs ~20-30% wall-clock. On a card that fits without it, turn it off and train faster. On a 15GB card training 8B, you have no choice. Know which regime you're in.

---

## Common misconceptions & failure modes

- **"QLoRA is lower quality than LoRA because the base is 4-bit."** In practice the gap is small — the frozen base is only ever *read*, and the bf16 adapters learn to compensate. QLoRA was explicitly designed to nearly match 16-bit LoRA. Don't reach for full precision reflexively.
- **"NF4 is just int4 with a different name."** No — the buckets are non-uniform, shaped for a normal distribution. Set `bnb_4bit_quant_type="nf4"`. Leaving it on the `fp4` default gives you worse quality for free.
- **"I'll merge the adapter into my 4-bit training model and export it."** The headline failure. Merge into **fp16/bf16**, then quantize. Merging into 4-bit produces precision loss or garbage.
- **"Double quantization hurts accuracy, I'll disable it."** It's essentially free quality-wise and saves the few hundred MB that decide whether you fit. Leave it on.
- **"Paged optimizer makes training slow."** Only during the moments it's actually paging. The alternative to a brief slowdown is an OOM crash. Keep it on for tight-VRAM runs.
- **"Bigger `max_seq_len` is safer."** It's linearly more activation memory and KV for *no benefit* if your data is short. Size it to the ~95th-percentile token length of your dataset.
- **"Gradient checkpointing is on by default."** It usually isn't. If your measured VRAM is way over your estimate, check this first.
- **"Full fine-tuning would be better if I had the GPUs."** For behavior/format/tone tasks on a few thousand examples, QLoRA is not a poor-person's compromise — it's often the *right* tool, because it localizes change to adapters and resists catastrophic forgetting (Week 4).

---

## Rules of thumb / cheat sheet

*(All numbers approximate — measure your own run.)*

**QLoRA VRAM back-of-envelope:**
```
base_GB  ≈ params_B × 0.5 × 1.1      (NF4 + metadata)
train_GB ≈ base_GB + ~0.3 (adapters) + activations + ~1.5 (CUDA/frag)
8B on 15GB card: expect ~8-11 GB with checkpointing + sl≤1024
```

**The lab's default config (start here, tune to fit):**
```python
# bitsandbytes 4-bit config
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NOT fp4
    bnb_4bit_use_double_quant=True,     # free savings
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# LoRA
r=16, lora_alpha=32, lora_dropout=0.05,
target_modules=["q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"]
# Trainer
optim="paged_adamw_8bit"
gradient_checkpointing=True
per_device_train_batch_size=1
gradient_accumulation_steps=8           # effective batch 8
learning_rate=2e-4                      # cosine schedule
num_train_epochs=1-2
max_seq_len=<your data's ~p95 token length>   # e.g. 1024, not 4096
```

**Why each knob fights OOM:**
| Knob | Attacks | Effect |
|---|---|---|
| 4-bit NF4 base | frozen weights | 4× off the biggest term |
| double quant | scale metadata | ~few hundred MB |
| gradient checkpointing | activations | 3-5× less, +20-30% compute |
| paged AdamW 8-bit | optimizer + spike | small states, no end-of-epoch OOM |
| batch=1 + grad-accum | activations | fit; keep effective batch via accum |
| max_seq_len sized to data | activations + KV | linear savings, no quality cost |

**Tool decision (choose one, don't collect all three):**
| Tool | Pick it when | Trade-off |
|---|---|---|
| **Unsloth** | You want fastest + lowest-VRAM + free-Colab notebook; solo/small-team; fitting on a T4/L4 | Custom kernels; some edge models lag support; less config surface |
| **Axolotl** | Production/repeatable runs; one YAML per experiment; multi-GPU; want configs in git | YAML learning curve; heavier setup |
| **raw TRL + PEFT** | You need maximum control, custom loss/collator, or to understand every layer | Most boilerplate; you own the OOM debugging |

Default for this course: **Unsloth** for the QLoRA run (it just works on free Colab), knowing Axolotl and TRL+PEFT exist for when you productionize.

---

## Connect to the lab

Week 2's lab is exactly this lecture made real: write `vram_estimate.py`, predict your 8B QLoRA peak, then run the Unsloth notebook with the config above and check your prediction against `torch.cuda.max_memory_allocated()` (Definition of Done: within ~20%). You'll save the adapter, hot-swap it at inference, then **merge into bf16** (`save_pretrained_merged` / `merge_and_unload` on a fresh bf16 base) — deliberately proving to yourself that the merged fp16 model reproduces the adapter's outputs. The "merge into 4-bit = garbage" rule is what you must not violate; the merged bf16 model is the artifact you hand to Week 3 for AWQ/GPTQ/GGUF serving quantization.

---

## Going deeper (optional)

- **QLoRA paper** — Dettmers et al., 2023, *"QLoRA: Efficient Finetuning of Quantized LLMs."* The source of NF4, double quantization, and paged optimizers. Search: `QLoRA paper arxiv Dettmers`.
- **`bitsandbytes` docs** — the library that implements NF4 and paged 8-bit optimizers. Root: `huggingface.co/docs/bitsandbytes`. Read the 4-bit / `BitsAndBytesConfig` section.
- **Hugging Face PEFT docs** — LoRA/QLoRA usage, `merge_and_unload`. Root: `huggingface.co/docs/peft`.
- **Unsloth docs & notebooks** — fastest QLoRA path, free-Colab notebooks for Llama-3.1-8B / Qwen2.5-7B, VRAM tables. Root: `docs.unsloth.ai` and their GitHub. Search: `Unsloth QLoRA Colab notebook`.
- **Axolotl** — YAML-driven fine-tuning. Search: `Axolotl fine-tuning github` and `Axolotl qlora example yaml`.
- **HF TRL `SFTTrainer` docs** — the raw-control path. Root: `huggingface.co/docs/trl`.
- **HF "Model Memory Utility" / accelerate memory estimator** — sanity-check your VRAM math. Search: `Hugging Face model memory usage estimator`.

---

## Check yourself

1. An 8B model needs ~128GB for full fine-tuning but ~9GB under QLoRA. Walk through *which VRAM terms* QLoRA eliminates or shrinks, and which one it does *not* touch.
2. Why does NF4 preserve model quality better than naive int4 at the same 4 bits? What property of trained weights does it exploit?
3. You merge your trained adapter directly into the 4-bit base and inference produces gibberish. Explain the mechanism, and state the correct sequence.
4. Your run trained fine for 3 hours then OOM'd at the end of the epoch. What's happening, and which two config changes address it most directly?
5. Your teammate sets `max_seq_len=4096` "to be safe," but the dataset's longest example is 700 tokens. What's the cost, and why is there no benefit?
6. When would you reach for Axolotl over Unsloth, and when for raw TRL+PEFT?

### Answer key

1. QLoRA **freezes** the base, so it eliminates the per-parameter gradient, both fp32 Adam states, and the fp32 master copy for 8B params (the bulk of the 16 B/param bill). It shrinks the frozen weights themselves 4× via NF4 (16GB → ~4GB). Gradients/optimizer states still exist but *only for the ~50M adapter params* — negligible. What it does **not** eliminate: activation memory from the forward/backward pass through the full base — that's why gradient checkpointing and `max_seq_len` still matter.
2. Trained weights are roughly **normally distributed** — dense near zero, sparse in the tails. Uniform int4 spends buckets on empty tails and under-resolves the crowded center. NF4's buckets are **non-uniform**, packed near zero where the mass is, so the same 16 levels carry more information about the weights that actually matter. It's distribution-matched quantization.
3. Merging adds `(B·A)·(alpha/r)` onto the base weights. Into a 4-bit base that means adding a bf16 delta onto lossy dequantized weights and re-rounding to 4-bit — quantization error compounds, and the adapter (trained to correct the *specific dequantized* base) is now applied to re-quantized numbers, so its correction misfires → garbage. Correct sequence: load base in **fp16/bf16**, `merge_and_unload` into that, save the merged fp16 model, then quantize *that* for serving (AWQ/GPTQ/GGUF).
4. VRAM **spikes** during the optimizer step; a card with no headroom OOMs at that moment (often end of epoch). Most direct fixes: enable the **paged** optimizer (spills the spike to CPU RAM) and **gradient checkpointing** (cuts the activation baseline so the spike stays under the ceiling). Also `batch_size=1` + grad-accum and a tighter `max_seq_len`.
5. Activation and KV memory scale **linearly** with `max_seq_len`, so 4096 vs ~1024 reserves ~4× the activation budget for sequences that never exist — pure waste that can push you into OOM or force gradient checkpointing you didn't need. No benefit because nothing in the data is longer than 700 tokens; padding/reservation beyond the real length trains on nothing. Size to ~p95 of actual token lengths.
6. **Axolotl** when you want production-repeatable runs captured in a YAML config (in git), multi-GPU, or standardized experiments. **raw TRL+PEFT** when you need maximum control — custom loss, custom data collator, non-standard training loop — or to understand/debug every layer yourself. **Unsloth** remains the default for fitting a single-GPU QLoRA fast and cheap.
