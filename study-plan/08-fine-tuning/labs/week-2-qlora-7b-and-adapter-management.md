# Week 2 Lab: QLoRA a 7-8B Model and Manage Adapters

> This week you fit a **real 7-8B model** onto a single free/cheap GPU with **QLoRA** (4-bit NF4 base + tiny bf16 LoRA adapters + a paged optimizer) and drive the *entire adapter lifecycle*: predict VRAM before you spend a cent, train, hot-swap the adapter at inference, merge it into a bf16 base, and run a quick eval against the base-instruct model. The headline skill is that you no longer *guess* whether a run will fit — you compute the budget by hand, then prove your estimate was within ~20% of the measured peak. Everything runs on a **free Colab T4 (15 GB)** or a ~$0.50-1.00/hr L4/A10; no paid API is required.
>
> Read the matching lectures first — this guide assumes them:
> - [../lectures/04-lora-mechanics-and-adapter-lifecycle.md](../lectures/04-lora-mechanics-and-adapter-lifecycle.md) — `r`, `alpha`, `target_modules`, dropout; hot-swap vs merge; why you never merge into a 4-bit base.
> - [../lectures/05-qlora-4bit-training.md](../lectures/05-qlora-4bit-training.md) — 4-bit NF4, paged optimizer, where every gigabyte goes, the train-4bit/serve-bf16 rule.
> - [../lectures/06-vram-math-and-peft-vs-full-ft.md](../lectures/06-vram-math-and-peft-vs-full-ft.md) — the four memory terms (weights, gradients, optimizer, activations) and the estimator you build in Step 1.
> - [../lectures/07-dataset-quality-and-decontamination.md](../lectures/07-dataset-quality-and-decontamination.md) — quality over quantity, and the hash-diff contamination check.
>
> It builds directly on the `llm-finetune-lab/` repo and the **week-1 dataset** (`data/prepared/train.jsonl` + `eval.jsonl`). Keep working in the same repo.

**Est. time:** ~9 hrs · **You will need:** the `llm-finetune-lab/` repo from Week 1, a Hugging Face account (Llama-3.1 is gated; Qwen2.5 is not), and a GPU. **Free/local path:** [Google Colab free T4 (15 GB)](https://colab.research.google.com) runs the whole lab — this is the recommended venue. Your Windows laptop is for data prep, `vram_estimate.py`, and reading diffs; the training/inference cells run in Colab. Paid-but-cheap alternative: an L4 or A10 (~$0.50-1.00/hr) on Modal/RunPod for ~3x faster runs.

---

## Before you start (setup)

**What / Why.** QLoRA needs a CUDA GPU with `bitsandbytes`, which does not work on Windows CPU or Apple Silicon. Rather than fight local installs, do the GPU work in Colab where the CUDA/`bitsandbytes`/`unsloth` stack is pre-baked. You keep the *scripts* in your repo (so they're graded and version-controlled) and paste/import them into a Colab notebook to run. `vram_estimate.py` is pure Python and runs anywhere, so you write and test it locally first.

**Do it (local, Windows/Git-Bash — for the pure-Python + data pieces):**
```bash
cd llm-finetune-lab            # the repo from Week 1
mkdir -p adapters models scripts
uv add "torch" "transformers>=4.44" "peft>=0.11" "datasets" "pydantic"   # runs locally for vram_estimate + data checks
```

**Do it (Colab — the GPU venue).** Open a new Colab notebook, set **Runtime → Change runtime type → T4 GPU**, then in the first cell:
```python
# Colab cell 1 — install the Unsloth stack (pins CUDA-compatible bitsandbytes/peft/trl)
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "trl<0.9.0" peft accelerate bitsandbytes
import torch; print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0))
```

**Verify:**
```python
# Colab — should print CUDA: True | GPU: Tesla T4  (or L4/A10 if you rented one)
# Also confirm ~15 GB total:
print(f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM")
```
```bash
# Local — vram_estimate deps only need pure Python + torch:
uv run python -c "import torch, peft, transformers; print('local deps ok')"
```

**Get your data into Colab.** Easiest: `git clone` your repo in a Colab cell, or mount Google Drive and copy `data/prepared/`. You need `train.jsonl` and the 100-row held-out `eval.jsonl` from Week 1 available in the Colab filesystem.

**Windows / macOS / Linux note.** All `uv run` commands are identical across platforms. The **only** platform-specific fact: `bitsandbytes` 4-bit training does not run on Windows-CPU or macOS — that is *why* the GPU steps live in Colab. Don't try to `uv add bitsandbytes` locally on Windows and block on a build error; you don't need it locally.

**Gated-model note.** If you use `meta-llama/Llama-3.1-8B-Instruct` you must accept its license on the HF hub and `huggingface-cli login` (or `notebook_login()` in Colab). `Qwen/Qwen2.5-7B-Instruct` is **not gated** — use it if you want zero friction. This guide shows Qwen2.5-7B as the default and notes Llama-3.1 equivalents.

---

## Step-by-step

### Step 1 — Write `vram_estimate.py` and predict your budget *first*

**What.** Write `scripts/vram_estimate.py` with a function `estimate_vram(params_billion, dtype_bytes, mode)` that returns the four memory terms (weights, gradients, optimizer states, activations) and their sum for `mode in {"full", "lora", "qlora"}`. Run it for your 8B model and **write the QLoRA prediction into README before you train**.

**Why.** The most expensive first-fine-tune mistake is renting the wrong GPU — over-provisioning an A100 you don't need, or launching full-FT on a T4 and OOMing three minutes in. Doing the arithmetic by hand (lecture 06) turns "I hope it fits" into "I predict 9.8 GB, and the run must stay ≤15 GB." The Definition of Done requires your prediction to land within ~20% of the measured peak, so this script is the acceptance instrument, not busywork.

**Do it.** In `scripts/vram_estimate.py`:
```python
"""Estimate training VRAM for full-FT / LoRA / QLoRA before renting a GPU.

Memory has four terms (see lecture 06):
  weights            = params * bytes_per_param(base)
  gradients          = trainable_params * 2   (bf16 grads)
  optimizer states   = trainable_params * K   (Adam fp32 m+v ~= 8 B/param full;
                                                paged AdamW-8bit ~= 2 B/param)
  activations        = empirical overhead (grad-checkpointing shrinks this a lot)
"""
from dataclasses import dataclass

GB = 1e9

@dataclass
class VramEstimate:
    weights: float
    gradients: float
    optimizer: float
    activations: float
    @property
    def total(self) -> float:
        return self.weights + self.gradients + self.optimizer + self.activations

def estimate_vram(params_billion: float, dtype_bytes: float, mode: str,
                  lora_frac: float = 0.005, act_gb: float = 2.0) -> VramEstimate:
    """params_billion: e.g. 8.0.  dtype_bytes: base weight bytes/param
    (2=fp16/bf16, 0.5=4-bit NF4).  mode: 'full' | 'lora' | 'qlora'.
    lora_frac: trainable fraction for PEFT (~0.5% is typical for r=16).
    act_gb: activation/working-set estimate (grad-checkpointing ON assumed for qlora)."""
    p = params_billion * 1e9
    if mode == "full":
        weights   = p * dtype_bytes / GB          # usually fp16 -> 2 B
        grads     = p * 2 / GB                     # bf16 grads for every param
        optim     = p * 8 / GB                     # Adam fp32 m+v+master ~= 8 B/param
        acts      = act_gb * 3                      # bigger working set, no PEFT savings
    elif mode == "lora":
        trainable = p * lora_frac
        weights   = p * dtype_bytes / GB          # base kept in fp16 (2 B)
        grads     = trainable * 2 / GB
        optim     = trainable * 8 / GB
        acts      = act_gb
    elif mode == "qlora":
        trainable = p * lora_frac
        weights   = p * 0.5 / GB                   # 4-bit NF4 base -> 0.5 B/param
        grads     = trainable * 2 / GB
        optim     = trainable * 2 / GB             # paged AdamW 8-bit -> ~2 B/param
        acts      = act_gb                          # grad-checkpointing keeps this low
    else:
        raise ValueError(f"mode must be full|lora|qlora, got {mode!r}")
    return VramEstimate(weights, grads, optim, acts)

if __name__ == "__main__":
    B = 8.0
    for mode, dt in [("full", 2.0), ("lora", 2.0), ("qlora", 0.5)]:
        e = estimate_vram(B, dt, mode)
        print(f"{mode:6s}  weights={e.weights:5.1f}  grads={e.gradients:5.2f}  "
              f"optim={e.optimizer:5.2f}  acts={e.activations:4.1f}  "
              f"TOTAL~={e.total:5.1f} GB")
```

**Do it (run it):**
```bash
uv run python scripts/vram_estimate.py
```

**Expected result.** Roughly:
```
full    weights= 16.0  grads=16.00  optim=64.00  acts=6.0  TOTAL~=102.0 GB
lora    weights= 16.0  grads= 0.08  optim= 0.32  acts=2.0  TOTAL~= 18.4 GB
qlora   weights=  4.0  grads= 0.08  optim= 0.16  acts=2.0  TOTAL~=  6.2 GB
```
This is the whole point of the phase in three lines: full-FT of an 8B is ~100 GB (a multi-GPU node), plain LoRA still keeps the base in fp16 (~18 GB, won't fit a T4), but **QLoRA lands around 6-10 GB** — comfortably inside 15 GB. Your measured peak in Step 3 will be a bit higher than the bare estimate because of the CUDA context, cached kernels, and the end-of-epoch optimizer spike; that's expected and is exactly what the ~20% tolerance covers.

**Verify.** Add the prediction to `README.md` now, e.g.:
```markdown
## VRAM budget (Qwen2.5-7B, QLoRA, r=16)
- Predicted peak: ~9.8 GB (est. 6.2 GB terms + ~3.5 GB CUDA/context/spike headroom)
- Measured peak: (fill in after Step 3)
- Within 20%? (fill in)
```

**Troubleshoot.**
- Your predicted total looks *too low* (e.g. 6 GB) and you're nervous — good instinct. The raw term-sum omits the ~1.5-2 GB CUDA context and kernel cache plus the transient optimizer-step spike. Add a fixed headroom in your README prediction (as above) so "predicted" is the number you actually compare against measured. The ~20% band absorbs the rest.
- Forgot which `dtype_bytes` to pass: **4-bit NF4 = 0.5**, fp16/bf16 = 2, int8 = 1. Passing 2 for QLoRA is the classic error and inflates the estimate 4x.

---

### Step 2 — Load the 7-8B base in 4-bit NF4 with Unsloth

**What.** In your Colab notebook, load `Qwen/Qwen2.5-7B-Instruct` (or `meta-llama/Llama-3.1-8B-Instruct`) in 4-bit and attach a LoRA config with `r=16, alpha=32, dropout=0.05`, targeting **both attention and MLP** projections. Unsloth's `FastLanguageModel` wraps the `bitsandbytes` + PEFT setup and patches kernels for lower VRAM and ~2x speed.

**Why.** 4-bit NF4 is what shrinks the frozen base from 16 GB (fp16) to ~4 GB — the single biggest lever. `r=16/alpha=32` gives effective scale `alpha/r = 2.0` (a sane default; lecture 04). Targeting attention **and** MLP (`gate/up/down_proj`) gives the adapter enough capacity for a formatting/routing task — attention-only often underfits, and "target everything" wastes VRAM and overfits (see the pitfalls). We start from the official Unsloth Colab notebook precisely because it pins a known-good stack; Axolotl (YAML config) and raw TRL+PEFT are the production-grade alternatives you should *know exist* but not fight this week.

**Do it.** Colab cell:
```python
from unsloth import FastLanguageModel

MODEL   = "Qwen/Qwen2.5-7B-Instruct"      # non-gated; Llama-3.1: "meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ = 1024                             # size to YOUR data, not 8192 — see Step 3 note

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = MODEL,
    max_seq_length = MAX_SEQ,
    dtype         = None,                  # None = auto (bf16 on Ampere+, fp16 on T4)
    load_in_4bit  = True,                  # 4-bit NF4 base
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    lora_alpha     = 32,                   # effective scale = alpha/r = 2.0
    lora_dropout   = 0.05,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",   # attention
                      "gate_proj", "up_proj", "down_proj"],     # MLP
    bias           = "none",
    use_gradient_checkpointing = "unsloth",  # essential for fitting on 15 GB
    random_state   = 3407,
)
model.print_trainable_parameters()
```

**Expected result.** `print_trainable_parameters()` prints something like `trainable params: 40,370,176 || all params: 7,655,...  || trainable%: 0.53` — i.e. **~0.5% trainable**, which matches the `lora_frac=0.005` you plugged into `vram_estimate.py`. Loading in 4-bit should occupy ~5-6 GB before training (`nvidia-smi` or `torch.cuda.memory_allocated()/1e9`).

**Verify.**
```python
import torch
print(f"post-load allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")  # expect ~5-6 GB
# Sanity: trainable% should be ~0.3-0.7 for r=16 on a 7-8B model.
```

**Troubleshoot.**
- **Gated repo 401 / access error** (Llama): run `from huggingface_hub import notebook_login; notebook_login()` and accept the license on the model page. Or just switch to Qwen2.5-7B.
- **`target_modules` name mismatch** ("module not found"): different architectures name projections differently. Qwen/Llama use `q_proj…down_proj`; if you use another model, print `model` and grep the linear layer names, or pass `target_modules="all-linear"` and let PEFT resolve them.
- **OOM on load** on T4: confirm `load_in_4bit=True` actually took (a plain fp16 load is ~16 GB and OOMs immediately). Reduce nothing else yet — if 4-bit load itself OOMs, the runtime isn't a GPU runtime; recheck Runtime type.

---

### Step 3 — QLoRA train on the week-1 dataset and record peak VRAM

**What.** Format the week-1 chat JSONL with the tokenizer's chat template, then train with TRL's `SFTTrainer` using **paged AdamW 8-bit, LR 2e-4, cosine schedule, 1-2 epochs, gradient checkpointing on**. Reset the CUDA peak counter before training and read `torch.cuda.max_memory_allocated()` after. Save the adapter to `adapters/ticket-router-v1/`.

**Why.** These hyperparameters are the QLoRA canon (lecture 05): the **paged** optimizer is what survives the end-of-epoch optimizer-step spike without OOM; LR 2e-4 with cosine decay is the standard LoRA starting point (an order of magnitude higher than full-FT because you're only moving tiny matrices); 1-2 epochs avoids memorizing a small dataset. Sizing `max_seq_len` to your actual data (routing tickets are short — 512-1024 is plenty) directly saves activation VRAM; defaulting to 8192 wastes memory you measured you don't have. Capturing `max_memory_allocated()` is the "actual" half of the predicted-vs-actual acceptance gate.

**Do it.** Colab cell:
```python
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1) Load week-1 chat-format JSONL and render with the model's chat template.
ds = load_dataset("json", data_files={"train": "data/prepared/train.jsonl"})["train"]

def to_text(row):
    # row["messages"] = [system, user, assistant] from Week 1
    return {"text": tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False)}
ds = ds.map(to_text, remove_columns=ds.column_names)

# 2) Reset the peak counter so we measure THIS run's peak.
torch.cuda.reset_peak_memory_stats()

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,     # effective batch 8; lower batch = lower peak
        warmup_ratio        = 0.05,
        num_train_epochs    = 2,             # 1-2 epochs
        learning_rate       = 2e-4,
        lr_scheduler_type   = "cosine",
        optim               = "paged_adamw_8bit",   # survives the optimizer-step spike
        gradient_checkpointing = True,
        bf16 = torch.cuda.is_bf16_supported(),      # T4 -> False -> fp16 path
        fp16 = not torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        output_dir = "outputs",
        seed = 3407,
    ),
)
stats = trainer.train()

peak = torch.cuda.max_memory_allocated() / 1e9

print(f"PEAK VRAM during training: {peak:.2f} GB")   # MUST be <= 15 GB (T4)

# 3) Save ONLY the adapter (a few tens of MB), not the base.
model.save_pretrained("adapters/ticket-router-v1")
tokenizer.save_pretrained("adapters/ticket-router-v1")
```

> **TRL version note (same drift you hit in Week 1).** The `SFTTrainer` signature moved across versions. Older TRL (0.9–0.11) takes `tokenizer=`, `dataset_text_field=`, and `max_seq_length=` as trainer args (as above). Newer TRL (>=0.12) wants `processing_class=tok` and folds `dataset_text_field`/`max_seq_length` into `SFTConfig`. If you get a `TypeError` on an unexpected keyword, move those three into `SFTConfig(...)` and pass `processing_class=tokenizer`. The Unsloth Colab pins a compatible `trl<0.9.0`-era stack, so the args above work as written there.

**Expected result.** Training loss decreases over the run (e.g. from ~1.5 toward <0.5 on a clean routing dataset). The printed **peak VRAM is roughly 9-12 GB on a T4** — under the 15 GB gate. `adapters/ticket-router-v1/` contains `adapter_model.safetensors` (tens of MB) + `adapter_config.json` — **not** a multi-GB base model. A 300-800 example dataset trains in ~5-20 min on a T4.

**Verify.**
```python
import os, glob
mb = sum(os.path.getsize(f) for f in glob.glob("adapters/ticket-router-v1/*")) / 1e6
print(f"adapter size: {mb:.1f} MB")   # tens of MB, NOT thousands
assert peak <= 15.0, f"peak {peak:.1f} GB exceeds the 15 GB gate"
```
Then close the loop on Step 1: write the measured `peak` into README next to your prediction and compute `abs(pred-actual)/actual`. It must be **≤ ~20%**.

**Troubleshoot.**
- **OOM at the very end of an epoch** (the classic optimizer-step spike): confirm `optim="paged_adamw_8bit"` and `gradient_checkpointing=True` are actually set, then lower `per_device_train_batch_size` to 1 and raise `gradient_accumulation_steps` to keep the effective batch. Also drop `max_seq_length` if your data is short.
- **Predicted vs actual off by >20%:** almost always because your README "predicted" omitted the CUDA-context/spike headroom (see Step 1) or you left `max_seq_length` at 8192 (activations balloon). Re-predict with the seq len you actually used.
- **Loss flat / not decreasing:** check the chat template rendered correctly (print one `ds[0]["text"]` — you should see the model's special tokens and an EOS after the assistant turn). A flat loss usually means template corruption or LR far too low.
- **`max_memory_allocated` looks lower than `nvidia-smi`:** expected — `nvidia-smi` includes the CUDA context + cached (unallocated) blocks. Report the `torch` number as your "peak allocated"; note in README that `nvidia-smi` reserved is higher.

---

### Step 4 — Hot-swap the adapter and diff against the bare base

**What.** Generate on the **same 20 eval inputs** twice: once with the 4-bit base + adapter attached, once with the 4-bit base *without* the adapter. Diff the outputs to prove the adapter changed behavior. Save both sets to `eval/results/`.

**Why.** This is the operational payoff of LoRA (lecture 04): adapters are hot-swappable — you keep one 4-bit base resident and attach/detach tiny adapters per task, no reload. But "I trained something" is not evidence it *does* anything. Diffing adapter-on vs adapter-off on identical inputs is the cheapest possible proof that training moved the model, and the Definition of Done requires a *measurable* difference on 20 inputs.

**Do it.** Colab cell:
```python
from unsloth import FastLanguageModel
import json

# Load the first 20 held-out eval inputs (user turns only).
eval_rows = [json.loads(l) for l in open("data/prepared/eval.jsonl")][:20]

def build_prompt(row):
    msgs = [m for m in row["messages"] if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def generate(m, prompt):
    FastLanguageModel.for_inference(m)     # Unsloth 2x inference mode
    ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = m.generate(**ids, max_new_tokens=128, do_sample=False)  # greedy = deterministic
    return tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

# (a) WITH adapter — `model` already has the adapter attached from Step 3.
with_adapter = [generate(model, build_prompt(r)) for r in eval_rows]

# (b) WITHOUT adapter — disable the LoRA layers on the SAME model object.
with model.disable_adapter():          # PEFT: temporarily route around the adapter
    no_adapter = [generate(model, build_prompt(r)) for r in eval_rows]

diffs = sum(a != b for a, b in zip(with_adapter, no_adapter))
print(f"{diffs}/20 outputs differ between adapter-on and adapter-off")
json.dump({"with_adapter": with_adapter, "no_adapter": no_adapter},
          open("eval/results/hotswap_diff.json", "w"), indent=2)
```
> If your PEFT/Unsloth version lacks `disable_adapter()`, load a **second** clean 4-bit copy of the base (`FastLanguageModel.from_pretrained(MODEL, load_in_4bit=True)` without `get_peft_model`) and generate on that instead. On a 15 GB T4 you may need to free the first model (`del`, `torch.cuda.empty_cache()`) between the two passes.

**Expected result.** A clear majority of the 20 differ — typically the base-instruct produces prose or loosely-formatted JSON, while the adapter produces your exact target schema. `diffs` should be well above 0 (aim for most of them); if the base already nails the format, differences may be subtler (field values, key order) but should still be measurable.

**Verify.** Eyeball 3-4 pairs in `hotswap_diff.json`: the adapter outputs should look like your training targets (strict JSON, right categories), the no-adapter outputs more freeform. `diffs >= 1` at minimum; a healthy fine-tune changes most of them.

**Troubleshoot.**
- **0/20 differ:** either the adapter didn't load (check `model.peft_config`), or `disable_adapter()` silently no-op'd — verify by printing one output from each pass; they should not be byte-identical. Also confirm you saved and are using the trained adapter, not a fresh untrained one.
- **Non-deterministic outputs:** you left sampling on. Use `do_sample=False` (greedy) so the diff reflects the adapter, not RNG.
- **OOM loading a second base:** use `disable_adapter()` instead of a second model, or free the first with `del model; torch.cuda.empty_cache()`.

---

### Step 5 — Merge into a bf16 base and confirm it reproduces adapter outputs

**What.** Merge the adapter weights into a **bf16 (not 4-bit)** copy of the base and save to `models/ticket-router-v1-merged/`. Then generate on the same 20 inputs and confirm the merged model reproduces the adapter outputs (**≥18/20 identical, or explain the diffs**).

**Why.** Hot-swapping keeps the base 4-bit and the adapter separate — great for serving many tasks off one base, but it carries a small runtime cost and requires PEFT at inference. Merging folds `A·B` back into the base weights so you get a **plain model** with no PEFT dependency, ready to quantize for serving (week 3). The rule that trips everyone (lecture 05): **merge into fp16/bf16, never into the 4-bit base** — merging high-precision deltas into de-quantized 4-bit weights loses precision and produces garbage. The ≥18/20 check proves the merge was faithful.

**Do it (Unsloth path — recommended).** Colab cell:
```python
# Unsloth merges into a 16-bit base in one call and writes a standard HF model dir.
model.save_pretrained_merged(
    "models/ticket-router-v1-merged",
    tokenizer,
    save_method = "merged_16bit",     # bf16/fp16 merged base — NOT "merged_4bit"
)
```

**Do it (raw PEFT path — know this exists / if not using Unsloth).**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(          # reload base in bf16, NOT 4-bit
    MODEL, torch_dtype=torch.bfloat16, device_map="auto")
merged = PeftModel.from_pretrained(base, "adapters/ticket-router-v1")
merged = merged.merge_and_unload()                    # fold A·B into base weights
merged.save_pretrained("models/ticket-router-v1-merged")
AutoTokenizer.from_pretrained(MODEL).save_pretrained("models/ticket-router-v1-merged")
```

**Do it (confirm reproduction):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
mtok = AutoTokenizer.from_pretrained("models/ticket-router-v1-merged")
mmod = AutoModelForCausalLM.from_pretrained(
    "models/ticket-router-v1-merged", torch_dtype=torch.bfloat16, device_map="auto")

def gen_merged(prompt):
    ids = mtok(prompt, return_tensors="pt").to("cuda")
    out = mmod.generate(**ids, max_new_tokens=128, do_sample=False)
    return mtok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

merged_outs = [gen_merged(build_prompt(r)) for r in eval_rows]   # same 20 inputs
identical = sum(a == b for a, b in zip(with_adapter, merged_outs))
print(f"{identical}/20 merged outputs identical to hot-swapped adapter")
```

**Expected result.** `models/ticket-router-v1-merged/` contains full model shards (`*.safetensors`, several GB — this is a *real* model now, not an adapter) + config + tokenizer. `identical` is **≥18/20**. The 0-2 that differ are usually tail-token or bf16-vs-4bit numerical rounding — acceptable and explainable, since the hot-swap ran on a 4-bit base while the merged model is bf16.

**Verify.**
```python
assert identical >= 18, f"only {identical}/20 identical — investigate before merging further"
```
If it's <18, document *why* in README (e.g. "4-bit base rounding vs bf16 merged; the differing 3 differ only in whitespace/field order").

**Troubleshoot.**
- **Merged model outputs garbage:** you merged into a **4-bit** base. Reload the base in `torch_dtype=torch.bfloat16` (raw path) or use `save_method="merged_16bit"` (Unsloth) — never `merged_4bit` for the serving artifact.
- **`identical` much lower than 18** and outputs look plausible but different: this is the 4-bit(hot-swap) vs bf16(merged) precision gap; it's expected to shift a few. If *most* differ, the merge didn't apply — confirm `merge_and_unload()` ran and you're loading from the merged dir, not the base.
- **Out of disk in Colab:** the merged 16-bit model is ~15 GB on disk. Save straight to mounted Google Drive, or push to the HF hub with `push_to_hub`, rather than the ephemeral Colab disk.

---

### Step 6 — Quick eval: base-instruct vs fine-tuned on the held-out 100

**What.** Run the **100 held-out** examples through both the base-instruct model and your fine-tuned model, and record two numbers for each: **JSON-valid rate** and **category accuracy**. Report the deltas. (The full eval harness with LLM-judge and retention is week 4 — this is the quick, honest read.)

**Why.** Loss going down is not evidence the model is better (lecture 07 / week 4). The two things fine-tuning is *supposed* to fix for a routing task are format adherence (does it emit parseable JSON every time?) and task accuracy (does it pick the right category?). Measuring both on data the model never saw is the minimum bar before you'd ever ship. Doing it now also forces the **contamination check**: the held-out 100 must be disjoint from train.

**Do it (contamination guard — run this first).**
```python
import hashlib, json
def norm_hash(row):  # hash the user turn so train/eval overlap is caught
    u = next(m["content"] for m in row["messages"] if m["role"] == "user")
    return hashlib.sha256(u.strip().lower().encode()).hexdigest()

train = {norm_hash(json.loads(l)) for l in open("data/prepared/train.jsonl")}
heldout = [json.loads(l) for l in open("data/prepared/eval.jsonl")][:100]
leaked = sum(norm_hash(r) in train for r in heldout)
assert leaked == 0, f"CONTAMINATION: {leaked} held-out rows appear in train"
print("contamination check passed: train and held-out are disjoint")
```

**Do it (score both models).**
```python
import json

def gold_category(row):
    return json.loads(next(m["content"] for m in row["messages"] if m["role"]=="assistant"))["category"]

def score(gen_fn, rows):
    valid = acc = 0
    for r in rows:
        out = gen_fn(build_prompt(r))
        try:
            pred = json.loads(out)
            valid += 1
            if pred.get("category") == gold_category(r):
                acc += 1
        except json.JSONDecodeError:
            pass
    n = len(rows)
    return {"json_valid_pct": 100*valid/n, "category_acc_pct": 100*acc/n}

# base-instruct: load a clean 4-bit base (no adapter) OR use model.disable_adapter()
ft   = score(gen_merged, heldout)                     # merged fine-tune from Step 5
with model.disable_adapter():
    base = score(lambda p: generate(model, p), heldout)

print("base-instruct :", base)
print("fine-tuned    :", ft)
print("delta json_valid:", ft["json_valid_pct"]-base["json_valid_pct"],
      "| delta acc:",       ft["category_acc_pct"]-base["category_acc_pct"])
json.dump({"base": base, "fine_tuned": ft}, open("eval/results/quick_eval.json","w"), indent=2)
```

**Expected result.** The fine-tuned model shows a clear win on **JSON-valid rate** (the format is exactly what it trained on — often base-instruct 60-90% → fine-tuned 95-100%) and a positive or at least non-negative **category accuracy** delta. Record all four numbers + the two deltas in README.

**Verify.** `eval/results/quick_eval.json` exists with both models' `json_valid_pct` and `category_acc_pct`, and the contamination assert passed (0 leaked).

**Troubleshoot.**
- **Fine-tuned JSON-valid < base:** usually you're parsing the wrong text (trailing prose, code fences). Strip ```` ```json ```` fences before `json.loads`, or check the model learned to stop (EOS present in training — a week-1 concern).
- **Accuracy identical to base:** the task may already be promptable (a valid, common finding — note it), or the adapter underfit (bump `r` to 32 or add epochs), or you're accidentally scoring the base twice (confirm `gen_merged` loads the merged dir).
- **Contamination assert fails:** you leaked eval into train. Fix the split in week-1 `prepare_dataset.py`; do not proceed with inflated numbers.

---

## Putting it together — short end-to-end run

From a fresh Colab T4 runtime, the full lab is:
```python
# 1. Predict (locally, before Colab):  uv run python scripts/vram_estimate.py
#    -> write predicted ~9.8 GB into README.

# 2. Load base 4-bit + LoRA (Step 2)
model, tokenizer = FastLanguageModel.from_pretrained(MODEL, MAX_SEQ, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=[...attn+mlp...], use_gradient_checkpointing="unsloth")

# 3. Train + measure peak (Step 3)
torch.cuda.reset_peak_memory_stats()
SFTTrainer(...).train()
peak = torch.cuda.max_memory_allocated()/1e9         # -> write into README, assert <=15
model.save_pretrained("adapters/ticket-router-v1")

# 4. Hot-swap diff (Step 4)   -> diffs/20  (adapter-on vs adapter-off)
# 5. Merge to bf16 (Step 5)   -> models/ticket-router-v1-merged/, identical>=18/20
# 6. Quick eval (Step 6)      -> eval/results/quick_eval.json, JSON-valid + acc deltas
```
End state: adapter in `adapters/`, merged model in `models/`, three result JSONs in `eval/results/`, and README carrying **predicted vs measured VRAM (within 20%)** plus the base-vs-fine-tuned quick-eval table.

---

## Definition of Done

Restating the spine checklist as verifiable checks:
- [ ] **`vram_estimate.py` prediction within ~20% of measured peak.** README shows predicted GB, measured `torch.cuda.max_memory_allocated()` GB, and the computed % gap ≤ 20%.
- [ ] **QLoRA fine-tune of a 7-8B model completes on one GPU at ≤15 GB VRAM; adapter saved.** `adapters/ticket-router-v1/adapter_model.safetensors` exists (tens of MB), and `peak <= 15.0` asserted.
- [ ] **Hot-swap demo: outputs on 20 inputs differ measurably from the no-adapter base.** `eval/results/hotswap_diff.json` exists; `diffs > 0` (aim for most).
- [ ] **Merged bf16 model saved and reproduces adapter outputs (≥18/20 identical or explained).** `models/ticket-router-v1-merged/` holds a full bf16 model; `identical >= 18` or the shortfall is explained in README as bf16-vs-4bit rounding.
- [ ] **JSON-valid rate and category accuracy recorded for base vs fine-tuned on the 100 held-out.** `eval/results/quick_eval.json` + README table with both models' numbers and deltas; contamination check passed (0 leaked).

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at end of epoch | optimizer-step spike | `optim="paged_adamw_8bit"` + `gradient_checkpointing=True`; batch=1, more grad-accum |
| OOM on model load | base loaded in fp16, not 4-bit | confirm `load_in_4bit=True`; confirm runtime is a GPU runtime |
| Predicted vs actual > 20% | forgot CUDA/context headroom, or `max_seq_len` too big | add ~2-3.5 GB headroom to prediction; size seq len to data |
| Loss flat / not decreasing | chat template corrupted or LR too low | print `ds[0]["text"]`; verify EOS after assistant turn; keep LR 2e-4 |
| 0/20 differ in hot-swap | adapter not loaded / `disable_adapter` no-op | check `model.peft_config`; print one output per pass; use trained (not fresh) adapter |
| Merged model outputs garbage | **merged into 4-bit base** | reload base in `bfloat16`; Unsloth `save_method="merged_16bit"` |
| `identical` < 18 after merge | 4-bit(hot-swap) vs bf16(merged) rounding | expected for a few; if most differ, `merge_and_unload()` didn't apply |
| Non-deterministic diffs | sampling on | `do_sample=False` everywhere you compare |
| Out of Colab disk on merge | ephemeral disk too small for ~15 GB | save to mounted Drive or `push_to_hub` |
| Gated repo 401 (Llama) | license not accepted / not logged in | `notebook_login()` + accept license, or switch to Qwen2.5-7B |
| `target_modules` not found | wrong projection names for this arch | print `model`; use correct names or `target_modules="all-linear"` |

**The five pitfalls to flag explicitly (spine):**
1. **`alpha` set without regard to `r`** — effective scale is `alpha/r`. `r=16,alpha=32` → scale 2.0. Changing `r` to 64 while keeping `alpha=32` silently halves your scale.
2. **`target_modules` too narrow or too wide** — attention-only often underfits a formatting task; "target everything" overfits and blows VRAM. Attention + MLP is the sane middle.
3. **Merging into a 4-bit base** — precision loss → garbage. Merge into bf16, quantize *afterward* (week 3).
4. **End-of-epoch OOM spike** — the optimizer step is the peak, not the forward pass. Paged optimizer + grad checkpointing + smaller batch.
5. **Eval contamination** — hash user turns and diff train vs eval; a leaked example inflates every number you report.

---

## Stretch goals (optional)

- **Sweep `r` to see over/under-fit.** Retrain with `r=8`, `r=16`, `r=64` (alpha = 2r each) and plot category accuracy on the held-out 100. On ~400 examples you'll often see `r=64` *lose* to `r=16` — memorization, not capacity. This directly answers the week-2 self-check.
- **Know the other two stacks by porting the config.** Write the *same* run as an **Axolotl** YAML (`adapter: qlora`, `lora_r: 16`, `lora_target_modules: [...]`, `load_in_4bit: true`) and as **raw TRL+PEFT** (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` + `LoraConfig` + `SFTTrainer`). You don't have to run them — producing the configs proves you understand what Unsloth abstracts.
- **Adapter registry.** Save a second task's adapter and write a tiny loader that hot-swaps between `ticket-router-v1` and the other on one resident 4-bit base — the multi-tenant serving pattern that makes LoRA economically interesting.
- **DoRA / rsLoRA one-liner.** Flip `use_rslora=True` (or PEFT `use_dora=True`) and compare held-out accuracy at the same `r`. Awareness-level — just note whether it moved the needle.
- **Measure the merged model's inference speed** (tokens/sec) vs the hot-swapped 4-bit adapter, to feel the hot-swap runtime cost before you quantize in week 3.
