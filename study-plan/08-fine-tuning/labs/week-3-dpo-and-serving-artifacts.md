# Week 3 Lab: DPO Preference Tuning and Compressed Serving Artifacts

> This week you add the **second training stage** — preference tuning with **DPO** — on top of last week's SFT/QLoRA adapter, then turn the winning model into **serving-ready artifacts**. You build a `(prompt, chosen, rejected)` dataset, run TRL's `DPOTrainer` with your SFT model as *both* policy and frozen reference, and prove the DPO model beats SFT-only with a **position-swapped pairwise LLM judge** (not vibes). Then you cross into serving: **merge → quantize → re-evaluate**. You produce three artifacts — a bf16-merged model, a 4-bit **AWQ** (GPU) build, and a **GGUF q4_k_m** (CPU/laptop) build you load in **Ollama** — and score all three on JSON-validity, task accuracy, and tokens/sec so you can *see* what quantization costs. The headline skill: you never ship a preference-tuned or quantized model you haven't re-measured.
>
> Read the matching lectures first — this guide assumes them:
> - [../lectures/08-dpo-preference-tuning.md](../lectures/08-dpo-preference-tuning.md) — the DPO loss, the implicit reward, the reference model as a KL leash, and the one knob (`beta`) that matters.
> - [../lectures/09-preference-method-zoo-and-distillation.md](../lectures/09-preference-method-zoo-and-distillation.md) — KTO/ORPO/GRPO in one line each, and response-level distillation + its licensing caveats.
> - [../lectures/10-quantization-for-serving.md](../lectures/10-quantization-for-serving.md) — GPTQ/AWQ/GGUF/fp8, the merge→quantize→re-eval pipeline, and why you never quantize the 4-bit training base.
>
> It builds directly on the `llm-finetune-lab/` repo and your **Week 2 SFT adapter** (`adapters/ticket-router-v1/`) and merged bf16 model (`models/ticket-router-v1-merged/`). Keep working in the same repo.

**Est. time:** ~9 hrs · **You will need:** the `llm-finetune-lab/` repo (Weeks 1-2), your Week 2 SFT LoRA adapter, a Hugging Face account, and an LLM-judge (Phase 7). **Free/local path:** DPO + AWQ run on a **free Colab T4 (15 GB)**; **GGUF + Ollama run locally on your Windows/Mac laptop with no GPU** — that is your zero-cost serving demo. Default model: `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen2.5-1.5B-Instruct` (fast, not gated) for a full end-to-end pass; scale to your Week 2 7-8B merged model if you have GPU time. The judge can be a small strong model via any API you already use, or a local Ollama model.

---

## Before you start (setup)

**What / Why.** DPO is trainer-ergonomics-identical to SFT (same LoRA setup, same TRL install), so the GPU environment is the one you built in Week 2 — do the DPO + AWQ work in **Colab**. Quantization-for-serving is a *different* toolchain: AWQ needs a GPU (`autoawq`), but **GGUF conversion and Ollama run on CPU on your laptop**. So this lab has two venues on purpose: Colab for training/AWQ, your laptop for GGUF/Ollama. Keep all *scripts* in the repo so they're version-controlled; run the GPU cells in Colab.

**Do it (local, Windows/Git-Bash — pure-Python + data + local serving):**
```bash
cd llm-finetune-lab                     # the repo from Weeks 1-2
mkdir -p adapters models scripts eval/results data/prepared
# data prep, judge, and GGUF/Ollama pieces run locally:
uv add "transformers>=4.44" "datasets" "pydantic" "huggingface_hub"
```

**Do it (Colab — the GPU venue for DPO + AWQ).** New notebook, **Runtime → Change runtime type → T4 GPU**, first cell:
```python
# Colab cell 1 — TRL for DPO (+ peft/bitsandbytes for the 4-bit base)
!pip install -q "trl>=0.11" "peft>=0.11" "transformers>=4.44" "datasets" "accelerate" "bitsandbytes"
import torch; print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0))
```

**Verify (Colab):**
```python
import trl, peft, transformers
print("trl", trl.__version__, "| peft", peft.__version__, "| tf", transformers.__version__)
# TRL API note: DPOConfig/DPOTrainer live in trl>=0.9; the `beta` arg moved onto DPOConfig
# in newer versions. Check `from trl import DPOTrainer, DPOConfig` imports cleanly.
```

**Verify (local — Ollama installed):**
```bash
ollama --version           # install from https://ollama.com/download if missing
```

**Get your data + adapter into Colab.** `git clone` your repo in a Colab cell (or mount Drive). You need: Week 1 `data/prepared/eval.jsonl` (the 100 held-out rows), your Week 2 `adapters/ticket-router-v1/`, and the `data/prepared/dpo.jsonl` you build in Step 1.

**Windows / Git-Bash notes.**
- All `uv run` commands are identical across platforms.
- `bitsandbytes`/`autoawq` do **not** run on Windows-CPU — that is why DPO + AWQ live in Colab. Don't `uv add autoawq` locally on Windows and block on a build error.
- `llama.cpp` builds and runs on Windows, but the simplest path is `pip install llama-cpp-python` for the Python conversion helper, or use the prebuilt release binaries. Ollama has a native Windows installer.
- In Git-Bash, prefer forward slashes in paths; if a tool rewrites `/model` to a Windows path, prefix with `MSYS_NO_PATHCONV=1`.

---

## Step-by-step

### Step 1 — Build the preference dataset `(prompt, chosen, rejected)`

**What.** Create `data/prepared/dpo.jsonl` with ~300-1000 rows, each `{"prompt": <chat-rendered or messages>, "chosen": <better answer>, "rejected": <worse answer>}`. Assert `chosen != rejected` for every row.

**Why.** SFT only ever showed the model one gold target per prompt (lecture 08). DPO needs a *contrast* — a better and a worse answer for the **same** prompt — so it can learn comparative judgement (tone, concision, strict JSON, suppressing bad-but-plausible outputs). If `chosen` and `rejected` are near-identical there's no learning signal; the contrast must be *meaningful*.

**Do it.** The cheapest high-signal source is your own Week 2 model's mistakes:
```python
# scripts/build_dpo_data.py  — turn your model's imperfect outputs into rejected samples
import json, random
from pathlib import Path

def make_triple(prompt_messages, chosen_json: str, rejected_json: str) -> dict:
    """A DPO row. `chosen` should be the correct/strict-JSON answer; `rejected`
    a plausible-but-worse one (wrong field, extra prose, missing key, wrong priority)."""
    assert chosen_json != rejected_json, "chosen and rejected must differ"
    return {"prompt": prompt_messages, "chosen": chosen_json, "rejected": rejected_json}

# Three cheap ways to get a `rejected`:
#  (a) MODEL MISTAKES: prompts where week-2 model emitted invalid/wrong JSON -> that IS the rejected.
#  (b) RULE-BASED CORRUPTION: take a gold `chosen`, programmatically break it
#      (drop a key, add a chatty prefix "Sure! Here is the JSON:", flip priority) -> rejected.
#  (c) STRONGER MODEL: ask a bigger model to produce `chosen`, keep the flawed one as `rejected`.

def corrupt(chosen_json: str) -> str:
    d = json.loads(chosen_json)
    op = random.choice(["prose", "dropkey", "flip"])
    if op == "prose":
        return "Sure! Here is the classification:\n" + chosen_json     # breaks strict-JSON
    if op == "dropkey" and len(d) > 1:
        d.pop(random.choice(list(d.keys())))
    if op == "flip" and "priority" in d:
        d["priority"] = random.choice(["low", "medium", "high"])
    return json.dumps(d)
```
Aim for a mix: ~50% real model mistakes (highest signal), ~50% rule-based corruptions to hit volume. Save with:
```python
rows = [...]  # list of make_triple(...) dicts
Path("data/prepared/dpo.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
```

**Expected result.** `data/prepared/dpo.jsonl` with 300-1000 lines, each parseable JSON with `prompt`, `chosen`, `rejected`.

**Verify.**
```bash
uv run python -c "
import json
rows=[json.loads(l) for l in open('data/prepared/dpo.jsonl', encoding='utf-8') if l.strip()]
assert all(r['chosen']!=r['rejected'] for r in rows), 'chosen==rejected somewhere'
print(len(rows),'rows | all chosen!=rejected OK')
print('sample chosen:', rows[0]['chosen'][:80])
print('sample rejected:', rows[0]['rejected'][:80])
"
```

**Troubleshoot.**
- *All rows look identical / tiny contrast* → your corruptions are too mild. Make rejected clearly worse (prose wrapper, wrong priority, missing key).
- *TRL complains about `prompt` format* → TRL's `DPOTrainer` accepts either a rendered string prompt or a `messages` list; be consistent. If you pass `messages`, ensure your tokenizer has a chat template. Rendering `prompt` with `apply_chat_template(..., add_generation_prompt=True)` and giving raw string `chosen`/`rejected` is the least surprising.

---

### Step 2 — Run DPO on top of your SFT adapter (`scripts/train_dpo.py`)

**What.** Fine-tune with TRL `DPOTrainer`, starting from your **SFT model as both the policy and the reference**. Config: `beta=0.1`, LR ~5e-6 to 5e-5, 1 epoch. Save the new adapter to `adapters/ticket-router-dpo-v1/`.

**Why.** DPO teaches "chosen > rejected" via the implicit reward `beta * log(π_θ/π_ref)` (lecture 08). The frozen reference is the **KL leash**: `beta ~ 0.1` is the standard middle — too low and the model drifts off the SFT distribution and *breaks the JSON formatting SFT taught it*; too high and nothing moves. DPO LR is **much lower than SFT** (you're nudging, not teaching from scratch).

**Do it.** In `scripts/train_dpo.py` (run the training in Colab):
```python
# scripts/train_dpo.py  — DPO on top of the week-2 SFT LoRA adapter
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig

BASE   = "Qwen/Qwen2.5-0.5B-Instruct"          # or your week-2 7-8B base
SFT_ADAPTER = "adapters/ticket-router-v1"       # week-2 SFT LoRA
OUT    = "adapters/ticket-router-dpo-v1"

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token

# Load base (4-bit if you're on a 7-8B model; fp16/bf16 is fine for 0.5-1.5B) ...
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
# ... attach your SFT adapter as the *policy* (trainable):
model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)

ds = load_dataset("json", data_files="data/prepared/dpo.jsonl", split="train")

cfg = DPOConfig(
    output_dir=OUT,
    beta=0.1,                       # the KL leash; 0.1 is the safe default
    learning_rate=5e-6,             # DPO LR is ~1-2 orders lower than SFT
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_length=1024, max_prompt_length=512,
    logging_steps=10, save_strategy="epoch",
    bf16=True,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,                 # None => TRL uses the base *without* the adapter as reference
    args=cfg,
    train_dataset=ds,
    processing_class=tok,           # newer TRL; older versions use tokenizer=tok
    peft_config=None,               # adapter already attached; or pass a fresh LoraConfig
)
trainer.train()
trainer.save_model(OUT)             # saves the DPO adapter
```
> Note on the reference model: with a PEFT policy, passing `ref_model=None` makes TRL use the **base model with the adapter disabled** as the reference — memory-efficient and exactly what you want (reference = your SFT model frozen). Watch the printed `rewards/chosen`, `rewards/rejected`, and `rewards/margins` — the margin should grow positive.

**Expected result.** Training completes in a few minutes (0.5B) to ~1 hr (7-8B on T4). `adapters/ticket-router-dpo-v1/` written. Logs show `rewards/margins` trending up and `rewards/accuracies` climbing above 0.5.

**Verify.**
```python
# in Colab after training
import glob; print("saved:", glob.glob("adapters/ticket-router-dpo-v1/*"))
# generate one sample and eyeball it's still valid JSON (DPO didn't break format)
```

**Troubleshoot.**
- *`rewards/margins` stays ~0* → contrast too weak (revisit Step 1) or `beta` too high. Lower to `0.05` only if formatting holds.
- *JSON formatting breaks after DPO* → `beta` too low / LR too high. Raise `beta` toward 0.2, drop LR to 5e-6. Re-check task metrics (Step 3) — style wins that tank accuracy are the classic DPO trap.
- *OOM on 7-8B* → keep the 4-bit base, `per_device_train_batch_size=1`, gradient checkpointing on, shorter `max_length`.
- *`DPOConfig` has no `beta`* → very new/old TRL split this. If `beta` isn't on `DPOConfig`, pass it to `DPOTrainer(..., beta=0.1)`; check `from trl import DPOConfig` and the installed docstring.

---

### Step 3 — Win-rate eval: position-swapped pairwise judge (`eval/judge.py`)

**What.** On 50 held-out prompts, generate from **SFT-only** and **SFT+DPO**, then use an LLM judge to pick the better answer *twice per pair with the order swapped*. Report win / tie / loss and net win-rate.

**Why.** "Loss went down" proves nothing about preference quality (lecture 12). A pairwise judge does — but LLM judges have a **position bias** (they favor whichever answer is shown first). Swapping the order and only counting a win when it's consistent across both orders de-biases the result. This is the acceptance instrument for the DPO stage.

**Do it.** In `eval/judge.py`:
```python
# eval/judge.py — position-swapped pairwise judge
import json

JUDGE_PROMPT = """You compare two assistant answers to the same task.
Task/prompt:
{prompt}

Answer A:
{a}

Answer B:
{b}

Which answer is better for a strict-JSON ticket-routing task (valid JSON, correct
category/priority, no extra prose)? Reply with exactly one token: A, B, or TIE."""

def judge_once(call_llm, prompt: str, a: str, b: str) -> str:
    """call_llm(str)->str is your judge model (API or local Ollama). Returns 'A'|'B'|'TIE'."""
    out = call_llm(JUDGE_PROMPT.format(prompt=prompt, a=a, b=b)).strip().upper()
    return "A" if out.startswith("A") else "B" if out.startswith("B") else "TIE"

def pairwise_swapped(call_llm, prompt, sft_out, dpo_out) -> str:
    """Run both orders; a real win must hold in BOTH. Returns 'DPO'|'SFT'|'TIE'."""
    r1 = judge_once(call_llm, prompt, a=dpo_out, b=sft_out)   # DPO is A
    r2 = judge_once(call_llm, prompt, a=sft_out, b=dpo_out)   # DPO is B
    dpo_wins = (r1 == "A") + (r2 == "B")
    sft_wins = (r1 == "B") + (r2 == "A")
    if dpo_wins == 2: return "DPO"
    if sft_wins == 2: return "SFT"
    return "TIE"                                              # disagreement => tie (position bias caught)

def net_win_rate(results: list[str]) -> dict:
    n = len(results)
    w = results.count("DPO"); l = results.count("SFT"); t = results.count("TIE")
    return {"n": n, "dpo_wins": w, "sft_wins": l, "ties": t,
            "net_win_rate": round((w - l) / n, 3)}
```
Wire `call_llm` to whatever judge you used in Phase 7 (a strong API model, or a local `ollama run` model — keep it *different* from the model under test). Generate the 50 SFT and 50 DPO outputs first (load each adapter, `model.generate`), then loop `pairwise_swapped`.

**Expected result.** A dict like `{"n":50, "dpo_wins":21, "sft_wins":9, "ties":20, "net_win_rate":0.24}`. Save to `eval/results/dpo_winrate.json`.

**Verify.** Net win-rate > 0 with DPO wins clearly exceeding SFT wins. Spot-check 3 "DPO win" pairs by hand — do you agree? Then confirm **task metrics didn't regress**: re-run your Week 2 JSON-valid rate + category accuracy on SFT vs DPO. DPO must not drop task accuracy.

**Troubleshoot.**
- *All ties* → judge output not parsed (returns prose, not A/B/TIE). Constrain harder or parse the first non-space char.
- *DPO wins style but task accuracy dropped* → the classic trap (lecture 08). Raise `beta`, lower LR, retrain. Style win + accuracy loss is a **reject**.
- *Judge always picks A* → that's exactly the position bias the swap catches; if swapped runs disagree it correctly becomes TIE. Don't "fix" by removing the swap.

---

### Step 4 — Merge the winning adapter, then quantize (AWQ + GGUF)

**What.** Merge the chosen adapter (DPO if it won, else SFT) into a **bf16** base → `models/ticket-router-v2-merged/`. Then produce (a) **AWQ 4-bit** via AutoAWQ (GPU), and (b) **GGUF q4_k_m** via llama.cpp (laptop).

**Why.** The one rule you never violate (lecture 10): **merge into fp16/bf16 FIRST, then quantize that merged model.** Quantizing the 4-bit *training* base gives double-quantization damage — subtly broken output. AWQ targets GPU servers (vLLM/transformers); GGUF targets laptops/Ollama.

**Do it — merge (Colab, bf16):**
```python
# merge the DPO (or SFT) adapter into a clean bf16 base
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "adapters/ticket-router-dpo-v1"     # the winner from Step 3
OUT = "models/ticket-router-v2-merged"

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()   # bf16 base, NOT 4-bit
merged.save_pretrained(OUT); AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
```

**Do it — AWQ (Colab, GPU):**
```python
!pip install -q autoawq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODEL = "models/ticket-router-v2-merged"; OUT = "models/ticket-router-v2-awq"
model = AutoAWQForCausalLM.from_pretrained(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
# Calibrate on data that LOOKS LIKE YOUR TRAFFIC (real ticket prompts), not random web text:
calib = [json.loads(l)["prompt"] for l in open("data/prepared/eval.jsonl")][:128]
model.quantize(tok, quant_config={"w_bit": 4, "q_group_size": 128, "zero_point": True},
               calib_data=calib)
model.save_quantized(OUT); tok.save_pretrained(OUT)
```

**Do it — GGUF (local laptop, CPU):**
```bash
# clone llama.cpp once; it ships the converter + quantizer
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && pip install -r requirements.txt
# 1) convert merged HF model -> fp16 GGUF
python convert_hf_to_gguf.py ../models/ticket-router-v2-merged \
       --outfile ../models/ticket-router-v2-f16.gguf --outtype f16
# 2) quantize fp16 GGUF -> q4_k_m  (build llama-quantize first: `cmake -B build && cmake --build build`)
./build/bin/llama-quantize ../models/ticket-router-v2-f16.gguf \
       ../models/ticket-router-v2-q4_k_m.gguf Q4_K_M
```
> Windows/Git-Bash: build with `cmake -B build && cmake --build build --config Release`; the binary is `build/bin/Release/llama-quantize.exe`. If cmake is a hassle, download a prebuilt release from the llama.cpp releases page and use its `llama-quantize` binary. The Python `convert_hf_to_gguf.py` script runs the same on all platforms.

**Expected result.** Three artifacts on disk: `models/ticket-router-v2-merged/` (bf16, ~1 GB for 0.5B), `models/ticket-router-v2-awq/` (~0.3-0.4 GB), `models/ticket-router-v2-q4_k_m.gguf` (~0.3-0.4 GB).

**Verify.**
```bash
ls -la models/ticket-router-v2-merged/ models/ticket-router-v2-awq/ 2>/dev/null
ls -la models/ticket-router-v2-q4_k_m.gguf
# sanity: file sizes shrink merged(fp16) > awq(~4bit) ~ gguf(q4)
```

**Troubleshoot.**
- *Garbage output from quantized model* → you likely quantized the 4-bit training base, not the bf16-merged one. Re-merge into bf16 first.
- *AWQ calibration slow/OOM* → fewer calib samples (64) or shorter prompts; still keep them domain-representative.
- *`convert_hf_to_gguf.py` unknown architecture* → update llama.cpp (`git pull`); newer model families need a recent converter.
- *AutoAWQ install fails on Windows* → correct; run AWQ in Colab. GGUF is your local path.

---

### Step 5 — Serve GGUF in Ollama and score all three (`eval/quant_scorecard.py`)

**What.** Load the GGUF in **Ollama** locally, then run your 100 held-out examples through **bf16-merged**, **AWQ**, and **GGUF-q4** and build a table of JSON-valid %, task accuracy, and tokens/sec.

**Why.** Quantization is lossy and *silently* so — it can cost 3-4 points of accuracy while still producing fluent text (lecture 10). The only way to know is to re-score. Ollama proves the zero-cost local serving path actually answers.

**Do it — Ollama (local):**
```bash
# Modelfile points at your GGUF
cat > Modelfile <<'EOF'
FROM ./models/ticket-router-v2-q4_k_m.gguf
PARAMETER temperature 0
TEMPLATE """{{ .Prompt }}"""
EOF
ollama create ticket-router-v2 -f Modelfile
ollama run ticket-router-v2 "Classify: my invoice is wrong and I'm furious"   # should emit JSON
```

**Do it — scorecard (`eval/quant_scorecard.py`):**
```python
# eval/quant_scorecard.py — JSON-valid %, accuracy, tokens/sec across artifacts
import json, time

def score_backend(name, generate, eval_rows) -> dict:
    """generate(prompt)->(text, n_tokens). Reuses your week-2 task metrics."""
    ok_json = correct = total = tot_tok = 0; t0 = time.time()
    for r in eval_rows:
        text, ntok = generate(r["prompt"]); tot_tok += ntok; total += 1
        try:
            pred = json.loads(text); ok_json += 1
            if pred.get("category") == r["gold"]["category"]: correct += 1
        except Exception:
            pass
    dt = time.time() - t0
    return {"backend": name, "json_valid_pct": round(100*ok_json/total, 1),
            "category_acc_pct": round(100*correct/total, 1),
            "tokens_per_sec": round(tot_tok/dt, 1)}

# backends:
#   bf16   -> transformers .generate on models/ticket-router-v2-merged (Colab GPU)
#   awq    -> transformers/vLLM load of models/ticket-router-v2-awq   (Colab GPU)
#   gguf   -> ollama.generate("ticket-router-v2", prompt)  OR llama-cpp-python (laptop CPU)
rows = [...]
table = [score_backend("bf16", gen_bf16, rows),
         score_backend("awq",  gen_awq,  rows),
         score_backend("gguf-q4", gen_gguf, rows)]
json.dump(table, open("eval/results/quant_scorecard.json","w"), indent=2)
print(f"{'backend':10}{'json%':>8}{'acc%':>8}{'tok/s':>8}")
for t in table: print(f"{t['backend']:10}{t['json_valid_pct']:>8}{t['category_acc_pct']:>8}{t['tokens_per_sec']:>8}")
# STATE the delta: bf16 acc - gguf acc = the price of quantization.
```
For the GGUF backend without Ollama you can use `llama-cpp-python`: `pip install llama-cpp-python`, then `Llama(model_path=..., n_ctx=1024)`.

**Expected result.** A 3-row table. Typical shape: AWQ/GGUF within a few points of bf16 on accuracy, GGUF slowest on CPU but *runs with no GPU*. Save to `eval/results/quant_scorecard.json`.

**Verify.** All three rows populated; you can state one sentence: "quantization cost N points of category accuracy (bf16 X% → GGUF-q4 Y%)." If a quantized backend collapses (JSON-valid crashes), that's a real finding — investigate, don't hide it.

**Troubleshoot.**
- *Ollama emits prose, not JSON* → your `TEMPLATE` is wrapping the prompt; use `TEMPLATE """{{ .Prompt }}"""` to pass your already-formatted prompt through untouched, and set `temperature 0`.
- *tokens/sec not comparable* → measure each backend on the same 100 rows, same max_new_tokens; report CPU vs GPU explicitly (GGUF on CPU is expected to be slower — that's the trade for zero cost).
- *Big accuracy drop after quant* → options: try a higher-bit GGUF (`Q5_K_M`/`Q6_K`), GPTQ instead of AWQ, better/more-representative AWQ calibration data, or accept it and note it in the model card.

---

## Putting it together — a short end-to-end run

A single pass, small model, ~90 min of active work:

```bash
# 1) LOCAL: build preference data from week-2 mistakes + corruptions
uv run python scripts/build_dpo_data.py           # -> data/prepared/dpo.jsonl  (assert chosen!=rejected)

# 2) COLAB: DPO on top of the SFT adapter (beta=0.1, LR 5e-6, 1 epoch)
python scripts/train_dpo.py                        # -> adapters/ticket-router-dpo-v1/

# 3) COLAB/LOCAL: position-swapped judge, SFT vs SFT+DPO on 50 prompts
uv run python eval/judge.py                        # -> eval/results/dpo_winrate.json (net_win_rate > 0)

# 4) COLAB: merge WINNER into bf16, then AWQ ; LOCAL: GGUF q4_k_m
#    (merge_and_unload into bf16, then autoawq, then convert_hf_to_gguf + llama-quantize)

# 5) LOCAL: serve GGUF in Ollama, score all three backends
ollama create ticket-router-v2 -f Modelfile
uv run python eval/quant_scorecard.py              # -> eval/results/quant_scorecard.json
```

You end the week with: a preference dataset, a DPO adapter that *provably* beats SFT on a de-biased judge, three serving artifacts (bf16 / AWQ / GGUF), a live local Ollama model, and a post-quant scorecard that states exactly what quantization cost.

---

## Definition of Done

Restating the spine's Week 3 checklist as verifiable checks:

- [ ] **`dpo.jsonl` valid.** Every row parses; `chosen != rejected` asserted in code (Step 1 verify block passes).
- [ ] **DPO beats SFT on the judge.** DPO run completes; SFT+DPO beats SFT-only on the pairwise judge **with position-swap de-biasing**; you report **net win-rate and n** (`eval/results/dpo_winrate.json`), and task accuracy did **not** regress.
- [ ] **Three serving artifacts exist.** `models/ticket-router-v2-merged/` (bf16), `...-awq/` (or GPTQ) 4-bit, and `...-q4_k_m.gguf`.
- [ ] **GGUF runs in Ollama locally.** `ollama run ticket-router-v2 "..."` returns a valid answer on your laptop, no GPU.
- [ ] **Post-quant scorecard table.** JSON-valid %, task accuracy, tokens/sec for **bf16 vs AWQ vs GGUF**, with the accuracy delta from quantization **stated in one sentence**.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `rewards/margins` stuck at ~0 | contrast too weak / `beta` too high | stronger rejected samples; lower `beta` to 0.05 only if format holds |
| JSON format breaks after DPO | `beta` too low / LR too high — drifted off SFT | raise `beta` toward 0.2, drop LR to 5e-6, re-check task metrics |
| DPO wins style but accuracy drops | the classic DPO trap | that's a **reject** — retrain tighter; always re-run task metrics after DPO |
| Judge returns all TIEs | judge output not parsed / position bias caught | constrain judge to `A/B/TIE`, parse first char; swap disagreement→TIE is correct |
| Quantized model outputs garbage | quantized the 4-bit training base | re-merge adapter into **bf16** first, then quantize |
| AutoAWQ won't install on Windows | GPU-only toolchain | run AWQ in Colab; use GGUF for the local path |
| `convert_hf_to_gguf.py` unknown arch | old llama.cpp | `git pull` llama.cpp; use a recent converter |
| Ollama emits prose not JSON | Modelfile `TEMPLATE` re-wrapping prompt | `TEMPLATE """{{ .Prompt }}"""`, `PARAMETER temperature 0` |
| Big accuracy drop after quant | lossy quant / bad calibration | try `Q5_K_M`/`Q6_K`, GPTQ, or domain-representative AWQ calibration data |
| OOM during DPO on 7-8B | batch/seq too large | 4-bit base, batch=1, grad-checkpoint on, shorter `max_length` |

---

## Stretch goals (optional)

- **Try a zoo member (awareness → practice).** Swap `DPOTrainer` for **ORPO** (`ORPOTrainer` — combines SFT + preference in one stage, *no reference model*) or **KTO** (`KTOTrainer` — binary good/bad labels, easier data collection). Compare win-rate and training time against DPO. One line each on when you'd pick them (lecture 09).
- **Response-level distillation.** Generate `chosen` answers from a strong teacher model to build a cleaner preference set, then note the **licensing caveat** — some model terms forbid using outputs to train a competing model (lecture 09). Write the caveat into your README.
- **Serve AWQ on vLLM.** `vllm serve models/ticket-router-v2-awq --quantization awq` and hit it with an OpenAI-compatible request; compare tokens/sec against transformers and the GGUF/Ollama path.
- **Sweep `beta`.** Train `beta ∈ {0.05, 0.1, 0.3}`, plot win-rate vs task-accuracy — see the leash trade-off in your own numbers.
- **fp8 (if you have H100-class access).** Produce an fp8 build and add it as a fourth scorecard row; note it's H100-class only (lecture 10).
- **GGUF quant ladder.** Build `Q4_K_M`, `Q5_K_M`, `Q6_K` and chart the accuracy-vs-size-vs-speed curve to pick your laptop-serving sweet spot.
