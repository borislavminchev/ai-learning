# Phase 08 - Model Adaptation & Fine-Tuning

Turn a generic model into a cheaper, faster, more reliable specialist - but only after prompting and RAG have *provably* failed. Fine-tuning changes **behavior, format, and style**, not facts. By the end of this phase you can decide (with numbers) between prompt/RAG/LoRA, ship a QLoRA fine-tune of a 7-8B model on free/cheap GPUs, and prove it didn't get dumber via a capability-retention scorecard.

Nav: Prev: `07-evaluation-observability.md` · Next: `09-architecture-system-design.md`

## Prerequisites
- Phase 0 foundations: VRAM math (params x bytes + KV cache), tokenization, chat templates, quantization basics (fp16/int8/int4).
- Phase 2: structured outputs / Pydantic (your fine-tune target is usually clean JSON).
- Phase 4: a working RAG baseline (you will A/B against it).
- Phase 7: an eval harness. **You cannot fine-tune responsibly without one.** If you don't have a golden set + an LLM-judge, go back.
- Comfort with Python, `uv`/`pip`, Hugging Face `transformers`/`datasets`, and reading a `nvidia-smi` output.
- A Hugging Face account (for gated models like Llama/Gemma) and a free Google Colab or Modal account.

## Time budget
4 weeks x ~10-15 hrs/week (~35% theory / 65% hands-on). Each week states its own breakdown. If you have less time, do every **Lab** and skip half the reading - the labs are the point.

## How to use this file
Work top to bottom; each week's Definition of Done is a hard gate - don't advance until every box is checked. Keep one git repo (`llm-finetune-lab/`) for the whole phase; every lab adds to it. Treat the milestone project as the thing you show in interviews.

> **📚 Course materials.** This file is the **spine** — a weekly plan and a *recap*. Deep material lives alongside it:
> - **Lectures**: [`lectures/`](lectures/00-index.md) — read for the *why*/mechanism.
> - **Lab guides**: [`labs/`](labs/) — follow to *build*.
> Each week: read the linked lectures first (the Theory here is their recap), then work the linked lab guide.

---

## Week 1 - Decide, then learn SFT properly

### Objectives
By the end of the week you can:
- Apply a written **prompt vs RAG vs fine-tune decision framework** to a real task and justify it in one page.
- Explain why fine-tuning is *bad* at injecting facts and *good* at behavior/format/tone, with a concrete example.
- Prepare an SFT dataset in chat format and render it with `tokenizer.apply_chat_template`, verifying EOS tokens and which tokens are masked.
- Run a first supervised fine-tune (SFT) with HF TRL's `SFTTrainer` on a small model and observe train/eval loss curves.
- State the effect of **response-only loss masking** and show the before/after on your data.

### Theory (~4 hrs)
> **📖 Deep lectures for this week** (read first): [1 · Prompt vs RAG vs Fine-Tune decision](lectures/01-prompt-vs-rag-vs-finetune-decision.md) · [2 · SFT mechanics & the three silent killers](lectures/02-sft-mechanics-and-the-three-silent-killers.md) · [3 · Base vs instruct & small-data hyperparameters](lectures/03-base-vs-instruct-and-small-data-hyperparameters.md). The bullets below are the recap.

- **The decision framework.** Read the Hugging Face TRL docs (`SFTTrainer`) and the OpenAI "fine-tuning" guide intro for the standard heuristic. Internalize the rule: prompt first, RAG for knowledge, fine-tune for **behavior you can't reliably prompt** (format adherence, tone, a narrow classification, latency/cost by shrinking prompts, distillation of a big model into a small one). Signals *for* tuning: >=1k clean examples, stable task, cost/latency pressure, consistent structured output. Fine-tuning does **not** reliably teach new facts - it teaches a *style of answering*.
- **SFT mechanics.** Next-token prediction on (prompt, completion) pairs. The three silent killers: (1) **loss masking** - only compute loss on assistant/response tokens, never on the prompt, or the model learns to parrot instructions; (2) **EOS token** - if completions don't end with EOS the model never learns to stop and rambles at inference; (3) **chat template** - you must use the *exact* template the base/instruct model expects (`apply_chat_template`), or you silently corrupt training. Read the HF blog "chat templates" explainer (search: "Hugging Face chat templates blog").
- **Base vs instruct starting point.** Instruct/chat models already know the template and how to follow instructions - usually your best SFT base. Base models need more data and you own the template design.
- **Hyperparameters that matter for small data.** LR (1e-4 to 2e-4 for LoRA is a common start, ~an order of magnitude lower for full FT), 1-3 epochs (more overfits fast on <5k examples), warmup, cosine schedule, packing (concatenate short samples to fill context - speeds training but watch cross-contamination and template boundaries).
- Skim only: the LoRA paper's *intuition* (low-rank deltas), not the math - you'll build it next week.

### Lab (~8 hrs)
> 🛠️ Full step-by-step guide: [Repo Skeleton and Your First SFT with Masking A/B](labs/week-1-repo-skeleton-and-first-sft.md). The steps below are the summary.

Build the repo skeleton and run your first SFT.

```
llm-finetune-lab/
  pyproject.toml            # uv-managed
  data/
    raw/                    # source data (e.g. support tickets)
    prepared/               # chat-format JSONL
  scripts/
    prepare_dataset.py
    inspect_template.py
    train_sft.py
  eval/                     # (grows in week 4)
  README.md
```

1. **Env.** `uv init && uv add "transformers>=4.44" "trl>=0.9" "datasets" "accelerate" "peft" "bitsandbytes" "pydantic"`. On Windows/CPU-only laptops, training runs on **Google Colab (free T4)** - put `train_sft.py` behind a `--device` flag and run the actual training there. Local machine is for data prep + inspection.
2. **Pick one real task.** Recommended: classify/route support tickets into `{category, priority, needs_human}` as strict JSON. Get or synthesize ~300-800 examples. Keep 100 held-out for eval (never train on them).
3. **`prepare_dataset.py`.** Emit chat-format JSONL: each row `{"messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant","content": <JSON>}]}`. Assistant content is the exact target JSON. Save to `data/prepared/train.jsonl` and `eval.jsonl`.
4. **`inspect_template.py`.** Load the tokenizer for your chosen model (e.g. `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Llama-3.1-8B-Instruct`). Run `apply_chat_template(msgs, tokenize=False)` and **print it** - eyeball the special tokens, confirm an EOS ends the assistant turn. Then tokenize and print which token IDs would be masked (label = -100) under response-only masking.
5. **`train_sft.py`.** Use TRL `SFTTrainer` (or `SFTConfig`) with `Qwen2.5-0.5B-Instruct` **first** (fast, fits a free GPU/CPU) to debug the pipeline end to end. Enable response-only loss (TRL's `DataCollatorForCompletionOnlyLM` or the `assistant_only_loss`/`completion_only` option in your TRL version - check the installed docs). Log train + eval loss.
6. **A/B the masking.** Train once with response-only masking, once without (loss on the whole sequence). Compare eval loss and eyeball 10 generations. Write the difference into `README.md`.

Free/cheap GPU: Colab free T4 handles 0.5B-3B QLoRA easily; the 7-8B QLoRA run is week 2 (still fits a T4/L4). No paid API required.

### Definition of Done
- [ ] `data/prepared/*.jsonl` validates: every row parses, assistant content is valid JSON, train and eval are disjoint (assert in code).
- [ ] `inspect_template.py` prints a rendered template showing an EOS after the assistant turn, and prints the count of masked vs unmasked label tokens for one example.
- [ ] `train_sft.py` completes on the 0.5B model with eval loss logged for >=2 eval steps and a decreasing train loss.
- [ ] The masking A/B is written up: response-only masking yields lower eval loss OR you can explain why not, with numbers.
- [ ] One-page **decision-framework note** in `README.md` for your task: why you're even considering fine-tuning vs prompt vs RAG.

### Pitfalls
- Forgetting EOS -> model never stops generating at inference. Verify in the rendered template, not just in theory.
- Hand-building the chat format instead of `apply_chat_template` -> silent template mismatch, garbage results, and you'll blame the model.
- Training loss on prompt tokens -> the model learns to echo instructions; looks "fine" on loss but degrades output quality.
- Too many epochs on a tiny dataset -> memorization; eval loss turns up while train loss keeps dropping.
- Testing on training examples -> fake wins. Assert disjointness in code.

### Self-check
1. Give one task where fine-tuning is the right tool and one where RAG is, and say why.
2. What exactly does response-only loss masking change in the loss computation?
3. Why can two models with identical architectures still require different chat templates?
4. Your fine-tune rambles and won't stop - name the two most likely causes.

---

## Week 2 - PEFT for real: LoRA, QLoRA, and VRAM math

### Objectives
By the end of the week you can:
- Explain and set LoRA `r`, `alpha`, `target_modules`, and dropout, and say what each knob does.
- Run a **QLoRA** fine-tune of a **7-8B model on a single free/cheap GPU** (4-bit NF4 + paged optimizer).
- Compute the VRAM budget for full-FT vs LoRA vs QLoRA of a given model *before* renting a GPU, and be within ~20% of reality.
- Hot-swap a LoRA adapter at inference and separately **merge** it into an fp16 base.
- Prepare a small, high-quality, decontaminated dataset and justify quality over quantity.

### Theory (~4 hrs)
> **📖 Deep lectures for this week** (read first): [4 · LoRA mechanics & adapter lifecycle](lectures/04-lora-mechanics-and-adapter-lifecycle.md) · [5 · QLoRA 4-bit NF4 training](lectures/05-qlora-4bit-training.md) · [6 · VRAM math & PEFT vs full FT](lectures/06-vram-math-and-peft-vs-full-ft.md) · [7 · Dataset quality & decontamination](lectures/07-dataset-quality-and-decontamination.md). The bullets below are the recap.

- **LoRA** (HF PEFT docs). You freeze the base weights and train small low-rank matrices (A x B) injected into chosen linear layers. `r` = rank/capacity (8-64 typical; higher = more capacity + more VRAM/overfit risk); `alpha` = scaling (common: alpha = r or 2r; effective scale = alpha/r); `target_modules` = which projections get adapters (start with attention `q_proj,k_proj,v_proj,o_proj`; add MLP `gate/up/down_proj` for more capacity). Adapters are tiny (MBs), **hot-swappable**, and **mergeable** back into the base. Awareness only: rsLoRA, DoRA (better-behaved variants) - know they exist.
- **QLoRA** (bitsandbytes + PEFT docs; search "QLoRA Unsloth notebook"). Base weights quantized to **4-bit NF4**, LoRA adapters in bf16, **paged optimizer** to survive memory spikes. This is what lets a 7-8B model fine-tune on a 16GB (or even free T4 15GB) GPU. Rule: **quantize the base for training, but merge the adapter into an fp16/bf16 base for serving**, then quantize *that* for deployment (week 3).
- **Full-FT vs PEFT.** Full FT needs weights + gradients + optimizer states (Adam ~= 2x params in fp32 states) + activations -> often 12-20x model size in VRAM. PEFT trains <1% of params -> gradients/optimizer states only for adapters. For almost all industry work, **PEFT is the default**; full FT is for continued pretraining or when you truly reshape the model.
- **VRAM math (do it by hand).** Inference fp16 ~= 2 bytes x params (+ KV cache). QLoRA base ~= 0.5 byte x params (4-bit) + adapter + optimizer(adapter) + activations. Read the Unsloth README's VRAM tables and the HF "Model Memory" utility to calibrate.
- **Dataset prep.** Quality dominates: clean, diverse, correctly-formatted, and **decontaminated** (no eval/benchmark examples leaked into train). 500 great examples beat 50k noisy ones. Synthetic data is fine *with verification/filtering* (avoid model-collapse feedback loops).

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [QLoRA a 7-8B Model and Manage Adapters](labs/week-2-qlora-7b-and-adapter-management.md). The steps below are the summary.

Fine-tune a 7-8B model with QLoRA and manage adapters.

1. **Tooling.** Two viable stacks - do it with **Unsloth** (fastest, lowest VRAM, great free-Colab notebooks) and know that **Axolotl** (YAML-config, production-friendly) and raw **TRL+PEFT** exist. `uv add unsloth` in Colab, or start from the official Unsloth Colab notebook for Llama-3.1-8B / Qwen2.5-7B QLoRA.
2. **VRAM estimate first.** In `scripts/vram_estimate.py`, write a function that takes `params_billion, dtype_bytes, mode` and prints estimated training VRAM for full-FT / LoRA / QLoRA. Predict for your 8B model, then run and record actual peak (`torch.cuda.max_memory_allocated()`). Log both in README.
3. **QLoRA train.** Config: 4-bit NF4, `r=16, alpha=32, dropout=0.05`, target attention + MLP projections, paged AdamW 8-bit, LR 2e-4, cosine, 1-2 epochs, gradient checkpointing on, max_seq_len sized to your data (don't waste VRAM). Train on your week-1 dataset. Save adapter to `adapters/ticket-router-v1/`.
4. **Hot-swap inference.** Load the 4-bit base + attach adapter with PEFT; generate on 20 eval inputs. Then load base **without** adapter and generate on the same inputs - diff the outputs to prove the adapter is doing something.
5. **Merge.** `model.merge_and_unload()` (or Unsloth `save_pretrained_merged`) into a **bf16** base; save to `models/ticket-router-v1-merged/`. Confirm merged model reproduces adapter outputs.
6. **Quick eval.** Run your held-out 100 through both base-instruct and fine-tuned; measure JSON-valid rate and category accuracy. Record deltas (full eval harness is week 4).

Cost note: the whole week runs on **free Colab T4** or a ~$0.50-1.00/hr L4/A10 on Modal/RunPod if you want speed. Nothing here needs a paid API.

### Definition of Done
- [ ] `vram_estimate.py` prediction is within ~20% of measured peak VRAM for the QLoRA run (log both numbers).
- [ ] QLoRA fine-tune of a 7-8B model completes on a single GPU with <=15GB VRAM; adapter saved.
- [ ] Adapter hot-swap demo: outputs on 20 inputs differ measurably from the no-adapter base.
- [ ] Merged bf16 model saved and reproduces adapter outputs on the same 20 inputs (>=18/20 identical or explained).
- [ ] JSON-valid rate and category accuracy recorded for base vs fine-tuned on the 100 held-out examples.

### Pitfalls
- Setting `alpha` without regard to `r` -> effective scale surprises; remember effective scale = alpha/r.
- `target_modules` too narrow (attention only) when the task needs capacity -> underfit; too wide -> overfit/VRAM blowup.
- Merging a LoRA adapter into a **4-bit** base -> precision loss/garbage. Merge into fp16/bf16, quantize afterward.
- OOM at the very end of an epoch (optimizer step spike) -> use paged optimizer + gradient checkpointing + smaller batch/grad-accum.
- Leaking eval examples into train ("contamination") -> inflated scores. Hash and diff train vs eval.

### Self-check
1. Why does QLoRA let you train an 8B model on 15GB when full FT needs >100GB?
2. What's the difference between hot-swapping and merging an adapter, and when do you use each?
3. You set `r=64` and accuracy drops vs `r=16` on 400 examples - what happened?
4. Estimate training VRAM for a 3B model under QLoRA. Show the terms.

---

## Week 3 - Alignment (DPO) & compression for serving

### Objectives
By the end of the week you can:
- Explain when to reach for **preference tuning** (DPO) after SFT, and prepare a `(prompt, chosen, rejected)` dataset.
- Run a **DPO** fine-tune with TRL on top of your SFT adapter and measure a win-rate improvement.
- Name KTO/ORPO/GRPO and say in one line when each applies (awareness, not implementation).
- Produce **serving-optimized** variants of your merged model: GPTQ/AWQ (GPU) and GGUF (llama.cpp/Ollama), and re-evaluate after quantizing.
- Explain response-level distillation and its licensing caveats.

### Theory (~4 hrs)
> **📖 Deep lectures for this week** (read first): [8 · DPO preference tuning](lectures/08-dpo-preference-tuning.md) · [9 · The preference-method zoo & distillation](lectures/09-preference-method-zoo-and-distillation.md) · [10 · Quantization for serving](lectures/10-quantization-for-serving.md). The bullets below are the recap.

- **Why preference tuning.** SFT teaches "a good answer"; DPO teaches "this answer is *better* than that one" - it sharpens tone, safety, formatting, and reduces bad-but-plausible outputs. **DPO is the practical default** (no separate reward model, no RL loop, stable). Read the TRL `DPOTrainer` docs. Key knob: `beta` (KL strength) - low beta drifts far from the reference model, high beta stays close. You need a reference model (usually your SFT model frozen).
- **The zoo, briefly.** KTO (binary good/bad labels - easier data collection), ORPO (combines SFT+preference in one stage, no reference model), GRPO (RL for *reasoning*/verifiable-reward tasks - what powers reasoning-model training). One line each; don't implement. DPO covers ~90% of industry preference needs.
- **Distillation.** Response-level: generate outputs from a big teacher, SFT a small student on them. Cheap, effective for narrowing a small model onto a task. Watch **licensing** - some model terms forbid using outputs to train competitors. Logit-level exists but needs teacher logits + more infra.
- **Quantization for serving** (distinct from QLoRA-for-training). After you **merge** to fp16/bf16: **GPTQ/AWQ** (4-bit, GPU inference via vLLM/transformers), **GGUF** (llama.cpp/Ollama, CPU+GPU, great for local), **fp8** (H100-class). Rule: **quantize after merging**, then **re-evaluate** - quantization can silently cost a few points; measure it. Read the AutoAWQ and llama.cpp READMEs.

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [DPO Preference Tuning and Compressed Serving Artifacts](labs/week-3-dpo-and-serving-artifacts.md). The steps below are the summary.

Add a preference stage and ship compressed artifacts.

1. **Preference data.** Build ~300-1000 `(prompt, chosen, rejected)` triples. Cheap source: take prompts where your week-2 model was imperfect, use a stronger model (or a rule) to produce a `chosen`, and use the flawed output as `rejected`. Save `data/prepared/dpo.jsonl`.
2. **DPO train** (`scripts/train_dpo.py`). TRL `DPOTrainer`, start from your SFT (LoRA) model as both policy and reference, `beta=0.1`, LR ~5e-6 to 5e-5, 1 epoch. Save `adapters/ticket-router-dpo-v1/`.
3. **Win-rate eval.** On 50 held-out prompts, generate from SFT-only and SFT+DPO; use an **LLM judge** (from Phase 7) doing pairwise comparison with **position-swap** to de-bias. Report win/tie/loss and net win-rate.
4. **Merge + quantize.** Merge the chosen adapter into bf16. Produce: (a) **AWQ** 4-bit via AutoAWQ (or GPTQ), (b) **GGUF** q4_k_m via `llama.cpp` `convert` + `quantize`, and load the GGUF in **Ollama** (`ollama create` from a Modelfile) to prove local serving.
5. **Re-eval after quant.** Run the 100 held-out through bf16-merged, AWQ, and GGUF-q4. Record JSON-valid rate, task accuracy, and tokens/sec for each. Build a small table.

Free/cheap path: AWQ/GPTQ need a GPU (Colab/Modal free tier fine). GGUF + Ollama run **locally on your laptop** with no GPU - this is your zero-cost serving demo.

### Definition of Done
- [ ] `dpo.jsonl` has valid `(prompt, chosen, rejected)` triples; chosen != rejected asserted.
- [ ] DPO run completes; SFT+DPO beats SFT-only on the pairwise judge with **position-swap de-biasing** (report net win-rate and n).
- [ ] Three serving artifacts exist: bf16-merged, AWQ (or GPTQ) 4-bit, GGUF q4_k_m.
- [ ] GGUF model loads and answers in **Ollama** locally.
- [ ] Post-quant scorecard table: JSON-valid %, task accuracy, tokens/sec for bf16 vs AWQ vs GGUF, with the accuracy delta from quantization stated.

### Pitfalls
- DPO LR too high or beta too low -> the model drifts, breaks formatting, and "wins" on style while losing task accuracy. Always re-check task metrics after DPO.
- `chosen` and `rejected` too similar -> no learning signal. Make the contrast meaningful.
- Quantizing the 4-bit training base instead of a merged fp16 model -> double quantization damage.
- Skipping re-eval after quantization -> shipping a silently-worse model. Always re-measure.
- Judge position bias -> without swapping order, "win-rates" are noise.

### Self-check
1. When does DPO help but SFT alone doesn't?
2. What does `beta` control in DPO, and what breaks if it's too small?
3. You quantized to 4-bit AWQ and JSON-valid rate dropped 4 points - what are your options?
4. Why GGUF for a laptop demo but AWQ/GPTQ for a vLLM server?

---

## Week 4 - Ops, evaluation & the retention scorecard

### Objectives
By the end of the week you can:
- Plan GPU/VRAM and pick a training venue (Colab/Modal/RunPod/provider FT API) with a cost estimate.
- Evaluate a fine-tune the right way: **task metrics + LLM-judge win-rate + a capability-RETENTION regression suite** to catch catastrophic forgetting.
- Deliberately over-train to *show* forgetting, then recover it with lower LR + replay data.
- Fine-tune an **embedding model** with contrastive learning + hard-negative mining and measure the RAG recall win.
- Produce a one-page **model card** with task gain and retention evidence.

### Theory (~4 hrs)
> **📖 Deep lectures for this week** (read first): [11 · GPU planning & training venues](lectures/11-gpu-planning-and-training-venues.md) · [12 · Three-legged fine-tune evaluation](lectures/12-three-legged-finetune-evaluation.md) · [13 · Catastrophic forgetting & mitigation](lectures/13-catastrophic-forgetting-and-mitigation.md) · [14 · Embedding-model fine-tuning](lectures/14-embedding-model-finetuning.md). The bullets below are the recap.

- **GPU/VRAM planning & venues.** Estimate before renting (you built the estimator in week 2). Venues: free Colab (T4/L4, session limits), Modal/RunPod/Lambda (per-second GPU, cheap A10/L4/A100), or **provider FT APIs** (OpenAI/Vertex/Bedrock/Together/Predibase) when you want zero infra and accept the lock-in/cost. Read the Modal docs "fine-tune" example and the Together/Predibase FT docs to know the managed option.
- **Evaluating a fine-tune - the core discipline.** *Loss down != better.* Three legs: (1) **task metrics** (accuracy, JSON-valid rate, field F1 on your held-out set); (2) **LLM-judge win-rate** vs the baseline (pairwise, de-biased); (3) **capability-retention regression** - run a fixed suite of *general* capabilities before and after to catch **catastrophic forgetting** (the model got great at your task and worse at everything else). Use `lm-evaluation-harness` (EleutherAI) for a small slice: e.g. a subset of **MMLU**, **IFEval** (instruction-following), and a **safety** check. A fine-tune that gains 15% on your task but loses 20% on IFEval is often a *reject*.
- **Catastrophic forgetting & mitigation.** Causes: too high LR, too many epochs, narrow data. Fixes: PEFT (adapters localize change), low LR, fewer epochs, and **replay** - mix ~10-20% general instruction data back into training. Re-run safety evals after tuning.
- **Embedding-model fine-tuning (often the bigger RAG win).** Read `sentence-transformers` training docs. Contrastive objective (MultipleNegativesRankingLoss): pull query and positive doc together, push negatives apart. **Hard-negative mining** = use your retriever to find *plausible-but-wrong* docs as negatives - this is where the gains are. For many RAG systems, a domain-tuned embedder beats fine-tuning the generator.

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [Retention Harness, Forgetting Demo, Embedder Tune, and Milestone Deliverables](labs/week-4-eval-harness-forgetting-embedder-and-milestone.md). The steps below are the summary.

Build the eval harness, prove/fix forgetting, and tune an embedder.

1. **`eval/retention.py`.** Wire `lm-eval` (EleutherAI `lm-evaluation-harness`) to score your base-instruct and fine-tuned models on a *small* slice: MMLU (a few subjects), IFEval, and one safety/toxicity probe. Emit a JSON scorecard.
2. **`eval/task_eval.py`.** Your held-out 100: JSON-valid rate, category accuracy, priority F1. Plus the pairwise LLM-judge win-rate vs base (reuse week-3 judge).
3. **Show forgetting.** Deliberately over-train (5+ epochs, LR 5e-4) a variant. Run both eval scripts - task metric up, retention down. Record it.
4. **Recover it.** Retrain with LR 1e-4, 2 epochs, and **20% replay** (mix a general instruction dataset like a slice of a public SFT set into your data). Show retention recovers while task metric stays strong. This contrast is the headline of your model card.
5. **Embedding fine-tune.** With `sentence-transformers`, take a base embedder (e.g. `BAAI/bge-small-en-v1.5`), build `(query, positive)` pairs from your domain, **mine hard negatives** with the base retriever, train with MultipleNegativesRankingLoss for 1-3 epochs. Measure **recall@5** on a held-out query set before/after. Runs on Colab or even CPU for `bge-small`.
6. **Model card.** `models/MODEL_CARD.md`: base model, data summary, hyperparameters, task gain, retention scorecard (before/after/forgetting/recovered), quantized-serving metrics, and known limitations.

### Definition of Done
- [ ] `retention.py` produces a before/after scorecard on >=3 capabilities (MMLU-slice, IFEval, safety).
- [ ] `task_eval.py` reports JSON-valid %, accuracy, F1, and a de-biased pairwise win-rate vs base.
- [ ] The over-trained variant **demonstrably** loses >=~10% on a retention metric; the recovered variant restores it (numbers shown for all three: base / over-trained / recovered).
- [ ] Fine-tuned embedder improves **recall@5 by a measurable margin** over the base embedder on the held-out query set (state the delta).
- [ ] `MODEL_CARD.md` complete with task gain **and** retention evidence.

### Pitfalls
- Judging a fine-tune on train/eval loss alone -> you'll ship a forgetful model that "trained well."
- Running the full MMLU (57 subjects) -> hours of compute; use a subject subset for the loop, full only for a final report.
- No replay data on a narrow dataset -> catastrophic forgetting you won't notice until users do.
- Mining negatives that are actually positives (false negatives) -> you teach the embedder the wrong thing; sanity-check mined negatives.
- Fine-tuning the generator when the real RAG bottleneck was retrieval -> tune the embedder first; it's often the bigger, cheaper win.

### Self-check
1. Why is "eval loss went down" insufficient evidence a fine-tune is good?
2. What are the three legs of a fine-tune eval and what does each catch?
3. How does 20% replay data fight catastrophic forgetting?
4. What is hard-negative mining and why does it drive embedding-tuning gains?
5. Given a weak RAG system, why might you fine-tune the embedder before the generator?

---

## Phase milestone project

**Deliverable A - "The 3 ways" decision memo (proven, not asserted).**
Solve one real task (support-ticket -> structured JSON, or your own) **three ways** and measure all three on the *same* held-out set:
1. Few-shot **prompt** on a frozen instruct model.
2. **RAG** over labeled examples (retrieve similar past tickets as context).
3. A **LoRA/QLoRA fine-tune**.

For each: accuracy (+ JSON-valid rate, F1), **p50/p95 latency**, and **cost per 1k requests**. Write a one-page memo recommending which ships and why - with numbers, not vibes. Include a **with/without response-only-loss-masking** comparison for the fine-tune.

**Deliverable B - QLoRA fine-tune + retention scorecard.**
QLoRA a 7-8B model, produce **fp16-merged / AWQ / GGUF** variants benchmarked on VRAM, tokens/sec, and task accuracy. Ship a one-page **model card** showing task gain **and** MMLU/IFEval/safety **retention**. Deliberately over-train to exhibit forgetting, then recover it with **20% replay + lower LR** and show the recovery in the numbers.

### Acceptance criteria
- [ ] Memo table with accuracy, p50/p95 latency, cost/1k for prompt vs RAG vs LoRA - filled from real runs.
- [ ] Clear ship recommendation justified by the numbers (including the case where prompt/RAG wins - that's a valid, common outcome).
- [ ] Masking A/B result included.
- [ ] QLoRA model trained on a single free/cheap GPU (<=15GB); adapter + merged + AWQ + GGUF artifacts all present.
- [ ] GGUF runs locally in Ollama.
- [ ] Retention scorecard shows base / over-trained (forgetting visible) / recovered, across >=3 capabilities.
- [ ] Model card complete and honest about limitations.

### Suggested repo layout
```
llm-finetune-lab/
  pyproject.toml
  data/{raw,prepared}/            # train.jsonl, eval.jsonl, dpo.jsonl
  scripts/
    prepare_dataset.py
    inspect_template.py
    vram_estimate.py
    train_sft.py
    train_dpo.py
    merge_and_quantize.py
  adapters/                       # LoRA/DPO adapters (hot-swappable)
  models/
    ticket-router-v1-merged/      # bf16
    ticket-router-v1-awq/
    ticket-router-v1-gguf/
    MODEL_CARD.md
  eval/
    task_eval.py
    retention.py                  # lm-eval slice: MMLU/IFEval/safety
    judge.py                      # pairwise, position-swapped
    results/                      # scorecards, tables
  embeddings/
    train_embedder.py             # sentence-transformers + hard negatives
    recall_eval.py
  memo/DECISION_MEMO.md
  README.md
```

## You are ready to move on when...
- [ ] You can, without notes, decide prompt vs RAG vs fine-tune for a described task and defend it with the numbers you'd need.
- [ ] You've trained SFT + DPO with correct chat templates, EOS, and response-only loss masking - and can explain each.
- [ ] You've fit a 7-8B QLoRA fine-tune on a single free/cheap GPU and predicted its VRAM within ~20%.
- [ ] You can hot-swap and merge LoRA adapters, and quantize (AWQ/GPTQ/GGUF) *after* merging, then re-evaluate.
- [ ] You never call a fine-tune "good" on loss alone - you produce task metrics + judge win-rate + a **retention** scorecard, and you know how to fix catastrophic forgetting.
- [ ] You've fine-tuned an embedding model with hard negatives and measured the RAG recall win.
- [ ] You have a shippable repo: the decision memo, the QLoRA model + serving variants, and an honest model card.

Next: `09-architecture-system-design.md` - wiring these models into routed, cached, streaming, observable production systems.
