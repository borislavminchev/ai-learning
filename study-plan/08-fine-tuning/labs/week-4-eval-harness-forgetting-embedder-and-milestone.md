# Week 4 Lab: Retention Harness, Forgetting Demo, Embedder Tune, and Milestone Deliverables

> This is the week you stop trusting the loss curve. You build the **three-legged evaluation** for a fine-tune — (1) task metrics on a locked holdout, (2) a position-swap-de-biased **LLM-judge win-rate** vs the base, and (3) a **capability-retention** regression via EleutherAI's `lm-evaluation-harness` — then *use* it to catch a real failure: you deliberately **over-train to make the model forget** general instruction-following, watch retention crater, and **recover it with lower LR + 20% replay**. Then you pivot to the RAG side and fine-tune an **embedding model** (`bge-small`) with **hard-negative mining**, measuring the recall@5 win. The week closes by assembling the phase's two milestone deliverables: the **"3 ways" decision memo** (prompt vs RAG vs LoRA, measured on the same holdout) and the **QLoRA fine-tune + retention scorecard** with a one-page model card. Everything runs on a **free Colab T4**; the embedder even runs on CPU.
>
> Read the matching lectures first — this guide assumes them:
> - [../lectures/11-gpu-planning-and-training-venues.md](../lectures/11-gpu-planning-and-training-venues.md) — VRAM→GPU→hours→dollars; free Colab vs per-second rental vs managed FT APIs; checkpointing.
> - [../lectures/12-three-legged-finetune-evaluation.md](../lectures/12-three-legged-finetune-evaluation.md) — task metrics (JSON-valid as a gate, macro-F1), position-swap judging, retention regression, and the **reject rule**.
> - [../lectures/13-catastrophic-forgetting-and-mitigation.md](../lectures/13-catastrophic-forgetting-and-mitigation.md) — the three levers (LR, epochs, data breadth), why PEFT bounds damage, and why 20% replay works.
> - [../lectures/14-embedding-model-finetuning.md](../lectures/14-embedding-model-finetuning.md) — contrastive `MultipleNegativesRankingLoss`, in-batch negatives, hard-negative mining, and the false-negative trap.
>
> It builds directly on the `llm-finetune-lab/` repo: the **week-1** chat dataset + held-out `eval.jsonl`, the **week-2** QLoRA adapter + merged model, the **week-3** DPO adapter + judge + serving artifacts, and (for the memo) a **Phase-4 RAG baseline**. Keep working in the same repo.

**Est. time:** ~9 hrs · **You will need:** the `llm-finetune-lab/` repo (weeks 1-3), a Hugging Face account (Qwen2.5 is non-gated), a GPU for the LLM legs, and an LLM-judge (a small local model *or* an API key from Phase 7). **Free/local path:** [Google Colab free T4 (15 GB)](https://colab.research.google.com) runs the retention harness, forgetting demo, and embedder tune; the `bge-small` embedder fine-tune also runs on a **CPU laptop** in ~15-30 min. Paid-but-cheap alternative: an L4/A10 (~$0.50-1.00/hr) on Modal/RunPod for faster `lm-eval` sweeps. No paid API is *required* — you can use a small local judge (e.g. the base-instruct model itself, or a merged model from week 3) if you have no key.

---

## Before you start (setup)

**What / Why.** This week is mostly *evaluation code* (pure Python + a benchmarking harness) plus two short training runs (the over-trained/recovered variants and the embedder). Keep the scoring scripts in your repo so they're version-controlled and reusable; run the GPU-bound `lm-eval` and training cells in Colab. The `lm-evaluation-harness` install is the one new heavy dependency — pin it, because task names and flags drift across versions (confirm against `lm-eval --tasks list`, never a memorized flag).

**Do it (local, Windows/Git-Bash — the pure-Python eval + data pieces):**
```bash
cd llm-finetune-lab                 # the repo from weeks 1-3
mkdir -p eval/results embeddings memo models
uv add "scikit-learn" "datasets" "pydantic"   # F1, data handling — run locally for task_eval + recall_eval logic
uv add "sentence-transformers>=3.0"           # embedder fine-tune (CPU-capable for bge-small)
```

**Do it (Colab — the GPU venue for retention + training).** New Colab notebook, **Runtime → Change runtime type → T4 GPU**, then:
```python
# Colab cell 1 — eval + training stack
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "trl<0.9.0" peft accelerate bitsandbytes
!pip install -q "lm-eval>=0.4.3"              # EleutherAI lm-evaluation-harness (the `lm_eval` package)
!pip install -q "sentence-transformers>=3.0" "scikit-learn"
import torch; print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0))
```

**Verify:**
```bash
# Local — confirm the pure-Python eval deps import:
uv run python -c "import sklearn, datasets, sentence_transformers; print('local eval deps ok')"
```
```python
# Colab — confirm lm-eval CLI is on PATH and can see tasks:
!lm-eval --tasks list 2>/dev/null | grep -E "ifeval|mmlu_high_school_mathematics|toxigen" | head
```

**Judge configuration (Leg 2).** You reuse the **week-3 pairwise judge** (`eval/judge.py`). If you have an API key (Anthropic/OpenAI), set it as an env var and pin the judge model + `temperature=0`. If you have *no* key, use a **local judge** — a strong-ish instruct model you can run in Colab (e.g. `Qwen/Qwen2.5-7B-Instruct` un-tuned, or a merged model). The mechanism (position-swap) is identical either way; only the `judge_completion()` backend changes.

**Windows / macOS / Linux note.** All `uv run` commands are identical across platforms. `lm-eval` and QLoRA training need CUDA and live in Colab. The **embedder fine-tune runs anywhere** — on Windows/CPU it's slower but works; `sentence-transformers` has no `bitsandbytes` dependency. Use forward slashes in paths inside Python; they work on all three OSes.

**Get your artifacts into Colab.** `git clone` your repo (or mount Drive) so Colab sees: `data/prepared/{train,eval}.jsonl`, `adapters/ticket-router-v1/` (week 2), the merged model or its HF-hub id, and `eval/judge.py` (week 3).

---

## Step-by-step

### Step 1 — Plan the venue and cost (the 10-minute discipline)

**What.** Before spending GPU time, write a 4-line venue plan into `README.md`: predicted VRAM (from week-2's `vram_estimate.py`), the GPU class that fits with headroom, a *measured* throughput estimate, and the resulting hours/dollars for the runs you're about to do this week (two short QLoRA variants + several `lm-eval` sweeps).

**Why.** Lecture 11's pipeline is `VRAM → GPU class → tokens/sec → hours → dollars → venue`. The whole point is to not learn your bill *after* the fact. The classic horror story is a rented A10 left running over a weekend for $70 of zero training — the defense is a written plan + "shut the box down the instant the run ends." For this week the honest answer is almost always "free Colab T4," but you state *why* (small QLoRA fits ~11-14 GB; retention sweeps are minutes on a subset), and you note the one T4 gotcha: **no bf16** (fp16 only → occasional NaNs; an L4/A10 fixes it if you hit them).

**Do it.** Add to `README.md`:
```markdown
## Week-4 venue plan
- Runs: over-trained QLoRA (5 ep), recovered QLoRA (2 ep + 20% replay), ~6 lm-eval sweeps, 1 embedder tune.
- Predicted peak VRAM (QLoRA 7-8B, r=16): ~9.8 GB  (from scripts/vram_estimate.py)
- GPU class w/ headroom (1.3-1.5x): T4 16GB (free Colab) fits; L4 24GB if fp16 NaNs.
- Throughput: measure it/s over 20 steps in Colab, then hours = tokens / (tok_s * 3600). (fill in)
- Cost: Colab free = $0. (If rented: hours * $/hr + ~15 min overhead; kill the box after.)
- Retention loop cost control: MMLU *subset* (4-6 subjects) + IFEval + 1 safety task = minutes, not the 57-subject hours.
```

**Expected result.** A written plan that says "free Colab T4, $0, subset MMLU in the loop." You should be able to defend the GPU choice from the VRAM number, not vibes.

**Verify.** The plan names a specific GPU, a specific dollar figure (even if $0), and explicitly commits to a *subset* MMLU in the iteration loop. That last line is the one that saves you a lost work-week (see Troubleshoot).

**Troubleshoot.**
- *Tempted to wire full MMLU into the loop?* Don't. Full MMLU is 57 subjects / ~14k questions / hours per run; you iterate a dozen times this week. Subset for iteration, full only for the final card.
- *fp16 NaN loss on T4:* the T4 has no bf16. Lower LR slightly, or move to an L4/A10 (which support bf16's wide exponent range). Note it in the plan.

---

### Step 2 — Build `eval/task_eval.py` (Leg 1: task metrics on the holdout)

**What.** Write `eval/task_eval.py` that runs the locked **held-out 100** through a model, parses each output, and reports the layered metrics: **JSON-valid rate (Layer 0 gate)**, category accuracy, **priority macro-F1**, `needs_human` F1, and exact-match. It must count unparseable output as *wrong on every field* — no "drop the bad ones." Emit a JSON scorecard to `eval/results/`.

**Why.** Lecture 12: loss down ≠ better. JSON-valid rate is Layer 0 because if the output doesn't parse there's no category to score — an unparseable output is a full task failure. **Macro-F1** (not micro/weighted) is mandatory because a model that never predicts the rare `P0` outage class still scores ~99% accuracy while missing every outage; macro-F1 exposes that as `P0 F1 = 0`. This script is your acceptance instrument for the milestone's task numbers.

**Do it.** In `eval/task_eval.py`:
```python
"""Leg 1 — task metrics on the locked holdout. Unparseable output = wrong on every field."""
import json, re
from sklearn.metrics import f1_score

def parse_output(text: str) -> dict | None:
    """Strict-ish JSON parse. Strips a leading ```json fence if the model added one
    (a schema-drift bug you should fix in prep, but don't let it silently zero your score)."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None

def gold(row: dict) -> dict:
    return json.loads(next(m["content"] for m in row["messages"] if m["role"] == "assistant"))

def build_prompt(row, tokenizer):
    msgs = [m for m in row["messages"] if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def task_scorecard(gen_fn, rows, tokenizer) -> dict:
    """gen_fn(prompt)->str. rows: list of chat-format dicts. Returns a JSON-able scorecard."""
    n = len(rows)
    valid = cat_ok = exact = 0
    prio_gold, prio_pred, nh_gold, nh_pred = [], [], [], []
    for r in rows:
        g = gold(r)
        out = gen_fn(build_prompt(r, tokenizer))
        p = parse_output(out)
        if p is None:                      # Layer 0 gate: unparseable => wrong everywhere
            # still record gold vs a sentinel so F1 denominators stay honest
            prio_gold.append(g["priority"]); prio_pred.append("__invalid__")
            nh_gold.append(bool(g["needs_human"])); nh_pred.append(None)
            continue
        valid += 1
        cat_ok += int(p.get("category") == g["category"])
        prio_gold.append(g["priority"]); prio_pred.append(p.get("priority", "__invalid__"))
        nh_gold.append(bool(g["needs_human"])); nh_pred.append(bool(p.get("needs_human")))
        exact += int(p.get("category")==g["category"] and p.get("priority")==g["priority"]
                     and bool(p.get("needs_human"))==bool(g["needs_human"]))
    # macro-F1 over priority classes (rare P0 counts as much as common P3)
    prio_f1 = f1_score(prio_gold, prio_pred, average="macro", zero_division=0)
    # binary F1 for needs_human; map None (unparseable) to the wrong label
    nh_pred_bin = [(False if x is None else x) for x in nh_pred]
    nh_f1 = f1_score([int(x) for x in nh_gold], [int(x) for x in nh_pred_bin], zero_division=0)
    return {
        "n": n,
        "json_valid_pct": round(100*valid/n, 1),
        "category_acc_pct": round(100*cat_ok/n, 1),   # denominator = ALL n, not just valid
        "priority_macro_f1": round(prio_f1, 3),
        "needs_human_f1": round(nh_f1, 3),
        "exact_match_pct": round(100*exact/n, 1),
    }

if __name__ == "__main__":
    # Wired up in Colab against your models; this block is a local smoke test on a dummy gen_fn.
    print("import ok — call task_scorecard(gen_fn, rows, tokenizer) from your Colab eval cell")
```

**Do it (run a real scorecard in Colab, base vs fine-tuned):**
```python
from eval.task_eval import task_scorecard
import json
heldout = [json.loads(l) for l in open("data/prepared/eval.jsonl")][:100]

base_card = task_scorecard(lambda p: gen_base(p), heldout, tokenizer)      # base-instruct
ft_card   = task_scorecard(lambda p: gen_merged(p), heldout, tokenizer)    # week-2/3 fine-tune
json.dump({"base": base_card, "fine_tuned": ft_card},
          open("eval/results/task_eval.json", "w"), indent=2)
print("base :", base_card); print("ft   :", ft_card)
```

**Expected result.** A scorecard like `{"json_valid_pct": 96.0, "category_acc_pct": 90.0, "priority_macro_f1": 0.71, ...}`. The fine-tune should beat base on JSON-valid rate (often base 60-90% → FT 95-100%). Watch the **per-class** story: even at 90% accuracy, `P0` F1 may be low — that's the real finding, not the headline.

**Verify.** `eval/results/task_eval.json` exists with `json_valid_pct`, `category_acc_pct`, `priority_macro_f1`, `needs_human_f1`, `exact_match_pct` for both models. Confirm the category-accuracy denominator is `n` (all rows), not the parseable subset — flip one output to garbage in a test and accuracy must drop.

**Troubleshoot.**
- *Accuracy computed over parseable-only:* the flattering lie from lecture 12. A 70%-valid model shows "96% accuracy on the 70%" while failing 30% of traffic. Denominator = all `n`.
- *Priority F1 suspiciously high with a broken rare class:* you used `average="micro"` or `"weighted"`. Use `"macro"`.
- *Code-fence outputs tanking JSON-valid:* the `parse_output` fence strip catches them, but log how many needed stripping — it's a prep bug to fix, not something to hide.

---

### Step 3 — Build `eval/retention.py` (Leg 3: capability-retention via `lm-eval`)

**What.** Wire EleutherAI's `lm-evaluation-harness` (`lm_eval`) to score a model on a **fixed, small slice** across three capability axes — MMLU **subset** (4-6 subjects), **IFEval** (instruction-following), and one **safety/toxicity** probe — and emit a JSON scorecard. You'll run it on the base-instruct model and on each fine-tuned variant and diff.

**Why.** Lecture 12/13: catastrophic forgetting lives *outside* your task. Retention is a **regression test** — same tasks, same few-shot, same harness version, before and after — or it measures nothing. IFEval is the canary that most directly catches the classic injury ("tuned to always emit ticket-JSON → can't follow 'reply in French, no punctuation'"); the safety probe catches eroded refusal, which is a release blocker on its own. The subset is a *smoke detector* (minutes), not a certification; full MMLU is for the final card only.

**Do it (define the fixed suite — commit these exact task names):**
```python
# eval/retention.py
"""Leg 3 — capability-retention regression via lm-evaluation-harness.
FIXED SUITE: change these names once and never again, or the regression test is meaningless."""
import json, subprocess, sys, pathlib

# Confirm each name exists in YOUR installed harness: `lm-eval --tasks list | grep <name>`.
RETENTION_TASKS = [
    "mmlu_high_school_mathematics",   # knowledge/reasoning
    "mmlu_professional_law",
    "mmlu_college_biology",
    "mmlu_philosophy",
    "ifeval",                          # instruction-following (the key forgetting canary)
    "toxigen",                         # safety/toxicity probe (or another safety task in your harness)
]
FEWSHOT = 0   # keep IDENTICAL across every run; IFEval is 0-shot by design

def run_retention(model_args: str, out_json: str, tasks=RETENTION_TASKS, limit=None):
    """model_args: e.g. 'pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16,load_in_4bit=True'
                   or 'pretrained=models/ticket-router-overtrained,dtype=bfloat16'.
    limit: cap examples per task for a fast in-loop smoke run (e.g. 100). None = full slice."""
    cmd = [
        "lm-eval", "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(FEWSHOT),
        "--batch_size", "auto",
        "--output_path", out_json,
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print("RUN:", " ".join(cmd)); subprocess.run(cmd, check=True)

def load_scores(results_dir: str) -> dict:
    """lm-eval writes results/<model>/results_*.json. Pull the primary metric per task."""
    p = sorted(pathlib.Path(results_dir).rglob("results_*.json"))[-1]
    res = json.loads(p.read_text())["results"]
    out = {}
    for task, metrics in res.items():
        # pick acc / acc_norm / prompt_level_strict_acc depending on task; keep it simple:
        for key in ("acc,none", "acc_norm,none", "prompt_level_strict_acc,none", "acc"):
            if key in metrics:
                out[task] = round(metrics[key], 4); break
    return out
```

**Do it (run base + a fine-tuned variant, in-loop with `--limit`):**
```python
# Colab — FAST in-loop smoke run (limit=100 per task); drop --limit for the final card run.
from eval.retention import run_retention, load_scores, RETENTION_TASKS

run_retention("pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=float16,load_in_4bit=True",
              out_json="eval/results/retention_base", limit=100)
run_retention("pretrained=models/ticket-router-v1-merged,dtype=bfloat16",
              out_json="eval/results/retention_ft", limit=100)

base = load_scores("eval/results/retention_base")
ft   = load_scores("eval/results/retention_ft")
scorecard = {t: {"base": base.get(t), "ft": ft.get(t),
                 "delta_pp": round(100*((ft.get(t) or 0)-(base.get(t) or 0)), 1)}
             for t in {**base, **ft}}
import json; json.dump(scorecard, open("eval/results/retention_scorecard.json","w"), indent=2)
print(json.dumps(scorecard, indent=2))
```

**Expected result.** A per-task before/after with a `delta_pp`. For a *sane* fine-tune, deltas are near-flat (within a couple points = noise). Example shape:
```
mmlu_* (avg)   base 0.61  ft 0.59   -2.4pp  (OK, noise)
ifeval         base 0.74  ft 0.71   -2.3pp  (OK)
toxigen        base 0.97  ft 0.96   -0.7pp  (OK)
```

**Verify.** `eval/results/retention_scorecard.json` covers **≥3 capabilities** (MMLU-slice + IFEval + safety) with base, ft, and delta. Re-running base twice with the same `--limit` and seed should give near-identical numbers — that's your proof the suite is a stable regression test.

**Troubleshoot.**
- *`lm-eval: task not found`:* the name isn't in your harness version. Run `lm-eval --tasks list | grep -i mmlu` (and `ifeval`, `toxigen`/`truthfulqa`) and use the exact strings printed. Do **not** trust a memorized flag.
- *IFEval errors on missing deps:* it needs `langdetect`/`nltk`/`immutabledict` in some versions — `pip install langdetect nltk immutabledict` and retry.
- *Different subset before vs after:* fatal — you measured nothing. The `RETENTION_TASKS` list and `FEWSHOT` must be byte-identical every run.
- *Full MMLU by accident (`--tasks mmlu`):* that's all 57 subjects / hours. Use the explicit `mmlu_*` subject names.

---

### Step 4 — Wire Leg 2: the position-swapped judge win-rate

**What.** Reuse the **week-3** `eval/judge.py` pairwise judge on ~50-150 held-out prompts: generate from **fine-tuned (A)** and **base-instruct (B)**, judge each pair **in both orders**, and count a WIN only if the fine-tune wins *both* orders (LOSS if it loses both; disagreement → TIE). Report `wins/ties/losses`, `n`, and **net win-rate**.

**Why.** Lecture 12: LLM judges have systematic **position bias**. An un-swapped win-rate is a mix of quality and seating order — literal noise. Swapping every pair and discarding order-disagreements as ties removes the artifact. And you must report `n`: `+18%` at n=20 is inside the noise band (≈±10pp), while at n≈200 it's real (noise ≈±2.5pp shrinks like 1/√n).

**Do it.** In `eval/judge.py` add (or confirm) a swap-aware scorer:
```python
def judge_pairwise(prompt, ans_a, ans_b, judge_completion) -> str:
    """Return 'A', 'B', or 'TIE'. judge_completion(msgs)->str is your week-3 backend
    (API or local), pinned model + temperature=0."""
    rubric = (f"Question:\n{prompt}\n\nResponse A:\n{ans_a}\n\nResponse B:\n{ans_b}\n\n"
              "Which response is better for this task? Reply with exactly one token: A, B, or TIE.")
    verdict = judge_completion([{"role": "user", "content": rubric}]).strip().upper()
    return verdict if verdict in ("A", "B", "TIE") else "TIE"

def win_rate_swapped(prompts, gen_ft, gen_base, judge_completion) -> dict:
    wins = ties = losses = 0
    for p in prompts:
        ft, base = gen_ft(p), gen_base(p)
        v1 = judge_pairwise(p, ft,  base, judge_completion)   # round 1: FT is A
        v2 = judge_pairwise(p, base, ft,  judge_completion)   # round 2: order flipped, FT is B
        ft_wins_1 = (v1 == "A")
        ft_wins_2 = (v2 == "B")
        if ft_wins_1 and ft_wins_2:      wins += 1     # consistent across both orders
        elif (v1 == "B") and (v2 == "A"): losses += 1  # base wins both
        else:                             ties += 1     # disagreement = position bias -> tie
    n = len(prompts)
    return {"n": n, "wins": wins, "ties": ties, "losses": losses,
            "net_win_rate_pct": round(100*(wins-losses)/n, 1)}
```

**Do it (run it):**
```python
from eval.judge import win_rate_swapped
import json
prompts = [build_prompt(r, tokenizer) for r in heldout[:100]]   # aim for n in the low hundreds
wr = win_rate_swapped(prompts,
        gen_ft=lambda p: gen_merged(p),
        gen_base=lambda p: gen_base(p),
        judge_completion=my_judge)          # week-3 judge backend, temperature=0, pinned model
json.dump(wr, open("eval/results/win_rate.json","w"), indent=2)
print(wr)
```

**Expected result.** Something like `{"n": 100, "wins": 62, "ties": 23, "losses": 15, "net_win_rate_pct": 47.0}`. A healthy fine-tune shows a clearly positive net win-rate; a chunk of the ties are order-disagreements (the position bias, made visible and discarded).

**Verify.** `eval/results/win_rate.json` reports `wins/ties/losses`, `n`, and net win-rate — never a bare percentage. Sanity-check that a model judged against *itself* yields ~0% net (all ties/near-random) — that confirms your swap logic isn't leaking a position win.

**Troubleshoot.**
- *Win-rate wobbles between runs:* judge non-determinism. Set `temperature=0` and **pin the judge model version**.
- *Suspiciously high win-rate:* you didn't actually swap, or you counted single-order wins. Confirm both `v1` and `v2` must agree for a WIN.
- *No API key:* use a local instruct model as the judge (Qwen2.5-7B-Instruct un-tuned). Slower, but the mechanism is identical. Budget ~2 calls/pair (n=100 → 200 calls).

---

### Step 5 — Show forgetting: deliberately over-train

**What.** Fine-tune a **forgetting** variant on purpose: **5+ epochs, LR 5e-4**, 0% replay, on your week-1 dataset. Save to `models/ticket-router-overtrained/`. Run `task_eval.py` and `retention.py` on it. Task metric should go **up**; retention (especially IFEval and safety) should go **down**.

**Why.** Lecture 13: forgetting is `distance traveled ∝ LR × steps`. LR 5e-4 × 5 epochs travels ~12× farther through weight space than the recovered recipe — an order-of-magnitude push out of the "general basin." You do this deliberately so you *recognize the signature on a dashboard*: the **scissors** where task accuracy rises while retention falls. This contrast is the headline of your model card.

**Do it (Colab — same Unsloth setup as week 2, aggressive knobs):**
```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                        max_seq_length=1024, dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            use_gradient_checkpointing="unsloth", random_state=3407)

ds = load_dataset("json", data_files={"train":"data/prepared/train.jsonl"})["train"]
ds = ds.map(lambda r: {"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)},
            remove_columns=ds.column_names)

SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds,
    dataset_text_field="text", max_seq_length=1024,
    args=SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=5,               # AGGRESSIVE — this causes forgetting
        learning_rate=5e-4,               # AGGRESSIVE — 5x the sane rate
        lr_scheduler_type="cosine", optim="paged_adamw_8bit",
        gradient_checkpointing=True, fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(), logging_steps=5, output_dir="outputs_overtrain",
        seed=3407)).train()

model.save_pretrained_merged("models/ticket-router-overtrained", tokenizer, save_method="merged_16bit")
```

**Do it (eval it with both scripts):**
```python
# Task metrics (should be UP vs the good week-2 fine-tune) ...
over_task = task_scorecard(lambda p: gen_from("models/ticket-router-overtrained", p), heldout, tokenizer)
# ... but retention should be DOWN:
run_retention("pretrained=models/ticket-router-overtrained,dtype=bfloat16",
              out_json="eval/results/retention_overtrained", limit=100)
over_ret = load_scores("eval/results/retention_overtrained")
print("overtrained task:", over_task)
print("overtrained retention:", over_ret)
```

**Expected result.** The over-trained model *wins* on task accuracy (e.g. +2-3pp over the good fine-tune — the trap) while IFEval **drops ~15-20pp** and the safety probe drops measurably. That scissors is the whole point; if retention *doesn't* drop, push harder (more epochs / higher LR) until it does — you need a visible ≥~10pp drop for the DoD.

**Verify.** Compare `eval/results/retention_overtrained` vs `retention_base`: at least one core axis (ideally IFEval) is down **≥~10pp**. Record all numbers.

**Troubleshoot.**
- *Retention barely moved:* your dataset may be too small/close to general text, or LoRA's frozen base is protecting you (a feature!). Crank epochs to 8-10 and/or LR to 1e-3, or widen `target_modules` — you're intentionally breaking it here.
- *fp16 NaN on T4 at LR 5e-4:* expected risk at high LR without bf16. Nudge LR to 3e-4 (still forgetting-inducing) or use an L4.
- *Task metric didn't rise either:* the run diverged (NaN) — check the loss log; if it exploded, lower LR just enough to train but still over-fit.

---

### Step 6 — Recover it: lower LR, fewer epochs, 20% replay

**What.** Retrain the *recovered* variant: **LR 1e-4, 2 epochs**, with **~20% general-instruction replay** mixed into your task data. Save to `models/ticket-router-recovered/`. Re-run both eval scripts: retention should return to ~baseline while task metrics stay strong.

**Why.** Lecture 13's recovery recipe stacks three levers: LR 5e-4→1e-4 (5× smaller steps), 5→2 epochs (fewer steps + less memorization), and **replay** (≈1 in 5 gradients points back toward general behavior — a leash keeping `θ` in the general basin). 10-20% is the sweet band: <5% is too loose, >30% dilutes the task signal. This is *rehearsal* done with a data mixer, not a math trick.

**Do it (build the replay mix — ~20% general SFT data):**
```python
from datasets import load_dataset, concatenate_datasets, Dataset

task = load_dataset("json", data_files={"train":"data/prepared/train.jsonl"})["train"]
n_task = len(task)
n_replay = int(0.20 * n_task / 0.80)      # so replay is ~20% of the FINAL mix

# A public general-instruction set (choose one you can access; Dolly is small & permissive):
gen = load_dataset("databricks/databricks-dolly-15k", split=f"train[:{n_replay}]")

def dolly_to_chat(ex):   # normalize into the same {"messages":[...]} shape as your task data
    user = ex["instruction"] + (("\n\n"+ex["context"]) if ex.get("context") else "")
    return {"messages":[{"role":"user","content":user},
                        {"role":"assistant","content":ex["response"]}]}
gen = gen.map(dolly_to_chat, remove_columns=gen.column_names)

mixed = concatenate_datasets([task, gen]).shuffle(seed=3407)
print(f"task={n_task}  replay={len(gen)}  mixed={len(mixed)}  replay_frac={len(gen)/len(mixed):.0%}")
mixed = mixed.map(lambda r: {"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)},
                  remove_columns=mixed.column_names)
```

**Do it (train gently):**
```python
model, tokenizer = FastLanguageModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                        max_seq_length=1024, dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            use_gradient_checkpointing="unsloth", random_state=3407)

SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=mixed,
    dataset_text_field="text", max_seq_length=1024,
    args=SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=2,               # GENTLE
        learning_rate=1e-4,               # GENTLE — 5x lower than the over-trained run
        lr_scheduler_type="cosine", optim="paged_adamw_8bit",
        gradient_checkpointing=True, fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(), logging_steps=5, output_dir="outputs_recovered",
        seed=3407)).train()
model.save_pretrained_merged("models/ticket-router-recovered", tokenizer, save_method="merged_16bit")
```

**Do it (produce the three-model contrast):**
```python
run_retention("pretrained=models/ticket-router-recovered,dtype=bfloat16",
              out_json="eval/results/retention_recovered", limit=100)
rec_ret  = load_scores("eval/results/retention_recovered")
rec_task = task_scorecard(lambda p: gen_from("models/ticket-router-recovered", p), heldout, tokenizer)

import json
contrast = {"base": base, "overtrained": over_ret, "recovered": rec_ret,
            "task": {"overtrained": over_task, "recovered": rec_task}}
json.dump(contrast, open("eval/results/forgetting_contrast.json","w"), indent=2)
print(json.dumps(contrast, indent=2))
```

**Expected result.** The signature three-row story (illustrative; your numbers come from your run):
```
Model         Task acc   IFEval   Safety
Base          82%        71       96%
Over-trained  95%        48  ↓    74%  ↓     <- task up, retention CRATERED
Recovered     93%        70       95%        <- ~2 task pts for ~all retention back
```

**Verify.** `eval/results/forgetting_contrast.json` shows all three models on the retention axes; the recovered variant restores the dropped metric to within a couple points of base while task accuracy stays strong (giving up ≤~2-3pp is fine). This is a DoD item and the model-card headline.

**Troubleshoot.**
- *Replay ended up >30% or <10%:* recompute `n_replay`; the `0.20/0.80` formula makes replay 20% of the *final* mix, not 20% of the task set.
- *Recovery didn't restore retention:* replay fraction too low, or LR still too high. Bump replay toward 20%, confirm LR=1e-4, confirm 2 epochs.
- *Task accuracy dropped a lot (>5pp) after recovery:* too much replay diluted the signal, or too few epochs — nudge replay down toward 10-15% or add one epoch. It's a trade; document where you landed.
- *Can't access Dolly/your chosen set:* any small permissive SFT set works (OASST, Alpaca, a Tulu slice). The *breadth* matters, not the specific corpus.

---

### Step 7 — Fine-tune the embedder with hard-negative mining

**What.** Take `BAAI/bge-small-en-v1.5`, build `(query, positive)` pairs from your domain, **mine hard negatives** with your Phase-4 retriever, train with `MultipleNegativesRankingLoss` for 1-3 epochs, **re-embed the corpus**, and measure **recall@5** before/after on a held-out query set. Write `embeddings/train_embedder.py` and `embeddings/recall_eval.py`.

**Why.** Lecture 14: for many RAG systems a domain-tuned embedder is the bigger, cheaper win than tuning the generator — no generator change can add missing context. MNRL manufactures negatives from the batch (so batch size is a *quality* knob), but random in-batch negatives are *easy*; the gains come from **hard negatives** — plausible-but-wrong docs your own retriever ranks highly. The trap: some mined "negatives" are actually relevant (unlabeled positives), and training on them teaches the *opposite* — so you filter by score margin and eyeball a sample. Critically, a fine-tuned embedder changes every vector, so you **must re-embed the whole corpus** before measuring, and never mix embedder versions across index and query.

**Do it (measure the baseline FIRST — wire `recall_eval.py`):**
```python
# embeddings/recall_eval.py
"""recall@k on a hand-labeled, contamination-free held-out query set. Measure BEFORE you tune."""
import numpy as np
from sentence_transformers import SentenceTransformer

def recall_at_k(model, corpus_texts, corpus_ids, queries, gold_doc_ids, k=5) -> float:
    """queries[i]'s correct chunk id is gold_doc_ids[i]. Cosine over normalized embeddings."""
    corp = model.encode(corpus_texts, normalize_embeddings=True, convert_to_numpy=True)
    q    = model.encode(queries,      normalize_embeddings=True, convert_to_numpy=True)
    sims = q @ corp.T                                   # (n_queries, n_docs)
    topk = np.argsort(-sims, axis=1)[:, :k]
    hits = sum(gold_doc_ids[i] in {corpus_ids[j] for j in topk[i]} for i in range(len(queries)))
    return hits / len(queries)

if __name__ == "__main__":
    base = SentenceTransformer("BAAI/bge-small-en-v1.5")
    # load your Phase-4 corpus + held-out (query, gold_chunk_id) set (disjoint from training pairs!)
    # r = recall_at_k(base, corpus_texts, corpus_ids, queries, gold_ids, k=5)
    # print("baseline recall@5:", r)
```

**Do it (mine hard negatives — use the built-in utility with margin filtering):**
```python
# embeddings/train_embedder.py
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.util import mine_hard_negatives

base = SentenceTransformer("BAAI/bge-small-en-v1.5")

# pairs: your domain (query, positive) — e.g. past tickets paired with the doc that resolved them.
# STRICTLY disjoint from the recall_eval held-out queries (contamination inflates recall).
pairs = Dataset.from_dict({"query": queries_train, "positive": positives_train})

# Mine ranks ~10-50, filter out docs scoring too close to the positive (false-negative guard),
# keep a few negatives per query. This weaponizes your CURRENT retriever against itself.
train_ds = mine_hard_negatives(
    pairs, base,
    corpus=corpus_texts,          # your real Phase-4 corpus
    num_negatives=3,
    range_min=10, range_max=50,   # skip the very top ranks where true-positives cluster
    margin=0.1,                   # reject a "negative" within 10% of the positive's score
    sampling_strategy="top",
    output_format="triplet",      # (anchor, positive, negative) rows
)

# EYEBALL 20 before training — read them; if >1-2 look like real answers, tighten margin & re-mine.
for row in train_ds.select(range(20)):
    print("Q:", row["anchor"][:80], "\n  NEG:", row["negative"][:120], "\n")

loss = MultipleNegativesRankingLoss(base)
trainer = SentenceTransformerTrainer(
    model=base, train_dataset=train_ds, loss=loss,
    # batch size is a QUALITY knob (more in-batch negatives); maximize within memory: 64-128 on a T4
    args=dict(per_device_train_batch_size=64, num_train_epochs=2, learning_rate=2e-5,
              warmup_ratio=0.1, fp16=True, output_dir="embeddings/bge-small-ft"),
)
trainer.train()
base.save("embeddings/bge-small-ft")
```
> `SentenceTransformerTrainer`'s `args` is really a `SentenceTransformerTrainingArguments` object in v3+ — construct it explicitly (`from sentence_transformers import SentenceTransformerTrainingArguments`) if the dict shorthand errors on your version. On CPU, drop `fp16=True` and shrink batch to 16-32; `bge-small` still trains in ~15-30 min.

**Do it (re-embed + re-measure — the step people forget):**
```python
from sentence_transformers import SentenceTransformer
from embeddings.recall_eval import recall_at_k

ft = SentenceTransformer("embeddings/bge-small-ft")
base = SentenceTransformer("BAAI/bge-small-en-v1.5")
r_base = recall_at_k(base, corpus_texts, corpus_ids, queries, gold_ids, k=5)
r_ft   = recall_at_k(ft,   corpus_texts, corpus_ids, queries, gold_ids, k=5)  # re-embeds inside
print(f"recall@5  base={r_base:.3f}  ft={r_ft:.3f}  "
      f"abs=+{r_ft-r_base:.3f}  rel=+{100*(r_ft-r_base)/r_base:.0f}%")
```

**Expected result.** A measurable recall@5 lift — lecture 14's worked example goes 0.64 → 0.84 (+0.20 absolute, +31% relative). Even a smaller domain often moves double-digit percentage points, in ~8-12 min on a T4. Report **both** absolute and relative delta with `n`.

**Verify.** `recall@5` for the fine-tuned embedder is measurably above baseline on the **held-out** query set (disjoint from training pairs). You re-embedded the whole corpus with the *fine-tuned* model before measuring (not compared new-query vectors against old-doc vectors).

**Troubleshoot.**
- *Recall went DOWN:* the #1 cause is false negatives (true positives mined as negatives). Tighten `margin` (0.1→0.15), mine deeper (`range_min=15`), and re-read 20 samples. #2 cause is eval leakage — confirm training pairs are disjoint from the held-out queries.
- *Recall didn't move at all:* you forgot to re-embed the corpus with the fine-tuned model, or you're searching new-model queries against an old-model index (a silent, brutal version-mismatch bug).
- *`mine_hard_negatives` not found:* update `sentence-transformers>=3.0`; older versions lack it — you can hand-roll it (encode corpus, cosine-rank each query, take ranks 10-50 minus the positive, apply the margin filter).
- *Cosine vs dot mismatch:* `bge` expects **normalized** embeddings + cosine. Keep `normalize_embeddings=True` everywhere and let MNRL use its cosine default.

---

### Step 8 — Assemble the milestone deliverables (memo + model card)

**What.** Produce the two phase-milestone artifacts:
- **Deliverable A — `memo/DECISION_MEMO.md`:** solve the ticket task **three ways** (few-shot prompt on a frozen instruct model, RAG over labeled examples, your LoRA/QLoRA fine-tune), measured on the **same held-out 100**: accuracy + JSON-valid + F1, **p50/p95 latency**, and **cost per 1k requests**. Recommend which ships. Include the **with/without response-only-loss-masking** A/B from week 1.
- **Deliverable B — `models/MODEL_CARD.md`:** base model, data summary, hyperparameters, task gain, the **base/over-trained/recovered retention scorecard**, quantized-serving metrics (bf16/AWQ/GGUF from week 3), and honest limitations.

**Why.** These are the interview artifacts the whole phase builds toward. The memo proves you can decide **prompt vs RAG vs fine-tune with numbers, not vibes** — and "prompt or RAG wins" is a valid, common, respectable outcome (lecture 01). The model card proves you never call a fine-tune "good" on loss alone: task gain **and** retention evidence, with the forgetting/recovery contrast as the honest centerpiece.

**Do it (measure the three ways on the same 100 — latency + cost too):**
```python
import time, json, numpy as np
from eval.task_eval import task_scorecard, build_prompt

def timed_scorecard(gen_fn, rows, tokenizer):
    lat = []
    def timed(p):
        t0 = time.perf_counter(); out = gen_fn(p); lat.append(time.perf_counter()-t0); return out
    card = task_scorecard(timed, rows, tokenizer)
    card["p50_ms"] = round(1000*float(np.percentile(lat, 50)), 1)
    card["p95_ms"] = round(1000*float(np.percentile(lat, 95)), 1)
    return card

prompt_card = timed_scorecard(gen_fewshot_prompt, heldout, tokenizer)   # frozen instruct + few-shot
rag_card    = timed_scorecard(gen_rag_over_examples, heldout, tokenizer) # Phase-4 retrieve similar tickets
lora_card   = timed_scorecard(lambda p: gen_from("models/ticket-router-recovered", p), heldout, tokenizer)

# cost/1k: for local models estimate from tokens/sec * $/hr(GPU); for API use the priced per-token rate.
json.dump({"prompt": prompt_card, "rag": rag_card, "lora": lora_card},
          open("eval/results/three_ways.json","w"), indent=2)
```

**Do it (memo skeleton — `memo/DECISION_MEMO.md`):**
```markdown
# Decision Memo: Ticket Router — Prompt vs RAG vs LoRA

## Task & holdout
Support ticket -> {category, priority, needs_human} strict JSON. Measured on the SAME 100 held-out rows.

## Results (same holdout, real runs)
| Approach | JSON-valid % | Cat acc % | Priority macro-F1 | p50 ms | p95 ms | $/1k req |
|----------|-------------:|----------:|------------------:|-------:|-------:|---------:|
| Few-shot prompt (frozen instruct) |  |  |  |  |  |  |
| RAG over labeled examples         |  |  |  |  |  |  |
| LoRA/QLoRA fine-tune (recovered)  |  |  |  |  |  |  |

## Masking A/B (from Week 1)
response-only masking vs full-sequence loss: <eval-loss / quality delta with numbers>.

## Recommendation
<Which ships and why — in numbers. If prompt or RAG wins, say so plainly; that's a valid outcome.>
```

**Do it (model card skeleton — `models/MODEL_CARD.md`):**
```markdown
# Model Card: ticket-router (QLoRA, Qwen2.5-7B-Instruct)

## Overview
Base: Qwen/Qwen2.5-7B-Instruct · Method: QLoRA (r=16, alpha=32) · Task: ticket -> JSON routing.

## Data
<size, sources, synthesis+filtering, decontamination (hash-diff train vs eval)>.

## Hyperparameters
Recovered run: LR 1e-4, 2 epochs, 20% replay (Dolly slice), paged AdamW 8-bit, cosine, max_seq 1024.

## Task gain (Leg 1 + Leg 2)
JSON-valid, category acc, priority macro-F1 (base vs FT), + de-biased judge net win-rate (n).

## Retention scorecard (Leg 3) — the headline
| Model | Task acc | MMLU(subset) | IFEval | Safety |
|-------|---------:|-------------:|-------:|-------:|
| Base         |  |  |  |  |
| Over-trained (5ep, LR 5e-4) |  |  |  |  |
| Recovered (2ep, LR 1e-4, 20% replay) |  |  |  |  |

## Serving variants (Week 3)
bf16-merged / AWQ / GGUF q4_k_m: VRAM, tokens/sec, task-accuracy delta after quant.

## Limitations
<rare-class (P0) recall, domain scope, what it forgets, where NOT to use it>.
```

**Expected result.** `eval/results/three_ways.json` filled from real runs; `memo/DECISION_MEMO.md` with a completed table and a clear, numbers-justified recommendation; `models/MODEL_CARD.md` with task gain **and** the three-row retention scorecard.

**Verify.** The memo table has *no empty cells* and a recommendation sentence that cites the numbers. The model card's retention table shows base/over-trained/recovered across ≥3 capabilities and is honest about limitations (e.g. low P0 recall).

**Troubleshoot.**
- *RAG "few-shot from examples" is just prompt-with-context:* that's fine — retrieve the k most similar labeled tickets as in-context exemplars; measure it as its own row.
- *Cost/1k hard to pin for local models:* estimate `$/1k = (tokens_per_req / tokens_per_sec) * ($/hr / 3600) * 1000`. State the assumption.
- *All three approaches tie on accuracy:* a real and common finding — recommend the cheapest/lowest-latency one and say so. The memo's value is the honest comparison, not a forced fine-tune win.

---

## Putting it together — a short end-to-end run

From a fresh Colab T4 runtime, the full week is:
```python
# 0. Venue plan written to README (Step 1): free Colab T4, $0, subset MMLU in the loop.

# 1. Build the three eval legs (Steps 2-4):
task_card = task_scorecard(gen_merged, heldout, tokenizer)          # Leg 1
run_retention("pretrained=models/ticket-router-v1-merged,dtype=bfloat16",
              "eval/results/retention_ft", limit=100)               # Leg 3
wr = win_rate_swapped(prompts, gen_merged, gen_base, my_judge)      # Leg 2 (position-swapped)

# 2. Forgetting demo (Steps 5-6): over-train -> recover -> three-row contrast
#    over-trained: 5 ep, LR 5e-4    -> task UP, IFEval DOWN >=10pp
#    recovered:    2 ep, LR 1e-4, 20% replay -> retention restored, task strong
json.dump(contrast, open("eval/results/forgetting_contrast.json","w"), indent=2)

# 3. Embedder tune (Step 7): mine hard negatives -> MNRL -> RE-EMBED -> recall@5 delta
#    baseline recall@5 -> fine-tuned recall@5 (report abs + rel delta, n)

# 4. Milestone (Step 8): three-ways memo (prompt/RAG/LoRA, same holdout, latency+cost)
#    + MODEL_CARD.md with the base/over-trained/recovered retention scorecard.
```
End state: `eval/results/` holds `task_eval.json`, `retention_scorecard.json`, `win_rate.json`, `forgetting_contrast.json`, `three_ways.json`; `embeddings/bge-small-ft/` + a recall delta; `memo/DECISION_MEMO.md` and `models/MODEL_CARD.md` complete.

---

## Definition of Done

Restating the spine's Week 4 checklist as verifiable checks:
- [ ] **`retention.py` produces a before/after scorecard on ≥3 capabilities (MMLU-slice, IFEval, safety).** `eval/results/retention_scorecard.json` exists with base + ft + delta per task; the `RETENTION_TASKS` list and `FEWSHOT` are fixed across runs.
- [ ] **`task_eval.py` reports JSON-valid %, accuracy, F1, and a de-biased pairwise win-rate vs base.** `eval/results/task_eval.json` (Layer-0 gate applied, macro-F1) + `eval/results/win_rate.json` (wins/ties/losses, n, net win-rate, position-swapped).
- [ ] **The over-trained variant demonstrably loses ≥~10% on a retention metric; the recovered variant restores it** (numbers shown for all three: base / over-trained / recovered). See `eval/results/forgetting_contrast.json`.
- [ ] **Fine-tuned embedder improves recall@5 by a measurable margin** over base on the held-out query set (state absolute + relative delta and n); corpus re-embedded with the fine-tuned model.
- [ ] **`MODEL_CARD.md` complete with task gain AND retention evidence** (the three-row base/over-trained/recovered table).

Milestone acceptance (phase-level) covered by this lab:
- [ ] **Memo table** with accuracy, p50/p95 latency, cost/1k for prompt vs RAG vs LoRA — filled from real runs (`memo/DECISION_MEMO.md`).
- [ ] **Clear ship recommendation** justified by the numbers (including the valid case where prompt/RAG wins).
- [ ] **Masking A/B** result included in the memo (from week 1).
- [ ] **Retention scorecard** shows base / over-trained (forgetting visible) / recovered across ≥3 capabilities.
- [ ] **Model card** complete and honest about limitations.
- [ ] (Carried from weeks 2-3) QLoRA model on a single free/cheap GPU (≤15 GB); adapter + merged + AWQ + GGUF present; GGUF runs in Ollama.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `lm-eval: task not found` | task name not in your harness version | `lm-eval --tasks list \| grep <name>`; use the exact printed string |
| IFEval crashes on import | missing `langdetect`/`nltk`/`immutabledict` | `pip install langdetect nltk immutabledict` and retry |
| Retention "changed" between runs | different subset / few-shot / harness version | freeze `RETENTION_TASKS` + `FEWSHOT`; pin `lm-eval` version |
| Full MMLU eats hours | `--tasks mmlu` = all 57 subjects | use explicit `mmlu_*` subject names; subset in loop, full only for card |
| Win-rate wobbles run-to-run | judge non-determinism | `temperature=0`, pin the judge model version |
| Win-rate suspiciously high | didn't actually swap orders | WIN only if FT wins **both** orders; disagreement = TIE |
| Category accuracy flatters the model | computed over parseable-only | denominator = ALL n; unparseable = wrong on every field |
| Priority F1 hides broken P0 | used micro/weighted F1 | `average="macro"` |
| Over-train didn't cause forgetting | data too small / LoRA protecting base | more epochs / higher LR / wider `target_modules` |
| Recovery didn't restore retention | replay too low or LR too high | replay ~20% of final mix, LR 1e-4, 2 epochs |
| fp16 NaN loss at high LR | T4 has no bf16 | lower LR, or move to L4/A10 (bf16) |
| Embedder recall went DOWN | false negatives (true positives mined as negatives) | tighten `margin`, mine ranks 10-50, eyeball 20 samples |
| Embedder recall didn't move | forgot to re-embed corpus / version mismatch | re-embed whole corpus with FT model; never mix versions |
| `mine_hard_negatives` missing | old sentence-transformers | `pip install -U "sentence-transformers>=3.0"` |

**The five pitfalls to flag explicitly (spine):**
1. **Judging on train/eval loss alone** — you'll ship a forgetful model that "trained well." Use the three legs.
2. **Running full MMLU (57 subjects) in the loop** — hours of compute. Subject subset for iteration, full only for the final report.
3. **No replay on a narrow dataset** — catastrophic forgetting you won't notice until users do. Mix ~10-20% general instruction data.
4. **Mining false negatives** — true positives mined as negatives teach the embedder the opposite. Margin-filter and sanity-check.
5. **Tuning the generator when retrieval was the bottleneck** — tune the embedder first; it's often the bigger, cheaper win.

---

## Stretch goals (optional)

- **Full MMLU for the card.** Run the retention suite once with `--limit` removed and full `mmlu` for a defensible, complete number in the model card (only once — it's hours).
- **A retention *floor* in CI.** Turn the reject rule into code: a script that reads `retention_scorecard.json` and exits non-zero if any core axis (esp. IFEval) drops >10pp or safety drops >2pp. This is model-selection-as-a-gate, not a note.
- **Sweep the forgetting curve.** Train at 1/2/3/5 epochs (or a LR ladder) and plot task accuracy vs IFEval on the same axes — reproduce lecture 13's "scissors" and mark the crossover point.
- **Judge-cost accounting.** Log judge API calls (n pairs × 2 orders) and dollars for the win-rate run; add a cache so re-runs are free. Report the cost in the memo.
- **MRR / nDCG alongside recall@5.** Add ranking metrics to `recall_eval.py` — recall@5 is binary-in-top-k; MRR rewards ranking the right chunk *higher*, which sometimes moves when recall@5 doesn't.
- **Managed-FT comparison.** Fine-tune the same dataset via a provider FT API (OpenAI/Together/Predibase), run *your* `task_eval.py` + `retention.py` on it, and add a row to the memo — quantify the infra-vs-lock-in trade from lecture 11.
- **Cross-encoder reranker for mining.** Add a cross-encoder pass over mined negatives to drop accidental true-positives, then re-train and see if recall improves further — the strongest false-negative defense.
