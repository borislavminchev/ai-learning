# Week 1 Lab: Repo Skeleton and Your First SFT with Masking A/B

> This week you stand up the `llm-finetune-lab/` repo you will use for the whole phase, then run your **first real supervised fine-tune (SFT) end to end**. You pick one narrow, honest task — routing support tickets into strict JSON `{category, priority, needs_human}` — build a clean chat-format dataset, and *see with your own eyes* the three things that silently ruin fine-tunes: the **chat template**, the **EOS token**, and **loss masking**. The headline experiment is a controlled A/B: train the same tiny model once with **response-only loss masking** and once with **loss on the whole sequence**, then compare eval loss and eyeball generations. You debug on `Qwen2.5-0.5B-Instruct` (fast, fits a free GPU or even CPU) so the pipeline is proven before you ever spend money on a 7-8B run in Week 2. No paid API or GPU is required — data prep and template inspection run on your Windows laptop; the actual training runs on a **free Google Colab T4** behind a `--device` flag.
>
> Read the matching lectures first — this guide assumes them:
> - [../lectures/01-prompt-vs-rag-vs-finetune-decision.md](../lectures/01-prompt-vs-rag-vs-finetune-decision.md) — the prompt/RAG/fine-tune heuristic and the one-page decision note you write into the README.
> - [../lectures/02-sft-mechanics-and-the-three-silent-killers.md](../lectures/02-sft-mechanics-and-the-three-silent-killers.md) — next-token loss on (prompt, completion) pairs; loss masking, EOS, and chat templates — the entire spine of this lab.
> - [../lectures/03-base-vs-instruct-and-small-data-hyperparameters.md](../lectures/03-base-vs-instruct-and-small-data-hyperparameters.md) — why an instruct model is your best SFT base, and the LR / epochs / schedule choices for small data.
>
> This lab **creates** the `llm-finetune-lab/` repo. Every later week (QLoRA, DPO, quantization, retention eval) builds on the dataset and repo you produce here. Keep it in git from day one.

**Est. time:** ~8 hrs · **You will need:** Python 3.10+, [`uv`](https://docs.astral.sh/uv/), a Hugging Face account (Qwen2.5 is **not** gated — zero friction; Llama-3.1 is gated and needs a license accept + login), and a browser. **Free/local path:** your Windows/Git-Bash laptop does all data prep, dataset validation, and template inspection on **CPU**; the two training runs happen on a **[free Google Colab T4 (15 GB)](https://colab.research.google.com)**. The `0.5B` model is small enough to also train on a decent CPU/16 GB laptop if you have no Colab — slower, but it works. No paid API, no local GPU required.

---

## Before you start (setup)

**What / Why.** The whole phase lives in one git repo so every lab is version-controlled and gradeable. You manage dependencies with `uv` (fast, reproducible, lockfile-backed). The key architectural decision this week: **separate "runs anywhere" work from "needs a GPU" work**. Data prep, JSONL validation, and template inspection are pure CPU and run on your Windows laptop. Only `train_sft.py` needs a GPU, so it sits behind a `--device` flag — you develop it locally against `--device cpu` on the 0.5B model (it will actually train, just slowly), and run the real thing on Colab with `--device cuda`. This keeps you off the Windows `bitsandbytes` rabbit hole (it doesn't build on Windows-CPU) — but note **Week 1 does not need `bitsandbytes` at all**; full-precision SFT of a 0.5B model needs no 4-bit quantization. We install it now only so the repo's env matches the phase.

**Do it (local, Windows/Git-Bash):**
```bash
# 1. Create and enter the repo
mkdir llm-finetune-lab && cd llm-finetune-lab
git init

# 2. Initialize uv and add the phase dependencies
uv init --python 3.11
uv add "transformers>=4.44" "trl>=0.9" "datasets" "accelerate" "peft" "bitsandbytes" "pydantic"

# 3. Scaffold the directory layout from the spine
mkdir -p data/raw data/prepared scripts eval
touch scripts/prepare_dataset.py scripts/inspect_template.py scripts/train_sft.py README.md
```

**Windows note on `bitsandbytes`.** If `uv add bitsandbytes` errors on Windows (no prebuilt wheel / CUDA missing), don't fight it — you don't need it for Week 1. Drop it from this command and add it in Colab in Week 2 instead:
```bash
uv add "transformers>=4.44" "trl>=0.9" "datasets" "accelerate" "peft" "pydantic"   # bitsandbytes deferred to Colab
```
On macOS/Linux with a CUDA GPU the full command works as written.

**Verify the env:**
```bash
uv run python -c "import transformers, trl, datasets, pydantic; print('trl', trl.__version__, '| transformers', transformers.__version__)"
```
Expected: it prints your `trl` and `transformers` versions with no ImportError. **Write down the `trl` version** — the response-only masking API differs between TRL versions (see Step 5), and you'll pick the right code path based on this number.

**Add a `.gitignore`** so you don't commit multi-GB model caches or datasets you shouldn't:
```bash
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
outputs/
runs/
data/raw/*.csv   # keep raw source out of git if it's large or sensitive
!data/prepared/  # DO commit the prepared JSONL — it's small and it's the deliverable
EOF
git add . && git commit -m "Week 1: repo skeleton"
```

**Gated-model note.** This guide defaults to `Qwen/Qwen2.5-0.5B-Instruct` (train) and `Qwen/Qwen2.5-7B-Instruct` (template inspection) — **neither is gated**, so no login needed. If you prefer Llama-3.1 for inspection, accept its license on the HF hub and run `uv run huggingface-cli login` first.

---

## Step-by-step

### Step 1 — Get or synthesize the ticket dataset (and hold out 100)

**What.** Produce ~300-800 raw support-ticket examples, each labeled with the target `{category, priority, needs_human}`, in `data/raw/tickets.jsonl`. Then split off **exactly 100** rows for eval that you will *never* train on.

**Why.** Fine-tuning is only honest if you measure it on data the model never saw. Lecture 01's decision framework says fine-tune for *behavior you can't reliably prompt* — a narrow classification with strict structured output is the textbook case. You need enough examples to teach the format (a few hundred is plenty for a 0.5B debug run) and a clean, disjoint held-out set so "it works" isn't a memorization illusion (a Week 1 pitfall: *testing on train → fake wins*).

**Do it.** If you have no real tickets, synthesize them with a schema-first generator. Create `scripts/make_synth_data.py`:
```python
"""Synthesize support tickets with strict labels. Deterministic seed so runs are reproducible."""
import json, random
from pathlib import Path

random.seed(7)
CATEGORIES = ["billing", "technical", "account", "shipping", "feature_request"]
PRIORITIES = ["low", "medium", "high", "urgent"]

# Templates keyed by category so the label is derivable from the text (a learnable signal).
TEMPLATES = {
    "billing":         ["I was charged twice for order {n}.", "Why is my invoice {n} higher this month?", "Please refund the duplicate payment on {n}."],
    "technical":       ["The app crashes when I open report {n}.", "Getting a 500 error on the dashboard.", "Login page is stuck loading forever."],
    "account":         ["I can't reset my password.", "Please delete my account {n}.", "How do I change the email on file?"],
    "shipping":        ["Where is my package {n}?", "My order {n} arrived damaged.", "Can I change the delivery address for {n}?"],
    "feature_request": ["Please add dark mode.", "Can you support CSV export?", "I'd love a mobile app."],
}
# Priority/needs_human heuristics make labels non-random but rule-derived.
URGENT_HINTS = ["crash", "500", "urgent", "damaged", "charged twice", "stuck"]

def label_for(cat: str, text: str) -> dict:
    t = text.lower()
    urgent = any(h in t for h in URGENT_HINTS)
    priority = "urgent" if urgent and cat in {"technical", "billing"} else random.choice(PRIORITIES)
    needs_human = cat in {"billing", "account"} or priority in {"high", "urgent"}
    return {"category": cat, "priority": priority, "needs_human": needs_human}

def gen(n_total: int = 600):
    rows = []
    for _ in range(n_total):
        cat = random.choice(CATEGORIES)
        text = random.choice(TEMPLATES[cat]).format(n=random.randint(1000, 9999))
        rows.append({"text": text, "label": label_for(cat, text)})
    return rows

if __name__ == "__main__":
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    rows = gen(600)
    with open("data/raw/tickets.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} raw tickets -> data/raw/tickets.jsonl")
```
Run it:
```bash
uv run python scripts/make_synth_data.py
```

**Expected result.** `data/raw/tickets.jsonl` with 600 lines, each `{"text": ..., "label": {"category","priority","needs_human"}}`.

**Verify.**
```bash
uv run python -c "import json; rows=[json.loads(l) for l in open('data/raw/tickets.jsonl',encoding='utf-8')]; print(len(rows), 'rows'); print(rows[0])"
```
Expected: `600 rows` and a sample row with a valid label dict.

**Troubleshoot.**
- *Labels look random / not learnable* → make the label a function of the text (as above). If category can't be inferred from the text, even a big model can't learn it and your loss won't drop meaningfully.
- *Using real tickets with PII* → scrub names/emails/order numbers before writing to `data/raw/`, and keep `data/raw/` out of git (see `.gitignore`).
- *Only a handful of distinct texts* → the model memorizes them; add more template variety so train and eval genuinely differ.

---

### Step 2 — Define the JSON schema with Pydantic

**What.** Write a Pydantic model for the target output in `scripts/schema.py`. This is the single source of truth for "valid assistant JSON."

**Why.** Your fine-tune target *is* strict JSON (lecture 01: fine-tuning is great at format adherence). If you validate against a real schema — not just "is it JSON" — you catch label typos (`"pariority"`), out-of-range enums, and wrong types *before* they poison training. You reuse this exact model in Week 2/4 eval to compute JSON-valid rate.

**Do it.** In `scripts/schema.py`:
```python
"""Single source of truth for the ticket-routing target JSON."""
from typing import Literal
from pydantic import BaseModel

class TicketLabel(BaseModel):
    category: Literal["billing", "technical", "account", "shipping", "feature_request"]
    priority: Literal["low", "medium", "high", "urgent"]
    needs_human: bool

    def to_json(self) -> str:
        # Compact, key-ordered, deterministic — the model learns ONE canonical string.
        return self.model_dump_json()  # e.g. {"category":"billing","priority":"high","needs_human":true}
```

**Expected result.** `TicketLabel(**row["label"]).to_json()` returns a compact JSON string for every raw row.

**Verify.**
```bash
uv run python -c "from scripts.schema import TicketLabel; print(TicketLabel(category='billing',priority='high',needs_human=True).to_json())"
```
Expected: `{"category":"billing","priority":"high","needs_human":true}`

**Troubleshoot.**
- `ModuleNotFoundError: scripts` → run from the repo root, and either add an empty `scripts/__init__.py` or run with `PYTHONPATH=. uv run python ...`.
- *Non-deterministic key order* → `model_dump_json()` preserves field declaration order; don't hand-build the JSON with `json.dumps(dict)` from an unordered dict, or the target string drifts between rows.

---

### Step 3 — `prepare_dataset.py`: emit chat-format JSONL and assert everything

**What.** Convert raw rows into chat-format JSONL where each row is `{"messages": [system, user, assistant]}` and the **assistant content is the exact canonical target JSON**. Write `data/prepared/train.jsonl` (500 rows) and `data/prepared/eval.jsonl` (100 rows). Assert: every row parses, assistant content is valid `TicketLabel` JSON, and train/eval are **disjoint**.

**Why.** SFT trains on `(prompt, completion)` pairs rendered through the model's chat template (lecture 02). The `messages` list is the portable, template-agnostic representation — you never hand-build `<|im_start|>` strings (a headline pitfall). The assertions are the Definition-of-Done acceptance gate in code: *validated JSONL, valid JSON assistant content, disjoint splits*.

**Do it.** In `scripts/prepare_dataset.py`:
```python
"""Raw tickets -> chat-format JSONL for SFT. Asserts validity + disjointness."""
import json, hashlib, random
from pathlib import Path
from schema import TicketLabel   # run from scripts/ dir, or set PYTHONPATH=.

random.seed(13)
SYSTEM = (
    "You are a support-ticket router. Read the ticket and reply with ONLY a JSON object "
    'with keys "category", "priority", "needs_human". '
    "category in [billing, technical, account, shipping, feature_request]; "
    "priority in [low, medium, high, urgent]; needs_human is a boolean. No prose."
)

def to_chat_row(raw: dict) -> dict:
    label = TicketLabel(**raw["label"])            # validates the label
    return {"messages": [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": raw["text"]},
        {"role": "assistant", "content": label.to_json()},   # EXACT canonical JSON string
    ]}

def row_hash(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()

def main():
    raw = [json.loads(l) for l in open("data/raw/tickets.jsonl", encoding="utf-8")]
    rows = [to_chat_row(r) for r in raw]
    random.shuffle(rows)
    eval_rows, train_rows = rows[:100], rows[100:]      # exactly 100 held out

    # --- Assertions = the acceptance gate ---
    for split_name, split in [("train", train_rows), ("eval", eval_rows)]:
        for r in split:
            assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"], "bad role order"
            TicketLabel.model_validate_json(r["messages"][-1]["content"])  # assistant content is valid JSON+schema
    train_hashes = {row_hash(r) for r in train_rows}
    eval_hashes  = {row_hash(r) for r in eval_rows}
    assert train_hashes.isdisjoint(eval_hashes), "TRAIN/EVAL OVERLAP — contamination!"

    Path("data/prepared").mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train_rows), ("eval", eval_rows)]:
        with open(f"data/prepared/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"OK: {len(train_rows)} train / {len(eval_rows)} eval, disjoint, all assistant JSON valid.")

if __name__ == "__main__":
    main()
```
Run it (note the `cd scripts` so the `from schema import` works, or set `PYTHONPATH=.`):
```bash
PYTHONPATH=scripts uv run python scripts/prepare_dataset.py
```

**Expected result.** `data/prepared/train.jsonl` (500 rows) and `eval.jsonl` (100 rows), and the printout `OK: 500 train / 100 eval, disjoint, all assistant JSON valid.`

**Verify.**
```bash
uv run python -c "import json; r=json.loads(open('data/prepared/train.jsonl',encoding='utf-8').readline()); print([m['role'] for m in r['messages']]); print('assistant:', r['messages'][-1]['content'])"
```
Expected: `['system', 'user', 'assistant']` and a compact JSON assistant string.

**Troubleshoot.**
- `AssertionError: TRAIN/EVAL OVERLAP` → your raw data has exact-duplicate rows; dedupe raw before splitting (`{row_hash(r): r for r ...}.values()`).
- `pydantic ValidationError` on an assistant row → a raw label has a typo or out-of-enum value; fix the generator/source, not the schema.
- *Assistant content is a Python dict, not a string* → TRL expects `content` to be a **string**. Always store `label.to_json()`, never the dict.

---

### Step 4 — `inspect_template.py`: render the template, confirm EOS, count masked tokens

**What.** Load a real instruct tokenizer, print `apply_chat_template(msgs, tokenize=False)`, confirm an **EOS token ends the assistant turn**, then tokenize and print the count of **masked (-100)** vs **unmasked** label tokens for one example under response-only masking.

**Why.** This is the single most important debugging step in the whole phase. Two of the three silent killers (lecture 02) are *visible right here*: (1) if the rendered template has no EOS after the assistant turn, your fine-tune will never learn to stop and will ramble at inference; (2) if you get response-only masking wrong, the model computes loss on the prompt and learns to parrot instructions. You *look at the actual bytes* instead of trusting theory. The Definition of Done requires this printout.

**Do it.** In `scripts/inspect_template.py`:
```python
"""Render the chat template, confirm EOS after the assistant turn, and count
masked vs unmasked LABEL tokens under response-only masking — for ONE example."""
import json, argparse
from transformers import AutoTokenizer

def main(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    row = json.loads(open("data/prepared/train.jsonl", encoding="utf-8").readline())
    msgs = row["messages"]

    # 1) RENDER (no tokenize) — eyeball the special tokens.
    rendered = tok.apply_chat_template(msgs, tokenize=False)
    print("=== RENDERED TEMPLATE ===\n" + rendered + "\n=========================")
    print("EOS token:", repr(tok.eos_token), "| id:", tok.eos_token_id)
    print("Ends with EOS?", rendered.rstrip().endswith(tok.eos_token))

    # 2) MASK — build labels the way response-only masking does: everything up to and
    #    including the assistant HEADER is masked (-100); only the assistant CONTENT + EOS is learned.
    full_ids = tok.apply_chat_template(msgs, tokenize=True)
    prompt_ids = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)
    n_prompt = len(prompt_ids)
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    masked   = sum(1 for x in labels if x == -100)
    unmasked = sum(1 for x in labels if x != -100)
    print(f"\ntotal tokens: {len(full_ids)} | masked(-100): {masked} | unmasked(learned): {unmasked}")
    print("learned token slice decodes to:", repr(tok.decode(full_ids[n_prompt:])))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Use the 7B tokenizer to inspect the template you'll scale to; it's identical to 0.5B's for Qwen2.5.
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    main(ap.parse_args().model)
```
Run it:
```bash
uv run python scripts/inspect_template.py --model Qwen/Qwen2.5-7B-Instruct
# Llama alternative (gated — login first): --model meta-llama/Llama-3.1-8B-Instruct
```

**Expected result.** You see the Qwen ChatML template with `<|im_start|>system … <|im_end|>`, `<|im_start|>user … <|im_end|>`, `<|im_start|>assistant … <|im_end|>`, and a printout like:
```
EOS token: '<|im_end|>' | id: 151645
Ends with EOS? True
total tokens: 118 | masked(-100): 96 | unmasked(learned): 22
learned token slice decodes to: '{"category":"billing","priority":"high","needs_human":true}<|im_end|>\n'
```
The exact counts depend on the row; what matters is **masked >> unmasked** (the prompt is masked) and the learned slice is your JSON + EOS.

**Verify.** Two checks: (1) `Ends with EOS? True` — for Qwen the assistant turn ends with `<|im_end|>` which is the EOS; (2) the "learned token slice" is your target JSON **plus** the EOS marker, and nothing from the system/user prompt.

**Troubleshoot.**
- `Ends with EOS? False` → some templates only add EOS when `add_generation_prompt=False` and you passed the assistant turn; make sure your final message is the assistant turn and you did **not** set `add_generation_prompt=True` on the full render. If the base model's template genuinely omits EOS, you must append it in the collator (Step 5) or the model won't learn to stop.
- *masked count is 0 or equals total* → your `n_prompt` slice is wrong; print `tok.decode(prompt_ids)` and confirm it's exactly system+user+assistant-header (no assistant content).
- `apply_chat_template` raises `no chat template` → you loaded a **base** (non-instruct) model; use the `-Instruct` variant, or you own the template design (lecture 03).
- *Different tokenizer, different template* → that's expected and is the point of lecture 02's "same architecture, different template." Always inspect the tokenizer you'll actually train.

---

### Step 5 — `train_sft.py`: SFTTrainer on 0.5B, behind a `--device` flag, with response-only loss

**What.** Write `scripts/train_sft.py` using TRL `SFTTrainer`/`SFTConfig` with `Qwen2.5-0.5B-Instruct`. Put device selection behind `--device {cpu,cuda}`. Enable **response-only loss masking** and add a `--mask {response,full}` flag so you can run the A/B in Step 6. Log train and eval loss with `>=2` eval steps.

**Why.** You debug on the 0.5B model *first* (lecture 03: start small, prove the pipeline, then scale) — it trains in minutes on a T4 and even runs on CPU. The `--device` flag is what lets you develop on Windows and run for real on Colab without editing code. Response-only masking is the killer you're studying; the `--mask` flag turns the DoD's A/B into a one-argument change.

**TRL version matters.** The response-only API changed across TRL versions — check the version you recorded in setup:
- **TRL < 0.12 (e.g. 0.9-0.11):** use `DataCollatorForCompletionOnlyLM` with the assistant response template string.
- **TRL >= 0.12 / >=0.13:** `SFTConfig(assistant_only_loss=True)` (needs a tokenizer whose chat template emits `{% generation %}` blocks) or keep using the collator. When in doubt, the collator path below works across a wide range.

**Do it.** In `scripts/train_sft.py`:
```python
"""First SFT: Qwen2.5-0.5B-Instruct, response-only vs full-sequence loss (A/B).
Runs on CPU (--device cpu, slow) for local debug or CUDA (--device cuda) on Colab."""
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from trl import DataCollatorForCompletionOnlyLM

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# ChatML assistant header — tokens AFTER this are the response we learn on.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

def build_collator(tok):
    # Only compute loss on tokens after the assistant header -> response-only masking.
    return DataCollatorForCompletionOnlyLM(response_template=RESPONSE_TEMPLATE, tokenizer=tok)

def main(a):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token          # avoid pad==eos loss surprises; set explicitly
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    ds = load_dataset("json", data_files={
        "train": "data/prepared/train.jsonl",
        "eval":  "data/prepared/eval.jsonl",
    })

    cfg = SFTConfig(
        output_dir=f"outputs/sft-{a.mask}",
        num_train_epochs=a.epochs,             # 1-2 for small data; more overfits (pitfall)
        per_device_train_batch_size=8 if a.device == "cuda" else 2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="steps", eval_steps=20,  # >=2 eval steps over the run (DoD)
        save_strategy="no",
        max_seq_length=512,                    # our rows are short; don't waste memory
        bf16=(a.device == "cuda"), fp16=False,
        report_to="none",
        packing=False,                         # keep template boundaries clean for the A/B
    )

    collator = build_collator(tok) if a.mask == "response" else None  # None -> loss on WHOLE sequence
    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=ds["train"], eval_dataset=ds["eval"],
        data_collator=collator,                # response-only when set; full-sequence when None
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"[{a.mask}] final eval_loss = {metrics['eval_loss']:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--mask", choices=["response", "full"], default="response")
    ap.add_argument("--epochs", type=float, default=2.0)
    main(ap.parse_args())
```

**Note on `SFTTrainer` chat handling.** Recent TRL auto-applies the chat template to `messages`-format datasets and passes `processing_class=tok` instead of `tokenizer=`. If your TRL version complains, pass the tokenizer as `processing_class=tok` and/or `SFTConfig(dataset_text_field=None)`. The collator's `response_template` must match your tokenizer's assistant header **exactly** (from Step 4's printout) — if you use Llama-3.1, change it to `"<|start_header_id|>assistant<|end_header_id|>\n\n"`.

**Do it (local smoke test — proves the code runs before Colab):**
```bash
# CPU, tiny, just to confirm it trains at all. Ctrl-C after you see loss logging.
uv run python scripts/train_sft.py --device cpu --mask response --epochs 1
```

**Do it (Colab — the real run).** Push the repo, then in a T4 notebook:
```python
!git clone <your-repo-url> && cd llm-finetune-lab && pip install -q "transformers>=4.44" "trl>=0.9" datasets accelerate peft
!cd llm-finetune-lab && python scripts/train_sft.py --device cuda --mask response --epochs 2
```

**Expected result.** Loss logs every 5 steps with **decreasing train loss**, at least 2 eval-loss readings (steps 20, 40, …), and a final line like `[response] final eval_loss = 0.42`.

**Verify.** Train loss trends down across logging steps; `eval_loss` is printed for `>=2` eval steps. Save the console log — you need those numbers for the README A/B.

**Troubleshoot.**
- `DataCollatorForCompletionOnlyLM` warns *"Could not find response key"* → the `response_template` string doesn't match the tokenized template. Copy the exact assistant-header string from Step 4's rendered output; for multi-token headers, pass the **token ids** instead of the string.
- *Loss is `nan`* → LR too high for full FT, or fp16 on CPU. Use `bf16` on GPU / fp32 on CPU, and drop LR to `1e-4`.
- `KeyError: 'messages'` / dataset not templated → your TRL version needs `processing_class=tok` and possibly explicit template application; upgrade TRL or map the dataset with `tok.apply_chat_template(..., tokenize=False)` into a `text` field first.
- *CPU run takes forever* → that's expected; the CPU run is only a smoke test. Do the real 2-epoch run on Colab.
- *OOM on T4* → lower `per_device_train_batch_size` to 4 and raise `gradient_accumulation_steps`.

---

### Step 6 — Run the masking A/B and write the difference into the README

**What.** Train the identical setup twice — `--mask response` and `--mask full` — on the same data and seed. Compare final eval loss, generate on 10 held-out inputs from each, and write the comparison (with numbers) into `README.md`.

**Why.** This is the DoD headline: *response-only masking yields lower eval loss OR you can explain why not, with numbers.* Full-sequence loss makes the model spend capacity learning to reproduce the (fixed) system+user prompt — wasted signal that often shows as worse or noisier eval loss and, more tellingly, generations that echo instructions or drift from strict JSON. Eyeballing 10 generations catches quality problems that a single loss number hides (lecture 02).

**Do it (Colab, both runs):**
```python
!cd llm-finetune-lab && python scripts/train_sft.py --device cuda --mask response --epochs 2
!cd llm-finetune-lab && python scripts/train_sft.py --device cuda --mask full     --epochs 2
```
Add a tiny generation helper to eyeball outputs — `scripts/generate_sample.py`:
```python
"""Load a trained checkpoint and print generations for the first 10 eval prompts."""
import json, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(ckpt):
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt)
    rows = [json.loads(l) for l in open("data/prepared/eval.jsonl", encoding="utf-8")][:10]
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=64, do_sample=False, eos_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print("GOLD:", r["messages"][-1]["content"], "\nGEN :", gen.strip(), "\n---")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
    main(ap.parse_args().ckpt)
```
```python
!cd llm-finetune-lab && python scripts/generate_sample.py --ckpt outputs/sft-response
!cd llm-finetune-lab && python scripts/generate_sample.py --ckpt outputs/sft-full
```

**Expected result.** Two final eval-loss numbers and two sets of 10 generations. Typical outcome: the response-only model has **lower or comparable eval loss** and produces cleaner strict JSON that stops at EOS; the full-sequence model more often adds stray prose or repeats the instruction.

**Verify.** Write the numbers into `README.md`, e.g.:
```markdown
## Masking A/B (Qwen2.5-0.5B, 2 epochs, LR 2e-4)
| run           | final eval_loss | valid JSON in 10 gens | notes                         |
|---------------|-----------------|-----------------------|-------------------------------|
| response-only | 0.42            | 10/10                 | stops at EOS, clean JSON      |
| full-sequence | 0.55            | 7/10                  | 3 gens echo the instruction   |
Conclusion: response-only masking lowered eval loss by 0.13 and fixed 3 malformed outputs.
```
If full-sequence somehow *won* (small data, short prompts can make the gap tiny or reversed), **explain why with the numbers** — that satisfies the DoD too.

**Troubleshoot.**
- *Both eval losses nearly identical* → expected when prompts are short relative to the response; note it and lean on the generation quality diff, which is where the visible difference shows.
- *Generations don't stop / ramble* → EOS problem (revisit Step 4); confirm `eos_token_id` is passed to `generate` and the template ended with EOS.
- *Both models output garbage* → the pipeline, not the masking, is broken; re-run Step 4 and confirm the template + learned-slice are correct.

---

### Step 7 — Write the one-page decision-framework note

**What.** In `README.md`, write a one-page note: for the ticket-routing task, *why are you even considering fine-tuning vs prompt vs RAG?*

**Why.** The DoD requires it, and lecture 01 is the whole point of Week 1: you don't fine-tune until prompting and RAG have provably failed or don't fit. Writing this forces the honest answer — for a narrow, stable, high-volume classification with strict JSON output and latency/cost pressure, a small fine-tune is the *right* tool; for injecting product facts, it's the wrong one (fine-tuning teaches behavior, not facts).

**Do it.** Add to `README.md`:
```markdown
## Decision note: prompt vs RAG vs fine-tune (ticket routing)
- **Task:** classify each ticket to strict JSON {category, priority, needs_human}.
- **Prompt-first?** A few-shot prompt on a 7B instruct model works but costs ~800 prompt
  tokens/request and JSON-valid rate is ~90% under load — format drift is the failure mode.
- **RAG?** No new *facts* are needed (labels are derivable from the ticket text), so retrieval
  adds latency without accuracy. RAG is the wrong tool here.
- **Fine-tune?** Signals FOR tuning are present: stable task, >=500 clean examples, strict
  structured output, and cost/latency pressure (we want a tiny model + short prompt in prod).
  Fine-tuning trades a one-time train cost for cheaper, more reliable inference.
- **Decision:** SFT a small model to lock the format; measure against the prompt baseline in the
  Week-4 memo before shipping. Fine-tuning here is for BEHAVIOR (format adherence), not facts.
```

**Expected result.** A ~1-page section in `README.md` a reviewer can read and agree the choice is justified.

**Verify.** It names the task, states why prompt and RAG are insufficient/wrong, lists concrete signals for tuning, and lands a decision — all in one page.

**Troubleshoot.**
- *You can't justify fine-tuning over prompting* → good; that's a valid outcome. Say so and note you'd ship the prompt baseline. The DoD asks for the *reasoning*, not a forced "yes."

---

## Putting it together — short end-to-end run

From a clean checkout, the whole week runs like this:
```bash
# --- Local (Windows/Git-Bash): build + validate the dataset, inspect the template ---
cd llm-finetune-lab
uv run python scripts/make_synth_data.py                       # 600 raw tickets
PYTHONPATH=scripts uv run python scripts/prepare_dataset.py     # -> train.jsonl (500) + eval.jsonl (100), asserts pass
uv run python scripts/inspect_template.py --model Qwen/Qwen2.5-7B-Instruct   # EOS + mask-count printout
uv run python scripts/train_sft.py --device cpu --mask response --epochs 1   # smoke test only; Ctrl-C after loss logs

# --- Colab (free T4): the two real training runs + generations ---
# (git clone the repo, pip install, then:)
python scripts/train_sft.py --device cuda --mask response --epochs 2
python scripts/train_sft.py --device cuda --mask full     --epochs 2
python scripts/generate_sample.py --ckpt outputs/sft-response
python scripts/generate_sample.py --ckpt outputs/sft-full
# --- Write the A/B table + decision note into README.md, then commit. ---
git add data/prepared scripts README.md && git commit -m "Week 1: first SFT + masking A/B"
```
End state: a version-controlled `llm-finetune-lab/` repo with validated chat-format data, a template+mask printout you understand, two trained 0.5B checkpoints, and a README documenting the masking A/B (with numbers) and the decision note.

---

## Definition of Done — verifiable checks

Restating the spine's acceptance gate; each is checkable from your repo:

- [ ] **`data/prepared/*.jsonl` validates.** `prepare_dataset.py` runs and prints `OK: 500 train / 100 eval, disjoint, all assistant JSON valid.` — every row parses, assistant content is valid `TicketLabel` JSON, and train ∩ eval = ∅ (asserted via row hashes).
- [ ] **`inspect_template.py` prints a rendered template with EOS after the assistant turn**, `Ends with EOS? True`, and the masked(-100) vs unmasked label-token counts for one example.
- [ ] **`train_sft.py` completes on the 0.5B model** with `eval_loss` logged for `>=2` eval steps and a **decreasing train loss** (save the log).
- [ ] **The masking A/B is written up in `README.md`** with numbers: response-only eval loss vs full-sequence eval loss, plus the 10-generation eyeball — response-only wins, OR you explain why not.
- [ ] **A one-page decision-framework note is in `README.md`** justifying fine-tune vs prompt vs RAG for the ticket task.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `uv add bitsandbytes` fails on Windows | no prebuilt wheel / no CUDA locally | Skip it this week; install in Colab (Week 2). Week 1 needs no 4-bit. |
| `apply_chat_template` raises "no chat template" | loaded a **base** model | Use the `-Instruct` variant, or design your own template (lecture 03). |
| `Ends with EOS? False` | wrong render args / template omits EOS | Ensure final message is the assistant turn, no `add_generation_prompt=True`; else append EOS in the collator. |
| `DataCollatorForCompletionOnlyLM`: "response key not found" | `response_template` != actual header tokens | Copy the exact assistant-header string from Step 4; for multi-token headers pass token IDs. |
| Train loss `nan` | LR too high / fp16 on CPU | Drop LR to `1e-4`; use bf16 on GPU, fp32 on CPU. |
| `KeyError: 'messages'` in SFTTrainer | TRL version handles chat differently | Pass `processing_class=tok`; or pre-render dataset into a `text` field with `apply_chat_template`. |
| Eval loss keeps rising while train loss drops | over-epoching on tiny data | Use 1-2 epochs; this is the memorization pitfall. |
| "It works great!" (tested on train) | testing on training examples | Only ever eval on the disjoint `eval.jsonl`; the hash assert guards this. |
| Generations ramble / never stop | missing/ignored EOS | Confirm EOS in template (Step 4) and pass `eos_token_id` to `generate`. |
| Assistant content is a dict, not a string | stored `label` not `label.to_json()` | TRL wants string `content`; always serialize. |

---

## Stretch goals (optional)

- **Prompt baseline, no training.** Run the frozen `Qwen2.5-0.5B-Instruct` (and a 7B) on `eval.jsonl` with just the system prompt; record JSON-valid rate and accuracy. This is the "prompt" leg you'll formalize in the Week-4 decision memo — knowing it now tells you whether fine-tuning even helps.
- **Third A/B arm: `packing=True`.** Re-run with packing on and watch for template-boundary cross-contamination; note the throughput gain vs any quality cost.
- **Field-level accuracy, not just loss.** Parse each generation with `TicketLabel` and compute per-field accuracy (category / priority / needs_human) and JSON-valid rate on the held-out 100. Loss is a proxy; this is the real metric and previews Week-4 `task_eval.py`.
- **Try the Llama-3.1 template.** Re-run `inspect_template.py` with `meta-llama/Llama-3.1-8B-Instruct` (after license accept + login) and diff the special tokens against Qwen's — concrete evidence for lecture 02's "same task, different template."
- **Seed sensitivity.** Train the response-only run at 3 seeds and report eval-loss variance, so your A/B conclusion isn't a single-seed fluke.
