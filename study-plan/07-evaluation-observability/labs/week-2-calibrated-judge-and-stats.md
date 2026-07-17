# Week 2 Lab: A Calibrated, De-biased Judge with Statistical Rigor

> This week you turn "the output looked good" into a number you can defend. You build an **LLM-as-judge** with a strict, criteria-driven rubric and a low-cardinality structured `Verdict` (reasoning first, score 1–4, pass/fail); make it **pairwise** with position-bias control; **calibrate** it against your own human labels with Cohen's kappa and improve it; add **RAG metrics** (RAGAS faithfulness / answer-relevancy / context-precision) or a **hallucination tripwire**; and finally wrap every result in a **bootstrap 95% CI** and a **paired McNemar test** so you never call a 2%-on-50 delta a win. By the end you have `src/judge.py` and `evals/stats.py` that any later CI gate (Week 3) can call.
>
> Read the matching lectures first — this guide assumes them:
> - [../lectures/05-llm-as-judge-mechanics.md](../lectures/05-llm-as-judge-mechanics.md) — modes, CoT-first, structured `Verdict`, Ollama path
> - [../lectures/06-judge-biases.md](../lectures/06-judge-biases.md) — position / verbosity / self-enhancement bias and mitigations
> - [../lectures/07-judge-calibration-kappa.md](../lectures/07-judge-calibration-kappa.md) — hand-labeling and Cohen's kappa
> - [../lectures/08-rag-eval-hallucination.md](../lectures/08-rag-eval-hallucination.md) — RAGAS triad + NLI/SelfCheckGPT tripwires
> - [../lectures/09-statistical-rigor.md](../lectures/09-statistical-rigor.md) — bootstrap CIs and paired tests

**Est. time:** ~9 hrs (the densest week) · **You will need:** the `llm-evals` repo from Week 1 (with `evals/data/golden_v1.jsonl`); Python 3.11+ with `uv`; a judge LLM. **Free/local path:** run [Ollama](https://ollama.com) and use its OpenAI-compatible endpoint at `http://localhost:11434/v1` — the pipeline is byte-for-byte identical to a paid API, only judge quality drops. NLI tripwire runs on CPU. **Critical constraint:** the judge model family must differ from the family that *generated* the outputs you are judging (self-enhancement bias — see Step 1).

---

## Before you start (setup)

**WHAT:** Confirm you are in the Week-1 repo, install this week's dependencies, and set the judge provider.

**WHY:** Every step below extends the same `llm-evals` package. RAGAS/DeepEval pull heavy transitive deps; installing them up front avoids mid-lab surprises.

**Do it:**
```bash
cd llm-evals            # the repo from Week 1
# core stats + judging deps
uv add instructor scipy statsmodels numpy scikit-learn pydantic python-dotenv rich
# RAG eval (skip if you have no RAG system, but installing is harmless)
uv add ragas datasets
# hallucination tripwire (CPU-friendly NLI cross-encoder)
uv add sentence-transformers
# optional: deepeval as an alternative judge/metric library
uv add deepeval

mkdir -p evals src
```

Pick ONE judge backend and put it in `.env`:

```bash
# Option A — paid API (strong judge). Family MUST differ from your generator.
#   If your system generated with GPT, judge with Claude (or vice-versa).
JUDGE_PROVIDER=openai
JUDGE_MODEL=gpt-4o-2024-11-20          # pin a dated snapshot, never a bare alias
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...         # if judging with Claude

# Option B — FREE / LOCAL via Ollama (OpenAI-compatible endpoint)
#   ollama pull llama3.1:8b   (or qwen2.5:7b-instruct)
# JUDGE_PROVIDER=ollama
# JUDGE_MODEL=llama3.1:8b
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_API_KEY=ollama                # any non-empty string; Ollama ignores it
```

**Windows / Git-Bash note:** `uv` commands are identical. For Ollama on Windows, install the native app from ollama.com, then `ollama pull llama3.1:8b` in any terminal; the server listens on `localhost:11434` automatically. Verify with `curl http://localhost:11434/v1/models`.

**Expected result:** `uv sync` succeeds; `uv run python -c "import instructor, ragas, sklearn, statsmodels, sentence_transformers; print('ok')"` prints `ok`.

**Verify:** `uv run python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('JUDGE_MODEL'))"` prints your model id.

**Troubleshoot:**
- `ragas` install fails on Windows building a wheel → ensure Python 3.11/3.12 (not 3.13, which some deps lag on): `uv python pin 3.12 && uv sync`.
- `sentence-transformers` pulls a large torch wheel → that is expected (~2 GB CPU torch). If disk is tight, defer it until Step 4.
- Ollama endpoint refused → the app/server isn't running. Start it (`ollama serve` if not auto-started) and re-`curl`.

---

## Step-by-step

### Step 1 — Build the rubric judge with a structured `Verdict`

**WHAT:** Create `src/judge.py`: a Pydantic `Verdict` with fields in the order **reasoning → score → pass_**, a strict criteria-driven system prompt that explicitly ignores length, and a `judge()` function that uses `instructor` to force the model to return a well-formed `Verdict`.

**WHY:** Field order is load-bearing. If `score` came before `reasoning`, the model commits to a number first and the "reasoning" becomes post-hoc rationalization (see [../lectures/05-llm-as-judge-mechanics.md](../lectures/05-llm-as-judge-mechanics.md)). A 1–4 scale is deliberately low-cardinality — LLMs cannot reliably discriminate on a 1–100 scale, and a fine scale gives an illusion of precision that is pure noise. `instructor` (or a provider's native `response_format`) guarantees a parseable object instead of you regexing JSON out of prose.

**Do it:**
```python
# src/judge.py
from __future__ import annotations
import os
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

class Verdict(BaseModel):
    """Field ORDER matters: reasoning is emitted FIRST so the score is grounded in it."""
    reasoning: str = Field(..., description="Step-by-step evaluation against EACH criterion, before any score.")
    score: int = Field(..., ge=1, le=4, description="1=fails criteria, 2=major gaps, 3=minor gaps, 4=fully meets.")
    pass_: bool = Field(..., description="True only if the output satisfies the criteria; typically score>=3.")

JUDGE_SYSTEM = """You are a strict, impartial evaluator.
You are given INPUT, OUTPUT, and CRITERIA.
Rules:
1. First reason step by step, checking the OUTPUT against EACH item in CRITERIA.
2. Then assign a score from 1 to 4: 1=fails, 2=major gaps, 3=minor gaps, 4=fully meets.
3. Then decide pass/fail: pass only when the criteria are actually satisfied (usually score>=3).
IMPORTANT: Judge ONLY correctness and adherence to CRITERIA. Ignore answer length, verbosity,
tone, and formatting flourish entirely. A long answer is not better; a short correct answer passes."""

def _client() -> instructor.Instructor:
    base_url = os.getenv("OPENAI_BASE_URL")          # set for Ollama; None for OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    raw = OpenAI(api_key=api_key, base_url=base_url)
    # JSON mode works for both OpenAI and Ollama's OpenAI-compatible server.
    return instructor.from_openai(raw, mode=instructor.Mode.JSON)

def judge(input_text: str, output_text: str, criteria: str,
          model: str | None = None, max_retries: int = 2) -> Verdict:
    model = model or os.getenv("JUDGE_MODEL", "gpt-4o-2024-11-20")
    client = _client()
    user = f"INPUT:\n{input_text}\n\nOUTPUT:\n{output_text}\n\nCRITERIA:\n{criteria}"
    return client.chat.completions.create(
        model=model,
        response_model=Verdict,
        max_retries=max_retries,           # instructor re-asks the model on a bad parse
        temperature=0,                     # judging should be as deterministic as possible
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": user}],
    )

if __name__ == "__main__":
    from dotenv import load_dotenv; load_dotenv()
    v = judge(
        input_text="What is the capital of France?",
        output_text="The capital of France is Paris.",
        criteria="Names the correct capital (Paris). No fabricated extra facts.",
    )
    print(v.model_dump())
```

Now the acceptance check — **20/20 sampled cases must return a well-formed `Verdict`**:
```python
# evals/smoke_judge.py
import json, pathlib, random
from dotenv import load_dotenv; load_dotenv()
from src.judge import judge, Verdict

rows = [json.loads(l) for l in pathlib.Path("evals/data/golden_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
sample = random.sample(rows, min(20, len(rows)))
ok = 0
for r in sample:
    crit = r.get("criteria") or "Answer is correct and directly addresses the input."
    out = r.get("output") or r.get("expected") or ""      # judge the system output for this case
    try:
        v = judge(r["input"], out, crit)
        assert isinstance(v, Verdict) and 1 <= v.score <= 4 and isinstance(v.pass_, bool) and v.reasoning
        ok += 1
    except Exception as e:
        print("BAD:", r["id"], repr(e))
print(f"well-formed Verdicts: {ok}/{len(sample)}")
```
```bash
uv run python -m evals.smoke_judge
```

**Expected result:** `well-formed Verdicts: 20/20` (or `N/N` if your golden set has fewer than 20 cases). Each `Verdict` has non-empty `reasoning`, `1<=score<=4`, boolean `pass_`.

**Verify:** Print one full verdict (`uv run python src/judge.py`) and read the `reasoning` — it should reference the *criteria*, not the answer's length. Confirm `score` is always in `{1,2,3,4}`.

**Troubleshoot:**
- Parse/validation errors on Ollama → small models sometimes emit prose around the JSON. Keep `mode=instructor.Mode.JSON` and `max_retries>=2`; if it still fails, switch to a more instruction-following local model (`qwen2.5:7b-instruct`).
- Model returns `score` as a string → the `ge/le` validator + instructor retry usually fix it; if not, add `Field(..., strict=False)` is unnecessary — instead lower temperature to 0 and keep the retry.
- **Self-enhancement guard:** if `JUDGE_MODEL` is the same family as your generator, stop and switch. Judging GPT-4o output with GPT-4o inflates scores; use Claude-judges-GPT, or Llama-judges-both ([../lectures/06-judge-biases.md](../lectures/06-judge-biases.md)).

---

### Step 2 — Add pairwise mode with position-bias control

**WHAT:** Add a `judge_pairwise(input, a, b, criteria)` that asks the judge which of two outputs is better, runs it **both** as (A,B) and (B,A), counts a win **only** when the two orderings agree, and returns the **position-disagreement rate** across a batch.

**WHY:** LLM judges have a documented position bias: they favor whichever answer they read first. If you ask once, you are measuring "which slot won" as much as "which answer is better." Running both orders and requiring cross-order consistency neutralizes it; the fraction of pairs where the two orders *disagree* is a direct empirical measurement of the residual bias — you log it, you don't assume it's zero ([../lectures/06-judge-biases.md](../lectures/06-judge-biases.md)).

**Do it:**
```python
# append to src/judge.py
from typing import Literal
from pydantic import BaseModel

class PairwiseVerdict(BaseModel):
    reasoning: str
    winner: Literal["FIRST", "SECOND", "TIE"]   # position-relative, NOT "A"/"B"

PAIRWISE_SYSTEM = """You are a strict, impartial evaluator comparing two candidate answers.
You are given INPUT, two answers (FIRST and SECOND), and CRITERIA.
1. Reason step by step, checking BOTH answers against EACH criterion.
2. Then output the winner: FIRST, SECOND, or TIE.
Judge ONLY correctness and adherence to CRITERIA. Ignore length, verbosity, tone, formatting."""

def _pairwise_once(input_text, first, second, criteria, model=None) -> PairwiseVerdict:
    model = model or os.getenv("JUDGE_MODEL", "gpt-4o-2024-11-20")
    client = _client()
    user = f"INPUT:\n{input_text}\n\nFIRST:\n{first}\n\nSECOND:\n{second}\n\nCRITERIA:\n{criteria}"
    return client.chat.completions.create(
        model=model, response_model=PairwiseVerdict, max_retries=2, temperature=0,
        messages=[{"role": "system", "content": PAIRWISE_SYSTEM},
                  {"role": "user", "content": user}])

def judge_pairwise(input_text, a, b, criteria, model=None) -> dict:
    """Run (A,B) then (B,A). Consistent winner only. Returns which of A/B won, or TIE/INCONSISTENT."""
    ab = _pairwise_once(input_text, a, b, criteria, model)   # A is FIRST
    ba = _pairwise_once(input_text, b, a, criteria, model)   # B is FIRST
    # map position winners back to A/B
    win_ab = {"FIRST": "A", "SECOND": "B", "TIE": "TIE"}[ab.winner]
    win_ba = {"FIRST": "B", "SECOND": "A", "TIE": "TIE"}[ba.winner]  # positions swapped
    if win_ab == win_ba and win_ab != "TIE":
        result = win_ab                    # cross-order consistent real win
    elif win_ab == "TIE" and win_ba == "TIE":
        result = "TIE"
    else:
        result = "INCONSISTENT"            # position bias flipped the verdict
    return {"winner": result, "order_ab": win_ab, "order_ba": win_ba,
            "consistent": result not in ("INCONSISTENT",)}

def pairwise_batch(cases, model=None) -> dict:
    """cases: list of {input, a, b, criteria}. Returns wins + position-disagreement rate."""
    results = [judge_pairwise(c["input"], c["a"], c["b"], c["criteria"], model) for c in cases]
    n = len(results)
    disagree = sum(1 for r in results if r["winner"] == "INCONSISTENT")
    a_wins = sum(1 for r in results if r["winner"] == "A")
    b_wins = sum(1 for r in results if r["winner"] == "B")
    ties = sum(1 for r in results if r["winner"] == "TIE")
    return {"n": n, "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
            "position_disagreement_rate": disagree / n if n else 0.0, "results": results}
```

Run it on ~15–20 pairs (reuse two system variants, or pair each golden output against a deliberately weaker paraphrase):
```python
# evals/run_pairwise.py
import json, pathlib
from dotenv import load_dotenv; load_dotenv()
from src.judge import pairwise_batch

rows = [json.loads(l) for l in pathlib.Path("evals/data/golden_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:20]
cases = [{"input": r["input"],
          "a": r.get("output") or r.get("expected", ""),
          "b": (r.get("expected", "") + " (also, unrelated tangent added).") or "n/a",
          "criteria": r.get("criteria", "Correct and directly answers the input.")} for r in rows]
summary = pairwise_batch(cases)
print(f"A:{summary['a_wins']} B:{summary['b_wins']} TIE:{summary['ties']} "
      f"position_disagreement_rate={summary['position_disagreement_rate']:.2%}")
```
```bash
uv run python -m evals.run_pairwise
```

**Expected result:** A line reporting wins and a `position_disagreement_rate` (typically 5–25% for a strong judge; higher for a small local model). The number is what matters — you have *measured* the bias.

**Verify:** Manually inspect a couple of `INCONSISTENT` results in `summary["results"]` — the judge should have literally flipped its winner when you swapped slots. That confirms the control is doing real work.

**Troubleshoot:**
- Disagreement rate ~50% → the judge is essentially guessing (pairs are too close, or the model is too weak). Sharpen the criteria or use a stronger judge.
- Rate is 0% on a tiny sample → not necessarily bias-free; run ≥15 pairs before trusting it.
- Always keep winners **position-relative** in the schema (`FIRST`/`SECOND`). If you let the model output `A`/`B`, it may anchor on the labels and you lose the swap's diagnostic power.

---

### Step 3 — Calibrate against human labels with Cohen's kappa (and improve it)

**WHAT:** Hand-label 30–50 golden cases pass/fail yourself, run the Step-1 judge on the same cases, compute `cohen_kappa_score(human, judge)`. If kappa < 0.6, revise the rubric (tighten vague criteria) and re-run. Record **before and after** kappa.

**WHY:** A judge is a measuring instrument, and an uncalibrated instrument is worse than none — it hands you an authoritative-feeling number that points anywhere. Cohen's kappa measures agreement *corrected for chance* (raw agreement % is inflated when one label dominates). Kappa > 0.6 is decent, > 0.8 strong. A judge at kappa 0.3 is a random number generator with good grammar; you must calibrate before you trust any downstream metric ([../lectures/07-judge-calibration-kappa.md](../lectures/07-judge-calibration-kappa.md)).

**Do it:** First, hand-label. Reuse Week-1 annotations where you can; otherwise a quick CLI:
```python
# evals/hand_label.py  -> writes evals/data/human_labels.jsonl
import json, pathlib, random
rows = [json.loads(l) for l in pathlib.Path("evals/data/golden_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
sample = random.sample(rows, min(40, len(rows)))   # 30-50 cases
out = []
for r in sample:
    print("\n== id:", r["id"]); print("INPUT:", r["input"][:600])
    print("OUTPUT:", (r.get("output") or r.get("expected", ""))[:600])
    print("CRITERIA:", r.get("criteria", "(none)"))
    lab = ""
    while lab not in ("p", "f"):
        lab = input("PASS or FAIL? [p/f]: ").strip().lower()
    out.append({"id": r["id"], "human_pass": lab == "p"})
pathlib.Path("evals/data/human_labels.jsonl").write_text(
    "\n".join(json.dumps(o) for o in out), encoding="utf-8")
print(f"labeled {len(out)} cases")
```
```bash
uv run python -m evals.hand_label     # do the labeling honestly, by the criteria
```

Then compute kappa:
```python
# evals/calibration.py
import json, pathlib
from dotenv import load_dotenv; load_dotenv()
from sklearn.metrics import cohen_kappa_score
from src.judge import judge

def load(p): return {json.loads(l)["id"]: json.loads(l)
                     for l in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if l.strip()}

def run_calibration() -> float:
    gold = load("evals/data/golden_v1.jsonl")
    human = load("evals/data/human_labels.jsonl")
    ids = list(human.keys())
    human_labels, judge_labels = [], []
    for cid in ids:
        r = gold[cid]
        v = judge(r["input"], r.get("output") or r.get("expected", ""),
                  r.get("criteria", "Correct and directly answers the input."))
        human_labels.append(int(human[cid]["human_pass"]))
        judge_labels.append(int(v.pass_))
    kappa = cohen_kappa_score(human_labels, judge_labels)
    agree = sum(int(a == b) for a, b in zip(human_labels, judge_labels)) / len(ids)
    print(f"n={len(ids)}  raw_agreement={agree:.2%}  cohen_kappa={kappa:.3f}")
    return kappa

if __name__ == "__main__":
    k = run_calibration()
    print("PASS (>=0.6)" if k >= 0.6 else "LOW — revise rubric and re-run")
```
```bash
uv run python -m evals.calibration     # this is your BEFORE kappa
```

If kappa < 0.6, diagnose *where* human and judge diverge (print the disagreeing ids), then tighten `JUDGE_SYSTEM`/criteria — the usual culprit is vague criteria ("is it good?") or an undefined pass threshold. Common fixes: enumerate concrete pass conditions, define what "grounded"/"correct" means for your task, state the score→pass mapping explicitly. Re-run and record the **after** kappa. Log both in `evals/data/calibration_log.md`:

```markdown
# Judge calibration log
- 2026-07-09  rubric v1 (vague "is the answer good"): n=40, kappa=0.41  ← BEFORE
- 2026-07-09  rubric v2 (enumerated 4 pass conditions, defined "grounded"): n=40, kappa=0.71  ← AFTER
```

**Expected result:** A before kappa (often 0.3–0.5 for a first vague rubric), a rubric revision, and an after kappa **≥ 0.6 and higher than before**, both written to the log.

**Verify:** Re-open `calibration_log.md` — two dated entries, kappa improved. `run_calibration()` prints `PASS (>=0.6)`.

**Troubleshoot:**
- Kappa is high but raw agreement is ~50/50 → good, that is exactly what kappa is for; trust kappa.
- Kappa stuck low after several rubric edits → check for a *systematic directional* disagreement (judge always stricter/looser). If the judge is consistently one-directional, adjust the pass threshold (e.g., pass on `score>=3` vs `>=2`), not just the wording.
- Kappa is `nan` → one rater gave a single constant label (all pass or all fail). Re-sample to include both classes; kappa is undefined without variance.
- Don't over-fit the rubric to these 40 cases. Keep the rubric general; if you tuned hard, re-validate on a fresh held-out handful.

---

### Step 4 — RAG metrics (RAGAS) or a hallucination tripwire

**WHAT:** If your system is RAG: build a RAGAS dataset from `golden_v1` and report **faithfulness + answer_relevancy + context_precision**. Regardless of RAG or not: add a cheap **NLI entailment tripwire** (or SelfCheckGPT-style consistency check) that flags **≥1 known-bad case**.

**WHY:** A single end-to-end score can't tell you *which* half of a RAG stack broke. The triad decomposes it: faithfulness = is the answer grounded in the retrieved context (hallucination signal); answer_relevancy = does it actually address the question; context_precision = did retrieval put relevant chunks up top (retrieval signal). The most dangerous failure — a fluent answer that invents a fact — is invisible to "is this good?" metrics, so you pair the semantic metric with a cheap deterministic tripwire that can run in production ([../lectures/08-rag-eval-hallucination.md](../lectures/08-rag-eval-hallucination.md)).

**Do it (RAG path):**
```python
# evals/run_ragas.py   (requires golden cases with contexts + ground_truth)
import json, pathlib
from dotenv import load_dotenv; load_dotenv()
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

rows = [json.loads(l) for l in pathlib.Path("evals/data/golden_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
ds = Dataset.from_list([{
    "question":     r["input"],
    "answer":       r.get("output") or r.get("expected", ""),
    "contexts":     r.get("context") or r.get("contexts") or [],   # list[str] of retrieved chunks
    "ground_truth": r.get("expected", ""),
} for r in rows if (r.get("context") or r.get("contexts"))])

result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)   # e.g. {'faithfulness': 0.83, 'answer_relevancy': 0.91, 'context_precision': 0.77}
```
```bash
uv run python -m evals.run_ragas
```
RAGAS uses an LLM under the hood — point it at your judge model. For the free path, configure RAGAS with an Ollama LLM + a local embedding model (`ragas`'s `evaluate(..., llm=..., embeddings=...)`); see the RAGAS docs for the wrapper. Expect it to be slower on CPU.

**Do it (tripwire — everyone, RAG or not):** a small NLI cross-encoder checks whether the context *entails* the answer. Low entailment ⇒ likely unsupported ⇒ trip.
```python
# src/tripwire.py
from sentence_transformers import CrossEncoder
_MODEL = None
def _model():
    global _MODEL
    if _MODEL is None:
        # labels: [contradiction, entailment, neutral]
        _MODEL = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    return _MODEL

def hallucination_tripwire(context: str, answer: str, entail_threshold: float = 0.5) -> dict:
    """Returns {'tripped': bool, 'entailment': float}. Tripped = answer NOT entailed by context."""
    import numpy as np
    scores = _model().predict([(context, answer)])          # logits over 3 classes
    probs = np.exp(scores[0]) / np.exp(scores[0]).sum()
    entailment = float(probs[1])
    return {"tripped": entailment < entail_threshold, "entailment": round(entailment, 3)}
```
Prove it fires on a **known-bad** case (answer contradicting/ungrounded in the context):
```python
# evals/run_tripwire.py
from src.tripwire import hallucination_tripwire
good = hallucination_tripwire("Paris is the capital of France.", "The capital of France is Paris.")
bad  = hallucination_tripwire("Paris is the capital of France.", "The capital of France is Berlin.")
print("good:", good)   # tripped=False, high entailment
print("bad :", bad)    # tripped=True,  low entailment  <-- must flag >=1 known-bad case
assert bad["tripped"], "tripwire failed to flag a known hallucination"
print("TRIPWIRE OK: flagged the known-bad case")
```
```bash
uv run python -m evals.run_tripwire
```

For a **SelfCheckGPT-style** alternative (no NLI model, works for non-RAG generation): sample the answer 3× at temperature > 0 and flag low mutual consistency (e.g., pairwise embedding similarity below a threshold) — an unstable answer is likely fabricated.

**Expected result:** RAG path → a dict of three scores in [0,1]. Everyone → the tripwire returns `tripped=True` on the known-bad case (`assert` passes) and `tripped=False` on the grounded one.

**Verify:** RAG — sanity-check that a case you *know* has an ungrounded answer has low `faithfulness`. Tripwire — the `assert` in `run_tripwire.py` succeeds, satisfying "flags ≥1 known-bad case."

**Troubleshoot:**
- No `contexts` in your golden set → you can't run RAGAS faithfulness/precision. Either add retrieved chunks to the cases, or (non-RAG system) skip RAGAS and satisfy the DoD via the **tripwire requirement** alone.
- `KeyError`/`ValidationError` on the columns above → you have **RAGAS ≥ 0.2**, which renamed the dataset fields. Map `question→user_input`, `answer→response`, `contexts→retrieved_contexts`, `ground_truth→reference`, and build an `EvaluationDataset` (`from ragas import EvaluationDataset; EvaluationDataset.from_list([...])`). The metric names (`faithfulness`, `answer_relevancy`, `context_precision`) are unchanged. If you prefer the code above verbatim, pin the old API with `uv add "ragas<0.2"`.
- RAGAS `evaluate` errors on OpenAI rate/creds → set the LLM explicitly and lower concurrency; on the free path wire an Ollama LLM + local embeddings.
- CrossEncoder download slow/offline → it's a one-time ~500 MB download; pre-fetch on a connected machine or cache under `HF_HOME`.
- NLI on very long context → truncate or chunk the context per claim; the base model has a modest max length.

---

### Step 5 — Statistical rigor: bootstrap CI + paired McNemar

**WHAT:** Write `evals/stats.py` with `bootstrap_ci(scores)` and a paired `mcnemar_pvalue(a_correct, b_correct)`. Run your system twice over the golden set — **baseline** vs a **tweaked prompt** — print each accuracy **with its 95% CI** and the **paired p-value**, then explain why a 2% delta at n=50 is not shippable.

**WHY:** A single accuracy is a point estimate with noise; reporting it alone is like quoting a stock price to eight decimals. The bootstrap gives an error bar with no distributional assumptions. Because A and B are evaluated on the *same* cases, you must use a **paired** test (McNemar for pass/fail) — it conditions on the cases where the two systems *differ*, extracting far more signal than an unpaired test, which throws away power and can hide a real win ([../lectures/09-statistical-rigor.md](../lectures/09-statistical-rigor.md)).

**Do it:**
```python
# evals/stats.py
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def bootstrap_ci(scores, n: int = 10000, alpha: float = 0.05, seed: int = 0):
    """95% CI for the mean of per-case scores (0/1 pass or 0-1 score) via resampling."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    means = rng.choice(scores, size=(n, len(scores)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(scores.mean()), float(lo), float(hi)

def mcnemar_pvalue(a_correct, b_correct) -> float:
    """Paired test for A vs B on the SAME cases (binary pass/fail)."""
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)   # A wrong, B right
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)   # A right, B wrong
    # exact binomial for small discordant counts; table diagonal is ignored by McNemar
    return float(mcnemar([[0, b01], [b10, 0]], exact=True).pvalue)

def paired_bootstrap_diff_ci(a_scores, b_scores, n: int = 10000, alpha: float = 0.05, seed: int = 0):
    """95% CI for the paired mean difference (B - A). If it straddles 0, no proven difference."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a_scores, float), np.asarray(b_scores, float)
    idx = rng.integers(0, len(a), size=(n, len(a)))     # resample CASES, keep pairs together
    diffs = (b[idx] - a[idx]).mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float((b - a).mean()), float(lo), float(hi)
```

Now the A/B run. Score each golden case with your judge under two prompts (or two system variants):
```python
# evals/run_ab.py
import json, pathlib
from dotenv import load_dotenv; load_dotenv()
from src.judge import judge
from evals.stats import bootstrap_ci, mcnemar_pvalue, paired_bootstrap_diff_ci

rows = [json.loads(l) for l in pathlib.Path("evals/data/golden_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

def run_variant(output_key):
    """output_key selects which precomputed system output to grade, e.g. 'output_baseline' / 'output_tweak'.
       (Generate these by running YOUR system under each prompt and storing outputs per case.)"""
    correct = []
    for r in rows:
        out = r.get(output_key) or r.get("output") or r.get("expected", "")
        v = judge(r["input"], out, r.get("criteria", "Correct and directly answers the input."))
        correct.append(int(v.pass_))
    return correct

a = run_variant("output_baseline")
b = run_variant("output_tweak")

acc_a, lo_a, hi_a = bootstrap_ci(a)
acc_b, lo_b, hi_b = bootstrap_ci(b)
p = mcnemar_pvalue(a, b)
d, dlo, dhi = paired_bootstrap_diff_ci(a, b)

print(f"n = {len(rows)}")
print(f"baseline: {acc_a:.1%}  95% CI [{lo_a:.1%}, {hi_a:.1%}]")
print(f"tweak:    {acc_b:.1%}  95% CI [{lo_b:.1%}, {hi_b:.1%}]")
print(f"paired diff (tweak - baseline): {d:+.1%}  95% CI [{dlo:+.1%}, {dhi:+.1%}]")
print(f"McNemar paired p-value: {p:.3f}")
print("SHIP" if (p < 0.05 and dlo > 0) else "DO NOT SHIP — difference not proven")
```
```bash
uv run python -m evals.run_ab
```

**Expected result (illustrative, n=50):**
```
n = 50
baseline: 82.0%  95% CI [70.0%, 92.0%]
tweak:    84.0%  95% CI [72.0%, 94.0%]
paired diff (tweak - baseline): +2.0%  95% CI [-6.0%, +10.0%]
McNemar paired p-value: 0.687
DO NOT SHIP — difference not proven
```

**Why a 2% delta at n=50 is not shippable — the explanation you must be able to give:** At n=50 the bootstrap CI on each accuracy spans roughly ±12 points, and the paired difference CI *straddles zero* — so the true effect is consistent with anything from "the tweak is worse" to "modestly better." The McNemar p-value (~0.69) says the discordant cases (where the two prompts disagreed) are far too few to reject "no difference." A CI that includes zero means **"no proven difference," full stop** — you are not allowed to peek at the point estimate and declare victory. To *see* a real 2% effect you would need on the order of hundreds to a few thousand cases; on 50 you are inside the noise.

**Verify:** Confirm the diff CI includes 0 and p > 0.05 for a small delta; then artificially widen the gap (flip a few `b` outputs to correct) and watch p drop below 0.05 and the diff CI clear zero — proving the harness reacts correctly.

**Troubleshoot:**
- Used an unpaired t-test / chi-square instead of McNemar → wrong tool for paired data; it discards the pairing and loses power. Use `mcnemar_pvalue` here.
- p-value is `nan` or 1.0 with tiny discordant counts → expected when almost no cases differ; report it honestly as "no signal," don't hunt for a different test.
- CI looks implausibly narrow → you probably passed *summary* accuracies instead of the per-case 0/1 list. `bootstrap_ci` needs the raw per-case scores.
- Bootstrap non-deterministic across runs → seed it (as above) so CI is reproducible in CI/CD.

---

## Putting it together — short end-to-end run

Run the whole Week-2 pipeline top to bottom:
```bash
# 1. judge returns valid Verdicts on 20/20
uv run python -m evals.smoke_judge
# 2. pairwise + position-disagreement rate
uv run python -m evals.run_pairwise
# 3. calibrate: BEFORE kappa -> revise rubric -> AFTER kappa (logged)
uv run python -m evals.calibration
# 4. RAG metrics (if RAG) AND/OR hallucination tripwire fires on a known-bad case
uv run python -m evals.run_ragas      # RAG systems only
uv run python -m evals.run_tripwire
# 5. A/B with 95% CIs + paired p-value + shippability verdict
uv run python -m evals.run_ab
```
Artifacts produced: `src/judge.py` (rubric + pairwise), `src/tripwire.py`, `evals/stats.py`, `evals/data/human_labels.jsonl`, `evals/data/calibration_log.md`. Commit them: `git add src evals && git commit -m "week2: calibrated de-biased judge + stats harness"`.

---

## Definition of Done — verifiable checks

Restating the spine's acceptance gate; each is checkable with the commands above:

- [ ] **Valid Verdicts:** the judge returns a well-formed `Verdict` (reasoning **first**, score **1–4**, pass/fail) for **20/20** sampled cases — `smoke_judge` prints `20/20`.
- [ ] **Position bias reported:** pairwise mode runs **both** orderings, counts a win only on cross-order consistency, and reports a **position-disagreement rate** — `run_pairwise` prints it.
- [ ] **Kappa before/after, improved:** Cohen's kappa vs human labels is computed, the rubric was revised, and both **before and after** are logged in `calibration_log.md` with after > before and **≥ 0.6**.
- [ ] **RAG metrics OR firing tripwire:** RAGAS **faithfulness + answer_relevancy + context_precision** reported (RAG systems), **OR** the hallucination tripwire flags **≥1 known-bad case** (`run_tripwire` assert passes).
- [ ] **CI + paired p-value + correct call:** `run_ab` prints each accuracy with its **95% CI** and the **paired McNemar p-value**, and you can explain why a **2% delta at n=50 is not shippable** (CI straddles 0 / p not < 0.05).

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `instructor` parse/validation errors (esp. Ollama) | small model wraps JSON in prose | `mode=instructor.Mode.JSON`, `max_retries>=2`, temp 0; try `qwen2.5:7b-instruct` |
| Scores feel arbitrary / cluster at 3 | scale too fine or criteria vague | keep 1–4 (never 1–100); enumerate concrete pass conditions in the rubric |
| Judge loves your own model's answers | **self-enhancement** — judge family == generator family | switch judge family (Claude↔GPT, or Llama judging both) |
| Pairwise flips on swap ~50% of the time | judge too weak or pairs too close | stronger judge / sharpen criteria; report the rate honestly |
| Kappa low after edits | systematic directional bias | adjust pass threshold (score≥3 vs ≥2), not just wording |
| Kappa = `nan` | one rater all-pass or all-fail | re-sample to include both classes |
| RAGAS errors / no scores | missing `contexts`/`ground_truth` or LLM creds | add retrieved chunks; set RAGAS llm+embeddings (Ollama on free path) |
| p-value `nan`/1.0 | almost no discordant cases | honest "no signal"; do not switch to a test that fakes significance |
| CI implausibly narrow | passed summary accuracy, not per-case list | feed raw per-case 0/1 scores to `bootstrap_ci` |
| Used t-test for A/B on same cases | **unpaired test on paired data** | use `mcnemar_pvalue` / `paired_bootstrap_diff_ci` |
| "84% > 82%, ship it!" | peeking at point estimate | if diff CI includes 0 → "no proven difference," full stop |

---

## Stretch goals (optional)

- **Reference-guided mode:** add a third judge mode that receives a gold answer alongside the criteria; compare its kappa to the pure-rubric judge.
- **Verbosity-bias probe:** pad a batch of correct answers with irrelevant filler and confirm the length-ignoring rubric doesn't raise their scores — quantify the residual verbosity effect.
- **Power analysis:** given your observed pass rates, compute the n needed to detect a true 2% / 5% / 10% effect at 80% power; use it to set golden-set size targets.
- **SelfCheckGPT tripwire:** implement the sample-3×-and-measure-consistency variant and compare its catch rate to the NLI tripwire on your known-bad cases.
- **Judge cost/latency budget:** time and price a 50-case judge run; this feeds Week 3's decision to run the judge only on changed prompts / nightly in CI.
- **Wire into CI (preview of Week 3):** expose `run_ab`'s verdict as a `pytest` that asserts `p < 0.05 and diff_lo > 0` before allowing a "this is better" claim to merge — see [../lectures/14-ci-eval-gate.md](../lectures/14-ci-eval-gate.md).
