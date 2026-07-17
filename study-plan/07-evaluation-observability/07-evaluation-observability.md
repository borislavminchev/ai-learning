# Phase 7 — Evaluation, Testing & Observability

Evals are the steering wheel of every LLM system. Without a trustworthy measurement loop you cannot ship, iterate, or defend a single change against "but it looked good in the playground." This phase turns the evals-first mindset you've carried since Phase 1 into a rigorous, automated discipline: golden sets mined from real traces, calibrated LLM judges you don't blindly trust, statistical rigor that refuses to celebrate a 2% delta on 50 examples, and an observability + data-flywheel stack that catches silent regressions in production.

Prev: [06-agents.md](../06-agents/06-agents.md) · Next: [08-fine-tuning.md](../08-fine-tuning/08-fine-tuning.md)

## Prerequisites
- Phases 1–6: you can build a prompt, a structured-output extractor (Pydantic/instructor), a RAG pipeline, and a tool-using agent. You have at least one of these running that we can evaluate.
- Python 3.11+ with `uv` (Phase 0 default), comfort with `pytest`, and a working provider key (OpenAI/Anthropic/Gemini) **or** a local Ollama install for cost-free judge/model calls.
- Basic Docker (for running Langfuse/Phoenix locally) and a GitHub repo with Actions enabled.
- You do **not** need statistics beyond high-school level — we teach the engineering-relevant intuition (bootstrap, paired tests) and hand the math to `scipy`/`statsmodels`.

## Time budget
3 weeks × ~10–15 hrs/week (~35–40 hrs total). Roughly 35% theory / 65% hands-on. Each week states its own hours breakdown.

## How to use this file
Do the weeks in order — each builds a layer of the milestone (Week 1 = golden set + error analysis, Week 2 = judges + statistical rigor, Week 3 = observability + CI gate). Timebox theory; the labs are the point. If you fall behind, cut theory reading first and keep the build moving — you learn evals by shipping an eval, not by reading about one.

> **📚 Course materials.** This file is the **spine** — a weekly plan and a *recap*. Deep material lives alongside it:
> - **Lectures**: [`lectures/`](lectures/00-index.md) — read for the *why*/mechanism.
> - **Lab guides**: [`labs/`](labs/) — follow to *build*.
>
> Each week: read the linked lectures first (the Theory here is their recap), then work the linked lab guide.

---

## Week 1 — Eval-driven development: traces, error analysis & golden sets

**Hours breakdown:** ~4 hrs theory / ~8 hrs lab (12 total). Bump to 15 if your system's traces are messy.

### Objectives
By the end of the week you can:
1. Explain the eval-driven loop (trace → error-analysis → failure-taxonomy → test-cases → change one thing → re-run) and why building the eval *before* optimizing is non-negotiable.
2. Collect real (or realistic) traces from one of your Phase 1–6 systems and do **open-coding error analysis** on 30–50 of them.
3. Turn that analysis into a written **failure taxonomy** (5–10 named failure modes with counts).
4. Build a **stratified, versioned golden set** of 50–100 cases stored as JSONL in git, decontaminated against anything you'll use as few-shot examples.
5. Distinguish reference-based vs criteria-based vs human eval and pick the right one per case.

### Theory (~4 hrs)
> **📖 Deep lectures for this week** (read first): [1 · Eval-Driven Development & the Goal vs Guardrail Split](lectures/01-eval-driven-development.md) · [2 · Error Analysis: Raw Traces to a Failure Taxonomy](lectures/02-error-analysis-to-taxonomy.md) · [3 · The Three Eval Families](lectures/03-three-eval-families.md) · [4 · Building Golden Sets Done Right](lectures/04-golden-sets-done-right.md). The bullets below are the recap.

- **Eval-driven development.** Read Hamel Husain's "Your AI Product Needs Evals" and his "Creating a LLM-as-a-Judge That Drives Business Value" (search: `Hamel Husain evals` — his blog `hamel.dev` is the canonical practitioner reference). Internalize: most teams' #1 mistake is optimizing prompts before they can measure. The loop is: *look at your data* → categorize failures → encode as tests → change one variable → re-measure.
- **The three eval families.** Reference-based (compare to a gold answer — good for extraction/classification, weak for open-ended text). Criteria-based (a rubric scores an output with no single right answer — the workhorse for generative tasks). Human (the ground truth you calibrate everything else against, and expensive). Skim the DeepEval and RAGAS docs' "metrics" overviews to see how each tool frames this.
- **Golden sets done right.** Stratified (cover the real distribution of inputs *and* the hard/rare cases, not just happy paths), versioned (a golden set is a git-tracked artifact like code), grown by mining production failures, and **decontaminated** (never let a case leak into your few-shot exemplars or fine-tune data — that inflates scores and fools you). Read the "Evals" chapter framing in Chip Huyen's *AI Engineering* (O'Reilly, 2024) for the mental model.
- **Guardrail vs goal metrics.** Goal = the thing you're trying to improve (answer accuracy). Guardrail = the things you must not regress (cost, p95 latency, refusal rate, toxicity). A "win" that blows the guardrail is not a win.

### Lab (~8 hrs)
> **🛠️ Full step-by-step guide:** [Traces to Golden Set: Error Analysis and a Versioned Eval Corpus](labs/week-1-golden-set-and-error-analysis.md). The steps below are the summary.

Pick ONE system you already built (RAG bot, extraction service, or agent). We'll evaluate it end to end.

**1. Project scaffold.**
```bash
uv init llm-evals && cd llm-evals
uv add pandas pydantic python-dotenv rich
uv add --dev pytest
mkdir -p evals/{data,traces,taxonomy} src
```
Layout:
```
llm-evals/
  src/system_under_test.py   # thin wrapper that calls YOUR Phase 1-6 system
  evals/
    traces/raw_traces.jsonl  # collected inputs+outputs
    data/golden_v1.jsonl     # your versioned golden set
    taxonomy/failures.md      # the written failure taxonomy
    error_analysis.py         # helper to sample & annotate
  .env
```

**2. Collect traces.** Wrap your system so every call appends a JSONL line `{id, ts, input, output, context?, meta}`. If you lack real traffic, generate 60–80 realistic inputs (vary difficulty, length, edge cases, adversarial phrasings) and run them through. Target ≥50 traces.

**3. Open-coding error analysis.** Sample 40 traces. For each, read input+output and write a free-text note on what's wrong (or "OK"). Then cluster your notes into named failure modes. A minimal annotation helper:
```python
# evals/error_analysis.py
import json, pathlib, random
rows = [json.loads(l) for l in pathlib.Path("evals/traces/raw_traces.jsonl").read_text().splitlines()]
for r in random.sample(rows, 40):
    print("\n== id:", r["id"]); print("INPUT:", r["input"][:500]); print("OUTPUT:", r["output"][:500])
    note = input("failure note (or OK): ")
    r["annotation"] = note
pathlib.Path("evals/traces/annotated.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
```

**4. Write the taxonomy.** In `evals/taxonomy/failures.md`, list each failure mode with a definition, a real example, and a count (e.g. `hallucinated_citation: 7`, `format_drift: 4`, `wrong_field_extracted: 9`, `unhelpful_refusal: 3`). Rank by frequency — this is your prioritized backlog.

**5. Build the golden set.** Convert your annotated traces into 50–100 cases in `evals/data/golden_v1.jsonl`. Each case: `{id, input, expected? , criteria?, stratum, source}`. Stratify with a `stratum` tag (e.g. `easy|hard|adversarial|rare_entity`) and ensure each named failure mode is represented by ≥3 cases. Add a `source` field (`prod` vs `synthetic`) so you can weight later. Commit it: `git add evals/data/golden_v1.jsonl && git commit -m "golden set v1"`.

**6. Decontamination check.** Write a tiny script that asserts no golden-set input string appears in your few-shot exemplars or system prompt. Fail loudly if it does.

### Definition of Done
- [ ] `raw_traces.jsonl` has ≥50 real/realistic traces with inputs and outputs.
- [ ] `failures.md` lists ≥5 named failure modes, each with a definition, an example, and a count.
- [ ] `golden_v1.jsonl` has 50–100 cases, each tagged with a `stratum`, and every failure mode has ≥3 cases.
- [ ] Golden set is committed to git with a version in the filename/tag.
- [ ] Decontamination script runs green (no golden input leaks into prompts/exemplars).
- [ ] You can state, in one sentence, the single most frequent failure mode and the metric you'll use to track it.

### Pitfalls
- **Skipping the "look at your data" step.** Jumping straight to a fancy metric without reading 40 outputs by hand is the classic failure. The taxonomy only comes from actually reading.
- **Only-happy-path golden sets.** If every case is easy, your eval will show 98% forever and catch nothing. Deliberately over-sample hard and adversarial cases.
- **Contamination.** Reusing golden inputs as few-shot examples silently inflates scores. Keep them physically separate and assert it.
- **Unversioned golden sets.** Editing the set in place makes historical scores incomparable. Bump to `golden_v2.jsonl` and note what changed.
- **Golden set too big to maintain.** 50–100 well-chosen stratified cases beat 5,000 random ones you never look at.

### Self-check
1. Why must the eval exist before you optimize the prompt? What goes wrong if you optimize first?
2. Give a task where reference-based eval is right and one where criteria-based is the only option.
3. What does "stratified" mean for a golden set, and why does it matter more than raw size?
4. How does golden-set contamination inflate your scores, and how do you prevent it?

---

## Week 2 — LLM-as-judge, its biases, and statistical rigor

**Hours breakdown:** ~5 hrs theory / ~9 hrs lab (14 total). This is the densest week.

### Objectives
By the end of the week you can:
1. Build an **LLM-as-judge** in all three modes (rubric/absolute, pairwise, reference-guided) with CoT-before-verdict and low-cardinality structured output.
2. Detect and mitigate the three core judge biases (position, verbosity, self-enhancement).
3. **Calibrate** the judge against human labels and report agreement with Cohen's kappa.
4. Run **RAG-specific eval** (faithfulness, context relevance, answer relevance) with RAGAS, and add a hallucination tripwire (NLI/SelfCheckGPT-style).
5. Apply **statistical rigor**: bootstrap confidence intervals and a paired significance test, so you never call a 2% delta on 50 examples a win.

### Theory (~5 hrs)
> **📖 Deep lectures for this week** (read first): [5 · LLM-as-Judge Mechanics](lectures/05-llm-as-judge-mechanics.md) · [6 · Judge Biases and Their Mitigations](lectures/06-judge-biases.md) · [7 · Calibrating the Judge with Cohen's Kappa](lectures/07-judge-calibration-kappa.md) · [8 · RAG Evaluation & Hallucination Detection](lectures/08-rag-eval-hallucination.md) · [9 · Statistical Rigor: Bootstrap CIs & Paired Tests](lectures/09-statistical-rigor.md). The bullets below are the recap.

- **LLM-as-judge mechanics.** Read the RAGAS docs and DeepEval's LLM-judge/G-Eval docs. Key rules: use a strong judge model; make it reason *before* emitting the verdict; use a **low-cardinality scale** (pass/fail or 1–4, never 1–100 — LLMs can't discriminate that finely); return structured output (Pydantic) so you get a machine-readable score + rationale. Skim the original G-Eval and "LLM-as-a-Judge" (MT-Bench / Zheng et al.) framing — search `LLM-as-a-judge MT-Bench` — but only for the *intuition*, not the math.
- **Judge biases (the engineering-critical part).** **Position bias**: in pairwise, the judge favors whichever answer is first — mitigate by swapping order and averaging both runs. **Verbosity bias**: longer answers score higher regardless of quality — control for length or instruct the rubric to ignore it. **Self-enhancement bias**: a model rates its own family's outputs higher — use a *different* model family as judge than the one generating.
- **Calibration & Cohen's kappa.** A judge is only trustworthy if it agrees with humans. Label ~30–50 cases yourself, run the judge on the same cases, and compute **Cohen's kappa** (agreement corrected for chance; >0.6 is decent, >0.8 is strong). If kappa is low, fix the rubric and re-measure — meta-evaluate the judge like any other component.
- **RAG eval & hallucination detection.** Faithfulness (is the answer grounded in retrieved context?), context relevance/precision, answer relevance — RAGAS computes these. Hallucination detection via NLI/entailment (does the context *entail* each claim?) or SelfCheckGPT (sample N answers, low consistency ⇒ likely fabricated). Pair a cheap deterministic tripwire with the semantic/judge metric.
- **Statistical rigor.** Read enough to internalize: a single accuracy number is a point estimate with noise. **Bootstrap** resampling gives you a confidence interval. For A vs B on the *same* cases, use a **paired** test (McNemar for binary, paired bootstrap for scores) — paired is far more powerful than treating them as independent. Rule of thumb: a 2% delta on 50 examples is almost always inside the noise. `scipy.stats` and `statsmodels` have everything.

### Lab (~9 hrs)
> **🛠️ Full step-by-step guide:** [A Calibrated, De-biased Judge with Statistical Rigor](labs/week-2-calibrated-judge-and-stats.md). The steps below are the summary.

Continue in the `llm-evals` repo.

**1. Install.**
```bash
uv add ragas deepeval scipy statsmodels numpy
# free/cheap judge option: use Ollama locally instead of a paid API
# ollama pull llama3.1:8b   (judge quality is lower but the pipeline is identical)
```

**2. Build a rubric judge with structured output.**
```python
# src/judge.py
from pydantic import BaseModel
class Verdict(BaseModel):
    reasoning: str          # CoT FIRST
    score: int              # 1..4, low cardinality
    pass_: bool
JUDGE_PROMPT = """You are a strict evaluator. Given INPUT, OUTPUT, and CRITERIA,
first reason step by step, then give a score 1-4 and pass/fail.
Ignore answer length; judge only correctness and adherence to criteria."""
# call your provider (or Ollama's OpenAI-compatible endpoint at http://localhost:11434/v1)
# with response_format=Verdict via instructor. Return the parsed Verdict.
```

**3. Add pairwise mode with position-bias control.** Run each comparison twice (A,B) and (B,A); only count it as a win if the judge is consistent across both orders. Log the disagreement rate — that number *is* your position-bias measurement.

**4. Calibrate against humans.** Hand-label 30–50 golden cases pass/fail (reuse Week 1's annotations where possible). Run the judge on the same cases. Compute kappa:
```python
from sklearn.metrics import cohen_kappa_score  # uv add scikit-learn
kappa = cohen_kappa_score(human_labels, judge_labels)
```
If kappa < 0.6, revise the rubric (usually the criteria are vague) and re-run. Record before/after kappa.

**5. RAG metrics (if your system is RAG; else skip to 6).**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
# build a ragas dataset {question, answer, contexts, ground_truth} from golden_v1
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)
```
Add a cheap NLI tripwire: for each answer claim, check entailment against context using a small NLI model (`cross-encoder/nli-deberta-v3-base` via sentence-transformers) or a SelfCheckGPT-style consistency check (sample the answer 3× at temp>0, flag low agreement).

**6. Statistical rigor harness.** Write `evals/stats.py`:
```python
import numpy as np
def bootstrap_ci(scores, n=10000, alpha=0.05):
    scores = np.asarray(scores)
    means = [np.random.choice(scores, len(scores), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])

from statsmodels.stats.contingency_tables import mcnemar
def paired_binary(a_correct, b_correct):   # McNemar for A vs B on same cases
    b01 = sum(1 for a,b in zip(a_correct,b_correct) if not a and b)
    b10 = sum(1 for a,b in zip(a_correct,b_correct) if a and not b)
    return mcnemar([[0,b01],[b10,0]], exact=True).pvalue
```
Run your system twice (baseline vs a tweaked prompt) over the golden set, print each accuracy **with its 95% CI**, and the paired p-value. Observe how wide the CI is at n=50.

### Definition of Done
- [ ] Judge returns a valid `Verdict` (reasoning-first, score 1–4, pass/fail) for 20/20 sampled cases.
- [ ] Pairwise mode runs both orderings and reports a position-disagreement rate.
- [ ] Cohen's kappa vs human labels is computed and recorded; you improved it by revising the rubric (before/after both logged).
- [ ] RAGAS faithfulness + answer relevance + context precision reported on the golden set (RAG systems), OR a hallucination tripwire flags ≥1 known-bad case.
- [ ] `bootstrap_ci` prints a 95% CI for accuracy and a paired p-value for baseline vs tweak; you can explain why a 2% delta at n=50 is not shippable.

### Pitfalls
- **Judge on a 1–100 scale.** It looks precise and is pure noise. Use 1–4 or pass/fail.
- **Verdict before reasoning.** If the model emits the score first, the "reasoning" is post-hoc rationalization. Force CoT first in the schema field order.
- **Self-enhancement.** Judging GPT-4o outputs with GPT-4o inflates scores. Use a different family (e.g. Claude judging GPT, or Llama judging both).
- **Trusting an uncalibrated judge.** A judge with kappa 0.3 vs humans is a random number generator with good grammar. Always calibrate first.
- **Independent tests on paired data.** Using an unpaired t-test for A vs B on the same cases throws away power and can hide real wins. Use McNemar/paired bootstrap.
- **Peeking at CIs to declare victory.** A CI that includes zero difference means "no proven difference," full stop.

### Self-check
1. Name the three judge biases and the concrete mitigation for each.
2. What does Cohen's kappa correct for that raw agreement percentage does not?
3. Why is faithfulness measured separately from answer relevance in RAG?
4. You see baseline 82% vs new prompt 84% on 50 cases. What two numbers do you compute before shipping, and what would convince you it's real?
5. Why prefer a paired test over an independent one when comparing two prompts on the same golden set?

---

## Week 3 — Observability, the data flywheel & a CI eval gate

**Hours breakdown:** ~4 hrs theory / ~9 hrs lab (13 total). Bump to 15 to polish the milestone.

### Objectives
By the end of the week you can:
1. Instrument an LLM/RAG/agent app with **OpenTelemetry GenAI**-convention tracing (via OpenLLMetry or a platform SDK) capturing per-span prompt, tokens, cost, and latency.
2. Stand up a local **observability backend** (Langfuse or Arize Phoenix in Docker) and view a cost/latency/quality dashboard.
3. Capture user **feedback** (thumbs/regenerate) linked to trace IDs and build a **data flywheel** query that auto-routes low-rated traces into a review queue that grows the golden set.
4. Detect **drift / silent model swaps** and run **sampled online evals** on production traffic.
5. Wire a **CI eval gate** (promptfoo or pytest + your judge) into GitHub Actions that blocks a merge when a deliberately-worse prompt regresses accuracy or p95 cost.

### Theory (~4 hrs)
> **📖 Deep lectures for this week** (read first): [10 · OpenTelemetry GenAI Tracing with OpenLLMetry](lectures/10-otel-genai-openllmetry.md) · [11 · Observability Platforms: Langfuse, Phoenix & Hosted Options](lectures/11-observability-platforms.md) · [12 · Feedback Capture and the Data Flywheel](lectures/12-feedback-and-flywheel.md) · [13 · Drift, Silent Model Swaps & Sampled Online Evals](lectures/13-drift-online-eval.md) · [14 · CI Eval Gates with promptfoo and pytest](lectures/14-ci-eval-gate.md). The bullets below are the recap.

- **OpenTelemetry GenAI & OpenLLMetry.** Read the OpenTelemetry "GenAI semantic conventions" docs and the OpenLLMetry (Traceloop) README on GitHub. Understand spans as a tree: one root request span with child spans per LLM call / retrieval / tool call, each carrying attributes (`gen_ai.request.model`, prompt/completion, token counts, cost, latency). Standard conventions mean your traces are portable across backends.
- **The observability platforms.** Skim docs for **Langfuse** (open-source, self-hostable, strong on prompt management + evals), **Arize Phoenix** (OSS, OTel-native, great local dev), and **LangSmith** / **Braintrust** (hosted, polished eval + experiment tracking). Opinionated default: **Langfuse or Phoenix self-hosted** for learning and privacy; reach for Braintrust/LangSmith when a team needs hosted experiment tracking. All ingest OTel.
- **Feedback & the data flywheel.** Explicit signals (thumbs up/down, edits) and implicit ones (regeneration, copy, dwell, edit-distance from the shown answer) are your quality proxies. Link every signal to a trace ID, redact PII at write time, and periodically pull low-rated traces into your review queue → golden set → next eval run. This loop is the whole reason to instrument.
- **Drift & silent model swaps.** Providers change models behind `latest` aliases; input distributions shift. Pin model *snapshots* in code, tag the resolved model on every trace, and run **sampled online evals** (judge a random 1–5% of prod traffic) with alerting on quality/cost drift. Read the monitoring sections of the Langfuse and Phoenix docs.
- **CI eval gates.** Read the promptfoo docs (`promptfoo` — `assert` types, `providers`, CI usage) and think of prompts/models as versioned artifacts with pass thresholds, exactly like unit tests. Pin model snapshots so the gate isn't flaky.

### Lab (~9 hrs)
> **🛠️ Full step-by-step guide:** [Milestone: Observability, Data Flywheel, and a Blocking CI Eval Gate](labs/week-3-observability-flywheel-ci-gate.md). The steps below are the summary.

**1. Stand up a backend (pick one).**
```bash
# Langfuse (self-hosted)
git clone https://github.com/langfuse/langfuse && cd langfuse && docker compose up -d   # http://localhost:3000
# OR Arize Phoenix (lighter, single container)
uv add arize-phoenix && python -m phoenix.server.main serve   # http://localhost:6006
```

**2. Instrument with OpenLLMetry (vendor-neutral OTel).**
```bash
uv add traceloop-sdk openai
```
```python
from traceloop.sdk import Traceloop
Traceloop.init(app_name="llm-evals")   # points at your OTLP endpoint (Phoenix/Langfuse)
# now normal OpenAI/Anthropic calls are auto-traced with tokens, cost, latency
```
Run 30+ requests through your system. Open the dashboard and confirm you see a span tree with per-call tokens and latency. Build/verify a view showing p50/p95 latency, total cost, and token counts per request.

**3. Feedback API + flywheel.** Add a tiny FastAPI endpoint:
```python
# uv add "fastapi[standard]"
@app.post("/feedback")
def feedback(trace_id: str, rating: int, comment: str = ""):
    # store {trace_id, rating, comment, ts}; redact PII from comment before write
    ...
```
Write a nightly-style script `evals/flywheel.py` that queries traces with `rating <= 2` (or implicit regenerate signals), redacts PII, and appends them to `evals/data/review_queue.jsonl`. Manually promote a few into `golden_v2.jsonl` — proving the loop closes.

**4. Sampled online eval + drift check.** Add a hook that judges a random 5% of production traces with your Week-2 judge and logs the score as a span attribute. Add an assertion/alert (log a WARNING, or post to a Slack webhook if you have one) when the rolling online score drops below a threshold or when the resolved `gen_ai.request.model` differs from the pinned snapshot (silent-swap detection).

**5. The CI eval gate (milestone core).** Create `promptfooconfig.yaml`:
```yaml
prompts: [file://prompts/current.txt]
providers: [openai:gpt-4o-2024-11-20]     # PIN the snapshot
tests: file://evals/data/golden_v1.jsonl  # convert to promptfoo test format
defaultTest:
  assert:
    - type: llm-rubric
      value: "Answer is correct and grounded per the criteria field"
    - type: cost
      threshold: 0.01      # per-call cost guardrail
    - type: latency
      threshold: 5000
```
```bash
uv add --dev promptfoo   # or: npx promptfoo@latest eval
npx promptfoo eval -c promptfooconfig.yaml --output results.json
```
Then GitHub Actions `.github/workflows/eval-gate.yml`:
```yaml
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npx promptfoo@latest eval -c promptfooconfig.yaml --share=false
      # promptfoo exits non-zero when assertions fail -> blocks the merge
```
(Prefer a pure-Python gate? Use `pytest` that loads the golden set, runs your judge, computes accuracy + bootstrap CI, and `assert accuracy >= THRESHOLD and p95_cost <= BUDGET`.)

**6. Prove it blocks a bad prompt.** Open a PR that swaps in a deliberately-worse prompt (e.g. strip the output-format instructions or the "say I don't know" rule). Watch the Action fail and block the merge. Screenshot it — that's your milestone evidence.

### Definition of Done
- [ ] Dashboard shows a span tree with per-step tokens, cost, and latency for ≥30 requests.
- [ ] `/feedback` endpoint stores ratings linked to trace IDs with PII redaction; `flywheel.py` produces a `review_queue.jsonl` of low-rated traces.
- [ ] At least one low-rated trace has been promoted into `golden_v2.jsonl` (the loop demonstrably closes).
- [ ] Sampled online eval logs a judge score on ≥5% of traces; a drift/silent-swap check fires a WARNING/alert when triggered (test it by changing the pinned model).
- [ ] CI eval gate runs on every PR against the pinned model snapshot.
- [ ] A PR containing a deliberately-worse prompt is **blocked** by the gate (screenshot/logs saved); the baseline prompt PR passes.

### Pitfalls
- **Logging raw PII into traces.** Prompts and outputs often contain user data. Redact at write time (Presidio or a regex pass) — traces are a data-governance surface.
- **Un-pinned models in the gate.** If CI hits `gpt-4o` (alias), the gate is flaky and non-reproducible. Always pin a dated snapshot.
- **A gate with no cost/latency guardrail.** Accuracy can go up while cost/p95 explodes. Assert guardrail metrics too.
- **Judge cost in CI.** Running an LLM judge on 100 cases per PR adds latency and $. Use a tiered gate: cheap deterministic checks always, judge only on changed prompts or on a nightly schedule.
- **Flywheel that never closes.** Capturing feedback but never promoting it to the golden set means no learning. Schedule the promotion step.
- **Instrumenting everything at once.** Start with LLM-call spans; add retrieval/tool spans incrementally.

### Self-check
1. What are the parts of an OpenTelemetry GenAI span tree for a RAG request, and which attributes matter most for cost debugging?
2. How does linking feedback to trace IDs enable the data flywheel, and what must you redact?
3. What is a silent model swap and how does tagging the resolved model on each trace catch it?
4. Why must the CI gate pin a model snapshot and include cost/latency assertions, not just accuracy?
5. How would you keep judge cost in CI under control while still catching prompt regressions?

---

## Phase milestone project — CI eval gate + observability & flywheel stack

Build one integrated system that proves the whole phase. Take a Phase 4/6 app (RAG bot or tool-using agent) and wrap it in a measurement + observability layer.

**What it must do and prove:**
1. **Golden set (Week 1):** a 50–100-case stratified, versioned, decontaminated golden set mined from real/realistic traces, with a written failure taxonomy driving case selection.
2. **Calibrated judge (Week 2):** an LLM-as-judge (rubric + pairwise with position-bias control) calibrated against human labels, reporting **Cohen's kappa before and after de-biasing the rubric**, plus RAG faithfulness/relevance metrics or a hallucination tripwire.
3. **Statistical rigor (Week 2):** every reported score comes with a bootstrap 95% CI, and A/B decisions use a paired test — no 2%-on-50 "wins."
4. **CI eval gate (Week 3):** GitHub Actions runs the eval on every PR against a **pinned model snapshot**, posts a score diff, and **blocks the merge** on accuracy or p95-cost regression — demonstrated by a PR with a deliberately-worse prompt getting blocked while the baseline passes.
5. **Observability + flywheel (Week 3):** OpenTelemetry/OpenLLMetry tracing into self-hosted Langfuse or Phoenix with per-step tokens/cost/latency, PII redaction, a feedback API linking signals to traces, a sampled online judge eval with drift/silent-swap alerting, and a flywheel query that auto-adds low-rated traces to a review queue that grows the golden set.

**Acceptance criteria:**
- [ ] Judge kappa vs humans ≥ 0.6 (or documented reason + remediation if not), with before/after values shown.
- [ ] Reported metrics include 95% CIs; at least one A/B decision is made with a paired p-value.
- [ ] The deliberately-worse-prompt PR is blocked by CI; evidence (Action logs/screenshot) is in the repo.
- [ ] Dashboard shows cost/latency/quality for ≥50 requests; ≥1 trace has traveled feedback → review queue → golden set.
- [ ] A `README.md` explains every tradeoff and "what would make us regret this."

**Suggested repo layout:**
```
eval-observability-stack/
  src/
    system_under_test.py     # your RAG/agent app
    judge.py                 # rubric + pairwise judge, structured output
    feedback_api.py          # FastAPI: /feedback -> trace-linked store
    online_eval.py           # sampled prod-trace judging + drift/swap alert
    tracing.py               # OpenLLMetry init -> Langfuse/Phoenix
  evals/
    data/golden_v1.jsonl golden_v2.jsonl review_queue.jsonl
    taxonomy/failures.md
    stats.py                 # bootstrap CI + McNemar
    calibration.py           # kappa before/after
    flywheel.py              # low-rated traces -> review queue
  promptfooconfig.yaml
  .github/workflows/eval-gate.yml
  docker-compose.yml         # Langfuse or Phoenix
  README.md                  # tradeoffs + "what would make us regret this"
```

## You are ready to move on when...
- [ ] You build the eval **before** touching the prompt — reflexively, without being told.
- [ ] You can turn a pile of raw traces into a named, counted failure taxonomy and a stratified golden set in an afternoon.
- [ ] You never trust an LLM judge you haven't calibrated against humans, and you can name and neutralize its three biases.
- [ ] You refuse to call a small delta a win without a confidence interval and a paired test, and you can explain why to a skeptical PM.
- [ ] You can instrument any LLM app with OTel tracing and read a span tree to find where tokens, cost, and latency go.
- [ ] You have a CI gate that has actually blocked a bad change, and a flywheel that turns production failures into new eval cases.
