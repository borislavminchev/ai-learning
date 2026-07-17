# Week 2 Lab: Domain Eval, Currency Loop, and the Shippable Portfolio Project

> Week 1 gave you a model-agnostic smoke-test harness that ranks models by **`$/correct`**. This week you make it *yours* and make it *durable*: you bolt on a **domain eval** (20–30 real cases from a domain you actually care about) so the harness ranks models for **your task, not a leaderboard's**; you generate a **model-card cheat-sheet** so you can read any pricing page in five seconds; and you wire a **currency loop** (`make smoke MODEL=<id>` + a `RUNBOOK.md`) so that when a new model drops next month you make a mechanical, data-driven adopt/skip call in under an hour. Then you do the last-mile work that actually gets you hired: you **package your strongest Phase 4–12 system into a shippable portfolio project** — deployed to a public URL, with a real eval suite, a unit-economics writeup, a `TRADEOFFS.md`, and two rehearsed interview stories. This is **Part B of the phase milestone**.
>
> The through-line of the whole week: **numbers you generated beat numbers someone else marketed.** A leaderboard ranks *general* ability; you ship a *specific* task. A public URL a stranger can open beats ten notebooks nobody can run.
>
> Read these first — this guide assumes you have:
> - [../lectures/06-benchmark-literacy-and-limits.md](../lectures/06-benchmark-literacy-and-limits.md) — what MMLU/GPQA/SWE-bench/BFCL/LMArena actually measure; contamination, format sensitivity, usable≠advertised context; why a 25-case private eval wins.
> - [../lectures/07-reading-model-cards-and-pricing-fast.md](../lectures/07-reading-model-cards-and-pricing-fast.md) — the single decision line (ctx / out / in-$ / out-$), the four places pricing overcharges (caching, batch, cached-input, reasoning tokens), and `litellm.get_model_info()`.
> - [../lectures/08-staying-current-without-drowning.md](../lectures/08-staying-current-without-drowning.md) — primary changelogs over hype; the "new model dropped → run your smoke test → decide" runbook.
> - [../lectures/09-portfolio-and-interview-prep.md](../lectures/09-portfolio-and-interview-prep.md) — shipped AND evaluated; deployed URL + eval numbers + unit economics + tradeoffs; the AI-eng system-design interview.

**Est. time:** ~9 hrs (2.5 domain eval · 1.5 model cards · 1 currency loop · 4 portfolio) · **You will need:** the Week-1 `new-model-smoke-test` repo (its `src/client.py`, `src/runner.py`, `src/scoring.py`, `models.yaml`, `smoke/reports/latest.md`); Python 3.11+ with `uv`; `make` (macOS/Linux ships it; on Windows use Git-Bash + `choco install make`, or the `just` alternative shown inline); a GitHub account with Actions; and one deploy target (Streamlit Community Cloud or Hugging Face Spaces are fastest and free). **Free/local path:** every eval step runs against `ollama/llama3.1` for $0; the judge model can be `ollama/llama3.1` too; deployment on Streamlit Community Cloud / HF Spaces is free.

---

## Before you start (setup)

### 0.1 — Confirm the Week-1 harness still runs

**What:** re-run last week's smoke test end to end.
**Why:** this week *extends* that repo. If `runner.py` is broken, every step below inherits the break. Fix it now, not at hour 3.

```bash
cd new-model-smoke-test
uv run python -m src.runner            # regenerates smoke/reports/latest.md
```

**Expected result:** `smoke/reports/latest.md` contains a table with `extract | tools | longctx | reason | $/correct | p50 lat` columns for ≥3 models including one Ollama model.
**Verify:** `cat smoke/reports/latest.md` shows fresh numbers (check the file mtime with `ls -l smoke/reports/latest.md`).
**Troubleshoot:** if hosted models error on auth, confirm `.env` has your keys and you `load_dotenv()` at the top of `runner.py`. If Ollama errors, `ollama serve` in another terminal and `ollama pull llama3.1`.

### 0.2 — Add this week's dependencies

**What:** add a YAML reader (if not already present) and a judge helper — nothing heavy.
**Why:** the domain eval's rubric cases are judged by a *cheap model*, which is just another `litellm.completion` call, so you likely need nothing new. Add `pyyaml` only if Week 1 didn't.

```bash
uv add pyyaml            # if models.yaml is not already parsed with it
uv run python -c "import yaml, litellm; print('deps ok')"
```

**Verify:** prints `deps ok`.
**Troubleshoot:** if `import yaml` fails, it's `pyyaml` (the package), not `yaml` — the import name differs from the pip name.

### 0.3 — Decide your domain now (this gates Step 1)

**What:** pick the one domain your domain-eval will cover. Options: your real product's task (support-ticket classification, contract-clause extraction), a hobby corpus (board-game rules Q&A, recipe scaling), or your Phase 4–12 project's core task (best choice — it doubles as your portfolio project's eval later).
**Why:** the disagreement between *your* ranking and a leaderboard's is the entire lesson. That only lands if the domain is one you have real intuition about. A synthetic domain teaches nothing.
**Verify:** you can write down, right now, 3 example inputs and their known-correct outputs without looking anything up. If you can't, the domain is too unfamiliar — pick another.

---

## Step-by-step

Each step names which Definition-of-Done (DoD) item it satisfies.

### Step 1 — Build the domain eval (~2.5 hrs) · DoD #1

**What:** add `smoke/tasks/domain.py` encoding **20–30 real cases** from your chosen domain. Each case has an input and a *checkable* expected outcome. Run it across the same models and add a `domain` column to `latest.md`.

**Why:** MMLU/LMArena rank *general* ability; you ship *one specific task*. A model can top the Arena and lose on your 25 cases (contamination, format sensitivity, and "the leaderboard's distribution ≠ yours" all conspire). This step is where "trust your own eval" stops being a slogan and becomes a column in your table.

**Do it.** Three case types cover almost everything; use whichever fits each case:
1. **exact field** — the answer is one canonical string/label (classification, extraction).
2. **numeric** — the answer is a number, scored with a tolerance (calculations, counts).
3. **rubric** — open-ended, judged by a *cheap model* against a **low-cardinality** rubric (e.g. `pass|partial|fail`, or `1|2|3`). Low-cardinality is the trick that keeps a cheap judge reliable.

Create the cases file (JSONL keeps cases out of code and easy to grow):

```jsonl
// smoke/tasks/domain_cases.jsonl   — 20–30 lines; these are ILLUSTRATIVE, replace with YOUR domain
{"id": "d01", "type": "exact",   "input": "Ticket: 'card declined at checkout, tried twice'", "expected": "billing"}
{"id": "d02", "type": "numeric", "input": "A recipe for 4 needs 300g flour. Scale to 7 servings. Grams?", "expected": 525, "tol": 1}
{"id": "d03", "type": "rubric",  "input": "Explain to a new user why their export is empty.", "rubric": "PASS if it mentions checking the date-range filter; else FAIL", "expected": "PASS"}
```

Now the task module. It reuses your Week-1 `complete()` client and returns `(n_correct, n_total)` just like the other tasks so `runner.py` can treat it uniformly:

```python
# smoke/tasks/domain.py
import json, re, pathlib
from src.client import complete   # your Week-1 thin LiteLLM wrapper -> Call

CASES = [json.loads(l) for l in
         pathlib.Path(__file__).with_name("domain_cases.jsonl").read_text().splitlines() if l.strip()]

JUDGE_MODEL = "ollama/llama3.1"      # cheap/free; swap to gpt-4o-mini if you want a stronger judge

def _judge(rubric: str, answer: str) -> bool:
    """Low-cardinality rubric judged by a cheap model. Returns True iff PASS."""
    msg = [{"role": "user", "content":
            f"Rubric: {rubric}\n\nCandidate answer:\n{answer}\n\n"
            f"Reply with exactly one word: PASS or FAIL."}]
    verdict = complete(JUDGE_MODEL, msg, temperature=0).text.strip().upper()
    return verdict.startswith("PASS")

def _num(text: str):
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None

def run(model: str):
    """Run all domain cases against `model`. Returns (n_correct, n_total, cost, latencies)."""
    n_correct, cost, lats = 0, 0.0, []
    for c in CASES:
        call = complete(model, [{"role": "user", "content": c["input"]}], temperature=0)
        cost += call.cost_usd; lats.append(call.latency_s)
        out = call.text.strip()
        if c["type"] == "exact":
            ok = c["expected"].lower() in out.lower()
        elif c["type"] == "numeric":
            got = _num(out)
            ok = got is not None and abs(got - float(c["expected"])) <= c.get("tol", 0)
        elif c["type"] == "rubric":
            ok = _judge(c["rubric"], out)
        else:
            ok = False
        n_correct += int(ok)
    return n_correct, len(CASES), cost, lats
```

Wire it into `runner.py` next to the other tasks and add a `domain` column to the report:

```python
# in src/runner.py, alongside extraction/toolloop/longctx/reasoning
from smoke.tasks import domain
# ... inside the per-model loop:
d_correct, d_total, d_cost, d_lats = domain.run(model)
# add `f"{d_correct}/{d_total}"` to that model's row, fold d_cost into total cost,
# and include d_lats when you compute p50 latency.
```

**Expected result:** `smoke/reports/latest.md` now has a `domain` column:

```
| model                      | extract | tools | longctx | reason | domain | $/correct | p50 lat |
|----------------------------|---------|-------|---------|--------|--------|-----------|---------|
| openai/gpt-4o-mini         | 15/15   |  1/1  |  1/1    |  7/8   | 24/26  | $0.0011   | 0.9s    |
| anthropic/claude-3-5-haiku | 15/15   |  1/1  |  1/1    |  8/8   | 26/26  | $0.0018   | 1.1s    |
| ollama/llama3.1            | 12/15   |  1/1  |  0/1    |  4/8   | 17/26  | $0.0000   | 3.2s    |
```

**Verify:** the `domain` column is present for **≥3 models**, and the counts are ≤ your case count. Spot-check three cases by hand — run one input through a model in a REPL and confirm the pass/fail your scorer assigned matches your own judgment. Then write the required **two sentences** comparing your ranking to a public leaderboard's (put them in `README.md` under a "Domain eval vs leaderboard" heading):

> *Example:* "On LMArena, Model X ranks above Model Y overall, but on my 26 support-ticket cases Y beat X (26/26 vs 24/26) because X over-explains and trips my exact-label check. For my routing task, the Arena ranking is actively misleading — I'd ship Y."

**Troubleshoot:**
- **Rubric judge is flaky** → your rubric isn't low-cardinality or is ambiguous. Force a one-word verdict (as above), set `temperature=0`, and make the rubric a single testable condition. If a stronger judge disagrees with the cheap one on >2 cases, note it — that's a real finding about judge cost/quality.
- **Everything scores 26/26** → your cases are too easy; they don't discriminate. Add 5 hard/adversarial cases from real failures you've seen.
- **`in` substring match too lenient for exact** → for labels with overlap (e.g. `billing` vs `billing-error`), anchor with word boundaries: `re.search(rf"\b{re.escape(c['expected'])}\b", out, re.I)`.

### Step 2 — Model-card cheat-sheet generator (~1.5 hrs) · DoD #2

**What:** write `src/modelcard.py` using `litellm.get_model_info(model)` to print a one-line summary (context, max output, in/out $/M) for every model in `models.yaml`, and commit the output as `smoke/reports/modelcards.md`.

**Why:** every few weeks someone asks "should we switch?" The people who answer in 30 seconds have automated the extraction. Pricing pages hide decimals (per-token vs per-million) and quietly bill for cached-input, batch, and reasoning tokens (Lecture 7). A generator kills the "pasted the price with the decimal in the wrong place" class of error forever.

**Do it.**

```python
# src/modelcard.py
import sys, yaml, pathlib, litellm

def summarize(model: str) -> str:
    try:
        m = litellm.get_model_info(model)
    except Exception as e:
        return f"{model}: (no metadata: {e})"
    return (f"{model}: ctx={m.get('max_input_tokens')} "
            f"out={m.get('max_output_tokens')} "
            f"in=${(m.get('input_cost_per_token') or 0)*1e6:.2f}/M "
            f"out=${(m.get('output_cost_per_token') or 0)*1e6:.2f}/M")

def main():
    models = yaml.safe_load(pathlib.Path("models.yaml").read_text())["models"]
    lines = ["# Model cards (generated by src/modelcard.py)\n"]
    lines += [f"- {summarize(mdl)}" for mdl in models]
    out = pathlib.Path("smoke/reports/modelcards.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
```

```bash
uv run python -m src.modelcard
```

**Expected result:** `smoke/reports/modelcards.md`:

```
# Model cards (generated by src/modelcard.py)

- openai/gpt-4o-mini: ctx=128000 out=16384 in=$0.15/M out=$0.60/M
- anthropic/claude-3-5-haiku-latest: ctx=200000 out=8192 in=$0.80/M out=$4.00/M
- gemini/gemini-2.0-flash: ctx=1048576 out=8192 in=$0.10/M out=$0.40/M
- ollama/llama3.1: (no metadata: ...)      # local models have no pricing — expected
```

**Verify:** every model in `models.yaml` has a line; the `$/M` figures match the provider's public pricing page (open one and confirm — this is also the Lecture-7 skill). The Ollama line showing no pricing is **correct**, not a bug.
**Troubleshoot:**
- **`get_model_info` raises "model not found"** → LiteLLM's metadata DB may lag brand-new models. Update: `uv add -U litellm`. If still missing, the `except` clause degrades gracefully — note the model needs a manual line until LiteLLM catches up.
- **Numbers look 1000× off** → you double-multiplied. `input_cost_per_token` is per *single* token; `×1e6` gives $/M. Don't also divide.

### Step 3 — Wire up the currency loop (~1 hr) · DoD #3

**What:** add a `make smoke MODEL=<id>` target that runs the full suite against **one** new model and appends to `latest.md`; write a 5-line `RUNBOOK.md`; optionally add a GitHub Actions `workflow_dispatch`.

**Why:** staying current is a *mechanical loop*, not a vibe (Lecture 8). When a model ships, you don't read hot takes — you run one command and read your own table. Making it one command is what makes you actually do it in a year.

**Do it.** First make `runner.py` accept an optional single-model override and an append mode:

```python
# src/runner.py — argparse tail
import argparse
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="run just this one model id")
    ap.add_argument("--append", action="store_true", help="append a row instead of rewriting")
    args = ap.parse_args()
    models = [args.model] if args.model else load_models("models.yaml")
    run_all(models, append=args.append)   # append=True -> add rows to latest.md, don't truncate
```

Then the `Makefile` (tabs, not spaces, for recipe lines):

```makefile
# Makefile
smoke:
	uv run python -m src.runner --model $(MODEL) --append
	@echo ">> appended $(MODEL) to smoke/reports/latest.md"

cards:
	uv run python -m src.modelcard

report:
	uv run python -m src.runner        # full rewrite of latest.md
```

On **Windows** without `make`, use [`just`](https://github.com/casey/just) (`cargo install just` or `choco install just`) with a `justfile`:

```makefile
# justfile  (run: just smoke openai/o4-mini)
smoke MODEL:
	uv run python -m src.runner --model {{MODEL}} --append
cards:
	uv run python -m src.modelcard
```

The 5-line runbook:

```markdown
# RUNBOOK.md — "A new model just shipped"
1. Add the model id to `models.yaml` (one line).
2. `make smoke MODEL=<id>`   (or `just smoke <id>`) — runs the full suite, appends to smoke/reports/latest.md.
3. `make cards` — refresh smoke/reports/modelcards.md so you have its ctx/output/pricing line.
4. Read two columns only: **$/correct** and **domain**. Ignore launch-day benchmarks and Twitter.
5. Decide: adopt only if it beats your current pick on $/correct OR domain for a task you actually run.
```

**Optional — trigger from the browser.** Add a `workflow_dispatch` job so you can kick a live run from GitHub's UI (this one *does* need a provider key, stored as a repo secret — distinct from your free cassette-based CI):

```yaml
# .github/workflows/smoke.yml
name: smoke
on:
  workflow_dispatch:
    inputs:
      model: { description: "model id", required: true, default: "openai/gpt-4o-mini" }
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - env: { OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }} }
        run: uv run python -m src.runner --model "${{ inputs.model }}" --append
```

**Expected result:** `make smoke MODEL=openai/gpt-4o-mini` runs the whole suite for just that model and adds a row to `latest.md`; the GitHub Actions tab shows a "smoke" workflow with a "Run workflow" button.
**Verify:** run `make smoke MODEL=<a-model-already-in-your-table>` and confirm a **new appended row** appears (don't overwrite existing rows). `wc -l smoke/reports/latest.md` grows by one row.
**Troubleshoot:**
- **`Makefile:2: *** missing separator`** → recipe lines must start with a real TAB, not spaces. In many editors, paste then convert. Git-Bash `cat -A Makefile` shows `^I` for tabs.
- **`make` not found on Windows** → use the `just` path above, or `choco install make`, or add the target to a small `run.sh` and call `bash run.sh smoke <id>`.
- **`workflow_dispatch` button missing** → the workflow file must be on your default branch for GitHub to show the manual trigger.

### Step 4 — Package the headline portfolio project (~4 hrs) · DoD #4–#7 · Milestone Part B

This is the last mile that gets you hired. Take the **strongest system from Phases 4–12** (RAG bot, agent, or multimodal pipeline) and make it portfolio-grade. Four sub-deliverables — do them in this order.

#### 4a — Deploy it to a public URL (non-negotiable)

**What:** put the app behind a URL a stranger can open. **Why:** "a portfolio of undeployed notebooks" is the #1 pitfall — recruiters can't and won't run a notebook. Deployment forces you to fix the "works on my machine" gaps (env vars, cold start, secrets) that interviews probe.

**Do it — fastest free paths:**

- **Streamlit Community Cloud** (fastest for a Python app). Wrap your system in `app/streamlit_app.py`, push to a public GitHub repo, then at [share.streamlit.io](https://share.streamlit.io) point it at the repo. Add provider keys under **App → Settings → Secrets**.
  ```python
  # app/streamlit_app.py
  import streamlit as st
  from your_project import answer   # your Phase 4-12 entrypoint
  st.title("Ask my docs")
  q = st.text_input("Question")
  if q:
      with st.spinner("thinking"):
          st.write(answer(q))
  ```
  ```bash
  uv add streamlit
  uv run streamlit run app/streamlit_app.py   # test locally first
  ```
- **Hugging Face Spaces** — same idea; create a Space (SDK: Streamlit or Gradio), push, set keys in Space **Settings → Secrets**.
- **Render / Fly.io free tier** — for a FastAPI backend; add a `Dockerfile`, deploy from the repo.
- **Modal (free credits)** — only if a piece genuinely needs a GPU (e.g. a local embedding/vision model). Otherwise a hosted API keeps the app CPU-only and cheaper to run.

**Expected result:** opening the URL in an incognito window shows a working app you can query.
**Verify:** ask it a real question from an incognito/other device — no local server running. Put the URL at the top of the README.
**Troubleshoot:** blank page or 500 → check the platform's build logs; usually a missing secret or a dependency not in `pyproject.toml`/`requirements.txt`. Cold-start timeout on free tier → add a tiny warm-up call or note "first request ~15s cold start" in the README (honesty about limits reads as maturity).

#### 4b — Attach an eval suite with real numbers

**What:** reuse your **Phase 7 golden set + judge** and put real numbers in the README: a quality metric (accuracy / recall@k / faithfulness) plus **p50/p95 latency**.
**Why:** "shipped AND evaluated" is the bar. A URL with no numbers is a demo; numbers make it evidence you think like an engineer.

**Do it.** Copy your Phase 7 golden set into `evals/` and run it against the *deployed path* (or the same code the app calls):

```bash
mkdir -p evals && cp <phase7>/golden.jsonl evals/
uv run python evals/run.py    # your Phase 7 harness: scores quality + records latencies
```

Report in the README:

```markdown
## Eval (n=50 golden questions, judge=gpt-4o-mini)
| metric        | value |
|---------------|-------|
| faithfulness  | 0.92  |
| recall@5      | 0.88  |
| p50 latency   | 1.3s  |
| p95 latency   | 3.8s  |
```

**Verify:** the numbers regenerate by running `evals/run.py` (not hand-typed). p95 > p50 (if equal, you aren't measuring the tail — you need ≥20 samples).
**Troubleshoot:** judge disagreeing with you → recalibrate against a few hand-labeled cases (Phase 7 skill). Latency all identical → you're measuring a cached path; disable cache for the latency run or measure cold.

#### 4c — Cost/latency + unit-economics section

**What:** write the unit-economics section: **$/request, $/1k requests, cache hit rate, $/useful-answer**, plus **one measured optimization** and its effect.
**Why:** this is exactly the back-of-envelope math the AI-eng system-design interview asks for. `$/useful-answer` = `$/request ÷ (quality metric)` — the honest cost, since a cheap wrong answer costs you a retry or a churned user.

**Do it.** Pull real per-request token counts from your eval run (your Week-1 `Call` dataclass already carries `cost_usd`):

```markdown
## Unit economics
- Avg tokens/request: 4,200 in / 380 out
- $/request: $0.0021   ·   $/1k requests: $2.10
- Cache hit rate: 46% (semantic cache on retrieved chunks)
- $/useful-answer: $0.0021 / 0.92 faithfulness = $0.0023
- **Optimization:** added a semantic cache on the retrieval step → cost -42%, p50 latency -0.6s (measured over the 50-question golden set).
```

**Verify:** every figure traces to a measurement (eval run, cache-stats log), not a guess. The optimization has a *before/after* number.
**Troubleshoot:** no cache to report → that's fine, report cache hit rate 0% and propose the cache as your "at 100× scale" move in TRADEOFFS. But if you have ≥1 hr, adding a `functools`/dict semantic cache is the cheapest way to earn a real optimization number.

#### 4d — TRADEOFFS.md and the private INTERVIEW.md

**What:** write `TRADEOFFS.md` (why this model/framework/architecture, what at 100× scale, what would make us regret this) and a **private** `INTERVIEW.md` with two stories.
**Why:** naming your own tradeoffs — including "what would make us regret this" — is the single strongest signal of engineering maturity in an interview. The interview stories are your rehearsal script.

```markdown
# TRADEOFFS.md
## Why this stack
- Model: gpt-4o-mini — $/correct won on my domain eval (Week-2 table); good enough faithfulness at 1/10th Sonnet's cost.
- Framework: raw LiteLLM + a thin RAG loop, not LangChain — I could see and tune every prompt (Week-1 rubric: ejectability, opacity).
- Architecture: FAISS + hosted LLM — no vector-DB ops to run at this scale.
## At 100x scale
- FAISS in-process → managed Qdrant/pgvector; add a read replica.
- Add request queue + batch the embedding calls (batch discount).
- Semantic cache becomes load-bearing (today it's a nice-to-have at 46%).
## What would make us regret this
- If queries drift multi-hop, single-shot RAG degrades — would need a re-ranker or an agentic retriever.
- gpt-4o-mini deprecation: mitigated because swapping models is a one-line models.yaml + `make smoke` decision.
```

```markdown
# INTERVIEW.md  (PRIVATE — add to .gitignore)
## Story 1 — System-design walkthrough (5 min)
- Problem, users, QPS assumption (say 5 QPS peak).
- Token math: 4.2k in + 380 out /req → ~23k tok/s at peak → $X/day.
- Retrieval → context-assembly → LLM → validation; where the cache sits; fallback model.
## Story 2 — Debugging non-determinism
- The bug (flaky eval / intermittent tool-call failure).
- How I made it reproducible (temperature=0, seeded, VCR cassette).
- The fix and how I proved it (eval went from X to Y).
```

```bash
echo "INTERVIEW.md" >> .gitignore   # keep it private
```

**Expected result:** `TRADEOFFS.md` answers all three questions; `INTERVIEW.md` exists locally and is git-ignored.
**Verify:** rehearse Story 1 out loud with a timer — it must land in ~5 minutes and include real QPS/token/cost numbers from 4c, not hand-waving.
**Troubleshoot:** can't fill "what would make us regret this" → you haven't stress-tested the design; ask "what input distribution breaks my retrieval / my prompt / my cost model?" and answer that.

---

## Putting it together — end-to-end run

```bash
# 1. Domain eval + full table + model cards, regenerated from scratch
cd new-model-smoke-test
uv run python -m src.runner            # writes latest.md WITH the domain column
uv run python -m src.modelcard         # writes modelcards.md

# 2. Simulate a new model dropping (the currency loop)
#    (edit models.yaml to add e.g. openai/o4-mini, then:)
make smoke MODEL=openai/o4-mini        # or: just smoke openai/o4-mini
cat smoke/reports/latest.md            # read $/correct and domain columns -> decide per RUNBOOK.md

# 3. Portfolio project: deployed + evaluated
uv run streamlit run app/streamlit_app.py   # local smoke test, then it's live at your public URL
uv run python evals/run.py                   # regenerate README eval numbers
```

Then open your **live URL** in an incognito window and ask it a real question. That single click — a stranger querying your evaluated, cost-instrumented system — is the whole point of the phase.

---

## Definition of Done — verifiable checks

Restating the spine checklist as things you can literally verify:

- [ ] **Domain eval across ≥3 models.** `smoke/reports/latest.md` has a `domain` column with counts for ≥3 models (incl. one Ollama). — *verify:* `grep -i domain smoke/reports/latest.md` and count model rows.
- [ ] **Leaderboard comparison in writing.** README (or latest.md) has two sentences comparing your ranking to a public leaderboard's, naming the disagreement. — *verify:* the sentences exist and name a specific benchmark and a specific model flip.
- [ ] **`modelcards.md` correct for every model.** One line per `models.yaml` entry with ctx / max-output / in-$/M / out-$/M. — *verify:* line count == model count; a hosted model's $/M matches its public pricing page.
- [ ] **RUNBOOK + `make smoke MODEL=<id>` works end to end.** — *verify:* running it appends exactly one row to `latest.md`; `RUNBOOK.md` has the 5 steps.
- [ ] **Live public URL.** — *verify:* opens in an incognito window with no local server; URL is at the top of the README.
- [ ] **README shows real eval + cost numbers.** Quality metric + p50/p95 latency + $/request (or $/1k). — *verify:* every number regenerates from `evals/run.py`, none hand-typed.
- [ ] **`TRADEOFFS.md` complete.** Answers "why this stack," "what at 100× scale," "what would make us regret this." — *verify:* all three headings present and specific.
- [ ] **5-minute system-design walkthrough rehearsed.** — *verify:* time yourself; it includes QPS/token/cost math and a non-determinism debugging story (from `INTERVIEW.md`).

**Milestone Part B acceptance (from the spine):** the portfolio project is reachable at a public URL, its README shows real eval + cost numbers, and you can give the 5-minute walkthrough with cost/token math and a failure/degradation story. All four are covered by the checks above.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Domain scorer passes everything | cases too easy / substring match too loose | add adversarial cases; use `\b` word-boundary regex for exact labels |
| Rubric judge flip-flops | rubric not low-cardinality / temp>0 | one-word PASS\|FAIL verdict, `temperature=0`, single testable condition |
| `get_model_info` "not found" | LiteLLM metadata lags new model | `uv add -U litellm`; the `except` clause degrades gracefully meanwhile |
| Model-card $/M off by 1000× | double-multiplied per-token price | multiply per-token by `1e6` once; don't divide |
| `Makefile missing separator` | spaces instead of TAB in recipe | real TAB; verify with `cat -A Makefile` (`^I`) |
| `make` not on Windows | not installed | use `justfile` + `just`, or `choco install make`, or `bash run.sh` |
| `workflow_dispatch` button absent | workflow not on default branch | merge the workflow file to `main` |
| Deployed app blank / 500 | missing secret or dep | check platform build logs; add key to Secrets; pin deps |
| p50 == p95 in eval | too few samples / cached path | ≥20 samples; disable cache for the latency run |
| Can't answer "what would make us regret this" | design not stress-tested | ask what input distribution breaks retrieval/prompt/cost, answer that |

---

## Stretch goals (optional)

- **Judge-vs-judge check.** Run the rubric cases with both a cheap judge (`ollama/llama3.1`) and a strong one (`gpt-4o-mini`); report where they disagree. This is a real, cited finding about judge cost/quality tradeoffs you can put in TRADEOFFS.
- **Contamination probe.** Add 3 domain cases you *know* are post-cutoff or private; if a model aces public benchmarks but flops these, you've demonstrated contamination in your own table (Lecture 6).
- **Usable-context test in your domain.** Bury a domain fact at your real context length (not 20k — *your* app's actual size) and confirm recall holds; advertised context ≠ usable context (Lecture 6).
- **Currency automation.** Add a scheduled GitHub Action (weekly `workflow_dispatch` cron) that runs `make cards` and opens a PR if any model's pricing changed — the "stay current" loop, fully mechanical.
- **Cost dashboard.** Add a tiny `/metrics` route or a Streamlit sidebar that shows live $/request and cache hit rate from the last N requests — turns your unit-economics writeup into a running instrument.
- **A/B the optimization.** Deploy two variants (with/without semantic cache) behind a flag and show the cost/latency delta on live traffic, not just the golden set.
