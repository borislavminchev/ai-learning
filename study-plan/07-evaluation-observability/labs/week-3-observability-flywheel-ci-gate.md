# Week 3 Lab: Observability, the Data Flywheel & a Blocking CI Eval Gate (Phase Milestone)

> This is the final week of Phase 7, so this guide **folds in the phase milestone**: you assemble everything from Weeks 1–2 into one integrated, defensible measurement + observability stack. You instrument your app with **OpenTelemetry GenAI** conventions via **OpenLLMetry**, ship traces into a self-hosted **Langfuse** *or* **Arize Phoenix** in Docker, and read a per-step cost/latency/token dashboard. You add a **feedback API** that links thumbs/ratings to trace IDs (with PII redaction), then a **data flywheel** that routes low-rated traces into a review queue and grows `golden_v2.jsonl`. You add a **sampled online eval** (judge a random ~5% of prod traffic) with **drift / silent-model-swap** alerting. Finally — the milestone core — you wire a **CI eval gate** (promptfoo, and a pure-Python `pytest` alternative) into GitHub Actions that **blocks a merge** when a deliberately-worse prompt regresses accuracy or blows the p95-cost guardrail. The evidence that the gate blocked a bad PR is your milestone artifact.
>
> Read the matching lectures first — this guide assumes them:
> - [../lectures/10-otel-genai-openllmetry.md](../lectures/10-otel-genai-openllmetry.md) — span trees, GenAI semantic conventions, OpenLLMetry init
> - [../lectures/11-observability-platforms.md](../lectures/11-observability-platforms.md) — Langfuse vs Phoenix vs hosted, when to pick which
> - [../lectures/12-feedback-and-flywheel.md](../lectures/12-feedback-and-flywheel.md) — explicit/implicit signals, trace-linking, PII redaction, the loop
> - [../lectures/13-drift-online-eval.md](../lectures/13-drift-online-eval.md) — model pinning, resolved-model tagging, sampled online eval, alerting
> - [../lectures/14-ci-eval-gate.md](../lectures/14-ci-eval-gate.md) — promptfoo assert types, pinned snapshots, tiered gates, pytest gate

**Est. time:** ~9 hrs (bump to ~15 to polish the full milestone) · **You will need:** the `llm-evals` repo from Weeks 1–2 (with `evals/data/golden_v1.jsonl`, `src/judge.py`, `evals/stats.py`); Python 3.11+ with `uv`; Docker Desktop (for Langfuse/Phoenix); Node.js 18+ (for promptfoo via `npx`); a GitHub repo with Actions enabled. **Free/local path:** Phoenix runs as a single local process (no Docker, no cloud). Langfuse self-hosts in Docker. The judge and the app can run against **Ollama** (`http://localhost:11434/v1`) so no paid API is required — pin an Ollama tag as your "snapshot." CI runs on GitHub's free `ubuntu-latest` runners. **Critical constraint (carried from Week 2):** the online-eval judge family must differ from the generator family (self-enhancement bias).

---

## Before you start (setup)

**WHAT:** Confirm you are in the Week-1/2 repo, install this week's dependencies, and choose an observability backend.

**WHY:** Every step extends the same `llm-evals` package. OpenLLMetry, the observability backend, FastAPI, and promptfoo pull heavy or cross-ecosystem deps (Docker images, a Node toolchain); installing/pulling them up front avoids mid-lab stalls.

**Do it:**
```bash
cd llm-evals            # the repo from Weeks 1-2

# tracing + app + feedback API + drift check
uv add traceloop-sdk openai "fastapi[standard]" httpx
# PII redaction (choose ONE path)
uv add presidio-analyzer presidio-anonymizer   # heavier, spaCy-backed, high recall
#   OR skip presidio and use the regex fallback in Step 3 (zero extra deps)

# observability backend — PICK ONE:
#  A) Arize Phoenix (lightest: single local process, no Docker) — recommended to start
uv add arize-phoenix openinference-instrumentation-openai
#  B) Langfuse (self-hosted in Docker; stronger prompt mgmt + eval UI) — Step 1 option B

mkdir -p src evals prompts .github/workflows
```

Verify Docker and Node are available (needed for Langfuse and promptfoo respectively):
```bash
docker --version        # Docker Desktop running?  (only needed for Langfuse)
node --version          # v18+  (for `npx promptfoo`)
```

Add the app + judge + tracing config to `.env` (extends Week 2's `.env`):
```bash
# --- generator (your system under test) ---
APP_PROVIDER=openai
APP_MODEL=gpt-4o-2024-11-20            # PIN a dated snapshot — never a bare alias
OPENAI_API_KEY=sk-...

# --- FREE / LOCAL path via Ollama (OpenAI-compatible) ---
# APP_PROVIDER=ollama
# APP_MODEL=llama3.1:8b                 # your pinned "snapshot" is the exact tag
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_API_KEY=ollama

# --- judge (Week 2) — family MUST differ from generator ---
JUDGE_MODEL=gpt-4o-2024-11-20          # or claude-*/llama3.1 per Week 2

# --- OTLP export target (set by Step 1 depending on backend) ---
# Phoenix: http://localhost:6006/v1/traces      Langfuse: http://localhost:3000/api/public/otel/v1/traces
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006
```

**Windows / Git-Bash note:** `uv`, `docker`, `node`, `npx` are identical in Git-Bash. Docker Desktop must be running (WSL2 backend) before `docker compose up`. For Ollama on Windows, install the native app from ollama.com; the server auto-listens on `localhost:11434`. If `npx promptfoo` is slow first-run, that's the one-time download — subsequent runs are cached.

**Expected result:** `uv sync` succeeds; `uv run python -c "import traceloop, fastapi, phoenix; print('ok')"` prints `ok`.

**Verify:** `uv run python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('APP_MODEL'), os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT'))"` prints your pinned model and OTLP endpoint.

**Troubleshoot:**
- `presidio-analyzer` import fails needing a spaCy model → `uv run python -m spacy download en_core_web_lg`, or just use the regex redactor in Step 3 (no spaCy).
- Docker Desktop not running → Langfuse `docker compose up` hangs; start Docker Desktop or use Phoenix (Option A, no Docker).
- `node`/`npx` missing → install Node 18+ (nvm-windows on Windows) or defer promptfoo and use the pure-Python pytest gate in Step 5b.

---

## Step-by-step

### Step 1 — Stand up an observability backend

**WHAT:** Run either Arize Phoenix (single local process) or self-hosted Langfuse (Docker Compose), and note its OTLP ingest endpoint.

**WHY:** You need a place for spans to land and a dashboard to read them. Because OpenLLMetry emits standard OTel GenAI spans (see [../lectures/10-otel-genai-openllmetry.md](../lectures/10-otel-genai-openllmetry.md)), your instrumentation is portable — you can swap backends later without touching app code ([../lectures/11-observability-platforms.md](../lectures/11-observability-platforms.md)). Start local for privacy and zero cost.

**Do it — Option A: Phoenix (lightest, no Docker):**
```bash
# starts a local collector + UI on http://localhost:6006, OTLP on the same port
uv run python -m phoenix.server.main serve
# leave this running in its own terminal
```

**Do it — Option B: Langfuse (self-hosted, Docker):**
```bash
git clone https://github.com/langfuse/langfuse
cd langfuse && docker compose up -d          # UI at http://localhost:3000
cd ..
# Create a project + API keys in the UI, then add to .env:
#   LANGFUSE_PUBLIC_KEY=pk-lf-...
#   LANGFUSE_SECRET_KEY=sk-lf-...
#   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:3000/api/public/otel
```

**Expected result:** Phoenix prints `🚀 Phoenix ... http://localhost:6006` (or Langfuse UI loads at `:3000` and `docker compose ps` shows containers `Up`).

**Verify:** open the URL in a browser — you see an empty traces/projects view (no spans yet, that's fine).

**Troubleshoot:**
- Port already in use (`6006`/`3000`) → stop the other process or set `PHOENIX_PORT=6007` / edit Langfuse compose ports; update `OTEL_EXPORTER_OTLP_ENDPOINT`.
- Langfuse containers crash-loop → usually the Postgres volume; `docker compose down -v && docker compose up -d` to reset (dev only, wipes data).
- Phoenix `command not found` → you ran it outside `uv run`; use `uv run python -m phoenix.server.main serve`.

---

### Step 2 — Instrument the app with OpenLLMetry and confirm a span tree

**WHAT:** Add `src/tracing.py` that initializes OpenLLMetry (Traceloop) pointing at your OTLP endpoint, wrap your Phase 1–6 system as `src/system_under_test.py`, tag every trace with the **resolved** model, run 30+ requests, and confirm a span tree with per-call tokens/cost/latency.

**WHY:** OpenLLMetry auto-instruments OpenAI/Anthropic/etc. so each LLM call, retrieval, and tool call becomes a child span carrying `gen_ai.request.model`, prompt/completion, token counts, and latency — the raw material for cost debugging and later drift detection. Tagging the *resolved* model (what the provider actually served) is what makes silent-swap detection possible in Step 4.

**Do it:**
```python
# src/tracing.py
from __future__ import annotations
import os
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

def init_tracing() -> None:
    """Point OpenLLMetry at Phoenix/Langfuse via OTLP. Idempotent."""
    Traceloop.init(
        app_name="llm-evals",
        api_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:6006"),
        disable_batch=True,   # flush eagerly in dev so spans show up immediately
    )
```

```python
# src/system_under_test.py  — thin wrapper over YOUR Phase 1-6 system
from __future__ import annotations
import os
from openai import OpenAI
from traceloop.sdk.decorators import workflow
from opentelemetry import trace

_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,   # Ollama endpoint or default
)
PINNED_MODEL = os.getenv("APP_MODEL", "gpt-4o-2024-11-20")

SYSTEM_PROMPT_PATH = "prompts/current.txt"   # single source of truth for the prompt

def _system_prompt() -> str:
    return open(SYSTEM_PROMPT_PATH, encoding="utf-8").read()

@workflow(name="answer")           # creates the ROOT request span
def answer(question: str) -> dict:
    resp = _client.chat.completions.create(
        model=PINNED_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": _system_prompt()},
                  {"role": "user", "content": question}],
    )
    # Tag the RESOLVED model on the current span (silent-swap detection in Step 4).
    resolved = getattr(resp, "model", PINNED_MODEL)
    span = trace.get_current_span()
    span.set_attribute("gen_ai.response.model", resolved)
    span.set_attribute("app.pinned_model", PINNED_MODEL)
    usage = resp.usage
    return {
        "answer": resp.choices[0].message.content,
        "resolved_model": resolved,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
```

Create the baseline prompt and a driver:
```bash
mkdir -p prompts
cat > prompts/current.txt <<'EOF'
You are a precise assistant. Answer the user's question using only well-established facts.
If you are not sure, say "I don't know." Return a concise, correctly-formatted answer.
EOF
```
```python
# evals/replay.py — push 30+ golden inputs through the traced system
import json, pathlib
from src.tracing import init_tracing
from src.system_under_test import answer

init_tracing()
rows = [json.loads(l) for l in
        pathlib.Path("evals/data/golden_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
for r in rows[:40]:
    out = answer(r["input"])
    print(r["id"], out["resolved_model"], out["prompt_tokens"], "->", out["completion_tokens"])
```
```bash
uv run python -m evals.replay
```

**Expected result:** 30+ lines print with token counts; the Phoenix/Langfuse dashboard fills with traces. Each trace is a **span tree**: a root `answer` workflow span with an LLM-call child span carrying model, token counts, and latency.

**Verify:** in the dashboard, open one trace and confirm you see (a) the model name, (b) prompt+completion token counts, (c) latency in ms. Build/open a view showing **p50/p95 latency, total cost, and tokens per request** across the run (Phoenix: the project's metrics tab; Langfuse: the dashboard/traces filters).

**Troubleshoot:**
- No spans appear → `OTEL_EXPORTER_OTLP_ENDPOINT` wrong or backend down; with `disable_batch=True` spans flush immediately, so a mismatch is the usual cause. For Langfuse the path must end in `/api/public/otel`.
- Tokens are `null` on the Ollama path → Ollama's usage reporting is partial; latency/spans still work, and you can estimate cost from tokens×price later.
- Cost is blank → cost is derived from model pricing; on Phoenix set the model's price or compute cost yourself from token counts (fine for learning).

---

### Step 3 — Feedback API + the data flywheel

**WHAT:** Add `src/feedback_api.py` (FastAPI `/feedback` linking a rating to a `trace_id`, redacting PII before write) and `evals/flywheel.py` (pulls `rating <= 2` traces into `evals/data/review_queue.jsonl`), then manually promote ≥1 into `golden_v2.jsonl`.

**WHY:** Feedback linked to trace IDs is what turns production into training signal — the whole reason to instrument ([../lectures/12-feedback-and-flywheel.md](../lectures/12-feedback-and-flywheel.md)). Redacting PII **at write time** matters because traces + feedback are a data-governance surface. The flywheel only pays off if the promotion step actually runs, so we close the loop end-to-end here.

**Do it:**
```python
# src/redact.py — PII redaction with a zero-dep regex fallback
from __future__ import annotations
import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_SSN   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD  = re.compile(r"\b(?:\d[ -]?){13,16}\b")

def redact(text: str) -> str:
    if not text:
        return text
    text = _EMAIL.sub("[EMAIL]", text)
    text = _SSN.sub("[SSN]", text)
    text = _CARD.sub("[CARD]", text)
    text = _PHONE.sub("[PHONE]", text)
    return text

# Optional higher-recall path (if presidio installed):
# from presidio_analyzer import AnalyzerEngine
# from presidio_anonymizer import AnonymizerEngine
# def redact(text): ... analyze -> anonymize ...
```
```python
# src/feedback_api.py
from __future__ import annotations
import json, pathlib, time
from fastapi import FastAPI
from pydantic import BaseModel
from src.redact import redact

app = FastAPI()
STORE = pathlib.Path("evals/data/feedback.jsonl")
STORE.parent.mkdir(parents=True, exist_ok=True)

class Feedback(BaseModel):
    trace_id: str
    rating: int            # 1 (bad) .. 5 (great); thumbs-down maps to 1
    comment: str = ""

@app.post("/feedback")
def feedback(fb: Feedback):
    row = {"trace_id": fb.trace_id, "rating": fb.rating,
           "comment": redact(fb.comment), "ts": time.time()}   # redact BEFORE write
    with STORE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return {"ok": True}
```
```bash
# run the API and post a low-rated feedback linked to a trace id
uv run uvicorn src.feedback_api:app --port 8000 &
curl -s -X POST http://localhost:8000/feedback \
  -H 'content-type: application/json' \
  -d '{"trace_id":"trace-123","rating":1,"comment":"wrong answer, email me a@b.com 555-123-4567"}'
```
```python
# evals/flywheel.py — low-rated feedback -> review queue (dedup by trace_id)
import json, pathlib

fb = pathlib.Path("evals/data/feedback.jsonl")
queue = pathlib.Path("evals/data/review_queue.jsonl")
seen = {json.loads(l)["trace_id"] for l in queue.read_text().splitlines()} if queue.exists() else set()

promoted = 0
with queue.open("a", encoding="utf-8") as out:
    for line in fb.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["rating"] <= 2 and r["trace_id"] not in seen:      # explicit low rating
            out.write(json.dumps({"trace_id": r["trace_id"], "comment": r["comment"],
                                  "source": "flywheel", "stratum": "prod_failure"}) + "\n")
            seen.add(r["trace_id"]); promoted += 1
print(f"promoted {promoted} low-rated trace(s) to review_queue.jsonl")
```
```bash
uv run python -m evals.flywheel
# Then MANUALLY promote a reviewed case into the versioned golden set:
#   inspect review_queue.jsonl, fetch the trace's input/output, write a proper case
#   {id, input, expected?, criteria, stratum:"prod_failure", source:"prod"} into golden_v2.jsonl
cp evals/data/golden_v1.jsonl evals/data/golden_v2.jsonl   # start v2 from v1, then append the promoted case
git add evals/data/golden_v2.jsonl && git commit -m "golden set v2: +prod failure from flywheel"
```

**Expected result:** `curl` returns `{"ok":true}`; `feedback.jsonl` has the row with `comment` showing `[EMAIL]`/`[PHONE]` (raw PII gone); `flywheel.py` prints `promoted 1 low-rated trace(s)`; `golden_v2.jsonl` contains the new case.

**Verify:** `grep -c '\[EMAIL\]' evals/data/feedback.jsonl` ≥ 1 and `grep -c 'a@b.com' evals/data/feedback.jsonl` == 0 (redaction happened before write). `wc -l evals/data/golden_v2.jsonl` > `wc -l evals/data/golden_v1.jsonl`.

**Troubleshoot:**
- `curl` connection refused → uvicorn not up; drop the `&` and run it in its own terminal, or check port 8000.
- Windows `&` backgrounding flaky in Git-Bash → run uvicorn in a separate terminal window instead of backgrounding.
- Raw PII still in the file → you wrote before redacting; ensure `redact()` wraps `comment` in the row dict, not after.
- Duplicate cases piling up → the `seen` dedup set relies on stable `trace_id`s; ensure feedback carries the real trace id from the dashboard.

---

### Step 4 — Sampled online eval + drift / silent-model-swap check

**WHAT:** Add `src/online_eval.py` that (a) judges a random ~5% of traffic with your Week-2 judge and logs the score as a span attribute, and (b) fires a WARNING/alert when the rolling online score drops below a threshold **or** the resolved model differs from the pinned snapshot.

**WHY:** Providers silently rotate models behind `latest` aliases and input distributions drift ([../lectures/13-drift-online-eval.md](../lectures/13-drift-online-eval.md)). Sampling keeps online-judge cost bounded while still surfacing quality regressions; tagging the resolved model on every trace (Step 2) is exactly what lets you catch a silent swap by comparison to the pin.

**Do it:**
```python
# src/online_eval.py
from __future__ import annotations
import os, random, logging
from opentelemetry import trace
from src.judge import judge          # Week 2 rubric judge -> Verdict
from src.system_under_test import answer, PINNED_MODEL

log = logging.getLogger("online_eval")
logging.basicConfig(level=logging.INFO)

SAMPLE_RATE = float(os.getenv("ONLINE_SAMPLE_RATE", "0.05"))   # judge ~5%
QUALITY_FLOOR = float(os.getenv("ONLINE_QUALITY_FLOOR", "0.7"))
DEFAULT_CRITERIA = "Answer is correct, grounded, and says 'I don't know' when unsure."

def _alert(msg: str) -> None:
    log.warning("ALERT: %s", msg)
    url = os.getenv("SLACK_WEBHOOK_URL")
    if url:
        import httpx
        httpx.post(url, json={"text": msg}, timeout=5)

def observe(question: str, criteria: str = DEFAULT_CRITERIA) -> dict:
    out = answer(question)                              # traced, tags resolved model
    # (a) silent-swap check — resolved vs pinned
    if out["resolved_model"] and PINNED_MODEL not in out["resolved_model"]:
        _alert(f"silent model swap: pinned={PINNED_MODEL} resolved={out['resolved_model']}")
    # (b) sampled online judge
    if random.random() < SAMPLE_RATE:
        v = judge(question, out["answer"], criteria)
        trace.get_current_span().set_attribute("eval.online_score", v.score)
        trace.get_current_span().set_attribute("eval.online_pass", v.pass_)
        if (v.score / 4.0) < QUALITY_FLOOR:
            _alert(f"online quality below floor: score={v.score}/4 q='{question[:60]}'")
    return out
```
Test the drift alert deterministically by forcing a mismatch:
```bash
# force sampling to 100% and set an impossible pin to prove BOTH alerts fire
ONLINE_SAMPLE_RATE=1.0 APP_MODEL=gpt-4o-DOES-NOT-EXIST-2099 \
  uv run python -c "from src.online_eval import observe; print(observe('What is the capital of France?'))" 2>&1 | grep -i alert || true
```

**Expected result:** with a wrong pin, a `WARNING ALERT: silent model swap...` line prints; at 100% sample rate, an `eval.online_score` span attribute is set (visible in the dashboard), and low scores emit a quality alert.

**Verify:** in the dashboard, filter traces where `eval.online_score` is present — that confirms ≥5% (here 100% for the test) carry a judge score. The forced-mismatch run must print the silent-swap alert. Reset `APP_MODEL` to the real pin afterward.

**Troubleshoot:**
- No alert on mismatch → the substring check assumes the resolved model contains the pin; if your provider returns a totally different string, compare exact strings you control instead.
- Online judge too slow/expensive → lower `ONLINE_SAMPLE_RATE`; on the free path use Ollama as the judge.
- `eval.online_score` never appears → sampling is probabilistic; set `ONLINE_SAMPLE_RATE=1.0` for the verification run.

---

### Step 5a — CI eval gate with promptfoo (milestone core)

**WHAT:** Create `promptfooconfig.yaml` that runs your prompt over the golden set with an `llm-rubric` correctness assert plus `cost` and `latency` guardrails against a **pinned** provider snapshot, wire it into `.github/workflows/eval-gate.yml` on `pull_request`.

**WHY:** Prompts and models are versioned artifacts with pass thresholds, exactly like unit tests ([../lectures/14-ci-eval-gate.md](../lectures/14-ci-eval-gate.md)). promptfoo exits non-zero when an assertion fails, which is what blocks the merge. Pinning the snapshot keeps the gate reproducible (un-pinned = flaky). Guardrails on cost/latency stop "accuracy up, cost exploded" from sneaking through.

**Do it:**
```bash
# convert golden_v1.jsonl -> promptfoo test cases {vars, assert.value from criteria}
cat > evals/to_promptfoo.py <<'PY'
import json, pathlib, yaml
rows = [json.loads(l) for l in pathlib.Path("evals/data/golden_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
tests = [{"vars": {"question": r["input"]},
          "assert": [{"type": "llm-rubric",
                      "value": r.get("criteria", "Answer is correct and grounded.")}]}
         for r in rows]
pathlib.Path("evals/promptfoo_tests.yaml").write_text(yaml.safe_dump(tests, sort_keys=False))
print(f"wrote {len(tests)} tests")
PY
uv add pyyaml && uv run python evals/to_promptfoo.py
```
```yaml
# promptfooconfig.yaml
prompts:
  - id: current
    raw: "file://prompts/current.txt"        # system prompt as an artifact
providers:
  - id: openai:gpt-4o-2024-11-20             # PIN the dated snapshot — never a bare alias
    config:
      temperature: 0
tests: file://evals/promptfoo_tests.yaml
defaultTest:
  vars:
    question: ""
  assert:
    - type: cost
      threshold: 0.01                        # per-call cost guardrail ($)
    - type: latency
      threshold: 5000                         # ms
# free/local path: set provider to `ollama:chat:llama3.1:8b` and drop the cost assert
```
```bash
# run locally first (Node toolchain via npx; no global install needed)
npx promptfoo@latest eval -c promptfooconfig.yaml --output results.json
echo "exit code: $?"     # 0 = all asserts passed, non-zero = a failure (this is the gate)
```
```yaml
# .github/workflows/eval-gate.yml
name: eval-gate
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - name: Run promptfoo eval gate (pinned snapshot)
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: npx promptfoo@latest eval -c promptfooconfig.yaml --share=false
        # promptfoo exits non-zero on any failed assert -> blocks the merge
```

**Expected result:** local `npx promptfoo eval` prints a pass/fail table and exits `0` on the baseline prompt. In GitHub, the `eval-gate` check appears on PRs.

**Verify:** `echo "exit code: $?"` is `0` for the baseline. Add `OPENAI_API_KEY` under repo Settings → Secrets → Actions; open a trivial PR and confirm the `eval-gate` check runs and passes.

**Troubleshoot:**
- `llm-rubric` needs an LLM and fails in CI → it uses `OPENAI_API_KEY`; ensure the secret is set. To keep judge cost down, gate the judge to changed prompts (tiered gate) and always run the cheap `cost`/`latency`/deterministic asserts.
- promptfoo can't read `prompts/current.txt` → paths are relative to the config file; keep `promptfooconfig.yaml` at repo root.
- Flaky pass/fail run-to-run → you used a bare alias (`gpt-4o`); pin the dated snapshot as shown.

---

### Step 5b — Pure-Python gate alternative (no Node) + Step 6: prove it blocks a bad prompt

**WHAT:** Provide a `pytest`-based gate (loads the golden set, runs your judge, computes accuracy + bootstrap CI, asserts `accuracy >= THRESHOLD and p95_cost <= BUDGET`), then open a PR with a deliberately-worse prompt and capture the blocked Action.

**WHY:** A pure-Python gate avoids the Node dependency and reuses your Week-2 `judge`/`stats` code directly. Proving the gate blocks a genuinely-worse prompt is the milestone's decisive evidence — a gate that never fails proves nothing.

**Do it (pytest gate):**
```python
# tests/test_eval_gate.py
import json, os, pathlib, statistics
import pytest
from src.system_under_test import answer
from src.judge import judge
from evals.stats import bootstrap_ci      # Week 2

ACC_THRESHOLD = float(os.getenv("ACC_THRESHOLD", "0.75"))
P95_COST_BUDGET = float(os.getenv("P95_COST_BUDGET", "0.01"))
PRICE = {"prompt": 2.5e-6, "completion": 1.0e-5}   # $/token for the pinned model

def _cost(o): return (o["prompt_tokens"] or 0)*PRICE["prompt"] + (o["completion_tokens"] or 0)*PRICE["completion"]

def _golden():
    p = pathlib.Path("evals/data/golden_v1.jsonl")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def test_eval_gate():
    correct, costs = [], []
    for r in _golden():
        out = answer(r["input"])
        v = judge(r["input"], out["answer"], r.get("criteria", "Answer is correct and grounded."))
        correct.append(1 if v.pass_ else 0)
        costs.append(_cost(out))
    acc = statistics.mean(correct)
    lo, hi = bootstrap_ci(correct)                       # 95% CI on accuracy
    p95_cost = sorted(costs)[int(0.95*len(costs))-1]
    print(f"accuracy={acc:.3f} 95%CI=[{lo:.3f},{hi:.3f}] p95_cost=${p95_cost:.4f}")
    assert acc >= ACC_THRESHOLD, f"accuracy {acc:.3f} < {ACC_THRESHOLD}"
    assert p95_cost <= P95_COST_BUDGET, f"p95 cost ${p95_cost:.4f} > budget ${P95_COST_BUDGET}"
```
Wire it (either replace the promptfoo step or add a second job):
```yaml
      - name: Python eval gate
        env: { OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }} }
        run: uv run pytest tests/test_eval_gate.py -x -q
```

**Do it (prove it blocks — Step 6):**
```bash
git checkout -b bad-prompt
cat > prompts/current.txt <<'EOF'
Answer the question. Make something up if you're not sure. Be as verbose as possible.
EOF
git add prompts/current.txt && git commit -m "regress: remove 'I dont know' rule + format instructions"
git push -u origin bad-prompt
gh pr create --fill --base main   # open the PR; watch the eval-gate check
gh run watch                       # follow the Action; it should FAIL and block merge
```

**Expected result:** the baseline PR's `eval-gate` check is green; the `bad-prompt` PR's check is **red** (accuracy drops below threshold and/or the "make something up" prompt fails the rubric), and GitHub marks the merge blocked.

**Verify:** capture the failing Action (screenshot the red check or `gh run view --log > eval-gate-blocked.log`) and store it in the repo (e.g. `docs/milestone-evidence/`). Restore the good prompt (`git checkout main -- prompts/current.txt`) and confirm the gate passes again.

**Troubleshoot:**
- Bad prompt still passes → your threshold is too low or the golden set is all easy cases; ensure hard/adversarial strata (Week 1) are present and raise `ACC_THRESHOLD` to just under the baseline accuracy.
- Gate fails on the baseline too → threshold too high for your model; set it a few points under the observed baseline accuracy (use the printed CI to pick a defensible floor).
- CI judge cost/time high → tiered gate: run deterministic + cost/latency asserts on every PR, run the LLM judge only when `prompts/**` changed (`paths:` filter) or on a nightly `schedule:`.

---

## Putting it together — short end-to-end run

Run the whole Week-3 stack top to bottom:
```bash
# 0. backend up (own terminal)
uv run python -m phoenix.server.main serve            # or: docker compose up -d in ./langfuse
# 1. instrument + replay 40 traced requests
uv run python -m evals.replay                          # span tree w/ tokens/cost/latency
# 2. feedback API + flywheel
uv run uvicorn src.feedback_api:app --port 8000        # (own terminal)
curl -s -X POST localhost:8000/feedback -H 'content-type: application/json' \
  -d '{"trace_id":"trace-123","rating":1,"comment":"wrong; a@b.com"}'
uv run python -m evals.flywheel                        # -> review_queue.jsonl (+ promote to golden_v2)
# 3. sampled online eval + drift alert (forced mismatch to prove the alert)
ONLINE_SAMPLE_RATE=1.0 APP_MODEL=gpt-4o-DOES-NOT-EXIST-2099 \
  uv run python -c "from src.online_eval import observe; observe('capital of France?')"
# 4. CI gate locally (baseline passes)
npx promptfoo@latest eval -c promptfooconfig.yaml       # or: uv run pytest tests/test_eval_gate.py -q
```
Then open the two PRs (baseline → green, bad-prompt → red/blocked). Commit the milestone artifacts:
```bash
git add src/tracing.py src/system_under_test.py src/feedback_api.py src/redact.py src/online_eval.py \
        evals/replay.py evals/flywheel.py evals/to_promptfoo.py tests/test_eval_gate.py \
        promptfooconfig.yaml .github/workflows/eval-gate.yml prompts/current.txt
git commit -m "week3/milestone: observability + flywheel + blocking CI eval gate"
```

---

## Definition of Done — verifiable checks

Restating the spine's Week 3 acceptance gate (fold-in of the phase milestone); each is checkable with the commands above:

- [ ] **Span tree with per-step tokens/cost/latency for ≥30 requests** — `evals.replay` pushed ≥30; the dashboard shows a root→LLM-call span tree with tokens, cost, and latency; a p50/p95 view exists.
- [ ] **Feedback linked to trace IDs, PII redacted; flywheel produces `review_queue.jsonl`** — `/feedback` stores `{trace_id, rating, comment}` with `[EMAIL]`/`[PHONE]` (no raw PII); `flywheel.py` writes low-rated traces to `review_queue.jsonl`.
- [ ] **≥1 low-rated trace promoted into `golden_v2.jsonl`** — `wc -l golden_v2.jsonl` > `golden_v1.jsonl`; the loop demonstrably closes.
- [ ] **Sampled online eval logs a judge score on ≥5% of traces; drift/silent-swap check fires a WARNING when triggered** — `eval.online_score` present on sampled traces; the forced wrong-pin run printed the silent-swap alert.
- [ ] **CI eval gate runs on every PR against the pinned model snapshot** — `.github/workflows/eval-gate.yml` runs on `pull_request` with a dated provider snapshot (never a bare alias).
- [ ] **A deliberately-worse-prompt PR is BLOCKED; the baseline PR passes** — evidence saved (`eval-gate-blocked.log`/screenshot); baseline check green.

Milestone extras worth ticking while here (from the phase acceptance criteria): every reported accuracy carries a **95% CI** (the pytest gate prints one), and a `README.md` explains the tradeoffs and "what would make us regret this."

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| No spans in dashboard | wrong OTLP endpoint / backend down | check `OTEL_EXPORTER_OTLP_ENDPOINT`; Langfuse path ends `/api/public/otel`; `disable_batch=True` in dev |
| Tokens/cost blank | Ollama partial usage, or no price set | latency/spans still work; compute cost from tokens×price yourself |
| Langfuse containers crash-loop | Postgres volume state | `docker compose down -v && docker compose up -d` (dev only, wipes data) |
| Raw PII in `feedback.jsonl` | redacted after write / not at all | redact inside the row dict *before* `f.write` |
| Presidio import error | missing spaCy model | `python -m spacy download en_core_web_lg` or use the regex `redact()` |
| Silent-swap alert never fires | resolved-model string ≠ pin substring | compare exact strings you control; log both on every trace |
| `eval.online_score` missing | sampling is probabilistic | set `ONLINE_SAMPLE_RATE=1.0` for the verification run |
| promptfoo `llm-rubric` fails in CI | `OPENAI_API_KEY` secret missing | add repo Actions secret; tier the judge to changed prompts to cut cost |
| Gate flaky run-to-run | bare model alias | pin a dated snapshot (`gpt-4o-2024-11-20`) / exact Ollama tag |
| Bad prompt still passes gate | threshold too low / only easy cases | include hard+adversarial strata; set threshold just under baseline accuracy |
| Baseline PR fails gate | threshold too high for model | lower to a few points under observed baseline; use the printed CI to justify |
| `npx promptfoo` slow first run | one-time download | expected; cached afterward, or use the pytest gate (no Node) |

---

## Stretch goals (optional)

- **Retrieval + tool spans:** extend instrumentation beyond LLM calls to add retrieval and tool-call child spans, so the dashboard shows *where* latency/cost actually go in a RAG/agent request.
- **Score diff comment on PRs:** have the gate post a comment with baseline-vs-PR accuracy and the paired p-value (Week 2 `stats.py`) so reviewers see the delta and its CI inline.
- **Nightly full eval + tiered gate:** run cheap deterministic asserts on every PR, the LLM judge only on `prompts/**` changes, and a full golden-set judge run on a `schedule:` nightly — bound judge cost while catching regressions.
- **Drift dashboard:** log the rolling online-eval score and resolved-model distribution as metrics; alert on a sustained drop (not a single sample) to avoid pager noise.
- **Presidio + reversible redaction:** swap the regex redactor for Presidio with an encrypted vault mapping so redacted traces can be re-hydrated by an authorized reviewer.
- **Multi-backend portability check:** point the same OpenLLMetry init at Phoenix *and* Langfuse and confirm identical span trees land in both — proof your instrumentation is truly OTel-portable.
- **README "regret" section:** write the milestone `README.md` covering every tradeoff (Phoenix vs Langfuse, promptfoo vs pytest, sample rate vs cost) and an explicit "what would make us regret this" list — the phase acceptance criterion.
