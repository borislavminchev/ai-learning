# Week 1 Lab: Build the Model-Agnostic New-Model Smoke-Test Harness

> You build a small, reusable repo — `new-model-smoke-test` — that runs four probing tasks (structured extraction, a tool-calling loop, long-context recall, and multi-step reasoning) against **any** OpenAI-compatible endpoint through a single LiteLLM interface, then prints a comparison table whose headline column is **cost-per-correct-answer** (`$/correct`). This is the artifact you re-run every time a new model ships, so you never have to guess whether it's worth switching. Along the way you enforce notebook-to-production discipline: one task becomes a typed module whose LLM calls are recorded (VCR.py) so `pytest` runs green, offline, and free in CI.
>
> **Read these first (this lab is the hands-on payoff for them):**
> [01 — The Raw-SDK-vs-Framework Decision](../lectures/01-framework-vs-raw-sdk-decision.md) ·
> [02 — Provider SDKs In Depth: The Shapes a Gateway Hides](../lectures/02-provider-sdks-in-depth.md) ·
> [03 — Gateways: One Interface, and What Leaks Through It](../lectures/03-gateways-litellm-and-the-escape-hatch.md) ·
> [04 — The Framework Landscape: Survey to Choose, Not to Adopt](../lectures/04-framework-landscape-survey.md) ·
> [05 — Notebook-to-Production Discipline: Typed Modules and Recorded LLM Calls](../lectures/05-notebook-to-production-discipline.md)
>
> Spine: [../13-frameworks-career.md](../13-frameworks-career.md) (Week 1). Lectures index: [../lectures/00-index.md](../lectures/00-index.md).

**Est. time:** ~9 hrs (+2 hrs optional Next.js/Vercel AI SDK track) · **You will need:** Python 3.11+ with [`uv`](https://docs.astral.sh/uv/), Git, a GitHub account with Actions enabled, and **at least one** of: an OpenAI/Anthropic/Gemini API key **or** a local [Ollama](https://ollama.com) install. **Free/local path:** install Ollama, `ollama pull llama3.1`, and point LiteLLM at its OpenAI-compatible endpoint — you can complete every required step spending **$0**. Hosted keys are optional and cost pennies for this suite.

---

## Before you start (setup)

**1. Tools.** Confirm your toolchain (Git-Bash on Windows, or any POSIX shell / PowerShell):

```bash
python --version      # need 3.11+
uv --version          # https://docs.astral.sh/uv/  (install: pipx install uv, or see docs)
git --version
```

- **Windows note:** install `uv` with `winget install astral-sh.uv` or `pipx install uv`. All shell snippets below assume **Git-Bash**; where a command differs on PowerShell/`cmd`, a *(Windows)* note follows.

**2. Local model (the free path).** Install Ollama, pull a small general model, and confirm the server is up:

```bash
ollama pull llama3.1          # ~4.7 GB; also try qwen2.5:7b or llama3.2:3b if RAM-constrained
ollama list                   # confirm the model is present
curl http://localhost:11434/v1/models   # Ollama exposes an OpenAI-compatible API on :11434
```

- Ollama serves an **OpenAI-compatible** endpoint at `http://localhost:11434/v1`. That is exactly the "any OpenAI-compatible endpoint" claim you're proving. Keep `ollama serve` running (it auto-starts as a background service on most installs).

**3. Optional hosted keys.** If you want to compare against hosted models, have one or more of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` ready. You'll put them in `.env` (never commit it). If you have none, the lab still completes fully against Ollama.

**4. Mental model.** LiteLLM gives you *one* function — `litellm.completion(model=..., messages=...)` — that speaks to OpenAI, Anthropic, Gemini, Bedrock, and Ollama. The lectures explain what that uniformity *hides* (Anthropic content blocks + `cache_control`, OpenAI `reasoning_effort`, Bedrock SigV4 auth). Keep that in mind: this lab deliberately keeps a direct-SDK escape hatch in view (Stretch goal 2).

---

## Step-by-step

### Step 1 — Scaffold the repo

**What:** Create the project skeleton with `uv` and the directory layout.

**Why:** A fixed layout (`src/` for code, `smoke/` for tasks+fixtures+reports) is what makes the harness reusable and the CI wiring predictable. This matches the spine's suggested layout so Week 2 can bolt on cleanly.

**Do it:**

```bash
uv init new-model-smoke-test && cd new-model-smoke-test
uv add litellm pydantic python-dotenv rich pyyaml tiktoken
uv add --dev pytest vcrpy
mkdir -p smoke/tasks smoke/fixtures smoke/reports src tests
touch src/__init__.py smoke/__init__.py smoke/tasks/__init__.py
```

- *(Windows / Git-Bash):* `mkdir -p` and `touch` work in Git-Bash. In PowerShell use `New-Item -ItemType Directory -Force smoke/tasks` etc., and `ni src/__init__.py` to create empty files.

Target layout:

```
new-model-smoke-test/
  src/
    __init__.py
    runner.py          # loads models, runs all tasks, prints + writes the table
    client.py          # thin LiteLLM wrapper + cost/latency accounting
    scoring.py         # small correctness helpers
  smoke/
    __init__.py
    tasks/
      __init__.py
      extraction.py    # structured JSON extraction + gold answers
      toolloop.py      # 2-step tool-calling loop (calculator + fake lookup)
      longctx.py       # needle-in-haystack recall at ~20k tokens
      reasoning.py     # multi-step word problems with numeric answers
    fixtures/          # recorded responses for CI (VCR cassettes)
    reports/latest.md  # generated comparison table
  tests/
    test_extraction.py
  models.yaml          # the list of models to test
  .env                 # OPENAI_API_KEY etc. — NEVER commit
  .gitignore
```

**Expected result:** `uv run python -c "import litellm, pydantic, yaml; print('ok')"` prints `ok`.

**Verify:** `ls src smoke smoke/tasks` shows the files/dirs above.

**Troubleshoot:** If `uv add` fails behind a corporate proxy, set `UV_INDEX_URL` / `HTTPS_PROXY` env vars, or fall back to `python -m venv .venv && pip install ...`.

---

### Step 2 — Ignore secrets and generated churn

**What:** Add a `.gitignore` so keys and volatile artifacts never land in Git.

**Why:** The `.env` holds API keys; committing it is the classic leak. Generated reports change on every run and pollute diffs.

**Do it** — create `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
smoke/reports/latest.md
```

Create `.env` (fill only the keys you actually have; Ollama needs none):

```dotenv
# .env  — never commit
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
# Ollama needs no key. Optional override if not on the default port:
# OLLAMA_API_BASE=http://localhost:11434
```

**Expected result:** `git status` does **not** list `.env`.

**Verify:** `git check-ignore .env` prints `.env`.

**Troubleshoot:** If `.env` was already staged, `git rm --cached .env` before your first commit.

---

### Step 3 — The model-agnostic client (LiteLLM wrapper)

**What:** Write `src/client.py`: one `complete()` function that calls any model and returns a typed `Call` with text, token counts, cost, and latency.

**Why:** This is the single choke point that makes the harness provider-agnostic. Every task calls `complete()` and never touches a provider SDK. Cost and latency are captured here so the report is honest (`litellm.completion_cost` returns `$0` for local models — correct, and the reason the free path works).

**Do it** — `src/client.py`:

```python
# src/client.py
import os, time
from dataclasses import dataclass
import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.drop_params = True  # silently drop params a given provider doesn't support

@dataclass
class Call:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    raw: object  # keep the full response for tool-call parsing

def complete(model: str, messages: list[dict], **kw) -> Call:
    """Call any LiteLLM-supported model. Returns a typed Call with cost+latency."""
    t0 = time.perf_counter()
    r = litellm.completion(model=model, messages=messages, **kw)
    dt = time.perf_counter() - t0
    u = r.usage
    try:
        cost = litellm.completion_cost(completion_response=r) or 0.0
    except Exception:
        cost = 0.0  # unknown/local model -> treat as free
    return Call(
        text=r.choices[0].message.content or "",
        prompt_tokens=getattr(u, "prompt_tokens", 0),
        completion_tokens=getattr(u, "completion_tokens", 0),
        cost_usd=cost,
        latency_s=dt,
        raw=r,
    )
```

Key signatures to keep stable across the repo:

```python
def complete(model: str, messages: list[dict], **kw) -> Call: ...
```

**Expected result:** A quick smoke of the client against Ollama returns text and `cost_usd == 0.0`:

```bash
uv run python -c "from src.client import complete; c=complete('ollama/llama3.1',[{'role':'user','content':'Say hi in one word.'}]); print(repr(c.text), c.cost_usd, round(c.latency_s,2))"
```

**Verify:** You see a one-word reply, `0.0` cost, and a latency in seconds.

**Troubleshoot:**
- `litellm.exceptions.APIConnectionError` against Ollama → is `ollama serve` running? Test `curl http://localhost:11434/v1/models`. If on a non-default host/port, set `OLLAMA_API_BASE`.
- Provider auth errors → the corresponding key is missing/typo'd in `.env`. Ollama should work with no key.

---

### Step 4 — `models.yaml`: the one-line-swap model list

**What:** List the models to test in `models.yaml`. Adding/removing a model is a one-line edit — no code change (a Definition-of-Done requirement).

**Why:** Decoupling "which models" from "what tests" is the whole point of the harness. The runner reads this file.

**Do it** — `models.yaml`:

```yaml
models:
  - ollama/llama3.1                    # free, local — always keep at least one local target
  # Uncomment any you have keys for:
  # - openai/gpt-4o-mini
  # - anthropic/claude-3-5-haiku-latest
  # - gemini/gemini-2.0-flash
```

- LiteLLM's model string is `provider/model`. `ollama/llama3.1` routes to your local server. See LiteLLM provider docs at `docs.litellm.ai` for exact ids per provider.

**Expected result:** The file parses: `uv run python -c "import yaml; print(yaml.safe_load(open('models.yaml'))['models'])"`.

**Verify:** Output is a Python list of your model strings.

**Troubleshoot:** YAML is whitespace-sensitive — use spaces, not tabs. Keep the leading `- ` on each list item.

---

### Step 5 — Scoring helpers

**What:** Write `src/scoring.py` with tiny, deterministic helpers tasks share.

**Why:** Correctness must be *checkable* — that's what lets you compute `$/correct` instead of eyeballing. Deterministic scorers also make CI meaningful.

**Do it** — `src/scoring.py`:

```python
# src/scoring.py
import json, re
from dataclasses import dataclass

@dataclass
class TaskResult:
    name: str
    n_correct: int
    n_total: int
    cost_usd: float
    latencies: list[float]

    @property
    def p50_latency(self) -> float:
        s = sorted(self.latencies)
        return s[len(s) // 2] if s else 0.0

def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of model text (handles ```json fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def extract_number(text: str) -> float | None:
    """Last number in the text — models often restate the answer at the end."""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return float(nums[-1]) if nums else None
```

**Expected result:** Importable helpers.

**Verify:**

```bash
uv run python -c "from src.scoring import extract_json, extract_number; print(extract_json('x {\"a\":1} y'), extract_number('the answer is 42.'))"
```

prints `{'a': 1} 42.0`.

**Troubleshoot:** If a model wraps JSON in prose with braces elsewhere, tighten the regex or ask the model for JSON-only output (Step 6 does this via the prompt).

---

### Step 6 — Task 1: structured extraction

**What:** `smoke/tasks/extraction.py` — feed messy strings, ask for `{name, date, amount}` JSON, score schema-valid rate + field exact-match against gold.

**Why:** Extraction is the most common real LLM job and the fastest way to see if a model follows a schema. It exposes JSON-mode differences a gateway papers over.

**Do it** — `smoke/tasks/extraction.py`:

```python
# smoke/tasks/extraction.py
from src.client import complete
from src.scoring import TaskResult, extract_json

# (input string, gold dict). Keep ~15 cases; 4 shown — add your own messy ones.
CASES = [
    ("Invoice from Acme Corp dated 2025-03-14, total due $1,240.00.",
     {"name": "Acme Corp", "date": "2025-03-14", "amount": 1240.00}),
    ("Hi, this is Beta LLC. Your 01/09/2025 order came to 89 dollars 50 cents.",
     {"name": "Beta LLC", "date": "2025-01-09", "amount": 89.50}),
    ("RECEIPT — Gamma Ltd — 2024-12-01 — amount: 5.00 USD",
     {"name": "Gamma Ltd", "date": "2024-12-01", "amount": 5.00}),
    ("From: Delta Inc <billing@delta.io>  Paid 2025/07/02 : $300",
     {"name": "Delta Inc", "date": "2025-07-02", "amount": 300.00}),
]

PROMPT = (
    "Extract the vendor name, ISO date (YYYY-MM-DD), and amount as a number. "
    "Respond with ONLY a JSON object: "
    '{{"name": str, "date": "YYYY-MM-DD", "amount": number}}. Text: {text}'
)

def run(model: str) -> TaskResult:
    correct, valid, cost, lats = 0, 0, 0.0, []
    for text, gold in CASES:
        c = complete(model, [{"role": "user", "content": PROMPT.format(text=text)}],
                     temperature=0)
        cost += c.cost_usd; lats.append(c.latency_s)
        obj = extract_json(c.text)
        if obj is None:
            continue
        valid += 1
        try:
            if (obj.get("name") == gold["name"]
                    and obj.get("date") == gold["date"]
                    and abs(float(obj.get("amount")) - gold["amount"]) < 0.01):
                correct += 1
        except (TypeError, ValueError):
            pass
    r = TaskResult("extract", correct, len(CASES), cost, lats)
    r.schema_valid_rate = valid / len(CASES)  # attribute used by the CI test
    return r
```

**Expected result:** `run("ollama/llama3.1")` returns a `TaskResult` with `n_total == len(CASES)`.

**Verify:**

```bash
uv run python -c "from smoke.tasks.extraction import run; r=run('ollama/llama3.1'); print(r.n_correct, r.n_total, r.schema_valid_rate)"
```

**Troubleshoot:** Small local models sometimes emit prose before JSON — the `extract_json` regex handles a fenced block. If `schema_valid_rate` is low, that's a *real finding* about the model, not a bug. Grow `CASES` to ~15 for a stable signal.

---

### Step 7 — Task 2: the tool-calling loop

**What:** `smoke/tasks/toolloop.py` — register `calculator` and a fake `get_price` tool, ask a question needing both, run the tool loop, score numeric correctness.

**Why:** Tool calling is where provider API shapes diverge most (OpenAI `tool_calls` + `role:"tool"` with args as a JSON *string*; Anthropic `tool_use`/`tool_result` content blocks with parsed args). LiteLLM normalizes to the OpenAI shape — this task proves the normalized loop works end-to-end.

**Do it** — `smoke/tasks/toolloop.py`:

```python
# smoke/tasks/toolloop.py
import json
from src.client import complete
from src.scoring import TaskResult, extract_number

PRICES = {"widget": 3.0, "gadget": 4.5}

def get_price(item: str) -> float:
    return PRICES.get(item.lower().strip(), 0.0)

def calculator(expression: str) -> float:
    return float(eval(expression, {"__builtins__": {}}, {}))  # sandboxed eval; demo only

TOOLS = [
    {"type": "function", "function": {
        "name": "get_price", "description": "Unit price in USD for an item.",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}},
                       "required": ["item"]}}},
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
]
DISPATCH = {"get_price": get_price, "calculator": calculator}

# question -> gold numeric answer
CASES = [("What is the total cost of 3 widgets and 2 gadgets?", 18.0)]

def run(model: str, max_turns: int = 4) -> TaskResult:
    correct, cost, lats = 0, 0.0, []
    for question, gold in CASES:
        messages = [{"role": "user", "content": question}]
        answer = None
        for _ in range(max_turns):
            c = complete(model, messages, tools=TOOLS, temperature=0)
            cost += c.cost_usd; lats.append(c.latency_s)
            msg = c.raw.choices[0].message
            calls = getattr(msg, "tool_calls", None)
            if not calls:
                answer = extract_number(c.text)
                break
            messages.append(msg.model_dump())  # echo the assistant tool-call turn back
            for tc in calls:
                args = json.loads(tc.function.arguments)
                result = DISPATCH[tc.function.name](**args)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "name": tc.function.name, "content": str(result)})
        if answer is not None and abs(answer - gold) < 0.01:
            correct += 1
    return TaskResult("tools", correct, len(CASES), cost, lats)
```

**Expected result:** With a tool-capable model, `run(...)` returns `1/1`.

**Verify:** `uv run python -c "from smoke.tasks.toolloop import run; r=run('ollama/llama3.1'); print(r.n_correct, r.n_total)"`

**Troubleshoot:**
- Not every local model supports tool calling. `llama3.1` does; if yours doesn't, LiteLLM raises or returns no `tool_calls` — switch to a tool-capable model (`qwen2.5`, or a hosted `gpt-4o-mini`). This is itself a useful smoke-test result.
- `eval` is used only for a trusted, sandboxed calculator here — never `eval` untrusted input in real code.

---

### Step 8 — Task 3: long-context recall (needle-in-haystack)

**What:** `smoke/tasks/longctx.py` — bury a unique fact in ~20k tokens of filler, ask for it, score exact recall.

**Why:** Advertised context ≠ usable context (the RULER / "lost in the middle" problem). This is the cheapest way to test whether a model actually *reads* a long prompt.

**Do it** — `smoke/tasks/longctx.py`:

```python
# smoke/tasks/longctx.py
from src.client import complete
from src.scoring import TaskResult

NEEDLE = "The access code is PLUM-7741."
FILLER = ("The quarterly logistics review noted steady throughput. " * 8 + "\n")

def _haystack(target_tokens: int = 20000) -> str:
    # ~4 chars/token heuristic; place the needle in the middle (the hardest spot).
    approx_chars = target_tokens * 4
    body = FILLER * (approx_chars // len(FILLER))
    mid = len(body) // 2
    return body[:mid] + "\n" + NEEDLE + "\n" + body[mid:]

def run(model: str) -> TaskResult:
    doc = _haystack()
    prompt = (f"{doc}\n\nQuestion: What is the access code? "
              "Answer with only the code.")
    c = complete(model, [{"role": "user", "content": prompt}], temperature=0)
    ok = "PLUM-7741" in c.text.upper()
    return TaskResult("longctx", int(ok), 1, c.cost_usd, [c.latency_s])
```

**Expected result:** Hosted long-context models return `1/1`; small local models may return `0/1` — an honest, useful signal.

**Verify:** `uv run python -c "from smoke.tasks.longctx import run; r=run('ollama/llama3.1'); print(r.n_correct, r.n_total)"`

**Troubleshoot:**
- Ollama's default context window may be smaller than 20k and will *silently truncate*. Raise it with `num_ctx` via extra params, or lower `target_tokens` for local runs: `complete(model, msgs, num_options={"num_ctx": 32768})` is model-dependent — simplest is to test a smaller haystack locally and the full 20k against hosted models. Note the difference in your README (this is the "usable context" lesson made concrete).
- Use `tiktoken` to measure actual token counts if you want precision instead of the 4-char heuristic.

---

### Step 9 — Task 4: multi-step reasoning

**What:** `smoke/tasks/reasoning.py` — 8 short multi-step word problems with known numeric answers; score exact match.

**Why:** Reasoning separates models that "sound right" from those that *are* right, and it's where cost-per-correct diverges sharply (a cheap model that fails half the problems is expensive per correct answer).

**Do it** — `smoke/tasks/reasoning.py`:

```python
# smoke/tasks/reasoning.py
from src.client import complete
from src.scoring import TaskResult, extract_number

# (problem, numeric answer) — add up to 8. Keep answers unambiguous integers/decimals.
CASES = [
    ("A train travels 60 km in 1.5 hours. What is its average speed in km/h?", 40),
    ("If 3 pens cost $2.40, how much do 7 pens cost in dollars?", 5.60),
    ("A rectangle is 8 by 5. What is its perimeter?", 26),
    ("15% of 240 is what number?", 36),
    ("Sum of the first 10 positive integers?", 55),
    ("A shirt is $50 with a 20% discount, then 10% tax on the discounted price. Final price?", 44),
    ("If x + 7 = 19, what is 3x?", 36),
    ("A car uses 6 liters per 100 km. How many liters for 250 km?", 15),
]
PROMPT = "Solve step by step, then end with 'Answer: <number>'.\n\n{q}"

def run(model: str) -> TaskResult:
    correct, cost, lats = 0, 0.0, []
    for q, gold in CASES:
        c = complete(model, [{"role": "user", "content": PROMPT.format(q=q)}], temperature=0)
        cost += c.cost_usd; lats.append(c.latency_s)
        got = extract_number(c.text)
        if got is not None and abs(got - gold) < 0.01:
            correct += 1
    return TaskResult("reason", correct, len(CASES), cost, lats)
```

**Expected result:** A `TaskResult` with `n_total == 8`.

**Verify:** `uv run python -c "from smoke.tasks.reasoning import run; r=run('ollama/llama3.1'); print(r.n_correct, '/', r.n_total)"`

**Troubleshoot:** If a model rambles past its "Answer:" line, `extract_number` takes the *last* number — usually correct, but verify a couple by hand the first time.

---

### Step 10 — The runner: loop models × tasks, compute `$/correct`, write the report

**What:** `src/runner.py` runs all four tasks for every model in `models.yaml`, aggregates cost and correctness, and writes `smoke/reports/latest.md` with the `$/correct` and p50-latency columns.

**Why:** This is the one command you run every time a model drops. `$/correct = total_cost / total_correct` is the single decision number — cheap-per-token means nothing if the model is wrong.

**Do it** — `src/runner.py`:

```python
# src/runner.py
import yaml
from rich.console import Console
from rich.table import Table
from smoke.tasks import extraction, toolloop, longctx, reasoning

TASKS = [("extract", extraction), ("tools", toolloop),
         ("longctx", longctx), ("reason", reasoning)]

def run_model(model: str) -> dict:
    results, cost, lats = {}, 0.0, []
    for name, mod in TASKS:
        r = mod.run(model)
        results[name] = (r.n_correct, r.n_total)
        cost += r.cost_usd
        lats.extend(r.latencies)
    total_correct = sum(nc for nc, _ in results.values())
    dpc = (cost / total_correct) if total_correct else float("inf")
    p50 = sorted(lats)[len(lats)//2] if lats else 0.0
    return {"model": model, "results": results, "cost": cost,
            "dollars_per_correct": dpc, "p50": p50}

def main():
    models = yaml.safe_load(open("models.yaml"))["models"]
    rows = [run_model(m) for m in models]

    console = Console()
    table = Table(title="new-model smoke test")
    for col in ["model", "extract", "tools", "longctx", "reason", "$/correct", "p50 lat"]:
        table.add_column(col)
    lines = ["| model | extract | tools | longctx | reason | $/correct | p50 lat |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = [f"{r['results'][n][0]}/{r['results'][n][1]}"
                 for n in ("extract", "tools", "longctx", "reason")]
        dpc = "$0.0000" if r["dollars_per_correct"] == 0 else \
              ("n/a" if r["dollars_per_correct"] == float("inf")
               else f"${r['dollars_per_correct']:.4f}")
        p50 = f"{r['p50']:.1f}s"
        table.add_row(r["model"], *cells, dpc, p50)
        lines.append(f"| {r['model']} | " + " | ".join(cells) + f" | {dpc} | {p50} |")
    console.print(table)

    with open("smoke/reports/latest.md", "w", encoding="utf-8") as f:
        f.write("# Smoke-test report\n\n" + "\n".join(lines) + "\n")
    console.print("[green]wrote smoke/reports/latest.md[/green]")

if __name__ == "__main__":
    main()
```

**Do it — run it:**

```bash
uv run python -m src.runner
```

**Expected result:** A Rich table in your terminal and a Markdown file at `smoke/reports/latest.md` like:

```
| model            | extract | tools | longctx | reason | $/correct | p50 lat |
|------------------|---------|-------|---------|--------|-----------|---------|
| ollama/llama3.1  | 12/15   |  1/1  |  0/1    |  4/8   | $0.0000   | 3.1s    |
```

(With a hosted key uncommented in `models.yaml`, you'll see a second row with a nonzero `$/correct` — e.g. `openai/gpt-4o-mini … $0.0009 … 0.8s`.)

**Verify:** `cat smoke/reports/latest.md` shows the table with a `$/correct` and a `p50 lat` column.

**Troubleshoot:**
- `ModuleNotFoundError: smoke` → run from the repo root with `-m` (i.e. `uv run python -m src.runner`), and make sure the `__init__.py` files from Step 1 exist.
- One task erroring shouldn't kill the run — wrap `mod.run(model)` in try/except and record `0/n` if you want resilience across many models.

---

### Step 11 — Notebook-to-production discipline: typed module + recorded test

**What:** Make the extraction task testable offline. Add a Pydantic response model, then record the LLM calls with VCR.py so `pytest` replays them for free.

**Why:** This is the lab's production-discipline requirement (Definition of Done: green `pytest` offline, no API key in CI). Recorded cassettes make LLM tests deterministic and cost-free — the difference between a notebook and a shippable module.

**Do it — (a) add a typed model** (append to `smoke/tasks/extraction.py`):

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    name: str
    date: str      # YYYY-MM-DD
    amount: float
```

Use it to validate in `run()` (replace the ad-hoc dict checks with `Invoice(**obj)` inside a `try/except ValidationError`) — this gives you real schema validation and a clean `schema_valid_rate`.

**Do it — (b) record the calls** in `tests/test_extraction.py`:

```python
# tests/test_extraction.py
import vcr

my_vcr = vcr.VCR(
    cassette_library_dir="smoke/fixtures",
    record_mode="once",                      # record first run, replay thereafter
    filter_headers=["authorization", "x-api-key", "api-key"],  # never store secrets
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)

@my_vcr.use_cassette("extraction.yaml")
def test_extraction_schema_valid():
    from smoke.tasks.extraction import run
    results = run("openai/gpt-4o-mini")   # or ollama/llama3.1 for the free path
    assert results.schema_valid_rate >= 0.9
```

**Do it — record once, then replay:**

```bash
# First run performs REAL calls and writes smoke/fixtures/extraction.yaml (needs a key OR Ollama running):
uv run pytest tests/test_extraction.py -q
# Every subsequent run replays the cassette — offline, free, deterministic:
uv run pytest -q
```

Commit the cassette: `git add smoke/fixtures/extraction.yaml`.

**Expected result:** First run records and passes; later runs pass with **no network and no key**.

**Verify:** Turn off networking (or unset keys / stop Ollama) and re-run `uv run pytest -q` — it still passes, reading `smoke/fixtures/extraction.yaml`.

**Troubleshoot:**
- **Secrets in the cassette:** open `smoke/fixtures/extraction.yaml` and confirm no `authorization`/`x-api-key` values — `filter_headers` should have redacted them. Never commit a cassette that contains a key.
- **VCR + local models:** Ollama calls are HTTP too, so VCR records them the same way. If you recorded against Ollama, the test replays without Ollama running.
- **Cassette drift:** if you change prompts/cases, delete the cassette and re-record (`record_mode="once"` won't overwrite an existing one). Re-record *intentionally*, never in CI.

---

### Step 12 — Wire up GitHub Actions (green CI, no API key)

**What:** Add `.github/workflows/ci.yml` that installs deps and runs `pytest` on push — using the committed cassettes, so it needs no secret.

**Why:** A green CI badge that costs $0 and never flakes is the payoff of Step 11. It's a Definition-of-Done item and the thing a hiring manager notices.

**Do it** — `.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5      # installs uv
      - run: uv sync --dev
      - run: uv run pytest -q            # replays cassettes; no API key needed
```

**Do it — push:**

```bash
git add -A && git commit -m "smoke-test harness: client, 4 tasks, runner, recorded CI"
git branch -M main
git remote add origin <your-repo-url>   # create the repo on GitHub first
git push -u origin main
```

- *(Windows / Git-Bash):* these Git commands are identical in Git-Bash.

**Expected result:** The Actions tab shows a green run of `ci`.

**Verify:** Open the repo's **Actions** tab → the latest run is green; the log shows `pytest` passing with no key configured.

**Troubleshoot:**
- Red because a test tried a live call → the cassette wasn't committed or `record_mode` isn't `once`. Confirm `smoke/fixtures/extraction.yaml` is in the repo.
- `uv sync` fails → ensure `pyproject.toml` and `uv.lock` are committed (both are created/updated by `uv add`).

---

### Step 13 — README paragraph: the rubric-based decision

**What:** Add a short README section stating, for **one** real task, whether you'd use a raw SDK or a framework — and why, using the rubric.

**Why:** The whole point of Week 1 is *judgment*, not tooling. This paragraph is the Definition-of-Done proof that you can apply the rubric (velocity, breaking-change cadence, abstraction cost, ejectability, dependency weight, license, maintainer risk — see [lecture 01](../lectures/01-framework-vs-raw-sdk-decision.md)).

**Do it** — add to `README.md`, for example:

```markdown
## Raw SDK vs framework: the extraction task

For structured extraction I'd use **raw provider SDK + Instructor/Pydantic**, not a
chain framework. Rubric read: the task is ~40 lines, so *abstraction cost* and
*dependency weight* outweigh any velocity gain; *breaking-change cadence* on chain
frameworks is high and I need *ejectability* to debug the exact prompt sent. I'd only
reach for a framework (LangGraph) once this becomes a multi-step stateful workflow
with retries and human-in-the-loop — i.e. when it removes *repeated* work, not 40 lines.
```

**Expected result:** A README with a defensible, rubric-anchored paragraph.

**Verify:** The paragraph names at least three rubric axes and states the escape hatch.

**Troubleshoot:** If you can't justify it in the rubric's terms, re-read lecture 01 — the axes *are* the answer.

---

## Putting it together — a short end-to-end run

From a clean checkout, the whole harness runs in a few commands:

```bash
# 0) local model up (free path)
ollama serve &                 # if not already running as a service
ollama pull llama3.1

# 1) install
uv sync --dev

# 2) run all four tasks against every model in models.yaml, write the report
uv run python -m src.runner
cat smoke/reports/latest.md    # <- $/correct + p50 latency table

# 3) deterministic offline tests (replays recorded cassettes)
uv run pytest -q

# 4) add a brand-new model = ONE line in models.yaml, then re-run step 2
echo '  - openai/gpt-4o-mini' >> models.yaml   # (only if you have a key)
uv run python -m src.runner
```

That's the loop you'll repeat for years: a model drops → one line in `models.yaml` → `uv run python -m src.runner` → read `$/correct` → decide. (Week 2 wraps this in a `make smoke MODEL=<id>` target and a `RUNBOOK.md`.)

---

## Definition of Done

Restating the spine's Week 1 checklist as things you can literally verify:

- [ ] **`uv run python -m src.runner` runs all four tasks against ≥3 models (incl. one local Ollama) and writes `smoke/reports/latest.md`.** → `cat smoke/reports/latest.md` shows ≥3 rows, one being `ollama/...`.
- [ ] **The report has a `$/correct` column and a p50 latency column from real usage/cost data.** → both columns present and populated (local row shows `$0.0000`).
- [ ] **Swapping a model is a one-line edit in `models.yaml` — no code change.** → add/remove a line, re-run, table changes; `git diff` touches only `models.yaml`.
- [ ] **At least one task runs through a local OpenAI-compatible endpoint (Ollama).** → an `ollama/...` row exists with nonzero correctness on at least one task.
- [ ] **`pytest` runs green offline via committed VCR cassettes (no API key).** → disable network / unset keys, `uv run pytest -q` still passes.
- [ ] **A GitHub Actions run is green on a push.** → Actions tab shows a green `ci` run with no secret configured.
- [ ] **A one-paragraph README section gives a rubric-based raw-SDK-vs-framework call for one real task.** → the paragraph from Step 13 exists and names ≥3 rubric axes.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `APIConnectionError` on `ollama/*` | `ollama serve` not running / wrong port | `curl http://localhost:11434/v1/models`; set `OLLAMA_API_BASE` if non-default |
| `AuthenticationError` on a hosted model | missing/typo'd key in `.env` | check `.env`; `load_dotenv()` runs in `client.py`; Ollama needs no key |
| `cost_usd` is `0.0` for a hosted model | LiteLLM lacks pricing for that model id | update `litellm`; verify the exact model string in `docs.litellm.ai` |
| Local `longctx` always `0/1` | Ollama silently truncated the 20k prompt (small `num_ctx`) | raise `num_ctx`, or test a smaller haystack locally + full size on hosted; note the gap (usable ≠ advertised context) |
| No `tool_calls` returned | model doesn't support tools | use a tool-capable model (`qwen2.5`, `gpt-4o-mini`); record it as a smoke-test finding |
| `ModuleNotFoundError: smoke` / `src` | wrong CWD or missing `__init__.py` | run from repo root with `uv run python -m src.runner`; ensure the `__init__.py` files exist |
| `pytest` makes live calls in CI | cassette not committed / wrong `record_mode` | commit `smoke/fixtures/*.yaml`; keep `record_mode="once"` |
| Secrets appear in a cassette | `filter_headers` missing | add `filter_headers=["authorization","x-api-key","api-key"]`; delete + re-record |
| YAML won't parse | tabs instead of spaces | use spaces; keep `- ` on each item |
| *(Windows)* `mkdir -p` / `touch` fail | using PowerShell, not Git-Bash | use Git-Bash, or PowerShell equivalents (`New-Item`) |

---

## Stretch goals (optional)

1. **Provider fallback & cost tracking via a LiteLLM Router / proxy.** Configure `litellm.Router` (or run the LiteLLM proxy) with a fallback list so a failed primary model auto-retries on a secondary, and log per-request spend. See `docs.litellm.ai`.
2. **Direct-SDK escape hatch (the lesson made real).** Add `smoke/tasks/caching.py` that calls **Anthropic's** SDK directly with `cache_control` breakpoints, or **OpenAI's** with `reasoning_effort` — features LiteLLM may not expose. Measure the cached-token cost drop. This proves you know *what the gateway hides* (lectures 02–03).
3. **Token-accurate long-context.** Replace the 4-char heuristic in `longctx.py` with real `tiktoken` counts, and sweep the needle position (start / middle / end) to reproduce "lost in the middle."
4. **Add reasoning-effort / thinking rows.** For models that support it, run `reasoning.py` at low vs high reasoning effort and add both to the table — watch `$/correct` move.
5. **Optional production-frontend track (~2 hrs):** scaffold a Next.js app with the **Vercel AI SDK** (`npx create-next-app`, `npm i ai @ai-sdk/openai`) and stream one smoke-test task to the browser with `streamText`. Then build the *same* feature in **Streamlit** (`uv add streamlit`, ~20 lines) and note in your README what each medium is good for (prototyping UI vs production frontend). See `sdk.vercel.ai` and `docs.streamlit.io`.
