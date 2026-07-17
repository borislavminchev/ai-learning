# Week 1 Lab: Build a Bounded ReAct Agent on the Raw SDK

> This week you hand-write a tool-using **ReAct agent** in ~150 lines on the **raw Anthropic SDK** — no LangChain, no LlamaIndex, no CrewAI. The point is to *feel* the control loop: an agent is a `while` loop around an LLM that emits tool-call requests, your harness runs the tools, and you feed results back until the model is done. You will wire in **four independent budgets** (max-steps, wall-clock, token, dollar) plus a **kill switch**, turn every tool failure into an **observation** the model recovers from, and emit a **structured per-step trace** you can read back to debug the run. A free **Ollama** local path is given so you can iterate at zero cost.
>
> Read these lectures first — this guide assumes them:
> - [../lectures/01-the-agent-loop.md](../lectures/01-the-agent-loop.md) — perceive → plan → act → observe; why a loop beats a prompt
> - [../lectures/02-react-native-tool-calling.md](../lectures/02-react-native-tool-calling.md) — structured `tool_use` blocks, not regex-scraped `Action:/Observation:` text
> - [../lectures/03-errors-as-observations.md](../lectures/03-errors-as-observations.md) — tools return `ERROR: …` strings and never raise
> - [../lectures/04-budgets-and-kill-switches.md](../lectures/04-budgets-and-kill-switches.md) — defense in depth against runaway loops
> - [../lectures/05-determinism-and-tracing.md](../lectures/05-determinism-and-tracing.md) — pin the model, force `tool_choice`, and log every step

**Est. time:** ~9 hrs · **You will need:** Python 3.10+, a terminal (Git-Bash on Windows), and **one** of:
- **Paid path (default):** an Anthropic API key with access to `claude-opus-4-8`. Budget is tiny — the whole lab costs well under $1.
- **Free/local path:** [Ollama](https://ollama.com) running `llama3.1` or `qwen2.5` on CPU. No API key, no GPU, no spend. Every step below works on either backend; the Ollama swap is [Step 6](#step-6--free-local-path-run-the-same-loop-on-ollama).

---

## Before you start (setup)

**What:** Create the project folder, an isolated virtual environment, and install the two dependencies (`anthropic`, `httpx`).

**Why:** A venv keeps the SDK versions pinned to this lab and off your system Python. `httpx` is what the `web_fetch` tool uses to make HTTP requests. `python-dotenv` loads your API key from `.env` so it never lives in your shell history or the code.

**Do it:**

```bash
mkdir agent-lab && cd agent-lab
python -m venv .venv

# Activate the venv:
source .venv/Scripts/activate     # Windows (Git-Bash)
# source .venv/bin/activate       # macOS / Linux

pip install "anthropic>=0.40" httpx python-dotenv
```

Create the sandbox file the `read_file` tool will read, and the `.env` with your key:

```bash
mkdir -p sandbox
printf 'Project Apollo notes\nBudget line item: 42 units at 17 dollars each.\nOwner: Bo.\n' > sandbox/notes.txt

printf 'ANTHROPIC_API_KEY=sk-ant-...\n' > .env      # paste your real key
printf '.env\n.venv/\ntrace.jsonl\nKILL\n__pycache__/\n' > .gitignore
```

**Expected result:** `pip list` shows `anthropic`, `httpx`, `python-dotenv`. `cat sandbox/notes.txt` prints the three lines. Your target folder layout:

```
agent-lab/
  .env                 # ANTHROPIC_API_KEY=sk-ant-...
  agent.py             # ~150 lines: the loop + 4 budgets + kill switch  (Step 4)
  tools.py             # calculator, web_fetch, read_file                (Step 2)
  trace.py             # structured per-step logger -> trace.jsonl       (Step 3)
  sandbox/notes.txt    # a file for read_file to read                    (done above)
  KILL                 # (absent normally) touch this to abort a run
```

**Verify:**
```bash
python -c "import anthropic, httpx, dotenv; print('deps ok')"
```
Prints `deps ok`.

**Troubleshoot:**
- `source: .venv/Scripts/activate: No such file` on Windows → you probably ran the macOS line; Windows venvs put activate under `Scripts/`, not `bin/`.
- `python: command not found` → try `python3`. Confirm `python --version` is ≥ 3.10 (the walrus-free code here runs on 3.9 too, but `str.removeprefix`-style helpers and modern typing assume 3.10+).
- Don't have a key yet? Skip to [Step 6](#step-6--free-local-path-run-the-same-loop-on-ollama) and do the whole lab on Ollama for free, then come back.

---

## Step-by-step

### Step 1 — Understand the loop you are about to build (no code)

**What:** Fix the shape of the loop in your head before writing it.

**Why:** Everything else is detail. The one idea (lecture 01): **the model never runs your tools.** It emits a *request* (`stop_reason == "tool_use"` with `tool_use` blocks); your harness executes the matching Python function and feeds the result back as a `tool_result` block. You repeat until the model stops asking for tools and returns text.

**Do it:** Trace this pseudocode by hand for the task *"Read notes.txt, then compute 17*23":*

```
messages = [user task]
loop up to max_steps:
    check budgets + KILL file        # ← every iteration, before spending anything
    resp = model(messages, tools)    # model decides: call a tool, or finish
    if resp is NOT a tool request:   # stop_reason != "tool_use"
        return resp.text             # DONE
    append assistant turn (the tool_use blocks)
    for each tool_use block:
        obs = DISPATCH[name](**args) # run the real Python function
        append a matching tool_result block (is_error if obs starts "ERROR:")
    append the tool_results as one user turn
return "ABORTED: max-steps budget"   # loop fell through
```

**Expected result:** You can name the two messages you must append per tool round (assistant `tool_use` turn, then user `tool_result` turn) and their order.

**Verify:** Answer self-check #2 from the spine out loud: *given a `tool_use` response, what two messages must you append and in what order, and what happens if a `tool_use_id` goes unanswered?* (Answer: assistant turn with the `tool_use` blocks, then a user turn whose content is the matching `tool_result` blocks; every `tool_use_id` must be answered or the next API call 400s.)

**Troubleshoot:** If "the model runs the tool" still feels natural, re-read lecture 02 — that misconception is the #1 source of broken agent loops.

---

### Step 2 — `tools.py`: three tools that return a string and NEVER raise

**What:** Write the calculator (AST-eval, no `eval()`), `web_fetch` (scheme validation + truncation), and `read_file` (sandbox path-traversal guard). Each returns a `str`; on any failure it returns an `"ERROR: …"` string instead of raising.

**Why:** This is **errors-as-observations** (lecture 03). A tool that raises kills the loop with a traceback. A tool that *returns* `ERROR: …` lets the model see the failure and retry with different args or give up gracefully — the difference between a demo and something you'd run unattended. Using `ast` instead of `eval()` means the calculator can't execute arbitrary code; `startswith(("http://","https://"))` blocks `file://`/`ftp://` SSRF vectors; `is_relative_to(SANDBOX)` after `resolve()` blocks `../../etc/passwd` traversal.

**Do it:** Create `tools.py`:

```python
import ast, operator, pathlib
import httpx

SANDBOX = pathlib.Path("sandbox").resolve()
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}

def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression via AST (no eval())."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp):
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"ERROR: cannot evaluate {expression!r}: {e}"

def web_fetch(url: str) -> str:
    """Fetch text from an http(s) URL, truncated to 4000 chars."""
    if not url.startswith(("http://", "https://")):
        return "ERROR: url must start with http:// or https://"
    try:
        r = httpx.get(url, timeout=10, follow_redirects=True)
        r.raise_for_status()
        return r.text[:4000]
    except Exception as e:
        return f"ERROR: fetch failed: {e}"

def read_file(name: str) -> str:
    """Read a file from ./sandbox by name; block path traversal."""
    try:
        p = (SANDBOX / name).resolve()
        if not p.is_relative_to(SANDBOX):        # blocks ../ traversal & absolute paths
            return "ERROR: path escapes sandbox"
        return p.read_text(encoding="utf-8")[:4000]
    except Exception as e:
        return f"ERROR: {e}"
```

**Expected result:** Importable module with three functions.

**Verify:** Exercise every branch — happy path, error path, and the security guards — from the REPL:

```bash
python - <<'PY'
from tools import calculator, web_fetch, read_file
print(calculator("17*23"))              # 391
print(calculator("1/0"))                # ERROR: cannot evaluate '1/0': division by zero
print(calculator("__import__('os')"))   # ERROR: ... unsupported expression   (eval() would have run it!)
print(read_file("notes.txt")[:20])      # Project Apollo note
print(read_file("../.env"))             # ERROR: path escapes sandbox
print(web_fetch("file:///etc/passwd"))  # ERROR: url must start with http:// or https://
print(web_fetch("https://example.com")[:15])  # <!doctype html   (or ERROR: if offline)
PY
```

Every line returns a string; **none raises**. That "none raises" property is the whole point.

**Troubleshoot:**
- `AttributeError: 'PosixPath' object has no attribute 'is_relative_to'` → you're on Python < 3.9. Upgrade, or replace the guard with `if SANDBOX not in p.parents and p != SANDBOX:`.
- `web_fetch` returns `ERROR: fetch failed` → you're offline; that's fine, the calculator + read_file tools carry the required 2-tool DoD task. You'll use the error deliberately in [Step 8](#step-8--exercise-2-recover-from-a-broken-tool-call-doda-4).
- Calculator returns a value for `__import__(...)` → you used `eval()` somewhere; the AST walker must reject any node type that isn't a constant/BinOp/UnaryOp.

---

### Step 3 — `trace.py`: one structured JSON line per step

**What:** A `log_step(...)` function that appends one JSON object per step to `trace.jsonl` (and prints a one-line human summary).

**Why:** Without a per-step log you cannot answer *"why did it do that?"* (lecture 05). One record per step — index, thought, tool, args, truncated observation, tokens in/out, cumulative dollars — is the seed of the observability you formalize in later weeks. JSONL (one JSON object per line) is append-friendly and trivially greppable.

**Do it:** Create `trace.py`:

```python
import json, time

def log_step(step, thought, tool, args, obs, usage, dollars, is_error=False):
    rec = {
        "ts": round(time.time(), 3),
        "step": step,
        "thought": (thought or "")[:200],
        "tool": tool,
        "args": args,
        "observation": (obs or "")[:200],
        "is_error": is_error,
        "tok_in": usage.input_tokens,
        "tok_out": usage.output_tokens,
        "cum_dollars": round(dollars, 5),
    }
    with open("trace.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[step {step}] tool={tool} in={rec['tok_in']} out={rec['tok_out']} ${rec['cum_dollars']}")
    return rec
```

**Expected result:** Importable `log_step`. Note the signature takes a `usage` object with `.input_tokens` / `.output_tokens` — the Anthropic `resp.usage` shape. (The Ollama path in Step 6 wraps its counts in a tiny shim with the same two attributes, so `trace.py` needs no changes.)

**Verify:**
```bash
python - <<'PY'
from types import SimpleNamespace
from trace import log_step
u = SimpleNamespace(input_tokens=100, output_tokens=20)
log_step(1, "thinking...", "calculator", {"expression":"17*23"}, "391", u, 0.001)
PY
cat trace.jsonl
```
Prints a `[step 1] …` line and appends one JSON object to `trace.jsonl`. Then `rm trace.jsonl` — the real run should start with a fresh file.

**Troubleshoot:** `TypeError: Object of type ... is not JSON serializable` → `args` here are plain dicts from the model's `tool_use.input`, which are always JSON-safe. If you pass something exotic, wrap with `str(args)`.

---

### Step 4 — `agent.py`: the loop + four budgets + kill switch

**What:** The core file. Accumulate `messages`, call the model with the tool schemas, on `stop_reason == "tool_use"` dispatch each block through a `DISPATCH` map and append matching `tool_result` blocks (set `is_error` on `ERROR:` observations), else return the final text. Enforce four **independent** budgets plus a KILL-file check at the top of every iteration, each returning a **distinct** `ABORTED: <reason>`.

**Why:** The budgets are *defense in depth* (lecture 04): max-steps catches oscillation, wall-clock catches slow tools, token catches context bloat, dollar is the one finance cares about, and the KILL file is the human's emergency stop mid-run. They're independent because a single looping tool can blow your dollar budget long before it hits the step cap. The DISPATCH map is how native tool calling stays clean — no `if name == ...` ladder.

**Do it:** Create `agent.py`:

```python
import os, time, pathlib
from dotenv import load_dotenv
from anthropic import Anthropic
from tools import calculator, web_fetch, read_file
from trace import log_step

load_dotenv()

MODEL = "claude-opus-4-8"              # PINNED version = a determinism lever
PRICE_IN, PRICE_OUT = 5 / 1e6, 25 / 1e6   # $/token for opus-4-8 ($5 / $25 per 1M)
KILL_FILE = pathlib.Path("KILL")

TOOLS = [
    {"name": "calculator", "description": "Evaluate an arithmetic expression.",
     "input_schema": {"type": "object",
                      "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]}},
    {"name": "web_fetch", "description": "Fetch text from an http(s) URL.",
     "input_schema": {"type": "object",
                      "properties": {"url": {"type": "string"}},
                      "required": ["url"]}},
    {"name": "read_file", "description": "Read a file from the sandbox by name.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"}},
                      "required": ["name"]}},
]
DISPATCH = {"calculator": calculator, "web_fetch": web_fetch, "read_file": read_file}


def run(task, *, max_steps=8, max_seconds=60, max_tokens_budget=50_000,
        max_dollars=0.25, tool_choice=None):
    client = Anthropic()
    messages = [{"role": "user", "content": task}]
    start, tok_in, tok_out, dollars = time.time(), 0, 0, 0.0

    for step in range(1, max_steps + 1):
        # --- budgets + kill switch, checked BEFORE we spend anything ---
        if KILL_FILE.exists():
            return "ABORTED: kill switch"
        if time.time() - start > max_seconds:
            return "ABORTED: wall-clock budget"
        if tok_in + tok_out > max_tokens_budget:
            return "ABORTED: token budget"
        if dollars > max_dollars:
            return "ABORTED: dollar budget"

        kwargs = dict(model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages)
        if tool_choice:                       # e.g. {"type":"tool","name":"calculator"}
            kwargs["tool_choice"] = tool_choice
        resp = client.messages.create(**kwargs)

        tok_in += resp.usage.input_tokens
        tok_out += resp.usage.output_tokens
        dollars = tok_in * PRICE_IN + tok_out * PRICE_OUT

        thought = "".join(b.text for b in resp.content if b.type == "text")
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if resp.stop_reason != "tool_use":    # model is done — return final text
            log_step(step, thought, None, None, None, resp.usage, dollars)
            return thought

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in tool_uses:                  # native tool calling via the DISPATCH map
            obs = DISPATCH[tu.name](**tu.input)      # tool returns a string, never raises
            is_err = obs.startswith("ERROR:")
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": obs, "is_error": is_err})
            log_step(step, thought, tu.name, tu.input, obs, resp.usage, dollars, is_err)
        messages.append({"role": "user", "content": results})

    return "ABORTED: max-steps budget"


if __name__ == "__main__":
    print(run("Read notes.txt from the sandbox, then compute 17*23 and tell me both results."))
```

**Expected result:** `agent.py` is ~90 lines — comfortably under the ~150-line DoD ceiling. It imports cleanly:
```bash
python -c "import agent; print('agent ok', len(agent.TOOLS), 'tools')"   # agent ok 3 tools
```

**Verify:** `wc -l agent.py` shows well under 150. Read the loop once more and confirm each of the five aborts (`kill switch`, `wall-clock budget`, `token budget`, `dollar budget`, `max-steps budget`) returns a **distinct** string — the DoD requires 5 runs, 5 reasons.

**Troubleshoot:**
- `anthropic.AuthenticationError` → `.env` key is wrong/missing, or `load_dotenv()` didn't find `.env` (run from the `agent-lab/` folder).
- `400 ... temperature` if you added `temperature=0` → current Claude models (Opus 4.8 / Sonnet 5) **reject** sampling params with a 400. Remove it; see [Step 9](#step-9--exercise-3-forced-tool_choice--and-what-determinism-you-actually-get) for what determinism levers *do* work here.
- `400 ... each tool_use block must have a corresponding tool_result` → you appended the assistant turn but not the user `tool_result` turn (or missed a `tool_use_id`). The loop above answers every block; if you edited it, ensure `results` has one entry per `tool_use`.

---

### Step 5 — First real run: the 2-tool task (DoD #1)

**What:** Run the default task, which forces **two different tools** (read a file AND do arithmetic) and prints the correct answer.

**Why:** This is the acceptance test for the loop itself. If the model reads `notes.txt`, computes `17*23`, and reports both, your append-order, dispatch, and result-plumbing are all correct.

**Do it:**
```bash
rm -f trace.jsonl KILL           # clean slate
python agent.py
```

**Expected result:** Something like:
```
[step 1] tool=read_file in=520 out=95 $0.00499
[step 2] tool=calculator in=1130 out=110 $0.00840
[step 3] tool=None in=1780 out=145 $0.01252
notes.txt says ... and 17 * 23 = 391.
```
The final line names both results (the notes content **and** `391`). Exact token counts and step numbering vary — the model may call both tools in one step or across two.

**Verify:**
```bash
wc -l trace.jsonl                       # one record per step
python - <<'PY'
import json
d = [json.loads(l) for l in open("trace.jsonl")]
cum = [r["cum_dollars"] for r in d]
print("records:", len(d), "cum_dollars:", cum)
assert cum == sorted(cum), "cum_dollars must be monotonically non-decreasing"
assert any(r["tool"] == "read_file" for r in d) and any(r["tool"] == "calculator" for r in d)
print("DoD #1 + trace checks PASS")
PY
```
This confirms DoD #1 (correct answer from a 2-tool task) **and** the trace requirement (one record per step, `cum_dollars` monotonically increasing, showing `tool`/`args`/`observation`/`tok_in`/`tok_out`).

**Troubleshoot:**
- Model answers arithmetic without calling `calculator` → strengthen the task: *"Use the calculator tool to compute 17*23; use the read_file tool to read notes.txt."* Frontier models sometimes do trivial math in their head.
- `cum_dollars` not sorted → you recomputed `dollars` from a per-call delta instead of cumulative `tok_in`/`tok_out`. The formula must use the running totals.
- Runs but final line is empty → the model ended on a `tool_use` you didn't answer, or `max_steps` was too low and it returned an `ABORTED:` string. Raise `max_steps` or read the trace.

---

### Step 6 — Free local path: run the same loop on Ollama

**What:** Swap the backend to a local Ollama model so you can iterate at zero cost. Same loop, same budgets, same trace.

**Why:** So you never spend money debugging the *loop* (as opposed to model quality), and so you can run the budget-tripping exercises hundreds of times for free. Ollama also honors `temperature=0`, which the Claude models reject — you'll use that in Step 9.

**Do it:** Install Ollama from [ollama.com](https://ollama.com), then:
```bash
ollama pull llama3.1        # or: ollama pull qwen2.5  — both do native tool calling
pip install ollama
```

Add a second entry point to `agent.py` (or a small `agent_ollama.py`) — a `run_ollama` that keeps the identical loop and only changes the model call + result-block shape. Ollama returns `response.message.tool_calls`, and its usage counts live on `prompt_eval_count` / `eval_count`:

```python
# agent_ollama.py
import time, pathlib
from types import SimpleNamespace
import ollama
from tools import calculator, web_fetch, read_file
from trace import log_step

MODEL = "llama3.1"
KILL_FILE = pathlib.Path("KILL")
PRICE_IN = PRICE_OUT = 0.0          # local = free; dollar budget stays a no-op-ish 0

OLLAMA_TOOLS = [{"type": "function", "function": {
    "name": n, "description": d,
    "parameters": {"type": "object",
                   "properties": {p: {"type": "string"}}, "required": [p]}}}
    for n, d, p in [("calculator", "Evaluate an arithmetic expression.", "expression"),
                    ("web_fetch", "Fetch text from an http(s) URL.", "url"),
                    ("read_file", "Read a file from the sandbox by name.", "name")]]
DISPATCH = {"calculator": calculator, "web_fetch": web_fetch, "read_file": read_file}


def run_ollama(task, *, max_steps=8, max_seconds=120, max_tokens_budget=50_000):
    messages = [{"role": "user", "content": task}]
    start, tok_in, tok_out = time.time(), 0, 0
    for step in range(1, max_steps + 1):
        if KILL_FILE.exists():                       return "ABORTED: kill switch"
        if time.time() - start > max_seconds:        return "ABORTED: wall-clock budget"
        if tok_in + tok_out > max_tokens_budget:     return "ABORTED: token budget"

        resp = ollama.chat(model=MODEL, messages=messages, tools=OLLAMA_TOOLS,
                           options={"temperature": 0})     # Ollama honors temperature=0
        tok_in += resp.get("prompt_eval_count", 0) or 0
        tok_out += resp.get("eval_count", 0) or 0
        usage = SimpleNamespace(input_tokens=tok_in, output_tokens=tok_out)

        msg = resp["message"]
        calls = msg.get("tool_calls") or []
        if not calls:                                        # model is done
            log_step(step, msg.get("content", ""), None, None, None, usage, 0.0)
            return msg.get("content", "")

        messages.append(msg)                                 # assistant turn
        for c in calls:
            fn = c["function"]
            obs = DISPATCH[fn["name"]](**fn["arguments"])
            is_err = obs.startswith("ERROR:")
            messages.append({"role": "tool", "content": obs, "name": fn["name"]})
            log_step(step, "", fn["name"], fn["arguments"], obs, usage, 0.0, is_err)
    return "ABORTED: max-steps budget"


if __name__ == "__main__":
    print(run_ollama("Read notes.txt from the sandbox, then compute 17*23 and report both."))
```

**Expected result:** `python agent_ollama.py` prints step lines and a final answer naming both results, exactly like Step 5 — but with `$0.0` and no API key.

**Verify:** `ollama list` shows `llama3.1`. The run produces `trace.jsonl` with one record per step. If the 8B model flubs tool arguments, retry or `ollama pull qwen2.5` (often better at tool-call formatting).

**Troubleshoot:**
- `ConnectionError` → the Ollama server isn't running. Start it (`ollama serve`, or just open the Ollama app) and confirm `curl http://localhost:11434/api/tags` responds.
- Model narrates `Action: calculator` as text instead of emitting a real tool call → that's the legacy regex-ReAct failure the loop must *not* rely on (lecture 02). Use `llama3.1`/`qwen2.5` (native tool calling) and keep tools in the `tools=` param; don't parse text.
- **State which determinism levers your backend supports** (DoD requirement): on **Ollama**, `temperature=0` is honored and is your main knob; on **Claude opus-4-8**, `temperature` is rejected — see Step 9.

> The remaining exercises are written against the paid `agent.py` but work identically on `run_ollama` (the token/wall-clock/kill budgets are all present there; only the dollar budget is Claude-only, since local inference is free).

---

### Step 7 — Exercise 1: trip each budget and confirm its reason string (DoD #3)

**What:** Force **each** of the four budgets plus the kill switch to fire, and confirm the exact `ABORTED: <reason>` string. Five runs, five distinct reasons.

**Why:** A budget you've never seen fire is a budget you don't trust. This is DoD #3.

**Do it:** From a REPL (each call passes overrides that guarantee that one budget trips first):

```bash
rm -f KILL
python - <<'PY'
from agent import run
TASK = "Read notes.txt from the sandbox, then compute 17*23 and report both."

# 1) max-steps: only one step allowed, task needs >=2 → loop falls through
print("max_steps :", run(TASK, max_steps=1))

# 2) dollar: any real opus call costs > $0.0001, so step 2's top-of-loop check trips
print("dollars  :", run(TASK, max_dollars=0.0001))

# 3) token: budget of 1 token; after step 1's usage, step 2's check trips
print("tokens   :", run(TASK, max_tokens_budget=1))

# 4) wall-clock: 0s budget trips at the top of step 1, before any spend
print("wallclock:", run(TASK, max_seconds=0))
PY
```

Then the kill switch — create the flag file, run, delete it:

```bash
touch KILL
python -c "from agent import run; print('kill     :', run('anything'))"
rm KILL
```

**Expected result:**
```
max_steps : ABORTED: max-steps budget
dollars   : ABORTED: dollar budget
tokens    : ABORTED: token budget
wallclock : ABORTED: wall-clock budget
kill      : ABORTED: kill switch
```
Five runs, five distinct strings. ✅ DoD #3.

**Verify:** Each line matches a *different* reason. To watch the KILL switch abort a run *mid-flight* (not just pre-flight), start a long task in one terminal and `touch KILL` from another; because the check is at the top of every iteration, the run stops at the next step boundary and returns `ABORTED: kill switch`.

**Troubleshoot:**
- `max_dollars=0.0001` returned the finished answer instead of aborting → the task completed in a single cheap step before the step-2 check. Lower it further or use a task that needs ≥2 model calls (the read+compute task does).
- Token budget never trips → same cause; the check is at the *top* of the loop, so it fires on step 2 after step 1's usage lands. Confirm your task needs a second model call.
- Kill switch didn't fire → the `KILL` file isn't in the working directory the process runs from. `ls KILL` from the same folder you launched `python`.

---

### Step 8 — Exercise 2: recover from a broken tool call (DoD #4)

**What:** Point `web_fetch` at a garbage URL and confirm the model *sees the `ERROR:` observation, adapts, and the run still terminates cleanly* — no traceback, and the failing `tool_result` carries `is_error: true`.

**Why:** This is the payoff of errors-as-observations (lecture 03) and DoD #4. A raised exception would end the run; a returned `ERROR:` string becomes context the model reasons about.

**Do it:**
```bash
rm -f trace.jsonl
python -c "from agent import run; print(run('Fetch https://not-a-real-domain-xyz-42.invalid and summarize it; if it fails, explain what went wrong.'))"
```

**Expected result:** The trace shows a `web_fetch` step with an `ERROR: fetch failed …` observation and `is_error: true`, then a final text turn where the model explains the fetch failed rather than crashing. The process exits 0 with a real answer — **no Python traceback**.

**Verify:**
```bash
python - <<'PY'
import json
d = [json.loads(l) for l in open("trace.jsonl")]
assert any(r["is_error"] for r in d), "expected an is_error:true tool_result"
print("is_error present:", [r["is_error"] for r in d], "→ DoD #4 PASS")
PY
echo "exit code: $?"     # 0 — clean termination
```

**Troubleshoot:**
- You got a traceback → a tool raised instead of returning `ERROR:`. Re-check Step 2: every tool body is wrapped in `try/except` returning a string.
- `is_error` is `False` on the failed fetch → your loop sets `is_error` off `obs.startswith("ERROR:")`; confirm `web_fetch` prefixes failures with exactly `ERROR:`.
- The model retries forever → good instinct to notice; that's what the budgets in Step 7 are for. In practice the model gives up after seeing the error and returns text.

---

### Step 9 — Exercise 3: forced `tool_choice` — and what determinism you actually get

**What:** Force a specific tool with `tool_choice={"type":"tool","name":"calculator"}` and observe that the model's first action is deterministically that tool. Then articulate which determinism levers your backend supports.

**Why:** Forcing `tool_choice` is a determinism lever (lecture 05): when a step *must* call a specific tool, you remove the model's discretion about *which* tool and *whether* to call one. Combined with pinning the exact model version, this is how you get reproducible debugging on Claude — because current Claude models (Opus 4.8, Sonnet 5) **reject `temperature`/`top_p` with a 400**, so `temperature=0` is *not* available there.

**Do it:**
```bash
rm -f trace.jsonl
python -c "from agent import run; print(run('What is 42 times 17?', tool_choice={'type':'tool','name':'calculator'}))"
head -1 trace.jsonl
```

**Expected result:** The first trace record has `"tool": "calculator"` every time — the model is forced to call it rather than answering from memory or picking `web_fetch`.

**Verify — state the levers (DoD #6).** Write these three sentences into your notes; the DoD requires you can state each and say which your backend supports:
- **Pin the model version** (`claude-opus-4-8`, never a floating alias): removes silent behavior drift between runs when the provider updates a mutable alias. *Supported on both backends.*
- **Force `tool_choice`**: makes a step call a named tool deterministically instead of choosing. *Supported on both backends.*
- **`temperature=0`**: eliminates sampling variance. *Supported on **Ollama** (`options={"temperature":0}`), **rejected with 400 on Claude opus-4-8/Sonnet 5** — there, determinism comes from pinning the model + `output_config.effort` + adaptive thinking, not from a temperature knob.*

**Troubleshoot:**
- `400 invalid_request_error` mentioning `tool_choice` → check the shape: `{"type":"tool","name":"calculator"}`. `{"type":"auto"}` (the default) and `{"type":"any"}` are the other valid forms.
- Adding `temperature=0` to the Claude call → `400`. Remove it; that's the whole point of this step. Keep `temperature=0` only on the Ollama path.
- Forced tool but the model still answered in text afterward → that's expected: `tool_choice` forces the *next* action to be that tool; after the tool result comes back, the model is free to finish with text.

---

## Putting it together — short end-to-end run

Run the canonical acceptance flow start to finish:

```bash
cd agent-lab
source .venv/Scripts/activate        # or .venv/bin/activate on macOS/Linux
rm -f trace.jsonl KILL

# 1. The 2-tool task prints the correct answer (DoD #1)
python agent.py

# 2. Trace has one record per step with monotonic cum_dollars (DoD #2)
python - <<'PY'
import json
d=[json.loads(l) for l in open("trace.jsonl")]
cum=[r["cum_dollars"] for r in d]
print("records:",len(d)," monotonic:",cum==sorted(cum))
print("tools used:", sorted({r["tool"] for r in d if r["tool"]}))
PY

# 3. Five distinct ABORTED reasons across 5 runs (DoD #3)
python - <<'PY'
from agent import run
T="Read notes.txt then compute 17*23 and report both."
for label,kw in [("steps",{"max_steps":1}),("dollars",{"max_dollars":1e-4}),
                 ("tokens",{"max_tokens_budget":1}),("wallclock",{"max_seconds":0})]:
    print(label, "->", run(T, **kw))
PY
touch KILL; python -c "from agent import run;print('kill ->',run('x'))"; rm KILL

# 4. Broken tool call → is_error:true and clean exit (DoD #4)
python -c "from agent import run; print(run('Fetch https://nope.invalid and summarize; explain any failure.'))"

# 5. agent.py is <= ~150 lines (DoD #5)
wc -l agent.py
```

Watching those five blocks pass in sequence *is* the lab.

---

## Definition of Done — verifiable checks

Restating the spine's acceptance gate. Each is checkable with the commands above.

- [ ] **DoD #1 — 2-tool task, correct answer.** `python agent.py` reads `notes.txt` **and** computes `17*23`, printing both (Step 5).
- [ ] **DoD #2 — trace integrity.** `trace.jsonl` has **one record per step**, each showing `tool`, `args`, `observation`, `tok_in`/`tok_out`, and **monotonically increasing `cum_dollars`** (Step 5 verify block).
- [ ] **DoD #3 — all four budgets + kill switch fire with distinct reasons.** 5 runs → `max-steps budget`, `dollar budget`, `token budget`, `wall-clock budget`, `kill switch` (Step 7).
- [ ] **DoD #4 — broken tool call is recoverable.** A garbage URL yields a `tool_result` with `is_error: true` and the agent **terminates cleanly, no traceback** (Step 8).
- [ ] **DoD #5 — `agent.py` ≤ ~150 lines.** `wc -l agent.py` (~90).
- [ ] **DoD #6 — you can state the determinism levers.** One sentence each on pinning the model / forcing `tool_choice` / `temperature=0`, and **which your chosen backend supports**: Claude opus-4-8 supports pinning + `tool_choice` but **rejects `temperature`**; Ollama supports all three including `temperature=0` (Step 9).

---

## Troubleshooting cheatsheet

| Symptom | Cause | Fix |
|---|---|---|
| `anthropic.AuthenticationError` | Missing/wrong key, or wrong CWD | Put a real key in `.env`; run from `agent-lab/`; confirm `load_dotenv()` ran |
| `400 ... temperature`/`top_p` | Claude 4.x rejects sampling params | Remove `temperature`; determinism = pin model + `tool_choice` (+ `output_config.effort`) |
| `400 ... tool_use must have a corresponding tool_result` | Missed the user `tool_result` turn, or an unanswered `tool_use_id` | Append assistant turn **then** a user turn with one `tool_result` per `tool_use` block |
| Traceback from inside a tool | A tool raised instead of returning `ERROR:` | Wrap every tool body in `try/except` returning a string (Step 2) |
| `cum_dollars` not monotonic | Recomputed from per-call delta | Recompute from cumulative `tok_in`/`tok_out` each step |
| Budget never trips | Check is at top of loop; task finished first | Use a task needing ≥2 model calls; lower the threshold |
| Kill switch ignored | `KILL` file not in process CWD | `ls KILL` from the launch folder; check is at loop top only |
| Model answers math without the tool | Frontier model does trivial math in-head | Tell it explicitly to *use the calculator tool*, or force `tool_choice` |
| Ollama `ConnectionError` | Server not running | `ollama serve` / open the app; `curl localhost:11434/api/tags` |
| Ollama emits `Action:` text, no tool call | Regex-ReAct instead of native | Use `llama3.1`/`qwen2.5`; pass tools via `tools=`; never parse text |
| `is_relative_to` AttributeError | Python < 3.9 | Upgrade, or use `SANDBOX in p.parents` check |

---

## Stretch goals (optional)

- **Real search tool.** Swap `web_fetch` for a real search API — [Tavily](https://tavily.com) free tier, or DuckDuckGo via `pip install ddgs`. Keep the errors-as-observations contract: return a string, never raise.
- **Parallel tool calls.** The model can emit multiple `tool_use` blocks in one turn. The loop already handles this (it iterates `tool_uses`), but time the dispatch and run independent tools concurrently with `concurrent.futures.ThreadPoolExecutor` — then measure the latency win.
- **`output_config.effort` sweep.** On Claude, add `output_config={"effort": "low"|"medium"|"high"}` and observe how tool-call count and token spend change on the same task. This is the Claude-side determinism/cost knob that replaces `temperature`.
- **Retry-with-backoff wrapper.** Wrap `client.messages.create` so 429/5xx retry with exponential backoff (the SDK already does 2 retries; make it explicit and log each retry to the trace).
- **A `--dry-run` cost estimator.** Before running, call `client.messages.count_tokens(...)` on the initial prompt to print an estimated dollar cost, so you can catch an accidentally huge task before spending.
- **Trace viewer.** Write a 20-line script that pretty-prints `trace.jsonl` as a table (step, tool, args, truncated obs, cum $) — the seed of the observability dashboard you build in Phase 7.
- **Second kill mechanism.** Add a SIGINT handler (Ctrl-C) that sets a module-level flag the loop checks alongside the KILL file — a human's *other* emergency stop.
