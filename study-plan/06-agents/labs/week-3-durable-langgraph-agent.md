# Week 3 Lab: Port to LangGraph — Durable, Interruptible, Crash-Safe

> This week you take the hand-rolled ReAct loop from Week 1 and port it to **LangGraph** — not because graphs are trendy, but because LangGraph is a **durable executor**: it checkpoints state after every super-step, lets you pause *inside a node* for a human, and resumes a killed process from the last committed step. You will then prove durability the only way that counts: **hard-kill the process mid-tool-call and re-invoke the same `thread_id`**, asserting a side-effect ledger has *zero* duplicates. The lesson you carry out: **checkpointing resumes you after committed steps; idempotency keys save you in the commit gap.** You need both. A free local path (Ollama) and a free "prod" path (Postgres in Docker) are given — no API key, no GPU, no spend required.
>
> Read these lectures first — this guide assumes them:
> - [../lectures/11-frameworks-by-control-model.md](../lectures/11-frameworks-by-control-model.md) — graph vs conversation-orchestration vs typed-loop; who owns control flow
> - [../lectures/12-durable-execution-checkpointing.md](../lectures/12-durable-execution-checkpointing.md) — state durability, super-steps, checkpointers, when you graduate to Temporal
> - [../lectures/13-idempotency-safe-replay.md](../lectures/13-idempotency-safe-replay.md) — LLM/tool calls as non-deterministic side effects; the commit gap; idempotency keys
> - [../lectures/14-hitl-interrupts-time-travel.md](../lectures/14-hitl-interrupts-time-travel.md) — `interrupt()`, `Command(resume=...)`, time-travel, async agents

**Est. time:** ~9 hrs · **You will need:** Python 3.10+, a terminal (Git-Bash on Windows), `uv` (or plain `pip`/`venv`), and **one** of:
- **Free/local path (recommended):** [Ollama](https://ollama.com) running `llama3.1` on CPU. No key, no GPU, no spend. Native tool calling works fine for this lab's tiny task.
- **Paid path:** an Anthropic key (`claude-opus-4-8`) or OpenAI key (`gpt-4o-mini`). Cents, not dollars.
- **Optional "prod" durability path:** Docker, to run Postgres for `PostgresSaver` instead of SQLite. Also free.

You are **porting**, not rewriting from zero — keep your Week-1 `tools.py` logic and model call handy. LangGraph changes *who drives the loop* (the graph runtime, not your `while`), not what the tools do.

---

## Before you start (setup)

**What:** Create a `week3/` subfolder in your phase-6 repo, an isolated environment, and install LangGraph + the SQLite checkpointer + your model adapter.

**Why:** LangGraph's core (`langgraph`, `langchain-core`) is the graph runtime; the checkpointer library (`langgraph-checkpoint-sqlite`) is a *separate* package — forgetting it is the #1 "why won't `SqliteSaver` import" issue. `pydantic` types your state cleanly; `pytest` runs the durability tests.

**Do it (uv — recommended):**

```bash
mkdir -p phase6-agents/week3 && cd phase6-agents/week3
uv init
uv add langgraph langchain-core "langgraph-checkpoint-sqlite" pydantic pytest

# Pick ONE model backend:
uv add langchain-ollama          # free/local (default) — then: ollama pull llama3.1
# uv add langchain-anthropic     # paid: needs ANTHROPIC_API_KEY
# uv add langchain-openai        # paid: needs OPENAI_API_KEY
```

**Do it (plain pip, if you don't have `uv`):**

```bash
mkdir -p phase6-agents/week3 && cd phase6-agents/week3
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git-Bash)
# source .venv/bin/activate       # macOS / Linux
pip install langgraph langchain-core langgraph-checkpoint-sqlite pydantic pytest langchain-ollama
```

Pull the local model (free path) and lay down a `.gitignore` so the checkpoint DB and ledger never get committed:

```bash
ollama pull llama3.1     # ~4.7 GB; skip if using a paid API backend

printf '.venv/\n__pycache__/\ncheckpoints.sqlite*\nside_effects.log\n.env\n' > .gitignore
```

**Expected result:** target folder layout (files are created across the steps below):

```
phase6-agents/week3/
  pyproject.toml        # from `uv init` (or requirements via pip)
  model.py              # your model, .bind_tools(TOOLS) — the escape hatch lives here   (Step 1)
  tools.py              # tools + idempotent side-effecting send_invoice                 (Step 3)
  ledger.py             # append-only record of applied side effects                     (Step 3)
  graph.py              # StateGraph: agent node, tool node, conditional edges           (Step 2)
  run.py                # CLI: start / resume a thread_id, compiled with a checkpointer   (Step 4)
  hitl.py               # interrupt() + Command(resume=...) approval demo                 (Step 6)
  crash_test.py         # hard-kills mid-run, resumes, asserts no dup side effects        (Step 7)
  checkpoints.sqlite    # created at runtime (gitignored)
  tests/
    test_durability.py                                                                     (Step 8)
```

**Verify:**
```bash
python -c "import langgraph, langchain_core; from langgraph.checkpoint.sqlite import SqliteSaver; print('deps ok')"
```
Prints `deps ok`. (If you're on the paid path, `langchain_ollama` will be missing — swap the import for your backend.)

**Troubleshoot:**
- `ModuleNotFoundError: No module named 'langgraph.checkpoint.sqlite'` → you skipped `langgraph-checkpoint-sqlite`. It ships separately from `langgraph`. `uv add langgraph-checkpoint-sqlite`.
- `source: .venv/Scripts/activate: No such file` on Windows → you ran the macOS line; Windows venvs put `activate` under `Scripts/`, not `bin/`.
- `ollama pull` hangs → confirm the Ollama app/daemon is running (`ollama list` should respond).

---

## Step-by-step

### Step 1 — The model node and your escape hatch

**What:** Create `model.py` — one place that builds a chat model with your tools bound to it. This is the "drop-to-raw" boundary: everything framework-specific lives behind `llm_with_tools`.

**Why:** Lecture 11's load-bearing rule — *pick a framework by control model, but always keep a drop-to-raw escape hatch.* LangGraph nodes are just Python functions; nothing stops you from calling the raw SDK inside one. Isolating the model here means that if LangChain's chat wrapper ever fights you, you swap this file (not the graph). It also makes the model backend a one-line change.

**Do it:**

```python
# model.py — the ONLY place the concrete model backend is named. Escape hatch lives here.
import os

def build_llm_with_tools(tools):
    """Return a chat model with `tools` bound. Swap the backend in one place."""
    backend = os.environ.get("LLM_BACKEND", "ollama")

    if backend == "ollama":
        from langchain_ollama import ChatOllama
        # temperature=0 is honored by Ollama (Claude Opus/Sonnet reject it — see Week 1).
        llm = ChatOllama(model="llama3.1", temperature=0)
    elif backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-opus-4-8", max_tokens=1024)  # no temperature on Opus 4.8
    elif backend == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        raise ValueError(f"unknown LLM_BACKEND={backend!r}")

    return llm.bind_tools(tools)
```

**Expected result:** no output yet — this is a library module. Set the backend once per shell:

```bash
export LLM_BACKEND=ollama        # or anthropic / openai
```

**Verify:** after `tools.py` exists (Step 3) you'll do a smoke test. For now:
```bash
python -c "import model; print('model module ok')"
```

**Troubleshoot:**
- `ValidationError: temperature ... not permitted` on Anthropic → current Claude models reject sampling params. Remove `temperature` for the `anthropic` branch (already done above).
- Ollama backend errors with `model 'llama3.1' not found` → run `ollama pull llama3.1` first.

---

### Step 2 — Typed state + the graph skeleton

**What:** Define the graph in `graph.py`: a typed `State`, an `agent` node (calls the model), a prebuilt `ToolNode` (runs whatever tool the model asked for), and conditional edges that loop `agent → tools → agent` until the model stops asking for tools.

**Why:** This is the Week-1 `while` loop expressed as a **state machine**. In Week 1 *you* owned the loop; here the LangGraph runtime owns it — and that's precisely what makes it checkpointable and resumable (it knows where "here" is). `add_messages` is a reducer that appends new messages instead of overwriting, so state accumulates the conversation. `tools_condition` is the prebuilt router: if the last AI message contains tool calls, go to `tools`; otherwise go to `END`.

**Do it:**

```python
# graph.py — the StateGraph. Same control flow as Week 1, but the runtime drives it.
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOLS               # your tools, wrapped as @tool (Step 3)
from model import build_llm_with_tools

llm_with_tools = build_llm_with_tools(TOOLS)


class State(TypedDict):
    # add_messages = reducer: new messages are APPENDED, not replaced. This is the state
    # LangGraph checkpoints after every super-step.
    messages: Annotated[list, add_messages]


def agent(state: State):
    # Escape hatch: this body can be raw SDK. Here we invoke the tool-bound chat model.
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def build_graph():
    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))     # runs the tool the model requested
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)   # -> "tools" if tool_calls else END
    g.add_edge("tools", "agent")             # loop back after a tool runs
    return g                                  # NOTE: NOT compiled here — run.py adds the checkpointer


GRAPH = build_graph()
```

**Why compile later:** the checkpointer is bound at **compile** time, and the SQLite saver is a context manager (it opens a DB connection). Compiling in `run.py` inside the `with` block keeps the connection lifetime correct. Compiling here would force a checkpointer choice on every importer.

**Expected result:** `graph.py` imports cleanly once `tools.py` exists.

**Verify:**
```bash
python -c "from graph import GRAPH; print(type(GRAPH).__name__)"   # -> StateGraph
```

**Troubleshoot:**
- `ImportError: cannot import name 'tools_condition'` → old LangGraph. `uv add -U langgraph` (need a 0.2+/0.3+ line where prebuilts live under `langgraph.prebuilt`).
- Graph "ends immediately" with no tool call → your model isn't emitting tool calls. Confirm `bind_tools` ran and the backend supports native tool calling (llama3.1 does; some smaller models don't).

---

### Step 3 — An idempotent, side-effecting tool + the ledger

**What:** Give the agent at least one tool with a **real, observable side effect** (`send_invoice` writes a line to `side_effects.log`), guarded by an **idempotency key**. Add the append-only ledger in `ledger.py`. Include one harmless read tool so the task is realistic.

**Why:** This is the heart of the week. A durable system's whole promise — "resume without re-doing work" — is only *testable* if there is a side effect you can count. The ledger is your source of truth for "did this actually happen?" The idempotency key (`sha256(action + args)`) makes the tool a **no-op on replay**: if the process dies *after* the side effect fires but *before* the checkpoint commits (the **commit gap**), replay re-enters the tool — and only the key stops a duplicate. Lecture 13: checkpointing alone does *not* save you here.

**Do it:**

```python
# ledger.py — append-only; the single source of truth for "did this side effect happen?"
import json, os, hashlib

LEDGER = "side_effects.log"

def key(*parts) -> str:
    """Deterministic idempotency key. Same action + same args => same key => dedup."""
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]

def applied(k: str) -> bool:
    if not os.path.exists(LEDGER):
        return False
    with open(LEDGER) as f:
        return any(json.loads(line)["key"] == k for line in f if line.strip())

def record(k: str, action: str) -> None:
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"key": k, "action": action}) + "\n")

def count() -> int:
    if not os.path.exists(LEDGER):
        return 0
    with open(LEDGER) as f:
        return sum(1 for line in f if line.strip())
```

```python
# tools.py — tools wrapped as LangChain @tool so ToolNode can run them.
import os
from langchain_core.tools import tool
from ledger import key, applied, record

@tool
def send_invoice(customer: str, amount: float) -> str:
    """Send an invoice to a customer. SIDE-EFFECTING and MUST be idempotent."""
    k = key("send_invoice", customer, amount)
    if applied(k):
        return f"already-sent:{k}"          # replay-safe no-op — the commit-gap defense
    # --- the real side effect (here: append to the ledger; in prod: POST to Stripe, etc.) ---
    record(k, f"invoice {customer} {amount}")
    # OPTIONAL for Step 7: simulate a crash AFTER the side effect, BEFORE returning.
    if os.environ.get("CRASH_IN_GAP") == "1":
        os._exit(1)                          # hard kill — no cleanup, no checkpoint commit
    return f"sent:{k}"

@tool
def get_balance(customer: str) -> str:
    """Look up a customer's outstanding balance (read-only, safe to replay)."""
    balances = {"acme": "1200.00", "globex": "0.00", "initech": "875.50"}
    return balances.get(customer.lower(), "unknown customer")

TOOLS = [send_invoice, get_balance]
```

**Expected result:** importing `tools` works; `TOOLS` has two entries. No ledger file exists yet (it's created on first `send_invoice`).

**Verify:**
```bash
python - <<'PY'
from tools import send_invoice, TOOLS
from ledger import count
print("tools:", [t.name for t in TOOLS])
print(send_invoice.invoke({"customer": "acme", "amount": 100}))   # sent:...
print(send_invoice.invoke({"customer": "acme", "amount": 100}))   # already-sent:... (dedup!)
print("ledger lines:", count())                                    # 1, not 2
PY
```
Expect `sent:...` then `already-sent:...` and `ledger lines: 1`. That one line proves idempotency works *outside* the graph before you trust it *inside* it.

**Troubleshoot:**
- `send_invoice.invoke(...)` raises `TypeError` about a dict → LangChain `@tool` objects take a single dict of args to `.invoke()`, not kwargs. Use `.invoke({"customer": ..., "amount": ...})`.
- Two ledger lines instead of one → your key isn't deterministic. `100` vs `100.0` produce different keys; normalize `amount` (e.g. `round(float(amount), 2)`) if the model sends varying formats.

---

### Step 4 — Compile with a checkpointer and run a thread

**What:** Write `run.py` — compile the graph with `SqliteSaver`, run a task under a fixed `thread_id`, and stream the steps. Support both "start" and "resume" of the same thread.

**Why:** Lecture 12 — durability = state persisted per super-step to a store, keyed by `thread_id`. The `thread_id` **is** the resume handle: re-invoking with the same id continues from the last checkpoint; a new id starts fresh. `SqliteSaver` is a real on-disk store (unlike `MemorySaver`, which evaporates on restart and is useless for durability). Streaming with `stream_mode="values"` lets you watch state evolve.

**Do it:**

```python
# run.py — compile with a checkpointer, run/resume a thread_id from the CLI.
import sys
from langgraph.checkpoint.sqlite import SqliteSaver
from graph import GRAPH

TASK = ("Look up the balance for customer 'acme', then send them an invoice "
        "for 100.00 dollars. Report what you did.")

def main():
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "demo-1"
    resume = "--resume" in sys.argv

    # SqliteSaver is a context manager: it owns the DB connection for this block.
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
        app = GRAPH.compile(checkpointer=cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        # Resume = invoke with None input; LangGraph continues from the last checkpoint.
        payload = None if resume else {"messages": [("user", TASK)]}

        for ev in app.stream(payload, cfg, stream_mode="values"):
            last = ev["messages"][-1]
            who = getattr(last, "type", "?")
            text = getattr(last, "content", last)
            print(f"[{who}] {text}")

        # Show how many checkpoints exist for this thread.
        n = sum(1 for _ in app.get_state_history(cfg))
        print(f"\n--- checkpoints for thread '{thread_id}': {n} ---")

if __name__ == "__main__":
    main()
```

**Expected result:**
```bash
python run.py demo-1
```
prints the user task, then the agent's tool-call turn, then a `tool` message (`sent:<key>`), then a final assistant summary, then `--- checkpoints for thread 'demo-1': N ---` with N ≥ 3 (one per super-step). `side_effects.log` now has exactly one line.

**Verify:**
```bash
python -c "from ledger import count; print('ledger lines:', count())"   # 1
# Prove the checkpoint DB has real rows:
python -c "import sqlite3; c=sqlite3.connect('checkpoints.sqlite'); \
print('checkpoint rows:', c.execute('select count(*) from checkpoints').fetchone()[0])"
```

**Troubleshoot:**
- `sqlite3.OperationalError: no such table: checkpoints` → you queried before any run wrote to it, or you used `MemorySaver`. Run `run.py` once first.
- Resume re-runs everything from scratch → you passed a *new* `thread_id`, or omitted `--resume` so `payload` wasn't `None`. The id must match and input must be `None`.
- `RuntimeError: ... checkpointer ... outside context` → you compiled the app outside the `with SqliteSaver(...)` block. Keep compile + stream inside it.

---

### Step 5 — The free "prod" path: Postgres checkpointer (optional but recommended)

**What:** Swap `SqliteSaver` for `PostgresSaver` backed by a Docker Postgres. One import + one `.setup()` call.

**Why:** SQLite is perfect locally, but production agents run as services with concurrent workers — that's Postgres territory. The point of this step is to *feel* how little changes: the graph, tools, and ledger are untouched; only the checkpointer line moves. This is the "one import swap" the spine promises.

**Do it:**

```bash
docker run -d --name pg -e POSTGRES_PASSWORD=pw -p 5432:5432 postgres:16
uv add langgraph-checkpoint-postgres     # or: pip install langgraph-checkpoint-postgres
```

```python
# run_pg.py — same as run.py, but Postgres. Note the one-time .setup().
from langgraph.checkpoint.postgres import PostgresSaver
from graph import GRAPH

DB = "postgresql://postgres:pw@localhost:5432/postgres"

with PostgresSaver.from_conn_string(DB) as cp:
    cp.setup()                              # ONE TIME: creates the checkpoint tables
    app = GRAPH.compile(checkpointer=cp)
    cfg = {"configurable": {"thread_id": "pg-demo-1"}}
    for ev in app.stream({"messages": [("user", "Send acme an invoice for 50.00")]},
                         cfg, stream_mode="values"):
        print(ev["messages"][-1])
```

**Expected result:** identical run behavior; state now lives in Postgres. `docker exec pg psql -U postgres -c '\dt'` lists LangGraph's checkpoint tables.

**Verify:**
```bash
docker exec pg psql -U postgres -c "select count(*) from checkpoints;"
```
Non-zero after a run.

**Troubleshoot:**
- `relation "checkpoints" does not exist` → you skipped `cp.setup()`. It must run once before the first stream.
- `connection refused` on 5432 → container not up (`docker ps`), or port already taken. Change host port: `-p 5433:5432` and update the URL.
- On the psycopg side, if you hit `ModuleNotFoundError: psycopg` → `uv add "psycopg[binary]"`.

---

### Step 6 — Human-in-the-loop interrupt before the dangerous tool

**What:** Insert an `approval` node before the side-effecting tool. It calls `interrupt(payload)` to **pause the graph inside the node**, surfaces a question to your app, and resumes with `Command(resume=value)`. Prove that state survives a **process restart** while paused.

**Why:** Lecture 14 — `interrupt()` persists state and hands the payload to your caller; because state is checkpointed, a human can take a coffee break (or the server can redeploy) while the graph waits. Uses: approve a spend, edit a plan, answer a clarifying question. The restart-while-paused test is what separates a real HITL from a `input()` prompt.

**Do it:** add an approval gate. The cleanest wiring for the lab is a dedicated node that runs before `tools` only when the last message requested `send_invoice`:

```python
# hitl.py — an approval gate that interrupts before a side-effecting tool.
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

from tools import TOOLS
from model import build_llm_with_tools

llm_with_tools = build_llm_with_tools(TOOLS)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def route_after_agent(state: State) -> str:
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if any(c["name"] == "send_invoice" for c in calls):
        return "approval"          # dangerous tool -> ask a human first
    if calls:
        return "tools"             # safe tool -> run directly
    return END

def approval(state: State):
    last = state["messages"][-1]
    # interrupt() PAUSES here, persists state, and surfaces this payload to the caller.
    decision = interrupt({"ask": "Approve send_invoice?", "tool_calls": last.tool_calls})
    if str(decision).lower().startswith("y"):
        return {}                  # approved: fall through to the tools node
    # denied: drop the tool call by appending a refusal so the loop ends cleanly
    return {"messages": [("assistant", "Invoice cancelled by human reviewer.")]}

def route_after_approval(state: State) -> str:
    # If the human approved, the last message is still the AI tool-call turn -> run tools.
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

def build_hitl_graph():
    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("approval", approval)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, ["approval", "tools", END])
    g.add_conditional_edges("approval", route_after_approval, ["tools", END])
    g.add_edge("tools", "agent")
    return g

if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "hitl-1"}}
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
        app = build_hitl_graph().compile(checkpointer=cp)
        task = "Send customer 'acme' an invoice for 100.00 dollars."
        # 1) Run until it interrupts.
        for ev in app.stream({"messages": [("user", task)]}, cfg, stream_mode="values"):
            pass
        state = app.get_state(cfg)
        if state.next:                       # graph is paused (there is a next node)
            intr = state.tasks[0].interrupts[0].value
            print("PAUSED, asking:", intr["ask"])
            # ---> at this point you could kill and restart the process; state is on disk <---
            decision = input("approve? [y/N] ")
            # 2) Resume the SAME thread with the human's answer.
            for ev in app.stream(Command(resume=decision), cfg, stream_mode="values"):
                print(ev["messages"][-1])
```

**Expected result:**
```bash
python hitl.py
```
prints `PAUSED, asking: Approve send_invoice?`, waits for input; typing `y` runs the tool and finishes, `n` cancels and ends without a ledger write.

**Restart-while-paused proof (the real test):** run it, and at the pause **Ctrl-C to kill the process**. Then run a tiny resume script against the same `thread_id`:

```python
# resume_hitl.py
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from hitl import build_hitl_graph
cfg = {"configurable": {"thread_id": "hitl-1"}}
with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
    app = build_hitl_graph().compile(checkpointer=cp)
    for ev in app.stream(Command(resume="yes"), cfg, stream_mode="values"):
        print(ev["messages"][-1])
```
```bash
python resume_hitl.py     # completes the run even though hitl.py's process is long dead
```

**Verify:** the resume completes and `side_effects.log` gains exactly one line (only after approval).

**Troubleshoot:**
- Resume raises `no interrupt to resume` → the thread wasn't actually paused (the model may not have chosen `send_invoice`). Check `state.next` was non-empty before killing.
- `interrupt` throws on second call in the same process → `interrupt()` must be driven via `stream`/`invoke`; don't call the node function directly.
- Human approves but tool never runs → `route_after_approval` didn't find `tool_calls`. Ensure your denial branch is the only one that appends a message; approval should return `{}` and leave the AI tool-call turn as the last message.

---

### Step 7 — The crash test (the whole point of the week)

**What:** `crash_test.py` hard-kills the process **mid-tool-call**, then re-invokes the same `thread_id` and asserts the ledger line count is **identical** before and after. Do it at both kill positions and observe the difference.

**Why:** This is where checkpointing and idempotency divide the labor (lecture 13):
- **(a) Kill *after* the tool node commits its checkpoint:** LangGraph resumes *after* that node — the tool never re-runs. Checkpointing alone is sufficient.
- **(b) Kill in the *commit gap*** (side effect fired, checkpoint not yet written): replay re-enters the tool, and **only the idempotency key** turns it into a no-op. Checkpointing does *not* save you here.

Proving both, and seeing that (b) would double-fire without the guard, is the lesson.

**Do it:**

```python
# crash_test.py — run the case-b crash as a subprocess so we can hard-kill it.
import os, sys, subprocess
from ledger import count

THREAD = "crash-1"

RUN_ONCE = """
from langgraph.checkpoint.sqlite import SqliteSaver
from graph import GRAPH
cfg = {"configurable": {"thread_id": "crash-1"}}
task = "Send customer 'acme' an invoice for 100.00 dollars, then confirm."
with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
    app = GRAPH.compile(checkpointer=cp)
    for ev in app.stream({"messages": [("user", task)]}, cfg, stream_mode="values"):
        pass
"""

RESUME = """
from langgraph.checkpoint.sqlite import SqliteSaver
from graph import GRAPH
cfg = {"configurable": {"thread_id": "crash-1"}}
with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
    app = GRAPH.compile(checkpointer=cp)
    for ev in app.stream(None, cfg, stream_mode="values"):   # None = resume
        print(ev["messages"][-1])
"""

def main():
    # Fresh state for a clean measurement.
    for f in ("side_effects.log", "checkpoints.sqlite"):
        if os.path.exists(f):
            os.remove(f)

    # 1) Start the run with CRASH_IN_GAP=1: send_invoice fires the side effect, then os._exit(1).
    env = {**os.environ, "CRASH_IN_GAP": "1"}
    proc = subprocess.run([sys.executable, "-c", RUN_ONCE], env=env)
    assert proc.returncode != 0, "expected the crash to kill the process"
    n_before = count()
    print(f"after crash: ledger lines = {n_before}")     # 1 — the side effect DID happen

    # 2) Resume the SAME thread_id (no crash this time).
    proc = subprocess.run([sys.executable, "-c", RESUME], env=os.environ)
    assert proc.returncode == 0, "resume should complete"
    n_after = count()
    print(f"after resume: ledger lines = {n_after}")      # STILL 1 — idempotency no-op'd the replay

    assert n_after == n_before, f"DUPLICATE SIDE EFFECT! {n_before} -> {n_after}"
    print("PASS: no duplicate side effect across the commit-gap crash.")

if __name__ == "__main__":
    main()
```

**Expected result:**
```bash
python crash_test.py
```
prints `after crash: ledger lines = 1`, then the resumed final message, then `after resume: ledger lines = 1`, then `PASS: no duplicate side effect...`.

**Prove the guard is load-bearing (do this once):** temporarily comment out the `if applied(k): return ...` early-return in `send_invoice`, rerun `crash_test.py`, and watch it fail with `DUPLICATE SIDE EFFECT! 1 -> 2`. That failure is the entire justification for idempotency keys. Restore the guard afterward.

**Case (a) — post-commit kill:** run `python run.py crash-2` to completion (no `CRASH_IN_GAP`), note the ledger line, then `python run.py crash-2 --resume`. Because the tool node's checkpoint committed, the resume is a no-op with *nothing left to do* — the tool never re-enters. Contrast this with case (b) where the tool *does* re-enter and the key saves you.

**Verify:** `crash_test.py` exits 0 and the two printed counts are equal.

**Troubleshoot:**
- `after crash: ledger lines = 0` → `CRASH_IN_GAP` fired *before* `record(...)`, or the model didn't call `send_invoice`. Confirm the `os._exit(1)` sits *after* `record(k, ...)` in `tools.py`, and that the task clearly asks to send an invoice.
- Resume raises `no checkpoint found` → the crash killed before *any* checkpoint wrote. Ensure the agent node (which decides the tool call) committed first; the crash should be inside the *tool*, not the first agent step.
- `1 -> 2` even with the guard → your key isn't stable across processes (e.g. it includes a timestamp or `id()`). Keys must be pure functions of action + args.

---

### Step 8 — Durability tests + time-travel

**What:** Write `tests/test_durability.py` asserting (1) a resume produces a final answer, (2) the side-effect count is invariant across a crash+resume, and (3) the idempotency guard returns the `already-*` path on replay. Then use `get_state_history` to fork an alternate branch (time-travel).

**Why:** The Definition of Done requires `pytest` green with these three assertions. Time-travel (lecture 14) turns "why did it do that?" into a reproducible experiment: list a thread's checkpoints, pick an earlier `checkpoint_id`, and resume from it to explore a different branch.

**Do it:**

```python
# tests/test_durability.py
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ledger import count, key, applied, record

def _fresh():
    for f in ("side_effects.log", "checkpoints.sqlite"):
        if os.path.exists(f):
            os.remove(f)

def test_idempotency_guard_replays_as_noop():
    _fresh()
    from tools import send_invoice
    first = send_invoice.invoke({"customer": "acme", "amount": 100.0})
    second = send_invoice.invoke({"customer": "acme", "amount": 100.0})
    assert first.startswith("sent:")
    assert second.startswith("already-sent:")     # (3) guard returns the already-* path
    assert count() == 1

def test_crash_resume_invariant():
    # Runs crash_test.py which asserts the count invariant internally.
    root = os.path.dirname(os.path.dirname(__file__))
    proc = subprocess.run([sys.executable, os.path.join(root, "crash_test.py")],
                          cwd=root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr   # (2) count invariant holds

def test_resume_produces_answer():
    _fresh()
    from langgraph.checkpoint.sqlite import SqliteSaver
    from graph import GRAPH
    cfg = {"configurable": {"thread_id": "test-answer"}}
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
        app = GRAPH.compile(checkpointer=cp)
        final = None
        for ev in app.stream({"messages": [("user", "What is acme's balance?")]},
                             cfg, stream_mode="values"):
            final = ev["messages"][-1]
        assert final is not None and getattr(final, "content", "")   # (1) final answer exists
```

Run:
```bash
pytest -q
```

**Time-travel snippet** (run interactively, not a test):

```python
# time_travel.py
from langgraph.checkpoint.sqlite import SqliteSaver
from graph import GRAPH
cfg = {"configurable": {"thread_id": "demo-1"}}
with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
    app = GRAPH.compile(checkpointer=cp)
    history = list(app.get_state_history(cfg))
    print(f"{len(history)} checkpoints")
    fourth = history[3]                     # pick an earlier checkpoint (index from newest)
    fork_cfg = fourth.config                # carries its checkpoint_id
    # Resume from that earlier point to explore an alternate branch:
    for ev in app.stream(None, fork_cfg, stream_mode="values"):
        print(ev["messages"][-1])
```

**Expected result:** `pytest -q` reports 3 passed. `time_travel.py` prints the checkpoint count and replays from checkpoint #4.

**Verify:** `pytest -q` exit code 0; `python time_travel.py` prints `N checkpoints` with N ≥ number of super-steps in `demo-1`.

**Troubleshoot:**
- Tests interfere with each other via shared files → each test calls `_fresh()`; the crash-resume test runs in a subprocess with its own working dir. If you parallelize with `pytest -n`, give each test a distinct `thread_id` and ledger file.
- `get_state_history` returns fewer entries than expected → you queried a thread that only ran one super-step. Use `demo-1` after a full `run.py demo-1`.
- `IndexError` on `history[3]` → the thread has < 4 checkpoints; pick a smaller index or run a longer task.

---

## Putting it together — short end-to-end run

```bash
export LLM_BACKEND=ollama            # free/local; or anthropic / openai

# 1) Clean state, run the ported agent end-to-end under a thread_id.
rm -f side_effects.log checkpoints.sqlite
python run.py demo-1
#   -> agent looks up balance, sends invoice, summarizes
#   -> "--- checkpoints for thread 'demo-1': N ---"  (N >= 3)

# 2) Confirm exactly one real side effect happened.
python -c "from ledger import count; print('ledger:', count())"      # 1

# 3) HITL: pause for approval, kill the process, resume across the restart.
python hitl.py            # prints "PAUSED, asking: ...", Ctrl-C to kill at the pause
python resume_hitl.py     # completes the dead run from the checkpoint

# 4) Crash proof: kill mid-tool-call, resume, assert no duplicate side effect.
python crash_test.py      # -> "PASS: no duplicate side effect across the commit-gap crash."

# 5) Tests green.
pytest -q                 # 3 passed
```

You now have the Week-1 agent running under a durable executor: state survives restarts, a human can gate the dangerous tool, and a mid-flight crash resumes without double-firing.

---

## Definition of Done — verifiable checks

Restating the spine's Week 3 checklist as things you can actually run:

- [ ] **Graph compiles & solves the Week-1 task.** `python run.py demo-1` completes end-to-end (balance lookup + invoice) and prints a correct final answer. `python -c "from graph import GRAPH; print(GRAPH)"` imports cleanly.
- [ ] **Checkpointer works.** After a run, the store holds multiple checkpoints for the `thread_id`: `python -c "import sqlite3; print(sqlite3.connect('checkpoints.sqlite').execute('select count(*) from checkpoints').fetchone())"` is ≥ number of super-steps, or `get_state_history(cfg)` length ≥ 3.
- [ ] **HITL across a restart.** `hitl.py` pauses at `interrupt()`; killing the process and running `resume_hitl.py` with `Command(resume=...)` on the **same `thread_id`** completes the run.
- [ ] **Crash proof, both kill positions.** `crash_test.py` prints `PASS` — `side_effects.log` line count is identical before and after resume for the commit-gap crash (b); the post-commit crash (a) resumes with the tool never re-entering. Removing the idempotency guard makes (b) fail with `1 -> 2` (demonstrated once, then restored).
- [ ] **`pytest` green.** `test_durability.py` asserts (1) resume produces a final answer, (2) side-effect count invariant across crash+resume, (3) guard returns the `already-*` path on replay. `pytest -q` = 3 passed.
- [ ] **Framework fluency.** You can state each control model in one line — *graph/state-machine (LangGraph), conversation-orchestration (CrewAI/AutoGen/OpenAI Agents SDK), typed loop (Pydantic AI/Claude Agent SDK)* — and give a "use X not Y because…" default (e.g. *"LangGraph over OpenAI Agents SDK here because I need inspectable, checkpointable control flow and a crash-safe resume, not emergent handoffs."*).

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `No module named 'langgraph.checkpoint.sqlite'` | checkpointer package not installed | `uv add langgraph-checkpoint-sqlite` (ships separately) |
| Resume re-runs everything from scratch | new `thread_id`, or input wasn't `None` | reuse the exact id; pass `None` as input to resume |
| `no such table: checkpoints` | queried before any run, or used `MemorySaver` | run once with `SqliteSaver`/`PostgresSaver` first |
| `MemorySaver` "loses" state on restart | in-memory checkpointer evaporates | never use it for durability — use SQLite/Postgres |
| Duplicate side effect (`1 -> 2`) after crash | idempotency key not stable / guard missing | key must be a pure fn of action+args; keep the `applied()` early-return |
| `after crash: ledger lines = 0` | crashed before `record()`, or no tool call | put `os._exit(1)` after `record()`; make the task clearly need the tool |
| Postgres `relation "checkpoints" does not exist` | skipped one-time setup | call `cp.setup()` once before the first stream |
| `ValidationError: temperature not permitted` | set `temperature` on Claude Opus/Sonnet | remove it for Anthropic; keep `temperature=0` only on Ollama/OpenAI |
| Graph ends with no tool call | model not emitting tool calls | confirm `bind_tools(TOOLS)` and a tool-calling-capable model (llama3.1 ok) |
| `interrupt` "no interrupt to resume" | thread wasn't actually paused | check `state.next` non-empty before killing; drive via `stream`/`invoke` |
| `compiled ... outside checkpointer context` | compiled outside the `with SqliteSaver(...)` block | compile + stream inside the context manager |

---

## Stretch goals (optional)

- **Async service shape.** Stand the same graph behind a FastAPI endpoint that accepts a `thread_id` per request (`POST /invoke`, `POST /resume`), so any worker process can resume any thread. This is the event-driven shape from lecture 14 — the step that turns "a script" into "a service." Use `app.ainvoke` / `app.astream` and an async checkpointer (`AsyncSqliteSaver` / `AsyncPostgresSaver`).
- **Idempotency with a real external side effect.** Replace the ledger write in `send_invoice` with a call to a local fake HTTP endpoint (a 5-line Flask app) and pass the idempotency key as an `Idempotency-Key` header — mirroring how Stripe actually dedups. Verify the server sees the key only once across a crash+resume.
- **Time-travel debugging.** Build a tiny CLI that lists a thread's checkpoints with their node names, lets you pick one, edit the state (`app.update_state`), and fork a corrected branch — the reproducible "why did it do that, and what if it hadn't" workflow.
- **Graduate the mental model to Temporal.** Read the lecture 12 section on execution durability, then sketch (don't build) how this agent would map to Temporal: which parts are deterministic *workflow* code and which are non-deterministic *activities*. Write two sentences on when you'd actually make that jump (multi-service, multi-day, strict exactly-once across systems) — and why LangGraph checkpointing is enough for a single process today.
