# Week 2 Lab: Plan-and-Execute vs ReWOO — Instrument, Compare, Route

> You build the **same multi-hop research task two ways** — Plan-and-Execute (planner → execute loop → replanner) and ReWOO (one planning call → dumb worker → one solve call) — then let an instrumented trace decide the winner on **tokens, latency, and quality**, not vibes. You finish with a one-page decision memo and a tiny router that dispatches queries to the cheapest topology that can serve them. The whole point of the week: **the simplest topology that solves the task wins**, and you prove it with numbers.
>
> Read these lectures **first** (they are the *why*; this lab is the *build*):
> [6 · Control-Flow Landscape & ReAct's Cost Model](../lectures/06-control-flow-landscape-react-cost.md) · [7 · Plan-and-Execute](../lectures/07-plan-and-execute.md) · [8 · ReWOO](../lectures/08-rewoo.md) · [9 · Parallelism, Reflection & Search](../lectures/09-parallelism-reflection-search.md) · [10 · Workflow-vs-Autonomous Spectrum & Routing](../lectures/10-workflow-autonomous-spectrum-routing.md)

**Est. time:** ~9 hrs · **You will need:** Python 3.10+, a terminal, and an LLM backend. **Free/local path (default): [Ollama](https://ollama.com) serving `llama3.1:8b` over its OpenAI-compatible endpoint (`http://localhost:11434/v1`) — no API key, no GPU required, runs CPU-only on a laptop.** Optional paid fallback: `gpt-4o-mini` (a few cents) if the 8B planner is too flaky for the ReWOO `#E` format. `tiktoken` for token counting, `rich` for tables, `pytest` for tests.

---

## Before you start (setup)

We give you **two paths**. The primary path is **framework-free**: a hand-rolled Plan-and-Execute and ReWOO on the raw OpenAI-compatible SDK pointed at Ollama. This keeps every LLM call and every token in *your* code so the comparison is honest and reproducible. (The spine also allows LangGraph; if you prefer it, do the framework-free version first so you understand what the graph hides, then port — notes at the end.)

**1. Install Ollama and pull the model (free/local path).**

```bash
# Install from https://ollama.com  (macOS/Linux/Windows installers)
ollama pull llama3.1:8b        # ~4.7 GB; native tool calling, runs CPU-only
# sanity check the server is up (Ollama auto-starts a daemon on :11434):
curl http://localhost:11434/v1/models
```

> **Windows / Git-Bash notes.** `curl` ships with Git-Bash and Windows 10/11. If Ollama isn't running, launch the "Ollama" app from the Start menu (it runs a tray daemon), or run `ollama serve` in a separate terminal. The daemon listens on `127.0.0.1:11434` on all platforms.

**2. Create the project and virtualenv.**

```bash
mkdir week2-planning && cd week2-planning
python -m venv .venv
source .venv/bin/activate          # Windows (Git-Bash): source .venv/Scripts/activate
                                   # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install openai tiktoken rich python-dotenv pytest
```

> We install the **`openai`** SDK, not `langchain-*`. Ollama exposes an OpenAI-compatible API, so the *same* client talks to Ollama locally or to OpenAI in the cloud — you switch backends by changing `base_url` and `api_key`, nothing else.

**3. Create `.env` so the backend is a one-line switch.**

```bash
cat > .env <<'EOF'
# Free/local default (Ollama OpenAI-compatible endpoint):
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama            # Ollama ignores the value but the SDK requires a non-empty string
MODEL=llama3.1:8b

# Paid fallback (uncomment, comment the block above):
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...
# MODEL=gpt-4o-mini
EOF
```

**4. Folder layout you'll fill in (matches the spine):**

```
week2-planning/
  .env
  tools.py            # shared mock (later real) tools + a CALLS ledger
  llm.py              # one place that creates the client + a metered chat() call
  instrument.py       # Meter: token + LLM-call + latency counting
  plan_execute.py     # Pattern A: planner -> execute loop -> replanner
  rewoo.py            # Pattern B: planner (1 call) -> worker (0 LLM) -> solver (1 call)
  router.py           # bonus: dispatch a query to the cheapest topology
  run_compare.py      # runs both 5x, prints median comparison table
  tests/test_patterns.py
  RESULTS.md          # your decision memo
```

**The shared task — use exactly this so results are comparable (from the spine):**

> "Which of these three cities — the capital of France, the most populous city in Japan, and the city hosting the 2028 Summer Olympics — has the largest metro-area population, and what is that number?"

It is deliberately chosen: it needs ~6 tool calls, the plan is **predictable** (great for ReWOO), and the lookups are **mostly independent** (so later you can *see* why LLM Compiler would help). Correct answer: **Tokyo, 37.4 million**. Start with **mocked** facts so token/latency numbers are reproducible; swapping in a real search API is a stretch goal.

**Golden rule for a fair fight:** use the **same model, same tool, same temperature** for both patterns. Different anything = meaningless numbers. (See lecture 8's "unfair comparison" trap.)

---

## Step-by-step

### Step 1 — Shared tools + a call ledger

**What.** A single `search(query)` tool both patterns call, plus a global `CALLS` list that records every invocation and its latency.

**Why.** Both patterns must hit the *identical* tool so the only variable is the topology. The ledger gives you `tool_calls` and per-call latency for free — no guessing. Mocked facts make runs deterministic (lecture 6: "mock is fine and makes token/latency numbers reproducible").

**Do it.** Create `tools.py`:

```python
# tools.py
import time

CALLS = []  # list of (tool_name, query, latency_s)

FACTS = {  # mocked knowledge so runs are reproducible
    "capital of france": "Paris",
    "most populous city in japan": "Tokyo",
    "2028 summer olympics host": "Los Angeles",
    "paris metro population": "11.2 million",
    "tokyo metro population": "37.4 million",
    "los angeles metro population": "12.9 million",
}

def search(query: str) -> str:
    """Look up a fact. Returns a short string, never raises (errors-as-observations)."""
    t0 = time.perf_counter()
    time.sleep(0.15)                       # simulate network so latency deltas are visible
    q = query.lower().strip()
    ans = next((v for k, v in FACTS.items() if k in q), "no result")
    CALLS.append(("search", query, time.perf_counter() - t0))
    return ans
```

**Expected result.** `python -c "import tools; print(tools.search('capital of France')); print(tools.CALLS)"` prints `Paris` and a one-row ledger.

**Verify.**
```bash
python -c "import tools; print(tools.search('tokyo metro population'))"   # -> 37.4 million
```

**Troubleshoot.** If a lookup returns `no result`, your query text doesn't contain any `FACTS` key substring. The matcher is intentionally simple (substring, lowercased) — keep planner-emitted queries close to the keys, or add synonyms to `FACTS`.

---

### Step 2 — One metered LLM entry point

**What.** A single `chat()` function that every pattern uses, wired to a `Meter` that counts LLM calls and input/output tokens.

**Why.** If both patterns route every model call through the same instrumented function, your counts are apples-to-apples. Centralizing the client also makes the Ollama↔OpenAI switch a one-liner. Note the honesty caveat from the spine: `tiktoken` on prompt strings **approximates** — it ignores system/tool-schema overhead. For OpenAI you can read exact `usage` off the response; state whichever method you used in `RESULTS.md`.

**Do it.** Create `instrument.py`:

```python
# instrument.py
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def ntok(s: str) -> int:
    return len(enc.encode(s or ""))

class Meter:
    """Counts LLM calls and approximate in/out tokens across a single run."""
    def __init__(self):
        self.llm_calls = 0
        self.in_tok = 0
        self.out_tok = 0

    def record(self, prompt_text: str, completion_text: str, usage=None):
        self.llm_calls += 1
        if usage is not None:                     # exact path (OpenAI returns usage)
            self.in_tok  += usage.prompt_tokens
            self.out_tok += usage.completion_tokens
        else:                                     # approximate path (tiktoken)
            self.in_tok  += ntok(prompt_text)
            self.out_tok += ntok(completion_text)

    def report(self):
        return dict(llm_calls=self.llm_calls, in_tok=self.in_tok,
                    out_tok=self.out_tok, total_tok=self.in_tok + self.out_tok)
```

Create `llm.py`:

```python
# llm.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                 api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.environ["MODEL"]

def chat(messages, meter=None, temperature=0) -> str:
    """One metered chat completion. Returns the assistant text.
    temperature=0 works on Ollama and gpt-4o-mini (determinism lever)."""
    resp = _client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature)
    text = resp.choices[0].message.content or ""
    if meter is not None:
        prompt_text = "\n".join(m["content"] for m in messages)
        # Ollama's OpenAI-compatible API also returns usage; prefer it when present.
        meter.record(prompt_text, text, usage=getattr(resp, "usage", None))
    return text
```

**Expected result.** A quick smoke test returns text and increments the meter.

**Verify.**
```bash
python -c "from llm import chat; from instrument import Meter; m=Meter(); print(chat([{'role':'user','content':'Reply with the single word: ok'}], meter=m)); print(m.report())"
```
You should see `ok` (roughly) and `llm_calls=1` with non-zero tokens.

**Troubleshoot.**
- **`Connection refused` / timeout** → Ollama isn't running. Start the app or `ollama serve`; re-check `curl http://localhost:11434/v1/models`.
- **`model not found`** → `ollama pull llama3.1:8b` first; make sure `MODEL` in `.env` matches exactly (tag included).
- **`usage` is `None` on Ollama** → fine, you fall back to `tiktoken` (approximate). Note that in the memo.

---

### Step 3 — Pattern A: Plan-and-Execute

**What.** Planner LLM writes a step list up front; an executor works one step at a time (calling `search`); a **replanner** LLM looks at results-so-far after each step and decides *continue with a revised plan* or *finish*.

**Why.** This is the pattern that **can adapt mid-course** — the replanner sees observations and can change the remaining plan (lecture 7). That flexibility is exactly what it *pays for* in tokens and latency: the replanner re-feeds growing context and calls are serial. You want to *measure* that cost.

**Do it.** Sketch of `plan_execute.py` — implement the bodies; signatures and control flow given so you don't get stuck:

```python
# plan_execute.py
import json, re
from llm import chat
from tools import search

PLANNER_SYS = (
    "You are a planner. Given a question, output a JSON list of short search-query "
    'steps needed to answer it. Example: ["capital of France", "Paris metro population"]. '
    "Output ONLY the JSON list."
)

def _parse_steps(text: str) -> list[str]:
    """Extract a JSON list of steps from planner output; tolerate stray prose."""
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0)) if m else []

def run_plan_execute(question: str, meter=None, max_iters: int = 8) -> str:
    # 1) PLAN (1 LLM call): produce initial steps
    plan = _parse_steps(chat(
        [{"role": "system", "content": PLANNER_SYS},
         {"role": "user", "content": question}], meter=meter))

    evidence = []                                  # (step, observation)
    for _ in range(max_iters):
        if not plan:
            break
        step = plan.pop(0)
        obs = search(step)                         # EXECUTE one step (no LLM)
        evidence.append((step, obs))

        # 2) REPLAN (1 LLM call PER STEP): the replanner SEES observations -> token cost grows
        rp = chat([{"role": "system", "content":
            "You are a replanner. Given the question, evidence gathered, and remaining plan, "
            'output JSON: {"done": bool, "steps": [remaining search queries]}. '
            "If evidence is enough to answer, set done=true and steps=[]."},
            {"role": "user", "content":
             f"Question: {question}\nEvidence: {evidence}\nRemaining plan: {plan}"}],
            meter=meter)
        m = re.search(r"\{.*\}", rp, re.S)
        decision = json.loads(m.group(0)) if m else {"done": True, "steps": []}
        if decision.get("done"):
            break
        plan = decision.get("steps", plan)

    # 3) SOLVE (1 LLM call): compose the final answer from all evidence
    return chat([{"role": "system", "content":
        "Answer the question using ONLY the evidence. Give the city AND the number."},
        {"role": "user", "content": f"Question: {question}\nEvidence: {evidence}"}],
        meter=meter)
```

Notice the cost signature: **1 plan + N replans + 1 solve** LLM calls, and each replan re-feeds `evidence`. That's the point.

**Expected result.** Running it end-to-end returns a sentence naming **Tokyo** and **37.4 million**, and `meter.report()` shows roughly `llm_calls ≈ 5–8` with a comparatively **high** `total_tok`.

**Verify.**
```bash
python -c "from plan_execute import run_plan_execute; from instrument import Meter; m=Meter(); print(run_plan_execute('Which of these three cities — the capital of France, the most populous city in Japan, and the city hosting the 2028 Summer Olympics — has the largest metro-area population, and what is that number?', meter=m)); print(m.report())"
```

**Troubleshoot.**
- **Planner emits prose, not JSON** → the `_parse_steps` regex grabs the first `[...]`; if 8B still refuses, add "Output ONLY the JSON list, no prose." and lower `temperature=0`. Log the raw text when parsing fails.
- **Infinite-ish looping** → `max_iters` caps it; if the replanner never says `done`, tighten its instruction ("set done=true as soon as you can answer").
- **Wrong number** → the planner likely searched `"tokyo population"` (not in FACTS) instead of `"tokyo metro population"`. Nudge the planner prompt toward the phrasing the tool understands, or enrich `FACTS`.

---

### Step 4 — Pattern B: ReWOO (Reasoning WithOut Observation)

**What.** Three nodes: **Planner** emits the *entire* plan with variable placeholders (`#E1`, `#E2`, …) in **one** LLM call; a **Worker** (NO LLM) executes each `search[...]`, substituting already-resolved `#E` variables; a **Solver** composes the final answer in **one** LLM call given plan + all evidence.

**Why.** The planner **never sees tool observations** — that's the entire reason ReWOO exists (lecture 8). It collapses to **exactly 2 LLM calls total** regardless of step count, versus Plan-and-Execute's per-step replans. Big token win *when the plan is predictable*. The tradeoff it gives up: **no mid-course correction** — if step 2's result should change the plan, ReWOO can't adapt.

**Do it.** Sketch of `rewoo.py` — the planner prompt, the parse regex, and the substitution logic are the load-bearing parts:

```python
# rewoo.py
import re
from llm import chat
from tools import search

PLANNER_SYS = (
    "You are a ReWOO planner. Produce a plan where each line binds a variable to a tool call. "
    "Use ONLY this format, one per line:\n"
    "#E1 = search[the query text]\n"
    "You may reference earlier results by writing #E1 inside a later query, e.g. "
    "#E4 = search[#E1 metro population]. Emit ONLY the plan lines."
)

STEP_RE = re.compile(r"(#E\d+)\s*=\s*(\w+)\[([^\]]+)\]")   # -> (var, tool, arg)

def _parse_plan(plan: str):
    """Return ordered [(var, tool, arg_template), ...]. One repair retry lives in run_rewoo."""
    return STEP_RE.findall(plan)

def run_rewoo(question: str, meter=None) -> str:
    # 1) PLAN — ONE LLM call, never sees observations
    plan = chat([{"role": "system", "content": PLANNER_SYS},
                 {"role": "user", "content": question}], meter=meter)
    steps = _parse_plan(plan)
    if not steps:                                   # one repair retry (see Pitfalls in spine)
        plan = chat([{"role": "system", "content": PLANNER_SYS +
            "\nYour last output had NO valid '#En = tool[arg]' lines. Try again, format ONLY."},
            {"role": "user", "content": question}], meter=meter)
        steps = _parse_plan(plan)

    # 2) WORK — NO LLM. Substitute resolved vars, call the tool, store results.
    DISPATCH = {"search": search}
    results: dict[str, str] = {}
    for var, tool, arg in steps:
        for done_var, val in results.items():       # fill #E1, #E2, ... into this arg
            arg = arg.replace(done_var, val)
        results[var] = DISPATCH[tool](arg)

    # 3) SOLVE — ONE LLM call, given plan + evidence
    evidence = "\n".join(f"{v} = {results[v]}" for v, _, _ in steps)
    return chat([{"role": "system", "content":
        "Using the plan and evidence, answer the question. Give the city AND the number."},
        {"role": "user", "content":
         f"Question: {question}\nPlan:\n{plan}\nEvidence:\n{evidence}"}], meter=meter)
```

**Expected result.** Returns **Tokyo / 37.4 million** and `meter.report()` shows **`llm_calls == 2`** (or 3 if the repair retry fired) with a markedly **lower** `total_tok` than Plan-and-Execute.

**Verify.**
```bash
python -c "from rewoo import run_rewoo; from instrument import Meter; m=Meter(); print(run_rewoo('Which of these three cities — the capital of France, the most populous city in Japan, and the city hosting the 2028 Summer Olympics — has the largest metro-area population, and what is that number?', meter=m)); print(m.report())"
```
Confirm `llm_calls` is 2 (the DoD hinges on this).

**Troubleshoot.**
- **`_parse_plan` returns `[]`** → the 8B model botched the `#E` format (the classic ReWOO failure, lecture 8). The one repair retry helps; if it *still* fails, switch `MODEL=gpt-4o-mini` in `.env` — this is exactly the "spend a few cents" signal. **Always log the raw `plan` string** so you can see what it emitted.
- **`llm_calls == 3` unexpectedly** → the repair retry fired; fix the planner prompt rather than accepting it, or your token comparison will be slightly unfair.
- **Worker `KeyError`** → the model invented a tool name other than `search`; constrain the prompt to "use ONLY the tool `search`".
- **Variables not substituting** → your later query must contain the literal `#E1` token; check the planner used `#E1` (not `E1` or `{E1}`). The regex only recognizes `#E\d+`.

---

### Step 5 — Measure and compare (median of 5 runs)

**What.** A harness that runs each pattern 5× and prints a comparison table of median `llm_calls`, `total_tok`, `wall_s`, `tool_calls`, and correctness.

**Why.** LLMs are noisy; a single run lies. The DoD requires **median of 5**. This table is the evidence your memo cites (lecture 6/10: measure, don't guess).

**Do it.** Create `run_compare.py`:

```python
# run_compare.py
import time, statistics
from rich.table import Table
from rich.console import Console
import tools
from instrument import Meter
from plan_execute import run_plan_execute
from rewoo import run_rewoo

QUESTION = ("Which of these three cities — the capital of France, the most populous city "
            "in Japan, and the city hosting the 2028 Summer Olympics — has the largest "
            "metro-area population, and what is that number?")

def _correct(ans: str) -> bool:
    a = ans.lower()
    return "tokyo" in a and "37.4" in a

def one_run(fn):
    tools.CALLS.clear()
    m = Meter(); t0 = time.perf_counter()
    ans = fn(QUESTION, meter=m)
    return dict(answer=ans, wall_s=round(time.perf_counter() - t0, 2),
                tool_calls=len(tools.CALLS), correct=_correct(ans), **m.report())

def median_runs(fn, n=5):
    runs = [one_run(fn) for _ in range(n)]
    agg = lambda k: statistics.median(r[k] for r in runs)
    return dict(llm_calls=agg("llm_calls"), total_tok=agg("total_tok"),
                wall_s=agg("wall_s"), tool_calls=agg("tool_calls"),
                correct=sum(r["correct"] for r in runs), n=n)

if __name__ == "__main__":
    rows = {"plan_execute": median_runs(run_plan_execute),
            "rewoo":        median_runs(run_rewoo)}
    t = Table(title="Plan-and-Execute vs ReWOO (median of 5)")
    for c in ["pattern", "llm_calls", "total_tok", "wall_s", "tool_calls", "correct/5"]:
        t.add_column(c)
    for name, r in rows.items():
        t.add_row(name, str(r["llm_calls"]), str(r["total_tok"]), str(r["wall_s"]),
                  str(r["tool_calls"]), f'{r["correct"]}/{r["n"]}')
    Console().print(t)
```

**Expected result.** A table roughly like:

| pattern | llm_calls | total_tok | wall_s | tool_calls | correct/5 |
|---|---|---|---|---|---|
| plan_execute | ~5–8 | (high) | (high) | 6 | 5/5 |
| rewoo | 2 | (low) | (lower) | 6 | 5/5 |

ReWOO should show **≥2× fewer total tokens** and fewer LLM calls.

**Verify.**
```bash
python run_compare.py
```

**Troubleshoot.**
- **ReWOO shows ~the same tokens as Plan-and-Execute** → your ReWOO is leaking observations into the planner (the #1 bug in the spine's self-check). The planner must be called **once** with only the question; evidence goes to the **solver** only. Re-check Step 4.
- **`correct` is low for both** → mock keys / planner phrasing mismatch (see Step 3/4 troubleshooting). Fix correctness before trusting token numbers.
- **`wall_s` dominated by `time.sleep(0.15)`** → expected; both patterns make 6 tool calls so sleep cancels out. The *LLM* time is the real latency delta.

---

### Step 6 — The decision memo (`RESULTS.md`)

**What.** A one-page writeup: the numbers table, the winner per axis, and at least one **flip condition**.

**Why.** The engineering skill this week isn't building either pattern — it's *deciding* and being able to defend it, including when the answer flips (lecture 10). A result without a flip condition is over-generalized.

**Do it.** Paste your table into `RESULTS.md` and answer, in prose:
1. **Which won on tokens?** (Expect ReWOO, ~2–5× fewer — because it makes 2 LLM calls and the planner never re-reads observations.)
2. **Which won on latency?** (Expect ReWOO — fewer serial LLM round-trips.)
3. **Which won on quality/robustness?** (Tie on *this* task since the plan is predictable.)
4. **State the measurement method** you used (`tiktoken` approximate vs provider `usage`) — being honest about measurement is part of the exercise.
5. **The flip condition**, e.g.: *"ReWOO wins tokens ~3–5× here because the plan is predictable; the moment a step result should change the plan (a lookup fails and needs a fallback query, or one city's number is missing), Plan-and-Execute's replanner wins on correctness."*
6. One sentence pointing forward: *"If I needed the independent lookups faster, the next move is **LLM Compiler** (parallel DAG), not more autonomy."*

**Expected result.** `RESULTS.md` has the table + memo naming a winner and ≥1 flip condition.

**Verify.** Re-read it: could a teammate pick the right pattern for a *new* task from your memo alone? If not, sharpen the flip condition.

**Troubleshoot.** If you can't articulate a flip condition, force one empirically: make `FACTS["tokyo metro population"]` return `"no result"` and observe ReWOO confidently produce a wrong/empty answer (no adaptation) while Plan-and-Execute's replanner *could* issue a fallback query. That contrast is your flip condition.

---

### Step 7 — Router (bonus)

**What.** `router.py`: a tiny classifier that reads a query and returns one of `{"single_react", "plan_execute", "rewoo"}`, then dispatches to the matching handler.

**Why.** This is Anthropic's **routing** building block made concrete (lecture 10) — the pattern that makes the rest economical: send each request to the *cheapest handler that can serve it*. Encode the policy from your memo.

**Do it.** Start with a cheap keyword/heuristic router (no LLM needed for the DoD); upgrade to a one-call LLM classifier as a stretch.

```python
# router.py
import re
from plan_execute import run_plan_execute
from rewoo import run_rewoo
# from your Week-1 agent, or a stub, for the single-fact case:
# from single_react import run_single_react

def classify(query: str) -> str:
    q = query.lower()
    multi_hop = len(re.findall(r"\b(and|,|compare|largest|which of)\b", q)) >= 2
    adaptive  = any(w in q for w in ["if", "depending", "otherwise", "fallback", "latest"])
    if not multi_hop:
        return "single_react"            # single fact -> cheapest
    return "plan_execute" if adaptive else "rewoo"   # predictable multi-hop -> rewoo

DISPATCH = {"rewoo": run_rewoo, "plan_execute": run_plan_execute,
            # "single_react": run_single_react,
            }

def route(query: str, meter=None) -> str:
    choice = classify(query)
    return DISPATCH[choice](query, meter=meter)
```

**Expected result.** The shared multi-hop question classifies to `"rewoo"`; a single-fact query ("What is the capital of France?") classifies to `"single_react"`; an adaptive query ("Find the latest host city; if the lookup fails, try the fallback source") classifies to `"plan_execute"`.

**Verify.**
```bash
python -c "from router import classify; print(classify('What is the capital of France?')); print(classify('Which of these three cities ... largest metro-area population, and the number?')); print(classify('Find the latest data; if it fails, use a fallback query'))"
```
Expect `single_react`, `rewoo`, `plan_execute`.

**Troubleshoot.** Heuristics are brittle by design — if a query misroutes, that's your cue for the stretch (a one-call LLM classifier returning strictly one of the three labels). Keep the label set closed; validate the LLM's output against it and default to `plan_execute` (the safe, adaptive choice) on any parse miss.

---

### Step 8 — Tests

**What.** `tests/test_patterns.py`: assert both patterns answer correctly, ReWOO makes exactly 2 LLM calls, and the router sends the multi-hop question to `rewoo`.

**Why.** The DoD requires `pytest -q` green with ≥4 assertions including "ReWOO == 2 LLM calls" — that assertion is what catches the observation-leak regression.

**Do it.** Create `tests/test_patterns.py`:

```python
# tests/test_patterns.py
from instrument import Meter
from plan_execute import run_plan_execute
from rewoo import run_rewoo
from router import classify

Q = ("Which of these three cities — the capital of France, the most populous city in Japan, "
     "and the city hosting the 2028 Summer Olympics — has the largest metro-area population, "
     "and what is that number?")

def _ok(ans): a = ans.lower(); return "tokyo" in a and "37.4" in a

def test_plan_execute_correct():
    assert _ok(run_plan_execute(Q))

def test_rewoo_correct():
    assert _ok(run_rewoo(Q))

def test_rewoo_makes_two_llm_calls():
    m = Meter(); run_rewoo(Q, meter=m)
    assert m.report()["llm_calls"] == 2      # allow ==3 only if you accept the repair retry

def test_router_sends_multihop_to_rewoo():
    assert classify(Q) == "rewoo"
```

**Expected result.** `pytest -q` is green (4 passing).

**Verify.**
```bash
pytest -q
```

**Troubleshoot.**
- **`test_rewoo_makes_two_llm_calls` fails with 3** → the repair retry fired; the 8B planner is unreliable on the `#E` format. Either switch to `gpt-4o-mini` (deterministic format) or relax the assertion to `in (2, 3)` *and note it in the memo* — but prefer fixing the planner.
- **Correctness tests flaky** → set `temperature=0` (already the default in `chat()`), and pin the exact model tag. Flakiness in the *answer* usually means the planner phrasing drifts from the mock keys.
- **Import errors under pytest** → run from the project root so `tools`, `llm`, etc. are importable; add an empty `tests/__init__.py` if needed, or `python -m pytest -q`.

---

## Putting it together — a short end-to-end run

```bash
# from week2-planning/ with .venv active and Ollama running:
python run_compare.py        # median-of-5 table: rewoo should show >=2x fewer tokens & fewer calls
pytest -q                    # 4 assertions green (incl. "ReWOO == 2 LLM calls")
python -c "from router import route; from instrument import Meter; m=Meter(); \
print(route('What is the capital of France?', meter=m)); print(m.report())"
```

Then open `RESULTS.md` and confirm it contains the numbers table, the per-axis winner, and a concrete flip condition. That sequence *is* the week: build both, measure both, decide with evidence, and encode the decision in a router.

---

## Definition of Done — verifiable checks (restated from the spine)

- [ ] **Both patterns solve the same question.** `plan_execute.py` and `rewoo.py` run end-to-end on the identical question and return the correct answer (**Tokyo, 37.4M**). → `pytest -q` passes `test_*_correct`.
- [ ] **Real instrumentation, not guesses.** A trace/printout shows **per-run token counts, LLM-call count, and wall-clock latency** for each pattern, all from `Meter` + `time.perf_counter()`. → `python run_compare.py` prints them.
- [ ] **Median-of-5 comparison table** where **ReWOO shows ≥2× fewer total tokens and fewer LLM calls** than Plan-and-Execute. (If it doesn't, your ReWOO is leaking observations into the planner — fix it.) → the `run_compare.py` table.
- [ ] **`RESULTS.md`** contains the numbers table + a memo naming the winner **and ≥1 concrete flip condition**, plus which token-measurement method you used.
- [ ] **`router.py` correctly dispatches ≥3 differently-shaped sample queries** (single-fact → single_react, predictable multi-hop → rewoo, adaptive → plan_execute).
- [ ] **`pytest -q` green with ≥4 assertions**, including **"ReWOO == 2 LLM calls."**
- [ ] **You didn't build LATS or LLM Compiler.** You can state in one sentence *why not* for this task — that restraint is the week's actual lesson.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` on any LLM call | Ollama daemon not running | Launch Ollama app or `ollama serve`; verify `curl http://localhost:11434/v1/models` |
| `model not found` | model not pulled / `MODEL` typo | `ollama pull llama3.1:8b`; match `MODEL` in `.env` exactly (with tag) |
| ReWOO `_parse_plan` returns `[]` | 8B botched the `#E` format | One repair retry (built in); else set `MODEL=gpt-4o-mini`; always log raw `plan` |
| ReWOO tokens ≈ Plan-and-Execute | observations leaking into the planner | Planner called **once** with only the question; evidence goes to the **solver** |
| Wrong number in answer | planner query ≠ mock key | Nudge planner phrasing toward `FACTS` keys or enrich `FACTS` |
| `test_rewoo_makes_two_llm_calls` == 3 | repair retry fired | Fix planner prompt / use gpt-4o-mini; don't silently accept 3 |
| Flaky correctness | temperature / floating model | `temperature=0` (default); pin exact model tag; same model for BOTH patterns |
| Unfair comparison | different model/temp/tool per pattern | Use the *same* `chat()`, model, temperature, and `search` tool everywhere |
| `usage` is `None` (Ollama) | Ollama may omit usage | Falls back to `tiktoken` (approximate) — state this in `RESULTS.md` |

---

## Stretch goals (optional)

- **Real search instead of mocks.** Swap `search()` for the [Tavily](https://tavily.com) free tier or DuckDuckGo via the `ddgs` package. Re-run the comparison — note how variance and latency change and whether your flip condition now triggers naturally (a failed lookup needing a fallback).
- **LLM router.** Replace the keyword `classify()` with a single cheap LLM call that returns strictly one of the three labels; validate against the closed label set and default to `plan_execute` on any parse miss.
- **Exact token accounting.** On the OpenAI path, read `resp.usage` for exact counts (already wired in `Meter.record`); compare against the `tiktoken` estimate and quantify the gap in `RESULTS.md`.
- **See why LLM Compiler would help (don't build it fully).** Add per-tool-call timestamps to the `CALLS` ledger and compute how much wall-clock you'd save if the 4 independent lookups ran concurrently. Report the number — *that* is the case for LLM Compiler (lecture 9), and naming it is the point, not building it.
- **Port to LangGraph.** Now that you understand the control flow, rebuild both patterns as LangGraph graphs (planner/executor/replanner nodes; planner/worker/solver nodes) with `config={"callbacks":[meter]}`. Confirm the token/call numbers match your framework-free version — if they don't, the framework is hiding calls you should account for.
