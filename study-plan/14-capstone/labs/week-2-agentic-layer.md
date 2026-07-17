# Week 2 Lab: A Supervisor That Acts, Survives a Crash, and Proves Who Asked

> **What you build:** you take last week's tenant-isolated retrieval spine and bolt on the part that *does things* — a **LangGraph supervisor** routing between three *typed* tools (`sql_query` read-only, `doc_search` retrieval, `submit_action` write), where the write is gated by a **human-in-the-loop** approval, the whole run is durably **checkpointed to Postgres** so it survives a crash mid-run with **zero duplicated writes**, a **budget/kill guard** hard-stops a runaway loop between steps, and the agent reaches the outside world — **exposing one skill over A2A** and **consuming one external MCP tool** — both authorized by **OAuth scopes keyed to the end user**. Autonomy is easy; *safe, durable, auditable* autonomy is the deliverable.
>
> **Why:** a write that fires twice in a regulated domain is a duplicate claim or a double refund — a compliance incident. A prompt that says "please only SELECT" is a wish, not a control. A god-token that lets you prove *an* action happened but not *whose authority* it carried is unshippable. This lab makes each of those walls real on a laptop.
>
> **Read first (assumed done):** [05 — Supervisor topology & typed tool contracts](../lectures/05-supervisor-topology-and-typed-tool-contracts.md) · [06 — Durability: checkpointing + idempotent writes](../lectures/06-durability-checkpointing-and-idempotent-writes.md) · [07 — HITL gating & the budget kill-switch](../lectures/07-hitl-gating-and-budget-kill-switch.md) · [08 — End-user OAuth scopes across MCP-in and A2A-out](../lectures/08-end-user-oauth-scopes-across-mcp-and-a2a.md).

**Est. time:** ~9 hrs (Setup 0.5 · DB scope 1.25 · typed tools 1.25 · supervisor+HITL+checkpoint 2.5 · budget/kill 0.75 · MCP-in 1 · A2A-out+OAuth 1.25 · tests 0.75) · **You will need:** Python 3.11+, Docker, and an LLM for tool-calling. **Free/local path (no GPU, no paid key):** [Ollama](https://ollama.com) with `llama3.1:8b` for the LLM, **local Postgres in Docker** for checkpoints/memory/RLS, and **FastMCP** (the `mcp` SDK) + a plain FastAPI **A2A server** running on `localhost`. `curl` and `pytest` round it out. **Default (a few cents/run, more reliable tool-args):** `gpt-4o-mini` via `OPENAI_API_KEY`.

---

## Before you start (setup)

This week **extends the `capstone/` repo from Week 1** — you reuse the tenant-filtered retriever as `doc_search`. If you're starting fresh, a stub retriever is fine; the graded parts are the agentic controls, not retrieval quality.

**1. Install the stack** (in your existing venv):

```bash
cd capstone
python -m venv .venv && source .venv/bin/activate   # Git-Bash on Windows: same line works
# Native Windows PowerShell/cmd instead: .venv\Scripts\activate
pip install -U langgraph langchain langchain-core \
    langgraph-checkpoint-postgres \
    langchain-openai langchain-ollama langchain-mcp-adapters \
    "mcp[cli]" authlib "psycopg[binary]" pydantic \
    fastapi "uvicorn[standard]" httpx pytest python-dotenv rich
```

**2. Start Postgres** (reuse Week 1's container if it's already up on 5432):

```bash
docker run -d --name capstone-pg -p 5432:5432 \
  -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg16
# verify
docker exec -it capstone-pg psql -U postgres -c "select version();"
```

**3. Pick your model.** Default is `gpt-4o-mini` (set `OPENAI_API_KEY` in `.env`). Free path:

```bash
ollama pull llama3.1:8b     # ~4.7 GB; tool-calling works, keep tool schemas tiny
```

> **Git-Bash / Windows notes.** (a) Use forward slashes in paths inside Python; Git-Bash accepts them for CLI too. (b) If a Python process needs to reach Ollama or a service the *container* can't see, use `http://host.docker.internal:11434`; from the host itself use `localhost`. (c) `docker exec -it ... psql` is the portable way to run SQL — you don't need a local `psql` binary. (d) Killing a process "for real" (the crash test) differs by shell — commands given per-step below.

**4. Environment file** (`.env`, git-ignored):

```bash
DATABASE_URL=postgresql://postgres:pg@localhost:5432/postgres
OPENAI_API_KEY=sk-...        # omit if using Ollama
JWT_DEV_SECRET=dev-secret    # HS256, DEV ONLY — see Step 6
```

**New files you'll create this week:**

```
capstone/
  agent/
    graph.py          # supervisor + nodes + wiring + checkpointer
    tools.py          # typed tools: sql_query, doc_search, submit_action
    db_scope.py       # per-user read-only role + RLS session helper
    budget.py         # token/$ meter + guard node
    auth.py           # OAuth: mint + verify JWT, scope checks
    memory.py         # PostgresStore long-term memory helper (optional-but-cheap)
  interop/
    ext_mcp_server.py # a tiny local FastMCP server to consume (stand-in)
    mcp_client.py     # scope-gated loader of the external MCP tool
    a2a_server.py     # exposes ONE skill + Agent Card, scope-checked
  sql/
    01_roles_rls.sql  # read-only role, RLS policies, writes/idempotency table
  tests/
    test_hitl.py  test_resume.py  test_scopes.py  test_budget.py  test_db_scope.py
```

---

## Step-by-step

### Step 1 — The read-only wall: enforce DB scope AT the database

**What.** A Postgres role that can only `SELECT`, plus Row-Level Security so text-to-SQL physically cannot read another user's rows, plus a durable `writes` table with an idempotency primary key.

**Why.** "Please only SELECT" is theater — text-to-SQL *will* eventually emit a `DELETE` via a jailbreak or an instruction injected into a retrieved doc. Push the control to the lowest layer a prompt can't reach (lecture [05](../lectures/05-supervisor-topology-and-typed-tool-contracts.md)). Three independent walls must all be bypassed for a mutation to land: no write grant, read-only transaction, RLS on `app.user_id`.

**Do it.** Create `sql/01_roles_rls.sql`:

```sql
-- a demo table with owner-scoped rows
CREATE TABLE IF NOT EXISTS claims (
  id text PRIMARY KEY, owner_id text NOT NULL, amount numeric, status text);
INSERT INTO claims VALUES
  ('c1','u_alice',100,'open'), ('c2','u_alice',250,'paid'),
  ('c3','u_bob',999,'open') ON CONFLICT DO NOTHING;

-- (1) read-only role: no write grant exists
CREATE ROLE app_readonly NOLOGIN;
GRANT USAGE ON SCHEMA public TO app_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT TO app_readonly;

-- (2) row-level isolation keyed to a per-session setting
ALTER TABLE claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE claims FORCE ROW LEVEL SECURITY;   -- apply even to table owner
DROP POLICY IF EXISTS claims_user_isolation ON claims;
CREATE POLICY claims_user_isolation ON claims
  USING (owner_id = current_setting('app.user_id', true));

-- durable write log w/ idempotency: the dedup wall for crash-resume
CREATE TABLE IF NOT EXISTS writes (
  idempotency_key text PRIMARY KEY,
  user_id text NOT NULL, action text NOT NULL,
  payload jsonb NOT NULL, created_at timestamptz DEFAULT now());
```

Apply it:

```bash
docker exec -i capstone-pg psql -U postgres < sql/01_roles_rls.sql
```

Create `agent/db_scope.py`:

```python
import os, psycopg
from psycopg.rows import dict_row
DSN = os.environ["DATABASE_URL"]

def readonly_conn(user_id: str):
    """A connection dropped to app_readonly, scoped to one user, write-refusing."""
    c = psycopg.connect(DSN, row_factory=dict_row)
    c.execute("SET ROLE app_readonly;")                              # (1) no write grant
    c.execute("SELECT set_config('app.user_id', %s, false);", (user_id,))  # (2) RLS key
    c.execute("SET default_transaction_read_only = on;")             # (3) txn refuses writes
    return c
```

**Expected result.** As `app_readonly` with `app.user_id='u_alice'`, `SELECT * FROM claims` returns only Alice's 2 rows; a `DELETE`/`UPDATE`/`INSERT` raises a Postgres error.

**Verify.**

```bash
docker exec -it capstone-pg psql -U postgres -c \
"SET ROLE app_readonly; SELECT set_config('app.user_id','u_alice',false);
 SET default_transaction_read_only=on; SELECT id,owner_id FROM claims;"
# -> only c1, c2 (u_alice). Then prove a write errors:
docker exec -it capstone-pg psql -U postgres -c \
"SET ROLE app_readonly; DELETE FROM claims WHERE id='c1';"
# -> ERROR: permission denied for table claims  (or read-only transaction)
```

**Troubleshoot.** *RLS returns all rows / no rows:* you forgot `FORCE ROW LEVEL SECURITY` (owner bypasses policies otherwise), or `app.user_id` wasn't set in the same session. *`SET ROLE` fails:* run the grant script as the `postgres` superuser (the `docker exec` above does). *Rows leak across users:* the policy's `USING` clause must reference `current_setting('app.user_id', true)`, not a hardcoded value.

---

### Step 2 — Three typed tools, three trust levels

**What.** Define `sql_query` (read-only), `doc_search` (tenant-filtered retrieval), and `submit_action` (destructive write) as Pydantic-typed LangChain tools. Critically, `submit_action` **does not commit** — it only declares intent.

**Why.** The typed schema *is* the contract and the audit record (lecture [05](../lectures/05-supervisor-topology-and-typed-tool-contracts.md)): it makes each call validated, loggable, authorizable, rate-limitable, and unit-testable. Splitting three tools lets you attach *different* controls to each. The write's actual commit belongs in the graph node **after** approval — put it in the tool body and you've deleted your HITL gate.

**Do it.** Create `agent/tools.py`:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from .db_scope import readonly_conn

class SqlArgs(BaseModel):
    sql: str = Field(description="a single read-only SELECT statement")

@tool(args_schema=SqlArgs)
def sql_query(sql: str, *, user_id: str) -> list[dict]:
    """Run a read-only SELECT scoped to the current user."""
    with readonly_conn(user_id) as c:
        rows = c.execute(sql).fetchall()
    return rows[:50]                       # dict_row already gives dicts

class DocArgs(BaseModel):
    query: str = Field(description="natural-language search over the user's docs")

@tool(args_schema=DocArgs)
def doc_search(query: str, *, user_id: str) -> list[str]:
    """Retrieve top-k tenant-filtered docs (reuse Week-1 retriever)."""
    # from retrieval.search import search   # <- your Week-1 hybrid+ACL retriever
    # return [h["text"] for h in search(query, tenant_id=user_id, roles=[...])]
    return [f"[stub] doc snippet relevant to: {query}"]   # replace with Week-1 call

class ActionArgs(BaseModel):
    action: str = Field(description="the destructive action name, e.g. 'void_claim'")
    payload: dict = Field(description="action arguments")
    idempotency_key: str = Field(description="stable key; MUST be generated upstream")

@tool(args_schema=ActionArgs)
def submit_action(action: str, payload: dict, idempotency_key: str, *, user_id: str) -> str:
    """DESTRUCTIVE write. Commit is DEFERRED to the post-approval graph node.
    This tool only PROPOSES the write; do not add a DB mutation here."""
    return f"proposed:{action}"   # the graph node commits after HITL — see Step 3
```

**Expected result.** `sql_query.invoke({"sql": "SELECT * FROM claims", "user_id": "u_alice"})` returns Alice's rows; `submit_action` returns a `proposed:...` string and writes nothing.

**Verify.**

```bash
python -c "from agent.tools import sql_query; \
print(sql_query.invoke({'sql':'SELECT id FROM claims','user_id':'u_alice'}))"
# -> [{'id': 'c1'}, {'id': 'c2'}]
```

**Troubleshoot.** *`user_id` shows up as an LLM-fillable arg:* keyword-only args after `*` are injected by your code, not the model — but with some model backends you must pass them via config injection or a `functools.partial`; the simplest robust pattern is to bind `user_id` when you build the node (Step 3) rather than let the schema expose it. *Ollama emits malformed args:* keep schemas tiny (they are) and add a one-line description per field — the model uses them.

---

### Step 3 — The supervisor graph: HITL interrupt + durable Postgres checkpoint

**What.** A `StateGraph` with a supervisor LLM node that binds the three tools and routes to them or to an `action_node` that calls `interrupt()` **before** the DB mutation. Compile with `PostgresSaver`.

**Why.** This is the week's spine (lecture [06](../lectures/06-durability-checkpointing-and-idempotent-writes.md)). Checkpointing after every super-step means a crash resumes from the last checkpoint; `interrupt()` persists state and parks the run (possibly for days) until a human resumes — which only works because the checkpointer is durable (lecture [07](../lectures/07-hitl-gating-and-budget-kill-switch.md)). The commit is `INSERT ... ON CONFLICT DO NOTHING` on the idempotency key generated **upstream**, so an at-least-once replay writes exactly one row.

**Do it.** Create `agent/graph.py`:

```python
import os, hashlib, json, operator
from typing import TypedDict, Annotated
import psycopg
from psycopg.types.json import Json
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from .tools import sql_query, doc_search, submit_action
from .budget import guard, meter_llm            # Step 4
from .auth import scopes_from_state             # Step 6

DSN = os.environ["DATABASE_URL"]

class S(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str
    user_scopes: list[str]
    idempotency_key: str        # generated UPSTREAM, lives in checkpointed state
    tokens: int
    usd: float

def make_model():
    # default: reliable tool-calling
    try:
        from langchain_openai import ChatOpenAI
        if os.environ.get("OPENAI_API_KEY"):
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except Exception:
        pass
    from langchain_ollama import ChatOllama            # free/local fallback
    return ChatOllama(model="llama3.1:8b", temperature=0)

TOOLS = [sql_query, doc_search, submit_action]

def supervisor(state: S):
    model = make_model().bind_tools(TOOLS)
    resp = model.invoke(state["messages"])
    upd = meter_llm(resp)                     # accumulate tokens/usd (Step 4)
    return {"messages": [resp], **upd}

def route(state: S):
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    name = last.tool_calls[0]["name"]
    return "action" if name == "submit_action" else "tools"

def tools_node(state: S):
    """Execute read tools with user_id injected by us (never by the model)."""
    call = state["messages"][-1].tool_calls[0]
    fn = {"sql_query": sql_query, "doc_search": doc_search}[call["name"]]
    out = fn.invoke({**call["args"], "user_id": state["user_id"]})
    return {"messages": [ToolMessage(content=json.dumps(out, default=str),
                                     tool_call_id=call["id"])]}

def action_node(state: S):
    call = state["messages"][-1].tool_calls[0]
    args = call["args"]
    # (Step 6) scope check BEFORE we ever page a human
    if "action.write" not in state.get("user_scopes", []):
        return {"messages": [ToolMessage(content="403: needs action.write",
                                         tool_call_id=call["id"])]}
    decision = interrupt({                       # <-- PAUSES + checkpoints; no write yet
        "type": "approval_required",
        "action": args["action"], "payload": args["payload"]})
    if not decision.get("approved"):             # explicit approval only; silence = reject
        return {"messages": [ToolMessage(content="rejected by human",
                                         tool_call_id=call["id"])]}
    key = state["idempotency_key"]               # stable across resume
    with psycopg.connect(DSN) as c:
        c.execute("""INSERT INTO writes(idempotency_key,user_id,action,payload)
                     VALUES(%s,%s,%s,%s) ON CONFLICT (idempotency_key) DO NOTHING""",
                  (key, state["user_id"], args["action"], Json(args["payload"])))
        c.commit()
    return {"messages": [ToolMessage(content=f"committed:{args['action']}",
                                     tool_call_id=call["id"])]}

def build_graph(checkpointer):
    g = StateGraph(S)
    g.add_node("guard", guard)                   # Step 4: budget kill-switch
    g.add_node("supervisor", supervisor)
    g.add_node("tools", tools_node)
    g.add_node("action", action_node)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard",
        lambda s: END if s.get("_killed") else "supervisor", {"supervisor": "supervisor", END: END})
    g.add_conditional_edges("supervisor", route, {"tools": "tools", "action": "action", END: END})
    g.add_edge("tools", "guard")                 # loop back through the guard
    g.add_edge("action", END)
    return g.compile(checkpointer=checkpointer)

def upstream_key(user_id: str, action: str, payload: dict) -> str:
    """Deterministic idempotency key — computed BEFORE the interrupt, stored in state."""
    blob = json.dumps({"u": user_id, "a": action, "p": payload}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()

def open_saver():
    """Context manager yielding a PostgresSaver with tables ensured."""
    cm = PostgresSaver.from_conn_string(DSN)
    cp = cm.__enter__()
    cp.setup()                                   # idempotent; creates checkpoint tables
    return cm, cp
```

Run it once, driving to an interrupt and approving:

```python
# scripts/run_once.py
from dotenv import load_dotenv; load_dotenv()
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from agent.graph import build_graph, open_saver, upstream_key

cm, cp = open_saver()
try:
    app = build_graph(cp)
    key = upstream_key("u_alice", "void_claim", {"claim_id": "c1"})
    cfg = {"configurable": {"thread_id": "run-123"}}
    state = {"messages": [HumanMessage("Void claim c1 for me.")],
             "user_id": "u_alice", "user_scopes": ["kb.read", "action.write"],
             "idempotency_key": key, "tokens": 0, "usd": 0.0}
    out = app.invoke(state, cfg)
    print("interrupted?", "__interrupt__" in out)          # -> True, run parked, DB untouched
    out = app.invoke(Command(resume={"approved": True}), cfg)  # human approves -> commits
    print("done:", out["messages"][-1].content)
finally:
    cm.__exit__(None, None, None)
```

**Expected result.** First `invoke` returns with an `__interrupt__` marker and **no** row in `writes`. After `Command(resume={"approved": True})`, exactly one `writes` row exists.

**Verify.**

```bash
python -m scripts.run_once
docker exec -it capstone-pg psql -U postgres -c "SELECT idempotency_key, action FROM writes;"
# -> exactly one row for void_claim
```

**Troubleshoot.** *No `__interrupt__` returned:* you're on `MemorySaver`, or you never routed to `action` — confirm the model actually called `submit_action` (with Ollama, prompt more directly: "Use submit_action to void claim c1"). *`interrupt` import error:* it's `from langgraph.types import interrupt, Command` in current LangGraph. *`cp.setup()` fails:* the DB user needs create-table rights — use the `postgres` superuser DSN, not `app_readonly`.

---

### Step 4 — Budget meter + kill-switch guard node

**What.** A `Meter` that accumulates `tokens`/`usd` from each LLM response, and a `guard` node (runs every loop, before the supervisor) that routes to `END` past a `CAP_USD`, logging a `BUDGET_KILL` line.

**Why.** A looping agent burns real money. Stop it *between* steps at a node boundary so you leave a clean, resumable checkpoint — never a `SIGKILL` mid-write (lecture [07](../lectures/07-hitl-gating-and-budget-kill-switch.md)). Enforce at the graph level, not a wall-clock API timeout that can land anywhere.

**Do it.** Create `agent/budget.py`:

```python
CAP_USD = 0.05
PRICE = {"in": 0.15/1e6, "out": 0.60/1e6}   # gpt-4o-mini $/token; 0 for local Ollama

def meter_llm(resp) -> dict:
    """Read usage off the LLM response; price in/out separately. Reuse your agents-phase Meter."""
    u = getattr(resp, "usage_metadata", None) or {}
    it, ot = u.get("input_tokens", 0), u.get("output_tokens", 0)
    return {"tokens": it + ot, "usd": it*PRICE["in"] + ot*PRICE["out"]}

def guard(state) -> dict:
    if state.get("usd", 0.0) > CAP_USD:
        return {"_killed": True,
                "messages": [{"role": "system",
                              "content": f"BUDGET_KILL @ ${state['usd']:.4f} (cap ${CAP_USD})"}]}
    return {}
```

> `tokens`/`usd` accumulate because `S` declares no reducer for them — LangGraph overwrites with the returned value, so return the **running total**. If you prefer additive accumulation, make them `Annotated[int, operator.add]` and return only the *delta*. Pick one and be consistent. (The `supervisor` above returns the delta from `meter_llm`; use the `operator.add` variant for that to be a true cumulative total.)

**Expected result.** With `CAP_USD` set below one run's cost (e.g. `0.0`), the graph halts at `guard`, routes to `END`, and the last message is a `BUDGET_KILL` line; the checkpoint is valid and resumable.

**Verify.** In a test, set `CAP_USD = 0.0`, seed `usd` above 0 (or run one LLM turn), invoke, and assert the last message contains `BUDGET_KILL` and no new `writes` row appeared.

**Troubleshoot.** *Cap never trips:* Ollama reports no cost — for the free path, meter **tokens** and cap on those (`CAP_TOKENS`), or seed `usd` in the test. *Killed mid-write:* impossible if the guard only runs before `supervisor`/after `tools`; never put it inside `action_node`.

---

### Step 5 — Consume an external MCP tool (MCP-in), scope-gated

**What.** A tiny local **FastMCP** server exposing one canned tool, and a loader that only returns it when the user's scopes include `mcp.read`.

**Why.** MCP standardizes "here are the tools a server exposes" so any client can call them (lecture [08](../lectures/08-end-user-oauth-scopes-across-mcp-and-a2a.md)). Gate at **load time**: an unscoped user never receives the tool, so the supervisor can't call — or hallucinate a call to — a tool that isn't in its action space.

**Do it.** Create `interop/ext_mcp_server.py`:

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("ext")

@mcp.tool()
def fx_rate(base: str, quote: str) -> float:
    """Return a (canned) FX rate."""
    return 0.92

if __name__ == "__main__":
    mcp.run()          # stdio transport by default
```

Create `interop/mcp_client.py`:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async def load_mcp_tools(user_scopes: set[str]):
    if "mcp.read" not in user_scopes:
        raise PermissionError("403: needs mcp.read")        # gate BEFORE connecting
    client = MultiServerMCPClient({
        "ext": {"command": "python", "args": ["interop/ext_mcp_server.py"],
                "transport": "stdio"}})
    return await client.get_tools()      # LangChain tools ready to add to TOOLS
```

**Expected result.** `load_mcp_tools({"mcp.read"})` returns a list containing an `fx_rate` tool; `load_mcp_tools(set())` raises `PermissionError("403: needs mcp.read")`.

**Verify.**

```bash
python - <<'PY'
import asyncio; from interop.mcp_client import load_mcp_tools
tools = asyncio.run(load_mcp_tools({"mcp.read"}))
print([t.name for t in tools])                       # -> ['fx_rate']
try: asyncio.run(load_mcp_tools(set()))
except PermissionError as e: print("blocked:", e)    # -> blocked: 403: needs mcp.read
PY
```

**Troubleshoot.** *Server won't spawn:* run `python interop/ext_mcp_server.py` directly — it should block waiting on stdio (Ctrl-C to exit). *Adapter import error:* package is `langchain-mcp-adapters`; class is `MultiServerMCPClient`. *Windows stdio hangs:* ensure the `command` is `python` (on your PATH in the venv) and the `args` path is relative to your repo root / cwd.

---

### Step 6 — Expose one skill over A2A + end-user OAuth scopes

**What.** A `mint`/`verify` JWT helper (HS256, dev-only), and a FastAPI **A2A server** that serves an **Agent Card** at `/.well-known/agent-card.json` and a scope-checked `POST /skills/answer_kb_question` (200 with `kb.read`, 403 without).

**Why.** The token crossing every boundary must carry the *end user's* scopes, not a fat service account — "the agent did it" is not an audit answer (lecture [08](../lectures/08-end-user-oauth-scopes-across-mcp-and-a2a.md)). Discovery (the card) is public; **invocation** is authorized. Scopes complement, never replace, the Week-1 tenant filter.

**Do it.** Create `agent/auth.py`:

```python
import os
from authlib.jose import jwt
KEY = os.environ.get("JWT_DEV_SECRET", "dev-secret").encode()   # HS256 — DEV ONLY

def mint(user_id: str, scopes: list[str]) -> str:
    return jwt.encode({"alg": "HS256"},
                      {"sub": user_id, "scope": " ".join(scopes)}, KEY).decode()

def verify(token: str, need: str) -> str:
    claims = jwt.decode(token, KEY); claims.validate()          # enforces exp/nbf if present
    if need not in claims["scope"].split():
        raise PermissionError(f"403: needs {need}")
    return claims["sub"]

def scopes_from_state(token: str) -> list[str]:
    claims = jwt.decode(token, KEY); claims.validate()
    return claims["scope"].split()
```

Create `interop/a2a_server.py`:

```python
from fastapi import FastAPI, Header, HTTPException
from dotenv import load_dotenv; load_dotenv()
from agent.auth import verify
app = FastAPI()

@app.get("/.well-known/agent-card.json")     # public discovery — no secrets here
def card():
    return {"name": "capstone-kb", "version": "1", "url": "http://localhost:8100",
            "skills": [{"id": "answer_kb_question", "name": "Answer KB question",
                        "security": [{"oauth2": ["kb.read"]}]}]}

@app.post("/skills/answer_kb_question")       # authorized invocation
def answer(q: dict, authorization: str = Header(None)):
    try:
        user = verify((authorization or "").removeprefix("Bearer "), "kb.read")
    except PermissionError as e:
        raise HTTPException(403, str(e))
    # run the graph for this user (kb.read only -> read tools; tenant filter still applies)
    return {"answer": f"[kb] {user} asked: {q.get('question')}"}
```

Run and prove the three cases:

```bash
uvicorn interop.a2a_server:app --port 8100 &     # Git-Bash: '&' backgrounds it
# 1) Agent Card is public:
curl -s http://localhost:8100/.well-known/agent-card.json
# 2) mint a kb.read token and call the skill -> 200:
TOKEN=$(python -c "from dotenv import load_dotenv; load_dotenv(); from agent.auth import mint; print(mint('u_alice',['kb.read']))")
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8100/skills/answer_kb_question \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"question":"hi"}'   # -> 200
# 3) token WITHOUT kb.read -> 403:
BAD=$(python -c "from dotenv import load_dotenv; load_dotenv(); from agent.auth import mint; print(mint('u_alice',['other']))")
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8100/skills/answer_kb_question \
  -H "Authorization: Bearer $BAD" -d '{"question":"hi"}'    # -> 403
```

**Expected result.** Card returns JSON listing one skill and its required `kb.read` scope; skill returns `200` with a `kb.read` token, `403` without. `submit_action` rejects before HITL for a caller lacking `action.write` (wired in Step 3's `action_node`).

**Verify.** The three `curl` calls above print `200`-body, `200`, `403`.

**Troubleshoot.** *Everything 403s:* the `JWT_DEV_SECRET` used to mint must equal the one used to verify — load `.env` in both processes. *`authlib` decode error:* pass the raw token (strip `Bearer `); `claims.validate()` only checks `exp`/`nbf` if you set them. *Port busy:* pick another port and update the card's `url`.

> **A2A SDK note.** For real interoperability you'd publish via the `a2a-sdk` (typed Agent Card, task lifecycle). The FastAPI version above is the transparent, dependency-light stand-in that satisfies the DoD on a laptop; swap it for `a2a-sdk` as a stretch. **Production auth:** HS256 with a shared secret is symmetric — anyone who can verify can forge. Ship RS256/ES256 with a JWKS endpoint, short `exp`, and `aud` binding. Flag the dev secret loudly in `DECISIONS.md`.

---

### Step 7 — Optional: persistent long-term memory (PostgresStore)

**What.** Cross-thread facts keyed by `(namespace, key)` — e.g. "this tenant says *claimant*, not *patient*."

**Why.** Checkpoints are *this run's* execution state; the **Store** is durable cross-thread memory (lecture [06](../lectures/06-durability-checkpointing-and-idempotent-writes.md)). Sort every fact by lifetime: per-run scratch → checkpoint; reusable facts → Store.

**Do it.** Create `agent/memory.py`:

```python
import os
from langgraph.store.postgres import PostgresStore
DSN = os.environ["DATABASE_URL"]

def remember(tenant_id: str, key: str, value: dict):
    with PostgresStore.from_conn_string(DSN) as store:
        store.setup()
        store.put((tenant_id, "vocab"), key, value)

def recall(tenant_id: str, key: str):
    with PostgresStore.from_conn_string(DSN) as store:
        store.setup()
        item = store.get((tenant_id, "vocab"), key)
        return item.value if item else None
```

**Verify.** `remember("acme","term",{"prefers":"claimant"})` then `recall("acme","term")` returns the dict across a fresh process.

**Troubleshoot.** *Import path:* `from langgraph.store.postgres import PostgresStore` (package `langgraph-checkpoint-postgres` ships it). *Leaking per-run state here:* don't — put transient turn data in the checkpoint, not the Store.

---

### Step 8 — The four proofs (tests)

**What.** `pytest` tests: HITL-block, crash-resume with no dup write, scope-403, budget-kill (plus a DB-scope test).

**Why.** These *are* the deliverable — each maps to a DoD line. The crash-resume test must simulate a **new process** (or at least a new saver connection), not a re-invoked object, or `MemorySaver` would fake a green.

**Do it (key signatures + the load-bearing assertions).**

```python
# tests/test_hitl.py
def test_unapproved_write_leaves_db_untouched(app, cfg, count_writes, seed_state):
    before = count_writes()
    out = app.invoke(seed_state("void_claim", {"claim_id": "c1"}), cfg)
    assert "__interrupt__" in out                    # parked at interrupt
    assert count_writes() == before                  # nothing written yet
    app.invoke(Command(resume={"approved": False}), cfg)
    assert count_writes() == before                  # reject -> still nothing

def test_approved_write_inserts_exactly_one(app, cfg, count_writes, seed_state):
    before = count_writes()
    app.invoke(seed_state("void_claim", {"claim_id": "c2"}), cfg)
    app.invoke(Command(resume={"approved": True}), cfg)
    assert count_writes() == before + 1
```

```python
# tests/test_resume.py  — durability proven across a NEW saver/process
def test_crash_resume_no_duplicate(dsn, seed_state):
    key = upstream_key("u_alice", "void_claim", {"claim_id": "c3"})
    cfg = {"configurable": {"thread_id": "run-crash"}}
    # process A: run to the interrupt, then "crash" (drop app + saver)
    cmA, cpA = open_saver()
    appA = build_graph(cpA)
    appA.invoke({**seed_state("void_claim", {"claim_id": "c3"}), "idempotency_key": key}, cfg)
    cmA.__exit__(None, None, None)                   # <-- simulate crash: saver gone
    # process B: brand-new saver + graph, SAME thread_id, resume to completion
    cmB, cpB = open_saver()
    appB = build_graph(cpB)
    appB.invoke(Command(resume={"approved": True}), cfg)
    cmB.__exit__(None, None, None)
    assert count_key(dsn, key) == 1                  # exactly one, not two
```

> **Real OS-kill (bonus).** For the strongest proof, run process A in a subprocess and `kill -9` it (Git-Bash: `kill -9 <pid>`; Windows native: `taskkill /F /PID <pid>`) *between* the approved insert and END, then resume in a fresh `python` process with the same `thread_id`. `count_key == 1` must still hold — that's the idempotency key + `ON CONFLICT` closing the commit gap.

```python
# tests/test_scopes.py
def test_a2a_requires_kb_read(client):     # FastAPI TestClient(app)
    assert client.post("/skills/answer_kb_question",
        headers={"Authorization": f"Bearer {mint('u','[]'.strip('[]') and [] or ['other'])}"},
        json={"question":"hi"}).status_code == 403
    assert client.post("/skills/answer_kb_question",
        headers={"Authorization": f"Bearer {mint('u',['kb.read'])}"},
        json={"question":"hi"}).status_code == 200

def test_write_scope_gate_before_hitl(app, cfg, count_writes):
    # user lacks action.write -> rejected before interrupt, no write, no pause
    state = {"messages": [HumanMessage("void claim c1")], "user_id": "u_alice",
             "user_scopes": ["kb.read"], "idempotency_key": "k1", "tokens": 0, "usd": 0.0}
    before = count_writes()
    out = app.invoke(state, cfg)
    assert "__interrupt__" not in out and count_writes() == before
```

```python
# tests/test_budget.py
def test_over_cap_halts_at_guard(monkeypatch):
    import agent.budget as b; monkeypatch.setattr(b, "CAP_USD", 0.0)
    cmB, cp = open_saver(); app = build_graph(cp)
    state = {"messages": [HumanMessage("hi")], "user_id": "u_alice",
             "user_scopes": ["kb.read"], "idempotency_key": "k", "tokens": 0, "usd": 0.01}
    out = app.invoke(state, {"configurable": {"thread_id": "run-budget"}})
    assert any("BUDGET_KILL" in str(getattr(m, "content", m.get("content", "")))
               for m in out["messages"])
    cmB.__exit__(None, None, None)
```

```python
# tests/test_db_scope.py  — the wall is at the DB, not the prompt
import psycopg, pytest
from agent.db_scope import readonly_conn
def test_delete_raises_at_db():
    with readonly_conn("u_alice") as c:
        with pytest.raises(psycopg.Error):          # permission denied / read-only txn
            c.execute("DELETE FROM claims WHERE id='c1'")
def test_rls_hides_other_users_rows():
    with readonly_conn("u_alice") as c:
        rows = c.execute("SELECT owner_id FROM claims").fetchall()
    assert all(r["owner_id"] == "u_alice" for r in rows)   # never u_bob
```

**Expected result.** `pytest -q` green with ≥4 passing proofs.

**Verify.**

```bash
pytest -q
```

**Troubleshoot.** *`test_resume` shows 2 rows:* either `MemorySaver` is wired in (must be `PostgresSaver`), or the idempotency key is minted inside `action_node` (must be `upstream_key`, in state). *Fixtures:* put `app`, `cfg`, `count_writes`, `seed_state`, `dsn`, `client` in `tests/conftest.py`; keep `thread_id` distinct per test to avoid cross-test state bleed.

---

## Putting it together — a short end-to-end run

1. `docker start capstone-pg` and apply `sql/01_roles_rls.sql`.
2. Start the A2A server: `uvicorn interop.a2a_server:app --port 8100 &`.
3. Mint a full-scope user token and hit the KB skill:
   ```bash
   TOKEN=$(python -c "from dotenv import load_dotenv;load_dotenv();from agent.auth import mint;print(mint('u_alice',['kb.read','action.write','mcp.read']))")
   curl -s -X POST http://localhost:8100/skills/answer_kb_question \
     -H "Authorization: Bearer $TOKEN" -d '{"question":"what claims are open?"}'
   ```
4. Drive the graph end-to-end: `python -m scripts.run_once` — watch it (a) route a read to `sql_query` under Alice's RLS scope, (b) route a write to `action_node`, **pause at `interrupt`** with `writes` empty, (c) commit exactly one row on `{"approved": True}`.
5. Kill the run mid-flight and resume with the same `thread_id`; confirm `SELECT count(*) FROM writes WHERE idempotency_key=?` is **1**, not 2.
6. Repeat with a `kb.read`-only token — the A2A skill still answers reads (200), but `submit_action` is rejected before HITL, and `load_mcp_tools` raises 403 without `mcp.read`.

You now have a supervisor that reads under a DB-enforced per-user scope, writes only behind a human gate, survives a crash without double-writing, stops itself at a budget cap, and reaches outward only with the end user's authority.

---

## Definition of Done — verifiable checks

Restating the spine's Week 2 DoD as things you can *demonstrate*, not assert:

- [ ] **`pytest -q` green** with ≥4 tests: HITL-block, crash-resume, scope-403, budget-kill.
- [ ] **HITL blocks:** on a destructive payload the run pauses at `interrupt` (an `__interrupt__` is returned, `writes` row-count **unchanged**); resuming `{"approved": false}` leaves the DB untouched; `{"approved": true}` inserts **exactly one** row.
- [ ] **Crash-resume, no dup write:** run past the approved insert, simulate a crash (drop the saver / **new process**) with the **same `thread_id`**, resume from the `PostgresSaver` checkpoint to completion, and assert `SELECT count(*) FROM writes WHERE idempotency_key=?` **== 1** (not 2). Bonus: `kill -9` the OS process for real between checkpoint and END.
- [ ] **DB scope enforced at the DB:** a generated `DELETE`/`UPDATE`, or a cross-user `SELECT`, raises a Postgres permission/RLS error — proven by a test that runs one as `app_readonly` (not a prompt refusal).
- [ ] **Budget kill:** with `CAP_USD` below one run's cost, the graph halts at `guard`, routes to `END`, logs a `BUDGET_KILL` line with the overage, and leaves a **valid resumable** checkpoint.
- [ ] **A2A + MCP under scope:** the A2A skill returns `200` only with a `kb.read` token (`403` otherwise); the MCP tool loads only when scopes include `mcp.read`. The Agent Card is served at `/.well-known/agent-card.json` and lists the one skill + its required scope.
- [ ] **Trace evidence:** a run shows **per-step token counts and cumulative $** (your `Meter` printout or LangSmith).
- [ ] **Persistent memory (optional-but-cheap):** a fact written to `PostgresStore` under `(tenant_id, "vocab")` is recalled from a fresh process.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `test_resume` finds **2** rows | `MemorySaver` wired in, **or** key minted inside `action_node` | Use `PostgresSaver`; generate `idempotency_key` **upstream**, keep it in state |
| Resume "works" but proves nothing | You re-invoked the **same** graph object | Resume in a **new process / new saver connection**, same `thread_id` |
| DELETE succeeds against `claims` | Running as superuser / role not set | `SET ROLE app_readonly` + `default_transaction_read_only=on`; grant SELECT only |
| RLS returns all rows | Missing `FORCE ROW LEVEL SECURITY` or `app.user_id` unset | Add `FORCE`; set `app.user_id` in the same session |
| No `__interrupt__` returned | Model didn't call `submit_action`, or wrong saver | Prompt directly; confirm `PostgresSaver`; check `route()` sends writes to `action` |
| A2A always 403 | Mint/verify secrets differ | Load the same `JWT_DEV_SECRET` in both processes |
| Budget never trips (Ollama) | Local model reports `$0` | Cap on **tokens** for the free path, or seed `usd` in the test |
| MCP tool never loads | Missing `mcp.read`, or server didn't spawn | Pass `{"mcp.read"}`; run `python interop/ext_mcp_server.py` standalone to debug |
| `cp.setup()` permission error | Using `app_readonly` DSN | Use the `postgres` superuser DSN for checkpointer/store setup |
| Windows: `&` / `kill` not working | Native shell vs Git-Bash differ | Git-Bash: `cmd &`, `kill -9 <pid>`; native: start in new window, `taskkill /F /PID` |

---

## Stretch goals (optional)

- **Real A2A over `a2a-sdk`.** Replace the FastAPI stand-in with the typed `a2a-sdk` server: proper Agent Card schema, task lifecycle, and streaming — then drive it from a second client agent that discovers you by URL only.
- **Streamable-HTTP MCP.** Swap the MCP transport from `stdio` to streamable-http and require the same end-user scope on the MCP call, not just at load — carry the token through the hop (on-behalf-of).
- **Production-grade tokens.** Move `auth.py` to RS256/ES256 with a JWKS endpoint, short `exp`, and `aud` binding; write the dev-vs-prod delta into `DECISIONS.md`.
- **Realtime voice front-end.** Put LiveKit Agents or Pipecat in front of the same graph for speech-in/speech-out — only after the core DoD is green.
- **`langgraph-supervisor` prebuilt.** Now that you've hand-wired the supervisor, refactor onto the prebuilt and confirm your DB-role, HITL-placement, and idempotency decisions are unchanged (the prebuilt does *not* make them for you).
- **Rate-limit `submit_action`.** "Max N writes per run" enforced as a counted, named event in state — cheap because the call is a discrete typed contract.
