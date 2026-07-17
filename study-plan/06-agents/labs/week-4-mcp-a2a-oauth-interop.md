# Week 4 Lab: MCP + A2A + OAuth — A Multi-Agent Interop Repo

> **What you build:** a single laptop-runnable repo that makes your Week-3 agent stop being an island. By the end you will (1) run a **FastMCP** server exposing a tool + resource + prompt (plus one *sensitive* tool), (2) have an agent **consume** that MCP server as a real over-the-wire tool, (3) **expose your Week-3 agent over A2A** behind a valid Agent Card, (4) drive a **second client agent** that discovers it *by URL only* and streams the task lifecycle, and (5) **gate the sensitive tool behind an end-user OAuth-scoped JWT** and prove the 401 / 403 / 200 cases.
>
> **Why:** the mental model to burn in is **MCP connects an agent to tools/resources (vertical); A2A connects agents to each other as peers (horizontal)**. And the whole agent-security story reduces to one sentence: *the agent should only ever act with the caller's authority, minimally scoped, for a short time.* This lab makes both concrete on a laptop with no GPU and no paid key.
>
> **Read first (assumed done):** [15 — MCP deep dive](../lectures/15-mcp-deep-dive.md) · [16 — MCP security failure modes](../lectures/16-mcp-security-failure-modes.md) · [17 — A2A protocol](../lectures/17-a2a-protocol.md) · [18 — MCP vs A2A & composition](../lectures/18-mcp-vs-a2a-composition.md) · [19 — Agent identity & auth](../lectures/19-agent-identity-and-auth.md) · [20 — Protocol landscape & payments](../lectures/20-protocol-landscape-and-payments.md). Framework/durability background: [11 — frameworks by control model](../lectures/11-frameworks-by-control-model.md).

**Est. time:** ~9 hrs (Env 0.25 · MCP server 1.5 · consume MCP 1.5 · A2A server 2.5 · A2A client 1.5 · OAuth gate 1.5 · glue/tests 0.75) · **You will need:** Python 3.11+, [`uv`](https://docs.astral.sh/uv/), and an LLM. **Free/local path:** [Ollama](https://ollama.com) with `llama3.1:8b` (CPU is fine). **Alt:** any OpenAI-compatible base URL (set `OPENAI_BASE_URL` + `OPENAI_API_KEY` to a cheap gateway) — **no GPU, no paid key required**. `curl` and a browser (for the MCP Inspector) round it out.

---

## Before you start (setup)

**What you should have:** your **Week-3 LangGraph agent** (`graph.py` + tools + a model wrapper). This lab wraps that agent — you are not building a new brain, you are giving it a mouth (A2A) and new hands (MCP).

**Reality check on the SDKs.** `a2a-sdk` and `mcp`/`fastmcp` move fast; symbol names have churned across releases (e.g. `capabilities.streaming` vs `capabilities.streaming=True`, `send_message_streaming` vs a client factory). This guide targets the widely-documented stable shape. **Pin your versions** so a background `uv` upgrade doesn't break you mid-lab, and if an import fails, introspect the installed package (`python -c "import a2a, inspect; print(dir(a2a.types))"`) rather than guessing. Troubleshooting at the bottom shows how.

### Step 0 — Environment

**What:** create the repo and install the stack. **Why:** one venv, pinned deps, one model backend — so every later step is reproducible.

```bash
uv init week4-interop && cd week4-interop
uv add "mcp[cli]" fastmcp a2a-sdk httpx uvicorn pydantic "pyjwt[crypto]" rich
uv add langgraph langchain-core            # reuse your Week-3 agent stack
uv add --dev pytest pytest-asyncio         # tests are async (A2A client is async)

# free/local model:
ollama pull llama3.1:8b
# ...OR point at an OpenAI-compatible gateway instead (no Ollama needed):
#   export OPENAI_BASE_URL=https://your-gateway/v1
#   export OPENAI_API_KEY=whatever-the-gateway-wants
```

**Target layout** (matches the spine — build the files as you go):

```
week4-interop/
  pyproject.toml
  notes/
    demo.md                 # backing store for the notes:// resource
  mcp_server/
    server.py               # FastMCP: word_count + notes://{name} + summarize + delete_note
  client/
    use_mcp.py              # agent consumes the MCP server over the wire
    a2a_client.py           # discovers card BY URL, message/send + message/stream
  a2a_agent/
    agent_executor.py       # wraps the Week-3 agent in an AgentExecutor
    server.py               # A2A server + Agent Card at /.well-known/agent-card.json
  auth/
    tokens.py               # mint/verify scoped JWTs (end-user keyed)
    authz.py                # scope + audience gate for delete_note
  tests/
    test_interop.py         # MCP call, A2A discover+stream, 3 auth cases
  run_all.sh                # green = DoD met
  README.md
```

**Expected result:** `uv run python -c "import mcp, fastmcp, a2a, jwt, langgraph; print('ok')"` prints `ok`.

**Verify:** `uv pip list | grep -E "mcp|a2a|pyjwt|fastmcp|langgraph"` shows all packages with versions.

**Troubleshoot:** if `uv add a2a-sdk` fails, your Python is < 3.10 — `uv python install 3.11 && uv venv --python 3.11`. If Ollama isn't reachable later, `ollama serve` in a separate terminal and confirm `curl http://localhost:11434/api/tags`.

---

## Step-by-step

### Step 1 — Build the MCP server (tool + resource + prompt + a sensitive tool)

**What:** a FastMCP server exposing the three MCP primitives — a **tool** (`word_count`, model-invoked), a **resource** (`notes://{name}`, read-only app-controlled context), a **prompt** (`summarize`, a user-selected template) — plus one **sensitive** tool (`delete_note`) that Step 5 will gate. It must run over **stdio** *and* **streamable-http** at `/mcp`.

**Why:** MCP is how you stop hand-writing bespoke tool adapters — one server, many hosts. stdio is the local, zero-network-surface transport (a subprocess speaking JSON-RPC over stdin/stdout); streamable-http is the current *remote* transport (a single `/mcp` endpoint that upgrades to a stream when needed). **SSE is legacy** — deprecated in the 2025 spec in favor of streamable-http; do not build new on it (see [15 — MCP deep dive](../lectures/15-mcp-deep-dive.md)).

**Do it:** first the backing note, then the server.

```bash
mkdir -p notes && printf '# Demo Note\n\nAgents speak MCP to tools and A2A to peers.\n' > notes/demo.md
```

```python
# mcp_server/server.py
from pathlib import Path
from mcp.server.fastmcp import FastMCP

NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"

# stateless_http=True keeps the streamable-http path simple for a laptop/tests.
mcp = FastMCP("week4-tools", stateless_http=True)


@mcp.tool()
def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in text."""
    return len(text.split())


@mcp.resource("notes://{name}")  # read-only, app-controlled context (URI-addressed)
def get_note(name: str) -> str:
    """Return the contents of the note called <name>."""
    path = (NOTES_DIR / f"{name}.md").resolve()
    if NOTES_DIR.resolve() not in path.parents:      # block path traversal
        raise ValueError("note escapes NOTES_DIR")
    return path.read_text(encoding="utf-8")


@mcp.prompt()  # user-selected template
def summarize(topic: str) -> str:
    """A prompt template that asks for a 5-bullet summary of a topic."""
    return f"Summarize what we know about {topic} in 5 bullet points."


@mcp.tool()
def delete_note(name: str) -> str:
    """SENSITIVE: delete a note. Gated behind an end-user OAuth scope in Step 5."""
    # In Step 5 the HTTP transport rejects this before we get here unless the
    # caller presents a JWT with scope 'notes:delete'. The tool body stays dumb.
    path = (NOTES_DIR / f"{name}.md").resolve()
    if NOTES_DIR.resolve() not in path.parents:
        raise ValueError("note escapes NOTES_DIR")
    if path.exists():
        path.unlink()
        return f"deleted:{name}"
    return f"missing:{name}"


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        # serves POST/GET at /mcp on 127.0.0.1:8000 by default
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio (default): reads/writes JSON-RPC over the pipe
```

**Expected result:**
- `uv run mcp dev mcp_server/server.py` boots the **MCP Inspector** (a local web UI). Under **Tools** you see `word_count` and `delete_note`; under **Resources**, `notes://{name}`; under **Prompts**, `summarize`. Calling `word_count` with `{"text":"one two three"}` returns `3`.
- `uv run python mcp_server/server.py http` starts Uvicorn on `http://127.0.0.1:8000` serving `/mcp`.

**Verify (both transports):**
1. Inspector shows **1 primary tool** (`word_count`) + **1 resource** + **1 prompt** (plus the sensitive `delete_note`). *DoD counts `word_count` as the tool; `delete_note` is the gated extra.*
2. streamable-http answers `tools/list`. MCP requires an `initialize` handshake and a session, so the cleanest check is a tiny client (you'll reuse this in Step 2):

```bash
uv run python - <<'PY'
import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
anyio.run(main)
PY
# EXPECT: TOOLS: ['word_count', 'delete_note']
```

**Troubleshoot:**
- `mcp dev` opens the browser but shows nothing → you passed the wrong path; it must point at the file, not a package. Also check the terminal for a tracebacks-on-import.
- streamable-http `tools/list` hangs → you hit `/` not `/mcp`, or forgot `await s.initialize()`. The handshake is mandatory.
- `notes://demo` errors "escapes NOTES_DIR" → run from the repo root so the relative `NOTES_DIR` resolves; or hardcode an absolute path while debugging.
- Windows note: `mcp dev` needs Node.js on PATH (the Inspector is a Node app). Install Node LTS if the Inspector won't launch; the HTTP verify above needs no Node.

---

### Step 2 — Consume the MCP server from your agent

**What:** `client/use_mcp.py` opens an MCP **client session**, calls `list_tools()`, adapts the MCP tools into your agent's tool interface, and asks a question that *forces* the `word_count` call. Then you prove the trace shows an **MCP round-trip**, not a local Python function.

**Why:** this is the "vertical" half of interop — the agent gains a capability it did not have to hand-code. Using the official MCP client session (not re-implemented JSON-RPC) is the opinionated default; it handles the handshake, framing, and session for you.

**Do it (adapter approach — shows the round-trip explicitly):**

```python
# client/use_mcp.py
import anyio, contextlib
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

SERVER = ["python", "mcp_server/server.py"]  # stdio subprocess


@contextlib.asynccontextmanager
async def mcp_session(mode="stdio"):
    if mode == "http":
        async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                yield s
    else:
        params = StdioServerParameters(command=SERVER[0], args=SERVER[1:])
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                yield s


async def ask_word_count(paragraph: str, mode="stdio") -> int:
    async with mcp_session(mode) as s:
        tools = await s.list_tools()
        names = [t.name for t in tools.tools]
        print(f"[MCP] discovered tools over {mode}: {names}")
        assert "word_count" in names, "server did not expose word_count"

        # THE MCP ROUND-TRIP: this call is JSON-RPC over the transport,
        # NOT len(paragraph.split()) in this process.
        print(f"[MCP -> server] tools/call word_count")
        result = await s.call_tool("word_count", {"text": paragraph})
        value = int(result.content[0].text)
        print(f"[server -> MCP] result = {value}")
        return value


if __name__ == "__main__":
    para = "how many words are in this exact paragraph right here"
    got = anyio.run(ask_word_count, para)   # stdio; pass "http" for streamable-http
    print("ANSWER:", got, "| local check:", len(para.split()))
```

**Wire it into the agent (LangGraph):** the honest way to "force the tool" is to give the model *only* the MCP-backed tool and a task that requires it. The shortcut is [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) (`uv add langchain-mcp-adapters`), which converts an MCP session's tools into LangChain tools your Week-3 `create_react_agent` can bind:

```python
# inside use_mcp.py, agent variant
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from model import get_llm            # your Week-3 model wrapper (Ollama/OpenAI-compatible)

async def agent_uses_mcp(question: str):
    async with mcp_session("stdio") as s:
        tools = await load_mcp_tools(s)                 # MCP tools -> LC tools
        agent = create_react_agent(get_llm(), tools)
        result = await agent.ainvoke({"messages": [("user", question)]})
        return result["messages"][-1].content
# question = "Use your tools to count the words in: 'the quick brown fox jumps'"
```

**Expected result:** stdout shows `[MCP] discovered tools`, `[MCP -> server] tools/call word_count`, `[server -> MCP] result = 10`, and `ANSWER: 10 | local check: 10`. In the agent variant, the LangGraph message trace contains a `ToolMessage` whose tool call name is `word_count` — the model chose it and it executed on the server.

**Verify:** the printed round-trip lines appear **before** the answer, and the count matches the local check. For the DoD "trace shows the MCP round-trip not a local function": add a print inside `mcp_server/server.py`'s `word_count` (`print("SERVER word_count called", flush=True)` to `stderr`) — you'll see it fire in the *server* process/log, proving execution crossed the transport.

**Troubleshoot:**
- `result.content[0].text` errors → newer MCP returns structured content; use `result.structuredContent` or `result.content` and inspect its type. Print `result` once to see the shape.
- stdio subprocess exits immediately → the server import raised; run `python mcp_server/server.py` alone and read the traceback.
- Model never calls the tool → it answered from its own counting. Force it: give it *only* `word_count`, phrase the task as "use your word_count tool", and (Ollama) confirm the model does native tool calling (`llama3.1` does).

---

### Step 3 — Expose your Week-3 agent over A2A

**What:** wrap the Week-3 agent in an **AgentExecutor**, serve it with `a2a-sdk`, and publish a valid **Agent Card** at `/.well-known/agent-card.json` advertising a `research` **AgentSkill** and `capabilities.streaming=True`. In the executor, drive the task lifecycle **working → artifact → completed** with streaming status updates.

**Why:** this is the "horizontal" half — other agents (any vendor, any framework) discover and call yours as an **opaque peer**, by its skills, not by importing its code. The Agent Card *is* the contract (see [17 — A2A protocol](../lectures/17-a2a-protocol.md)).

**Do it — the executor** (`a2a_agent/agent_executor.py`). The `AgentExecutor` interface is `execute(context, event_queue)` + `cancel(...)`. Use a `TaskUpdater` to emit lifecycle events:

```python
# a2a_agent/agent_executor.py
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TaskArtifactUpdateEvent, Part, TextPart
from a2a.utils import new_task, new_agent_text_message

# your Week-3 agent, compiled LangGraph app
from week3_agent import build_agent   # returns a compiled graph with .ainvoke / .astream


class ResearchAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agent = build_agent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # 1) submitted -> working
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Researching…", task.context_id, task.id),
        )

        # 2) do the work; stream >=1 incremental update so the client sees progress
        chunks = []
        async for step in self.agent.astream({"messages": [("user", query)]}):
            note = _summarize_step(step)          # e.g. "called tool word_count"
            if note:
                chunks.append(note)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(note, task.context_id, task.id),
                )

        final = _final_text(step)                 # last message content

        # 3) emit an artifact (the durable result), then complete
        await updater.add_artifact(
            [Part(root=TextPart(text=final))], name="research-result",
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")   # fine for the lab
```

`_summarize_step` / `_final_text` are small helpers over your graph's stream shape. If you don't want to stream real steps yet, emit two synthetic `working` updates ("planning…", "answering…") so the client sees ≥2 events — the DoD needs ≥2 stream events, not real reasoning.

**Do it — the server + Agent Card** (`a2a_agent/server.py`):

```python
# a2a_agent/server.py
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from agent_executor import ResearchAgentExecutor

skill = AgentSkill(
    id="research",
    name="Research assistant",
    description="Answers questions using tools and memory.",
    tags=["research", "qa"],
    examples=["What changed in the Q3 report?"],
)

card = AgentCard(
    name="Week3 Research Agent",
    description="Tool-using agent from Week 3, exposed over A2A.",
    url="http://localhost:9999/",                 # where clients POST tasks
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True, push_notifications=False),
    default_input_modes=["text"],
    default_output_modes=["text"],
    skills=[skill],
)

handler = DefaultRequestHandler(
    agent_executor=ResearchAgentExecutor(),
    task_store=InMemoryTaskStore(),
)
app = A2AStarletteApplication(agent_card=card, http_handler=handler)

if __name__ == "__main__":
    uvicorn.run(app.build(), host="localhost", port=9999)
```

**Expected result:** `uv run python a2a_agent/server.py` serves on `:9999`. `curl http://localhost:9999/.well-known/agent-card.json` returns JSON with `"name": "Week3 Research Agent"`, a `skills` array containing the `research` skill, and `"capabilities": {"streaming": true, ...}`.

**Verify:**
```bash
curl -s http://localhost:9999/.well-known/agent-card.json | python -m json.tool | grep -E 'research|streaming'
# EXPECT: the research skill id/name AND "streaming": true
```

**Troubleshoot:**
- Import errors (`A2AStarletteApplication` / `TaskUpdater` not found) → **version drift**. Introspect: `python -c "from a2a.server import apps; import inspect; print([n for n in dir(apps)])"`. Pin what you have (`uv add "a2a-sdk==<installed>"`) and adjust names. Some releases keyword the card as `streaming=True` in `AgentCapabilities` (snake) vs the JSON `streaming` (camel is the wire form) — the SDK handles the JSON casing; you pass snake_case in Python.
- Card served at `/agent-card.json` but not under `/.well-known/` → older SDKs used `/.well-known/agent.json`. The 2025 spec is `agent-card.json`; upgrade or set the path explicitly. Check what your build registered: `curl -s http://localhost:9999/.well-known/agent.json` as a fallback probe.
- 500 on `message/send` → your `build_agent()` import failed or `astream` shape differs. Test the agent standalone first.

---

### Step 4 — Second client agent discovers and calls it (by URL only)

**What:** `client/a2a_client.py` uses the SDK **resolver** to fetch the card *by URL*, constructs an `A2AClient`, runs `message/send` (one-shot), then `message/stream` printing **≥2 incremental events** with `rich`. **It must not import any server code** — only the URL.

**Why:** this is the discovery → delegate → stream loop that makes agents composable peers. "By URL only" is the load-bearing DoD constraint: if the client imports the server, you have a monolith, not interop.

**Do it:**

```python
# client/a2a_client.py
import anyio
from uuid import uuid4
import httpx
from rich.console import Console
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Message, Part, TextPart, Role,
    MessageSendParams, SendMessageRequest, SendStreamingMessageRequest,
)

console = Console()
BASE_URL = "http://localhost:9999"   # the ONLY thing we know about the server


def _msg(text: str) -> Message:
    return Message(role=Role.user, message_id=uuid4().hex,
                   parts=[Part(root=TextPart(text=text))])


async def main():
    async with httpx.AsyncClient(timeout=30) as http:
        # 1) DISCOVER by URL — read the card, no server imports
        resolver = A2ACardResolver(httpx_client=http, base_url=BASE_URL)
        card = await resolver.get_agent_card()
        console.print(f"[bold green]Discovered:[/] {card.name} | "
                      f"skills={[s.id for s in card.skills]} | "
                      f"streaming={card.capabilities.streaming}")

        client = A2AClient(httpx_client=http, agent_card=card)

        # 2) one-shot message/send
        req = SendMessageRequest(id=uuid4().hex,
                                 params=MessageSendParams(message=_msg(
                                     "How many words in 'alpha beta gamma delta'?")))
        resp = await client.send_message(req)
        console.print("[bold]send result:[/]", resp.model_dump(exclude_none=True))

        # 3) message/stream — print >=2 incremental events
        console.rule("streaming")
        sreq = SendStreamingMessageRequest(id=uuid4().hex,
            params=MessageSendParams(message=_msg("Research the demo note briefly.")))
        n = 0
        async for event in client.send_message_streaming(sreq):
            n += 1
            console.print(f"[cyan]event {n}[/]", event.model_dump(exclude_none=True))
        console.print(f"[bold]total stream events:[/] {n}")
        assert n >= 2, "expected >=2 incremental stream events"


if __name__ == "__main__":
    anyio.run(main)
```

**Expected result:** with the Step-3 server running, you see `Discovered: Week3 Research Agent | skills=['research'] | streaming=True`, a one-shot `send result`, then a `streaming` rule followed by **≥2** `event N` lines (working updates → artifact → completed), ending `total stream events: N` with N ≥ 2.

**Verify:** `grep -n "import a2a_agent\|from a2a_agent\|from week3_agent" client/a2a_client.py` returns nothing — the client only knows `BASE_URL`. The assertion `n >= 2` passes.

**Troubleshoot:**
- `send_message_streaming` doesn't exist → version drift; some releases expose streaming via the same `send_message` with a `ClientConfig(streaming=True)` factory (`create_client`). Introspect `dir(A2AClient)` and adapt; the *concept* (consume incremental events) is what the DoD checks.
- Only 1 event → your executor emitted a single terminal update. Add an explicit `working` status before the artifact (see Step 3 note) so there are ≥2.
- `get_agent_card()` 404 → the well-known path differs (see Step 3 troubleshooting); point the resolver at the path your server actually serves via `agent_card_path=`.
- Hangs forever → server is blocking (a sync `.invoke` inside `execute`); make the agent call `await ...astream/ainvoke` so the event loop can flush stream events.

---

### Step 5 — Gate the sensitive tool behind an end-user OAuth-scoped token

**What:** mint a short-lived **JWT** with `sub=<end_user_id>` and `scope="notes:delete"`, verify the **audience** matches this server, and check the **scope BEFORE** executing `delete_note`. Demonstrate three cases: **401** (no token), **403 insufficient_scope** (`notes:read` only), **200** (correctly scoped).

**Why:** this is the confused-deputy defense made concrete. A shared admin key means any poisoned/compromised tool acts with god-authority; a **short-lived, end-user-scoped** token means a compromised tool can only ever act within *this user's* permissions, for a short time (see [19 — Agent identity & auth](../lectures/19-agent-identity-and-auth.md), [16 — MCP security](../lectures/16-mcp-security-failure-modes.md)). Full RFC 8693 token exchange is stretch; the scoped-per-user JWT captures the principle on a laptop.

**Do it — mint/verify** (`auth/tokens.py`). Use a symmetric key for the lab (HS256); note RS256/JWKS is the real-world path.

```python
# auth/tokens.py
import time, jwt   # PyJWT

SECRET = "lab-only-not-a-real-secret"
ALG = "HS256"
ISSUER = "week4-auth"
AUDIENCE = "week4-mcp-server"          # this server's identity; tokens are audience-bound


def mint(sub: str, scope: str, ttl_seconds: int = 300) -> str:
    now = int(time.time())
    return jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": sub, "scope": scope,
         "iat": now, "exp": now + ttl_seconds},          # short-lived
        SECRET, algorithm=ALG,
    )


class AuthError(Exception):
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code
        super().__init__(f"{status} {code}")


def verify(token: str | None, required_scope: str) -> dict:
    if not token:
        raise AuthError(401, "missing_token")            # no credential
    try:
        claims = jwt.decode(token, SECRET, algorithms=[ALG],
                            audience=AUDIENCE, issuer=ISSUER)   # checks aud+exp+sig
    except jwt.PyJWTError as e:
        raise AuthError(401, f"invalid_token:{e.__class__.__name__}")
    scopes = claims.get("scope", "").split()
    if required_scope not in scopes:
        raise AuthError(403, "insufficient_scope")       # authenticated but not authorized
    return claims
```

**Do it — the gate** (`auth/authz.py`), a thin wrapper the sensitive tool calls before doing anything:

```python
# auth/authz.py
from auth.tokens import verify, AuthError

def require_scope(token: str | None, scope: str) -> dict:
    """Raise AuthError(401|403) or return the verified claims. Log every decision."""
    try:
        claims = verify(token, scope)
        print(f"[authz] ALLOW sub={claims['sub']} scope={scope}")
        return claims
    except AuthError as e:
        print(f"[authz] DENY {e.status} {e.code} (needed {scope})")
        raise
```

**Wire it into the request path.** The clean design keeps the tool body dumb and enforces at the transport edge — the caller presents the JWT as a **Bearer** header on the MCP HTTP transport, and a middleware/dependency runs `require_scope(token, "notes:delete")` before dispatching `delete_note`. For a laptop demo you can enforce inside the tool by threading the token through, but the *principle* is: **check audience + scope before the side effect**. Minimal driver that proves the three cases without needing full middleware:

```python
# tests-style driver (also lives in tests/test_interop.py)
from auth.tokens import mint, AuthError
from auth.authz import require_scope

def gated_delete(token, name):
    require_scope(token, "notes:delete")   # 401/403 raised here, BEFORE delete
    from mcp_server.server import delete_note
    return delete_note(name)

# (a) no token -> 401
try: gated_delete(None, "demo")
except AuthError as e: assert e.status == 401

# (b) notes:read only -> 403 insufficient_scope
ro = mint("alice", "notes:read")
try: gated_delete(ro, "demo")
except AuthError as e: assert e.status == 403 and e.code == "insufficient_scope"

# (c) correctly scoped -> success
rw = mint("alice", "notes:delete")
assert gated_delete(rw, "demo").startswith(("deleted:", "missing:"))
```

**Expected result / Verify:** stdout logs three decisions —
```
[authz] DENY 401 missing_token (needed notes:delete)
[authz] DENY 403 insufficient_scope (needed notes:delete)
[authz] ALLOW sub=alice scope=notes:delete
```
and the three assertions pass.

**Troubleshoot:**
- `jwt.decode` raises `InvalidAudienceError` on the valid token → your `mint` `aud` and `verify` `audience` disagree; they must both be `AUDIENCE`. This is the audience-binding check working — it prevents a token minted for another server being replayed here.
- All tokens pass regardless of scope → you split `scope` wrong or checked `in` against the whole string; `"notes:delete" in "notes:read"` is `False` for the split-list check but `True` for a substring check — use the `.split()` list, never substring.
- Want the real Bearer-over-HTTP path → put `require_scope` in a Starlette middleware on the streamable-http app, reading `Authorization: Bearer <jwt>`; return `JSONResponse(status_code=e.status, ...)` on `AuthError`. That makes `delete_note` return a real 401/403 over the wire.

---

## Putting it together — short end-to-end run

Four terminals (or background them), from the repo root:

```bash
# T1: MCP server over streamable-http
uv run python mcp_server/server.py http                 # -> :8000/mcp

# T2: A2A server (your Week-3 agent behind a card)
uv run python a2a_agent/server.py                        # -> :9999

# T3: consume MCP (vertical) — forces word_count over the wire
uv run python client/use_mcp.py                          # prints MCP round-trip + answer

# T3: discover + call the A2A agent BY URL (horizontal) — >=2 stream events
uv run python client/a2a_client.py

# T3: the three auth cases (401 / 403 / 200)
uv run python -m pytest tests/test_interop.py -k auth -q

# one shot: everything
bash run_all.sh
```

A minimal `run_all.sh` starts the two servers, waits for health, runs `use_mcp.py`, `a2a_client.py`, and `pytest`, and exits non-zero if any step fails — green means the DoD below is met.

**Composition to notice (write it in the README):** the A2A `research` agent, when asked a question, *internally* uses the MCP `word_count` tool. That is the canonical composition — an **A2A peer whose internal capabilities are backed by MCP tools**. The client agent delegates a goal (A2A, horizontal); the research agent calls a tool (MCP, vertical).

---

## Definition of Done — verifiable checks

Restated from the spine. Each line is a command you can run.

- [ ] **MCP primitives + two transports.** `uv run mcp dev mcp_server/server.py` lists **1 tool** (`word_count`) **+ 1 resource** (`notes://{name}`) **+ 1 prompt** (`summarize`) in the Inspector; the same server run as `http` serves **`tools/list` over streamable-http at `/mcp`** (Step 1 verify snippet returns the tool names).
- [ ] **Agent consumes MCP.** `use_mcp.py` calls `word_count` and the **trace/log shows the MCP round-trip** (`[MCP -> server] tools/call` + server-side print), not a local function.
- [ ] **Valid Agent Card.** `curl .../.well-known/agent-card.json` returns valid JSON with the **`research` skill** and **`capabilities.streaming: true`**.
- [ ] **Client discovers by URL only + streams.** `a2a_client.py` fetches the card **by URL** (no server imports — grep proves it), runs `message/send`, and prints **≥2 incremental `message/stream` events**.
- [ ] **Auth gate: 401 / 403 / 200.** The gated `delete_note` path returns **401** (no token), **403 insufficient_scope** (`notes:read` only), and **success** (correctly scoped `notes:delete`) — **all three logged**.
- [ ] **README distinction + composition + auth rationale.** README states the **MCP-vs-A2A distinction in ≤3 sentences**, names **one composition example**, and adds **two sentences** on how an end-user-scoped short-lived token defeats the confused-deputy attack. Also note *why stdio (local) vs streamable-http (remote)* and *why SSE is legacy*.
- [ ] **Green suite.** `pytest` (or `run_all.sh`) is green, with assertions covering the **MCP call**, **A2A discover+stream**, and the **three auth cases**.

**README skeleton to drop in** (fills the doc DoD):

```markdown
## MCP vs A2A
MCP connects an agent to tools and resources — a vertical tool call you orchestrate.
A2A connects agents to each other as peers — you hand an opaque peer a goal and it
reasons on its own. Rule: deterministic function you own -> MCP; autonomous peer with
its own reasoning -> A2A.

**Composition:** our A2A `research` agent internally calls the MCP `word_count` tool;
a client agent discovers the research agent by its card and delegates — A2A on the
outside, MCP on the inside.

**Auth (confused-deputy):** `delete_note` requires a short-lived JWT with `sub=<user>`
and `scope=notes:delete`, audience-bound to this server. Because the token carries the
*end user's* least-privilege authority for ~5 minutes, a poisoned or compromised tool
can never exceed what that user could do — unlike a shared admin key, which would let
any tricked tool delete anything on everyone's behalf.

**Transports:** stdio = local subprocess, zero network surface (desktop/CLI hosts);
streamable-http = the current remote transport (single `/mcp`, upgrades to a stream).
SSE is the legacy HTTP transport, deprecated in the 2025 spec — do not build new on it.
```

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError` on `a2a.*` / `mcp.*` symbols | SDK version drift (fast-moving packages) | `python -c "import a2a.types, inspect; print(dir(a2a.types))"`; pin `uv add "a2a-sdk==<installed>"`; adapt names |
| `mcp dev` shows blank / won't open | Node.js missing (Inspector is a Node app) or wrong file path | Install Node LTS; point `mcp dev` at `mcp_server/server.py` directly |
| streamable-http `tools/list` hangs | Missing `await session.initialize()` or hitting `/` not `/mcp` | Add the handshake; use the exact `/mcp` URL |
| Model answers without calling `word_count` | Model counted itself / weak tool-calling | Give it *only* the MCP tool; phrase task to require it; use `llama3.1:8b` (native tool calls) |
| Card 404 at `/.well-known/agent-card.json` | Older SDK uses `/.well-known/agent.json` | Upgrade SDK, or probe the old path and set `agent_card_path=` on the resolver |
| Only 1 stream event | Executor emits one terminal event | Emit an explicit `working` status before the artifact/complete |
| A2A client hangs | Sync `.invoke` blocking the event loop in `execute` | Use `await agent.astream/ainvoke`; keep `execute` fully async |
| `InvalidAudienceError` on a valid token | `mint` `aud` ≠ `verify` `audience` | Make both `AUDIENCE`; this check is *supposed* to reject cross-server replay |
| Scope check passes when it shouldn't | Substring match instead of split-list | `required_scope in claims["scope"].split()`, never substring |
| Ollama refused / slow | `ollama serve` not running / CPU cold start | `ollama serve`; `curl localhost:11434/api/tags`; first call warms the model |
| Windows path/`\n` issues in Git-Bash | CRLF or backslash paths | Use forward slashes and `pathlib`; run scripts with `uv run python ...` |

---

## Stretch goals (optional)

1. **Real Bearer-over-HTTP gate.** Move `require_scope` into a Starlette middleware on the streamable-http MCP app so `delete_note` returns a genuine **401/403 over the wire** with a `WWW-Authenticate` header — the MCP authorization-spec shape (Protected Resource Metadata, audience-bound Bearer).
2. **AG-UI last mile.** Emit the agent's tool-call start/args/result + a final state as an SSE stream that a tiny static HTML page renders live — the UI layer MCP/A2A don't cover ([20 — protocol landscape](../lectures/20-protocol-landscape-and-payments.md)).
3. **Card-advertised auth.** Add `securitySchemes` to the Agent Card so the client *discovers* it must present a token, and have `a2a_client.py` satisfy it — auth requirements flowing through discovery.
4. **RFC 8693 token exchange.** Add a tiny "auth server" endpoint that trades a user token for a downstream `notes:delete` token audience-bound to the MCP server — the on-behalf-of pattern beyond a single minted JWT.
5. **Tool poisoning demo.** Add a second MCP tool whose `description` contains an injected instruction; show it reaching the model, then mitigate (show descriptions to the user / require approval for side effects) per [16 — MCP security](../lectures/16-mcp-security-failure-modes.md).
