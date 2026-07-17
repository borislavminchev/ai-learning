# Week 1 Lab: Stand Up the Four Stores, Stateful Chat, and Provable GDPR Delete

> You build the foundation for the whole phase: a `llm-gateway-lab/` repo with four storage tiers (Redis, Postgres, MinIO, a vector store), a **stateful chat** endpoint where deterministic code owns the flow and the LLM sits at a single leaf, idempotent multi-turn storage with **context compaction**, and a **GDPR cascade delete** that provably purges a user from *every* store — proven by a test that even runs a semantic search and gets zero hits.
>
> Read these first — this lab is where you *build* what they explain:
> - [1. The Five Reference Architectures and the Escalation Ladder](../lectures/01-five-reference-architectures.md)
> - [2. LLM Proposes, Code Disposes: The Deterministic Boundary](../lectures/02-deterministic-code-vs-llm-leaves.md)
> - [3. Four-Tier State: Redis, Postgres, Object Storage, Vector](../lectures/03-state-and-storage-tiers.md)
> - [4. Context-Window Management: Windowing, Compaction, and Retrieval Memory](../lectures/04-context-window-management.md)
> - [5. Right-to-Erasure as an Architecture Constraint: Provable Cascade Delete](../lectures/05-gdpr-erasure-as-architecture.md)

**Est. time:** ~9 hrs · **You will need:** Docker Desktop (or Docker Engine + Compose v2), Python 3.11+, `uv` ([astral.sh/uv](https://docs.astral.sh/uv/)), and a terminal. **Free/local path (default):** everything runs in Docker on a laptop — `redis:7`, `postgres:16` with the `pgvector` image, `minio/minio` for object storage, and **Ollama** for the model. No paid API and no cloud account are required. A provider key (OpenAI/Anthropic/Gemini) is optional and only swaps the model leaf.

---

## Before you start (setup)

**Check your toolchain.**

```bash
docker --version          # Docker 24+; "docker compose version" should be v2.x
uv --version              # install: https://docs.astral.sh/uv/getting-started/installation/
python --version          # 3.11+
```

**Install Ollama and pull a small model** (this is your free model leaf).

```bash
# macOS/Linux: see https://ollama.com/download ; Windows: install the OllamaSetup.exe
ollama pull llama3.1          # ~4.7 GB; or "ollama pull qwen2.5:3b" if RAM-constrained
ollama pull nomic-embed-text  # 768-dim embeddings for the vector store
ollama serve                  # runs on http://localhost:11434 (usually auto-starts)
```

> **Windows / Git-Bash notes.** Run Docker Desktop with the WSL2 backend. In Git-Bash, a leading-slash path like `-v /data:/data` can get mangled by MSYS path conversion — prefix the command with `MSYS_NO_PATHCONV=1` if a volume mount misbehaves. `curl` ships with Git-Bash; if a here-doc (`<<'EOF'`) misbehaves, put the JSON in a file and use `curl -d @body.json`. Ollama on Windows runs as a background service after install — check the tray icon rather than running `ollama serve` in a terminal.

**Create the repo skeleton.** This is the ONE repo you keep for the whole phase (Weeks 2–3 build on it).

```bash
mkdir llm-gateway-lab && cd llm-gateway-lab
uv init --python 3.11
uv add fastapi "uvicorn[standard]" redis "psycopg[binary,pool]" pydantic pydantic-settings boto3 httpx litellm
uv add --dev pytest pytest-asyncio
mkdir -p app tests
```

Target layout for this week:

```
llm-gateway-lab/
  docker-compose.yml        # redis, postgres(+pgvector), minio
  .env                      # local config (gitignored)
  pyproject.toml            # uv-managed
  app/
    __init__.py
    config.py               # settings (env-driven)
    deps.py                 # store clients: redis, pg pool, s3, ollama embed
    models.py               # Pydantic DTOs + SQL DDL
    llm.py                  # thin LLM leaf — ONE function, no business logic
    memory.py               # session buffer + token-threshold compaction
    chat.py                 # stateful chat endpoint (deterministic flow)
    gdpr.py                 # cascade delete + receipt
    main.py                 # FastAPI app + /health
  tests/
    conftest.py
    test_chat.py
    test_gdpr_delete.py
  README.md
```

Create `.env` and a `.gitignore` entry for it:

```bash
cat > .env <<'EOF'
POSTGRES_DSN=postgresql://lab:lab@localhost:5432/lab
REDIS_URL=redis://localhost:6379/0
S3_ENDPOINT=http://localhost:9000
S3_KEY=minioadmin
S3_SECRET=minioadmin
S3_BUCKET=lab-artifacts
OLLAMA_BASE=http://localhost:11434
CHAT_MODEL=ollama/llama3.1
EMBED_MODEL=nomic-embed-text
COMPACT_TOKEN_THRESHOLD=1500
EOF
echo ".env" >> .gitignore
```

---

## Step-by-step

### Step 1 — Bring up the four stores with Docker Compose

**What.** A `docker-compose.yml` that runs Redis, Postgres (with pgvector), and MinIO. Ollama runs on the host (simpler for GPU access), so it is your fourth "leaf" but not a compose service.

**Why.** Each store maps to a tier from Lecture 3: Redis = hot/volatile session + dedup + cache; Postgres = durable source of truth *and* (via pgvector) the semantic memory index; MinIO = S3-compatible object storage for blobs/archives. Running them locally means the whole lab is free and reproducible.

**Do it.** Use the `pgvector/pgvector` image so the `vector` extension is available without building anything.

```yaml
# docker-compose.yml
services:
  redis:
    image: redis:7
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      retries: 10

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: lab
      POSTGRES_PASSWORD: lab
      POSTGRES_DB: lab
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U lab -d lab"]
      interval: 5s
      retries: 10

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports: ["9000:9000", "9001:9001"]
    volumes: ["miniodata:/data"]
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      retries: 10

volumes:
  pgdata:
  miniodata:
```

```bash
docker compose up -d
docker compose ps          # all should be "healthy" after ~15s
```

**Expected result.** `docker compose ps` shows `redis`, `postgres`, `minio` all healthy. MinIO console is at `http://localhost:9001` (login `minioadmin`/`minioadmin`).

**Verify.**

```bash
docker exec -it $(docker compose ps -q redis) redis-cli ping     # -> PONG
docker exec -it $(docker compose ps -q postgres) psql -U lab -d lab -c "SELECT 1;"
```

**Troubleshoot.**
- *Port already in use* (5432/6379/9000): another Postgres/Redis is running. Stop it, or remap the left side of the port (e.g. `"5433:5432"`) and update `.env`.
- *Postgres unhealthy on first boot*: give it 20–30s; the volume initializes on first run.
- *Windows*: if `docker compose` can't reach the daemon, confirm Docker Desktop is running and WSL integration is enabled for your distro.

---

### Step 2 — Create the schema and enable pgvector

**What.** Create tables `users`, `conversations`, `messages`, `spend_ledger`, and a `memories(user_id, embedding vector(768), text)` table; enable the `vector` extension.

**Why.** Postgres is the durable source of truth; the schema is the contract the whole app and the GDPR delete rely on. Putting `user_id` on every table is what makes cascade delete tractable (Lecture 5's core lesson: design for erasure up front).

**Do it.** Put the DDL in `app/models.py` as a string you can run at startup, and also apply it once by hand now.

```sql
-- app/models.py  (SCHEMA_SQL)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
    content         TEXT NOT NULL,
    idempotency_key TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, idempotency_key)     -- dedup at the DB level too
);

CREATE TABLE IF NOT EXISTS spend_ledger (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    tenant_id   TEXT NOT NULL,
    tokens_in   INT NOT NULL DEFAULT 0,
    tokens_out  INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memories (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    tenant_id   TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
```

> Note the `UNIQUE (conversation_id, idempotency_key)` constraint — a second layer of idempotency defense behind the Redis check in Step 5. `nomic-embed-text` returns 768-dim vectors, matching `vector(768)`.

Apply it once:

```bash
docker exec -i $(docker compose ps -q postgres) psql -U lab -d lab < <(uv run python -c "from app.models import SCHEMA_SQL; print(SCHEMA_SQL)")
```

**Expected result.** `\dt` in psql lists all five tables; `SELECT * FROM pg_extension WHERE extname='vector';` returns a row.

**Verify.**

```bash
docker exec -it $(docker compose ps -q postgres) psql -U lab -d lab -c "\dt"
```

**Troubleshoot.**
- *`extension "vector" is not available`*: you're not on the `pgvector/pgvector:pg16` image. Fix the image in compose and `docker compose up -d --force-recreate postgres`.
- *`type "vector" does not exist`*: the extension wasn't created before the `memories` table. Run `CREATE EXTENSION vector;` first (the DDL orders it correctly).

---

### Step 3 — Store clients in `deps.py` and config in `config.py`

**What.** One place that constructs the Redis client, a Postgres connection pool, an S3 (MinIO) client, and a helper to embed text via Ollama.

**Why.** Centralizing clients keeps every module (chat, gdpr, memory) using the same connections and makes them injectable in tests. The embed helper lives here because both `memory.py` (write memories) and the GDPR test (semantic search) need it.

**Do it.**

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    postgres_dsn: str
    redis_url: str
    s3_endpoint: str
    s3_key: str
    s3_secret: str
    s3_bucket: str = "lab-artifacts"
    ollama_base: str = "http://localhost:11434"
    chat_model: str = "ollama/llama3.1"
    embed_model: str = "nomic-embed-text"
    compact_token_threshold: int = 1500
    class Config:
        env_file = ".env"

settings = Settings()
```

```python
# app/deps.py
import boto3, httpx
import redis.asyncio as aioredis
from psycopg_pool import AsyncConnectionPool
from app.config import settings

redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
pg_pool = AsyncConnectionPool(settings.postgres_dsn, open=False)

def s3():
    return boto3.client(
        "s3", endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_key,
        aws_secret_access_key=settings.s3_secret,
    )

async def embed(text: str) -> list[float]:
    """768-dim embedding via Ollama. Used for semantic memory + GDPR search test."""
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{settings.ollama_base}/api/embeddings",
                         json={"model": settings.embed_model, "prompt": text})
        r.raise_for_status()
        return r.json()["embedding"]

def user_key_prefix(user_id: str) -> str:
    """The ONE namespace convention every Redis writer must use (Step 6 relies on it)."""
    return f"u:{user_id}:"
```

**Expected result.** `uv run python -c "from app import deps"` imports without error.

**Verify.** Create the bucket now so later steps have somewhere to write:

```bash
uv run python -c "from app.deps import s3; s3().create_bucket(Bucket='lab-artifacts')"
```

**Troubleshoot.**
- *`ModuleNotFoundError: psycopg_pool`*: `uv add "psycopg[binary,pool]"` installs the pool extra — confirm it's in `pyproject.toml`.
- *S3 `EndpointConnectionError`*: MinIO isn't up or `S3_ENDPOINT` is wrong (must be `http://localhost:9000`, not the 9001 console).

---

### Step 4 — The thin LLM leaf (`llm.py`)

**What.** Exactly ONE async function: `complete(messages, model) -> str`. No branching on content, no persistence, no business rules.

**Why.** This is the deterministic boundary from Lecture 2 made concrete: "LLM proposes, code disposes." Everything else in the app is testable without a model because the model touches the system only here. The moment you `if "yes" in response` *inside* this file, you've lost the boundary.

**Do it.** Use LiteLLM so the same function works against Ollama (free) or a provider key by only changing the model string.

```python
# app/llm.py
import litellm

async def complete(messages: list[dict], model: str) -> str:
    """The ONLY place the app talks to a model. No business logic here."""
    resp = await litellm.acompletion(model=model, messages=messages)
    return resp.choices[0].message.content

async def summarize(text: str, model: str) -> str:
    """Compaction helper — still just a model call, no flow control."""
    messages = [
        {"role": "system", "content":
         "Summarize the conversation below. PRESERVE every ID, order number, "
         "date, and money amount VERBATIM. Do not paraphrase identifiers."},
        {"role": "user", "content": text},
    ]
    return await complete(messages, model)
```

**Expected result.** A quick smoke test returns text.

**Verify.**

```bash
uv run python -c "import asyncio; from app.llm import complete; \
print(asyncio.run(complete([{'role':'user','content':'say hi in 3 words'}], 'ollama/llama3.1')))"
```

**Troubleshoot.**
- *LiteLLM can't reach Ollama*: LiteLLM's Ollama provider defaults to `http://localhost:11434`; if yours differs, set `OLLAMA_API_BASE` env var. Confirm `curl http://localhost:11434/api/tags` lists your model.
- *Slow first call*: the model loads into RAM on first use — subsequent calls are faster.

---

### Step 5 — Stateful chat with idempotency and buffer (`chat.py`, `memory.py`)

**What.** `POST /chat` taking `{tenant_id, user_id, conversation_id, message, idempotency_key}`. Deterministic flow with a single model call at step (d).

**Why.** This is the stateful-chatbot rung of the escalation ladder (Lecture 1) with correct state handling (Lecture 4): dedup, hot-buffer read with durable fallback, threshold-triggered compaction that preserves entities verbatim, single leaf call, then dual-write (durable + hot).

**Do it.** First `memory.py` — the buffer and compaction.

```python
# app/memory.py
import json
from app.deps import redis_client, embed, pg_pool, user_key_prefix
from app.llm import summarize
from app.config import settings

def _buf_key(user_id: str, conv: str) -> str:
    return f"{user_key_prefix(user_id)}sess:{conv}"    # note: under the user namespace

async def load_buffer(user_id: str, conv: str) -> list[dict]:
    raw = await redis_client.lrange(_buf_key(user_id, conv), 0, -1)
    if raw:
        return [json.loads(x) for x in raw]
    # miss -> rebuild from Postgres (durable source of truth)
    async with pg_pool.connection() as cx:
        rows = await (await cx.execute(
            "SELECT role, content FROM messages WHERE conversation_id=%s ORDER BY id",
            (conv,))).fetchall()
    return [{"role": r, "content": c} for r, c in rows]

async def push_turn(user_id: str, conv: str, role: str, content: str):
    key = _buf_key(user_id, conv)
    await redis_client.rpush(key, json.dumps({"role": role, "content": content}))
    await redis_client.expire(key, 3600)      # hot buffer TTL = 1h (volatile tier)

def estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages) // 4     # ~4 chars/token heuristic

async def maybe_compact(user_id, tenant_id, conv, messages) -> list[dict]:
    """If over threshold, summarize old turns (entities verbatim), keep recent verbatim."""
    if estimate_tokens(messages) < settings.compact_token_threshold:
        return messages
    keep = messages[-4:]                         # keep last 2 turns verbatim
    old = messages[:-4]
    text = "\n".join(f"{m['role']}: {m['content']}" for m in old)
    summary = await summarize(text, settings.chat_model)
    # persist the summary as long-term semantic memory (vector tier)
    vec = await embed(summary)
    async with pg_pool.connection() as cx:
        await cx.execute(
            "INSERT INTO memories(user_id, tenant_id, text, embedding) VALUES (%s,%s,%s,%s)",
            (user_id, tenant_id, summary, str(vec)))
    return [{"role": "system", "content": f"Summary so far: {summary}"}] + keep
```

Now `chat.py` — the deterministic orchestrator.

```python
# app/chat.py
from fastapi import APIRouter
from app.models import ChatIn
from app.deps import redis_client, pg_pool, user_key_prefix
from app.memory import load_buffer, push_turn, maybe_compact
from app.llm import complete
from app.config import settings

router = APIRouter()

@router.post("/chat")
async def chat(body: ChatIn):
    dedup = f"{user_key_prefix(body.user_id)}idem:{body.idempotency_key}"
    # (a) dedup — return prior response if this key was already handled
    prior = await redis_client.get(dedup)
    if prior is not None:
        return {"reply": prior, "cached": True}

    # (b) load recent turns (Redis hot, Postgres fallback)
    history = await load_buffer(body.user_id, body.conversation_id)
    # (c) compact if over token threshold (entities preserved verbatim)
    history = await maybe_compact(body.user_id, body.tenant_id,
                                  body.conversation_id, history)

    messages = history + [{"role": "user", "content": body.message}]
    # (d) THE ONLY model call
    reply = await complete(messages, settings.chat_model)

    # (e) persist durably (Postgres) + push hot (Redis)
    async with pg_pool.connection() as cx:
        await cx.execute(
            "INSERT INTO conversations(conversation_id,user_id,tenant_id) VALUES (%s,%s,%s) "
            "ON CONFLICT DO NOTHING",
            (body.conversation_id, body.user_id, body.tenant_id))
        await cx.execute(
            "INSERT INTO messages(conversation_id,user_id,tenant_id,role,content,idempotency_key) "
            "VALUES (%s,%s,%s,'user',%s,%s) ON CONFLICT DO NOTHING",
            (body.conversation_id, body.user_id, body.tenant_id, body.message, body.idempotency_key))
        await cx.execute(
            "INSERT INTO messages(conversation_id,user_id,tenant_id,role,content) "
            "VALUES (%s,%s,%s,'assistant',%s)",
            (body.conversation_id, body.user_id, body.tenant_id, reply))
        await cx.execute(
            "INSERT INTO spend_ledger(user_id,tenant_id,tokens_in,tokens_out) VALUES (%s,%s,%s,%s)",
            (body.user_id, body.tenant_id, len(body.message)//4, len(reply)//4))
    await push_turn(body.user_id, body.conversation_id, "user", body.message)
    await push_turn(body.user_id, body.conversation_id, "assistant", reply)
    await redis_client.set(dedup, reply, ex=3600)   # remember this idempotency key
    # (f) return
    return {"reply": reply, "cached": False}
```

Add the DTO to `models.py`:

```python
# app/models.py  (append)
from pydantic import BaseModel

class ChatIn(BaseModel):
    tenant_id: str
    user_id: str
    conversation_id: str
    message: str
    idempotency_key: str
```

**Expected result.** `POST /chat` returns `{"reply": "...", "cached": false}` and, on a repeated idempotency key, `"cached": true` with no second model call.

**Verify.** After Step 7 wires `main.py`, hit it:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{
  "tenant_id":"t1","user_id":"u1","conversation_id":"c1",
  "message":"My order is #A-7788 for $42.50. Remember it.",
  "idempotency_key":"k1"}' | python -m json.tool
```

**Troubleshoot.**
- *Everything blocks/hangs*: you're mixing sync psycopg with async. Use `async with pg_pool.connection()` as shown, and ensure the pool is opened at startup (Step 7).
- *`UniqueViolation` surfaces as a 500*: the `ON CONFLICT DO NOTHING` clauses handle the dup insert; make sure they're present on the user-message insert.

---

### Step 6 — GDPR cascade delete (`gdpr.py`)

**What.** `DELETE /users/{user_id}` that deletes across **all four stores in order** and returns a per-store receipt.

**Why.** Lecture 5's thesis: erasure must cascade to *derived* copies (caches, embeddings, indexes, blobs), not just the primary row. The receipt makes the delete auditable. The Redis part uses `SCAN` (never `KEYS`) against the per-user prefix designed in Step 3.

**Do it.**

```python
# app/gdpr.py
from fastapi import APIRouter
from app.deps import redis_client, pg_pool, s3, user_key_prefix
from app.config import settings

router = APIRouter()

async def _delete_redis(user_id: str) -> int:
    """SCAN (never KEYS) over the per-user prefix, DEL in batches."""
    deleted, match = 0, f"{user_key_prefix(user_id)}*"
    async for key in redis_client.scan_iter(match=match, count=500):
        await redis_client.delete(key)
        deleted += 1
    return deleted

@router.delete("/users/{user_id}")
async def delete_user(user_id: str):
    receipt = {}
    # 1. Postgres rows (durable) — messages, conversations, memories, spend_ledger
    async with pg_pool.connection() as cx:
        for table in ("messages", "conversations", "memories", "spend_ledger"):
            cur = await cx.execute(f"DELETE FROM {table} WHERE user_id=%s", (user_id,))
            receipt[f"pg_{table}"] = cur.rowcount
        cur = await cx.execute("DELETE FROM users WHERE user_id=%s", (user_id,))
        receipt["pg_users"] = cur.rowcount
    # 2. Redis (all caches, sessions, dedup keys under the user namespace)
    receipt["redis_keys"] = await _delete_redis(user_id)
    # 3. Vector rows — here pgvector's `memories` is already covered by step 1;
    #    if you run Qdrant, issue a filtered delete on payload.user_id here too.
    receipt["vector_rows"] = receipt["pg_memories"]
    # 4. Object storage — delete every object under the user's prefix
    cli, n = s3(), 0
    paginator = cli.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=f"users/{user_id}/"):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objs:
            cli.delete_objects(Bucket=settings.s3_bucket, Delete={"Objects": objs})
            n += len(objs)
    receipt["s3_objects"] = n
    return {"user_id": user_id, "deleted": receipt}
```

> **Qdrant variant (optional).** If you added `qdrant/qdrant` to compose instead of using pgvector for memory, the vector delete is a filtered delete:
> ```python
> from qdrant_client import QdrantClient, models
> QdrantClient(url="http://localhost:6333").delete(
>     collection_name="memories",
>     points_selector=models.Filter(must=[
>         models.FieldCondition(key="user_id",
>             match=models.MatchValue(value=user_id))]))
> ```

**Expected result.** `{"user_id":"u1","deleted":{"pg_messages":N,...,"redis_keys":M,"vector_rows":K,"s3_objects":J}}`.

**Verify.** Covered by the test in Step 8; a manual smoke:

```bash
curl -s -X DELETE localhost:8000/users/u1 | python -m json.tool
```

**Troubleshoot.**
- *Redis keys survive*: your writers didn't use `user_key_prefix`. Every session/cache/dedup key MUST start with `u:{user_id}:` or `SCAN` won't find it — this is the single most common GDPR bug.
- *`delete_objects` fails on empty page*: guard with `if objs:` (shown) — `Delete` with an empty list is invalid.

---

### Step 7 — Wire the app and `/health`

**What.** `main.py` mounts the routers, opens the pg pool at startup, runs the schema, and exposes `GET /health` that pings each store.

**Why.** The Definition of Done requires `/health` to report each store reachable. A lifespan hook is the clean place to open/close the pool and ensure the schema + bucket exist.

**Do it.**

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.deps import redis_client, pg_pool, s3
from app.config import settings
from app.models import SCHEMA_SQL
from app import chat, gdpr

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pg_pool.open()
    async with pg_pool.connection() as cx:
        await cx.execute(SCHEMA_SQL)
    try:
        s3().create_bucket(Bucket=settings.s3_bucket)
    except Exception:
        pass    # already exists
    yield
    await pg_pool.close()

app = FastAPI(lifespan=lifespan)
app.include_router(chat.router)
app.include_router(gdpr.router)

@app.get("/health")
async def health():
    status = {}
    try:
        status["redis"] = await redis_client.ping()
    except Exception as e:
        status["redis"] = f"down: {e}"
    try:
        async with pg_pool.connection() as cx:
            await cx.execute("SELECT 1")
        status["postgres"] = True
    except Exception as e:
        status["postgres"] = f"down: {e}"
    try:
        s3().head_bucket(Bucket=settings.s3_bucket)
        status["object_storage"] = True
    except Exception as e:
        status["object_storage"] = f"down: {e}"
    status["ok"] = all(v is True for v in status.values())
    return status
```

**Expected result.** `GET /health` returns all stores `true` and `"ok": true`.

**Verify.**

```bash
uv run uvicorn app.main:app --reload &
sleep 3
curl -s localhost:8000/health | python -m json.tool
```

**Troubleshoot.**
- *`SCHEMA_SQL` re-run errors*: everything uses `IF NOT EXISTS`, so it's idempotent — if it errors, a table's DDL is missing that guard.
- *Windows Git-Bash `&` backgrounding*: run uvicorn in a separate terminal instead of backgrounding; job control differs from bash.

---

### Step 8 — The two proving tests (`test_chat.py`, `test_gdpr_delete.py`)

**What.** (1) An idempotency + compaction test; (2) the GDPR test that seeds all four stores and asserts zero residue, including a semantic search that returns no user vectors.

**Why.** The Definition of Done is verified by tests, not by eyeballing. Counting model calls proves idempotency; the semantic search proves the vector index is truly purged (the #1 missed store).

**Do it.** In `conftest.py`, monkeypatch `llm.complete` to a counter so tests are fast and model-free.

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock
import app.chat as chat_mod

@pytest.fixture
def mock_llm(monkeypatch):
    m = AsyncMock(return_value="OK reply keeping order #A-7788 for $42.50")
    monkeypatch.setattr(chat_mod, "complete", m)
    return m
```

```python
# tests/test_chat.py  (sketch — fill the client setup)
import pytest

@pytest.mark.asyncio
async def test_idempotency_one_call_one_row(client, mock_llm):
    body = {"tenant_id":"t1","user_id":"ut","conversation_id":"cc",
            "message":"order #A-7788 $42.50","idempotency_key":"same"}
    await client.post("/chat", json=body)
    await client.post("/chat", json=body)      # same key twice
    assert mock_llm.call_count == 1            # exactly one model call
    # assert exactly one 'user' message row in Postgres for this conversation
```

```python
# tests/test_gdpr_delete.py  (sketch)
import pytest
from app.deps import pg_pool, redis_client, s3, embed, user_key_prefix
from app.config import settings

@pytest.mark.asyncio
async def test_delete_purges_every_store(client):
    uid = "gdpr-user"
    # seed: message + memory(vector) + cached redis key + object
    async with pg_pool.connection() as cx:
        await cx.execute("INSERT INTO users(user_id,tenant_id) VALUES (%s,'t1')", (uid,))
        vec = await embed("secret note about order #Z-9 for $100")
        await cx.execute("INSERT INTO memories(user_id,tenant_id,text,embedding) "
                         "VALUES (%s,'t1',%s,%s)", (uid, "secret note", str(vec)))
    await redis_client.set(f"{user_key_prefix(uid)}cache:x", "their text")
    s3().put_object(Bucket=settings.s3_bucket, Key=f"users/{uid}/a.txt", Body=b"blob")

    r = (await client.delete(f"/users/{uid}")).json()
    assert r["deleted"]["redis_keys"] >= 1

    # assert ZERO residue everywhere
    async with pg_pool.connection() as cx:
        for t in ("messages","memories","spend_ledger","users"):
            n = (await (await cx.execute(f"SELECT count(*) FROM {t} WHERE user_id=%s",(uid,))).fetchone())[0]
            assert n == 0, t
    assert [k async for k in redis_client.scan_iter(f"{user_key_prefix(uid)}*")] == []
    # semantic search must return NO user-owned vector
    q = await embed("order #Z-9 $100")
    async with pg_pool.connection() as cx:
        hits = await (await cx.execute(
            "SELECT count(*) FROM memories WHERE user_id=%s "
            "ORDER BY embedding <-> %s LIMIT 5", (uid, str(q)))).fetchone()
    assert hits[0] == 0
    objs = s3().list_objects_v2(Bucket=settings.s3_bucket, Prefix=f"users/{uid}/")
    assert objs.get("KeyCount", 0) == 0
```

> For `client`, use `httpx.ASGITransport` with `app.main.app` (or FastAPI's `TestClient`), sharing the same store connections. `<->` is pgvector's L2 distance operator.

**Expected result.** `uv run pytest -q` is green: idempotency (one call, one row), and GDPR (zero rows/keys/objects/vector hits).

**Verify.**

```bash
uv run pytest -q
```

**Troubleshoot.**
- *`pytest-asyncio` "async def not natively supported"*: add `asyncio_mode = "auto"` under `[tool.pytest.ini_options]` in `pyproject.toml`, or mark tests with `@pytest.mark.asyncio`.
- *GDPR test flaky on the vector search*: you asserted on distance instead of ownership — filter by `user_id` and assert count 0, as shown.

---

## Putting it together — a short end-to-end run

```bash
docker compose up -d
uv run uvicorn app.main:app --reload    # separate terminal

# 1. health: all stores up
curl -s localhost:8000/health | python -m json.tool

# 2. a stateful conversation that carries an ID + amount forward
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{
  "tenant_id":"t1","user_id":"demo","conversation_id":"c1",
  "message":"My order #A-7788 is $42.50 — remember it.","idempotency_key":"m1"}'
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{
  "tenant_id":"t1","user_id":"demo","conversation_id":"c1",
  "message":"What was my order number and total?","idempotency_key":"m2"}'
# -> reply should still contain #A-7788 and $42.50

# 3. idempotency: replay m1 — same reply, cached:true, no new row
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{
  "tenant_id":"t1","user_id":"demo","conversation_id":"c1",
  "message":"My order #A-7788 is $42.50 — remember it.","idempotency_key":"m1"}'

# 4. GDPR delete — receipt with per-store counts
curl -s -X DELETE localhost:8000/users/demo | python -m json.tool

# 5. prove residue is gone
docker exec -it $(docker compose ps -q postgres) psql -U lab -d lab \
  -c "SELECT count(*) FROM messages WHERE user_id='demo';"     # -> 0
uv run pytest -q
```

---

## Definition of Done

Restating the spine's Week 1 checklist as verifiable checks:

- [ ] **All stores up + health.** `docker compose up` brings up Redis, Postgres, MinIO; `GET /health` returns `true` for each store and `"ok": true`.
- [ ] **10-turn coherence + compaction preserves entities.** `POST /chat` holds a coherent 10-turn conversation; turn 11 crosses `COMPACT_TOKEN_THRESHOLD` and triggers compaction, and a test asserts the summary still contains every ID/amount from earlier turns (verbatim).
- [ ] **Idempotency.** Same `idempotency_key` twice → exactly one persisted `messages` row (user role) and exactly one model call (`mock_llm.call_count == 1`).
- [ ] **GDPR test green.** `test_gdpr_delete.py` passes: after delete, Postgres returns 0 rows, Redis `SCAN` finds 0 keys, the vector store returns 0 rows AND a semantic search returns 0 user-owned hits, and object storage returns 0 objects for the user.
- [ ] **Deletion receipt.** `DELETE /users/{id}` returns a JSON receipt with per-store deletion counts (`pg_messages`, `redis_keys`, `vector_rows`, `s3_objects`, …).
- [ ] **README tier justification.** A one-paragraph note in `README.md` naming, for each store, *what* lives there and *why* (the four-tier justification from Lecture 3).

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `type "vector" does not exist` | wrong Postgres image | use `pgvector/pgvector:pg16`, recreate the container |
| Redis keys survive a delete | writer didn't use `user_key_prefix` | route every session/cache/dedup key through `u:{user_id}:` |
| GDPR test passes but semantic search still hits | asserted distance, not ownership | filter `WHERE user_id=%s` and assert count 0 |
| `/chat` hangs | sync DB calls in async path | use `async with pg_pool.connection()`; open pool in lifespan |
| Second identical request creates a new row | Redis dedup miss + no DB constraint | keep the `UNIQUE(conversation_id, idempotency_key)` + `ON CONFLICT DO NOTHING` |
| Compaction drops an order number | free-form summary | the `summarize` system prompt forces verbatim IDs; keep last N turns raw |
| LiteLLM 404 to Ollama | wrong base URL | `curl localhost:11434/api/tags`; set `OLLAMA_API_BASE` if non-default |
| MinIO `EndpointConnectionError` | wrong port | endpoint is `:9000` (API), console is `:9001` |
| Git-Bash mangles `-v /path` | MSYS path conversion | prefix command with `MSYS_NO_PATHCONV=1` |

---

## Stretch goals (optional)

- **Swap pgvector for Qdrant.** Add `qdrant/qdrant` to compose, write memories to a `memories` collection with `user_id` in the payload, and implement the filtered delete shown in Step 6. Now your GDPR delete spans a *separate* vector service — closer to production and a better interview story.
- **Two-store dedup race test.** Fire the same idempotency key concurrently (e.g. `asyncio.gather` of two `/chat` calls) and prove the `UNIQUE` constraint + `ON CONFLICT` still yields exactly one row under a race.
- **Retention TTL sweep.** Add a `created_at`-based purge job that deletes `messages`/`memories` older than a retention window, and a test that it doesn't touch fresh rows — a preview of the "backups and archives" cascade from Lecture 5.
- **Archive raw turns to object storage.** On each turn, write the raw request/response JSON to `users/{user_id}/turns/{id}.json` in MinIO, then extend the GDPR test to assert those blobs are gone too.
- **Rebuild-from-Postgres drill.** `FLUSHDB` Redis mid-conversation and prove the next `/chat` transparently rebuilds the buffer from Postgres (the durable-fallback path in `load_buffer`).
