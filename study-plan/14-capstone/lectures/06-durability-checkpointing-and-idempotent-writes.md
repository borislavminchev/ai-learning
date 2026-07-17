# Lecture: Durable Execution — Checkpointing + Idempotency for Crash-Safe Writes

> This is the spine of Capstone Week 2. Your supervisor can now query, retrieve, and — behind a human gate — *write*. A write that fires twice in a regulated domain is a duplicate claim, a double refund, a compliance incident. This lecture is the design note for making the agent survive a crash mid-run and resume with **exactly one** write on the books. You will leave able to place a checkpointer, design an idempotency key that is stable across resume, build the dedup wall in Postgres, and write a crash-resume proof that a fake in-memory setup cannot pass.

**Prerequisites:** Phase 6 Lectures 12 (Durable Execution & Checkpointing), 13 (Idempotency & Safe Replay), 14 (HITL Interrupts); Capstone Week 1 (the served retrieval spine); Week 2 Steps 1–3 (DB scope, typed tools, the supervisor graph). · **Reading time:** ~14 min · **Part of:** Capstone Week 2

## The integration problem

Week 1 gave you a retrieval surface that *reads*. Week 2's supervisor *acts* — and acting is where durability stops being a nicety and becomes the deliverable. Three facts collide:

1. **Agents are long-running and side-effecting.** A supervisor run spans several LLM calls, tool calls, and a human-approval pause that may last minutes to days. Each of those is an opportunity for the process to die: OOM kill, rolling deploy, spot reclaim, a hung vendor call the orchestrator kills.
2. **A crash mid-run must not lose work *or* repeat it.** Restarting a 12-step run from zero re-bills every LLM call and — worse — risks re-firing the `submit_action` write. Both are unacceptable; the second is a data-integrity bug.
3. **The write is gated by a human.** The HITL `interrupt()` (Lecture 14) *is* a pause that might sit overnight. A pause that lives only in RAM is not a pause; it is a time bomb that a server restart detonates.

The integration job this week is to wire three subsystems so these facts become non-events: a **durable checkpointer** (state survives the process), an **idempotency key generated upstream** (the write is deduplicable), and a **dedup wall in Postgres** (the second attempt no-ops). The subtle truth binding them, and the single most-skipped insight in production agents:

> Checkpointing gives you **at-least-once** execution of a node on resume. It does **not** give you exactly-once side effects. "Exactly-once" is *engineered* — with a dedup key and a unique constraint — not selected from a checkbox.

Get this wrong in one of two classic ways and your "durable" agent double-writes: use `MemorySaver` (state dies with the process — the exact failure you claim to defend against), or mint the idempotency key *inside* the retried node (a fresh key on resume defeats the dedup wall). Both pass a naive test and fail in production. This note designs around both.

## Architecture & how the pieces connect

Three stores, one graph. The write path threads through all of them.

```mermaid
flowchart TD
  user[user turn]
  subgraph graph[LangGraph StateGraph thread_id=run-123]
    state[state = messages, user_id, idempotency_key, tokens, usd<br/>key lives IN state]
    sup[supervisor]
    tools[sql_query / doc_search]
    action[action_node<br/>interrupt PAUSE + checkpoint, waits for human<br/>Command resume<br/>INSERT ON CONFLICT DO NOTHING]
    state --> sup
    sup --> tools
    sup --> action
  end
  saver[PostgresSaver checkpoints<br/>per-thread state]
  writes[writes table<br/>idempotency_key PK<br/>the dedup wall]
  store[PostgresStore namespace,key<br/>long-term memory, cross-thread facts]
  user --> sup
  graph -->|after every super-step| saver
  graph -->|the write itself| writes
  saver --> store
```

**The flow, step by step.** The supervisor loop runs; after *every super-step* LangGraph serializes the full state and writes a checkpoint row keyed by `thread_id`. When the model decides to write, control reaches `action_node`, which calls `interrupt()` — this **persists state and parks the run**. A human later resumes with `Command(resume={"approved": true})`. Only then does the node execute the INSERT. The INSERT targets a `writes` table whose **primary key is the idempotency_key**, using `ON CONFLICT DO NOTHING` — so a second attempt with the same key inserts zero rows.

**Where the key comes from is the whole game.** The idempotency key must be part of the *checkpointed state* — generated **upstream**, before the interrupt, so it is captured in a checkpoint and is byte-identical on resume. If `action_node` did `key = uuid4()` at the moment of writing, a resume would re-enter the node (the crash left no checkpoint of the write), mint a *new* uuid4, miss the conflict, and insert a **second** row. The key's stability across resume is what turns the unique constraint into a wall instead of a speed bump. (This is the same discipline as Phase 6 Lecture 13's `sha256(thread_id + step + args)` — deterministic inputs only — applied to the capstone's `submit_action` payload.)

**Two Postgres roles, do not confuse them:**
- **Checkpoints** (`PostgresSaver`, package `langgraph-checkpoint-postgres`): *this thread's* execution state, short-term, per-run. Keyed by `thread_id`. Answers "where was run-123 when it crashed?"
- **Store** (`PostgresStore`): long-term, cross-thread facts keyed by `(namespace, key)`. Answers "what does this tenant call a claimant?" across all future runs. More on the split below.

## Key decisions & tradeoffs

**Decision 1 — `PostgresSaver`, never `MemorySaver`.** `MemorySaver` keeps checkpoints in the process heap. It dies exactly when the process dies — which is the precise failure durability exists to defend against. It is fine for a notebook and lethal for a "durable" claim. The trap is that `MemorySaver` **passes a fake resume test**: if your test creates one graph object, invokes it, then re-invokes the *same object*, state was never gone, so resume "works." The test proves nothing. `PostgresSaver` costs you a `cp.setup()` call and a Docker Postgres; that is the entire price of real durability. (Weigh SQLite for single-node local dev, Postgres for the shared/multi-worker capstone — one import swap; see Phase 6 Lecture 12.)

**Decision 2 — generate the idempotency key upstream, in state.** Options: (a) `uuid4()` inside `action_node` — **broken**, new key per resume; (b) deterministic hash of `thread_id + step + payload` computed anywhere and stored in state — safe; (c) a `uuid4()` minted *once* in an upstream node and written to state before the interrupt — also safe, because the checkpoint freezes it. Prefer (b) or (c). The tradeoff of (b) is that two *intentionally distinct* writes in one run must differ in `step`/args or they collapse to one key; (c) sidesteps that but requires you to remember to generate before the interrupt. Either way: **the key must be captured in a checkpoint before the write is attempted.**

**Decision 3 — `INSERT … ON CONFLICT DO NOTHING` with the key as PRIMARY KEY.** This is the dedup wall. The database, not the application, enforces uniqueness — the same "controls live at the lowest layer" principle as Week 1's server-side ACL filter and Week 2's Postgres RLS. Alternatives (a `SELECT` then `INSERT` in Python) race under concurrency; the unique constraint does not. Tradeoff: `DO NOTHING` silently swallows the second attempt, so if you need to distinguish "I just wrote it" from "it was already there," check the affected-row count or use `ON CONFLICT … DO NOTHING RETURNING` and branch on whether a row came back.

**Decision 4 — interrupt *before* the mutation, resume on explicit approval.** The `interrupt()` sits in front of the INSERT (Lecture 14). Because the checkpointer is durable, the parked run survives a restart during the human's lunch. Never infer approval from a null/timeout resume — require `{"approved": true}` and treat everything else as reject, or you have built an auto-approver.

**Decision 5 — checkpoints vs Store, by fact lifetime.** A fact scoped to *one run* (the current plan, this turn's tool results, the pending approval) belongs in **checkpoint state**. A fact that should outlive the run and be reused by future threads (tenant vocabulary: "this tenant says *claimant*, not *patient*"; a resolved entity mapping; a user preference) belongs in the **Store** under `(namespace, key)`. Put a long-term fact in a checkpoint and it dies with the thread; put per-run scratch in the Store and you leak state across runs and tenants.

## How it fails in production & how to prevent it

- **In-memory checkpointer masquerading as durable.** `MemorySaver` in a "durability" test passes because state never left the process. **Prevent:** use `PostgresSaver`, and prove durability by resuming in a **new process / new connection**, not a re-invoked object. Bonus: kill the OS process for real between the checkpoint and END.
- **Key minted inside the retried node.** `uuid4()` after the crash point produces a new key on resume → the conflict never triggers → **two rows**. This is the top cause of a failing resume test showing `count(*) == 2`. **Prevent:** generate upstream, store in state, assert it is byte-identical across resume.
- **Approval inferred from silence.** A null/timeout resume treated as "approved" is an auto-approver wearing a HITL costume. **Prevent:** explicit `approved: true`; default-reject.
- **The commit gap (Lecture 13).** Even with checkpointing, there is a sliver between "INSERT committed in Postgres" and "checkpoint recording it committed." A crash there re-enters the node on resume — and *only* the idempotency key + unique constraint saves you, because the DB already holds that key. **Prevent:** this is exactly why the dedup wall is non-optional; checkpointing alone cannot close the gap.
- **Fresh `thread_id` on resume.** Resuming with a new `thread_id` silently starts a brand-new run, re-executing everything and re-attempting the write. No error, just a doubled bill and a possible duplicate. **Prevent:** persist the `thread_id`; it is the run's primary key, not a log tag.
- **Checkpoint bloat.** Every checkpoint serializes the full state (including `messages`) after every super-step. Many threads × long transcripts is real Postgres storage. **Prevent:** prune finished threads; compact `messages` before it becomes a bloat ticket.

## Checklist / cheat sheet

- [ ] Compile with `PostgresSaver` (package `langgraph-checkpoint-postgres`); call `cp.setup()` once at deploy/migration time. Never `MemorySaver` outside a notebook.
- [ ] Run with a **stable, persisted `thread_id`**; resume = same `thread_id`, no new input.
- [ ] Idempotency key generated **upstream**, part of checkpointed state, deterministic (or minted once before the interrupt). Never `uuid4()` inside the retried node.
- [ ] `writes` table: `idempotency_key text PRIMARY KEY`; write via `INSERT … ON CONFLICT (idempotency_key) DO NOTHING`.
- [ ] `interrupt()` sits **before** the mutation; resume only on explicit `{"approved": true}`.
- [ ] Crash-resume proof: run past the approved insert → kill the process (new process, **same** `thread_id`) → resume → assert `SELECT count(*) FROM writes WHERE idempotency_key=? == 1`, not 2.
- [ ] Checkpoints hold per-run execution state; `PostgresStore` holds cross-thread `(namespace, key)` facts. Sort every fact by lifetime.
- [ ] Remember the rule: **checkpointing + idempotency, never one alone.**

## Connect to the build

This lecture is the design behind Week 2's Step 3 graph and `tests/test_resume.py`. Concretely: your `submit_action` payload already carries `idempotency_key` (Step 2) — make sure it is populated *upstream* of `action_node`, not inside it. The `sql/01_roles_rls.sql` `writes` table already declares `idempotency_key text PRIMARY KEY`; the `action_node` INSERT uses `ON CONFLICT (idempotency_key) DO NOTHING`. The Definition-of-Done crash-resume test is the proof: run the graph to just past the approved insert, simulate a crash (drop the app object / start a new process) with the **same** `thread_id`, resume from the `PostgresSaver` checkpoint to completion, and assert exactly one row. If that test shows two rows, the two suspects are (1) `MemorySaver` still wired in, or (2) the key minted inside the node — the two failure modes this note exists to prevent. When you add long-term memory, put tenant vocabulary in `PostgresStore` under `(tenant_id, "vocab")`, not in the thread's checkpoint.

## Going deeper (optional)

- **LangGraph docs — "Persistence."** Canonical reference for checkpointers, `thread_id`, super-steps, and `get_state_history`. Root: `langchain-ai.github.io/langgraph`. Search: `langgraph persistence checkpointer`.
- **`langgraph-checkpoint-postgres`** on the `langchain-ai/langgraph` GitHub repo — read `PostgresSaver` source and its `setup()` to see what a checkpoint row actually stores.
- **LangGraph docs — "Human-in-the-loop."** `interrupt()` / `Command(resume=…)` semantics and why the DB checkpointer is a prerequisite. Search: `langgraph human in the loop interrupt`.
- **LangGraph docs — "Long-term memory" / `Store`.** The `(namespace, key)` model and `PostgresStore`. Search: `langgraph long-term memory store`.
- **PostgreSQL docs — `INSERT … ON CONFLICT`.** The upsert/dedup primitive. Root: `postgresql.org/docs`. Search: `postgres insert on conflict do nothing`.
- **Stripe docs — "Idempotent requests."** The canonical API-level idempotency-key design; how a vendor closes the commit gap for you when you pass a key through. Search: `stripe idempotent requests`.

## Check yourself

1. Your resume test shows **two** rows in `writes` after a crash-and-resume. Name the two most likely causes and the one-line fix for each.
2. Why does `MemorySaver` pass a "resume" test that isn't actually testing durability — and what change to the test would expose it?
3. Precisely why must the idempotency key be generated *upstream* of `action_node` rather than inside it? Reference what a checkpoint stores.
4. Checkpointing gives at-least-once node execution on resume. Explain the commit gap that makes this true even with a durable checkpointer, and name the mechanism (not checkpointing) that prevents the duplicate write.
5. Give one fact that belongs in a **checkpoint** and one that belongs in the **Store**, and state the rule that sorts them.

### Answer key

1. **(a) `MemorySaver` is still wired in** — state never survived the crash, so resume replayed from scratch and re-attempted the write; fix: compile with `PostgresSaver`. **(b) The idempotency key is minted inside the retried node** (`uuid4()` in `action_node`) — resume computes a new key, misses the `ON CONFLICT`, inserts a second row; fix: generate the key upstream and carry it in checkpointed state.

2. `MemorySaver` keeps checkpoints in the process heap. A test that invokes one graph object and then re-invokes the *same object* never destroyed that heap state, so resume "works" — but nothing was persisted. Expose it by resuming in a **new process / new connection** (ideally after killing the OS process) with the same `thread_id`; `MemorySaver` then has nothing to load and the run either restarts from zero or fails, while `PostgresSaver` resumes correctly.

3. A checkpoint stores the *serialized state* after each super-step. If the key is part of state and was generated upstream, it is frozen in a checkpoint *before* the write is attempted, so on resume the re-entered node reads the **identical** key and the `ON CONFLICT` fires. If instead the node mints the key at write time (`uuid4()`), the pre-write checkpoint never captured it; a resume re-enters the node, generates a *different* key, the conflict never triggers, and a duplicate row is written.

4. The INSERT to Postgres and the checkpoint recording that it happened are separate systems and cannot be truly atomic — there is a window where the row is committed but the checkpoint is not. A crash in that window leaves LangGraph's last durable checkpoint *before* the write, so on resume it re-enters `action_node` and re-attempts the INSERT (at-least-once). Checkpointing cannot close this. The **idempotency key + `idempotency_key PRIMARY KEY` unique constraint** does: the row is already present, so `ON CONFLICT DO NOTHING` no-ops and the count stays at 1.

5. **Checkpoint:** the current run's pending approval, this turn's tool results, the in-flight `messages` — anything scoped to one `thread_id`. **Store:** tenant vocabulary ("this tenant says *claimant* not *patient*"), a user preference, a resolved entity mapping — anything meant to be reused by future threads, keyed by `(namespace, key)`. **Rule:** sort by *fact lifetime* — per-run execution state goes in the checkpoint; cross-thread durable facts go in the Store.
