# Lecture 3: Change Data Capture vs Webhooks, and Tombstone Deletes

> A source system just deleted a customer record. Three weeks later a compliance officer asks you to prove that customer's data no longer appears in any RAG answer — and you discover the delete never propagated, because your pipeline treated "the row is gone" as "nothing happened." This lecture exists because *propagating change* — especially deletes — is the part of data engineering that silently rots and then explodes at audit time. After this lecture you will be able to choose between log-based CDC, watermark CDC, and webhooks for a given source; build the hand-rolled watermark CDC this phase uses; and explain why a delete must travel as an explicit **tombstone** record, never as absence.

**Prerequisites:** SQL + Postgres basics; the idempotency/landing-zone material from Lectures 1–2 of this phase; comfort with transactions and `updated_at` timestamps · **Reading time:** ~24 min · **Part of:** Phase 5 (Data Engineering for AI) Week 1

---

## The core idea (plain language)

You have a **source of truth** (a Postgres table, a SaaS app, an API) that keeps changing, and a **downstream** (your landing zone, then your RAG index, then answers) that must stay in sync with it. The entire problem is: *how do changes travel from source to downstream, reliably, in order, without losing any — including deletes?*

There are two fundamentally different shapes:

1. **Pull / Change Data Capture (CDC)** — *you* go ask the source "what changed since I last looked?" You are in control of when and how much you read. Two sub-flavors:
   - **Log-based CDC** (Debezium, logical replication): read the database's write-ahead log, which records *every* row change — insert, update, delete — in commit order.
   - **Watermark-based polling CDC**: query `WHERE updated_at > last_watermark`, remember the high-water mark, repeat. Lightweight, no special DB privileges.
2. **Push / Webhooks (event push)** — the source *pushes* an HTTP event to you the moment something happens ("order.created", "user.deleted"). You run an endpoint and react.

The deep asymmetry that drives every decision: **with pull, the source's log/table is the authority and you reconcile against it; with push, the network delivery is the authority — and networks lose, duplicate, and reorder messages.** A webhook that never arrives leaves you silently, permanently out of sync with no way to notice. A CDC reader that falls behind is still reading an ordered, complete log; it just catches up.

And the sharpest edge of all, the one this phase is built around: **a deleted row is a change too.** If your propagation mechanism only carries "here is a row," then a delete looks like *nothing* — the row simply stops appearing — and downstream has no signal to remove it. The fix is a **tombstone**: an explicit record that says "entity X is deleted," which travels through the exact same pipe as inserts and updates.

---

## How it actually works (mechanism, from first principles)

### Log-based CDC: the WAL is already a change stream

Every serious database already writes an ordered, durable log of changes before it touches the actual table pages — Postgres calls it the **WAL** (write-ahead log). This exists for crash recovery and replication, but it happens to be *exactly* the change stream you want: for every committed transaction it records, in commit order, "row 17 in table `documents` changed from A to B," including deletes.

Log-based CDC tools (Debezium, or Postgres **logical replication** with a publication + replication slot) attach to this log and emit one event per row change:

```
WAL (commit order)                CDC events emitted
  INSERT documents id=1     -->   {op:"c", id:1, after:{...}}
  UPDATE documents id=1     -->   {op:"u", id:1, before:{...}, after:{...}}
  DELETE documents id=1     -->   {op:"d", id:1, before:{...}, after:null}
```

The properties you get for free are the reason people reach for it:

- **Completeness** — every change is captured, because the DB *cannot* commit without writing the log first. You physically cannot miss an update the way a polling job can miss a fast-flipping value.
- **Deletes are first-class** — the `op:"d"` event *is* the tombstone. The DB tells you the row died.
- **Ordering** — events arrive in commit order, so `before/after` transitions are consistent.

The cost: you need elevated DB privileges (a replication slot), a running connector (Debezium usually means Kafka Connect + Kafka), and operational care — **an un-consumed replication slot pins the WAL and can fill the disk on the source database**, which is a genuine way to take down production. We are *not* running Debezium this week; we are learning the mental model and then building the lightweight cousin.

### Watermark-based polling CDC: the version we build

When you don't control the source DB deeply, or you want zero extra infrastructure, you approximate CDC with a **watermark**. The recipe:

1. Keep a stored high-water mark: the largest `updated_at` (or monotonic version/id) you have already ingested.
2. Each run: `SELECT * FROM documents WHERE updated_at > :last_watermark ORDER BY updated_at`.
3. Write those rows to the CDC landing zone.
4. **Advance the watermark to the max `updated_at` you just read — transactionally, tied to the durable write.**

```
run 1: watermark = 1970-01-01T00:00Z
   SELECT ... WHERE updated_at > 1970...   -> 4 rows, max updated_at = 10:15:04
   write 4 rows to landing
   set watermark = 10:15:04
run 2 (nothing changed):
   SELECT ... WHERE updated_at > 10:15:04  -> 0 rows
   watermark unchanged                      -> idempotent no-op
```

That last line is the whole idempotency guarantee from Lecture 1, applied to a live table: re-running lands zero new rows if nothing changed.

Two mechanism details that bite if you get them wrong:

- **Use `>` on the watermark, and prefer a strict monotonic cursor, but understand the boundary risk.** If two rows share the exact same `updated_at` and one lands in this batch while the other commits a microsecond later with the identical timestamp, `> max_seen` will skip the second one forever. Defenses: use `updated_at` plus a tiebreaker (`(updated_at, id) > (:wm_ts, :wm_id)`), or re-read a small overlap window `updated_at >= last_watermark - safety_lag` and dedup on primary key. Overlap + dedup is the pragmatic default because it also absorbs clock skew.
- **Advance the watermark only after the write is durable, in the same transaction (or after fsync).** If you advance first and crash before the write commits, those rows are lost forever — the next run's `WHERE updated_at > wm` will never see them. This is the "watermark stored non-atomically" pitfall from the spine.

### Webhooks: push, and everything that can go wrong with a network

A webhook flips control: the source calls *your* HTTPS endpoint with a payload the instant something happens. Latency is excellent (milliseconds) and there's no polling waste. But the delivery is a single network call, and single network calls have three failure modes you must design around:

- **Loss.** Your endpoint is down for 30 seconds during a deploy; the provider's retry budget expires; the event is gone. You have no record that anything was missed. This is the killer — silent, undetectable data loss.
- **Duplication.** The provider sent the event, your handler processed it, but your `200 OK` was lost on the way back. The provider retries; you get the same event twice. **At-least-once delivery is the norm**, so your consumer *must* be idempotent — dedup on the provider's event id.
- **Reordering.** `user.updated (name=Bob)` and `user.updated (name=Bob Smith)` can arrive out of order under retries/parallel delivery, so last-write-wins by arrival time gives you the *wrong* final state. You need a version/timestamp *inside* the payload to order by, not arrival order.

```
Webhook reality (at-least-once, unordered)
  source --X-->  you      (lost: deploy window)
  source ---->   you ->200 lost-> source retries -> you (duplicate)
  source ==v2==> you                                 } arrive
  source ==v1==> you  (network reorder)              } out of order
```

So a "correct" webhook consumer is: verify signature → dedup on event id → order by an in-payload version → apply idempotently → and, because loss is undetectable, **reconcile periodically against the source anyway** (which is... a poll). Most robust webhook systems end up bolting a CDC-style reconciliation on the side.

---

## Worked example

You own a `documents(id, body, updated_at, deleted bool)` table feeding a RAG corpus. Here is the full watermark-CDC lifecycle with numbers, including the delete.

**State at start:** watermark = `2026-07-01T00:00:00Z`. Table:

| id | body            | updated_at            | deleted |
|----|-----------------|-----------------------|---------|
| 1  | "Refund policy" | 2026-07-01T09:00:00Z  | false   |
| 2  | "Shipping FAQ"  | 2026-07-01T09:05:00Z  | false   |

**Run A.** `SELECT ... WHERE updated_at > '2026-07-01T00:00Z'` returns rows 1 and 2. We write two records to `landing/cdc/dt=2026-07-01/part-A.jsonl`, each `{"id":1,"body":"Refund policy","updated_at":"...09:00:00Z","deleted":false}`. Advance watermark → `2026-07-01T09:05:00Z`.

**A user edits doc 1** at `09:30`. Now `id=1, body="Refund policy v2", updated_at=09:30, deleted=false`.

**Run B.** `WHERE updated_at > 09:05` returns only row 1. Land one record (the update). Advance watermark → `09:30`. Downstream re-embeds doc 1; doc 2 untouched. Good — updates flow.

**Now the critical part: doc 2 is deleted.** The application does **not** run `DELETE FROM documents WHERE id=2`. If it did, the row would vanish and our `WHERE updated_at > wm` query would simply never return it again — **the delete would be invisible to CDC**. "No row" is indistinguishable from "never existed" and from "unchanged and below the watermark."

Instead the app does a **soft delete**:

```sql
UPDATE documents SET deleted = true, updated_at = now() WHERE id = 2;
-- id=2, body="Shipping FAQ", updated_at=2026-07-01T10:00:00Z, deleted=true
```

**Run C.** `WHERE updated_at > 09:30` returns row 2 — because the soft delete *bumped `updated_at`*. We land it as a record with `deleted:true`: that is the **tombstone**.

```json
{"id":2,"body":"Shipping FAQ","updated_at":"2026-07-01T10:00:00Z","deleted":true}
```

Advance watermark → `10:00`. Downstream sees `deleted:true` for id 2 and *acts on it*: deletes doc 2's chunks from the vector index. The fact in "Shipping FAQ" is now unretrievable.

The mechanism in one sentence: **soft-delete flag + a bumped `updated_at` turns a delete into just another row change that watermark CDC already knows how to capture** — and the `deleted:true` payload is the tombstone that tells downstream to remove, not upsert.

Contrast the numbers: if you had hard-deleted, Run C returns 0 rows, watermark still advances harmlessly, and doc 2's vectors live in your index forever. The corpus and the source have silently diverged by exactly one document — the one someone asked you to delete.

---

## How it shows up in production

- **The compliance time bomb (the reason this is a milestone).** GDPR/CCPA "right to erasure" isn't satisfied by deleting the source row; it's satisfied when the data is gone *everywhere it flowed* — your landing zone, your Parquet, your vector index, your caches. If deletes don't propagate, you are non-compliant and can't even tell. This lecture's tombstone path is precisely what makes Week 3's governance proof possible: delete a doc → tombstone → indexer removes vectors → the answer changes.
- **Webhook gaps you find months late.** A provider had a 4-minute outage during your deploy last quarter; ~2,000 events never arrived. Nobody noticed because *missing data doesn't throw errors*. You discover it when totals don't reconcile against a monthly export. The cost isn't the outage — it's the trust in every downstream number computed since.
- **The replication slot that filled the disk.** A Debezium/logical-replication consumer crashed and nobody restarted it. Postgres dutifully retained WAL for the un-consumed slot; a week later the source DB ran out of disk and *production writes failed*. Log-based CDC is powerful but it couples your reliability to the source DB's health.
- **Duplicate-driven double counting.** A webhook `payment.succeeded` was delivered twice; a non-idempotent handler credited the account twice. At-least-once delivery guarantees this happens eventually — idempotency keyed on event id is not optional.
- **The "phantom answer."** A RAG bot keeps citing a policy doc that was deleted weeks ago, because the delete never reached the index. Users trust the bot; the bot is confidently serving retracted information. This is a support and legal incident, not a "data quality" ticket.

---

## Common misconceptions & failure modes

- **"If the row is gone, downstream will figure it out."** No. Downstream only sees what you send it. Absence is not an event. Without a tombstone, a delete is literally undetectable to a pull pipeline reading `WHERE changed > wm`.
- **"Webhooks are real-time so they're strictly better than polling."** Webhooks win on latency and lose on *guarantees*. They can be lost (undetectably), duplicated, and reordered. Low latency is worthless if you can't trust completeness.
- **"Exactly-once delivery solves this."** Exactly-once delivery across a network is effectively a myth (Lecture 1). What you actually engineer is **at-least-once delivery + idempotent, dedup-keyed writes**. Design for duplicates; don't pray they won't happen.
- **"Timestamp watermarks are exact."** Ties on `updated_at`, clock skew, and long-running transactions (a row committed *after* your query started but with an earlier `updated_at`) can drop rows. Use a tiebreaker cursor or a small overlap window + primary-key dedup.
- **"Hard delete is cleaner."** Hard deletes are the enemy of CDC and auditability. If the source must hard-delete, you need log-based CDC (which captures the `op:"d"`) or a DB trigger that writes a tombstone to an outbox — a plain watermark poll *cannot* see a hard delete.
- **"Advance the watermark first, then write."** Reversed and dangerous: a crash in between loses data permanently. Durable write first (or same transaction), then advance.
- **"Soft-deleted rows are just filtered out.”** They must be *carried* as tombstones through ingestion, then filtered/actioned downstream. Filtering them at the source query (`WHERE deleted=false`) re-hides the delete and reintroduces the original bug.

---

## Rules of thumb / cheat sheet

*(Numbers below are approximate engineering defaults, not benchmarks — tune to your source.)*

- **Choose log-based CDC when:** you control the DB, need *every* change and hard-deletes captured, need commit-order guarantees, and can operate a connector + monitor replication-slot lag. Best completeness; highest ops cost.
- **Choose watermark CDC when:** you have `updated_at` (or a monotonic version), can enforce soft-deletes, want zero extra infra, and can tolerate poll-interval latency (seconds to minutes). This is our Week 1 default.
- **Choose webhooks when:** you *don't* control the source (it's a SaaS), you need low latency, and you can add idempotent dedup + periodic reconciliation. Never rely on webhooks alone for completeness.
- **Always pair webhooks with a reconciliation poll.** Push for latency, pull for correctness.
- **Watermark hygiene:** compare with `>` plus a tiebreaker, or re-read `>= wm - safety_lag` and dedup on PK. Advance the watermark only after a durable write, in the same transaction.
- **Deletes:** soft-delete (`deleted=true` + bump `updated_at`) so watermark CDC captures them; emit an explicit tombstone record; never `WHERE deleted=false` at ingest.
- **Consumers are idempotent, always.** Dedup on a stable key (primary key for CDC, event id for webhooks). Order by an in-payload version, never by arrival time.
- **Latency ladder (rough):** log-based CDC ~sub-second–seconds · watermark poll ~poll interval (e.g., 1–5 min) · webhooks ~milliseconds but lossy.
- **Watch the replication slot.** Un-consumed slot = growing WAL = source disk risk. Alert on slot lag.

---

## Connect to the lab

Week 1's lab has you build `ingest_cdc.py`: seed a `documents(id, body, updated_at, deleted bool)` table, `SELECT` rows where `updated_at > last_watermark`, write them to the CDC landing zone, and advance the watermark **transactionally**. The load-bearing step is modeling a delete: soft-delete a row (`deleted=true`, bump `updated_at`), re-run, and verify a **tombstone JSONL record** (`"deleted":true`) lands — never a silently dropped row. Store the watermark so a crash between fetch and commit can't lose or double-write data. That tombstone is not busywork: in Week 3 you *consume* it in `indexer.py` to delete the doc's vectors from Qdrant, and `reports/governance_proof.md` proves the deleted fact becomes unanswerable. If the tombstone doesn't land here in Week 1, the governance proof is impossible in Week 3.

---

## Going deeper (optional)

- **Debezium documentation** (debezium.io) — read the "Introduction to Debezium" and the Postgres connector overview. Note how it emits `op: c/u/d` events and how deletes plus *tombstone events* work for log compaction. The canonical CDC reference even if you never run it.
- **PostgreSQL documentation** (postgresql.org/docs) — "Logical Replication" and "Write-Ahead Logging (WAL)" chapters, plus `CREATE PUBLICATION` and replication slots. This is the machinery under log-based CDC.
- **dlt (dltHub) docs** (dlthub.com/docs) — "Incremental loading" gives you watermark/cursor-based extraction with state management for free; compare it to your hand-rolled version.
- **Kafka documentation** (kafka.apache.org/documentation) — "Message Delivery Semantics" section for the at-least-once / exactly-once mental model; and the "log compaction" concept where a null-value record *is* the tombstone.
- **Martin Kleppmann, *Designing Data-Intensive Applications*** — chapters on replication and "the log" (Ch. 5, 11). The best conceptual grounding for why the log is the change stream.
- Search queries: *"Debezium tombstone event delete"*, *"watermark based CDC updated_at boundary problem"*, *"webhook idempotency at-least-once dedup"*, *"logical replication slot WAL disk full"*, *"log compaction tombstone Kafka"*.

---

## Check yourself

1. Your source hard-deletes a row (real `DELETE`, no flag). Why can a watermark poll never see it, and what two mechanisms *can* capture it?
2. A webhook provider promises delivery of every event. Why is this still not enough for a completeness-critical pipeline, and what do you add?
3. Two rows commit with the identical `updated_at`, and one lands in this batch. Explain how `WHERE updated_at > :wm` can drop the other one, and give two fixes.
4. Why must you advance the watermark *after* (or within) the durable write, never before? Walk through a crash.
5. Explain, with the exact field involved, how a soft delete lets watermark CDC capture a deletion, and what the tombstone record looks like.
6. Downstream (the RAG indexer) receives a record with `deleted:true`. Why is it wrong to filter these out at ingest time, and what should happen instead?

### Answer key

1. A watermark poll only sees rows still present with `updated_at > wm`; a hard-deleted row is *gone*, indistinguishable from "never existed" or "unchanged," so the query never returns it. Capture it with **log-based CDC** (the WAL records the `op:"d"` delete) or a **DB trigger/outbox** that writes a tombstone row on delete.
2. Webhooks are at-least-once and can be **lost undetectably** (your endpoint was down, retries expired), **duplicated**, and **reordered**. Delivery is the authority and networks fail silently. Add an **idempotent, dedup-keyed consumer** (dedup on event id, order by in-payload version) *plus a periodic reconciliation poll* against the source for completeness.
3. If both rows share `updated_at = T` and you advance the watermark to `T` after landing the first, `WHERE updated_at > T` will forever skip the second (it's equal, not greater). Fixes: (a) use a composite cursor `(updated_at, id) > (:wm_ts, :wm_id)`; or (b) re-read an overlap window `updated_at >= wm - safety_lag` and dedup on primary key.
4. If you advance the watermark first and the process crashes before the write commits, the next run queries `WHERE updated_at > new_wm` and **never sees the uncommitted rows** — permanent data loss. Writing durably first (same transaction, or after fsync) means a crash just re-runs harmlessly from the old watermark.
5. The app sets `deleted = true` **and bumps `updated_at = now()`**. The bumped `updated_at` makes the row satisfy `WHERE updated_at > wm`, so the next poll returns it like any change. The landed record — `{"id":2, ..., "deleted":true}` — is the tombstone: an explicit "this entity is deleted" signal.
6. Filtering `deleted:true` out at ingest re-hides the delete — downstream never learns it should remove the doc, so its vectors/facts persist (the original bug). Instead the indexer must **act on the tombstone**: delete that doc's chunks from the index. The tombstone must be carried through, then actioned, not dropped.
