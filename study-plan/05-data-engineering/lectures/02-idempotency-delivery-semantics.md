# Lecture 2: Idempotency and Delivery Semantics — Engineering Around 'Exactly-Once'

> Every junior engineer eventually asks the vendor "does your queue guarantee exactly-once delivery?" and every senior engineer winces, because the honest answer is "no, and neither does anyone else's — we *engineer* the appearance of exactly-once out of two cheaper, achievable parts." This lecture exists because the entire replay-safety of your ingestion pipeline — the thing that lets you re-run a job 5× and get byte-identical output — rests on this one idea, and getting it wrong doesn't crash: it silently double-counts revenue, resurrects deleted documents in your RAG answers, or drops rows nobody notices for a month. After this lecture you will be able to explain why exactly-once is a fiction, place any pipeline on the delivery-semantics spectrum, build an idempotent write keyed on a stable content hash, order your durable-write-then-advance-watermark steps so a crash can't lose or duplicate data, and *prove* idempotency with a repeatable hash test.

**Prerequisites:** Basic distributed-systems intuition (network calls can fail and retry); SQL transactions; hashing basics; comfort reading Python · **Reading time:** ~26 min · **Part of:** Phase 5 (Data Engineering for AI) Week 1

---

## The core idea (plain language)

You want a fact to land in your system **exactly once**: one row, one event, one document — no duplicates, no losses. That is the business requirement. The trap is believing the *transport layer* can give it to you as a delivery guarantee.

It cannot, and the reason is almost stupidly simple. Any time system A sends something to system B over an unreliable channel, there is a moment where A has sent the message and is waiting for an acknowledgement. If the ack doesn't arrive, A faces an unanswerable question: *did B never receive it, or did B receive it and process it but the ack got lost on the way back?* A has no way to tell these two cases apart. It only has two moves:

- **Retry** (assume the message was lost). If it was actually delivered, you now have a **duplicate**.
- **Don't retry** (assume it was delivered). If it was actually lost, you now have a **loss**.

There is no third option at the transport layer. This is not a limitation of any particular vendor — it is a property of unreliable channels, formalized as the Two Generals problem. So the practical target is not "exactly-once delivery." It is:

> **at-least-once delivery + idempotent writes keyed on a stable dedup key = *effective* exactly-once.**

You let the transport be honest and choose to over-deliver (retry aggressively, never lose). Then you make the *receiving write* idempotent: applying the same record twice has the same effect as applying it once, because the second write recognizes "I already have this exact thing" and does nothing. Duplicates hit a wall at the write boundary and vanish. The system as a whole behaves as if each fact landed once — not because the wire promised it, but because you engineered the write to absorb the redundancy.

Hold that sentence in your head for the whole lecture: **exactly-once is an *end-to-end property you build*, not a *delivery guarantee you buy*.**

---

## How it actually works (mechanism, from first principles)

### The delivery-semantics spectrum

Three points, defined by **when you acknowledge/commit relative to when you do the work.** This is the whole game.

```
                 at-most-once            at-least-once           effective exactly-once
                 ------------            -------------           ----------------------
order of ops:    ack THEN process        process THEN ack        process idempotently
                                                                  THEN ack
on crash:        work may be lost         work may repeat         work neither lost
                 (already acked)          (not yet acked)          nor visibly repeated
duplicates:      never                    possible                dedup'd at the write
losses:          possible                 never                   never
you get:         speed, no dedup cost     safety, must dedup      safety + correctness
```

- **At-most-once:** acknowledge the message *first*, then try to process it. If you crash after ack but before the work commits, the work is gone and nobody will retry — it was already acked. Zero duplicates, possible loss. This is fine for high-volume, low-value telemetry (a dropped metrics sample doesn't matter) and disastrous for anything you must not lose.

- **At-least-once:** do the work and commit it *first*, acknowledge *last*. If you crash after committing but before acking, the sender never got the ack and will redeliver — so the work happens again. Zero loss, possible duplicates. **This is the correct default for ingestion**, because loss is usually unrecoverable (the source may not keep the data) while duplicates are something you can neutralize.

- **Effective exactly-once:** at-least-once transport, plus the write itself is idempotent. Redelivered duplicates are recognized and skipped, so the *observable outcome* is exactly-once even though the wire delivered "≥ once."

The critical realization: **you don't choose exactly-once on the spectrum — you choose at-least-once and then build idempotency on top.** Exactly-once isn't a fourth transport mode; it's at-least-once with a dedup layer bolted on at the destination.

### The Kafka mental model (without running Kafka)

You will not run Kafka in this week's lab, but Kafka's model is the cleanest mental scaffold, so borrow it. A Kafka topic is an append-only log of messages, each with a monotonically increasing **offset** (0, 1, 2, 3, …). A consumer reads messages and periodically **commits its offset** — a durable bookmark saying "I've processed everything up to here."

The delivery semantics fall directly out of *when you commit the offset relative to when you process the message*:

```
message:   [42: "order#A"] [43: "order#B"] [44: "order#C"]
                                 ^ consumer processing 43

AT-LEAST-ONCE:  process(43) -> write to DB -> THEN commit offset 43
                crash after write, before commit  ->  restart re-reads from 43
                                                   ->  order#B processed AGAIN  (duplicate)

AT-MOST-ONCE:   commit offset 43 -> THEN process(43)
                crash after commit, before write  ->  restart reads from 44
                                                   ->  order#B never written    (loss)
```

That committed offset is exactly the same idea as the **watermark** you'll store in `state.json` this week: a durable bookmark that says "I've consumed the source up to here." Kafka's "exactly-once semantics" (EOS, since 0.11) doesn't repeal physics — it wraps the produce-and-commit into a single **transaction** (offset commit + output write commit atomically) plus an idempotent producer (sequence numbers dedup retries on the broker). It is at-least-once + idempotence + transactions, productized. The lesson transfers whole: **commit your bookmark and your output atomically, or in the safe order, and dedup on a stable key.**

### Building an idempotent write, step by step

An idempotent write needs to answer one question fast: *"have I already landed this exact record?"* To answer it you need (1) a stable identity for the record and (2) a stable fingerprint of its content.

**Step 1 — Choose a stable dedup key (identity).** Use the source's natural/primary key: `issue_id`, `order_id`, `document_id`. This identifies *which entity* the record is about. It must be stable across re-fetches — the same GitHub issue must always have the same `id` whether you fetched it today or last week. Do **not** invent a key from ingestion metadata (row number, fetch order, autoincrement on your side) — those change on replay and defeat dedup.

**Step 2 — Content-hash only the stable content (fingerprint).** You also need to detect *changes* to an entity. Hash the record's meaningful fields into a `content_sha256`. The single most important rule here:

> **Hash only stable content. Never hash mutable/ingestion fields like `fetched_at`, `ingested_at`, a request ID, or a page cursor.**

Why this bites: suppose you include `fetched_at` in the hash. You fetch issue #42 at 09:00 → hash `A`. Nothing about the issue changed, but you re-fetch at 10:00 → now the record contains `fetched_at=10:00` → hash `B ≠ A`. Your pipeline concludes "issue #42 changed!" and lands a new copy. Every single re-fetch looks new. Your "idempotent" pipeline now grows without bound and your 5×-rerun hash test never stabilizes. The fix is to compute the hash over a canonical projection of *only the source's own content*:

```python
import hashlib, json

def content_hash(record: dict) -> str:
    # Project ONLY stable, source-owned fields. Exclude fetched_at, ingest_run_id, etc.
    stable = {k: record[k] for k in ("id", "title", "body", "state", "updated_at")}
    # Canonical serialization: sorted keys, no whitespace ambiguity, stable types.
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

Two subtle points. First, `updated_at` here is the *source's* modification timestamp — a legitimate content field, because when the source bumps it the entity genuinely changed. That's different from *your* `fetched_at`, which changes on every poll regardless of content. Second, serialization must be **canonical**: `sort_keys=True` and fixed separators so `{"a":1,"b":2}` and `{"b":2,"a":1}` hash identically. If your dict ordering wobbles, your hashes wobble, and idempotency dies for a reason that looks like a ghost.

**Step 3 — Skip records already landed with identical content-hash.** The write logic becomes:

```
for record in fetched_records:
    key  = record["id"]              # stable identity
    h    = content_hash(record)      # stable fingerprint
    seen = index.get(key)            # (key -> last content_hash) we've persisted
    if seen == h:
        skip                          # identical content already landed -> no-op (idempotent!)
    else:
        write(record, h)              # new entity OR changed content -> land it
        index[key] = h
```

The hash intuition you need (no proofs): SHA-256 produces a 256-bit digest. The chance two *different* contents collide to the same digest is astronomically small (~2⁻¹²⁸ for a birthday collision across ~2⁶⁴ distinct records). For a corpus of millions of documents, a hash collision is *far* less likely than a cosmic-ray bit-flip in your RAM. So "same hash ⇒ same content" is safe to treat as true in engineering practice. (Use SHA-256 for content identity; `xxhash` is fine and faster for *non-security* bucketing where you can tolerate the rare collision, e.g. a first-pass dedup, but for the canonical `content_sha256` you paste into your DoD, use SHA-256.)

### The watermark pattern for incremental fetch

You don't want to re-download the whole source every run — that's slow, expensive, and rate-limited. You want **incremental fetch**: pull only what's new since last time. The mechanism is a **watermark** — the high-water mark of what you've consumed, stored durably in `state.json`:

```json
{ "source": "github_issues", "watermark_updated_at": "2026-07-08T14:30:00Z", "watermark_id": 10456 }
```

Next run, you fetch only records with `updated_at > watermark_updated_at` (or `id > watermark_id` for append-only sources). This is your hand-rolled CDC: the source's `updated_at` column is your change-data-capture signal.

### The ordering requirement — the part that actually loses data

Here is where most home-grown pipelines have a latent data-loss bug. You have two durable side-effects per run: **(W) write the fetched records to the landing zone**, and **(A) advance the watermark in `state.json`**. The order and atomicity of W and A decide whether a crash between them loses or duplicates data.

```
WRONG (advance watermark first):
   fetch rows [updated_at up to 14:30]
   A: state.json = 14:30      <-- watermark advanced
   *** CRASH ***              <-- before rows are durably written
   W: (never happens)
   next run: fetch updated_at > 14:30  ->  the rows we fetched are SKIPPED FOREVER = LOSS
```

Advancing the watermark before the write is durable means a crash in the gap permanently skips those records — silent, unrecoverable loss. The rule:

> **Durably write the data (write + fsync) BEFORE advancing the watermark — or advance the watermark in the same transaction as the write.**

```
RIGHT (write durably, then advance):
   fetch rows [updated_at up to 14:30]
   W: write rows to landing, fsync   <-- data is durable on disk
   *** CRASH here? ***                <-- watermark still at old value
   A: state.json = 14:30              <-- only now advance
   crash between W and A:  next run re-fetches updated_at > OLD  ->  re-writes the same rows
                           ->  DUPLICATES, which idempotent write ABSORBS (skip on same hash)
```

Notice the asymmetry that makes this the correct order: crashing *before* W loses nothing (we hadn't claimed progress). Crashing *between* W and A produces **duplicates, not loss** — and duplicates are exactly what your idempotent content-hash write neutralizes. You've converted the one unrecoverable failure (loss) into the one recoverable failure (duplication). That is the entire trick, in miniature: **order your steps so the only possible failure is the kind idempotency can clean up.**

The `fsync` matters and is easy to forget. A plain `write()` puts bytes in the OS page cache, not on the platter/SSD; a power loss can lose them even though your code "wrote" the file. `fsync` (or `os.fsync(fd)` after `flush()`) forces the durability that makes "W is done" actually true. Advancing a watermark against a write that only *looked* durable is the same loss bug wearing a disguise.

The even cleaner option, when your landing store supports it: put W and A in **one transaction** (e.g. write both the rows and the watermark row to Postgres and `COMMIT` once). Then there is no gap at all — either both happen or neither does. This is the local, hand-rolled version of Kafka's transactional EOS.

### The failure taxonomy: dup, reorder, loss — and why idempotent writes neutralize two of three

Every delivery failure over an unreliable channel reduces to exactly three primitives. Name them, because "how does my design handle each of these three?" is the entire correctness checklist:

```
 failure    what happened                          idempotent-content-hash write handles it?
 --------   -----------------------------------    ------------------------------------------
 DUP        same record delivered/fetched twice     YES — 2nd arrival hashes identical -> skip
 REORDER    records arrive out of source order      YES (for state) — last-writer-by-content,
                                                     not by arrival order; converges to source
 LOSS       record never arrives / is skipped        NO — nothing to dedup against; must be
                                                     PREVENTED upstream (at-least-once + ordering)
```

- **Dup** is the easy one and the whole point of the content-hash write: the second copy of issue #102 with the same stable fields hashes to the same digest, the index already has that `(key → hash)`, and the write is a no-op. Duplicates die at the write boundary.

- **Reorder** is the subtle one. Suppose the source updated issue #102 twice — `state=open` then `state=closed` — and your fetch or your queue delivers them out of order (closed first, then open). A naive "last write wins by *arrival* order" would leave you at `open`, which is *wrong* (stale). Two engineering fixes, both compatible with idempotency: (1) key the write as an **upsert on the natural key** and let a monotonic source field (`updated_at`, a version/sequence number) decide the winner — write only if the incoming `updated_at` is newer than what you've stored, so out-of-order arrivals can't overwrite fresher state; (2) if you land *every version* (append-only history), reorder doesn't corrupt anything — you sort by the source's `updated_at` at read time, and the content-hash still dedups exact repeats. Either way the converged state is a pure function of the *source's* ordering, not the *network's*. Idempotency alone dedups; idempotency **plus a monotonic version guard** also makes you reorder-tolerant.

- **Loss** is the one idempotency *cannot* fix, and this is the critical asymmetry to internalize. If a record never arrives, there is no second copy to recognize and no hash to compare — there is simply a hole. You cannot dedup your way out of missing data. Loss must be **prevented**, not repaired, and prevention is precisely the two disciplines from the sections above: **at-least-once delivery** (retry until acked, never drop) so the record always eventually arrives, and **write-before-advance-watermark ordering** so a crash converts would-be loss into a recoverable duplicate. This is why the ordering rule matters so much: it is the mechanism that keeps every failure in the "dup or reorder" column — the two columns idempotency can clean up — and out of the "loss" column, which it cannot.

The design goal, stated as one sentence: **arrange your transport and write ordering so the only failures that can occur are dup and reorder, then let idempotent content-hash writes (plus a version guard) absorb both.**

---

## Worked example

Concrete numbers. You're ingesting GitHub issues for a repo. The source currently has 3 issues:

```
id=101  updated_at=2026-07-07T10:00Z  title="Login bug"      state=open
id=102  updated_at=2026-07-07T11:00Z  title="Slow query"     state=open
id=103  updated_at=2026-07-08T09:00Z  title="Typo in docs"   state=closed
```

**Run 1 (cold start).** `state.json` has no watermark, so fetch all. Compute content hashes over `{id,title,body,state,updated_at}` only:

```
101 -> sha256 = a1b2...   (land)
102 -> sha256 = c3d4...   (land)
103 -> sha256 = e5f6...   (land)
```

Write 3 records + fsync. **Then** advance watermark to the max `updated_at` seen = `2026-07-08T09:00Z`, `id=103`. Landed rows this run: **3**. Reconciled dataset hash (sha256 over the sorted list of the 3 content hashes) = `H1`.

**Run 2 (immediate re-run, nothing changed at source).** Fetch `updated_at > 2026-07-08T09:00Z` → but wait, issue 103's `updated_at` *equals* the watermark. This is the classic **boundary bug**: if you use strict `>` on `updated_at` alone and two records share that timestamp, or the watermark equals a record's timestamp, you either re-fetch 103 (harmless — idempotent write skips it) or skip a same-timestamp sibling (loss!). Defensive design: fetch `>=` the watermark and rely on the content-hash to dedup the boundary record, OR use a composite `(updated_at, id)` cursor. Suppose we fetch `>=` and get 103 back:

```
103 -> sha256 = e5f6...   index says 103 already landed with e5f6... -> SKIP
```

Landed rows this run: **0.** Reconciled hash = `H1` (unchanged). This is the DoD condition: *second run lands zero new rows, hash identical.*

**Runs 3, 4, 5 (still nothing changed).** Same as run 2: 0 landed, hash = `H1` every time. Paste `H1 == H1 == H1 == H1 == H1` into your README — **that is your proof of idempotency**, not an assertion of it.

**Run 6 (issue 102 gets a new comment; source bumps its `updated_at`).**

```
102  updated_at=2026-07-09T08:00Z  state=open  (body changed)
```

Fetch `>= 2026-07-08T09:00Z` → returns 103 (skip, same hash) and 102 (new `updated_at` → new hash `c3d4' ≠ c3d4`). Land the updated 102. Landed rows: **1.** Reconciled hash changes to `H2` — *correctly*, because the corpus genuinely changed. Watermark advances to `2026-07-09T08:00Z`. Idempotency doesn't mean "the output never changes" — it means "the output is a pure function of the source state; re-running against the *same* source state gives the *same* output."

**Now the crash scenario.** During Run 6, you write the updated 102 to landing and fsync (W done), then the process is killed *before* advancing the watermark (A not done). `state.json` still says `2026-07-08T09:00Z`. Run 7 starts, re-fetches 102 (updated_at `2026-07-09T08:00Z` ≥ old watermark), recomputes hash `c3d4'`, finds it already landed → **skip, 0 new rows**, then advances the watermark. No loss, no duplicate in the final dataset. The crash cost you one redundant fetch-and-hash, nothing else. Contrast: had you advanced the watermark *before* writing, Run 7 would fetch `> 2026-07-09T08:00Z`, find nothing, and issue 102's update would be **lost forever** — a bug you'd discover weeks later when someone asks why the corpus is stale.

---

## How it shows up in production

- **Double-counted money.** A payments webhook is delivered twice (the sender's ack timed out and it retried). Without an idempotent write keyed on the provider's `event_id`, you insert two ledger rows and charge/credit twice. This is the canonical "at-least-once bit you" story, and it's why Stripe et al. put an `Idempotency-Key` header on write APIs — they're pushing the dedup key to you because they *cannot* promise exactly-once delivery.

- **The resurrected document.** In your RAG corpus (Week 3), a doc is deleted at the source. If your delete path isn't idempotent/ordered correctly, a retried ingestion re-lands the deleted doc, its chunks get re-embedded, and the model starts answering from data you legally deleted. The governance proof you build later depends on the exactly-once discipline established here.

- **The slow leak (unstable hash).** Someone adds `fetched_at` to the hashed payload "for debugging." Nothing errors. But every nightly run now re-lands every record because every hash looks new. The landing zone grows 100×/month, storage bills climb, downstream dedup jobs slow to a crawl, and the idempotency test that used to pass now flaps. The root cause — one mutable field in the hash — takes an afternoon to find because *nothing is broken*, it's just quietly wrong.

- **Silent loss from watermark-before-write.** The most expensive one, because loss is invisible. Records fetched right before a crash are skipped forever. There's no error, no duplicate, no anomaly in row counts you'd notice — just a permanent hole in history. You find it during an audit or a backfill comparison, long after the source has aged the data out and it's unrecoverable.

- **Boundary-timestamp bugs.** Two records share the exact `updated_at` at the watermark boundary; strict `>` skips one. On a busy source (thousands of updates/second) this happens constantly. The fix (>= plus content-hash dedup, or a composite cursor) is cheap; the diagnosis is not, because it's data-dependent and won't reproduce in your quiet test environment.

- **Rate-limit and cost amplification.** Non-incremental "just re-fetch everything" pipelines hammer the source API. GitHub gives you 5,000 requests/hour authenticated; a full re-crawl of a large repo can blow that in one run, get you 403'd, and stall the pipeline. The watermark is what keeps each run's fetch proportional to *what changed*, not *how much exists*.

---

## Common misconceptions & failure modes

- **"Kafka/SQS/our broker gives exactly-once, so I don't need dedup."** No transport gives true exactly-once delivery. Kafka's EOS is at-least-once + idempotent producer + transactions, and it only spans Kafka-to-Kafka; the moment you write to an external system (your DB, your landing zone) *you* own idempotency. Assume at-least-once and dedup at the write.

- **"Idempotent means the output never changes."** Idempotent means *re-applying the same input yields the same result*. If the *source* changes, the output should change — that's correct behavior, not a violation. The invariant is "same source state in → same output out," proven by re-running against unchanged source and getting an identical hash.

- **"I dedup on the primary key, so I'm safe."** Keying on identity alone means you'll *miss updates*: if entity 102 changes, the key is still 102, and a naive "skip if key seen" drops the update. You need identity (key) *for locating the record* **and** content-hash *for detecting change*. Key says "which row"; hash says "same content or not."

- **"I'll advance the watermark first so I don't reprocess on retry."** That's optimizing for the wrong failure. Advancing first trades recoverable duplication for unrecoverable loss. Always write-then-advance (or same transaction). Reprocessing is cheap; a hole in history is forever.

- **"`write()` is durable."** It isn't until `fsync` (or a transaction commit) returns. Bytes sit in the OS cache; a power/kernel crash loses them. Any durability-ordering argument that skips fsync is reasoning about a lie.

- **"Hash collisions will corrupt my dedup."** For SHA-256 over your content, a collision is astronomically less likely than hardware failure. This is a non-problem; don't add complexity to defend against it. (Do use a canonical serialization so *identical content always hashes identically* — that's the real failure mode, and it's about determinism, not collisions.)

- **"Randomly ordered JSON is fine to hash."** No — dict/key ordering, float formatting, and unicode normalization all change the bytes and thus the hash. Non-canonical serialization makes identical content look different. Sort keys, fix separators, pin encoding.

---

## Rules of thumb / cheat sheet

- **Target = at-least-once + idempotent write.** Never design for "exactly-once delivery"; design for over-delivery you can absorb.
- **Dedup key = source's natural/primary key.** Stable across re-fetches. Never derive it from ingest metadata.
- **Content hash = SHA-256 over *stable source fields only*.** Exclude `fetched_at`, `ingested_at`, run IDs, cursors, page numbers — anything that changes without the entity changing.
- **Canonical serialization before hashing.** `json.dumps(..., sort_keys=True, separators=(",",":"))`, fixed encoding. Determinism is the whole point.
- **Write + fsync BEFORE advancing the watermark** — or commit both in one transaction. Order so the only possible failure is a duplicate, never a loss.
- **Fetch `>=` watermark, dedup the boundary with the hash** (or use a composite `(updated_at, id)` cursor). Strict `>` on a timestamp alone is a latent loss bug at same-timestamp boundaries.
- **Model deletes as tombstones, not absence.** "No row" ≠ "deleted." A retried run must be able to re-apply a delete idempotently. (Detailed in Week 3.)
- **Prove idempotency, don't assert it.** Re-run 5×; require zero new landed rows and a byte-identical reconciled `content_sha256`. Paste the 5 identical hashes.
- **Watermark lives in durable state** (`state.json` fsync'd, or a state table), never only in memory. In-memory watermark + crash = full re-scan or loss.
- *(Collision-probability and rate-limit figures above are order-of-magnitude engineering approximations, not measured benchmarks.)*

---

## Connect to the lab

This is the theory spine under **Week 1's `ingest_api.py`, `manifest.py`, and `tests/test_idempotent.py`.** Your paginated-API ingest stores the watermark in `state.json`, dedups fetched records on the primary key with a content hash over stable fields, and skips records already landed with an identical hash — exactly the Step 1–3 write above. Your `test_idempotent.py` runs the ingest twice against a recorded response and asserts the second run lands **0 rows** with an unchanged reconciled `content_sha256`; the Definition of Done pushes that to **5 identical hashes** pasted into the README — that's the "prove idempotency" bar made concrete. When you build `ingest_cdc.py`, apply the ordering rule literally: advance the Postgres watermark *in the same transaction* as the CDC write, and emit tombstone rows for deletes so Week 3's governance proof can propagate them.

---

## Going deeper (optional)

- **Apache Kafka documentation — "Design → Message Delivery Semantics"** (kafka.apache.org/documentation). The canonical, precise treatment of at-most/at-least/exactly-once and how EOS is built from idempotent producers + transactions. Read this even though you won't run Kafka this week.
- **Confluent blog — "Exactly-Once Semantics Are Possible: Here's How Kafka Does It"** (search: *Confluent exactly once semantics blog*). Engineering-level walkthrough of the transaction + idempotent-producer machinery.
- **Stripe API docs — "Idempotent requests"** (docs.stripe.com/api/idempotent_requests). The textbook production example of pushing a dedup key (`Idempotency-Key`) to the client because delivery can't be exactly-once. Short and clarifying.
- **Debezium documentation — "Introduction to Change Data Capture"** (debezium.io/documentation). For the CDC half of Week 1; the watermark you hand-roll is the lightweight version of what Debezium reads from the WAL.
- **dlt (data load tool) docs — "Incremental loading"** (dlthub.com). The library-provided version of watermarks + state + dedup you're building by hand; read after the lab to see what you'd otherwise reach for.
- *Designing Data-Intensive Applications* by Martin Kleppmann — Chapter 11 ("Stream Processing"), sections on fault tolerance and exactly-once, and Chapter 8 on unreliable networks/the Two Generals framing. The best long-form treatment.
- Search queries: *"two generals problem exactly once delivery"*, *"idempotent consumer pattern dedup key"*, *"watermark incremental extraction CDC"*, *"fsync durability write ordering crash consistency"*.

---

## Check yourself

1. A vendor's sales engineer promises "exactly-once delivery." In one sentence, why is that claim impossible at the transport layer, and what's the real property you can build?
2. Your teammate added `fetched_at` to the fields fed into `content_sha256` "for traceability." Your idempotency test now fails and the landing zone is growing every run. Explain the mechanism.
3. You store the watermark in `state.json`. Give the correct order of the two durable operations (write records, advance watermark) and explain what each of the two possible crash points costs.
4. Why do you need *both* a stable dedup key and a content hash? What goes wrong if you use only one?
5. Your ingest uses `WHERE updated_at > watermark`. On a busy source you occasionally lose a record. What's the bug and two ways to fix it?
6. Define, operationally, what it means to "prove idempotency" for this week's lab — what exactly do you measure and what result passes?

### Answer key

1. Because an unreliable channel makes "message lost" and "ack lost" indistinguishable to the sender, so it must either retry (risking a duplicate) or not (risking a loss) — there's no exactly-once *delivery*; you build **effective exactly-once** as at-least-once delivery + an idempotent write keyed on a stable dedup key.
2. `fetched_at` changes on every poll regardless of whether the entity changed, so every re-fetch produces a new content hash → the write logic sees "changed content" and lands a fresh copy each run. The dataset grows unboundedly and the 5×-rerun hash never stabilizes. Fix: hash only stable source-owned fields; exclude all ingestion metadata.
3. **Write records + fsync FIRST, then advance the watermark** (or commit both in one transaction). Crash *before* the write: nothing was claimed as done → re-fetched next run → no loss. Crash *between* write and advance: records are durable but the watermark is old → next run re-fetches and re-writes them → **duplicates**, which the idempotent content-hash write skips. The order guarantees the only possible failure is recoverable duplication, never unrecoverable loss.
4. The **key** locates the entity (which row); the **hash** detects whether its content changed. Key-only: you'll skip genuine updates (same key, changed content) — silent staleness. Hash-only: you lose stable identity for locating/updating the record and can't cheaply answer "is this the same entity, changed?" — you need both: identity + fingerprint.
5. Strict `>` skips records whose `updated_at` exactly equals the watermark (same-timestamp siblings, or the boundary record) — a same-timestamp loss bug that only shows on high-frequency sources. Fix (a): fetch `>=` the watermark and let the content-hash dedup the boundary record; fix (b): use a composite cursor `(updated_at, id)` so ties are broken deterministically.
6. Re-run the ingest **5× against unchanged source state** and require **zero new landed rows** on runs 2–5 and a **byte-identical reconciled `content_sha256`** across all runs. Passing = five identical hashes (pasted into the README) and no growth in landed rows — the output is a pure function of source state.
