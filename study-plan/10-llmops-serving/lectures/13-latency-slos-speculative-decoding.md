# Lecture 13: Latency SLOs and Speculative Decoding — Promising a Speed, Then Buying It

> You can already measure TTFT and TPOT at percentiles (Lecture 8). But a measurement is not a promise. A latency SLO turns "here's what we saw on Tuesday" into "here's what we commit to, here's the alert that fires when we break it, and here's the lever we pull to fix it." This lecture is about setting latency targets you can defend to a PM, alerting on the number users actually feel (not the flattering average), mapping every breach back to a physical cause, and then reaching for the biggest *lossless* TPOT lever we have — speculative decoding, where a small draft model proposes tokens the big model verifies in one pass. After this you can write a two-line SLO, justify each threshold, wire an alert that fires before users complain, and decide from your load-test numbers whether speculative decoding will help you or just add overhead.

**Prerequisites:** Lecture 8 (TTFT/TPOT, percentiles, load testing), Lecture 7 (memory-bound decode), Week 1 vLLM serving · **Reading time:** ~24 min · **Part of:** Phase 10 (LLMOps) Week 3

---

## The core idea (plain language)

Two ideas, one thread.

**First: an SLO is a contract with a number in it.** Latency has a shape — a queue, a prefill, a first token, then a drip of tokens. An SLO (Service Level Objective) picks the parts of that shape users feel, attaches a *percentile* target to each, and commits to it. For LLM serving the two that matter are almost always:

- **p95 TTFT < 800 ms** — Time To First Token: *perceived responsiveness*, the gap between "I hit send" and "something appeared." Dominated by queueing plus prefill.
- **p95 TPOT < 50 ms** — Time Per Output Token (a.k.a. inter-token latency, ITL): *streaming smoothness*, how fast tokens arrive once they start. 50 ms/token ≈ 20 tokens/sec, comfortably faster than people read.

Those numbers are defensible defaults, not laws — you set them from what the *product* needs and what your *hardware* can deliver, then defend the gap between the two.

The non-negotiable part is the percentile. **You alert on p95 (or p99), never the average.** The average — and p50, the median — describe the *typical* request, and the typical request is fine. The users who churn, file the ticket, or tweet the spinner screenshot live in the tail. A p50 of 300 ms with a p99 of 6 s looks healthy on a "mean latency" dashboard while quietly bleeding your worst-served 1%. **The average is where pain hides.**

**Second: once you have a TPOT target, speculative decoding is the main lever to hit it.** Decode is memory-bandwidth-bound (Lecture 7): generating one token means dragging the entire model's weights from VRAM to compute one token of math. That's wasteful — you moved billions of parameters to produce one token. Speculative decoding fixes the waste: a small, cheap **draft** model guesses the next several tokens, and the big **target** model checks all of them in a *single* forward pass. Every guess the target agrees with is a free token you got for one weight-read instead of several. The target is the arbiter — it only accepts tokens it would have produced itself — so **output quality is unchanged.** You speed up *how* you produce the same tokens, not *which* tokens you produce.

The lecture: define the two SLOs, justify the numbers, map breaches to causes, then use speculative decoding (plus levers you already know) to meet the TPOT target.

---

## How it actually works (mechanism, from first principles)

### Part 1 — Why two SLOs, and why percentiles

Recall the anatomy of a streaming request:

```
  send                                                         done
   │                                                             │
   ▼                                                             ▼
   ├── queue ──┬──── prefill ────┬─ tok1 ─┬─ tok2 ─┬ … ┬ tokN ───┤
   │           │                 │        │        │   │         │
   └───────────┴─────────────────┘        └────────┴───┘
        TTFT = queue + prefill              TPOT = mean gap between tokens
        (perceived responsiveness)          (streaming smoothness)
```

TTFT and TPOT are governed by *different physics*, which is exactly why they need separate targets:

- **TTFT** = time in the scheduler's queue + time for the big matrix-multiplies of **prefill** to chew through your prompt. It scales with prompt length and server busyness. A *compute-and-queue* number.
- **TPOT** = time for one decode step, a *memory-bandwidth* number: read the weights, read the KV cache, emit one token. Barely depends on prompt length; depends on batch size, model size, and memory bandwidth.

A single "latency" SLO would jam these two unrelated physics onto one threshold and tell you nothing about *which* broke. Two SLOs give you a diagnostic split for free.

**Why the percentile, not the average.** A percentile answers "what fraction of requests were at least this good?" p95 TTFT < 800 ms means 95% of requests got their first token within 800 ms; up to 5% may be slower. Consider ten TTFT samples (ms):

```
280  300  310  320  330  340  350  360  380   5200
```

- Average = **817 ms** — dragged up by one bad sample; describes *no actual request*.
- p50 (median) = **335 ms** — looks great, and is a lie about your worst traffic.
- p90 ≈ **380 ms**, p99 ≈ **5200 ms** — the tail is right there; one in a hundred users stares at a 5-second spinner.

Alert on the average (817 ms) and you'd think you were near budget for *everyone*. Alert on p50 (335 ms) and you'd think you had huge headroom. Only p95/p99 shows the cliff. **Set the SLO on the percentile that represents the users you refuse to lose** — usually p95 for a baseline, p99 when the product is latency-critical (voice, live coding). And at high concurrency the tail *is* the story: continuous batching (Lecture 8) makes a request that arrives when the batch is full wait for a slot, and that queueing lands squarely in the TTFT tail. Averages smear it away; p99 catches it.

### Part 2 — Mapping a breach to a cause

An alert that says "p95 TTFT is 1.4 s, budget is 800 ms" is only useful if you can turn it into an action. Because TTFT and TPOT come from different physics, *which SLO broke* already narrows the cause:

```
TTFT SLO breached (slow first token)         TPOT SLO breached (slow streaming)
────────────────────────────────────        ───────────────────────────────────
• Queueing: arrivals outpace batch          • Memory-bandwidth saturation: batch
  admission → requests wait → add              is large, every step drags weights +
  replicas, raise --max-num-seqs (if           a big KV cache → each token slower
  VRAM allows), alert on queue time         • Batch too large: you traded TPOT for
• Prefill cost: prompts got longer            throughput → back off max-num-seqs or
  (bigger RAG context, longer history)         add a replica → smaller batch/request
  → chunked prefill, trim prompts           • Contention: another process / noisy
• Cold prefix cache: shared system            neighbor stealing the GPU → isolate,
  prompt changed (a timestamp!) so             check nvitop for stray processes
  caching stopped hitting → make the        • Long context: KV cache grew huge →
  prefix byte-stable                           each step reads more per token →
• Oversized prompts: someone pasted 40k        shorter contexts, paged/quantized KV,
  tokens → cap input, chunked prefill,         or accept higher TPOT for long
  route long prompts to a bigger card          sessions
```

Two levers appear on both sides for opposite reasons — the trap worth internalizing:

- **Batch size / `--max-num-seqs`.** *Bigger* batches raise throughput and can *lower* TTFT queueing (more admitted at once) but *raise* TPOT (each step reads a bigger KV cache). *Smaller* batches do the reverse. No free setting — you're choosing where to sit on the latency/throughput curve, and the two SLOs pull opposite ways.
- **Replicas.** A data-parallel replica behind the load balancer helps *both* — shrinks queues (TTFT) and lets each replica run a smaller batch (TPOT) — at the cost of more GPUs. When in doubt and you have budget, add a replica: the lever with no nasty tradeoff except money.

### Part 3 — Speculative decoding, mechanism-level

Start from the waste. In ordinary autoregressive decode, one token costs one full forward pass of the big model, and that pass is dominated by *reading the weights from VRAM*, not by the arithmetic. You moved ~14 GB of weights to compute one token. If you could check *several* candidate tokens in that same single weight-read, you'd amortize the expensive memory traffic across multiple tokens.

That's the trick. Two models cooperate:

- **Draft model** — small and fast (e.g. a 0.5B). It cheaply generates a short *guess* of the next k tokens, one at a time. Tiny model, cheap passes.
- **Target model** — the real one (e.g. 7B). It takes the prompt-so-far *plus* the draft's k guessed tokens and does **one** forward pass scoring all k+1 positions at once. A transformer scores many positions in a single pass (that's what prefill already does), so verifying k guesses costs about the same as generating one token normally.

The target then **accepts the longest correct prefix** of the guesses — every leading token matching what it would have produced itself — and rejects the rest, correcting the first wrong one with its own token.

```
Step: draft proposes 4 tokens, target verifies in ONE pass
──────────────────────────────────────────────────────────
draft guesses :   "the"   "cat"   "sat"   "on"
target verify :    ✓       ✓       ✓       ✗   (target wanted "under")
                  └───── accept 3 ─────┘   └ reject; emit target's "under"

Result of this one target forward pass: 4 tokens committed
  ("the", "cat", "sat", "under")  ← 3 accepted guesses + 1 correction
Normal decode would commit 1 token for the same weight-read.
```

**Why quality is untouched.** The target is the arbiter. A token survives only if it matches what the target would have emitted under its own sampling rule. The draft never *chooses* the output — it only *saves time* on tokens the target agrees with. Genius draft → accept 4 of 4, ~4× faster that step. Useless draft → accept 0, emit the target's one correction, spend one target pass for one token — same tokens you'd have gotten, just with wasted draft effort. **The output distribution is identical to plain decoding.** That's what "no quality loss" means, and why speculative decoding is safe to turn on in production in a way aggressive quantization is not.

**Where the win comes from, and where it evaporates.** The speedup is roughly "average tokens committed per target pass," which depends on the **acceptance rate** (how often the draft guesses right) and on being in a **memory-bound** regime where a verify pass over k tokens costs about the same as producing one.

- **Helps when decode is memory-bound and acceptance is high:** small batches / low concurrency, a draft genuinely predictive of the target (same family/tokenizer, on-distribution text like code or repetitive structured output), a big target. The extra tokens per pass are nearly free.
- **Hurts (or does nothing) when decode is compute-bound:** large batches already saturate compute, so the verify pass is no longer "free capacity waiting on a memory read" — verifying k tokens genuinely costs more, and rejected drafts plus the draft's own runtime become pure overhead. At high concurrency, speculative decoding can make you *slower*.

The mental model: **speculative decoding spends spare compute to buy back wasted memory bandwidth.** Spare compute (small batch, memory-bound) → great trade. No spare compute (large batch, compute-bound) → nothing to trade, just overhead. (We stay mechanism-level; the acceptance-rate-to-speedup math is a rabbit hole you won't debug in prod. What you need: high acceptance + memory-bound = win; low acceptance or compute-bound = loss.)

---

## Worked example

You run `Qwen2.5-7B-Instruct` on a single L4 (24 GB). A Lecture-8-style load test at **concurrency 4** gives baseline decode:

- TPOT p50 ≈ 22 ms/token, **TPOT p95 ≈ 45 ms/token** — just inside a 50 ms budget, no headroom.
- TTFT p95 ≈ 620 ms — comfortably inside 800 ms.

Add a draft — a 0.5B Qwen from the same family, so tokenizer and distribution match. In vLLM (2025 config style):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --speculative-config '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 4}'
```

(Older vLLM used flat `--speculative-model ... --num-speculative-tokens 4`; newer builds fold it into `--speculative-config` JSON. Check `vllm serve --help` for your version — the flag surface has churned across 2024→2026.)

Your workload is a coding assistant — predictable, repetitive tokens — so the draft guesses well; you measure ~2.5 tokens committed per target pass. Re-run the *same* k6 pass at concurrency 4:

- **TPOT p95 drops to ~20 ms/token.** Real headroom under the 50 ms SLO; streaming visibly speeds up.
- TTFT is roughly unchanged — speculative decoding is a *decode* optimization, doing nothing for prefill or queueing. It can even nudge TTFT up slightly as the draft's memory footprint eats KV-cache space; watch `--max-num-seqs`.

Now run the *same* server at **concurrency 64** for a throughput test. At that batch size decode is compute-bound. You measure TPOT p95 *worse* with speculation on than off — the verify passes now cost real compute and rejected drafts are wasted work. **The knob that won at concurrency 4 lost at concurrency 64.** In production you'd enable speculative decoding on the low-concurrency, latency-sensitive replica pool and leave it off on the batch/high-throughput pool — or gate it dynamically.

The before/after row for `slo/spec_decode.md`:

| config | concurrency | TPOT p50 | TPOT p95 | note |
|---|---|---|---|---|
| baseline | 4 | 22 ms | 45 ms | at SLO edge |
| +spec (0.5B draft, k=4) | 4 | ~11 ms | ~20 ms | comfortable headroom |
| +spec | 64 | — | *worse than baseline* | compute-bound, overhead dominates |

(Illustrative of the *shape* of the result — measure your own; acceptance rate is workload-specific.)

---

## How it shows up in production

- **The dashboard that lies.** A team ships, watches "average latency: 340 ms," declares victory. Tickets pile up about "the app freezing." The p99 was 8 seconds the whole time — a slow prefix-cache-miss path on 2% of requests. The fix was one line of Grafana (chart p99, alert on it), not one line of code. **If your latency panel shows a mean, you don't have latency monitoring.**
- **SLOs you can't meet on this hardware.** A PM says "make it 100 ms TTFT." Your model's prefill for a 2k-token prompt on an L4 is 400 ms with *zero* queue. No alerting fixes physics. The SLO is a two-way negotiation between product need and hardware reality, with prefix caching, a smaller/quantized model, or a bigger card as the levers to close the gap. **An SLO you structurally can't meet is a scheduled outage.**
- **The wrong lever for the wrong SLO.** TTFT is breaching, so someone lowers `--max-num-seqs` "to reduce load." That *raises* queueing (fewer admitted) and makes TTFT worse. TTFT breaches usually want *more* replicas or chunked prefill. Diagnosing which SLO broke tells you which lever to pull — the whole point of splitting them.
- **Speculative decoding on globally, regretted at peak.** Beautiful in the demo (low traffic, memory-bound), a regression at 5pm peak (high traffic, compute-bound). Treat it as per-pool / per-regime, and re-measure at real peak concurrency, not at concurrency 1. The draft also occupies VRAM and runs every step, so on a tight card it can cost a few `--max-num-seqs` slots — not literally free.
- **Reach for the cheap levers first.** Before/alongside speculative decoding: **prefix caching** (`--enable-prefix-caching`) crushes TTFT for shared system prompts; **chunked prefill** stops one giant prompt stalling everyone's stream (helps TTFT tail and TPOT under mixed loads); **quantization** (AWQ/int4) shrinks weights so each decode step reads less from VRAM (helps TPOT — re-eval quality); **more replicas** help both SLOs at the cost of GPUs. Speculative decoding is the specialist TPOT tool once the general ones are in place.

---

## Common misconceptions & failure modes

- **"Alert on the average, it's simpler."** The average is the one statistic guaranteed to hide the users you're losing. Alert on p95/p99; use the median only as a sanity glance.
- **"Speculative decoding lowers quality because a small model is involved."** No. The target verifies and accepts only its own tokens; the output distribution is identical to plain decoding. The draft saves time, never chooses content — a *lossless* speedup, unlike quantization or distillation.
- **"Bigger k (more speculative tokens) is always faster."** No. Larger k means more wasted work when the draft is wrong and more compute per verify pass. There's a sweet spot (often ~3–5); past it, or in a compute-bound regime, bigger k is slower.
- **"Speculative decoding will fix my slow first token."** No — it's a decode/TPOT optimization. TTFT is prefill + queue; speculation doesn't touch prefill and can slightly worsen TTFT via VRAM pressure. Use replicas / chunked prefill / prefix caching for TTFT.
- **"It'll help at any batch size."** The opposite risk: at high concurrency (compute-bound) it can regress TPOT. Always measure at your real peak concurrency before committing.
- **"A random small model makes a good draft."** Acceptance depends on the draft predicting the target. A same-family draft with the *same tokenizer*, trained on similar data, accepts far more often than an unrelated 0.5B. A mismatched tokenizer is often a hard incompatibility, not just low acceptance.
- **"SLO met in the load test = SLO met forever."** SLOs drift: prompts lengthen, traffic grows, a prefix-cache-busting change ships. The SLO is real only if there's a *live* alert on production percentiles, not a one-time benchmark.

---

## Rules of thumb / cheat sheet

*(All numbers are engineering defaults to start from and tune — not universal truths.)*

- **Default SLOs to open the conversation:** p95 TTFT < 800 ms, p95 TPOT < 50 ms (≈20 tok/s). Tighten toward ~300 ms TTFT / ~25 ms TPOT for latency-critical UX (voice, live coding); loosen for batch/async.
- **Always alert on p95 or p99, never the average.** Chart p50 too, but only as context.
- **TTFT breach → think queue + prefill:** more replicas, chunked prefill, prefix caching, trim/cap prompts. *Don't* shrink the batch — that worsens queueing.
- **TPOT breach → think memory bandwidth + batch:** smaller `--max-num-seqs` or more replicas (smaller per-request batch), quantize weights, shorten context, or turn on speculative decoding.
- **Speculative decoding helps when:** small batch / low concurrency, memory-bound decode, high acceptance (same-family draft, same tokenizer, predictable output like code/structured text), big target.
- **Speculative decoding hurts when:** large batch / high concurrency (compute-bound), low acceptance, mismatched draft. Re-measure at peak.
- **Draft-size heuristic:** ~10× smaller than target is a common start (0.5B for a 7B; 1–2B for a 70B). `num_speculative_tokens` ≈ 3–5 to start.
- **Order of levers for latency:** prefix caching → chunked prefill → right-size batch → add replicas → quantize (re-eval quality) → speculative decoding. Cheap and general first; specialist last.
- **The SLO is real only if it's alerted in prod.** A benchmark number without a live alert is a hope, not an objective.

---

## Connect to the lab

This is the theory behind **Week 3, Steps 2 and 5**. In Step 2 you relaunch vLLM with a small draft model (0.5B drafting for the 7B) and re-run a short k6 pass to record **TPOT before vs after** in `slo/spec_decode.md` — the worked example above, on your hardware, at *your* concurrency (do it at both a low and a high tier so you *see* the regime flip). In Step 5 you write the **SLO alert**: a script that reads recent latencies from your gateway metrics or the shadow log and exits non-zero / posts an alert when **p95 TTFT** breaches your target — the "alert on the percentile, not the average" rule made executable. Together they close the loop: measure the tail, promise a target, buy the speedup that meets it, get paged when you don't.

## Going deeper (optional)

- **vLLM docs — Speculative Decoding** (docs.vllm.ai): the authoritative, current flag surface (changed across versions — `--speculative-model`/`--num-speculative-tokens` vs newer `--speculative-config` JSON). Also the pages on automatic prefix caching and chunked prefill. Search: `vllm speculative decoding docs`, `vllm speculative config`.
- **Original speculative decoding papers** — "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., Google) and "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., DeepMind). Read the intro and the mechanism figure; skip the proofs. Search: `speculative decoding Leviathan`, `speculative sampling Chen DeepMind`.
- **Medusa** and **EAGLE** — newer draft-free / lightweight-head takes on "propose-then-verify," referenced by modern serving stacks. Search: `Medusa speculative decoding`, `EAGLE speculative decoding`.
- **Google SRE Book — "Service Level Objectives"** chapter (free online, sre.google): the canonical treatment of SLIs/SLOs/error budgets and why you alert on the objective. Search: `Google SRE book service level objectives`.
- **Horace He, "Making Deep Learning Go Brrrr From First Principles"**: the memory-bound vs compute-bound mental model that explains *why* speculative decoding helps in one regime and hurts in the other. Search that exact title.

## Check yourself

1. You have p50 TTFT = 300 ms and p99 TTFT = 6 s. Which do you alert on for your SLO, and what real-world experience does the other number hide?
2. In one or two sentences, why does speculative decoding **not** change output quality, and what role does the target model play?
3. Your TPOT SLO is breaching at peak (concurrency 64) after you turned on speculative decoding, even though it helped in testing at concurrency 4. What happened?
4. TTFT is breaching but TPOT is fine. Name two likely causes and the matching fix for each — and say why shrinking `--max-num-seqs` would be the wrong move.
5. Why is a 0.5B model from the *same family* a better draft than an unrelated 0.5B, in terms of the mechanism?
6. You need to cut TPOT and you're memory-bound at low concurrency. List two levers besides speculative decoding, and one reason you might prefer them first.

### Answer key

1. **Alert on p99 (or p95) — the tail.** The p50 (300 ms) describes the typical, fine request and hides that your worst ~1% of users wait 6 seconds for a first token — the ones who churn or file tickets. Average- or median-based alerting would look healthy while that tail bleeds.
2. Because the **target model is the arbiter**: it accepts only draft tokens matching what it would have produced under its own sampling rule, and emits its own token whenever the draft is wrong. The draft only saves time on tokens the target already agrees with; it never chooses the output, so the output distribution is identical to plain decoding.
3. At concurrency 64 decode is **compute-bound**, not memory-bound. The whole win is spending *spare* compute to reclaim wasted memory bandwidth; when the batch already saturates compute, verifying k drafted tokens genuinely costs more, and rejected drafts plus the draft's own runtime become pure overhead — so TPOT regresses. Enable it per-regime (low-concurrency/latency pool), not globally.
4. (a) **Queueing** — arrivals outpace batch admission → add replicas (and possibly *raise* `--max-num-seqs` if VRAM allows); (b) **prefill cost / oversized prompts / cold prefix cache** → chunked prefill, trim/cap prompts, fix a prefix-cache-busting change (e.g. a timestamp in the system prompt). Shrinking `--max-num-seqs` is wrong because it admits *fewer* concurrent requests, *increasing* queue wait and making TTFT worse.
5. Acceptance depends on the draft predicting the target's next tokens. A same-family draft shares the **tokenizer** (a mismatch is often a hard incompatibility, not just low acceptance) and was trained on similar data, so it guesses tokens the target agrees with far more often — more accepted tokens per verify pass, hence a bigger speedup. An unrelated 0.5B guesses poorly, so acceptance and speedup collapse.
6. **Quantize the weights** (AWQ/int4 — each decode step reads less from VRAM, directly lowering memory-bound TPOT; re-eval quality after) and **add a replica / shrink the per-request batch** (each request sees a smaller KV cache per step). Prefer these first because they're more general and don't add the draft's VRAM footprint or the compute-bound-regime risk; speculative decoding is the specialist tool once the general levers are in place.
