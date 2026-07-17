# Lecture 2: Continuous batching, PagedAttention, prefix caching, and chunked prefill

> Everyone repeats "vLLM is 5–20× faster than a `generate()` loop" without being able to say *why*. This lecture is the why. Four mechanisms — continuous batching, PagedAttention, automatic prefix caching, and chunked prefill — do almost all of that work, and every one of them shows up as a number you'll watch in next week's load tests. After this you'll be able to give the two-sentence throughput story to a skeptical staff engineer, predict which flag moves which metric, and recognize the failure modes (the cache-buster, the OOM cliff, the prefill stall) *before* they page you at 2am.

**Prerequisites:** what a KV cache is and why autoregressive decoding is one-token-at-a-time; rough transformer anatomy (layers, attention, heads); comfort with GB/VRAM arithmetic; you've seen `docker run vllm/vllm-openai` from Lecture 1. · **Reading time:** ~28 min · **Part of:** Phase 10 (LLMOps: Serving, Optimization & Deployment) Week 1

---

## The core idea (plain language)

A naive inference server treats each request like a job with a fixed crew: you gather a batch, everyone runs the same number of steps together, and nobody leaves until the *slowest* member finishes. LLM requests are wildly uneven — one asks for 5 tokens, another for 800 — so that crew model wastes enormous amounts of GPU time waiting.

vLLM's throughput comes from refusing to wait. It rethinks four things:

1. **Continuous batching** — reshuffle the batch *every single decode step*, not every request. A finished request leaves instantly; a new one joins instantly. No waiting for the batch to drain.
2. **PagedAttention** — stop storing each sequence's KV cache as one big contiguous block. Store it as small fixed-size *pages*, like an operating system does with RAM. This kills the wasted memory that otherwise caps how many sequences you can hold at once.
3. **Automatic prefix caching** — if two requests share the same opening tokens (a system prompt, a few-shot block, a RAG document), compute that KV once and *reuse* it. The shared work happens one time instead of once per request.
4. **Chunked prefill** — when a huge prompt arrives, don't process all 30,000 tokens in one blocking gulp. Slice it and interleave the slices with everyone else's decoding so one whale doesn't freeze the whole tank.

Here is the two-sentence version to memorize: **"Continuous batching keeps the GPU busy by adding and evicting sequences every decode step instead of waiting for a static batch to finish, so no request is held hostage by the slowest one in its batch. PagedAttention stores the KV cache in OS-style pages instead of one buffer per sequence, so we pack far more concurrent requests into the same VRAM — and prefix caching plus chunked prefill make shared prompts and long prompts nearly free."** That's the whole 5–20× story.

## How it actually works (mechanism, from first principles)

### Continuous batching (iteration-level / in-flight scheduling)

To feel the win, first feel the pain of the alternative.

**Static batching.** Collect N requests, run them as one batch, generate until *all* of them emit a stop token or hit `max_tokens`, then return the batch and pick up the next N. The problem is that generation lengths are uneven and you pay for the max.

Worked numbers. Batch of 4 requests. Output lengths turn out to be 20, 40, 60, and 400 tokens. Static batching runs **400 decode steps** for all four (the short ones just emit padding / sit idle after they finish). Total useful token-steps = 20+40+60+400 = 520. Total token-steps the GPU actually spent = 4 × 400 = 1600. Utilization = 520/1600 ≈ **33%**. Two-thirds of your decode compute went to babysitting finished sequences. This waiting-on-the-slowest effect is **head-of-line blocking**: the 20-token request can't return its answer or free its slot until the 400-token request is done, even though it finished 380 steps earlier.

**Continuous batching (iteration-level scheduling).** The scheduler runs one decode step across whatever sequences are currently active, then *re-decides the batch*. A sequence that emitted its stop token is evicted immediately and its slot is handed to a waiting request that same step.

```
Static batching (batch of 4, lengths 20/40/60/400):

req A (20) ####----------------------------------  idle 380 steps, can't return
req B (40) ########------------------------------  idle 360 steps
req C (60) ############--------------------------  idle 340 steps
req D (400)######################################  the whole batch waits on this
           |------------ 400 decode steps ---------|

Continuous batching:

req A (20) ####  -> returns at step 20, slot reused
req E      ....####  (joined at step 20 in A's freed slot)
req B (40) ########  -> returns at step 40
req F      ........########  (joined at step 40)
req C ...  req D still going, but nobody waits on it to leave
```

Same GPU, same model, same batch *size* — the difference is that the batch *membership* churns every step, so the GPU is doing useful work for a finished-eligible sequence essentially all the time. That's the single biggest lever versus a "for req in requests: model.generate(req)" loop, where each request also monopolizes the GPU with a batch size of 1 and leaves most of the memory bandwidth idle.

**The metric it moves:** aggregate **throughput (tokens/sec)** goes up hugely under concurrency, and per-request latency stops depending on unlucky batch-mates. In Week 2 you'll see throughput climb steeply from concurrency 1 → 8 → 64 precisely because continuous batching is filling freed slots.

### PagedAttention (paged KV cache)

The KV cache is the memory of already-generated tokens: for every token in a sequence, every layer stores a key vector and a value vector so future tokens can attend to it. It grows by a fixed amount per token and it is the thing that actually limits how many sequences you can serve at once.

**The naive layout and why it's wasteful.** The obvious implementation reserves one contiguous buffer per sequence, sized to the *maximum* possible length. Say `max_model_len = 8192`. Every admitted sequence reserves 8192 tokens' worth of KV even if it only ever uses 300. That's **internal fragmentation** — reserved-but-unused space. Worse, when sequences finish and free their buffers, you're left with contiguous holes of odd sizes that a new 8192-slot request can't fit into even though the total free memory is plenty — **external fragmentation**. Real measurements from the vLLM authors found naive serving wasted a large majority of KV memory to these two effects.

**The OS analogy (this is the whole idea).** Operating systems solved exactly this for RAM decades ago: don't give a process one giant contiguous chunk, give it fixed-size *pages* and keep a page table mapping the process's logical addresses to scattered physical pages. PagedAttention does the same for the KV cache:

- VRAM for KV is carved into fixed-size **blocks** (a block holds the KV for a fixed number of tokens — commonly 16).
- Each sequence gets a **block table** mapping its logical token positions to physical blocks scattered anywhere in memory.
- A sequence allocates blocks **on demand** as it generates, not up front. A 300-token sequence uses ~19 blocks (300/16 ≈ 18.75) and not one block more.

Because blocks are uniform and allocated lazily, internal fragmentation shrinks to at most one partially-filled block per sequence (< 16 tokens wasted), and external fragmentation disappears — any free block fits any sequence. The practical payoff: you fit **far more concurrent sequences in the same VRAM**, which is the raw material continuous batching needs to keep the GPU fed.

```
Naive: one contiguous buffer per seq, sized to max_len
[ seq A used | reserved reserved reserved reserved ]   <- huge internal waste
[ seq B used | reserved reserved reserved ]

Paged: fixed-size blocks, allocated on demand, scattered
Physical blocks: [B3:A][B7:A][B1:B][B9:A][B2:B][B5:free][B8:free]...
  A's block table -> [B3, B7, B9, ...]
  B's block table -> [B1, B2, ...]
```

**Copy-on-write sharing.** Because a block is just an entry in a block table, two sequences can *point at the same physical block*. This is how parallel sampling and beam search share the prompt's KV: one copy, N pointers. When a sequence needs to diverge (write into a shared block), vLLM copies just that block first — copy-on-write, exactly like `fork()`. This same pointer-sharing machinery is what makes the next mechanism possible.

**The metric it moves:** **KV-block occupancy** (vLLM's `gpu_cache_usage_perc` / free-vs-used block count in the logs) and the max `--max-num-seqs` you can sustain without OOM. Watch the "# GPU blocks" line vLLM prints at startup — that number, times the block size, is your entire concurrency budget.

### Automatic prefix caching (`--enable-prefix-caching`)

Prefix caching notices that KV for a given token depends only on that token and everything before it. So if two requests begin with the *identical* token sequence, the KV blocks for that shared span are identical too — compute them once, reuse forever (until evicted).

**The block-hash mechanism.** vLLM hashes each KV block by its content: the hash of block *i* is a function of the tokens in block *i* **and the hash of block *i-1*** (a chained/rolling hash). So a block's identity encodes its entire prefix. When a new request arrives:

1. vLLM tokenizes it and walks its blocks, computing each block hash.
2. For each hash, it checks a table: is a block with this hash already cached?
3. Every leading block that hits is reused directly (its KV is already computed) — the request's prefill *skips* those tokens entirely.
4. The first block that misses (the point where the prompt diverges) and everything after it gets computed normally.

Because the hash is chained, reuse is a strict **prefix** match: you reuse from the start up to the first differing block, then stop. This is why placement matters enormously (see the gotcha below).

Worked example. Fat shared system prompt of 2000 tokens, block size 16 → 125 blocks. Two users send different questions but the same system prompt. First request: computes all 125 system-prompt blocks + its own question blocks. Second request: the 125 system-prompt blocks all *hit* → prefill skips ~2000 tokens and only computes the ~10–30 tokens of the new question. If prefill is your TTFT bottleneck, you just cut ~2000 tokens of prefill down to ~20 — a roughly two-orders-of-magnitude reduction in prefill work for that request.

**The metric it moves:** **TTFT (time to first token)** for requests with shared prefixes drops dramatically, because TTFT is dominated by prefill and you've deleted most of the prefill. vLLM exposes prefix-cache hit-rate metrics; a healthy chat workload with a big system prompt should show a high hit rate.

### Chunked prefill

Prefill (processing the prompt) and decode (generating tokens) compete for the same GPU. Prefill is one big parallel matmul over all prompt tokens; decode is a stream of tiny one-token steps. If you let a giant prefill run to completion before resuming decode, every already-streaming user's token flow **freezes** for the duration of that prefill.

Worked feel for it. A 30,000-token prompt arrives while 40 users are mid-stream. Prefilling 30k tokens in one shot might take, say, a few hundred milliseconds to over a second depending on GPU — and for that entire window, none of the 40 streaming users gets a token. Their **TPOT (time per output token)** spikes; the stream visibly stutters.

**Chunked prefill** splits that 30k-token prefill into chunks (e.g., 512 or 2048 tokens each) and **interleaves** them with the ongoing decode steps. Each scheduler iteration does a bit of the big prefill *plus* one decode step for everyone else. The whale still gets processed, just spread across many iterations, so no one else's stream stalls.

```
Without chunked prefill:
[......... 30k-token prefill (everyone frozen) .........][decode][decode]...

With chunked prefill (chunk = 2k):
[2k prefill + decode-all][2k prefill + decode-all][2k prefill + decode-all]...
   ^ big prompt advances a chunk    ^ streaming users still get a token each step
```

The tradeoff: the big prompt's *own* TTFT gets slightly worse (it's spread over more iterations instead of blasted through at once), and each iteration does a little more scheduling work. In exchange, decode latency stays smooth and *fair* across all requests. This is the right trade for a multi-tenant server, which is why **chunked prefill is on by default in recent vLLM** (controlled via `--enable-chunked-prefill` and the `--max-num-batched-tokens` budget that sets the chunk size).

**The metric it moves:** it stabilizes **TPOT / inter-token latency** for concurrent decoders (kills the stutter), at the cost of a small increase in the long prompt's TTFT.

## Worked example

Put all four together on one box. Say vLLM starts and logs `# GPU blocks: 4096`, block size 16 → capacity for 4096 × 16 = 65,536 KV tokens in flight at once.

A chat product sends every request a shared **1,500-token system prompt** + few-shot block, then a short user turn (~50 tokens), and generates ~150 tokens.

- **Without any of this (static batch of 8, no paging, no prefix cache):** each sequence reserves `max_model_len` worth of KV (say 8192 tokens), so 8 sequences alone reserve 65,536 tokens — you're already at capacity with just 8 requests, most of it reserved-and-unused. Every request re-prefills the full 1,550 tokens. Short requests wait on the 150-token max. Throughput is poor and TTFT is high.

- **With PagedAttention:** a request uses only what it touches — ~(1500+50+150)/16 ≈ 107 blocks ≈ 1,712 tokens, not 8,192. Now 65,536 / 1,712 ≈ **38 concurrent sequences** fit instead of 8. Continuous batching keeps all ~38 slots productive, evicting finished ones each step.

- **With prefix caching:** the 1,500-token system prefix is computed once and shared. Requests 2..N skip ~94 blocks of prefill (1500/16). TTFT for those requests collapses to "prefill 50 new tokens + first decode" instead of "prefill 1,550 tokens." On a fat-system-prompt chat workload this is often the difference between ~600ms and ~80ms TTFT (illustrative, not a benchmark — measure yours).

- **With chunked prefill:** when one user pastes a 25k-token document, the other ~37 streaming users don't stall — the big prefill is diced and interleaved, so their per-token latency stays flat.

Net: same GPU, ~5× the concurrency (8 → ~38), most requests skip ~94% of their prefill, and a single long prompt no longer freezes the fleet. That's where "5–20×" comes from — it stacks.

## How it shows up in production

- **Throughput scales with concurrency until KV runs out, then it cliffs.** As you raise load, throughput climbs (continuous batching filling slots) right up until KV blocks are exhausted. Past that, vLLM must **preempt**: it evicts a running sequence's KV (recompute or swap) to make room, which shows up as latency spikes and, in logs, preemption warnings. The fix is the Week-1 flag triad: lower `--max-model-len`, lower `--max-num-seqs`, or raise `--gpu-memory-utilization` (carefully).

- **Prefix caching is a TTFT superpower and a silent correctness non-issue.** It only reuses *identical* prefixes and the KV is mathematically the same, so it never changes outputs — it's pure speed. The place it "fails" is when your prompt structure defeats reuse (below).

- **The cache-buster gotcha (memorize this one).** Putting anything per-request at the **top** of the prompt destroys all downstream reuse. A `Current time: 2026-07-09T14:03:22Z` line, a `request_id`, a per-user UUID, or a randomized greeting at position 0 changes block 0's hash, which changes every chained hash after it → **zero prefix hits**, full prefill every time, TTFT back to slow. Rule: **stable content first (system prompt, few-shot, shared RAG context), volatile content last.** Move the timestamp to the end or drop it. Teams routinely "lose" prefix caching this way and blame vLLM.

- **Chunked prefill trades a little long-prompt TTFT for fleet-wide fairness.** If you serve a latency-sensitive single-user workload with huge prompts and *no* concurrency, chunking can look like it slightly hurt your one TTFT number. For multi-tenant serving it's almost always correct. Tune the chunk size via `--max-num-batched-tokens`.

- **What to watch in `/metrics` (Prometheus, on by default).** `vllm:gpu_cache_usage_perc` (KV-block occupancy — your headroom), `vllm:num_requests_running` vs `vllm:num_requests_waiting` (queueing → TTFT pressure), prefix-cache hit rate (is your shared prompt actually being reused?), and TTFT/TPOT histograms. In Week 2's load tests these are exactly the panels you'll stare at.

## Common misconceptions & failure modes

- **"Continuous batching means bigger batches, so latency goes up."** No — it means the batch *membership* changes per step. Individual requests generally return *sooner* because they're no longer blocked by long batch-mates. Throughput and per-request latency both improve versus static batching.

- **"PagedAttention makes the model faster."** It doesn't speed up the math; it lets you fit more sequences in VRAM, which lets continuous batching keep the GPU busier. The speed comes from higher *occupancy*, not faster kernels.

- **"Prefix caching changes my outputs / it's a quality risk."** It reuses the exact same KV that would have been computed; outputs are identical. It's a pure latency/throughput optimization. (RadixAttention in SGLang is a more general form of the same idea — tree-structured prefix reuse.)

- **"I turned on prefix caching but TTFT didn't improve."** Almost always the cache-buster: a per-request token near the top of the prompt, or prompts that simply don't share a prefix. Check the hit-rate metric before assuming the flag is broken.

- **"Chunked prefill is a niche flag."** It's default in recent vLLM and it's why one user's 30k-token paste doesn't tank everyone's stream. If you *disable* it to shave a single prompt's TTFT, understand you're re-introducing prefill stalls under concurrency.

- **KV-cache OOM at startup vs mid-load.** Startup OOM = your `--max-model-len × --max-num-seqs` demands more KV than the card has after weights; vLLM tells you the block count it could allocate. Mid-load preemption = you oversubscribed and the scheduler is evicting; you'll see it as latency spikes, not a crash. Different symptoms, same lever set.

## Rules of thumb / cheat sheet

- **The two-sentence pitch:** continuous batching = churn the batch every decode step so no request waits on the slowest; PagedAttention = OS-style KV pages so you pack many more sequences into the same VRAM. Prefix caching + chunked prefill make shared and long prompts cheap.
- **Stable-first prompt order.** System prompt → few-shot → shared RAG context → *then* per-request/volatile stuff. Never put a timestamp/UUID at position 0 if you want prefix reuse.
- **`--enable-prefix-caching`** for any workload with a fat shared system prompt or shared RAG context (chat, agents, RAG). It's low-risk, high-TTFT-reward. (Default-on in recent vLLM builds; set it explicitly to be sure.)
- **Chunked prefill:** leave it on for multi-tenant serving. Tune chunk size via `--max-num-batched-tokens` if long-prompt TTFT vs decode fairness needs balancing.
- **Concurrency budget ≈ (# GPU blocks × block size) ÷ (avg tokens per sequence).** Read `# GPU blocks` from the startup log; that's your ceiling.
- **KV-OOM triage order:** lower `--max-model-len` first (biggest lever, capped by your actual context needs), then `--max-num-seqs`, then nudge `--gpu-memory-utilization` up (last resort — leave headroom for CUDA graphs / other procs).
- **Metric → mechanism map:** throughput ↔ continuous batching; KV-block occupancy ↔ PagedAttention; TTFT ↔ prefix caching (and prefill); TPOT/inter-token smoothness ↔ chunked prefill.
- All specific latency figures above are *illustrative* — measure your own in Week 2.

## Connect to the lab

Week 1 Step 3's "prefix caching win" experiment (50 requests sharing a 1,000-token system prompt, `--enable-prefix-caching` on vs off) is this lecture's prefix-caching section made real — watch the TTFT gap in the logs and confirm the hit-rate metric. The KV-cache OOM experiment is the PagedAttention concurrency budget hitting its wall; you'll pull exactly the flag levers from the cheat sheet. Then in Week 2's k6 load tests, the throughput curve you plot across concurrency 1/8/64 *is* continuous batching filling freed slots, and `gpu_cache_usage_perc` in `/metrics` is your PagedAttention occupancy gauge.

## Going deeper (optional)

- **vLLM official docs** (docs.vllm.ai) — read the "Automatic Prefix Caching", "Optimization and Tuning", and "Engine Arguments" pages. The canonical, current source for flags and defaults.
- **vLLM GitHub repo** (github.com/vllm-project/vllm) — the design docs and issues explain scheduler and chunked-prefill behavior; search the repo for "chunked prefill" and "prefix caching".
- **PagedAttention paper** — "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023, the vLLM paper). Skim the intro and the figures for the fragmentation numbers; skip the proofs. Search that title.
- **SGLang / RadixAttention** (github.com/sgl-project/sglang) — the tree-structured generalization of prefix caching; useful contrast for agent/structured workloads.
- **Horace He, "Making Deep Learning Go Brrrr From First Principles"** — the memory-bound vs compute-bound mental model that underpins why decode batching is "almost free." Search that title.
- Search queries that stay current: "vLLM continuous batching explained", "vLLM automatic prefix caching hash", "vLLM chunked prefill max-num-batched-tokens", "orca iteration-level scheduling" (the paper that introduced continuous batching).

## Check yourself

1. In the static-batch example (lengths 20/40/60/400), why does the 20-token request finish generating at step 20 but not *return* until step 400, and what is that phenomenon called?
2. PagedAttention doesn't make any matrix multiply faster. So where does its contribution to the 5–20× actually come from?
3. A colleague adds `Request received at {iso_timestamp}` as the first line of every system prompt "for logging." Your prefix-cache hit rate drops to ~0 and TTFT regresses. Explain the mechanism and the fix.
4. You serve a multi-tenant chat API. One user pastes a 40k-token document. With chunked prefill *off*, what do the other 30 streaming users experience, and which metric captures it?
5. vLLM logs `# GPU blocks: 2048`, block size 16, and your average sequence uses ~1,000 tokens of KV. Roughly how many concurrent sequences can you sustain, and which flag would you lower first if you OOM?
6. Why is it correct to say prefix caching never changes model outputs, whereas quantization might?

### Answer key

1. It finished *generating* at step 20, but under static batching the batch is returned as a unit only when the *slowest* member (the 400-token request) completes; the short request's slot and result are held the whole time. This is **head-of-line blocking** — and it's exactly what continuous batching eliminates by evicting finished sequences every step.
2. From **occupancy**. By eliminating internal and external KV fragmentation, PagedAttention fits many more concurrent sequences in the same VRAM. More resident sequences = more work continuous batching can pack into each GPU step = higher throughput. The kernels aren't faster; the GPU is just kept fuller.
3. Block hashes are chained (each block's hash depends on the previous block's hash), so reuse is a strict prefix match. A per-request timestamp at position 0 changes block 0's hash, which changes every subsequent block hash → no blocks hit → full prefill every request → TTFT regresses. Fix: move volatile content (timestamp/UUID/request-id) to the *end* of the prompt, keep the stable system prompt and few-shot at the top.
4. Without chunked prefill, the 40k-token prefill runs as one blocking operation; for its entire duration none of the 30 streaming users receives a token, so their stream stalls and stutters. The metric is **TPOT / inter-token latency** (it spikes), even though the big prompt's own TTFT might look fine.
5. Capacity ≈ 2048 × 16 = 32,768 KV tokens; at ~1,000 tokens/sequence that's ~**32 concurrent sequences**. If you OOM, lower **`--max-model-len`** first (it directly shrinks the per-sequence KV ceiling and is usually the biggest lever), then `--max-num-seqs`, then consider nudging `--gpu-memory-utilization`.
6. Prefix caching reuses the *exact same* KV values that would have been recomputed from the identical tokens — the arithmetic is bit-for-bit the same path, so outputs are unchanged; it only saves time. Quantization changes the numeric precision of the weights/activations themselves, which can shift outputs, so it needs a quality re-eval; prefix caching does not.
