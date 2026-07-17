# Lecture 7: Memory-bound decode and reading the GPU — nvidia-smi and nvitop

> There is a moment, the first time you load-test your own vLLM server, when you glance at the GPU monitor mid-generation and see "GPU utilization: 34%." Your gut says *the card is bored, I'm wasting money, something is misconfigured*. That gut is wrong, and acting on it — throwing hardware at the problem, panicking about a bug — is one of the most common and expensive mistakes in inference ops. This lecture rewires that instinct. You'll learn *why* generating tokens is limited by how fast the GPU can read from its own memory rather than by how fast it can compute, why that makes a low utilization number look scary but be perfectly healthy, and — the payoff — why batching more users onto a decode-bound server is nearly free throughput, which is the entire reason continuous batching exists. Then you'll learn to *read the instruments*: `nvidia-smi` for a snapshot and `nvitop` for a live dashboard, and a small diagnostic table that turns two numbers (memory used, SM utilization) into a correct one-word label — *memory-bound, under-utilized,* or *KV-pressured* — so that during your load test you say the right thing about your server.

**Prerequisites:** what a KV cache is and why decode is one-token-at-a-time (Lecture 2); the VRAM equation — weights vs KV cache (Lecture 6); you've launched vLLM at least once (Lecture 1); comfort with GB and GB/s arithmetic. · **Reading time:** ~30 min · **Part of:** Phase 10 (LLMOps: Serving, Optimization & Deployment) Week 2

---

## The core idea (plain language)

A GPU does two very different things when it serves an LLM, and they have opposite bottlenecks.

**Prefill** is the phase where the model reads your whole prompt at once. If your prompt is 2,000 tokens, the GPU shoves all 2,000 through every layer in big, dense matrix multiplications. Lots of tokens, lots of math, all the arithmetic units light up. Prefill is **compute-bound** — the limiting resource is FLOPs, and the GPU's SM (streaming multiprocessor) utilization sits near 100%.

**Decode** is the phase where the model generates the answer, one token at a time. To produce token N+1, it has to read *the entire model's weights* out of VRAM — every layer, every matrix — and multiply them against a *single* token's worth of activation. Then it does it again for token N+2. And again. The math per step is tiny (one token), but the amount of memory it must *read* per step is enormous (all the weights, plus the whole KV cache). Decode is **memory-bandwidth-bound** — the limiting resource is GB/s of VRAM read speed, not FLOPs.

Here is the mental model, due to Horace He's "Making Deep Learning Go Brrrr From First Principles": every operation is either *compute-bound* (waiting on the math units) or *memory-bound* (waiting on data to arrive from memory). You figure out which by asking a simple question — **how much math am I doing per byte I read?** That ratio is called arithmetic intensity. Prefill does a lot of math per byte (high intensity → compute-bound). Decode does almost no math per byte, because it reads a giant weight matrix just to multiply it by one skinny vector (low intensity → memory-bound). You don't need the roofline math to use this; you need the reflex: *"Am I FLOP-limited or bandwidth-limited right now?"*

Three consequences fall out of this, and internalizing them is the whole point of the lecture:

1. **Low GPU utilization during decode is normal, not a bug.** The SMs are genuinely idle a lot of the time, waiting for weights to stream in from VRAM. On a *single* request you cannot go faster — the card is already reading memory as fast as it physically can.
2. **Bigger batches during decode are almost free.** When you read a weight matrix to serve one sequence, you can multiply it against *64* sequences' vectors for nearly the same memory cost. The expensive part (reading the weights) is shared; the cheap part (a little more math) is what grows. This is precisely why **continuous batching** wins so hard.
3. **Memory bandwidth (GB/s), not peak TFLOPs, predicts your decode throughput.** When you pick a GPU for serving, the bandwidth spec on the datasheet is the number that matters, and it reframes which cards are actually good value.

## How it actually works (mechanism, from first principles)

### Why decode reads the whole model for every single token

Start with the shape of the computation. A transformer is a stack of layers, and each layer is mostly big weight matrices: the attention projections (Q, K, V, output) and the MLP (two or three fat matrices). For a 7B model in fp16, all of those weights together are about **14 GB**.

During **prefill** of a 2,000-token prompt, the GPU loads a weight matrix once and multiplies it against all 2,000 tokens stacked into a matrix. That's a matrix × matrix multiply — dense, big, exactly what GPUs are built for. The weight was read once and reused across 2,000 tokens' worth of arithmetic. High math-per-byte.

During **decode**, you generate one token, then feed it back to generate the next. Each step is a matrix × *vector* multiply: the same 14 GB of weights, but now multiplied against a *single* token's activation vector. The weight matrix is read from VRAM in full, used for a tiny amount of math, and discarded. Next step: read all 14 GB again. Low math-per-byte.

So the floor on how fast you can decode on one stream is set by a memory-read budget, not a compute budget:

```
time per decode step  ≈  bytes read per step  /  memory bandwidth
```

### A worked bandwidth calculation (single stream)

Take a concrete card. An **NVIDIA A10** has roughly **600 GB/s** of memory bandwidth (approximate — check the datasheet for your exact SKU). Serve a 7B model in fp16, ~14 GB of weights.

Per decode step, ignore the KV cache for a moment and just count weights:

```
14 GB read  /  600 GB/s  ≈  0.0233 s per token  ≈  23 ms/token
→  ~43 tokens/sec  (single stream, theoretical ceiling)
```

That's a *ceiling*; real numbers are lower because you also read the KV cache each step and there's overhead. But notice what it tells you: at batch size 1, **you are limited by 14 GB ÷ bandwidth, and no amount of spare compute helps.** The A10 has plenty of idle FLOPs during this — that's exactly why its SM utilization reads low.

Now do the same on an **H100** (~3.35 TB/s ≈ 3,350 GB/s, approximate):

```
14 GB / 3,350 GB/s  ≈  4.2 ms/token  →  ~240 tokens/sec (single stream ceiling)
```

The H100 is ~5–6× faster per token than the A10 here — and if you look up the two cards, that ratio tracks their **bandwidth** ratio far more closely than their peak-FLOPs ratio. That's the "bandwidth predicts decode throughput" claim made concrete.

### Why bigger batches are almost free

Here's the lever. When you batch B sequences together in decode, each step still reads the weights **once** (they're shared across the batch), but now does B tokens' worth of math against them:

```
Batch 1:  read 14 GB, do 1 token of math    → ~23 ms →  1 token  →  ~43 tok/s
Batch 8:  read 14 GB, do 8 tokens of math    → ~24 ms →  8 tokens →  ~330 tok/s
Batch 32: read 14 GB, do 32 tokens of math   → ~28 ms →  32 tokens→  ~1140 tok/s
```

The per-step time barely moved (the extra math is cheap and the memory read is unchanged), but you got 8×, then ~26× the *throughput*. You're amortizing the same expensive weight-read across more customers. That is the whole magic trick of batched decode — and the reason continuous batching (packing every free slot with a waiting sequence, every step) is the single biggest throughput lever in serving.

It's "almost" free, not free, for two reasons. First, larger batches read **more KV cache** per step (each sequence has its own KV), so eventually the KV reads, not the weight reads, dominate and the free lunch ends. Second, at some batch size you cross back into **compute-bound** territory — you're finally doing enough math per byte that the arithmetic units become the bottleneck. That crossover point is exactly where "add more sequences" stops helping. Below it (the usual regime for a single GPU serving a big model), batching is nearly free.

```
throughput
   ▲
   │                         ┌───────── compute-bound: adding
   │                    ┌────┘           sequences stops helping
   │               ┌────┘
   │          ┌────┘   ← memory-bound: batching is
   │     ┌────┘          nearly free, throughput climbs fast
   │┌────┘
   └┴──────────────────────────────────►  batch size (concurrent seqs)
```

### The KV cache is the other thing being read

Decode reads two things from VRAM every step: the **weights** (shared across the batch) and the **KV cache** (private per sequence, and it *grows* as the sequence gets longer). Early in a generation the KV is small and weights dominate the read. But with long contexts and big batches, the KV cache can become the larger read — and since it's *not* shared across sequences, it doesn't amortize the way weights do. This is why a server humming along at batch 64 with short prompts can slow down and back up when the same batch is running 20k-token contexts: you've shifted the bottleneck from shared weight reads to un-shared KV reads. (The VRAM *capacity* side of this was Lecture 6; here it's the *bandwidth* side of the same coin.)

### What the two numbers on your monitor actually mean

The instruments report two things you care about:

- **Memory used** — how full VRAM is. Weights + KV cache + activations + overhead (Lecture 6). Mostly a *capacity* signal.
- **GPU / SM utilization (%)** — the fraction of recent time the SMs had *at least one* thing to compute. This is a coarse "was the engine turning over" gauge, **not** a measure of how much of the card's compute you're using.

That second definition is the trap. `nvidia-smi`'s "GPU-Util" says *"was the GPU doing anything?"*, not *"was the GPU busy doing math?"* During memory-bound decode the SMs fire briefly, then stall waiting for the next chunk of weights to arrive, then fire again. Averaged over a second that can read as 30–60% — the card *is* idle much of the time, but idle *waiting on memory*, which you cannot fix with more compute. So a low utilization number during generation is the expected symptom of a healthy memory-bound server, not a diagnosis of waste.

## Worked example

You launch `Qwen2.5-7B-Instruct` in fp16 on a rented A10 (24 GB) and run the Week-2 k6 load test at concurrency 1, 8, and 64. In a second terminal you keep `nvitop` open. Here's what you see and how you label it (throughput numbers are illustrative, matching the ceilings computed above):

| Concurrency | `nvitop` memory | SM util | tokens/sec (aggregate) | p95 TTFT | Label |
|---|---|---|---|---|---|
| 1 | 15 GB / 24 GB | ~30% | ~40 | 90 ms | **Memory-bound, single stream.** Card is fine; you literally can't go faster on one request. |
| 8 | 17 GB / 24 GB | ~45% | ~310 | 120 ms | **Memory-bound, batching working.** ~8× throughput for near-zero extra latency — the free lunch. |
| 64 | 23 GB / 24 GB | ~70% | ~1,050 | 480 ms | **Memory-bound, near KV capacity.** Throughput still climbing, but memory nearly full and TTFT rising — approaching the wall. |

The story the table tells: from concurrency 1 → 8, aggregate throughput roughly 8×'d while latency barely moved. That's the "batching is almost free" claim, live. From 8 → 64 you kept gaining throughput but memory filled and TTFT climbed — you're now trading tail latency for throughput and getting close to KV-cache pressure. SM utilization rose the whole way (more math per weight-read) but never pinned at 100%, because you never left the memory-bound regime.

Now the counterfactual that teaches the diagnostic. Suppose at concurrency 64 you instead saw **memory: 15 GB / 24 GB, SM util 20%, throughput flat at ~250 tok/s.** That's a *different* label: you've got tons of free VRAM and idle compute and you're *not* scaling with load — you're under-utilizing the card. The fix isn't a bigger GPU; it's raising `--max-num-seqs` (the server is capping concurrency below what the card can hold) or checking that your load generator is actually sending 64 concurrent requests.

## How it shows up in production

- **Someone opens a "GPU is only 40% utilized, we're wasting money" ticket.** The instinct is to downsize the card or cram more models on. If the 40% is *decode-time* utilization on a single-stream or lightly-batched workload, that's memory-bound and expected — the right move is to *raise concurrency* (more `--max-num-seqs`, more offered load) so you amortize weight reads, not to change hardware. You will field this ticket. Knowing the answer cold is worth real money.
- **GPU selection gets reframed.** Two cards with similar peak TFLOPs but very different memory bandwidth will give very different decode throughput — the higher-bandwidth card wins for token generation. When you compare an A10 vs an L4 vs an A100 vs an H100 for *serving*, sort by **GB/s** first. (Prefill-heavy or huge-batch workloads shift back toward compute; but the default serving bottleneck is decode bandwidth.)
- **Continuous batching's payoff becomes legible.** When your throughput jumps 8–20× going from a naive request-per-call loop to vLLM at load, this is *why*: you moved from batch-1 weight reads to amortized batch-N weight reads. If someone asks "why is vLLM so much faster," the honest one-sentence answer is "it keeps the decode batch full so the weight reads are shared."
- **Latency vs throughput tension.** Cranking concurrency to max throughput fills the KV cache and lengthens the decode batch, which *raises* per-token latency and TTFT (queueing). The memory-bound model tells you these pull against each other, so you must report and SLO **both** (Lecture 8), not just tokens/sec.
- **The "buy a bigger GPU won't help" conversation.** If you're memory-bound on a single stream and latency-sensitive (batch can't grow — e.g., one interactive user), the only lever that helps per-token latency is *more bandwidth* (a faster card) or *fewer bytes to read* (quantize the weights to int4/fp8, cutting the per-step read roughly in half or more). More SMs do nothing. This is a frequent, expensive misunderstanding in capacity planning.

## Common misconceptions & failure modes

- **"Low GPU-Util means the card is idle/wasted."** GPU-Util means "the SMs had *something* to do recently," not "the compute is saturated." Memory-bound decode legitimately shows low util while running flat-out against the memory bus. Don't downsize on this signal alone.
- **"Decode is slow because the GPU isn't powerful enough (needs more FLOPs)."** On a single stream, decode is bandwidth-limited. A card with 2× the FLOPs but the same bandwidth won't decode faster. Look at GB/s.
- **"Bigger batches cost proportionally more time."** In the memory-bound regime they cost *almost nothing* extra per step, because the dominant cost (weight reads) is shared. This inverts the intuition from CPU-bound code.
- **"`nvidia-smi` GPU-Util is like a CPU load percentage."** It isn't a utilization-of-capacity metric. For real per-SM occupancy and memory-throughput pressure you need finer tools (`nvitop`, DCGM, Nsight). Treat `nvidia-smi` util as a coarse on/off-ish gauge.
- **Reading the wrong process.** On a shared box, `nvidia-smi`'s memory total is everyone's. Always confirm *which PID* owns the VRAM (your vLLM process) before concluding "the model uses 22 GB" — a stray notebook or a zombie CUDA process may own some of it.
- **Chasing 100% util as a goal.** For serving, 100% SM util usually means you've become compute-bound — often at a batch size where latency is already bad. The healthy operating point for decode is *memory-bound with util well under 100% and throughput scaling with load*, not a pinned utilization meter.
- **Confusing "memory used" (capacity) with "memory bandwidth" (throughput).** VRAM can be only half full (capacity fine) while the memory *bus* is saturated (bandwidth maxed). They're different axes; decode stresses the bus.

## Rules of thumb / cheat sheet

**The diagnostic table (memorize this).** During *generation*:

| Memory used | SM util | Diagnosis | Action |
|---|---|---|---|
| **High** | **Low** | Memory-bound — expected/healthy | Nothing wrong. To gain throughput, raise concurrency (more offered load / `--max-num-seqs`) if memory allows. |
| **Low** | **Low** | Under-utilized — wasting the card | Raise `--max-num-seqs`; send more concurrent load; check the load generator; consider a smaller/cheaper GPU. |
| **High** | Rising, latency climbing | KV pressure / queueing | You're at capacity: TTFT/TPOT rising, near OOM. Lower `--max-num-seqs` or `--max-model-len`, add a replica, or shard. |
| **Low/med** | **~100%** | Compute-bound (rare for decode; common in prefill or huge batches) | More compute/FLOPs would help; batching won't. Often means big prefills dominating. |

**Reflexes:**
- Decode = **bandwidth-bound**. Prefill = **compute-bound**. Ask "FLOP-limited or bandwidth-limited?" before optimizing.
- For a single stream: `tokens/sec ≈ memory_bandwidth / bytes_read_per_step`. Bytes read ≈ weight bytes (+ KV). Sanity-check your server against this ceiling.
- Selecting a serving GPU? Sort candidates by **GB/s memory bandwidth** first, VRAM capacity second, TFLOPs last.
- To cut single-stream decode latency: **quantize** (fewer bytes/step) or get **more bandwidth** (faster card). More SMs won't help.
- To cut cost/throughput: **raise the batch** (continuous batching) until KV pressure or the compute-bound knee stops you.
- Snapshot → `nvidia-smi`. Live dashboard → `nvitop`. Always check *which PID* owns the memory on a shared box.

**Reading `nvidia-smi` (the fields that matter):**

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...        Driver: ...        CUDA Version: 12.x                  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A10          Off  | 00000000:...     Off |                  Off |
|  0%   58C    P0   120W / 150W |  15342MiB / 23028MiB |     34%      Default |  ← Memory-Usage + GPU-Util
+-------------------------------+----------------------+----------------------+
| Processes:                                                                  |
|  GPU   PID   Type   Process name                             GPU Memory     |
|    0  4821    C     .../python (vllm)                         15106MiB      |  ← which PID owns VRAM
+-----------------------------------------------------------------------------+
```

- **Memory-Usage** `15342MiB / 23028MiB` → capacity signal.
- **GPU-Util** `34%` → coarse "was it doing anything"; low during memory-bound decode is normal.
- **Processes** → confirm your vLLM PID owns the memory before drawing conclusions.
- `nvidia-smi -l 1` refreshes every second; `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1` for scriptable/loggable output.

**`nvitop` (live dashboard):** `pip install nvitop`, then run `nvitop`. Gives you per-process VRAM, live SM-util and memory-util *sparklines*, power, temp, and a process tree — far better than raw `nvidia-smi` for watching a load test. Watch **memory-util%** (a proxy for bandwidth pressure) alongside **GPU-util%** (SM activity): memory-bound decode shows memory-util high and GPU-util moderate.

## Connect to the lab

This is the theory behind Week-2 Lab **Step 2** ("keep `nvitop` visible during load tests") and **Step 3** (the k6 sweep at concurrency 1/8/64). As you run each tier, glance at `nvitop` and write the label from the diagnostic table in your notes — *memory-bound, under-utilized,* or *KV-pressured* — for that tier. The Definition of Done explicitly requires: *"You can point at `nvitop` during a run and correctly label the server memory-bound vs under-utilized."* This lecture is what lets you do that without guessing. It also feeds Step 4: knowing decode is bandwidth-bound is why your best throughput comes from the *highest* concurrency tier that hasn't hit KV pressure — that's the tokens/sec you plug into the cost-per-1M math.

## Going deeper (optional)

- **Horace He — "Making Deep Learning Go Brrrr From First Principles."** The canonical, engineering-first explanation of compute-bound vs memory-bound and arithmetic intensity, without roofline ceremony. Search: `Horace He Making Deep Learning Go Brrrr first principles`. (Published on horace.io.)
- **vLLM docs — performance & optimization.** Root: `docs.vllm.ai`. Search: `vLLM performance tuning docs`, `vLLM continuous batching`. Explains how the scheduler keeps the decode batch full.
- **NVIDIA GPU datasheets** for the cards you'll rent (A10, L4, A100, H100/H200). Look specifically at **memory bandwidth (GB/s)** and VRAM. Search: `NVIDIA A10 datasheet memory bandwidth`, `NVIDIA H100 datasheet bandwidth`.
- **`nvitop`** — GitHub repo `XuehaiPan/nvitop`. README documents the live panes and the interactive process view. Search: `nvitop github`.
- **NVIDIA `nvidia-smi` documentation** and **DCGM** for production GPU telemetry (Prometheus exporter). Search: `nvidia-smi query-gpu documentation`, `NVIDIA DCGM exporter Prometheus`.
- **Nsight Systems / Nsight Compute** if you ever need to *prove* an op is memory-bound at the kernel level (well beyond serving ops, but good to know it exists). Search: `NVIDIA Nsight Systems memory bound`.
- **The roofline model** — the formal version of the intuition, for later (optional; not needed for this phase). Search: `roofline model arithmetic intensity`.

## Check yourself

1. In two sentences, why is decode memory-bandwidth-bound while prefill is compute-bound? Use the phrase "math per byte."
2. Your monitoring shows 95% VRAM used but only 15% GPU-Util during generation. Is this a bug or expected, and what (if anything) should you change?
3. Explain, in terms of what gets *read* from VRAM per step, why raising the decode batch from 1 to 32 is "almost free" for throughput. What eventually ends the free lunch?
4. You must cut per-token latency for a *single* interactive user on a 7B fp16 model and the batch cannot grow. Which of these helps and which don't: (a) a GPU with 2× the TFLOPs, (b) a GPU with 2× the bandwidth, (c) int4-quantizing the weights, (d) raising `--max-num-seqs`?
5. You're choosing between two GPUs for a token-generation-heavy service. One has higher peak TFLOPs, the other higher memory bandwidth (same VRAM). Which do you lean toward for decode throughput, and why?
6. On `nvitop` you see low memory used, low GPU-util, and throughput that doesn't rise when you add load. What's the label and the fix?

### Answer key

1. Prefill multiplies a weight matrix against *many* prompt tokens at once, so it does a lot of math per byte read (high arithmetic intensity → the compute units are the bottleneck). Decode multiplies the same weights against *one* token per step, doing almost no math per byte read (low intensity → the limit is how fast VRAM can be read, i.e., bandwidth).
2. **Expected**, not a bug — that's the signature of a healthy memory-bound decode (SMs stalling on memory reads, so GPU-Util reads low). Don't downsize the card on this signal. If you want more *throughput*, raise concurrency / offered load (and `--max-num-seqs` if it's capping you) so weight reads amortize across more sequences; there's no per-request speedup to be had here.
3. Each decode step reads the full weight matrices from VRAM **once**, and those reads are *shared* across every sequence in the batch — so serving 32 sequences reads the same ~14 GB as serving 1, while doing 32× the (cheap) math. The dominant cost is unchanged, so throughput scales almost linearly. The free lunch ends when (a) per-sequence **KV-cache reads** (which are *not* shared and grow with context) start to dominate the memory traffic, or (b) the batch gets large enough that you become **compute-bound**.
4. Helps: **(b)** more bandwidth (directly cuts bytes-read time per step) and **(c)** int4 quantization (fewer bytes to read per step, roughly halving or better). No help: **(a)** more TFLOPs (you're not compute-bound on a single stream) and **(d)** raising `--max-num-seqs` (that grows batch/throughput, not single-stream latency, and the batch can't grow here anyway).
5. Lean toward the **higher memory bandwidth** card. Decode throughput on a single stream is set by `bandwidth / bytes_read_per_step`, so GB/s tracks tokens/sec far better than peak TFLOPs does; the extra FLOPs sit idle in the memory-bound regime.
6. Label: **under-utilized** (you're wasting the card). Fix: raise `--max-num-seqs` in case the server is capping concurrency below what the card can hold, and verify your load generator is actually sending concurrent requests; a smaller/cheaper GPU may also be the right call if the workload genuinely can't fill this one.
