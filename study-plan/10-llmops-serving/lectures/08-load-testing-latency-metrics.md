# Lecture 8: Load Testing and the Latency Metrics That Matter — TTFT, TPOT, Throughput, Percentiles

> You have a vLLM endpoint that answers `curl`. That proves it *works*; it tells you nothing about whether it works *under load*, how it *feels* to a user typing at it, or what it will *cost* you at 200 requests per second. This lecture is about measuring a serving endpoint honestly — the four numbers that actually describe an LLM's runtime behavior (TTFT, TPOT, end-to-end latency, throughput), why you must report them at percentiles instead of averages, why throughput and latency are in permanent tension, and how to run a load test that isn't quietly lying to you. After it you can point k6 or locust at your own server, produce a per-concurrency-tier table of tokens/sec, req/s, and TTFT/TPOT at p50/p95/p99, cross-check it against vLLM's `/metrics`, and defend every cell — including why cranking concurrency made your throughput chart look glorious and your users feel terrible.

**Prerequisites:** Week 1 (a running vLLM OpenAI-compatible endpoint), Lecture 7 (memory-bound decode: prefill vs decode), comfort reading JSON over HTTP, arithmetic and basic percentiles · **Reading time:** ~28 min · **Part of:** Phase 10 (LLMOps) Week 2

---

## The core idea (plain language)

An LLM response is not one event; it is a stream that unfolds over time. That single fact is why you cannot describe its performance with one number. When you `curl` a chat endpoint with streaming on, here is what actually happens on the wire:

1. You send the request. It may sit in a **queue** if the server is busy.
2. The server runs **prefill** — it reads your entire prompt and builds the KV cache.
3. The **first token** comes back. The user sees text start to appear.
4. The server runs **decode**, emitting one token at a time, each a little apart.
5. The **last token** arrives; the stream closes.

Two very different things determine how this *feels*. The gap from "I hit send" to "text starts appearing" is one experience — it's the "is this thing even alive?" moment. The rhythm of tokens after that — smooth and fast, or stuttering — is a completely different experience. A response that starts in 200 ms and then dribbles out over 15 seconds feels sluggish. A response that takes 2 seconds to start but then floods the screen feels snappy once it gets going. **One latency number cannot capture both, so we use two.**

- **TTFT — Time To First Token.** From request sent to first token received. This is *perceived responsiveness*. It is dominated by **queueing + prefill**. When the server is idle, TTFT is basically prefill time. When it's loaded, TTFT is mostly *waiting in line*.
- **TPOT — Time Per Output Token** (also called **ITL**, Inter-Token Latency). The average gap between successive tokens during decode. This is *streaming smoothness*. If your reading/generation speed target is "faster than a human reads" (~5–10 tokens/sec is readable; 30+ feels instant), TPOT is the metric.

From those two plus the token counts you can reconstruct everything else:

- **End-to-end (E2E) latency** = TTFT + (output_tokens − 1) × TPOT. The total wall-clock a request took. This is what a *non-streaming* client measures, and it's what matters for batch/agent workloads where nobody's watching tokens appear.
- **Throughput** = how much work the *system* does per unit time. Two flavors, and confusing them causes real damage: **tokens/sec** (system-wide generation rate — the capacity/cost number) and **requests/sec** (RPS — completed requests per second — the QPS number). These are system-level; do not confuse them with a single request's *per-request* token rate (1/TPOT).

The whole lecture is: define these precisely, measure them without fooling yourself, and always report them **together with percentiles**, because the interesting pain lives in the tail and in the trade-off between them.

---

## How it actually works (mechanism, from first principles)

### The anatomy of one streamed request

Here is a single request on a timeline. Time flows left to right; each `t` is a token arriving.

```
send                                                        stream closed
 │                                                                │
 ▼                                                                ▼
 ├──── queue ────┬──── prefill ────┬── t1  t2  t3  ...  tN ──┤
 │               │                 │   │                          
 │◄────────── TTFT ───────────────►│◄──── decode phase ─────►│
 │                                     │◄─►│                       
 │                                    TPOT (gap between tokens)    
 │◄──────────────────── E2E latency ───────────────────────►│
```

Read it carefully, because every measurement mistake is a mistake about *which segment you timed*:

- **TTFT** spans the queue **and** prefill. On an idle server there's no queue, so TTFT ≈ prefill time, which scales with **prompt length** (more input tokens = more prefill work). On a loaded server, the queue dominates and TTFT balloons even though prefill is unchanged. This is why TTFT is the metric that "explodes" under concurrency.
- **TPOT** is the steady-state cadence of decode. Because decode is memory-bandwidth-bound (Lecture 7), TPOT barely moves as you add concurrent requests *until the batch saturates memory bandwidth* — then it degrades for everyone at once. TPOT does **not** include the first token's arrival; it's the *inter*-token gap.
- **E2E** is the sum. For a 500-token answer with 40 ms TPOT, decode alone is ~20 seconds — so E2E is dominated by output length, not by TTFT. For a 10-token classification answer, TTFT dominates E2E. **The metric that matters depends on the workload.**

### Why you report percentiles, not averages

Averages lie about latency because latency distributions are **right-skewed with a long tail**. A handful of requests — the ones that hit a full queue, a garbage-collection pause, a cold KV-cache block, a 3,000-token prompt — take far longer than typical. The mean gets dragged toward those outliers *and* hides them at the same time.

A percentile p*N* is "N% of requests were at least this fast." p50 (median) is the typical experience; p95 is "the slowest 1 in 20"; p99 is "the slowest 1 in 100." Concretely, suppose you collect these 20 TTFT samples (ms):

```
120 130 130 140 140 150 150 150 160 160
170 170 180 190 200 210 240 280 900 1500
```

- **Mean** ≈ 273 ms. It looks like a quarter-second service. No single user experienced "273 ms" — it sits in a gap between the bulk (~150) and the outliers (900, 1500).
- **p50** = 165 ms. The honest typical.
- **p95** ≈ 900 ms. 1 in 20 users waited nearly a second.
- **p99** ≈ 1500 ms. 1 in 100 waited a second and a half.

Now the production reality: at 100 RPS, **p99 is one user every second**, and at scale each user issues many requests per session, so a p99 tail hits *most* sessions at least once. "p50 is fine" is how teams ship endpoints that feel broken to real users. **p50 hides tail pain; p95/p99 is where SLOs live.** The rule you will repeat forever: report p50 *and* p95 *and* p99, never the mean alone.

> A subtlety that bites: you cannot average percentiles. If tier A has p99 = 800 ms and tier B has p99 = 1200 ms, the combined p99 is **not** 1000 ms — you must merge the raw samples and recompute. Load-test tools compute percentiles from the full sample set per metric; trust the tool, don't average its outputs by hand.

### The throughput ↔ latency trade-off (the heart of it)

This is the single most important thing in the lecture, and it's where naive load tests produce dangerously optimistic charts.

Recall from Lecture 7: decode is memory-bandwidth-bound, so **batching is nearly free for throughput** — running 32 sequences through one weight-read gets you ~32× the tokens per weight-read. So as you raise concurrency, **system tokens/sec climbs steeply**. That's the good news and it's real.

The bad news: those 32 sequences share one GPU. Continuous batching interleaves them, but each request now waits its turn on every scheduler step, and new arrivals **queue** behind a full batch. So as concurrency climbs:

- **tokens/sec (system)** goes UP (good — more total work, lower cost/token)
- **TTFT p99** goes UP (bad — requests wait longer to even start)
- **TPOT** stays flat, then degrades once memory bandwidth saturates

Sketch of the curves as you sweep concurrency 1 → 8 → 64 → 256:

```
tokens/sec (system)          p99 TTFT
   │            ____            │              /
   │        ___/                │            /
   │     __/                    │         __/
   │   _/                       │   _____/
   │ _/                         │__/
   └────────────── conc         └────────────── conc
   climbs, then plateaus        flat, then hockey-sticks
```

There is a **knee**: below it, adding concurrency buys throughput almost for free; above it, you're just growing the queue — throughput plateaus (the GPU is saturated) while TTFT explodes. Your job in a load test is to *find that knee* for your model and hardware, because that's the operating point where you get near-max throughput while keeping tail latency inside your SLO.

This is why **you must always report throughput and latency together, at the same concurrency**. A vendor benchmark that says "12,000 tokens/sec!" without stating concurrency and the corresponding p99 TTFT is marketing, not engineering — they cranked concurrency to 512, blew p99 TTFT to 8 seconds, and quoted only the throughput. Your per-concurrency-tier table exists precisely to make this trade-off visible and un-hideable.

### Closed-loop vs open-loop load models

*How* you generate load changes what you measure. There are two fundamentally different models, and picking the wrong one makes your numbers describe a system nobody actually runs.

**Closed-loop (fixed concurrency / fixed VUs).** You have N virtual users. Each one sends a request, *waits for the full response*, then immediately sends the next. Concurrency is pinned at N; arrival rate is whatever the server can sustain. In k6 this is `constant-vus`.

- Models: a fixed-size worker pool, a batch job with N parallel workers, an internal service with a bounded client pool.
- The catch — **coordinated omission**: because a VU can't send its next request until the current one finishes, a slow server *automatically throttles the load*. If responses slow down, requests are sent *less often*, so the test never applies the backpressure a real burst would. Closed-loop tests systematically *understate* tail latency under overload. It measures capacity honestly at steady state but hides what happens when you exceed it.

**Open-loop (fixed arrival rate).** Requests arrive at a target rate (say 50 RPS) **regardless of whether prior ones finished**. Concurrency is an *output*, not an input — it rises on its own if the server can't keep up. In k6 this is `constant-arrival-rate` (with `preAllocatedVUs`).

- Models: real user traffic — users click when they click; they don't wait for *your* server to be free. This is what production actually looks like.
- The catch — if the server can't sustain the arrival rate, the queue grows without bound and latency runs away. That's not a bug in the test; **that's the test correctly showing you fall over at that rate.** Open-loop is how you find your true sustainable RPS and your real overload behavior.

Rule of thumb: use **closed-loop** to characterize capacity and build the per-concurrency table (it maps cleanly to "N concurrent users"); use **open-loop** to validate an SLO against a realistic arrival rate and to find the RPS where the server breaks. Serious capacity work uses both. If you only ever run closed-loop, you will be blindsided in production the first time a genuine traffic spike arrives faster than you drain it.

---

## Worked example

Let's measure a Qwen2.5-7B endpoint on one L4, closed-loop, three tiers. Fix the workload so runs are comparable: **prompt pinned at ~200 input tokens, `max_tokens: 200`, `temperature: 0`** (pinning temperature matters — sampling changes output length and therefore token counts and E2E; a comparable run needs deterministic-ish length). We stream (`stream: true`) so we can time the first SSE chunk.

**Measuring TTFT and TPOT correctly.** With streaming:

- `TTFT = t(first SSE data chunk) − t(request sent)`
- `TPOT = (t(last chunk) − t(first chunk)) / (output_tokens − 1)` — i.e. **(total − TTFT) / (tokens − 1)**. You divide by tokens *after* the first because TPOT is the *inter*-token gap.

Say for one request: sent at 0 ms, first token at 240 ms, last (200th) token at 8,240 ms, output = 200 tokens.

```
TTFT = 240 ms
decode span = 8240 − 240 = 8000 ms over 199 gaps
TPOT = 8000 / 199 ≈ 40.2 ms/token  → per-request rate ≈ 24.9 tok/s
E2E  = 8240 ms
```

Now run all three tiers for 60 s each and collect the distribution. A representative result table (numbers illustrative, shape realistic):

| Concurrency | tokens/sec (sys) | req/s | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p50 | TPOT p95 | TPOT p99 |
|-------------|-----------------:|------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 1           |               25 |  0.12 |   240 ms |   260 ms |   280 ms |   40 ms  |   42 ms  |   45 ms  |
| 8           |              180 |  0.90 |   310 ms |   520 ms |   690 ms |   43 ms  |   48 ms  |   55 ms  |
| 64          |              620 |  3.10 |   1.9 s  |   4.8 s  |   7.2 s  |   58 ms  |   85 ms  |  120 ms  |

Read this like an engineer:

- **Throughput scaled ~25× (25 → 620 tok/s)** from conc 1 → 64. Decode batching working as advertised — this is the cost win.
- **TPOT barely moved at conc 8** (40 → 43 ms p50) — the batch fits comfortably in memory bandwidth. At conc 64 it degrades (40 → 58 p50, 120 p99): the batch is now large enough to contend for bandwidth *and* chunked-prefill of new arrivals is stealing decode steps.
- **TTFT p99 exploded 280 ms → 7.2 s.** That's pure queueing. At conc 64 you have far more requests than the server can prefill at once, so new requests wait seconds just to start. If your SLO is "p95 TTFT < 800 ms," **conc 8 passes and conc 64 fails**, even though conc 64 has 3.4× the throughput.

**The knee is between 8 and 64.** Your actual operating concurrency is the largest one that keeps p95 TTFT inside SLO — probably somewhere around 16–24 here, which you'd find by adding tiers. This table *is* the deliverable the lab and milestone ask for, and it's the artifact that lets you say "at conc 16 we do ~350 tok/s at p95 TTFT 750 ms" instead of a context-free "620 tok/s."

**Cross-check against vLLM `/metrics`.** vLLM exposes Prometheus counters/histograms that let you verify your client-side numbers from the *server's* point of view — invaluable when you suspect your generator is the bottleneck. Scrape `http://localhost:8000/metrics` and look for the histogram families (names as of recent vLLM):

- `vllm:time_to_first_token_seconds` — server-measured TTFT histogram
- `vllm:time_per_output_token_seconds` — server-measured TPOT/ITL
- `vllm:e2e_request_latency_seconds` — end-to-end
- `vllm:request_queue_time_seconds` — time spent queued (this is the piece of TTFT that isn't prefill — hugely diagnostic)
- `vllm:generation_tokens_total` / `vllm:prompt_tokens_total` — counters; diff over time = throughput
- `vllm:num_requests_running` / `vllm:num_requests_waiting` — live batch and queue depth

If your k6 TTFT p99 is 7 s but `request_queue_time` p99 is 6.5 s, you've *proven* the tail is queueing, not prefill — the fix is more replicas or lower concurrency, not a faster prompt. If server-side TTFT is 300 ms but your client sees 3 s, the gap is **network or client-side** — which is the next failure mode.

---

## How it shows up in production

**The laptop-over-the-internet benchmark (the classic).** You run k6 from your MacBook in a café against a GPU box in `us-east-1`. Round-trip network latency is 80–150 ms *each way*, plus TLS handshake, plus your home Wi-Fi jitter. Your measured TTFT is 350 ms; the server's `/metrics` says 90 ms. You just "measured" the internet, not your endpoint. Under real load the network term also has its own fat tail (a single dropped packet + retransmit adds hundreds of ms to p99), so your tail is now a convolution of two unrelated systems and completely uninterpretable. **Run the load generator in-region — ideally on the same box or same VPC/subnet as the server.** For pure endpoint characterization, run k6 on `localhost` against the container. When you *do* want to include realistic network latency (an SLO for real users), measure it deliberately and separately, and report both "on-box" and "from client region" numbers.

**Cost is a throughput number.** Your cost/1M tokens (next lecture) is `gpu_$/hr ÷ (tokens/sec × 3600) × 1e6`. That tokens/sec is the *system* number at your chosen operating concurrency — the same table cell. Pick the wrong concurrency and your cost model is off by the throughput ratio. This is why "report throughput at the concurrency that meets your latency SLO" is not academic: it's the input to the buy-vs-rent-vs-API decision.

**SLOs are p95/p99 statements.** A real serving SLO reads like "p95 TTFT < 800 ms and p95 TPOT < 60 ms at up to 20 concurrent requests." Every clause is a percentile at a stated load. Alerting on the mean means you page *after* half your users are already suffering; alerting on p99 (or on `num_requests_waiting` climbing) pages you *before* users complain. The load test is where you discover what those numbers *can* be on your hardware, so the SLO you promise is one you can keep.

**Un-pinned workloads produce un-comparable runs.** If run A used 50-token prompts and run B used 2,000-token prompts, run B's TTFT is legitimately higher (more prefill) — comparing them tells you nothing about a code change. If `temperature > 0` and you didn't cap output length, two runs generate different token counts, so tokens/sec and E2E differ for reasons unrelated to the server. **Pin input token count, pin `max_tokens`, pin temperature.** Comparability is the whole point of a benchmark; a benchmark you can't compare across is a random number generator.

---

## Common misconceptions & failure modes

- **"Our average latency is 400 ms, we're fine."** The average is the least useful latency statistic. Report p50/p95/p99 or you're hiding the tail that defines the user experience. p50 can be great while p99 is a disaster.
- **"tokens/sec is tokens/sec."** System throughput (all requests summed) vs per-request rate (1/TPOT) differ by roughly the concurrency factor. A vendor "10k tok/s" is system-wide at high concurrency; a single user still only gets ~30 tok/s. Never quote throughput without stating concurrency.
- **"Higher concurrency is strictly better."** It maximizes throughput and *destroys* p99 TTFT past the knee. The best operating point is the highest concurrency that still meets your latency SLO, not the highest concurrency period.
- **"I'll measure TTFT without streaming."** Impossible. A non-streaming request only returns when *complete*, so the earliest you can time is E2E. TTFT and TPOT **require `stream: true`** and timing individual SSE chunks. If your load script isn't parsing the stream, its "TTFT" is fiction.
- **Coordinated omission in closed-loop tests.** Fixed-VU tests throttle themselves when the server slows, hiding overload tail latency. If you need to know "what happens at 100 RPS," use an open-loop (constant-arrival-rate) model.
- **Cold-start / warm-up contamination.** The first few requests hit an un-warmed CUDA graph, empty prefix cache, and JIT paths. Discard a warm-up window (e.g., first 10–15 s) or your p99 is measuring startup, not steady state.
- **Timing the wrong segment.** Dividing decode time by *all* tokens instead of (tokens − 1), or including the queue in "prefill," or counting TTFT inside TPOT. Draw the timeline; label each segment; then compute.
- **Load-testing from far away.** Covered above — network latency and its own tail swamp TTFT. Generate load in-region/on-box and cross-check with server `/metrics`.
- **Ignoring the tokenizer in token counts.** "150 words" is not 150 tokens. Compute tokens/sec from the *tokenizer's* count (or vLLM's reported `completion_tokens`), not word count, or your throughput is off by ~1.3×.

---

## Rules of thumb / cheat sheet

- **Always report four numbers, each at p50/p95/p99:** TTFT, TPOT, E2E, and system throughput (tokens/sec + req/s) — at a **stated concurrency**.
- **TTFT** = queue + prefill; blows up under concurrency. **TPOT** = decode cadence; flat until bandwidth saturates. **E2E** = TTFT + (out−1)×TPOT.
- **TTFT and TPOT need `stream:true`.** TTFT = time to first SSE chunk. TPOT = (total − TTFT) / (tokens − 1).
- **Never quote throughput without concurrency and its p99 TTFT.** They trade off; report them together.
- **Percentiles don't average.** Merge raw samples; let the tool compute.
- **Closed-loop (constant-vus)** for capacity/per-tier tables; **open-loop (constant-arrival-rate)** for SLO validation and finding the break point. Beware coordinated omission in closed-loop.
- **Pin the workload:** fixed input tokens, fixed `max_tokens`, fixed temperature (0 for comparability). Discard a warm-up window.
- **Run the generator in-region / on-box.** Cross-check client numbers against vLLM `/metrics` (`time_to_first_token_seconds`, `time_per_output_token_seconds`, `request_queue_time_seconds`, `num_requests_waiting`).
- **Find the knee:** sweep concurrency (1, 8, 32, 64, 128…); operate at the largest tier that keeps p95 TTFT inside SLO.
- **Approximate readability targets (rules of thumb, not laws):** p95 TTFT < ~800 ms feels responsive for chat; TPOT < ~50 ms (≈20+ tok/s) reads smoothly. Batch/agent workloads care about E2E and cost, not TTFT.

---

## Connect to the lab

This lecture is the theory behind **Week 2, Step 3** (load test with k6) and **Steps 2 and 5** (instrument `/metrics`, write the recommendation). In the lab you install k6, run `loadtest/script.js` with `constant-vus` scenarios at concurrency **1 / 8 / 64**, add a `stream: true` variant to split TTFT from TPOT, and write p50/p95/p99 for each to `results/latency.csv` — the exact per-concurrency table this lecture walks through. You'll keep `nvitop` and vLLM's `/metrics` visible to cross-check (Lecture 7's memory-bound story explains *why* TPOT stays flat then degrades). That table then feeds **Step 4's** cost/1M-tokens math and the phase milestone's break-even chart — where the throughput number you pick is only defensible because you reported the p99 TTFT next to it.

## Going deeper (optional)

- **k6 documentation** (grafana.com/docs/k6) — executors (`constant-vus`, `ramping-vus`, `constant-arrival-rate`), `Trend` metrics, scenarios/stages, thresholds. The open-vs-closed model distinction is documented under "Scenarios / executors."
- **locust documentation** (docs.locust.io) — the Python alternative; `HttpUser`, `@task`, `wait_time`; good when your team lives in Python and wants request logic in real code.
- **vLLM docs** (docs.vllm.ai) — search "metrics" and "benchmarks." vLLM ships `benchmarks/benchmark_serving.py` (and the newer `vllm bench serve`), which measures TTFT/TPOT/throughput properly with streaming — read it as a reference implementation of *how to time an LLM endpoint*. Also the Prometheus metrics reference for the exact counter/histogram names on your version.
- **Coordinated omission** — search "Gil Tene coordinated omission" (the canonical talk, "How NOT to Measure Latency"). Essential for understanding why closed-loop load tests understate tails.
- **"Latency numbers every programmer should know"** and general percentile/tail-latency writing — search "tail at scale Dean Barroso" (the Google paper on why p99 dominates at fan-out scale).
- **Prometheus histograms & quantiles** (prometheus.io/docs) — how `histogram_quantile` works and why you can't average pre-aggregated quantiles.

## Check yourself

1. Your endpoint has p50 TTFT of 180 ms and p99 TTFT of 6 s at concurrency 64. Throughput is 620 tok/s. Your SLO is "p95 TTFT < 800 ms." What do you do, and what does it cost you?
2. Why can't you measure TTFT with a non-streaming request? What's the earliest thing a non-streaming client can time?
3. A vendor advertises "8,000 tokens/sec." What three questions must you ask before that number means anything?
4. You run a fixed-VU (closed-loop) test at 64 VUs and latency looks fine, but production melts down during a traffic spike at the same average RPS. What measurement artifact likely fooled you, and which load model would have caught it?
5. Your k6 client reports p99 TTFT of 3.2 s, but vLLM's `/metrics` shows `time_to_first_token_seconds` p99 of 250 ms and `request_queue_time_seconds` p99 of 90 ms. Where is the 3 s going?
6. Given TTFT = 200 ms, output = 300 tokens, TPOT = 40 ms, what is E2E latency? If instead output were 10 tokens, which metric now dominates E2E?

### Answer key

1. Concurrency 64 fails the SLO (p95 will be well above 800 ms given p99 = 6 s — that's mostly queueing). Drop to a lower tier — sweep 8/16/32 to find the largest concurrency where **p95 TTFT < 800 ms** (the "knee"). The cost is throughput: you'll operate below 620 tok/s (maybe ~300–400), which raises your cost/1M tokens. That's the throughput↔latency trade-off made concrete — you're buying tail latency with throughput. If you need both, add replicas.
2. A non-streaming request only returns once the full response is generated, so the connection yields nothing until completion — the earliest you can time is **end-to-end latency**. TTFT and TPOT require `stream: true` so you can timestamp the first SSE chunk (TTFT) and the inter-chunk gaps (TPOT).
3. (a) **At what concurrency?** — throughput is meaningless without it. (b) **What's the corresponding p99 TTFT/TPOT?** — they may have blown latency to get that number. (c) **What input/output token counts and model/hardware?** — 8k tok/s on short outputs and a big batch is a different claim than on long outputs; and it must be the same model + GPU to compare to yours. (Bonus: system-wide vs per-request.)
4. **Coordinated omission.** In a closed-loop test each VU waits for its response before sending the next, so when the server slows the test *sends fewer requests* — it throttles itself and never applies true overload backpressure, hiding the tail. An **open-loop** model (`constant-arrival-rate`) sends at a fixed RPS regardless, so the queue grows and the runaway latency shows up in the test — matching the production spike.
5. Server-side TTFT (prefill) is 250 ms and queue time is only 90 ms, so the server is healthy and fast. The ~3 s is **outside the server** — network latency/jitter between your generator and the endpoint (and possibly client-side SSE parsing overhead). Fix: run the load generator in-region/on-box; the server isn't the problem.
6. E2E = TTFT + (output − 1) × TPOT = 200 + 299 × 40 = 200 + 11,960 = **12,160 ms ≈ 12.2 s** — decode dominates. With only 10 tokens: 200 + 9 × 40 = 560 ms, and now **TTFT (200 of 560 ms) dominates** — for short outputs, TTFT is most of E2E; for long outputs, TPOT × length is.
