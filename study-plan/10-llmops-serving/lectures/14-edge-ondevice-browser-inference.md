# Lecture 14: On-device, edge, and in-browser inference — when the server loses

> Every lecture in this phase so far assumed the model lives on a GPU you rent and reach over HTTP. This one asks the heretical question: what if there is no server at all? What if the model runs on the user's laptop, their phone, or inside the browser tab they already have open? For a real and growing slice of workloads that is not a downgrade — it is strictly better: the data never leaves the device, it works on a plane with no wifi, and your per-user compute bill is exactly zero because you are not renting the GPU, the user already bought it. After this lecture you will know the runtimes that matter (llama.cpp/GGUF, Ollama, LM Studio, MLX, WebLLM/transformers.js), what quantization actually costs you in quality and bytes, and — the part that pays your salary — a hard-edged decision framework for *when* the portable end of the spectrum wins and when reaching for it is a rookie mistake. And you will see why the OpenAI-compatibility theme from Week 1 means shipping to the edge is often a one-line `base_url` change.

**Prerequisites:** you can hit an OpenAI-compatible endpoint with the stock client (Week 1); you understand the VRAM equation and that quantization trades bytes for quality (Lectures 6, 9); a rough feel for tokens/sec, TTFT, and memory-bound decode (Lectures 7, 8). · **Reading time:** ~28 min · **Part of:** Phase 10 (LLMOps) Week 3

---

## The core idea (plain language)

There is a spectrum of *where* a token gets generated, and it runs from "someone else's datacenter GPU" all the way down to "the CPU in the phone in your pocket." Most of this phase lived at the datacenter end — vLLM on a rented card, priced per GPU-hour. This lecture is about the portable end, and, more importantly, the *decision boundary* between the two.

The portable end has three properties the server end structurally cannot match, no matter how much money you throw at it:

1. **Privacy by physics.** The data is processed on the device it originated on. It never crosses a network. There is no server log to subpoena, no vendor to trust, no VPC to secure, no data-processing agreement to negotiate. For medical notes, legal drafts, personal journals, source code under NDA — anything a user would hesitate to upload — this is not a feature you bolt on. It is the default, and it is unbeatable because there is nothing to breach.
2. **Offline capability.** No network, no problem. A model on the device works on an airplane, in a hospital basement, on a factory floor, in a warzone, in a country where your API is blocked.
3. **Per-user marginal cost = 0.** You are not renting the GPU — the user already owns it. Whether you have 10 users or 10 million, your inference bill is identical: nothing. The cost moved off your income statement and onto their electricity bill and their battery.

And it has three properties that are structurally *worse* — not "worse until we optimize," but worse by construction:

1. **You are capped at small, quantized models.** A laptop has maybe 8–32 GB of RAM shared with everything else; a phone has less. You are running a 1B–8B model at 4-bit, not a 70B and certainly not a frontier model. The quality ceiling is real, and it is low.
2. **Output consistency and quality are worse and vary by device.** Quantization degrades quality (how much depends on the level and the task), and a small model is a small model. Worse: the same model can behave differently across GPUs and driver versions.
3. **Update and version control across heterogeneous hardware is a nightmare.** Your server model is one thing you control. Your edge model runs on ten thousand different CPUs, GPUs, OS versions, and RAM sizes, each holding a possibly-stale copy of the weights. You lost the single knob that makes server-side rollout and rollback safe (which is exactly the *next* lecture's topic).

The whole lecture is about learning to feel that trade in your gut, so you reach for edge when it wins and a server when it doesn't — and know the hybrid pattern that gets you both.

## How it actually works (mechanism, from first principles)

### The runtimes, and where each fits

A handful of tools do 95% of the work at the portable end. They are not really competitors; they are different rungs on one ladder, and most of them sit on the same foundation.

**llama.cpp + GGUF — the portable CPU/edge runtime.** llama.cpp is a C/C++ inference engine with essentially no dependencies that runs LLMs on CPUs (and optionally on GPUs, Apple Metal, Vulkan, etc.) fast enough to be genuinely useful. It is the bedrock almost everything else at this end is built on. Its companion is **GGUF**, a single-file model format that packs the weights, the tokenizer, the chat template, and metadata into one `.gguf` file you can copy around like any other file. The magic of llama.cpp is that it turned "run a real LLM on a normal laptop with no GPU" into a `make && ./llama-cli -m model.gguf` reality. When you download a quantized model off Hugging Face to run locally, you are almost always downloading a GGUF someone built for llama.cpp.

**Ollama — the ergonomic wrapper over llama.cpp.** Ollama takes llama.cpp and wraps it in the developer experience you actually want. `ollama pull qwen2.5:3b` fetches a model from a registry (the mental model is `docker pull`); `ollama run qwen2.5:3b` drops you into a chat REPL; and — the load-bearing part for this phase — it exposes an **OpenAI-compatible API on `http://localhost:11434/v1`**. That means the exact client code you wrote in Week 1 against vLLM works against Ollama by changing *one string*:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",  # <- the only change from your vLLM client
    api_key="ollama",                       # any non-empty string; Ollama ignores it
)
r = client.chat.completions.create(
    model="qwen2.5:3b",
    messages=[{"role": "user", "content": "one word: hi"}],
)
print(r.choices[0].message.content)
```

Same SDK, same `/v1/chat/completions`, same request and response shapes. This is the punchline of the entire "OpenAI-compatible" theme running through the phase: **vLLM (Week 1), Ollama (this week), and a hosted API like OpenAI/Together are interchangeable behind one `base_url`.** You write the client once; you choose *where the tokens come from* at deploy time.

**LM Studio — the GUI.** Same engine family (it also builds on llama.cpp, plus an MLX backend on Macs), but delivered as a polished desktop app: browse a model catalog, download with a click, chat in a window, and flip on a local OpenAI-compatible server — all with buttons instead of a terminal. It is what you hand a non-engineer, or what you use yourself to eyeball a model's vibe before scripting against it. Under the hood it serves the same `/v1` endpoint.

**MLX — Apple-silicon-native.** Apple's array framework, purpose-built for M-series chips. Its edge is **unified memory**: on a Mac, the CPU and GPU share one physical pool of RAM, so there is no PCIe copy of weights from system RAM into separate VRAM — the GPU reads the weights where they already live. On a 32 GB or 64 GB Mac this is a real advantage: you can run models that would need an expensive discrete GPU on a PC, and MLX squeezes more tokens/sec out of the silicon than a generic CPU path. If your users are on Macs, MLX (or llama.cpp's Metal backend) is the fast lane. The trade: it is Apple-only, so it is a per-platform optimization, not a portable base.

**WebLLM / transformers.js — fully in the browser over WebGPU.** This is the far end of the spectrum: the model runs *inside the web page*, executed on the GPU through the **WebGPU** browser API, with **zero server**. The user visits a URL, the browser downloads the quantized weights once (hundreds of MB to a couple GB — the one real cost), and after that every token is generated locally at **zero marginal cost and total privacy** — nothing you type ever leaves the tab. **WebLLM** (from the MLC project) targets chat/LLM workloads and, naturally, exposes an OpenAI-compatible JS API. **transformers.js** is Hugging Face's broader "run Transformers models in JavaScript" library (embeddings, classification, ASR, small LLMs). The catch: the model must fit what a browser will download and what WebGPU can hold, so you are at the small-and-quantized end of even the edge spectrum.

```
     server end  <──────────────────────────────────────────>  portable end
   ┌───────────┐   ┌──────────┐   ┌────────┐   ┌───────┐   ┌────────────┐
   │  vLLM on  │   │  Ollama  │   │   LM   │   │  MLX  │   │  WebLLM /  │
   │ rented GPU│   │ (laptop) │   │ Studio │   │ (Mac) │   │  browser   │
   └───────────┘   └────┬─────┘   └───┬────┘   └───┬───┘   └─────┬──────┘
    70B, fp16           └──── all built on llama.cpp ────┘         │
    full control                                            WebGPU, no server
    $$$/hour             <──── 1B–8B, quantized, $0/user ──>  first-load download

    All five speak /v1/chat/completions → swap them with base_url alone.
```

### Quantization: GGUF and the Q4/Q5/Q8 tradeoff

Edge lives or dies on quantization, so you need to reason about it numerically, not by feel. Recall the weights math from Lecture 6: memory = params × bytes/param. At fp16 that is 2 bytes/param. Quantization stores each weight in *fewer* bits, which shrinks two things at once: the file you must download and, critically for a memory-bandwidth-bound decode (Lecture 7), the bytes you must read from RAM per generated token. On a laptop with slow memory, that second effect is often *why* the smaller model is faster, not just smaller.

GGUF quantization levels are labeled roughly by bits-per-weight. The ones you will actually meet:

- **Q8_0 (~8 bits/weight, ~1 byte):** near-lossless. Quality essentially indistinguishable from fp16 on most tasks. Biggest file of the practical set.
- **Q6_K (~6 bits):** a notch down from Q8, still very high quality; a reasonable pick when Q4 worries you but Q8 won't fit.
- **Q5_K_M (~5 bits):** small, well-controlled quality loss. A common "I care about quality" choice.
- **Q4_K_M (~4 bits/weight, ~0.5 byte):** the **default sweet spot**. Roughly half the size of Q8 for a quality drop most users won't notice on everyday tasks. When someone says "just grab the Q4," this is the file they mean.
- **Below Q4 (Q3_K, Q2_K):** quality degrades faster and more visibly — the model gets noticeably dumber, especially on reasoning and code. Use only when you are genuinely out of RAM.

The `_K_M` suffix marks the **K-quants**, which are smarter than flat/legacy quantization: they keep the more sensitive weights (attention projections, some layers) at higher precision and squeeze the rest, so you get better quality per byte than the raw bit count suggests. `_S`/`_M`/`_L` denote small/medium/large variants of that scheme. Default to `_M`.

Do the arithmetic for a 7B model:

```
7B params × 2 bytes  (fp16)   = 14.0 GB   — needs a datacenter GPU; won't sit on an 8/16 GB laptop
7B params × 1 byte   (Q8_0)   =  7.0 GB   — fits a 16 GB machine, near-lossless
7B params × 0.5 byte (Q4_K_M) ≈  3.5 GB   — fits an 8 GB machine, small quality hit
```

That 14 GB → 3.5 GB collapse is *the* enabling trick of the entire edge story. It is the difference between "needs a rented GPU" and "runs on the laptop the user already owns." But weights are not the whole footprint. Add the KV cache and runtime overhead on top (Lecture 6). The rule of thumb: **the quantized weights should fit in well under your available RAM**, leaving room for the OS, the browser, the KV cache, and your actual application. On an 8 GB laptop, a 3–4B model at Q4 is comfortable; a 7B at Q4 (~3.5 GB weights + cache + OS) is tight; a 13B is a swap-thrashing disaster that "loads" and then crawls at 1–2 tok/s.

### Why "zero marginal cost" is exactly true — and where the cost hides

On a server, every request burns GPU-seconds you pay for (Lecture 9's `cost_per_1M`). On the edge, the user's hardware does the compute, so *your* marginal cost per inference is genuinely, arithmetically zero. The cost did not vanish, though — it moved and changed shape:

- **First-load download.** The weights ship to the device once. A 3.5 GB Q4 model is a 3.5 GB download the first time; after that it's cached and inference is free.
- **The user's battery and thermals.** A phone or thin laptop doing sustained decode gets hot and drains fast. Free to you, not free to them.
- **The user's RAM.** You are squatting in memory they would rather give to Chrome or their IDE.

None of these hit *your* bill. That is precisely why edge economics are so attractive at scale — and precisely why the costs are easy to ignore until a user complains their fan is screaming.

## Worked example

You are building a **private journaling app**. Users write personal reflections and want an AI to summarize their week and surface recurring themes. This is about the most privacy-sensitive data a person produces. Two architectures:

**Option A — server.** Ship entries to your vLLM box (or a hosted API) for summarization. From Lecture 9, say you sustain 1,500 tok/s on an A10 at $0.75/hr → about $0.14 per 1M tokens *at full utilization*. Each user writes ~2,000 tokens/week and you generate a ~300-token summary; call it ~2,300 tokens/user/week. At 100,000 active users that is 230M tokens/week ≈ 1B tokens/month. Even at the cheap, fully-pegged floor that is ~$140/month of raw inference — realistically 2–4× that once you account for real-world utilization and the ops tax (Lecture 9), so call it $300–500/month. And you are now the legal custodian of 100,000 people's private diaries: a compliance surface, a breach liability, and a trust problem you can never fully retire.

**Option B — edge (WebLLM in the browser, or a bundled Ollama/MLX on desktop).** Ship a Q4 3B model to the browser. First visit: the user downloads ~2 GB once (show a progress bar; cache it). Every summarization after that runs on their GPU via WebGPU. Your inference bill: **$0, at any user count.** The diary text **never leaves their machine** — there is no server log because there is no server call. Your compliance surface for that feature is essentially empty.

The numbers that decide it:

```
                         Server (Option A)          Edge (Option B)
inference cost/month     ~$300–500 (grows w/ users)  $0 (flat, any scale)
data leaves device?      yes (liability)             no (private by physics)
works offline?           no                          yes
model quality            can use 70B / frontier      capped ~3B Q4
first-use latency        instant                     ~2 GB download once
per-device consistency   one model, controlled       varies by browser/GPU
version control          instant central rollout     stale clients forever
```

For *this* app — private, small-model-sufficient (summarizing a week of journaling does not need a 70B), and cost-sensitive at scale — **edge wins decisively.** A 3B Q4 summary is "good enough," and the privacy story is a genuine product differentiator you can put on the landing page: *"Your journal never leaves your device."*

Now change one variable: the app becomes a **legal-contract analyzer** where a misread clause has real financial consequences. Suddenly the "capped at 3B Q4" line is disqualifying — you need frontier reasoning and tight consistency. Edge loses; you go server and solve the privacy problem with a VPC and a BAA instead. **Same framework, opposite answer, because the quality requirement moved.** That is the whole skill: the decision is not "edge good / server good," it is a function of privacy sensitivity, quality bar, offline need, and user count.

## How it shows up in production

- **"It ran great on my M3 Max, ships broken on a 4-year-old ThinkPad."** Your dev machine has 64 GB of unified memory and a fast GPU. Your user has 8 GB and integrated graphics. The 7B Q4 that flew for you swaps to disk and dribbles out 2 tok/s for them. **Edge performance is a distribution, not a number** — benchmark on the *worst* hardware you claim to support, not the best, and pick the model size for the floor, not the ceiling.
- **The version-control nightmare is real and it bites late.** You fix a bad prompt or a bad model on the server and every user has the fix in seconds. On the edge, users run whatever they last downloaded — last week's model, last month's app build. There is no `rollback.sh` (that is the next lecture's server luxury). You need an explicit update mechanism (re-download on version bump, app auto-update, a server-checked version gate) and you must assume a long tail of stale clients *forever*.
- **First-load download is a conversion killer if you're careless.** A 2 GB download before the app does anything will lose users on mobile data or a hotel wifi. Mitigations: lazy-load the model only when the AI feature is first used, cache aggressively (it is a one-time cost), pick the smallest model that clears the quality bar, and *show a real progress bar* so it does not look hung.
- **WebGPU availability is not universal.** It is solid in current Chrome/Edge and shipping in recent Safari/Firefox, but a user on an older browser, a locked-down enterprise machine, or certain mobile configs may not have it. You need a fallback path (a server call, or a clear "please update your browser") — never assume the browser can run the model.
- **Battery and thermals are your users' complaint, not your dashboard's metric.** Sustained on-device inference heats phones and drains laptops, and it will never show up in your server-side monitoring because there is no server involved. Cap generation length, avoid running the model when a rule or a tiny classifier would do, and don't pin the GPU in the background.
- **The privacy claim must be *true*, end to end.** "Runs on-device" loses all its value the moment you also phone home with the prompt for "analytics" or crash reporting. If you are selling privacy, the data genuinely cannot leave — audit your own telemetry so you do not quietly break the one promise that justified going edge in the first place.

## Common misconceptions & failure modes

- **"Edge is just a cheaper server."** No — it is a *different* set of trade-offs. It is cheaper *and* more private *and* offline-capable, but with a hard, low quality ceiling and no central control. You do not pick edge to save money on a workload that needs a 70B; you *cannot run* the 70B on the device at all.
- **"Q4 is basically fp16, it's fine."** Q4_K_M is the sweet spot, not a free lunch. It is a real quality drop — usually invisible on easy tasks, sometimes very visible on hard reasoning, long code, or long context. Re-eval on *your* task (same "re-run evals after quantizing" discipline as Lecture 9); do not assume.
- **"Ollama is a different API, I'll have to rewrite my client."** The opposite — Ollama exposes an OpenAI-compatible endpoint on `:11434/v1`. Your Week-1 client works with a `base_url` swap. Interchangeability is the entire design goal.
- **"In-browser means it re-downloads the model every request."** First load only; the weights are cached in the browser afterward. Marginal inference cost after that is zero. Budget the *one* download, not one per request.
- **"MLX is just llama.cpp for Mac."** Related niche, different tool. MLX is Apple's own framework exploiting unified memory; llama.cpp also has a Metal backend. On Apple silicon either can be the fast path — the point is the unified-memory advantage, not the specific library.
- **"If it fits in RAM, it'll be fast."** Fitting and being fast are different. A model that *barely* fits leaves no room for KV cache and the OS, spills to swap, and crawls. Leave headroom. And because decode is memory-bandwidth-bound (Lecture 7), a laptop's slow memory caps your tokens/sec even when the model "fits" comfortably.
- **"I can't have both edge and server."** The hybrid pattern is very often the right answer — see the cheat sheet.

## Rules of thumb / cheat sheet

*(All specific numbers are engineering defaults to start from and tune — not universal truths. Benchmark your own device.)*

**Use edge when:**
- The data is sensitive and users would balk at uploading it (privacy is a feature, not overhead).
- The app must work offline.
- You have many users and want per-user inference cost = 0 (server cost scales with *your* users; edge cost does not).
- A small (1B–8B) quantized model clears your quality bar — verify with an eval, don't assume.
- A round-trip to a server is a latency tax you'd rather not pay once the model is loaded.

**Use a server when:**
- You need a big model, frontier quality, or tight output consistency.
- You must control the exact model/prompt version and roll changes or rollbacks centrally (next lecture).
- The workload is heavy or bursty in a way that would melt or drain a user's device.
- You need capabilities the device can't give: huge context, heavy tool use, multi-model routing, or the newest model the day it ships.

**The hybrid pattern (usually the mature answer):** run a small model **on the edge for the cheap, private, offline, common-case paths**, and **fall back to a server** for the hard queries the small model flags as low-confidence or out-of-scope. Most requests cost you nothing and stay private; only the hard minority hits your paid GPU. This is Lecture 9's model-cascade idea with the cheap tier relocated onto the user's hardware — and because both tiers speak `/v1/chat/completions`, the fallback is a `base_url` swap, not a rewrite.

**Quantization defaults:** start at **Q4_K_M** (sweet spot). Bump to **Q5_K_M / Q6_K / Q8_0** if an eval shows Q4 hurts your task. Go below Q4 only when you are out of RAM. Then **re-run your evals** — quantization is a quality knob, not just a size knob.

**Sizing:** quantized weights should fit *well under* available RAM (leave room for OS + KV cache + your app). 8 GB laptop → ~3B Q4 comfortable, 7B Q4 tight, 13B don't. Test on your *worst* supported device.

**Portability:** Ollama / LM Studio / vLLM / hosted APIs are all OpenAI-compatible — one `base_url` swaps between them. Write the client once, choose the backend at deploy time.

## Connect to the lab

Week 3, Step 1 *is* this lecture in 15 minutes: install Ollama, `ollama run qwen2.5:3b`, then hit `http://localhost:11434/v1/chat/completions` with your **unchanged** Week-1 OpenAI client — proving the `base_url`-only portability claim with your own hands. The optional `edge/webllm/index.html` page loads a small quantized model fully in the browser over WebGPU so you can watch the one-time model download, then generate tokens with zero server involvement — the "zero marginal cost, total privacy" claim made concrete. In your README, note which paths of *your* app you would put on the edge and which need the server, using the hybrid framework above.

## Going deeper (optional)

- **llama.cpp** — the canonical repo `ggml-org/llama.cpp` (formerly `ggerganov/llama.cpp`) on GitHub. Read the README and the quantization notes. Search: `llama.cpp GGUF quantization types` for the full Q-level table, and `GGUF format spec`.
- **Ollama** — official site `ollama.com` and the `ollama/ollama` GitHub repo. Read the "OpenAI compatibility" doc page specifically. Search: `Ollama OpenAI compatible API`.
- **LM Studio** — `lmstudio.ai` for the app and its local-server docs.
- **MLX** — Apple's `ml-explore/mlx` and `ml-explore/mlx-examples` GitHub repos; docs at `ml-explore.github.io/mlx`. Search: `MLX LLM Apple silicon`, `MLX unified memory`.
- **WebLLM** — `mlc-ai/web-llm` GitHub repo and `webllm.mlc.ai`; its parent project MLC LLM is at `llm.mlc.ai`. Search: `WebLLM WebGPU in browser inference`.
- **transformers.js** — Hugging Face's `huggingface/transformers.js`; docs under `huggingface.co/docs/transformers.js`.
- **WebGPU** — the browser standard itself. Search: `WebGPU browser support caniuse` for current availability before you rely on it.
- **Concept reading** — search `on-device LLM inference tradeoffs` and `hybrid edge cloud LLM cascade` for architecture writeups. Treat every specific tokens/sec and quality number you find as approximate and hardware-dependent — benchmark your own device.

## Check yourself

1. Name the three structural advantages of edge inference and the three structural disadvantages. For each disadvantage, explain why you can't just "engineer around it" with a server-class fix.
2. A 13B model at fp16 is 26 GB. Estimate its size at Q8_0 and at Q4_K_M. Which (if any) fits comfortably on a 16 GB laptop with room for the OS and KV cache, and why?
3. Your Week-1 client points at vLLM. What is the *complete* set of changes needed to make it talk to a local Ollama instance instead, and why is that set so small?
4. A user says your in-browser AI app is "slow to start but fast after that, and it works on the subway." Explain each of those three observations in terms of how WebLLM actually works.
5. You're choosing between edge and server for a feature. Give two concrete signals that push you toward edge and two that push you toward a server. Then describe a hybrid design that uses both.
6. Why is "update/version control across heterogeneous client hardware" a harder problem for edge than the instant rollback you get on a server, and what is one mitigation?

### Answer key

1. **Advantages:** (a) privacy — data is processed on-device and never crosses a network; (b) offline capability; (c) per-user marginal inference cost = 0, since the user's hardware does the compute. **Disadvantages:** (a) capped at small quantized models — a phone/laptop has limited RAM shared with everything, so you *can't run* a 70B/frontier model at all, not merely run it slower; a server-class fix doesn't exist because the constraint is the user's device; (b) worse and device-variable output quality — quantization degrades quality and a small model is inherently weaker, and you can't engineer a bigger brain into hardware that can't hold one; (c) update/version control across thousands of heterogeneous devices — you don't own the fleet, so you can't push a fix and *know* everyone got it the way one server lets you.
2. Q8_0 ≈ 1 byte/param → ~13 GB. Q4_K_M ≈ 0.5 byte/param → ~6.5 GB. On a 16 GB laptop, the Q8 (~13 GB) leaves almost nothing for the OS, KV cache, and other apps — it will swap and crawl. The Q4 (~6.5 GB) fits comfortably with headroom for cache and the OS. So Q4_K_M is the realistic choice; Q8 technically "fits" the number but leaves no room to actually run well.
3. Change the `base_url` to `http://localhost:11434/v1`, use the local model id (e.g. `qwen2.5:3b`), and pass any non-empty string as `api_key`. That's it. The set is tiny because Ollama exposes an **OpenAI-compatible** endpoint, so the SDK, the request/response shapes, and all your surrounding code are identical — portability across vLLM/Ollama/hosted is the entire point of the OpenAI-compatible standard.
4. **Slow to start:** the first visit downloads the quantized weights (hundreds of MB to a couple GB) once. **Fast after that:** the weights are cached locally and every token is generated on the user's own GPU via WebGPU with no network round-trip. **Works on the subway:** once downloaded, inference is fully local — zero server calls — so no connectivity is required.
5. **Toward edge:** sensitive data users won't upload (privacy), a need to work offline, a huge user count where per-user cost=0 matters, or a small model that already clears the quality bar. **Toward server:** need for a big/frontier model or tight consistency, or a requirement for central control over the exact model/prompt version with instant rollback. **Hybrid:** run a small quantized model on-device for the common, private, offline-friendly cases; when it flags low confidence or an out-of-scope query, fall back to a server-hosted larger model. Most requests are free and private; only the hard minority costs you GPU time — and because both speak `/v1`, the fallback is a `base_url` swap.
6. On a server you control the single running instance, so flipping `ACTIVE_VERSION` fixes everyone instantly. On the edge the model runs on thousands of devices you don't control, each with a possibly-stale copy and different OS/GPU/RAM, so there is no central switch — you must ship an update mechanism and still expect a long tail of clients on old versions indefinitely. Mitigation: a server-checked version gate that forces a re-download of weights/prompt (or blocks the feature) when the app detects it is outdated.
