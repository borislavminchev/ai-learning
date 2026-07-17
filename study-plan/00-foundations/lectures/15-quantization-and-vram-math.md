# Lecture 15: Quantization & VRAM Math

> Every question that matters when you self-host a model — "Will this fit on my GPU?", "Which download do I pick?", "Why did it OOM at 8k tokens when it loaded fine empty?", "Why is the int4 version almost as good but a third of the size?" — comes down to a single arithmetic identity and a handful of formats. This lecture takes the one-paragraph VRAM recap from the Week 3 spine and turns it into a mental model you can compute in your head at a whiteboard. After it you can estimate the VRAM of any open model at any precision, split that estimate into weights / KV cache / activations, explain *why* int4 is the local sweet spot and *where* it starts to hurt, tell GGUF from AWQ from GPTQ from bitsandbytes and know when each applies, and pick a quantization for a specific GPU without downloading anything first.

**Prerequisites:** Lecture on the transformer loop & KV cache (Week 2), tokenization, basic NumPy dtypes · **Reading time:** ~26 min · **Part of:** Phase 0 Week 3

---

## The core idea (plain language)

A large language model *is* a big pile of numbers called **parameters** (weights). "7B" means seven billion of them. When you run the model, every one of those numbers has to live in memory that the compute device can reach — for a GPU that's **VRAM** (video RAM), the fast memory soldered next to the chip. If the numbers don't fit in VRAM, the model won't run there. Full stop.

So the first question in local inference is always: *how many bytes does each parameter take?* That's a choice, not a law of nature. The model was trained in a high-precision format, but for **inference** (just running it, not training it) you can store each weight in fewer bits and it still works — often astonishingly well. Shrinking the number of bits per weight is called **quantization**.

The whole lecture reduces to one identity you should be able to recite:

```
weights VRAM (bytes)  =  number of parameters  ×  bytes per parameter
```

with three standard "bytes per parameter" values you must memorize:

| Precision | bits/param | bytes/param |
|-----------|-----------:|------------:|
| fp16 / bf16 | 16 | **2** |
| int8 | 8 | **1** |
| int4 | 4 | **0.5** |

That's it. A 7B model at fp16 is `7 × 2 = 14 GB` of weights. At int4 it's `7 × 0.5 = 3.5 GB`. Everything else in this lecture is (a) the corrections you add on top of weights — KV cache and activations — and (b) the engineering vocabulary (GGUF, AWQ, GPTQ, bitsandbytes) for the *how*. But the load-bearing idea is that one multiplication.

Why does throwing away bits not destroy the model? Because a weight's exact value barely matters. The model's behavior comes from billions of numbers voting together; rounding each vote from "0.37418..." to "roughly 0.375" moves the aggregate almost not at all. Quantization is *lossy compression tuned to a distribution that tolerates loss*. Push it too far (int2, int1) and the rounding error finally swamps the signal — which is exactly why int4 is the floor for "still good," not int1.

---

## How it actually works (mechanism, from first principles)

### What "precision" actually means

A parameter is a real number like `0.3741828`. To store it you pick a numeric format:

- **fp32** (4 bytes) — full 32-bit float. How small models were classically trained. Rarely used for inference now; wastes memory.
- **fp16 / bf16** (2 bytes) — 16-bit floats. `bf16` (bfloat16) trades mantissa bits for a wider exponent range, which is why it's the modern training/serving default — it doesn't overflow as easily. For VRAM math they're identical: **2 bytes**. This is the "native" precision you compare everything against.
- **int8** (1 byte) — the float range is squeezed into 256 integer buckets, plus a scale factor per group of weights so you can reconstruct approximate floats. **1 byte.**
- **int4** (0.5 byte) — 16 buckets per weight, again with per-group scales. Two weights share a byte. **0.5 bytes.**

The key move in integer quantization: you don't store one global scale for all seven billion weights (that would be far too coarse). You chop the weights into small **groups** (commonly 32, 64, or 128 weights) and store one scale — sometimes a scale *and* a zero-point — per group. That per-group metadata is why a "4-bit" model is never *exactly* 0.5 bytes/param in practice; it's more like 0.5 + a small overhead (often quoted as ~4.3–4.5 effective bits). For back-of-envelope work we use 0.5; when the real file is 10–15% bigger than your estimate, this overhead (plus the format container) is why.

```
fp16 weight:  [ 16 bits, exact-ish float ]                      2.0 B
int8 group:   [w][w][w]...[w]  + one scale         →  ~1.0 B / weight
int4 group:   [w|w][w|w]...    + scale (+zero)     →  ~0.5 B / weight  (+ overhead)
```

### The three consumers of VRAM

Weights are the biggest and most predictable slice, but they are not the whole bill. When a model is actually serving requests, VRAM holds **three** things:

```
        TOTAL VRAM
   ┌──────────────────────────────────────────────┐
   │  WEIGHTS         KV CACHE        ACTIVATIONS   │
   │  (fixed:         (grows with     (transient    │
   │   params ×        context ×       working       │
   │   bytes/param)    batch)          memory)       │
   └──────────────────────────────────────────────┘
       ~constant     grows with use   ~overhead
```

1. **Weights** — fixed the moment you load the model. `params × bytes/param`. Doesn't change with prompt length or batch.
2. **KV cache** — the attention keys/values for every token currently in context (from Week 2: the cache that lets each new token avoid reprocessing the whole prompt). This **grows linearly with the number of tokens and with batch size**. On long contexts it can rival or exceed the weights.
3. **Activations** — the transient tensors flowing through the layers during a forward pass, plus framework/CUDA overhead. Roughly a fixed-ish tax. The spine's rule of thumb: budget **20–40% headroom** to cover activations + fragmentation + the runtime itself.

The KV cache size has its own formula worth internalizing:

```
KV cache bytes ≈ 2 × n_layers × seq_len × d_model × bytes_per_elem × batch
                 ↑                                    ↑
             K and V                           usually fp16 = 2 B
```

The leading `2` is because you cache both **K**eys and **V**alues. `d_model` is the model's hidden size. Note the two multipliers that bite in production: **seq_len** (a 32k-token conversation caches 32× more than a 1k one) and **batch** (serving 20 users at once means 20× the cache). This is why a model that loads happily at idle can OOM the instant real traffic with long prompts arrives — the weights were never the problem, the cache was.

### A worked KV-cache number

Take a Llama-3-8B-shaped model: ~32 layers, hidden size 4096, fp16 KV. For one sequence of 4096 tokens:

```
2 × 32 × 4096 × 4096 × 2 bytes × 1  =  ~2.15 GB
```

So ~2 GB of KV cache for a *single* 4k-token request on top of the weights. Bump to 8 concurrent users at 4k each and that's ~17 GB of KV cache alone — more than the int4 weights. That single calculation explains most "it OOMs under load" tickets you'll ever see.

---

## Worked example

You have a **24 GB** GPU (say an RTX 4090 / 3090). You want to run a **13B** model with room for real conversations (~4k-token context, single stream to start). Which precision fits, and with how much headroom?

**Step 1 — weights at each precision** (`params × bytes/param`):

| Precision | Weights |
|-----------|--------:|
| fp16 | 13 × 2 = **26 GB** |
| int8 | 13 × 1 = **13 GB** |
| int4 | 13 × 0.5 = **6.5 GB** |

fp16 is already 26 GB — it doesn't even fit the empty weights in 24 GB. Dead on arrival.

**Step 2 — add KV cache** for 4k tokens. A 13B model is ~40 layers, hidden ~5120. KV cache ≈ `2 × 40 × 4096 × 5120 × 2 = ~3.4 GB` per sequence. (You don't need this exact — "a few GB per 4k stream" is the instinct.)

**Step 3 — add activation/overhead headroom** (~25%).

Now assemble the realistic totals:

| Precision | Weights | +KV(4k) | +25% headroom | Fits 24 GB? |
|-----------|--------:|--------:|--------------:|:-----------:|
| fp16 | 26 | 29.4 | ~37 GB | No |
| int8 | 13 | 16.4 | ~20.5 GB | Yes, tight |
| int4 | 6.5 | 9.9 | ~12.4 GB | **Yes, comfortably** |

**Decision:** load int4. You get the 13B at ~12 GB total, leaving ~12 GB of headroom — which you can spend on **longer context** or **higher batch/concurrency**, the two things that actually scale your serving. int8 fits but leaves little room to grow; fp16 is impossible without a second GPU. This is the reasoning the `econ vram` CLI command automates: it does exactly these three lines and then checks the total against a list of common GPU sizes.

---

## How it shows up in production

- **"Will it fit" is a hiring/architecture decision, not a runtime surprise.** The VRAM identity lets you answer "can we serve Llama-70B on one A100-80GB?" *before* provisioning. 70B fp16 = 140 GB → no single 80 GB card. int4 = 35 GB → yes, with room for KV cache. That single calc decides whether you buy one GPU or two, which is thousands of dollars.

- **KV cache is the silent OOM.** Teams size the box to the weights, deploy, and fall over the first time a user pastes a long document or concurrency spikes. The fix is capacity planning that includes `2 × layers × seq_len × d_model × 2 × batch`, plus serving-side tricks (paged attention in vLLM, KV-cache quantization) you'll meet in later phases. Know now that context length and batch size are *memory* knobs, not just latency knobs.

- **int4 is the default local sweet spot for a reason.** Going fp16 → int8 → int4 roughly halves memory each step for a quality cost that stays small down to 4-bit (typical rule-of-thumb: a few percent on benchmarks, often imperceptible in chat). Below 4-bit the curve turns sharply — int3/int2 lose coherence fast. So int4 is where the memory savings are huge and the quality tax is still cheap. A 7B that needed 14 GB now runs in ~3.5 GB, which is the difference between "needs a datacenter GPU" and "runs on a laptop or an 8 GB consumer card."

- **Where int4 quality actually degrades (so you can spot it).** Quantization damage is not uniform. It shows up most on: long multi-step reasoning and math (small rounding errors compound over many steps), code generation (one wrong token breaks a program), rare/long-tail knowledge, and very long contexts. For chat, summarization, and extraction it's usually invisible. So: benchmark int4 *on your task*, not on a generic leaderboard, and be extra suspicious for reasoning/code workloads — that's where you might step up to int8 or a better 4-bit method (AWQ/GPTQ with calibration beats naive rounding).

- **The format you download dictates your whole stack.** Pick a GGUF and you're on llama.cpp/Ollama (CPU or GPU, Mac-friendly). Pick AWQ/GPTQ and you're on a CUDA GPU with vLLM/TGI/transformers. Pick bitsandbytes and you're quantizing on-the-fly in transformers — CUDA only. Choosing the wrong one for your hardware means a download that won't load. This is the single most common "why won't this run" for beginners.

- **Cost story at scale (previewed in Phase 10).** Quantization is what makes self-hosting economically competitive: fitting a capable model on a cheaper/single GPU changes the $/token math versus a hosted API. The VRAM estimate is the input to that whole spreadsheet.

---

## Common misconceptions & failure modes

- **"The VRAM number is `params × bytes` and I'm done."** That's weights only. Real serving adds KV cache (grows with context × batch) and activation overhead. Budget 20–40% on top, more for long context or high concurrency. Sizing to the weights-only number is the #1 cause of production OOM.

- **"A 4-bit model file is exactly 0.5 bytes/param."** No — per-group scales/zero-points and the format container add overhead, so real 4-bit files land around 4.3–4.5 effective bits (10–15% above the naive estimate). Use 0.5 for napkin math; expect the download to be a bit larger.

- **"Quantization retrains or damages the model permanently."** For inference quantization the original fp16 weights are untouched on disk; quantization produces a *separate* compressed artifact. It's compression, not lobotomy — and it's reversible in the sense that you can always go back to the fp16 source.

- **"int8 is always safer than int4, so use int8."** int8 costs 2× the memory of int4 for a quality gain that's often negligible on chat/extraction tasks. Defaulting to int8 wastes VRAM you could spend on context or batch. Start int4; step up only if *your* eval shows degradation.

- **"bitsandbytes will let me run int4 on my Mac / CPU."** bitsandbytes is **CUDA-only**. On a Mac or CPU you use **GGUF via llama.cpp/Ollama**. Expecting `load_in_4bit=True` to work without an NVIDIA GPU is a guaranteed error and a classic beginner trap (it's called out in the Week 3 pitfalls too).

- **"All 4-bit quants are equal."** A naive round-to-nearest 4-bit is worse than a **calibration-based** 4-bit (GPTQ, AWQ) that uses sample data to protect the weights that matter most. Same bit-width, meaningfully different quality. The method matters, not just the number.

- **"Lower bits also make it proportionally faster."** Memory scales cleanly; speed doesn't. Sometimes int4 is faster (less memory to move — LLM decode is memory-bandwidth-bound), sometimes slower (dequantization overhead, unoptimized kernels). Treat quantization as a *memory* lever first; measure latency separately.

---

## Rules of thumb / cheat sheet

- **The identity:** `weights VRAM = params(B) × bytes/param`. Memorize bytes/param: **fp16 = 2, int8 = 1, int4 = 0.5**.
- **Instant table:** 7B → 14 / 7 / 3.5 GB. 13B → 26 / 13 / 6.5 GB. 70B → 140 / 70 / 35 GB (fp16/int8/int4).
- **Add headroom:** multiply the weights estimate by **~1.2–1.4** for activations + runtime, then add KV cache separately for long context / batching.
- **KV cache:** `≈ 2 × layers × seq_len × d_model × 2 bytes × batch`. Scales with context and concurrency — this is your OOM-under-load term.
- **Default precision:** **int4** for local. It's the memory/quality sweet spot. Step to int8 only if your task (reasoning, code, long-tail facts) shows degradation on *your* eval.
- **Don't go below int4** unless you've measured it — quality falls off a cliff at int3/int2.
- **Format → stack:** **GGUF** = llama.cpp/Ollama (CPU+GPU, Mac). **AWQ / GPTQ** = CUDA GPU serving (vLLM/TGI/transformers), calibration-based 4-bit. **bitsandbytes** = on-the-fly int8/int4 in transformers, **CUDA-only**.
- **Calibration vs on-the-fly:** GPTQ/AWQ *calibrate* on sample data ahead of time (slower to make, better quality, produces a saved artifact). bitsandbytes quantizes *on load* (zero prep, convenient, slightly lower quality, nothing to distribute).
- **Picking a quant for a GPU:** compute weights at int4 → add KV for your target context/batch → ×1.3 → compare to VRAM. If it fits with margin, try int8 for quality; if it doesn't fit, drop params or context, not below int4.
- **When in doubt on hardware:** no NVIDIA GPU → GGUF/Ollama. NVIDIA GPU serving many users → AWQ/GPTQ in vLLM. Quick experiment in a notebook → bitsandbytes.

---

## Connect to the lab

This lecture is the theory behind **Week 3, Lab task 2 (`econ vram` command)** and it's validated by **task 3 (`econ validate`)**. In `vram` you'll implement exactly the three slices above — weights (`params × bytes(quant)`), KV cache (`2 × layers × seq_len × hidden × bytes`), and a ~20% activation headroom — then print whether the total fits 8/16/24 GB GPUs. In `validate` you'll load a genuinely small model (Qwen2.5-0.5B, or Llama-3.2 via Ollama) and compare *measured* memory to your estimate. Watch for two things: (1) your estimate will read a bit low versus reality because of the per-group quant overhead and CUDA/runtime footprint — that gap is expected, document it; (2) the int4/int8 `bitsandbytes` path needs CUDA, so on a laptop you validate the int4 path through **GGUF/Ollama** instead and note the CUDA caveat in the CLI output (Week 3 pitfalls).

---

## Going deeper (optional)

- **Hugging Face `transformers` — Quantization docs** (huggingface.co/docs/transformers) — the *Quantization* overview plus per-method pages for bitsandbytes, GPTQ, and AWQ. The authoritative reference for what runs where.
- **llama.cpp** GitHub repo (`ggml-org/llama.cpp`) — the canonical GGUF engine; read the README section on quant types (Q4_K_M, Q5_K_M, etc.) to understand real-world 4-/5-bit variants. Search: *"llama.cpp quantization Q4_K_M"*.
- **Ollama model library** (ollama.com/library) — ready-made GGUF quants; the fastest way to feel the size/quality tradeoff hands-on.
- **AWQ** — *"AWQ: Activation-aware Weight Quantization"* paper and the `mit-han-lab/llm-awq` repo. **GPTQ** — the original GPTQ paper and the `AutoGPTQ` / GPTQModel repos. Search those names; don't guess URLs.
- **bitsandbytes** GitHub repo (`bitsandbytes-foundation/bitsandbytes`) — the CUDA-only on-the-fly quantizer; the README states the hardware requirements plainly.
- **vLLM docs** (docs.vllm.ai) — for the serving-side story (PagedAttention, KV-cache management) that this lecture's KV math motivates; you'll use it in later phases.
- Talk: search *"Tim Dettmers 8-bit quantization"* and *"GPTQ AWQ explained"* for engineer-level walkthroughs of why low-bit inference works.

---

## Check yourself

1. Estimate the weights-only VRAM for a **34B** model at fp16, int8, and int4. Which of those fits on a single 24 GB GPU with room for a KV cache?
2. Your 7B int4 model loads fine (~3.5 GB weights) on a 12 GB card but crashes after a user pastes a 30k-token document. What ran out, and which term in the VRAM budget did you forget?
3. Why is int4 the usual local sweet spot rather than int8 or int2? Frame it as a memory-vs-quality curve.
4. You're on a MacBook (no NVIDIA GPU). A tutorial says `load_in_4bit=True` with bitsandbytes. What happens, and what should you use instead?
5. What's the difference between a GPTQ/AWQ quant and a bitsandbytes quant in terms of *when* quantization happens and what you can distribute?
6. Name two task types where you'd be extra suspicious of int4 quality and consider stepping up to int8, and one where int4 is almost certainly fine.

### Answer key

1. `params × bytes/param`: fp16 = 34 × 2 = **68 GB**, int8 = 34 × 1 = **34 GB**, int4 = 34 × 0.5 = **17 GB**. Only **int4 (~17 GB)** fits a 24 GB card, and it leaves ~7 GB (before the ~1.3× headroom) for KV cache and activations — enough for modest context/batch. int8 (34 GB) and fp16 (68 GB) both exceed 24 GB.
2. The **KV cache** ran out. Weights are fixed at ~3.5 GB, but a 30k-token context caches keys/values for all 30k tokens (`≈ 2 × layers × 30000 × d_model × 2 bytes`), which can be many GB — enough to blow past 12 GB on top of weights + activations. You sized to weights-only and forgot that KV grows linearly with context length (and batch).
3. Each step fp16 → int8 → int4 roughly **halves memory**. The quality cost stays small down to 4-bit (often a few percent, frequently imperceptible on chat/extraction), so int4 captures nearly all the memory savings for a cheap quality tax. Below 4-bit (int3/int2), rounding error finally swamps the signal and quality **falls off a cliff**, while the incremental memory saving shrinks. int4 sits at the knee of the curve: maximum savings still on the flat part of the quality line.
4. **bitsandbytes is CUDA-only**, so on a Mac (Apple Silicon / no NVIDIA GPU) it errors out — it can't load. Use **GGUF via llama.cpp or Ollama** instead, which runs 4-bit inference on CPU/Metal without CUDA.
5. **GPTQ/AWQ** quantize **ahead of time** using calibration data, producing a **saved, distributable artifact** (a pre-quantized model you download and serve). **bitsandbytes** quantizes **on-the-fly at load time** from the fp16 weights — nothing pre-quantized to distribute, just a flag (`load_in_4bit=True`); convenient, but you ship/load the full-precision source and quantize each time, and quality is typically a touch below a well-calibrated GPTQ/AWQ.
6. Be suspicious on **long multi-step reasoning/math** (errors compound over steps) and **code generation** (one wrong token breaks the program); also rare long-tail facts and very long contexts. int4 is almost certainly fine for **casual chat, summarization, or simple extraction/classification**, where the small rounding error is invisible.
