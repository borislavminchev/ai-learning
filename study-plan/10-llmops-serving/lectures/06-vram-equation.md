# Lecture 6: The VRAM equation — weights, KV cache, activations, and max concurrency

> This is the most valuable hour of the phase because it lets you answer, *on paper, before you spend a dollar*, two questions that otherwise cost you a rented GPU and a wasted afternoon: "Will this model even fit?" and "How many concurrent users can I serve at my target context length?" Almost every engineer new to serving assumes the weights are the whole story, rents a 24GB card for a 7B model, sets `--max-num-seqs 256`, and gets a KV-cache OOM on startup. After this lecture you'll decompose VRAM into its four real parts, compute the KV cache per token straight from a model's `config.json`, invert the budget to solve for max concurrency, and confirm your hand estimate against vLLM's own startup log — landing within ~20% every time.

**Prerequisites:** what a KV cache is and why decoding is one-token-at-a-time (Lecture 2); rough transformer anatomy (layers, attention heads, hidden size); comfort with GB arithmetic; you've launched vLLM once (Lecture 1). · **Reading time:** ~30 min · **Part of:** Phase 10 (LLMOps: Serving, Optimization & Deployment) Week 2

---

## The core idea (plain language)

When you put a model on a GPU, the VRAM gets spent on four things, and only four:

1. **Weights** — the model's parameters, loaded once, sitting there the whole time. Fixed. Predictable to the byte.
2. **KV cache** — the running memory of the conversation so far, one chunk per token *per active sequence*. This grows with how long the prompts/outputs are and how many users you're serving at once. **This is the part that moves, and it is almost always what caps concurrency.**
3. **Activations** — scratch space for the math of the current forward pass. Transient, comparatively small at inference, freed immediately. Leave headroom; don't over-model it.
4. **Framework overhead** — CUDA context, the inference engine's buffers, CUDA graphs, fragmentation slack. A gigabyte or two you budget for and forget.

The headline you must internalize: **weights decide whether the model fits at all; the KV cache decides how many people it can serve.** Newcomers obsess over the first and get blindsided by the second. A 7B model in fp16 is ~14GB of weights — but on a 24GB card, whether you serve 4 users or 40 depends entirely on how you budget the remaining ~8GB of KV cache against your context length.

And there's one architectural fact that changed the whole economics of serving: **GQA (grouped-query attention)**. Modern models use far fewer *key/value* heads than *attention* heads, which shrinks the KV cache by a large multiple. That single design choice is *why* a Llama-3 or Qwen-2.5 model can serve dozens of concurrent users on one card where an old MHA (multi-head attention) model of the same size could serve a handful. Once you see the KV formula, you'll see exactly why.

## How it actually works (mechanism, from first principles)

### Part 1 — Weights: params × bytes-per-param

This is the easy, fixed part. Every parameter is a number, and the number of bytes it takes depends on the datatype you load it in:

| Datatype | Bytes/param | Notes |
|---|---|---|
| fp32 | 4 | Rarely used for inference; wasteful |
| fp16 / bf16 | 2 | The default for serving open models |
| fp8 | 1 | Native on H100/H200; near-lossless with care |
| int4 (AWQ / GPTQ) | ~0.5 | 4 bits + small quant-scale overhead |

So `weights_GB ≈ params_billions × bytes_per_param`. Worked numbers to memorize:

- **7B fp16:** 7 × 2 = **~14 GB**
- **7B int4:** 7 × 0.5 = **~3.5 GB**
- **70B fp16:** 70 × 2 = **~140 GB** (does not fit on one 80GB card — needs tensor parallelism or quantization)
- **70B int4:** 70 × 0.5 = **~35 GB** (now fits comfortably on one 80GB card)

The int4 rows are approximate — AWQ/GPTQ carry a little metadata (group-wise scales/zeros), so real footprints run maybe 5–10% above the clean 0.5 bytes/param. Close enough for capacity planning.

That's the whole weights story. The instant you've done this arithmetic you know whether the model fits *before* KV cache. If `weights_GB` alone is already close to your card size, stop — you have no room to serve anyone, and you need a smaller model, a quant, or more GPUs.

### Part 2 — KV cache: the part that actually caps concurrency

Every token that's already been processed leaves behind, at every layer, a **key** vector and a **value** vector so future tokens can attend to it. That's the KV cache. It grows by a fixed amount *per token, per sequence*, and it lives in VRAM for as long as that sequence is active.

The per-token cost:

```
kv_bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_elem
                     ↑
                     K and V (two of them)
```

Everything on the right you read straight out of the model's `config.json` on the Hugging Face Hub:

- `num_hidden_layers` → **n_layers**
- `num_key_value_heads` → **n_kv_heads** (this is the GQA number — the one that matters)
- `hidden_size` and `num_attention_heads` → **head_dim** = `hidden_size / num_attention_heads`
- `bytes_per_elem` = 2 for an fp16/bf16 KV cache (the common default, even for quantized *weights* — the KV cache is usually kept at fp16 unless you explicitly enable fp8 KV)

> **Watch the two head counts.** `num_attention_heads` (query heads) sets `head_dim`. `num_key_value_heads` (KV heads) sets how many K/V vectors you actually store. In a GQA model these differ — and the KV cache is driven by the *smaller* one. Using `num_attention_heads` in the KV formula is the single most common way people over-estimate KV by 4–8×.

Once you have `kv_bytes_per_token`, the pool you need is just multiplication:

```
kv_pool_bytes ≈ kv_bytes_per_token × context_length × concurrent_sequences
```

Here `context_length` is the *total* tokens per sequence you're budgeting for (prompt + generated). If you tune vLLM with `--max-model-len 8192`, that 8192 is your per-sequence ceiling.

### GQA vs MHA — why modern models serve so many users

Let's make the GQA point concrete with a Llama-3-8B-shaped config:

- `num_hidden_layers` = 32
- `num_attention_heads` = 32
- `num_key_value_heads` = **8** (GQA: 8 KV heads shared across 32 query heads)
- `hidden_size` = 4096 → `head_dim` = 4096 / 32 = 128
- bytes = 2

**GQA (real config, 8 KV heads):**
```
kv/token = 2 × 32 × 8 × 128 × 2  = 131,072 bytes ≈ 128 KB/token
```

Now imagine the *same model* built with old-style MHA, where every query head has its own KV head — `n_kv_heads` = 32:

**MHA (hypothetical, 32 KV heads):**
```
kv/token = 2 × 32 × 32 × 128 × 2 = 524,288 bytes ≈ 512 KB/token
```

Same layer count, same head_dim — but MHA costs **4× the KV memory per token** (32/8). Flip that around: with the same KV pool, GQA lets you serve **4× as many concurrent sequences**, or 4× the context length, or some mix. That 4× (some models go 8×) is not a micro-optimization — it is *the* reason a single 24GB card can hold dozens of chat sessions on a modern model. When you read "this model uses GQA" in a release blog, read it as "this model was built to be cheap to serve at concurrency."

```
     32 query heads                 32 query heads
     |  |  |  | ... (32)            |  |  |  | ... (32)
     v  v  v  v                      \ | /   \ | /   ...
   [K,V][K,V][K,V]...(32)            [K,V]   [K,V] ... (8 groups)
   MHA: 32 KV heads stored        GQA: 8 KV heads stored  -> 4x smaller cache
```

### Part 3 — Activations: leave headroom, don't over-model

Activations are the transient tensors of the current forward pass — the intermediate results flowing through each layer. During training they're huge (you keep them for the backward pass). During *inference* they're small and short-lived: computed, used, freed, layer by layer. For a decode step you're processing very few tokens at a time, so activation memory is a fraction of a GB. Even a chunked prefill of a long prompt touches only a chunk's worth at once.

The engineering move is *not* to model this precisely. Reserve a headroom buffer — call it ~1–2 GB for a 7B-class model on a single card — and move on. vLLM's `--gpu-memory-utilization` already bakes a fudge factor in: it caps how much of the card the engine will touch (e.g., 0.90 = 90%), leaving the rest for activations, the CUDA context, and fragmentation slack. Over-modeling activations is a classic newcomer time-sink with no payoff.

### Part 4 — Framework overhead

The CUDA context alone is a few hundred MB before you load a single weight. Add the engine's own workspace, NCCL buffers if you're multi-GPU, and CUDA graph capture. Budget ~1–2 GB and stop thinking about it. Again, `--gpu-memory-utilization < 1.0` is your safety margin for this plus activations.

### Assembling the budget and inverting it

Put it together:

```
total_VRAM ≈ weights + kv_pool + activations + framework_overhead
```

The whole reason we care is to solve for concurrency. Rearrange to isolate the KV pool, then divide by the per-sequence KV cost:

```
kv_pool_available   = (card_VRAM × gpu_mem_util) − weights − headroom
max_num_seqs        ≈ kv_pool_available / (kv_bytes_per_token × context_length)
```

That `max_num_seqs` is your predicted ceiling on concurrent sequences — the number you'd hand to vLLM's `--max-num-seqs`. Everything in it is knowable before you rent anything.

## Worked example

**Target:** serve `Llama-3-8B` (fp16) on a single **24 GB** card (A10/L4), at **8k** context, and find max concurrency.

**Step 1 — Weights.** 8B × 2 bytes ≈ **16 GB**. (Slightly over the clean 7×2; use the real param count. Call it 16 GB.)

**Step 2 — Available for KV.** With `--gpu-memory-utilization 0.90`:
```
usable        = 24 × 0.90            = 21.6 GB
headroom      (activations + overhead) ≈ 1.5 GB
kv_pool_avail = 21.6 − 16 − 1.5      = 4.1 GB
```

**Step 3 — KV per token** (from the config above): **128 KB/token** = 131,072 bytes.

**Step 4 — KV per sequence at 8k context:**
```
128 KB/token × 8192 tokens ≈ 1,024 MB = 1.0 GB per full sequence
```

**Step 5 — Max concurrency:**
```
max_num_seqs ≈ 4.1 GB / 1.0 GB ≈ 4 sequences
```

Four concurrent full-length sequences. That feels low — and it teaches the real lesson: **at long context, the KV cache per sequence is enormous, so concurrency is tiny.** Now watch what levers do:

- **Halve the context to 4k:** KV/seq drops to ~0.5 GB → **~8 sequences**. Concurrency doubles for free by right-sizing `--max-model-len`.
- **Quantize weights to int4 (AWQ):** weights drop 16 → ~4.5 GB, freeing ~11.5 GB more for KV → kv_pool ≈ 15.6 GB → at 8k, **~15 sequences**. Roughly 4× the concurrency from quantization alone (re-eval quality — see the note below).
- **Now the MHA counterfactual:** if this model used MHA (512 KB/token), each 8k sequence would cost 4 GB, and your 4.1 GB pool would serve **one** sequence. GQA is the difference between "serves 4" and "serves 1" here.

**Step 6 — Confirm against vLLM's log.** On startup vLLM prints something like:
```
GPU blocks: 2618, CPU blocks: 2048
# or, in newer builds:
Maximum concurrency for 8192 tokens per request: 5.11x
```
vLLM's KV cache is paged into **blocks of ~16 tokens** each (Lecture 2's PagedAttention). Convert its block count to a sanity check:
```
tokens_of_kv = gpu_blocks × block_size = 2618 × 16 ≈ 41,888 tokens of KV capacity
seqs_at_8k   = 41,888 / 8192 ≈ 5.1 sequences
```
Our hand estimate said ~4; vLLM says ~5. **Within 20% — target met.** The gap is our conservative headroom guess; tighten `headroom` and you converge. If vLLM's "Maximum concurrency" line is printed directly, even better — compare it to your `max_num_seqs` number straight up.

If your hand number and vLLM's differ by *more* than ~20%, the usual culprits are: you used `num_attention_heads` instead of `num_key_value_heads` (over-estimate), you forgot the KV cache is fp16 not the weights' int4 (under-estimate), or your headroom guess was way off.

## How it shows up in production

- **The startup OOM cliff.** `--max-num-seqs 256 --max-model-len 32768` on a 24GB card doesn't fail gracefully — vLLM computes the KV pool it would need, sees it can't reserve the blocks, and dies (or crashes mid-load). You *predicted* this on paper. The fix is always one of: lower `--max-model-len`, lower `--max-num-seqs`, quantize the weights, or (last resort, risky) raise `--gpu-memory-utilization`.
- **The mysterious concurrency ceiling.** Requests past a certain concurrency don't error — they *queue*. vLLM admits sequences up to the KV pool, then the rest wait, and your p95 TTFT climbs. If you didn't do the KV math, this looks like "the server got slow" when really you hit the KV wall you could have computed.
- **Long context silently murders concurrency.** A product decision to "support 32k context" is a decision to cut concurrency ~4× versus 8k on the same hardware, because KV pool per sequence scales linearly with context. Finance and product often don't know they made that trade. You do — put it in the capacity doc.
- **The quantization ↔ concurrency lever.** When throughput is the goal and you're KV-bound, quantizing weights isn't just about fitting the model — it *donates the freed VRAM to the KV pool*, buying you more concurrent users. This is a real, common production tuning move.
- **Quantizing the KV cache itself.** There's a second, orthogonal lever the weights-only story misses: shrink the KV cache elements. vLLM exposes `--kv-cache-dtype fp8`, which stores K and V at 1 byte instead of 2 — halving `bytes_per_elem` in the KV formula and therefore *doubling* the tokens your pool holds (≈2× concurrency or 2× context). It's cheaper to reach for than re-quantizing weights and is often near-lossless, but it's not free: KV quant can nick quality on long-context recall, so re-eval the same as with weight quant. On H100/H200 the fp8 KV path is well-supported; treat it as "quantize the fast-growing term, not just the fixed one."
- **GQA models are why your cost/user is low.** When you compare a modern model to a legacy MHA one and the new one serves far more users per dollar, the KV formula is the receipt.

## Common misconceptions & failure modes

- **"Weights are the VRAM budget."** Weights are the *floor*. Concurrency lives in what's left over, and it's spent on KV cache. Fits-vs-serves are different questions.
- **"I'll model activations precisely."** Waste of time at inference. Reserve headroom, let `--gpu-memory-utilization` absorb it.
- **Using `num_attention_heads` in the KV formula.** On a GQA model this over-estimates KV by the GQA ratio (4×, 8×) and makes you buy too much GPU. Always use `num_key_value_heads`.
- **Assuming the KV cache follows the weight quantization.** int4 *weights* do not imply int4 *KV*. Unless you explicitly turn on fp8/int8 KV cache, KV stays fp16 (2 bytes/elem). Budget it that way.
- **Forgetting KV is per-*token*, and that output tokens count too.** A sequence's KV grows as it generates. Budget `max_model_len` as prompt + completion, not just the prompt.
- **Cranking `--gpu-memory-utilization` to 0.98 to "fit more."** You steal from activations/CUDA/fragmentation slack and OOM under load, not at startup — the worst time. 0.90 is a sane default; 0.95 is aggressive.
- **Ignoring that vLLM reserves the KV pool up front.** vLLM grabs its KV blocks at startup based on your flags. It's not lazy. That's *why* it OOMs on launch, not on the 50th request — and why the startup log is your ground truth.

## Rules of thumb / cheat sheet

- **Weights GB ≈ params_B × bytes/param.** fp16=2, fp8=1, int4≈0.5. (7B fp16≈14GB, 70B fp16≈140GB, 70B int4≈35GB.)
- **KV/token = 2 × n_layers × n_kv_heads × head_dim × bytes.** Read `num_hidden_layers`, `num_key_value_heads`, `hidden_size`/`num_attention_heads` from `config.json`. KV is fp16 (2 bytes) by default.
- **7B/8B-class GQA models:** ballpark **~100–130 KB/token** of KV. At 8k context that's ~0.8–1.0 GB *per sequence*.
- **max_num_seqs ≈ (card_GB × util − weights − ~1.5GB headroom) / (KV_per_token × context_len).**
- **GQA ratio** = `num_attention_heads / num_key_value_heads`. That's your KV savings (and concurrency multiplier) vs MHA. Commonly 4× or 8×.
- **vLLM block size ≈ 16 tokens.** `gpu_blocks × 16 = total KV tokens`; divide by context_len for concurrency. Target agreement with your estimate within **~20%**.
- **Default `--gpu-memory-utilization 0.90`.** Raise to 0.95 only knowingly; never near 1.0.
- **KV-OOM at startup → in order: lower `--max-model-len`, lower `--max-num-seqs`, quantize, then (last) raise util.**
- **Quantizing weights frees VRAM for KV = more concurrency — but re-run your eval suite; a cheaper cost/1M is worthless if quality dropped.**
- **`--kv-cache-dtype fp8` halves the KV element size (2→1 byte) = ~2× the tokens per pool = ~2× concurrency/context.** Orthogonal to weight quant; re-eval long-context quality after enabling.

## Connect to the lab

This lecture is the direct spec for **Week 2, Step 1 — the VRAM estimator** (`week2-economics/vram_estimate.py`). Your script reads a model's `config.json`, prints weights GB, KV GB per 1k-tokens-per-sequence, and **max concurrent sequences** for a target context length and card size — exactly the invert-the-budget arithmetic above. The Definition of Done is that your predicted `max-num-seqs` lands **within ~20%** of what your Week-1 vLLM server reported in its startup KV-block log. Build the estimator, then diff it against the log; when they agree, you've earned the ability to spec hardware before renting it.

## Going deeper (optional)

- **vLLM docs** (docs.vllm.ai) — read the "conserving memory," "optimization/performance tuning," and PagedAttention pages. The startup logging of KV blocks / max concurrency is documented there and in the engine's log output.
- **Model `config.json` on the Hugging Face Hub** — open the files tab of any model (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`) and read the real numbers. This is the primary source; do the arithmetic against actual configs.
- **GQA original paper:** "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al.) — search that title; it's the canonical explanation of why KV heads < query heads.
- **Horace He, "Making Deep Learning Go Brrrr From First Principles"** — the memory-bound-vs-compute-bound mental model that underpins *why* KV bandwidth dominates decode (pairs with next lecture). Search the title.
- **AWQ and GPTQ papers/repos** — search "AWQ activation-aware weight quantization" and "GPTQ" for the int4 methods behind the ~0.5 bytes/param row.
- **Search queries:** "vLLM available KV cache blocks log", "transformer KV cache memory formula GQA", "vLLM gpu-memory-utilization max-num-seqs OOM".

## Check yourself

1. On a 24GB card serving a 7B fp16 model, you raise concurrency from 4 to 40 users. Which VRAM term grows, and which stays flat?
2. A model's `config.json` shows `num_attention_heads: 32`, `num_key_value_heads: 8`, `hidden_size: 4096`, `num_hidden_layers: 32`. Compute `head_dim` and the KV bytes per token (fp16). Which head count did you use and why?
3. Your hand estimate says max 4 concurrent sequences at 8k context; vLLM's log implies ~5. Is your estimate acceptable, and what's the most likely source of the gap?
4. You quantize a 7B model from fp16 to int4 on a 24GB card. Does that let you serve more concurrent users, and through what mechanism? What must you check afterward?
5. Same model built with MHA instead of GQA (32 KV heads vs 8): what happens to per-token KV cost and to max concurrency at a fixed KV pool?
6. Why does vLLM OOM at *startup* when you set `--max-num-seqs` too high, rather than on the 100th request?

### Answer key

1. The **KV cache** term grows (linearly with concurrent sequences); **weights** stay flat (loaded once, ~14GB regardless of users). Activations/overhead stay roughly flat. This is the whole "KV caps concurrency" point.
2. `head_dim` = 4096 / 32 = **128**. KV/token = 2 × 32 (layers) × 8 (**kv** heads) × 128 × 2 bytes = **131,072 bytes ≈ 128 KB**. Use `num_key_value_heads` (8), not `num_attention_heads` (32), because GQA stores only the KV heads; using 32 would over-estimate by 4×.
3. **Acceptable** — 4 vs 5 is within the ~20% target. The most likely gap is a **conservative headroom guess** (you reserved more for activations/overhead than vLLM actually needed); tighten it and the numbers converge. (Other suspects: block-size rounding, exact weight size.)
4. **Yes.** int4 cuts weights from ~14GB to ~3.5GB, and that **freed VRAM is donated to the KV pool**, which is what concurrency scales with — roughly 4× more room for KV, so many more concurrent sequences. Afterward you **must re-run your eval suite**: quantization can degrade quality, and cheaper serving is meaningless if the outputs got worse.
5. Per-token KV cost **quadruples** (32/8 = 4×: from ~128KB to ~512KB/token). At a fixed KV pool, **max concurrency drops ~4×**. This is exactly why GQA models serve so many more users per card than legacy MHA models.
6. vLLM **reserves the entire KV block pool up front** at startup based on your flags (it's not lazy). If `--max-num-seqs × --max-model-len` implies more KV blocks than fit in the card after weights, it can't reserve them and fails immediately — that's why the startup log is your ground-truth capacity check, and why the OOM is a launch-time event, not a mid-traffic surprise.
