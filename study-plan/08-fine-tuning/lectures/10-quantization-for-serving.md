# Lecture 10: Quantization for Serving — GPTQ, AWQ, GGUF, and fp8

> You trained a QLoRA adapter on a 4-bit base, merged it into a clean bf16 model, and it scores well. Now you have to *serve* it — cheaply, fast, and without silently getting dumber. This is a different problem from QLoRA-for-training, and the tools have different names for good reasons. This lecture draws the sharp line between the two, nails down the one pipeline order that keeps you out of trouble (**merge → quantize → re-evaluate**), and walks the three serving formats you will actually reach for in 2025-2026: **GPTQ/AWQ** for GPU servers, **GGUF** for laptops and Ollama, and **fp8** for H100-class hardware. After this you can pick the right format for a given venue in one sentence, run the AutoAWQ and llama.cpp flows end to end, and — the part that separates engineers from tinkerers — build the bf16-vs-AWQ-vs-GGUF scorecard that proves your quantized model didn't lose the accuracy you worked for.

**Prerequisites:** Lecture 5 (QLoRA — 4-bit NF4 training), the LoRA adapter-lifecycle lecture (merge vs hot-swap), Phase 0 quantization basics (fp16/int8/int4, what "bits per weight" means), a working eval harness from Phase 7 (JSON-valid rate + task accuracy) · **Reading time:** ~30 min · **Part of:** Phase 08 (Model Adaptation & Fine-Tuning) Week 3

---

## The core idea (plain language)

There are two completely different reasons to quantize a model, and confusing them is the single most common way people ship a broken fine-tune.

**Quantization for *training* (QLoRA)** compresses the frozen base to 4-bit NF4 so a fine-tune fits on a 15GB card. The 4-bit weights are scaffolding — they exist to make gradient descent affordable, and the *thing you actually keep* is the tiny bf16 adapter trained on top. NF4 is optimized for "survives being dequantized back to bf16 for the forward pass," not for fast inference.

**Quantization for *serving* (this lecture)** compresses a *finished* model so it costs less VRAM and runs faster *at inference time*. There is no adapter, no gradient, no training. You take one merged fp16/bf16 checkpoint and produce a smaller artifact optimized for a specific runtime — vLLM, transformers, llama.cpp, Ollama, TensorRT-LLM. The formats here (GPTQ, AWQ, GGUF, fp8) are inference formats. You would never train through them.

The bridge between the two worlds is a single rule you must never violate: **merge the adapter into an fp16/bf16 base FIRST, then quantize *that* merged model.** Quantize the 4-bit training base and you get **double-quantization damage** — you're quantizing something that's already lossy, compounding two rounding steps, and the adapter (trained to correct one specific dequantized base) is now bolted onto differently-rounded numbers. The output is garbage, and it can be *subtly* garbage, which is worse.

The other half of the discipline: **quantization is lossy, so you re-evaluate after doing it.** Dropping from 16 bits to 4 bits throws away ~75% of the numeric information in every weight. Most of the time the model barely notices. Sometimes it costs you 3-4 points of task accuracy or JSON-validity — and it does so *silently*, because the model still produces fluent, plausible text. The only way to know is to run your held-out set through the quantized model and compare. Never ship a quantized model you haven't re-scored.

---

## How it actually works (mechanism, from first principles)

### What "quantize a weight" actually means

A weight is a floating-point number. fp16/bf16 store it in 16 bits. Int4 quantization stores it in 4 bits — 16 possible values instead of 65,536. You can't represent arbitrary numbers in 16 buckets, so you group weights into blocks, and for each block store a **scale** (and often a **zero-point**) that maps the 16 integer levels onto the block's actual range.

```
original block (fp16):   [-0.42, 0.10, -0.05, 0.31, ...]   (16 bits each)
                         ↓ find range, pick scale
scale = 0.42/7                          (map to int4 range -8..7)
quantized (int4):        [ -7,    2,    -1,    5, ...]      (4 bits each)
stored per block:        the 4-bit codes + one fp16 scale
dequantize at runtime:   code × scale ≈ original
```

Bits-per-weight isn't exactly 4 because of that per-block scale metadata. A common "4-bit" scheme with group size 128 costs roughly **4.5 bits/weight** once you count scales. That's why an "8B 4-bit" model is ~4.5-5GB on disk, not exactly 4GB. Memorize the shape: **fp16 ≈ 2 bytes/param, 8-bit ≈ 1 byte/param, 4-bit ≈ 0.5-0.6 byte/param.** An 8B model: ~16GB fp16, ~8GB int8, ~5GB int4.

The whole art of a quantization *method* is: given that you must throw away information, **which** information do you throw away, and how do you protect the weights that matter most?

### GPTQ and AWQ — calibration-based 4-bit for GPUs

Naive "round every weight to the nearest 4-bit level" (called RTN, round-to-nearest) works okay but leaves accuracy on the table because it treats every weight as equally important. GPTQ and AWQ are smarter, and both are **calibration-based**: you feed the quantizer a few hundred sample texts, run them through the model, and use the observed *activations* to decide how to quantize.

- **GPTQ** (2022) quantizes weights one column at a time, and after rounding each column it *adjusts the not-yet-quantized weights* to compensate for the error just introduced (using second-order/Hessian information from the calibration activations). Effect: the quantization error gets absorbed rather than accumulating. Strong at 4-bit, well supported everywhere.
- **AWQ** (Activation-aware Weight Quantization, 2023) starts from a sharp observation: a *small fraction* of weight channels (the ones multiplied by large-magnitude activations) matter far more than the rest. AWQ finds those salient channels from the calibration data and **scales them up before quantizing** so they land on finer-grained levels, protecting them from rounding damage. In practice AWQ often edges out GPTQ on quality at the same 4 bits and is the current default for many vLLM deployments.

Both need calibration data. Use ~128-512 samples that *look like your real traffic* — for a JSON-emitting ticket router, calibrate on real ticket prompts, not random web text. Calibrating on the wrong distribution is a quiet way to lose accuracy on exactly the inputs you care about. Both target **GPU inference** and are consumed by vLLM and transformers.

### GGUF — one file, CPU+GPU, built for local

GGUF is the file format of **llama.cpp** (and therefore Ollama, LM Studio, Jan). It is not one quantization algorithm — it's a container that packs weights, tokenizer, chat template, and metadata into a **single file**, with many quant levels to choose from. The naming looks cryptic but decodes cleanly:

```
q4_k_m  →  4-bit, "k-quant" (mixed precision within the file), Medium size
q5_k_m  →  5-bit k-quant, medium
q8_0    →  8-bit, legacy simple scheme
q2_k    →  2-bit — tiny, noticeably degraded, avoid for real work
```

The `_k` "k-quants" are the modern default: they don't use one bit-width for the whole model. They spend more bits on the layers that matter (attention, some MLP) and fewer elsewhere. **`q4_k_m` is the workhorse** — the standard "just give me a good small model" choice, roughly 4.5-5 bits/weight effective. `q5_k_m` if you have room and want a hair more fidelity; `q8_0` when you want near-lossless and don't care about size.

GGUF's superpower is that llama.cpp runs it on **CPU**, or CPU+GPU with layers offloaded to whatever VRAM you have. That's why it's the format for a laptop demo with no GPU — and why Ollama, built on llama.cpp, uses it.

### fp8 — for the newest hardware

fp8 is 8-bit *floating point* (not integer): an 8-bit number with an exponent and mantissa, so it keeps floating-point dynamic range at a quarter of fp16's size. It matters because **H100/H200/Ada-class GPUs have native fp8 tensor cores** — the hardware does fp8 matmuls at roughly 2× fp16 throughput. On that hardware, fp8 is close to lossless (much better fidelity than 4-bit int) *and* faster. On older cards (A100, T4, consumer 30-series) there's no fp8 acceleration, so it's pointless. fp8 is the "I have H100s and want max throughput with minimal quality loss" answer, served by vLLM and TensorRT-LLM.

---

## Worked example

You have `ticket-router-v1-merged/` — an 8B model, bf16, ~16GB on disk, that scores **94% category accuracy** and **99% JSON-valid** on your 100 held-out tickets. Now produce serving artifacts.

**Path A — AWQ for a vLLM GPU server (AutoAWQ):**

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "models/ticket-router-v1-merged"   # the MERGED bf16 model
quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tok   = AutoTokenizer.from_pretrained(model_path)

# calibrate on ~128-512 samples that resemble real traffic
model.quantize(tok, quant_config=quant_config, calib_data=my_ticket_prompts)
model.save_quantized("models/ticket-router-v1-awq")
tok.save_pretrained("models/ticket-router-v1-awq")
```

Result: ~5GB on disk. Serve with `vllm serve models/ticket-router-v1-awq --quantization awq`.

**Path B — GGUF q4_k_m for a laptop (llama.cpp + Ollama):**

```bash
# 1. convert the merged HF model to a full-precision GGUF
python llama.cpp/convert_hf_to_gguf.py models/ticket-router-v1-merged \
    --outfile models/ticket-router-v1-f16.gguf --outtype f16

# 2. quantize that GGUF down to q4_k_m
./llama.cpp/llama-quantize \
    models/ticket-router-v1-f16.gguf \
    models/ticket-router-v1-q4_k_m.gguf q4_k_m
```

Note the two-step shape: **convert → quantize**. You first produce a big f16 GGUF, then compress it. Load it into Ollama with a Modelfile:

```
# Modelfile
FROM ./models/ticket-router-v1-q4_k_m.gguf
TEMPLATE """{{ .System }}<|user|>{{ .Prompt }}<|assistant|>"""
PARAMETER temperature 0
PARAMETER stop "<|user|>"
```

```bash
ollama create ticket-router -f Modelfile
ollama run ticket-router "Route this ticket: ..."
```

Now **re-evaluate** — the non-negotiable step. Run the same 100 held-out tickets through all three and build the scorecard:

| Variant | Bits/wt | Disk | JSON-valid | Category acc | tokens/sec* |
|---|---|---|---|---|---|
| bf16 (merged) | 16 | ~16 GB | 99% | 94% | 45 (L4 GPU) |
| AWQ 4-bit | ~4.5 | ~5 GB | 98% | 93% | 110 (L4 GPU) |
| GGUF q4_k_m | ~4.5 | ~5 GB | 97% | 92% | 22 (laptop CPU) |

*\*Illustrative numbers to show the shape of the tradeoff — measure your own; do not cite these.*

This table is the deliverable. It shows the pattern you should *expect*: 4-bit costs ~1-2 points here (acceptable), gives ~3× less VRAM, and on GPU roughly doubles throughput. The laptop GGUF is slower in tokens/sec but runs on *zero* GPU budget — that's its whole point. If AWQ had dropped JSON-valid to 90%, this table is exactly what stops you from shipping it.

---

## How it shows up in production

**VRAM and cost.** Quantization is the lever that fits a model on cheaper hardware. An 8B bf16 model (~16GB weights) plus KV cache barely fits a 24GB card and can't fit a 16GB one with real batch sizes. At 4-bit (~5GB) you leave 10-19GB for KV cache and concurrency — which means higher batch sizes, higher throughput, and often you can drop from an A100 to an L4/A10 and cut your GPU bill by more than half. That's the actual business case.

**Latency and throughput.** For batch-1 interactive inference, LLM decoding is **memory-bandwidth bound** — the bottleneck is reading weights from VRAM each token. Half the bytes ≈ roughly up to ~2× faster decode. That gain shrinks as batch size grows (you become compute-bound) and depends on kernel quality. fp8 on H100 is the exception that also speeds up the *compute*.

**The silent-quality trap.** A quantized model rarely fails loudly. It produces fluent text that's *slightly* more likely to drop a required JSON field, mis-route an edge-case ticket, or lose a few points on your hardest examples. Loss curves and eyeballing 5 outputs won't catch a 3-point JSON-valid regression across a real distribution. Your held-out scorecard is the only instrument that will. Teams that skip it ship silently-worse models and find out from users.

**Format lock-in to venue.** AWQ/GPTQ artifacts run in vLLM/transformers on GPU; they do *not* run in Ollama. GGUF runs in llama.cpp/Ollama; it does *not* run in vLLM. Pick the format for the runtime you're actually deploying to — you can't cross the streams. If you need both a GPU server and a laptop demo, you produce *both* artifacts from the same merged model.

---

## Common misconceptions & failure modes

- **"I trained in 4-bit QLoRA, so I'll just serve the 4-bit training base."** No. That base is NF4 scaffolding *plus* a separate adapter. Serving it means either running the training-time quant (slow, not an inference format) or — worse — merging the adapter into the 4-bit base. Merge into **bf16**, then quantize with an inference method.
- **"Quantizing twice is fine, it's already small."** Double quantization compounds two rounding errors and applies the adapter's correction to numbers it wasn't trained against. This is the double-quantization damage the pitfalls warn about. One quantization, applied to a clean fp16 merge.
- **"4-bit is 4-bit — the method doesn't matter."** RTN, GPTQ, and AWQ at the same 4 bits can differ by several accuracy points. AWQ's activation-aware scaling and GPTQ's error compensation exist precisely because naive rounding loses more.
- **"Skip calibration data / use random text."** GPTQ and AWQ tune to the calibration distribution. Calibrate on text unlike your traffic and you protect the wrong weights — quiet accuracy loss on your real inputs.
- **"GGUF q2_k to save space."** 2-bit is visibly degraded. For anything you care about, `q4_k_m` is the floor; `q5_k_m`/`q8_0` when fidelity matters.
- **"fp8 will speed up my T4/A100 deploy."** Only H100/Ada-class hardware has native fp8 acceleration. On older cards it buys you nothing.
- **"It generated valid text, so quantization was fine."** Fluency is not correctness. Re-run the held-out set and compare the numbers.

---

## Rules of thumb / cheat sheet

- **Pipeline order, always:** merge adapter → fp16/bf16 base → quantize *that* → **re-evaluate** on held-out.
- **Never** quantize or merge into the 4-bit training base.
- **Decision rule:**
  - Zero-cost laptop / local demo, CPU-only, Ollama → **GGUF `q4_k_m`**.
  - GPU server via vLLM/transformers → **AWQ** (or GPTQ) 4-bit.
  - H100/Ada hardware, want max throughput near-lossless → **fp8**.
  - Need both laptop and GPU server → produce **both** artifacts from the same merge.
- **Bits/param cheat:** fp16 ≈ 2 B, int8 ≈ 1 B, int4 ≈ 0.5-0.6 B. 8B model: ~16 / ~8 / ~5 GB.
- **GGUF names:** `q4_k_m` = default workhorse; `q5_k_m` = a bit sharper; `q8_0` = near-lossless/big; avoid `q2_k`.
- **Calibration:** 128-512 samples that resemble real traffic; wrong distribution = quiet accuracy loss.
- **Expected 4-bit cost (approximate):** ~0-2 points on task/JSON metrics is normal; >3-4 points means try AWQ instead of GPTQ, a smaller quant group size (finer scales), 5-bit/`q5_k_m`, or 8-bit. Always driven by the scorecard, never by faith.
- **Re-eval metrics, minimum three:** JSON-valid rate, task accuracy, tokens/sec — per variant.

---

## Connect to the lab

Week 3's lab step 4-5 is this lecture executed: take your DPO-chosen adapter, merge it into bf16, then produce (a) **AWQ 4-bit** via AutoAWQ and (b) **GGUF `q4_k_m`** via llama.cpp `convert` + `quantize`, and load the GGUF in **Ollama** from a Modelfile to prove local serving. Then run your 100 held-out tickets through **bf16-merged / AWQ / GGUF** and fill in the scorecard — JSON-valid %, task accuracy, tokens/sec — stating the accuracy delta quantization cost you. That table, plus "GGUF runs in Ollama with no GPU," is the Definition of Done.

---

## Going deeper (optional)

- **AutoAWQ** — the standard AWQ quantization library and workflow. Search: `AutoAWQ github` and `vLLM AWQ quantization docs`.
- **AWQ paper** — Lin et al., *"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration."* Search: `AWQ paper arxiv activation-aware`.
- **GPTQ paper** — Frantar et al., *"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers."* Search: `GPTQ paper arxiv Frantar`.
- **llama.cpp** — the GGUF reference implementation, `convert_hf_to_gguf.py`, `llama-quantize`, and the quant-type table. Root: `github.com/ggml-org/llama.cpp`. Search: `llama.cpp quantize k-quants readme`.
- **Ollama** — Modelfile reference and importing GGUF. Root: `ollama.com` and `github.com/ollama/ollama`. Search: `Ollama Modelfile FROM gguf import`.
- **vLLM docs** — serving quantized models (AWQ/GPTQ/fp8) and supported hardware. Root: `docs.vllm.ai`. Search: `vLLM fp8 quantization H100`.
- **Hugging Face quantization overview** — GPTQ/AWQ/bitsandbytes in transformers. Root: `huggingface.co/docs/transformers/main/en/quantization`.

---

## Check yourself

1. State the four-step serving pipeline in order, and explain precisely what goes wrong if you quantize the 4-bit QLoRA training base instead of a merged fp16 model.
2. Why do GPTQ and AWQ need calibration data, but a naive round-to-nearest int4 scheme does not? What does AWQ do with that calibration data specifically?
3. Your teammate wants to demo the model on their GPU-less laptop *and* serve it in a vLLM cluster. Which artifact(s) do you produce, and why can't one file do both jobs?
4. You quantized to AWQ 4-bit and JSON-valid rate dropped from 99% to 94%. List three concrete options to recover accuracy without abandoning quantization.
5. When is fp8 the right call, and why is it pointless on a T4 or A100?
6. Decode `q4_k_m` and explain why it's the default over `q4_0` and over `q2_k`.

### Answer key

1. **Merge adapter → into fp16/bf16 base → quantize that merged model with an inference method → re-evaluate on held-out.** Quantizing the 4-bit training base means either the adapter gets merged onto already-lossy dequantized weights (compounding rounding, and the adapter's correction — trained against one specific dequantization — misfires) or you're re-quantizing something already quantized. Either way you get double-quantization damage: subtly-wrong, fluent-looking output.
2. Naive RTN just rounds each weight independently — no data needed. GPTQ/AWQ use calibration activations to decide *which weights matter and how to protect them*: GPTQ compensates remaining weights for each column's rounding error using second-order info; **AWQ identifies the salient weight channels (those multiplied by large-magnitude activations) and scales them up before quantizing** so they get finer resolution. Wrong calibration distribution → wrong weights protected → quiet accuracy loss on real inputs.
3. Produce **both**: a **GGUF `q4_k_m`** for the laptop (llama.cpp/Ollama, CPU-capable) and an **AWQ (or GPTQ) 4-bit** for the vLLM cluster (GPU). One file can't do both because the formats are tied to runtimes — vLLM doesn't load GGUF, and llama.cpp/Ollama doesn't load AWQ/GPTQ. Both are made from the same merged bf16 model.
4. (a) Try **AWQ instead of GPTQ** (or re-tune calibration data to match traffic); (b) go to a **higher-fidelity quant** — 5-bit / `q5_k_m` or 8-bit — trading some size/speed for accuracy; (c) reduce quant **group size** for finer per-block scales, or fix the calibration set to resemble real prompts. Then re-score. If nothing recovers it and the drop is unacceptable, ship bf16/fp8.
5. fp8 is right on **H100/H200/Ada-class GPUs**, which have native fp8 tensor cores: ~2× fp16 throughput *and* near-lossless quality (far better than 4-bit int). On a T4 or A100 there's no fp8 hardware acceleration, so you pay the quality/complexity without the speedup — use int4 (AWQ/GPTQ) there instead.
6. `q4_k_m` = **4-bit**, **k-quant** (mixed precision *within* the file — more bits on layers that matter, fewer elsewhere), **M**edium size variant. It beats `q4_0` (a legacy uniform 4-bit scheme) because k-quants allocate bits by importance, and it beats `q2_k` because 2-bit is visibly degraded — `q4_k_m` is the sweet spot of size vs quality, hence the community default.
