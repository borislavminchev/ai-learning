# Lecture 12: Diffusion Fundamentals — The Knobs That Matter (Steps, CFG, Sampler, Seed, Negative Prompt)

> You have spent this whole curriculum treating models as functions that take input and produce output. A diffusion model breaks that mental model in an interesting way: it does not compute an image in one shot, it *carves* one out of static over many small steps, and you get to stand at the controls of that carving process. The difference between a demo image and a production asset is almost never the prompt — it is whether you understand what the five real knobs (steps, CFG, sampler, seed, negative prompt) actually do to that loop, and which combinations are wasting your GPU minutes for zero quality. After this lecture you can drive SDXL or FLUX with intent, read a `diffusers` pipeline call and predict its cost and behavior, explain why cranking guidance "fries" an image, know why FLUX ignores your negative prompt, and set up the contact-sheet experiment that turns knob-tuning from superstition into measurement.

**Prerequisites:** basic Python, comfort with the HF ecosystem, the "an image is just tokens/tensors" intuition from this phase's VLM lectures · **Reading time:** ~30 min · **Part of:** Multimodal & Specialized Modalities, Week 3

---

## The core idea (plain language)

A diffusion model is a **denoiser run in a loop**. That is the whole thing. It was trained on one narrow task: "here is an image with some Gaussian noise added; predict the noise." Do that well enough, at every noise level from "barely fuzzy" to "pure static," and you get something surprising for free — you can *generate*. Start from 100% random noise, ask the model "what noise is in this?", subtract a fraction of its answer, and repeat. Each pass the picture is a little less random and a little more like a coherent image. After N passes you have something that was never in the training set but lives on the same manifold as the images that were.

The text prompt steers this. A **text encoder** (CLIP for SDXL, T5 for FLUX) turns your words into vectors, and those vectors are fed to the denoiser at every step as a *condition* — a nudge that says "of all the images you could carve out of this noise, aim for the ones that match *a red bicycle on a beach at sunset*." The denoiser was trained on image+caption pairs, so it has learned to let that condition bias its noise predictions.

Everything you control is either **how many denoising steps you run** (steps), **how hard the prompt is allowed to push the process around** (CFG scale), **the specific math used to take each step** (sampler/scheduler), **what random static you started from** (seed), and **what to actively steer away from** (negative prompt). Five knobs. Get an engineering feel for each and you can debug 90% of "why does my output look wrong" without touching the model internals.

The one mental shift from LLM-land: this is a **budget you spend per image**, not per token. Every step is a full forward pass through a large UNet or transformer. Twenty steps is twenty forward passes. That is where your latency and GPU cost live, and it is why "just use more steps" is not free.

---

## How it actually works (mechanism, from first principles)

### The denoising loop

Picture the loop as a schedule of decreasing noise levels. The **scheduler** hands you a list of timesteps, each with an associated noise level, from most-noisy to least-noisy:

```
  step:   1        2        3       ...      N
  noise: 100% ──▶ 82% ──▶ 66% ──▶  ...  ──▶  ~0%
          │        │        │                 │
   latent x_T ─▶ x_{t-1} ─▶  ...  ──────────▶ x_0
          ▲        ▲
        UNet     UNet    (each arrow = one full forward pass,
     predicts  predicts   conditioned on your text embedding)
      noise     noise
```

Concretely, one iteration:

1. Feed the current noisy latent `x_t` **plus** the text embedding **plus** the timestep `t` into the model.
2. The model outputs a noise prediction.
3. The sampler uses that prediction to compute a slightly-cleaner latent `x_{t-1}`.
4. Repeat until you reach `x_0`, then a VAE decoder turns that latent into actual pixels.

Two efficiency facts matter here. First, this all happens in a compressed **latent space** (SDXL works on a 128×128×4 latent that decodes to 1024×1024 pixels) — that is why it is tractable at all. Second, the **decode to pixels happens once, at the end**; the loop itself never touches full-resolution pixels.

### Steps — resolution of the integration

The denoising loop is numerically solving a differential equation that flows from noise to image. **Steps = how many chunks you slice that path into.** More steps means smaller, more accurate chunks, so fine detail resolves better — up to a point. Past that point the image has converged and extra steps just burn compute redrawing the same thing.

Numeric intuition: going from 10 → 20 steps is usually a *visible* jump in coherence and detail. 20 → 30 is a modest refinement. 30 → 50 is often indistinguishable in a blind test, and 50 → 100 is almost always wasted money. The diminishing-returns curve is steep. Typical production range: **20–40 steps** for SDXL with a good sampler; some samplers get usable results at 15. Latency scales linearly — 40 steps costs roughly 2× the wall-clock of 20 steps, because it is literally 2× the forward passes.

### CFG scale — the prompt's leverage

Classifier-free guidance is the cleverest knob and the one people most misuse. Here is the mechanism without the math: at each step the model actually makes **two** predictions — one *conditioned* on your prompt, one *unconditioned* (empty prompt). CFG then extrapolates:

```
  final = unconditional + cfg_scale × (conditional − unconditional)
```

The term `(conditional − unconditional)` is "the direction the prompt pulls." `cfg_scale = 1` means ignore the prompt's extra pull entirely. `cfg_scale = 7` means push **7× harder** in the prompt's direction than the difference alone suggests. This is why it costs roughly 2× per step — two forward passes — unless the model is guidance-distilled (see FLUX below).

The tradeoff: **low CFG** (2–4) gives diverse, sometimes loosely-related, softer images that may drift from your prompt. **High CFG** (12+) slams hard toward the prompt but pushes the latents into regions the model never saw in training, producing the classic **"fried" look** — blown-out saturation, hard contrast edges, halos, a burnt HDR feel. The sweet spot for SDXL is roughly **5–8**.

Crucial and counterintuitive: **prompt adherence is not monotonic in CFG.** People assume "more guidance = follows my prompt more, always." It does not. Past the sweet spot, the extrapolation over-shoots into artifact territory and the image often adheres to your prompt *worse* while looking worse — you get saturated garbage that has *less* of what you asked for, not more. When a knob's effect reverses on you, that is exactly the kind of thing that wastes an afternoon if you did not know it could happen.

### Sampler / scheduler — how each step is taken

The sampler is the numerical method that turns a noise prediction into the next latent. Different samplers are different ODE solvers, and they trade **quality-per-step** against **speed** and **determinism**:

- **Euler / Euler a** — simple, fast, reliable. "Euler a" (ancestral) injects fresh noise each step, so it keeps changing even at high step counts and is less reproducible; plain Euler converges.
- **DPM++ 2M Karras** — a strong modern default: high quality at low step counts (great at 20–25), widely used in production. The "Karras" part is a noise-schedule choice that spends steps where they matter.
- **DDIM** — older, deterministic, still fine.
- **LCM / Turbo / Lightning** — distilled fast paths that get a usable image in **1–8 steps** by trading some quality/diversity for speed. This is what powers "real-time" generation.

The engineering point: the sampler changes **how many steps you need for a given quality**. A good sampler at 20 steps can beat a weak one at 40 — so picking the sampler is a cost lever, not just an aesthetic one.

### Seed — the initial static

The whole loop starts from a specific block of random noise, and that noise is drawn from a pseudo-random generator you seed. **Same seed + same everything = the same image, bit-for-bit** (on the same hardware/library version). This is your single most important debugging tool: **fix the seed and you can change one other knob at a time and see *only* that knob's effect.** Change the seed and you are looking at a different image *and* different knobs, and you can conclude nothing. Every honest knob experiment holds the seed fixed.

### Negative prompt — steering away

Remember CFG's unconditional prediction? A negative prompt **replaces that empty baseline with something you actively want to avoid.** Instead of "push away from nothing," it becomes "push away from `blurry, extra fingers, watermark, low quality`." Mechanically it is the same subtraction, just with a poisoned reference point. This is why negative prompts are a CFG-era feature: **they only exist because the model runs that second, unconditional pass.**

This is exactly why **FLUX largely does not support negative prompts.** FLUX is *guidance-distilled*: to be fast, it was trained to produce guided-looking output in a **single** forward pass, folding the guidance signal into the model itself rather than computing it at inference from two passes. No second pass, no reference point to poison, no meaningful negative prompt. SDXL, which does run true CFG with two passes, supports negative prompts fully. (FLUX exposes a `guidance_scale` parameter, but it is a distilled embedding-conditioning knob, not the two-pass extrapolation SDXL does — same name, different mechanism.)

---

## Worked example

Let's cost out and reason through a real SDXL call before writing any code.

Target: a 1024×1024 SDXL image, DPM++ 2M Karras, 30 steps, CFG 7, fixed seed.

- **Forward passes:** 30 steps × 2 (CFG's conditional + unconditional) = **60 UNet passes**, then 1 VAE decode. On an A100-class GPU that is roughly 3–6 seconds; on a T4 (free Colab tier) more like 20–40 seconds. Memorize the *shape*, not the exact numbers — they move with hardware and library versions.
- **If I drop to 20 steps:** 40 passes, ~33% faster, and for most prompts I will not see the difference. That is a free cost cut.
- **If I crank CFG to 14 to "force" adherence:** same cost (still 2 passes/step) but the image comes back saturated and *less* faithful. I have spent the same money to make it worse.
- **If I switch to FLUX-schnell (a distilled turbo variant):** ~4 steps, single pass per step, so ~4 passes total — an order of magnitude cheaper, at some cost to fine control and with no negative prompt.

Now the code, using HF `diffusers`:

```python
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,          # fp16 on GPU; halves memory
).to("cuda")                            # SDXL needs a GPU — see below
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True   # DPM++ 2M Karras
)

g = torch.Generator("cuda").manual_seed(1234)   # <-- reproducibility

image = pipe(
    prompt="a red bicycle leaning on a wall, golden hour, photorealistic",
    negative_prompt="blurry, deformed, watermark, low quality",
    num_inference_steps=30,     # steps knob
    guidance_scale=7.0,         # CFG knob
    generator=g,                # seed knob
).images[0]
image.save("out.png")
```

Every knob from this lecture maps to one argument: `num_inference_steps`, `guidance_scale`, `negative_prompt`, `generator` (seed), and the scheduler object (sampler). For FLUX you would swap in `FluxPipeline`, drop `negative_prompt`, and use far fewer steps.

---

## How it shows up in production

- **GPU is non-negotiable.** SDXL and FLUX need a GPU with real VRAM (SDXL comfortably in ~8–12GB fp16; FLUX-dev is larger, often wanting 16–24GB). **Local CPU is not "slow," it is impractical** — minutes per image versus seconds. Your realistic options: a **free Colab/Modal** notebook for experimentation, or a **hosted endpoint (fal, Replicate)** that runs it for a few cents per image and hands you a URL. For the lab, a hosted endpoint is often the sanest choice — no CUDA install, no VRAM tetris.
- **Steps and CFG are your latency/cost dials.** In a product that generates images on a request path, dropping from 40 to 25 steps or moving to a distilled model can be the difference between a 2-second and a 12-second response. Measure the quality delta on *your* prompts before assuming you need the high setting.
- **Batching amortizes overhead.** Generating 4 images in one pipeline call is much cheaper per image than 4 separate calls — the model load and setup are paid once.
- **Reproducibility bugs are seed bugs.** "It looked great yesterday and I can't get it back" almost always means the seed was not pinned. Log the seed (and library/model versions) alongside every generated asset you care about.
- **Text-in-image and hands are the classic failure surfaces.** Older diffusion models (SD 1.5, and SDXL to a lesser degree) reliably mangle **rendered text** (garbled letterforms on signs, logos, UI mockups) and **hands** (six fingers, fused digits). This is not a knob you can tune away — it is a model-capability limit. **FLUX and other newer models are markedly better at rendered text**, which is exactly why FLUX took off for anything involving words in the image. If your use case needs legible text or clean hands, that constraint should drive your *model* choice, not your CFG value.

---

## Common misconceptions & failure modes

- **"More steps always = better."** False past ~30–40. You are paying linearly for gains that flatten to nothing. The failure mode is a slow, expensive pipeline that looks identical to a fast one.
- **"More CFG = better prompt adherence."** False and non-monotonic. Above ~10–12 you get fried, over-saturated images that often follow the prompt *worse*. Sweet spot ~5–8.
- **"Negative prompts work everywhere."** No — they depend on the two-pass CFG mechanism. **FLUX largely ignores them** because it is guidance-distilled. Writing an elaborate negative prompt for FLUX is wasted effort.
- **"Sampler is just aesthetics."** No — it is a cost lever. The right sampler hits your quality target in fewer steps.
- **"I'll just run it on my laptop CPU."** For SDXL/FLUX this is a dead end. Use a GPU (Colab/Modal) or a hosted endpoint.
- **Comparing images with different seeds.** If you change steps *and* the seed, you learn nothing about steps. Pin the seed for every knob comparison.
- **Expecting perfect text/hands from old models.** That is a capability wall; switch models (FLUX) rather than tuning knobs.

---

## Rules of thumb / cheat sheet

*(All approximate defaults, current to 2025–2026 tooling — measure on your own prompts.)*

- **Steps:** SDXL 20–40 (DPM++ 2M Karras is happy at ~25). Distilled/turbo/LCM/FLUX-schnell: 1–8. Rarely go above 50.
- **CFG scale:** SDXL **5–8**. Below 3 = drifts from prompt; above ~12 = fried. Not monotonic — tune, don't crank.
- **Sampler:** default to **DPM++ 2M Karras** for quality-per-step; **Euler a** for fast/varied; distilled samplers for real-time.
- **Seed:** **always pin it** for experiments and for any asset you might need to reproduce. Log it.
- **Negative prompt:** use on **SDXL** (`blurry, deformed, extra fingers, watermark, text, low quality`). Skip it on **FLUX** — it won't listen.
- **Model choice:** need legible **text-in-image** or clean **hands** → **FLUX** or newer. General photoreal, lots of community LoRAs/ControlNets, tight VRAM → **SDXL**.
- **Hardware:** GPU only. No GPU → Colab/Modal or fal/Replicate. Never CPU for SDXL/FLUX.
- **First move when tuning:** fix seed, then sweep **steps × CFG** on a contact sheet before touching anything else.

---

## Connect to the lab

Week 3, Part 1 is exactly this lecture made concrete: you'll build a **contact sheet** — the same prompt rendered across `steps ∈ {10, 20, 40}` × `cfg ∈ {3, 7, 12}` at a **fixed seed** — so you can *see* diminishing returns on steps and the frying at CFG 12 with your own eyes. Do it on a free Colab/Modal GPU or a hosted fal/Replicate endpoint. Once the grid is in front of you, the "rules of thumb" above stop being memorized facts and become things you've observed. (The next lecture takes the same pipeline into ControlNet and inpainting for structural control and masked edits.)

---

## Going deeper (optional)

- **Hugging Face `diffusers` docs** — the canonical API reference for pipelines, schedulers, and every knob here. Root: `huggingface.co/docs/diffusers`. Read the "Stable Diffusion XL" and "FLUX" pipeline pages and the "Schedulers" overview.
- **SDXL and FLUX model cards** on Hugging Face (`stabilityai/stable-diffusion-xl-base-1.0`, `black-forest-labs/FLUX.1-dev`) — capabilities, resolutions, and guidance-distillation notes straight from the source.
- **The original DDPM and classifier-free guidance papers** — for the "why" behind the loop and CFG. Search queries: `Denoising Diffusion Probabilistic Models Ho 2020` and `Classifier-Free Diffusion Guidance Ho Salimans`.
- **"What are Diffusion Models?" by Lilian Weng** — a well-known, engineer-readable deep dive. Search: `Lilian Weng diffusion models blog`.
- **The Illustrated Stable Diffusion** by Jay Alammar — visual, intuition-first walkthrough of the latent loop. Search: `Illustrated Stable Diffusion Alammar`.
- Search queries for practice: `DPM++ 2M Karras vs Euler comparison`, `SDXL CFG scale sweet spot`, `FLUX negative prompt why not supported`.

---

## Check yourself

1. In one sentence, what task was the diffusion model actually trained on, and how does that single task make *generation* possible?
2. You raise `num_inference_steps` from 30 to 80 and the image barely changes but takes 2.7× longer. Explain both observations mechanically.
3. A colleague sets `guidance_scale=16` "to make it follow the prompt exactly." Predict what happens and why, and give the term for the resulting look.
4. Why does a negative prompt work on SDXL but do essentially nothing on FLUX?
5. You want to compare two samplers fairly. Name every setting you must hold constant, and say specifically why the seed is on that list.
6. A product requirement says the generated poster must show a legible slogan and a person with correct hands. Which knob do you tune, and which do you *not* — what's the real lever?

### Answer key

1. It was trained only to **predict the noise** added to an image at a given noise level. Because it can do that at *every* level, you can start from pure noise and iteratively subtract predicted noise, walking from static to a coherent image — that iterative denoising *is* generation.
2. Steps is the resolution of a numerical integration from noise to image; by ~30 steps the image has essentially **converged**, so extra steps redraw the same result (no visible change). Cost is linear in steps because each step is a full forward pass, so ~80/30 ≈ 2.7× the passes ≈ 2.7× the time.
3. The image gets **fried** — over-saturated, blown-out contrast, halos — and often adheres to the prompt *worse*, not better. CFG extrapolates `uncond + scale × (cond − uncond)`; at scale 16 it overshoots into latent regions the model never trained on. Adherence is **not monotonic** in CFG.
4. A negative prompt works by **replacing the unconditional prediction** in CFG's two-pass extrapolation with something to steer away from. SDXL runs that second pass, so it has a reference point to poison. FLUX is **guidance-distilled** — guidance is baked into a single forward pass — so there is no second pass and nothing for the negative prompt to act on.
5. Hold constant: prompt, negative prompt, steps, CFG scale, resolution, model, library/hardware, **and the seed**. The seed fixes the **initial noise** so both samplers start from the identical latent; without that you'd be comparing two *different* images and could not attribute any difference to the sampler.
6. Legible text and correct hands are **model-capability** limits, not knob settings — no value of steps/CFG/sampler fixes them. The real lever is **model choice**: pick **FLUX** (or another newer model) known to be markedly better at rendered text and hands. Tuning CFG here would just waste time.
