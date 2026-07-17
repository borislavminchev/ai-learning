# Week 3 Lab: Generation, Video, and the Milestone Glue — Eval Harness + 60% Cost Cut

> This week you do three things and then bind them into the phase milestone. **Part 1** you drive a **diffusion model** with intent — steps, CFG, sampler, seed, negative prompt — plus one **inpainting** and one **ControlNet** edit, so image *generation* stops being a black box. **Part 2** you build a **video-understanding** mini-pipeline (scene sampling + ASR fusion) and prove it costs a fraction of feeding every frame. **Part 3** — the real deliverable — you build a **task-specific multimodal eval harness** (field-F1, CER/WER, latency, $/1000 docs) on a hand-labeled gold set, run a **3-model bake-off**, then **cut your Week 1 `docextract` pipeline's cost ≥60% with F1 held within 2 points**, proven by committed before/after eval runs. This closes **Milestone A** (receipt→JSON with a 60% cut) and gives the shared eval harness that also scores **Milestone B** (SlideChat / ColPali). The discipline is the one you carried since Phase 7: *leaderboards lie about your data — measure the metric that matches the job, on your gold set, and pick on cost-per-correct-answer, not vibes.*
>
> Read these first — this guide assumes you have:
> - [../lectures/12-diffusion-fundamentals-knobs.md](../lectures/12-diffusion-fundamentals-knobs.md) — denoising over N steps; CFG/sampler/seed/negative prompt; SDXL vs FLUX and the text-in-image/hands caveats.
> - [../lectures/13-control-editing-controlnet-inpainting.md](../lectures/13-control-editing-controlnet-inpainting.md) — ControlNet (structure-conditioned), inpainting (masked edits), IP-Adapter, LoRA.
> - [../lectures/14-video-understanding-frame-sampling.md](../lectures/14-video-understanding-frame-sampling.md) — why every-frame is a token bomb; scene sampling + ASR fusion.
> - [../lectures/15-multimodal-eval-harness.md](../lectures/15-multimodal-eval-harness.md) — CER/WER, field-level precision/recall/F1, recall@k, latency, cost-per-correct-answer.
> - [../lectures/16-cost-latency-engineering.md](../lectures/16-cost-latency-engineering.md) — the lever stack (downscale/crop → detail:low → cheap pre-filter → prompt caching → batch/async → self-host) and the "within 2 points" honesty guardrail.

**Est. time:** ~9 hrs (Part 1 ~2.5 · Part 2 ~1.5 · Part 3 ~5) · **You will need:** Python 3.11+, `uv`, your **Week 1 `docextract` repo** (Part 3 extends it), `ffmpeg` on PATH, and a VLM you can call from `litellm`. **Free/local path throughout:** Ollama + `qwen2.5vl` for extraction (no key, no cost), `faster-whisper` on CPU for ASR, `jiwer`/`scikit-learn` for metrics. **Generation (Part 1) needs a GPU** — the free path is a Colab/Modal notebook or a hosted endpoint (**fal**/Replicate) for a few cents; local CPU diffusion is impractically slow. For the bake-off, **Gemini Flash** is the cheapest hosted per-image option with a generous free tier; `gpt-4o-mini` is the easy cheap alternative.

---

## Before you start (setup)

### 0.1 — Confirm the Week 1 repo and its gold-set seed

**What:** you are extending `docextract`, not starting fresh. Part 3 reuses `src/schema.py`, `src/detect.py`, `src/extract_vlm.py`, and `src/route.py`.
**Why:** the milestone is *harden and cost-cut the pipeline you already built*, not rebuild it.

```bash
cd docextract        # your Week 1 repo
uv run python -c "from src.extract_vlm import extract; print('week-1 extract importable')"
git status           # you will commit before/after eval runs, so start clean
```

**Verify:** the import prints without error and `git status` is clean (or has only expected work-in-progress).
**Troubleshoot:** if you skipped/renamed Week 1 files, adjust the import paths below to match your actual module names — the logic is identical.

### 0.2 — Add Week 3 dependencies

```bash
# Eval + video (CPU-friendly, install everywhere)
uv add jiwer scikit-learn pandas scenedetect
uv add faster-whisper            # if not already added in Week 2
# Generation (Part 1) — install ONLY where you have a GPU (or skip, use hosted)
uv add diffusers transformers accelerate torch
```

Layout you will add to `docextract/`:

```
docextract/
  eval/
    gold.jsonl          # 25-40 hand-labeled receipts: {id, image, gold_fields}
    eval.py             # field-F1 + CER + latency + cost; writes runs/*.json
    cost.py             # $/1000-docs model from token counts + provider prices
    runs/               # baseline.json, optimized.json (both committed)
  gen/                  # Part 1 scratch: contact_sheet.py, inpaint.py, controlnet.py, out/
  video/                # Part 2 scratch: pipeline.py, sample.mp4, out/
```

### 0.3 — Provider price table (edit to today's numbers)

**What:** cost cannot be measured without a price sheet. Put it in one place.
**Why:** `$/1000 docs` in your bake-off must come from *real* per-token / per-image prices, not guesses. Prices change — **look them up on the provider's own pricing page** (do not trust a number memorized months ago).

```python
# eval/cost.py  — EDIT these to current published prices (USD per 1M tokens)
PRICES = {
    # model:                  (input_per_1M, output_per_1M)
    "gemini/gemini-2.0-flash": (0.10, 0.40),
    "gpt-4o-mini":             (0.15, 0.60),
    "ollama/qwen2.5vl":        (0.00, 0.00),   # local = zero marginal $ (your electricity aside)
}

def dollars(model: str, in_tok: int, out_tok: int) -> float:
    pi, po = PRICES[model]
    return (in_tok / 1e6) * pi + (out_tok / 1e6) * po
```

> **Windows / Git-Bash note:** run everything through `uv run` from Git-Bash. If `ffmpeg` isn't found, install via `winget install Gyan.FFmpeg` (or `choco install ffmpeg`) and reopen the shell so PATH refreshes. Use forward slashes in paths; `uv run python script.py` works identically to macOS/Linux.

---

## Step-by-step

### Step 1 — Diffusion knobs: build a steps × CFG contact sheet

**What:** generate the *same prompt + fixed seed* across `steps ∈ {10,20,40}` and `cfg ∈ {3,7,12}`, tile the 9 images into one contact sheet.
**Why:** you must *see* what each knob does. Fixed seed isolates the variable; without it you can't tell a knob's effect from noise. This is the whole point of the diffusion half — engineering intuition, not training a model.

**Do it** (GPU — Colab/Modal or a GPU box). Free hosted alternative in Troubleshoot.

```python
# gen/contact_sheet.py
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16",
).to("cuda")

PROMPT = "a photograph of a red vintage bicycle leaning on a brick wall, golden hour"
NEG    = "blurry, distorted, extra wheels, text, watermark"
STEPS  = [10, 20, 40]
CFGS   = [3.0, 7.0, 12.0]
SEED   = 42

imgs = []
for s in STEPS:
    for c in CFGS:
        g = torch.Generator("cuda").manual_seed(SEED)   # fixed seed => reproducible
        img = pipe(PROMPT, negative_prompt=NEG, num_inference_steps=s,
                   guidance_scale=c, generator=g).images[0]
        imgs.append(img.resize((384, 384)))

# tile 3x3
W, H = 384, 384
sheet = Image.new("RGB", (W*3, H*3))
for i, im in enumerate(imgs):
    sheet.paste(im, ((i % 3)*W, (i // 3)*H))
sheet.save("gen/out/contact_sheet.png")
print("saved gen/out/contact_sheet.png  (rows=steps 10/20/40, cols=cfg 3/7/12)")
```

**Expected result:** a 3×3 grid. Low steps (10) look under-baked/soft; CFG 3 is loose/creative; CFG 12 looks over-saturated and "fried." A sweet spot around steps 20–30, CFG 6–8 is visible.
**Verify:** open `gen/out/contact_sheet.png`; you can point to the fried column (CFG 12) and the mushy row (10 steps). Adherence is *not* monotonic in CFG — note that in your writeup.
**Troubleshoot:**
- **No GPU:** use a hosted endpoint. `fal` or Replicate expose SDXL/FLUX over HTTP for a few cents; wrap the call in the same loop and save the returned images. Do not attempt SDXL on CPU (minutes per image).
- **`FLUX` instead of SDXL:** FLUX renders text-in-image far better but **largely ignores negative prompts** (guidance works differently) — drop `NEG` and note it. Use `diffusers`' `FluxPipeline`.
- **OOM on a small GPU:** call `pipe.enable_model_cpu_offload()` and/or drop resolution to 768.

### Step 2 — One inpainting edit and one ControlNet run

**What:** (a) mask a region of an image and regenerate only it (inpaint); (b) extract Canny edges from an image and generate a new style that keeps the structure (ControlNet).
**Why:** these are the two "control" tools you will actually reach for. You only need to prove you can drive them once each.

**Do it — inpainting:**

```python
# gen/inpaint.py
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
).to("cuda")

init = Image.open("gen/in/room.jpg").convert("RGB").resize((1024, 1024))
mask = Image.open("gen/in/mask.png").convert("L").resize((1024, 1024))  # white=edit, black=keep
out = pipe(prompt="a potted monstera plant", image=init, mask_image=mask,
           num_inference_steps=25, guidance_scale=7.0).images[0]
out.save("gen/out/inpaint.png")
```

> Make the mask quickly: any paint tool, or `PIL.ImageDraw` to fill a white rectangle over the region on a black canvas.

**Do it — ControlNet (Canny → new style):**

```python
# gen/controlnet.py
import cv2, numpy as np, torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

edges = cv2.Canny(cv2.imread("gen/in/room.jpg"), 100, 200)
control = Image.fromarray(np.stack([edges]*3, axis=-1))   # 3-channel edge map

cn = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, torch_dtype=torch.float16,
).to("cuda")
img = pipe("the same room, watercolor painting style", image=control,
           num_inference_steps=25, controlnet_conditioning_scale=0.7).images[0]
img.save("gen/out/controlnet.png")
```

**Expected result:** inpaint changes only the masked region (rest is untouched); ControlNet keeps the room's layout/edges but restyles it.
**Verify:** overlay/flip between input and inpaint output — only the masked area changed. For ControlNet, the edge structure of the original is clearly preserved in the new style.
**Troubleshoot:** if inpaint bleeds outside the mask, your mask polarity is wrong (white = region to change). If ControlNet ignores structure, raise `controlnet_conditioning_scale` toward 1.0; if it looks like a traced photocopy, lower it.

### Step 3 — Video understanding: scene sampling + ASR fusion (no token bomb)

**What:** take a short video → extract audio and transcribe (faster-whisper) → detect scene boundaries (PySceneDetect) → grab one frame per scene → send *sampled frames + transcript* to a VLM for a timestamped summary. Record the token count vs a naive every-frame approach.
**Why:** feeding every frame of even a 60s clip is thousands of image-token blocks. Scene-sampling + ASR fusion gives the model the *content* at a fraction of the cost. This is Video RAG's core move.

**Do it:**

```bash
# 1) extract the audio track (Git-Bash / macOS / Linux identical)
ffmpeg -i video/sample.mp4 -vn -ac 1 -ar 16000 video/out/audio.wav -y
```

```python
# video/pipeline.py
from faster_whisper import WhisperModel
from scenedetect import detect, ContentDetector
import subprocess, os

# 2) transcribe (CPU-friendly local ASR; int8 keeps it fast)
asr = WhisperModel("base.en", compute_type="int8")
segments, _ = asr.transcribe("video/out/audio.wav", vad_filter=True, word_timestamps=True)
transcript = [(round(s.start, 1), s.text.strip()) for s in segments]

# 3) scene boundaries -> one representative frame per scene
scenes = detect("video/sample.mp4", ContentDetector(threshold=27.0))  # list of (start, end)
os.makedirs("video/out/frames", exist_ok=True)
frame_paths = []
for i, (start, _end) in enumerate(scenes):
    t = start.get_seconds()
    p = f"video/out/frames/scene_{i:02d}.jpg"
    subprocess.run(["ffmpeg", "-ss", str(t), "-i", "video/sample.mp4",
                    "-frames:v", "1", "-q:v", "3", p, "-y"], check=True)
    frame_paths.append((round(t, 1), p))

print(f"{len(frame_paths)} scenes sampled  (naive every-frame would be ~{int(scenes[-1][1].get_seconds()*24)} frames @24fps)")
```

Then reuse Week 1's `litellm` VLM client: build a message with the transcript as text plus each sampled frame as an `image_url` (`detail:"low"`), prompt *"Summarize this video. Use the transcript for what's said and the frames for what's shown; give timestamps."*

**Expected result:** a coherent timestamped summary. Your print line shows e.g. **8 sampled frames vs ~1440 every-frame** — roughly a 180× reduction in image tokens.
**Verify:** the summary references things only visible in frames (not just the audio), proving fusion works; you have a recorded frame-count comparison.
**Troubleshoot:** too many/few scenes → tune `ContentDetector(threshold=...)` (lower = more sensitive). Whisper emitting phantom "thanks for watching" on silence → confirm `vad_filter=True`. `ffmpeg -ss` before `-i` is fast (keyframe seek); good enough for a representative frame.

### Step 4 — Build the gold set (25–40 labeled receipts)

**What:** hand-label 25–40 of your Week 1 receipts with the *true* field values, stratified across clean-digital / phone-photo / faded / foreign-currency. Store as JSONL.
**Why:** every metric downstream is measured against this. This is your Phase 7 golden-set discipline applied to pixels. Stratifying means your aggregate F1 can't be gamed by an easy-sample-heavy set.

**Do it** — one line per doc:

```jsonl
{"id":"r001","image":"samples/r001.jpg","stratum":"digital","gold_fields":{"merchant":"Whole Foods","date":"2025-11-03","currency":"USD","subtotal":"42.10","tax":"3.79","total":"45.89"}}
{"id":"r014","image":"samples/r014.jpg","stratum":"phone_faded","gold_fields":{"merchant":"Café Central","date":"2025-10-22","currency":"EUR","subtotal":"18.00","tax":"1.80","total":"19.80"}}
```

Label the scalar fields (merchant/date/currency/subtotal/tax/total); line-items are optional for F1 but recommended for the hardest 5. **Normalize as you label** (dates ISO, amounts as plain decimals) so the metric doesn't punish formatting.

**Verify:** `uv run python -c "import json;print(sum(1 for _ in open('eval/gold.jsonl')))"` prints 25–40; every line parses as JSON; each stratum has ≥3 docs.
**Troubleshoot:** don't have enough real receipts? Photograph a few more — synthetic receipts hide exactly the failures (EXIF rotation, faded totals) the eval must catch.

### Step 5 — `eval.py`: field-F1, CER, latency, and cost

**What:** for each gold doc, run extraction, then compute **field-level precision/recall/F1** (normalized exact match per field), **CER** on text fields with `jiwer`, plus per-doc **latency** and **token counts**. Aggregate and write a run file.
**Why:** this is the harness the whole milestone leans on. It must be a *function of (model, config)* so you can call it for the bake-off and for every cost-cut step.

**Do it** — key signatures and the metric core:

```python
# eval/eval.py
import json, time
from jiwer import cer
from src.extract_vlm import extract           # Week 1
from eval.cost import dollars

SCALAR = ["merchant","date","currency","subtotal","tax","total"]

def norm(s: str | None) -> str:
    return "" if s is None else "".join(str(s).lower().split()).strip(".")

def field_scores(gold: dict, pred: dict) -> dict:
    """Per-field exact-match TP/FP/FN for micro-F1, and CER on text fields."""
    tp = fp = fn = 0
    cers = []
    for f in SCALAR:
        g, p = norm(gold.get(f)), norm(pred.get(f))
        if g and p and g == p: tp += 1
        elif g and p and g != p: fp += 1; fn += 1     # wrong value = both a miss and a false hit
        elif g and not p: fn += 1
        elif p and not g: fp += 1
        if g:                                          # CER only where we have a gold string
            cers.append(cer(g, p) if p else 1.0)
    return {"tp": tp, "fp": fp, "fn": fn, "cer": sum(cers)/len(cers) if cers else 0.0}

def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f1

def run(model: str, config: dict, gold_path="eval/gold.jsonl") -> dict:
    tp=fp=fn=0; lat=[]; cers=[]; in_tok=out_tok=0
    for line in open(gold_path, encoding="utf-8"):
        d = json.loads(line)
        t0 = time.perf_counter()
        rec = extract(d["image"], model=model, **config)     # returns Receipt + usage
        lat.append(time.perf_counter()-t0)
        pred = {f: getattr(rec, f).value for f in SCALAR}
        s = field_scores(d["gold_fields"], pred)
        tp+=s["tp"]; fp+=s["fp"]; fn+=s["fn"]; cers.append(s["cer"])
        in_tok += rec._usage.prompt_tokens; out_tok += rec._usage.completion_tokens
    p, r, f1 = prf(tp, fp, fn)
    lat.sort(); n=len(lat)
    return {
        "model": model, "config": config,
        "precision": round(p,4), "recall": round(r,4), "f1": round(f1,4),
        "cer": round(sum(cers)/len(cers),4),
        "p50_ms": round(lat[n//2]*1000), "p95_ms": round(lat[min(n-1,int(n*0.95))]*1000),
        "usd_per_1000": round(dollars(model, in_tok, out_tok)/n*1000, 4),
    }
```

> **Wiring note:** your Week 1 `extract()` must return the token usage. `litellm` responses carry `.usage`; have `extract` stash it (e.g. `rec._usage = resp.usage`) or return a `(Receipt, usage)` tuple and adapt the harness. If you can't get real usage, use `litellm.token_counter(...)` on the request/response as a fallback and note it.

**Expected result:** `run("gemini/gemini-2.0-flash", {})` returns a dict with `f1`, `cer`, `p50_ms`, `p95_ms`, `usd_per_1000`.
**Verify:** F1 is in [0,1] and sane (>0.7 on your set for a decent model); deliberately corrupt one gold value and confirm F1 drops.
**Troubleshoot:** F1 suspiciously low → your normalization differs between gold and pred (dates/currency symbols). Log a few `(gold, pred)` pairs and fix `norm()` on **both** sides consistently. `jiwer.cer` errors on empty strings → guard as shown.

### Step 6 — Model bake-off: 3 VLMs on the same gold set

**What:** run `eval.run(...)` for `gemini/gemini-2.0-flash`, `gpt-4o-mini`, and `ollama/qwen2.5vl`; print one comparison table.
**Why:** you pick the default on **cost-per-correct-answer**, not a public leaderboard. Same gold set, same harness = the only fair comparison.

**Do it:**

```python
# eval/bakeoff.py
import json, pandas as pd
from eval.eval import run

MODELS = ["gemini/gemini-2.0-flash", "gpt-4o-mini", "ollama/qwen2.5vl"]
rows = [run(m, {}) for m in MODELS]
df = pd.DataFrame(rows)[["model","f1","cer","p50_ms","p95_ms","usd_per_1000"]]
print(df.to_string(index=False))
df.to_json("eval/runs/bakeoff.json", orient="records", indent=2)
```

**Expected result:** a table like:

| model | f1 | cer | p50_ms | p95_ms | usd_per_1000 |
|---|---|---|---|---|---|
| gemini-2.0-flash | 0.91 | 0.04 | 1400 | 2600 | 0.85 |
| gpt-4o-mini | 0.89 | 0.05 | 1700 | 3100 | 1.90 |
| ollama/qwen2.5vl | 0.83 | 0.08 | 8200 | 14000 | 0.00 |

(Your numbers will differ — that's the point.)
**Verify:** you can write one sentence justifying the default (e.g. "Gemini Flash: best F1 at lowest hosted cost; local Qwen is the $0 fallback but too slow for interactive use").
**Troubleshoot:** local model much worse → give it the *same* prompt, and if it can't do bboxes/confidence reliably, score only the scalar fields for it and note the limitation. Ollama slow/OOM → use a smaller quant or run fewer gold docs for the local row only (state N).

### Step 7 — Baseline cost run, committed

**What:** run the eval with the **naive** config (full-res image, `detail:"high"`, no caching, realtime, single model) and commit it as `eval/runs/baseline.json`.
**Why:** a 60% cut is meaningless without a fixed, committed baseline to measure against. This is the "before."

**Do it:**

```bash
uv run python -c "import json; from eval.eval import run; \
json.dump(run('gemini/gemini-2.0-flash', {'detail':'high','max_edge':None}), open('eval/runs/baseline.json','w'), indent=2)"
git add eval/runs/baseline.json && git commit -m "eval: committed baseline run (naive config)"
```

**Verify:** `baseline.json` exists with `f1` and `usd_per_1000`; note both numbers — they are your targets to beat (cost) and hold (F1).
**Troubleshoot:** if `extract()` doesn't accept `detail`/`max_edge` kwargs yet, add them (Step 8 needs them anyway).

### Step 8 — Stack the cost levers, re-running eval after each

**What:** apply the levers *in priority order*, re-running `eval.run(...)` after each so you see cost fall and can catch any F1 drop immediately. Target **≥60% cost cut, F1 within 2 points of baseline.**
**Why:** optimizing cost before measuring accuracy is how F1 quietly drops 8 points unnoticed. The re-run-after-each-change loop *is* the honesty guardrail.

**Do it — the lever stack (highest bang-for-buck first):**

1. **Downscale + crop.** Downscale the long edge to ≤1024px and crop to the receipt bbox (reuse Week 1 `detect.py`). Biggest single win — image tokens scale with pixels/tiles.
2. **`detail: "low"`.** One flag; often identical extraction quality on a receipt at a fraction of the tokens.
3. **Cheap-model pre-filter.** A tiny/cheap classifier call ("is this a receipt/invoice? yes/no") rejects junk before the expensive extraction model runs on it.
4. **Prompt caching.** Keep the long instruction prefix *stable and first* so the provider caches it across calls (cached input tokens are much cheaper).
5. **Batch/async for the easy 80%.** Route clean/high-confidence docs through the **batch API** (often ~50% off, but minutes–hours latency); keep only low-confidence docs on the realtime endpoint.

```python
# sketch: config-driven so eval.run can call each variant
CONFIGS = {
    "baseline":   {"detail":"high", "max_edge":None,  "crop":False, "prefilter":False, "cache":False, "batch":False},
    "downscale":  {"detail":"high", "max_edge":1024,  "crop":True,  "prefilter":False, "cache":False, "batch":False},
    "low_detail": {"detail":"low",  "max_edge":1024,  "crop":True,  "prefilter":False, "cache":False, "batch":False},
    "prefilter":  {"detail":"low",  "max_edge":1024,  "crop":True,  "prefilter":True,  "cache":False, "batch":False},
    "cached":     {"detail":"low",  "max_edge":1024,  "crop":True,  "prefilter":True,  "cache":True,  "batch":False},
    "optimized":  {"detail":"low",  "max_edge":1024,  "crop":True,  "prefilter":True,  "cache":True,  "batch":True},
}
# for name,cfg in CONFIGS.items(): print(name, run("gemini/gemini-2.0-flash", cfg))
```

Commit the final winner as `eval/runs/optimized.json`.

**Expected result:** a monotone-ish descent in `usd_per_1000` (e.g. baseline 0.85 → optimized 0.30, a ~65% cut) while `f1` stays within 2 points (0.91 → 0.90).
**Verify:** compute the reduction: `(baseline.usd - optimized.usd)/baseline.usd ≥ 0.60` **and** `abs(baseline.f1 - optimized.f1) ≤ 0.02`. Both run files committed.
**Troubleshoot:**
- **F1 dropped >2 points after a lever** → that lever went too far. Most common: over-aggressive downscale (bump max_edge to 1280) or a crop that clips a field. Back it off; a slightly smaller cut with intact F1 beats a big cut that regresses.
- **Batch didn't help** → batch discount only applies if you actually submit to the batch endpoint; confirm you're not still hitting realtime. Never route interactive traffic to batch (minutes–hours).
- **Caching shows no savings** → the instruction prefix must be byte-identical and first; per-doc text must come *after* it.

### Step 9 — Recalibrate the confidence-routing threshold against the gold set

**What:** using the gold set, find the confidence threshold at which the "auto-approved" bucket hits your target precision; state that precision.
**Why:** in Week 1 you set the threshold (0.75) by guess. Self-reported confidence is uncalibrated — now you have ground truth, so *measure* where auto-approval is safe. This closes the "calibrated confidence" DoD item.

**Do it:** for each doc, record `(min_field_confidence, all_fields_correct?)`. Sweep the threshold; for each, compute precision of the auto-approved bucket (docs above threshold that are fully correct / all docs above threshold) and its coverage (fraction auto-approved). Pick the threshold meeting your precision target (e.g. ≥0.98) at the best coverage.

```python
# eval/calibrate.py (sketch)
def sweep(pairs, targets=(0.95,0.98,0.99)):
    # pairs: list[(min_conf, correct_bool)]
    for t in [i/100 for i in range(50,100,2)]:
        auto = [c for mc,c in pairs if mc>=t]
        if not auto: continue
        prec = sum(auto)/len(auto); cov = len(auto)/len(pairs)
        print(f"thr={t:.2f}  auto_precision={prec:.3f}  coverage={cov:.2f}")
```

**Expected result:** a table letting you say e.g. "at confidence ≥0.88, the auto-approved bucket is 98.5% precise and covers 74% of docs; the rest go to review."
**Verify:** you can state the auto-bucket precision and coverage in one sentence; the threshold is written into `route.py`.
**Troubleshoot:** if confidence barely correlates with correctness (flat precision across thresholds), fall back to the *arithmetic* validator as the primary gate and treat confidence as a weak tiebreaker — and say so.

### Step 10 — Green the whole repo

**What:** `pytest` across `docextract`, including the Week 1 tests and a new eval-smoke test.
**Why:** the DoD requires the whole repo green after the Week 3 changes — cost cuts must not have broken extraction.

**Do it:**

```python
# tests/test_eval_smoke.py
def test_field_scores_perfect():
    from eval.eval import field_scores, prf
    g = {"merchant":"Acme","total":"9.99"}
    s = field_scores(g, dict(g))
    p,r,f1 = prf(s["tp"], s["fp"], s["fn"])
    assert f1 == 1.0 and s["cer"] == 0.0

def test_field_scores_catches_wrong_value():
    from eval.eval import field_scores
    s = field_scores({"total":"9.99"}, {"total":"99.9"})
    assert s["fp"] == 1 and s["fn"] == 1     # wrong value counts as miss + false hit
```

```bash
uv run pytest -q
```

**Expected result:** all tests pass.
**Verify:** exit code 0.
**Troubleshoot:** import errors → run pytest from the repo root with `uv run` so `src`/`eval` are importable; add empty `__init__.py` where needed.

---

## Putting it together — a short end-to-end run

```bash
# 1) Generation intuition (GPU / Colab / hosted)
uv run python gen/contact_sheet.py && uv run python gen/inpaint.py && uv run python gen/controlnet.py

# 2) Video: sampled frames + ASR -> timestamped summary
ffmpeg -i video/sample.mp4 -vn -ac 1 -ar 16000 video/out/audio.wav -y
uv run python video/pipeline.py

# 3) The milestone glue
uv run python eval/bakeoff.py                          # 3-model table -> pick default
uv run python -c "import json;from eval.eval import run;json.dump(run('gemini/gemini-2.0-flash',{'detail':'high','max_edge':None}),open('eval/runs/baseline.json','w'),indent=2)"
uv run python eval/optimize.py                         # stacks levers -> eval/runs/optimized.json
uv run python -c "import json;b=json.load(open('eval/runs/baseline.json'));o=json.load(open('eval/runs/optimized.json'));print(f\"cost cut {(b['usd_per_1000']-o['usd_per_1000'])/b['usd_per_1000']:.0%}, F1 {b['f1']}->{o['f1']}\")"
uv run pytest -q
git add eval/runs/*.json && git commit -m "milestone A: 60% cost cut, F1 within 2 points (before/after evals)"
```

The final print should read something like `cost cut 65%, F1 0.91->0.90` — that one line is the whole job.

---

## Definition of Done — verifiable checks

Restating the spine's Week 3 checklist and the Milestone A acceptance as things you can point at:

- [ ] **Contact sheet** shows the visible effect of steps/CFG/seed; **one inpaint** and **one ControlNet** result saved (`gen/out/`).
- [ ] **Video summary** produced from sampled frames + ASR, with a **recorded token/frame-count comparison** vs every-frame (your print line: N scenes vs ~M frames).
- [ ] `eval.py` outputs **per-field F1** and **CER** on a **25–40 doc gold set**, and a **3-model** comparison table with **F1, p50/p95 latency, and $/1000 docs** (`eval/runs/bakeoff.json`).
- [ ] Receipt pipeline **cost cut ≥60%** vs baseline with **F1 within 2 points** — proven by **`baseline.json` and `optimized.json`, both committed**; the reduction line prints ≥60%.
- [ ] Confidence-routing threshold is **calibrated against the gold set** — you can state the **precision and coverage of the auto-approved bucket**.
- [ ] `pytest` **green** across the whole `docextract` repo.
- [ ] **Milestone A acceptance:** 20/20 samples produce valid JSON (Week 1 still passes); eval reports per-field F1 ≥ your stated target; the review queue contains exactly the docs that failed validation; a review UI renders the image with bbox highlights on flagged fields (Week 1 asset — confirm it still works after the cost cut).

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| SDXL images look "fried"/saturated | CFG too high (>~12) | Drop to 6–8; adherence isn't monotonic in CFG |
| Diffusion painfully slow / OOM | Running on CPU or small GPU | Use Colab/Modal or fal/Replicate; `enable_model_cpu_offload()`; drop to 768px |
| FLUX ignores your negative prompt | FLUX doesn't do CFG-style negatives | Remove neg prompt; steer via the positive prompt; note the caveat |
| Inpaint changes the whole image | Mask polarity inverted | White = region to change, black = keep |
| Video summary misses on-screen content | Only ASR fused, frames too sparse | Lower `ContentDetector` threshold; ensure frames actually sent to VLM |
| Whisper emits "thanks for watching" | Hallucination on silence | `vad_filter=True` |
| F1 near zero on a good model | Gold/pred normalization mismatch | Apply the *same* `norm()` (dates ISO, strip currency/whitespace) to both sides |
| `jiwer.cer` crashes | Empty prediction string | Guard: `cer(g,p) if p else 1.0` |
| Cost cut but F1 dropped >2 pts | A lever overshot (usually downscale/crop) | Re-run eval per lever; back off the offender (max_edge 1280; loosen crop) |
| Batch API shows no savings | Still hitting realtime endpoint | Submit to the batch endpoint; only route non-realtime docs |
| Caching shows no savings | Prefix not stable/first | Put byte-identical instruction prefix first; per-doc text after |
| `ffmpeg`/torch not found on Windows | PATH not refreshed / wrong Python | `winget install Gyan.FFmpeg`, reopen Git-Bash; use 64-bit Python 3.11/3.12 |

---

## Stretch goals (optional)

- **Self-host the winner** with vLLM (quantized Qwen2.5-VL) and add a `$/1000 docs` row at your real volume — cross-reference Phase 10 serving economics; find the break-even vs the hosted API.
- **Extend the harness to Milestone B (SlideChat):** add **recall@k** against a small labeled `query→page` gold set and run the ColPali app through the *same* harness so one report scores both milestones.
- **A/B the ColPali app vs an OCR→text RAG baseline** on visual questions (charts/tables) and report which wins and why — the Milestone B acceptance item.
- **Per-field cost attribution:** which fields drive the token cost (line items!) and whether a two-pass extract (scalars cheap, line-items only if needed) beats the single-pass cost.
- **Calibration plot:** bin model confidence vs empirical correctness (reliability diagram) to *show* how uncalibrated the raw confidence was before Step 9.
