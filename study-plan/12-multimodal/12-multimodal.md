# Phase 12 — Multimodal & Specialized Modalities

Most real product value now lives outside plain text: phone photos of receipts, scanned contracts, screenshots, slide decks, voice calls, and video. This phase wires **VLMs, OCR, multimodal retrieval, speech, and image generation** into low-latency, cost-controlled systems that ship. You will treat images and audio the same way you treated text since Phase 1 — as untrusted input that flows through instrumented pipelines with structured output, evals, and a cost budget — and you will finish with two artifacts a real product would use: a receipt→JSON extractor with bounding boxes, confidence routing, and a 60% cost cut, and a "chat with your slide decks" ColPali app.

Prev: [11-safety-security.md](../11-safety-security/11-safety-security.md) · Next: [13-frameworks-career.md](../13-frameworks-career/13-frameworks-career.md)

## Prerequisites
- Phases 0–7 at minimum: you can build a structured-output extractor with **Pydantic + instructor** (Phase 2), a **RAG pipeline** with reranking and citations (Phases 3–4), and an **eval harness** with a golden set + LLM judge (Phase 7). We reuse all of these.
- The **Agents** phase (Phase 6) is cross-referenced heavily in Week 2's realtime voice agent — you should already understand the tool-calling loop, bounded loops, and latency budgets.
- Python 3.11+ with `uv`, a provider key for at least one vision model (**Gemini** is the cheapest per image and has a generous free tier; OpenAI GPT-4o/Claude also fine) **or** a local install of **Ollama** running a small VLM (`llama3.2-vision`, `qwen2.5vl`) for cost-free work.
- `ffmpeg` installed and on PATH (audio/video). A microphone for the voice week. No GPU required for the core labs; where one helps (self-hosted Whisper, diffusion, ColPali indexing) we give a free Colab/Modal fallback.

## Time budget
3 weeks × ~10–15 hrs/week (~35–40 hrs total). Roughly 35% theory / 65% hands-on. Each week states its own hours breakdown and fits a part-time schedule alongside a full-time job.

## How to use this file
Do the weeks in order — Week 1 (vision + document extraction) builds the receipt milestone, Week 2 (multimodal retrieval + speech) builds the slide-deck app and a voice agent, Week 3 (generation, evaluation, cost engineering) hardens both and adds the eval harness + 60% cost cut. Timebox the theory; the labs are the deliverable. If you fall behind, cut the generation half of Week 3 first — the receipt and slide-deck milestones plus their evals are the non-negotiable core.

> **📚 Course materials.** This file is the **spine** — a weekly plan and a *recap*. Deep material lives alongside it:
> - **Lectures**: [`lectures/`](lectures/00-index.md) — read for the *why*/mechanism.
> - **Lab guides**: [`labs/`](labs/) — follow to *build*.
> Each week: read the linked lectures first (the Theory here is their recap), then work the linked lab guide.

---

## Week 1 — Vision: VLMs, structured image understanding, OCR & document extraction

**Hours breakdown:** ~4 hrs theory / ~9 hrs lab (13 total). Bump to 15 if your documents are messy scans.

### Objectives
By the end of the week you can:
1. Explain the VLM architecture (vision encoder → projector → LLM), why images become **token blocks**, and estimate the token cost of an image before you send it — including why a 4K screenshot can silently cost thousands of tokens via tiling.
2. Send an image to a VLM and get back **schema-validated JSON** (Pydantic + instructor), with per-field values, **bounding boxes**, and a per-field **confidence** score.
3. Choose correctly between **dedicated OCR** (PaddleOCR/Textract), **VLM reading**, and the **hybrid** (OCR text + image → VLM) approach, and justify it per document type.
4. Detect **born-digital vs scanned** documents and route accordingly; extract page-spanning table structure.
5. Route **low-confidence fields to human review** deterministically instead of trusting the model blindly.

### Theory (~4 hrs)
> 📚 Lectures for this week: [How VLMs Actually Work](lectures/01-vlm-architecture.md) · [Image Tokens, Tiling, and Estimating Cost](lectures/02-image-tokens-tiling-cost.md) · [Schema-Validated Extraction](lectures/03-structured-vlm-extraction.md) · [OCR vs VLM vs Hybrid Routing](lectures/04-ocr-vs-vlm-hybrid-routing.md) · [LLM Proposes, Code Disposes](lectures/05-confidence-routing-validation.md). Read them first. The bullets below are the recap.

- **How VLMs actually work.** A vision encoder (usually a ViT, e.g. SigLIP/CLIP-style) turns an image into patch embeddings; a small **projector/connector** (MLP or cross-attention, as in LLaVA/Qwen2.5-VL) maps those into the LLM's token space; the LLM then attends over image tokens + text tokens jointly. The engineering consequence: **an image is just more tokens**, and more/larger images = more tokens = more cost and latency. Read the **LLaVA** project README (search `LLaVA GitHub`) for the canonical encoder+projector+LLM picture, and the **Qwen2.5-VL** model card on Hugging Face for a 2025 production-grade example with dynamic resolution.
- **Tiling & the cost blowup.** Providers split large images into fixed tiles (OpenAI's `detail: high` tiles a 512px grid; Gemini and Claude have their own schemes). Read the **OpenAI vision guide** and **Anthropic vision docs** (root docs: `platform.openai.com/docs`, `docs.anthropic.com`) sections on image token counting. The key intuition: a 3000×2000 photo can expand to thousands of tokens, and `detail: low` (or downscaling to ~768–1024px on the long edge) often gives identical extraction quality at a fraction of the cost. You will exploit this in Week 3.
- **VQA & grounding limits.** VLMs are strong at reading and describing but **unreliable at counting, precise spatial reasoning, and exact pixel coordinates**. Bounding boxes from a VLM are approximate — good enough for a review-UI highlight, not for pixel-perfect cropping. When you need real coordinates at scale, dedicated OCR/layout models (PaddleOCR, Textract, Docling) give you geometry the VLM can't.
- **OCR decision framework.** Dedicated OCR engines (PaddleOCR, Tesseract, AWS Textract, Google Document AI) return text **with coordinates and confidence** and are cheap/fast at scale, but struggle with context and layout semantics. VLMs read messy, rotated, low-quality, or context-heavy images well and can output structured JSON directly, but cost more and hallucinate plausible values. **Hybrid** (run OCR, then feed the OCR text *plus* the image to a VLM) is often the best of both: the VLM gets grounded text to reduce hallucination and the image for layout/context. Skim the **Docling** and **PaddleOCR** GitHub READMEs.
- **Born-digital vs scanned.** A born-digital PDF has an embedded text layer (extract it directly with PyMuPDF — near-perfect, free, instant). A scanned PDF is just images and needs OCR/VLM. Detecting which you have first saves you from OCR-ing text you could have read for free.

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [Build docextract: Document Image to Validated JSON with BBoxes, Confidence, and Review Routing](labs/week-1-docextract-service.md). The steps below are the summary.

Build `docextract`, a service that turns a document image into validated JSON with boxes and confidence. This is the seed of the Week 3 milestone.

**1. Scaffold.**
```bash
uv init docextract && cd docextract
uv add pydantic instructor "litellm>=1.50" pymupdf pillow python-dotenv rich
uv add paddleocr paddlepaddle   # dedicated OCR; heavy, install once
uv add --dev pytest
mkdir -p src samples out tests
```
Layout:
```
docextract/
  src/
    schema.py        # Pydantic models: field value + bbox + confidence
    detect.py        # born-digital vs scanned; auto-rotate/downscale
    ocr.py           # PaddleOCR wrapper -> tokens with coords+conf
    extract_vlm.py   # image -> JSON via VLM (instructor)
    extract_hybrid.py# OCR text + image -> JSON
    route.py         # confidence routing to review queue
  samples/           # 15-20 receipts/invoices (mix digital + phone photos)
  out/               # extracted JSON + review-queue items
  .env
```

**2. Define the schema with boxes + confidence.**
```python
# src/schema.py
from pydantic import BaseModel, Field
from typing import Optional

class BBox(BaseModel):
    x: float; y: float; w: float; h: float   # normalized 0-1

class FieldVal(BaseModel):
    value: Optional[str] = Field(None, description="extracted text, null if absent")
    bbox: Optional[BBox] = Field(None, description="normalized location on the page")
    confidence: float = Field(..., ge=0, le=1, description="model self-reported 0-1")

class LineItem(BaseModel):
    description: FieldVal; qty: FieldVal; unit_price: FieldVal; amount: FieldVal

class Receipt(BaseModel):
    merchant: FieldVal; date: FieldVal; currency: FieldVal
    subtotal: FieldVal; tax: FieldVal; total: FieldVal
    line_items: list[LineItem]
```

**3. Detect + preprocess.** In `detect.py`, use PyMuPDF (`fitz`) to check for a text layer: `page.get_text("text")` non-empty ⇒ born-digital (extract directly). Otherwise render to image. For phone photos: auto-rotate via EXIF (Pillow `ImageOps.exif_transpose`), and **downscale the long edge to 1600px** before any VLM call (keeps receipts readable, slashes tokens).

**4. VLM extraction with instructor.** Use `litellm` so you can swap models with one string. Ask for the `Receipt` schema; instruct the model to emit `confidence` per field and normalized `bbox`.
```python
# src/extract_vlm.py
import base64, instructor
from litellm import completion
from .schema import Receipt

def _img_url(path):  # inline base64 data URL
    b = base64.b64encode(open(path,"rb").read()).decode()
    return f"data:image/jpeg;base64,{b}"

client = instructor.from_litellm(completion)

def extract(path, model="gemini/gemini-2.0-flash"):
    return client.chat.completions.create(
        model=model, response_model=Receipt, max_retries=2,
        messages=[{"role":"user","content":[
            {"type":"text","text":"Extract every field. Give normalized bbox (0-1) "
             "and a calibrated 0-1 confidence per field. Use null value + confidence 0 "
             "if a field is absent. Do not invent totals."},
            {"type":"image_url","image_url":{"url":_img_url(path),"detail":"low"}},
        ]}],
    )
```
> **Free/local alt:** point `model="ollama/qwen2.5vl"` at a local Ollama (`ollama pull qwen2.5vl`). Slower, no cost, no key.

**5. Hybrid path.** In `extract_hybrid.py`, run PaddleOCR first, get `[(text, bbox, conf), ...]`, format it as a text block, and pass **both** the OCR dump and the image to the VLM. Compare hallucination rate vs the pure-VLM path on your 5 hardest samples.

**6. Confidence routing.** In `route.py`, for each extracted field: if `confidence < 0.75` **or** an arithmetic check fails (`sum(line_items.amount) ≈ subtotal`, `subtotal + tax ≈ total`), flag it. Write flagged docs to `out/review_queue.jsonl` with the offending fields; write clean docs to `out/auto.jsonl`. This is "LLM proposes, code disposes" for vision.

**7. Test.** In `tests/`, assert every sample produces schema-valid `Receipt` JSON and that the arithmetic validator catches a deliberately corrupted total.

### Definition of Done
- [ ] `extract()` returns a schema-valid `Receipt` for **20/20** sample images (pure-VLM path), verified by `pytest`.
- [ ] Born-digital PDFs are read via PyMuPDF text layer (no OCR/VLM call) and scanned/photo inputs are routed to the image path — proven with one of each in the tests.
- [ ] Every field carries a `confidence` in [0,1] and a normalized `bbox`; you can render at least one box back onto the image (Pillow `ImageDraw`) and it lands on the right region.
- [ ] The arithmetic validator flags a corrupted-total document into `review_queue.jsonl`.
- [ ] You have a one-paragraph written comparison of pure-VLM vs hybrid hallucination on your 5 hardest samples, with a recommendation.
- [ ] Downscaling to ≤1600px is applied before VLM calls and you recorded the token count before/after for one image.

### Pitfalls
- **Sending full-resolution phone photos.** A 12MP image at `detail: high` can cost thousands of tokens for zero quality gain on a receipt. Downscale and use low detail first; only escalate resolution if a field is unreadable.
- **Trusting VLM bounding boxes as exact.** They drift. Use them to *highlight* a region for a human, not to auto-crop for a downstream OCR that assumes pixel precision.
- **No arithmetic validation.** VLMs happily emit totals that don't sum. Deterministic checks catch fabrications the model is confident about.
- **Treating model `confidence` as calibrated probability.** Self-reported confidence is a weak signal — validate it against actual correctness on your golden set (Week 3) before you set the routing threshold.
- **Forgetting EXIF rotation.** Phone photos are often stored rotated; the model reads sideways text badly. Always `exif_transpose` first.

### Self-check
1. Walk through what happens to a 2000×1500 photo from bytes to LLM tokens. Where does the cost blow up?
2. When would you choose dedicated OCR over a VLM, and when is the hybrid worth its extra latency?
3. Why are VLM bounding boxes and object counts unreliable, and what do you do about it?
4. How do you decide a PDF is born-digital vs scanned, and why does it matter for cost?
5. Why is model-reported confidence not enough for routing, and what would make it trustworthy?

---

## Week 2 — Multimodal retrieval (ColPali) & speech: STT, TTS, and a realtime voice agent

**Hours breakdown:** ~4 hrs theory / ~10 hrs lab (14 total). The densest week — the voice agent is fiddly.

### Objectives
By the end of the week you can:
1. Explain **multimodal embeddings** (CLIP/SigLIP contrastive training) and **ColPali/ColQwen** late-interaction page-image retrieval, and say why ColPali can **skip OCR** entirely for visually rich documents.
2. Build **multimodal RAG over slide/PDF page images**: index pages, retrieve top-k, answer with a VLM, and return **page-number + bbox citations** and thumbnails.
3. Transcribe audio with **faster-whisper**, applying **VAD** gating, **word timestamps**, and speaker **diarization**.
4. Synthesize speech (**TTS**) with streaming and low time-to-first-byte.
5. Build a **cascaded realtime voice agent** (STT→LLM→TTS) with **barge-in**, **endpointing**, and a stated **<800ms** response budget — and articulate when speech-to-speech beats the cascade.

### Theory (~4 hrs)
> 📚 Lectures for this week: [Multimodal Embeddings: CLIP/SigLIP](lectures/06-multimodal-embeddings-clip-siglip.md) · [ColPali & Late Interaction](lectures/07-colpali-late-interaction.md) · [Multimodal RAG Architecture](lectures/08-multimodal-rag-architecture.md) · [Speech-to-Text: Whisper, VAD, Diarization](lectures/09-stt-asr-whisper-vad-diarization.md) · [Text-to-Speech: Streaming Synthesis](lectures/10-tts-streaming-synthesis.md) · [Realtime Voice Agents](lectures/11-realtime-voice-agent-architecture.md). Read them first. The bullets below are the recap.

- **Multimodal embeddings.** CLIP/SigLIP learn a shared image-text space via contrastive training, so `cosine(image, caption)` is meaningful. This powers text→image search. Read the **OpenCLIP** and **SigLIP** model cards (search `SigLIP Hugging Face`). Limitation: a single global vector per image loses fine detail — fine for "find the photo of a cat," weak for "which slide mentions Q3 churn."
- **ColPali / late interaction.** ColPali (and ColQwen2) treat a **document page as an image** and produce **multi-vector** embeddings (one per patch, ColBERT-style), scored by MaxSim late interaction. This captures text, tables, and layout **without an OCR/parsing pipeline** — you embed the rendered page and query it directly. Read the **ColPali paper's** intuition and the `vidore/colpali` model card + the **Byaldi** wrapper README (search `ColPali Byaldi GitHub`). Engineering payoff: for slide decks, scanned reports, and figure-heavy PDFs, ColPali often beats an OCR→text RAG baseline with far less pipeline. Cross-reference Phase 4 RAG — the *generation, citation, and grounding* discipline is identical; only the retrieval substrate changed.
- **Multimodal RAG shape.** Three patterns: (a) caption images → text-embed → text RAG (cheap, lossy); (b) multimodal-embed the raw modality; (c) ColPali page-image retrieval → feed retrieved **page images** to a VLM. Retrieve **tightly** (top-3, not top-20) because each retrieved page is expensive image tokens for the VLM.
- **STT/ASR.** **Whisper** (OpenAI) is the workhorse; **faster-whisper** (CTranslate2) is the fast local implementation; **WhisperX** adds accurate word timestamps + diarization. **VAD** (voice activity detection, e.g. Silero) gates silence — critical because Whisper **hallucinates text on silence**. **Diarization** (pyannote) answers "who spoke when." Read the **faster-whisper** and **WhisperX** GitHub READMEs.
- **Realtime voice agents.** Two architectures: **cascaded** STT→LLM→TTS (modular, debuggable, swap any component, but latency stacks) vs **speech-to-speech** (OpenAI Realtime, Gemini Live — lower latency, preserves prosody, but a black box). The hard problems are **endpointing** (deciding the user *finished* talking), **barge-in** (user interrupts; you must stop TTS instantly), and a **<800ms** turn budget so it feels conversational. Production stacks: **LiveKit Agents**, **Pipecat**, **OpenAI Realtime API**; transport is **WebRTC** (jitter/packet-loss handling) not raw WebSocket for real deployments. Read the **Pipecat** and **LiveKit Agents** GitHub docs. This is the Agents phase (Phase 6) with a real-time I/O layer bolted on — the loop, tool calls, and budgets are the same.

### Lab (~10 hrs) — two builds
> 🛠️ Full step-by-step guide: [Build SlideChat (ColPali RAG) and a Cascaded Realtime Voice Agent](labs/week-2-slidechat-and-voice-agent.md). The steps below are the summary.

**Build A: "Chat with your slide decks" (ColPali multimodal RAG) — ~5 hrs.**
```bash
uv init slidechat && cd slidechat
uv add byaldi pdf2image pillow litellm python-dotenv rich
# poppler needed by pdf2image: mac `brew install poppler`, ubuntu `apt install poppler-utils`
```
> **GPU note:** ColPali indexing wants a GPU. On a laptop, index a small deck (≤30 pages) on CPU (slow but works) **or** run indexing in a **free Colab/Modal** notebook and download the index. Query-time retrieval is cheap.

Steps:
1. Convert 2–3 real PDF decks to page images (`pdf2image.convert_from_path`, 150 DPI).
2. Index with Byaldi: `from byaldi import RAGMultiModalModel; model = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0"); model.index(input_path="decks/", index_name="decks", store_collection_with_index=True)`.
3. Retrieve: `results = model.search(query, k=3)` → each result has a page image + score + source page number.
4. Answer: send the retrieved **page images** + the question to a VLM (reuse Week 1's `litellm` client), instructing it to answer **and cite the page number** for each claim.
5. Return the answer + the top-3 page thumbnails + page-number citations. Have the VLM also give an approximate bbox on the cited page (reuse Week 1) for a highlight.

**Build B: cascaded realtime voice agent — ~5 hrs.**
```bash
uv add faster-whisper sounddevice numpy silero-vad
# TTS: pick ONE — piper (free local), or an API (ElevenLabs/Cartesia/OpenAI)
```
Minimal cascade (start non-streaming, then add streaming):
1. **Capture + VAD:** record mic with `sounddevice`; run **Silero VAD** to detect speech start/stop. Endpoint when you see ~500–700ms of trailing silence (tune this — too short cuts users off, too long feels laggy).
2. **STT:** feed the captured segment to `faster-whisper` (`model = WhisperModel("base.en", compute_type="int8")` for CPU) with `vad_filter=True` and `word_timestamps=True`.
3. **LLM:** send the transcript to your Phase 6 tool-using agent (or a plain chat call). Keep responses short — long TTS kills the latency budget.
4. **TTS:** stream synthesis (Piper locally, or ElevenLabs/Cartesia streaming API) and play as chunks arrive.
5. **Barge-in:** while TTS is playing, keep the VAD running; if the user starts speaking, **stop playback immediately** and start a new capture. This is the single most important UX feature.
6. **Measure the budget:** log timestamps at each stage boundary and print end-of-speech → first-audio-out latency. Target **<800ms**; report your p50/p95.
> **Level-up (optional):** re-implement Build B on **Pipecat** or the **OpenAI Realtime API** and compare latency + code complexity vs your hand-rolled cascade. Note where speech-to-speech wins.

### Definition of Done
- [ ] **SlideChat:** indexes ≥2 decks, and for **10/10** visual questions ("what's the number on the revenue slide?") returns an answer **with a correct page-number citation** and the top-3 thumbnails.
- [ ] SlideChat retrieves top-3 (not more) and you recorded the VLM image-token cost per query.
- [ ] **Voice agent:** transcribes your speech with word timestamps, and VAD filtering demonstrably suppresses a Whisper hallucination on a silent clip (show before/after).
- [ ] Diarization labels ≥2 speakers on a 2-speaker clip (WhisperX or pyannote).
- [ ] **Barge-in works:** you can interrupt the agent mid-sentence and it stops speaking within ~200ms and listens.
- [ ] You logged end-of-speech→first-audio latency across ≥10 turns and report p50/p95 against the 800ms budget.

### Pitfalls
- **Whisper hallucinating on silence/music.** Without VAD gating it emits phantom "Thank you for watching" text. Always `vad_filter=True` and gate on real speech.
- **ColPali retrieving too many pages.** Each page is a fat image for the VLM. Top-3 keeps cost and latency sane; top-20 is a bill and a slowdown.
- **Endpointing too aggressively.** Cutting the user off after 200ms of silence feels broken; 500–700ms is a better default. Tune on real speech, not on yourself reading a script.
- **Not stopping TTS on barge-in.** If the agent keeps talking over the user, the whole thing feels dead. Kill the audio buffer the instant VAD fires.
- **WebSocket in production voice.** Raw WebSocket has no jitter/packet-loss handling; real deployments use **WebRTC** (via LiveKit/Pipecat). Fine to prototype on WebSocket, but know why it won't survive a mobile network.

### Self-check
1. Why can ColPali skip the OCR/parse pipeline that a text RAG system needs, and when would you still prefer text RAG?
2. What is late interaction (MaxSim) and why does a single CLIP vector per image lose to it on document search?
3. Why does Whisper hallucinate, and how does VAD gating prevent it?
4. Name the three hard problems in realtime voice and the mechanism you used for each.
5. Cascaded vs speech-to-speech: give one concrete reason to pick each.

---

## Week 3 — Generation, model evaluation & cost/latency engineering

**Hours breakdown:** ~5 hrs theory / ~9 hrs lab (14 total). Ties everything into the milestones with an eval harness and the 60% cost cut.

### Objectives
By the end of the week you can:
1. Generate images with a diffusion model and explain the knobs that matter — **sampler, steps, CFG scale, seed, negative prompt** — plus where **SDXL/FLUX** fit and their text-in-image/hands caveats.
2. Apply **control & editing**: ControlNet (structure-conditioned) and **inpainting** (masked edits).
3. Do basic **video understanding** via **frame sampling + scene detection + ASR fusion**, and reason about its token-cost blowup.
4. Build a **task-specific multimodal eval harness** on a gold set reporting **CER/WER** (OCR/ASR), **field-level F1** (extraction), and **voice latency** — not leaderboard scores.
5. Cut the receipt pipeline's cost **≥60%** (downscale/crop, low-detail, cheap-model pre-filter, prompt caching, batch API) **without regressing accuracy**, proven on the eval set.

### Theory (~5 hrs)
> 📚 Lectures for this week: [Diffusion Fundamentals: The Knobs That Matter](lectures/12-diffusion-fundamentals-knobs.md) · [Controlled Generation & Editing](lectures/13-control-editing-controlnet-inpainting.md) · [Video Understanding Without a Token Bomb](lectures/14-video-understanding-frame-sampling.md) · [Evaluating Multimodal Systems on YOUR Task](lectures/15-multimodal-eval-harness.md) · [Cutting Cost 60% Without Regressing Accuracy](lectures/16-cost-latency-engineering.md). Read them first. The bullets below are the recap.

- **Diffusion fundamentals (engineering intuition only).** A diffusion model denoises random noise toward an image over N **steps**, guided by your prompt. **CFG scale** (classifier-free guidance) trades prompt-adherence vs diversity/artifacts (~5–8 typical; too high = fried images). **Sampler** (Euler, DPM++ 2M Karras, etc.) affects speed/quality per step. **Seed** makes generation reproducible. **Negative prompt** steers *away* from unwanted content (SDXL supports it; FLUX largely doesn't). Read the **Hugging Face `diffusers`** docs (root: `huggingface.co/docs/diffusers`) and the **SDXL** and **FLUX** model cards. Known caveats: older models mangle **text-in-image** and **hands**; FLUX and newer models are much better at text.
- **Control & editing.** **ControlNet** conditions generation on structure (edges, depth, pose) so you keep a layout while changing style. **Inpainting/outpainting** edits only a masked region. **IP-Adapter** conditions on a reference image; **LoRAs** inject a style/subject. You just need to know these exist and when to reach for them — skim the `diffusers` ControlNet + inpainting guides.
- **Video understanding.** You almost never feed every frame — that's a token catastrophe. Instead: **sample frames** (fixed interval or on **scene changes** via PySceneDetect), extract the **audio track and transcribe** (Week 2 Whisper), then fuse sampled-frame captions + transcript for the VLM/LLM. Read the **PySceneDetect** docs and `ffmpeg` frame-extraction examples. Video RAG = this, indexed.
- **Evaluating multimodal models on YOUR task.** Leaderboards lie about your data. Build a **gold set** and measure the metric that matches the job: **CER/WER** (character/word error rate) for OCR/ASR, **field-level precision/recall/F1** for structured extraction, **recall@k** for retrieval, **latency + task-success** for voice. This is the Phase 7 discipline applied to pixels and audio. Use `jiwer` for WER/CER. Evaluate ≥3 candidate models (e.g. Gemini Flash vs GPT-4o-mini vs a local Qwen2.5-VL) on the **same** gold set and pick on cost-per-correct-answer, not vibes.
- **Cost/latency control.** The levers, in order of bang-for-buck: **downscale + crop** to the region of interest before sending; **`detail: low`**; a **cheap-model pre-filter** (classify/reject junk before the expensive model); **prompt caching** (stable instruction prefix); **batch/async APIs** (often ~50% discount for non-realtime); and **self-hosting** a quantized VLM with vLLM at high volume. Cross-reference Phase 10 for the serving economics.

### Lab (~9 hrs) — three parts
> 🛠️ Full step-by-step guide: [Generation, Video, and the Milestone Glue: Eval Harness + 60% Cost Cut](labs/week-3-generation-eval-and-cost-milestone.md). The steps below are the summary.

**Part 1: image gen + control (~2.5 hrs).**
```bash
uv add diffusers transformers accelerate torch
```
> **GPU note:** SDXL/FLUX need a GPU. Use **free Colab/Modal** if your laptop lacks one, or call a hosted endpoint (**fal**/Replicate) for a few cents. Local CPU is impractically slow.
1. Generate the same prompt across `steps ∈ {10,20,40}`, `cfg ∈ {3,7,12}`, fixed `seed` — build a contact sheet and see each knob's effect.
2. Do one **inpainting** edit (mask a region, change it) with the `diffusers` inpaint pipeline.
3. One **ControlNet** run (Canny edge → new style) keeping structure.

**Part 2: video understanding (~1.5 hrs).**
```bash
uv add scenedetect
```
1. `ffmpeg`-extract audio from a short video; transcribe with faster-whisper (Week 2).
2. PySceneDetect to find scene boundaries; grab one frame per scene.
3. Send sampled frames + transcript to a VLM: "summarize this video with timestamps." Note the token count vs a naive every-frame approach.

**Part 3: eval harness + 60% cost cut (~5 hrs) — the milestone glue.**
1. **Gold set.** Hand-label 25–40 of your Week 1 receipts with the true field values (and known totals). Store as JSONL: `{id, image, gold_fields}`. Reuse your Phase 7 golden-set discipline (stratify: clean digital, phone photo, faded, foreign-currency).
2. **Metrics.** Write `eval.py`: for each doc, run extraction and compute **field-level precision/recall/F1** (exact match per field, normalized) and **CER** on text fields with `jiwer`. Report per-field F1 and an aggregate.
3. **Model bake-off.** Run the harness across **3 VLMs** (e.g. `gemini/gemini-2.0-flash`, `gpt-4o-mini`, `ollama/qwen2.5vl`). Produce a table: **F1, p50/p95 latency, $/1000 docs**. Pick a default with justification.
4. **Cost cut.** Establish a baseline $/1000 docs, then stack: (a) downscale to ≤1024px + crop to the receipt bbox; (b) `detail: low`; (c) a cheap-model **pre-filter** that rejects non-receipts before the main model; (d) **prompt caching** on the fixed instruction; (e) route the clean/high-confidence 80% through the **batch API** and only low-confidence docs through realtime. Re-run the eval after each change. **Target ≥60% cost reduction with F1 within 2 points of baseline.**

### Definition of Done
- [ ] Contact sheet shows the visible effect of steps/CFG/seed; one inpaint and one ControlNet result saved.
- [ ] Video summary produced from sampled frames + ASR, with a recorded token-count comparison vs every-frame.
- [ ] `eval.py` outputs per-field **F1** and **CER** on a **25–40 doc gold set**, and a **3-model** comparison table with **F1, p50/p95 latency, and $/1000 docs**.
- [ ] Receipt pipeline cost cut **≥60%** vs baseline with **F1 within 2 points** — proven by before/after eval runs, both committed.
- [ ] Confidence-routing threshold is now **calibrated against the gold set** (you can state precision of the "auto-approved" bucket).
- [ ] `pytest` green across the whole `docextract` repo.

### Pitfalls
- **CFG cranked too high.** Above ~12 SDXL images get saturated/fried. Adherence isn't monotonic in CFG.
- **Every-frame video.** Feeding all frames of even a 60s clip is a token bomb. Scene-sample + ASR fusion is the only sane default.
- **Optimizing cost before measuring accuracy.** Cut cost and F1 quietly drops 8 points and you don't notice. Always re-run the eval after each optimization — the "within 2 points" guard is the point.
- **WER/CER without normalization.** Case, punctuation, and whitespace differences inflate error. Normalize (lowercase, strip punctuation) consistently on both sides with `jiwer` transforms.
- **Batch API for latency-sensitive traffic.** Batch is cheap but async/slow (minutes–hours). Only route non-realtime docs to it; keep interactive requests on the realtime endpoint.

### Self-check
1. What does CFG scale trade off, and what happens at the extremes?
2. Why is negative prompting available on SDXL but not really on FLUX, and when do you need it?
3. How do you keep video understanding from exploding your token budget?
4. Which metric (CER/WER vs field-F1 vs recall@k vs latency) matches which multimodal task, and why not just use a leaderboard?
5. List your cost levers in priority order and the one guardrail that keeps a cost cut honest.

---

## Phase milestone project

Ship **two** integrative artifacts (they share the `docextract` and `slidechat` repos you built across the weeks) and one eval harness that binds them.

### Milestone A — Receipt/Invoice → JSON pipeline
A phone photo goes in; validated ERP-ready JSON comes out.
- **Input:** phone photo or PDF → auto-rotate (EXIF) → born-digital-vs-scanned detection → downscale/crop.
- **Extraction:** VLM (or hybrid) → schema-validated `Receipt` JSON with **per-field bounding boxes** and **calibrated confidence**.
- **Validation:** arithmetic checks (line items sum to subtotal; subtotal + tax = total); low-confidence or failed-arithmetic fields **routed to a review queue**.
- **Cost:** documented **≥60% cost reduction** vs the naive baseline (downscale/crop + low-detail + cheap pre-filter + prompt caching + batch), with **F1 held within 2 points**, proven by committed before/after eval runs.
- **Acceptance:** 20/20 samples produce valid JSON; the eval harness reports per-field F1 ≥ your stated target; the review queue contains exactly the docs that failed validation; a review UI (even a simple HTML/Streamlit page) renders the image with bounding-box highlights on flagged fields.

### Milestone B — "Chat with your slide decks" (ColPali)
- Index PDF/deck **page images** with ColPali/ColQwen (Byaldi); retrieve **top-3** pages; answer with a VLM including **page-number citations + thumbnails**.
- **A/B baseline:** compare against a text-only OCR→text RAG baseline on a set of **visual** questions (charts, tables, figures) and report which wins and why.
- **Acceptance:** 10/10 visual questions answered with correct page citations; recall@3 measured against a small labeled query→page gold set; per-query VLM image-token cost recorded.

### Shared eval harness
- One reusable harness reporting, across ≥3 VLMs: **field accuracy/F1** (extraction), **CER/WER** where relevant, **recall@k** (retrieval), **p50/p95 latency**, and **$/1000 docs**. This is your Phase 7 discipline made multimodal — leaderboards do not ship your product.

### Suggested repo layout
```
multimodal-milestone/
  docextract/            # Milestone A (Weeks 1 & 3)
    src/{schema,detect,ocr,extract_vlm,extract_hybrid,route}.py
    eval/{gold.jsonl,eval.py,runs/}       # baseline vs optimized
    review_ui/                             # streamlit/html box-highlight viewer
    tests/
  slidechat/             # Milestone B (Week 2)
    src/{index,retrieve,answer}.py
    eval/{queries.jsonl,eval.py}
    decks/
  voice_agent/           # Week 2 Build B (bonus, cross-ref Phase 6)
    src/{capture_vad,stt,agent,tts,bargein}.py
  eval_harness/          # shared multimodal metrics (jiwer, F1, recall@k, latency, cost)
  README.md              # architecture, model choices, cost writeup, tradeoffs
```

## You are ready to move on when...
- [ ] You can explain, end to end, how an image becomes LLM tokens and predict where cost blows up before you send it.
- [ ] You can get **schema-validated JSON with boxes + confidence** out of a VLM and route low-confidence fields to humans deterministically.
- [ ] You can justify OCR vs VLM vs hybrid, and born-digital vs scanned routing, per document type.
- [ ] You can stand up **ColPali page-image RAG** with page+bbox citations and A/B it against an OCR baseline.
- [ ] You can build a **cascaded voice agent** with VAD gating, endpointing, barge-in, and a measured <800ms budget — and say when speech-to-speech wins.
- [ ] You can drive a **diffusion** model with intent (steps/CFG/seed/negative), do inpainting/ControlNet, and sample video sanely.
- [ ] You can build a **task-specific multimodal eval harness** (CER/WER, field F1, recall@k, latency, $/doc) and pick models on cost-per-correct-answer.
- [ ] You **cut a real pipeline's cost ≥60%** without regressing accuracy, proven by evals — the whole job in one line.
