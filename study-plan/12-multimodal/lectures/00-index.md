# Phase 12 — Lectures Index

Deep, textbook-style lectures backing the [Phase 12 spine](../12-multimodal.md). The spine's **Theory** sections are short recaps; these are the full treatment — mechanism from first principles, worked numeric examples, production consequences, misconceptions, cheat sheets, and a self-check with answers.

## How to use

- Read the week's lectures **before** the matching [lab guide](../labs/). The spine's Theory recap tells you the headline; the lecture gives you the depth; the lab makes you build it.
- Don't rabbit-hole. Read once for the mechanism + the cheat sheet, do the self-check, then move to the lab. You'll cement the rest by building.

## Week 1 — Vision: VLMs, structured image understanding, OCR & document extraction

| # | Lecture |
|---|---------|
| 1 | [How VLMs Actually Work: Encoder, Projector, LLM](01-vlm-architecture.md) |
| 2 | [Image Tokens, Tiling, and Estimating Cost Before You Send](02-image-tokens-tiling-cost.md) |
| 3 | [Schema-Validated Extraction: Pydantic + instructor, BBoxes, Confidence, and Grounding Limits](03-structured-vlm-extraction.md) |
| 4 | [OCR vs VLM vs Hybrid, and Born-Digital vs Scanned Routing](04-ocr-vs-vlm-hybrid-routing.md) |
| 5 | [LLM Proposes, Code Disposes: Deterministic Validation and Human-Review Routing](05-confidence-routing-validation.md) |

## Week 2 — Multimodal retrieval (ColPali) & speech: STT, TTS, and a realtime voice agent

| # | Lecture |
|---|---------|
| 6 | [Multimodal Embeddings: CLIP/SigLIP and the Shared Image-Text Space](06-multimodal-embeddings-clip-siglip.md) |
| 7 | [ColPali & Late Interaction: OCR-Free Page-Image Retrieval](07-colpali-late-interaction.md) |
| 8 | [Multimodal RAG: Three Patterns, Tight Retrieval, and Grounded VLM Answers](08-multimodal-rag-architecture.md) |
| 9 | [Speech-to-Text: Whisper, faster-whisper, VAD Gating, Timestamps, and Diarization](09-stt-asr-whisper-vad-diarization.md) |
| 10 | [Text-to-Speech: Streaming Synthesis and Time-to-First-Byte](10-tts-streaming-synthesis.md) |
| 11 | [Realtime Voice Agents: Cascaded vs Speech-to-Speech, Endpointing, Barge-In, Latency Budget](11-realtime-voice-agent-architecture.md) |

## Week 3 — Generation, model evaluation & cost/latency engineering

| # | Lecture |
|---|---------|
| 12 | [Diffusion Fundamentals: The Knobs That Matter (Steps, CFG, Sampler, Seed, Negative Prompt)](12-diffusion-fundamentals-knobs.md) |
| 13 | [Controlled Generation & Editing: ControlNet, Inpainting, IP-Adapter, LoRA](13-control-editing-controlnet-inpainting.md) |
| 14 | [Video Understanding Without a Token Bomb: Frame Sampling, Scene Detection, ASR Fusion](14-video-understanding-frame-sampling.md) |
| 15 | [Evaluating Multimodal Systems on YOUR Task: CER/WER, Field-F1, Recall@k, Latency, Cost](15-multimodal-eval-harness.md) |
| 16 | [Cutting Cost 60% Without Regressing Accuracy: The Lever Stack and the Honesty Guardrail](16-cost-latency-engineering.md) |

## Labs (step-by-step guides)

- [Week 1 Lab — Build docextract: Document Image to Validated JSON with BBoxes, Confidence, and Review Routing](../labs/week-1-docextract-service.md)
- [Week 2 Lab — Build SlideChat (ColPali RAG) and a Cascaded Realtime Voice Agent](../labs/week-2-slidechat-and-voice-agent.md)
- [Week 3 Lab — Generation, Video, and the Milestone Glue: Eval Harness + 60% Cost Cut](../labs/week-3-generation-eval-and-cost-milestone.md)
