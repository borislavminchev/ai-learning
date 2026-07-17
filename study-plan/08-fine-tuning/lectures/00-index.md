# Phase 08 Lectures — Model Adaptation & Fine-Tuning

These are the deep, textbook-style lectures that back the phase **spine** ([`../08-fine-tuning.md`](../08-fine-tuning.md)). The spine is a weekly plan and a *recap*; these lectures are the *why* and the mechanism behind every bullet in it.

**Read each week's lectures BEFORE that week's lab.** The Theory section in the spine is a compressed recap of the lectures below — work the lab only once the mechanism is clear.

## Lectures by week

| Week | # | Lecture |
|------|---|---------|
| 1 | 01 | [The Adaptation Decision Framework: Prompt vs RAG vs Fine-Tune](01-prompt-vs-rag-vs-finetune-decision.md) |
| 1 | 02 | [SFT Mechanics and the Three Silent Killers](02-sft-mechanics-and-the-three-silent-killers.md) |
| 1 | 03 | [Base vs Instruct Starting Points and Hyperparameters for Small Data](03-base-vs-instruct-and-small-data-hyperparameters.md) |
| 2 | 04 | [LoRA: Ranks, Targets, and the Adapter Lifecycle](04-lora-mechanics-and-adapter-lifecycle.md) |
| 2 | 05 | [QLoRA: 4-bit NF4 Training on a Single Cheap GPU](05-qlora-4bit-training.md) |
| 2 | 06 | [VRAM Math and PEFT vs Full Fine-Tuning Budgets](06-vram-math-and-peft-vs-full-ft.md) |
| 2 | 07 | [Dataset Quality, Curation, and Decontamination](07-dataset-quality-and-decontamination.md) |
| 3 | 08 | [DPO: Preference Tuning After SFT](08-dpo-preference-tuning.md) |
| 3 | 09 | [The Preference-Method Zoo and Response-Level Distillation](09-preference-method-zoo-and-distillation.md) |
| 3 | 10 | [Quantization for Serving: GPTQ, AWQ, GGUF, and fp8](10-quantization-for-serving.md) |
| 4 | 11 | [GPU Planning and Choosing a Training Venue](11-gpu-planning-and-training-venues.md) |
| 4 | 12 | [Evaluating a Fine-Tune: Task Metrics, Judge Win-Rate, and Retention](12-three-legged-finetune-evaluation.md) |
| 4 | 13 | [Catastrophic Forgetting: Causing It and Fixing It](13-catastrophic-forgetting-and-mitigation.md) |
| 4 | 14 | [Fine-Tuning Embedding Models with Hard-Negative Mining](14-embedding-model-finetuning.md) |

## Lab guides

Follow these to *build*. Each lab assumes you have read that week's lectures above.

- **Week 1** — [Repo Skeleton and Your First SFT with Masking A/B](../labs/week-1-repo-skeleton-and-first-sft.md)
- **Week 2** — [QLoRA a 7-8B Model and Manage Adapters](../labs/week-2-qlora-7b-and-adapter-management.md)
- **Week 3** — [DPO Preference Tuning and Compressed Serving Artifacts](../labs/week-3-dpo-and-serving-artifacts.md)
- **Week 4** — [Retention Harness, Forgetting Demo, Embedder Tune, and Milestone Deliverables](../labs/week-4-eval-harness-forgetting-embedder-and-milestone.md)
