# Phase 09 — Lectures Index

Deep, textbook-style lectures backing the [Phase 09 spine](../09-architecture-system-design.md). The spine's **Theory** sections are short recaps; these are the full treatment — mechanism from first principles, worked numeric examples, production consequences, misconceptions, cheat sheets, and a self-check with answers.

## How to use

- Read the week's lectures **before** the matching [lab guide](../labs/). The spine's Theory recap tells you the headline; the lecture gives you the depth; the lab makes you build it.
- Each lecture is ~20 min. Budget the week's "Theory hrs" (from the spine) on lecture reading, then spend the rest of the week's hours in the lab.
- Don't rabbit-hole. Read once for the mechanism + the cheat sheet, do the self-check, move to the lab. You'll cement the rest by building.

## Week 1 — Reference architectures, state, and provable GDPR delete

| # | Lecture |
|---|---------|
| 1 | [The Five Reference Architectures and the Escalation Ladder](01-five-reference-architectures.md) |
| 2 | [LLM Proposes, Code Disposes: The Deterministic Boundary](02-deterministic-code-vs-llm-leaves.md) |
| 3 | [Four-Tier State: Redis, Postgres, Object Storage, Vector](03-state-and-storage-tiers.md) |
| 4 | [Context-Window Management: Windowing, Compaction, and Retrieval Memory](04-context-window-management.md) |
| 5 | [Right-to-Erasure as an Architecture Constraint: Provable Cascade Delete](05-gdpr-erasure-as-architecture.md) |

## Week 2 — The resilient LLM gateway: routing, fallback, caching, limits

| # | Lecture |
|---|---------|
| 6 | [The LLM Gateway: Unified Router, Provider Abstraction, and Leaky Seams](06-gateway-router-pattern.md) |
| 7 | [Resilience: Fallback Chains and Circuit Breakers](07-circuit-breakers-and-fallback.md) |
| 8 | [Streaming Tokens: SSE vs WebSocket, TTFT, and Upstream Cancellation](08-sse-streaming-ttft-cancellation.md) |
| 9 | [Async and Queue-Based Inference for Long and Batch Jobs](09-async-queue-based-inference.md) |
| 10 | [Three Caching Layers and the Tenant-Isolation Iron Rule](10-caching-layers.md) |
| 11 | [Token-Aware Rate Limiting and the Hard Spend Kill-Switch](11-rate-limiting-and-spend-governance.md) |
| 12 | [Secrets, Provider Keys, and Per-Tenant BYOK](12-secrets-and-key-management.md) |

## Week 3 — Evaluability, observability, degradation, and system-design practice

| # | Lecture |
|---|---------|
| 13 | [Designing for Evaluability: Tracing, Versioning, and the Feedback Flywheel](13-evaluability-and-observability.md) |
| 14 | [Capacity Planning, Provider Quota, and Graceful Degradation](14-scalability-capacity-and-degradation.md) |
| 15 | [Build-vs-Buy and Compliance-Architecture Trust Boundaries](15-build-vs-buy-and-privacy-boundaries.md) |
| 16 | [The 45-Minute LLM System-Design Method](16-system-design-method.md) |
| 17 | [PII Redaction Across the Observability and Feedback Pipeline](17-privacy-in-observability-pipelines.md) |

## Labs (step-by-step guides)

- [Week 1 Lab — Stand Up the Four Stores, Stateful Chat, and Provable GDPR Delete](../labs/week-1-state-and-gdpr-delete.md)
- [Week 2 Lab — Build the Resilient Gateway: Router, Fallback, Streaming, Cache, Limits](../labs/week-2-resilient-gateway.md)
- [Week 3 Lab — Observability, Degradation Mode, Load Test, Capacity Math, and Three Mock Designs (Milestone)](../labs/week-3-observability-degradation-and-designs.md)
