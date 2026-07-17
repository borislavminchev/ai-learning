# Capstone — Lectures Index

Deep **design/decision** lectures backing the [Capstone spine](../14-capstone.md) — a 4-week integrative build of a multi-tenant, production-grade **Enterprise Knowledge & Action Platform** for a regulated B2B domain. These lectures are integration notes (how the subsystems compose, the architecture decisions, the tradeoffs, the failure modes) — not first-principles re-teaching; they reference the earlier phases that taught each piece.

## How to use

- Read the week's lectures **before** starting that week's [lab guide](../labs/).
- Each week's lab has hard proof gates (crash-and-resume, GDPR delete purges every store, tenant isolation, injection red-team fails). The lectures tell you *how to design for* those gates.
- Keep the suggested monorepo (`apps/`, `packages/`, `infra/`, `evals/`, `docs/`) from the Week 1 lab as the single home for all four weeks.

## Week 1 — Data pipeline + retrieval backbone

| # | Lecture |
|---|---------|
| 1 | [Capstone Integration Map](01-capstone-integration-map.md) — how the four weeks fit together and why the retrieval spine comes first |
| 2 | [Ingestion, OCR & Chunk Provenance Decisions](02-ingestion-ocr-and-chunk-provenance-decisions.md) — scan detection, OCR routing, bbox provenance, contextual chunking |
| 3 | [Tenant Isolation & ACL as a Server-Side Boundary](03-tenant-isolation-and-acl-as-a-server-side-boundary.md) — enforcing per-tenant access in the store query, not in Python |
| 4 | [Mutation, Versioning & CAG-vs-Retrieval](04-mutation-versioning-and-cag-vs-retrieval.md) — tombstone deletes, blue-green re-embedding, dataset versioning, when to choose CAG |

→ Lab: [Week 1 — Data & Retrieval Backbone](../labs/week-1-data-and-retrieval-backbone.md)

## Week 2 — Agentic layer

| # | Lecture |
|---|---------|
| 5 | [Supervisor Topology & Typed Tool Contracts](05-supervisor-topology-and-typed-tool-contracts.md) — orchestrator-worker, typed tools as contract + audit record |
| 6 | [Durability: Checkpointing & Idempotent Writes](06-durability-checkpointing-and-idempotent-writes.md) — crash-safe resume with no duplicated side effects |
| 7 | [HITL Gating & the Budget Kill-Switch](07-hitl-gating-and-budget-kill-switch.md) — human approval on destructive actions, per-run token/$ budgets |
| 8 | [End-User OAuth Scopes across MCP and A2A](08-end-user-oauth-scopes-across-mcp-and-a2a.md) — least-privilege tool access keyed to the end user |

→ Lab: [Week 2 — Agentic Layer](../labs/week-2-agentic-layer.md)

## Week 3 — Architecture & serving

| # | Lecture |
|---|---------|
| 9 | [The Gateway as Single Egress](09-the-gateway-as-single-egress.md) — one door for every model call; app imports zero vendor SDKs |
| 10 | [Fallback, Cascade & Caching Decisions](10-fallback-cascade-and-caching-decisions.md) — reliability fallback vs cost cascade; exact + semantic cache |
| 11 | [Multi-Tenant Fairness, Quotas & Kill-Switch](11-multi-tenant-fairness-quotas-and-kill-switch.md) — per-tenant RPM/TPM + spend caps; the fairness proof |
| 12 | [Safe Release: Eval-Gated CI/CD and Rollout](12-safe-release-eval-gated-cicd-and-rollout.md) — eval gate, shadow → canary → flag, instant rollback |

→ Lab: [Week 3 — Serving & CI/CD](../labs/week-3-serving-and-cicd.md)

## Week 4 — Eval, observability, safety & governance

| # | Lecture |
|---|---------|
| 13 | [Eval as a Set, with Confidence Intervals](13-eval-as-a-set-with-confidence-intervals.md) — split metrics by concern; the ship rule as a CI-checkable function |
| 14 | [Calibrated Judges & the Observability Roll-Up](14-calibrated-judges-and-observability-rollup.md) — pinned/calibrated LLM judges; OTel tracing + cost/latency/quality dashboards |
| 15 | [Threat Model, the Lethal Trifecta & Injection Defense](15-threat-model-trifecta-and-injection-defense.md) — break one trifecta leg per action; PII redaction at every sink |
| 16 | [Governance: GDPR Erasure and Audit](16-governance-gdpr-erasure-and-audit.md) — cascade delete across every store, NIST-AI-RMF, model card, audit logging |

→ Lab: [Week 4 — Eval, Safety, Governance & Final Deliverable](../labs/week-4-eval-safety-governance-and-final.md)
