# Phase 7 — Lectures Index (Evaluation, Testing & Observability)

These are the **deep lectures** that back the spine: [`../07-evaluation-observability.md`](../07-evaluation-observability.md). The spine is a weekly plan and a *recap*; each lecture below is the textbook-style treatment of one concept — the *why* and the mechanism behind the bullets in the spine's Theory sections.

**How to read this phase:** for each week, read that week's lectures **before** you start that week's lab. The spine's Theory section is the recap of these lectures; the spine's Lab section is the summary of the lab guide. Read deep here first, then build.

## Lectures by week

### Week 1 — Eval-driven development: traces, error analysis & golden sets
| # | Lecture |
|---|---------|
| 1 | [Eval-Driven Development and the Goal vs Guardrail Split](01-eval-driven-development.md) |
| 2 | [Error Analysis: From Raw Traces to a Failure Taxonomy](02-error-analysis-to-taxonomy.md) |
| 3 | [The Three Eval Families: Reference, Criteria, and Human](03-three-eval-families.md) |
| 4 | [Building Golden Sets: Stratified, Versioned, Decontaminated](04-golden-sets-done-right.md) |

### Week 2 — LLM-as-judge, its biases, and statistical rigor
| # | Lecture |
|---|---------|
| 5 | [LLM-as-Judge Mechanics: Modes, CoT-First, Structured Verdicts](05-llm-as-judge-mechanics.md) |
| 6 | [Judge Biases and Their Mitigations](06-judge-biases.md) |
| 7 | [Calibrating the Judge Against Humans with Cohen's Kappa](07-judge-calibration-kappa.md) |
| 8 | [RAG Evaluation and Hallucination Detection](08-rag-eval-hallucination.md) |
| 9 | [Statistical Rigor: Bootstrap CIs and Paired Tests](09-statistical-rigor.md) |

### Week 3 — Observability, the data flywheel & a CI eval gate
| # | Lecture |
|---|---------|
| 10 | [OpenTelemetry GenAI Tracing with OpenLLMetry](10-otel-genai-openllmetry.md) |
| 11 | [Observability Platforms: Langfuse, Phoenix, and Hosted Options](11-observability-platforms.md) |
| 12 | [Feedback Capture and the Data Flywheel](12-feedback-and-flywheel.md) |
| 13 | [Drift, Silent Model Swaps, and Sampled Online Evals](13-drift-online-eval.md) |
| 14 | [CI Eval Gates with promptfoo and pytest](14-ci-eval-gate.md) |

## Lab guides

Work each week's lab **after** reading that week's lectures:

- **Week 1** — [Traces to Golden Set: Error Analysis and a Versioned Eval Corpus](../labs/week-1-golden-set-and-error-analysis.md)
- **Week 2** — [A Calibrated, De-biased Judge with Statistical Rigor](../labs/week-2-calibrated-judge-and-stats.md)
- **Week 3** — [Milestone: Observability, Data Flywheel, and a Blocking CI Eval Gate](../labs/week-3-observability-flywheel-ci-gate.md)
