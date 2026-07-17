# Phase 1 — Lectures Index

Deep, textbook-style lectures backing the [Phase 1 spine](../01-prompting-context.md). The spine's **Theory** sections are short recaps; these are the full treatment — mechanism from first principles, worked numeric examples, production consequences, misconceptions, cheat sheets, and a self-check with answers.

## How to use

- Read the week's lectures **before** the matching [lab guide](../labs/). The spine's Theory recap tells you the headline; the lecture gives you the depth; the lab makes you build it.
- Each lecture is ~20 min. Budget ~2 hrs of lecture reading per week (the spine's "Theory hrs"), then spend the rest of the week's hours in the lab.
- Don't rabbit-hole. Read once for the mechanism + the cheat sheet, do the self-check, move to the lab. You'll cement the rest by building.

## Week 1 — Prompt anatomy, provider idioms, and output-format control

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 1 | [Prompt Anatomy and Section Ordering](01-prompt-anatomy.md) | prompt structure, recency/primacy, caching order |
| 2 | [Few-Shot Exemplar Engineering](02-few-shot-exemplar-engineering.md) | byte-identical exemplars, zero- vs few-shot |
| 3 | [Provider Idioms and Reasoning Controls](03-provider-idioms-reasoning-controls.md) | Anthropic/OpenAI/Gemini idioms, effort |
| 4 | [Forcing and Validating Structured JSON Output](04-structured-json-output.md) | native JSON mechanisms, validation |
| 5 | [Per-Provider Token Counting](05-token-counting.md) | count_tokens vs tiktoken |

## Week 2 — Reasoning prompting, templating & versioning, decomposition/chaining

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 6 | [Chain-of-Thought, Correctly](06-chain-of-thought.md) | when to hand-write CoT vs defer |
| 7 | [Self-Consistency and the Cost Multiplier](07-self-consistency.md) | sample-N majority vote, cost tradeoff |
| 8 | [Prompt Management as Software: Templating and Versioning](08-prompt-templating-versioning.md) | Jinja2 templates, versioning, cache drift |
| 9 | [Task Decomposition and Prompt Chaining](09-decomposition-chaining.md) | typed steps, error propagation |
| 10 | [Eval-Driven Prompt Development with promptfoo](10-promptfoo-eval-gates.md) | eval suites, assertions, gates |

## Week 3 — Context engineering: budgeting, caching, compaction, and the failure taxonomy

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 11 | [Context Engineering and Token Budgeting](11-context-budgeting.md) | per-category token budget, context rot |
| 12 | [Prompt Caching Mechanics](12-prompt-caching.md) | prefix match, cache_control, invalidators |
| 13 | [Rolling Compaction and Summarization](13-rolling-compaction.md) | threshold-triggered, entity-preserving |
| 14 | [Long-Context Handling vs RAG](14-long-context-vs-rag.md) | chunking, when full-context beats RAG |
| 15 | [Prompt Failure Taxonomy and Matched Fixes](15-failure-taxonomy-fixes.md) | injection, format drift, hallucination fixes |

## Labs (step-by-step guides)

- [Week 1 Lab — Provider-Agnostic Invoice Extraction Harness](../labs/week-1-invoice-extraction-harness.md)
- [Week 2 Lab — Jinja2 Prompt Registry, Reasoning Experiments, and promptfoo Gate](../labs/week-2-templating-reasoning-eval.md)
- [Week 3 Lab — Context-Budget and Cache-Optimization Instrument (Phase Milestone)](../labs/week-3-context-cache-instrument.md)
