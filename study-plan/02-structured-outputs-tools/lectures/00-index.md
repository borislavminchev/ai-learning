# Phase 2 — Lectures Index

Deep, textbook-style lectures backing the [Phase 2 spine](../02-structured-outputs-tools.md). The spine's **Theory** sections are short recaps; these are the full treatment — mechanism from first principles, worked examples, provider specifics, production consequences, misconceptions, and self-checks.

## How to use

- Read the week's lectures **before** the matching [lab guide](../labs/). The spine's Theory recap tells you the headline; the lecture gives you the depth; the lab makes you build it.
- Each lecture is ~20 min. Budget the spine's stated "Theory hrs" per week for lecture reading, then spend the rest of the week's hours in the lab.
- Don't rabbit-hole. Read once for the mechanism + the gotchas, do the self-check, then go build.

## Week 1 — Structured Outputs: from prompt-and-pray to schema-constrained contracts

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 1 | [When (and When Not) to Force Structured Output](01-when-to-use-structured-output.md) | decision framework, reasoning cost |
| 2 | [The Three Reliability Tiers and Which Provider Knob Delivers Each](02-three-reliability-tiers.md) | prompt-and-pray → JSON mode → Structured Outputs |
| 3 | [Provider JSON-Schema Subsets and Strict-Mode Gotchas](03-provider-schema-subsets-and-strict-mode.md) | OpenAI/Anthropic/Gemini schema quirks |
| 4 | [Designing LLM-Friendly Schemas with Pydantic v2 (and a Zod Primer)](04-llm-friendly-schema-design.md) | flat schemas, enums, descriptions, Zod |
| 5 | [Runtime Resilience: Repair Loops, Error Classification, and Streaming Partial JSON](05-repair-loops-and-streaming.md) | retry/repair, error classes, partial JSON |

## Week 2 — Tool / Function Calling: the universal loop, hardening, constrained decoding, and MCP

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 6 | [The Universal Tool-Calling Loop](06-universal-tool-calling-loop.md) | send tools → request → execute → return → loop |
| 7 | [Provider Wire-Formats and the Normalizing Adapter](07-provider-wire-formats-and-adapters.md) | tool_calls vs tool_use vs functionCall |
| 8 | [Tool Definitions as Prompt Engineering and the tool_choice Knob](08-tool-definitions-and-tool-choice.md) | naming/descriptions, auto/none/required/forced |
| 9 | [Parallel vs Serial Tool Execution](09-parallel-vs-serial-tool-execution.md) | concurrency, id matching, dependency serialization |
| 10 | [Security: Tool Arguments Are Untrusted Input](10-securing-tool-calls.md) | allowlists, parameterization, sandboxing, HITL |
| 11 | [Constrained Decoding and Grammars on Self-Hosted Models](11-constrained-decoding-and-grammars.md) | Outlines/XGrammar, logit masking, limits |
| 12 | [MCP: Standardizing Tools, the Server/Client Model, and Its Security Surface](12-mcp-server-and-client.md) | MCP protocol, FastMCP, security surface |

## Labs (step-by-step guides)

- [Week 1 Lab — Build structio: A Schema-Constrained Extraction Toolkit](../labs/week-1-structio-extraction-toolkit.md)
- [Week 2 Lab — Extend structio into a Hardened, Provider-Agnostic Tool-Loop Extraction Service (Phase Milestone)](../labs/week-2-hardened-tool-loop-and-extraction-service.md)
