# Phase 6 — Lectures Index

Deep, textbook-style lectures backing the [Phase 6 spine](../06-agents.md) (*AI Agents & Agentic Systems — Expanded Deep Track*). The spine's **Theory** sections are short recaps; these lectures are the full treatment — mechanism from first principles, worked examples, production consequences, failure modes, and self-checks.

## How to use

- Read each week's lectures **before** you start that week's [lab guide](../labs/). The spine's Theory recap gives the headline; the lecture gives the depth; the lab makes you build it.
- Don't rabbit-hole. Read for the mechanism and the tradeoffs, then move to the lab — you cement the rest by building the week's shippable artifact.

## Week 1 — Agent Fundamentals From Scratch: The Loop, Not the Prompt

| # | Lecture |
|---|---------|
| 1 | [The Agent Loop: Loop, Not Prompt](01-the-agent-loop.md) |
| 2 | [ReAct with Native Tool Calling](02-react-native-tool-calling.md) |
| 3 | [Errors-as-Observations & Robust Tool Design](03-errors-as-observations.md) |
| 4 | [Bounded Loops, Budgets & Kill Switches (Defense in Depth)](04-budgets-and-kill-switches.md) |
| 5 | [Determinism Levers & Structured Tracing](05-determinism-and-tracing.md) |

## Week 2 — Planning & Control-Flow Patterns

| # | Lecture |
|---|---------|
| 6 | [Control-Flow Patterns: The Landscape & ReAct's Cost Model](06-control-flow-landscape-react-cost.md) |
| 7 | [Plan-and-Execute](07-plan-and-execute.md) |
| 8 | [ReWOO: Reasoning WithOut Observation](08-rewoo.md) |
| 9 | [Parallelism, Reflection & Search: LLM Compiler, Reflexion, LATS](09-parallelism-reflection-search.md) |
| 10 | [The Workflow-vs-Autonomous Spectrum, Building Blocks & Routing](10-workflow-autonomous-spectrum-routing.md) |

## Week 3 — Frameworks by Control Model + Durable, Crash-Safe Execution

| # | Lecture |
|---|---------|
| 11 | [Choosing a Framework by Control Model](11-frameworks-by-control-model.md) |
| 12 | [Durable Execution & Checkpointing](12-durable-execution-checkpointing.md) |
| 13 | [Idempotency & Safe Replay of Side Effects](13-idempotency-safe-replay.md) |
| 14 | [Human-in-the-Loop Interrupts, Time-Travel & Async Agents](14-hitl-interrupts-time-travel.md) |

## Week 4 — Interop Protocols: MCP, A2A, Identity & the "Protocol Soup"

| # | Lecture |
|---|---------|
| 15 | [MCP Deep Dive: Primitives, Transports & Discovery](15-mcp-deep-dive.md) |
| 16 | [MCP Security: The Four Failure Modes & Mitigations](16-mcp-security-failure-modes.md) |
| 17 | [A2A: Agent Cards, Task Lifecycle & Streaming](17-a2a-protocol.md) |
| 18 | [MCP vs A2A: Vertical vs Horizontal & Composition](18-mcp-vs-a2a-composition.md) |
| 19 | [Agent Identity & Auth: OAuth 2.1, Scoped Tokens & Token Exchange](19-agent-identity-and-auth.md) |
| 20 | [The Protocol Landscape: ACP, AGNTCY, AG-UI & Agentic Payments](20-protocol-landscape-and-payments.md) |

## Week 5 — Memory, Multi-Agent Orchestration & Context Engineering at Scale

| # | Lecture |
|---|---------|
| 21 | [Short-Term Memory as a Token Budget](21-short-term-memory-token-budget.md) |
| 22 | [Long-Term Memory: The Write Path & Memory Systems](22-long-term-memory-write-path.md) |
| 23 | [Memory Poisoning: A Persistent Attack Surface](23-memory-poisoning-defenses.md) |
| 24 | [Context Engineering for Long-Horizon Agents](24-context-engineering-long-horizon.md) |
| 25 | [Multi-Agent Orchestration Topologies & When a Single Agent Wins](25-multiagent-topologies.md) |

## Week 6 — Production-Grade Agents: Computer Use, Coding Agents, Guardrails & Trajectory Eval

| # | Lecture |
|---|---------|
| 26 | [Computer Use & Browser Control: Vision vs Accessibility Tree](26-computer-use-browser-control.md) |
| 27 | [Sandboxing Untrusted Agent Execution](27-sandboxing-untrusted-execution.md) |
| 28 | [Coding Agents as a Category](28-coding-agents-category.md) |
| 29 | [Prompt Injection & Guardrails: The Lethal Trifecta](29-prompt-injection-lethal-trifecta.md) |
| 30 | [Trajectory Evaluation, Variance & Trace Harvesting](30-trajectory-evaluation-variance.md) |

## Labs (step-by-step guides)

- [Week 1 Lab — Build a Bounded ReAct Agent on the Raw SDK](../labs/week-1-bounded-react-agent.md)
- [Week 2 Lab — Plan-and-Execute vs ReWOO: Instrument, Compare, Route](../labs/week-2-plan-execute-vs-rewoo.md)
- [Week 3 Lab — Port to LangGraph: Durable, Interruptible, Crash-Safe](../labs/week-3-durable-langgraph-agent.md)
- [Week 4 Lab — MCP + A2A + OAuth: A Multi-Agent Interop Repo](../labs/week-4-mcp-a2a-oauth-interop.md)
- [Week 5 Lab — Supervisor + Specialists with Persistent, Guarded Memory](../labs/week-5-memory-multiagent.md)
- [Week 6 Lab — Coding Agent, Trajectory Eval, Red-Team & the Warden Capstone](../labs/week-6-coding-agent-eval-redteam-warden.md)
