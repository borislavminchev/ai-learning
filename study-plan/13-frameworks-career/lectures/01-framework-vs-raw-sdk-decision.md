# Lecture 1: The Raw-SDK-vs-Framework Decision

> Every AI engineer eventually hits the same fork: a shiny framework promises to turn 200 lines of glue into 20, and a quiet voice asks "but what happens when it breaks at 2am?" This lecture is not about which framework to learn — it's about building the *judgment* to decide, per task, whether a framework earns its place or whether raw provider SDKs plus a thin gateway will ship faster and debug easier. After it, you will be able to state a defensible default posture, name the three concrete costs of the "framework tax" and unmask each one, fill out a seven-axis scorecard for any real task, and defend "raw SDK" or "adopt" with specifics instead of vibes — always keeping an escape hatch.

**Prerequisites:** You've built at least one LLM feature end-to-end (a prompt, a structured extractor, or a tool loop) and have felt the pain of a dependency breaking. · **Reading time:** ~26 min · **Part of:** Frameworks, Ecosystem, Team Practice & Career — Week 1

---

## The core idea (plain language)

The specifics of any given LLM framework have a half-life measured in months. LangChain's public API has been rewritten more than once; the "production default" for agent control flow has migrated from chains to graphs; new entrants (Pydantic AI, the vendor Agents SDKs) arrive quarterly. If your value as an engineer is "I know LangChain," you depreciate with it. If your value is "I can look at a task and correctly decide whether *any* framework earns its keep, then adopt it in an afternoon," you appreciate.

So the skill this phase builds is a decision procedure, not a tool. And the decision has a **default posture** you hold until a specific task argues you out of it:

> **Start with raw provider SDKs plus a thin gateway. Adopt a framework only when it removes work you are repeating across many features — and only if you can eject.**

Read that twice, because every clause is load-bearing.

- **Raw provider SDKs** (`openai`, `anthropic`, `google-genai`) are thin, well-documented, and maintained by the people who ship the models. When a new capability lands, it lands here first.
- **A thin gateway** (LiteLLM, or your own ~40-line wrapper) gives you model-swapping and cost tracking without hiding the request shape.
- **"Removes repeated work across many features"** is the *only* justification that survives scrutiny. Adopting a framework to avoid writing 30 lines *once* is a losing trade — you pay the tax forever to save an afternoon once.
- **"If you can eject"** means: for any single call, you can drop to the raw SDK without unwinding your whole architecture. This is the escape hatch, and it is non-negotiable.

This is the same conclusion Anthropic reaches in *Building Effective Agents*: **start with the simplest thing that works, compose small patterns, and only add complexity (frameworks, agentic loops) when the simpler pattern demonstrably falls short.** It's also Hamel Husain's repeated argument: frameworks optimize for the *demo* — the 15-line "look how easy" tweet — but production is 95% the un-demoed work (evals, logging, error handling, prompt iteration), and frameworks often make that 95% *harder* by hiding the prompt.

---

## How it actually works (mechanism, from first principles)

### Why the default leans toward raw SDKs

An LLM feature is, mechanically, four steps: **(1) build a prompt/messages payload, (2) send an HTTP request, (3) parse the response, (4) maybe loop (tool calls, retries).** That's it. A raw SDK is a thin typed wrapper over exactly those four steps. A framework inserts *abstraction layers* between you and each step — a `PromptTemplate` object between you and step 1, a `Chain`/`Runnable`/`Agent` between you and steps 2–4.

Each abstraction layer buys you *convenience* and charges you *distance from the wire*. The core question of this entire lecture is: **is the convenience worth the distance for this task?** To answer it you must price the distance. That price is the framework tax.

### The framework tax: three concrete costs

**Cost 1 — Indirection: you can't see the exact prompt/messages sent on the wire.**

This is the most common and most dangerous. You write:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])
chain = prompt | model | StrOutputParser()
answer = chain.invoke({"question": user_q})
```

What actually went over the wire? You *think* it's two messages. But the framework may have injected a format instruction, wrapped your question, added few-shot exemplars from a retriever, or rendered a tool schema into the system prompt. When the model misbehaves, your first debugging question — "what did the model actually see?" — has no answer in your code. The prompt is assembled somewhere three layers down.

**Unmask it by logging the HTTP body.** Every SDK ultimately calls an HTTP client (usually `httpx`). You can enable transport-level logging and read the real payload:

```python
import logging, httpx

# Broad approach: turn on transport-level DEBUG logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)

# Decisive approach: sniff the actual JSON body with a request hook
def log_request(request: httpx.Request):
    print("WIRE >>>", request.content.decode())

client = httpx.Client(event_hooks={"request": [log_request]})
# pass `client` into the SDK / gateway that accepts a custom http client
```

The first time you do this on a framework chain, you often find the prompt is 3× longer than you thought (injected instructions, serialized schemas) — which explains both the quality drift *and* the token bill. **The rule this teaches:** if you cannot answer "what exact bytes went to the model?" in under a minute, your indirection is too high.

**Cost 2 — Churn: breaking changes every few weeks.**

Fast-moving frameworks reorganize public APIs constantly to chase new model capabilities. A concrete, representative deprecation-driven refactor (LangChain's real trajectory over 2023–2024, generalized):

```python
# v0.0.x — the demo everyone copied
from langchain import LLMChain, PromptTemplate, OpenAI
chain = LLMChain(llm=OpenAI(), prompt=PromptTemplate(...))
chain.run(question=q)          # <-- .run() everywhere

# a few months later: LLMChain deprecated, LCEL is "the way"
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
chain = prompt | ChatOpenAI() | StrOutputParser()
chain.invoke({"question": q})  # <-- .run() gone, imports moved packages

# later still: split into langchain-core / langchain-community / provider packages
# your imports break again; agent APIs migrate toward LangGraph
```

Each arrow is a maintenance window your team pays for: re-reading migration guides, updating imports across the codebase, re-testing, re-pinning. The raw `openai` SDK over the same period added *parameters* (new endpoints, `reasoning_effort`) but rarely broke the call you already wrote — additive, not destructive. **The rule:** count how often the framework has forced a non-trivial refactor in the last 12 months. Three or more is a red flag for anything on your critical path.

**Cost 3 — Opacity: debugging a black box at 2am with no visibility into retries/parsing.**

Production incident, 2am: your agent is returning empty answers intermittently. With raw SDK code you'd add three print statements and see it — the model returned a tool call you didn't handle, or a retry swallowed a 429. With a framework, the retry logic, the JSON parsing, the error handling, and the loop control all live *inside* the abstraction. You're now reading the framework's source to learn whether it retries on `429`, how many times, whether it re-parses malformed tool args, whether a parse failure silently returns `None` or raises. The mean-time-to-resolution of an incident is dominated by **how quickly you can see the actual behavior** — and opacity multiplies it. Every layer you didn't write is a layer you must reverse-engineer under pressure.

### A diagram of the distance

```
RAW SDK + THIN GATEWAY              FRAMEWORK
------------------------            ------------------------
your code                           your code
  |  (you build messages)             |  PromptTemplate  <- indirection
  v                                    v
gateway (swap model, log $)          Chain / Agent / Runnable
  |                                    |  (retries, parsing, loop hidden here)
  v                                    v
provider HTTP  <-- you see this      provider HTTP  <-- three layers away
  |                                    |
  v                                    v
model                                model
```

The left column: two hops to the wire, both yours. The right: the wire is behind machinery you didn't write and can't see without effort. Neither is wrong — but you must know which you're standing in.

---

## Worked example: the seven-axis scorecard

The decision isn't a gut call; it's a scorecard you fill out *per task*. Score each axis, then read the pattern. The seven axes:

| # | Axis | The question it answers |
|---|------|-------------------------|
| 1 | **Velocity** | Does it get me to a *working, shippable* feature faster — not just a faster demo? |
| 2 | **Breaking-change cadence** | How often has it forced a non-trivial refactor in the last 12 months? |
| 3 | **Abstraction cost** | What does it hide that I actually need to see (the prompt, retries, tokens)? |
| 4 | **Ejectability** | Can I drop to the raw SDK for *one* call without unwinding everything? |
| 5 | **Dependency weight** | Transitive deps, install size, cold-start time. |
| 6 | **License** | Permissive (MIT/Apache) vs copyleft/commercial-restricted; does it fit my deployment? |
| 7 | **Maintainer risk** | Bus factor, release health, issue-response, funding. |

Score each **Good (✓) / Neutral (~) / Bad (✗)** for the specific task. Now apply it to three concrete tasks.

### Task A — "Extract `{name, date, amount}` as typed JSON from 10k messy emails/day"

You're doing one structured-output call, repeated at scale, needing schema validation and retries-on-invalid.

| Axis | Raw SDK (`openai` structured outputs) | Framework (Instructor / Pydantic AI) |
|------|----------------------------------------|----------------------------------------|
| Velocity | ~ (you write the retry-on-invalid loop) | ✓ (validation + reprompt built in) |
| Breaking-change | ✓ (stable) | ✓ (Instructor is small, stable) |
| Abstraction cost | ✓ (you see everything) | ✓ (thin — it *shows* you the prompt) |
| Ejectability | ✓ | ✓ (it wraps the SDK, doesn't replace it) |
| Dependency weight | ✓ | ✓ (one small dep) |
| License | ✓ | ✓ (MIT) |
| Maintainer risk | ✓ | ~ (smaller project) |

**Verdict: adopt the thin library (Instructor/Pydantic AI).** This is the *right* kind of framework: it removes work you'd repeat on every extraction (validation + reprompt-on-failure), it's *thin* (it shows you the exact prompt), and it's trivially ejectable because it's a wrapper over the raw SDK, not a replacement for it. The tax is near zero; the repeated-work savings are real and recurring. This is the pattern to imitate — adopt libraries that reduce indirection, be suspicious of ones that add it.

### Task B — "A two-step tool-calling loop: model calls a calculator, then a price lookup, then answers"

One agent, two tools, a loop you'll run in one place.

| Axis | Raw SDK | Framework (LangChain agent / LangGraph) |
|------|---------|------------------------------------------|
| Velocity | ✓ (the loop is ~40 lines you fully control) | ~ (fast to start, slow to debug) |
| Breaking-change | ✓ | ✗ (agent APIs churn) |
| Abstraction cost | ✓ | ✗ (hides tool-arg parsing + loop control) |
| Ejectability | ✓ | ~ |
| Dependency weight | ✓ | ✗ (heavy transitive tree) |
| License | ✓ | ✓ |
| Maintainer risk | ✓ | ✓ (well-funded) |

**Verdict: raw SDK.** A two-tool loop is ~40 lines: send messages + tool schemas, check for a `tool_use`/`tool_calls` block, dispatch, append the result, loop. Adopting a full agent framework to avoid those 40 lines pays the whole tax (churn, opacity on exactly the parsing/retry logic you'll debug) to save work you do *once*. The "removes repeated work across many features" test fails — you have one loop, not fifty.

### Task C — "A stateful multi-agent system: 6 nodes, branching, human-in-the-loop approvals, resumable after a crash"

Genuinely complex control flow, replicated across several product surfaces.

| Axis | Raw SDK | Framework (LangGraph) |
|------|---------|------------------------|
| Velocity | ✗ (you'd rebuild a state machine + persistence) | ✓ |
| Breaking-change | ✓ | ~ (stabilizing) |
| Abstraction cost | ✓ | ~ (hides orchestration, but *shows* node internals) |
| Ejectability | ✓ | ✓ (each node can call the raw SDK directly) |
| Dependency weight | ✓ | ~ |
| License | ✓ | ✓ (MIT) |
| Maintainer risk | ✓ | ✓ |

**Verdict: adopt (LangGraph).** Here the framework removes *genuinely repeated, genuinely hard* work — durable state, checkpointing, branching, resume-after-crash — that you'd otherwise reimplement (badly) across every surface. Crucially it stays **ejectable**: individual nodes are just functions that can call the raw SDK, so when one node needs Anthropic prompt caching the gateway won't expose, you drop to the raw SDK *in that node only*. The tax is real but the savings recur across many features and the escape hatch is intact. **Adopt.**

The pattern across A/B/C: **adopt when the work is repeated AND hard AND the abstraction stays ejectable; stay raw when the work is one-off, cheap, or lives exactly where you'll debug.**

---

## How it shows up in production

- **The token bill you didn't expect.** Indirection hides prompt bloat. A team migrated a RAG chain to raw SDK, logged the wire body, and found the framework was re-injecting the full tool schema and format instructions on every turn — the *real* prompt was ~2.5× their estimate. Cost tracks tokens; you were paying for text you never wrote. (Approximate, illustrative — but this class of surprise is routine.)
- **The 2am MTTR multiplier.** Opacity doesn't cost you on the happy path; it costs you during incidents, when every minute is customer-facing. "Add a print, see the wire" (raw) vs "read the framework's retry source to learn if it swallowed the 429" (framework) can be the difference between a 10-minute and a 2-hour outage.
- **The migration tax as a recurring line item.** Churn means every framework on your critical path adds a *scheduled* maintenance cost — the quarter you spend chasing a `0.1 → 0.2` migration is a quarter you didn't ship features. Budget it explicitly before adopting.
- **The gateway leak.** Even the thin gateway you *do* keep will fail to expose provider-specific levers (Anthropic `cache_control`, OpenAI `reasoning_effort`). Prompt caching can cut input cost dramatically on repeated context; if your gateway silently drops the cache header, you pay full price and never know. This is *precisely* why ejectability is non-negotiable — you drop to the raw SDK for the cached call and keep the gateway for the rest.

---

## Common misconceptions & failure modes

- **"Frameworks are always bloat."** No — the *right* framework (a thin, ejectable one like Instructor) reduces indirection and removes real repeated work. The tax is about *heavy, opaque, high-churn* abstractions, not all abstraction. Task A adopts a framework and is correct to.
- **"Raw SDK means no abstraction at all."** You still write a thin gateway/wrapper — that's abstraction you *own* and can see through. "Raw" means *your* layers, not *zero* layers.
- **"I'll just eject later if I need to."** You can only eject if the architecture was built ejectable from day one. Bolting an escape hatch onto a codebase that assumed the framework's control flow everywhere is a rewrite, not an edit. Design for ejection up front or you don't have it.
- **"The demo was 15 lines, so it'll be simple in production."** The demo is the 5% the framework optimizes for. Production is the 95% (evals, logging, error handling, prompt iteration) — and hidden prompts make that 95% *harder*. Judge on the 95%.
- **"Adopting it saves us writing 30 lines."** Thirty lines written *once* is cheaper than a framework taxed *forever*. The test is *repeated* work across *many* features, not line count on one.
- **"Everyone uses framework X, so it's safe."** Popularity isn't bus factor or release health. Check who maintains it, how they respond to issues, whether it's funded — a popular project with one burned-out maintainer is maintainer risk.

---

## Rules of thumb / cheat sheet

- **Default: raw provider SDK + thin gateway.** Make the framework *argue its way in*, not the reverse.
- **The one justification that survives:** it removes work you're repeating across *many* features. One-off work → write it raw.
- **The framework tax = indirection + churn + opacity.** Price all three before adopting.
- **Ejectability is non-negotiable.** If you can't drop to the raw SDK for one call, don't adopt. Design for it on day one.
- **Unmask indirection in 60 seconds:** log the `httpx` request body. If you can't see the exact bytes sent, indirection is too high.
- **Churn check:** ≥3 forced refactors in 12 months = red flag on your critical path.
- **Prefer thin, ejectable libraries** (Instructor, Pydantic AI) over thick, opaque frameworks. Adopt things that *reduce* distance to the wire.
- **Simplest thing first** (Anthropic *Building Effective Agents*): a single well-crafted prompt beats a chain; a workflow of fixed steps beats an autonomous agent; add agentic complexity only when simpler patterns demonstrably fail.
- **Score the seven axes per task**, not once for your whole career. The same framework can be "adopt" for Task C and "no" for Task B.

---

## Connect to the lab

This week's lab builds the `new-model-smoke-test` repo — and it *is* the decision made concrete. You'll write a thin LiteLLM gateway (`src/client.py`), the escape hatch made real, and prove you can hit any OpenAI-compatible endpoint through one interface. When you write the README paragraph justifying "raw SDK vs framework" for one real task, fill out the seven-axis scorecard from this lecture and cite specific costs — that paragraph is the deliverable that proves the judgment landed. The pitfall "adopting a framework to avoid writing 30 lines" in the lab is Task B from this lecture.

## Going deeper (optional)

- **Anthropic — *Building Effective Agents*** (the canonical "workflows vs agents, simple composable patterns first" essay). Root domain: `anthropic.com`. Search: `Anthropic building effective agents`.
- **Hamel Husain — writing on frameworks vs raw code / "Fuck You, Show Me The Prompt"** (the argument that frameworks hide the prompt and optimize for demos). Root domain: `hamel.dev`. Search: `Hamel Husain fuck you show me the prompt`.
- **LangChain / LangGraph docs** (to understand what the heavy framework actually offers before rejecting or adopting it). Search: `LangGraph docs stateful agents`.
- **Instructor / Pydantic AI docs** (the thin, ejectable structured-output libraries). Search: `Instructor python structured outputs` and `Pydantic AI docs`.
- **LiteLLM docs** (the gateway). Root domain: `docs.litellm.ai`.
- **Simon Willison's blog** (running commentary on LLM tooling from a build-it-simple perspective). Root domain: `simonwillison.net`.

## Check yourself

1. State the default posture in one sentence, and explain why the clause "and only if you can eject" is separate from "removes repeated work."
2. Name the three components of the framework tax, and for **indirection**, describe the exact 60-second technique to unmask it.
3. You're adding a *single* two-tool calling loop that runs in one place. The scorecard says "raw SDK." Which two axes most drive that verdict, and why?
4. Give one task where adopting a framework is the *correct* call, and explain which axis flips it from "no" to "yes."
5. Why is ejectability something you must design for on day one rather than "add later"?
6. A teammate says "the demo was 15 lines, it'll be simple." What's wrong with that reasoning, and what should you judge on instead?

### Answer key

1. *"Start with raw provider SDKs plus a thin gateway; adopt a framework only when it removes work you're repeating across many features — and only if you can eject."* Ejectability is separate because a framework can genuinely remove repeated work (justifying adoption) yet still trap you (no escape hatch) — you need *both* the value **and** the exit. A framework that removes work but can't be ejected fails the test regardless of its savings.
2. **Indirection** (can't see the wire prompt), **churn** (frequent breaking changes forcing refactors), **opacity** (can't see retries/parsing when debugging). Unmask indirection by logging the `httpx` request body (transport-level DEBUG logging or an `httpx` request event hook that prints `request.content`) to read the exact JSON sent to the provider.
3. **Breaking-change cadence** and **abstraction cost** most drive it. Agent-framework APIs churn (repeated forced refactors), and the abstraction hides exactly the tool-arg parsing and loop control you'll need to debug — while the raw loop is ~40 lines you fully control. The savings are one-off (one loop, one place), so the "repeated work across many features" test fails.
4. A **stateful multi-agent system with branching, human-in-the-loop, and resume-after-crash**, replicated across several surfaces (adopt LangGraph). The **velocity** axis flips it: reimplementing durable state/checkpointing/branching by hand is genuinely hard, repeated work — and the framework stays ejectable (nodes can call the raw SDK), so the escape hatch survives.
5. Because ejecting requires the architecture to *not assume* the framework's control flow everywhere. If every path routes through the framework's loop/state, dropping to the raw SDK for one call is a rewrite, not an edit. The escape hatch only exists if you built the seams for it from the start.
6. The 15-line demo is the ~5% the framework optimizes to showcase; production is the ~95% (evals, logging, error handling, prompt iteration), which hidden prompts make *harder*. Judge on the 95% — specifically whether you can still see the wire, debug retries, and eject — not on demo line count.
