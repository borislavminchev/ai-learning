# Phase 13 — Frameworks, Ecosystem, Team Practice & Career (Ongoing)

The half-life of specifics in this phase is measured in months, so the skill you're building is not "know LangChain" — it's **judgment**: when to reach for a framework and when a raw SDK plus a thin gateway is simpler, faster, and easier to debug at 2am. This phase turns you into the engineer who can adopt any new provider, model, or framework in an afternoon, defend the choice with a rubric and numbers, and ship a public, evaluated portfolio project that a hiring manager can actually run. You'll build two durable artifacts you keep forever: a **reusable "new-model smoke test" repo** (one command, any OpenAI-compatible endpoint) and a **public end-to-end project** with a cost/latency writeup.

Prev: [12-multimodal.md](../12-multimodal/12-multimodal.md) · Next: [14-capstone.md](../14-capstone/14-capstone.md)

## Prerequisites
- Phases 1–12: you can build and eval a prompt, a structured-output extractor (Pydantic/instructor), a RAG pipeline, a tool-using agent, and you have at least one running system you can turn into a portfolio piece.
- Python 3.11+ with `uv` (Phase 0 default), comfort with `pytest`, and at least one working provider key (OpenAI/Anthropic/Gemini) **or** a local Ollama install for cost-free runs.
- Basic Docker, a GitHub account with Actions enabled, and a place to deploy a small app (Vercel/Render/Fly free tiers, or Modal/Hugging Face Spaces).
- Node 20+ if you do the Vercel AI SDK production-frontend track in Week 1 (optional but recommended).

## Time budget
2 weeks × ~10–15 hrs/week (~24–28 hrs total). Roughly 35% theory / 65% hands-on. Each week states its own hours breakdown. This phase is explicitly **ongoing** — the milestone repos are meant to be maintained for years, and the "stay current" habits run in parallel with every other phase.

## How to use this file
Do the two weeks in order: Week 1 builds framework judgment and the model-agnostic smoke-test harness; Week 2 builds benchmark/model-card literacy, the currency habit, and packages your headline portfolio project for interviews. Timebox the theory — the labs are the deliverables you'll actually show people. If you fall behind, cut the optional Vercel/Next.js track first and keep the LiteLLM smoke-test repo moving, because that's the artifact you'll reuse every time a new model drops.

> **📚 Course materials.** This file is the **spine** — a weekly plan and a *recap*. Deep material lives alongside it:
> - **Lectures**: `lectures/` ([index](lectures/00-index.md)) — read for the *why*/mechanism.
> - **Lab guides**: `labs/` — follow to *build*.
> Each week: read the linked lectures first (the Theory here is their recap), then work the linked lab guide.

---

## Week 1 — Framework judgment & the model-agnostic smoke-test harness

**Hours breakdown:** ~4 hrs theory / ~9 hrs lab (13 total). Add 2 hrs if you do the optional Next.js + Vercel AI SDK track.

### Objectives
By the end of the week you can:
1. Apply a **raw-SDK-vs-framework decision rubric** to a concrete task and justify the call with the "framework tax" (indirection, churn, opacity) vs the repeated work it removes — always keeping an escape hatch.
2. Call **three provider SDKs directly** (OpenAI Responses/Chat, Anthropic content-blocks + prompt caching, Gemini or Bedrock Converse) and name the concrete API-shape differences that a gateway papers over.
3. Stand up a **LiteLLM gateway** and hit any OpenAI-compatible endpoint (including a local Ollama model) through one interface.
4. Ship a reusable **`new-model-smoke-test` repo**: one command runs structured extraction, a tool-calling loop, long-context recall, and a reasoning task against N models and prints a comparison table with **cost-per-correct-answer**.
5. Enforce **notebook-vs-production discipline**: extract a typed module, record LLM calls, and run them in CI without spending a cent.

### Theory (~4 hrs)
> 📖 Lectures for this week: [The Raw-SDK-vs-Framework Decision](lectures/01-framework-vs-raw-sdk-decision.md) · [Provider SDKs In Depth](lectures/02-provider-sdks-in-depth.md) · [Gateways: One Interface, and What Leaks Through It](lectures/03-gateways-litellm-and-the-escape-hatch.md) · [The Framework Landscape](lectures/04-framework-landscape-survey.md) · [Notebook-to-Production Discipline](lectures/05-notebook-to-production-discipline.md). Read them first — the bullets below are the recap.
- **The raw-SDK-vs-framework decision.** Read Anthropic's "Building Effective Agents" (search: `Anthropic building effective agents` on `anthropic.com`) and Hamel Husain's writing on frameworks vs raw code (`hamel.dev`). Internalize the default: **start with raw provider SDKs + a thin gateway; adopt a framework only when it removes work you're repeating, and only if you can eject.** The "framework tax" is three costs: indirection (you can't see the prompt actually sent), churn (breaking changes every few weeks), and opacity (debugging a black box). The rubric to memorize: *velocity, breaking-change cadence, abstraction cost, ejectability, dependency weight, license, maintainer risk.*
- **Provider SDKs, deeply.** Skim the official docs for the shapes, not the prose: OpenAI **Responses API** and Chat Completions (`platform.openai.com/docs`), Anthropic **Messages API** with content blocks + `cache_control` breakpoints + the Agent SDK (`docs.anthropic.com`; if you're on Claude, the `claude-api` skill has current model ids and pricing), Google **Gemini / Vertex** (`ai.google.dev`), AWS **Bedrock Converse API** + IAM auth (`docs.aws.amazon.com/bedrock`). The engineering point: these disagree on message shape (OpenAI `tool_calls` + `role:"tool"` with args-as-JSON-string vs Anthropic `tool_use`/`tool_result` content blocks with parsed args vs Gemini `functionDeclarations`), on caching, and on auth (Bedrock needs IAM/SigV4, not a bearer key). A gateway hides this — but you must know what it hides.
- **Gateways.** LiteLLM (`docs.litellm.ai`), OpenRouter, Portkey. The value: one OpenAI-compatible interface, provider fallback, cost tracking, key management. The trap: gateway abstractions leak (caching, reasoning-effort, provider-specific params), so keep the escape hatch to call a provider SDK directly.
- **The framework landscape (survey, don't adopt).** Know what each is *for*: **LangChain/LCEL** (composable chains — and its well-known failure modes: opaque prompts, heavy deps), **LangGraph** (stateful graphs, the production default for complex control flow), **LlamaIndex** (RAG-first), **Haystack** (pipelines), **DSPy** (compile/optimize prompts + few-shot exemplars instead of hand-tuning — read the DSPy docs' intro), **Pydantic AI / Instructor** (typed structured output), **Semantic Kernel** (.NET-first). You are surveying so you can *choose*, not learning all of them.
- **Notebooks vs production.** The discipline: extract typed modules, move config out of cells, and make LLM calls testable by **recording** them (VCR.py-style cassettes or a saved-response fixture) so CI runs deterministically and for free.

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [Build the Model-Agnostic New-Model Smoke-Test Harness](labs/week-1-model-agnostic-smoke-test-harness.md). The steps below are the summary.
Build the smoke-test repo — the artifact you'll reuse every time a new model ships.

**1. Scaffold.**
```bash
uv init new-model-smoke-test && cd new-model-smoke-test
uv add litellm pydantic python-dotenv rich tiktoken
uv add --dev pytest vcrpy
mkdir -p smoke/{tasks,fixtures,reports} src
```
Layout:
```
new-model-smoke-test/
  src/
    runner.py          # loads models, runs all tasks, prints table
    client.py          # thin LiteLLM wrapper + cost accounting
    scoring.py         # per-task correctness scorers
  smoke/
    tasks/
      extraction.py    # structured JSON extraction task + gold answers
      toolloop.py      # a 2-step tool-calling loop (calculator + lookup)
      longctx.py       # needle-in-haystack recall at ~20k tokens
      reasoning.py     # a handful of multi-step word problems
    fixtures/          # recorded responses for CI (VCR cassettes)
    reports/latest.md  # generated comparison table
  models.yaml          # the list of models to test
  .env                 # OPENAI_API_KEY etc. (never commit)
```

**2. Model-agnostic client via LiteLLM.** One function, any provider:
```python
# src/client.py
import time, litellm
from dataclasses import dataclass

@dataclass
class Call:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float

def complete(model: str, messages: list[dict], **kw) -> Call:
    t0 = time.perf_counter()
    r = litellm.completion(model=model, messages=messages, **kw)
    dt = time.perf_counter() - t0
    u = r.usage
    return Call(
        text=r.choices[0].message.content or "",
        prompt_tokens=u.prompt_tokens,
        completion_tokens=u.completion_tokens,
        cost_usd=litellm.completion_cost(completion_response=r) or 0.0,
        latency_s=dt,
    )
```
`models.yaml` lists what to test — mix hosted and local:
```yaml
models:
  - openai/gpt-4o-mini
  - anthropic/claude-3-5-haiku-latest
  - gemini/gemini-2.0-flash
  - ollama/llama3.1        # free, local via Ollama
```
**Free/cheap path:** run everything against `ollama/llama3.1` (install Ollama, `ollama pull llama3.1`) so you owe nothing while building. Hosted models are pennies for this suite — but the local target proves the "any OpenAI-compatible endpoint" claim.

**3. Four tasks, each with a deterministic scorer.**
- **Extraction:** give 15 messy invoice/email strings, ask for `{name, date, amount}` as JSON, score schema-valid rate + field exact-match against gold.
- **Tool loop:** register a `calculator` and a fake `get_price` tool; ask a question needing both; score whether the final answer is numerically correct and the tool-call ids linked correctly.
- **Long-context recall:** bury a unique fact ("the access code is PLUM-7741") in ~20k tokens of filler; ask for it; score exact recall.
- **Reasoning:** 8 multi-step problems with known numeric answers; score exact match.
Each task returns `(n_correct, n_total)` and aggregates the `Call` cost/latency.

**4. The report.** `runner.py` loops models × tasks and writes `smoke/reports/latest.md`:
```
| model                        | extract | tools | longctx | reason | $/correct | p50 lat |
|------------------------------|---------|-------|---------|--------|-----------|---------|
| openai/gpt-4o-mini           | 15/15   |  1/1  |  1/1    |  7/8   | $0.0009   | 0.8s    |
| ollama/llama3.1              | 12/15   |  1/1  |  0/1    |  4/8   | $0.0000   | 3.1s    |
```
`$/correct = total_cost / total_correct` — the single number that matters when comparing models for a real task.

**5. Notebook-to-production discipline.** Take one task you'd normally prototype in a notebook and (a) move it into a typed module with a Pydantic response model, (b) record its LLM responses with VCR.py so `pytest` replays them offline:
```python
# tests/test_extraction.py
import vcr
my_vcr = vcr.VCR(cassette_library_dir="smoke/fixtures", record_mode="once")

@my_vcr.use_cassette("extraction.yaml")
def test_extraction_schema_valid():
    from smoke.tasks.extraction import run
    results = run("openai/gpt-4o-mini")
    assert results.schema_valid_rate == 1.0
```
First run records real calls; every run after is free and deterministic. Wire it into a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs `pytest` on push — no API key needed because the cassettes are committed.

**Optional production-frontend track (~2 hrs):** scaffold a Next.js app with the **Vercel AI SDK** (`npx create-next-app`, `npm i ai @ai-sdk/openai`) and stream one of your smoke-test tasks to a browser with `streamText`. This is your "prototyping UI (Streamlit/Gradio) vs production frontend" contrast — build the same feature in Streamlit (`uv add streamlit`) in 20 lines and note what each medium is good for.

### Definition of Done
- [ ] `uv run python -m src.runner` runs all four tasks against ≥3 models (including one local Ollama model) and writes `smoke/reports/latest.md`.
- [ ] The report includes a **`$/correct`** column and a p50 latency column, computed from real usage/cost data.
- [ ] Swapping a model is a **one-line edit** in `models.yaml` — no code change.
- [ ] At least one task is called through a **local OpenAI-compatible endpoint** (Ollama) proving provider-agnosticism.
- [ ] `pytest` runs green **offline** using committed VCR cassettes (no API key in CI).
- [ ] A GitHub Actions run is green on a push.
- [ ] A one-paragraph `README` section states, for one real task, whether you'd use a raw SDK or a framework and why (rubric-based).

### Pitfalls
- **Comparing models on cost/token instead of cost-per-correct-answer.** A model that's 3× cheaper per token but wrong twice as often is more expensive per useful answer. Always divide by correctness.
- **Letting the gateway hide too much.** LiteLLM won't expose every provider feature (Anthropic `cache_control`, OpenAI `reasoning_effort`). Keep a direct-SDK escape hatch and test caching/reasoning natively when it matters.
- **Non-deterministic CI.** If your tests call live APIs they'll be flaky and cost money. Record responses (VCR) and replay; only re-record intentionally.
- **Adopting a framework to avoid writing 30 lines.** The framework tax (deps, churn, opacity) usually exceeds the 30 lines. Adopt when it removes *repeated* work across many features.
- **Ollama tokenizer/pricing gaps.** `completion_cost` is $0 for local models — that's correct, but don't compare a local model's "free" against a hosted one without also weighing your own hardware/latency cost.

### Self-check
1. Name the seven axes of the framework-evaluation rubric and give a task where the rubric says "raw SDK."
2. What exactly does a gateway like LiteLLM abstract away that differs across OpenAI, Anthropic, and Bedrock — and what does it *fail* to abstract?
3. Why is `$/correct` a better model-selection metric than price-per-1M-tokens?
4. How do you make a test that calls an LLM run for free and deterministically in CI?
5. Give one concrete case where you'd keep a framework and one where the "framework tax" makes you eject.

---

## Week 2 — Benchmark literacy, staying current & the shippable portfolio

**Hours breakdown:** ~5 hrs theory / ~9 hrs lab (14 total).

### Objectives
By the end of the week you can:
1. Read a **benchmark** critically — know what MMLU/GPQA/SWE-bench/BFCL/LMArena actually measure, and name their limits (contamination, format sensitivity, "context window ≠ usable quality window").
2. Build a small **domain eval** of your own and explain why it beats a leaderboard score for your use case.
3. Read a **model card + pricing page fast**: extract context/output caps, modalities, cutoff, and per-token + cached + batch + reasoning-token pricing into a one-line summary.
4. Set up a lightweight **"stay current" pipeline** (changelogs/RSS, not hype) and hook your Week-1 smoke-test repo into it.
5. Package a **public end-to-end portfolio project** with a deployed app, an eval suite, and a **cost/latency writeup with unit economics** — and prep the AI-engineering interview stories around it.

### Theory (~5 hrs)
> 📖 Lectures for this week: [Benchmark Literacy and Its Limits](lectures/06-benchmark-literacy-and-limits.md) · [Reading Model Cards and Pricing Pages Fast](lectures/07-reading-model-cards-and-pricing-fast.md) · [Staying Current Without Drowning](lectures/08-staying-current-without-drowning.md) · [The Shippable Portfolio and AI-Engineering Interview Prep](lectures/09-portfolio-and-interview-prep.md). Read them first — the bullets below are the recap.
- **Benchmark literacy & limits.** Learn what each measures: **MMLU** (broad knowledge MCQ — largely saturated/contaminated now), **GPQA** (hard graduate science, "Google-proof"), **SWE-bench / SWE-bench Verified** (resolve real GitHub issues — the closest to agentic coding reality), **BFCL** (Berkeley Function-Calling Leaderboard — tool-calling accuracy, directly relevant to you), **LMArena** (human pairwise preference — good for vibe/helpfulness, gameable and style-biased). Read the **Artificial Analysis** site (`artificialanalysis.ai`) for cross-model cost/latency/quality at a glance and the SWE-bench and BFCL project pages. The engineering takeaways: **contamination** (test data leaks into training — high MMLU means little now), **format sensitivity** (a 5-point swing from prompt formatting alone), and **usable-context ≠ advertised-context** (a 1M-token window can still "lose the middle" — see RULER / needle-in-haystack framing). The conclusion the whole roadmap keeps hammering: **trust your own domain eval over any leaderboard.**
- **Reading model cards & pricing fast.** Practice the extraction: from any provider's model page, pull context window, max output, modalities, knowledge cutoff, and the full pricing shape (input, output, **cached-input**, **batch discount**, **reasoning/thinking-token** billing, deprecation date). Do OpenAI, Anthropic, and Gemini pages and reduce each to one line. (On Claude, the `claude-api` skill gives current model ids/pricing directly — use it rather than guessing.)
- **Staying current without drowning.** The habit: follow **primary changelogs**, not Twitter hype. OpenAI/Anthropic/Google changelog pages, the `simonw/llm` project and Simon Willison's blog (`simonwillison.net`), the Latent Space and Interconnects newsletters. The mechanism: when a model drops, you don't read hot takes — you run your Week-1 smoke test against it and read your own table.
- **Portfolio & interview prep.** What separates a portfolio that gets you hired: **shipped + evaluated**, not toy demos. Each project needs a deployed URL, an eval suite with numbers, a cost/latency writeup, and a README that explains every tradeoff and "what would make us regret this." For interviews, the AI-eng loop is **LLM-app system design** (not ML-research): design a RAG/agent system with back-of-envelope QPS/token/cost math, a caching/fallback/eval story, and a debugging-non-determinism answer. Read the reference-architecture framing in Chip Huyen's *AI Engineering* (O'Reilly) and your own Phase 9 notes.

### Lab (~9 hrs)
> 🛠️ Full step-by-step guide: [Domain Eval, Currency Loop, and the Shippable Portfolio Project](labs/week-2-domain-eval-and-shippable-portfolio.md). The steps below are the summary.
Two deliverables: a domain eval bolted onto your smoke-test repo, and your packaged headline portfolio project.

**1. Build a domain eval (~2.5 hrs).** Add a `smoke/tasks/domain.py` that encodes 20–30 cases from a domain you care about (your real product, a hobby corpus, whatever). Each case has an input and a *checkable* expected outcome (exact field, numeric answer, or a low-cardinality rubric judged by a cheap model). Run it across the same models and add a `domain` column to your report. Write two sentences comparing your ranking to a public leaderboard's ranking — they will usually disagree, and that disagreement is the whole point.

**2. Model-card cheat-sheet generator (~1.5 hrs).** Write `src/modelcard.py` that, given a model id, prints a one-line summary pulled from LiteLLM's model metadata (`litellm.get_model_info(model)` exposes context window and per-token pricing):
```python
# src/modelcard.py
import litellm
def summarize(model: str) -> str:
    m = litellm.get_model_info(model)
    return (f"{model}: ctx={m.get('max_input_tokens')} "
            f"out={m.get('max_output_tokens')} "
            f"in=${m.get('input_cost_per_token',0)*1e6:.2f}/M "
            f"out=${m.get('output_cost_per_token',0)*1e6:.2f}/M")
```
Run it over your `models.yaml` list and commit the output as `smoke/reports/modelcards.md`. This is your "read a pricing page in 5 seconds" tool.

**3. Wire up the currency loop (~1 hr).** Add a `Makefile` / justfile target `make smoke MODEL=<id>` that runs the full suite against one new model and appends to `latest.md`. Write a 5-line `RUNBOOK.md`: "When a new model ships → add it to `models.yaml` → `make smoke` → read `$/correct` and `domain` columns → decide." Optionally add a GitHub Actions `workflow_dispatch` so you can trigger a smoke run from the browser.

**4. Package the headline portfolio project (~4 hrs).** Take the strongest system from Phases 4–12 (RAG bot, agent, or multimodal pipeline) and make it *portfolio-grade*:
- **Deploy it.** Free/cheap options: Streamlit Community Cloud or Hugging Face Spaces (fastest), Render/Fly free tier, or Modal (free credits) for a GPU-needing piece. A public URL is non-negotiable.
- **Attach an eval suite.** Reuse your Phase 7 golden set + judge; the README must show real numbers (accuracy/recall@k/faithfulness + p50/p95 latency).
- **Write the cost/latency + unit-economics section.** Cost per request, cost per 1k requests, p50/p95 latency, cache hit rate, and a "$/useful-answer" figure. Include the one optimization you made and its measured effect (e.g., "semantic cache cut cost 42%").
- **Write the tradeoffs doc.** A `TRADEOFFS.md`: why this model/framework/architecture, what you'd do at 100× scale, and "what would make us regret this."
- **Record two interview stories** in a private `INTERVIEW.md`: one system-design walkthrough of this project (with the QPS/token/cost math), and one "how I debugged a non-deterministic failure" war story from any phase.

### Definition of Done
- [ ] The domain eval runs across ≥3 models and its ranking is compared, in writing, to a public leaderboard's ranking.
- [ ] `modelcards.md` has a correct one-line summary (context, output cap, in/out $/M) for every model in `models.yaml`.
- [ ] `RUNBOOK.md` documents the "new model dropped" workflow and `make smoke MODEL=<id>` works end to end.
- [ ] The portfolio project has a **live public URL** anyone can open.
- [ ] Its README shows real eval numbers (quality metric + p50/p95 latency) and a cost-per-request / cost-per-1k figure.
- [ ] `TRADEOFFS.md` answers "why this stack," "what at 100× scale," and "what would make us regret this."
- [ ] You can deliver a 5-minute spoken system-design walkthrough of the project with back-of-envelope cost/token math.

### Pitfalls
- **Trusting leaderboards for your use case.** A model can top MMLU/LMArena and lose on your 25-case domain eval. Benchmarks rank *general* ability; you ship a *specific* task.
- **Ignoring contamination.** A shockingly high score on an old benchmark (MMLU) often means the test leaked into training. Prefer fresh/held-out benchmarks (SWE-bench Verified, GPQA) and your own private eval.
- **Confusing advertised context with usable context.** "1M tokens" doesn't mean quality holds across all of it — test recall at your real context length before you rely on it.
- **A portfolio of undeployed notebooks.** Recruiters and hiring managers can't run a notebook and won't. A deployed URL + eval numbers + cost writeup beats ten Jupyter files.
- **Hype-driven currency.** Reacting to Twitter threads wastes time and misleads. Follow primary changelogs and let your smoke-test table, not a viral demo, decide whether a model is worth switching to.

### Self-check
1. What does SWE-bench Verified measure that MMLU doesn't, and why does that matter for an AI *engineer* specifically?
2. Give two mechanisms by which a benchmark score overstates real-world quality.
3. Reduce one real model's pricing/card to a single line covering context, output cap, and input/output/cached pricing.
4. Describe your exact steps, in order, from "a new model was announced this morning" to "decided whether to adopt it."
5. What three artifacts turn a demo into a portfolio project that gets you hired?

---

## Phase milestone project

**The reusable model-smoke-test harness + one public, evaluated end-to-end project.** These are the two things you keep and maintain for years; together they prove you can adopt any model and ship something real.

**Part A — `new-model-smoke-test` (from Week 1, hardened in Week 2).**
- One command runs structured extraction, a tool-calling loop, long-context recall, reasoning, and your domain eval against N models listed in `models.yaml`.
- Works against **any OpenAI-compatible endpoint** via LiteLLM — hosted providers and a local Ollama model.
- Emits a comparison table with a **`$/correct`** column, p50 latency, and a domain-eval column.
- `pytest` runs green offline via recorded cassettes; GitHub Actions is green on push.
- A `RUNBOOK.md` documents the "new model dropped → `make smoke` → decide" loop.

**Part B — public end-to-end portfolio project.**
- A deployed, publicly reachable app (your best Phase 4–12 system).
- An eval suite with real numbers in the README (quality metric + p50/p95 latency).
- A cost/latency writeup with **unit economics** ($/request, $/1k, cache hit rate, $/useful-answer) and one measured optimization.
- A `TRADEOFFS.md` covering model/framework/architecture choices, behavior at 100× scale, and "what would make us regret this."

**Acceptance criteria:**
- [ ] Adding a brand-new model to the smoke test is a one-line change and produces a full comparison table in one command.
- [ ] The smoke-test CI is green with no API key (cassettes) and can be triggered on demand for a live model.
- [ ] The portfolio project is reachable at a public URL and its README shows real eval + cost numbers.
- [ ] You can give a 5-minute system-design walkthrough of the portfolio project with QPS/token/cost math and a failure/degradation story.

**Suggested repo layout:**
```
new-model-smoke-test/
  src/{runner,client,scoring,modelcard}.py
  smoke/tasks/{extraction,toolloop,longctx,reasoning,domain}.py
  smoke/fixtures/*.yaml        # VCR cassettes for CI
  smoke/reports/{latest,modelcards}.md
  models.yaml
  Makefile                     # `make smoke MODEL=<id>`
  RUNBOOK.md
  .github/workflows/ci.yml

portfolio-<project>/           # your headline project
  app/                         # deployed app (Streamlit/Next.js/FastAPI)
  evals/                       # golden set + judge from Phase 7
  README.md                    # eval numbers + cost/latency + unit economics
  TRADEOFFS.md
  INTERVIEW.md                 # private: system-design + debugging stories
```

## You are ready to move on when...
- [ ] You can state the raw-SDK-vs-framework decision as a rubric and apply it to a new task in minutes, keeping an escape hatch.
- [ ] You can call OpenAI, Anthropic, and one of Gemini/Bedrock directly and name the API-shape and auth differences a gateway hides.
- [ ] Your smoke-test repo runs against any OpenAI-compatible endpoint and ranks models by `$/correct` and your own domain eval.
- [ ] You read benchmarks and model cards skeptically and trust your own eval over any leaderboard.
- [ ] You have a "new model dropped" runbook and a low-noise, primary-source currency habit.
- [ ] You have at least one public, deployed, evaluated project with a cost/latency writeup and a tradeoffs doc — and interview stories built around it.
