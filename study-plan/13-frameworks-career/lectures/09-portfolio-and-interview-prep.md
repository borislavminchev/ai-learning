# Lecture 9: The Shippable Portfolio and AI-Engineering Interview Prep

> You have spent twelve phases learning to build and *evaluate* LLM systems. This lecture is about the last mile that decides whether any of it gets you hired: turning an evaluated system into an artifact a stranger can open in a browser, probe with numbers, and use as evidence that you think like an engineer and not a prompt-tinkerer. A hiring-grade portfolio is *shipped AND evaluated* — a public URL, an eval suite with real numbers, a unit-economics writeup, and a README that names its own tradeoffs including "what would make us regret this." After this you'll be able to package a headline project to that bar and walk into an AI-engineering interview able to design an LLM app live with back-of-envelope cost math, tell a caching/fallback/eval story, and answer "how do you debug non-determinism" without flinching.

**Prerequisites:** You have at least one running RAG/agent/multimodal system from Phases 4–12, a Phase 7 golden set + judge, basic Docker, and a GitHub account · **Reading time:** ~28 min · **Part of:** Frameworks, Ecosystem, Team Practice & Career — Week 2

---

## The core idea (plain language)

A recruiter or hiring manager will spend somewhere between thirty seconds and five minutes on your project before deciding whether to keep reading. In that window they **will not** `git clone` your repo, create a virtualenv, hunt for an API key, and run a notebook. If the first interaction requires any of that, the project is worth zero to them regardless of how good the code is.

So the entire game is: **compress the evidence of your engineering into things that survive a thirty-second skim.** There are exactly four:

1. **A deployed public URL** — proof it runs, and that you know how to ship, not just prototype.
2. **An eval suite with real numbers** — proof you measure quality instead of vibes.
3. **A cost/latency writeup with unit economics** — proof you think about money and SLAs, which is what separates an engineer from a hobbyist.
4. **A README that explains every tradeoff, ending with "what would make us regret this"** — proof you have judgment and can be trusted to own a system.

A "toy demo" has at most one of these (usually a Streamlit screenshot). A hireable artifact has all four. The gap between them is not more model quality or a fancier framework — it is roughly a day of packaging work that most candidates skip, which is *exactly why doing it makes you stand out.*

The second half of the game is the *stories*. The artifacts get you the interview; two rehearsed narratives get you the offer: a **5-minute system-design walkthrough** of the headline project with the cost/token math live, and a **debugging-a-non-deterministic-failure war story**. This lecture builds both.

## How it actually works (mechanism, from first principles)

### Why "deployed" is a hard gate, not a nice-to-have

The claim you are making with a portfolio is "I can ship LLM systems that work in production." A notebook proves you can make a call succeed once, on your machine, with your keys, in your Python version. A deployed URL proves you handled the things production actually demands: dependency pinning, secret management, a request/response boundary, a cold-start path, and *someone else's* network hitting it. Those are the skills the job is about. The URL is a proxy for all of them, which is why it's non-negotiable.

**Deployment options and their fit** (2025–2026 landscape):

```
Need                          Best fit                         Why
---------------------------   ------------------------------   ---------------------------------
Fastest to a public URL,      Streamlit Community Cloud        Push to GitHub, point at app.py,
Python UI, no infra                                            secrets in a UI box. Minutes.
Fastest URL + a "Space"       Hugging Face Spaces              Gradio/Streamlit/Docker; ML-crowd
credibility, ML audience                                       visibility; free CPU tier.
A real HTTP API / backend,    Render or Fly.io free tiers      You control the Dockerfile and
FastAPI, custom routes                                         routes; closer to prod shape.
A piece that NEEDS a GPU      Modal (free credits)             Serverless GPU, pay-per-second,
(local embeddings, a model)                                    scale-to-zero; wrap a fn as an API.
```

The decision rule: **if the app is a Python UI over hosted model APIs, use Streamlit Community Cloud or HF Spaces — you can be live in under an hour.** Only reach for Render/Fly when you need a real backend shape, and only reach for Modal when something genuinely needs a GPU (self-hosted embeddings, a fine-tuned model, an image pipeline). Do not spin up a Kubernetes cluster to serve a chatbot; over-engineering the deploy is itself a red flag.

One free-tier reality to design around: **cold starts and scale-to-zero.** Free tiers spin your app down after idle. The recruiter who opens your link after it's been idle for a day may wait 20–40 seconds for the container to wake. Mitigations: a lightweight health-check pinger, a "warming up…" UI state, or accepting it and *documenting* it in the README (which itself signals you understand the tradeoff).

### The unit-economics writeup, in depth

This is the section that most separates you from other candidates, because almost nobody does it and every hiring manager probes it. Unit economics is answering, in numbers, "what does one useful answer from this system cost, and how fast is it?"

The figures hiring managers ask for, and how to compute each:

- **Cost per request.** For an LLM app this is dominated by tokens:
  `cost_req = (prompt_tokens/1e6 × input_price) + (completion_tokens/1e6 × output_price)`
  Add embedding and rerank costs if your RAG path calls them per request.
- **Cost per 1,000 requests.** Just `1000 × cost_req`. Managers think in per-1k or per-million because per-request numbers look deceptively tiny ("$0.004? who cares") until you multiply by traffic.
- **p50 / p95 latency.** The *median* and the *95th percentile* end-to-end wall-clock time. p95 matters more than p50 because it's the experience your unluckiest 1-in-20 users get, and it's what blows SLAs. Measure it, don't estimate it — collect ~100 real timings and sort.
- **Cache hit rate.** Fraction of requests served from a cache (exact-match or semantic) instead of a fresh model call. Directly scales cost down: a 40% hit rate on a cache that's ~free means ~40% cost reduction on the cached slice.
- **$/useful-answer.** The honest figure: `total_cost / number_of_answers_that_were_actually_correct`. A model that's cheaper per token but wrong more often can be *more* expensive per useful answer. This ties your cost writeup back to your eval suite — you need the correctness number from the eval to compute it.

Then the thing that makes it *engineering* rather than a table: **one measured optimization and its effect.** Not "I could add caching" — "I added a semantic cache with a 0.92 cosine threshold; on my eval traffic it hit 38% of requests and cut cost per 1k from \$4.10 to \$2.38, a 42% reduction, with p95 latency dropping from 3.1s to 1.9s on cache hits." Before/after numbers, one lever, stated cleanly.

### Worked cost arithmetic (do this by hand once and it sticks)

Suppose a RAG endpoint per request:

```
Prompt:  3,000 tokens (system + 5 retrieved chunks + question)
Output:    250 tokens
Model input price:   $0.15 / 1M tokens
Model output price:  $0.60 / 1M tokens
```

Cost per request:
```
input  = 3000/1e6 × $0.15 = 0.0000045 × 1000... let's keep units clean:
       = (3000 / 1,000,000) × 0.15 = 0.003   × 0.15 = $0.00045
output = (250  / 1,000,000) × 0.60 = 0.00025 × 0.60 = $0.00015
cost_req = $0.00045 + $0.00015 = $0.00060
```

Cost per 1k requests: `1000 × $0.00060 = $0.60`.

Now layer eval: your suite says the system answers **90%** of questions usefully. Then:
```
$/useful-answer = $0.00060 / 0.90 = $0.00067
```

Now the optimization: you add a semantic cache. Measured hit rate 38%, cached responses cost ~\$0 (an embedding lookup, call it negligible). Effective cost per request:
```
cost_req' = (1 - 0.38) × $0.00060 = 0.62 × $0.00060 = $0.00037
reduction = 1 - (0.00037 / 0.00060) = 1 - 0.62 = 38%
```

That 38% number is *real and defensible* because it falls straight out of the hit rate — this is exactly the kind of arithmetic you do live in the interview. (Round numbers and label estimates as estimates; never invent a benchmark figure you didn't measure.)

### The TRADEOFFS doc structure

A `TRADEOFFS.md` has three sections, and interviewers essentially ask these three questions verbatim:

```
## Why this model / framework / architecture
- Model:      why THIS one (cost/quality/latency from YOUR eval, not a leaderboard)
- Framework:  raw SDK + thin gateway vs LangGraph/etc — the rubric call, with the escape hatch
- Architecture: RAG vs fine-tune vs long-context; sync vs streaming; why the retrieval choice

## What changes at 100x scale
- Where does it break first? (rate limits, vector DB latency, cost line item, a single-node cache)
- What's the next architecture? (batch embeddings, a real vector DB, a queue, provider fallback)
- What stays fine? (say so — knowing what NOT to change is judgment too)

## What would make us regret this
- The honest failure modes: contamination in the eval, a domain the retrieval can't cover,
  a provider dependency with no fallback, a cost that explodes if inputs get longer,
  a latency cliff under concurrency. Name them BEFORE the interviewer finds them.
```

That third section is the strongest signal in the whole portfolio. Naming your own system's weaknesses before anyone asks demonstrates the exact senior-engineer trait teams are screening for: you own the thing end to end, including its flaws. Chip Huyen's *AI Engineering* frames the reference architecture as layers (model API → context/retrieval → orchestration → eval/guardrails → deployment/observability); a good TRADEOFFS doc walks the layer where each decision lives.

### The interview loop is LLM-APP SYSTEM DESIGN, not ML research

Internalize this framing or you'll prepare for the wrong interview. AI-engineering interviews rarely ask you to derive backprop or explain attention math. They ask you to **design an LLM application as a system** — the same shape as a classic distributed-systems interview, with LLM-specific twists. The loop, live on a whiteboard:

1. **Clarify the task and constraints.** QPS? Latency budget? Accuracy bar? Cost ceiling? Is it a chatbot, a batch pipeline, an agent?
2. **Draw the reference architecture.** Ingress → retrieval/context → model call(s) → post-processing/guardrails → response, plus eval and observability off to the side.
3. **Do back-of-envelope QPS/token/cost math** out loud (see below).
4. **Tell the caching / fallback / eval story** — how you cut cost, what happens when the primary model errors or rate-limits, how you know it works.
5. **Answer "how do you handle non-determinism."**

**Back-of-envelope live math.** Say the interviewer says "100 QPS, RAG chatbot." You reason aloud:
```
100 QPS × 3,000 in + 250 out tokens/req
tokens/sec in  = 100 × 3000 = 300,000 tok/s → 300k × 60 × 60 = ~1.08B tok/hr input
cost/hr input  = 1.08e9 / 1e6 × $0.15 = 1080 × $0.15 = $162/hr
cost/hr output = (100 × 250 × 3600) / 1e6 × $0.60 = 90 × $0.60 = $54/hr
≈ $216/hr → ~$5.2k/day → the room now cares about caching a LOT.
```
The point is not perfect numbers; it's that you *reach for the arithmetic reflexively* and it changes the design (now a 40% cache hit rate is \$2k/day saved, and it's obvious you must batch embeddings and add provider fallback for 100 QPS).

**The caching/fallback/eval story** is one connected narrative: "I cache exact + semantic to cut cost and tail latency; I fall back to a cheaper/alternate provider on 429s and 5xxs with a circuit breaker so one provider outage doesn't take me down; and I gate deploys on an eval suite so a prompt change that regresses faithfulness never ships."

**"How I debug non-determinism"** — the crisp answer interviewers want: *"I make the boundary reproducible and the failure observable. First, pin what I can: `temperature=0`, seeds where supported, fixed model version — knowing even then providers don't guarantee bit-identical output. Second, log the full request and response for every call (prompt, params, model id, token counts) so I can replay the exact failing input. Third, I don't chase single anomalies — I run the failing case N times to see if it's a distribution problem or a one-off, and I lean on my eval suite to tell me whether a change moved the aggregate, because with a stochastic system you debug at the level of rates, not single runs."*

## Worked example

You pick your Phase 8 RAG bot over a documentation corpus as the headline project. Packaging it to hireable grade, end to end:

**Deploy (45 min).** It's a Streamlit UI over hosted model + embedding APIs → Streamlit Community Cloud. Push repo, point at `app/main.py`, paste keys into the secrets box. Live URL in the README's first line. Note the free-tier cold start in the README.

**Eval (reuse Phase 7, 1 hr).** Golden set of 40 Q/A pairs. Run the judge: **faithfulness 0.91, answer-relevance 0.88, recall@5 0.83.** Collect 100 real request timings: **p50 1.4s, p95 3.2s.** These go in the README as a table, not prose.

**Unit economics (1 hr).** Per request: 3,000 in / 250 out tokens → **\$0.0006/req**, **\$0.60/1k**. Useful-answer rate from eval ≈ 0.90 → **\$0.00067/useful answer.** Optimization: add semantic cache, measure **38% hit rate**, cost/1k drops \$0.60 → \$0.37 (**~38% cut**), cache-hit p95 1.9s. All measured, all in the README.

**TRADEOFFS.md (45 min).** *Why:* gpt-4o-mini class model because on *my* 40-case eval it matched the frontier model within 2 points at 1/15th the cost; raw SDK + LiteLLM gateway because the app is simple enough that a framework's tax exceeds its savings; RAG over fine-tune because the corpus updates weekly. *At 100x:* the in-process cache and single vector index break first → move to a managed vector DB + a shared Redis cache + batch the embedding calls; add a second provider for fallback. *What would make us regret this:* the corpus has no coverage for install-error questions, so recall silently drops there; and cost scales with retrieved-chunk count, so a "use 15 chunks" tweak would ~triple input cost.

**INTERVIEW.md (private, 30 min).** Story 1: the 5-minute walkthrough with the numbers above. Story 2: "I had a flaky test where extraction passed 4 of 5 runs; I logged the raw response, found the model occasionally wrapped JSON in a ```` ```json ```` fence, added a tolerant parser + a schema-valid-rate metric over 20 runs, and it went to 100%." That's a real non-determinism war story.

## How it shows up in production

- **The cold-start recruiter bounce.** Your link is dead on arrival because the free tier scaled to zero and they gave up at 15 seconds. Fix: warm it, or set expectations in the UI, or use a tier that doesn't sleep. This is a *real* production concern (perceived availability) in miniature.
- **The cost number that only exists per-request.** Teams get blindsided when "$0.0006" becomes "$5,000/day" at scale. The per-1k and per-million framing exists precisely to make the future bill visible now. Interviewers probe this because they've been burned by it.
- **p50 looks great, p95 is a disaster.** Averages hide tail latency. In production the p95/p99 is what triggers timeouts, retries (which cost more tokens), and angry users. If you only report an average, a sharp interviewer knows you haven't run a real load.
- **The eval that lied.** You reported 0.95 faithfulness but the golden set was contaminated or too easy. In production the real distribution is harder and the number collapses. Naming this risk in TRADEOFFS ("what would make us regret this") is the same discipline that stops it in prod.
- **The single-provider outage.** No fallback means your provider's bad afternoon is your outage. The fallback story isn't hypothetical; provider 429s and 5xxs are routine at any real volume.

## Common misconceptions & failure modes

- **"My GitHub with clean code is my portfolio."** Code quality is table stakes; it's not *evidence of shipping*. The URL + numbers are. Recruiters skim, they don't code-review.
- **"More projects is better."** One deployed, evaluated, cost-analyzed project beats ten notebooks. Depth signals seniority; breadth of toy demos signals the opposite.
- **"I'll estimate the cost/latency."** Estimated numbers get exposed the moment an interviewer asks "how did you measure that?" Measure it. A measured 38% with a stated method beats a hand-waved "about half."
- **"Non-determinism means the system is broken."** No — it means you debug at the level of *rates over N runs* and aggregate eval deltas, not single traces. Treating a one-off as a bug you can single-step is the rookie tell.
- **"System design = draw boxes."** Without the token/cost/QPS arithmetic, the boxes are decoration. The math is what turns a diagram into a design.
- **"TRADEOFFS is where I sell the project."** It's where you're *honest* about it. The 'what would make us regret this' section losing its teeth (empty or fake-humble) is a wasted signal.
- **Over-engineered deploy as a flex.** K8s + Terraform for a chatbot reads as poor judgment, not sophistication. Match the deploy to the need.

## Rules of thumb / cheat sheet

- **The four hireability gates:** deployed URL · eval numbers · unit-economics writeup · TRADEOFFS with "what would make us regret this." All four or it's a toy.
- **Deploy pick:** Python UI over hosted APIs → **Streamlit Cloud / HF Spaces**. Real backend → **Render/Fly**. Needs a GPU → **Modal**. Never K8s for a chatbot.
- **Cost formula:** `(in_tok/1e6 × in_price) + (out_tok/1e6 × out_price)`. Report **per-1k**, not per-request (per-request hides the scale).
- **Always report p95, not just p50/average.** The tail is the SLA.
- **`$/useful-answer` = total_cost / correct_answers** — ties cost to eval; the only fair cross-model cost metric.
- **One measured optimization** with before/after numbers beats ten proposed ones. State the lever, hit rate, and the % effect.
- **Interview design loop:** clarify → draw reference architecture → out-loud QPS/token/cost math → caching/fallback/eval story → non-determinism answer.
- **Non-determinism answer in one breath:** pin (temp/seed/version), log full request+response, replay, judge at the level of rates over N runs and eval deltas.
- **Two rehearsed stories:** 5-min system-design walkthrough *with live math* + a debugging-a-flaky-failure war story. Rehearse both out loud.
- **Numbers are approximate** unless you measured them — label them as such; never fabricate a benchmark.

## Connect to the lab

This lecture is the theory behind the Week 2 portfolio-packaging lab: you'll take your strongest Phase 4–12 system, **deploy it to a public URL**, bolt on your Phase 7 eval suite so the README shows real quality + p50/p95 numbers, write the **unit-economics section** (cost/req, cost/1k, cache hit rate, \$/useful-answer + one measured optimization), and author `TRADEOFFS.md` and a private `INTERVIEW.md`. The phase milestone's acceptance test — *"deliver a 5-minute system-design walkthrough with back-of-envelope cost/token math"* — is exactly the interview craft rehearsed here.

## Going deeper (optional)

- **Chip Huyen, *AI Engineering* (O'Reilly, 2025)** — the reference-architecture framing (model → context/retrieval → orchestration → eval → deployment/observability) this lecture leans on. Also her blog `huyenchip.com`.
- **Streamlit Community Cloud docs** (`docs.streamlit.io`) and **Hugging Face Spaces docs** (`huggingface.co/docs/hub/spaces`) — the two fastest paths to a public URL.
- **Modal docs** (`modal.com/docs`) — serverless GPU, free credits; search: `Modal web endpoint tutorial`.
- **Render** (`render.com/docs`) and **Fly.io** (`fly.io/docs`) free-tier deploy guides.
- **Simon Willison's blog** (`simonwillison.net`) for current, no-hype model/pricing commentary; search: `Simon Willison LLM pricing`.
- Talks/search queries (don't trust invented links; search these): `AI engineering system design interview`, `LLM app cost per request optimization semantic cache`, `RAG evaluation faithfulness recall@k`, `p95 latency LLM serving`.

## Check yourself

1. A recruiter opens your project link and it 500s after 20 seconds. Name the most likely free-tier cause and two fixes.
2. Your endpoint uses 3,000 input and 300 output tokens per request at \$0.15/M in and \$0.60/M out. Compute cost per request and cost per 1,000 requests.
3. Why is `$/useful-answer` a fairer cross-model comparison than cost per 1M tokens, and what other artifact must you have to compute it?
4. In a live system-design interview for a 50-QPS RAG chatbot, what four things do you do, in order, before drawing a single box — and what is the one number that most changes the design?
5. Give the crisp three-part answer to "how do you debug a non-deterministic LLM failure."
6. What are the three sections of a TRADEOFFS doc, and why is the third one the strongest hiring signal?

### Answer key

1. **Cold start / scale-to-zero:** the container was spun down after idle and either timed out waking or hit a memory/timeout limit on cold boot. Fixes: keep it warm with a periodic health-check ping; show a "warming up…" state and raise the client timeout; or move to a tier that doesn't sleep. (Documenting the limit in the README also counts as handling it.)
2. `input = 3000/1e6 × 0.15 = $0.00045`; `output = 300/1e6 × 0.60 = $0.00018`; **cost/req = \$0.00063**; **cost/1k = \$0.63**.
3. Because a model can be cheaper per token yet wrong more often, so it costs more per *correct* answer; dividing total cost by correct answers normalizes for quality. You need an **eval suite** (a correctness/usefulness number) to compute the denominator.
4. **Clarify constraints (QPS, latency budget, accuracy bar, cost ceiling) → draw the reference architecture → do the QPS/token/cost back-of-envelope out loud → tell the caching/fallback/eval story.** The number that most changes the design is the **cost/hour at target QPS** (it forces caching, batching, and fallback into the picture).
5. **(1) Pin** what you can — `temperature=0`, seeds where supported, fixed model version — while knowing bit-identical output isn't guaranteed. **(2) Log** the full request and response (prompt, params, model id, token counts) so you can replay the exact failing input. **(3) Judge at the level of rates** — run the case N times and use aggregate eval deltas, because you debug a stochastic system by distributions, not single traces.
6. **Why this model/framework/architecture · what changes at 100x scale · what would make us regret this.** The third is strongest because naming your own system's failure modes before you're asked demonstrates end-to-end ownership and senior judgment — the exact trait the interview is screening for.
