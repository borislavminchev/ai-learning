# Lecture 12: Long context vs RAG vs CAG — context strategy as an economics decision

> Modern models advertise 200k, 1M, even 2M-token context windows, and the naive reading is "great, I'll just paste everything in." That instinct hands you a slow, expensive, quietly-wrong system. Getting knowledge *into* the model is not a feature to max out — it's a cost/latency/reliability tradeoff you make per query, and there are exactly three strategies to choose from: stuff it all in (**long context**), fetch only what's needed (**RAG**), or preload a fixed corpus once and reuse it (**CAG**). The choice is driven by two properties of your knowledge — how *big* it is and how *often it changes* — not by which feature sounds most powerful. After this lecture you can look at a workload, place it on that two-axis grid, pick the cheapest reliable strategy, and defend the choice with tokens-per-request, TTFT, and dollars-per-month numbers instead of "the model supports 1M tokens, so let's use it."

**Prerequisites:** you know prompt vs completion tokens are billed separately (Lecture 09); you understand automatic prefix caching and the KV cache from Lecture 02; you've seen RAG (retrieve-then-generate) in Phase 6; rough TTFT/TPOT literacy (Lecture 08). · **Reading time:** ~28 min · **Part of:** Phase 10 (LLMOps) Week 3

---

## The core idea (plain language)

Every LLM call needs the model to *know* something it wasn't trained on: your docs, this user's tickets, today's prices. There are only three ways that knowledge can reach the model, and they differ in **who pays for the prefill, and how often.**

1. **Long context** — paste all the knowledge into the prompt on *every* request. Simple, no infrastructure, the model sees everything. But you pay to prefill every one of those tokens on every single call, and long prompts degrade quality (the model loses track of the middle).
2. **RAG (retrieval-augmented generation)** — keep the knowledge in a vector store / search index; at query time, retrieve only the handful of chunks relevant to *this* question and put just those in the prompt. Cheapest per request and scales to enormous, changing corpora — at the cost of a retrieval pipeline and the risk that retrieval misses the right chunk.
3. **CAG (cache-augmented generation)** — take a small, stable corpus, prefill it *once* into the KV/prompt cache, and reuse that cached prefix across all future requests. Near-zero *repeated* prefill cost, no retrieval pipeline, no recall risk — but only works when the corpus is small enough to fit and stable enough that you're not constantly re-caching.

The one sentence that ties them together: **long context pays the full prefill every call; RAG pays a small prefill every call plus a retrieval step; CAG pays the full prefill once and ~nothing thereafter.** The decision is a rubric, not a preference — read it off two axes, corpus **size** and corpus **volatility**:

```
                      corpus is SMALL (fits in window)   corpus is LARGE (doesn't fit)
                    +----------------------------------+------------------------------+
   changes RARELY   |  CAG  (prefill once, reuse)      |  RAG  (must retrieve anyway) |
   (static)         |                                  |                              |
                    +----------------------------------+------------------------------+
   changes OFTEN    |  long context OR RAG             |  RAG  (only viable option)   |
   (dynamic)        |  (CAG cache thrashes)            |                              |
                    +----------------------------------+------------------------------+

   Special case: need EVERYTHING at once, one-shot, can't reuse  ->  long context, and pay for it.
```

That's the whole lecture. The rest is *why*, and how to put numbers on each so you can defend the choice in a design review.

## How it actually works (mechanism, from first principles)

### Where the cost lives: prefill vs decode

Recall from Lectures 02 and 08 that a request has two phases. **Prefill** processes the entire input prompt — every token, through every layer — to build the KV cache; it is compute-heavy and its cost scales with *input length*. **Decode** then generates output tokens one at a time, reading that KV cache; it is memory-bandwidth-bound and scales with *output length*. **TTFT (time to first token) is dominated by prefill.** So the length of the knowledge you inject directly drives both your input-token bill *and* your TTFT.

This is the crux. Injecting 100k tokens of context isn't "free because the window supports it" — it is 100k tokens pushed through the full model stack, on every call, before the user sees a single output token. Two independent costs fall out of that one length:

- **The money cost** scales *linearly* with input tokens: double the context, double the input bill.
- **The latency cost** also scales with input tokens, but you feel it as a wall: prefill of a big prompt can add whole seconds to TTFT. Order of magnitude, a mid-tier GPU prefills at a few thousand tokens/sec for a 7B-class model, so 100k tokens is on the order of *seconds* of pure prefill before decode begins. (Chunked prefill, Lecture 02, stops one whale from freezing everyone else's stream — it does **not** make the whale's own prefill cheap.)

Hold those two facts. Every strategy below is just a different answer to "how many input tokens do I prefill, and how many times do I re-pay for the same ones?"

### Long context: pay for everything, every time

You put the whole corpus in the prompt. Say it's **100,000 tokens** of documentation and the user's actual question is 50 tokens.

- **Tokens per request:** ~100,050 input tokens, *every call*.
- **Cost:** at an approximate hosted price of **$0.20 per 1M input tokens** (a cheap open-model API — see Lecture 09), that's `100,050 × $0.20/1M ≈ $0.020` **per request, just for input.** A thousand queries a day is ~$20/day ≈ **$600/month in prefill alone**, and you re-read the same 100k tokens a thousand times to get it.
- **TTFT:** prefill of 100k tokens can add **seconds** before generation starts. On a tight SLO (p95 TTFT < 800ms, Lecture 08) this is an automatic breach — no amount of decode tuning saves you.
- **Reliability — the subtle killer:** models don't attend uniformly across a long context. The well-documented **"lost in the middle"** effect (Liu et al., 2023) shows accuracy is highest for facts at the *start* and *end* of the context and sags for facts buried in the middle. And **context rot** — degradation as the window fills with more, partially-relevant tokens — means a fact the model would nail in a 2k-token prompt can be missed in a 100k-token one. Benchmarks like **RULER** and simple **needle-in-a-haystack** tests measure exactly this; they consistently show real, model-specific degradation *well before* the advertised max length. Read *about* these findings (they set your expectations); don't reimplement them.

So long context is dead simple and sees everything, but you pay full prefill every call *and* reliability drops as the window grows. It earns its place only when you genuinely need the whole thing in one shot and can't reuse it.

### RAG: pay only for what's relevant

Instead of shipping the whole corpus, you index it once (chunk → embed → store in a vector DB) and at query time retrieve the top-`k` chunks most relevant to the question. Only those go in the prompt.

- **Tokens per request:** say `k = 5` chunks × ~400 tokens = 2,000 tokens of context + a 50-token question ≈ **2,050 input tokens.** That's **~50× smaller** than the long-context prompt.
- **Cost:** `2,050 × $0.20/1M ≈ $0.0004` per request — roughly **1/50th** of long context's input cost — plus a tiny query-embedding cost (typically fractions of a cent).
- **TTFT:** prefill of 2k tokens is fast — sub-second on most setups. You *add* a retrieval hop (embed the query + vector search), typically tens of milliseconds, which is far cheaper than the prefill you avoided.
- **Reliability:** better on the lost-in-the-middle axis (short, focused context = the model attends well) but it introduces a new failure mode: **recall risk.** If the retriever doesn't surface the chunk containing the answer, the model can't use it — and it will confidently answer from priors anyway. **RAG quality is capped by retrieval quality.**
- **Scales to huge, changing corpora:** the index can be terabytes; you only ever prompt with a few KB. Update a document → re-embed just that document → the next query sees it. This incremental-update property is why RAG is the default for large or frequently-changing knowledge bases.

### CAG: pay for the prefill once

CAG leans directly on the **automatic prefix caching** you learned in Lecture 02. Recall: if two requests share the same opening tokens, vLLM computes that prefix's KV *once* and reuses the cached KV blocks for every later request with the same prefix. CAG is that mechanism used deliberately as an architecture: put your entire (small, stable) corpus at the *front* of the prompt, byte-identical on every request, and let the cache do the rest.

```
Request 1:  [ 20k-token corpus ][ question A ]   <- full prefill of 20k (COLD)
Request 2:  [ 20k-token corpus ][ question B ]   <- 20k prefix is a CACHE HIT
Request 3:  [ 20k-token corpus ][ question C ]   <- cache hit again
            \____ prefilled ONCE ____/  \_ only this is new prefill work _/
```

- **Tokens per request (billed):** depends on where you run. With **self-hosted vLLM** prefix caching, a cache hit skips the *compute* of re-prefilling the corpus (a big TTFT + throughput win) though the tokens still notionally "exist." With **hosted prompt caching** (Anthropic, OpenAI, Google Gemini), cached input tokens are billed at a **steep discount** — e.g., Anthropic charges cache *reads* at roughly 10% of the normal input price, and a *write* (the cold fill) at a small premium (~1.25×). (Approximate — check current provider docs; the /claude-api skill has the live numbers.) Either way, the expensive full prefill happens **once**, not per request.
- **TTFT:** after the cache is warm, TTFT collapses to roughly the cost of prefilling only the new question (tens of tokens) + decode. This is CAG's headline win: **long-context-level knowledge at RAG-level (or better) TTFT**, with no retrieval hop and no chunking.
- **Reliability:** the model sees the *entire* corpus every time (no recall risk like RAG) but the corpus must be small enough that lost-in-the-middle doesn't bite. Small + fully-in-context is the sweet spot.
- **The catch — invalidation:** the cache is keyed on the *exact* prefix. Change one token of the corpus and the whole cached prefix is invalid; the next request pays a full cold prefill again to rebuild it. There is also a **TTL**: hosted caches expire (Anthropic's default cache lifetime is ~5 minutes unless refreshed; self-hosted blocks are evicted under memory pressure), so a low-traffic CAG endpoint can go cold *between* requests and re-pay the cold fill even without a corpus change. This is why CAG demands both a **stable** corpus *and* enough traffic to keep the cache warm. And beware the classic cache-buster from Lecture 02: a timestamp or per-request token placed *before* the corpus destroys the shared prefix entirely.

### The cold/warm arithmetic that decides CAG vs long context

CAG's whole value is the ratio of cache **hits** to cache **misses**. Model it directly. Let a full cold prefill of the corpus cost `C_cold` and a warm request cost `C_warm` (≈ the discounted read or skipped compute). If your corpus changes (or the cache expires) every `N` requests:

```
avg cost/request (CAG) = (C_cold + (N-1)·C_warm) / N
```

As `N → ∞` (perfectly static, always warm), this trends to `C_warm` — CAG wins big. As `N → 1` (corpus changes every request, or traffic is too sparse to stay warm), it trends to `C_cold` — **CAG has collapsed into long context** and you carry the cache machinery for nothing. That single formula is why "how often does it change relative to how often you query it?" is the question that makes or breaks CAG.

### The three, side by side

| | Long context | RAG | CAG |
|---|---|---|---|
| Prompt content | whole corpus, every call | top-`k` relevant chunks | whole (small) corpus, every call |
| Prefill cost | full, **every** request | small, every request | full **once**, then cached |
| TTFT | worst (huge prefill) | good (small prefill + retrieval hop) | best after warm-up |
| Infra needed | none | retrieval pipeline + vector store + eval | prompt/prefix cache (built-in) |
| Corpus size fit | up to window limit | effectively unbounded | must fit in window |
| Update handling | trivial (just change the text) | re-embed changed docs, incremental | full cache rebuild on any change |
| Main failure mode | lost-in-the-middle / context rot | retrieval recall miss | cache invalidation / TTL churn |

## Worked example

You're building an internal assistant. Two knowledge sources, two different right answers.

**Source A — the company's employee handbook.** ~30,000 tokens. Updated maybe twice a year. Any employee query could touch any part of it (policies interlock). Traffic: 5,000 queries/day, steady during business hours.

**Source B — the full product documentation + all support tickets.** ~40 million tokens, dozens of docs edited daily, thousands of tickets added weekly. Any given question touches a tiny slice.

**Handbook → CAG.** It's *small* (30k fits comfortably in-window) and *static* (twice a year), and 5,000 queries/day is more than enough to keep a cache warm. Put the whole handbook as a fixed prefix; the question appends at the end.

- Cold prefill (first request of a cache lifetime, or after an update / TTL expiry): ~30k tokens, paid once.
- Every subsequent warm request: cache hit. On a hosted API with prompt caching at ~10% read cost, input cost ≈ `30,000 × $0.20/1M × 0.10 + question ≈ $0.0006` per request, vs. **$0.006** if you re-prefilled at full price every time — a **~10× per-request saving**, across 5,000 queries/day (≈ $27/day saved on this route alone). Twice-a-year edits mean two cold rebuilds a year. Using the `N` formula: `N` is enormous, so avg cost ≈ `C_warm`. Ideal CAG.

**Product docs + tickets → RAG.** 40M tokens **cannot** fit in any context window, and it changes daily — CAG would be cold constantly (`N ≈ 1`), long context is physically impossible. Index everything; retrieve top-5 chunks (~2k tokens) per query.

- Per request: ~2k input tokens ≈ `$0.0004` + a query embedding. Fast TTFT.
- A doc edit re-embeds just that doc; the next query sees it. No cache to invalidate, no window limit. This is the *only* viable strategy here.

**When would long context win?** Suppose a lawyer uploads a **single 80k-token contract** and asks ten cross-cutting questions that genuinely require reasoning over the *whole* document at once (RAG's chunking would fragment the cross-references, and there's no stable reusable corpus *across different* contracts). Here you pay the full 80k prefill — but notice: if it's ten questions about the *same* contract in one session, you actually have a **CAG situation** (cache the contract prefix for the session's lifetime). Pure long-context-and-pay is the answer only when you truly need everything in one shot, once, and can't reuse it across requests.

Notice the pattern: **the rubric fell straight out of corpus size and update rate** — not novelty, not which feature sounded most powerful.

## How it shows up in production

- **The 1M-token-window bill surprise.** A team ships "just paste the docs in" with a 128k-token context. It works in the demo. Then the input-token line on the invoice is 50× the output-token line and someone finally reads Lecture 09's "price both input and output." The fix is almost always RAG or CAG — stop re-prefilling the same tokens on every call.
- **TTFT SLO blown by prefill.** Your p95 TTFT target is 800ms (Lecture 08). A 60k-token context can't prefill that fast on your GPU, so p95 TTFT is 3s and users think the app hung. Shrinking context (RAG) or warming the cache (CAG) is the lever — **not** a bigger GPU, which barely moves single-request prefill latency.
- **"The model ignored the doc I gave it."** Classic lost-in-the-middle. The relevant paragraph was at token 55,000 of an 80k prompt. Move critical content to the start or end, or switch to RAG so the model only sees the relevant slice. Debugging tip: if quality is fine at 5k tokens and bad at 80k with the *same* facts, it's context rot, not a knowledge gap.
- **CAG cache thrash from a moving corpus.** Someone points CAG at a corpus a nightly job rewrites. Every morning the first request eats a cold prefill and TTFT spikes; worse, if updates land mid-day you rebuild repeatedly and the "near-zero prefill" promise evaporates. Either the corpus wasn't actually static, a cache-buster (timestamp, request id) crept into the prefix, or traffic is too sparse to beat the cache TTL — check all three.
- **The low-traffic CAG cold-start trap.** An internal tool gets 20 queries/day, spread out. With a 5-minute cache TTL, nearly every request arrives cold and pays the full prefill — CAG gave you *zero* benefit and you're effectively on long context. CAG needs query density, not just a static corpus; if traffic is sparse, RAG (or accepting long-context cost) is often cheaper.
- **RAG recall miss that reads as a hallucination.** The user asks something the retriever fails to surface; the model answers from priors, confidently wrong. This looks like a model problem but it's a *retrieval* problem. Log the retrieved chunks alongside every answer so you can tell "model ignored good context" from "retrieval never fetched it."
- **Hybrid is common and correct.** Real systems mix strategies within one prompt: CAG for stable system-level knowledge (glossary, policies, tool schemas, few-shot examples) as a cached prefix, *plus* RAG for the large dynamic corpus appended after it. You get one cheap warm prefix and a small per-query retrieval on top. This is the workhorse pattern for production assistants.

```
[ cached CAG prefix: system prompt + glossary + tool schemas ]  <- warm, ~free
[ RAG chunks retrieved for THIS query ]                          <- small, per-request
[ user question ]                                                <- tiny
```

## Common misconceptions & failure modes

- **"The window is 1M tokens, so using 1M tokens is fine."** The window is a *ceiling*, not a target. Cost scales with tokens *used*, every call, and reliability *drops* long before the ceiling (RULER/needle tests show it). Use the fewest tokens that answer the question.
- **"Long context replaces RAG now."** It replaces RAG only when the corpus is small enough to always fit *and* you're willing to pay full prefill every call *and* quality holds at that length. For large or dynamic corpora, RAG is still cheaper and more scalable by a wide margin.
- **"CAG is just long context."** No — CAG's whole point is prefilling **once** and reusing. If your corpus changes often (or traffic is too sparse to stay warm), the reuse never happens and CAG collapses *into* long context. CAG's advantage lives entirely in the cache-hit rate, which is a function of corpus stability and query density.
- **"Prompt caching means those tokens are free."** They're *discounted* (typically ~10% read cost on hosted APIs) or *compute-skipped* (self-hosted vLLM), not free — and the write/cold path can cost slightly *more* than a normal token. The first (cold) request always pays full price, and the cache can expire. Model the warm/cold mix, not just the warm case.
- **"RAG is always the answer."** RAG adds a whole pipeline (chunking, embeddings, a vector store, retrieval eval) and a recall failure mode. For a 20k-token static corpus, that's over-engineering — CAG is simpler *and* has no recall risk.
- **"Put the important stuff anywhere; the model reads it all."** Position matters. In long context, front and end are attended best; the middle is where facts go to die. If you must use long context, order deliberately.
- **"CAG and reasoning tokens compose for free."** They don't (Lecture 11). Prompt caching caches *input*; thinking tokens are freshly generated *output* every call and never cache. A CAG route that also thinks re-pays the thinking cost on every request regardless of how warm the prefix is.

## Rules of thumb / cheat sheet

- **The rubric:** small + static → **CAG**; large + dynamic → **RAG**; genuinely need everything at once and can pay → **long context**. Memorize it; it settles 90% of arguments.
- **Cost intuition:** long context pays full prefill *every* call; RAG pays a small prefill *every* call; CAG pays full prefill *once*. Pick based on how often you'd re-pay.
- **"Small enough for CAG"** ≈ fits in the window with room for the question *and* is well within where the model still attends reliably (test it — often far under the max).
- **"Static enough for CAG"** ≈ updates far less often than you get queries (`N` is large) *and* traffic is dense enough to beat the cache TTL. Twice a year at 5k/day: perfect. Hourly, or 20 queries/day: use RAG.
- **Always price input tokens** at real per-request context size × calls/month. A fat context is a recurring bill, not a one-time cost.
- **Watch TTFT, not just cost:** prefill dominates TTFT, so a huge context blows your responsiveness SLO even if you can afford the tokens.
- **Keep the cached prefix byte-identical** — no timestamps, request ids, or per-user tokens *before* the corpus, or you kill the cache (Lecture 02's cache-buster).
- **Log retrieved chunks with every RAG answer** so recall misses are debuggable and not mistaken for hallucinations.
- **Hybrid is legitimate and usually best at scale:** CAG the stable prefix + RAG the dynamic bulk. Don't force one strategy on the whole prompt.
- **Numbers here are approximate** and drift with provider pricing and model behavior — re-measure for your model and corpus.

## Connect to the lab

Week 3's theory block sits right on this decision, and the release-pipeline lab is where it gets teeth. When you build `gateway/router.py`, a "version" is a prompt+model pair — and *how that prompt sources its knowledge* (long context vs a RAG retrieval step vs a cached CAG prefix) is exactly the kind of change you'd shadow and A/B behind a flag. Wire your `tenant_id` cost tagging (Week 2 FinOps) to record input-token counts per strategy so you can prove the cost delta with real traces, not this lecture's estimates. If the GPU is up, re-run a short k6 pass with `--enable-prefix-caching` against a fat shared prefix (that's CAG in miniature) and watch TTFT drop between the cold first request and the warm rest — the same experiment as Week 1 Step 3, now reframed as an architecture choice you can defend with numbers.

## Going deeper (optional)

- **Lost in the middle:** Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023) — the canonical position-bias paper. Search the title.
- **RULER benchmark:** NVIDIA's long-context evaluation suite — search "RULER long context benchmark github". Read *about* the results (degradation before max length); don't reimplement.
- **Needle in a Haystack:** Greg Kamradt's original pressure test — search "needle in a haystack LLM test". Awareness-level only.
- **Prompt caching docs (the ground truth for CAG economics):** Anthropic prompt caching (`docs.anthropic.com`), OpenAI prompt caching (`platform.openai.com/docs`), Google Gemini context caching (`ai.google.dev`). Read the *pricing* and *TTL/invalidation* sections — that's the whole CAG cost model.
- **CAG paper:** search "Cache-Augmented Generation" / "Don't Do RAG: When Cache-Augmented Generation is All You Need" — the source of the term and its "small stable corpus" framing.
- **vLLM automatic prefix caching:** `docs.vllm.ai` — the self-hosted mechanism CAG rides on (ties to Lecture 02).
- **RAG foundations:** revisit your Phase 6 material and search "RAG vs long context tradeoffs 2025" for current practitioner writeups.

## Check yourself

1. State the three-way rubric in one line each: which strategy for small+static, large+dynamic, and need-everything-at-once?
2. You have a 100k-token corpus and serve 2,000 queries/day. Roughly how much *more* do you pay in input tokens with long context vs a RAG setup that retrieves 2k tokens/query, at $0.20/1M input? (Order of magnitude is fine.)
3. Why does CAG collapse into "long context with extra steps" when the corpus updates frequently — and give a *second* reason CAG can collapse even when the corpus never changes.
4. A fact sits at token 60,000 of an 80k-token prompt and the model misses it, but nails the same fact in a 3k-token prompt. What's the phenomenon called, and what are two fixes?
5. A RAG system confidently gives a wrong answer. Why might this be a *retrieval* bug rather than a model bug, and what should you have logged to tell the difference?
6. Which cache-buster from Lecture 02 silently destroys CAG's savings, and where must it not appear?

### Answer key

1. **Small + static → CAG** (prefill the corpus once, reuse the cached prefix). **Large + dynamic → RAG** (retrieve only the relevant chunks per query). **Need everything at once and can pay → long context** (accept full prefill every call).
2. Long context: `100,000 × $0.20/1M ≈ $0.020`/request × 2,000 = **~$40/day**. RAG: `2,000 × $0.20/1M ≈ $0.0004`/request × 2,000 = **~$0.80/day** (plus tiny embedding cost). Long context costs **~50× more** in input tokens — the ratio of context sizes (100k vs 2k). Roughly $40 vs under $1 a day.
3. CAG's entire benefit is the **cache hit**: prefill the corpus once, reuse it for all later requests. The cache is keyed on the exact prefix, so any change to the corpus invalidates it and forces a full cold prefill; frequent updates mean most requests hit a cold/freshly-rebuilt cache — that *is* long context, minus the simplicity. **Second reason:** even with a never-changing corpus, a cache has a **TTL** (and can be evicted under memory pressure). If traffic is sparse enough that requests arrive after the cache expires, each one pays a cold prefill — the reuse never happens, so CAG again collapses into long context.
4. **Lost in the middle** (position bias / context rot). Fixes: (a) switch to RAG so the model only sees the small relevant slice; (b) if you must use long context, reorder so critical content sits at the **start or end**, not buried in the middle; (c) shrink the overall context so you're well inside the model's reliable-attention range.
5. RAG quality is capped by retrieval: if the retriever never surfaces the chunk containing the answer, the model answers from its priors — confidently wrong. That's a **recall miss**, not the model reasoning badly. You should have **logged the retrieved chunks alongside each answer**; then you can distinguish "answer chunk was never fetched" (retrieval bug) from "chunk was fetched but ignored" (model/prompt bug).
6. A **per-request-varying token before the corpus** — a timestamp, request id, or per-user token placed *ahead* of the shared corpus. It makes every prefix unique, so nothing is ever a cache hit. The cached, byte-identical corpus must come **first**; put any variable content (the user's question, timestamps) **after** it.
