# Lecture 14: Long-Context Handling vs RAG

> There is a reflex, learned from a thousand blog posts, that goes: "I have a document and a question, therefore I need a vector database." That reflex is wrong often enough to cost you weeks. The honest first question is not "which embedding model?" — it is "does the document even *fit* in the window, and if it does, is retrieval buying me anything I can measure?" For a single document that fits under the context window, stuffing the whole thing into the prompt frequently *beats* retrieval on accuracy, ships in an afternoon, and costs a knowable number of cents per answer — while RAG adds a chunker, an embedding pipeline, an index, a retriever, and a whole new class of "the right chunk wasn't retrieved" bugs. This lecture installs the *decision*, not the machinery. After it you'll be able to compute cost-per-answer for both strategies, reason about chunking tradeoffs well enough to argue about them, and — for the invoice task — write the one-sentence justification for *not* building RAG yet.

**Prerequisites:** Lecture 1 (prompt anatomy, lost-in-the-middle), Lecture 11 (context rot / relevance over volume), Lecture 5 (token counting), comfort with arithmetic on token prices · **Reading time:** ~28 min · **Part of:** Phase 1 Week 3

---

## The core idea (plain language)

You have text the model needs to see to answer a question. There are exactly two ways to get it in front of the model:

1. **Stuff it.** Put the whole document in the prompt. The model attends over all of it. Simple, no infrastructure, deterministic — the model literally sees every token you sent.

2. **Retrieve it.** Chop the document (or corpus) into chunks ahead of time, embed each chunk into a vector, and at query time fetch the top-*k* chunks most similar to the question. Put only those chunks in the prompt. This is **RAG** (retrieval-augmented generation), and it is a Phase 3–4 topic — we are *not* building it here.

The industry defaults to (2) because of a scaling story: "you can't fit a million documents in a window, so you must retrieve." That's true *at corpus scale*. But it quietly smuggles in a false premise — that retrieval is the default and stuffing is the fallback. For a **single document that fits under the window**, the premise inverts. Stuffing wins on the axes you actually care about:

- **Accuracy** — the model sees the whole document, so it can never miss the relevant passage because a retriever ranked it #6 when you asked for top-5. Retrieval's ceiling is bounded by "did the right chunk get fetched?"; stuffing has no such ceiling.
- **Engineering cost** — zero new moving parts. No chunker to tune, no index to keep in sync, no embedding model to version.
- **Debuggability** — a wrong stuffed answer has two suspects: the prompt or the model. A wrong RAG answer has six: the chunker, the embedder, the retriever's *k*, the similarity metric, the re-ranker, *or* the prompt.

The one axis retrieval can win is **cost-per-answer at repeated-query scale** — if you ask 10,000 questions against the same fixed corpus, paying to send the whole corpus 10,000 times is absurd, and retrieval amortizes the ingestion cost across queries. That is the real decision boundary, and it's a number you can compute, not a feeling.

> **The claim to internalize:** for a single document under the window, benchmark full-context cost-per-answer *first*. Only add RAG plumbing when a repeated-query cost calculation or a document-exceeds-window constraint forces it.

---

## How it actually works (mechanism, from first principles)

### The window is the whole game

Every model has a **context window** — a hard token ceiling on input + output for a single call. In 2025–2026 the numbers that matter:

```
Model family (approx, 2025-2026)     input window
Claude (Sonnet/Opus current)         ~200K tokens
GPT-4-class / GPT-5-class            ~128K–400K tokens (varies by model)
Gemini 1.5/2.x Pro                   ~1M–2M tokens   ← the long-context outlier
```

*(Windows change with each release — confirm against the provider's current model page; these are order-of-magnitude.)*

The single most important sizing fact: **a page of dense text is roughly 500–800 tokens.** So a 200K window holds ~300–400 pages; a 1M-token Gemini window holds a small book or a large codebase. The decision "stuff vs retrieve" starts with one division:

```
document_tokens  vs  window_tokens (minus your prompt overhead + desired output)
```

If `document_tokens` comfortably fits — say under ~50% of the window, leaving room for instructions, few-shot, and output — stuffing is *on the table*. If it blows past the window, you must chunk and retrieve (or map-reduce, a different lecture). The invoice task lives firmly in "fits": a single invoice is a few hundred to a couple thousand tokens. There is no window pressure at all — which is precisely why reaching for RAG here would be pure over-engineering.

### Why retrieval has an accuracy ceiling stuffing doesn't

Retrieval answers a question by first *guessing* which chunks are relevant, using vector similarity between the query embedding and each chunk embedding. That guess is imperfect:

- The relevant passage might be phrased so differently from the question that its embedding sits far away (the classic "lexical gap" — the question says "amount owed," the invoice says "balance due").
- The answer might be **split across two chunks** so neither alone contains it.
- You fetch top-*k*; the right chunk ranks *k+1* and never makes it in.

Every one of these is a *silent miss*: the model answers confidently from the wrong chunks and you get a fluent wrong answer. Stuffing eliminates the retrieval step entirely — the model attends over the full document, so "the passage wasn't retrieved" is not a failure mode that exists. You've traded a *retrieval* accuracy risk for an *attention* accuracy risk (lost-in-the-middle, below), and the attention risk is usually smaller and always cheaper to mitigate.

### Chunking vocabulary at working depth

You won't *build* a chunker this phase, but you must be able to reason about the tradeoffs, because the day you do reach for RAG the chunking decision dominates its accuracy. Four terms:

**Structural chunking** — split on the document's own boundaries: headings, paragraphs, markdown sections, code functions, table rows. Cheap (no model call), respects the author's structure, keeps coherent units together *when the document is well-structured*. Fails on unstructured walls of text where a paragraph break isn't a topic break.

**Semantic chunking** — split where the *meaning* shifts, typically by embedding sentences and cutting where adjacent-sentence similarity drops below a threshold. Produces topically-tight chunks, but costs an embedding pass over every sentence at ingestion and can be finicky to tune. Buys you fewer chunks that straddle two topics (which is what pollutes retrieval).

**Overlap** — let adjacent chunks share a sliding window of tokens (e.g., 512-token chunks with 64-token overlap). Buys insurance against the "answer split across a boundary" failure: if the key sentence sits at a chunk edge, the overlap means it appears whole in one of the two chunks. **Token cost:** pure duplication — 64-token overlap on 512-token chunks inflates your stored/embedded token count by ~12.5%, paid at ingestion and (indirectly) at retrieval.

**Parent-child chunking** — embed *small* chunks (precise retrieval — a tight sentence matches a query well) but, once retrieved, send the *parent* (surrounding paragraph or section) to the model. You retrieve on the child and answer on the parent — the best of both. **Token cost:** the answer prompt carries parent-sized context per hit, so your per-answer bill goes up relative to sending bare children — trading tokens for accuracy.

```
Chunking strategy      retrieval precision   answer context   ingest cost   token cost/answer
structural             medium                medium           low           low
semantic               high                  medium           high (embed)  low
+ overlap              (any) +insurance      +boundary safety  +~10-15%      +overlap dup
parent-child           high (small child)    high (big parent) medium        higher (parent)
```

The meta-point: **every chunking knob is a token-cost-vs-accuracy trade**, and all of them are complexity you *don't* incur when you stuff. That's the weight on the "don't build RAG yet" side of the scale.

### Lost-in-the-middle: the tax you pay for stuffing

Stuffing isn't free of accuracy risk — it has exactly one, and it's the one from Lecture 1. Transformers attend *unevenly* across position: they attend strongly to the **start** (primacy) and **end** (recency) of the context, and *under-attend to the middle* — the "lost-in-the-middle" trough. Empirically, retrieval accuracy for a single fact plotted against its position in a long context looks like a **U**:

```
accuracy
  ^
1.0|●                                             ●
   | ●                                           ●
   |   ●                                       ●
   |      ●                                 ●
0.6|          ●●●        ●●●        ●●●          ← the trough
   +----------------------------------------------> position
    start            middle                 end
```

*(Shape is well-documented and robust; exact depth varies by model and context length — measure yours, don't quote a fixed number.)* This is why the operative policy from Lecture 1 is **critical facts to the edges**. When you stuff a long document and need the model to nail one buried number, don't rely on it being read from the middle — restate the key instruction/question at the *end*, right before generation.

Crucially, **the trough deepens as context grows.** A fact at 50% depth in a 4K-token prompt is nearly always read; the same fact at 50% depth in a 400K-token prompt is meaningfully more likely to be missed. So "it fits in the window" is necessary but not sufficient — a document that fits in a 1M window but sits at 900K tokens is *fitting* but *fighting the trough hard*. This is the subtle boundary where stuffing starts to lose to retrieval even below the window limit: not because it doesn't fit, but because the middle of a near-full window is a bad place to keep the answer.

---

## Worked example

The invoice task, made concrete. One invoice, and you want `{vendor, invoice_number, date, total_amount, currency}`.

**Sizing.** A messy invoice text blob is ~400–1,500 tokens. Your prompt (system + schema + a couple exemplars) is ~600 tokens. Output is ~60 tokens. Total per call: **~1,100–2,200 tokens**, against a 200K window — you are using ~1% of the window. There is *no* fit problem, *no* trough problem (the whole document sits in the high-attention zone), and *no* reason retrieval could improve accuracy — the model already sees every token. RAG here would add an embedding pipeline to search a document that fits in a tweet-and-a-half.

**Cost-per-answer, stuffing.** Use illustrative pricing **$3 / 1M input, $15 / 1M output**:

```
input:  1,800 tokens × $3 / 1e6   = $0.0054
output:    60 tokens × $15 / 1e6  = $0.0009
                                    ---------
cost-per-answer (stuff)           ≈ $0.0063   (~0.63 cents)
```

That is the number that ends the argument. **~0.6 cents per invoice, zero infrastructure.** Write it in the README: "we do not use RAG for single-invoice extraction because full-context stuffing costs ~$0.006/answer with no retrieval accuracy risk."

**Now stress-test the boundary: when would RAG win?** Imagine the task changed — not one invoice, but a **fixed 5,000-page vendor contract corpus** (~3M tokens) you answer questions against repeatedly.

*Stuffing the corpus per query* isn't even possible (3M > 200K window). Even on a 1M Gemini window it wouldn't fit, and if it did:

```
stuff 3M tokens × $3 / 1e6 = $9.00 PER QUESTION   ← absurd at repeat scale
```

*RAG:* embed the corpus once (a one-time ingestion cost), then per query fetch, say, 6 chunks × 500 tokens = 3,000 tokens of context:

```
per query: ~3,600 input × $3/1e6 + 60 output × $15/1e6 ≈ $0.0117
```

Retrieval is ~$0.012/answer vs $9/answer for stuffing — and stuffing doesn't even fit. *That* is the decision boundary, and it turned on two things: **document exceeds window** and **repeated queries over a fixed corpus**. The invoice task has *neither* — it's a single small document, and each invoice is a *one-shot extraction*, not repeated Q&A over a persistent corpus. Both signals point at stuffing.

**The query-pattern axis, stated plainly:**

| Signal | Points to stuffing | Points to RAG |
|---|---|---|
| Document size vs window | fits (< ~50% window) | exceeds window |
| Query pattern | one-shot extraction | repeated Q&A over a corpus |
| Corpus persistence | ephemeral (this doc, this call) | fixed corpus, many queries |
| Cost-per-answer | low (small doc) | stuffing cost × N queries is the killer |
| Accuracy risk you fear | none from retrieval | lost-in-the-middle in a near-full window |

The invoice task lands in the left column on every row. That's the justification.

---

## How it shows up in production

**The premature vector DB.** A team ships invoice extraction and, because "everyone uses RAG for documents," stands up a vector store, an embedding job, and a retriever — to search a 1,200-token document. They now maintain an index that must stay in sync with the source, pay embedding costs, debug "why didn't it retrieve the total line," and have *lower* accuracy than a five-line stuffing prompt would have delivered. The fix is deletion of the entire subsystem. This is the single most common over-engineering in document pipelines: **RAG applied to documents that fit.**

**The cost cliff at corpus scale.** The mirror-image mistake: a team stuffs a growing knowledge base into every call because "stuffing was simpler." It worked at 20 documents. At 2,000 documents each query sends 400K tokens, blows the window, and the bill is $5+/query. They didn't watch the *repeated-query cost-per-answer* number, so they didn't see the cliff coming. The lesson cuts both ways: stuffing is the right *default for single docs*, and the *wrong* default the moment you're doing repeated Q&A over a corpus that grows.

**Context rot from irrelevant chunks (Lecture 11).** When teams *do* build RAG and set *k* too high "to be safe," they stuff 15 chunks in when 3 were relevant. The 12 irrelevant chunks don't just cost tokens — they *degrade accuracy* by distracting the model (the context-rot effect from Lecture 11). This is the crucial link: retrieval doesn't only risk *missing* the right chunk, it risks *including* wrong ones. A stuffed single document has no irrelevant chunks by construction — another quiet accuracy advantage of stuffing when the doc is small and on-topic.

**Gemini and the long-context play.** Gemini's ~1M–2M token window changes the *fit* threshold dramatically. Documents that force RAG on a 200K model — a 300-page contract, a full codebase — fit whole in Gemini. For those, "stuff into Gemini" is a genuine, competitive alternative to building RAG, often winning on accuracy and simplicity for one-shot or low-volume queries. The caveat is the trough: at 900K tokens of a 1M window, lost-in-the-middle is real and deep, so put the question at the very end. Long context doesn't repeal the attention curve; it just moves the fit boundary.

**Latency.** Stuffing a big document means the model reads every token before answering — input-processing time scales with input length, so a 400K-token stuff can add seconds of time-to-first-token vs a 3K-token retrieved prompt. At corpus scale this is a *second* reason retrieval wins; at single-small-document scale it's a non-issue.

---

## Common misconceptions & failure modes

- **"Documents mean RAG."** No — *corpora that exceed the window, queried repeatedly* mean RAG. A single document under the window means *stuff it and measure*.
- **"Stuffing is the naive choice; RAG is the real engineering."** Backwards. Stuffing is the *baseline you must beat*. Adding RAG you can't justify with a cost-per-answer or fit number is the amateur move.
- **"If it fits in the window, stuffing is always fine."** Only mostly. A document that fits but sits near the top of a large window pays the lost-in-the-middle tax hard — watch *depth*, not just fit.
- **"More retrieved chunks = safer."** High *k* imports irrelevant chunks that cause context rot (Lecture 11) — lower accuracy *and* higher cost. Retrieval quality is about the *right* chunks, not *more* chunks.
- **"Overlap and parent-child are free accuracy."** They're accuracy bought with tokens: overlap duplicates ~10–15% of your text; parent-child sends parent-sized context per hit. Every chunking knob is a token-cost trade.
- **"Long-context models made RAG obsolete."** No — they moved the fit boundary. At true corpus scale, stuffing is still cost-prohibitive even in a 1M window, and the trough still bites near the top.
- **"I'll decide stuff-vs-retrieve by intuition."** The whole point is that it's a *computation*: document_tokens vs window, query count × cost-per-answer, and the accuracy risk of each. Compute it.

---

## Rules of thumb / cheat sheet

- **Default for a single document under the window: STUFF IT.** Add RAG only when fit or repeated-query cost forces it.
- **First two questions, always:** (1) Does `document_tokens` fit under ~50% of the window (leaving room for prompt + output)? (2) Is this one-shot, or repeated Q&A over a fixed corpus?
- **Compute cost-per-answer both ways before building anything.** Stuffing: `(input × in_price) + (output × out_price)`. RAG: amortized ingestion + `(retrieved + prompt) × in_price` per query. The number decides.
- **The RAG-wins signal is a product:** `stuffing_cost_per_answer × number_of_queries` becomes large *and* the corpus is fixed/reused. Single one-shot doc → that product is tiny → stuff.
- **Fits-but-huge? Watch the trough.** Put the question/critical instruction at the *end*; don't trust a fact buried at 50% depth in a near-full window.
- **Chunking knobs are token-cost trades:** semantic = better precision, ingest cost; overlap = boundary insurance, +10–15% tokens; parent-child = precise retrieval + rich answer context, bigger answer prompts. Know them to argue; don't build them this phase.
- **Keep retrieved *k* small.** Irrelevant chunks = context rot = lower accuracy + higher cost (Lecture 11).
- **Gemini's ~1M+ window** is the long-context tool of choice — it makes "stuff the whole book/codebase" viable for one-shot/low-volume queries. Still mind the trough.
- **For the invoice task specifically:** one tiny document, one-shot extraction → **stuff, ~$0.006/answer, no RAG.** Write that sentence in the README.

---

## Connect to the lab

Week 3's theory bullet says it directly: *for a single document under the window, stuffing full context often beats retrieval — benchmark cost-per-answer before adding RAG machinery.* Your Definition-of-Done line for the phase milestone — the README must document "the *when full-context beats RAG* cost note for this task" — **is** the deliverable this lecture prepares. Compute the ~$0.006/answer stuffing number from your own token-budget instrument (the per-category breakdown you build in the lab), and write the one-sentence justification for not building RAG. The "lost in the middle" ablation you run this week (fact at start / middle / end across ≥20 trials) is the *same trough* that sets the upper boundary on stuffing.

---

## Going deeper (optional)

- **"Lost in the Middle: How Language Models Use Long Contexts"** (Liu et al., 2023) — the canonical paper on the U-shaped position curve. Read once for the mechanism behind "critical facts to the edges." Search that exact title.
- **Anthropic docs — "Long context tips" / "Context windows"** (docs.anthropic.com) — practical guidance on structuring very long prompts, including putting the query at the end. Search: `Anthropic long context tips`.
- **Google Gemini docs — "Long context"** (ai.google.dev) — the 1M–2M token window and how Google frames stuff-vs-retrieve; note their own long-context-vs-RAG discussion. Search: `Gemini long context documentation`.
- **LlamaIndex / LangChain docs — chunking & node parsers** (root docs) — for when you *do* build RAG in Phase 3–4: sentence-window (semantic), parent-document (parent-child), and overlap settings live here. Search: `LlamaIndex parent document retriever`, `LangChain text splitter overlap`.
- **Anthropic — "Contextual Retrieval"** (Anthropic engineering blog) — a strong 2024 treatment of chunk-context loss and how to fix it; read it *before* building RAG, not now. Search: `Anthropic contextual retrieval`.
- Search query for the current debate: **"long context vs RAG when to use 2025"** — to see the evolving consensus as windows grow.

---

## Check yourself

1. State the decision boundary between stuffing and RAG in terms of the two signals that actually flip it, and place the invoice task on both.
2. Compute cost-per-answer for stuffing a 1,800-token prompt with 60 output tokens at $3/1M input and $15/1M output. Then argue why this number ends the "should we add RAG?" debate for single-invoice extraction.
3. Retrieval has an accuracy *ceiling* that stuffing does not. Explain the mechanism, and name two distinct ways retrieval silently returns the wrong context.
4. A document *fits* in a 1M-token window but sits at 850K tokens. Why might stuffing still lose accuracy here, and what's the mechanism? What's your mitigation?
5. Define overlap and parent-child chunking, and state the *token cost* each one adds and the accuracy problem each one buys down.
6. Your teammate proposes setting retrieval `k=20` "to be safe." Using the context-rot idea from Lecture 11, explain why this can *lower* accuracy, not just raise cost.

### Answer key

1. The boundary flips on **(a) document size vs window** — fits under it vs exceeds it — and **(b) query pattern / corpus persistence** — one-shot extraction over an ephemeral document vs repeated Q&A over a fixed, reused corpus. RAG wins when the document exceeds the window *and/or* you query a fixed corpus many times (so `cost_per_answer × N_queries` gets large). The invoice task fits easily (~1% of a 200K window) and is one-shot per invoice, so it lands on the *stuffing* side of both signals.
2. `(1,800 × $3/1e6) + (60 × $15/1e6) = $0.0054 + $0.0009 = $0.0063`, about 0.6 cents. It ends the debate because RAG would add a chunker, embedder, index, and retriever (plus retrieval-miss bugs and *lower* accuracy from dropped chunks) to search a document that costs less than a cent to send in full — there's no cost or accuracy problem for RAG to solve here.
3. Retrieval first *guesses* which chunks are relevant via vector similarity, and only the model sees the fetched chunks; if the guess is wrong the answer is bounded by that error. Stuffing has no guess — the model sees every token — so "the right passage wasn't retrieved" isn't a possible failure. Two silent-miss modes: (a) the relevant passage is phrased so differently from the query that its embedding is far away (lexical gap), and (b) the answer is split across a chunk boundary so no single retrieved chunk contains it (or the right chunk ranks *k+1* and is dropped).
4. Even though it fits, a fact at ~50% depth in a near-full window sits in the **lost-in-the-middle trough** — transformers attend strongly to start and end and under-attend to the middle, and the trough *deepens* as context grows, so a fact at 850K/1M is meaningfully more likely to be missed. Mitigation: put the question/critical instruction at the very end (critical facts to the edges) and don't assume a mid-context fact will be read.
5. **Overlap:** adjacent chunks share a sliding window of tokens; it buys down the "answer split across a boundary" miss, at a token cost of duplicating ~10–15% of the text at ingestion/embedding. **Parent-child:** embed and retrieve on small child chunks (precise matching) but send the larger parent to the model for answering; it buys both retrieval precision *and* enough surrounding context to answer, at a token cost of parent-sized context per retrieved hit.
6. High *k* imports chunks that are only weakly relevant; those irrelevant chunks cause **context rot** (Lecture 11) — the model is distracted by off-topic tokens and its answer quality *drops*, not just its token bill. Retrieval quality is about fetching the *right* few chunks, not the *most*; `k=20` trades accuracy away while also paying for 17 chunks you didn't need.
