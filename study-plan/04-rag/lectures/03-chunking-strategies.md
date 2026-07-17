# Lecture 3: Chunking Strategies — The Highest-Leverage Knob

> Chunking is the one offline decision that quietly sets the ceiling on everything downstream, and almost nobody spends enough time on it. Teams will A/B three rerankers and tune prompt wording for a week while shipping the LangChain default `chunk_size` they never questioned — a default that may be silently truncating half their chunks. This lecture reframes chunking as a single hard question: *what pieces of your corpus are allowed to be retrieved together?* Because whatever a chunk boundary separates can never co-retrieve, and whatever a chunk dilutes will never rank. After this lecture you'll be able to pick a chunking strategy from the actual shape of your documents (not habit), reason about the size tradeoff in tokens rather than vibes, name the one gotcha that causes phantom recall loss on almost every first RAG build, and wire every strategy behind one interface so you can swap them by config name and measure which wins.

**Prerequisites:** Lecture 1 (two pipelines, 7 failure points), Lecture 2 (layout-aware parsing), embeddings + cosine/IP similarity, tokenization basics. · **Reading time:** ~26 min · **Part of:** Retrieval-Augmented Generation, Week 1

## The core idea (plain language)

A chunk is the atomic unit of retrieval. Your vector index doesn't store documents; it stores chunks. When a query comes in, the retriever returns *chunks*. The LLM answers from *chunks*. So the boundaries you draw at index time are the boundaries of what can possibly be retrieved as a unit.

That gives us the single sentence to hang this whole lecture on:

> **Chunking decides what CAN co-retrieve. Anything a boundary splits apart, no retriever can put back together. Chunking is therefore a hard ceiling on recall — set before a single query is ever asked.**

This is why chunking is the highest-leverage knob. A reranker can only reorder chunks that were retrieved. Query rewriting can only fetch chunks that exist. Both operate *downstream* of the boundaries you already froze. If the fact the user needs was split across two chunks that never appear together in top-k, or buried inside a giant chunk whose embedding is too diluted to rank, then no online technique reaches it. The information loss happened in a batch job, and (per Lecture 1) the only fix is a reindex.

There is also a two-sided tension that governs *every* chunking choice:

- **Chunks too small** → context is fragmented. A fact that needs three sentences of surrounding setup gets split from that setup; the answer straddles a boundary and only half co-retrieves. This is **failure point #7 (incomplete)**.
- **Chunks too big** → the embedding is diluted. An embedding is *one* fixed-length vector summarizing the *whole* chunk. Cram five topics into one chunk and the vector becomes a mushy average of all five, matching every query weakly and no query strongly. Precision collapses; the right chunk ranks below k. This is **failure point #2 (missed top-k)**.

Every strategy below is, at bottom, a different answer to "how do I keep chunks small enough to be precise but big enough to be complete?"

## How it actually works (mechanism, from first principles)

### Why one vector per chunk forces the whole problem

Start from the mechanism. A bi-encoder embedding model takes a chunk of text and produces **one** vector of fixed dimension `d` (e.g. 384 for `bge-small-en-v1.5`, 768 or 1024 for larger models). That single vector has to represent everything in the chunk. Retrieval ranks chunks by cosine similarity between the query vector and each chunk vector.

Two consequences fall directly out of "one vector, whole chunk":

1. **A chunk covering many topics has a vector that is roughly the average of those topics.** Averages sit in the middle, far from any specific point. So a broad chunk is a weak match to *every* specific query. That's the dilution mechanism behind #2.
2. **A fact split across two chunks lives in two different vectors.** The query might match one strongly and the other not at all, so only half the answer surfaces in top-k. That's the fragmentation mechanism behind #7.

Good chunking maximizes *within-chunk topical coherence* (so the vector points somewhere specific) while keeping *semantically-joined material together* (so complete answers survive). Those two goals pull in opposite directions, which is why there is no single right chunk size — only a right strategy for a given document shape.

### The strategy ladder

Here is the ladder from crudest to most sophisticated. Read it as increasing respect for the document's structure and increasing cost.

```
  crude / cheap                                              smart / expensive
  ────────────────────────────────────────────────────────────────────────▶
  fixed-size → recursive-char → structure-aware → semantic → parent-doc /
  (structure   (respects        (uses the doc's   (splits on   sentence-window /
   blind)       separators)      own headings)     meaning)     contextual
```

**1. Fixed-size (hard cut).** Slice every `N` characters (or tokens), blind to everything. `text[i:i+800]`. Zero dependencies, deterministic, trivially fast.
- *Mechanism:* a knife every N chars. It will cut mid-word, mid-sentence, mid-table, mid-thought.
- *When to use:* as a **baseline only** — the thing you measure everything else against. Never ship it as your primary strategy on prose.
- *Cost:* nearly free; the cost is *quality*, paid downstream as #7.

**2. Recursive-character (the sensible default).** LangChain's `RecursiveCharacterTextSplitter`. It tries a *hierarchy of separators* in order and only falls back to a coarser cut when a piece is still too big.
- *Mechanism:* the separator list defaults to `["\n\n", "\n", " ", ""]`. It first tries to split on paragraph breaks (`\n\n`). If a resulting piece exceeds `chunk_size`, it recurses and splits that piece on single newlines, then spaces, then finally raw characters as the last resort. The effect: it keeps paragraphs whole when it can, sentences whole when it can't keep paragraphs, and only ever breaks a word when there is no other option.
- *When to use:* your **default** for general prose. It's the right first thing to reach for on 90% of corpora.
- *Cost:* cheap, no model needed, pure string ops.

**3. Semantic splitting.** LlamaIndex's `SemanticSplitterNodeParser`. Split where the *topic* shifts, detected by embedding similarity between adjacent sentences.
- *Mechanism:* embed each sentence (or small sentence group). Walk the document computing cosine similarity between consecutive sentence embeddings. Where similarity *drops* below a threshold (a percentile of the distribution, e.g. the 95th percentile of "distance"), that's a topic boundary — cut there. Coherent runs of on-topic sentences stay together; a hard subject change starts a new chunk.
- *When to use:* documents with uneven, unmarked topical structure (interview transcripts, meeting notes, unstructured articles) where headings don't exist but topics genuinely shift.
- *Cost:* **you must run an embedder at index time just to decide boundaries** — before the real embedding pass. Roughly doubles index-time embedding work and adds latency. Often not worth it over recursive-char; measure before adopting.

**4. Structure-aware.** Split on the document's *own* hierarchy — Markdown headings, HTML sections. LangChain `MarkdownHeaderTextSplitter`, LlamaIndex `MarkdownNodeParser`.
- *Mechanism:* the document already encodes human-authored topic boundaries as headings (`#`, `##`, `###`). Split on those. Each chunk is one section, and the heading path (`H1 > H2 > H3`) rides along as metadata — free provenance *and* free context about where the chunk sits.
- *When to use:* the best choice **when your parser preserved structure** (Docling/Unstructured emitting clean Markdown). A section is a unit a human already decided is coherent — you're borrowing their judgment for free.
- *Cost:* cheap. The catch is upstream: it's only as good as your parser. Feed it a flat PDF dump with no headings and it degrades to one giant chunk. This is why Lecture 2 (parsing) comes before this one — structure-aware chunking is the *payoff* for layout-aware parsing.
- *Caveat:* a single section can still be huge. Real pipelines run structure-aware **then** recursive-char *within* each oversized section — a two-level split.

**5. Overlap (a modifier, not a strategy).** Carry the last ~10–20% of each chunk into the start of the next.
- *Mechanism:* with `chunk_size=800, chunk_overlap=120`, chunk N ends at char 800 and chunk N+1 *starts* at char 680, replaying the last 120 chars. A fact sitting right on a boundary now appears *complete* in at least one chunk instead of severed in both.
- *When to use:* almost always, as cheap insurance against boundary-straddling answers (**failure #7**). Standard is 10–20% of chunk size.
- *Cost:* index size grows by the overlap fraction (20% overlap ≈ 20% more chunks ≈ 20% more vectors and storage). Cheap insurance, but not free — don't set overlap to 50% "to be safe."

**6. Parent-document / small-to-big.** Embed *small* chunks for precise retrieval, but return the *larger parent* to the LLM. LangChain `ParentDocumentRetriever`, LlamaIndex `AutoMergingRetriever`.
- *Mechanism:* this directly resolves the size tradeoff by *decoupling the retrieval unit from the context unit.* Index tiny children (say 200–400 chars) so their vectors are sharp and specific — great precision, they rank well. But store a pointer from each child to its parent (the full paragraph or section). At query time you match on the child, then *return the parent* to the LLM so it gets full context. Small for finding, big for reading.
- *When to use:* when precise retrieval AND rich context both matter — which is most serious QA. A top-tier default when you've outgrown plain recursive-char.
- *Cost:* two-tier storage (children in the vector index, parents in a docstore) and a join at query time. More moving parts, modestly more code.

**7. Sentence-window.** LlamaIndex `SentenceWindowNodeParser`. The purest form of small-to-big: embed a *single sentence*, but return that sentence *plus a window of k neighbors* around it.
- *Mechanism:* embed each sentence alone (maximally precise vector). On a hit, expand to sentence ± window (e.g. ±3 sentences) and hand that to the LLM. The neighbors provide the context the single sentence lacked.
- *When to use:* factoid QA where the answer is one sentence but needs its surroundings to be understood. Excellent precision/recall balance.
- *Cost:* more chunks (one per sentence → many vectors), and the window size is a knob you must tune.

**8. Contextual chunking (Anthropic Contextual Retrieval, 2024).** Before embedding a chunk, prepend a 1–2 sentence LLM-generated summary of *where this chunk sits in the whole document.*
- *Mechanism:* the classic failure is a chunk like "The timeout was raised to 30 seconds in this release." Embedded alone, it doesn't say *which* timeout or *which* product — so a query naming the product won't match it. Contextual chunking sends the chunk **plus the whole document** to an LLM and asks for a short situating blurb — e.g. "This chunk is from the ACME Router v4.2 release notes, discussing the `connect()` call timeout." That blurb is prepended *before embedding*, so the chunk's vector now carries the missing context and matches product-specific queries. Anthropic reported large retrieval-failure reductions from this (treat the exact figures as their reported numbers, not a universal guarantee).
- *When to use:* corpora where chunks are riddled with unresolved pronouns/references ("the system," "this version," "the above") — release notes, legal, technical manuals with heavy cross-referencing.
- *Cost:* **one LLM call per chunk at index time.** For 10,000 chunks that's 10,000 calls. It's designed to be **prompt-cache-friendly**: the big shared document goes in the cached prefix so you pay full price for it once and cached price for every chunk after — which is the difference between "expensive" and "prohibitive." Still, this is the priciest strategy on the ladder; reserve it for corpora where the recall win justifies it, and measure.

### The size tradeoff, with arithmetic

The mechanism is worth making numeric. Suppose a chunk is 200 tokens of tightly on-topic text about `connect()` timeouts. Its embedding points squarely at "connect timeout." A query "what is the connect timeout" lands close — high cosine, ranks #1.

Now merge four adjacent 200-token sections (connect timeouts, read buffers, TLS setup, logging config) into one 800-token chunk. The embedding is now roughly a blend of four directions. The same query matches the "connect" component but is dragged toward the average by the other three-quarters. Cosine drops; the chunk that *contains* the answer now ranks #8 instead of #1 — **retrieved-but-missed-top-k, #2.** The answer was in the index the whole time. Nobody sees it.

Go the other way — chunk at 50 tokens — and "The default is 30 seconds" lands in a chunk that no longer contains the word `connect()` (that was two sentences earlier). The query matches weakly, or matches the wrong "30 seconds" elsewhere. **Fragmentation, #7.** Same information, destroyed by a boundary.

## The gotcha that bites almost every first build: chars ≠ tokens

This is the single most common silent bug in beginner RAG, and it deserves its own section.

**`chunk_size` in the character-based splitters (`RecursiveCharacterTextSplitter`, fixed-size slicing) counts CHARACTERS, not TOKENS.** But your embedding model measures its input limit in *tokens*. English averages roughly **~4 characters per token**, so:

```
  chunk_size (chars)     ≈ tokens
  ─────────────────────────────────
        800 chars        ≈ 200 tokens     ← fine for bge-small
       2048 chars        ≈ 512 tokens     ← right at bge-small's ceiling
       4000 chars        ≈ 1000 tokens    ← DOUBLE the ceiling → truncated!
```

`bge-small-en-v1.5` has a **max sequence length of ~512 tokens.** Feed it a chunk longer than that and the model does **not error** — it **silently truncates** the input to 512 tokens and embeds only that prefix. Everything past the cutoff is invisible to the vector.

Here's why this is so nasty. Say you set `chunk_size=4000` (feels reasonable — "bigger chunks, more context"). That's ~1000 tokens. The embedder silently keeps the first ~512 tokens (~2048 chars) and drops the rest. Now:

- The **second half of every chunk is not represented in its embedding.** Any answer living in the back half of a chunk is *unsearchable* — it exists in the index text but not in the vector that indexes it.
- Retrieval quietly misses those answers. Recall craters for a reason that never appears in any log. This is **phantom recall loss** — the chunk is "there," but its embedding is a lie about its contents.
- You'll debug everything *except* the real cause, because there's no error message pointing at truncation.

**The fix:** know your embedder's token limit, convert to a char budget with the ~4x rule (or better, use a **token-based** splitter — `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)` or LlamaIndex's token-aware parsers — so `chunk_size` means tokens directly), and keep `chunk_size + max prepended context` safely *under* the limit. For `bge-small`, keep chunks around ~200–400 tokens (~800–1600 chars), leaving headroom for overlap and any contextual prefix. When you adopt contextual chunking, remember its prepended blurb *also counts against the 512-token budget.*

## Worked example

**Corpus:** the ACME Router v4.2 admin manual, parsed by Docling into clean Markdown with headings and pipe-tables. **Embedder:** `bge-small-en-v1.5` (max 512 tokens, 384-dim). **Query:** *"What's the default connect() timeout on the v4.2 router?"* The answer lives in a table under `## Connection Settings > ### Timeouts`.

**Config A — fixed-size, 800 chars, no overlap.** The knife falls mid-table. Chunk 14 ends `... | connect() | 30`; chunk 15 starts `seconds | ...`. The pairing `connect() ↔ 30 seconds` is severed. Top-5 returns chunk 14 (has `connect()` but a stranded `30`) — the LLM can't safely answer. **Failure #7.** Recall for this question: miss.

**Config B — recursive-character, 800 chars, 120 overlap.** Better on prose, but the manual's tables are large and the recursive splitter still cuts inside the timeouts table, and the 120-char overlap isn't enough to replay the whole table. Marginal improvement, still fragile on the tabular question.

**Config C — structure-aware (Markdown headers) → recursive within section.** Splits on `###`, so the entire `### Timeouts` section — including the intact pipe-table — is one chunk, carrying metadata `heading_path: "Connection Settings > Timeouts"`. The table survives whole; `connect() | 30 seconds` is one row inside one chunk. Top-1 hit. **Recall: hit.**

**Config D — but oversized.** Suppose the `### Timeouts` section is 3,200 chars ≈ 800 tokens. Config C looks great in the text, *but* `bge-small` truncates at ~512 tokens (~2048 chars). If the `connect()` row sits at char 2,600 — past the cutoff — its embedding **doesn't encode that row at all.** The chunk text has the answer; the vector doesn't. Retrieval misses, and you'd swear the chunk "contains it." **Phantom recall loss.** Fix: split the section further so each chunk stays under ~400 tokens, or use a token-aware splitter.

**Config E — contextual chunking on top of C.** Prepend "This section is from the ACME Router v4.2 manual, Connection Settings, covering default timeouts." Now even the query's `v4.2 router` phrasing matches strongly, and a bare chunk that only said "the default is 30 seconds" becomes findable. Highest recall — at the cost of one (cached) LLM call per chunk at index time.

The lesson: the *same table* is a miss under fixed-size, a fragile maybe under naive recursive, a clean hit under structure-aware — *unless* you trip the token ceiling, in which case even the "good" strategy silently fails.

## How it shows up in production

- **The default you never questioned is truncating your corpus.** A team ships `chunk_size=1000` (a common tutorial value) with a 512-token embedder and wonders why recall plateaus. Half of every chunk is unembedded. The tell: recall is mediocre *and unmovable* by online tuning. Check the token count of your chunks against the embedder limit *first*.
- **Reindex cost gates experimentation.** Every chunking change is an offline reindex (Lecture 1): re-chunk, re-embed everything, rebuild the index. So you can't casually "try semantic chunking in prod." You iterate on a **golden set offline**, pick a winner by recall@k, *then* pay for the full reindex once. This is why the interface abstraction below matters — you want to swap strategies by a config string, not a rewrite.
- **Contextual chunking's bill is real but cache-shaped.** One LLM call per chunk sounds terrifying at 100k chunks. With prompt caching (shared document in the cached prefix) the marginal cost per chunk drops sharply, but you still pay it once per chunk *and again on every reindex.* Budget it; don't discover it.
- **Overlap silently inflates your index.** 20% overlap ≈ 20% more vectors, storage, and ANN memory. On a 10M-chunk corpus that's 2M extra vectors. Usually worth it for the #7 insurance — but it's a line item, not free.
- **Structure-aware chunking is only as good as last week's parser.** If parsing degraded and headings vanished, your structure-aware splitter quietly falls back to giant chunks and recall drops — with no code change on your side. Chunking quality is *downstream* of parsing quality; regressions propagate.
- **Provenance must be attached at chunk time or it's gone.** Every chunk needs `{source, page, heading_path, char_span, chunker}` metadata *when you cut it*. Miss it here and you can't cite, can't debug "where did this come from," and can't score recall against a gold chunk. Bolt it on later and you're reindexing.

## Common misconceptions & failure modes

- **"Bigger chunks give the LLM more context, so they're better."** Backwards for *retrieval*. Bigger chunks dilute the embedding and *lower* the odds the right chunk ranks in top-k (#2). If you want big context AND precise retrieval, that's exactly what parent-document/sentence-window buy you — small to find, big to read. Don't conflate "context the LLM reads" with "unit you embed."
- **"`chunk_size` is tokens."** It is **characters** in the character splitters. ~4 chars/token. Oversized chunks get silently truncated at the embedder's max sequence length → phantom recall loss. The number one first-build bug.
- **"Semantic chunking is the smart choice, so use it by default."** It needs an embedder pass *just to find boundaries*, doubling index-time embedding work, and often barely beats recursive-char. Reach for it only when your docs have real unmarked topic shifts. Default is still recursive-character.
- **"Overlap fixes fragmentation, so more overlap is safer."** Overlap helps at boundaries but inflates index size linearly and never fixes a *too-big* chunk's dilution. 10–20% is the band; 50% is waste.
- **"Chunking is a one-time setup detail."** It's the highest-leverage recall knob in the whole system and the thing you should A/B *first*, before rerankers and prompts — because it sets the ceiling those can't exceed.
- **"Structure-aware is always best."** Only when structure *exists and survived parsing*, and even then a huge section still needs a second-level recursive split to respect the token ceiling.

## Rules of thumb / cheat sheet

- **Default:** `RecursiveCharacterTextSplitter`, ~200–400 tokens, 10–20% overlap. Change it only with evidence from a golden set.
- **The law:** chunking sets the recall ceiling. What a boundary splits can't co-retrieve; what a chunk dilutes won't rank. Fix chunking *before* rerankers/prompts.
- **Size tradeoff:** too small → fragments context (#7); too big → dilutes embedding, kills precision (#2). Aim for one coherent topic per chunk.
- **CHARS ≠ TOKENS:** ~4 chars/token. `800 chars ≈ 200 tokens`. Keep chunks safely under the embedder's max seq length (`bge-small` ≈ 512 tokens) or they truncate silently → **phantom recall loss.** Prefer a token-aware splitter.
- **Have headings that survived parsing?** Structure-aware first, recursive-char *within* oversized sections.
- **Need precision + context?** Parent-document (small-to-big) or sentence-window: embed small, return big.
- **Chunks full of "this/the system/that version"?** Contextual chunking — but budget one (cache-friendly) LLM call per chunk per reindex.
- **Always attach provenance at cut time:** `{source, page, heading_path, char_span, chunker}`. You cannot add it later without reindexing.
- **Every strategy behind one interface**, swappable by config name, so you grid-search chunkers on the golden set instead of rewriting.

### One interface, many strategies

The engineering pattern that makes the grid-search cheap: a single `chunk(text, meta) -> list[Chunk]` contract, dispatched by config name, with every strategy attaching provenance the same way.

```python
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    meta: dict = field(default_factory=dict)   # source, page, heading_path, char_span, chunker

def make_chunker(name: str, **cfg):
    if name == "fixed":       return FixedChunker(**cfg)
    if name == "recursive":   return RecursiveChunker(**cfg)     # RecursiveCharacterTextSplitter
    if name == "structure":   return StructureChunker(**cfg)     # MarkdownHeaderTextSplitter
    if name == "sentence_window": return SentenceWindowChunker(**cfg)
    if name == "parent":      return ParentDocChunker(**cfg)
    raise ValueError(name)

# every chunker returns List[Chunk] with identical provenance keys,
# so index.py / evaluate.py never know or care which strategy ran.
chunker = make_chunker(CONFIG.chunker, chunk_size=800, chunk_overlap=120)
chunks  = chunker.chunk(doc_text, base_meta={"source": path, "parser": "docling"})
```

The payoff: `evaluate.py` runs `{parser} × {chunker} × {k}` as a config grid, and swapping `structure` for `parent` is a one-line change, not a refactor.

## Connect to the lab

This week's lab (`src/chunk.py`) is exactly this interface: **three chunkers behind one contract**, selectable by name, each attaching `{source, page/heading, char_span}` provenance. You'll run the grid `{pymupdf, docling} × {fixed, recursive, structure} × k∈{3,5,10}` and watch recall@k move — that movement *is* the ceiling this lecture describes, made visible. Set your `chunk_size` in tokens (or convert with the 4x rule) and sanity-check that no chunk exceeds `bge-small`'s 512-token limit, or your "good" configs will post phantom-low recall for reasons no log explains.

## Going deeper (optional)

- **LangChain docs — "Text splitters" / "Recursively split by character" / "ParentDocumentRetriever"** at `python.langchain.com`. The canonical recursive-char and small-to-big APIs; note `from_tiktoken_encoder` for token-based sizing.
- **LlamaIndex docs — "Node Parsers", "SemanticSplitterNodeParser", "SentenceWindowNodeParser", "AutoMergingRetriever"** at `docs.llamaindex.ai`. Semantic, sentence-window, and auto-merging strategies.
- **Anthropic, "Introducing Contextual Retrieval" (2024)** — search that exact title on `anthropic.com`. The contextual-chunking method, its reported recall gains, and the prompt-caching cost structure. Read the "cost" section carefully.
- **BAAI FlagEmbedding / `BAAI/bge-small-en-v1.5` model card** on Hugging Face (`huggingface.co`) — confirm the max sequence length (~512 tokens) and normalization requirement for whatever embedder you actually deploy; the number changes per model.
- **Barnett et al. (2024), "Seven Failure Points When Engineering a RAG System"** — search that title; re-map #2 and #7 to the size tradeoff in this lecture.
- Search queries when you hit friction: "chunk size vs recall RAG benchmark", "tiktoken token count chunk size", "sentence window vs parent document retriever", "semantic chunking percentile threshold".

## Check yourself

1. State, in one sentence, why chunking caps recall in a way that no reranker or prompt change can lift. Trace it through the two pipelines.
2. You set `chunk_size=4000` with `bge-small-en-v1.5` and recall is mediocre and refuses to improve no matter how you tune k or the prompt. What's the most likely cause, why is there no error, and what's the fix?
3. Your corpus is clean Markdown from Docling with good headings, but individual sections range from 100 to 1,500 tokens. Describe the two-level chunking strategy you'd use and why one level isn't enough.
4. Explain the mechanism by which a *too-big* chunk and a *too-small* chunk fail — and name the failure point (from the 7) each triggers.
5. When would you pay for contextual chunking over plain structure-aware, and what exactly are you paying (per what, how often, and what makes it affordable)?
6. Parent-document/small-to-big "decouples the retrieval unit from the context unit." Explain what that means mechanically and which half of the size tradeoff each unit optimizes.

### Answer key

1. Chunk boundaries are frozen at index time in the offline pipeline, and the online pipeline can only ever retrieve, rerank, and prompt over the chunks that already exist. If a boundary split a fact apart (or a big chunk diluted it below top-k), the online stages operate downstream of that loss and cannot reconstruct or re-rank information that isn't represented in a retrievable vector. So chunking is a hard ceiling; lifting it requires re-chunking and reindexing.
2. 4000 chars ≈ 1000 tokens, but `bge-small` truncates input at ~512 tokens (~2048 chars) and embeds only the prefix — silently, with no error, because truncation is normal model behavior. The back half of every chunk is absent from its embedding (phantom recall loss), so any answer living there is unsearchable regardless of k or prompt. Fix: use a token-aware splitter (or the 4x char rule) and keep chunks under ~400 tokens for this model, leaving headroom for overlap/context prefixes.
3. Two-level: first split **structure-aware** on `###` headings so each chunk is a human-authored coherent section with a `heading_path`; then run **recursive-character within** any section that exceeds the token budget (e.g. the 1,500-token ones), so no chunk trips the 512-token embedder ceiling. One level isn't enough because structure-aware alone leaves oversized sections that get truncated (phantom recall loss), and recursive-alone throws away the free, high-quality boundaries the headings already gave you.
4. **Too big:** one vector must summarize many topics, so the embedding becomes an average pointing nowhere specific; it matches every query weakly and the answer-bearing chunk ranks below k → **#2 missed top-k**. **Too small:** a fact and its necessary context land in separate chunks that don't co-retrieve, so only part of the answer surfaces → **#7 incomplete**.
5. Pay for contextual chunking when chunks are full of unresolved references ("this version," "the system," bare numbers) so that, embedded alone, they don't match queries naming the entity/product. You pay **one LLM call per chunk**, incurred **at index time and again on every reindex.** It's affordable because it's designed for **prompt caching**: the shared full document sits in the cached prefix, so you pay full price for that context once and cached (much cheaper) price for each chunk's incremental call.
6. Mechanically, you embed **small children** (e.g. single sentences or 200–400-char pieces) into the vector index for matching, but store a pointer to a **larger parent** (paragraph/section) in a docstore; on a child hit you return the parent to the LLM. The small child optimizes the **precision** side (a sharp, specific vector that ranks well — fights #2), and the large parent optimizes the **completeness** side (full surrounding context so the answer isn't fragmented — fights #7). One knob for each half of the tradeoff.
