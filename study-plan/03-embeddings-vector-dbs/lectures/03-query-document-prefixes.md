# Lecture 3: Query/Document Asymmetry and Instruction Prefixes

> There is a single bug that quietly costs more beginner RAG systems their retrieval quality than any other, and it is not a bug in your code's logic — it's a bug in what you *fed the model*. You picked a good model (Lecture 2), your vectors normalize, your index returns results, `pytest` is green, and yet recall is mysteriously mediocre. Then you learn that E5- and BGE-family models were **trained with instruction prefixes** — `"query: "`, `"passage: "`, a query-side instruction — and that omitting them silently drops several recall points with no error and no warning. This lecture makes that failure impossible for you again. After it you can state each model family's exact prefix convention, explain *why* the asymmetry exists without touching heavy math, build an `encode(kind=...)` abstraction that makes it structurally impossible to forget a prefix on one code path, run a prefix-ablation experiment that quantifies the recall delta on your own data, and recognize all four ways prefixing goes wrong.

**Prerequisites:** Lecture 1 (embeddings, cosine/dot, normalization, one shared space per model) · Lecture 2 (model catalog: E5, BGE, all-MiniLM, OpenAI) · comfortable Python · **Reading time:** ~24 min · **Part of:** Phase 3 Week 1

---

## The core idea (plain language)

Some embedding models were trained to treat a **query** and a **document** as two different *kinds* of text, and they expect you to tell them which is which. You do that by prepending a short string — a **prefix** or **instruction** — before you encode.

E5 wants you to encode a search query as `"query: how do I reset my password"` and a document as `"passage: To reset your password, go to Settings…"`. BGE wants a longer instruction glued to the *query only* (`"Represent this sentence for searching relevant passages: how do I reset my password"`) and the raw document with **nothing** prepended. And a third group of models — `all-MiniLM`, OpenAI's `text-embedding-3-*` — want **no prefix at all**; adding one to them makes things slightly worse.

Two facts do all the damage:

1. **It's silent.** Feeding E5 a query without its `"query: "` prefix does not raise. It returns a perfectly shaped, normalized vector. That vector just sits in a slightly wrong spot, so the right documents rank a little lower, and your recall@10 quietly drops 2–5 points. Nothing in your logs says why.
2. **It's per-family *and* per-role.** The scheme depends on which model family you're using (E5 vs BGE vs none) *and* whether the text is a query or a document. Get the family wrong or the role wrong and you've mislabeled your vectors.

The engineering takeaway is a discipline: **encode your entire corpus once, tagged as documents; encode each incoming query tagged as a query; and make the "which prefix?" decision in exactly one function so no code path can forget it.** The rest of this lecture is that discipline, from first principles.

---

## How it actually works (mechanism, from first principles)

### Why an embedding model would care whether text is a query or a document

Recall Lecture 1's central fact: a model only "knows" the notion of similarity it was *trained* to produce. So to understand prefixes, look at how retrieval models are trained.

These models are trained on **pairs**: a short query and the longer passage that answers it — millions of them, harvested from search logs, Q&A sites, and question-answering datasets. The training objective ("contrastive learning") pushes the query vector and its correct passage vector **close together**, and pushes the query away from *wrong* passages. The model's whole job becomes: *put a question near its answer.*

Here's the subtlety. A question and its answer usually **do not look alike**. "How do I reset my password?" and "Navigate to Settings › Security and click Reset Credentials" share almost no words. A naive similarity model would score them low. So the retrieval objective is teaching the model something harder than "similar text → close vectors" — it's teaching "**a query and the passage that answers it → close vectors**," which is an *asymmetric* relationship. The query is short and interrogative; the passage is longer and declarative. They play different roles.

To help the model keep those roles straight, the training data tags each side with a marker — `"query: "` on one side, `"passage: "` on the other, or an instruction sentence on the query. During training the model learns that text after `"query: "` should be embedded *the way questions get embedded* (projected toward where their answers live), and text after `"passage: "` should be embedded *the way answers get embedded*.

```
  Training pairs (E5-style), pushed together in vector space:

    "query: how do I reset my password"   ●───┐
                                               │  pulled close
    "passage: To reset your password, go  ●───┘
     to Settings and click Reset..."

    "query: how do I reset my password"   ●───┐
                                               │  pushed apart
    "passage: Our office hours are 9 to 5" ●╌╌╌┘
```

So the prefix is not decoration. It is a **switch that selects which of the two learned behaviors the model applies.** At inference time, if you omit `"query: "`, you're running the model in a mode it was rarely trained in for that text — an out-of-distribution input — and its output vector lands in a slightly wrong region. Slightly wrong, times every query, equals measurable recall loss. No deep math required: you asked the model to do something a little different from what it practiced, and it's a little worse at it.

### The exact conventions — per family, per role

This is the part to memorize. The three regimes you'll meet constantly in 2025–26:

**E5 family** (`intfloat/e5-*`, `intfloat/multilingual-e5-*`) — **short symmetric-looking prefixes, one per role, on both sides:**

```
query    → "query: "     + text
document → "passage: "   + text
```

Both a query and a document get a prefix; they're just *different* prefixes. Note the exact spelling: lowercase, a colon, and **a trailing space**. `"query:hello"` is not the same tokenization as `"query: hello"`.

**BGE family** (`BAAI/bge-*-en-v1.5`, etc.) — **an instruction on the query only, nothing on documents:**

```
query    → "Represent this sentence for searching relevant passages: " + text
document → text            (no prefix)
```

BGE's asymmetry is starker: the query carries a whole instruction sentence, and the corpus is embedded raw. (Historical note worth knowing: for **short-text English** BGE-v1.5, the authors have said the query instruction gives a small gain and can even be skipped; for anything retrieval-serious, use it. Multilingual/M3 variants have their own guidance — check the card.) The important structural point is *documents get nothing.*

**No-prefix family** (`sentence-transformers/all-MiniLM-L6-v2`, most classic `all-*` models, OpenAI `text-embedding-3-small/large`, many others) — **encode both sides raw:**

```
query    → text
document → text
```

These were trained differently (often on symmetric similarity, or the role handling is internal to the API), so adding an E5/BGE prefix is *wrong* here — you'd be embedding the literal word "query" into every vector. OpenAI handles any query/document asymmetry inside its own service; you send plain text.

```
 ┌───────────┬──────────────────────────────────┬─────────────────────┐
 │ family    │ query side                        │ document side       │
 ├───────────┼──────────────────────────────────┼─────────────────────┤
 │ E5        │ "query: " + text                  │ "passage: " + text  │
 │ BGE       │ "Represent this sentence for      │ text  (raw)         │
 │           │  searching relevant passages: "   │                     │
 │           │  + text                           │                     │
 │ MiniLM /  │ text  (raw)                       │ text  (raw)         │
 │ OpenAI    │                                   │                     │
 └───────────┴──────────────────────────────────┴─────────────────────┘
   Always confirm on the model card — conventions vary by version.
```

The one rule that generalizes across all of them: **the corpus is encoded once in its document role; each query is encoded in its query role; the two roles use whatever this specific model family's card prescribes.** Never assume — the exact string is on the model card, and it matters down to the trailing space.

### The abstraction that makes forgetting impossible

The reason beginners get bitten is architectural: they call the raw model in two different places — a batch ingest script for the corpus, and a request handler for queries — and the prefix logic gets duplicated, drifts, or is simply forgotten on one of them. The fix is to funnel *all* encoding through a single function that takes the **role** as an argument and owns the prefix decision:

```python
# emb/encode.py
_PREFIX = {
    # family        (query_prefix,                                              doc_prefix)
    "e5":    ("query: ",                                                   "passage: "),
    "bge":   ("Represent this sentence for searching relevant passages: ", ""),
    "none":  ("",                                                          ""),
}

def _family(model_name: str) -> str:
    n = model_name.lower()
    if "e5" in n:  return "e5"
    if "bge" in n: return "bge"   # note: bge-m3 has its own rules — special-case if you use it
    return "none"

def encode(texts, model_name, kind, model=None, normalize=True, batch_size=64):
    assert kind in ("doc", "query"), f"kind must be 'doc' or 'query', got {kind!r}"
    q_pre, d_pre = _PREFIX[_family(model_name)]
    prefix = q_pre if kind == "query" else d_pre
    prepared = [prefix + t for t in texts]
    vecs = model.encode(prepared, batch_size=batch_size, normalize_embeddings=normalize)
    return vecs
```

The whole point is the `kind` parameter. Ingest calls `encode(docs, m, kind="doc")`; the query path calls `encode([q], m, kind="query")`. Neither caller ever types a prefix string. Swap the model and the family dispatch handles it — the harness *cannot forget the prefix on one path*, because there's only one path. This is the same `encode()` the Week 1 lab spec asks for, and it's why the spec put `kind` in the signature. (Special-case oddballs like `bge-m3` explicitly rather than letting the substring matcher guess.)

---

## Worked example

You're building search over a 20k-doc knowledge base with `intfloat/multilingual-e5-large`. You have an 80-query held-out eval with known relevant IDs (from Lecture 2's method). You run flat cosine search — exact, no ANN — so any recall difference is purely the model input, not index error. **The numbers below are illustrative of the *shape* of the effect, not measured results; E5's own paper and card describe prefixes as required, and community ablations typically show a few points.**

**Run A — no prefixes (the bug).** Corpus and queries both encoded raw:

```
recall@1  0.48   recall@5  0.69   recall@10  0.76   MRR@10  0.57
```

Looks like a middling model. A beginner concludes "e5-large isn't great on our data" and starts shopping for a different model — chasing the wrong variable.

**Run B — correct prefixes.** Same model, same vectors math, same eval — the *only* change is `"query: "` on queries and `"passage: "` on documents:

```
recall@1  0.55   recall@5  0.77   recall@10  0.83   MRR@10  0.65
```

recall@10 jumps from 0.76 to 0.83 — **+7 points** — and MRR@10 from 0.57 to 0.65. Nothing about the model changed; you just ran it in the mode it was trained for. That delta is the entire prefix-ablation experiment, and it's why the Week 1 Definition of Done makes you state the number.

Now the instructive mistake. **Run C — prefixes swapped:** documents get `"query: "`, queries get `"passage: "`. Every vector is still shaped and normalized correctly, search still runs, no error — and recall is *worse than Run A*, because now every vector is confidently in the wrong role. This is the tell that prefixing failures are silent and directional: the pipeline's health checks all pass while retrieval quietly degrades.

The reasoning to carry forward: a "bad model" result should trigger *"did I feed it the way it was trained?"* before *"is there a better model?"* One is an afternoon-saving one-line fix; the other is a re-benchmarking rabbit hole.

---

## How it shows up in production

**The "we tried E5 and it was mediocre" false negative.** A team ablates models, forgets prefixes for the E5/BGE candidates (because their previous model was all-MiniLM, which needs none), and concludes the prefix-based models are weak. They ship a worse model and never learn the real ranking. The prefix bug doesn't just hurt one system — it corrupts your *model-selection decision* from Lecture 2, because you compared a correctly-run model against a crippled one.

**The mixed-index catastrophe.** The worst production form: you ingest half your corpus through an old code path (no prefix) and half through a new one (with `"passage: "`), or you re-embed some documents after a refactor changed the prefix. Now a single index contains vectors from two different input regimes. Cosine distances between a query and these vectors are **no longer comparable** — some docs are "close" only because they were embedded in the same mode as others, not because they're relevant. There's no error; ranking just becomes partly random, and it's maddening to debug because *most* results look plausible. **One index must contain vectors produced by exactly one (model, prefix-scheme) combination.** This is the same principle behind Week 3's blue-green re-embedding rule: never mix vector spaces in one collection.

**The copy-paste-across-families leak.** An engineer copies a working E5 pipeline to a new service that uses `all-MiniLM` or OpenAI, and carries the `"query: "` / `"passage: "` prefixing along. Now every MiniLM vector literally encodes the token "query" or "passage," polluting the space. Recall sags a couple points and nobody connects it to the copied prefix logic. The `_family()` dispatch above prevents this — the prefix is a function of the model name, not of which file you copied.

**The trailing-space and casing gotcha.** `"query:"` vs `"query: "` vs `"Query: "` tokenize differently, and the model saw exactly one form in training. A stray missing space is a small but real recall leak that passes every test. Copy the prefix string verbatim from the model card; don't retype it from memory.

**Caching interacts with prefixes.** If your on-disk cache (Week 1) keys on the raw text, a query and a document with identical text would collide even though they must produce different vectors under E5. Key the cache on the **prepared** string (post-prefix) or include `kind` in the key, so `"query: X"` and `"passage: X"` are distinct cache entries.

---

## Common misconceptions & failure modes

- **"Skipping prefixes just throws an error, so I'd know."** No — it's completely silent. You get a valid, normalized vector in a slightly wrong place. The only symptom is a few points of recall you never measured. This silence is the whole reason the bug is so common.
- **"Prefixes are optional / a micro-optimization."** For E5 and BGE they're part of how the model was trained; several recall points is not a micro-optimization for a retrieval system. Treat them as mandatory unless the card says otherwise.
- **"One prefix scheme works for all models."** Prefixing is **per-family**. E5's `query:/passage:`, BGE's query-only instruction, and MiniLM/OpenAI's *no prefix* are mutually incompatible. Applying one family's scheme to another is a bug, not a default.
- **"Documents need a query prefix too."** Role matters. Under E5 documents get `"passage: "`; under BGE they get **nothing**. Prefixing the corpus with the *query* prefix is a classic failure that mislabels every document's role.
- **"I'll add the prefix in the query handler and I'm done."** Then your corpus was likely ingested without the document prefix (or with the wrong one), and now the two sides don't match. Both sides must be handled by the same `encode(kind=...)` abstraction, encoded under the same scheme.
- **"Mixing a few unprefixed vectors into the index is harmless."** It quietly makes distances non-comparable for the affected vectors. One index = one (model, prefix-scheme). Re-embed the whole corpus, don't patch part of it.
- **"bge-m3 uses the same instruction as bge-v1.5."** Don't assume across versions. `bge-m3` and newer variants have their own guidance (often *no* query instruction for dense retrieval). Read the specific card and special-case it in `_family()`.

---

## Rules of thumb / cheat sheet

- **E5:** `"query: "` on queries, `"passage: "` on documents. Both sides prefixed, exact spelling incl. trailing space.
- **BGE (v1.5):** `"Represent this sentence for searching relevant passages: "` on the **query only**; documents raw. (Newer/M3 variants differ — check the card.)
- **all-MiniLM, other classic `all-*`, OpenAI `text-embedding-3-*`:** **no prefix** on either side.
- **The one rule that always holds:** encode the corpus once in its **document** role; encode each query in its **query** role; use the *exact* strings from this model's card.
- **Funnel all encoding through one `encode(texts, model, kind="doc"|"query")`** that owns the prefix decision via family dispatch. No caller ever types a prefix string. This kills "forgot it on one path."
- **One index = one (model, prefix-scheme).** Never mix prefixed and unprefixed (or two-model) vectors in a single index — distances stop meaning anything.
- **When a prefix-based model looks "mediocre," check the prefix before you swap models.** It's a one-line fix vs a re-benchmarking detour.
- **Cache key must include the role/prefix** (key on the prepared string), so `"query: X"` and `"passage: X"` don't collide.
- **Copy prefix strings verbatim from the model card.** Trailing space, lowercase, colon all matter to tokenization.
- **Prefix-ablation is a required experiment, not a curiosity:** run E5 with and without prefixes on the same eval, report the recall@10 delta (Week 1 DoD).

---

## Connect to the lab

This lecture is the "why" behind **Week 1 Lab steps 3 and 6**. Step 3 builds exactly the `encode(texts, model_name, kind="doc"|"query", ...)` function above — the `kind` argument exists so the prefix decision lives in one choke-point and can't be forgotten. Step 6 is the **prefix ablation**: run `intfloat/e5-small-v2` with and without the `query:`/`passage:` prefixes on the same eval set and record the recall delta — that's your Run A vs Run B, and the Definition of Done makes you *state the number*. As you build it, key your `diskcache` layer on the prepared (post-prefix) string so query and document vectors for identical text don't collide, and remember `all-MiniLM-L6-v2` in the same bench must get **no** prefix — the family dispatch is what keeps that straight.

---

## Going deeper (optional)

- **E5 model cards & paper** — `huggingface.co/intfloat/e5-large-v2` and `intfloat/multilingual-e5-large` state the `query:`/`passage:` convention explicitly. For the training story, search: *"Text Embeddings by Weakly-Supervised Contrastive Pre-training E5 paper"*.
- **BGE model cards & FlagEmbedding repo** — `huggingface.co/BAAI/bge-large-en-v1.5` (query instruction guidance) and the canonical repo (search: *"FlagEmbedding BAAI GitHub"*) which documents per-model instruction usage and the `bge-m3` differences.
- **sentence-transformers documentation** (`sbert.net`) — *Computing Embeddings* and the model-specific notes; many prefix-based models carry a `prompts`/`prompt_name` config so ST can apply the prefix for you. Search: *"sentence-transformers prompts prompt_name"*.
- **OpenAI Embeddings guide** (`platform.openai.com/docs`) — confirms plain-text input (no manual prefixes) and how it handles input types.
- **Contrastive learning intuition** (no heavy math) — search: *"contrastive learning query passage dual encoder retrieval explained"* for the pairs-trained-together picture behind the asymmetry.
- **MTEB / model cards** — when shortlisting (Lecture 2), each candidate's card is the authoritative source for its prefix scheme; make reading it part of the selection afternoon.

---

## Check yourself

1. You encode your corpus and queries with `intfloat/e5-large-v2` but skip all prefixes. There's no error and vectors look normal. What has gone wrong, and what symptom will you see?
2. State the exact query-side and document-side treatment for each of: E5, BGE-v1.5, all-MiniLM-L6-v2. Which family prefixes *documents*, and which prefixes *only queries*?
3. Explain, without heavy math, *why* E5 needs different prefixes for queries and passages. What is different about how these models were trained?
4. Why does putting a `kind="doc"|"query"` argument on a single `encode()` function prevent an entire class of production bugs that duplicated encode logic invites?
5. You re-embedded 30% of your corpus after a refactor accidentally dropped the `"passage: "` prefix, leaving 70% correctly prefixed in the *same* index. Ranking is now subtly broken with no errors. Explain precisely why, and state the fix.
6. A colleague copies your working E5 service to a new project that uses OpenAI `text-embedding-3-small`, keeping the `"query: "`/`"passage: "` logic. Recall drops ~2 points. What's the bug, and what design would have prevented it?

### Answer key

1. E5 was trained with `"query: "` / `"passage: "` prefixes; without them you're running the model out-of-distribution, so every vector lands in a slightly wrong region. The failure is **silent** — valid, normalized vectors — and the only symptom is depressed recall/MRR (typically a few points) that you'll misread as "e5 is a mediocre model."
2. **E5:** query → `"query: " + text`, document → `"passage: " + text` (both prefixed). **BGE-v1.5:** query → `"Represent this sentence for searching relevant passages: " + text`, document → raw (query only). **all-MiniLM-L6-v2:** no prefix on either side. E5 prefixes documents; BGE prefixes only queries.
3. Retrieval models are trained contrastively on **query–passage pairs**, pulling a question and its answer close even though they don't look alike (short/interrogative vs long/declarative — an *asymmetric* relationship). The prefixes tag each side's role during training, so the model learns two behaviors and the prefix at inference selects the right one. Omit it and you invoke the wrong (untrained) mode for that text.
4. Duplicated encode logic (a batch ingest path and a query path) lets the prefix drift or be forgotten on one side, producing a corpus and queries embedded under mismatched schemes. A single `encode(kind=...)` funnel owns the prefix decision in one place, so there is only one code path to get right — the harness *structurally cannot* forget the prefix on one route.
5. The index now holds vectors from two different input regimes (prefixed vs unprefixed) of the same model. Cosine distances across regimes are **no longer comparable**, so relevance ranking becomes partly meaningless for the affected 30% — with no error, since all vectors are well-formed. Fix: re-embed the **entire** corpus under one consistent scheme (one index = one model + one prefix-scheme); don't patch a subset.
6. OpenAI models take **plain text and use no manual prefixes**; the copied code embeds the literal tokens "query"/"passage" into every vector, polluting the space and shaving recall. Prevention: make the prefix a function of the model (family dispatch in `encode()`), not something carried along by whichever file was copied — swapping the model name then automatically selects "no prefix."
