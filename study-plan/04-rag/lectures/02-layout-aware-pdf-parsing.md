# Lecture 2: Layout-Aware Parsing — Why PDFs Fight Back

> Every RAG system you will ever build starts with a `load → parse` step that looks like a one-liner and is actually where half your recall is silently lost. A PDF is a *drawing* format, not a *document* format: it stores where to paint each glyph, not what the reading order or table structure is. Naive text extraction throws away exactly the structure your chunker needs to keep facts together. This lecture takes the Week-1 spine's "PDFs are the enemy" bullet and turns it into the real mental model. After it you can explain why a two-column page comes out interleaved, pick between PyMuPDF, Unstructured, Docling, and LlamaParse on purpose, keep tables as pipe-tables *inside* the chunk so tabular questions stay answerable, and design a provenance schema that lets you cite, debug, and evaluate every chunk you ever emit.

**Prerequisites:** Lecture 1 (the two pipelines + 7 failure points), embeddings & chunking intuition (Phase 0), basic Python · **Reading time:** ~26 min · **Part of:** Phase 4 (RAG) Week 1

---

## The core idea (plain language)

Open a Word document and the file literally contains "this is a Heading 1, then a paragraph, then a 3-column table." The structure is *in the bytes*. A PDF contains almost none of that. The PDF spec (ISO 32000) is a **page-description language**: its job is to tell a printer or a screen *"draw the glyph 'A' at coordinates (72, 400) in Helvetica 11pt; draw a line from (100,380) to (500,380)."* It is a set of drawing instructions optimized for one thing — pixel-perfect reproduction on any device — and it succeeds brilliantly at that. It just was never designed to answer the question you actually have: *"what does this document say, in what order, and how is it organized?"*

So when your parser opens a PDF, there is no `.paragraphs` or `.tables` to read. There are only positioned glyphs and vector strokes. Everything a human eye reconstructs instantly — this is a heading, these numbers form a table, that gray strip at the bottom of every page is a footer, this page has two columns — has to be **inferred** from raw (x, y) coordinates. Different parsers infer it with wildly different sophistication, and the ones that guess badly hand your chunker a scrambled mess.

Here is the chain of consequences that makes this a *retrieval* problem, not a cosmetic one:

- Reading order wrong → sentences get spliced across columns → embeddings encode nonsense → the right chunk never ranks (failure point #2, missed top-k).
- Table flattened to prose → the row/column association that made "Region: EMEA, Q3 revenue: $4.2M" a single fact is gone → the question about EMEA Q3 revenue retrieves nothing useful (failure points #4 and #7).
- Headers/footers repeated into every chunk → the same "Confidential — Page 14" boilerplate pollutes hundreds of chunks → dilutes embeddings and wastes your token budget.

You cannot prompt-engineer your way out of any of these later. Whatever you embed wrong offline is *permanently* wrong online until you reindex. That is why parsing is the first thing you get right, and why it deserves a full lecture.

---

## How it actually works (mechanism, from first principles)

### A PDF page is a bag of positioned glyphs

Strip a PDF down and each page is a content stream of operators. Simplified, a line of text is stored as something like:

```
BT                      % begin text
/F1 11 Tf               % font F1, size 11
72 400 Td               % move cursor to (x=72, y=400)
(Revenue grew 12%) Tj   % paint this string of glyphs here
ET                      % end text
```

Crucial detail: **the order of these instructions is drawing order, not reading order.** A PDF generator (LaTeX, a browser's print engine, InDesign, a scanner's OCR layer) is free to emit glyphs in whatever order was convenient when laying out the page. It might paint all the body text, then go back and paint the sidebar, then the page number. The (x, y) coordinates are ground truth; the *sequence* is not. This single fact is the root cause of most PDF pain.

There is also frequently **no explicit space character.** The renderer positions the glyph for "R" then positions "e" a few points to the right; the visual gap *is* the space. A dumb extractor that just concatenates glyph strings gives you `Revenuegrew12%`. Extractors reconstruct spaces by measuring inter-glyph gaps against the font's space width — a heuristic that fails on justified text, tight kerning, and ligatures.

### Why multi-column pages interleave

Consider a two-column page. Human reading order is: all of column 1 top-to-bottom, *then* all of column 2. But the y-coordinates don't know about columns. A naive extractor that sorts glyphs top-to-bottom, then left-to-right (the obvious thing) produces this:

```
PAGE LAYOUT                     NAIVE TOP-TO-BOTTOM, LEFT-TO-RIGHT READ
┌───────────┬───────────┐       line 1 col1 ... line 1 col2
│ The quick │ jumps over│       line 2 col1 ... line 2 col2
│ brown fox │ the lazy  │       →  "The quick jumps over brown fox
│ sat down  │ dog ran   │          the lazy sat down dog ran"
└───────────┴───────────┘       (every line spliced across the gutter!)
```

The output is grammatical garbage: each visual row zips the two columns together. Embeddings of that text are meaningless, and a chunk built from it will never match a real query. A **layout-aware** parser first *detects the column boundary* (a wide vertical whitespace gap, or a layout ML model), then reads each column as its own block. Same coordinates, completely different — and correct — reading order.

### Why tables are the hardest case

A table in a PDF is, at the byte level, two unrelated things drawn on top of each other: (1) a set of vector line strokes (the grid), and (2) a pile of floating text glyphs (the cell contents). **Nothing links a number to its row and column** except the fact that it happens to sit inside a rectangle formed by four lines. Worse, many "tables" have *no* drawn lines at all — they're held together purely by aligned whitespace columns.

To reconstruct the table an engine must: find the horizontal and vertical rulings (or infer column boundaries from text alignment), compute the grid of cells, assign each floating text run to a cell by geometry, then decide the cell order. This is genuinely hard computer vision / geometry, which is exactly why table quality is *the* axis that separates the parsers below. When it fails, you get the numbers dumped as a flat run — `EMEA 4.2 APAC 3.1 AMER 5.8` — with the column headers three lines away and no way to know that 4.2 belongs to EMEA's Q3 revenue.

### Headers, footers, and the repetition tax

Running headers and footers are painted on *every* page at roughly the same y-coordinate. A parser with no layout awareness treats them as body text, so "Acme Corp — Internal — Q3 2025" and "Page 14 of 210" get concatenated into the middle of your extracted text on every page. When you chunk, that boilerplate leaks into hundreds of chunks. It's not just ugly: it's repeated tokens that dilute the semantic signal of each chunk's embedding and eat into the token budget you'll fight over in Week 3. Layout-aware parsers detect repeated page-margin elements and can drop or tag them.

### Digital vs scanned PDFs (the OCR fork)

There are two species of PDF and they need different tools:

- **Digital / born-digital** — generated by software; the glyphs and their codes are *in* the file. Text is directly extractable (the mechanisms above). Fast, no ML strictly required.
- **Scanned / image-only** — the "page" is a photo of paper; there are *no* text glyphs, just pixels. Extracting text requires **OCR** (optical character recognition) to turn pixels into characters first. Many real corpora are a mix, and some are "digital" but were produced by scanning + an OCR layer of dubious quality.

You can detect a scanned page trivially: extract text and if you get ~0 characters on a page that clearly has writing, it's an image — route it to OCR. Getting this fork wrong ("why is my 200-page manual returning 4 pages of text?") is a classic first-day bug.

---

## Worked example: one table page, four parsers, one query

Take a single page from a product manual: two columns of prose and one 3×4 pricing table. Here's the table as a human sees it:

```
Plan      Monthly   Annual   Seats
Starter   $10       $96      3
Team      $30       $288     10
Business  $80       $768     50
```

The gold question against it: **"What is the annual price of the Team plan?"** Gold answer substring: `$288`. Now trace what each parser hands the chunker.

**PyMuPDF (naive `get_text("text")`):** reads glyphs in draw order. Two-column prose interleaves; the table dumps as a flat token stream:

```
Plan Monthly Annual Seats Starter $10 $96 3 Team $30 $288 10 Business $80 $768 50
```

`$288` is present *as a substring*, so a lucky `gold_substr` check might even pass — but the embedding of that scrambled run barely resembles the query "annual price of the Team plan," and at retrieval time the chunk ranks poorly. Worse, the model has no way to know $288 is Team's *annual* figure and not Business's monthly. This is the phantom-recall trap: the string is technically there, the *meaning* is destroyed.

**PyMuPDF via `pymupdf4llm` (the markdown path):** `pymupdf4llm.to_markdown()` applies layout heuristics — it detects headings and, critically, emits simple tables as Markdown pipe-tables:

```markdown
| Plan     | Monthly | Annual | Seats |
|----------|---------|--------|-------|
| Team     | $30     | $288   | 10    |
```

Now "$288" sits in a row labeled `Team` under a column headed `Annual`. Both the embedding model and the downstream LLM read pipe-tables natively, so the chunk's vector actually encodes "Team / Annual / 288." Retrieval works. This is why `pymupdf4llm` — not raw `get_text` — is your baseline.

**Docling:** runs a layout model + a dedicated table-structure model, reconstructs reading order across columns, and exports the same table as clean Markdown *and* preserves the heading path (e.g. `Pricing > Plans`). On a page like this it matches or beats `pymupdf4llm`, and it pulls further ahead as tables get gnarlier (merged cells, no ruling lines).

**LlamaParse (hosted):** on a *scanned* or visually chaotic version of this page — merged header cells, rotated text, a screenshot-quality scan — the cloud service typically produces the cleanest table of all. But it's a paid API call over the network. You use it as a *gold reference* to see how far your local parsers fall short on the ugly 5% of pages, not as your default.

Same page, same coordinates, four very different chunks. Only three of the four keep `$288` retrievable *with its meaning intact*. That is the entire game.

---

## The four parsers engineers actually reach for

```
                fast/local   layout   table→md   OCR       deps      cost
PyMuPDF+4llm      ★★★★★      ★★★☆☆    ★★★☆☆     no        tiny      free/local
Unstructured     ★★★☆☆      ★★★★☆    ★★★★☆     yes(hi_res) heavy    free/local
Docling          ★★★★☆      ★★★★★    ★★★★★     yes        medium    free/local
LlamaParse       n/a(cloud) ★★★★★    ★★★★★     yes        none(API) paid
```
*(Star ratings are approximate, opinionated 2025-2026 field impressions — not benchmark numbers.)*

**PyMuPDF (`pymupdf` / `fitz`), via `pymupdf4llm`.** Fast, pure-Python, no ML models, tiny install. Excellent on text-heavy born-digital PDFs. Its `pymupdf4llm.to_markdown()` helper is the LLM-friendly path: heading detection + Markdown tables + optional per-page chunking (`page_chunks=True`), giving you page-numbered records for free. Weak on complex tables (merged cells, no rulings) and it doesn't OCR. **This is your speed/baseline parser.** When someone asks "what should I try first," it's this.

**Unstructured (`unstructured`).** Element-based: it partitions a document into typed elements — `Title`, `NarrativeText`, `Table`, `ListItem`, `Header`, `Footer` — each with rich metadata. Its `hi_res` strategy runs a layout-detection model plus OCR, so it handles scanned pages and gives you semantic element types you can chunk on directly. The cost is heavy dependencies (a layout model, often `detectron2`/ONNX, Tesseract or Paddle for OCR) and slower runs. Reach for it when you need OCR + element typing in one batteries-included package.

**Docling (IBM, `docling`).** The 2024/2025 standout for **layout + table-structure** reconstruction, fully local, no API key. It runs purpose-built models for page layout and for table structure, preserves **reading order** across columns, and exports clean **Markdown/JSON** with tables rendered as Markdown and heading hierarchy intact. Its table→markdown quality is its headline feature and the reason it's your layout-aware challenger. Install is heavier than PyMuPDF (it pulls models) but far lighter operationally than a cloud dependency — it runs on a laptop CPU.

**LlamaParse (LlamaIndex, hosted API).** Best-in-class on the *gnarly* stuff: scanned documents, financial statements with dense merged-cell tables, complex forms. It's a **paid cloud API** (a free tier exists for small volumes). Because it's hosted, your documents leave the machine — a non-starter for many corpora — and every page costs money and network latency. **Use it as a gold reference**: run your 3-5 ugliest pages through it to see the ceiling, then judge how close your free local parsers get. Do not build your default laptop pipeline on it.

**The opinionated default:** **PyMuPDF-via-`pymupdf4llm` as the baseline, Docling as the layout-aware challenger.** Both are free, local, offline, no API key. Diff them on your own table-heavy pages and let recall@k pick the winner. Keep Unstructured `hi_res` in your back pocket for OCR, and LlamaParse only to sanity-check the ugliest pages.

---

## The two non-negotiable rules

### Rule 1: tables become Markdown/HTML pipe-tables *inside* the chunk text

Never flatten a table to prose. The single load-bearing property of a table is the **association between a cell, its row, and its column** — "the Annual price *of the Team plan* is $288." Flatten it and that association evaporates; you're left with orphaned numbers. Both embedding models and LLMs were trained on enormous amounts of Markdown and HTML and read pipe-tables natively, so keep the structure literally in the chunk string:

```markdown
| Plan     | Monthly | Annual | Seats |
|----------|---------|--------|-------|
| Team     | $30     | $288   | 10    |
```

A chunk containing that renders "Team / Annual / 288" as a coherent local fact — its embedding sits near the query, and at generation time the LLM can read straight across the row. Flatten the same data to `Team 30 288 10` and you've manufactured failure points #4 (LLM can't extract the answer from noise) and #7 (the answer is fragmented across a scrambled run). HTML `<table>` is an acceptable alternative when you have merged cells (`rowspan`/`colspan`) that Markdown can't express. **Keep the table with its chunk, never as a separate stripped artifact.**

### Rule 2: design the provenance schema NOW

Every chunk must carry, from the moment it's created, enough metadata to answer *"where exactly did this come from?"* The minimum viable schema:

```
{source_file, page, section/heading, element_id, char_span}
```

- **`source_file`** — which document (you will have many).
- **`page`** — the page number, for citations a human can verify.
- **`section` / `heading`** (heading path, e.g. `Pricing > Plans`) — the structural location, and a cheap relevance signal.
- **`element_id`** — a stable id for the source element, so re-chunking doesn't scramble your references.
- **`char_span`** — `[start, end]` offsets into the source text, so you can highlight the exact span.

Why *now* and not later? Because without provenance you **cannot cite** (Week 3 forces every claim to resolve to a real chunk id + page — no provenance, no citation), you **cannot debug** ("this answer is wrong — where did it come from?" has no answer), and you **cannot evaluate** (Week 1's recall@k compares retrieved chunks against a *gold chunk id*; Week 4's context-recall keys relevance labels on stable ids). Bolting provenance on after you've indexed 200 pages means reparsing and reindexing everything. This is a five-minute schema decision that saves you three painful reworks.

One subtlety that bites in Week 4: **key your ids on stable content, not row indices.** If `element_id` is just "the 47th chunk," then changing your chunker renumbers everything and every gold label silently reads as a miss (recall@k → 0 for reasons that have nothing to do with retrieval). Derive ids from a content hash or a `source_file:page:char_span` tuple so they survive re-chunking.

### The normalized record `parse.py` should emit

Both parsers should converge on one shape, so the rest of the pipeline (chunk, index, evaluate) never has to care which parser produced a record:

```python
# One normalized record per structural element
{
    "text": "| Plan | Monthly | Annual | Seats |\n|---|---|---|---|\n| Team | $30 | $288 | 10 |",
    "element_type": "table",          # title | narrative | table | list_item | ...
    "source_file": "manual.pdf",
    "page": 42,
    "heading_path": ["Pricing", "Plans"],
    "element_id": "manual.pdf:p42:e3",  # STABLE across re-chunking
    "char_span": [1840, 1993],          # offsets into the page's text
    "parser": "docling",                # provenance of the provenance
}
```

Note `element_type` and `parser` on every record: the first lets a structure-aware chunker split on real boundaries (Week 1 lab step 3), and the second lets your evaluation grid attribute a recall win or loss to the parser that caused it. This record is the contract between `parse.py` and everything downstream.

---

## How it shows up in production

**Cost and latency live in the offline parse, and they diverge by 10-100×.** PyMuPDF rips through a 200-page digital PDF in *seconds* on a laptop CPU with no model downloads. Docling on the same document takes *minutes* because it runs layout + table models per page — still fine for a batch job, painful if you thought it was interactive. Unstructured `hi_res` with OCR is slower still. LlamaParse adds network round-trips and a per-page bill. Because this is the offline pipeline (Lecture 1), latency is usually tolerable — but when you're iterating on chunking and reparsing the corpus 20 times a day, a 3-minute parse becomes a 1-hour tax. Cache parsed output to disk keyed on file hash; never reparse an unchanged document.

**"My recall is bad" is a parser bug three times out of five.** Before you touch embeddings or the chunker, eyeball 2-3 table pages and 2-3 multi-column pages from each parser. If a table came out as a flat run or a two-column page interleaved, no downstream tuning will recover it. This is the cheapest debugging win in the whole phase and engineers skip it constantly because "parsing is done, moving on."

**Scanned pages silently vanish.** A born-digital-except-for-3-scanned-appendix-pages document parsed with PyMuPDF returns *nothing* for those pages — no error, just missing content (failure point #1, missing content, the worst kind because it's invisible). Detect empty-text pages and route them to an OCR-capable parser.

**Table quality is where the parser choice actually pays off.** On plain prose, all four parsers produce nearly identical text and the choice barely matters. The differences that move recall@k concentrate entirely on tables and complex layouts. So benchmark on the pages that have them, not on the easy pages that flatter everyone.

**Provenance is the difference between a demo and a system.** The moment a stakeholder asks "where did the model get that number?" you either have `manual.pdf, page 42, Pricing > Plans` to show them, or you have nothing and they stop trusting the system. Citation-enforced generation (Week 3) is *built on* the schema you commit to here.

---

## Common misconceptions & failure modes

- **"PyMuPDF returned text, so parsing worked."** Returning text and returning *correctly-ordered, structure-preserving* text are different things. A scrambled two-column page and a flattened table both "return text." Check the *structure*, not the presence of characters.
- **"The `$288` is in the chunk, so recall is fine."** A substring match can pass while the embedding is useless because the surrounding text is scrambled. Phantom recall: the eval's `gold_substr` check succeeds but real semantic retrieval fails. Validate on realistic query phrasings, not exact strings.
- **"Docling is strictly better, so always use it."** Docling wins on tables/layout but is much slower and heavier. On a text-only corpus PyMuPDF is faster and just as good. Match the tool to the document; benchmark, don't cargo-cult.
- **"I'll add provenance metadata later."** Later means reparsing and reindexing the whole corpus, and every gold label built in the meantime is invalid. It's a schema decision, not a feature — make it before the first chunk.
- **"LlamaParse is the best, I'll just use it."** It's a paid cloud API; your documents leave the machine and every page costs money and latency. Fine as a gold reference, wrong as a laptop-lab default.
- **"Flattening the table is fine, the LLM is smart."** The LLM can only reason over what it receives. A flattened table has already destroyed the row/column association *before* the LLM sees it — no amount of model intelligence reconstructs which number was in which cell.
- **"Chunk on characters, tables will be fine."** A blind character splitter will happily cut a Markdown table in half mid-row, leaving a header with no data in one chunk and orphaned rows in the next. Structure-aware chunking (split on `element_type`/headings) keeps tables whole.
- **"OCR'd text is as good as digital text."** OCR introduces character errors (`5`→`S`, `0`→`O`), especially on tables and small fonts. Treat OCR output as noisier and expect lower recall on exact-string questions.

---

## Rules of thumb / cheat sheet

- **Default stack:** `pymupdf4llm.to_markdown()` as baseline, **Docling** as the layout-aware challenger. Both free, local, offline. Diff them on *your* table pages.
- **Baseline path is `pymupdf4llm`, not raw `get_text("text")`** — the markdown path gives you headings + pipe-tables; raw text gives you scrambled runs.
- **Benchmark on hard pages** (tables, multi-column), not easy prose — that's the only place parser choice moves recall.
- **Tables → Markdown pipe-tables inside the chunk**, always. HTML `<table>` if you have merged cells. Never flatten to prose.
- **Provenance schema up front:** `{source_file, page, section/heading, element_id, char_span}` on every chunk.
- **Stable ids** — hash content or `file:page:span`, never row index — so re-chunking doesn't zero your recall metrics.
- **Detect scanned pages** (near-zero extracted chars) and route to an OCR parser (Unstructured `hi_res`, or Docling with OCR on).
- **Cache parsed output** keyed on file hash; reparsing an unchanged 200-page doc every iteration is pure waste.
- **Reserve the heavy tools:** Unstructured `hi_res` for OCR; LlamaParse for spot-checking your 3-5 ugliest pages against a gold ceiling.
- **Eyeball before you optimize:** open 3 table pages per parser side by side. Five minutes here beats an afternoon of embedding tuning.

---

## Connect to the lab

This lecture is the theory behind **Week 1 lab step 2 (`src/parse.py`)**: you'll parse one ~200-page PDF with **both** PyMuPDF (via `pymupdf4llm`) and Docling, emit the normalized record shape above, and eyeball 2-3 table pages from each to pick a winner. The provenance schema you design here is what makes step 5's golden set and step 6's recall@k computable — every chunk must carry `source`/`page`/`heading`/`char_span` so a retrieved chunk can be matched against a gold chunk. When step 7's `pytest` recall gate fails, this lecture is your first suspect: check whether the losing config flattened a table or scrambled a column before you blame the chunker or the embedder.

---

## Going deeper (optional)

- **PyMuPDF & pymupdf4llm docs** (`pymupdf.readthedocs.io`) — read *"Text Extraction"* and the `pymupdf4llm` "to_markdown" guide (`page_chunks`, table strategy).
- **Docling** — GitHub `docling-project/docling` (formerly `DS4SD/docling`); read the README on the layout + table-structure models and Markdown/JSON export. Search: *"Docling IBM document conversion"* and the Docling technical report.
- **Unstructured docs** (`docs.unstructured.io`) — the *"Partitioning"* page, `partition_pdf(strategy="hi_res")`, and the element/metadata model.
- **LlamaParse** — LlamaIndex docs (`docs.llamaindex.ai`), the *"LlamaParse"* / LlamaCloud section. Use only to establish a gold ceiling.
- **PDF as a format** — the ISO 32000 / PDF Reference; you don't need to read it, but skim a "how PDF text extraction works" writeup to internalize the glyph-positioning model. Search: *"why is extracting text from PDF hard"*.
- **Barnett et al., "Seven Failure Points When Engineering a RAG System" (2024)** — search that exact title; map parser failures to points #1, #2, #4, #7.
- **Reading-order & layout analysis** — background on document layout analysis; search: *"document layout analysis reading order detection"*.

---

## Check yourself

1. A colleague says "PDF is just a document format like Word, extraction is trivial." Correct them from first principles: what does a PDF page actually store, and why does that make reading order and tables hard?
2. Your two-column page comes out with every sentence spliced across the gutter. Explain the exact mechanism, and what a layout-aware parser does differently.
3. A tabular question ("annual price of the Team plan?") retrieves the right *page* but the LLM can't answer. The parser flattened the table to prose. Name the two failure points in play and explain why keeping a Markdown pipe-table in the chunk fixes it.
4. Why must the provenance schema be designed before the first chunk is created, rather than added later? Name the three capabilities you lose without it.
5. You benchmark PyMuPDF vs Docling on plain-prose pages and see no recall difference, so you conclude the parser choice doesn't matter. What's wrong with that experiment?
6. Your recall@k for one parser config is 0 across the board, even on questions you know are answerable, right after you changed the chunker. What's the likely provenance bug?

### Answer key

1. A PDF page stores **positioned glyphs and vector strokes** — drawing instructions ("paint 'A' at (72,400)") in *draw order*, not reading order — not paragraphs/tables/headings the way Word does. Reading order must be inferred from (x,y) coordinates because draw order is arbitrary; tables are just floating text plus grid lines with **nothing linking a number to its row/column** except geometry, so both require reconstruction, not reading.
2. A naive extractor sorts glyphs top-to-bottom then left-to-right, so each visual row zips the two columns together (line-1-col-1 + line-1-col-2, …), producing spliced garbage. A layout-aware parser first **detects the column boundary** (wide vertical whitespace or a layout model), then reads each column top-to-bottom as its own block before moving to the next — reconstructing the human reading order.
3. **#4 (not extracted)** — the LLM has the context but can't pull the answer from a scrambled run of orphaned numbers — and **#7 (incomplete)** — the answer's pieces (label, column, value) are fragmented. A Markdown pipe-table keeps the row/column association literally in the chunk text ("Team | … | $288"), so the embedding encodes the coherent fact and the LLM reads straight across the row.
4. Because adding it later means **reparsing and reindexing the entire corpus**, and any gold labels built in the meantime are invalidated. Without provenance you cannot **cite** (Week 3 requires every claim to resolve to a chunk id + page), cannot **debug** ("where did this come from?" is unanswerable), and cannot **evaluate** (recall@k and context-recall compare retrieved chunks against gold chunk *ids*).
5. Plain prose comes out nearly identical from all parsers, so the experiment is measuring the one place parser choice *doesn't* matter. The differences that move recall@k concentrate on **tables and multi-column/complex layouts** — you must benchmark on the hard pages that actually have them, or you'll wrongly conclude the parser is irrelevant.
6. The `element_id`/chunk ids are almost certainly keyed on **row index / chunk position** rather than stable content. Re-chunking renumbered everything, so every gold label now points at a different (or nonexistent) chunk and reads as a miss — recall craters to 0 for reasons unrelated to retrieval quality. Fix: key ids on a content hash or `source_file:page:char_span`.
