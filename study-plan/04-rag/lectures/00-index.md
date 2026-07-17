# Phase 4 — Lectures Index

Deep, textbook-style lectures backing the [Phase 4 spine](../04-rag.md). The spine's **Theory** sections are short recaps; these are the full treatment — mechanism from first principles, worked examples, production consequences, misconceptions, cheat sheets, and a self-check with answers.

## How to use

- Read the week's lectures **before** that week's [lab guide](../labs/). The spine's Theory recap tells you the headline; the lecture gives you the depth; the lab makes you build it.
- Each lecture is ~20 min. Budget the spine's stated "Theory hrs" per week on lecture reading, then spend the rest of the week's hours in the lab.
- Don't rabbit-hole. Read once for the mechanism + the cheat sheet, do the self-check, move to the lab. You'll cement the rest by building.

## Week 1 — Ingestion & retrieval: parse, chunk, and measure

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 1 | [The Two RAG Pipelines and the 7 Failure Points](01-rag-pipelines-and-failure-points.md) | offline vs online pipelines, failure taxonomy |
| 2 | [Layout-Aware Parsing: Why PDFs Fight Back](02-layout-aware-pdf-parsing.md) | PyMuPDF/Docling/Unstructured, tables→markdown |
| 3 | [Chunking Strategies: The Highest-Leverage Knob](03-chunking-strategies.md) | fixed/recursive/structure-aware, overlap, small-to-big |
| 4 | [Measuring Retrieval in Isolation: Golden Sets, recall@k, MRR](04-retrieval-only-evaluation.md) | golden set, recall@k, MRR, no-generation eval |

## Week 2 — Retrieval quality: hybrid, reranking, query transformation

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 5 | [Hybrid Search: Dense + Sparse Fused with RRF](05-hybrid-search-dense-sparse-rrf.md) | dense vs sparse, RRF fusion |
| 6 | [Cross-Encoder Reranking: The Highest-ROI Change](06-cross-encoder-reranking.md) | bi- vs cross-encoder, top-50→top-5 |
| 7 | [Query Transformation: Rewriting the Query Before Retrieval](07-query-transformation.md) | multi-query, HyDE, step-back, decomposition |
| 8 | [Metadata Filtering, Self-Query, and Access Control](08-metadata-self-query-and-access-control.md) | self-query, server-enforced ACL pre-filter |
| 9 | [Late Interaction: ColBERT and ColPali (Conceptual)](09-late-interaction-colbert-colpali.md) | MaxSim, token-level retrieval, ColPali |

## Week 3 — Generation, grounding, and corrective RAG

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 10 | [Context Assembly and Token Budgeting](10-context-assembly-and-token-budgeting.md) | lost-in-the-middle, dedup, compression, budget |
| 11 | [Inline Citations and NLI Groundedness Verification](11-citations-and-nli-groundedness.md) | attribution, NLI entailment, abstention policy |
| 12 | [Corrective and Agentic RAG with LangGraph](12-corrective-and-agentic-rag-langgraph.md) | CRAG/Self-RAG, grade→rewrite→retrieve loop |
| 13 | [When NOT to Use RAG: Cost-Per-Answer Decisions](13-when-not-to-use-rag.md) | long-context, CAG, fine-tuning tradeoffs |
| 14 | [Semantic Caching and Incremental Indexing](14-semantic-caching-and-incremental-indexing.md) | embedding-similarity cache, upsert/delete/CDC |
| 15 | [Prompt Injection via Retrieved Content](15-prompt-injection-via-retrieved-content.md) | indirect injection, delimiting, mitigations |
| 16 | [Graph RAG for Multi-Hop and Global Questions](16-graph-rag.md) | knowledge graph, community summaries, global/local search |

## Week 4 — RAG evaluation and the CI regression gate

| # | Lecture | Backs spine topic |
|---|---------|-------------------|
| 17 | [Two-Stage Evaluation and Retrieval Metrics](17-two-stage-eval-and-retrieval-metrics.md) | split retrieval vs generation, recall@k/MRR/nDCG |
| 18 | [Generation Metrics with RAGAS](18-generation-metrics-with-ragas.md) | faithfulness, context precision/recall, answer relevancy |
| 19 | [Building and Versioning Golden Sets](19-golden-sets-construction-and-versioning.md) | manual vs synthetic, strata, versioning |
| 20 | [Failure Localization and the CI Regression Gate](20-failure-localization-and-ci-regression-gate.md) | decision rule, thresholds.yaml, GitHub Actions gate |

## Labs (step-by-step guides)

- [Week 1 Lab — Ingest, Chunk, and Measure Retrieval on a Real 200-Page PDF](../labs/week-1-ingestion-and-retrieval-measurement.md)
- [Week 2 Lab — Hybrid + Rerank + Query Transforms, Ablated on the Golden Set](../labs/week-2-hybrid-rerank-query-transform-ablation.md)
- [Week 3 Lab — Grounded Generation, NLI Verification, and Corrective RAG](../labs/week-3-grounded-generation-and-corrective-rag.md)
- [Week 4 Lab — RAG Evaluation, CI Gate, and the End-to-End Milestone Service](../labs/week-4-rag-evaluation-and-milestone-service.md)
