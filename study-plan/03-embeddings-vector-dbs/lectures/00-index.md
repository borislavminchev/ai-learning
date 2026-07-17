# Phase 3 — Lectures Index

Deep, textbook-style lectures backing the [Phase 3 spine](../03-embeddings-vector-dbs.md). The spine's **Theory** sections are short recaps; these are the full treatment — mechanism from first principles, worked examples, production consequences, and the tradeoffs behind each knob.

## How to use

- Read the week's lectures **before** that week's [lab guide](../labs/). The spine's Theory recap gives the headline; the lecture gives the depth; the lab makes you build it.
- Don't rabbit-hole — read once for the mechanism, then go build. You cement the rest in the lab.

## Week 1 — Embeddings: fundamentals, model choice, and generation at scale

| # | Lecture |
|---|---------|
| 1 | [Embedding Geometry and Similarity Metrics](01-embedding-geometry-and-metrics.md) |
| 2 | [Choosing an Embedding Model on Evidence](02-choosing-an-embedding-model.md) |
| 3 | [Query/Document Asymmetry and Instruction Prefixes](03-query-document-prefixes.md) |
| 4 | [Matryoshka Representation Learning and Dimension Truncation](04-matryoshka-truncation.md) |
| 5 | [Generating Embeddings at Scale: Batching, Caching, and Serving](05-embedding-generation-at-scale.md) |

## Week 2 — ANN indexes: HNSW, IVF/PQ, quantization, and the recall/latency/cost triangle

| # | Lecture |
|---|---------|
| 6 | [Exact vs Approximate Search and Honest Recall Measurement](06-exact-vs-approximate-and-honest-recall.md) |
| 7 | [HNSW Internals and Tuning](07-hnsw-internals-and-tuning.md) |
| 8 | [IVF Inverted-File Indexes](08-ivf-inverted-file-indexes.md) |
| 9 | [Quantization: PQ, Scalar, Binary, and Rescore](09-quantization-pq-scalar-binary-rescore.md) |
| 10 | [The Recall/Latency/Cost Triangle and Pareto Analysis](10-recall-latency-cost-pareto.md) |

## Week 3 — Vector databases in production: filtering, hybrid search, freshness, multi-tenancy

| # | Lecture |
|---|---------|
| 11 | [The Vector Database Landscape and How to Select](11-vector-db-landscape-and-selection.md) |
| 12 | [Metadata Filtering and the Pre/Post-Filter Recall Trap](12-metadata-filtering-and-the-recall-trap.md) |
| 13 | [Hybrid Search, RRF Fusion, and Reranking](13-hybrid-search-rrf-and-reranking.md) |
| 14 | [Freshness, Lifecycle, and Blue-Green Re-embedding](14-freshness-lifecycle-and-blue-green-reembedding.md) |
| 15 | [Multi-Tenancy and Tenant Isolation](15-multi-tenancy-and-isolation.md) |
| 16 | [Serving a Retrieval Service: Tracing, Caching, and Fallback](16-serving-tracing-caching-and-fallback.md) |

## Labs (step-by-step guides)

- [Week 1 Lab — Embedding Toolkit and Model-Selection Benchmark](../labs/week-1-embedding-toolkit-and-model-benchmark.md)
- [Week 2 Lab — Recall/QPS/RAM Pareto Lab](../labs/week-2-recall-qps-pareto-lab.md)
- [Week 3 Lab — Production Retrieval Service (Phase Milestone)](../labs/week-3-production-retrieval-service.md)
