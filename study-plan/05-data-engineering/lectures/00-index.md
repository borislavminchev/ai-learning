# Phase 5 — Lectures Index

Deep, textbook-style lectures backing the [Phase 5 spine](../05-data-engineering.md). The spine's **Theory** sections are short recaps; these lectures are the full treatment — mechanism from first principles, worked examples, production consequences, misconceptions, and cheat sheets.

## How to use

- Read each week's lectures **before** that week's [lab guide](../labs/). The spine's Theory recap tells you the headline; the lecture gives you the depth; the lab makes you build it.
- Budget the spine's "Theory hrs" (~4 hrs/week) for lecture reading, then spend the rest of the week's hours in the lab.
- Don't rabbit-hole. Read once for the mechanism and the cheat sheet, then move to the lab — you'll cement the rest by building.

## Week 1 — Ingestion & Orchestration

| # | Lecture |
|---|---------|
| 1 | [Ingestion Architecture: Batch, Micro-batch, Streaming, and the Immutable Landing Zone](01-ingestion-architecture-landing-zones.md) |
| 2 | [Idempotency and Delivery Semantics: Engineering Around 'Exactly-Once'](02-idempotency-delivery-semantics.md) |
| 3 | [Change Data Capture vs Webhooks, and Tombstone Deletes](03-cdc-vs-webhooks-tombstones.md) |
| 4 | [Orchestrating with Dagster: Assets, Asset Checks, Freshness SLAs, and Blocking Quality Gates](04-dagster-orchestration-quality-gates.md) |
| 5 | [dlt: Incremental Loading, Schema Evolution, and Pipeline State for Free](05-dlt-incremental-loading.md) |

## Week 2 — Extraction & Cleaning

| # | Lecture |
|---|---------|
| 6 | [Document Parsing with Provenance: Docling, Unstructured, and Structure Preservation](06-document-parsing-provenance.md) |
| 7 | [OCR Reality: Preprocessing, Engine Choice, and Confidence Routing](07-ocr-confidence-routing.md) |
| 8 | [Text Normalization and Cleaning: Mojibake, Boilerplate, Language Filtering, and Drop Reports](08-text-normalization-cleaning.md) |
| 9 | [Deduplication at Three Tiers: Exact Hash, MinHash-LSH, and Semantic](09-deduplication-three-tiers.md) |
| 10 | [PII Detection and True Redaction with Presidio](10-pii-detection-redaction.md) |

## Week 3 — Datasets & Lifecycle

| # | Lecture |
|---|---------|
| 11 | [Columnar Storage and DuckDB: Parquet, Predicate Pushdown, and the Small-Files Problem](11-columnar-storage-duckdb.md) |
| 12 | [Dataset Versioning and Lineage: DVC, dvc.yaml, and Linking Data to Runs](12-dataset-versioning-lineage.md) |
| 13 | [Data-Quality Contracts as Blocking Gates: Great Expectations and Pandera](13-data-quality-contracts-blocking-gates.md) |
| 14 | [Synthetic Data Generation and Curation: distilabel, Critic Filtering, and Argilla](14-synthetic-data-curation.md) |
| 15 | [Incremental Indexing and Governance: Content-Hash Diffing, Tombstone Deletes, and Blue-Green Re-embedding](15-incremental-indexing-governance.md) |

## Labs (step-by-step guides)

- [Week 1 Lab — Idempotent, Replay-Safe Ingestion with Dagster](../labs/week-1-idempotent-ingestion.md)
- [Week 2 Lab — Extraction, Cleaning, Three-Tier Dedup, and PII Redaction](../labs/week-2-extraction-cleaning-pii.md)
- [Week 3 Lab — Versioning, Quality Contracts, Incremental Indexing, and the Governance Proof (Phase Milestone)](../labs/week-3-versioning-governance-milestone.md)
