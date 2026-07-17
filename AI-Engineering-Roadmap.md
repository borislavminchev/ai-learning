# The Practical AI Engineer's North-Star Roadmap

*An exhaustive, phased path from competent programmer to expert industry AI engineer — the person who ships LLM systems that are measured, cheap, fast, safe, and debuggable.*

---

## Who this is for & the mindset

This roadmap is for **software engineers who want to build and operate LLM-powered products in industry** — not research scientists. You already write code; you're here to learn how to turn frozen and fine-tuned models into reliable systems that survive contact with real users, real data, and real bills.

Internalize four operating principles before you touch a line of code:

1. **Evals-first, not vibes-first.** You cannot improve what you cannot measure, and LLM outputs are non-deterministic, so "it looked good in the playground" generalizes to nothing. Build the measurement loop *before* you optimize the prompt/model/pipeline. Every phase below has an eval angle; treat the dedicated Evaluation phase as a discipline you thread through *all* the others.
2. **Ship and measure.** Instrument everything (tokens, cost, latency, quality, failures) from day one. Traces and feedback are your debugger for probabilistic systems and the fuel for your data flywheel.
3. **Do in code what code can do.** LLMs are for fuzzy language tasks. Math, lookups, control flow, validation, and business rules belong in deterministic code. "LLM proposes, code disposes." Model output is *untrusted input* to the next system.
4. **Reach for the simplest thing that works.** A prompt beats a chain; a chain beats an agent; a routing workflow beats an autonomous multi-agent swarm — until you can prove you need more. Complexity is a cost you pay in latency, dollars, and 2am debugging.

**How the phases are ordered.** The sequence follows learning-dependency order, not the order these topics get hyped. Foundations unlock everything. Prompting and structured output are the highest-leverage, lowest-cost levers, so they come next. Embeddings infrastructure precedes RAG because RAG is a *consumer* of it. Data engineering feeds both RAG and fine-tuning. Agents build on structured tool-calling. Evaluation is formalized once you have systems worth evaluating, but its mindset applies from Phase 1. Fine-tuning comes late because it's the tool of last resort. Architecture, serving/LLMOps, safety, and multimodal are the "make it real and keep it alive" layers. Frameworks and career are ongoing.

**Emphasis tags** used throughout: **[Theory]** = understand the mental model, **[Practical]** = hands-on build skill, **[Architecture]** = system-level design judgment.

---

## Phase 0 — Foundations of LLMs & ML for Engineers

*Every downstream decision — model choice, cost, latency, prompt design, output quality — traces back to how transformers tokenize, embed, sample, and generate. Understand the mechanics and you debug instead of guess.*

### Concepts to master

**The engineering environment [Practical]**
- Reproducible Python setup with `uv` (2025 default) or `conda` for CUDA stacks; notebooks for exploration but beware hidden-state (Restart & Run All before trusting anything); version pinning (torch/transformers/CUDA break constantly); secrets via `python-dotenv`, never hardcoded.
- Vectorized thinking with **NumPy** (dtype/shape/broadcasting/axis semantics) — the mental model for tensors. **pandas/Polars** for data wrangling; **JSONL** is the dominant LLM dataset format.

**Just-enough classical ML [Theory]**
- Supervised vs unsupervised; train/val/test discipline and *why*; **data leakage** as the #1 real-world killer. Baselines before models — most "AI" problems are still logistic regression / gradient-boosted trees; reach for LLMs when the input is unstructured language.
- Overfitting/underfitting, bias-variance, regularization — and why they matter for fine-tuning (tiny datasets overfit fast; catastrophic forgetting; benchmark contamination).
- **Metrics that match business cost:** precision/recall/F1, confusion matrix, PR-AUC for rare positives, MAE/RMSE; ranking metrics recall@k/MRR/nDCG (critical for RAG); why BLEU/ROUGE are weak for open-ended text and LLM-as-judge dominates.

**PyTorch & the Hugging Face ecosystem [Practical]**
- Tensors/devices/dtypes, autograd, `nn.Module`, `.eval()` vs `.train()`, `no_grad`/`inference_mode`, common OOM/device-mismatch bugs. You must *read* model code and debug shapes, not train from scratch.
- `transformers` (`AutoModel`/`AutoTokenizer`, `pipeline`, `device_map`), `datasets` (streaming/memory-mapping), `huggingface_hub` (caching, gated models, `trust_remote_code` risk).

**How LLMs actually work [Theory / Architecture]**
- What a **parameter** is; VRAM math (params × bytes/param + KV cache + activations); Chinchilla params-vs-data intuition.
- **Transformer at engineer depth:** decoder-only autoregressive stack; encoder-only (BERT) vs decoder-only (GPT/Llama) vs encoder-decoder (T5). Attention as Query-Key-Value; causal masking; multi-head; why context is quadratic and "lost in the middle" happens; FlashAttention as memory-efficient (not different math).
- **Next-token prediction** is the *one thing* LLMs do — hallucination is fluent guessing, not lookup. Logits → softmax → sampling.
- **Tokenization** (BPE/WordPiece/Unigram; ~4 chars ≈ 1 token in English, worse for code/non-English); billing is per token, output usually costlier; count tokens *before* sending; tokenizers are not interchangeable across families.
- **Embeddings & vector spaces** (dense vectors, cosine similarity, dimensionality, normalization, Matryoshka truncation); same model for corpus and query.
- **Context window & KV cache** (shared input+output budget; prefill slow, decode fast; long context eats VRAM; prompt caching reuses KV).
- **Sampling parameters** (temperature, top-p, top-k, max_tokens, stop sequences, penalties, seed) and matching them to task; temp 0 is *not* fully deterministic.

**Model & deployment landscape [Theory / Architecture]**
- Base vs instruct/chat vs **reasoning models** (o-series, Claude extended thinking, Gemini thinking, DeepSeek-R1) and when the extra "thinking tokens" are worth the cost.
- Open vs closed models; decision factors (data sensitivity, volume/cost curve, latency/SLA, licenses); gateways/routers (OpenRouter, LiteLLM).
- **Quantization basics** (fp16→int8→int4; GGUF/AWQ/GPTQ/bitsandbytes; int4 sweet spot; task-dependent quality loss).
- Chat formats/message roles and underlying chat templates; **structured output / JSON mode / tool calling** at a foundational level.
- **Reproducibility & cost/latency/throughput fundamentals** — the three-way quality/cost/latency tradeoff you will trade for the rest of your career.

**Key tools:** uv, JupyterLab, VS Code, Colab/Modal/RunPod, NumPy, pandas/Polars, scikit-learn, PyTorch, Hugging Face `transformers`/`datasets`/`tokenizers`, tiktoken, sentence-transformers, FAISS, Ollama, vLLM (intro).

### Portfolio projects
1. **Reproducible LLM harness repo:** uv-managed env, dotenv API key, a notebook that runs headless via nbconvert (proving no hidden-state), and a script that returns top-5 next-token candidates with logprobs to *see* generation.
2. **Model economics CLI:** given a text file, report token counts and $ cost across 3–4 models using their real tokenizers and current pricing, highlighting English vs code vs Chinese; extend it to estimate VRAM for a given param count + quantization and validate against actually loading 2–3 open models at fp16/int8/int4.

---

## Phase 1 — Prompting & Context Engineering

*The context window is the only interface you fully control on a frozen model — the highest-leverage, lowest-cost knob for quality, cost, and latency in every system you ship.*

### Concepts to master

**Prompt anatomy & core techniques [Theory / Practical]**
- Prompt structure (system/task/context/examples/input/output-spec); static-first for caching, volatile user turn last; positive imperatives; primacy/recency and the "lost-in-the-middle" dead zone.
- Zero-shot vs few-shot (byte-identical exemplar format, cover hard cases, balance/shuffle to avoid recency label bias; reasoning models often do *worse* with heavy few-shot).
- **Chain-of-thought & reasoning prompting** — and the modern rule: do *not* hand-write CoT for reasoning models; instead set reasoning-effort/thinking-budget. Self-consistency (sample N, majority-vote) for high-value hard queries only.
- Role/persona/system-prompt design (durable rules > persona theater; test that it survives long conversations); delimiters & structure markup (XML tags for Claude; wrap untrusted input as data).

**Context engineering [Architecture]** — the successor discipline to prompting: you are the model's working-memory manager. Curate the minimal high-signal token set per call (relevance over volume; "context rot"/distraction from irrelevant chunks). Context-window management: truncation, rolling summarization/compaction (preserve entities/IDs verbatim), retrieval memory, structured scratchpad state. Trigger compaction at a token threshold.

**Reliability levers [Practical]**
- **Prompt caching strategies** (stable prefix first, volatile last; Anthropic `cache_control` breakpoints; any byte change busts the cache; measure hit rate).
- Long-document handling & chunking (semantic/structural chunking, overlap, parent-child, rerankers, and knowing when full-context beats RAG).
- Prompt templates, variables & **prompt management** (never string-concat in business logic; Jinja2/registry; version prompts like code; template drift invalidates caches).
- Sampling/decoding control per task type; **provider-specific idioms** (Claude XML + prefill + extended thinking; OpenAI system/developer hierarchy + strict structured outputs + reasoning_effort; Gemini long context + responseSchema; open models need exact chat templates).

**Failure taxonomy & mitigation [Practical]:** instruction-ignoring, hallucination, format drift, prompt injection/jailbreak, sycophancy, agent looping, wording sensitivity — each with its matched fix. Prompt versioning/testing/eval as a CI gate (introduced here, formalized in Phase 7).

**Key tools:** OpenAI/Anthropic/Gemini SDKs, tiktoken, Jinja2, LiteLLM, promptfoo, Langfuse/PromptLayer, DSPy (for auto-optimized exemplars).

### Portfolio projects
1. **Prompt refactor + eval:** take one real task (invoice-field extraction), write it as a blob then refactor into labeled sections; measure accuracy on 30 held-out examples before/after with token counts.
2. **Context-budget policy for a RAG chatbot:** instrument every token category per call, run ablations (drop history, drop low-score chunks, reorder critical facts to the edges), quantify each decision, and add Anthropic cache_control breakpoints — reporting cached-token %, latency, and cost across 100 requests.

---

## Phase 2 — Structured Outputs & Function / Tool Calling

*Reliable machine-parseable output and tool invocation is the backbone of every agent, RAG pipeline, and LLM-in-a-loop. Get this wrong and everything downstream is brittle.*

### Concepts to master

**The structured-output stack [Practical / Architecture]**
- Decision framework: structured when a program consumes the output, freeform for prose/reasoning; forcing JSON too early hurts reasoning (allow a scratchpad field first, or reason-then-extract in two passes).
- Three reliability tiers: prompt-and-pray → JSON mode (valid JSON, *not* your schema) → **schema-constrained Structured Outputs** (OpenAI strict `json_schema`, Anthropic tool `input_schema`, Gemini `responseSchema`). Know each provider's honored JSON-Schema subset and strict-mode gotchas (`additionalProperties:false`, required-with-null for optionality, depth/enum limits, first-call compile latency).
- **Schema design for LLMs:** flat over nested, enums for closed sets, descriptions on every field (they're prompt real estate), semantic field names, avoid unbounded arrays and giant enums.
- Parsing/validation with **Pydantic** (v2, `model_json_schema`, strict-mode transforms, validators) and **Zod** (`z.infer`, zod-to-json-schema, discriminated unions); the **instructor** wrapper pattern (response_model + auto-repair retries).

**Tool/function calling mechanics [Practical]**
- The universal loop (send tools+message → model returns tool_call *request* → you execute → return result with correct id linkage → loop). Provider differences you must internalize (OpenAI `tool_calls` + `role:'tool'` + args-as-JSON-string; Anthropic `tool_use`/`tool_result` content blocks + parsed-object args; Gemini functionDeclarations).
- Tool definitions *are* prompt engineering (name/description/param docs = when-to-use guidance); keep tool count low; **tool_choice** (auto/none/required/forced) — forcing a specific tool guarantees structured extraction.
- **Parallel tool calls** (execute concurrently, match by id, stop when calls are dependent).

**Advanced [Theory / Architecture]:** constrained decoding / grammars (Outlines, GBNF, XGrammar, lm-format-enforcer — logit masking; why semantic keywords can't be honored; self-host only); **retry & repair loops** (feed exact validation errors back; distinguish transient/deterministic/unrecoverable; idempotency for side-effectful tools); **streaming structured output** (partial-JSON parsers; validate before acting); **MCP** as the emerging tool standard (server exposes tools/resources/prompts; discovery; security surface); enums/unions/recursion gotchas.

**Security [Architecture]:** tool-call arguments are untrusted user input — never pass unsanitized to shells/SQL/filesystems; allowlist tools per context, parameterize, sandbox, least-privilege credentials, human confirmation on destructive actions.

**Key tools:** Pydantic, Zod, instructor, Outlines, guidance, OpenAI/Anthropic/Gemini SDKs, Vercel AI SDK, LiteLLM, MCP SDKs, FastMCP, pytest/promptfoo/Braintrust for the eval suite.

### Portfolio projects
1. **Provider-agnostic tool loop:** one Pydantic model → correct schemas for OpenAI/Anthropic/Gemini via a thin adapter; run the same extraction on all three and diff honored JSON-Schema keywords.
2. **Hardened extraction service:** pull `{name, date, amount, category}` from messy invoices with instructor + a validator (line-item totals must sum), measure schema-valid rate and field accuracy across two models, and add a circuit breaker to human review after 3 repair attempts.

---

## Phase 3 — Embeddings Infrastructure & Vector Databases

*The storage-and-retrieval backbone. RAG, semantic search, recsys, and agent memory all live or die here — this is where latency, cost, and recall are actually won or lost.*

### Concepts to master

**Embedding fundamentals [Theory]:** dense vectors where similarity ≈ geometric proximity; dimensionality/normalization; cosine vs dot vs L2 (match the metric to how the model was trained); curse of dimensionality → ANN; query/document asymmetry (E5/BGE prefixes); VRAM/storage math.

**Choosing & generating embeddings [Practical]:** selection axes (MTEB *but validate on your data*, max length, dimension, multilingual, domain, cost, open vs API); the 2025–26 workhorses; **generation at scale** (GPU batching with TEI/vLLM, API concurrency with backoff, length-bucketing, token-aware batching); **caching/dedup** (key on model:version:normalized_text); chunking as a retrieval-quality problem (fixed/recursive/semantic/structure-aware, overlap, parent-document, **contextual retrieval**).

**Indexes & ANN [Theory / Architecture]:** exact vs approximate; the **recall/latency/cost triangle** (always measure recall@k vs a flat ground truth); **HNSW** internals & tuning (M/efConstruction/efSearch; memory-resident; soft deletes need rebuilds); **IVF, PQ/OPQ, scalar/binary quantization, DiskANN/ScaNN** for scale; **Matryoshka (MRL)** truncation + adaptive retrieval.

**Vector databases in production [Architecture]:** the landscape (FAISS/hnswlib = libraries; Chroma/LanceDB embedded; **pgvector** when you already run Postgres; Qdrant/Weaviate/Milvus/Pinecone/Vespa/Turbopuffer) and how to *select*; **metadata filtering** (pre/post/native filtered-HNSW and its recall pitfalls); **hybrid search** (dense + BM25/SPLADE, RRF fusion — normalize scores); upserts/deletes/**freshness** (tombstones, compaction, blue-green re-embedding); sharding/replication/scatter-gather tail latency; **multi-tenancy** (shared-filter vs namespace vs cluster-per-tenant; GDPR delete); **cost modeling** (HNSW RAM is the silent budget killer); **drift & model versioning** (store raw text; atomic re-embed; golden-set recall monitoring); serving architecture with tracing, semantic query cache, and lexical fallback.

**Key tools:** sentence-transformers, HF TEI, OpenAI/Cohere/Voyage embeddings, FAISS, hnswlib, pgvector/pgvectorscale, Qdrant, Weaviate, Milvus/Zilliz, Pinecone, VectorDBBench, ann-benchmarks, Cohere/BGE rerankers.

### Portfolio projects
1. **Recall/QPS Pareto lab:** build ground truth with a flat index over 1M vectors, sweep HNSW (M × efConstruction × efSearch) and IVF-PQ, and plot recall@10 vs QPS vs RAM; add binary quantization + rescore and show recall recovery.
2. **Production retrieval service:** async FastAPI endpoint with per-tenant access filters, hybrid + filtered search, a reranker, OTel per-stage tracing, a Grafana dashboard (p99, cost/query, empty-result rate), a sampled golden-set recall monitor, and a lexical fallback on vector-DB timeout — load-tested to a recall/latency SLO.

---

## Phase 4 — Retrieval-Augmented Generation (RAG)

*The default architecture for grounding LLMs in private, fresh, or domain-specific knowledge without retraining. Retrieval quality, chunking, reranking, and grounding are what separate a demo from a system users trust.*

### Concepts to master

**The pipeline & its failure surface [Architecture]:** the canonical stages (ingest → parse/clean → chunk → embed → index → query-transform → retrieve → filter → rerank → assemble → generate → cite/validate); RAG is two pipelines (offline indexing, online query); the "7 failure points"; measure **retrieval separately from generation** (most losses are retrieval).

**Ingestion & chunking [Practical]:** layout-aware document parsing (Unstructured, Docling, LlamaParse, PyMuPDF; tables → markdown; provenance for citations); table/OCR handling; chunking strategies (fixed/recursive/semantic/structure-aware/contextual) and parent-document/sentence-window/auto-merging retrieval.

**Retrieval quality [Practical / Architecture]:** hybrid search + fusion; **reranking** with cross-encoders (retrieve top-50 → rerank → keep top-5); late-interaction/multi-vector (ColBERT/ColPali for visually rich docs); **query transformation** (rewriting/condense-question, HyDE, multi-query, decomposition, step-back, routing); metadata/self-querying retrieval with server-side access control; **Graph RAG** for global/thematic and multi-hop questions (and when its indexing cost isn't worth it); **agentic RAG** (retrieve→grade→re-query, Self-RAG/CRAG with web fallback).

**Generation & production [Practical / Architecture]:** context assembly (order to edges, dedup, contextual compression, budget); **citation, grounding & hallucination mitigation** (force inline citations, "say I don't know," post-hoc NLI/entailment verification — models fabricate plausible citations); **when NOT to use RAG** (long-context stuffing, CAG, fine-tuning for behavior — benchmark cost-per-answer); multimodal/table RAG with routing to text-to-SQL for aggregation; production concerns (latency budget, streaming, semantic caching, incremental indexing/CDC/deletes, per-tenant security, prompt-injection via retrieved content, tracing).

**RAG evaluation [Practical]:** retrieval metrics (recall@k, MRR, nDCG) *and* generation metrics (faithfulness/groundedness, context precision/recall, answer relevance) via RAGAS/TruLens/Phoenix; build a golden query→relevant-chunk set; localize failures to retrieval vs generation.

**Key tools:** LlamaIndex, LangChain, Haystack, DSPy, Unstructured/Docling/LlamaParse, Cohere Rerank/bge-reranker, pgvector/Qdrant/Weaviate, Microsoft GraphRAG, LangGraph, RAGAS, TruLens, Arize Phoenix.

### Portfolio projects
1. **Instrumented end-to-end RAG** over a 200-page manual: compare fixed vs recursive-structural vs contextual chunking, add a reranker, enforce citations with an NLI groundedness verifier, and produce a one-page trace of where latency and relevance are lost — plus a full-context no-RAG baseline cost/quality comparison.
2. **Agentic corrective-RAG** (LangGraph): grade retrieved chunks, rewrite-and-retry on weak results, fall back to web search, and measure answer quality + cost vs single-shot RAG on multi-hop questions.

---

## Phase 5 — Data Engineering for AI

*80% of real AI engineering is ingesting, parsing, cleaning, chunking, deduping, versioning, and governing data — not model code. This phase feeds both RAG (Phase 4) and fine-tuning (Phase 8).*

### Concepts to master

**Ingestion & orchestration [Architecture]:** batch vs streaming vs micro-batch; pull (polling/CDC) vs push (webhooks/queues); idempotent ingestion with dedup keys; at-least-once vs "exactly-once is a lie you engineer around"; backpressure/rate-limit handling; immutable landing zones; DAG orchestration, backfills, freshness SLAs, and data-quality gates as first-class pipeline steps.

**Extraction & cleaning [Practical]:** document parsing (digital vs scanned; layout/reading-order; Unstructured/Docling/LlamaParse tradeoffs); **table extraction** (merged cells, page-spanning); **OCR** (preprocessing dominates quality; Tesseract/PaddleOCR/cloud/VLM-OCR, confidence routing to human review); cleaning & normalization for corpora (boilerplate stripping, unicode/mojibake fixes, Gopher/C4-style quality filters, language filtering); **deduplication** (exact hashing, MinHash-LSH near-dup, embedding semantic dedup); **PII detection & redaction** (Presidio, checksums, consistent pseudonymization, true content removal in PDFs/images).

**Datasets & lifecycle [Practical / Architecture]:** synthetic data generation (self-instruct/Evol-Instruct/distillation with critic filtering, avoid model collapse and eval contamination); curation/labeling workflows (guidelines, inter-annotator agreement, active learning, weak supervision, LLM-as-labeler + human verification); **incremental indexing/freshness** (content-hash change detection, tombstone deletes, embedding-version migration); **dataset versioning & lineage** (DVC/lakeFS/Delta/Iceberg time travel; link data → model runs); **data quality/validation/governance** (contracts, expectations as blocking gates, drift, catalogs, retention/consent/licensing).

**Scale & specialized data [Practical / Architecture]:** multimodal data handling (Whisper transcription + diarization, CLIP/ColPali, frame sampling); storage formats & lakehouse (Parquet/Arrow, Delta/Iceberg, small-files problem); large-scale processing (single-node Polars/DuckDB vs Spark/Ray Data/Dask; GPU curation); feature stores basics (offline/online split, point-in-time-correct joins, train/serve skew); web scraping at scale (robots.txt/legal, JS rendering, clean-markdown extraction with provenance); metadata design for retrieval + compliance (source/section/timestamps/ACL tags, access-controlled retrieval).

**Key tools:** Airbyte/dlt, Kafka/Debezium, Dagster/Airflow/Prefect, Unstructured/Docling/PaddleOCR/Textract, trafilatura/ftfy, datasketch/datatrove/NeMo Curator, Presidio, Argilla/Label Studio/Cleanlab, distilabel, DVC/lakeFS/Iceberg, Great Expectations/Pandera, Feast, Scrapy/Playwright/Firecrawl, Parquet/Arrow/DuckDB, Spark/Ray Data.

### Portfolio projects
1. **Reproducible corpus pipeline:** ingest a paginated API + Postgres CDC into an immutable landing zone (idempotent, replay-safe), orchestrate in Dagster with a freshness SLA and a quality gate that halts on anomalies, then clean/dedup/PII-redact into validated JSONL with a drop-report per stage.
2. **Incremental RAG indexer with governance:** content-hash change detection, tombstone deletes, blue-green re-embedding migration, a Great Expectations blocking gate ("PII must be redacted"), and DataHub lineage from source to index — proving that deleting a source doc removes it from answers.

---

## Phase 6 — AI Agents & Agentic Systems

*Agents are how LLMs stop being chat toys and start doing multi-step work against real tools and data — and where 90% of production pain (cost, loops, non-determinism, failures) lives.*

### Concepts to master

**Core patterns [Theory / Architecture]:** the agent loop (perceive→plan→act→observe; "agent" = a loop, not a prompt); **ReAct** with native tool calling (not regex parsing); tool-using agents (schemas, validation, parallel calls, errors-as-observations, confirmation gating on destructive tools); **planning & decomposition** (plan-and-execute vs step-by-step, typed plans, replanning, todo-list pattern); when a single tool-using agent beats multi-agent (usually).

**Memory & orchestration [Architecture]:** short-term/working memory (rolling window, summarization/compaction, scratchpad, cache-friendly ordering); **long-term/episodic/semantic memory** (vector + structured stores; the write path — what/when to persist, dedup/conflict resolution, TTL, per-user namespacing, memory poisoning); **multi-agent patterns** (supervisor/orchestrator-worker, hierarchical, swarm/handoff, group-chat) and their cost/latency/coordination tradeoffs.

**Frameworks & protocols [Practical]:** LangGraph (stateful graphs, checkpointing, interrupts — the production default for complex flows), OpenAI Agents SDK, CrewAI, AutoGen/AG2, Claude Agent SDK, Pydantic AI — pick by *control model*, and understand the primitives so you can drop to raw SDK. **MCP** for building/consuming reusable tools (security is the big deal: tool poisoning, over-broad scopes, confused deputy).

**Making agents production-grade [Architecture / Practical]:** **human-in-the-loop** (durable interrupt/resume, approval gates, escalation); **guardrails & prompt-injection defense** (untrusted tool/retrieved content is data not instructions; least privilege; the "lethal trifecta"); **reflection/self-critique** (prefer grounded feedback — tests/validators/tool errors — over pure self-eval; cap iterations); **computer use & browser automation** (vision vs DOM/accessibility-tree; verification-after-action; sandbox everything); **workflow vs autonomous** tradeoffs (use the simplest topology); **controlling loops/cost/runaway** (max-steps, wall-clock, token/$ budgets, repeated-call detection, model tiering, kill switches); **state/checkpointing/durable execution** (LangGraph checkpointers, Temporal; LLM calls as non-deterministic side effects); **agent evaluation** (outcome, trajectory, tool-call correctness, efficiency, safety — run N times for variance); **observability/tracing** (span trees; harvest traces into eval sets); reliability in non-deterministic systems (temp 0, forced tool choice, retries/fallbacks, model version pinning).

**Key tools:** LangGraph, OpenAI Agents SDK, CrewAI, AutoGen, Claude Agent SDK, Pydantic AI, MCP SDKs, Temporal/Restate, Mem0/Zep/Letta, Playwright/browser-use, Langfuse/LangSmith/Phoenix, NeMo Guardrails/Llama Guard.

### Portfolio projects
1. **Hardened supervisor agent:** researcher + coder + critic specialists (LangGraph) with max-handoffs guard, per-step tracing, durable checkpointing (kill the process mid-run and resume with no duplicated side effects), and A/B against a single tool-using agent on quality/latency/token cost.
2. **Injection-resistant action agent:** summarizes untrusted web pages and can send email; red-team it with a crafted page that tries to exfiltrate secrets, then add layered guardrails (content isolation, tool allowlist, output PII scan, HITL on send) and prove the attack fails — with per-run token/$ budgets and a repeated-tool-call kill switch.

---

## Phase 7 — Evaluation, Testing & Observability

*Evals are the steering wheel. Without a trustworthy measurement loop you cannot ship, iterate, or defend any change. Introduced as a mindset in Phase 1 and applied everywhere — this phase makes it a rigorous discipline.*

### Concepts to master

**Eval-driven development [Theory]:** build the eval *before* optimizing; the loop (collect traces → error-analyze into a failure taxonomy → encode as test cases → change one thing → re-run → ship on measured improvement); offline vs online; the eval taxonomy (reference-based, criteria-based, human); guardrail vs goal metrics.

**Building the measurement layer [Practical]:** **golden sets** from real traces (stratified, grown by mining prod failures, versioned, decontaminated); open-coding error analysis; **LLM-as-judge** (rubric/pairwise/reference-guided; strong judge, CoT-before-verdict, low-cardinality scales, structured output) and its **biases/calibration** (position/verbosity/self-enhancement bias — swap order, measure Cohen's/Krippendorff against humans, meta-evaluate the judge); reference-based metrics (BLEU/ROUGE/METEOR vs BERTScore/COMET — pair a cheap deterministic tripwire with a semantic/judge metric); structured-output/constraint validation (schema-valid rate, first-attempt validity, field precision/recall); classification & retrieval metrics; **RAG-specific eval** (faithfulness/context-relevance/answer-relevance) and **hallucination detection** (NLI/SelfCheckGPT/HHEM); **agent/trajectory eval**.

**Rigor & process [Theory / Practical / Architecture]:** **statistical rigor** (confidence intervals, paired tests/McNemar/bootstrap, power/sample size, multiple-comparisons, judge-noise averaging — don't call a 2% delta on 50 examples a win); human annotation workflows & inter-annotator agreement; **A/B testing & online experimentation** (randomization unit, MDE, guardrails, CUPED, peeking, proxy quality signals like regeneration/edit rate); **regression testing in CI/CD** (prompts/models as versioned artifacts, pass thresholds, pin model snapshots, tiered cheap-vs-judge evals).

**Observability & monitoring [Architecture / Practical]:** **distributed tracing** of every LLM/retrieval/tool span (resolved prompt, context, tokens, cost, latency, errors) via OpenTelemetry GenAI conventions/OpenLLMetry; **logging, feedback capture & the data flywheel** (explicit + implicit signals linked to trace IDs, PII redaction); **production monitoring** (input/embedding drift, silent model drift behind "latest" aliases, sampled online judge evals, alerting); **safety/red-team evaluation** (measure attack-success *and* over-refusal); **cost/latency/quality dashboards** on one screen (TTFT, p50/p95/p99, tokens/$ per feature/tenant, cache hit rate).

**Key tools:** pytest, promptfoo, DeepEval, RAGAS, Braintrust, LangSmith, Langfuse, Arize Phoenix, OpenAI Evals, Inspect, scipy/statsmodels, Argilla/Label Studio, Statsig/GrowthBook, garak/PyRIT, Helicone, Grafana/Prometheus, W&B Weave.

### Portfolio projects
1. **CI eval gate:** a 50–100-case stratified golden set for one feature, deterministic checks + a human-calibrated LLM judge (report Cohen's kappa before/after de-biasing), wired into GitHub Actions to post a score diff and block merges on accuracy/p95-cost regression — demonstrated catching a deliberately-worse prompt.
2. **End-to-end observability + flywheel:** instrument a RAG+agent app with OpenTelemetry tracing (Phoenix/Langfuse) capturing per-step tokens/cost/TTFT with PII redaction, a feedback API linking thumbs/regenerate signals to traces, a sampled online groundedness eval with Slack alerting on quality/cost drift, and a weekly query that auto-adds low-rated traces to a review queue.

---

## Phase 8 — Model Adaptation & Fine-Tuning

*How you turn a generic model into a cheaper, faster, more reliable specialist — but only when prompting and RAG have provably failed, because tuning adds a data pipeline, eval harness, and serving cost most teams underestimate.*

### Concepts to master

**When and what [Architecture / Practical]:** the **prompt vs RAG vs fine-tune decision framework** (weights/behavior vs knowledge/context; tuning is bad at injecting facts; common winning combo = fine-tune for behavior/format + RAG for facts; signals FOR tuning: ≥1k clean examples, stable task, latency/cost pressure, consistent structured output, distillation).
- **SFT fundamentals** (response-only loss masking — a silent quality killer if skipped; EOS tokens; packing; LR/epochs/schedule; small-data overfitting).
- **Instruction tuning & chat templates** (`apply_chat_template`; base vs instruct starting points; only mask assistant turns).

**PEFT & efficient training [Practical]:** **LoRA** (rank/alpha/target_modules; adapters are hot-swappable/mergeable; rsLoRA/DoRA); **QLoRA** (4-bit NF4 base + bf16 adapters, paged optimizers; merge into fp16 base); full FT vs PEFT (ZeRO/FSDP, gradient checkpointing, continued pretraining); **dataset preparation** (quality dominates — clean/diverse/decontaminated beats big/noisy; synthetic data with verification).

**Alignment & compression [Practical]:** **preference tuning** (RLHF→**DPO** as the practical default; beta/KL; IPO/KTO/ORPO/SimPO/GRPO for reasoning); **knowledge distillation** (response-level vs logit-level; on-policy; licensing caveats); **quantization for serving** (GPTQ/AWQ/GGUF/fp8; quantize after merging; re-eval post-quant).

**Operations [Practical / Theory]:** tooling landscape (TRL/PEFT, Axolotl, Unsloth, torchtune, DeepSpeed/FSDP, provider FT APIs); **GPU/VRAM planning** (estimate before renting; LoRA/QLoRA/full FT memory math; fit-levers); **evaluating a fine-tune** (task metrics + LLM-judge win-rate + a **capability-retention regression suite** — loss down ≠ better); **catastrophic forgetting** (PEFT + low LR + replay data + safety re-eval); **embedding-model fine-tuning** (contrastive learning, hard-negative mining — often a bigger RAG win than tuning the generator).

**Key tools:** HF TRL/PEFT, Axolotl, Unsloth, torchtune, bitsandbytes, DeepSpeed/FSDP, AutoGPTQ/AutoAWQ, llama.cpp, vLLM, sentence-transformers, OpenAI/Vertex/Bedrock/Together/Predibase FT APIs, lm-evaluation-harness.

### Portfolio projects
1. **The decision memo, proven:** solve one real task (classify support tickets → JSON) three ways — few-shot prompt, RAG over labeled examples, and a LoRA fine-tune — measuring accuracy, p50/p95 latency, and cost/1k across all three; write the memo justifying which ships, and include a with/without response-only-loss-masking comparison.
2. **Fine-tune + retention scorecard:** QLoRA a 7–8B model, produce fp16/AWQ/GGUF variants benchmarked on VRAM/tokens-per-sec/task-eval, and ship a one-page model card showing task gain **and** MMLU/IFEval/safety retention (deliberately over-train to show forgetting, then recover it with 20% replay + lower LR).

---

## Phase 9 — AI Application Architecture & System Design

*LLM apps live or die on architecture, not prompts: routing, caching, streaming, state, fallback, and observability determine cost, latency, reliability, and whether you can iterate at all.*

### Concepts to master

**Reference architectures [Architecture]:** single-turn feature; multi-turn stateful chatbot (session storage, context-window management, idempotency); RAG app; tool-using agent; copilot/in-product assistant (context harvesting, action grammars, acceptance telemetry).

**The serving/control layer [Architecture / Practical]:** **API gateway / LLM router & fallback** (unified interface, provider abstraction leaks, circuit breakers, tag the serving model); **orchestration** (deterministic control owns flow, LLM at the leaves; graph state machines; durable execution); **streaming** (SSE vs WebSocket, TTFT, cancellation that aborts the upstream call, proxy-buffering bugs); **async & queue-based inference** (idempotent workers, DLQs, batch APIs); **multi-model routing** (cheap-first cascades, task routing — measure quality per tier); **caching layers** (exact + semantic + provider prompt caching; never cache across auth boundaries); **rate limiting & cost governance** (token-aware limits, per-tenant quotas, hard spend kill-switches); **secrets/API-key management** (never client-side, rotation, per-tenant BYOK).

**Data & discipline [Architecture]:** state/session/conversation storage (hot Redis + durable Postgres + object storage + vector memory; GDPR cascade delete across *all* stores including caches/indexes); **separating deterministic code from LLM calls**; **designing for evaluability** (versioned prompts/models, full traces, feedback capture, offline+online eval); observability/tracing; guardrails & I/O validation (treat model output as tainted before it hits SQL/HTML/shell); scalability/HA/capacity planning (provider quota is the real bottleneck, request coalescing, degradation modes); **build-vs-buy** across the stack; data-flow/privacy boundaries & compliance architecture; and **system-design practice** (whiteboard end-to-end designs with back-of-envelope QPS/tokens/cost math and a failure/degradation story).

**Key tools:** FastAPI/Express, Redis, PostgreSQL, LiteLLM/Portkey/Cloudflare AI Gateway, LangGraph/Temporal/Step Functions, Vercel AI SDK, Celery/SQS/BullMQ, GPTCache, Vault/Secrets Manager, Langfuse/Phoenix, Presidio, Bedrock/Vertex/Azure OpenAI (private endpoints).

### Portfolio projects
1. **Gateway with resilience:** LiteLLM (or hand-rolled) in front of two providers with fallback chain, per-tenant token-aware rate limits + monthly spend kill-switch, two-layer (exact + per-tenant semantic) cache, and a degradation mode serving cached/cheaper responses during a provider outage — load-tested to prove one abusive tenant can't starve others.
2. **Three timed mock system designs** (enterprise RAG assistant, coding copilot, high-volume support-triage agent), each with architecture diagram, data model, routing/caching/fallback plan, eval+observability strategy, cost estimate, and a failure/privacy section — plus a GDPR-delete implementation that provably purges a user from every store including the vector index and semantic cache.

---

## Phase 10 — LLMOps: Serving, Optimization & Deployment

*The layer the app-builder curriculum skips: how to actually serve models yourself, reason about GPU/latency/cost economics, ship to edge, and operate releases.*

### Concepts to master

**Self-hosted inference [Architecture]:** engine landscape (**vLLM**, TGI, SGLang, TensorRT-LLM, LMDeploy) and their throughput mechanisms (**continuous batching**, PagedAttention/paged KV cache, automatic prefix caching, chunked prefill); OpenAI-compatible endpoints for portability; tensor vs pipeline vs data parallelism; tuning (max-num-seqs, gpu-memory-utilization, KV-cache OOM, chat-template mismatches, CUDA-graph warmup).

**Hardware & deployment economics [Theory / Architecture]:** **GPU fundamentals** (VRAM = weights + activations + KV cache; memory-bound decode; GPU tiers; nvidia-smi/nvitop literacy); **serverless/managed GPU** (Modal/Replicate/RunPod/Baseten/Together/Fireworks; cold starts, scale-to-zero, warm pools; the closed-API vs serverless vs dedicated vs on-prem decision); **multi-LoRA/adapter serving** (S-LoRA style, per-tenant fine-tunes on one base).

**Performance & economics [Practical / Architecture]:** **reasoning models & test-time compute** (effort/thinking-budget controls; billed thinking tokens; when it pays); **long-context engineering** (needle-in-haystack/RULER limits, position bias/context rot, prompt caching, long-context vs RAG vs CAG); **LLM cost engineering / FinOps** (per-task token accounting, Batch API discounts, cascades, unit economics, per-tenant attribution, budget alerts); **latency engineering & SLOs** (TTFT vs TPOT/ITL, p50/p95/p99, load-testing with locust/k6, speculative decoding); **on-device/edge/browser AI** (llama.cpp/GGUF, Ollama/LM Studio, MLX, WebLLM/transformers.js over WebGPU); **CI/CD for prompts and models (LLMOps)** (git-tracked prompts, eval gates, feature-flag model swaps, shadow/canary/A-B rollout adapted to non-determinism, instant rollback).

**Key tools:** vLLM, TGI, SGLang, TensorRT-LLM, LoRAX, Modal/RunPod/Baseten, nvidia-smi/nvitop, llama.cpp/Ollama/LM Studio/MLX, WebLLM/transformers.js, locust/k6, GitHub Actions, LaunchDarkly, LiteLLM/Helicone cost analytics.

### Portfolio projects
1. **Serve-it-yourself break-even study:** deploy an open model behind vLLM with an OpenAI-compatible endpoint, load-test tokens/sec and cost/1M-tokens at concurrency 1/8/64, serve 10 LoRA adapters on one base routed by tenant id, and chart the crossover volume vs a hosted API (and vs serverless scale-to-zero on a bursty profile).
2. **Prompt/model release pipeline:** a change opens a PR that runs the eval suite as a required check, deploys behind a feature flag, shadows 10% of prod traffic, and supports one-click rollback — plus a latency report (TTFT + TPOT at p50/p95/p99 across concurrency) with an SLO alert that fires before users notice.

---

## Phase 11 — AI Safety, Security, Guardrails & Governance

*LLM apps expose a new attack surface — untrusted natural-language input flowing into privileged tools — that no traditional AppSec control fully covers. Shipping agents without this is the fastest way to leak data, get jailbroken, and fail a compliance review.*

### Concepts to master

**Threats & attacks [Architecture / Practical]:** **threat modeling** (model output is untrusted even from trusted users; trust boundaries; blast-radius per tool; the **lethal trifecta** = private data + untrusted content + exfiltration ability); **direct prompt injection** & system-prompt hardening (delimiters, spotlighting/datamarking — defense-in-depth, never sole control); **indirect (data-borne) prompt injection** (malicious instructions in web pages/PDFs/emails/RAG docs; dual-LLM/quarantined-LLM and CaMeL patterns); **jailbreaks** (persona, many-shot, obfuscation, Crescendo, GCG); **data exfiltration** via markdown-image rendering, SSRF, tool calls (often client-side — audit the renderer; CSP, egress allowlists, param-stripping proxies).

**Defenses & controls [Practical / Architecture]:** **guardrails frameworks** (NeMo Guardrails, Guardrails AI, Llama Guard/Prompt Guard — input/output/action gates; measure the guardrail's own FP/FN); **input/output moderation** (managed content-safety APIs, streaming buffering tradeoff, CSAM reporting obligations); **PII handling & data minimization** (redact before prompts *and* logs/traces/vector stores; redact-then-restore); **secure tool use & least privilege** (per-tool scoped credentials keyed to the *end user*, human approval on destructive/financial actions, narrow typed schemas, MCP server vetting); **sandboxing code execution** (gVisor/Firecracker microVMs, no egress, blocked metadata endpoint, resource limits — Docker alone isn't a boundary); **rate limiting & denial-of-wallet** protection; **improper output handling** (LLM output → SQLi/XSS/RCE/SSRF — parameterize, encode, scope DB roles per user).

**Governance & operations [Theory / Practical]:** **OWASP Top 10 for LLM Apps (2025)** and OWASP Agentic threats; **red-teaming** as continuous CI (garak/PyRIT/promptfoo; track per-attack-family pass rates); **compliance** (GDPR lawful basis/erasure — favor retrieval over training on personal data; data residency; **EU AI Act** risk tiers; vendor zero-retention/DPAs); **hallucination mitigation & grounding**; **toxicity/bias/fairness** (measure harm *and* over-refusal); **model & data supply-chain security** (safetensors over pickle, ModelScan, signing, AI-BOM); **responsible AI governance** (NIST AI RMF, ISO 42001, model cards, enforceable deploy gates — not shelfware); **audit logging & incident response** (tamper-evident, PII-redacted-at-write; IR runbook); **membership inference / training-data extraction / model privacy** (canaries, DP-SGD, embedding-store access control); **guardrail evaluation & calibration** (confusion matrix, threshold sweep, hold out novel attack families).

**Key tools:** MITRE ATLAS, Microsoft Threat Modeling Tool, NeMo Guardrails, Guardrails AI, Llama Guard/Prompt Guard, Azure Content Safety/Bedrock Guardrails, Presidio, E2B/gVisor/Firecracker, garak/PyRIT/promptfoo, ModelScan/Sigstore, NIST AI RMF/ISO 42001, OWASP LLM Top 10, Langfuse/SIEM.

### Portfolio projects
1. **Indirect-injection kill chain, then defused:** build a RAG agent that can email, plant an exfiltration instruction in a retrieved document, demonstrate the leak, then re-architect with a quarantined-LLM (typed-data-only outputs), CSP + image proxy + egress allowlist, tool authZ keyed to the end user, and HITL on send — proving each layer blocks the attack.
2. **Governed, audited deployment:** run one app through a NIST-AI-RMF-mapped risk assessment + model card + pre-deploy gate (red-team results + data-residency evidence required to ship), wire OpenTelemetry audit logging with PII redaction-at-write and guardrail-block-spike alerts, and produce a balanced guardrail eval (benign/borderline/adversarial) with a justified operating point on the safety-vs-over-refusal curve.

---

## Phase 12 — Multimodal & Specialized Modalities

*Most real product value now sits in non-text inputs — documents, screenshots, photos, voice, video. Master wiring VLMs, speech, image generation, and multimodal retrieval into low-latency, cost-controlled systems.*

### Concepts to master

**Vision [Theory / Practical]:** how **VLMs** work (vision encoder + projector + LLM; images become token blocks; tiling drives cost — a 4K screenshot can silently cost thousands of tokens); image understanding & VQA (structured JSON output, grounding/bounding boxes, ROI cropping, count/spatial unreliability, EXIF/contrast preprocessing); **OCR** (dedicated engines for coordinates/scale vs VLM reading for messy/contextual; hybrid: OCR text + image → VLM); **document VQA / layout / structured extraction** (born-digital vs scanned detection, per-field bounding boxes, page-spanning tables, confidence-routed human review).

**Retrieval & embeddings [Theory / Architecture]:** **multimodal embeddings** (CLIP/SigLIP contrastive; ColPali/ColQwen multi-vector page-image retrieval that skips OCR); **multimodal RAG** (text-caption vs multimodal-embedding vs hybrid summary-index→raw-modality; page-level chunking; page+bbox citations; retrieve tightly to control VLM image-token cost).

**Speech & audio [Practical / Architecture]:** **STT/ASR** (Whisper variants, VAD gating against hallucination, word timestamps, diarization, domain-vocabulary biasing, streaming); **TTS** (neural voices, TTFB/streaming, SSML/phoneme overrides, cloning consent); **realtime voice agents** (cascaded STT→LLM→TTS vs speech-to-speech; barge-in, endpointing, <800ms budgets, WebRTC over WebSocket, telephony bridges); audio understanding beyond speech (CLAP/AST classification, audio LLMs).

**Generation & video [Theory / Practical / Architecture]:** **video understanding** (frame sampling + scene detection + ASR fusion; video RAG; token-cost blowup); **image generation** diffusion fundamentals (sampler/steps/CFG/seed/negative prompts; SDXL/FLUX/DALL·E/Ideogram; text-in-image and hands caveats); **control & editing** (ControlNet, inpainting/outpainting, IP-Adapter, LoRAs, instruction editing); **choosing & evaluating multimodal models** (task-specific gold sets — CER/WER, field F1, recall@k, voice latency/task-success — not leaderboards); **cost/latency/scaling** (downscale/crop before sending, "low detail," frame budgets, batch/async, prompt caching, self-host with vLLM+quantization).

**Key tools:** GPT-4o/Claude/Gemini vision, Qwen2.5-VL, SigLIP/OpenCLIP, ColPali, PaddleOCR/Textract/Docling/Mistral OCR, LayoutLMv3/Donut, faster-whisper/WhisperX/Deepgram, pyannote/Silero VAD, ElevenLabs/Cartesia, LiveKit/Pipecat/OpenAI Realtime, ffmpeg/PySceneDetect, diffusers/ComfyUI/FLUX/ControlNet, Modal/Replicate/fal.

### Portfolio projects
1. **Receipt/invoice-to-ERP pipeline:** phone photo → auto-rotate/crop/enhance → structured JSON with per-field bounding boxes and a confidence report, born-digital-vs-scanned detection, totals-arithmetic validation, and low-confidence fields routed to a review UI — then cut its cost 60% (server-side crop/downscale, prompt caching, cheap-model pre-filter, batch API) without losing accuracy.
2. **"Chat with your slide decks":** index PDF page images with ColPali, retrieve top-3 pages, answer with a VLM and page-number citations + thumbnails, and A/B against a text-only OCR RAG baseline on visual questions — plus a reusable multimodal eval harness reporting field accuracy, p50/p95 latency, and $/1000 docs across three VLMs.

---

## Ongoing — Frameworks, Ecosystem, Team Practice & Career

*The half-life of specifics here is months. This runs in parallel with every phase.*

### Concepts to master

**Framework judgment [Architecture / Practical]:** the **raw-SDK-vs-framework decision** ("framework tax" — indirection, churn, opacity; default to raw SDKs + thin gateway, adopt only when it removes repeated work, always keep an escape hatch); provider SDKs deeply (OpenAI Responses/Chat, Anthropic content-blocks/caching/Agent SDK, Gemini/Vertex, Bedrock Converse + IAM, Azure OpenAI); gateways (LiteLLM/OpenRouter/Portkey); the framework landscape (LangChain/LCEL and its failure modes, LangGraph, LlamaIndex, Haystack, **DSPy** prompt-optimization paradigm, Semantic Kernel, Pydantic AI/Instructor); **MCP authoring** (building servers, not just consuming); local/self-hosted tooling (Ollama/vLLM/TGI); prototyping UIs (Streamlit/Gradio/Chainlit) vs production frontends (Next.js + Vercel AI SDK with streaming/generative UI); prompt-management, observability, and structured-output abstractions across the stack; **notebooks-vs-production discipline** (extract typed modules, config, tests with recorded LLM calls, CI).

**Judgment & currency [Theory / Practical]:** **framework evaluation rubric** (velocity, breaking-change cadence, abstraction cost, ejectability, deps, license, maintainer risk — spike before adopting); **benchmark literacy & limits** (MMLU/GPQA/SWE-bench/BFCL/LMArena; contamination, format sensitivity, "context window ≠ usable quality window" — build your own domain eval); **reading model cards/pricing/release intelligence** fast (context/output caps, modalities, cutoff, per-token + cached + batch + reasoning-token pricing, deprecation); **staying current** (changelogs/RSS over hype; a personal "new-model smoke test" repo).

**Team & career [Practical]:** specs and acceptance criteria for probabilistic features; shared prompt/eval review; on-call/incident response for silent quality decay; operating the **data flywheel**; documenting model/prompt decisions; **portfolio of shipped, evaluated projects** (with cost/latency writeups and guardrails — not toy demos); paper-to-practice reading; OSS contributions; **AI-engineering interview prep** (LLM-app system design, debugging non-determinism, tradeoff reasoning — not ML-research interviews).

**Key tools:** all provider SDKs, LiteLLM/OpenRouter, LangGraph/LlamaIndex/DSPy/Pydantic AI, MCP SDKs/FastMCP, Streamlit/Chainlit, Next.js + Vercel AI SDK, Langfuse/LangSmith/Phoenix, uv/pytest/VCR.py, LMArena/Artificial Analysis, GitHub.

### Portfolio project
- **A reusable "new-model smoke test" repo** (structured extraction, tool loop, long-context recall, reasoning via LiteLLM against any OpenAI-compatible endpoint, printing a comparison + cost-per-correct-answer table) — and a headline **public end-to-end project**: deployed app + eval suite + cost/latency writeup + a README explaining every tradeoff and "what would make us regret this."

---

## 🏛️ Capstone — The Enterprise Knowledge & Action Platform

*A multi-tenant, production-grade assistant for a regulated B2B domain (e.g., a healthcare-adjacent or financial operations copilot) that integrates most of the stack.*

**What it must do and prove:**

- **Ingestion & data engineering (Phase 5):** an orchestrated, idempotent pipeline that parses messy PDFs (tables, scans/OCR), cleans/dedups/PII-redacts, versions the dataset (DVC/lakeFS), enforces a quality gate, and does incremental indexing with tombstone deletes and a blue-green re-embedding migration.
- **Retrieval backbone (Phases 3–4):** hybrid + reranked retrieval over a multi-tenant vector store with server-side access-control filters, contextual chunking, page+bbox citations, and a groundedness verifier — plus a documented "when we chose long-context/CAG instead" decision.
- **Multimodal (Phase 12):** answer questions over document *images* (ColPali) and support a realtime **voice** interface (STT→LLM→TTS with barge-in).
- **Agentic layer (Phases 2, 6):** a supervisor agent with typed tools (text-to-SQL for aggregates over a read-only per-user-scoped role, a document-retrieval tool, a write/action tool behind HITL approval), durable checkpointing (survives a crash mid-run), memory, and per-run token/$ budgets with a kill switch.
- **Fine-tuning where it earns its place (Phase 8):** a fine-tuned embedding model (hard-negative mining) for the domain, and optionally a small LoRA for consistent structured output — each justified by an A/B against the prompt/RAG baseline and a capability-retention scorecard.
- **Architecture & serving (Phases 9–10):** an LLM gateway with provider fallback, cheap-model-first routing/cascade, exact + semantic caching, per-tenant rate limits and spend caps, streaming UI (Next.js + Vercel AI SDK), self-hosted open-model serving (vLLM + multi-LoRA) with a documented cost break-even vs API, and a CI/CD pipeline that gates prompt/model changes on evals and rolls out via shadow → canary → flag with instant rollback.
- **Evaluation & observability (Phase 7):** a versioned golden set, a human-calibrated LLM judge, retrieval + faithfulness + trajectory metrics, statistical rigor on ship/no-ship decisions, end-to-end OpenTelemetry tracing, a cost/latency/quality dashboard, sampled online quality evals, and a data flywheel feeding failures back into evals/training data.
- **Safety, security & governance (Phase 11):** threat model + lethal-trifecta mitigation, layered prompt-injection defenses on retrieved/tool content, sandboxed execution, PII redaction in prompts *and* traces, a red-team suite in CI, GDPR cascade-delete across every store, a NIST-AI-RMF risk assessment + model card, audit logging, and an incident-response runbook.

**Deliverable:** the deployed system, an architecture + data-flow + threat-model diagram set, an eval report with confidence intervals, a cost/latency writeup with unit economics, and a "tradeoffs & what we'd do differently" document.

---

## ✅ What an expert practical AI engineer should be able to do

**Foundations & reasoning**
- [ ] Explain, from tokenization to sampling, *why* a given output/cost/latency happened — and debug quality by isolating prompt vs params vs context length vs tokenizer vs model.
- [ ] Compute VRAM, token cost, and latency (TTFT/TPOT) for a workload before building it, and pick the model/quantization/hardware that fits.
- [ ] Choose correctly among prompt, RAG, long-context/CAG, fine-tuning, and reasoning models — and defend it with numbers, not fashion.

**Building & shipping**
- [ ] Design and ship any of the reference architectures (single-turn, chatbot, RAG, agent, copilot, voice) with streaming, routing, caching, fallback, rate limits, and secrets handled correctly.
- [ ] Guarantee machine-parseable output and reliable tool calling across providers, with validation + repair loops and a security posture that treats model output and tool args as untrusted.
- [ ] Build a production retrieval system: layout-aware ingestion, tuned chunking/embeddings, hybrid + reranked + filtered retrieval, grounded cited generation, incremental indexing, and per-tenant access control.
- [ ] Build agents that don't run away: bounded loops/budgets, durable state, HITL on risky actions, injection-resistant tool use, and full tracing.
- [ ] Self-host and optimize serving (vLLM, continuous batching, multi-LoRA, quantization) and know the API-vs-serverless-vs-dedicated break-even.

**Measuring & operating**
- [ ] Stand up an eval harness *before* optimizing: golden sets from real traces, calibrated LLM judges, retrieval/faithfulness/trajectory metrics, and statistical rigor on ship/no-ship.
- [ ] Gate every prompt/model change on evals in CI, roll out via shadow/canary/flags, and roll back instantly.
- [ ] Instrument end-to-end tracing + cost/latency/quality dashboards, run a data flywheel, and detect silent drift and model swaps behind "latest" aliases.
- [ ] Model unit economics (cost/user, margin/feature) and cut cost 40–60% via caching, cascades, and batch without regressing quality.

**Securing & governing**
- [ ] Threat-model an LLM/agent system, break the lethal trifecta, and defend against direct/indirect injection, jailbreaks, exfiltration, and improper output handling — with measured guardrails (catch rate *and* over-refusal).
- [ ] Sandbox code execution, enforce least-privilege tools keyed to the end user, and pass a compliance review (GDPR erasure, data residency, EU AI Act tiering, supply-chain scanning, audit logging, model cards).

**Judgment & growth**
- [ ] Evaluate a framework against a rubric, drop to raw SDKs when abstractions leak, and keep an escape hatch.
- [ ] Read model cards, pricing, and benchmarks skeptically — and trust your own domain evals over leaderboards.
- [ ] Maintain a personal model-smoke-test harness and a shipped, evaluated public portfolio, and stay current from changelogs and source, not hype.

*Reach the simplest solution that meets the SLO, prove it with evals, ship it with observability, and secure it before it ships — that is the whole job.*
