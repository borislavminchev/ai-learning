# AI Engineering — Week-by-Week Study Plan

An engineering-first (not research-oriented) path from competent programmer to expert practical AI engineer. Every week pairs **theory** with a **runnable lab** that has measurable acceptance criteria. Companion to `../AI-Engineering-Roadmap.md` (the north-star), broken into weekly detail.

## How to use

- **Budget:** ~10–15 hrs/week, part-time. Each week states an hours breakdown.
- **Order:** Files are numbered in dependency order. Don't skip forward — later phases assume earlier labs.
- **Discipline:** Do the lab, hit the *Definition of Done*, answer the *Self-check*. If you can't, repeat before advancing.
- **Portfolio:** Keep every lab in a public repo. The shipped, evaluated projects ARE your credential.

## Timeline

| # | Phase | Weeks | File |
|---|-------|-------|------|
| 0 | Foundations of LLMs & ML for Engineers | 3 | [00-foundations.md](00-foundations/00-foundations.md) |
| 1 | Prompting & Context Engineering | 3 | [01-prompting-context.md](01-prompting-context/01-prompting-context.md) |
| 2 | Structured Outputs & Function/Tool Calling | 2 | [02-structured-outputs-tools.md](02-structured-outputs-tools/02-structured-outputs-tools.md) |
| 3 | Embeddings Infrastructure & Vector Databases | 3 | [03-embeddings-vector-dbs.md](03-embeddings-vector-dbs/03-embeddings-vector-dbs.md) |
| 4 | Retrieval-Augmented Generation (RAG) | 4 | [04-rag.md](04-rag/04-rag.md) |
| 5 | Data Engineering for AI | 3 | [05-data-engineering.md](05-data-engineering/05-data-engineering.md) |
| 6 | **AI Agents & Agentic Systems (expanded deep track)** | 6 | [06-agents.md](06-agents/06-agents.md) |
| 7 | Evaluation, Testing & Observability | 3 | [07-evaluation-observability.md](07-evaluation-observability/07-evaluation-observability.md) |
| 8 | Model Adaptation & Fine-Tuning | 4 | [08-fine-tuning.md](08-fine-tuning/08-fine-tuning.md) |
| 9 | AI Application Architecture & System Design | 3 | [09-architecture-system-design.md](09-architecture-system-design/09-architecture-system-design.md) |
| 10 | LLMOps: Serving, Optimization & Deployment | 3 | [10-llmops-serving.md](10-llmops-serving/10-llmops-serving.md) |
| 11 | AI Safety, Security, Guardrails & Governance | 3 | [11-safety-security.md](11-safety-security/11-safety-security.md) |
| 12 | Multimodal & Specialized Modalities | 3 | [12-multimodal.md](12-multimodal/12-multimodal.md) |
| 13 | Frameworks, Ecosystem, Team Practice & Career | 2 | [13-frameworks-career.md](13-frameworks-career/13-frameworks-career.md) |
| 14 | Capstone — Enterprise Knowledge & Action Platform | 4 | [14-capstone.md](14-capstone/14-capstone.md) |

**Total: ~49 weeks** (~1 year part-time). Phase 13 (frameworks/career) also runs *continuously* in the background.

## Notes

- The **Agents phase (6)** is deliberately the deepest: agent loop from scratch → planning patterns → durable state → **interop protocols (MCP, A2A, ACP, AGNTCY, AG-UI)** → identity/OAuth → memory & multi-agent → coding agents, computer use, guardrails, and trajectory eval.
- Labs prefer **free/local** options (Ollama, FAISS, pgvector/Qdrant in Docker, Colab/Modal free credits) wherever a paid API or GPU would otherwise be needed.
- If you have prior experience in a phase, use its "You are ready to move on when…" checklist to test-out and skip.
