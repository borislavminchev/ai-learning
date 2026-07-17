# Phase 11 — Lectures Index (AI Safety, Security, Guardrails & Governance)

These are the **deep lectures** that back the [phase spine](../11-safety-security.md). The spine is a weekly plan and a *recap*; these lectures carry the *why* and the mechanism behind each control.

**Read each week's lectures BEFORE that week's lab.** The spine's per-week **Theory** section is a condensed recap of the lectures below — the labs assume you have already worked through the mechanism here.

## Lectures by week

### Week 1 — Threat modeling & the indirect-injection kill chain
| # | Lecture |
|---|---------|
| 01 | [Threat Modeling LLM Systems & the Lethal Trifecta](01-threat-modeling-lethal-trifecta.md) |
| 02 | [Prompt Injection: Direct vs Indirect (Data-Borne)](02-prompt-injection-direct-indirect.md) |
| 03 | [Jailbreak Families: Recognizing and Testing Structural Attacks](03-jailbreak-families.md) |
| 04 | [Exfiltration Channels: How Data Actually Leaves](04-exfiltration-channels.md) |
| 05 | [OWASP LLM Top 10 (2025) & Agentic Threat Taxonomy](05-owasp-llm-top10-agentic.md) |

### Week 2 — Runtime defenses: quarantine, guardrails, sandboxing & measurement
| # | Lecture |
|---|---------|
| 06 | [Quarantined-LLM, Dual-LLM & CaMeL](06-quarantined-dual-llm-camel.md) |
| 07 | [Guardrail Frameworks: Prompt Guard, Llama Guard, NeMo & Guardrails AI](07-guardrail-frameworks.md) |
| 08 | [PII Redaction & Data Minimization with Presidio](08-pii-redaction-data-minimization.md) |
| 09 | [Secure Tool Use, Least Privilege, Output Handling & Cost Controls](09-secure-tool-use-least-privilege.md) |
| 10 | [Sandboxing Untrusted Code Execution](10-sandboxing-code-execution.md) |
| 11 | [Measuring Guardrails: Confusion Matrix, Catch-Rate & Over-Refusal](11-measuring-guardrails-confusion-matrix.md) |

### Week 3 — Governance: red-team-in-CI, audit logging, compliance & a deploy gate
| # | Lecture |
|---|---------|
| 12 | [Automated Red-Teaming as CI: garak, PyRIT & promptfoo](12-red-teaming-as-ci.md) |
| 13 | [Model & Data Supply-Chain Security](13-supply-chain-security.md) |
| 14 | [Compliance You Can Act On: GDPR, Data Residency & EU AI Act](14-compliance-gdpr-residency-eu-ai-act.md) |
| 15 | [Governance Frameworks, Tamper-Evident Audit Logging & the Deploy Gate](15-governance-audit-deploy-gate.md) |

## Lab guides

Work each week's lab *after* its lectures:

- **Week 1** — [Build & Fire an Indirect-Injection Kill Chain](../labs/week-1-indirect-injection-kill-chain.md)
- **Week 2** — [Defuse the Kill Chain: Quarantine, Guardrails, Sandbox & Measure](../labs/week-2-runtime-defenses-and-measurement.md)
- **Week 3** — [Governance, Red-Team-in-CI & the Enforceable Deploy Gate (Phase Milestone)](../labs/week-3-governance-ci-gate-and-milestone.md)
