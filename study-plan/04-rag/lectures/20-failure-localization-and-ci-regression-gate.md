# Lecture 20: Failure Localization and the CI Regression Gate

> By now you can compute retrieval metrics (recall@k, MRR, nDCG) and generation metrics (RAGAS faithfulness, context precision/recall, answer relevancy). This lecture is where those two families finally *pay off*: together they let you take any failing RAG case and route it — with one lookup in a table — to "fix retrieval" or "fix generation," instead of guessing. Then you turn that routing into infrastructure: a per-row localizer with concrete thresholds, a proof that it works by deliberately breaking retrieval and then generation and watching the right metric crater, a shareable eval report that a mean score can't lie in, and a CI gate that fails the build the day someone regresses faithfulness. After this lecture you will be able to author `localize.py`, seed two failures that prove your decision rule, emit a `report.md` with per-stratum breakdowns and worst-case traces, and wire a GitHub Actions gate whose thresholds are set for headroom against judge noise and whose runs cost minutes and cents, not hours and dollars.

**Prerequisites:** Lecture 4 (golden sets, recall@k/MRR), the RAGAS-metrics lecture (faithfulness, context precision/recall, answer relevancy, LLM-as-judge), basic YAML + GitHub Actions, pytest. · **Reading time:** ~28 min · **Part of:** Retrieval-Augmented Generation, Week 4

## The core idea (plain language)

A RAG answer can be wrong for two structurally different reasons, and the whole discipline of this week exists to tell them apart:

1. **The retriever never fetched the right evidence.** The chunk that contains the answer wasn't in the top-k you handed the LLM. No prompt, no model swap, no temperature can recover information that isn't in the context. This is a **retrieval** problem.
2. **The retriever fetched the right evidence and the LLM ignored, contradicted, or embroidered on it.** The answer chunk *was* in context; the model still hallucinated. This is a **generation** problem.

A single end-to-end "accuracy: 62%" number **cannot distinguish these**, so it cannot route a fix. You'll spend two weeks tuning a prompt while the real bug is a chunker that split the answer across a boundary — or you'll re-embed the whole corpus while the real bug is a prompt that never told the model to cite its context. Both wastes are avoidable.

The payoff of computing *both* metric families — a retrieval signal (context-recall, and your own recall@k) and a generation signal (faithfulness) — is that their **combination** localizes the failure. That's the entire reason you paid the cost of two eval harnesses. Here is the rule, and you should be able to recite it in your sleep:

> - **context-recall LOW (faithfulness anything) ⇒ RETRIEVAL problem.** The evidence isn't in the context; fix chunking / embeddings / k / reranker.
> - **context-recall HIGH but faithfulness LOW ⇒ GENERATION problem.** The evidence *is* there and the model isn't using it; fix prompt / model / instruction-to-cite.
> - **context-recall HIGH and faithfulness HIGH but the answer is still wrong ⇒ golden-set/label problem or a reasoning gap.** The pipeline did its job; suspect your ground truth or a genuinely hard multi-step inference.

This table turns the useless sentence *"the RAG is bad"* into a routed ticket: *"case #37 is a retrieval miss — the reranker demoted the gold chunk."* The rest of the lecture is about making that routing mechanical (per-row code with thresholds), **proving** it's real (seed a retrieval break, seed a generation break, watch the predicted metric collapse), reporting it so nobody can hide a bad stratum behind a good mean, and gating it in CI so a regression is caught before merge instead of after launch.

## How it actually works (mechanism, from first principles)

### Why the two signals are near-orthogonal

The reason the decision rule works is that **context-recall and faithfulness measure different halves of the pipeline and are computed against different references**:

- **context-recall** asks: *does the retrieved context contain the information needed to produce the ground-truth answer?* It compares **context vs ground_truth**. The generator is irrelevant to it. A retrieval-only signal.
- **faithfulness** asks: *is every claim in the generated answer entailed by the retrieved context?* It compares **answer vs context**. The ground truth is irrelevant to it. A generation-only signal (conditioned on whatever context retrieval happened to return).

Because one ignores the answer and the other ignores the ground truth, they move independently, and the *pattern* of the two tells you which stage broke. The 2×2 (plus the "both high, still wrong" corner) is the natural read:

```
                 faithfulness LOW        faithfulness HIGH
              +------------------------+------------------------+
ctx-recall    |  RETRIEVAL              |  RETRIEVAL             |
    LOW        |  (evidence missing;     |  (model faithfully     |
              |   model also drifting)  |   grounded on the      |
              |                         |   WRONG/insufficient   |
              |                         |   context)             |
              +------------------------+------------------------+
ctx-recall    |  GENERATION             |  GOLDEN/REASONING      |
    HIGH       |  (evidence present,     |  (pipeline did its job;|
              |   model ignores/        |   suspect label or     |
              |   contradicts it)       |   hard multi-hop)      |
              +------------------------+------------------------+
```

Note the top-right cell: **low context-recall dominates regardless of faithfulness.** A high faithfulness score there is a trap — the model is faithfully summarizing context that doesn't actually contain the answer. Faithful to the wrong evidence is still a retrieval failure. That's why the rule reads "context-recall low ⇒ retrieval, *faithfulness anything*."

### Turning the rule into thresholds

The rule is stated on "low" and "high," which a computer can't evaluate. You pick numeric cutoffs. The spine's defaults, which are sane starting points:

- `context_recall < 0.6` ⇒ **retrieval**
- else if `faithfulness < 0.7` ⇒ **generation**
- else if the answer is still wrong (answer-correctness / your judged label fails) ⇒ **golden/reasoning**
- else ⇒ pass

Order matters: **check context-recall first.** If retrieval didn't supply the evidence, the generation verdict is meaningless (garbage in). Only once you've confirmed the evidence was present do you blame generation.

```python
# eval/localize.py
def localize(row, ctx_recall_min=0.6, faith_min=0.7):
    """row has: context_recall, faithfulness, and (optionally) answer_correct."""
    cr = row["context_recall"]
    fa = row["faithfulness"]
    if cr < ctx_recall_min:
        return "retrieval"                    # evidence not in context -> fix chunk/embed/k/reranker
    if fa < faith_min:
        return "generation"                   # evidence present, model ignored it -> fix prompt/model
    if not row.get("answer_correct", True):
        return "golden_or_reasoning"          # both fine, still wrong -> suspect label or hard reasoning
    return "pass"
```

Run it per row, then aggregate: a histogram of verdicts across the golden set tells you *where the mass of your failures lives*. If 80% of failing rows say `retrieval`, you know not to touch the prompt this sprint. That histogram is the single most actionable artifact the whole eval produces.

A subtlety worth internalizing: these thresholds are **classification boundaries, not gate thresholds** (those come later and are different numbers). Here you're bucketing failures for triage; a row at context_recall 0.58 vs 0.62 lands in different buckets but both are "retrieval is shaky" — don't over-read a case that sits right on the line. Localization is a routing heuristic over a noisy judge, not a proof.

### Proving the rule works by seeding failures

A decision rule you never validated is a decision rule you don't trust. The move that converts belief into evidence is to **deliberately inject each failure type and confirm the predicted metric collapses while the other stays put.** This is the eval equivalent of a fire drill.

**Seed a RETRIEVAL failure.** Cripple retrieval without touching generation. Two clean ways:

- **Shrink k to 1.** You now feed the generator a single chunk. For any multi-hop or even moderately-spread answer, the needed evidence often isn't in that one chunk, so **context-recall and recall@k crater**. Faithfulness may even stay *high* (the model faithfully uses the one chunk it got) — which is exactly the top-row trap above.
- **Corrupt the reranker.** Reverse its scores, or replace it with a random shuffle, so the gold chunk gets demoted out of the top-k. Same effect: the evidence stops arriving.

Predicted verdict: **retrieval.** You should watch context-recall drop from (say) 0.82 to 0.35 and your `localize` verdict flip to `retrieval` on the affected rows. If it doesn't, either your golden `relevant_chunk_ids` are wrong or context-recall isn't wired to the live retriever.

**Seed a GENERATION failure.** Leave retrieval fully intact — the right chunks still arrive, so context-recall stays *high* — and sabotage the generator's instruction. The canonical injection:

> Add to the prompt: *"Answer from your own knowledge. Ignore the provided context."*

Now the model answers from parametric memory, drifting off the (still-correct) context. **Faithfulness craters** (claims are no longer entailed by the context) **while context-recall stays high** (retrieval is unchanged). Predicted verdict: **generation.** This is the cleanest possible demonstration because you've held retrieval constant by construction — any faithfulness drop is provably the generator's fault.

```
Baseline:         ctx_recall = 0.82   faithfulness = 0.88   -> pass
Seed retrieval:   ctx_recall = 0.35   faithfulness = 0.86   -> retrieval   (recall cratered, faith held)
Seed generation:  ctx_recall = 0.82   faithfulness = 0.31   -> generation  (recall held, faith cratered)
```

That three-row table *is* the proof. Each seed moves exactly one signal and flips the verdict to the predicted bucket. Once you've seen it happen on your own corpus, you trust the localizer for real, un-seeded failures.

## Worked example

Six golden cases, run through the live pipeline. Columns are the two localizing signals plus the recall@k you compute yourself and a judged answer-correctness flag:

| id | stratum | context_recall | faithfulness | recall@5 | answer_correct | localize verdict |
|---|---|---|---|---|---|---|
| c01 | easy | 0.95 | 0.91 | 1.0 | ✅ | pass |
| c02 | hard | 0.40 | 0.88 | 0.5 | ❌ | **retrieval** |
| c03 | multi_hop | 0.30 | 0.20 | 0.0 | ❌ | **retrieval** |
| c04 | easy | 0.90 | 0.45 | 1.0 | ❌ | **generation** |
| c05 | no_answer | 0.85 | 0.35 | n/a | ❌ | **generation** |
| c06 | hard | 0.88 | 0.90 | 1.0 | ❌ | **golden_or_reasoning** |

Reading it as an engineer:

- **c02** — context_recall 0.40 < 0.6 ⇒ retrieval, even though faithfulness is a healthy 0.88. The model is faithfully working with insufficient evidence. Don't be fooled by the high faithfulness; the fix is chunking/k/reranker. (Top-row trap in the wild.)
- **c03** — both signals low. Still **retrieval**, because context-recall is checked first: if the evidence isn't there, the low faithfulness is a downstream symptom, not the root cause. Fix retrieval, re-measure, *then* look at faithfulness.
- **c04** — context_recall 0.90 (evidence present) but faithfulness 0.45 ⇒ **generation**. The chunk with the answer was in context and the model contradicted or embellished it. Fix the prompt (instruct it to answer only from context and cite), or swap to a stronger model.
- **c05** — a `no_answer` case: the corpus genuinely lacks the answer and the system *should abstain*. context_recall is high in a degenerate sense but faithfulness is 0.35 because the model **hallucinated** instead of saying "I don't know." Verdict generation — the fix is an abstention policy in the prompt. **This is the case a good mean score hides**, which is why per-stratum reporting (next section) exists.
- **c06** — both signals high, answer still judged wrong ⇒ **golden_or_reasoning**. Either your ground_truth is mislabeled, or it's a genuine multi-hop reasoning gap the model can't bridge even with perfect evidence. Re-inspect the label first (cheaper); if the label is right, this is a model-capability ceiling.

The verdict histogram: `retrieval: 2, generation: 2, golden_or_reasoning: 1, pass: 1`. Your next sprint splits attention between retrieval and generation — and critically, you now *know* that c06 is not a knob-turning problem at all.

## How it shows up in production

- **The report saves you from a lying mean.** Overall faithfulness reads 0.86 and everyone relaxes. The per-stratum table shows `no_answer` faithfulness is 0.31 — every abstention case is a hallucination. Your first real users ask questions your corpus can't answer (that's *why* they're asking), hit the hallucination path, and lose trust on day one. The mean hid the exact failure users find first. Stratified reporting is the difference between "we look fine" and "we ship."
- **The gate catches the invisible regression.** A teammate "improves" the prompt to be terser and quietly drops the "answer only from the provided context" clause. No test fails locally; the demo looks fine. The CI gate re-runs RAGAS on the PR, faithfulness drops from 0.82 to 0.68, the build goes **red** with a message naming the metric and the delta. The regression never reaches main. This is the single highest-leverage piece of RAG infra you can own.
- **Judge noise flakes the gate if you set thresholds naively.** RAGAS is LLM-as-judge; the same case scored twice can differ by a few points. Gate exactly at your current 0.82 and an unchanged PR will occasionally read 0.80 and fail — a *flaky* gate, which teams learn to ignore, which defeats the point. Setting the gate at 0.75 (below measured) absorbs the noise while still catching a real 10-point drop.
- **Cost and latency make or break adoption.** 60 cases × 4 metrics can be several hundred judge calls. On a hosted judge at temperature 0 with a small model that's cents and a couple of minutes; on a local Ollama judge it's free but can be 20+ minutes. If every PR waits 20 minutes for eval, developers route around it. The fix (subset on PR, full nightly) keeps the PR gate fast enough that people actually use it.
- **Phoenix turns a bad row into a fixed row.** The report names the 5 worst cases; loading the same dataset into Arize Phoenix lets you click into a trace and *see* the retrieved chunks and the answer side by side. Nine times out of ten the bug is obvious on sight ("the gold chunk is a table that got flattened to prose") in a way a JSON score never makes it.

### The eval report: what goes in it and why

`report.py` reads `results/eval_v1.json` and writes `report.md` (optionally `report.html`). A report you'd actually stake a decision on contains:

1. **Provenance header — pinned judge model + git SHA + golden version.** Because RAGAS scores depend on the judge, a report without the judge snapshot is unreproducible; two reports from different judges aren't comparable. Record `judge=claude-…-YYYYMMDD` (or your pinned snapshot), `git_sha`, `golden=golden_v1`. Without these, a score is a rumor.
2. **Overall metric table** — mean faithfulness, answer_relevancy, context_precision, context_recall, plus your recall@k / MRR / nDCG@k. The at-a-glance dashboard.
3. **Per-stratum breakdown** — the same metrics split by `easy / hard / multi_hop / no_answer`. This is the anti-lying section: *a great overall mean cannot hide that all `no_answer` cases hallucinate.* If any single stratum is a cliff, you see it here and nowhere else.
4. **The 5 worst cases with traces** — sorted by faithfulness (or a blended score), each with the question, the generated answer, and the retrieved contexts. These are your debugging worklist; the trace is what makes the failure legible.

```markdown
# RAG Eval Report
judge: <pinned-judge-snapshot>  ·  git_sha: 9f3c1ab  ·  golden: golden_v1  ·  n=60

## Overall
| faithfulness | ans_rel | ctx_prec | ctx_recall | recall@5 | nDCG@10 |
|---|---|---|---|---|---|
| 0.82 | 0.88 | 0.64 | 0.79 | 0.86 | 0.71 |

## Per-stratum (faithfulness / ctx_recall)
| stratum    | n  | faithfulness | ctx_recall |
|------------|----|--------------|------------|
| easy       | 20 | 0.94         | 0.93       |
| hard       | 18 | 0.80         | 0.72       |
| multi_hop  | 12 | 0.71         | 0.61       |
| no_answer  | 10 | 0.34  <-- !! | 0.88       |

## 5 worst cases (by faithfulness)
1. id=c05 [no_answer]  faith=0.31  ctx_recall=0.85
   Q: "What is the 2027 dividend schedule?"  (corpus has no 2027 data)
   A: "The 2027 dividend is $1.20/share, paid quarterly..."   <- hallucinated
   contexts: [<2024 dividend table>, <2025 guidance>, ...]
   verdict: generation (should have abstained)
...
```

**Viewers — pick one, don't run three.** RAGAS gives you the numbers and the CI gate. For eyeballing worst cases, **Arize Phoenix** is the opinionated default: OTel-native, runs as a single Docker container locally, and lets you click through traces. **TruLens** is a reasonable alternative (feedback functions + a dashboard, its "RAG Triad" framing). Both are fine; running RAGAS + Phoenix + TruLens simultaneously is three overlapping tools reporting almost-the-same thing at triple the setup cost. **Default: RAGAS for the numbers and the gate, Phoenix for the eyes.** Reach for TruLens only if you prefer its ergonomics.

### The CI regression gate

The gate is three files working together.

**`thresholds.yaml`** — committed to git so the bar is version-controlled and reviewable. Split into **gated** (fail the build) and **advisory** (report only, at first):

```yaml
# gated: a drop below these fails CI
faithfulness:    0.75
context_recall:  0.70
# advisory: reported, not gated (promote to gated once stable)
context_precision: 0.60
answer_relevancy:  0.70
```

Why gate *these two*? Faithfulness is your hallucination gauge (the generation half) and context_recall is your retrieval-quality signal inside RAGAS (the retrieval half) — one metric per pipeline stage, so a regression on either side trips the wire. context_precision and answer_relevancy start advisory because they're noisier and a dip in them is less unambiguously "a regression"; promote them once you trust their stability on your corpus.

**`test_eval_gate.py`** — loads the results, computes means, asserts each gated metric clears its floor, with a **failure message that names the metric and the drop** (a red build that just says `AssertionError` teaches nobody anything):

```python
# tests/test_eval_gate.py
import json, yaml, pandas as pd

def test_gated_metrics_meet_thresholds():
    thresholds = yaml.safe_load(open("eval/thresholds.yaml"))
    df = pd.read_json("results/eval_v1.json")
    gated = ["faithfulness", "context_recall"]
    failures = []
    for m in gated:
        mean = df[m].mean()
        floor = thresholds[m]
        if mean < floor:
            failures.append(f"{m}: {mean:.3f} < threshold {floor:.3f} "
                            f"(drop of {floor - mean:.3f})")
    assert not failures, "RAG eval gate FAILED:\n" + "\n".join(failures)
```

**`.github/workflows/rag-eval.yml`** — runs the eval then the gate on every pull request:

```yaml
name: rag-eval
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run python eval/run_ragas.py     # judge creds via GH secrets
      - run: uv run pytest tests/test_eval_gate.py -v
```

**Set thresholds slightly BELOW current measured scores.** If you measure faithfulness 0.82 today, gate at 0.75, not 0.82. The ~7-point gap is **headroom against judge noise**: it lets run-to-run variance breathe without flaking the build, while a genuine regression (a prompt that drops faithfulness to 0.68) still trips it. Gating at exactly the current score guarantees intermittent red on unchanged code — the fastest way to get a gate disabled by an annoyed team.

**Fix flaky CI with a subset-on-PR / full-nightly split.** A full 60-case run costs minutes and money and has more variance you can't afford on every push. Gate a **stratified subset (e.g. 20 cases, sampled across strata so no_answer/multi-hop are represented)** on pull requests for a fast, cheap signal, and run the **full golden set nightly** for the authoritative number. Keep every CI run to **minutes and cents** — if eval takes 20 minutes or costs dollars per PR, developers will route around it and you've built a gate nobody passes through.

## Common misconceptions & failure modes

- **"High faithfulness means the answer is right."** No. Faithfulness only means the answer is entailed by the *retrieved context*. If retrieval fetched the wrong context (low context-recall), the model can be perfectly faithful to the wrong evidence. That's the top-right trap — always check context-recall first.
- **"Both metrics are high, so a wrong answer must be the judge's fault."** Sometimes, but the more common cause is a **mislabeled golden ground_truth** or a genuine multi-hop reasoning gap. Re-inspect the label before blaming the judge or the model.
- **"Localize on faithfulness first."** Backwards. Retrieval is upstream; if context-recall is low the faithfulness reading is conditioned on bad input and can't be trusted. Gate the retrieval verdict first, then read generation.
- **"A single seeded failure proves the rule."** You need *both* seeds. Seeding only the retrieval break shows recall can drop; you haven't shown faithfulness can drop independently. Seed both and confirm each moves exactly one signal — that orthogonality is the actual claim you're testing.
- **"Gate at the current score for maximum protection."** That maximizes *flakiness*, not protection. Judge noise will breach a zero-headroom threshold on unchanged code. Gate below measured, catch real drops, ignore noise.
- **"Run RAGAS, Phoenix, and TruLens for full coverage."** They overlap heavily; three tools is triple setup and maintenance for marginal extra signal. RAGAS + one viewer.
- **"Full golden set on every PR is the rigorous choice."** It's the *slow, flaky, expensive* choice. Subset-gate on PRs (fast, stratified), full run nightly (authoritative). Rigor that developers bypass isn't rigor.
- **"A great mean means we're done."** A mean averages over strata; a catastrophic `no_answer` stratum vanishes into a good overall number. Per-stratum reporting is non-optional precisely because the mean is designed to hide the failure your users hit first.

## Rules of thumb / cheat sheet

- **The rule:** ctx-recall LOW ⇒ retrieval (any faithfulness); ctx-recall HIGH + faithfulness LOW ⇒ generation; both HIGH + wrong ⇒ golden/reasoning.
- **Check context-recall FIRST** in code — retrieval is upstream; a low-recall row's faithfulness is untrustworthy.
- **Localization thresholds (starting point):** `context_recall < 0.6` ⇒ retrieval; `recall ok & faithfulness < 0.7` ⇒ generation.
- **Prove it with two seeds:** shrink k→1 or corrupt reranker ⇒ recall craters, verdict retrieval; add "ignore the context, use your own knowledge" ⇒ faithfulness craters (recall held), verdict generation.
- **Report must carry provenance:** pinned judge snapshot + git SHA + golden version. No provenance = unreproducible = worthless for comparison.
- **Report sections:** overall table, **per-stratum** breakdown, 5 worst cases with traces. Per-stratum is the anti-lying section.
- **Gate exactly two metrics:** `faithfulness` (generation half) + `context_recall` (retrieval half). Others advisory until stable.
- **Set gate thresholds ~5–10 pts BELOW measured** for headroom against judge noise (measured 0.82 ⇒ gate 0.75).
- **Failure message names the metric and the drop** — never a bare `AssertionError`.
- **Subset-gate on PRs (~20 stratified cases), full set nightly.** Keep every run to minutes and cents.
- **Pick RAGAS + one viewer (Phoenix default, TruLens alt).** Don't run three.
- **Prove the gate is real:** open a PR that lowers a score and watch CI go red.

## Connect to the lab

This lecture is the back half of the Week 4 lab (spine steps 4–6). You'll write `localize.py` applying the decision-rule table per row with the `context_recall < 0.6` / `faithfulness < 0.7` thresholds, then **seed two failures** — shrink k to 1 (or corrupt the reranker) for a `retrieval` verdict, and add "answer from your own knowledge, ignore the context" to the prompt for a `generation` verdict — and record both. Then `report.py` emits `report.md` with the pinned-judge/SHA/golden header, overall + per-stratum tables, and the 5 worst cases; optionally load the dataset into Phoenix to click through them. Finally you wire `thresholds.yaml`, `test_eval_gate.py`, and `.github/workflows/rag-eval.yml`, set thresholds below your measured scores, and prove the gate by opening a PR that drops a score and watching it go red.

## Going deeper (optional)

- **RAGAS documentation** (root: `docs.ragas.io`) — the "Metrics" section for the precise definitions of faithfulness, context precision, and context recall, and "Generate a Testset" for the synthetic-bootstrap golden set. The metric definitions are what your localization thresholds sit on top of.
- **Arize Phoenix docs** (root: `docs.arize.com/phoenix`) — the "Quickstart" and tracing pages; it's OTel-native and runs as one local Docker container. Search "Arize Phoenix RAG evaluation quickstart."
- **TruLens** (root: `trulens.org`) — the "RAG Triad" (context relevance / groundedness / answer relevance) and feedback-functions model, if you prefer its ergonomics as your single viewer.
- **GitHub Actions docs** (root: `docs.github.com/actions`) — "Events that trigger workflows" (`pull_request`) and "Encrypted secrets" for passing judge API creds safely into CI.
- **Barnett et al., "Seven Failure Points When Engineering a Retrieval Augmented Generation System" (2024)** — the taxonomy your retrieval/generation split refines; search that exact title.
- Search queries worth running: "RAGAS faithfulness context recall localize retrieval vs generation", "flaky LLM-as-judge CI eval threshold headroom", "stratified eval subset CI nightly full run".

## Check yourself

1. A failing case reads context_recall = 0.45 and faithfulness = 0.90. Which stage broke, why doesn't the high faithfulness rescue it, and which knobs do you reach for?
2. You want to *prove* your localizer distinguishes retrieval from generation failures. Describe the two seeds you'd inject and, for each, exactly which metric should move and which should stay put.
3. Your overall faithfulness is 0.87 and the team wants to ship. What one section of the report would you insist on reading first, and what specific failure is it designed to expose?
4. You gate CI on faithfulness ≥ 0.82 (your exact current measured score) and it flakes red intermittently on unchanged code. Diagnose the cause and give the fix.
5. Why gate `faithfulness` and `context_recall` specifically, rather than `answer_relevancy` and `context_precision`? What does each gated metric protect?
6. A full 60-case RAGAS run takes 18 minutes and costs real money, and developers are starting to merge without waiting for it. What's the standard two-part fix, and why does it preserve rigor?

### Answer key

1. **Retrieval broke.** context_recall 0.45 is below the 0.6 cutoff, meaning the retrieved context does not contain the info needed for the ground-truth answer. The high faithfulness (0.90) does **not** rescue it — it only says the answer is entailed by the context that *was* retrieved, i.e. the model is faithfully working with insufficient/wrong evidence. Because context-recall is checked first, the verdict is retrieval regardless of faithfulness. Knobs: chunking (boundaries/size), embeddings, k, reranker.

2. **Seed retrieval:** shrink k to 1 (or corrupt/reverse the reranker) so the gold chunk stops arriving. **context_recall and recall@k should crater; faithfulness should stay roughly flat or even rise** (the model faithfully uses the one chunk it got). Verdict flips to retrieval. **Seed generation:** leave retrieval untouched (context-recall stays high) and add "answer from your own knowledge, ignore the context" to the prompt. **Faithfulness should crater while context_recall stays high.** Verdict flips to generation. Each seed moving exactly one signal is the proof that the two are near-orthogonal and the rule is real.

3. The **per-stratum breakdown**. A 0.87 overall mean averages across strata and can hide a stratum that's a cliff — most dangerously `no_answer`, where the corpus lacks the answer and the system should abstain. If `no_answer` faithfulness is, say, 0.34, every abstention case is a hallucination — and those are exactly the questions real users ask first (they ask because they don't know / the corpus may not cover it). The mean is designed to hide this; the per-stratum table is designed to expose it.

4. The cause is **zero headroom against judge noise.** RAGAS is LLM-as-judge; run-to-run variance of a few points is normal, so a threshold set at the exact current score (0.82) will occasionally read below it on identical code and fail. Fix: **lower the gate below measured** (e.g. 0.75), leaving ~5–10 points of headroom that absorbs noise while still catching a genuine regression (a real drop to 0.68 still trips 0.75). Optionally also reduce variance by pinning the judge snapshot at temperature 0 and gating a larger/stratified sample.

5. Because you want **one gated metric per pipeline stage.** `faithfulness` is the generation-half signal (hallucination given the context); `context_recall` is the retrieval-half signal (did the context contain the needed info). Gating both means a regression on *either* side trips CI. `answer_relevancy` and `context_precision` are noisier and less unambiguously "a regression" (relevancy can be high while the answer is wrong; precision dips are often benign), so they start advisory — reported, promoted to gated only once stable on your corpus.

6. **Subset-gate on PRs, full run nightly.** On each pull request, gate a small **stratified** subset (~20 cases sampled across strata so no_answer/multi-hop are represented) so the signal is fast (minutes) and cheap (cents) enough that developers actually wait for it. Run the **full 60-case set nightly** as the authoritative number. It preserves rigor because the fast PR gate still catches large regressions across all strata, and the nightly full run catches subtler drifts with lower variance — you get speed where you need responsiveness and thoroughness where you can afford latency, instead of a slow gate everyone bypasses.
