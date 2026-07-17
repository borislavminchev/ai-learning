# Week 4 Lab: RAG Evaluation, a CI Regression Gate, and the End-to-End Milestone Service

> Weeks 1–3 gave you a working RAG pipeline: parse → chunk → embed → hybrid retrieve → rerank → grounded generate with citations and an NLI verifier. This week you **stop trusting vibes.** You build a versioned golden set of `{query, relevant_chunk_ids, ground_truth, stratum}` cases, compute **retrieval metrics from scratch** (recall@k / MRR@k / nDCG@k), wire **RAGAS** (faithfulness, context precision, context recall, answer relevancy) over the *live* pipeline, **localize** every failure to retrieval-vs-generation with a decision rule you *prove* on two seeded breaks, emit a shareable eval report, and put a **CI gate** in front of the whole thing so a prompt tweak can never silently regress faithfulness again. Then you **fuse all four weeks** into the phase milestone: an async, `docker compose up` RAG service over a real ≥200-page corpus that answers with enforced citations and reports its own quality in CI.
>
> Read these lectures first (this guide assumes them):
> - [Lecture 17 — Two-Stage Evaluation and Retrieval Metrics](../lectures/17-two-stage-eval-and-retrieval-metrics.md)
> - [Lecture 18 — Generation Metrics with RAGAS](../lectures/18-generation-metrics-with-ragas.md)
> - [Lecture 19 — Building and Versioning Golden Sets](../lectures/19-golden-sets-construction-and-versioning.md)
> - [Lecture 20 — Failure Localization and the CI Regression Gate](../lectures/20-failure-localization-and-ci-regression-gate.md)

**Est. time:** ~9 hrs for the eval build (Steps 0–6) + ~6–8 hrs to assemble the milestone service (Step 7). · **You will need:** Python 3.10+, your Week 1–3 repo, Docker (Qdrant + the milestone `docker compose`), Git, a GitHub repo (for Actions), ~3 GB disk for models. **Everything runs free and local on a CPU laptop** — the only piece that *could* cost money is the RAGAS judge LLM, and there is a fully local path (Ollama `llama3.1:8b` + `nomic-embed-text`). A cheap pinned API judge is an optional upgrade, never required.

---

## Before you start (setup)

**What you carry over from Weeks 1–3:**
- Your chunked corpus with **stable chunk ids** (each chunk `{id, text, source, page, heading/section}`) and a content hash per chunk. If your ids are ephemeral row indices, fix that first — Week-4 re-chunking will silently zero recall@k otherwise (Lecture 19). Key relevance labels on stable content hashes / source spans.
- Your retriever (`hybrid_search` + `rerank` from Week 2) and your generator (citation-enforced, NLI-verified, from Week 3), exposed behind two callables you can invoke per query.
- Docker Qdrant running (`docker start qdrant` if it exists), Ollama installed.

**Folder layout (extending the phase repo):**
```
phase4-rag/
  eval/
    golden_v1.jsonl        # {"id","query","relevant_chunk_ids":[...],"ground_truth","stratum"}
    build_golden.py        # synthetic bootstrap (RAGAS) + chunk-id mapping + manual merge
    retrieval_metrics.py   # recall@k, mrr@k, ndcg@k from scratch
    judge.py               # pinned judge LLM + embeddings (Ollama or API), one place
    run_ragas.py           # wire RAGAS over the LIVE pipeline; emit results/eval_v1.json
    localize.py            # apply the decision-rule table per row
    report.py              # -> results/report.md (+ optional report.html)
    thresholds.yaml        # gate thresholds, committed to git
  tests/
    test_retrieval_metrics.py
    test_eval_gate.py      # the CI gate
  results/
    eval_v1.json
    report.md
  .github/workflows/rag-eval.yml
```

**Install the eval add-ons.**
```bash
# from the phase repo root, venv active
# Windows Git-Bash: source .venv/Scripts/activate   |  macOS/Linux: source .venv/bin/activate
pip install -U ragas datasets pandas numpy pytest pyyaml
# local judge path (free, offline):
ollama pull llama3.1:8b        # judge LLM (~4.7 GB)
ollama pull nomic-embed-text   # judge embeddings (~275 MB)
ollama list                    # confirm both present
```
> If you use `uv` for the milestone repo (recommended for the CI job): `uv add ragas datasets pandas numpy pytest pyyaml`.

**Pin the judge in ONE place** so every script and the report use the same snapshot (Lecture 18: an un-pinned judge re-scores your history silently).
```python
# eval/judge.py  — the single source of truth for the judge
import os
JUDGE_KIND = os.getenv("RAGAS_JUDGE", "ollama")     # "ollama" | "api"

def get_judge():
    """Return (llm, embeddings, judge_name) for RAGAS. judge_name goes in the report."""
    if JUDGE_KIND == "ollama":
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b", temperature=0))
        emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
        return llm, emb, "ollama:llama3.1:8b@local"
    # API path (optional): pin the exact snapshot, temperature 0, record it.
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings   # or your provider
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    snapshot = os.environ["JUDGE_MODEL"]                 # e.g. a dated snapshot id
    llm = LangchainLLMWrapper(ChatOpenAI(model=snapshot, temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    return llm, emb, f"api:{snapshot}"
```
> RAGAS's API evolves across versions; the wrapper import paths above match `ragas>=0.2`. If your installed version differs, check `docs.ragas.io` "Customise Models" for the exact wrapper names — the *shape* (pass an `llm` and `embeddings` into `evaluate`) is stable.

---

## Step-by-step

### Step 1 — Build the versioned golden set (~2.5 hrs)

**What.** Produce `eval/golden_v1.jsonl` with **40–80** rows, each carrying `relevant_chunk_ids` **and** `ground_truth` **and** a `stratum`. Bootstrap ~60 draft rows synthetically with RAGAS `TestsetGenerator`, map each generated context back to *your* chunk ids, then do a mandatory human-review pass and hand-add `no_answer` and multi-hop cases.

**Why.** Retrieval metrics need labeled **chunk ids**; `context_recall` and answer-correctness need **reference answers** (Lecture 19). A set with only answers can't compute recall@k; a set with only ids can't compute context_recall. You need the full row. And a happy-path-only set reads 0.95 forever and never catches the hallucination-on-empty-context failure your users hit first — so ≥5 `no_answer` cases are non-negotiable.

**Do it.** First the schema (JSONL, one object per line — greppable, diffable, appendable):
```json
{"id":"q001","query":"What is the default connect() timeout?","relevant_chunk_ids":["doc3#c17"],"ground_truth":"The connect() call defaults to a 30 second timeout.","stratum":"easy_factoid"}
{"id":"q044","query":"Which release first shipped async pools, and what Python version did it require?","relevant_chunk_ids":["doc1#c05","doc7#c22"],"ground_truth":"Async pools shipped in v2.3, which required Python 3.9+.","stratum":"multi_hop"}
{"id":"q058","query":"What is the SLA for the on-prem appliance?","relevant_chunk_ids":[],"ground_truth":"The provided documents do not contain this information.","stratum":"no_answer"}
```

Synthetic bootstrap + the critical **chunk-id mapping** step:
```python
# eval/build_golden.py
import json, hashlib, pathlib
from datasets import Dataset
from ragas.testset import TestsetGenerator
from eval.judge import get_judge

def norm(t: str) -> str:
    return " ".join(t.split()).lower()

def chunk_hash(t: str) -> str:
    return hashlib.sha256(norm(t).encode()).hexdigest()

# 1) Load YOUR chunks and build a hash -> id map (the bridge).
CHUNKS = json.loads(pathlib.Path("data/chunks.json").read_text())  # [{id, text, ...}]
HASH2ID = {chunk_hash(c["text"]): c["id"] for c in CHUNKS}

def map_contexts_to_ids(contexts: list[str]) -> list[str]:
    """generated context TEXT -> YOUR chunk id (exact hash, fallback substring)."""
    ids = []
    for ctx in contexts:
        cid = HASH2ID.get(chunk_hash(ctx))
        if cid is None:                                   # generator paraphrased/trimmed
            cid = next((c["id"] for c in CHUNKS if norm(ctx)[:120] in norm(c["text"])), None)
        if cid:
            ids.append(cid)
    return ids

def bootstrap(n=60):
    llm, emb, _ = get_judge()
    # RAGAS wants LangChain-style Documents from your corpus:
    from langchain_core.documents import Document
    docs = [Document(page_content=c["text"], metadata={"id": c["id"]}) for c in CHUNKS]
    gen = TestsetGenerator(llm=llm, embedding_model=emb)
    testset = gen.generate_with_langchain_docs(docs, testset_size=n)
    rows = []
    for i, s in enumerate(testset.to_pandas().itertuples()):
        rows.append({
            "id": f"q{i:03d}",
            "query": s.user_input,
            "relevant_chunk_ids": map_contexts_to_ids(list(s.reference_contexts)),
            "ground_truth": s.reference,
            "stratum": "auto",                            # you will re-tag in review
        })
    pathlib.Path("eval/golden_draft.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    print(f"wrote {len(rows)} draft rows -> eval/golden_draft.jsonl")

if __name__ == "__main__":
    bootstrap(60)
```

Then the **mandatory human-review pass** (do it by hand in your editor on `golden_draft.jsonl`, save as `golden_v1.jsonl`):
1. **Delete** nonsense / ambiguous / verbatim-leaky questions (rewrite leaky ones to ask from *intent*, not the chunk text).
2. **Fix** wrong `ground_truth`s — the highest-value 20 minutes of the week.
3. **Re-tag** every `stratum`: `easy_factoid` / `hard` / `multi_hop` / `no_answer`.
4. **Hand-add** ≥5 `no_answer` cases (`relevant_chunk_ids: []`, ground_truth = "not in the docs") and a few genuine multi-hop cases (two+ `relevant_chunk_ids`).
Target 40–80 clean rows. `git add eval/golden_v1.jsonl && git commit` — **bump the version, never edit in place.**

**Expected result.** `eval/golden_v1.jsonl` with 40–80 rows; ≥5 `no_answer`; ≥2 multi-hop; every row has non-empty `ground_truth` and a `stratum`.

**Verify.**
```bash
python eval/build_golden.py         # writes golden_draft.jsonl (then you hand-edit -> golden_v1.jsonl)
python - <<'PY'
import json, collections
rows=[json.loads(l) for l in open("eval/golden_v1.jsonl") if l.strip()]
strata=collections.Counter(r["stratum"] for r in rows)
no_gt=[r["id"] for r in rows if not r["ground_truth"]]
print("rows:",len(rows),"strata:",dict(strata),"missing ground_truth:",no_gt)
assert len(rows)>=40 and strata.get("no_answer",0)>=5 and not no_gt
print("OK")
PY
```

**Troubleshoot.**
- **`relevant_chunk_ids` mostly empty** → the hash bridge missed. The generator returned trimmed/paraphrased context text. Lower the substring fallback threshold, or match on source+page span instead of full text. For `no_answer` rows, empty is *correct*.
- **`TestsetGenerator` API mismatch** → RAGAS renames these across versions (`generate_with_langchain_docs` vs `generate`). Check `docs.ragas.io` "Generate a Testset" for your installed version; the concept (corpus in, `(question, contexts, ground_truth)` out) is stable.
- **Ollama testset generation is slow / low quality** → expected on an 8B local judge. Generate a smaller `n` (e.g. 30) and hand-author the rest; quality of the review pass matters more than raw count.

---

### Step 2 — Retrieval metrics from scratch (~1.5 hrs)

**What.** Implement `recall@k`, `mrr@k`, `ndcg@k` over ranked id lists vs the golden `relevant_chunk_ids`, and unit-test them against a hand-computed toy example.

**Why.** These are your **retrieval floor and ranking-quality** signals, computed independently of the LLM (Lecture 17). recall@k = ceiling on answer quality (if the fact isn't retrieved, generation can't recover it); nDCG@k is the one to trust when position matters. You implement them from scratch so you *understand* them and so a hand-computed test proves the aggregates before you trust any number.

**Do it.**
```python
# eval/retrieval_metrics.py
import math

def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    if not rel:                        # no_answer case: define recall as 1.0 iff nothing was "required"
        return 1.0
    hits = sum(1 for c in retrieved[:k] if c in rel)
    return hits / len(rel)

def mrr_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    for i, c in enumerate(retrieved[:k], start=1):
        if c in rel:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:  # binary relevance, gain=1
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(retrieved[:k], start=1) if c in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
```

The unit test against a hand-computed example (relevant at ranks 2 and 5 ⇒ MRR = 1/2 = 0.5):
```python
# tests/test_retrieval_metrics.py
from eval.retrieval_metrics import recall_at_k, mrr_at_k, ndcg_at_k

RETRIEVED = ["a", "GOLD1", "c", "d", "GOLD2", "f"]   # gold at ranks 2 and 5
RELEVANT  = ["GOLD1", "GOLD2"]

def test_recall():
    assert recall_at_k(RETRIEVED, RELEVANT, k=5) == 1.0      # both found in top-5
    assert recall_at_k(RETRIEVED, RELEVANT, k=3) == 0.5      # only GOLD1 in top-3

def test_mrr():
    assert mrr_at_k(RETRIEVED, RELEVANT, k=5) == 0.5         # first hit at rank 2

def test_ndcg():
    import math
    dcg  = 1/math.log2(2+1) + 1/math.log2(5+1)               # hits at ranks 2 and 5
    idcg = 1/math.log2(1+1) + 1/math.log2(2+1)               # ideal: ranks 1 and 2
    assert abs(ndcg_at_k(RETRIEVED, RELEVANT, k=5) - dcg/idcg) < 1e-9
```

**Expected result.** `pytest tests/test_retrieval_metrics.py` is green.

**Verify.**
```bash
pytest tests/test_retrieval_metrics.py -v
```

**Troubleshoot.**
- **nDCG > 1.0** → your IDCG is wrong (using `len(relevant)` instead of `min(len(relevant), k)`, or off-by-one in the log). The ideal ranking places all relevant items at the top.
- **`no_answer` rows crash / read 0** → decide the convention explicitly. Above, empty `relevant` ⇒ recall 1.0 (nothing was required). Alternatively skip `no_answer` rows in the *retrieval* aggregate and judge them only via faithfulness/abstention. Document whichever you pick.
- **All recall@k reads 0.0 unexpectedly** → chunk-id mismatch between golden set and live retriever (Lecture 19 pitfall). Confirm the retriever emits the *same* ids you labeled; re-run Step 1's mapping if you re-chunked.

---

### Step 3 — Wire RAGAS over the LIVE pipeline (~2 hrs)

**What.** For each golden row, **actually call your Week-3 retriever and generator**, then build a `datasets.Dataset` and run RAGAS `evaluate(...)` with the four metrics + your pinned judge. Also compute recall@k / mrr@k / nDCG@k per row from the *retrieved ids*. Merge both families into `results/eval_v1.json`.

**Why.** The deadliest process error is hand-feeding RAGAS the "ideal" contexts — then faithfulness and precision look wonderful and you've evaluated a system you don't ship (Lecture 18). The `contexts` field **must** be exactly what your live retriever returned. Computing both families per row is what makes Step 4's localization possible.

**Do it.**
```python
# eval/run_ragas.py
import json, pathlib
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from eval.judge import get_judge
from eval.retrieval_metrics import recall_at_k, mrr_at_k, ndcg_at_k

# YOUR Week 1-3 components (adapt the imports to your repo):
from retrieval.pipeline import retrieve            # -> list of chunk objects w/ .id, .text
from generation.pipeline import answer             # (query, chunks) -> answer string

K = 5
GOLDEN = [json.loads(l) for l in pathlib.Path("eval/golden_v1.jsonl").read_text().splitlines() if l.strip()]

def build_rows():
    rows = []
    for case in GOLDEN:
        ctx_chunks = retrieve(case["query"], k=K)        # LIVE retriever
        ans = answer(case["query"], ctx_chunks)          # LIVE generator (citations + NLI from Wk3)
        rows.append({
            "id": case["id"],
            "stratum": case["stratum"],
            "user_input":       case["query"],           # RAGAS>=0.2 field names
            "response":         ans,
            "retrieved_contexts": [c.text for c in ctx_chunks],
            "reference":        case["ground_truth"],
            "retrieved_ids":    [c.id for c in ctx_chunks],
            "relevant_chunk_ids": case["relevant_chunk_ids"],
        })
    return rows

def main():
    rows = build_rows()
    llm, emb, judge_name = get_judge()
    ds = Dataset.from_list([{k: r[k] for k in
          ("user_input","response","retrieved_contexts","reference")} for r in rows])
    result = evaluate(ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm, embeddings=emb)
    rag = result.to_pandas()

    # merge RAGAS scores + from-scratch retrieval metrics + provenance, per row
    out = []
    for r, (_, m) in zip(rows, rag.iterrows()):
        out.append({**{k: r[k] for k in ("id","stratum")},
            "faithfulness":     float(m["faithfulness"]),
            "answer_relevancy": float(m["answer_relevancy"]),
            "context_precision":float(m["context_precision"]),
            "context_recall":   float(m["context_recall"]),
            "recall@5": recall_at_k(r["retrieved_ids"], r["relevant_chunk_ids"], K),
            "mrr@5":    mrr_at_k(r["retrieved_ids"], r["relevant_chunk_ids"], K),
            "ndcg@10":  ndcg_at_k(r["retrieved_ids"], r["relevant_chunk_ids"], 10),
            "judge": judge_name})
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path("results/eval_v1.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} rows -> results/eval_v1.json  (judge={judge_name})")

if __name__ == "__main__":
    main()
```

**Expected result.** `results/eval_v1.json` — one object per golden case with all four RAGAS metrics **plus** recall@5 / mrr@5 / nDCG@10 **plus** the pinned judge name.

**Verify.**
```bash
python eval/run_ragas.py
python -c "import json,statistics as s; d=json.load(open('results/eval_v1.json')); \
print('n=',len(d)); \
print({m: round(s.mean(x[m] for x in d),3) for m in ['faithfulness','context_recall','context_precision','answer_relevancy']})"
```

**Troubleshoot.**
- **`context_recall` is NaN / errors** → a row is missing `reference` (ground_truth). Context recall *requires* it. Fill it in Step 1.
- **RAGAS field-name errors (`question` vs `user_input`)** → version drift. `ragas>=0.2` uses `user_input / response / retrieved_contexts / reference`; older uses `question / answer / contexts / ground_truth`. Match your installed version (`pip show ragas`).
- **Ollama judge extremely slow** → 60 cases × 4 metrics × several claims each ≈ hundreds of judge calls; local 8B can take 20–40 min. Run once, cache `results/eval_v1.json`, iterate on downstream steps without re-running. For CI, use a subset (Step 6).
- **Faithfulness suspiciously perfect (1.0 everywhere)** → you probably hand-fed contexts. Confirm `retrieved_contexts` comes from `retrieve()`, not the golden `relevant_chunk_ids`.

---

### Step 4 — Localize failures and prove it on 2 seeded breaks (~1 hr)

**What.** Apply the decision-rule table per row, then **deliberately inject** one retrieval break and one generation break and watch the predicted metric collapse while the other holds.

**Why.** A decision rule you never validated is one you don't trust (Lecture 20). context_recall and faithfulness measure different halves against different references, so they move independently — seeding each failure and confirming *exactly one signal moves* is the proof the rule is real on your corpus.

**Do it.** The localizer (check context-recall **first** — retrieval is upstream):
```python
# eval/localize.py
def localize(row, ctx_recall_min=0.6, faith_min=0.7):
    cr, fa = row["context_recall"], row["faithfulness"]
    if cr < ctx_recall_min:
        return "retrieval"                    # evidence not in context -> chunk/embed/k/reranker
    if fa < faith_min:
        return "generation"                   # evidence present, model ignored it -> prompt/model
    if not row.get("answer_correct", True):
        return "golden_or_reasoning"
    return "pass"

if __name__ == "__main__":
    import json, collections
    rows = json.load(open("results/eval_v1.json"))
    hist = collections.Counter(localize(r) for r in rows)
    print("verdict histogram:", dict(hist))
```

**Seed a RETRIEVAL failure** — cripple retrieval without touching generation, then re-run `run_ragas.py`:
```bash
# Option A: shrink k to 1 (feed the generator a single chunk)
K=1 python eval/run_ragas.py                 # (make K read from env in run_ragas.py)
# Option B: corrupt the reranker (reverse its scores) so the gold chunk is demoted.
```
Predicted: **context_recall & recall@k crater; faithfulness holds** (the model faithfully uses the one chunk it got). Verdict flips to `retrieval`.

**Seed a GENERATION failure** — leave retrieval intact, sabotage the prompt:
> Add to the generation prompt: *"Answer from your own knowledge. Ignore the provided context."*

Predicted: **faithfulness craters while context_recall stays high.** Verdict flips to `generation`. Record the three-row before/after table:
```
Baseline:         ctx_recall=0.82  faithfulness=0.88  -> pass
Seed retrieval:   ctx_recall=0.35  faithfulness=0.86  -> retrieval
Seed generation:  ctx_recall=0.82  faithfulness=0.31  -> generation
```

**Expected result.** Each seed moves exactly one signal and flips the verdict to the predicted bucket. Screenshot / commit the table.

**Verify.**
```bash
python eval/localize.py                      # baseline verdict histogram
# then run the two seeded configs and diff the histograms
```

**Troubleshoot.**
- **Retrieval seed doesn't drop context_recall** → your golden `relevant_chunk_ids` may be wrong, or context_recall isn't reading the live retriever. Re-check Step 3's `retrieved_contexts` wiring.
- **Generation seed doesn't drop faithfulness** → the instruction may be overridden downstream (your NLI verifier stripping the ungrounded answer, or a cached answer served). Bypass the cache and verifier for this experiment.
- **Row sits right on the 0.6 boundary** → localization is a routing heuristic over a noisy judge, not a proof; don't over-read a case at 0.58 vs 0.62 (Lecture 20).

---

### Step 5 — Eval report with per-stratum breakdown (~1 hr)

**What.** `report.py` reads `results/eval_v1.json` and writes `results/report.md` with a **provenance header** (judge + git SHA + golden version), an **overall** metric table, a **per-stratum** breakdown, and the **5 worst cases** by faithfulness with their traces.

**Why.** A mean lies. Overall faithfulness 0.86 relaxes everyone while `no_answer` faithfulness is 0.31 — every abstention case is a hallucination, and those are exactly the questions users ask first (Lecture 20). Per-stratum is the anti-lying section. Provenance makes the report reproducible — without the judge snapshot, a score is a rumor.

**Do it.**
```python
# eval/report.py
import json, subprocess, statistics as st, collections, pathlib
from eval.localize import localize

rows = json.load(open("results/eval_v1.json"))
sha  = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
judge = rows[0].get("judge","?")
METRICS = ["faithfulness","answer_relevancy","context_precision","context_recall",
           "recall@5","mrr@5","ndcg@10"]

def mean(rs, m): 
    vals=[r[m] for r in rs if m in r]
    return st.mean(vals) if vals else 0.0

def table(rs, cols):
    head="| "+" | ".join(cols)+" |\n|"+"|".join(["---"]*len(cols))+"|\n"
    return head+"".join("| "+" | ".join(f"{mean(rs,c):.3f}" if c in METRICS else str(c)
                        for c in cols)+" |\n" for _ in [0])

L=[f"# RAG Eval Report",
   f"judge: {judge}  ·  git_sha: {sha}  ·  golden: golden_v1  ·  n={len(rows)}\n",
   "## Overall", table(rows, METRICS),
   "## Per-stratum (faithfulness / ctx_recall)",
   "| stratum | n | faithfulness | context_recall |","|---|---|---|---|"]
by = collections.defaultdict(list)
for r in rows: by[r["stratum"]].append(r)
for s, rs in sorted(by.items()):
    flag = "  <-- !!" if mean(rs,"faithfulness")<0.6 else ""
    L.append(f"| {s} | {len(rs)} | {mean(rs,'faithfulness'):.3f}{flag} | {mean(rs,'context_recall'):.3f} |")
L += ["\n## 5 worst cases (by faithfulness)"]
for r in sorted(rows, key=lambda x:x["faithfulness"])[:5]:
    L.append(f"- id={r['id']} [{r['stratum']}] faith={r['faithfulness']:.2f} "
             f"ctx_recall={r['context_recall']:.2f} verdict={localize(r)}")
pathlib.Path("results/report.md").write_text("\n".join(L))
print("wrote results/report.md")
```

**Expected result.** `results/report.md` with the provenance header, overall table, per-stratum table (a bad stratum flagged), and the worst-5 list with per-row localize verdicts.

**Verify.**
```bash
python eval/report.py && cat results/report.md
```
Confirm the header names your judge, the per-stratum table has a `no_answer` row, and the worst-5 carry verdicts.

**Troubleshoot.**
- **`git rev-parse` fails in a non-repo** → run inside the repo, or hardcode `sha="local"` for a throwaway run.
- **Want clickable traces** → `pip install arize-phoenix`, launch the local server (`phoenix serve`), and load the same dataset to click through the worst cases side-by-side (retrieved chunks vs answer). Pick **one** viewer (Phoenix default); don't also run TruLens.

---

### Step 6 — CI regression gate (~1 hr)

**What.** Commit `thresholds.yaml` (gated: faithfulness + context_recall; advisory: the other two), write `test_eval_gate.py` with a **named failure message**, and add `.github/workflows/rag-eval.yml` that runs the eval then the gate on PRs. Prove it by opening a PR that lowers a score and watching CI go red.

**Why.** The gate catches the invisible regression — a teammate drops "answer only from the provided context" from the prompt, no local test fails, the demo looks fine, but faithfulness quietly falls from 0.82 to 0.68 (Lecture 20). Gate `faithfulness` (generation half) + `context_recall` (retrieval half): one metric per pipeline stage, so a regression on *either* side trips the wire.

**Do it.**
```yaml
# eval/thresholds.yaml  — set slightly BELOW current measured scores for headroom
# gated: a drop below these fails CI
faithfulness:    0.75
context_recall:  0.70
# advisory: reported, not gated (promote once stable on your corpus)
context_precision: 0.60
answer_relevancy:  0.70
```
```python
# tests/test_eval_gate.py
import json, yaml, statistics as st

def test_gated_metrics_meet_thresholds():
    thresholds = yaml.safe_load(open("eval/thresholds.yaml"))
    rows = json.load(open("results/eval_v1.json"))
    gated = ["faithfulness", "context_recall"]
    failures = []
    for m in gated:
        mean = st.mean(r[m] for r in rows)
        floor = thresholds[m]
        if mean < floor:
            failures.append(f"{m}: {mean:.3f} < threshold {floor:.3f} (drop of {floor-mean:.3f})")
    assert not failures, "RAG eval gate FAILED:\n" + "\n".join(failures)
```
```yaml
# .github/workflows/rag-eval.yml
name: rag-eval
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run python eval/run_ragas.py     # judge creds via GH encrypted secrets
      - run: uv run pytest tests/test_eval_gate.py -v
```
> **CI judge choice.** A local Ollama judge can't run cheaply on `ubuntu-latest`. For CI use a **cheap pinned API judge** (set `RAGAS_JUDGE=api` and `JUDGE_MODEL` + provider key as GitHub encrypted secrets), and **gate a stratified ~20-case subset** on PRs while running the **full 60 nightly** (`on: schedule:`). Keep every PR run to minutes and cents.

**Expected result.** `pytest tests/test_eval_gate.py` is green locally; a PR that lowers a score turns the Actions run **red** with a message naming the metric and the drop.

**Verify.**
```bash
python eval/run_ragas.py && pytest tests/test_eval_gate.py -v     # green on current scores
# Prove the gate bites: temporarily set faithfulness: 0.99 in thresholds.yaml, re-run -> RED
```
Then open a real PR that adds "ignore the context" to the prompt and confirm the Actions job fails.

**Troubleshoot.**
- **Gate flakes red on unchanged code** → zero headroom. You gated at the exact measured score; judge noise breaches it. Lower the gate ~5–10 pts below measured (Lecture 20).
- **CI too slow / times out** → you're running the full set on every PR. Split subset-on-PR / full-nightly.
- **Secrets not available** → API judge creds must be GitHub **encrypted secrets** referenced as `${{ secrets.JUDGE_KEY }}` in `env:`; forks don't get secrets, so gate on same-repo PRs.
- **`uv sync` fails** → ensure `pyproject.toml` + `uv.lock` are committed; `pip`-only repos can swap the two `uv` lines for `pip install -r requirements.txt` + `python`/`pytest`.

---

### Step 7 — Assemble the phase milestone service (~6–8 hrs)

**What.** Fuse all four weeks into one deployable repo: an async FastAPI RAG service over a real **≥200-page corpus** (product manual, SEC 10-K, a Kubernetes docs export, a research-lab handbook — something with tables/headers/figures). It must run from `docker compose up`, answer with enforced citations and per-stage timings, support incremental indexing, semantic caching, and report its own quality via the RAGAS CI gate you just built. This is the interview artifact.

**Why.** Each week produced a shippable slice; the milestone proves you can integrate them into a service you could actually deploy — not a notebook. The acceptance criteria below are the spine's; treat them as your contract.

**Do it.** Suggested repo layout (the spine's):
```
phase4-rag/
  README.md                 # corpus choice, arch diagram, tradeoff writeups + eval numbers
  pyproject.toml
  docker-compose.yml        # app + Qdrant (+ optional reranker/NLI service)
  Makefile                  # ingest, serve, eval, bench targets
  thresholds.yaml           # RAGAS floors enforced in CI (Step 6)
  data/raw/doc.pdf          # the real 200-page source
  data/golden.jsonl         # -> your eval/golden_v1.jsonl
  ingest/                   # partition (hi_res/Docling), chunk, metadata, upsert (incremental)
  retrieval/                # dense+bm25, rrf, rerank, query_rewrite  (Weeks 1-2)
  generation/               # prompt, citation enforcement, nli_verifier (Week 3)
  cache/semantic_cache.py   # embedding-similarity answer cache (Week 3)
  agentic/crag_pipeline.py  # CRAG grade->correct->generate (stretch benchmark)
  app/                      # FastAPI: /query, /documents, /health; OTel tracing
  eval/                     # run_ragas.py, report.py (Steps 3+5)
  results/                  # ragas_report.md/json, crag_vs_singleshot.csv
  tests/                    # citation_resolves, incremental, cache_invalidation, eval_gate
  .github/workflows/eval.yml
```

**`docker-compose.yml`** — app + Qdrant, one command up:
```yaml
services:
  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333", "6334:6334"]
    volumes: ["./qdrant_storage:/qdrant/storage"]
  app:
    build: .
    ports: ["8000:8000"]
    environment:
      - QDRANT_URL=http://qdrant:6333
      - OLLAMA_HOST=http://host.docker.internal:11434   # local LLM/embeddings on the host
    depends_on: [qdrant]
    extra_hosts: ["host.docker.internal:host-gateway"]  # Linux; Docker Desktop resolves it natively
```
> **Free/local path:** run Ollama on the host (`ollama serve`) and point the container at it via `host.docker.internal`. No GPU, no paid API. Reranker (`bge-reranker-v2-m3`) and NLI (`cross-encoder/nli-deberta-v3-base`) run in-process on CPU.

**The FastAPI surface** (`app/main.py`) — async, structured JSON, per-stage timings:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

app = FastAPI()

class Query(BaseModel):
    q: str
    k: int = 5

class Citation(BaseModel):
    chunk_id: str
    page: int | None = None

class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    faithfulness: float
    timings_ms: dict[str, float]

@app.get("/health")
async def health(): return {"status": "ok"}

@app.post("/query", response_model=Answer)
async def query(req: Query):
    t = {}; t0 = time.perf_counter()
    cached = semantic_cache.get(req.q)                    # Week 3 cache
    if cached: return cached
    chunks = await retrieve(req.q, k=req.k)               # hybrid + rerank (Weeks 1-2)
    t["retrieve"] = (time.perf_counter()-t0)*1000; t1=time.perf_counter()
    ans, cites, faith = await generate_grounded(req.q, chunks)   # citations + NLI (Week 3)
    t["generate"] = (time.perf_counter()-t1)*1000
    # citation resolution: every cited id MUST map to a retrieved chunk
    retrieved_ids = {c.id for c in chunks}
    dangling = [c.chunk_id for c in cites if c.chunk_id not in retrieved_ids]
    if dangling: raise HTTPException(422, f"dangling citations: {dangling}")
    out = Answer(answer=ans, citations=cites, faithfulness=faith, timings_ms=t)
    semantic_cache.put(req.q, out)                        # invalidated on reindex
    return out

@app.post("/documents")     # incremental: upsert or tombstone a single doc, immediate effect
async def upsert_document(...): ...   # re-embed only this doc; new fact retrievable within the request cycle
```

**Wire the whole thing together in this order:**
1. **Ingest** the real PDF with a layout model (`unstructured` `partition_pdf(strategy="hi_res")` or Docling), chunk by structure (~512–1024 tokens, small overlap), carry `{source, page, section, heading_path}` + a content hash, persist a chunk manifest. (Week 1)
2. **Index** dense + BM25 sparse in Qdrant, fuse with RRF, rerank top-50→top-5. (Week 2)
3. **Query rewrite** (multi-query or HyDE) behind an **A/B flag** so you can *measure* its lift, not assume it. (Week 2)
4. **Grounded generation** with inline `[chunk_id]`/`[page]` citations; **reject/repair** answers whose citations don't resolve. (Week 3)
5. **NLI verifier** on every answer; flag/strip unentailed sentences; expose a per-answer faithfulness score. (Week 3)
6. **Semantic cache** keyed on rewritten-query embedding (cosine ≥ threshold); **invalidate on reindex**. (Week 3)
7. **Incremental indexing** via `POST /documents` (upsert + tombstone), no full rebuild.
8. **RAGAS eval + CI gate** from Steps 1–6, exposed as `make eval`.
9. **Tests**: citation resolves (0 dangling across the golden set), incremental (new doc answerable / deleted doc un-retrievable), cache invalidation after reindex.

**Makefile targets:**
```makefile
ingest:  ; python -m ingest.run data/raw/doc.pdf
serve:   ; docker compose up --build
eval:    ; python eval/run_ragas.py && python eval/report.py
bench:   ; python agentic/bench_crag_vs_singleshot.py
```

**Agentic stretch — CRAG, benchmarked.** Add `agentic/crag_pipeline.py` (grade retrieved context → rewrite+retry or web/keyword fallback → generate) and benchmark **CRAG vs single-shot** on the *same* golden set: quality delta (RAGAS faithfulness + answer correctness) **and** cost delta (tokens, extra LLM calls, p95 latency). Write the one-paragraph verdict: when is the extra loop worth it?

**Expected result.** `docker compose up` yields a live service; `POST /query` returns structured JSON (answer, resolved citations with chunk_id+page, per-stage timings) for 20/20 sample queries; `make eval` produces the RAGAS report; the CI job fails a PR that drops a gated metric.

**Verify.**
```bash
docker compose up --build -d
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/query -H 'content-type: application/json' \
  -d '{"q":"<a fact that lives in a TABLE on a known page>","k":5}' | python -m json.tool
# expect: answer + citations[{chunk_id,page}] + timings_ms{retrieve,generate}
make eval && cat results/report.md
pytest tests/ -v          # citation_resolves, incremental, cache_invalidation, eval_gate green
```

**Troubleshoot.**
- **Container can't reach Ollama** → on Linux add `extra_hosts: ["host.docker.internal:host-gateway"]` (shown above) and confirm `ollama serve` is up on the host. On Windows/macOS Docker Desktop resolves `host.docker.internal` natively.
- **Qdrant unreachable from app** → use the *service name* URL `http://qdrant:6333` inside compose, not `localhost`.
- **Table fact not retrieved/cited** → your parser flattened the table to prose (Week 1 failure). Re-parse that page with `hi_res`/Docling so the pipe-table survives in the chunk.
- **Dangling citations (422)** → the model cited an id not in the retrieved set. Confirm you inject `[chunk_id]` markers into the prompt and parse+resolve them; repair once then reject (Week 3).
- **Cache serves stale answer after reindex** → your invalidation isn't keyed on the corpus version. Bump a corpus-version tag on every reindex and include it in the cache key; add the `test_cache_invalidation` test.

---

## Putting it together — a short end-to-end run

```bash
# --- Eval build (Steps 1-6) ---
python eval/build_golden.py                 # draft -> hand-review -> eval/golden_v1.jsonl
pytest tests/test_retrieval_metrics.py -v   # metrics verified against hand-computed example
python eval/run_ragas.py                    # LIVE pipeline -> results/eval_v1.json
python eval/localize.py                      # baseline verdict histogram
#   then run the 2 seeded configs (k=1; "ignore context") and diff histograms
python eval/report.py && cat results/report.md
python eval/run_ragas.py && pytest tests/test_eval_gate.py -v   # gate green on current scores

# --- Milestone service (Step 7) ---
make ingest                                 # layout-aware ingest of the 200-page PDF
docker compose up --build -d                # app + Qdrant live
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/query -H 'content-type: application/json' \
  -d '{"q":"<table fact>","k":5}' | python -m json.tool
make eval                                   # RAGAS report over the deployed service
pytest tests/ -v                            # all acceptance tests green
```
Then open a PR that drops "answer only from the context" from the prompt and watch the `rag-eval` Actions job go **red** — that red build is the deliverable that proves the gate is real.

---

## Definition of Done — verifiable checks

**Week 4 eval checklist (from the spine):**
- [ ] **`golden_v1.jsonl` committed:** 40–80 cases, each with `relevant_chunk_ids`, `ground_truth`, and a `stratum`; **≥5 `no_answer`** cases; multi-hop represented. *(Verify: the Step 1 assertion script.)*
- [ ] **`pytest tests/test_retrieval_metrics.py` green** — recall@k / mrr@k / nDCG@k verified against a hand-computed example (relevant at ranks 2 and 5 ⇒ MRR 0.5).
- [ ] **`run_ragas.py` runs the LIVE retriever+generator** and emits faithfulness, answer_relevancy, context_precision, **context_recall**, plus recall@k / mrr@k / nDCG@k per case into `results/eval_v1.json`. *(Verify: `contexts` come from `retrieve()`, not the golden ids.)*
- [ ] **`report.md` exists** with provenance header (pinned judge + git SHA + golden version), overall + **per-stratum** tables, and the 5 worst cases.
- [ ] **Localization demonstrated on 2 seeded failures:** k→1 (or corrupt reranker) routes to `retrieval`; "ignore the context" routes to `generation`, matching the decision rule. Three-row before/after table recorded.
- [ ] **CI gate is real:** `test_eval_gate.py` fails when faithfulness or context_recall drops below `thresholds.yaml`; the GitHub Actions workflow runs it on PRs — proven by a PR that lowers a score going red.

**Milestone acceptance criteria (from the spine):**
- [ ] `docker compose up` yields a live service; `POST /query` returns structured JSON with an answer, resolved citations (chunk_id + page), and per-stage timings for **20/20** sample queries.
- [ ] Ingestion preserves layout: a **table** and a section-scoped fact are correctly retrieved and cited (show the query + source page).
- [ ] Every cited id resolves to a retrieved chunk; an automated test proves **0 dangling citations** across the golden set.
- [ ] NLI verifier runs on every answer; you can show **≥1 hallucinated sentence flagged/stripped**, with the faithfulness score reported.
- [ ] Hybrid+rerank+rewrite **beats a dense-only baseline** on RAGAS `context_recall`/`answer_relevancy`, with a before/after table in the README.
- [ ] Semantic cache reports hit rate and measured latency/cost saved; a test proves entries are **invalidated after a reindex**.
- [ ] Incremental indexing: adding a doc makes a new fact answerable within one request; deleting it makes the fact un-retrievable — **both proven by tests**.
- [ ] `make eval` produces a RAGAS report; the CI job **fails a PR** that drops any metric below `thresholds.yaml`.
- [ ] **CRAG vs single-shot** benchmark table exists (quality delta + cost delta) with a written recommendation. *(Stretch.)*
- [ ] `README.md` explains chunking strategy, retrieval config, why each knob was set, and every tradeoff — the "why," not just the "what."

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `context_recall` NaN / errors | row missing `ground_truth` | fill reference answers; context_recall requires them |
| faithfulness = 1.0 everywhere | hand-fed ideal contexts | `retrieved_contexts` must come from the LIVE retriever |
| recall@k reads 0.0 everywhere | chunk-id mismatch (re-chunked) | key labels on stable content hashes; re-run Step 1 mapping |
| nDCG > 1.0 | wrong IDCG | use `min(len(relevant), k)` items at the top |
| RAGAS field errors (`question`/`user_input`) | version drift | match `ragas` version fields (`pip show ragas`) |
| RAGAS painfully slow | local judge, hundreds of calls | run once, cache JSON; subset for CI |
| retrieval seed doesn't drop ctx_recall | golden ids wrong / not live | re-check `relevant_chunk_ids` + live wiring |
| generation seed doesn't drop faithfulness | verifier/cache masking it | bypass cache + NLI strip for the experiment |
| gate flakes red on unchanged code | zero headroom | gate ~5–10 pts below measured; pin judge, temp 0 |
| CI too slow / times out | full set on every PR | subset-on-PR (~20 stratified), full nightly |
| container can't reach Ollama | host networking | `host.docker.internal` + `ollama serve` on host |
| app can't reach Qdrant | using `localhost` in compose | use service name `http://qdrant:6333` |
| table fact not retrieved | parser flattened the table | re-parse page with `hi_res`/Docling; keep pipe-table |
| dangling citations (422) | model cited a non-retrieved id | inject `[id]` markers, parse+resolve, repair-then-reject |
| stale cache after reindex | cache key ignores corpus version | include a corpus-version tag in the cache key |

---

## Stretch goals (optional)

- **Promote advisory metrics to gated.** Once `context_precision` / `answer_relevancy` are stable on your corpus over several runs, move them into the gated block of `thresholds.yaml` and re-set floors below measured.
- **Calibrate the judge against humans.** Hand-label ~20 cases (grounded/not, relevant/not), compute Cohen's κ between judge and human, and record it in the report before you trust the gate (Lecture 18; deeper in Phase 7).
- **Arize Phoenix worst-case triage.** Load `results/eval_v1.json` into Phoenix (`pip install arize-phoenix`, local server) and click through the 5 worst cases' traces — usually the bug ("gold chunk is a flattened table") is obvious on sight.
- **Subset-vs-full variance study.** Run the 20-case PR subset 5× and the full 60 once; report the run-to-run variance of faithfulness to justify your gate headroom empirically.
- **CRAG cost/quality frontier.** Extend the milestone benchmark: sweep the CRAG `tries` cap (0/1/2) and plot faithfulness gain vs added p95 latency and tokens — find where the extra loop stops paying for itself.
- **`answer_correctness` in the panel.** Add RAGAS `answer_correctness` (needs ground_truth) to catch the "faithful to a wrong chunk" case that faithfulness alone can't — fills the bottom-right cell of the localization table.
