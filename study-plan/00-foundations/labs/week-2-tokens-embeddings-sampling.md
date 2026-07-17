# Week 2 Lab: Tokens, Cost, Embeddings & Sampling

> This week you turn the "how LLMs actually work" theory into tools you will reuse for the rest of the roadmap. You will build a **token counter + cost estimator** (so you can price a call *before* you send it), a **single `complete()` interface** over OpenAI / Anthropic / Ollama (with a fully free local path), a **logprobs viewer** that lets you *see* the next-token distribution, an **embeddings ranker** that reuses your Week-1 cosine function (plus a Matryoshka 128-dim storage trick), and a **sampling sweep** that shows temperature/top_p behavior and locks it down with an offline test. By the end, model-choice debates become numbers and generation stops being magic.
>
> Read these first — this guide assumes you have:
> - [../lectures/07-next-token-prediction.md](../lectures/07-next-token-prediction.md) — why generation is a sampled loop over a distribution.
> - [../lectures/08-attention-transformer-intuition.md](../lectures/08-attention-transformer-intuition.md) — attention, causal masking, why long prompts cost more.
> - [../lectures/09-tokenization.md](../lectures/09-tokenization.md) — BPE, why non-English/code cost more tokens, why tokenizers aren't interchangeable.
> - [../lectures/10-embeddings-and-similarity.md](../lectures/10-embeddings-and-similarity.md) — dense vectors, cosine similarity, Matryoshka truncation.
> - [../lectures/11-context-window-kv-cache.md](../lectures/11-context-window-kv-cache.md) — shared input+output budget, prefill vs decode.
> - [../lectures/12-sampling-parameters.md](../lectures/12-sampling-parameters.md) — temperature/top_p/top_k/seed and why temp=0 isn't fully deterministic.

**Est. time:** ~9 hrs · **You will need:** the `ai-foundations/` repo from Week 1 (uv-managed, `.env` loading, `metrics.py` with your NumPy `cosine`), Python 3.11+, and **Ollama** (free, local, CPU-friendly). OpenAI + Anthropic API keys are *optional* (~$0.01 for the whole lab); **every step has a free Ollama/CPU path**, so you can complete the entire Definition of Done with zero paid keys.

---

## Before you start (setup)

You are extending the Week 1 repo, not starting fresh. All commands assume you are in the repo root (`ai-foundations/`).

**1. Install the Python deps** (adds embeddings + provider SDKs on top of Week 1's `tiktoken`):

```bash
uv add openai anthropic sentence-transformers tiktoken
uv add --dev pytest            # already present from Week 1; harmless to re-run
```

**2. Install and warm up Ollama** (this is your free inference + free-path test backend):

```bash
# Install: download from https://ollama.com (Windows installer, or `brew install ollama` on macOS)
# On Windows the installer starts the background server automatically.
# On macOS/Linux you may need a server in one terminal:  ollama serve

ollama pull llama3.2:3b        # ~2 GB, runs on CPU; our default chat model
ollama pull nomic-embed-text   # optional: local embeddings if you want zero HF download
```

**3. Confirm the local server is up** (the OpenAI-compatible endpoint lives at port 11434):

```bash
curl http://localhost:11434/v1/models
```

Expected: a JSON blob listing `llama3.2:3b` (and any other pulled models). If you get "connection refused," start the server (`ollama serve` on macOS/Linux; check the tray icon on Windows).

**4. (Optional) Paid keys.** If you want to run the OpenAI/Anthropic branches, put real keys in `.env` (never commit it — it's gitignored from Week 1):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

You will finish this file this week (add three modules and one test):

```
src/ai_foundations/
  tokens.py      # NEW — tiktoken counting + dated PRICES + estimate_cost + lang breakdown
  llm.py         # NEW — unified complete(prompt, backend, **params)
  logprobs.py    # NEW — top-5 next-token viewer
  embed.py       # NEW — sentence-transformers + reuse NumPy cosine + Matryoshka 128d
  metrics.py     # from Week 1 — reuse cosine()
notebooks/
  02_llm_basics.ipynb   # NEW — cost table + lang breakdown + rankings, documented
tests/
  test_sampling.py      # NEW — offline temp=0 extraction assertion (Ollama)
data/
  sampling_sweep.jsonl  # NEW — generated output (gitignored)
```

---

## Step-by-step

### Step 1 — Token counter + cost estimator with a dated price snapshot

**What.** A module that counts tokens for any text and turns "this file + this model + N expected output tokens" into an exact dollar figure, plus a per-language token breakdown.

**Why.** Billing is per token, output usually costs more than input, and non-English/code use *more* tokens per character (see [../lectures/09-tokenization.md](../lectures/09-tokenization.md)). Counting *before* you send is how you avoid surprise bills and pick a model on cost, not vibes. Hardcoding prices with a **date comment** is deliberate: pricing drifts, so a stale number silently corrupts every estimate — the date makes the staleness visible.

**Do it.** Create `src/ai_foundations/tokens.py`:

```python
"""Token counting + cost estimation. Pricing is a DATED snapshot — verify before trusting."""
from __future__ import annotations
import tiktoken

# --- PRICES: USD per 1M tokens. SNAPSHOT copied 2026-07-09 from each vendor's
# --- pricing page. THIS WILL GO STALE. Re-check the vendor docs before quoting $.
PRICES = {
    # model               input $/1M   output $/1M   tokenizer family
    "gpt-4o":            {"in": 2.50,  "out": 10.00, "enc": "o200k_base"},
    "gpt-4o-mini":       {"in": 0.15,  "out": 0.60,  "enc": "o200k_base"},
    "claude-3-5-sonnet": {"in": 3.00,  "out": 15.00, "enc": "cl100k_base"},  # approx: Claude != tiktoken
    "llama3.2:3b":       {"in": 0.00,  "out": 0.00,  "enc": "cl100k_base"},  # local via Ollama = free
}

def get_encoding(model: str) -> tiktoken.Encoding:
    """Return the tiktoken encoding for a model. Falls back to o200k_base."""
    enc_name = PRICES.get(model, {}).get("enc", "o200k_base")
    return tiktoken.get_encoding(enc_name)

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    return len(get_encoding(model).encode(text))

def estimate_cost(text: str, model: str, expected_output_tokens: int) -> dict:
    """Return {input_tokens, output_tokens, input_usd, output_usd, total_usd}."""
    p = PRICES[model]
    n_in = count_tokens(text, model)
    n_out = expected_output_tokens
    in_usd = n_in / 1_000_000 * p["in"]
    out_usd = n_out / 1_000_000 * p["out"]
    return {
        "model": model, "input_tokens": n_in, "output_tokens": n_out,
        "input_usd": round(in_usd, 6), "output_usd": round(out_usd, 6),
        "total_usd": round(in_usd + out_usd, 6),
    }

def compare_models(text: str, expected_output_tokens: int, models=None) -> list[dict]:
    models = models or list(PRICES)
    return [estimate_cost(text, m, expected_output_tokens) for m in models]

def lang_breakdown(samples: dict[str, str], model: str = "gpt-4o") -> list[dict]:
    """samples = {label: text}. Reports tokens, chars, and tokens-per-char so you can
    *see* that code/non-English pack more tokens per character than English."""
    rows = []
    for label, text in samples.items():
        n_tok = count_tokens(text, model)
        n_chars = len(text)
        rows.append({"label": label, "tokens": n_tok, "chars": n_chars,
                     "tokens_per_char": round(n_tok / max(n_chars, 1), 3)})
    return rows
```

Then, in `notebooks/02_llm_basics.ipynb`, print a comparison table and a language breakdown. **Caveat to write in a comment:** tiktoken is accurate for OpenAI; Claude and Llama tokenize differently, so their token counts here are an *approximation* — flag it. Feed `lang_breakdown` three deliberately contrasting samples:

```python
from ai_foundations.tokens import compare_models, lang_breakdown
import pandas as pd

paragraph = "The quick brown fox jumps over the lazy dog. " * 5
code = "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n" * 5
non_english = "人工知能は世界を変えつつあります。これはテスト文です。" * 5   # Japanese

print(pd.DataFrame(compare_models(paragraph, expected_output_tokens=500)))
print(pd.DataFrame(lang_breakdown(
    {"english": paragraph, "code": code, "japanese": non_english})))
```

**Expected result.** The cost table shows `gpt-4o` costing more than `gpt-4o-mini`, and `llama3.2:3b` at `$0.00`. The breakdown shows `tokens_per_char` **highest for Japanese, then code, then English** (English English ≈ 0.25–0.3; code higher; CJK often > 1.0).

**Verify.** `count_tokens("hello world") == 2`. The DataFrame prints without KeyError. `tokens_per_char` for Japanese > that for English.

**Troubleshoot.** `KeyError` on a model → it's not in `PRICES`; add it or pass a subset to `compare_models`. First `tiktoken` call downloads the encoding — needs network once; after that it's cached.

---

### Step 2 — One interface, three backends: `complete()`

**What.** A single function `complete(prompt, backend, **params) -> str` that talks to OpenAI, Anthropic, or Ollama and returns the assistant text.

**Why.** You do not want three call sites with three SDK shapes scattered through your code. One seam means you swap backends by changing a string — and the **Ollama path is free**, so anyone without keys runs every later experiment locally. The trick: Ollama ships an **OpenAI-compatible endpoint**, so the same `openai` client hits it by just changing `base_url`.

**Do it.** Create `src/ai_foundations/llm.py`:

```python
"""Unified completion over openai / anthropic / ollama."""
from __future__ import annotations
import os
from openai import OpenAI
import anthropic

OLLAMA_BASE_URL = "http://localhost:11434/v1"

def complete(prompt: str, backend: str = "ollama", model: str | None = None,
             temperature: float = 0.7, max_tokens: int = 256, **params) -> str:
    if backend == "openai":
        client = OpenAI()  # reads OPENAI_API_KEY from env
        model = model or "gpt-4o-mini"
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, **params)
        return r.choices[0].message.content

    if backend == "ollama":
        # Same OpenAI client, different base_url. api_key can be any non-empty string.
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        model = model or "llama3.2:3b"
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, **params)
        return r.choices[0].message.content

    if backend == "anthropic":
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        model = model or "claude-3-5-sonnet-20241022"
        # Anthropic caps output with max_tokens (required) and uses .content[0].text
        r = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text

    raise ValueError(f"unknown backend: {backend!r}")
```

**Note on `max_tokens`:** always pass it (see [../lectures/11-context-window-kv-cache.md](../lectures/11-context-window-kv-cache.md)) — a looping model will otherwise generate to the context limit and bill you for it. Anthropic *requires* it.

Smoke-test the free path first, then (optionally) diff temperature behavior:

```python
from ai_foundations.llm import complete
print(complete("Say hello in one word.", backend="ollama"))          # free path
print(complete("Write a 6-word story about the sea.",
               backend="ollama", temperature=0.0))
print(complete("Write a 6-word story about the sea.",
               backend="ollama", temperature=1.0))
```

**Expected result.** A non-empty string from Ollama with **no API keys set**. The temp=0 story is stable across reruns; temp=1.0 varies. If you have keys, send the same prompt to all three and eyeball that they differ in style.

**Verify.** `complete("hi", backend="ollama")` returns a non-empty `str`. This satisfies the DoD line "returns a valid non-empty completion from Ollama with zero API keys set."

**Troubleshoot.** `Connection refused` → Ollama server not running (Step 0). `model not found` → `ollama pull llama3.2:3b`. OpenAI/Anthropic `AuthenticationError` → key missing/typo in `.env`; the point is you can skip these entirely and use `ollama`.

---

### Step 3 — See the distribution: top-5 next-token logprobs

**What.** A script that prints the top-5 candidate next tokens with their probabilities for a prompt like "The capital of France is".

**Why.** This makes [../lectures/07-next-token-prediction.md](../lectures/07-next-token-prediction.md) concrete: the model emits a distribution, you sample from it. Seeing the top-5 with probabilities is the portfolio artifact — proof you understand generation isn't a lookup. **Note:** OpenAI and Ollama expose logprobs; **Anthropic does not** the same way — write that in a comment.

**Do it.** Create `src/ai_foundations/logprobs.py`:

```python
"""Top-5 next-token viewer. Works with OpenAI and Ollama's OpenAI-compatible API.
NOTE: Anthropic's Messages API does not expose per-token logprobs the same way."""
from __future__ import annotations
import math
from openai import OpenAI

def top_next_tokens(prompt: str, backend: str = "ollama", model: str | None = None, k: int = 5):
    if backend == "ollama":
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        model = model or "llama3.2:3b"
    else:
        client = OpenAI()
        model = model or "gpt-4o-mini"

    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=k)

    # First generated token's alternatives:
    top = r.choices[0].logprobs.content[0].top_logprobs
    return [{"token": t.token, "logprob": t.logprob, "prob": math.exp(t.logprob)}
            for t in top]

if __name__ == "__main__":
    for row in top_next_tokens("The capital of France is", backend="ollama"):
        print(f"{row['token']!r:>12}  p={row['prob']:.4f}")
```

Run it:

```bash
uv run python -m ai_foundations.logprobs
```

**Expected result.** Five lines, each a candidate token and probability, with the highest-probability token being ` Paris` (or ` The`/` located` depending on model phrasing), e.g.:

```
      ' Paris'  p=0.9012
       ' The'  p=0.0203
    ' located'  p=0.0110
...
```

**Verify.** You get exactly `k` rows, probabilities are in `(0, 1]`, and the leading candidate is sensible for the prompt. The DoD asks that probabilities "sum sensibly" — the top-5 won't sum to exactly 1 (there's a long tail), but the top candidate should dominate for a factual prompt.

**Troubleshoot.** `logprobs` key missing / `None` → some Ollama model builds don't return logprobs; try `llama3.2:3b` explicitly, or run this branch against OpenAI. `top_logprobs` capped at 20 on OpenAI — `k=5` is safe.

---

### Step 4 — Embeddings + semantic ranking + Matryoshka truncation

**What.** Embed a query and ~10 short documents, rank documents by cosine similarity using your **Week-1 NumPy cosine** function, then re-rank with 128-dim truncated embeddings and show the top result is unchanged.

**Why.** This is the backbone of Phases 3–4 (vector DBs, RAG). Reusing your own `cosine` proves the metric isn't a black box (see [../lectures/10-embeddings-and-similarity.md](../lectures/10-embeddings-and-similarity.md)). Matryoshka embeddings let you keep the first 128 dims and drop the rest — a big storage/speed win for a tiny quality cost.

**Do it.** Create `src/ai_foundations/embed.py`. Reuse `cosine` from Week 1's `metrics.py`:

```python
"""Embeddings + semantic ranking. Reuses the NumPy cosine you wrote in Week 1."""
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from ai_foundations.metrics import cosine   # your Week-1 NumPy cosine(a, b)

# all-MiniLM-L6-v2: 384-dim, fast, CPU-friendly. For true Matryoshka use a model
# trained for it (e.g. nomic-embed-text-v1.5 or a Matryoshka MiniLM); truncating a
# non-Matryoshka model still *works* but degrades more — note that in your writeup.
_MODEL = None
def _model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(name)
    return _MODEL

def embed(texts: list[str], dims: int | None = None) -> np.ndarray:
    """Embed texts; if dims is set, truncate to the first `dims` (Matryoshka) and
    re-normalize so cosine stays meaningful."""
    vecs = _model().encode(texts, normalize_embeddings=True)
    if dims is not None:
        vecs = vecs[:, :dims]
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return np.asarray(vecs, dtype=np.float32)

def rank(query: str, docs: list[str], dims: int | None = None) -> list[tuple[int, float, str]]:
    q = embed([query], dims=dims)[0]
    d = embed(docs, dims=dims)
    scored = [(i, float(cosine(q, d[i])), docs[i]) for i in range(len(docs))]
    return sorted(scored, key=lambda x: x[1], reverse=True)
```

Drive it in the notebook with a hand-built set where the relevant doc is obvious:

```python
from ai_foundations.embed import rank

query = "How do I reset my password?"
docs = [
    "To reset your password, click 'Forgot password' on the login page.",  # target
    "Our office is open 9am to 5pm on weekdays.",
    "The capital of Peru is Lima.",
    "Bananas are a good source of potassium.",
    # ... ~6 more distractors
]
full = rank(query, docs)                 # 384-dim
trunc = rank(query, docs, dims=128)       # Matryoshka 128-dim

print("full  top-1:", full[0][2])
print("128d  top-1:", trunc[0][2])
assert full[0][0] == trunc[0][0], "128-dim truncation changed the top result!"
```

**Expected result.** Both rankings put the "Forgot password" doc at rank 1, with the target's cosine clearly above the distractors. The `assert` passes: truncation preserved top-1.

**Verify.** `full[0][0] == trunc[0][0]` (same doc index at rank 1). This is the DoD line "128-dim truncation keeps the top-1 the same." Storage dropped 3× (384→128 floats/doc).

**Troubleshoot.** First run downloads the model (~90 MB) — needs network once. `ImportError: cosine` → check your Week-1 signature; if yours is `cosine_similarity`, alias it. If truncation *does* change top-1, your docs are too similar — pick a clearer target, or use a Matryoshka-trained model (truncating a non-Matryoshka model degrades faster). **Fully offline alternative:** swap `sentence-transformers` for Ollama's `nomic-embed-text` via `POST http://localhost:11434/api/embeddings`.

---

### Step 5 — Sampling sweep to JSONL + an offline temp=0 extraction test

**What.** Run the same prompt across a grid of `temperature ∈ {0, 0.7, 1.2}` and `top_p ∈ {1.0, 0.5}`, save each output with its params to JSONL, and write a pytest that asserts a temp=0 **extraction** prompt returns the expected substring for ≥4/5 inputs — running **offline against Ollama** so it works in CI.

**Why.** This makes [../lectures/12-sampling-parameters.md](../lectures/12-sampling-parameters.md) tangible: low temp → stable/repetitive, high temp/low top_p → diverse. Extraction/classification wants low temp; brainstorming wants higher. The test encodes the practical rule: at temp=0, an extraction task should be reliably correct — and pins it so a regression fails loudly.

**Do it — the sweep.** A small script (or notebook cell) using your `write_jsonl` from Week 1:

```python
from itertools import product
from ai_foundations.llm import complete
from ai_foundations.io_jsonl import write_jsonl   # from Week 1

prompt = "Give me a one-sentence tagline for a coffee shop on Mars."
rows = []
for temp, top_p in product([0.0, 0.7, 1.2], [1.0, 0.5]):
    out = complete(prompt, backend="ollama", temperature=temp,
                   max_tokens=64, top_p=top_p)
    rows.append({"prompt": prompt, "temperature": temp, "top_p": top_p, "output": out})
write_jsonl("data/sampling_sweep.jsonl", rows)
print(f"wrote {len(rows)} rows")
```

**Do it — the test.** Create `tests/test_sampling.py`. Extraction prompt: pull the numeric amount out of a messy sentence; at temp=0 the model should return a string containing the number.

```python
import pytest
from ai_foundations.llm import complete

CASES = [
    ("Invoice total was $42 due Friday.", "42"),
    ("We shipped 7 boxes yesterday.", "7"),
    ("The meeting is at 3 pm sharp.", "3"),
    ("Order #128 contains 5 items.", "5"),
    ("Refund of 19 dollars was issued.", "19"),
]

def _extract(sentence: str) -> str:
    prompt = (f"Extract only the main number from this sentence, digits only, "
              f"no words:\n{sentence}")
    return complete(prompt, backend="ollama", temperature=0.0, max_tokens=8)

@pytest.mark.parametrize("sentence,expected", CASES)
def test_extraction_temp0(sentence, expected):
    assert expected in _extract(sentence)
```

Run:

```bash
uv run python scripts/sampling_sweep.py    # or run the notebook cell
uv run pytest tests/test_sampling.py -q
```

**Expected result.** `data/sampling_sweep.jsonl` has 6 lines; the two `temperature: 0.0` outputs are identical (or nearly), the `1.2` ones are visibly more varied. The test passes for ≥4/5 cases.

**Verify.** `wc -l data/sampling_sweep.jsonl` → 6. `pytest` reports at least 4 of 5 parametrized cases passing (DoD: "≥4/5 inputs"). Because it uses Ollama, it needs no keys and runs in CI.

**Troubleshoot.** All 5 fail → a 3B model may wrap the number in words; loosen `_extract` to strip non-digits, or accept `expected in "".join(filter(str.isdigit, out))`. If you want CI to *skip* when Ollama is absent, guard with `pytest.importorskip`-style check on the `/v1/models` endpoint and `pytest.skip("ollama not running")`. Add `data/*.jsonl` to `.gitignore` (generated artifact).

---

## Putting it together — a short end-to-end run

The pieces chain into one workflow: **price it → send it → inspect it → rank with it → tune it.**

```python
from ai_foundations.tokens import estimate_cost
from ai_foundations.llm import complete
from ai_foundations.logprobs import top_next_tokens
from ai_foundations.embed import rank

text = open("data/report.txt").read()

# 1) Price the call before sending (Step 1)
print(estimate_cost(text, "gpt-4o-mini", expected_output_tokens=300))

# 2) Send it — free local path (Step 2)
answer = complete("Summarize in one sentence:\n" + text, backend="ollama", max_tokens=80)
print(answer)

# 3) Peek at the model's next-token distribution (Step 3)
print(top_next_tokens("The capital of France is", backend="ollama"))

# 4) Rank docs semantically, reusing your NumPy cosine (Step 4)
print(rank("refund policy", ["Refunds within 30 days.", "We sell coffee.", "..."])[0])

# 5) Sweep sampling and save (Step 5) -> data/sampling_sweep.jsonl
```

Run headless to prove no hidden state (Week-1 discipline carries forward):

```bash
uv run jupyter execute notebooks/02_llm_basics.ipynb
uv run pytest -q
```

---

## Definition of Done — acceptance gate

Restated from the spine as verifiable checks. You are **not done** until every box is checked.

- [ ] **`tokens.py` cost table + language proof.** `compare_models` prints a table for ≥3 models, and `lang_breakdown` shows code and non-English have higher `tokens_per_char` than English — documented in a `02_llm_basics.ipynb` cell. *Verify:* the notebook cell shows Japanese `tokens_per_char` > English.
- [ ] **`llm.py` free path works.** `complete("hi", backend="ollama")` returns a non-empty string with **no API keys set**. *Verify:* run it in a shell with `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` unset.
- [ ] **Logprobs viewer.** `logprobs.py` prints the top-5 next-token candidates with probabilities for ≥1 prompt, top candidate dominating. *Verify:* `uv run python -m ai_foundations.logprobs`.
- [ ] **Embedding ranking + Matryoshka.** `embed.py` ranks the hand-built query→docs so the obviously-relevant doc is rank 1, and 128-dim truncation keeps top-1 identical. *Verify:* the `assert full[0][0] == trunc[0][0]` passes.
- [ ] **Offline sampling test.** A pytest (Ollama, no keys) shows a temp=0 extraction prompt returns the expected value for ≥4/5 inputs. *Verify:* `uv run pytest tests/test_sampling.py -q`.
- [ ] **You can explain out loud** why the same prompt costs more in Chinese than English (more tokens per character — Step 1 breakdown) and why prefill is slower per token than decode (prefill processes the whole prompt in parallel-but-compute-heavy; decode is one token at a time from the KV cache — [../lectures/11-context-window-kv-cache.md](../lectures/11-context-window-kv-cache.md)).

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `curl localhost:11434` refused | Ollama server not running | Windows: check tray icon / restart app. macOS/Linux: `ollama serve` in a terminal. |
| `model 'llama3.2:3b' not found` | model not pulled | `ollama pull llama3.2:3b` |
| `openai.AuthenticationError` | no/invalid key | Skip paid backends — use `backend="ollama"`; or fix `.env`. |
| `KeyError` in `estimate_cost` | model absent from `PRICES` | add it (with a dated comment) or pass a known-model subset. |
| Token counts feel wrong for Claude/Llama | tiktoken is OpenAI-only | It's an approximation — note it; exact counts need each vendor's tokenizer. |
| `logprobs.content` is `None` | model build doesn't emit logprobs | use `llama3.2:3b`, or run the OpenAI branch. |
| First `embed`/`tiktoken` call hangs | one-time model/encoding download | ensure network on first run; cached after. |
| 128-dim truncation changes top-1 | non-Matryoshka model + too-similar docs | use a Matryoshka model, or a clearer target doc. |
| Extraction test flaky | small model wraps number in words | strip to digits: `"".join(filter(str.isdigit, out))`. |
| Runaway generation / big bill | `max_tokens` unset | always pass `max_tokens` (Anthropic requires it). |
| `.env` shows in `git status` | not gitignored | confirm `.env` in `.gitignore` (Week 1); rotate any leaked key. |

---

## Stretch goals (optional)

- **Real tokenizers per family.** Replace the tiktoken approximation for Claude/Llama with the actual tokenizers (Anthropic's `count_tokens`, or the HF tokenizer for Llama) and quantify how far off tiktoken was (the pitfall says 20%+).
- **Cache-aware pricing.** Add input-cache pricing to `PRICES` and show the discount for a repeated long prompt (previews Phase 1 prompt caching).
- **Cosine vs dot-product benchmark.** Time normalized-dot-product vs full cosine over 10k vectors; confirm they agree when vectors are unit-normalized.
- **Matryoshka quality curve.** Sweep `dims ∈ {64, 128, 256, 384}` and plot ranking agreement (e.g., recall@1 vs full) to *see* the quality/storage tradeoff.
- **Wire the cost estimator into `complete()`.** Have `complete()` optionally return `(text, actual_cost)` by counting real output tokens after the call — the seed of Week 3's `econ cost` command.

---

**Next:** Week 3 builds on all of this — the **Model Economics CLI** (`econ cost / vram / validate`) reuses `tokens.py` directly, adds VRAM math and quantization, and finishes the Phase 0 milestone. See the Week 3 milestone section of [../00-foundations.md](../00-foundations.md).
