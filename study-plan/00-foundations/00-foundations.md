# Phase 0 — Foundations of LLMs & ML for Engineers

**Phase goal:** Build the mental model and the toolchain that every later phase assumes. By the end you can set up a reproducible Python/AI environment, reason about tokens/cost/VRAM/sampling from first principles, call hosted and local models with confidence, and ship a "model economics CLI" that turns fuzzy model-choice debates into numbers. You will understand *why* an output cost what it did and how to debug it — not guess.

Prev: (start here) · Next: `01-prompting-context.md`

## Prerequisites
- Comfortable writing Python (functions, classes, virtualenvs, pip) and using a terminal.
- Git basics (clone, branch, commit) and a GitHub account.
- A laptop with 16 GB RAM (8 GB works with smaller models). No GPU required — every lab has a CPU/local path. A credit card or free-tier keys for OpenAI + Anthropic help for one lab (~$2 total); an Ollama local model is the free substitute.
- You do NOT need calculus, linear algebra proofs, or ML research background. We teach the engineering-relevant intuition only.

## Time budget
3 weeks × ~10–15 hrs/week (≈ 33–45 hrs total). Split target per week: **~35% theory / ~65% hands-on**.

## How to use this file
Work top to bottom, one week at a time; do not skip the Labs — they are the point. Timebox theory to the stated hours and stop reading when the clock runs out; you learn the rest by building. Treat each **Definition of Done** as a hard gate: if you can't check every box, you're not done. Keep everything in one git repo (`ai-foundations/`) so Week 3's milestone can reuse Weeks 1–2.

> **📚 Course materials.** This file is the **spine** — a weekly plan and a *recap*. The deep material lives alongside it:
> - **Lectures** (textbook-style, one per concept): [`lectures/`](lectures/00-index.md) — read these for the *why* and the mechanism.
> - **Lab guides** (step-by-step, with explanations): [`labs/`](labs/) — follow these to *build*.
>
> Each week below: read the linked **lectures** first (the Theory section here is their recap), then work the linked **lab guide** (the Lab section here is its summary).

---

## Week 1 — Reproducible environment + just-enough classical ML

### Objectives
By the end of the week you can:
1. Create a fully reproducible Python AI project with `uv` (locked deps, `.env` secrets, no hardcoded keys) and explain why version pinning matters.
2. Run a Jupyter notebook **headless** via `nbconvert`/`jupyter execute` and prove it has no hidden state ("Restart & Run All" is green from the CLI).
3. Manipulate data with NumPy (shape/dtype/broadcasting/axis) and pandas, and read/write **JSONL** — the dominant LLM dataset format.
4. Split data into train/val/test correctly, name at least three concrete ways **data leakage** happens, and spot overfitting from a learning curve.
5. Compute precision/recall/F1 and a ranking metric (nDCG@k / recall@k) by hand and with scikit-learn, and say which metric matches which business cost.

### Theory (~4 hrs)
> **📖 Deep lectures for this week** (read first): [1 · Reproducible envs & uv](lectures/01-reproducible-envs-uv.md) · [2 · Notebooks without hidden state](lectures/02-notebooks-without-hidden-state.md) · [3 · NumPy / vectorized thinking](lectures/03-numpy-tensors-vectorized-thinking.md) · [4 · Just-enough classical ML](lectures/04-just-enough-classical-ml.md) · [5 · Data leakage](lectures/05-data-leakage.md) · [6 · Metrics that match cost](lectures/06-metrics-that-match-cost.md). The bullets below are the recap.

- **Reproducible envs & the `uv` toolchain.** `uv` is the 2025 default: a single fast Rust tool that replaces `pip`/`venv`/`pipenv`/`pyenv`. Why it matters: torch/transformers/CUDA break across versions constantly, so a committed lockfile is the difference between "works on my machine" and "works." Read the **Astral `uv` documentation** (docs.astral.sh/uv) — sections *Projects*, *Locking and syncing*. Skim the **Twelve-Factor App** principle "Store config in the environment" for the secrets rationale.
- **Notebooks without hidden state.** Notebooks run cells out of order, leaving variables alive that no longer exist in the code — the #1 source of "it worked yesterday." Rule: nothing is trusted until *Restart Kernel & Run All* passes, ideally from the CLI so CI can enforce it. Reference: **Jupyter documentation** (`jupyter execute` / `nbconvert --execute`) and Joel Grus's well-known talk "I Don't Like Notebooks" for the failure modes (search that title).
- **Vectorized thinking with NumPy.** The tensor mental model you'll use for the rest of the roadmap: everything is an n-dimensional array with a `dtype` and `shape`; operations broadcast along axes instead of looping. Why: it's how you'll read model code and debug shape/device bugs later. Read the **NumPy "Absolute Beginners" + "Broadcasting"** guides (numpy.org/doc). pandas for tabular wrangling; **JSONL** (one JSON object per line) because it streams, appends, and is what every LLM fine-tune/eval dataset uses.
- **Just-enough classical ML.** Supervised vs unsupervised; **train/val/test discipline** and *why* (val to tune, test touched once); **data leakage** as the real-world killer (fitting the scaler on all data, target leakage from post-outcome features, temporal leakage, duplicate rows across splits). Baselines before models — most "AI" tasks are still logistic regression or gradient-boosted trees; reach for LLMs only when the input is unstructured language. Overfitting/underfitting and the bias-variance idea (matters later: tiny fine-tune sets overfit fast). Reference: **scikit-learn User Guide** (*Cross-validation*, *Metrics*) and the book *Hands-On Machine Learning* (Géron), chapters 1–3.
- **Metrics that match business cost.** Precision (of what I flagged, how much was right) vs recall (of what mattered, how much I caught) vs F1; confusion matrix; PR-AUC for rare positives. Ranking metrics — **recall@k, MRR, nDCG@k** — matter enormously for RAG in Phase 3–4, so meet them now. Know why BLEU/ROUGE are weak for open-ended text and why LLM-as-judge dominates (Phase 7).

### Lab (~8 hrs)
> **🛠️ Full step-by-step guide:** [Week 1 Lab — Reproducible Env + Classical ML Basics](labs/week-1-repro-env-and-ml-basics.md). The steps below are the summary; the guide walks each one with what/why/expected-output/troubleshooting.

Build the repo skeleton the whole phase reuses.

**1. Project init with uv (30 min).**
```bash
# install uv (macOS/Linux); Windows: use the PowerShell installer on the uv docs page
curl -LsSf https://astral.sh/uv/install.sh | sh

uv init ai-foundations && cd ai-foundations
uv add numpy pandas scikit-learn jupyterlab python-dotenv tiktoken
uv add --dev pytest nbconvert
git init && git add -A && git commit -m "chore: uv project scaffold"
```
Create this layout:
```
ai-foundations/
  pyproject.toml        # uv-managed
  uv.lock               # committed — reproducibility
  .env                  # NEVER committed (put in .gitignore)
  .env.example          # committed template
  data/                 # gitignored raw/derived data
  notebooks/
    01_numpy_pandas.ipynb
  src/ai_foundations/
    __init__.py
    io_jsonl.py
    metrics.py
  tests/
    test_metrics.py
    test_io_jsonl.py
```
Add `.env`, `data/`, `.venv/`, `*.ipynb_checkpoints` to `.gitignore`. Put `OPENAI_API_KEY=` and `ANTHROPIC_API_KEY=` in `.env.example`.

**2. Secrets done right (20 min).** In `src/ai_foundations/__init__.py` load `.env` with `python-dotenv` and expose a helper that reads keys from the environment and raises a clear error if missing — never hardcode a key, never print it.

**3. JSONL I/O module (1 hr).** In `io_jsonl.py` write `read_jsonl(path) -> Iterator[dict]` (streaming, one `json.loads` per line, skips blank lines) and `write_jsonl(path, rows)`. This is infrastructure you'll reuse in every phase.

**4. NumPy/pandas notebook (2.5 hrs).** In `01_numpy_pandas.ipynb`: create arrays, demonstrate `shape`/`dtype`, broadcasting a `(3,1)` with `(1,4)`, `axis=0` vs `axis=1` reductions, and cosine similarity between two vectors *implemented with NumPy* (`np.dot(a,b)/(norm(a)*norm(b))`) — you'll reuse this exact function for embeddings in Week 2. Load a small CSV (e.g., the scikit-learn breast-cancer or Titanic dataset) into pandas, do a group-by, and export a slice to JSONL via your module.

**5. Classical ML + metrics (2.5 hrs).** In the same or a second notebook: `train_test_split` with `stratify`, fit `LogisticRegression` as a baseline, then plot a train-vs-val learning curve to *see* overfitting. Compute precision/recall/F1 and the confusion matrix with scikit-learn. Then, in `src/.../metrics.py`, **implement from scratch** (NumPy only): `precision_recall_f1(y_true, y_pred)` and `ndcg_at_k(relevances, k)`. Write `tests/test_metrics.py` asserting your functions match `sklearn.metrics` on a fixed example.

**6. Prove no hidden state (30 min).** Run headless and commit the executed output:
```bash
uv run jupyter execute notebooks/01_numpy_pandas.ipynb   # or: uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_numpy_pandas.ipynb
uv run pytest -q
```

### Definition of Done
- [ ] `git clone` + `uv sync` + `uv run pytest -q` passes green on a clean checkout; `uv.lock` is committed.
- [ ] No API key or secret appears anywhere in git history (`git log -p | grep -i key` finds only the `.env.example` placeholder).
- [ ] `uv run jupyter execute notebooks/01_numpy_pandas.ipynb` completes with exit code 0 (proves no hidden state).
- [ ] `read_jsonl`/`write_jsonl` round-trip 100 records losslessly (asserted in a test).
- [ ] Your hand-written `precision_recall_f1` and `ndcg_at_k` match scikit-learn / a hand-computed value to 1e-6 in tests.
- [ ] You can state, in one sentence each, three distinct data-leakage mechanisms and which metric you'd optimize for a fraud detector vs a document search box.

### Pitfalls
- Committing `.env` or pasting a key into a notebook cell — rotate the key immediately if you do; git history is forever.
- Fitting a scaler/encoder on the full dataset *before* splitting — classic leakage that inflates val scores.
- Trusting notebook results after running cells out of order; always Restart & Run All before believing a number.
- Reporting accuracy on an imbalanced dataset (99% "no fraud" gets 99% accuracy doing nothing) — use precision/recall/PR-AUC.
- Confusing `axis=0` (down columns) with `axis=1` (across rows) in NumPy/pandas reductions — the silent source of wrong aggregates.

### Self-check
1. Why do we need a separate val AND test set — what goes wrong with only two splits?
2. Your recall@5 is 0.95 but precision is 0.10. What does that mean and when is it acceptable?
3. What exactly does `uv.lock` guarantee that `pyproject.toml` alone does not?
4. Give a task where an LLM is the wrong tool and logistic regression is right.
5. How would you detect that a notebook has hidden state without reading every cell?

---

## Week 2 — How LLMs work for engineers (tokens, embeddings, sampling)

### Objectives
By the end of the week you can:
1. Explain, at engineer depth, the decoder-only transformer loop: input → tokens → attention → logits → softmax → sampling → next token, and why generation is autoregressive.
2. Tokenize text with `tiktoken`, count tokens *before* sending, and compute exact input/output cost for a call across models.
3. Generate embeddings with `sentence-transformers`, compute cosine similarity, and rank documents by semantic similarity to a query.
4. Explain the context window + KV cache: why input+output share a budget, why prefill is slow and decode fast, and why long context eats VRAM.
5. Predictably control output by choosing temperature/top-p/top-k/max_tokens/stop/seed for a given task, and explain why temp=0 is *not* fully deterministic.
6. Call OpenAI, Anthropic, and a local Ollama model through one interface, and inspect logprobs to *see* the model's next-token distribution.

### Theory (~5 hrs)
> **📖 Deep lectures for this week** (read first): [7 · Next-token prediction](lectures/07-next-token-prediction.md) · [8 · Attention & the transformer](lectures/08-attention-transformer-intuition.md) · [9 · Tokenization](lectures/09-tokenization.md) · [10 · Embeddings & similarity](lectures/10-embeddings-and-similarity.md) · [11 · Context window & KV cache](lectures/11-context-window-kv-cache.md) · [12 · Sampling parameters](lectures/12-sampling-parameters.md). The bullets below are the recap.

- **Next-token prediction is the one thing LLMs do.** Everything else is emergent. The model outputs a probability distribution over the vocabulary for the next token; you sample one, append it, and repeat. Hallucination is *fluent guessing from that distribution*, not a lookup failure. This single fact explains most model behavior. Reference: Andrej Karpathy's talk **"Intro to Large Language Models"** and his **"Let's build the GPT Tokenizer"** video (search those titles on YouTube) — the best engineer-level intuition available.
- **Transformer/attention intuition (no proofs).** Decoder-only autoregressive stack (GPT/Llama family). Attention = each token looks at earlier tokens via Query-Key-Value matching and pulls in relevant context; **causal masking** stops it peeking at the future; multi-head = several attention "views" at once. Why you care: attention is quadratic in sequence length (long prompts cost more and get slow), and "lost in the middle" — models attend best to the start and end of context — is a real retrieval bug you'll design around. FlashAttention is a memory-efficient *implementation*, not different math. Encoder-only (BERT) = understanding/embeddings; decoder-only (GPT) = generation. Reference: **Jay Alammar's "The Illustrated Transformer"** (jalammar.github.io) — read once for the picture, don't rabbit-hole.
- **Tokenization.** Models see tokens, not characters. BPE (used by GPT/tiktoken) merges frequent byte pairs; ~4 chars ≈ 1 token in English, worse for code and much worse for non-English (a Chinese sentence can be 2–3× the tokens of its English translation). **Billing is per token and output tokens usually cost more than input.** Tokenizers are NOT interchangeable across model families. Reference: OpenAI's **tiktoken** GitHub repo and the **"What are tokens"** page in the OpenAI docs.
- **Embeddings & cosine similarity.** An embedding maps text to a dense vector where geometric closeness ≈ semantic similarity. Cosine similarity (angle, ignores magnitude) is the usual metric; normalize vectors so dot product = cosine. Use the *same* model for corpus and query. Matryoshka embeddings let you truncate dimensions to trade a little quality for a lot of storage. This is the backbone of Phase 3 (vector DBs) and Phase 4 (RAG). Reference: **sentence-transformers documentation** (sbert.net) and the **MTEB leaderboard** on Hugging Face (as a starting point — always validate on your own data).
- **Context window & KV cache.** The context window is a shared budget for input + output; if the prompt is huge there's little room to answer. **Prefill** (processing your prompt) is compute-heavy and slow-per-token; **decode** (generating) is fast per token but sequential. The KV cache stores attention keys/values so each new token doesn't reprocess the whole prompt — it's why long contexts eat VRAM and why prompt caching (Phase 1) saves money.
- **Sampling parameters.** `temperature` (flattens/sharpens the distribution; 0 = greedy-ish), `top_p` (nucleus: smallest set of tokens summing to p), `top_k` (top k tokens), `max_tokens` (output cap — set it or you'll pay for runaway), `stop` sequences, `seed` (best-effort reproducibility). Match them to the task: extraction/classification → low temp; brainstorming → higher temp/top_p. **temp=0 is not fully deterministic** (GPU non-determinism, batching, MoE routing).

### Lab (~9 hrs)
> **🛠️ Full step-by-step guide:** [Week 2 Lab — Tokens, Cost, Embeddings & Sampling](labs/week-2-tokens-embeddings-sampling.md). The steps below are the summary.

**Setup (30 min).**
```bash
uv add openai anthropic sentence-transformers tiktoken
# Ollama: install from ollama.com, then:
ollama pull llama3.2:3b        # ~2GB, runs on CPU/laptop
ollama pull nomic-embed-text   # local embeddings, optional
```

**1. Token counter + cost estimator (2 hrs).** New module `src/ai_foundations/tokens.py`. Use `tiktoken.encoding_for_model(...)` to count tokens for a text file. Build a small `PRICES` dict (input $/1M, output $/1M) for 3–4 models — **hardcode today's published numbers and add a comment with the date you copied them; pricing changes, so this is a snapshot.** Function `estimate_cost(text, model, expected_output_tokens)` returns input tokens, projected output tokens, and total $. Print a comparison table across models for the same input. Feed it the same paragraph in English, a code snippet, and (paste) a non-English sentence to *see* the tokens-per-language difference.

**2. One interface, three backends (3 hrs).** New module `src/ai_foundations/llm.py` exposing `complete(prompt, backend, **params) -> str` where `backend ∈ {"openai","anthropic","ollama"}`.
- OpenAI via the official `openai` SDK (Chat Completions).
- Anthropic via the official `anthropic` SDK (Messages API).
- Ollama via its OpenAI-compatible endpoint (`base_url="http://localhost:11434/v1"`, any api_key string) using the same `openai` client — this is the free path if you have no API keys.
Send the same prompt to all three at temperature 0 and 1.0 and diff the outputs. **Cost note:** OpenAI+Anthropic here is ~$0.01; if you skip paid keys, run all experiments against Ollama only — the concepts are identical.

**3. See the distribution: logprobs (2 hrs).** Call OpenAI Chat Completions with `logprobs=True, top_logprobs=5` on a prompt like "The capital of France is". Print the top-5 candidate next tokens with their probabilities (exponentiate the logprobs). This is the portfolio artifact from the roadmap: a script that shows the top-5 next-token candidates so you can *see* generation. (Ollama also returns logprobs; Anthropic does not expose them the same way — note that in a comment.)

**4. Embeddings + semantic ranking (1.5 hrs).** `src/ai_foundations/embed.py`: load `sentence-transformers` model `all-MiniLM-L6-v2` (fast, CPU-friendly, 384-dim). Embed a query and ~10 short documents, reuse your Week-1 NumPy cosine function to rank documents by similarity, and print the ranking. Then re-embed with a Matryoshka-capable model truncated to 128 dims and show the ranking is nearly identical at a fraction of the storage.

**5. Sampling sweep (1 hr, wrap into a test).** Run the same creative prompt at temperature {0, 0.7, 1.2} and top_p {1.0, 0.5}; save outputs to JSONL with their params. Write a `pytest` that asserts a deterministic-ish extraction prompt at temp=0 returns the expected substring for 5/5 inputs (uses Ollama so it runs offline in CI).

### Definition of Done
- [ ] `tokens.py` prints a cost table for ≥3 models and correctly shows non-English/code text uses more tokens than equivalent English (documented in a notebook cell).
- [ ] `llm.py` returns a valid non-empty completion from Ollama with zero API keys set (the free path works).
- [ ] The logprobs script prints the top-5 next-token candidates with probabilities that sum sensibly for at least one prompt.
- [ ] `embed.py` ranks a hand-built query→documents set so the obviously-relevant doc is rank 1; 128-dim truncation keeps the top-1 the same.
- [ ] A `pytest` (offline, Ollama) shows a temp=0 extraction prompt returns the expected value for ≥4/5 inputs.
- [ ] You can explain, out loud, why the same prompt costs more in Chinese than English and why prefill is slower per token than decode.

### Pitfalls
- Using the wrong tokenizer for a model family (tiktoken counts for OpenAI; Llama/Claude tokenize differently) — your cost estimate can be off by 20%+.
- Forgetting to set `max_tokens` — a looping model can generate to the context limit and bill you for it.
- Comparing embeddings from *different* models, or comparing query and corpus embedded by different models — the vectors aren't in the same space.
- Assuming temperature 0 is byte-for-byte reproducible across runs/providers — it isn't; pin `seed` where supported and still expect drift.
- Not normalizing vectors before using dot product as cosine similarity.

### Self-check
1. Walk through what happens to the string "unhappiness" from raw text to a sampled next token.
2. Why does output typically cost more than input, and why does that change how you design prompts?
3. What is the KV cache and why does a 100k-token context need so much VRAM?
4. When would you pick top_p=0.9 over temperature=0.7, and when does it not matter?
5. Your semantic search returns garbage. Name three model/embedding causes before you blame the data.

---

## Week 3 — Model & deployment landscape + the model economics CLI

### Objectives
By the end of the week you can:
1. Explain base vs instruct/chat vs reasoning models and decide when the extra "thinking tokens" of a reasoning model are worth the cost/latency.
2. Reason about open vs closed models on the real axes (data sensitivity, cost-at-volume, latency/SLA, license) and name where gateways (OpenRouter, LiteLLM) fit.
3. Explain quantization families (GGUF/AWQ/GPTQ; fp16→int8→int4), pick int4 as the usual sweet spot, and compute VRAM for a given param count + quantization.
4. Load open models with Hugging Face `transformers` at fp16/int8/int4, apply the correct **chat template**, and observe the VRAM/quality tradeoff.
5. Get **structured output** and a basic **tool call** from a model — the foundation for Phases 2 and 6.
6. Ship the **Model Economics CLI**: given a text file, report token counts + $ cost across models and estimate VRAM for a param count + quantization, validated against actually loading a model.

### Theory (~5 hrs)
> **📖 Deep lectures for this week** (read first): [13 · Base vs instruct vs reasoning](lectures/13-model-types-base-instruct-reasoning.md) · [14 · Open vs closed & gateways](lectures/14-open-vs-closed-and-gateways.md) · [15 · Quantization & VRAM math](lectures/15-quantization-and-vram-math.md) · [16 · Chat templates, structured output & tools](lectures/16-chat-templates-structured-output-tools.md). The bullets below are the recap.

- **Base vs instruct vs reasoning models.** *Base* = raw next-token predictor, completes text, not helpful out of the box. *Instruct/chat* = fine-tuned to follow instructions in a message format (the default you use). *Reasoning* models (OpenAI o-series, Claude extended thinking, Gemini thinking, DeepSeek-R1) spend extra "thinking tokens" before answering — better on hard multi-step problems, but slower and you pay for the hidden tokens. Rule: use reasoning models when the task genuinely needs multi-step deduction; don't pay the thinking tax for simple extraction. Reference: the provider docs' model pages (platform.openai.com/docs/models, docs.anthropic.com) — read a model card for context window, output cap, and pricing.
- **Open vs closed & gateways.** Closed (OpenAI/Anthropic/Gemini): best quality, zero ops, data leaves your building, price set by vendor. Open (Llama, Qwen, Mistral, DeepSeek, Gemma): you host, control data, pay for GPUs, own the ops. Decide on data sensitivity, volume/cost curve (APIs cheap at low volume, self-host wins at scale — quantified in Phase 10), latency/SLA, and license (check commercial terms — not all "open" weights are). **Gateways** (OpenRouter, LiteLLM) give one API across many providers with fallback/routing — you'll use LiteLLM heavily later; know it exists now.
- **Quantization + VRAM math.** Parameters are stored as numbers; fewer bits = less memory, slightly less quality. fp16 = 2 bytes/param, int8 = 1, int4 = 0.5. **Rule of thumb VRAM (weights only) = params(B) × bytes/param.** So a 7B model ≈ 14 GB at fp16, ≈7 GB int8, ≈3.5 GB int4 — plus KV cache + activations (add ~20–40% headroom for real serving). int4 is the usual sweet spot for local. Formats: **GGUF** (llama.cpp/Ollama, CPU+GPU, the local default), **AWQ**/**GPTQ** (GPU-optimized 4-bit), **bitsandbytes** (on-the-fly int8/int4 in transformers). Reference: the **llama.cpp** and **Hugging Face `transformers` quantization** docs; **Ollama** model library for ready GGUF quants.
- **Chat templates.** Instruct models were trained with a *specific* format (special tokens marking system/user/assistant turns). Send the wrong format and quality silently craters. `transformers` handles this via `tokenizer.apply_chat_template(messages, ...)` — never hand-format. This is why raw open models need care that hosted chat APIs hide.
- **Structured output & tool calling (intro).** Two of the highest-leverage reliability features. *Structured output* forces the model to return JSON matching a schema (OpenAI strict `json_schema`, Anthropic tool `input_schema`); *tool calling* lets the model request that your code run a function and return the result. You'll go deep in Phase 2 — here you just make each work once and see the shape of the request/response. Reference: OpenAI **Structured Outputs** guide and Anthropic **Tool use** guide in their official docs.

### Lab (~9 hrs) — Milestone: the Model Economics CLI
> **🛠️ Full step-by-step guide:** [Week 3 Lab — Build the Model Economics CLI](labs/week-3-model-economics-cli.md). The steps below are the summary.

Build `src/ai_foundations/econ_cli.py`, a real CLI (use `argparse` or `typer` — `uv add typer`). This is the Phase 0 portfolio milestone from the roadmap.

**1. `cost` command (2 hrs).** `econ cost --file report.txt --models gpt-4o,claude-sonnet,llama3.1-8b --output-tokens 500`. Reuse Week 2's `tokens.py`. Print a table: model | input tokens | est output tokens | input $ | output $ | total $. Handle that different families need different tokenizers (use tiktoken for OpenAI; for others, note the approximation and use a documented chars/token ratio). Add `--lang-breakdown` that reports tokens-per-char so English vs code vs non-English is visible.

**2. `vram` command (2 hrs).** `econ vram --params 7 --quant int4 --seq-len 4096 --batch 1`. Compute: weights = params × bytes(quant); KV cache ≈ `2 × layers × seq_len × hidden × bytes` (use published config values for a known 7B, e.g. Llama-3-8B, or accept `--layers/--hidden` flags); activations headroom ≈ 20%. Print total GB and whether it fits common GPUs (e.g., 8/16/24 GB). Support quant ∈ {fp16, int8, int4}.

**3. `validate` command (2.5 hrs).** `econ validate --model <hf-id> --quant int4`. Actually load a small open model with `transformers` + `bitsandbytes` (`load_in_4bit=True` / `load_in_8bit=True`) or via Ollama, then report *measured* memory (`torch.cuda.max_memory_allocated()` if GPU; else RSS via `psutil` for CPU/GGUF) and compare to your `vram` estimate. Use a genuinely small model so it runs on a laptop:
```bash
uv add transformers bitsandbytes accelerate torch psutil typer
# CPU-friendly validation targets: Qwen2.5-0.5B-Instruct, or Llama-3.2-1B/3B via Ollama
```
If you have no GPU, validate the int4 path via **Ollama** (`ollama run llama3.2:3b` then read memory) and note in the CLI output that bitsandbytes int4/int8 needs CUDA — the free CPU path is GGUF through Ollama/llama.cpp. **Cloud alternative:** Google Colab free tier gives a T4 GPU for the bitsandbytes path if you want to see int4 vs int8 vs fp16 on real hardware.

**4. Chat template + structured output + tool call (2.5 hrs).** In a notebook `notebooks/03_landscape.ipynb`:
- Load a small instruct model's tokenizer and print `apply_chat_template(messages, tokenize=False)` to *see* the special tokens — then do the same with the wrong (base) formatting to feel the difference.
- Get **structured JSON** out: call OpenAI with a strict `json_schema` (or Anthropic with a tool `input_schema`) to extract `{name, amount, date}` from a messy sentence; assert it parses. Free path: use Ollama's `format: json` mode.
- Get **one tool call**: define a `get_weather(city)` tool, send it, receive the model's tool-call request, execute your Python stub, return the result, get the final answer. Print the raw request/response so you see the loop you'll industrialize in Phase 2.

### Definition of Done
- [ ] `econ cost` prints a valid comparison table for ≥3 models and `--lang-breakdown` shows code/non-English cost more tokens than English.
- [ ] `econ vram --params 7 --quant int4` and `--quant fp16` return numbers in the right ballpark (7B ≈ ~3.5 GB weights int4, ~14 GB fp16) and correctly say which GPUs fit.
- [ ] `econ validate` loads a real small model and reports measured memory within a stated tolerance of the estimate (or documents why the CPU/GGUF path is used instead).
- [ ] You produce valid schema-conforming JSON from a model for 5/5 test inputs, and complete one full tool-call round trip end to end.
- [ ] The notebook shows the correct chat template's special tokens and demonstrates that wrong formatting degrades output.
- [ ] `uv run pytest -q` green; CLI has `--help` for every command; README documents each command with an example invocation.

### Pitfalls
- Treating the VRAM rule of thumb as exact — KV cache and activations can add 30–50%; long sequences and big batches blow past the weights-only number.
- Hand-formatting chat prompts for open models instead of `apply_chat_template` — silent quality loss with no error.
- Expecting `bitsandbytes` int4/int8 to work on CPU/Mac — it needs CUDA; use GGUF/Ollama or Colab instead.
- Assuming a reasoning model is always better — you pay for hidden thinking tokens and latency on tasks that don't need it.
- `trust_remote_code=True` runs arbitrary code from the model repo — only for models you trust.
- Hardcoding prices without a date comment — pricing shifts; a stale number silently makes every estimate wrong.

### Self-check
1. When would you self-host an open int4 model instead of calling a hosted API, and what numbers decide it?
2. Estimate the fp16 and int4 weights-only VRAM for a 13B model. Now add KV cache intuition — what makes it bigger?
3. Why does sending an instruct model the wrong chat template hurt quality even though no error is thrown?
4. What's the difference between JSON mode and schema-constrained structured output, and why does the latter matter downstream?
5. Give a concrete task where a reasoning model earns its extra token cost, and one where it's a waste.

---

## Phase milestone project — The Reproducible LLM Harness + Model Economics CLI

Integrate all three weeks into one shippable repo that proves the Phase 0 competencies (this combines both portfolio projects from the roadmap's Phase 0).

**What it must do:**
1. **Reproducible harness:** `uv`-managed env with committed `uv.lock`, `.env`-based secrets (never committed), and a notebook that runs **headless via nbconvert/jupyter execute** in CI — proving no hidden state.
2. **See generation:** a script/command that returns the **top-5 next-token candidates with logprobs** for a prompt so a reader can *see* how the model generates.
3. **Model Economics CLI** (`econ`): `cost` (token counts + $ across 3–4 models using real tokenizers and dated pricing, English vs code vs non-English), `vram` (estimate VRAM for a param count + quantization), and `validate` (load 2–3 open models at fp16/int8/int4 — or GGUF via Ollama — and compare measured vs estimated memory).
4. **Foundations shown:** structured JSON output for 5/5 inputs and one full tool-call round trip.

**Acceptance criteria:**
- [ ] Fresh clone → `uv sync` → `uv run pytest -q` green, with no secrets in git history.
- [ ] CI step (GitHub Actions or a `make ci` target) executes the notebook headless and fails on any error.
- [ ] `econ cost` and `econ vram` produce correct, documented tables; `econ validate`'s measured memory is within a stated tolerance of the estimate on at least one model.
- [ ] Logprobs command prints top-5 candidates for ≥2 prompts.
- [ ] README explains every command, the VRAM formula, the pricing-snapshot date, and the free (Ollama/CPU) path for anyone without API keys or a GPU.

**Suggested repo layout:**
```
ai-foundations/
  pyproject.toml
  uv.lock
  .env.example
  README.md
  Makefile                      # ci: run pytest + execute notebooks headless
  .github/workflows/ci.yml      # optional but recommended
  data/
  notebooks/
    01_numpy_pandas.ipynb
    02_llm_basics.ipynb
    03_landscape.ipynb
  src/ai_foundations/
    __init__.py                 # dotenv secret loading
    io_jsonl.py                 # JSONL read/write
    metrics.py                  # precision/recall/f1, ndcg@k, cosine
    tokens.py                   # tiktoken counting + pricing snapshot
    embed.py                    # sentence-transformers + ranking
    llm.py                      # openai/anthropic/ollama unified complete()
    logprobs.py                 # top-5 next-token viewer
    econ_cli.py                 # typer CLI: cost / vram / validate
  tests/
    test_metrics.py
    test_io_jsonl.py
    test_econ.py                # offline via Ollama
```

## You are ready to move on when...
- [ ] You can set up a reproducible AI project with `uv` + `.env` from scratch in under 10 minutes without notes.
- [ ] You can count tokens and estimate the $ cost of a call across models *before* sending it, and explain why language and output length change the bill.
- [ ] You can trace an output from tokenization → attention → logits → sampling and explain why the model produced *that* token.
- [ ] You can generate embeddings, compute cosine similarity, and rank documents by semantic relevance.
- [ ] You can pick temperature/top-p/top-k/max_tokens/stop for a task and justify the choice.
- [ ] You can estimate VRAM for a param count + quantization and validate it by actually loading a model.
- [ ] You can call hosted and local models through one interface, get schema-valid JSON, and complete a tool-call loop.
- [ ] Your harness repo is green in CI with no hidden notebook state and no secrets in history.

Next: **`01-prompting-context.md`** — you have the mechanics; now learn to control the one interface you fully own on a frozen model: the context window.
