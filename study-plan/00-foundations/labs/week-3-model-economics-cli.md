# Week 3 Lab: Build the Model Economics CLI (Milestone)

> This week you turn fuzzy "which model should we use?" debates into **numbers**. You will build `econ`, a real `typer` CLI with three commands — `cost` (tokens + $ across models), `vram` (memory estimate for a param count + quantization, KV cache included), and `validate` (measured-vs-estimated memory by actually loading a model) — then prove three model-reliability foundations in a notebook: correct **chat templates**, **schema-constrained JSON extraction**, and one full **tool-call round trip**. Finally you wire the whole three-week repo into a **reproducible harness** (headless notebook in CI, no secrets, README) — this is the Phase 0 portfolio milestone.
>
> Read the matching lectures first — this guide assumes them:
> - [../lectures/13-model-types-base-instruct-reasoning.md](../lectures/13-model-types-base-instruct-reasoning.md) — when reasoning "thinking tokens" are worth the cost/latency.
> - [../lectures/14-open-vs-closed-and-gateways.md](../lectures/14-open-vs-closed-and-gateways.md) — open vs closed on real axes; where OpenRouter/LiteLLM fit.
> - [../lectures/15-quantization-and-vram-math.md](../lectures/15-quantization-and-vram-math.md) — GGUF/AWQ/GPTQ, fp16→int8→int4, and the VRAM formula this CLI implements.
> - [../lectures/16-chat-templates-structured-output-tools.md](../lectures/16-chat-templates-structured-output-tools.md) — `apply_chat_template`, structured output, tool calling.
>
> It also reuses Week 2's `tokens.py` and `llm.py`, so keep working in the same `ai-foundations/` repo.

**Est. time:** ~9 hrs (lab) + ~2 hrs (milestone wiring) · **You will need:** the `ai-foundations/` repo from Weeks 1–2, Python + `uv`, [Ollama](https://ollama.com) (the free/local path for `validate`, structured JSON, and tool calls — **no API keys or GPU required**). Optional: OpenAI/Anthropic keys (~$0.02 for this lab), or a free Google Colab T4 to see real `bitsandbytes` int4/int8/fp16.

---

## Before you start (setup)

**What / Why.** Week 3 adds a CLI framework, the HF loading stack, and a process-memory probe. The GPU-only pieces (`transformers` + `bitsandbytes`) are optional — you get a complete, gradable lab on CPU through Ollama. Install the local model first because the download runs while you code.

**Do it:**
```bash
cd ai-foundations          # the repo you built in Weeks 1–2

# CLI + memory probe (works everywhere, no GPU)
uv add typer psutil

# HF loading stack (GPU/Colab path; safe to install on CPU, just won't 4-bit there)
uv add transformers accelerate torch
uv add bitsandbytes            # CUDA-only at runtime; install now, use on GPU/Colab

# Free local path for validate + structured JSON + tool calls:
ollama pull llama3.2:3b        # ~2 GB, CPU-friendly, supports tools + format:json
ollama pull qwen2.5:0.5b       # ~400 MB, tiny — good for the smallest validate target
```

**Verify:**
```bash
uv run python -c "import typer, psutil; print('cli deps ok')"
ollama list          # llama3.2:3b and qwen2.5:0.5b should appear
ollama run llama3.2:3b "say hi in 3 words"   # confirms the daemon serves
```

**Windows / Git-Bash note.** Ollama installs as a background service on Windows (starts automatically). If `ollama list` errors with a connection refused, launch the Ollama app once from the Start menu, then retry in Git-Bash. All `uv run` commands below are identical across Windows/macOS/Linux.

**Troubleshoot.** If `uv add bitsandbytes` fails to build on Windows, don't block — it is only used on the GPU/Colab path. Comment it out of `pyproject.toml`, keep going with the Ollama path, and reach for Colab in Step 3 if you want to see real 4-bit.

---

## Step-by-step

### Step 1 — Scaffold the `typer` CLI skeleton

**What.** Create `src/ai_foundations/econ_cli.py` with a `typer` app and three stub commands (`cost`, `vram`, `validate`), and wire an entry point so `econ` runs from the shell.

**Why.** Getting the command surface and `--help` working *before* any logic means you develop against a real interface, and the Definition of Done's "`--help` for every command" is satisfied structurally from step one. `typer` gives you typed flags, help text, and validation for free from function signatures.

**Do it.** In `econ_cli.py`:
```python
import typer

app = typer.Typer(help="Model Economics CLI — turn model-choice debates into numbers.")

@app.command()
def cost(
    file: str = typer.Option(..., help="Path to the input text file."),
    models: str = typer.Option("gpt-4o,claude-sonnet,llama3.1-8b",
                               help="Comma-separated model keys."),
    output_tokens: int = typer.Option(500, help="Assumed output tokens per model."),
    lang_breakdown: bool = typer.Option(False, help="Show tokens-per-char."),
):
    """Token counts + $ cost across models for FILE."""
    raise typer.Exit()   # fill in Step 2

@app.command()
def vram(
    params: float = typer.Option(..., help="Billions of parameters, e.g. 7."),
    quant: str = typer.Option("int4", help="fp16 | int8 | int4"),
    seq_len: int = typer.Option(4096), batch: int = typer.Option(1),
    layers: int = typer.Option(32), hidden: int = typer.Option(4096),
):
    """Estimate VRAM for a param count + quantization (KV cache included)."""
    raise typer.Exit()   # fill in Step 2

@app.command()
def validate(
    model: str = typer.Option(..., help="HF id (GPU) or ollama:<tag> (CPU)."),
    quant: str = typer.Option("int4", help="fp16 | int8 | int4"),
):
    """Load a real model and compare measured vs estimated memory."""
    raise typer.Exit()   # fill in Step 3

if __name__ == "__main__":
    app()
```
Add a console script to `pyproject.toml` so `econ` is a command:
```toml
[project.scripts]
econ = "ai_foundations.econ_cli:app"
```
Re-sync so the entry point is installed: `uv sync`.

**Expected result.** `uv run econ --help` lists three commands; `uv run econ vram --help` shows every flag with its help string and default.

**Verify.**
```bash
uv run econ --help
uv run econ cost --help && uv run econ vram --help && uv run econ validate --help
```

**Troubleshoot.** If `econ` isn't found, run via module (`uv run python -m ai_foundations.econ_cli --help`) — it works without the entry point — then re-run `uv sync`. If `typer` complains about a missing option, note that `typer.Option(...)` (Ellipsis) marks a flag **required**; use a default value to make it optional.

---

### Step 2 — Implement `cost` and `vram`

**What.** Fill in the two pure-computation commands. `cost` reuses Week 2's `tokens.py`; `vram` implements the formula from [lecture 15](../lectures/15-quantization-and-vram-math.md).

**Why.** These are deterministic and testable with no network — the backbone of the CLI and the easiest Definition-of-Done boxes to lock down. Doing them before `validate` means you have an *estimate* to compare measured memory against.

**Do it — `cost`.** Reuse `estimate_cost` from Week 2. Different model families tokenize differently: use `tiktoken` for OpenAI models, and for non-OpenAI families fall back to a **documented chars/token ratio** (note the approximation in output). Build the per-row table:
```python
from ai_foundations.tokens import count_tokens, PRICES  # from Week 2

CHARS_PER_TOKEN = {"openai": None, "llama": 3.5, "claude": 3.6}  # None = use tiktoken

def _rows(text, model_keys, output_tokens):
    for key in model_keys:
        in_tok = count_tokens(text, key)          # tiktoken or chars/ratio inside
        p = PRICES[key]                            # {"in": $/1M, "out": $/1M, "date": ...}
        in_usd  = in_tok / 1_000_000 * p["in"]
        out_usd = output_tokens / 1_000_000 * p["out"]
        yield key, in_tok, output_tokens, in_usd, out_usd, in_usd + out_usd
```
Print it as an aligned table (plain f-strings are fine; `typer.echo`/`rich` if you want color). For `--lang-breakdown`, also print `len(text) / in_tok` (chars-per-token) per model so English vs code vs non-English is visible: lower chars/token = more expensive.

**Do it — `vram`.** Implement the formula from the spine and lecture 15:
```python
BYTES = {"fp16": 2, "int8": 1, "int4": 0.5}

def vram_estimate(params_b, quant, seq_len, batch, layers, hidden):
    weights = params_b * 1e9 * BYTES[quant]                 # bytes
    # KV cache: 2 (K and V) × layers × seq_len × hidden × batch × bytes/elem (fp16=2)
    kv = 2 * layers * seq_len * hidden * batch * 2
    activations = 0.20 * (weights + kv)                     # ~20% headroom
    total_gb = (weights + kv + activations) / 1e9
    return total_gb, weights/1e9, kv/1e9
```
Then print which common GPUs fit (`fits = [g for g in (8,16,24,48,80) if total_gb <= g]`).

**Expected result.**
```
$ uv run econ vram --params 7 --quant int4
weights ≈ 3.5 GB · KV cache ≈ 2.1 GB · +20% headroom → total ≈ 6.7 GB
Fits on: 8, 16, 24, 48, 80 GB GPUs
$ uv run econ vram --params 7 --quant fp16
weights ≈ 14.0 GB · ... total ≈ ~19–20 GB → fits 24, 48, 80
```

**Verify.** Weights-only must match the rule of thumb: 7B int4 ≈ 3.5 GB, fp16 ≈ 14 GB (params × bytes/param). Write a quick assertion into `tests/test_econ.py`:
```python
def test_vram_weights_ballpark():
    total, w, kv = vram_estimate(7, "int4", 4096, 1, 32, 4096)
    assert abs(w - 3.5) < 0.1          # int4 weights
    assert 6.0 < total < 8.0           # weights + KV + headroom
```

**Troubleshoot.** If your `cost` numbers look 10× off, check the `/1_000_000` — published prices are per **1M** tokens. If KV cache dwarfs weights, you likely passed a huge `--seq-len`; that's *correct* and is the whole point of the flag (long context eats VRAM).

---

### Step 3 — Implement `validate` (measured vs estimated)

**What.** Load a real, small model and report **measured** memory, then diff it against your `vram` estimate. Two paths: GPU via `transformers` + `bitsandbytes`, or CPU via Ollama (the free default).

**Why.** The rule of thumb is a starting point; the acceptance gate wants you to *check it against reality* and learn where it drifts (KV cache and activations can add 30–50%). This is the difference between "I read the formula" and "I measured it."

**Do it — free CPU path (Ollama, no GPU).** Load the model into Ollama's server, then read its resident memory. Ollama reports the loaded model's size directly:
```python
import subprocess, json, psutil

def measured_ollama(tag: str) -> float:
    subprocess.run(["ollama", "run", tag, "warm up"], capture_output=True, text=True)
    ps = subprocess.run(["ollama", "ps", "--format", "json"], capture_output=True, text=True)
    # each line is a JSON object with a "size" field in bytes
    sizes = [json.loads(l)["size"] for l in ps.stdout.splitlines() if l.strip()]
    return max(sizes) / 1e9 if sizes else psutil.Process().memory_info().rss / 1e9
```
In the command, print measured GB, the `vram` estimate for that model's param count/quant, the delta, and a clear note:
```
NOTE: bitsandbytes int4/int8 requires CUDA. On CPU we validate the GGUF path via Ollama.
model=llama3.2:3b  measured≈2.3 GB  estimated(int4,3B)≈1.9 GB  delta=+21% (KV+activations)
```

**Do it — GPU path (`transformers` + `bitsandbytes`; local CUDA box or Colab T4).**
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def measured_hf(hf_id: str, quant: str) -> float:
    cfg = None
    if quant == "int4": cfg = BitsAndBytesConfig(load_in_4bit=True)
    elif quant == "int8": cfg = BitsAndBytesConfig(load_in_8bit=True)
    torch.cuda.reset_peak_memory_stats()
    AutoModelForCausalLM.from_pretrained(
        hf_id, quantization_config=cfg,
        torch_dtype=torch.float16 if quant == "fp16" else None,
        device_map="cuda",
    )
    return torch.cuda.max_memory_allocated() / 1e9
```
CPU-friendly HF target if you have a small GPU or Colab: `Qwen/Qwen2.5-0.5B-Instruct`. **Cloud alternative:** open a free Colab notebook, `pip install transformers bitsandbytes accelerate`, and run this at int4/int8/fp16 to see the three tiers on a real T4.

**Detect the environment** so one command does the right thing:
```python
use_gpu = torch.cuda.is_available()
```

**Expected result.** Measured memory prints and lands within a stated tolerance (say ±40%) of the estimate — or the CPU path runs and clearly documents *why* it used GGUF/Ollama instead of bitsandbytes.

**Verify.**
```bash
uv run econ validate --model ollama:qwen2.5:0.5b --quant int4   # free CPU path
# GPU/Colab:
uv run econ validate --model Qwen/Qwen2.5-0.5B-Instruct --quant int4
```
Confirm the printed delta and the note about which path ran.

**Troubleshoot.**
- `bitsandbytes` import error on CPU/Mac → expected; the GPU path is unavailable, use the Ollama path. This is a pitfall the spine calls out explicitly.
- `torch.cuda.is_available()` is `False` on Colab → Runtime → Change runtime type → T4 GPU.
- `ollama ps` shows nothing → the model unloaded (idle timeout); the `ollama run ... "warm up"` call re-loads it right before you read `ps`.
- Never pass `trust_remote_code=True` to `from_pretrained` unless you trust the repo — it runs arbitrary code.

---

### Step 4 — Chat templates: correct vs wrong formatting

**What.** In `notebooks/03_landscape.ipynb`, print `tokenizer.apply_chat_template(messages, tokenize=False)` for a small instruct model, then hand-format the *same* messages the wrong (base-completion) way and compare.

**Why.** Instruct models were trained with a *specific* special-token layout (see [lecture 16](../lectures/16-chat-templates-structured-output-tools.md)). Send the wrong format and quality silently craters — no error, just worse answers. Seeing the tokens makes this concrete.

**Do it.**
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")  # tokenizer only, tiny
messages = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "Name the capital of France."},
]
correct = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
wrong   = "\n".join(m["content"] for m in messages)   # naive, no special tokens
print("CORRECT:\n", correct)
print("WRONG:\n", wrong)
```
Then generate from both formats (via Ollama or a loaded model) and eyeball the difference. Add a markdown cell noting the special tokens you see (`<|im_start|>`, role markers, `<|im_end|>`) and *why* skipping them hurts.

**Expected result.** `correct` contains the model's special-token scaffolding around each turn and a trailing assistant-generation prompt; `wrong` is bare text. Generation from the correct template is on-task; from the wrong one it drifts (rambles, ignores the system role, or continues your text instead of answering).

**Verify.** The Definition of Done wants the notebook to *show* the correct template's special tokens and *demonstrate* that wrong formatting degrades output — one printed template + one before/after generation cell covers it.

**Troubleshoot.** Downloading the full model is unnecessary here — `AutoTokenizer.from_pretrained` pulls only tokenizer files (a few MB). If offline, any instruct tokenizer you already cached works; the point is the template, not the specific model.

---

### Step 5 — Schema-constrained structured JSON extraction

**What.** Extract `{name, amount, date}` from a messy sentence and get back JSON that *parses and conforms*, for 5/5 test inputs. Free path: Ollama's `format: json`. Paid path: OpenAI strict `json_schema` or Anthropic tool `input_schema`.

**Why.** Structured output is one of the two highest-leverage reliability features (lecture 16). JSON *mode* only guarantees valid JSON; *schema-constrained* output guarantees the shape you need — the distinction the self-check asks about.

**Do it — free path (Ollama `format:json`).**
```python
import json, requests

SCHEMA_HINT = 'Return ONLY JSON: {"name": str, "amount": number, "date": "YYYY-MM-DD"}'
def extract(sentence: str) -> dict:
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": f"{SCHEMA_HINT}\nText: {sentence}"}],
        "format": "json",     # forces syntactically valid JSON
        "stream": False,
    })
    return json.loads(r.json()["message"]["content"])
```
Newer Ollama versions accept a full JSON Schema object in `format` (not just the string `"json"`) for true schema constraint — use it if available. **Paid path:** OpenAI `response_format={"type":"json_schema","json_schema":{...,"strict":True}}`, or Anthropic a tool whose `input_schema` is your JSON Schema and read `tool_use.input`.

Run it over 5 messy inputs and assert:
```python
CASES = [
  ("Invoice: Acme charged $1,240.50 on 2026-03-14.", {"name":"Acme"}),
  # ...4 more with varied phrasings/currencies/date formats
]
for text, expect in CASES:
    got = extract(text)
    assert {"name","amount","date"} <= got.keys()   # schema conforms
    json.dumps(got)                                  # parses / serializes
```

**Expected result.** All 5 return a dict with the three keys, parseable, with plausible values.

**Verify.** `assert` loop passes 5/5; drop it into `tests/test_econ.py` guarded to skip if Ollama isn't running (so CI without Ollama stays green — see Step 7).

**Troubleshoot.** If a value comes back as a string like `"$1,240.50"` instead of a number, tighten the prompt (`amount as a plain number, no currency symbol`) or use the schema-object form. If keys are missing, the model ignored the instruction — smaller models need the schema stated explicitly in the prompt even with `format:json`.

---

### Step 6 — One full tool-call round trip

**What.** Define a `get_weather(city)` tool, send it to the model, receive the model's tool-call *request*, execute your Python stub, return the result, and get the model's final natural-language answer. Print the raw request/response at each hop.

**Why.** This is the loop you industrialize in Phase 2. Doing it once by hand — and *seeing* the two-turn shape — demystifies "agents": the model doesn't run code, it *asks you* to, and you feed the result back.

**Do it — free path (Ollama tool calling; llama3.2 supports it).**
```python
import requests, json

def get_weather(city: str) -> dict:            # your stub
    return {"city": city, "temp_c": 21, "cond": "sunny"}

TOOLS = [{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "parameters": {"type":"object","properties":{"city":{"type":"string"}},
                   "required":["city"]},
}}]

msgs = [{"role":"user","content":"What's the weather in Paris?"}]
# turn 1: model asks for the tool
r1 = requests.post("http://localhost:11434/api/chat",
      json={"model":"llama3.2:3b","messages":msgs,"tools":TOOLS,"stream":False}).json()
call = r1["message"]["tool_calls"][0]["function"]
print("MODEL WANTS:", call)                    # {'name':'get_weather','arguments':{'city':'Paris'}}

result = get_weather(**call["arguments"])       # you run it
msgs += [r1["message"], {"role":"tool","content":json.dumps(result)}]

# turn 2: model answers using the tool result
r2 = requests.post("http://localhost:11434/api/chat",
      json={"model":"llama3.2:3b","messages":msgs,"stream":False}).json()
print("FINAL:", r2["message"]["content"])
```
**Paid path:** OpenAI Chat Completions `tools=[...]` → read `tool_calls` → append a `role:"tool"` message with `tool_call_id` → call again. Anthropic: model returns a `tool_use` block → you reply with a `tool_result` block.

**Expected result.** Turn 1 prints a structured tool-call request naming `get_weather` with `{"city":"Paris"}`; turn 2 prints a sentence like "It's 21°C and sunny in Paris."

**Verify.** You completed **one full round trip end to end** (request → execute → result → final answer) and printed the raw shapes — the Definition of Done box.

**Troubleshoot.** If `tool_calls` is missing from `r1`, the model answered directly (small models sometimes skip the tool) — make the question require external data ("right now", a specific city) or use `llama3.2:3b`/`qwen2.5` which reliably emit tool calls. Ensure you append **both** the assistant's tool-call message and the `role:"tool"` result before turn 2, or the model has no result to use.

---

### Step 7 — Wire the reproducible-harness milestone (CI, no secrets, README)

**What.** Integrate all three weeks into one shippable repo: headless notebook execution in CI, verified secret-free history, and a README documenting every command, the VRAM formula, the pricing-snapshot date, and the free path.

**Why.** This is the Phase 0 portfolio milestone — the artifact that proves reproducibility (no hidden state, no leaked keys) and makes the CLI usable by someone with no keys and no GPU.

**Do it — CI-safe tests.** Make network/Ollama tests skip gracefully so CI stays green without secrets or a daemon:
```python
import shutil, pytest
requires_ollama = pytest.mark.skipif(
    shutil.which("ollama") is None, reason="Ollama not installed (CI headless)")
```
Apply it to Steps 5/6 tests; keep `vram`/`cost` tests pure so they always run.

**Do it — Makefile + headless notebook.** Add a `ci` target that runs tests and executes every notebook headless (proving no hidden state):
```make
ci:
	uv run pytest -q
	uv run jupyter execute notebooks/01_numpy_pandas.ipynb
	uv run jupyter execute notebooks/02_llm_basics.ipynb
	uv run jupyter execute notebooks/03_landscape.ipynb
```
**Windows note:** if `make` isn't installed, run the three lines directly, or `choco install make` / use Git-Bash with `mingw32-make`.

**Do it — GitHub Actions** (`.github/workflows/ci.yml`):
```yaml
name: ci
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run pytest -q
      - run: uv run jupyter execute notebooks/03_landscape.ipynb
```
Notebooks must not require keys/Ollama to *execute* — guard those cells with `if os.getenv("OPENAI_API_KEY")` / a try-around-Ollama so headless execution never fails on a missing secret.

**Do it — README.** Document each `econ` command with an example invocation, the VRAM formula (weights + KV + ~20% headroom) with the KV expression, the **pricing-snapshot date**, and the free Ollama/CPU path. Confirm no secrets ever entered history:
```bash
git log -p | grep -i "sk-\|api_key\|secret" | grep -v "example"   # expect NO real key
```

**Expected result.** `make ci` (or the three commands) is green locally; the Actions run is green on push; README covers every command + formula + date + free path.

**Verify.** Simulate a fresh clone:
```bash
cd /tmp && git clone <your-repo> fresh && cd fresh
uv sync && uv run pytest -q          # green with zero API keys set
uv run econ --help
```

**Troubleshoot.** If CI fails inside a notebook, an ungated cell is calling a network/Ollama resource — wrap it in the key/daemon guard. If `uv sync` fails in CI over `bitsandbytes`, mark it optional in `pyproject.toml` (it's GPU-only) so the CPU CI runner installs cleanly.

---

## Putting it together — a short end-to-end run

Once all steps pass, this is the story your repo tells:

```bash
# 1. Cost: same file across models, see language effect
uv run econ cost --file data/report.txt \
  --models gpt-4o,claude-sonnet,llama3.1-8b --output-tokens 500 --lang-breakdown

# 2. VRAM: can a 7B int4 model serve 4k context on a 16 GB GPU?
uv run econ vram --params 7 --quant int4 --seq-len 4096

# 3. Validate: does reality match the estimate? (free CPU path)
uv run econ validate --model ollama:qwen2.5:0.5b --quant int4

# 4. Foundations (notebook 03): correct chat template, schema JSON 5/5, one tool round trip
uv run jupyter execute notebooks/03_landscape.ipynb

# 5. Prove reproducibility
make ci     # pytest + all notebooks headless, no secrets
```

The thread: `cost` and `vram` turn model choice into numbers → `validate` shows those numbers survive contact with a real model → the notebook proves you can drive open models correctly (template), get reliable structure (schema JSON), and let a model call your code (tool loop) → CI proves the whole thing is reproducible and secret-free.

---

## Definition of Done — verifiable checks

Restating the spine's gate. All boxes must pass:

- [ ] `econ cost` prints a valid comparison table for **≥3 models**, and `--lang-breakdown` shows code / non-English cost **more tokens** than English. *(Run the end-to-end command; compare chars/token per language.)*
- [ ] `econ vram --params 7 --quant int4` and `--quant fp16` return right-ballpark numbers (**7B ≈ ~3.5 GB int4, ~14 GB fp16** weights) and correctly say **which GPUs fit**. *(Assertion test from Step 2.)*
- [ ] `econ validate` loads a **real small model** and reports measured memory **within a stated tolerance** of the estimate — *or* documents why the CPU/GGUF (Ollama) path is used. *(Step 3 output includes the note + delta.)*
- [ ] **Valid schema-conforming JSON for 5/5** test inputs, and **one full tool-call round trip** end to end. *(Step 5 assert loop + Step 6 two-turn run.)*
- [ ] The notebook **shows the correct chat template's special tokens** and **demonstrates that wrong formatting degrades output**. *(Step 4 cells.)*
- [ ] `uv run pytest -q` **green**; CLI has **`--help` for every command**; **README documents each command with an example invocation.** *(Steps 1 and 7.)*

Milestone-specific acceptance (from the phase milestone section):

- [ ] Fresh clone → `uv sync` → `uv run pytest -q` **green, no secrets in git history**.
- [ ] **CI step** executes the notebook **headless** and fails on any error.
- [ ] README explains **every command, the VRAM formula, the pricing-snapshot date, and the free (Ollama/CPU) path**.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `econ` command not found | entry point not installed | `uv sync` after adding `[project.scripts]`, or run `uv run python -m ai_foundations.econ_cli` |
| `bitsandbytes` import/build error | it's **CUDA-only** | use the Ollama path (Step 3); make it optional in `pyproject.toml`; use Colab for real 4-bit |
| `torch.cuda.is_available()` is False on Colab | CPU runtime | Runtime → Change runtime type → **T4 GPU** |
| `ollama ps` shows nothing in `validate` | model unloaded (idle timeout) | the `ollama run ... "warm up"` re-loads it right before reading `ps` |
| `cost` numbers ~10× off | forgot per-**1M**-token pricing | divide token counts by `1_000_000` |
| VRAM estimate huge | large `--seq-len` inflates KV cache | that's correct — long context genuinely eats VRAM |
| Chat template looks like plain text | called on a base (non-instruct) tokenizer | use an **instruct** tokenizer; `add_generation_prompt=True` |
| JSON has string amounts / missing keys | small model + loose prompt | state the schema explicitly; use Ollama's schema-object `format` |
| No `tool_calls` in tool step | model answered directly | ask something needing live data; use `llama3.2:3b`; append **both** assistant + tool messages |
| CI fails inside a notebook | ungated cell hits network/Ollama | wrap in `if os.getenv(...)` / try-except so headless execution never needs a secret |
| stale pricing silently wrong | no date comment | keep the `"date"` field in `PRICES` and surface it in `cost` output + README |

---

## Stretch goals (optional)

- **Gateway awareness:** add a `--via openrouter` note or a thin LiteLLM call so `cost` can price a model through a gateway — foreshadows the LiteLLM you'll lean on later ([lecture 14](../lectures/14-open-vs-closed-and-gateways.md)).
- **Reasoning-token accounting:** add an `--reasoning-tokens` flag to `cost` so you can price the hidden "thinking" tokens of an o-series/extended-thinking model and *see* the thinking tax ([lecture 13](../lectures/13-model-types-base-instruct-reasoning.md)).
- **Three-tier validate on Colab:** run `validate` at fp16/int8/int4 for one HF model and produce a small table of measured memory vs quality (a couple of generations) — the clearest way to feel the quantization tradeoff.
- **Schema strictness comparison:** run the extraction on the same 5 inputs with plain `format:"json"` vs a full JSON-Schema constraint and count conformance failures — makes the "JSON mode vs structured output" distinction concrete.
- **`econ fit` command:** given a GPU size, invert the VRAM formula to report the largest params/quant/context that fits.
