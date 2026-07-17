# Week 1 Lab: Build `structio` — A Schema-Constrained Extraction Toolkit

> This week you build `structio/`, a structured-extraction toolkit that turns messy invoice text into a **validated `Invoice` object** across three providers (OpenAI strict `json_schema`, Anthropic forced tool, Gemini `responseSchema`). You'll go from "the model usually returns JSON" to a guaranteed, validated, observable contract — the primitive that RAG, agents, and every LLM-in-a-loop are built on. You'll also add a **repair loop** with error classification + a circuit breaker, and a **streaming partial-JSON** demo. This toolkit is the base you extend in Week 2 into a hardened tool-loop service.
>
> **Read the lectures first** (the spine's Theory is their recap):
> - [When (and When Not) to Force Structured Output](../lectures/01-when-to-use-structured-output.md)
> - [The Three Reliability Tiers](../lectures/02-three-reliability-tiers.md)
> - [Provider JSON-Schema Subsets and Strict-Mode Gotchas](../lectures/03-provider-schema-subsets-and-strict-mode.md)
> - [Designing LLM-Friendly Schemas with Pydantic v2 (and a Zod Primer)](../lectures/04-llm-friendly-schema-design.md)
> - [Runtime Resilience: Repair Loops, Error Classification, and Streaming Partial JSON](../lectures/05-repair-loops-and-streaming.md)

**Est. time:** ~8 hrs · **You will need:**
- Python 3.11+, `uv`, `pytest`, comfort with type hints.
- API keys for **OpenAI**, **Anthropic**, and a **Gemini** free-tier key. Total spend for this lab is well under ~$5 on `gpt-4o-mini` / `claude-haiku-4-5` / `gemini-2.0-flash` class models. No GPU.
- **Free / local path (no paid keys):** [Ollama](https://ollama.com) with its OpenAI-compatible endpoint (`ollama pull llama3.2`). You can run Tiers 1–2 fully locally; Tier-3 strict enforcement is weaker on Ollama — that's a deliberate teaching moment for Week 2's constrained decoding. See [Free/local path](#free--local-path-no-paid-keys) at the end of the steps.

> **Model-ID note.** The spine was written against `claude-3-5-haiku`, which has since been **retired**. This guide uses the current cheap Anthropic model **`claude-haiku-4-5`** instead. If a provider ID 404s, that model was likely retired — check the provider's current model list.

---

## Before you start (setup)

**What / Why.** A clean, reproducible env with pinned deps and secrets in `.env` (never in code). `uv` gives you fast, lockfile-backed installs.

**Do it (Windows Git-Bash and macOS/Linux are identical here):**

```bash
# Install uv if you don't have it
#   macOS/Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
#   Windows:      winget install astral-sh.uv     (or: pip install uv)

uv init structio && cd structio
uv add pydantic openai anthropic google-genai instructor python-dotenv httpx rich
uv add --dev pytest

mkdir -p structio/schemas structio/providers tests data
touch structio/__init__.py structio/schemas/__init__.py structio/providers/__init__.py
```

Create `.env` in the repo root (Git-Bash `printf` avoids the `echo -e` portability trap):

```bash
printf 'OPENAI_API_KEY=sk-...\nANTHROPIC_API_KEY=sk-ant-...\nGEMINI_API_KEY=...\n' > .env
echo ".env" >> .gitignore
```

**Target folder layout** (you'll fill these in step by step):

```
structio/
  structio/
    schemas/invoice.py        # Pydantic models
    providers/openai_so.py    # tier-3 structured output per provider
    providers/anthropic_so.py
    providers/gemini_so.py
    repair.py                 # retry/repair loop + error classification
    stream.py                 # partial-JSON streaming demo
    config.py                 # loads .env, shared model IDs
  tests/test_schema.py
  data/invoices.jsonl         # 20 messy invoice texts (you write these)
  docs/honored_keywords.md    # per-provider schema-keyword table (a DoD artifact)
```

**Expected result.** `uv run python -c "import pydantic, openai, anthropic, google.genai, instructor; print('ok')"` prints `ok`.

**Verify.** `uv run python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(all(os.getenv(k) for k in ['OPENAI_API_KEY','ANTHROPIC_API_KEY','GEMINI_API_KEY']))"` prints `True`.

**Troubleshoot.**
- `google.genai` import fails → the package is `google-genai` (the import is `from google import genai`), not the older `google-generativeai`. Re-run `uv add google-genai`.
- On Windows a bad `.env` from `echo -e` shows literal `\n`; use the `printf` form above.
- `uv` not found after install → open a new shell so PATH updates, or use `pip install uv`.

Add a tiny shared config so every module reads the same env and model IDs:

```python
# structio/structio/config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

def require(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"{key} not set — add it to .env")
    return v
```

---

## Step-by-step

### Step 1 — Design an LLM-friendly Pydantic schema

**What.** Define `Invoice` (flat, enums for closed sets, a `description` on every field, a `reasoning` scratchpad field first, and a business-rule validator that line items sum to the total).

**Why.** The schema *is* prompt real estate: field names and descriptions are read by the model. `reasoning` first is the "scratchpad-before-extraction" trick — forcing JSON too early hurts reasoning, so we let the model think in a string field before the structured fields. The `@model_validator` enforces a business rule **in code, not the prompt** ("LLM proposes, code disposes").

**Do it** — `structio/structio/schemas/invoice.py`:

```python
from enum import Enum
from pydantic import BaseModel, Field, model_validator


class Category(str, Enum):
    software = "software"
    hardware = "hardware"
    services = "services"
    other = "other"


class LineItem(BaseModel):
    description: str = Field(description="Human-readable line description")
    qty: float = Field(description="Quantity billed")
    unit_price_usd: float = Field(description="Price per unit in USD")


class Invoice(BaseModel):
    reasoning: str = Field(
        description="Brief notes on how fields were located; write this FIRST"
    )
    vendor_name: str = Field(description="Legal name of the vendor issuing the invoice")
    invoice_date: str = Field(description="ISO 8601 date, e.g. 2026-03-14")
    total_usd: float = Field(description="Grand total in USD")
    category: Category = Field(description="Best-fit spend category")
    line_items: list[LineItem] = Field(
        description="Individual billed lines", max_length=25
    )

    @model_validator(mode="after")
    def totals_must_sum(self):
        computed = round(sum(li.qty * li.unit_price_usd for li in self.line_items), 2)
        if self.line_items and abs(computed - self.total_usd) > 0.01:
            raise ValueError(
                f"line items sum to {computed} but total_usd is {self.total_usd}"
            )
        return self
```

**Expected result.** `Invoice.model_json_schema()` returns a dict with `properties`, `required`, and a `$defs` block for `LineItem` and `Category`.

**Verify.**

```bash
uv run python -c "from structio.schemas.invoice import Invoice; import json; s=Invoice.model_json_schema(); print(sorted(s['properties'])); assert all('description' in v for v in s['properties'].values()); print('every field has a description')"
```

Expected: the property list plus `every field has a description` (first DoD box).

**Troubleshoot.**
- `max_length` on a field → Pydantic v2 spelling; if you see a v1 error you have Pydantic v1 installed. `uv add "pydantic>=2"`.
- Enum serializes to the member name instead of the value → make it `class Category(str, Enum)` (inherit `str`) as shown.

---

### Step 2 — Write 20 messy invoice inputs

**What.** Author `data/invoices.jsonl` — one JSON object per line, each with a raw invoice `text` and an `expect` label (`ok` for extractable, `garbage` for impossible/nonsense).

**Why.** You need varied, adversarial inputs to prove the extractor is robust: different formats, some with arithmetic that *must* validate (to exercise the totals validator), and **at least 2 garbage inputs** that must route to escalation rather than crash.

**Do it** — create `data/invoices.jsonl` (18 `ok` + 2 `garbage`; abbreviated — write your own, vary formats):

```jsonl
{"id": 1, "expect": "ok", "text": "INVOICE Acme Software Ltd  Date: 2026-03-14  2 x Pro License @ $50.00 = $100.00  Total due: $100.00"}
{"id": 2, "expect": "ok", "text": "From: Bolt Hardware Inc. 03/02/2026. Items: 3 widgets 12.50 ea; 1 mount 9.00. TOTAL 46.50 USD"}
{"id": 3, "expect": "ok", "text": "Cloud Ops Services — consulting 10 hrs x $120 = 1200. Invoice date 2026-01-09. Amount: $1,200.00"}
{"id": 19, "expect": "garbage", "text": "the quick brown fox jumps over the lazy dog"}
{"id": 20, "expect": "garbage", "text": "lorem ipsum 42 %%% no vendor no total ###"}
```

A helper to load them:

```python
# structio/structio/data.py
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "invoices.jsonl"

def load_invoices() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
```

**Expected / Verify.**

```bash
uv run python -c "from structio.data import load_invoices; rows=load_invoices(); print(len(rows), sum(r['expect']=='garbage' for r in rows), 'garbage')"
```

Expected: `20 2 garbage`.

**Troubleshoot.** `json.decoder.JSONDecodeError` → one line isn't valid JSON (a stray newline inside a record, or a trailing comma). Each record must be on exactly one line.

---

### Step 3 — Tier-3 with OpenAI (strict `json_schema`)

**What.** One function that returns a validated `Invoice` using OpenAI's parse helper, and prints **first-call latency vs. subsequent calls** to observe schema-compile caching.

**Why.** OpenAI strict mode constrains decoding to your JSON Schema. Strict mode has hard rules (every object needs `additionalProperties: false`, **every property must be in `required`** — optional fields are modeled as nullable unions, not by omission) and a **one-time schema-compile latency** (the grammar is cached after the first call). The `.parse()` helper generates the strict schema for you from the Pydantic model.

**Do it** — `structio/structio/providers/openai_so.py`:

```python
import time
from openai import OpenAI
from structio.config import OPENAI_MODEL, require
from structio.schemas.invoice import Invoice

_client = OpenAI(api_key=require("OPENAI_API_KEY"))

SYS = "Extract invoice fields. Fill `reasoning` FIRST with how you located each field."


def extract_openai(text: str, model: str = OPENAI_MODEL) -> Invoice:
    """Tier-3: strict json_schema via the parse helper. Returns a validated Invoice."""
    completion = _client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": text}],
        response_format=Invoice,   # parse() derives a strict schema from the model
    )
    msg = completion.choices[0].message
    if msg.refusal:
        raise ValueError(f"model refused: {msg.refusal}")
    return msg.parsed  # already an Invoice instance (also runs your validators)


if __name__ == "__main__":
    from structio.data import load_invoices
    text = load_invoices()[0]["text"]
    for i in range(2):
        t = time.perf_counter()
        inv = extract_openai(text)
        print(f"call {i}: {time.perf_counter() - t:.2f}s  vendor={inv.vendor_name}")
```

**Expected result.** `uv run python -m structio.providers.openai_so` prints two lines; **call 0 is noticeably slower** than call 1 (schema-compile caching). Both extract the vendor.

**Verify.** The returned value `isinstance(inv, Invoice)` and `inv.total_usd` is a float. The `.parse()` path raises/validates on schema mismatch, so a successful return is already schema-valid.

**Troubleshoot.**
- `BadRequestError: ... additionalProperties` → you passed a hand-written schema without `additionalProperties: false`. Prefer `response_format=Invoice` (the helper handles it). If you must hand-roll, add `additionalProperties: false` to **every** object.
- Optional field rejected in strict mode → don't omit it from `required`; model it as `Optional[str]` (a `["string","null"]` union) and keep it required.
- Model 404 → `gpt-4o-mini` is a current cheap model; if unavailable set `OPENAI_MODEL` in `.env`.

---

### Step 4 — Tier-3 with Anthropic (forced tool)

**What.** Anthropic has no `json_schema` response format; the idiom is to define a **tool** whose `input_schema` is your JSON Schema and **force** it with `tool_choice`. Read the args from the `tool_use` block and validate with Pydantic.

**Why.** Forcing a specific tool guarantees the model emits arguments matching your schema. Crucially, Anthropic returns the args as an **already-parsed object** (a dict), not a JSON string — so no `json.loads` step (contrast with OpenAI tool args in Week 2).

**Do it** — `structio/structio/providers/anthropic_so.py`:

```python
import anthropic
from structio.config import ANTHROPIC_MODEL, require
from structio.schemas.invoice import Invoice

_client = anthropic.Anthropic(api_key=require("ANTHROPIC_API_KEY"))

SYS = "Extract invoice fields via the record_invoice tool. Fill `reasoning` FIRST."

TOOL = {
    "name": "record_invoice",
    "description": "Record the structured fields extracted from an invoice.",
    "input_schema": Invoice.model_json_schema(),
}


def extract_anthropic(text: str, model: str = ANTHROPIC_MODEL) -> Invoice:
    """Tier-3: forced tool. Returns a validated Invoice."""
    resp = _client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYS,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "record_invoice"},  # force this tool
        messages=[{"role": "user", "content": text}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "record_invoice":
            return Invoice.model_validate(block.input)  # block.input is a parsed dict
    raise ValueError("model did not call record_invoice")


if __name__ == "__main__":
    from structio.data import load_invoices
    inv = extract_anthropic(load_invoices()[0]["text"])
    print("vendor:", inv.vendor_name, "total:", inv.total_usd)
```

**Expected result.** `uv run python -m structio.providers.anthropic_so` prints the vendor and total.

**Verify.** `isinstance(block.input, dict)` (parsed, not a string) and the returned value validates. If a deliberately-broken input (line items not summing to total) reaches this function, `Invoice.model_validate` raises `pydantic.ValidationError` — that's the repair loop's job in Step 7.

**Troubleshoot.**
- `400 ... input_schema` → Anthropic honors a **subset** of JSON Schema. If your schema uses a keyword it rejects, strip it (log which ones — Step 6). The flat `Invoice` here is well within the subset.
- Empty `content` / no `tool_use` block → check `tool_choice` names the exact tool; a `stop_reason` of `"refusal"` means the model declined — treat as escalation.
- **Newer models also support native structured outputs** (`client.messages.parse()` / `output_config.format`) on the current top-tier models, but the **forced-tool idiom is what this lab teaches** and works on the cheap `claude-haiku-4-5`. Keep the forced tool.

---

### Step 5 — Tier-3 with Gemini (`responseSchema`)

**What.** Use `google-genai` with `response_mime_type="application/json"` + `response_schema=Invoice`, then validate. Log which schema keywords Gemini silently dropped.

**Why.** Gemini's `responseSchema` uses an OpenAPI-subset dialect; property ordering and `anyOf`/nullability support are narrower than OpenAI's. Passing the Pydantic model lets the SDK translate; you still validate the result yourself because the dialect is lossy.

**Do it** — `structio/structio/providers/gemini_so.py`:

```python
from google import genai
from google.genai import types
from structio.config import GEMINI_MODEL, require
from structio.schemas.invoice import Invoice

_client = genai.Client(api_key=require("GEMINI_API_KEY"))

SYS = "Extract invoice fields. Fill `reasoning` FIRST with how you located each field."


def extract_gemini(text: str, model: str = GEMINI_MODEL) -> Invoice:
    """Tier-3: responseSchema. Returns a validated Invoice."""
    resp = _client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYS,
            response_mime_type="application/json",
            response_schema=Invoice,   # SDK translates to the OpenAPI-subset dialect
        ),
    )
    # resp.parsed is an Invoice when the SDK could round-trip the schema;
    # fall back to validating the raw JSON text otherwise.
    if isinstance(resp.parsed, Invoice):
        return resp.parsed
    return Invoice.model_validate_json(resp.text)


if __name__ == "__main__":
    from structio.data import load_invoices
    inv = extract_gemini(load_invoices()[0]["text"])
    print("vendor:", inv.vendor_name, "total:", inv.total_usd)
```

**Expected result.** `uv run python -m structio.providers.gemini_so` prints the vendor and total.

**Verify.** Returned value is an `Invoice`. If `resp.parsed` is `None`, the `model_validate_json(resp.text)` fallback catches it — a good place to log the raw text when debugging.

**Troubleshoot.**
- The `reasoning` field comes back empty or the enum is rejected → Gemini's dialect can drop `description`/enum handling differently; log it (Step 6) and, if needed, restate the field intent in the prompt.
- `resp.parsed` is `None` but `resp.text` is valid JSON → keep the fallback; some schema shapes don't auto-parse.
- Rate limit on the free tier → add a short `time.sleep` between calls, or run fewer inputs while developing.

---

### Step 6 — Log honored-vs-dropped JSON-Schema keywords (DoD artifact)

**What.** A small script + a markdown table recording, **per provider**, which JSON-Schema keywords survived the round-trip and which were silently dropped.

**Why.** This is a required Definition-of-Done artifact and a real production skill: you cannot trust a schema keyword until you've verified the provider honors it. You reuse this finding in Week 2 for tool `parameters`.

**Do it** — probe with a schema that uses keywords providers commonly drop, then write `docs/honored_keywords.md`:

```python
# structio/structio/keyword_probe.py  (run, then hand-fill the table from what you observe)
from structio.schemas.invoice import Invoice
KEYWORDS = ["description", "enum", "$defs/$ref", "maxItems", "additionalProperties",
            "format(date)", "minimum/maximum", "anyOf/null"]
print("Base schema keywords present:", sorted(Invoice.model_json_schema().keys()))
print("Probe each provider by extending Invoice with the keyword and checking whether")
print("the returned object respects it (e.g. maxItems enforced? enum constrained?).")
```

```markdown
<!-- docs/honored_keywords.md -->
# Honored vs. dropped JSON-Schema keywords (Week 1 finding)

| Keyword                | OpenAI (strict) | Anthropic (tool) | Gemini (responseSchema) |
|------------------------|-----------------|------------------|-------------------------|
| `description`          | honored         | honored          | often dropped           |
| `enum`                 | honored         | honored          | honored                 |
| `$ref` / `$defs`       | honored         | honored          | flattened               |
| `additionalProperties` | **required**    | ignored          | ignored                 |
| `maxItems`             | honored         | not enforced     | not enforced            |
| `format: date`         | not constrained | not constrained  | not constrained         |
| nullable / `anyOf`     | `["T","null"]`  | limited          | narrow                  |

> Fill this from your OWN runs — values differ by model/date. The point is to
> record what you observed, not to trust the docs.
```

**Expected / Verify.** `docs/honored_keywords.md` exists and each cell reflects behavior you actually observed (checks a DoD box).

**Troubleshoot.** If you can't tell whether a keyword was honored, make it *violable*: e.g. set `maxItems: 2` and prompt for an invoice with 5 line items — if you get 5 back, it wasn't enforced.

---

### Step 7 — `instructor` + a hand-rolled repair loop with a circuit breaker

**What.** Two things: (a) wrap OpenAI and Anthropic with `instructor` to get typed objects + automatic re-ask on validation error; (b) hand-roll your **own** loop so you understand what instructor does — classify each failure as transient / deterministic / unrecoverable, cap deterministic repairs at 3, and raise `EscalateToHuman` on unrecoverable input.

**Why.** Real inputs fail three different ways and each needs a different response: **transient** (429/5xx → exponential backoff + retry), **deterministic** (schema/validation error → feed the exact error back and re-ask, capped), **unrecoverable** (garbage/refusal → escalate to a human, don't loop forever burning money). The circuit breaker is what keeps the 2 garbage inputs from crashing or spinning.

**Do it** — `structio/structio/repair.py`:

```python
import time
import instructor
from openai import OpenAI, RateLimitError, APIStatusError
from pydantic import ValidationError
from structio.config import OPENAI_MODEL, ANTHROPIC_MODEL, require
from structio.schemas.invoice import Invoice


class EscalateToHuman(Exception):
    """Unrecoverable: refusal, or repeated deterministic failure — route to review."""


# --- (a) the instructor way: patch the client, pass response_model + max_retries ---
_ins_openai = instructor.from_openai(OpenAI(api_key=require("OPENAI_API_KEY")))

def extract_with_instructor(text: str) -> Invoice:
    return _ins_openai.chat.completions.create(
        model=OPENAI_MODEL,
        response_model=Invoice,     # typed return
        max_retries=3,              # instructor re-asks, feeding the ValidationError back
        messages=[{"role": "user", "content": text}],
    )


# --- (b) the hand-rolled way: classify → transient / deterministic / unrecoverable ---
_raw_openai = OpenAI(api_key=require("OPENAI_API_KEY"))
SYS = "Extract invoice fields into the schema. Fill `reasoning` FIRST."

def extract_with_repair(text: str, max_deterministic: int = 3) -> Invoice:
    messages = [{"role": "system", "content": SYS}, {"role": "user", "content": text}]
    deterministic_attempts = 0
    transient_attempts = 0

    while True:
        try:
            comp = _raw_openai.beta.chat.completions.parse(
                model=OPENAI_MODEL, messages=messages, response_format=Invoice,
            )
            msg = comp.choices[0].message
            if msg.refusal:                                  # unrecoverable
                raise EscalateToHuman(f"refusal: {msg.refusal}")
            return msg.parsed

        except RateLimitError:                               # transient
            transient_attempts += 1
            if transient_attempts > 5:
                raise EscalateToHuman("rate limited after 5 backoffs")
            time.sleep(min(2 ** transient_attempts, 30))     # exponential backoff

        except APIStatusError as e:                          # transient iff 5xx
            if e.status_code >= 500:
                transient_attempts += 1
                if transient_attempts > 5:
                    raise EscalateToHuman("server errors after 5 backoffs")
                time.sleep(min(2 ** transient_attempts, 30))
            else:
                raise                                        # 4xx: our bug, don't loop

        except ValidationError as ve:                        # deterministic
            deterministic_attempts += 1
            if deterministic_attempts >= max_deterministic:  # circuit breaker
                raise EscalateToHuman(
                    f"failed schema after {deterministic_attempts} repairs: {ve}"
                )
            # feed the EXACT error back and re-ask
            messages.append({"role": "user",
                             "content": f"Your output failed validation:\n{ve}\nFix it and return valid data."})
```

**Expected result.** A valid input returns an `Invoice`; a garbage input eventually raises `EscalateToHuman` (after ≤3 deterministic attempts) rather than crashing or looping forever.

**Verify.**

```bash
uv run python -c "
from structio.repair import extract_with_repair, EscalateToHuman
from structio.data import load_invoices
rows = load_invoices()
ok = extract_with_repair(rows[0]['text']); print('ok:', ok.vendor_name)
try:
    extract_with_repair(next(r for r in rows if r['expect']=='garbage')['text'])
    print('BUG: garbage did not escalate')
except EscalateToHuman as e:
    print('escalated as expected:', str(e)[:60])
"
```

**Troubleshoot.**
- Garbage input returns an `Invoice` of made-up junk instead of escalating → the model hallucinated valid-looking fields. Add a rule to `SYS`: "If the text is not an invoice, you must refuse." Then the `msg.refusal` branch catches it. (Some garbage will still parse; that's fine as long as it doesn't crash — the 2 designated garbage rows should escalate.)
- Infinite loop → confirm the deterministic branch increments and the circuit breaker (`>= max_deterministic`) is reachable.
- To repair **Anthropic** too, swap in `instructor.from_anthropic(anthropic.Anthropic())` with `response_model=Invoice, max_retries=3` (add `max_tokens=1024`).

---

### Step 8 — Stream partial JSON safely

**What.** Stream one extraction, render the object as it fills in with `rich`, and **only accept the object after `Invoice.model_validate` passes on the completed result**.

**Why.** Providers emit partial JSON while streaming; a half-formed object can look parseable but be semantically wrong. Use a tolerant partial parser to *render* progress, but only *act* on a validated, complete object.

**Do it** — `structio/structio/stream.py` (uses instructor's partial-streaming API):

```python
import instructor
from openai import OpenAI
from rich.live import Live
from rich.pretty import Pretty
from structio.config import OPENAI_MODEL, require
from structio.schemas.invoice import Invoice

_client = instructor.from_openai(OpenAI(api_key=require("OPENAI_API_KEY")))


def stream_extract(text: str) -> Invoice:
    stream = _client.chat.completions.create_partial(   # yields growing partial objects
        model=OPENAI_MODEL,
        response_model=Invoice,
        messages=[{"role": "user", "content": text}],
    )
    last = None
    with Live(refresh_per_second=8) as live:
        for partial in stream:
            last = partial
            live.update(Pretty(partial))                # render progress; DO NOT act on it

    # Only now, on the completed object, enforce the full contract:
    final = Invoice.model_validate(last.model_dump())
    return final


if __name__ == "__main__":
    from structio.data import load_invoices
    inv = stream_extract(load_invoices()[0]["text"])
    print("\nACCEPTED (validated):", inv.vendor_name, inv.total_usd)
```

**Expected result.** You see the object populate field-by-field in the terminal, then a final `ACCEPTED (validated)` line. The partial objects have `Optional`/missing fields mid-stream; the final one passes full validation (including `totals_must_sum`).

**Verify.** Add an `assert isinstance(inv, Invoice)` and confirm the `totals_must_sum` validator runs on the final object (it does, because `model_validate` runs validators — a partial that violated it would raise here, proving you never "acted" on a half-object).

**Troubleshoot.**
- `create_partial` not found → your `instructor` is old; `uv add "instructor>=1.0"`. Alternative: stream raw and feed chunks to the `partial-json-parser` library, validating only the final assembled string.
- The `Live` display flickers on Windows Git-Bash → lower `refresh_per_second`, or drop the `Live` and just `print(partial)` each iteration.

---

### Step 9 — Wire a `pytest` that proves the contract

**What.** Parametrize over the 20 inputs × providers; assert the **schema-valid rate (≥18/20)**, that garbage escalates, and that a hand-crafted broken invoice is **rejected by the totals validator**.

**Why.** The DoD is a merge gate — a green `pytest` is the objective proof.

**Do it** — `structio/tests/test_schema.py`:

```python
import pytest
from pydantic import ValidationError
from structio.schemas.invoice import Invoice, LineItem, Category
from structio.data import load_invoices
from structio.repair import extract_with_repair, EscalateToHuman

ROWS = load_invoices()
VALID = [r for r in ROWS if r["expect"] == "ok"]
GARBAGE = [r for r in ROWS if r["expect"] == "garbage"]


def test_totals_validator_rejects_bad_sum():
    with pytest.raises(ValidationError):
        Invoice(reasoning="x", vendor_name="V", invoice_date="2026-01-01",
                total_usd=999.0, category=Category.other,
                line_items=[LineItem(description="a", qty=1, unit_price_usd=1.0)])


@pytest.mark.parametrize("row", GARBAGE, ids=lambda r: f"garbage-{r['id']}")
def test_garbage_escalates(row):
    with pytest.raises(EscalateToHuman):
        extract_with_repair(row["text"])


@pytest.mark.live  # hits the API — run explicitly; see marker note below
def test_schema_valid_rate_openai():
    ok = 0
    for row in VALID:
        try:
            inv = extract_with_repair(row["text"])
            assert isinstance(inv, Invoice)
            ok += 1
        except EscalateToHuman:
            pass
    assert ok >= 18, f"only {ok}/{len(VALID)} valid inputs extracted"
```

Register the `live` marker so it doesn't warn — add `tests/conftest.py`:

```python
def pytest_configure(config):
    config.addinivalue_line("markers", "live: hits real provider APIs (costs tokens)")
```

**Expected / Verify.**

```bash
uv run pytest -q -m "not live"          # fast, offline: validator + escalation logic
uv run pytest -q -m live                # slower: real API calls, asserts >=18/20
```

Expected: both green. `-m "not live"` runs instantly (no tokens); `-m live` spends a few cents and asserts the 18/20 gate.

**Troubleshoot.**
- `live` test flaky at exactly 18 → your dataset's `ok` rows may be too ambiguous; tighten the invoice texts so a competent model can extract them. The garbage rows should stay in the `garbage` set (they're allowed to escalate).
- Import errors in tests → run via `uv run pytest` from the repo root so `structio` is importable (or `uv pip install -e .`).

---

### Free / local path (no paid keys)

**What / Why.** Run Tiers 1–2 against a local Ollama model via its OpenAI-compatible endpoint, so you can do the whole exercise offline. Ollama's schema enforcement is weaker than provider strict mode — a deliberate contrast you'll fix with constrained decoding (Outlines) in Week 2.

**Do it:**

```bash
ollama pull llama3.2         # ~2GB; runs on CPU
# point the OpenAI client at Ollama:
```

```python
from openai import OpenAI
local = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key ignored
# Tier 2 (JSON mode): guarantees parseable JSON, NOT your schema.
resp = local.chat.completions.create(
    model="llama3.2",
    response_format={"type": "json_object"},
    messages=[{"role": "system", "content": "Return JSON matching: " + str(Invoice.model_json_schema())},
              {"role": "user", "content": text}],
)
# You must Invoice.model_validate_json(...) yourself, and expect it to fail more often.
```

**Expected result.** You get parseable JSON, but fields are often missing or mis-typed — `Invoice.model_validate_json` fails more than with strict mode. **That gap is the point:** Tier 2 ≠ your schema.

**Verify.** Run the same 20 inputs through the local Tier-2 path and record the valid rate; it will be materially below the strict-mode providers. Note this in your README as motivation for Week 2's Outlines lab.

**Troubleshoot.** Connection refused → `ollama serve` must be running (the desktop app or `ollama serve` in another terminal).

---

## Putting it together — one end-to-end run

Run all three providers over the dataset and print a per-provider valid-rate table:

```python
# structio/structio/run_all.py
from rich.table import Table
from rich.console import Console
from structio.data import load_invoices
from structio.providers.openai_so import extract_openai
from structio.providers.anthropic_so import extract_anthropic
from structio.providers.gemini_so import extract_gemini

PROVIDERS = {"openai": extract_openai, "anthropic": extract_anthropic, "gemini": extract_gemini}


def main():
    rows = load_invoices()
    valid = [r for r in rows if r["expect"] == "ok"]
    table = Table(title="Tier-3 schema-valid rate (valid inputs only)")
    table.add_column("provider"); table.add_column("valid"); table.add_column("escalated/errors")
    for name, fn in PROVIDERS.items():
        ok = err = 0
        for r in valid:
            try:
                fn(r["text"]); ok += 1
            except Exception:
                err += 1
        table.add_row(name, f"{ok}/{len(valid)}", str(err))
    Console().print(table)


if __name__ == "__main__":
    main()
```

```bash
uv run python -m structio.run_all
```

**Expected:** a table showing ≥18/20 valid for each provider (garbage inputs are excluded here; they belong to the escalation test).

---

## Definition of Done — verifiable checks

Restating the spine's Week 1 checklist. Each box maps to a command above.

- [ ] **`Invoice` emits valid JSON Schema; every field has a description.** → Step 1 verify one-liner prints `every field has a description`.
- [ ] **All three provider functions return a validated `Invoice` for ≥18/20 valid inputs; the 2 garbage inputs route to escalation, not crash.** → `uv run python -m structio.run_all` (≥18/20 each) + `uv run pytest -q -m live` (escalation test green).
- [ ] **Logged, per provider, which JSON-Schema keywords were honored vs silently dropped** (markdown table in the repo). → `docs/honored_keywords.md` filled from your own runs (Step 6).
- [ ] **The totals validator rejects a hand-crafted invoice whose line items don't sum to the total** (a passing test proves it). → `test_totals_validator_rejects_bad_sum` green.
- [ ] **The repair loop caps at 3 deterministic attempts and raises `EscalateToHuman` on unrecoverable input; transient 429s are retried with backoff.** → `test_garbage_escalates` green + read `repair.py` branches.
- [ ] **Streaming demo renders progressively and only accepts the final object after validation.** → `uv run python -m structio.stream` shows fill-in then `ACCEPTED (validated)`.
- [ ] **`pytest` is green.** → `uv run pytest -q -m "not live"` and `uv run pytest -q -m live` both pass.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `additionalProperties` 400 (OpenAI) | Hand-written schema missing `additionalProperties: false` on an object | Use `response_format=Invoice` (parse helper); or add it to every object |
| Optional field rejected in OpenAI strict | Field omitted from `required` | Model as `Optional[T]` (nullable union) and keep it in `required` |
| Anthropic returns no `tool_use` block | `tool_choice` not forcing the tool, or a refusal | Force `{"type":"tool","name":"record_invoice"}`; treat refusals as escalation |
| Anthropic tool args are a string | Wrong assumption — they're a **parsed dict** | Read `block.input` directly; no `json.loads` |
| Gemini `resp.parsed` is `None` | Dialect couldn't round-trip the schema | Fall back to `Invoice.model_validate_json(resp.text)`; log dropped keywords |
| Model 404 | Model retired (e.g. `claude-3-5-haiku`) | Use `claude-haiku-4-5` / current cheap IDs; set via `.env` |
| Repair loop never terminates | Deterministic counter/circuit breaker not reached | Ensure `deterministic_attempts >= max_deterministic` raises |
| Garbage input yields fake `Invoice` | Model hallucinated fields | Add "if not an invoice, refuse" to system prompt |
| Streaming acts on half-object | Validating a partial | Only `model_validate` the **final** assembled object |
| `create_partial` missing | Old `instructor` | `uv add "instructor>=1.0"` or use `partial-json-parser` |
| `import google.genai` fails | Wrong package | `uv add google-genai`; import `from google import genai` |
| Cache/latency panic on OpenAI | One-time schema compile | Expected; don't drop strict mode to "fix" it |

---

## Stretch goals (optional)

- **Zod parity (TypeScript).** Re-implement `Invoice` with Zod + `zod-to-json-schema`, and get schema-valid output via the Vercel AI SDK's `generateObject`. Confirm the same honored-keywords findings from a JS stack.
- **Confidence-aware repair.** Add an optional `confidence: float` field and route low-confidence extractions to human review even when they're schema-valid.
- **Cost/latency trace.** Emit a JSONL line per extraction (provider, first-call vs cached latency, input/output tokens from `usage`) — this is the reliability-eval fuel you'll expand in Week 2.
- **Reason-then-extract two-pass.** Compare the single-pass `reasoning`-first schema against a two-call pass (free-text reasoning, then extract) on the hardest 5 inputs; note the quality/cost tradeoff.
- **Ollama + Outlines preview.** Peek ahead: install `outlines`, force the `Invoice` schema on `llama3.2`, and confirm it can't emit malformed JSON even under an adversarial "ignore the format" prompt — the Week 2 payoff.
