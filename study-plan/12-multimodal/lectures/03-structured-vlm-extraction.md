# Lecture 3: Schema-Validated Extraction — Pydantic + instructor, BBoxes, Confidence, and Grounding Limits

> A VLM will happily read a crumpled phone photo of a receipt and hand you a clean-looking JSON object. The trap is that "clean-looking" and "correct" are two different things, and the model gives you no way to tell them apart unless you *engineer* one. This lecture is about turning a general-purpose vision model into a **reliable structured extractor**: a Pydantic schema that forces every field to carry its own value, location, and self-assessment; an `instructor`-over-`litellm` client that auto-repairs invalid JSON; a prompt that gives the model an honest "I don't have this" escape hatch; and — most importantly — a clear-eyed map of exactly where VLM grounding *breaks* so you never build a pipeline on top of a number the model was never able to produce. After this you will be able to design the `FieldVal` pattern, wire up retrying schema coercion, pass images correctly and cheaply, and state precisely which VLM outputs you can trust auto-approving versus which you must route to a human.

**Prerequisites:** Phase 2 Pydantic + instructor (forcing & validating structured JSON — the parses-vs-correct distinction), Week 1 Lectures 1–2 on VLM architecture and image → token blocks, basic probability intuition · **Reading time:** ~30 min · **Part of:** Multimodal & Specialized Modalities, Week 1

---

## The core idea (plain language)

You have a VLM that can *read*. You want a service that *extracts* — that takes a receipt image and returns `{merchant, date, total, line_items, ...}` your ERP can ingest without a human in the loop for the easy 80%.

Three things stand between "the model can read" and "the service is reliable":

1. **Shape.** The output must be machine-parseable, always, not 95% of the time. That is constrained decoding + schema validation (Phase 2's structured-output discipline), now applied to pixels.
2. **Self-assessment.** For each field you need three things, not one: the **value** ("$42.17"), *where* it came from (a **bounding box** so a human can glance at the region), and *how sure* the model is (a **confidence**). A bare `total: 42.17` tells you nothing about whether to trust it.
3. **Honesty about limits.** The VLM is strong at reading and describing and *weak* at counting, precise spatial reasoning, and exact coordinates. If you treat its bounding boxes as pixel-perfect or its confidence as a calibrated probability, you will ship a pipeline that fails silently and confidently.

The pattern that ties 1 and 2 together is what we'll call **`FieldVal`**: instead of a flat model where each field is a scalar, *every* extracted field is wrapped in an object carrying `value + bbox + confidence`. That wrapping is the single most important design decision in this lecture, because it turns "the model's opinion of the whole document" into "the model's opinion of each field, separately routable." Point 3 is the reliability discipline that keeps you from over-trusting what points 1 and 2 produce.

---

## How it actually works (mechanism, from first principles)

### The FieldVal pattern — why wrap every field

Compare two schemas for the same receipt.

**Flat (the naïve version):**

```python
class ReceiptFlat(BaseModel):
    merchant: str
    total: float
    tax: float
```

**Wrapped (`FieldVal`):**

```python
from pydantic import BaseModel, Field
from typing import Optional

class BBox(BaseModel):
    x: float; y: float; w: float; h: float          # normalized 0-1, top-left origin

class FieldVal(BaseModel):
    value: Optional[str] = Field(None, description="extracted text; null if absent")
    bbox: Optional[BBox]  = Field(None, description="normalized location on the page")
    confidence: float     = Field(..., ge=0, le=1, description="model self-reported 0-1")

class LineItem(BaseModel):
    description: FieldVal; qty: FieldVal; unit_price: FieldVal; amount: FieldVal

class Receipt(BaseModel):
    merchant: FieldVal; date: FieldVal; currency: FieldVal
    subtotal: FieldVal; tax: FieldVal; total: FieldVal
    line_items: list[LineItem]
```

The flat model gives you `total = 42.17`. It has thrown away everything you need to *act* on that number safely. The wrapped model gives you `total = FieldVal(value="42.17", bbox=BBox(0.71, 0.88, 0.18, 0.04), confidence=0.93)`.

Why does wrapping *every* field — even `merchant`, which is easy — buy you something? **Per-field routing.** Once every field self-reports, your downstream logic is a loop over fields, not a monolithic accept/reject on the whole document:

```
for field in receipt:
    if field.confidence < THRESHOLD or arithmetic_check_failed(field):
        route_to_review(field)   # human sees ONLY this field + its bbox highlight
    else:
        auto_approve(field)
```

A flat schema forces an all-or-nothing decision: the whole receipt is trusted or the whole receipt goes to a human. With `FieldVal`, a receipt where `merchant` (conf 0.98) and `date` (conf 0.97) are obvious but `total` (conf 0.61) is smudged sends *one field* to review. On a 12-field document that is the difference between a reviewer confirming one number in 3 seconds and re-keying the entire receipt. The wrapping is cheap now and unlocks granular routing you'd otherwise have to bolt on with a schema migration later.

Keep `value` a **string** even for numbers. `"1.238,00"` (European), `"$1,238.00"`, `"1238"` are all things the model will read verbatim; if you type `value: float` the model must silently normalize, and normalization is *your* job in code where you can test it, not the model's job where it hides parsing errors. Store raw, parse in Python.

### instructor over litellm — coercion + auto-repair

`litellm` gives you one function, `completion(...)`, that speaks to 100+ providers via a single string (`"gemini/gemini-2.0-flash"`, `"gpt-4o-mini"`, `"ollama/qwen2.5vl"`). `instructor` wraps that function and adds two things:

1. **Schema injection.** You pass `response_model=Receipt`. instructor serializes the Pydantic model to a JSON Schema and injects it — via the provider's native structured-output mechanism where available (function-calling / `response_format`), or as instructions in the prompt as a fallback. You never hand-write the schema string.
2. **Validate-and-retry.** instructor calls the model, `json.loads` the output, and runs `Receipt.model_validate(...)`. If Pydantic raises (missing field, `confidence` out of `[0,1]`, malformed bbox), instructor **feeds the validation error back to the model as a new message** and asks it to fix it — up to `max_retries` times.

The retry loop is the mechanism worth internalizing. It's not magic; it's a conversation:

```
attempt 1 → model returns {"total": {"value": "42.17", "confidence": 1.4}}
Pydantic  → ValidationError: total.confidence: Input should be <= 1
instructor→ sends the model: "Your response failed validation: total.confidence
             must be <= 1. Fix and return valid JSON."
attempt 2 → model returns confidence 0.9  ✓  → Receipt object handed back
```

```python
import instructor
from litellm import completion
from .schema import Receipt

client = instructor.from_litellm(completion)

receipt = client.chat.completions.create(
    model="gemini/gemini-2.0-flash",
    response_model=Receipt,
    max_retries=2,
    messages=[...],   # text + image, below
)   # returns a validated Receipt instance, not a dict
```

Two consequences. **Each retry is a full extra model call** — latency and cost multiply. `max_retries=2` means a pathological input can cost 3× tokens and 3× wall-clock. That's fine for a fallback; it's a problem if *most* of your traffic retries, which usually means your schema is too strict or your prompt is unclear (fix the prompt, don't raise retries). And **retries fix *shape*, never *content*.** instructor can coerce the model into a valid `confidence ∈ [0,1]`; it cannot make that confidence *true*. This is the pixels-version of Phase 2's parses-vs-correct gate — instructor closes the parses gate for you and does nothing for the correct gate.

### Passing images — data URL vs URL, the content block, `detail`

An image rides in the same `messages` list as text, as a content block of type `image_url`. Two ways to supply it:

```python
import base64

def data_url(path):                       # inline: bytes travel in the request body
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:image/jpeg;base64,{b}"

messages=[{"role": "user", "content": [
    {"type": "text", "text": "Extract every field. ..."},
    {"type": "image_url", "image_url": {"url": data_url(path), "detail": "low"}},
]}]
```

- **base64 data URL** — the image bytes are embedded in the JSON request. Works for local files, no hosting needed; the cost is a larger request body (base64 inflates size ~33%). This is the default for a local extraction service.
- **remote URL** (`"url": "https://.../receipt.jpg"`) — the provider fetches it. Smaller request, but the URL must be publicly reachable by the provider and the fetch adds latency and a failure mode (dead link, auth wall). Use only for images already on a public CDN.
- **`detail`** (OpenAI-style; Gemini/Claude have analogous tiling schemes) — `"low"` sends a single downscaled tile (~85 tokens on OpenAI, roughly fixed and cheap); `"high"` tiles the image into 512px blocks, each costing its own token block. For a receipt, `detail: "low"` after downscaling the long edge to ~1024–1600px almost always reads every field and costs a fraction of `high`. Reach for `high` only when a specific field is genuinely unreadable at low res. (This is the cost lever you'll exploit in Week 3.)

---

## Worked example

One receipt, faded thermal paper, phone photo. The true values:

```
merchant: "Blue Bottle Coffee"    subtotal: 18.50
date:     "2026-03-14"            tax:       1.61
                                  total:     20.11
line items: Latte x2 @ 5.50 = 11.00 ;  Croissant x1 @ 7.50 = 7.50
```

You send the downscaled image with `detail: "low"` and this prompt (next section explains each clause):

> Extract every field into the schema. Give a normalized bbox (0–1, top-left origin) and a 0–1 confidence per field. If a field is absent, return value=null and confidence=0. Do NOT compute or invent totals — copy them exactly as printed; if a total is unreadable, return null.

The model returns (abbreviated):

```json
{
  "merchant": {"value": "Blue Bottle Coffee", "bbox": {"x":0.12,"y":0.05,"w":0.55,"h":0.06}, "confidence": 0.97},
  "date":     {"value": "2026-03-14",         "bbox": {"x":0.10,"y":0.14,"w":0.30,"h":0.04}, "confidence": 0.91},
  "subtotal": {"value": "18.50", "bbox": {"x":0.70,"y":0.60,"w":0.16,"h":0.04}, "confidence": 0.88},
  "tax":      {"value": "1.61",  "bbox": {"x":0.70,"y":0.66,"w":0.16,"h":0.04}, "confidence": 0.86},
  "total":    {"value": "21.11", "bbox": {"x":0.70,"y":0.74,"w":0.16,"h":0.05}, "confidence": 0.94},
  "line_items": [ ... ]
}
```

Look at `total`: the model returned **21.11** with confidence **0.94** — a confident wrong answer (true total 20.11; it misread a smudged 0 as a 1). The confidence is *higher* than on the correctly-read `subtotal`. This is the whole reliability problem in one field: **self-reported confidence did not track correctness.** If your routing threshold were `confidence < 0.75 → review`, this field sails through at 0.94 and corrupts your ERP.

What catches it is not the model — it's your code. The arithmetic check:

```
subtotal + tax = 18.50 + 1.61 = 20.11
model total    = 21.11
|20.11 - 21.11| = 1.00  >  tolerance (say 0.02)   → FLAG total to review
```

The line items sum to 11.00 + 7.50 = 18.50 = subtotal ✓, and subtotal + tax = 20.11 ≠ 21.11 ✗. The deterministic check flags `total` regardless of the model's confidence. **This is "LLM proposes, code disposes."** The VLM proposes values; arithmetic and thresholds dispose of the ones you can verify.

Now the bboxes. The `total` bbox `(0.70, 0.74, 0.16, 0.05)` — is it exactly on the total? On a 1000px-tall image, `y=0.74` means 740px down. The real total line might be at 700px. **A 4% drift = 40px** — enough that if you auto-cropped that box and fed it to a downstream OCR expecting a tight line, you'd clip the top of the digits or grab the line below. The box is *good enough to draw a highlight rectangle a human can find at a glance*, and *not good enough to crop and re-OCR blindly*. That distinction is the next section.

---

## How it shows up in production

**The three failure modes you will actually hit:**

1. **Fabricated-but-plausible values.** Ask for a field that's smudged or off-frame and the model, absent an escape hatch, invents something that *looks* right — a merchant name that's a plausible chain, a date in the right format, a tax that's a believable percentage. It's plausible precisely because the model is good at "what would a receipt say here." Mitigation: the prompt's explicit `null + confidence 0` path (below) and never `required`-without-null on genuinely-optional fields.
2. **Confident wrong totals.** As in the worked example: the model reads a number wrong and reports high confidence. Totals are the highest-value field and the one most likely to be quietly wrong. Mitigation: arithmetic validation is non-negotiable. If the numbers don't sum, you *know* something is wrong even when the model swears otherwise.
3. **Bounding boxes off by 10–20%.** VLM coordinates drift because the model was never trained to emit pixel-precise geometry — it predicts coordinate *tokens* the way it predicts words, by plausibility, not by measurement. Expect boxes that are roughly-right in the correct region and systematically loose. On a 2000px image, 15% drift is 300px. Mitigation: use boxes for human-review *highlighting only*; when you need real geometry (auto-crop, layout), use a dedicated OCR/layout model (PaddleOCR, Textract, Docling) that returns measured coordinates + per-token confidence.

**Why VLM grounding is limited — the mechanism.** A VLM reads by attending over image *token blocks* and generating text tokens. "Reading the word on this receipt" is close to its training objective (describe/transcribe what's visible). "Report the pixel coordinates of that word" and "count how many line items there are" and "which box is to the left of which" are *spatial/quantitative* tasks the architecture handles poorly: coordinates are emitted as ordinary generated numbers (subject to the same plausibility bias as any token), and counting requires holding a precise tally the attention mechanism doesn't reliably maintain past small n. So: trust it to *read a value*; distrust it to *count objects*, *reason about precise spatial relationships*, or *emit exact coordinates*.

**Confidence is uncalibrated — treat it as a hint, not a probability.** "confidence 0.9" does *not* mean "right 90% of the time." The number is a self-report the model generates like any other token; it is systematically overconfident and its scale is arbitrary and model-specific. You may find that on *your* data, 0.9-confidence fields are right 82% of the time and 0.6-confidence fields right 78% — i.e., barely separable. **You cannot know your routing threshold is meaningful until you validate confidence against a gold set** (Week 3): label N receipts, bin predictions by reported confidence, and measure actual correctness per bin. Only then can you say "confidence ≥ 0.85 auto-approves at 97% field precision" and defend the number. Setting a threshold from vibes is setting it wrong. *(Forward reference: Week 3's eval harness is where this calibration lives.)*

**Cost/latency.** Each `instructor` retry is a full call — monitor your retry rate; a spike means schema or prompt drift, not a model problem. `detail: "low"` + downscaling is the dominant image-cost lever (a 12MP photo at `high` can be thousands of tokens for zero quality gain on a receipt). base64 inflates request size ~33% vs a URL — irrelevant for one image, real if you batch dozens per request.

---

## Common misconceptions & failure modes

- **"High confidence means correct."** No. Confidence is uncalibrated self-report; it is routinely high on wrong values (worked example). Validate it against ground truth before trusting a threshold.
- **"The bbox is where the field is."** Approximately. Expect 10–20% drift. Fine for highlighting a region to a human; wrong for pixel-perfect auto-cropping. For real coordinates use dedicated OCR/layout models.
- **"VLMs can count the line items."** Unreliably past small numbers. If you depend on an exact count, verify it (e.g., cross-check line-item sum against subtotal) rather than trusting the model's tally.
- **"instructor guarantees correct data."** It guarantees *schema-valid* data. Coercion and retries close the parses gate; the correct gate is your arithmetic checks and gold-set eval.
- **"Make every field `required` so nothing is missing."** That *causes* fabrication — a required, non-nullable field the model can't read forces a hallucinated value. Model absence explicitly: `value=null, confidence=0`.
- **"Type `value` as float so I get numbers."** Then the model normalizes silently and hides read errors. Keep `value` a string; parse and normalize in tested Python.
- **"Raising `max_retries` fixes flaky extraction."** It masks the symptom and multiplies cost. High retry rates mean the schema is too strict or the prompt is unclear — fix those.
- **"Bigger/`detail: high` image = better."** Usually not for receipts; it's a token bomb for no quality gain. Downscale + low detail first, escalate one field only if unreadable.

---

## Rules of thumb / cheat sheet

- **Wrap every field in `FieldVal` (value + bbox + confidence).** Even easy fields — it unlocks per-field routing without a later schema migration.
- **`value` is always a string.** Store raw as printed; normalize/parse in Python where you can test it.
- **Give the model an honest out:** prompt for `value=null, confidence=0` on absent fields; make those fields nullable in the schema. This kills fabricated-but-plausible values.
- **Prompt must say "do not invent totals — copy exactly; null if unreadable."** Then verify totals with arithmetic in code.
- **Arithmetic validation is mandatory:** `Σ line_items.amount ≈ subtotal` and `subtotal + tax ≈ total` (small tolerance). Catches confident-wrong totals the model won't flag.
- **BBoxes: highlight, never crop.** Assume 10–20% drift. Real geometry → PaddleOCR/Textract/Docling.
- **Confidence is a weak, uncalibrated hint.** Don't set a routing threshold until you've calibrated it against a gold set (Week 3).
- **Images:** base64 data URL for local files, remote URL only for public CDN. `detail: "low"` + downscale long edge to ~1024–1600px as the default; `high` only to rescue one unreadable field.
- **`instructor`:** `response_model=Receipt`, `max_retries=2`. Watch the retry rate; a spike = schema/prompt problem, not a model problem.
- **Swap models with one string** via `litellm` (`gemini/gemini-2.0-flash` ↔ `gpt-4o-mini` ↔ `ollama/qwen2.5vl`). Develop free/local before spending on frontier calls.

---

## Connect to the lab

This lecture is the theory behind Week 1's `src/schema.py`, `src/extract_vlm.py`, and `src/route.py`. In the lab you'll define exactly the `BBox / FieldVal / LineItem / Receipt` schema above, wire `instructor.from_litellm(completion)` with `response_model=Receipt` and `max_retries=2`, and pass downscaled images as `detail: "low"` data URLs. The Definition-of-Done — "every field carries a confidence in [0,1] and a normalized bbox; the arithmetic validator flags a corrupted-total document into `review_queue.jsonl`" — is precisely the FieldVal + code-disposes discipline here. When Week 3 has you calibrate the confidence threshold against a 25–40 doc gold set, that's this lecture's "don't trust confidence until you validate it" made concrete.

## Going deeper (optional)

- **instructor** docs and repo — the canonical source for `response_model`, `max_retries`, and the validate-retry loop: `python.useinstructor.com` (search: `instructor litellm response_model max_retries`).
- **litellm** docs for the unified completion interface and vision/`image_url` support: `docs.litellm.ai` (search: `litellm vision image_url provider prefix`).
- **Pydantic** docs for `Field(..., ge=0, le=1)`, `Optional`, and nested-model validation: `docs.pydantic.dev`.
- **OpenAI vision guide** and **Anthropic vision docs** for image token counting, tiling, and the `detail` parameter: `platform.openai.com/docs` and `docs.anthropic.com` (search: `OpenAI vision detail high low tokens`, `Anthropic vision image tokens`).
- **Qwen2.5-VL** model card on Hugging Face for a 2025 production-grade VLM with dynamic resolution and documented grounding behavior (search: `Qwen2.5-VL Hugging Face grounding`).
- Dedicated geometry when you need it: **PaddleOCR**, **AWS Textract**, and **Docling** GitHub READMEs (search: `Docling GitHub`, `PaddleOCR GitHub`).
- On why LLM/VLM confidence is uncalibrated: search `LLM calibration verbalized confidence` and `expected calibration error` for the background intuition (no proofs needed — you just need to know the number lies until measured).

## Check yourself

1. You could store `total` as a bare `float`. Why does wrapping it in `FieldVal` (value + bbox + confidence) change what your *routing* code can do?
2. `instructor` returns a schema-valid `Receipt` with `total.confidence = 0.96`. Your reviewer finds the total is wrong. Which of instructor's guarantees held, which never applied, and what code should have caught this?
3. Why must the extraction prompt explicitly offer a `null value + confidence 0` path for absent fields — what specific failure appears if you don't?
4. A teammate wants to auto-crop each field's bbox and run a tight OCR on the crop for a "second opinion." Why is this fragile, and what would you use instead for real coordinates?
5. You set the review threshold at `confidence < 0.75`. What must you do before you can defend that 0.75, and where in the course does that happen?
6. When would you pass an image as a remote URL instead of a base64 data URL, and what does `detail: "low"` buy you on a receipt?

### Answer key

1. A bare float supports only an all-or-nothing decision on the whole document. `FieldVal` lets you route **per field**: flag only `total` (low confidence or failed arithmetic) to a human while auto-approving `merchant`/`date`, so a reviewer confirms one number instead of re-keying the receipt. Wrapping every field now avoids a schema migration later.
2. The **shape/parses** guarantee held — instructor coerced and validated the JSON against the schema. The **correct** guarantee never existed; instructor and confidence say nothing about whether the value is right. The **arithmetic check** (`subtotal + tax ≈ total`, `Σ line_items ≈ subtotal`) should have caught it — deterministic verification, independent of the model's confidence.
3. Without an explicit "return null, confidence 0" path (and a nullable field), a field the model can't read becomes a **fabricated-but-plausible value** — a well-formatted invented merchant/date/amount — because the model fills the gap with what a receipt "should" say. The escape hatch converts a silent hallucination into an honest, routable "absent."
4. VLM bboxes drift 10–20%; a tight crop will clip digits or grab the wrong line, so the "second opinion" reads garbage and *looks* authoritative. Use a dedicated OCR/layout model (PaddleOCR, Textract, Docling) that returns **measured** coordinates + per-token confidence for anything requiring real geometry.
5. You must **calibrate** the threshold against a gold set: label receipts, bin predictions by reported confidence, and measure actual per-bin field correctness — only then can you state the auto-approved bucket's precision. That happens in **Week 3's eval harness**; a threshold picked from vibes is unjustified.
6. Use a **remote URL** only when the image is already on a public CDN the provider can fetch (smaller request body, but adds a fetch/latency/dead-link failure mode); use **base64 data URL** for local files. `detail: "low"` sends one downscaled tile (~fixed, cheap token cost) which reads every field on a downscaled receipt at a fraction of `high`'s tiled token cost.
