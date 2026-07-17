# Week 1 Lab: Build `docextract` — Document Image to Validated JSON with BBoxes, Confidence, and Review Routing

> This week you build **`docextract`**, a service that turns a document image (a phone photo of a receipt or a born-digital PDF invoice) into **schema-validated JSON** carrying a **value, a normalized bounding box, and a confidence score for every field** — then gates that JSON through **deterministic arithmetic checks** and routes anything shaky to a human review queue. This is the seed of **Milestone A** (the receipt→JSON pipeline you harden and cost-cut in Week 3), so build it like production code, not a notebook throwaway. The core discipline is the one you have carried since Phase 2: treat the image as **untrusted input**, force **structured output**, and let **code — not the model — decide** what ships.
>
> Read these first — this guide assumes you have:
> - [../lectures/01-vlm-architecture.md](../lectures/01-vlm-architecture.md) — encoder→projector→LLM; why an image is "just more tokens."
> - [../lectures/02-image-tokens-tiling-cost.md](../lectures/02-image-tokens-tiling-cost.md) — tiling, `detail:low`, and estimating token cost from pixel dimensions *before* you send.
> - [../lectures/03-structured-vlm-extraction.md](../lectures/03-structured-vlm-extraction.md) — the `FieldVal` pattern, `instructor`-over-`litellm`, and where VLM grounding (bboxes, counting) breaks.
> - [../lectures/04-ocr-vs-vlm-hybrid-routing.md](../lectures/04-ocr-vs-vlm-hybrid-routing.md) — OCR vs VLM vs hybrid, and born-digital vs scanned routing.
> - [../lectures/05-confidence-routing-validation.md](../lectures/05-confidence-routing-validation.md) — "LLM proposes, code disposes"; arithmetic validation and why self-reported confidence is uncalibrated.

**Est. time:** ~9 hrs (bump to ~11 if your documents are messy scans) · **You will need:** Python 3.11+, `uv`, and **one** of: a **Gemini** API key (cheapest per image, generous free tier — recommended) *or* **OpenAI/Claude** *or* a fully free/local path via **Ollama + `qwen2.5vl`** (no key, no cost, slower on CPU). PaddleOCR runs CPU-only for the hybrid step. You also need **15–20 real receipts/invoices** you collect yourself (see setup).

---

## Before you start (setup)

### 0.1 — Scaffold the project with `uv`

**What:** create the repo, add dependencies, lay out `src/`.
**Why:** a reproducible, isolated environment (Phase 0 discipline) so the heavy PaddleOCR install happens once and every teammate gets the same versions.

```bash
uv init docextract && cd docextract
uv add pydantic instructor "litellm>=1.50" pymupdf pillow python-dotenv rich
uv add paddleocr paddlepaddle          # dedicated OCR; heavy (~500MB+ of wheels), install once
uv add --dev pytest
mkdir -p src samples out tests
touch src/__init__.py
```

Target layout (you create these files across the steps below):

```
docextract/
  src/
    __init__.py
    schema.py         # Pydantic models: BBox, FieldVal, LineItem, Receipt
    detect.py         # born-digital vs scanned; EXIF auto-rotate; downscale
    ocr.py            # PaddleOCR wrapper -> [(text, bbox, conf), ...]
    extract_vlm.py    # image -> Receipt via VLM (instructor + litellm)
    extract_hybrid.py # OCR text + image -> Receipt
    route.py          # arithmetic + confidence gating -> review_queue.jsonl / auto.jsonl
  samples/            # 15-20 receipts/invoices (mix born-digital PDF + phone photos)
  out/                # extracted JSON, review-queue items, rendered bbox verification
  tests/
  .env
```

**Verify:** `uv run python -c "import pydantic, instructor, litellm, fitz, PIL, paddleocr; print('deps ok')"` prints `deps ok`.
**Troubleshoot:** if `paddlepaddle` fails to build on Windows, ensure you are on 64-bit Python 3.11/3.12 (PaddlePaddle wheels lag the newest Python). If it still fails, you can defer it — only Step 4 (hybrid) needs it; the pure-VLM Definition-of-Done items do not.

### 0.2 — Credentials and `.env`

```bash
printf "GEMINI_API_KEY=your-key-here\n" > .env
printf ".env\nout/\n__pycache__/\n.venv/\n" > .gitignore
```

`litellm` reads `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from the environment. Load it at the top of your scripts:

```python
from dotenv import load_dotenv; load_dotenv()
```

**Free/local path:** skip the key entirely. Install Ollama (Windows installer from https://ollama.com; `brew install ollama` on macOS) and pull the VLM once:

```bash
ollama pull qwen2.5vl        # ~6GB; runs CPU-only, slow but free, no key
curl http://localhost:11434/v1/models   # confirm the server is up
```

Then pass `model="ollama/qwen2.5vl"` wherever this guide uses `gemini/gemini-2.0-flash`.

### 0.3 — Collect your sample set (do this now — it gates everything)

**What:** gather **15–20 real receipts/invoices** into `samples/`. Deliberately mix:
- **5–8 born-digital PDFs** (email invoices, PDF receipts with a real text layer),
- **8–12 phone photos** (crumpled receipts, thermal-paper fades, rotated shots, one non-English/foreign-currency, one that is *not* a receipt for the pre-filter later).

**Why:** the Definition of Done requires 20/20 samples to extract cleanly, one born-digital + one scanned proven in tests, and 5 "hardest" samples for the hybrid comparison. A synthetic set hides exactly the failures (EXIF rotation, faded totals, tiling cost) this lab teaches you to catch.

**Verify:** `ls samples/ | wc -l` shows 15–20; you have at least one `.pdf` and one `.jpg`/`.png` phone photo.

---

## Step-by-step

Each step names which Definition-of-Done (DoD) item it satisfies.

### Step 1 — Define the schema: value + bbox + confidence per field

**What:** the Pydantic contract every extraction must satisfy. → *DoD: schema-valid `Receipt`; every field carries confidence in [0,1] and a normalized bbox.*

**Why:** the schema is the spec. By making `confidence` a **required** field and `value`/`bbox` **optional**, you force the model to self-assess every field and give it an honest "this field is absent" escape hatch (null value) instead of hallucinating. See [../lectures/03-structured-vlm-extraction.md](../lectures/03-structured-vlm-extraction.md) for the `FieldVal` pattern.

**Do it — `src/schema.py`:**

```python
# src/schema.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class BBox(BaseModel):
    """Normalized page coordinates, all in [0,1]. Origin = top-left."""
    x: float = Field(..., ge=0, le=1, description="left edge, fraction of page width")
    y: float = Field(..., ge=0, le=1, description="top edge, fraction of page height")
    w: float = Field(..., ge=0, le=1, description="width as fraction of page width")
    h: float = Field(..., ge=0, le=1, description="height as fraction of page height")


class FieldVal(BaseModel):
    value: Optional[str] = Field(None, description="extracted text; null if the field is absent")
    bbox: Optional[BBox] = Field(None, description="normalized location on the page; null if unknown")
    confidence: float = Field(..., ge=0, le=1, description="model self-reported 0-1 (uncalibrated)")


class LineItem(BaseModel):
    description: FieldVal
    qty: FieldVal
    unit_price: FieldVal
    amount: FieldVal


class Receipt(BaseModel):
    merchant: FieldVal
    date: FieldVal
    currency: FieldVal
    subtotal: FieldVal
    tax: FieldVal
    total: FieldVal
    line_items: list[LineItem]
```

**Expected result:** the module imports and a round-trip works.
**Verify:**

```bash
uv run python -c "from src.schema import Receipt, FieldVal, BBox; \
r = Receipt(merchant=FieldVal(value='Acme', confidence=0.9), date=FieldVal(confidence=0.0), \
currency=FieldVal(value='USD', confidence=0.9), subtotal=FieldVal(value='10.00', confidence=0.8), \
tax=FieldVal(value='0.80', confidence=0.8), total=FieldVal(value='10.80', confidence=0.8), line_items=[]); \
print(r.model_dump_json()[:80], '...')"
```

**Troubleshoot:** a `ValidationError` on `confidence` means you forgot it — it is required by design (`...`). If you want the `ge`/`le` bounds to reject a model returning `1.2`, that's intentional; `instructor` will re-prompt on it (Step 3).

---

### Step 2 — `detect.py`: born-digital vs scanned, EXIF rotate, downscale

**What:** decide whether a PDF already has readable text (skip OCR/VLM entirely) vs is just pixels; and normalize any image (EXIF auto-rotate + downscale long edge to ~1600px). → *DoD: born-digital PDFs read via PyMuPDF with no VLM call; downscaling to ≤1600px applied before VLM calls; token count recorded before/after.*

**Why:** OCR-ing text you could read for free is pure waste, and a 12MP phone photo at high detail costs thousands of tokens for zero quality gain on a receipt (see [../lectures/02-image-tokens-tiling-cost.md](../lectures/02-image-tokens-tiling-cost.md)). Forgetting EXIF means the model reads sideways text badly ([../lectures/04-ocr-vs-vlm-hybrid-routing.md](../lectures/04-ocr-vs-vlm-hybrid-routing.md)).

**Do it — `src/detect.py`:**

```python
# src/detect.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import fitz                      # PyMuPDF
from PIL import Image, ImageOps

MAX_EDGE = 1600                  # downscale long edge to this before any VLM call


@dataclass
class DetectResult:
    kind: str                    # "born_digital" | "scanned_pdf" | "image"
    text: str | None = None      # extracted text layer, if born-digital
    image_path: str | None = None  # normalized image on disk, if we need pixels


def _has_text_layer(doc: fitz.Document, min_chars: int = 20) -> bool:
    """Born-digital if any page yields a non-trivial text layer."""
    for page in doc:
        if len(page.get_text("text").strip()) >= min_chars:
            return True
    return False


def normalize_image(src: str | Path, out_dir: str | Path = "out") -> str:
    """EXIF auto-rotate, then downscale the long edge to MAX_EDGE. Returns new path."""
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)          # respect phone rotation metadata
    img = img.convert("RGB")
    long_edge = max(img.size)
    if long_edge > MAX_EDGE:
        scale = MAX_EDGE / long_edge
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    dst = out_dir / f"{src.stem}_norm.jpg"
    img.save(dst, "JPEG", quality=90)
    return str(dst)


def render_pdf_page(pdf_path: str | Path, page_no: int = 0, out_dir="out") -> str:
    """Rasterize one PDF page to an image for the scanned/image path."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    pix = doc[page_no].get_pixmap(dpi=200)
    raw = out_dir / f"{Path(pdf_path).stem}_p{page_no}.png"
    pix.save(raw)
    return normalize_image(raw, out_dir)        # still downscale/normalize


def detect(path: str | Path) -> DetectResult:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        if _has_text_layer(doc):
            text = "\n".join(p.get_text("text") for p in doc)
            return DetectResult(kind="born_digital", text=text)
        return DetectResult(kind="scanned_pdf", image_path=render_pdf_page(path))
    # plain image (phone photo)
    return DetectResult(kind="image", image_path=normalize_image(path))
```

**Expected result:** a born-digital PDF returns `kind="born_digital"` with text and **no image written**; a photo returns `kind="image"` with a normalized `*_norm.jpg` ≤1600px on the long edge.

**Verify:**

```bash
uv run python -c "from src.detect import detect; r=detect('samples/YOUR_DIGITAL.pdf'); \
print(r.kind, (r.text or '')[:60])"
uv run python -c "from src.detect import detect; from PIL import Image; \
r=detect('samples/YOUR_PHOTO.jpg'); print(r.kind, Image.open(r.image_path).size)"
```

**Record token count before/after downscaling** (a DoD item) — add this helper and run it on one big photo:

```python
# quick_token_probe.py  (throwaway; or fold into a notebook)
from dotenv import load_dotenv; load_dotenv()
from litellm import token_counter
from src.detect import normalize_image
import base64

def data_url(p):
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f"data:image/jpeg;base64,{b}"

def toks(path):
    msg = [{"role": "user", "content": [
        {"type": "text", "text": "extract"},
        {"type": "image_url", "image_url": {"url": data_url(path), "detail": "low"}}]}]
    return token_counter(model="gemini/gemini-2.0-flash", messages=msg)

orig = "samples/YOUR_BIG_PHOTO.jpg"
small = normalize_image(orig)
print("before:", toks(orig), "after:", toks(small))
```

Note both numbers in your writeup. (Even if `token_counter` estimates images coarsely for some providers, the point is to see the direction and magnitude.)

**Troubleshoot:** if a "born-digital" PDF is actually a scan wrapped with a junk text layer (some scanners embed a few garbage chars), raise `min_chars` or spot-check `page.get_text()` quality. If `ImageOps.exif_transpose` does nothing, the photo has no EXIF orientation tag — that's fine, it's a no-op.

---

### Step 3 — `extract_vlm.py`: image → validated `Receipt` via instructor

**What:** the pure-VLM extraction path — one model string, schema-coerced output, auto-retry. → *DoD: `extract()` returns schema-valid `Receipt` for 20/20 samples.*

**Why:** `instructor.from_litellm` wraps the raw completion so the model's JSON is validated against `Receipt` and **re-prompted up to `max_retries` times** if it violates the schema (e.g. `confidence=1.2`, missing field). Using `litellm` means swapping Gemini↔OpenAI↔Ollama is a one-string change. `detail:"low"` and the downscaled image keep tokens sane. See [../lectures/03-structured-vlm-extraction.md](../lectures/03-structured-vlm-extraction.md).

**Do it — `src/extract_vlm.py`:**

```python
# src/extract_vlm.py
from __future__ import annotations
import base64
from pathlib import Path
from dotenv import load_dotenv
import instructor
from litellm import completion
from .schema import Receipt

load_dotenv()
client = instructor.from_litellm(completion)

PROMPT = (
    "You are a precise document extractor. Extract every field of this receipt/invoice.\n"
    "Rules:\n"
    "- For each field give: the text value, a normalized bounding box (x,y,w,h each in 0-1, "
    "origin top-left), and a calibrated confidence in 0-1 reflecting how sure you are.\n"
    "- If a field is genuinely ABSENT from the document, set value to null and confidence to 0. "
    "Do NOT guess.\n"
    "- DO NOT invent or compute totals. Only report numbers you can actually read on the page. "
    "If the total is unreadable, null it.\n"
    "- Keep money as it appears (e.g. '12.99'); put the currency in the currency field."
)


def _img_url(path: str | Path) -> str:
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/jpeg;base64,{b}"


def extract(image_path: str | Path, model: str = "gemini/gemini-2.0-flash") -> Receipt:
    """Pure-VLM extraction. image_path must already be normalized (see detect.normalize_image)."""
    return client.chat.completions.create(
        model=model,
        response_model=Receipt,
        max_retries=2,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url",
             "image_url": {"url": _img_url(image_path), "detail": "low"}},
        ]}],
    )
```

**Free/local alt:** `extract(path, model="ollama/qwen2.5vl")`. Slower on CPU (tens of seconds/image) and its bboxes drift more, but zero cost and no key. Make sure `ollama serve` is running.

**Expected result:** a populated `Receipt` object; print it with `rich`.

**Verify:**

```bash
uv run python -c "from src.detect import detect; from src.extract_vlm import extract; \
from rich import print as rp; \
img = detect('samples/YOUR_PHOTO.jpg').image_path; \
rp(extract(img).model_dump())"
```

You should see merchant/date/total filled with plausible values, each with a `confidence` and (approximate) `bbox`. Absent fields should be `value=null, confidence=0`.

**Troubleshoot:**
- `instructor` raising after retries → the model can't satisfy the schema; loosen a constraint temporarily or bump `max_retries`, and check your image actually rendered (open the `*_norm.jpg`).
- Gemini `image_url` complaints → `litellm` handles the base64 data-URL for Gemini; ensure `litellm>=1.50`.
- All confidences suspiciously `0.99` → that's the whole point of Step 5; **do not trust it**, you'll validate arithmetically.
- Ollama connection refused → start the server; confirm `curl http://localhost:11434/v1/models`.

---

### Step 4 — `ocr.py` + `extract_hybrid.py`: OCR-grounded extraction, and the hallucination comparison

**What:** run PaddleOCR to get `(text, bbox, confidence)` tokens, dump them as text, and feed **both the OCR text and the image** to the VLM. Then compare hallucination on your 5 hardest samples. → *DoD: one-paragraph written comparison of pure-VLM vs hybrid on the 5 hardest samples with a recommendation.*

**Why:** the VLM reads layout/context well but fabricates plausible values; dedicated OCR gives grounded text with real geometry but no semantics. Feeding OCR text *into* the VLM anchors it to characters that actually exist on the page, cutting hallucination — the hybrid is often the best of both (see [../lectures/04-ocr-vs-vlm-hybrid-routing.md](../lectures/04-ocr-vs-vlm-hybrid-routing.md)).

**Do it — `src/ocr.py`:**

```python
# src/ocr.py
from __future__ import annotations
from functools import lru_cache
from paddleocr import PaddleOCR


@lru_cache(maxsize=1)
def _engine() -> PaddleOCR:
    # First call downloads models (~a few hundred MB) and caches them. CPU is fine.
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def ocr_tokens(image_path: str) -> list[tuple[str, list, float]]:
    """Return [(text, quad_bbox, confidence), ...]. quad_bbox is 4 (x,y) pixel points."""
    result = _engine().ocr(image_path, cls=True)
    tokens = []
    for page in (result or []):
        for line in (page or []):
            box, (text, conf) = line
            tokens.append((text, box, float(conf)))
    return tokens


def ocr_text_dump(image_path: str) -> str:
    """Flatten OCR tokens into a plain text block for the VLM prompt."""
    return "\n".join(f"{t}\t(ocr_conf={c:.2f})" for t, _, c in ocr_tokens(image_path))
```

**Do it — `src/extract_hybrid.py`:**

```python
# src/extract_hybrid.py
from __future__ import annotations
from pathlib import Path
from .extract_vlm import client, _img_url, PROMPT
from .ocr import ocr_text_dump
from .schema import Receipt


def extract_hybrid(image_path: str | Path, model: str = "gemini/gemini-2.0-flash") -> Receipt:
    ocr_dump = ocr_text_dump(str(image_path))
    grounded = (
        PROMPT
        + "\n\nAn OCR engine already read this page. Trust these tokens for the exact "
          "characters/numbers; use the image only for layout and to resolve ambiguity. "
          "Do not output any number that is absent from both the image and this OCR text:\n"
          f"---- OCR TOKENS ----\n{ocr_dump}\n---- END OCR ----"
    )
    return client.chat.completions.create(
        model=model, response_model=Receipt, max_retries=2,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": grounded},
            {"type": "image_url", "image_url": {"url": _img_url(image_path), "detail": "low"}},
        ]}],
    )
```

**Run the comparison** on your 5 hardest samples (faded, rotated, foreign-currency, crumpled):

```bash
uv run python -c "
from src.detect import detect
from src.extract_vlm import extract
from src.extract_hybrid import extract_hybrid
hard = ['samples/faded.jpg','samples/rotated.jpg','samples/foreign.jpg','samples/crumpled.jpg','samples/dense.jpg']
for p in hard:
    img = detect(p).image_path or p
    v = extract(img); h = extract_hybrid(img)
    print(p)
    print('  VLM    total:', v.total.value, 'conf', v.total.confidence)
    print('  HYBRID total:', h.total.value, 'conf', h.total.confidence)
"
```

**Expected result:** on faded/dense samples the hybrid should match the visible numbers more often; the pure VLM will sometimes emit a clean-looking but *wrong* total. Write a one-paragraph comparison (which won, on which sample types, and your recommendation — typically "hybrid for thermal/faded, pure-VLM for clean digital where OCR adds latency for no gain").

**Troubleshoot:**
- First `ocr()` call is slow → it's downloading models; cached afterward.
- PaddleOCR API differences across versions (`.ocr(img, cls=True)` vs `.predict()`) → check `paddleocr.__version__`; adjust the unpack of `line` if the tuple shape differs.
- Can't install Paddle at all → document that the hybrid path is CPU-optional and note it as a known gap; the *pure-VLM* DoD items still pass. (You can also substitute EasyOCR as a lighter fallback.)

---

### Step 5 — `route.py`: arithmetic + confidence gating

**What:** for every extracted `Receipt`, flag any field with `confidence < 0.75` **or** that fails an arithmetic check (line items sum to subtotal; subtotal + tax ≈ total). Write flagged docs to `out/review_queue.jsonl`, clean docs to `out/auto.jsonl`. → *DoD: arithmetic validator flags a corrupted-total doc into `review_queue.jsonl`.*

**Why:** self-reported confidence is **uncalibrated** — a model says 0.98 and is wrong ([../lectures/05-confidence-routing-validation.md](../lectures/05-confidence-routing-validation.md)). Deterministic arithmetic catches confident fabrications the model will never flag itself. This is "LLM proposes, **code disposes**."

**Do it — `src/route.py`:**

```python
# src/route.py
from __future__ import annotations
import json
from pathlib import Path
from .schema import Receipt

CONF_THRESHOLD = 0.75            # Week 3: calibrate this against the gold set
MONEY_TOL = 0.02                 # currency rounding slack


def _money(fv) -> float | None:
    if fv is None or fv.value is None:
        return None
    try:
        return float(str(fv.value).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def validate(r: Receipt) -> list[str]:
    """Return a list of human-readable reasons this doc should be reviewed. Empty = clean."""
    reasons: list[str] = []

    # 1) low-confidence fields
    named = {"merchant": r.merchant, "date": r.date, "currency": r.currency,
             "subtotal": r.subtotal, "tax": r.tax, "total": r.total}
    for name, fv in named.items():
        if fv.confidence < CONF_THRESHOLD:
            reasons.append(f"low_confidence:{name}={fv.confidence:.2f}")

    # 2) line items sum to subtotal
    subtotal, tax, total = _money(r.subtotal), _money(r.tax), _money(r.total)
    line_sum = sum(v for li in r.line_items if (v := _money(li.amount)) is not None)
    if subtotal is not None and r.line_items and abs(line_sum - subtotal) > MONEY_TOL:
        reasons.append(f"line_items_sum={line_sum:.2f}!=subtotal={subtotal:.2f}")

    # 3) subtotal + tax ~= total
    if None not in (subtotal, tax, total) and abs((subtotal + tax) - total) > MONEY_TOL:
        reasons.append(f"subtotal+tax={subtotal + tax:.2f}!=total={total:.2f}")

    return reasons


def route(doc_id: str, r: Receipt, out_dir: str = "out") -> bool:
    """Write r to auto.jsonl (clean) or review_queue.jsonl (flagged). Returns True if clean."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    reasons = validate(r)
    record = {"id": doc_id, "receipt": r.model_dump(), "reasons": reasons}
    target = "auto.jsonl" if not reasons else "review_queue.jsonl"
    with open(out / target, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return not reasons
```

**Expected result:** clean receipts land in `out/auto.jsonl`; anything with a low-confidence field or broken arithmetic lands in `out/review_queue.jsonl` **with the reasons attached** so a reviewer knows what to check.

**Verify:**

```bash
uv run python -c "
from src.schema import *
from src.route import validate
good = Receipt(merchant=FieldVal(value='A',confidence=0.9), date=FieldVal(value='2026-01-01',confidence=0.9),
  currency=FieldVal(value='USD',confidence=0.9), subtotal=FieldVal(value='10.00',confidence=0.9),
  tax=FieldVal(value='0.80',confidence=0.9), total=FieldVal(value='10.80',confidence=0.9),
  line_items=[LineItem(description=FieldVal(value='x',confidence=0.9), qty=FieldVal(value='1',confidence=0.9),
    unit_price=FieldVal(value='10.00',confidence=0.9), amount=FieldVal(value='10.00',confidence=0.9))])
print('clean reasons:', validate(good))            # []
bad = good.model_copy(deep=True); bad.total.value='99.99'
print('corrupted reasons:', validate(bad))         # subtotal+tax!=total
"
```

**Troubleshoot:** foreign currency with `.`/`,` swapped decimal separators will trip `_money` — normalize per-locale later; for now the tolerance and the review queue absorb it. If a legitimate receipt has a service charge/tip line the model didn't model, the arithmetic will fail correctly and route to review — that's a real gap in your schema, note it.

---

### Step 6 — Render one bbox back onto the image (verification)

**What:** draw a field's normalized bbox onto the original image with Pillow `ImageDraw` and eyeball that it lands on the right region. → *DoD: you can render at least one box back onto the image and it lands on the right region.*

**Why:** VLM bboxes are **approximate** — good enough to *highlight* a region for a human, never for pixel-perfect auto-cropping ([../lectures/03-structured-vlm-extraction.md](../lectures/03-structured-vlm-extraction.md)). This step proves your normalized→pixel conversion is correct and shows you how much they drift.

**Do it — add to `src/detect.py` (or a small `viz.py`):**

```python
# src/viz.py
from pathlib import Path
from PIL import Image, ImageDraw
from .schema import BBox


def draw_bbox(image_path: str, bbox: BBox, out_path: str = "out/bbox_check.png",
              label: str = "") -> str:
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    x0, y0 = bbox.x * W, bbox.y * H
    x1, y1 = (bbox.x + bbox.w) * W, (bbox.y + bbox.h) * H
    d = ImageDraw.Draw(img)
    d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=4)
    if label:
        d.text((x0, max(0, y0 - 14)), label, fill=(255, 0, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
```

**Verify:**

```bash
uv run python -c "
from src.detect import detect
from src.extract_vlm import extract
from src.viz import draw_bbox
img = detect('samples/YOUR_PHOTO.jpg').image_path
r = extract(img)
print('drew:', draw_bbox(img, r.total.bbox, label='total'))
"
```

Open `out/bbox_check.png` — the red box should sit **on or near** the total. If it's roughly in the right area (right zone of the page, right band), that's a pass; exact pixels are not expected.

**Troubleshoot:** box in a wildly wrong place → confirm you drew on the **same normalized image** you sent the model (not the original full-res, which has different dimensions). Box flipped vertically → check your origin assumption (this code uses top-left, matching the prompt). `r.total.bbox is None` → the model returned no box for that field; pick a field it did box, or re-prompt.

---

### Step 7 — Tests: 20/20 schema-valid + corrupted-total caught + routing proof

**What:** `pytest` that (a) every sample yields a schema-valid `Receipt`, (b) a born-digital PDF is read via the text layer with no image path, (c) the arithmetic validator flags a deliberately corrupted total. → *DoD: 20/20 samples schema-valid (verified by pytest); born-digital vs image routing proven; corrupted-total flagged.*

**Why:** the tests *are* the acceptance gate. To keep them fast and offline, cache each VLM extraction to disk once and let the schema/routing assertions run against the cache (real API calls in tests are slow, flaky, and cost money).

**Do it — `tests/conftest.py` (cache real extractions once):**

```python
# tests/conftest.py
import json, os
from pathlib import Path
import pytest
from src.detect import detect
from src.extract_vlm import extract
from src.schema import Receipt

CACHE = Path("out/extractions.json")


@pytest.fixture(scope="session")
def extractions():
    """Extract each sample once, cache to disk. Delete out/extractions.json to refresh."""
    samples = sorted(p for p in Path("samples").iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"})
    if CACHE.exists():
        raw = json.loads(CACHE.read_text())
        return {k: Receipt(**v) for k, v in raw.items()}
    model = os.getenv("DOCEXTRACT_MODEL", "gemini/gemini-2.0-flash")
    out = {}
    for p in samples:
        d = detect(p)
        if d.kind == "born_digital":
            continue                       # text-layer path exercised separately
        out[p.name] = extract(d.image_path, model=model)
    CACHE.write_text(json.dumps({k: v.model_dump() for k, v in out.items()}))
    return out
```

**Do it — `tests/test_docextract.py`:**

```python
# tests/test_docextract.py
from pathlib import Path
import pytest
from src.detect import detect
from src.schema import Receipt, FieldVal, LineItem
from src.route import validate


def test_all_samples_schema_valid(extractions):
    assert len(extractions) >= 12, "need your image samples extracted"
    for name, r in extractions.items():
        assert isinstance(r, Receipt), name
        for fv in (r.merchant, r.date, r.total):
            assert 0.0 <= fv.confidence <= 1.0, f"{name}:{fv}"


def test_born_digital_uses_text_layer():
    pdf = next((p for p in Path("samples").glob("*.pdf")
                if detect(p).kind == "born_digital"), None)
    assert pdf is not None, "add at least one born-digital PDF to samples/"
    r = detect(pdf)
    assert r.kind == "born_digital" and r.text and r.image_path is None


def test_scanned_or_photo_uses_image_path():
    photo = next((p for p in Path("samples").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
    assert photo is not None
    assert detect(photo).image_path is not None


def _receipt(total="10.80"):
    f = lambda v: FieldVal(value=v, confidence=0.9)
    return Receipt(merchant=f("Acme"), date=f("2026-01-01"), currency=f("USD"),
                   subtotal=f("10.00"), tax=f("0.80"), total=f(total),
                   line_items=[LineItem(description=f("x"), qty=f("1"),
                                        unit_price=f("10.00"), amount=f("10.00"))])


def test_clean_receipt_passes():
    assert validate(_receipt()) == []


def test_corrupted_total_is_flagged():
    reasons = validate(_receipt(total="99.99"))
    assert any("total" in r for r in reasons), reasons
```

**Expected result:** `pytest` green. The first run populates `out/extractions.json` (real API/Ollama calls); subsequent runs are instant and offline.

**Verify:**

```bash
uv run pytest -q
```

**Troubleshoot:** to force a fresh extraction, delete `out/extractions.json`. Free-path CI: `DOCEXTRACT_MODEL=ollama/qwen2.5vl uv run pytest`. If you have fewer than 20 usable samples, adjust the `>= 12` guard to your real image count but keep the born-digital + image tests — those prove the routing DoD.

---

## Putting it together — end-to-end run

A tiny driver that runs the whole pipeline over `samples/` and populates the queues:

```python
# run_all.py
from pathlib import Path
from rich import print as rp
from src.detect import detect
from src.extract_vlm import extract
from src.route import route

for p in sorted(Path("samples").iterdir()):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}:
        continue
    d = detect(p)
    if d.kind == "born_digital":
        rp(f"[green]{p.name}[/]: born-digital — read text layer, no VLM call")
        continue
    r = extract(d.image_path)
    clean = route(p.stem, r)
    rp(f"{'[green]AUTO' if clean else '[yellow]REVIEW'}[/] {p.name}: total={r.total.value}")
```

```bash
uv run python run_all.py
wc -l out/auto.jsonl out/review_queue.jsonl
```

**Expected:** born-digital PDFs print "no VLM call"; each photo prints AUTO or REVIEW; `out/auto.jsonl` + `out/review_queue.jsonl` together account for every image, and every line in `review_queue.jsonl` carries a `reasons` list.

---

## Definition of Done — verifiable checks

Restated from the spine ([../12-multimodal.md](../12-multimodal.md), Week 1). Each maps to a step above.

- [ ] **`extract()` returns a schema-valid `Receipt` for 20/20 sample images (pure-VLM path), verified by `pytest`.** → Step 3 + Step 7 (`test_all_samples_schema_valid`).
- [ ] **Born-digital PDFs read via PyMuPDF text layer (no OCR/VLM call); scanned/photo routed to the image path — proven with one of each in tests.** → Step 2 + Step 7 (`test_born_digital_uses_text_layer`, `test_scanned_or_photo_uses_image_path`).
- [ ] **Every field carries a confidence in [0,1] and a normalized bbox; you can render at least one box back onto the image and it lands on the right region.** → Step 1 + Step 6 (`out/bbox_check.png`).
- [ ] **The arithmetic validator flags a corrupted-total document into `review_queue.jsonl`.** → Step 5 + Step 7 (`test_corrupted_total_is_flagged`).
- [ ] **One-paragraph written comparison of pure-VLM vs hybrid hallucination on your 5 hardest samples, with a recommendation.** → Step 4.
- [ ] **Downscaling to ≤1600px applied before VLM calls; token count recorded before/after for one image.** → Step 2 (`quick_token_probe.py` output noted in your writeup).

You are done with Week 1 when all six boxes are checked and `uv run pytest -q` is green.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `paddlepaddle` won't install (Windows) | Python too new / 32-bit | Use 64-bit Python 3.11–3.12; or defer — only Step 4 needs it |
| `instructor` errors after retries | Model can't satisfy schema (e.g. `confidence>1`) | Bump `max_retries`; confirm the image rendered; loosen a constraint temporarily |
| Bbox drawn in wrong place | Drew on original full-res, not the normalized image | Always draw on the same `*_norm.jpg` you sent the model |
| Every field `confidence≈0.99` | Self-reported confidence is uncalibrated | Expected — rely on Step 5 arithmetic; calibrate the threshold in Week 3 |
| Huge token bill / slow calls | Full-res photo or `detail:high` | Downscale (Step 2) + `detail:"low"`; re-check `quick_token_probe.py` |
| Whisper-like "born-digital" false positive | Scanner embedded junk text layer | Raise `min_chars` in `_has_text_layer`; spot-check `get_text()` quality |
| Ollama "connection refused" | Server not running | `ollama serve` (macOS/Linux) / check tray (Windows); `curl :11434/v1/models` |
| Sideways text misread | Missing EXIF rotation | `ImageOps.exif_transpose` is in `normalize_image` — confirm you called it |
| Foreign-currency arithmetic false-flags | `,`/`.` decimal separator | Normalize in `_money` per locale; for now the review queue catches it |
| PaddleOCR unpack error | Version API drift | Check `paddleocr.__version__`; adjust `.ocr(...)` result unpacking |

---

## Stretch goals (optional)

- **Minimal review UI.** A Streamlit page (`review_ui/app.py`) that reads `out/review_queue.jsonl`, shows each image with **all** flagged-field bboxes highlighted (reuse Step 6), and lets you accept/correct a value. This is a required piece of Milestone A's acceptance — starting it now saves Week 3 time.
- **Multi-page invoices.** Extend `detect.py` to loop all PDF pages and merge line items across pages (page-spanning tables from the spine's objective 4).
- **Cheap-model pre-filter.** Add a tiny classifier call ("is this a receipt/invoice? yes/no") to reject non-receipts before the expensive extraction — this is the first cost lever you'll formalize in Week 3.
- **Confidence calibration preview.** Hand-label the true total for 10 samples and plot self-reported confidence vs actual correctness. You'll see it's poorly calibrated — motivating the gold-set calibration in Week 3.
- **Swap-model bake-off.** Run the same 5 hard samples through `gemini/gemini-2.0-flash`, `gpt-4o-mini`, and `ollama/qwen2.5vl`; note which reads faded totals best. Seeds the 3-model comparison table in Week 3.
