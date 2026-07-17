# Week 2 Lab: Build SlideChat (ColPali multimodal RAG) and a Cascaded Realtime Voice Agent

> This week you build **two** things. **Build A — SlideChat:** a "chat with your slide decks" app that indexes **PDF/deck page images** with ColPali/ColQwen (via **Byaldi**), retrieves the top-3 pages by **late-interaction (MaxSim)** scoring — *no OCR/parse pipeline at all* — and answers with a VLM that returns **page-number citations + thumbnails**. **Build B — a cascaded realtime voice agent:** microphone → **Silero VAD** endpointing → **faster-whisper** STT → LLM → **TTS**, with **barge-in** and a measured **<800ms** turn budget. Together they prove you can (a) retrieve over the visual substrate of a document and ground a VLM answer on it, and (b) wire the Phase 6 agent loop into a real-time speech I/O layer.
>
> Read these first — this guide assumes you have:
> - [../lectures/06-multimodal-embeddings-clip-siglip.md](../lectures/06-multimodal-embeddings-clip-siglip.md) — CLIP/SigLIP contrastive training, the shared image-text space, and why one global vector per image loses fine detail.
> - [../lectures/07-colpali-late-interaction.md](../lectures/07-colpali-late-interaction.md) — ColPali/ColQwen multi-vector embeddings, MaxSim late interaction, and why it skips OCR.
> - [../lectures/08-multimodal-rag-architecture.md](../lectures/08-multimodal-rag-architecture.md) — the three RAG patterns, tight top-k retrieval, and grounded VLM answers with citations.
> - [../lectures/09-stt-asr-whisper-vad-diarization.md](../lectures/09-stt-asr-whisper-vad-diarization.md) — Whisper vs faster-whisper vs WhisperX, VAD gating (why Whisper hallucinates on silence), word timestamps, diarization.
> - [../lectures/10-tts-streaming-synthesis.md](../lectures/10-tts-streaming-synthesis.md) — streaming synthesis and time-to-first-byte.
> - [../lectures/11-realtime-voice-agent-architecture.md](../lectures/11-realtime-voice-agent-architecture.md) — cascaded vs speech-to-speech, endpointing, barge-in, and the latency budget.

**Est. time:** ~10 hrs (Build A ~5, Build B ~5) · **You will need:** Python 3.11+, `uv`, `ffmpeg` on PATH, a working **microphone + speakers/headphones**, `poppler` (for `pdf2image`), and **2–3 real PDF decks**. **Free/local path throughout:** Byaldi runs **ColQwen locally** (GPU preferred; CPU works for a small deck — see the GPU note); STT is **faster-whisper on CPU** (`compute_type="int8"`, no key); TTS is **Piper** (free, local, offline); the answering VLM can be a local **Ollama** model (`llama3.2-vision` / `qwen2.5vl`) so the whole week can run with **no API key and no cost**. Where an API is faster (Gemini for answering, Cartesia/ElevenLabs for TTS) it is called out as optional.

---

## Before you start (setup)

### 0.1 — System prerequisites (ffmpeg + poppler)

**What:** install the two native tools both builds depend on.
**Why:** `pdf2image` shells out to poppler to rasterize PDF pages; audio capture/decoding and `faster-whisper` rely on `ffmpeg`.

```bash
# macOS
brew install ffmpeg poppler
# Ubuntu/Debian
sudo apt-get install -y ffmpeg poppler-utils
```

**Windows (Git-Bash):** the cleanest path is a package manager so both land on PATH:

```bash
winget install --id Gyan.FFmpeg -e        # ffmpeg
winget install --id oschwartz10612.Poppler # poppler (or download poppler-windows release and add its /bin to PATH)
```

If you cannot get poppler on PATH, pass its bin folder explicitly later: `convert_from_path(pdf, poppler_path=r"C:\path\to\poppler\bin")`.

**Verify:** `ffmpeg -version` and `pdftoppm -h` both print help (restart the shell after install so PATH refreshes).

### 0.2 — Scaffold both projects

**What:** two sibling `uv` projects so their (heavy, conflicting) dependency trees stay isolated. **Why:** ColPali pulls a large Torch/transformers stack; the voice agent pulls audio libs — keep them separate and reproducible (Phase 0 discipline).

```bash
# Build A
uv init slidechat && cd slidechat
uv add byaldi pdf2image pillow "litellm>=1.50" python-dotenv rich
uv add --dev pytest
mkdir -p src decks out eval tests && touch src/__init__.py
cd ..

# Build B
uv init voice_agent && cd voice_agent
uv add faster-whisper sounddevice numpy scipy "litellm>=1.50" python-dotenv rich
# VAD: the pip package ships the Silero ONNX model
uv add silero-vad
uv add --dev pytest
mkdir -p src clips out tests && touch src/__init__.py
cd ..
```

Target layouts:

```
slidechat/
  src/{__init__,index,retrieve,answer,viz}.py
  decks/            # your 2-3 real PDF decks
  out/             # thumbnails, rendered citations
  eval/            # queries.jsonl + eval.py (recall@k)
  tests/
voice_agent/
  src/{__init__,capture_vad,stt,tts,agent,bargein}.py
  clips/           # test wavs (a silent clip, a 2-speaker clip)
  out/
  tests/
```

**Verify:** `cd slidechat && uv run python -c "import byaldi, pdf2image, litellm; print('A ok')"` and `cd ../voice_agent && uv run python -c "import faster_whisper, sounddevice, numpy, silero_vad; print('B ok')"`.

**Troubleshoot:** `sounddevice` needs PortAudio — bundled in the wheel on Win/mac; on Linux run `sudo apt-get install libportaudio2`. If `silero-vad` import differs by version, see Step B1's fallback (torch.hub load).

### 0.3 — Credentials (optional) and `.env`

Only needed if you use a hosted answering VLM or TTS API. The free/local path needs none.

```bash
printf "GEMINI_API_KEY=\nCARTESIA_API_KEY=\nELEVENLABS_API_KEY=\n" > slidechat/.env
printf ".env\nout/\n__pycache__/\n.venv/\n*.wav\n" > slidechat/.gitignore
cp slidechat/.gitignore voice_agent/.gitignore
```

`litellm` reads `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from the environment; load with `from dotenv import load_dotenv; load_dotenv()`.

**Free/local answering VLM (recommended default):** install Ollama (Windows installer from https://ollama.com; `brew install ollama` on macOS), then:

```bash
ollama pull llama3.2-vision      # or: ollama pull qwen2.5vl
curl http://localhost:11434/v1/models
```

Pass `model="ollama/llama3.2-vision"` anywhere this guide uses `gemini/gemini-2.0-flash`.

### 0.4 — GPU note for ColPali indexing (read before Build A)

ColPali/ColQwen **indexing** (embedding every page) wants a GPU. Options, cheapest first:
- **CPU, small deck:** index a **≤30-page** deck on CPU. Slow (single-digit minutes) but works and needs nothing extra. Query-time retrieval is cheap either way.
- **Free Colab/Modal:** run *only the indexing cell* on a free GPU, then download the `.byaldi/` index folder and query locally.
- **Local GPU:** if you have CUDA, install a matching Torch build (`uv pip install torch --index-url https://download.pytorch.org/whl/cu121`) and indexing is seconds.

Keep your first deck tiny so you can iterate; scale up once the pipeline works.

---

## Step-by-step

Each step names which Definition-of-Done (DoD) item it satisfies.

## Build A — SlideChat (ColPali multimodal RAG)

### Step A1 — Convert decks to page images

**What:** rasterize each PDF deck to per-page images at ~150 DPI into a folder Byaldi can index. → *DoD: indexes ≥2 decks.*

**Why:** ColPali treats **a page as an image** — you never parse text. 150 DPI keeps slide text legible while staying small enough that CPU indexing is bearable; higher DPI mostly buys tokens, not accuracy ([../lectures/07-colpali-late-interaction.md](../lectures/07-colpali-late-interaction.md)).

**Do it — `src/index.py` (conversion half):**

```python
# src/index.py
from __future__ import annotations
from pathlib import Path
from pdf2image import convert_from_path

def deck_to_images(pdf_path: str | Path, out_root: str | Path = "decks_img", dpi: int = 150) -> list[str]:
    """Rasterize a PDF deck to page PNGs: decks_img/<deckname>/page_0001.png ..."""
    pdf_path = Path(pdf_path)
    out_dir = Path(out_root) / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=dpi)   # add poppler_path=... on Windows if needed
    paths = []
    for i, page in enumerate(pages, start=1):
        p = out_dir / f"page_{i:04d}.png"
        page.save(p, "PNG")
        paths.append(str(p))
    return paths
```

**Expected result:** `decks_img/<deck>/page_0001.png …` one file per slide.
**Verify:** `cd slidechat && uv run python -c "from src.index import deck_to_images; print(len(deck_to_images('decks/YOUR_DECK.pdf')), 'pages')"`.
**Troubleshoot:** `PDFInfoNotInstalledError` → poppler not on PATH; pass `poppler_path=`. Blank/black pages → some PDFs need `use_pdftocairo=True`.

### Step A2 — Index with Byaldi (ColQwen), the OCR-free step

**What:** build a late-interaction index over the page images with `RAGMultiModalModel`. → *DoD: indexes ≥2 decks.*

**Why:** Byaldi wraps ColQwen2 and stores **multi-vector** page embeddings (one vector per patch, ColBERT-style). `store_collection_with_index=True` keeps the base64 page image *inside* the index so retrieval hands you the image back directly — no separate page store to manage. This is the whole "skip OCR" payoff ([../lectures/07-colpali-late-interaction.md](../lectures/07-colpali-late-interaction.md)).

**Do it — add to `src/index.py`:**

```python
from byaldi import RAGMultiModalModel

def build_index(images_root: str = "decks_img",
                index_name: str = "decks",
                model_name: str = "vidore/colqwen2-v1.0") -> RAGMultiModalModel:
    """Index every page image under images_root. First call downloads the model (~a few GB)."""
    model = RAGMultiModalModel.from_pretrained(model_name)   # add device_map / index_root as needed
    model.index(
        input_path=images_root,             # a directory: Byaldi walks it recursively
        index_name=index_name,
        store_collection_with_index=True,   # keep page images in the index for retrieval
        overwrite=True,
    )
    return model
```

Driver to index all your decks once:

```bash
uv run python -c "
from src.index import deck_to_images, build_index
from pathlib import Path
for pdf in Path('decks').glob('*.pdf'):
    deck_to_images(pdf)
build_index()
print('indexed ->', list(Path('.byaldi').glob('*')))
"
```

**Expected result:** a persisted index under `.byaldi/decks/`. First run downloads ColQwen weights (slow, once).
**Verify:** `ls .byaldi/decks` shows index files; the driver prints them.
**Troubleshoot:**
- OOM / killed on CPU → your deck is too big; cut to ≤30 pages or index on Colab (0.4).
- `store_collection_with_index=True` balloons memory on huge decks → set it `False` and instead map page-id → file path yourself in Step A3.
- Model-name errors → `vidore/colqwen2-v1.0` is the current default; `vidore/colpali-v1.2` is the older alternative. Do not guess other names — check the `vidore` org on Hugging Face.
- Windows path oddities → keep `images_root` a relative forward-slash path.

### Step A3 — Retrieve top-3 pages (tight k)

**What:** load the persisted index and return the **top-3** pages for a query, each with its source deck, page number, score, and image. → *DoD: retrieves top-3 (not more).*

**Why:** every retrieved page becomes **fat image tokens** for the answering VLM. Top-3 keeps cost and latency sane; top-20 is a bill and a slowdown ([../lectures/08-multimodal-rag-architecture.md](../lectures/08-multimodal-rag-architecture.md)).

**Do it — `src/retrieve.py`:**

```python
# src/retrieve.py
from __future__ import annotations
import base64, io
from dataclasses import dataclass
from PIL import Image
from byaldi import RAGMultiModalModel

@dataclass
class Hit:
    doc_id: int
    page_num: int          # 1-based page within its deck
    score: float
    image: Image.Image     # the page image
    source: str            # deck/page label for citation

def load_index(index_name: str = "decks") -> RAGMultiModalModel:
    return RAGMultiModalModel.from_index(index_name)

def _to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def search(model: RAGMultiModalModel, query: str, k: int = 3) -> list[Hit]:
    raw = model.search(query, k=k)   # each result: doc_id, page_num, score, (base64) image
    hits: list[Hit] = []
    for r in raw:
        img = _to_image(r["base64"]) if r.get("base64") else None
        hits.append(Hit(
            doc_id=r["doc_id"], page_num=r["page_num"], score=float(r["score"]),
            image=img, source=f"doc{r['doc_id']}:p{r['page_num']}",
        ))
    return hits
```

**Expected result:** three `Hit`s ranked by MaxSim score, each carrying the page image.
**Verify:**

```bash
uv run python -c "
from src.retrieve import load_index, search
m = load_index()
for h in search(m, 'what was total revenue?', k=3):
    print(h.source, round(h.score,2), h.image.size if h.image else None)
"
```

**Troubleshoot:** result dict keys vary slightly by Byaldi version — if `r['base64']` is missing, you indexed with `store_collection_with_index=False`; either re-index with it `True`, or map `(doc_id, page_num)` back to your `decks_img/.../page_XXXX.png` file and open it from disk.

### Step A4 — Answer with a VLM, grounded on the retrieved pages, with page citations + bbox

**What:** send the top-3 **page images** + the question to a VLM; require a **page-number citation** per claim and an approximate bbox on the cited page. → *DoD: 10/10 visual questions answered with a correct page-number citation; per-query VLM image-token cost recorded.*

**Why:** this is Phase 4 RAG discipline — the retrieval substrate changed to page images, but *grounding + citation* is identical. Passing the image (not OCR text) lets the VLM read charts/tables/figures directly ([../lectures/08-multimodal-rag-architecture.md](../lectures/08-multimodal-rag-architecture.md)). Structured output makes the citation machine-checkable; the bbox reuses Week 1's "approximate box for a highlight" idea.

**Do it — `src/answer.py`:**

```python
# src/answer.py
from __future__ import annotations
import base64, io
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import instructor
from litellm import completion, token_counter
from .retrieve import Hit

load_dotenv()
client = instructor.from_litellm(completion)

class Citation(BaseModel):
    source: str = Field(..., description="the page label you used, exactly as given, e.g. 'doc0:p4'")
    bbox: list[float] | None = Field(None, description="approx [x,y,w,h] normalized 0-1 of the evidence")

class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(..., description="every page you actually used")

def _img_part(img) -> dict:
    buf = io.BytesIO(); img.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

def _messages(question: str, hits: list[Hit]) -> list[dict]:
    content = [{"type": "text", "text":
        "Answer ONLY from the slide images below. Each image is labeled with a page id.\n"
        f"Question: {question}\n"
        "Cite the page id(s) you used and give an approximate normalized bbox of the evidence. "
        "If the pages do not contain the answer, say so and cite nothing."}]
    for h in hits:
        content.append({"type": "text", "text": f"[page id: {h.source}]"})
        content.append(_img_part(h.image))
    return [{"role": "user", "content": content}]

def answer(question: str, hits: list[Hit], model: str = "gemini/gemini-2.0-flash") -> tuple[Answer, int]:
    msgs = _messages(question, hits)
    tokens = token_counter(model=model, messages=msgs)   # per-query image-token cost (record this)
    ans = client.chat.completions.create(model=model, response_model=Answer, max_retries=2, messages=msgs)
    return ans, tokens
```

> **Free/local alt:** `answer(q, hits, model="ollama/llama3.2-vision")`. Multi-image support varies by local model/version — if it only reliably takes one image, pass just `hits[:1]` for the local path and note the limitation.

**Expected result:** an `Answer` whose `citations[].source` is one of the labels you passed (e.g. `doc0:p4`), plus the token count for the query.
**Verify:**

```bash
uv run python -c "
from src.retrieve import load_index, search
from src.answer import answer
m = load_index()
hits = search(m, 'what was total revenue?', k=3)
a, toks = answer('what was total revenue?', hits)
print('TOKENS', toks); print(a.model_dump())
"
```

Check the cited `source` matches the page that actually shows the number, and note `TOKENS` (this is the DoD's per-query image-token cost).
**Troubleshoot:** citation label the model invented (not in your set) → tighten the prompt ("cite EXACTLY one of these ids: ...") and validate in code, dropping unknown labels. Model refuses/answers from nothing → your retrieval missed; inspect `search()` output first.

### Step A5 — Return thumbnails + render the citation highlight

**What:** save the top-3 page thumbnails and draw the cited bbox on the cited page. → *DoD: returns an answer with the top-3 thumbnails (+ correct page citation).*

**Why:** thumbnails + a highlighted region are the product surface a user trusts — it shows *where* the answer came from (same rationale as Week 1's bbox render; boxes are approximate, good for highlighting).

**Do it — `src/viz.py`:**

```python
# src/viz.py
from pathlib import Path
from PIL import ImageDraw

def save_thumbs(hits, out_dir="out", size=(320, 240)) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for h in hits:
        t = h.image.copy(); t.thumbnail(size)
        p = f"{out_dir}/thumb_{h.source.replace(':','_')}.png"
        t.save(p); paths.append(p)
    return paths

def draw_citation(hit, bbox, out_path="out/citation.png") -> str:
    img = hit.image.copy(); W, H = img.size
    if bbox:
        x, y, w, h = bbox
        ImageDraw.Draw(img).rectangle([x*W, y*H, (x+w)*W, (y+h)*H], outline=(255, 0, 0), width=5)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
```

**Verify:** run A4's driver, then `save_thumbs(hits)` and `draw_citation(hit_for_cited_source, ans.citations[0].bbox)`; open `out/citation.png` — the box should sit on/near the evidence.
**Troubleshoot:** box wildly off → VLM bboxes drift (expected); the highlight is a hint, not a crop. Match the cited `source` back to the right `Hit` before drawing.

### Step A6 — recall@3 eval + the OCR-text-RAG A/B baseline

**What:** a tiny labeled query→page gold set to measure **recall@3**, and a text-only OCR→RAG baseline to A/B on **visual** questions. → *DoD (milestone): recall@3 measured; A/B vs text RAG reported.*

**Why:** leaderboards don't ship your deck — measure retrieval on *your* pages (Phase 7 discipline). The A/B is the point of ColPali: on charts/tables/figures it should beat an OCR→text pipeline that can't "see" the layout ([../lectures/07-colpali-late-interaction.md](../lectures/07-colpali-late-interaction.md)).

**Do it — `eval/queries.jsonl`** (hand-label ~10 visual questions; `source` = the correct page label from Step A3):

```json
{"q": "what was total revenue in Q3?", "gold": "doc0:p4"}
{"q": "which region had the highest churn?", "gold": "doc0:p9"}
```

**Do it — `eval/eval.py`:**

```python
# eval/eval.py
import json
from pathlib import Path
from src.retrieve import load_index, search

def recall_at_k(k: int = 3) -> float:
    m = load_index()
    rows = [json.loads(l) for l in Path("eval/queries.jsonl").read_text().splitlines() if l.strip()]
    hits = 0
    for r in rows:
        got = {h.source for h in search(m, r["q"], k=k)}
        hits += int(r["gold"] in got)
        print(("HIT " if r["gold"] in got else "MISS"), r["q"], "->", got)
    print(f"recall@{k} = {hits}/{len(rows)} = {hits/len(rows):.2f}")
    return hits / len(rows)

if __name__ == "__main__":
    recall_at_k(3)
```

**Text-RAG baseline (A/B):** extract each page's text with `pdftotext` (poppler) or PyMuPDF, chunk per page, embed with any text embedder you already have, retrieve top-3, and answer. Run **the same visual questions** through both and record which retrieves the right page. Write a one-paragraph verdict (expected: ColPali wins on chart/figure questions where the number lives in a graphic, not the text layer).
**Verify:** `uv run python -m eval.eval` prints per-query HIT/MISS and the recall@3 fraction.
**Troubleshoot:** recall low → labels wrong (re-check the gold page id), deck too small, or queries too vague; make questions concrete ("the revenue number on the results slide").

---

## Build B — Cascaded realtime voice agent

Start **non-streaming** (record → transcribe → think → speak) so each stage is debuggable, then add streaming TTS + barge-in.

### Step B1 — Capture + VAD endpointing

**What:** record the mic and use **Silero VAD** to detect speech start/stop; endpoint after ~500–700ms of trailing silence. → *DoD: word-timestamp transcription later; VAD suppresses a silence hallucination.*

**Why:** VAD gates silence so Whisper doesn't get fed dead air (Whisper **hallucinates** "thank you for watching" on silence — [../lectures/09-stt-asr-whisper-vad-diarization.md](../lectures/09-stt-asr-whisper-vad-diarization.md)). Endpointing at 500–700ms is the conversational sweet spot: too short cuts users off, too long feels laggy ([../lectures/11-realtime-voice-agent-architecture.md](../lectures/11-realtime-voice-agent-architecture.md)).

**Do it — `src/capture_vad.py`:**

```python
# src/capture_vad.py
from __future__ import annotations
import queue, numpy as np, sounddevice as sd
from silero_vad import load_silero_vad, VADIterator

SR = 16000                 # Whisper wants 16 kHz mono
FRAME = 512                # samples per VAD frame at 16 kHz (~32 ms)

def record_utterance(max_s: float = 15.0, silence_ms: int = 600) -> np.ndarray:
    """Block until the user speaks then goes quiet for silence_ms. Returns float32 mono audio."""
    model = load_silero_vad()
    vad = VADIterator(model, sampling_rate=SR, min_silence_duration_ms=silence_ms)
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def cb(indata, frames, time_, status):
        q.put(indata[:, 0].copy())

    collected, started, trailing_silence = [], False, 0
    with sd.InputStream(samplerate=SR, channels=1, blocksize=FRAME, dtype="float32", callback=cb):
        import time; t0 = time.time()
        while time.time() - t0 < max_s:
            chunk = q.get()
            collected.append(chunk)
            evt = vad(chunk, return_seconds=True)     # {'start':..} or {'end':..} or None
            if evt and "start" in evt:
                started = True
            if started and evt and "end" in evt:
                break
    return np.concatenate(collected) if collected else np.zeros(0, np.float32)
```

> **Version fallback:** if `silero_vad`'s helper API differs, load via torch.hub: `model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')` and use `utils`' `VADIterator`. Keep 16 kHz mono either way.

**Expected result:** speak a sentence, pause — the function returns a float32 array of just your utterance.
**Verify:**

```bash
cd voice_agent && uv run python -c "
from src.capture_vad import record_utterance
import numpy as np
a = record_utterance(); print('captured samples:', a.shape[0], 'secs:', round(a.shape[0]/16000,2))
"
```

**Troubleshoot:** nothing captured → wrong input device: list with `python -c "import sounddevice as sd; print(sd.query_devices())"` and set `sd.default.device = <idx>`. Cuts you off mid-word → raise `silence_ms`. Windows mic permission → allow desktop apps to use the microphone in Privacy settings.

### Step B2 — STT with faster-whisper (VAD filter + word timestamps)

**What:** transcribe the captured audio on CPU with `vad_filter=True` and `word_timestamps=True`; and demonstrate VAD suppressing a hallucination on a silent clip. → *DoD: word-timestamp transcription; VAD filtering demonstrably suppresses a silence hallucination (before/after).*

**Why:** `faster-whisper` (CTranslate2) runs Whisper fast on CPU with `int8`; `vad_filter=True` drops non-speech so the model isn't tempted to invent text; `word_timestamps=True` gives per-word timing you'll need for barge-in analysis and diarization alignment ([../lectures/09-stt-asr-whisper-vad-diarization.md](../lectures/09-stt-asr-whisper-vad-diarization.md)).

**Do it — `src/stt.py`:**

```python
# src/stt.py
from __future__ import annotations
import numpy as np
from faster_whisper import WhisperModel

_model = None
def _get(model_size="base.en"):
    global _model
    if _model is None:
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")  # free/local CPU path
    return _model

def transcribe(audio: np.ndarray, vad_filter: bool = True) -> tuple[str, list]:
    segments, _ = _get().transcribe(
        audio, language="en", vad_filter=vad_filter, word_timestamps=True,
    )
    text, words = [], []
    for seg in segments:
        text.append(seg.text)
        for w in (seg.words or []):
            words.append((w.word, round(w.start, 2), round(w.end, 2)))
    return " ".join(t.strip() for t in text).strip(), words
```

**Prove the VAD win (before/after)** — make a silent clip and transcribe it both ways:

```bash
# 3 seconds of silence at 16 kHz
uv run python -c "
import numpy as np, scipy.io.wavfile as wav
wav.write('clips/silence.wav', 16000, np.zeros(16000*3, dtype=np.float32))
"
uv run python -c "
import scipy.io.wavfile as wav, numpy as np
from src.stt import transcribe
sr, a = wav.read('clips/silence.wav'); a = a.astype(np.float32)
print('NO VAD  ->', repr(transcribe(a, vad_filter=False)[0]))
print('WITH VAD->', repr(transcribe(a, vad_filter=True)[0]))
"
```

**Expected result:** `NO VAD` often prints a phantom phrase (e.g. "Thank you."); `WITH VAD` prints `''`. Screenshot both lines for the DoD.
**Verify:** run the two-liner above; also transcribe a real utterance and confirm `words` carries `(word, start, end)` triples.
**Troubleshoot:** first run downloads the model (once). If `NO VAD` doesn't hallucinate on your machine, try a clip with faint background hum or music — silence-adjacent noise is the reliable trigger. Slow on CPU → use `tiny.en` while iterating.

### Step B3 — Diarization on a 2-speaker clip

**What:** label ≥2 speakers on a 2-speaker clip. → *DoD: diarization labels ≥2 speakers.*

**Why:** "who spoke when" is a distinct model from ASR ([../lectures/09-stt-asr-whisper-vad-diarization.md](../lectures/09-stt-asr-whisper-vad-diarization.md)). WhisperX bundles diarization + word alignment; pyannote is the underlying engine.

**Do it (WhisperX path — one install, aligns words to speakers):**

```bash
uv add whisperx        # pulls pyannote; heavier install
```

```python
# src/diarize.py  (WhisperX)
import whisperx
def diarize(wav_path: str, hf_token: str, device="cpu") -> list[dict]:
    audio = whisperx.load_audio(wav_path)
    asr = whisperx.load_model("base.en", device, compute_type="int8")
    result = asr.transcribe(audio)
    dia = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diar = dia(audio)
    result = whisperx.assign_word_speakers(diar, result)
    return [{"speaker": s.get("speaker"), "text": s["text"]} for s in result["segments"]]
```

> pyannote's diarization model is gated — accept its terms on Hugging Face and pass a `hf_token` (free HF account). **No-token fallback:** you can still satisfy the DoD with a light **speaker-change** heuristic — embed 1-second windows with a speaker-embedding model (e.g. `resemblyzer`) and cluster into 2 groups with KMeans; label segments by cluster. Note the method you used.

**Expected result:** segments tagged `SPEAKER_00` / `SPEAKER_01` on a two-person clip (record yourself + a second voice, or use any 2-speaker sample you have rights to).
**Verify:** print the segments; confirm at least two distinct speaker labels appear.
**Troubleshoot:** gated-model 401 → accept terms + set token. WhisperX install heavy/conflicting → do it in a throwaway venv just for this DoD, or use the resemblyzer fallback.

### Step B4 — TTS with streaming (Piper local, or an API)

**What:** synthesize the agent's reply and play it, streaming so audio starts before the whole reply is ready. → *DoD (budget): low end-of-speech→first-audio latency.*

**Why:** time-to-first-byte dominates perceived latency — start speaking the first sentence while synthesizing the rest ([../lectures/10-tts-streaming-synthesis.md](../lectures/10-tts-streaming-synthesis.md)). Piper is free, local, offline, and fast on CPU.

**Do it (free/local — Piper):** download a Piper voice (`.onnx` + `.onnx.json`) from the Piper voices release page (find via the `rhasspy/piper` GitHub), then:

```bash
uv add piper-tts
```

```python
# src/tts.py  (Piper local)
from __future__ import annotations
import numpy as np, sounddevice as sd, threading
from piper.voice import PiperVoice

_voice = None
def _get(model_path="en_US-amy-medium.onnx"):
    global _voice
    if _voice is None:
        _voice = PiperVoice.load(model_path)
    return _voice

def speak(text: str, stop_event: threading.Event | None = None):
    """Stream synthesized audio to the speakers; abort promptly if stop_event is set (barge-in)."""
    v = _get(); sr = v.config.sample_rate
    stream = sd.OutputStream(samplerate=sr, channels=1, dtype="int16")
    stream.start()
    try:
        for chunk in v.synthesize_stream_raw(text):     # yields PCM bytes per sentence/segment
            if stop_event and stop_event.is_set():
                break
            stream.write(np.frombuffer(chunk, dtype=np.int16))
    finally:
        stream.stop(); stream.close()
```

> **API alt (lower latency, costs):** Cartesia or ElevenLabs streaming endpoints return audio chunks as they synthesize — play each chunk on arrival, honoring the same `stop_event`. Do not hardcode a specific URL; use the provider's official SDK. Keep replies short — long TTS blows the budget.

**Expected result:** you hear the reply; audio begins within a few hundred ms of calling `speak`.
**Verify:** `uv run python -c "import threading; from src.tts import speak; speak('Hello, this is your slide assistant.', threading.Event())"`.
**Troubleshoot:** `synthesize_stream_raw` name differs by piper-tts version → check `dir(PiperVoice)`; some versions expose `synthesize(text, wav_file)` (non-streaming) — use that first, add streaming after the loop works. No audio device → set `sd.default.device`.

### Step B5 — The LLM turn (reuse the Phase 6 agent loop)

**What:** send the transcript to your LLM/agent and get a **short** reply. → keeps the turn under budget.

**Why:** this is the Phase 6 tool-using loop with a real-time I/O layer bolted on — same bounded loop, same tool calls, just latency-critical ([../lectures/11-realtime-voice-agent-architecture.md](../lectures/11-realtime-voice-agent-architecture.md)). Long answers are the #1 budget killer, so cap length.

**Do it — `src/agent.py`:**

```python
# src/agent.py
from __future__ import annotations
from dotenv import load_dotenv
from litellm import completion
load_dotenv()

SYSTEM = ("You are a spoken voice assistant. Reply in ONE or TWO short sentences, "
          "plain conversational language, no markdown, no lists.")

def reply(user_text: str, history: list[dict], model="gemini/gemini-2.0-flash") -> str:
    msgs = [{"role": "system", "content": SYSTEM}, *history, {"role": "user", "content": user_text}]
    out = completion(model=model, messages=msgs, max_tokens=120)
    return out.choices[0].message.content.strip()
```

> **Free/local alt:** `model="ollama/llama3.1"` (text) or your Phase 6 agent entrypoint. To make it *agentic*, swap `completion` for your tool-calling loop — but keep replies short for voice.

**Verify:** `uv run python -c "from src.agent import reply; print(reply('what can you do?', []))"` → one/two short sentences.
**Troubleshoot:** rambling replies → lower `max_tokens`, strengthen the system prompt. Local model slow → it eats your budget; report it and prefer a fast hosted model for the latency DoD, noting the tradeoff.

### Step B6 — Barge-in: stop TTS the instant the user speaks

**What:** while TTS plays, keep VAD listening; if the user starts talking, **stop playback immediately** and start a new capture. → *DoD: interrupt mid-sentence, agent stops within ~200ms and listens.*

**Why:** barge-in is the single most important voice-UX feature — an agent that talks over you feels dead ([../lectures/11-realtime-voice-agent-architecture.md](../lectures/11-realtime-voice-agent-architecture.md)). Mechanism: run TTS in a thread with a `stop_event`; a concurrent VAD watcher sets it the instant it hears speech.

**Do it — `src/bargein.py`:**

```python
# src/bargein.py
from __future__ import annotations
import threading, queue, numpy as np, sounddevice as sd
from silero_vad import load_silero_vad, VADIterator
from .tts import speak

SR, FRAME = 16000, 512

def speak_with_bargein(text: str) -> bool:
    """Speak `text`; if the user starts talking, cut audio fast. Returns True if interrupted."""
    stop = threading.Event(); interrupted = threading.Event()
    t = threading.Thread(target=speak, args=(text, stop), daemon=True); t.start()

    model = load_silero_vad(); vad = VADIterator(model, sampling_rate=SR)
    q: "queue.Queue[np.ndarray]" = queue.Queue()
    with sd.InputStream(samplerate=SR, channels=1, blocksize=FRAME, dtype="float32",
                        callback=lambda i, f, tm, s: q.put(i[:, 0].copy())):
        while t.is_alive():
            evt = vad(q.get(), return_seconds=True)
            if evt and "start" in evt:
                stop.set(); interrupted.set()      # kill TTS immediately
                break
    t.join(timeout=0.3)
    return interrupted.is_set()
```

**Expected result:** start the agent talking, speak over it — audio cuts within ~200ms and the loop returns to listening.
**Verify:** call `speak_with_bargein("This is a long sentence that keeps going and going so you have time to interrupt it before it finishes.")` and talk over it; it should return `True` fast.
**Troubleshoot:** it hears *itself* (echo) and self-interrupts → use headphones (kills the acoustic loop), and/or ignore VAD for the first ~300ms and require 2 consecutive speech frames before triggering. Never fires → mic device/threshold; test capture alone (B1) first.

### Step B7 — Measure the latency budget (p50/p95)

**What:** log timestamps at each stage boundary and report **end-of-speech → first-audio-out** latency over ≥10 turns. → *DoD: p50/p95 reported against the 800ms budget.*

**Why:** "feels conversational" is a number, not a vibe — the target is **<800ms** end-of-speech to first audio ([../lectures/11-realtime-voice-agent-architecture.md](../lectures/11-realtime-voice-agent-architecture.md)).

**Do it — instrument the loop and record the gap:**

```python
# src/loop.py (the end-to-end turn, timed)
import time, numpy as np
from statistics import median
from .capture_vad import record_utterance
from .stt import transcribe
from .agent import reply
from .bargein import speak_with_bargein

def one_turn(history, model="gemini/gemini-2.0-flash") -> float:
    audio = record_utterance()             # returns AFTER end-of-speech (VAD endpoint)
    t_eos = time.perf_counter()            # <-- end of speech
    text, _ = transcribe(audio)
    resp = reply(text, history, model=model)
    t_first_audio = time.perf_counter()    # first audio-out is imminent in speak()
    latency_ms = (t_first_audio - t_eos) * 1000
    speak_with_bargein(resp)
    history += [{"role": "user", "content": text}, {"role": "assistant", "content": resp}]
    return latency_ms

def run(n=10):
    hist, lat = [], []
    for i in range(n):
        print(f"-- turn {i+1}/{n}, speak now --")
        lat.append(one_turn(hist)); print(f"   latency: {lat[-1]:.0f} ms")
    lat.sort()
    print(f"p50={median(lat):.0f}ms  p95={lat[int(0.95*len(lat))-1]:.0f}ms  target<800ms")
```

> Measure `t_first_audio` as tightly as you can — ideally set it at the moment `stream.write` receives its first chunk inside `speak()` (pass a callback or a shared timestamp). The version above brackets STT+LLM, which is the dominant, controllable cost.

**Expected result:** a p50/p95 line. Hosted fast models + short replies typically land near/under 800ms; local CPU STT+LLM will exceed it — **report the number honestly** and name the bottleneck (usually the LLM, then STT).
**Verify:** `uv run python -c "from src.loop import run; run(10)"` and record p50/p95.
**Troubleshoot:** p95 huge → find the stage: log per-stage deltas. STT dominates → `tiny.en`/`base.en`, keep utterances short. LLM dominates → smaller/faster model, shorter `max_tokens`, or stream tokens into sentence-wise TTS.

---

## Putting it together — a short end-to-end run

**SlideChat (one question, full path):**

```bash
cd slidechat
uv run python -c "
from src.retrieve import load_index, search
from src.answer import answer
from src.viz import save_thumbs, draw_citation
q = 'what was total revenue?'
m = load_index(); hits = search(m, q, k=3)
a, toks = answer(q, hits)
print('ANSWER:', a.answer)
print('CITES :', [c.source for c in a.citations], '| tokens:', toks)
print('THUMBS:', save_thumbs(hits))
cited = next((h for h in hits if h.source == a.citations[0].source), hits[0])
print('HILITE:', draw_citation(cited, a.citations[0].bbox))
"
```
You get an answer, page-number citation(s), three thumbnails in `out/`, and a highlight image — plus the per-query token count.

**Voice agent (a live conversation):**

```bash
cd voice_agent
uv run python -c "from src.loop import run; run(10)"
```
Speak, get spoken replies, interrupt one mid-sentence to see barge-in, and read the p50/p95 latency line at the end.

---

## Definition of Done — verifiable checks

Restated from the spine ([../12-multimodal.md](../12-multimodal.md), Week 2). Each maps to steps above.

- [ ] **SlideChat indexes ≥2 decks, and for 10/10 visual questions returns an answer with a correct page-number citation and the top-3 thumbnails.** → Steps A1–A5 (`answer.citations`, `save_thumbs`).
- [ ] **SlideChat retrieves top-3 (not more) and you recorded the VLM image-token cost per query.** → Step A3 (`k=3`) + Step A4 (`token_counter`).
- [ ] **Voice agent transcribes your speech with word timestamps, and VAD filtering demonstrably suppresses a Whisper hallucination on a silent clip (before/after).** → Step B2 (silence-clip before/after).
- [ ] **Diarization labels ≥2 speakers on a 2-speaker clip (WhisperX or pyannote).** → Step B3.
- [ ] **Barge-in works: you can interrupt the agent mid-sentence and it stops speaking within ~200ms and listens.** → Step B6.
- [ ] **You logged end-of-speech→first-audio latency across ≥10 turns and report p50/p95 against the 800ms budget.** → Step B7.

Bonus (milestone-facing): **recall@3** measured on a labeled query→page set and an **A/B vs OCR→text RAG** on visual questions → Step A6.

You are done with Week 2 when all six boxes are checked, SlideChat answers your 10 visual questions with correct citations, and the voice agent runs a live interruptible conversation with a reported latency number.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `PDFInfoNotInstalledError` / blank pages | poppler not on PATH | Install poppler; pass `poppler_path=`; try `use_pdftocairo=True` |
| ColPali indexing OOM / killed | Deck too big for CPU/RAM | ≤30-page deck, or index on free Colab/Modal and download `.byaldi/` |
| `search()` result has no image | Indexed with `store_collection_with_index=False` | Re-index `True`, or map `(doc_id,page_num)`→`page_XXXX.png` on disk |
| VLM cites a page id you never sent | Model invented a label | Prompt "cite EXACTLY one of these ids"; validate + drop unknowns in code |
| Whisper prints phantom "Thank you" | No VAD gating on silence | `vad_filter=True`; you proved this in B2 |
| `sounddevice` no input / silence | Wrong device / OS mic permission | `sd.query_devices()`; set `sd.default.device`; allow mic in OS privacy |
| Agent interrupts itself | TTS echo into mic (no headphones) | Use headphones; ignore first ~300ms; require 2 speech frames |
| Endpointing cuts you off | `silence_ms` too low | Raise to 600–700ms; tune on real speech, not a script |
| pyannote/WhisperX 401 gated | Didn't accept HF model terms | Accept terms + pass `hf_token`; or resemblyzer clustering fallback |
| Latency p95 huge | Local CPU STT/LLM | `tiny.en`/`base.en`, short replies, faster/hosted LLM; report bottleneck |
| `silero_vad` import/API differs | Version drift | torch.hub `snakers4/silero-vad` fallback; keep 16 kHz mono |
| `piper` `synthesize_stream_raw` missing | piper-tts version | `dir(PiperVoice)`; use non-streaming `synthesize` first, add streaming after |

---

## Stretch goals (optional)

- **Run Build B on Pipecat or the OpenAI Realtime API** and compare latency + code complexity vs your hand-rolled cascade. Note exactly where **speech-to-speech** wins (prosody, sub-500ms turns) and where the cascade wins (debuggability, component swap). (Spine's "Level-up".)
- **WebRTC transport.** Re-front the voice agent with **LiveKit Agents** (WebRTC) instead of local `sounddevice`, and observe jitter/packet-loss handling you don't get from raw WebSocket — the reason production voice isn't WebSocket.
- **Streaming STT (partials).** Feed audio to faster-whisper in a rolling window and emit partial transcripts so the LLM can start "thinking" before end-of-speech — shave the budget further.
- **SlideChat A/B, fully scored.** Extend Step A6: answer the *same* 10 questions through ColPali and the text-RAG baseline, LLM-judge both for correctness, and report answer accuracy (not just retrieval recall) — the milestone's A/B verdict with numbers.
- **Tool-using voice agent.** Wire Step B5 to your Phase 6 tool loop (e.g. "look that up in the deck" → calls SlideChat's `search`+`answer`) so the two builds fuse into one voice-driven document assistant.
- **Multi-image local answering.** If your Ollama VLM only takes one image, add a "reduce" step: answer per-page then combine, so the local path still uses all top-3 pages.
