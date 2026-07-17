# Week 1 Lab: Reproducible Env + Classical ML Basics

> **What you will build:** the `ai-foundations/` repo skeleton that *every* later phase reuses — a `uv`-managed, lockfile-pinned Python project with `.env` secrets, a streaming JSONL I/O module, a NumPy/pandas notebook (broadcasting, axes, cosine similarity), a classical-ML baseline (LogisticRegression + learning curve), and hand-rolled metrics (`precision_recall_f1`, `ndcg_at_k`) proven correct against scikit-learn. You finish by running the notebook **headless** to prove it has no hidden state.
>
> **Why it matters:** this is the toolchain the whole roadmap assumes. Reproducibility, JSONL, vectorized thinking, train/val/test discipline, and metrics-that-match-cost are the load-bearing habits behind every fine-tune, RAG pipeline, and eval you will build later. A number you can't reproduce is a number you can't trust.
>
> **Matching lectures — read these first:**
> - [`../lectures/01-reproducible-envs-uv.md`](../lectures/01-reproducible-envs-uv.md) — uv, lockfiles, secrets
> - [`../lectures/03-numpy-tensors-vectorized-thinking.md`](../lectures/03-numpy-tensors-vectorized-thinking.md) — shape/dtype/broadcasting/axes, cosine sim
> - [`../lectures/04-just-enough-classical-ml.md`](../lectures/04-just-enough-classical-ml.md) — train/val/test, baselines, overfitting & learning curves
> - [`../lectures/05-data-leakage.md`](../lectures/05-data-leakage.md) — the six leakage mechanisms
> - [`../lectures/06-metrics-that-match-cost.md`](../lectures/06-metrics-that-match-cost.md) — precision/recall/F1, nDCG@k
>
> (JSONL — one JSON object per line — is the dominant LLM dataset format; you build the reusable reader/writer in Step 4.)

**Est. time:** ~8 hrs · **You will need:** Python 3.11+, a terminal (Git-Bash on Windows), `git`, and `uv` (installed in Step 0). No GPU, no API keys, no paid services — this entire week runs offline on CPU.

---

## Before you start (setup)

- **Read the five lectures above.** This guide assumes you know *why* a lockfile beats `requirements.txt`, what broadcasting is, why val and test are separate, and what nDCG@k rewards. We won't re-derive the theory here — we'll build it.
- **Pick your repo home.** Everything lives in one repo, `ai-foundations/`, reused through Week 3. Create it somewhere you control, e.g. `~/code/ai-foundations` (macOS/Linux) or `~/Desktop/ai-foundations` (Windows). Avoid deeply nested OneDrive-synced paths — file locks can confuse `uv` and Jupyter.
- **Confirm Python is present:** `python --version` (Windows) or `python3 --version` (macOS/Linux) should print 3.11 or newer. `uv` can also fetch a Python for you, so this is a nice-to-have, not a hard requirement.
- **Have `git` configured:** `git config --global user.name` and `...user.email` should both return values. You will make several commits.

A note on shells: this user is on **Windows + Git-Bash**, so the primary commands below use POSIX/bash syntax (forward slashes, `export`). Where Windows genuinely differs from macOS/Linux, a **Windows** and a **macOS/Linux** line are both shown.

---

## Step-by-step

### Step 1 — Install `uv` and scaffold the project

**What:** Install the `uv` toolchain and create the `ai-foundations` project.

**Why:** `uv` is the 2025 default — one fast Rust binary that replaces `pip`/`venv`/`pipenv`/`pyenv`. It produces a committed **lockfile** (`uv.lock`) that pins the *exact* resolved version and hash of every transitive dependency. That is the difference between "works on my machine" and "works." (See [`../lectures/01-reproducible-envs-uv.md`](../lectures/01-reproducible-envs-uv.md).)

**Do it:**

```bash
# Install uv
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell — run once, then return to Git-Bash):
#   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart the shell so `uv` is on PATH, then confirm:
uv --version

# Scaffold the project
uv init ai-foundations
cd ai-foundations
```

**Expected result:** `uv --version` prints something like `uv 0.5.x`. `uv init` creates `ai-foundations/` containing `pyproject.toml`, a `.python-version`, a `README.md`, and a starter `main.py` or `hello.py`.

**Verify:** `ls -a` shows `pyproject.toml`. `cat pyproject.toml` shows a `[project]` table with your project name.

**Troubleshoot:** If `uv` is "command not found" after install, the installer added it to a shell profile that your current session hasn't sourced — open a *new* terminal, or add `~/.local/bin` (macOS/Linux) / `%USERPROFILE%\.local\bin` (Windows) to PATH manually.

---

### Step 2 — Add dependencies (locked)

**What:** Add runtime and dev dependencies. `uv` resolves them and writes `uv.lock`.

**Why:** `uv add` updates `pyproject.toml` *and* re-locks in one step, so the lockfile always matches your declared deps. Runtime deps (numpy/pandas/etc.) ship with the project; dev deps (pytest/nbconvert) are only needed to build and test.

**Do it:**

```bash
uv add numpy pandas scikit-learn jupyterlab python-dotenv tiktoken
uv add --dev pytest nbconvert
```

**Expected result:** `uv` prints a resolution summary ("Resolved N packages") and creates a `.venv/` plus `uv.lock`. `pyproject.toml` now lists your dependencies under `[project] dependencies` and dev ones under `[dependency-groups] dev` (or `[tool.uv] dev-dependencies`).

**Verify:**

```bash
uv run python -c "import numpy, pandas, sklearn, dotenv, tiktoken; print('imports OK')"
```

Prints `imports OK`. Also confirm `uv.lock` exists and is non-trivial: `wc -l uv.lock` shows hundreds of lines.

**Troubleshoot:** Slow or failing resolution is almost always a network/proxy issue — retry, or set `UV_HTTP_TIMEOUT=120`. If `scikit-learn`'s wheel won't install on an unusual platform, ensure you're on Python 3.11–3.12 (some wheels lag the newest Python).

---

### Step 3 — Create the repo layout, `.gitignore`, and `.env`

**What:** Lay out the package structure the whole phase reuses, and set up secrets hygiene *before* your first commit.

**Why:** Getting `.gitignore` right before committing means a secret can never enter history. Git history is forever — the only fix for a committed key is rotating it. (See the secrets section of [`../lectures/01-reproducible-envs-uv.md`](../lectures/01-reproducible-envs-uv.md).)

**Do it:**

```bash
# package + supporting dirs
mkdir -p src/ai_foundations notebooks data tests
touch src/ai_foundations/__init__.py \
      src/ai_foundations/io_jsonl.py \
      src/ai_foundations/metrics.py \
      tests/test_metrics.py \
      tests/test_io_jsonl.py
```

Target layout:

```
ai-foundations/
  pyproject.toml        # uv-managed
  uv.lock               # committed — reproducibility
  .env                  # NEVER committed
  .env.example          # committed template
  .gitignore
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

Create `.gitignore`:

```gitignore
# secrets & data
.env
data/
# python / venv
.venv/
__pycache__/
*.pyc
# notebooks
.ipynb_checkpoints/
```

Create `.env.example` (committed template — placeholders only, never real values):

```dotenv
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

Create your real `.env` (git will ignore it) — leave the values blank this week; you don't need keys until Week 2:

```bash
cp .env.example .env
```

Make the package importable as `ai_foundations`. In `pyproject.toml`, ensure the build finds `src/` — add this if `uv init` didn't:

```toml
[tool.uv]
package = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ai_foundations"]
```

Then re-sync so the package installs in editable/importable form:

```bash
uv sync
uv run python -c "import ai_foundations; print(ai_foundations.__file__)"
```

**Expected result:** the import prints a path inside `src/ai_foundations/__init__.py`.

**Verify:** `git status --porcelain` should **not** list `.env` (only `.env.example`). Confirm explicitly:

```bash
git add -A
git status --short | grep -E '(^|\s)\.env$' && echo "DANGER: .env staged" || echo ".env correctly ignored"
```

You want to see `.env correctly ignored`.

**Troubleshoot:** If `import ai_foundations` fails with `ModuleNotFoundError`, the package isn't declared. Confirm `src/ai_foundations/__init__.py` exists and the `[tool.hatch...packages]` line points at `src/ai_foundations`, then `uv sync` again. Prefer `uv run python ...` over bare `python` so you're inside the project venv.

---

### Step 4 — Secrets helper in `__init__.py`

**What:** Load `.env` and expose a helper that reads a key from the environment, raising a clear error if it's missing — never hardcoding, never printing the value.

**Why:** Config lives in the environment (Twelve-Factor). A helper that fails loudly on a missing key turns a confusing 401 deep in an API call into an obvious "you forgot to set OPENAI_API_KEY."

**Do it** — in `src/ai_foundations/__init__.py`:

```python
"""ai_foundations package: shared env + secrets loading."""
import os
from dotenv import load_dotenv

# Load .env once at import time; real env vars still win over the file.
load_dotenv(override=False)


def get_secret(name: str) -> str:
    """Return the secret `name` from the environment.

    Raises a clear error if it is missing or blank. Never prints the value.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required secret {name!r}. "
            f"Add it to your .env file (see .env.example)."
        )
    return value
```

**Expected result:** importing the package auto-loads `.env`. Calling `get_secret("OPENAI_API_KEY")` with a blank `.env` raises the helpful `RuntimeError`.

**Verify:**

```bash
uv run python -c "from ai_foundations import get_secret; \
  import sys; \
  try: get_secret('OPENAI_API_KEY'); \
  except RuntimeError as e: print('raised OK:', 'Missing' in str(e))"
```

Prints `raised OK: True` (blank key → error, as intended). If you temporarily `export OPENAI_API_KEY=test123` and re-run, it should *not* raise.

**Troubleshoot:** If the error never fires even with a blank `.env`, an env var of the same name is already exported in your shell — `unset OPENAI_API_KEY` and retry. Never `print(value)` to debug; check `bool(value)` instead.

---

### Step 5 — JSONL I/O module

**What:** In `io_jsonl.py`, implement a *streaming* reader and a writer for JSONL (one JSON object per line).

**Why:** JSONL is the format every LLM fine-tune and eval dataset uses because it streams, appends, and survives partial writes. A streaming reader (yields one row at a time) handles million-line files without loading them into memory. You'll reuse this in every phase.

**Do it** — in `src/ai_foundations/io_jsonl.py`:

```python
import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one dict per non-blank line. Streaming: O(1) memory per row."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:            # skip blank lines
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Write each dict as one JSON line. Returns the count written."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n
```

Key design choices to note: `encoding="utf-8"` (so non-English text round-trips), `ensure_ascii=False` (keeps characters readable, not `\uXXXX`), a trailing newline per row, and blank-line skipping on read.

**Expected result:** a reader you can loop over lazily and a writer that returns how many rows it wrote.

**Verify** — write `tests/test_io_jsonl.py`:

```python
from ai_foundations.io_jsonl import read_jsonl, write_jsonl


def test_roundtrip_100_records(tmp_path):
    rows = [{"id": i, "text": f"row-{i}", "unicode": "café ☕"} for i in range(100)]
    p = tmp_path / "data.jsonl"
    n = write_jsonl(p, rows)
    assert n == 100
    back = list(read_jsonl(p))
    assert back == rows          # lossless round-trip, order preserved


def test_skips_blank_lines(tmp_path):
    p = tmp_path / "blank.jsonl"
    p.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert list(read_jsonl(p)) == [{"a": 1}, {"a": 2}]
```

Run `uv run pytest tests/test_io_jsonl.py -q` — both tests pass. This satisfies the DoD "round-trip 100 records losslessly."

**Troubleshoot:** A `UnicodeDecodeError` means a file was written without UTF-8 somewhere — always pass `encoding="utf-8"`. If the round-trip test fails on the unicode field, you likely left `ensure_ascii=True` (default) — that still round-trips, but if you compare against a pre-escaped string it won't match; compare against the original Python objects as above.

---

### Step 6 — NumPy/pandas notebook (broadcasting, axes, cosine sim)

**What:** Create `notebooks/01_numpy_pandas.ipynb` and work through arrays, broadcasting, axis reductions, a NumPy cosine similarity, and a pandas group-by that you export to JSONL.

**Why:** This is the tensor mental model — shape/dtype/broadcasting/axes — that you'll use to read model code and debug shape bugs for the rest of the roadmap. The cosine function you write here is *literally reused* for embeddings in Week 2. (See [`../lectures/03-numpy-tensors-vectorized-thinking.md`](../lectures/03-numpy-tensors-vectorized-thinking.md).)

**Do it:**

```bash
uv run jupyter lab   # opens the browser; create notebooks/01_numpy_pandas.ipynb
```

Build these cells (paste and run top-to-bottom). **Cell 1 — shape & dtype:**

```python
import numpy as np
a = np.arange(12).reshape(3, 4)
print(a.shape, a.dtype)          # (3, 4) int64
print(a.astype(np.float32).dtype)  # float32 — half the memory
```

**Cell 2 — broadcasting `(3,1)` with `(1,4)`:**

```python
col = np.arange(3).reshape(3, 1)   # shape (3, 1)
row = np.arange(4).reshape(1, 4)   # shape (1, 4)
grid = col + row                    # broadcasts to (3, 4)
print(grid.shape)                   # (3, 4)
print(grid)
```

Explain in a markdown cell *why*: dimensions align from the right; a size-1 dim stretches to match. `(3,1)` + `(1,4)` → `(3,4)`.

**Cell 3 — axis reductions (the classic trap):**

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])
print(m.sum(axis=0))  # [5 7 9]  — down COLUMNS (collapses rows)
print(m.sum(axis=1))  # [6 15]   — across ROWS (collapses columns)
```

Note the rule: `axis=k` is the axis that *disappears*.

**Cell 4 — cosine similarity in pure NumPy** (write this exact function; you'll port it to `metrics.py` and reuse it in Week 2):

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

print(cosine_similarity([1, 0, 0], [1, 0, 0]))  # 1.0 (identical)
print(round(cosine_similarity([1, 0], [0, 1]), 6))  # 0.0 (orthogonal)
```

**Cell 5 — pandas group-by + export to JSONL** (uses your Step 5 module):

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
import sys; sys.path.insert(0, "../src")   # make ai_foundations importable from the notebook
from ai_foundations.io_jsonl import write_jsonl

ds = load_breast_cancer(as_frame=True)
df = ds.frame
# group-by: mean radius per target class
summary = df.groupby("target")["mean radius"].mean().reset_index()
print(summary)

# export a slice to JSONL via your module
write_jsonl("../data/cancer_slice.jsonl", df.head(50).to_dict(orient="records"))
```

**Expected result:** each cell prints the shapes/values above; the group-by shows malignant vs benign have clearly different mean radius; `data/cancer_slice.jsonl` appears with 50 lines.

**Verify:** `wc -l data/cancer_slice.jsonl` → `50`. `head -1 data/cancer_slice.jsonl` shows one JSON object.

**Troubleshoot:** If `from ai_foundations...` fails in the notebook, either add the `sys.path.insert` shim shown above, or (cleaner) rely on the installed package from Step 3 (`uv sync` with `package = true`) and drop the shim. Prefer launching Jupyter with `uv run jupyter lab` so the kernel is the project venv — a mismatched kernel is the #1 cause of "numpy not found" in an otherwise-working project.

---

### Step 7 — Classical ML baseline + learning curve

**What:** In the same notebook (or `notebooks/01b_ml.ipynb`), split with stratification, fit a `LogisticRegression` baseline, and plot a train-vs-validation learning curve to *see* overfitting.

**Why:** Baselines before models — most tasks are still logistic regression, and you only reach for an LLM when the input is unstructured language. `stratify` keeps class balance identical across splits (critical for imbalanced data). A learning curve is how you diagnose over/underfitting visually. Fit the scaler *inside* the training split only — fitting on all data is textbook leakage. (See [`../lectures/04-just-enough-classical-ml.md`](../lectures/04-just-enough-classical-ml.md) and [`../lectures/05-data-leakage.md`](../lectures/05-data-leakage.md).)

**Do it:**

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)

# stratify=y keeps the malignant/benign ratio identical in train and test
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline scales INSIDE cross-validation folds → no leakage.
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_te)
p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary")
print(f"precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
print("confusion matrix:\n", confusion_matrix(y_te, y_pred))
```

Now the learning curve (train vs validation score as training-set size grows):

```python
sizes, train_scores, val_scores = learning_curve(
    clf, X_tr, y_tr, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 6), scoring="f1"
)
plt.plot(sizes, train_scores.mean(axis=1), "o-", label="train")
plt.plot(sizes, val_scores.mean(axis=1), "o-", label="validation")
plt.xlabel("training examples"); plt.ylabel("F1"); plt.legend()
plt.title("Learning curve — gap = overfitting")
plt.show()
```

**Expected result:** test precision/recall/F1 all around ~0.95+. The learning curve shows a train curve near the top and a validation curve rising toward it; a persistent **gap** between them is the visual signature of overfitting (variance), while two curves that plateau low together signals underfitting (bias).

**Verify:** the plot renders inline; both curves are plotted; you can point to the gap and say "that's the variance." Write one markdown sentence interpreting *your* curve.

**Troubleshoot:** `ConvergenceWarning` → raise `max_iter` (5000 above) or you forgot the `StandardScaler`. If precision/recall look suspiciously perfect (1.000), you likely leaked — e.g. scaled before splitting, or reused test rows in train. Keep the `StandardScaler` *inside* the pipeline so it's re-fit per fold.

---

### Step 8 — Metrics from scratch + tests vs scikit-learn

**What:** In `src/ai_foundations/metrics.py`, implement `precision_recall_f1(y_true, y_pred)` and `ndcg_at_k(relevances, k)` with NumPy only, then prove they match scikit-learn / a hand-computed value in `tests/test_metrics.py`.

**Why:** You defend metric choices in design reviews for the rest of your career; implementing them once burns in what they actually count. precision = "of what I flagged, how much was right"; recall = "of what mattered, how much I caught." nDCG@k rewards putting *highly* relevant items *high*, with a `log2(i+1)` position discount. (See [`../lectures/06-metrics-that-match-cost.md`](../lectures/06-metrics-that-match-cost.md).)

**Do it** — in `src/ai_foundations/metrics.py`:

```python
import numpy as np


def precision_recall_f1(y_true, y_pred):
    """Binary precision/recall/F1 for the positive class (label 1).

    Returns (precision, recall, f1) as floats.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def dcg_at_k(relevances, k):
    """DCG@k = sum(rel_i / log2(i + 1)) for i = 1..k, positions 1-indexed."""
    rel = np.asarray(relevances, dtype=np.float64)[:k]
    positions = np.arange(1, rel.size + 1)          # 1, 2, 3, ...
    discounts = np.log2(positions + 1)              # log2(2), log2(3), ...
    return float(np.sum(rel / discounts))


def ndcg_at_k(relevances, k):
    """nDCG@k = DCG@k / IDCG@k. `relevances` are graded gains in ranked order."""
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)        # ideal ordering
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg else 0.0
```

Also port `cosine_similarity` from Step 6 into this module (the spine's suggested layout puts cosine in `metrics.py`).

**Expected result:** three functions matching the definitions above. Worked check from the lecture: `relevances = [2, 0, 3, 1, 0]` gives DCG@5 = 3.931, IDCG@5 = 4.762, **nDCG@5 = 0.826**.

**Verify** — write `tests/test_metrics.py` asserting agreement with sklearn to 1e-6:

```python
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, ndcg_score
from ai_foundations.metrics import precision_recall_f1, ndcg_at_k


def test_prf_matches_sklearn():
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    sp, sr, sf, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    assert abs(p - sp) < 1e-6
    assert abs(r - sr) < 1e-6
    assert abs(f1 - sf) < 1e-6


def test_ndcg_hand_computed():
    # From lecture 06 worked example.
    assert abs(ndcg_at_k([2, 0, 3, 1, 0], 5) - 0.826) < 1e-3


def test_ndcg_matches_sklearn():
    rels = [3, 2, 3, 0, 1, 2]
    mine = ndcg_at_k(rels, 6)
    # sklearn wants (n_samples, n_labels): true relevances vs the scores that
    # produced THIS order. A strictly-decreasing score reproduces the given order.
    scores = np.arange(len(rels), 0, -1).reshape(1, -1)
    truth = np.asarray(rels).reshape(1, -1)
    assert abs(mine - ndcg_score(truth, scores, k=6)) < 1e-6
```

Run `uv run pytest tests/test_metrics.py -q` — all pass. This satisfies the DoD "hand-written metrics match sklearn to 1e-6."

**Troubleshoot:** If `test_ndcg_matches_sklearn` fails, remember sklearn's `ndcg_score` takes `(y_true, y_score)` and derives the ranking from `y_score`; feed a strictly decreasing score so it ranks items in the same order your `relevances` list already assumes. A common off-by-one bug is using `log2(i)` (position 1 → `log2(1)=0` → division by zero) instead of `log2(i+1)`.

---

### Step 9 — Prove no hidden state (headless run)

**What:** Execute the notebook from the command line and run the full test suite. Commit the executed notebook.

**Why:** Notebooks run cells out of order, leaving zombie variables that no longer exist in the code — the #1 source of "it worked yesterday." Nothing is trusted until *Restart & Run All* passes, and running it *headless* means CI can enforce it. (See the notebooks section of [`../lectures/01-reproducible-envs-uv.md`](../lectures/01-reproducible-envs-uv.md).)

**Do it:**

```bash
# Execute the notebook top-to-bottom in a fresh kernel:
uv run jupyter execute notebooks/01_numpy_pandas.ipynb
# equivalent, writes outputs back into the file:
#   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_numpy_pandas.ipynb

# Run the whole test suite:
uv run pytest -q
echo "exit code: $?"
```

**Expected result:** `jupyter execute` finishes with **exit code 0** and no traceback. `pytest` prints all green (e.g. `5 passed`). If any cell relied on a variable defined only in a since-deleted cell, the headless run *fails here* — which is exactly the bug you want to catch.

**Verify:**

```bash
uv run jupyter execute notebooks/01_numpy_pandas.ipynb && echo "NO HIDDEN STATE: exit 0"
```

Then commit:

```bash
git add -A
git commit -m "feat: week-1 repro env, jsonl io, numpy notebook, classical ML + metrics"
```

**Troubleshoot:** A headless failure with `NameError` means real hidden state — open the notebook, *Restart Kernel & Run All* in the UI, fix the cell that references a missing variable, and re-run headless. If `jupyter execute` can't find a kernel, install the venv as a kernel or just prefix everything with `uv run` so the project environment is used.

---

## Putting it together — end-to-end run

Everything now hangs together. Simulate a fresh contributor cloning your repo:

```bash
cd ..                                  # leave the project dir
git clone ai-foundations ai-foundations-clone   # or clone from your remote
cd ai-foundations-clone
uv sync                                # rebuilds the EXACT env from uv.lock
uv run pytest -q                       # green from a clean checkout
uv run jupyter execute notebooks/01_numpy_pandas.ipynb   # exit 0, no hidden state
git log -p | grep -i key               # only the .env.example placeholder appears
```

The pieces connect like this: `__init__.py` loads secrets → `io_jsonl.py` is the data layer the notebook writes through → the notebook builds the NumPy/pandas intuition and the ML baseline → `metrics.py` holds the reusable, tested metrics (and cosine sim, reused in Week 2) → the headless run + pytest are the reproducibility gate. This same skeleton grows into the Week 3 milestone (Model Economics CLI) — you'll add `tokens.py`, `llm.py`, `embed.py`, and `econ_cli.py` to the exact same `src/ai_foundations/` package.

---

## Definition of Done — verifiable checks

Restate of the spine's acceptance gate. You are **not done** until every box is checked:

- [ ] **Clean-checkout green:** `git clone` + `uv sync` + `uv run pytest -q` passes on a fresh clone, and `uv.lock` is committed. *(Verify: run the end-to-end block above.)*
- [ ] **No secrets in history:** `git log -p | grep -i key` returns only the `.env.example` placeholder — no real key anywhere in git history.
- [ ] **No hidden state:** `uv run jupyter execute notebooks/01_numpy_pandas.ipynb` exits 0.
- [ ] **JSONL round-trips:** a test writes then reads 100 records losslessly (`tests/test_io_jsonl.py` passes).
- [ ] **Metrics match reference:** hand-written `precision_recall_f1` and `ndcg_at_k` match scikit-learn / a hand-computed value to 1e-6 (`tests/test_metrics.py` passes).
- [ ] **Concepts, out loud:** you can state, in one sentence each, three distinct data-leakage mechanisms (e.g. fitting a scaler on all data before splitting; target leakage from a post-outcome feature; temporal leakage / train rows dated after test rows; duplicate rows across splits), and say which metric you'd optimize for a **fraud detector** (recall / PR-AUC — missing fraud is costly, positives are rare) vs a **document search box** (precision@k / nDCG@k — the top results must be right).

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `uv: command not found` after install | PATH not refreshed | Open a new terminal; add `~/.local/bin` (mac/Linux) or `%USERPROFILE%\.local\bin` (Win) to PATH |
| `ModuleNotFoundError: ai_foundations` | package not declared / wrong kernel | `package = true` + hatch `packages = ["src/ai_foundations"]`, then `uv sync`; launch Jupyter via `uv run jupyter lab` |
| Notebook: `numpy not found` but CLI works | notebook using a different kernel | Start Jupyter with `uv run jupyter lab` so the kernel is the project venv |
| `.env` shows up in `git status` | `.gitignore` added after staging, or wrong path | Ensure `.env` is in `.gitignore`; `git rm --cached .env` if already tracked |
| `git log` shows a real key | key was committed | **Rotate the key immediately**; history is forever (scrubbing is a last resort) |
| `ConvergenceWarning` from LogisticRegression | unscaled features / too few iterations | Keep `StandardScaler` in the pipeline; raise `max_iter` |
| Suspiciously perfect (1.000) scores | data leakage | Split first, fit scaler/encoder on train only (use a `Pipeline`); check for duplicate rows across splits |
| nDCG `nan` or `inf` | `log2(i)` used (pos 1 → 0) or IDCG=0 | Use `log2(i+1)`; guard `idcg == 0` → return 0.0 |
| `ndcg_score` disagrees with yours | sklearn ranks by `y_score`, not list order | Pass a strictly-decreasing `y_score` matching your assumed ranking |
| `UnicodeDecodeError` reading JSONL | non-UTF-8 write | Always `open(..., encoding="utf-8")`; `ensure_ascii=False` on write |
| `jupyter execute` fails with `NameError` | genuine hidden state | Restart Kernel & Run All in UI, fix the offending cell, re-run headless |

---

## Stretch goals (optional)

- **Wire up CI.** Add `.github/workflows/ci.yml` that runs `uv sync`, `uv run pytest -q`, and `uv run jupyter execute notebooks/01_numpy_pandas.ipynb` on push — the reproducibility gate now enforced by a machine, not your memory. (Prep for the Week 3 milestone.)
- **Add `recall@k` and `MRR`** to `metrics.py` with tests — you'll lean on these heavily for RAG in Phases 3–4.
- **Threshold sweep.** For the LogisticRegression model, plot precision and recall as you move the decision threshold off 0.5, and pick the threshold on purpose for a "high-recall fraud" vs "high-precision search" scenario.
- **Deliberately leak, then measure.** Fit the `StandardScaler` on the *full* dataset before splitting and record how much the validation F1 inflates — feel leakage in numbers, not just theory (ties directly to [`../lectures/05-data-leakage.md`](../lectures/05-data-leakage.md)).
- **PR-AUC on an imbalanced slice.** Downsample the positives to ~5% and show why accuracy becomes a useless metric while PR-AUC still discriminates.
