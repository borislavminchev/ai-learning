# Lecture 2: Notebooks Without Hidden State

> Jupyter notebooks are the single most productive tool for exploring data and models — and the single most common source of results that cannot be reproduced. The reason is *hidden state*: the notebook you see on screen is not the program that produced the numbers in it. This lecture exists because "it worked in the notebook" is where a shocking number of AI bugs are born and buried. After this lecture you will understand the Jupyter execution model well enough to know exactly how hidden state accumulates, you will have a hard discipline (`Restart & Run All`, enforced headlessly in CI) that makes it impossible, and you will know when to stop exploring in a notebook and graduate code into a real module.

**Prerequisites:** Lecture 1 (reproducible envs); comfortable running Python and a terminal · **Reading time:** ~20 min · **Part of:** Phase 0 Week 1

---

## The core idea (plain language)

A Python *script* has one honest property: if you run it top to bottom, the result is a pure function of the source you can see. A **notebook breaks that contract.** A notebook is a list of code *cells* attached to a long-lived process called the **kernel**. You can run the cells in any order, run one cell five times, edit a cell after running it, or delete a cell whose variables are still alive in the kernel's memory. What you *see* is the cells' source; what actually *ran* is an invisible history of executions in whatever order you happened to click.

**Hidden state is the gap between "the code visible in the notebook" and "the state actually living in the kernel."** When that gap is nonzero — and it silently grows the longer you work — the notebook can show a beautiful result that *no clean run of the same cells will ever reproduce.* The number is real; the program that made it no longer exists on disk.

The fix is not a tool, it's a discipline with one rule: **nothing is trusted until `Restart Kernel & Run All` passes** — ideally from the command line, so a machine (CI) can enforce it, not just your good intentions.

## How it actually works (mechanism, from first principles)

### The kernel and the global namespace

When you open a notebook, Jupyter starts a **kernel**: a persistent Python interpreter that outlives any single cell. Every cell you execute runs `exec()` inside that kernel's **one shared global namespace**. Variables, imports, function definitions, monkeypatches, opened file handles, random-number-generator state — all of it accumulates in that namespace and *stays there* until the kernel restarts.

Two numbers make this concrete. Each executed cell gets a monotonically increasing **execution count**, the `[N]` in the margin. That number is the only visible trace of *order of execution*. The cells' *position on screen* is the order you'll read them in later. **When those two orders disagree, you have hidden state.**

```
On screen (top→bottom)      Execution count (what actually ran)
┌──────────────────────┐
│ [1] import pandas    │     1   ← ran first
│ [4] df = clean(df)   │     4   ← ran LAST, after you edited it twice
│ [2] df = load()      │     2   ← ran second
│ [3] df = load()      │     3   ← ran third (you re-ran load)
└──────────────────────┘
Read top-to-bottom it says: import → clean → load → load.
But `clean` ran at count 4, AFTER both loads, referencing a df
that a fresh top-to-bottom run would not have in that state.
A clean re-run executes in ON-SCREEN order and gets a different (or crashing) result.
```

### The three failure modes

**1. Out-of-order execution.** You run cell 5, realize you need a fix in cell 2, edit and run cell 2, then keep going from cell 6 — never re-running 3, 4, 5. The kernel now holds a Frankenstein state that no linear reading of the notebook describes. The classic tell: execution counts that don't increase down the page (`[7]` above `[3]`), or gaps and repeats.

**2. Deleted-cell ghosts.** You write `helper = lambda x: x*2` in a cell, use it everywhere, then delete that cell because it felt untidy. `helper` is *still in the kernel's namespace* — every downstream cell keeps working. You commit. A teammate runs it fresh and gets `NameError: name 'helper' is not defined`, because the definition exists nowhere in the file. The code that made your result is *gone* but the result looks fine to you.

**3. Mutated-in-place state.** AI notebooks are full of expensive, mutable objects: a DataFrame you filter, a tokenizer you configure, a model you move to GPU, a global RNG you seed. Re-running a cell that does `df = df[df.score > 0.5]` a second time filters the *already-filtered* df — now it's `score > 0.5` twice, i.e. a smaller df than a clean run produces. Re-run it three times and your "20k rows" is silently 4k. Nothing errors. The number is just wrong.

### Why "it worked yesterday" happens

Yesterday you had a kernel that had been alive for hours, accumulating exactly the right pile of state through a path of clicks you'll never remember. You closed your laptop (the kernel *keeps running* or is checkpointed by the UI, so tomorrow-you sees the outputs still rendered). Today the kernel is fresh — or you restart it — and the notebook cannot rebuild that pile from its visible cells. The outputs saved in the `.ipynb` file are *cached pictures of a state that no longer exists.* "Nothing changed" is true of the code and false of the kernel, and the kernel was where the truth lived.

### What's actually in the `.ipynb` file

An `.ipynb` is JSON: a list of cells, each with its `source` (the code), its `execution_count`, and its saved `outputs` (text, tables, base64 PNGs of plots). Critically, **the outputs are stored in the file independent of whether they match the current source.** You can have a cell whose code says `2+2` and whose saved output says `5`, committed to git, rendering `5` to everyone who opens it. This is why notebook outputs are evidence of *nothing* until you've done a clean run — and why committing notebooks with stale outputs quietly ships lies.

## Worked example

You're building a labeled dataset for a Phase 5 fine-tune. Your notebook, read top to bottom, looks like this:

```python
# Cell A
import pandas as pd
df = pd.read_json("raw.jsonl", lines=True)   # 20,000 rows

# Cell B
df = df.drop_duplicates(subset="text")        # dedup

# Cell C
df = df[df["lang"] == "en"]                    # keep English

# Cell D
print(len(df), "examples ready")               # → prints 12,431
df.to_json("clean.jsonl", lines=True, orient="records")
```

While working, you ran A, B, C, D. Then you tweaked Cell C's filter, re-ran **C**, re-ran **D**. Then you found a dedup bug, edited **B**, re-ran **B** — but *did not re-run C*. You glance at Cell D's output: `12,431 examples ready`. You commit `clean.jsonl` and the notebook.

What actually happened: after your last edit, `B` ran on the already-English-filtered `df` (because C had run before your B re-run), so dedup operated on 12,431 rows, not 20,000. The `df` written to `clean.jsonl` is some in-between state that **a clean top-to-bottom run will never produce** — that run would do load → dedup(20k) → filter → a *different* count. Your training set is now a number nobody can reproduce, and if the fine-tune later underperforms you have no way to tell whether it's the data, because the data-generating program doesn't exist.

Now the discipline. You hit **Restart Kernel & Run All**. The kernel is wiped; cells run strictly A→B→C→D. It prints `11,908`, not `12,431`. The discrepancy *is the bug* — surfaced instantly. You fix it, restart-run-all again, get a stable number, and only *now* commit. From the terminal you can prove it with one command:

```bash
uv run jupyter execute notebooks/build_dataset.ipynb
# exit code 0  → every cell ran clean, in order, no errors
# non-zero     → a fresh run does NOT reproduce; do not trust the outputs
```

## How it shows up in production

- **Irreproducible results / eval scores.** The most expensive version of this bug: a model or RAG eval score computed in a notebook that no clean run reproduces. You report `0.87 recall@5` to your team, someone tries to build on it, gets `0.79`, and you lose a day discovering the original was a hidden-state artifact. Any number that matters — an eval, a dataset size, a cost estimate — must come from a clean run or a script, never a hand-clicked notebook.

- **"Works in the notebook, breaks in the pipeline."** Deleted-cell ghosts and out-of-order state are invisible in the notebook and fatal the moment the logic is lifted into a scheduled job (Airflow/Dagster in Phase 5) or an API handler (Phase 9), which necessarily run top-to-bottom, once, in a fresh process. The notebook's forgiving execution model is exactly what production lacks.

- **Silent data corruption in training sets.** The mutated-in-place trap (double-filtering, double-normalizing) doesn't crash — it produces a plausible-but-wrong dataset. In Phase 8 a fine-tune trained on a subtly corrupted set fails mysteriously, and the corruption is un-diagnosable because it lives in a kernel state that's long gone.

- **Merge hell and unreviewable diffs.** `.ipynb` is JSON with embedded outputs (including base64 images). Two people editing the same notebook produce git conflicts in machine-generated JSON that are nearly impossible to resolve by hand, and PR diffs are dominated by output churn (a re-rendered plot changes hundreds of lines). This is a real team-velocity tax, addressed below.

- **Secrets and cost surprises baked into outputs.** A cell that prints an API response can serialize a key or PII into the notebook's saved outputs and commit it to git (see Lecture 1 on why that's forever). And a cell left running a paid-API loop in a live kernel keeps billing you while you're not looking.

### The discipline, operationalized

1. **`Restart Kernel & Run All` before you believe any number, and before every commit.** This is the whole game. If it doesn't pass clean, the notebook is lying.
2. **Enforce it headlessly so a machine checks, not your memory.** `jupyter execute nb.ipynb` (or `jupyter nbconvert --to notebook --execute --inplace nb.ipynb`) runs the whole notebook in a fresh kernel from the CLI and exits non-zero on any error. Put it in CI — this is literally a Phase 0 Definition-of-Done gate.
3. **Strip outputs before committing** to kill merge hell and secret-in-output leaks. Tools: `nbstripout` (a git filter that auto-strips outputs on commit) or `jupytext` (pairs each `.ipynb` with a plain `.py` that's the reviewable, diffable source of truth).
4. **Seed randomness explicitly in a cell**, not implicitly by kernel history, so a clean run is deterministic where it can be.
5. **Prefer forward-only execution while working:** after editing an upstream cell, re-run *everything below it* (Jupyter's "Run All Below"), not just the cell you touched.

### When to graduate out of the notebook

Notebooks are for **exploration**: looking at data, eyeballing model outputs, sketching a transform. The moment a piece of code is *reused* — imported by another notebook, called twice, or destined for a pipeline/API — it should **graduate into a plain `.py` module** under `src/`, with a function signature, a docstring, and a `pytest`. The rule of thumb: **if you'd be upset to lose it, or you've copy-pasted it once, it belongs in a module, not a cell.** The notebook then just imports and calls it. This is exactly the layout the Week 1 lab enforces (`src/ai_foundations/` for real code, `notebooks/` for exploration that *imports* from `src/`).

## Common misconceptions & failure modes

- **"The saved outputs prove the code works."** No — outputs are cached JSON stored independently of the source; they can show `5` for `2+2`. Only a fresh `Restart & Run All` (or headless execute) proves anything.
- **"I always run cells in order, so I'm fine."** You don't. The instant you edit an upstream cell and continue downward without re-running the middle, you've diverged. Execution counts out of sequence are the objective tell.
- **"Restarting loses my work."** It loses *kernel state*, which is the thing you *want* to lose — that's the point. Your code (the cells) and, if you seeded RNGs and read from files, your results are all rebuildable. If they're *not* rebuildable, that's the bug, and better to learn it now than in prod.
- **"Notebooks aren't for real engineers."** False dichotomy. Notebooks are superb for exploration and terrible as the system of record. The skill is knowing which mode you're in and graduating code at the boundary.
- **"I'll just commit the notebook with outputs so people see the results."** That's how you get merge conflicts in base64 PNGs and secrets in git history. Strip outputs (`nbstripout`) and commit a clean, executable notebook; let CI produce the outputs.
- **`%run` / `import` inside notebooks re-introduces state.** Importing a module caches it (`sys.modules`); editing that module and re-importing won't reload it unless you use `%load_ext autoreload` / `%autoreload 2`. A stale imported module is another flavor of hidden state.

## Rules of thumb / cheat sheet

- **Trust nothing until `Restart Kernel & Run All` is green.** Do it before every commit and before believing any number.
- **Enforce headlessly in CI:** `jupyter execute nb.ipynb` (or `nbconvert --to notebook --execute --inplace`) must exit 0. This is a Definition-of-Done gate.
- **Execution counts must increase down the page.** Out-of-order `[N]` = hidden state present. Fix by "Run All Below" after any upstream edit.
- **Strip outputs before commit** with `nbstripout`; or pair with `jupytext` for clean `.py` diffs and reviewable PRs.
- **Never `print()` secrets or raw API responses** in a cell you'll commit — outputs are serialized into the file.
- **Seed RNGs in a cell**, not by kernel history, so clean runs are reproducible.
- **Graduate reused code to `src/*.py` + a test.** Copy-pasted once or imported elsewhere → it's a module, not a cell.
- **`%autoreload 2`** when developing an imported module alongside a notebook, so edits actually take effect.

## Connect to the lab

This lecture is the deep version of Week 1 Lab step 6 ("Prove no hidden state"). In the lab you'll run `uv run jupyter execute notebooks/01_numpy_pandas.ipynb` and require exit code 0 — that's the machine-checked version of `Restart & Run All`. Watch for the spine's pitfall: *"Trusting notebook results after running cells out of order; always Restart & Run All before believing a number."* If your headless run fails but the notebook "looks right," you've just caught real hidden state — celebrate it, that's the lecture working.

## Going deeper (optional)

- **Jupyter documentation** — `docs.jupyter.org`. Read the execution model overview and the `nbconvert` / `jupyter execute` CLI pages (search **"jupyter nbconvert execute"** and **"jupyter execute command"**).
- **Joel Grus, "I Don't Like Notebooks"** — the canonical JupyterCon 2018 talk on exactly these failure modes; search that title on YouTube/Slides. Watch it once; it will make you paranoid in the correct way.
- **`nbstripout`** — search **"nbstripout git filter"** for auto-stripping outputs on commit.
- **`jupytext`** — search **"jupytext pair notebook py"** for pairing notebooks with reviewable plain-text `.py` files.
- **`papermill`** — search **"papermill parameterize notebook"** for running notebooks as parameterized, headless jobs (a bridge from exploration toward pipelines).

## Check yourself

1. What precisely is "hidden state" in a notebook, and what are the two on-screen signals that it's present?
2. A cell defining a helper function was deleted, but the notebook still runs for you. Why? What happens when a teammate runs it fresh?
3. Why are the outputs saved in an `.ipynb` file not evidence that the code works?
4. Give the exact command that machine-checks "no hidden state," and say what its exit code means.
5. You re-ran a `df = df[df.score > 0.5]` cell twice. What's wrong with `df` now, and why didn't it error?
6. What's the rule for when a piece of code should leave the notebook and become a module?

### Answer key

1. Hidden state is the gap between the code visible in the cells and the actual state living in the long-lived kernel's global namespace (accumulated via out-of-order/repeated/edited/deleted-cell executions). Signals: execution counts `[N]` that don't strictly increase down the page (gaps, repeats, or higher-above-lower), and cells whose results depend on variables not defined by any earlier visible cell.
2. The function is still in the kernel's namespace from when the (now-deleted) cell executed, so downstream cells keep resolving it. A teammate running fresh (or you after a restart) gets `NameError`, because the definition exists in no cell on disk — the code that produced your result is gone.
3. Outputs are stored in the `.ipynb` JSON independently of the source and are not re-validated against it; a cell can show a saved output that its current code would never produce. Only a fresh `Restart & Run All` / headless execute proves the code reproduces the outputs.
4. `jupyter execute nb.ipynb` (or `jupyter nbconvert --to notebook --execute --inplace nb.ipynb`). Exit code 0 = every cell ran, in order, in a fresh kernel, with no error (reproducible). Non-zero = a clean run fails; the notebook's outputs are not trustworthy.
5. `df` has been filtered twice, so it holds strictly fewer rows than one clean run produces (the second run filters the already-filtered frame). It didn't error because filtering a DataFrame that already satisfies the condition is perfectly valid — it just silently returns a smaller/wrong result.
6. If you'd be upset to lose it, or you've reused/copy-pasted it even once (or it's headed for a pipeline/API), it belongs in a `src/*.py` module with a signature, docstring, and a test — the notebook should import and call it, not own it.
