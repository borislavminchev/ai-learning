# Lecture 3: Vectorized Thinking — NumPy Arrays as the Tensor Mental Model

> Every model you will ever debug — a 7B Llama, an embedding batch, a training loop — is fundamentally a pile of n-dimensional arrays flowing through operations that must agree on *shape* and *dtype*. If you can look at a NumPy `ndarray` and instantly reason about its shape, its memory layout, and how it will broadcast against the next array, you have already built 80% of the mental model you need for PyTorch tensors, GPU memory math, and the shape-mismatch stack traces that will define your first months of real AI work. This lecture builds that model from first principles. After it you will be able to: read any array's `shape`/`dtype`/`strides` and predict what an operation does; apply the broadcasting rules by hand; never again confuse `axis=0` with `axis=1`; explain *why* vectorized code is 10–100× faster than a Python loop; choose float32 vs float16 with the precision/memory tradeoff in mind; and implement cosine similarity in a way you will literally reuse for embeddings next week.

**Prerequisites:** Comfortable Python (functions, lists, basic slicing); Lecture 1–2 (repro env, JSONL) helpful but not required · **Reading time:** ~28 min · **Part of:** Phase 0 Week 1

---

## The core idea (plain language)

A NumPy `ndarray` is not "a list of lists." It is three things bolted together:

1. A **flat, contiguous block of bytes** in memory (the *buffer*) — all the same type.
2. A **dtype** that says how to interpret each element (8-byte float? 4-byte int?).
3. A **shape** plus **strides** that say how to walk that flat buffer to pretend it is a grid, a cube, or a 5-D tensor.

That is the whole trick. The multidimensionality is a *view*, an addressing scheme layered on top of one long tape of numbers. Once you internalize "array = shape + dtype + strided buffer," a lot of confusing behavior becomes obvious: why reshaping is nearly free, why `a.T` doesn't copy, why some operations are fast and some secretly copy gigabytes, and why a shape mismatch is the single most common bug in ML code.

**Vectorized thinking** is the discipline of expressing computation as whole-array operations ("multiply these two 1000×1000 arrays") instead of element-by-element Python loops. You do this not because it's elegant, but because it moves the loop out of the slow Python interpreter and into optimized, compiled C that runs over contiguous memory at hardware speed. This is the exact same reason PyTorch and JAX exist. NumPy on CPU is the training-wheels version of the tensor programming model you'll use everywhere.

---

## How it actually works (mechanism, from first principles)

### 1. Shape + dtype + strided buffer

Make an array and interrogate it:

```python
import numpy as np
a = np.arange(12, dtype=np.float32).reshape(3, 4)
a.shape     # (3, 4)
a.dtype     # dtype('float32')
a.strides   # (16, 4)  <- bytes to step to next row, next column
a.nbytes    # 48        <- 12 elements * 4 bytes
```

The buffer is just `[0,1,2,...,11]` laid out flat. `shape=(3,4)` says "pretend this is 3 rows of 4." `strides=(16,4)` is the magic: to move one row down, jump 16 bytes (4 floats × 4 bytes); to move one column right, jump 4 bytes. Indexing `a[i,j]` is pure arithmetic on the buffer address: `base + i*16 + j*4`. No pointer-chasing, no per-element Python objects.

This is why these operations are **essentially free** (they only rewrite shape/strides metadata, no data copy):

```python
a.reshape(4, 3)   # same buffer, new (shape, strides)
a.T               # transpose: strides swapped to (4, 16) — a *view*
a[1:, :2]         # a slice — still a view into the same buffer
```

And why this **is not free** — it must physically reorder bytes:

```python
np.ascontiguousarray(a.T)   # forces a copy so memory is row-contiguous again
```

```
Logical view (shape 3x4)        Physical buffer (one flat tape)
  0  1  2  3
  4  5  6  7      <—— strides ——>   [0 1 2 3 4 5 6 7 8 9 10 11]
  8  9 10 11
```

Contrast with a Python list of lists: that is a box of pointers to pointers to boxed integer objects scattered across the heap — 28 bytes per `int`, cache-hostile, and the interpreter touches each one. The `ndarray` is 4 bytes per element, packed, and processed in compiled loops. Same data, wildly different machine.

### 2. Why vectorization beats Python loops

Sum a million numbers two ways:

```python
xs = np.arange(1_000_000, dtype=np.float64)

# Python loop: interpreter dispatches ~1M times, boxes/unboxes each float
total = 0.0
for x in xs:
    total += float(x)

# Vectorized: one C call over contiguous memory
total = xs.sum()
```

The vectorized version is typically **50–200× faster** (order-of-magnitude, hardware-dependent). Three compounding reasons:

- **Interpreter overhead gone.** The Python loop does a bytecode dispatch, a type check, and an object allocation *per element*. `xs.sum()` does one call into C and loops there.
- **Memory locality.** Contiguous 8-byte doubles stream through the CPU cache in cache-line-sized (64-byte) gulps. A list-of-objects thrashes the cache chasing pointers.
- **SIMD.** Modern CPUs have vector registers (SSE/AVX) that add 4, 8, or 16 floats in *one* instruction. NumPy's compiled kernels use them; a Python loop cannot. This is "single instruction, multiple data" — the CPU's own baby version of what a GPU does at massive scale.

The lesson generalizes: **if you wrote a `for` loop over array elements in hot AI code, you almost certainly made a mistake.** Express it as array ops and let the compiled kernel (and later, the GPU) do the work.

### 3. Broadcasting

Broadcasting is how NumPy lets arrays of *different but compatible* shapes combine without you writing loops or copying data to match sizes. The rules, applied by comparing shapes **right-aligned**, dimension by dimension:

1. Align shapes on the **right**. Pad the shorter shape with 1s on the **left**.
2. For each dimension, they are compatible if they are **equal**, or **one of them is 1**.
3. The result dimension is the **max** of the two. A size-1 dimension is *stretched* (virtually, no copy) to match.

Worked example — the classic `(3,1)` with `(1,4)` from the lab:

```
A.shape = (3, 1)
B.shape = (1, 4)
-----------------  right-aligned
result  = (3, 4)      # dim0: max(3,1)=3 ; dim1: max(1,4)=4
```

```python
A = np.array([[10], [20], [30]])   # (3,1)  a column
B = np.array([[1, 2, 3, 4]])       # (1,4)  a row
A + B
# [[11 12 13 14]
#  [21 22 23 24]
#  [31 32 33 34]]
```

Each element of the `(3,4)` output is `A[i,0] + B[0,j]`. No 3×4 copies of A or B are ever materialized; the size-1 axes are read with stride 0 (the same value re-read). This is the whole outer-product / grid pattern, for free.

A subtractive example that bites people — normalizing rows of a matrix:

```python
X = np.random.rand(100, 5)          # 100 rows, 5 features
mu = X.mean(axis=0)                 # shape (5,)  — per-column mean
X_centered = X - mu                 # (100,5) - (5,) -> (100,5)  OK
```

Here `(5,)` right-aligns under `(100,5)`, gets left-padded to `(1,5)`, and stretches to `(100,5)`. But watch:

```python
row_mean = X.mean(axis=1)           # shape (100,)
X - row_mean                        # (100,5) - (100,) -> ERROR
# operands could not be broadcast together with shapes (100,5) (100,)
```

`(100,)` right-aligns to the *last* axis of `(100,5)`, i.e. tries to match 100 against 5 — fail. The fix is to make the intent explicit with a size-1 axis:

```python
X - X.mean(axis=1, keepdims=True)   # (100,5) - (100,1) -> (100,5)  OK
```

**`keepdims=True` is your broadcasting seatbelt.** Use it whenever you reduce along an axis and then combine the result back with the original.

### 4. Axis semantics and the classic aggregation bug

`axis=n` means **"collapse dimension `n`; iterate over it and reduce it away."** The confusing part: reducing `axis=0` gives you a result indexed by the *remaining* axes.

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])           # shape (2, 3)

X.sum(axis=0)   # -> [5, 7, 9]      collapse rows: sum DOWN each column -> shape (3,)
X.sum(axis=1)   # -> [6, 15]        collapse cols: sum ACROSS each row  -> shape (2,)
X.sum()         # -> 21             collapse everything -> scalar
```

The mnemonic that actually sticks: **`axis=0` is the direction of increasing row index — so it moves vertically, down the columns. `axis=1` moves horizontally, across each row.** The result always has the reduced axis *removed* from the shape.

The **classic aggregation bug**: you have a batch of examples, shape `(batch, features)` = `(32, 768)` — say 32 embeddings of dimension 768. You want the average embedding of the batch. That is `X.mean(axis=0)` → `(768,)`. If you fat-finger `axis=1` you get `(32,)` — the mean *of each embedding's components*, a meaningless per-example scalar. **No error is raised.** The shapes are plausible, the numbers are finite, the pipeline runs, and your results are quietly wrong. This exact mistake — right operation, wrong axis, no crash — is one of the most expensive silent bugs in ML code. The habit that saves you: **assert the shape you expect after every reduction.**

```python
avg = X.mean(axis=0)
assert avg.shape == (768,), avg.shape
```

### 5. dtype, memory, and precision

The dtype sets bytes-per-element, which sets memory footprint, which sets what fits on your GPU:

| dtype | bytes/elem | ~decimal digits | typical use |
|-------|-----------|-----------------|-------------|
| float64 | 8 | ~15–16 | NumPy default; scientific CPU work |
| float32 | 4 | ~7 | ML default on CPU/GPU; "full precision" for training |
| float16 | 2 | ~3 | GPU inference/training; small dynamic range |
| bfloat16 | 2 | ~3 (wider range) | LLM training/inference standard |
| int8 | 1 | exact int | quantized inference |

The rule of thumb you will use constantly: **bytes = element_count × bytes_per_element.** A `7B`-parameter model at float16 (2 bytes) is ≈ 14 GB of weights (7e9 × 2). At float32 it would be ≈ 28 GB; at int8 ≈ 7 GB. This is the same arithmetic as `a.nbytes` on a tiny array, just scaled up — and it is exactly how you'll estimate VRAM in Week 3.

The precision tradeoff bites in two directions:

- **float16 has a tiny range.** Its max is ~65504; values above overflow to `inf`, and very small values underflow to 0. A sum of many positive numbers, or an `exp()` in a softmax, can overflow in fp16 while being fine in fp32. This is why mixed-precision training keeps a fp32 "master copy" and why `bfloat16` (same 2 bytes but a wider exponent, trading mantissa bits for range) became the LLM default — range matters more than precision for gradients.
- **Accumulation error.** Summing a million fp32 numbers accumulates rounding error; NumPy often accumulates reductions in a higher-precision type internally to mitigate this, but you should be aware that `arr.astype(np.float16).sum()` can differ visibly from the fp64 answer.

```python
big = np.float16(60000)
big + big          # inf   (overflowed fp16's ~65504 ceiling)
np.float32(60000) + np.float32(60000)   # 120000.0  fine
```

---

## Worked example — cosine similarity in NumPy

Cosine similarity measures the *angle* between two vectors, ignoring their magnitude — the workhorse metric for "how semantically similar are these two embeddings?" You'll reuse this exact function for embeddings in Week 2, so build it right.

Definition: `cos(a, b) = (a · b) / (‖a‖ · ‖b‖)`, ranging from -1 (opposite) through 0 (orthogonal) to 1 (identical direction).

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0                      # guard the zero-vector edge case
    return float(np.dot(a, b) / denom)
```

Trace it with numbers:

```python
a = np.array([1.0, 0.0, 1.0])
b = np.array([1.0, 1.0, 0.0])
# dot = 1*1 + 0*1 + 1*0 = 1
# ‖a‖ = sqrt(1+0+1) = sqrt(2) ≈ 1.414
# ‖b‖ = sqrt(1+1+0) = sqrt(2) ≈ 1.414
# cos = 1 / (1.414 * 1.414) = 1 / 2.0 = 0.5
cosine_similarity(a, b)   # 0.5
```

Now scale to the real use case — ranking many documents against one query — using **vectorization and broadcasting** instead of a Python loop over documents:

```python
query = np.random.rand(384).astype(np.float32)      # one embedding
docs  = np.random.rand(10, 384).astype(np.float32)  # 10 doc embeddings

# Normalize once, then a single matrix-vector product gives all 10 scores.
q_n = query / np.linalg.norm(query)
d_n = docs / np.linalg.norm(docs, axis=1, keepdims=True)   # <- keepdims!
scores = d_n @ q_n                                   # shape (10,)
ranking = np.argsort(scores)[::-1]                   # best first
```

Two things to notice, both from this lecture. First, `np.linalg.norm(docs, axis=1, keepdims=True)` produces shape `(10,1)`, which broadcasts cleanly against `(10,384)` to normalize each row — drop `keepdims` and you get the `(10,)`-vs-`(10,384)` broadcast error from earlier. Second, once vectors are unit-normalized, cosine similarity *is* just the dot product, so all 10 scores come from one `@` (matmul) call into optimized BLAS — no Python loop over documents. This pattern — normalize, then one matmul for the whole batch — is exactly how a vector database scores thousands of candidates per query.

---

## How it shows up in production

- **Shape-mismatch stack traces are your daily bread.** In PyTorch, `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x10)` is the same shape-algebra you practiced here. Engineers who "think in shapes" fix these in seconds; those who don't `print(x.shape)` at ten places and pray. The NumPy model *is* the PyTorch model — `.shape`, `.dtype`, broadcasting, and `.reshape` behave the same; PyTorch just adds `.device` (CPU/GPU) and autograd.
- **The silent axis bug costs real money.** Averaging a batch along the wrong axis, or computing a loss reduction over the wrong dimension, produces plausible finite numbers and no exception. You discover it days later when metrics are inexplicably flat. Shape assertions after reductions are cheap insurance.
- **dtype is VRAM, and VRAM is money.** Choosing fp16/bf16 over fp32 halves weight memory and often doubles throughput on GPUs with tensor cores — the difference between a model fitting on a 24 GB card or needing an 80 GB one (roughly 3–4× the hourly cost). The `bytes = count × bytes_per_element` arithmetic decides your hardware bill.
- **Precision bugs masquerade as model bugs.** fp16 overflow in a softmax or a loss that goes `NaN` after a few hundred steps is frequently a numerical-range problem, not a modeling one. Knowing fp16's ~65k ceiling and bf16's wider range lets you diagnose it instead of blaming the architecture.
- **Accidental copies blow up latency and memory.** A stray `np.ascontiguousarray`, a transpose fed into an op that needs contiguity, or converting a huge array's dtype can silently duplicate gigabytes. Understanding views-vs-copies keeps big pipelines from OOM-ing.

---

## Common misconceptions & failure modes

- **"An ndarray is a fast list."** No — it is a typed, fixed-size, contiguous buffer. Appending grows it by copying; mixing types upcasts silently (`np.array([1, 2.5])` becomes float64). It trades Python's flexibility for speed and memory density.
- **"Reshape/transpose copies the data."** Usually not — they return *views* sharing the buffer. This is fast but dangerous: mutating a view mutates the original. When you need independence, call `.copy()`.
- **"Broadcasting matches shapes left to right."** It matches **right to left** (right-aligned), padding the shorter shape with 1s on the *left*. Getting this backwards is why `(100,5) - (100,)` fails but `(100,5) - (5,)` works.
- **"`axis=0` means rows, so it operates on rows."** It *collapses* axis 0 — it operates *down columns*, producing one value per column. The axis you name is the one that *disappears*.
- **"fp16 is just fp32 with less precision."** It also has a drastically smaller *range* (~65k max). Overflow/underflow, not just rounding, is the failure mode — and it's why bf16 exists.
- **"A loop is fine, it's readable."** In hot numeric paths a Python loop is a 50–200× tax and blocks the eventual move to GPU. Vectorize.
- **"temp=0... " — wrong lecture, but same spirit: never assume; assert.** Print shapes and dtypes at boundaries until it's habit.

---

## Rules of thumb / cheat sheet

- **Every array is `shape + dtype + strided buffer`.** When confused, print all three: `x.shape, x.dtype, x.strides`.
- **Broadcasting:** right-align shapes, pad left with 1s; each dim must be equal or one is 1; size-1 dims stretch for free.
- **`axis=0` = down columns (per-column result); `axis=1` = across rows (per-row result). The named axis is removed.**
- **Use `keepdims=True`** whenever you reduce and then broadcast the result back against the original array.
- **Assert shapes after reductions and reshapes.** `assert out.shape == (B, D)` catches the silent axis bug.
- **Never loop over array elements in hot code.** Reach for vectorized ops, broadcasting, and `@` (matmul).
- **Memory math:** `bytes = element_count × bytes_per_element`. fp32=4, fp16/bf16=2, int8=1. A 7B model ≈ 14 GB at fp16 (approximate).
- **Default to float32 for ML on CPU; use bf16 on modern GPUs.** Remember fp16's ~65k range ceiling.
- **Reshape/slice/transpose = views (cheap, shared). Use `.copy()` for independence; expect a copy on dtype conversion and forced contiguity.**
- **Normalize once, then dot = cosine.** Batch-score with a single matmul, not a loop.

---

## Connect to the lab

This lecture is the theory behind Week 1 Lab exercise **4 (NumPy/pandas notebook)**: you'll create arrays and inspect `shape`/`dtype`, broadcast a `(3,1)` against a `(1,4)`, run `axis=0` vs `axis=1` reductions, and implement `cosine_similarity` with `np.dot(a,b)/(norm(a)*norm(b))` — the exact function you'll reuse for embeddings in Week 2 (`embed.py`) and again for `metrics.py`. Watch for two traps the lab quietly sets: the `axis=0`/`axis=1` mix-up in your group-by/aggregation (assert the result shape), and forgetting `keepdims=True` when normalizing rows of a document matrix. Also confirm your headless run (`jupyter execute`) proves no hidden state — a stale variable of the wrong shape is the same class of bug as the silent axis error.

---

## Going deeper (optional)

- **NumPy official docs** (numpy.org/doc) — read *"NumPy: the absolute basics for beginners"* and the *"Broadcasting"* page. These are canonical and short.
- **NumPy internals:** search *"NumPy strides tutorial"* and *"numpy as_strided"* to see how shape/strides create views; the concept transfers directly to PyTorch's `.stride()`.
- **PyTorch docs** (pytorch.org/docs) — the *"Tensor"* and *"Broadcasting semantics"* pages mirror NumPy almost exactly; skim them to see how the same model gains `.device` and autograd.
- **From Python to NumPy** by Nicolas P. Rougier (free online book) — search that title; excellent on vectorization patterns and the view/copy distinction.
- **Talk:** Jake VanderPlas, *"Losing your Loops: Fast Numerical Computing with NumPy"* (search the title on YouTube) — the definitive engineer-level case for vectorization.
- **Numerical precision:** search *"bfloat16 vs float16 deep learning"* and read the NVIDIA mixed-precision training docs for why bf16 dominates LLM work.

---

## Check yourself

1. An array has `shape=(3,4)` and `dtype=float32`. How many bytes is its buffer, and what does `.T` do to the buffer versus the metadata?
2. You have `X` of shape `(100, 5)` and want to subtract the per-row mean. Why does `X - X.mean(axis=1)` fail, and what's the one-word fix?
3. You call `X.sum(axis=0)` on a `(32, 768)` batch and get shape `(768,)`. What did you compute, and what would `axis=1` have given you instead?
4. Give two distinct reasons (beyond "C is faster") that `xs.sum()` beats a Python `for` loop over a million-element array.
5. A softmax over logits returns `NaN` in fp16 but works in fp32. What property of fp16 is the likely culprit, and which 2-byte dtype fixes it while staying 2 bytes?
6. After unit-normalizing two vectors, why does cosine similarity reduce to a plain dot product — and why does that let you score 10,000 documents with one matmul?

### Answer key

1. `3 × 4 × 4 = 48` bytes. `.T` leaves the buffer untouched and only swaps the strides (and shape) metadata, returning a view — no data is moved.
2. `X.mean(axis=1)` has shape `(100,)`, which right-aligns against the last axis of `(100,5)` — 100 vs 5 don't match, so broadcasting fails. Fix: `keepdims=True` (giving `(100,1)`, which broadcasts). One word: **keepdims**.
3. `axis=0` collapses the batch dimension, giving the **mean/sum per feature** — the aggregate over all 32 examples for each of the 768 dimensions. `axis=1` would collapse the feature dimension, giving one meaningless scalar **per example**, shape `(32,)` — the classic silent axis bug.
4. Any two of: (a) no Python interpreter/bytecode dispatch and no per-element object boxing; (b) contiguous memory streams through CPU cache with good locality; (c) SIMD vector instructions add multiple elements per instruction.
5. fp16's tiny dynamic range (~65504 max) — `exp()` of a large logit overflows to `inf`, poisoning the softmax to `NaN`. **bfloat16** fixes it: same 2 bytes but a wider exponent (larger range), trading mantissa precision for the range that prevents overflow.
6. Cosine = `(a·b)/(‖a‖‖b‖)`; if both vectors are already unit length the denominator is 1, so cosine equals the dot product. With all document vectors normalized into a `(10000, D)` matrix and the query normalized to `(D,)`, one `docs @ query` matmul computes all 10,000 dot products (= cosine scores) in a single optimized BLAS call — no per-document loop.
