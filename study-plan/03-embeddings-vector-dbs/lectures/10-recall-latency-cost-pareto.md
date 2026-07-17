# Lecture 10: The Recall/Latency/Cost Triangle and Pareto Analysis

> You've now built HNSW, IVF-Flat, IVF-PQ, and binary+rescore indexes and watched each knob move recall, speed, and memory in its own direction. This lecture is the synthesis: it turns that pile of individual observations into a *decision framework* you can defend in a design review. The central truth is uncomfortable and liberating at once — there is no single best index. Recall, latency, and cost trade against each other, and all any engineer can do is find the cheapest point on a frontier that still meets the service-level objective (SLO). After this lecture you'll be able to read a recall-vs-QPS scatter with RAM encoded as size/color, extract the Pareto frontier from a raw `sweep.csv`, work an SLO problem end to end ("recall@10 ≥ 0.95 at ≥ 500 QPS, minimize RAM"), and — crucially — refuse to ship a config that looks great on the plot but takes 40 minutes to build and 40 GB of RAM to serve.

**Prerequisites:** Lectures 6–9 (exact vs approximate search and honest recall; HNSW; IVF; PQ/quantization); big-O intuition; the fact that recall must always be measured against a flat ground truth · **Reading time:** ~28 min · **Part of:** Phase 3 — Embeddings Infrastructure & Vector Databases, Week 2

---

## The core idea (plain language)

Every index you built last week is a point in a three-dimensional space:

- **Recall** — what fraction of the true top-k neighbors the index actually returned. (Measured against a flat index. Always.)
- **Latency / throughput** — how fast you can answer, usually expressed as QPS (queries per second) on a fixed machine, or as p50/p95 latency.
- **Cost** — RAM per vector and dollars per month to serve it, *plus* build time, which is a one-time cost that nonetheless kills deployments.

The uncomfortable fact — the **no-free-lunch principle** for vector search — is that you cannot maximize all three at once. Push recall up and you either burn more CPU (lower QPS) or more RAM. Shrink RAM with quantization and recall drops unless you spend latency rescoring. Crank QPS with an aggressively pruned graph and recall sags. Nobody has an index that is simultaneously the most accurate, the fastest, and the cheapest. The knobs *are* the trade.

So "what's the best index?" is the wrong question. It has no answer, the same way "what's the best car?" has no answer — a pickup truck and a sports car are both optimal, for different owners. The right question is: **given my SLO, what is the cheapest configuration that satisfies it?** That reframes index selection from an argument about which algorithm is "better" into a constrained optimization: filter to the configs that meet the requirement, then minimize the resource you care about. This lecture teaches you to answer that question with a plot and a spreadsheet instead of a vibe.

The tool for reasoning about "you can't win on all axes at once" is the **Pareto frontier** — the set of configurations where you can't improve one axis without giving up another. Everything not on the frontier is strictly worse than something on it and should be deleted from consideration. Your entire job in the Week 2 lab reduces to: build the frontier, then walk it to the cheapest feasible point.

---

## How it actually works (mechanism, from first principles)

### Pareto dominance, precisely

Config **A dominates** config **B** if A is at least as good as B on *every* axis and strictly better on *at least one*. "Good" means higher recall, higher QPS, lower RAM, lower build time. If A dominates B, there is never a reason to ship B — A is better or equal everywhere. B is *dominated* and can be thrown away.

A config that is **not dominated by anything** is **Pareto-optimal**. The collection of all Pareto-optimal configs is the **Pareto frontier**. Reading it: for any point on the frontier, to get more recall you must pay with QPS or RAM; there's no free improvement left. That's exactly the no-free-lunch principle drawn as a curve.

Worked micro-example. Suppose you have five configs (ignore RAM for a second, just recall and QPS):

```
config   recall@10   QPS
  A        0.90       2000
  B        0.95       1200
  C        0.97        400
  D        0.93        300    <-- dominated
  E        0.88        900    <-- dominated
```

- **D** (0.93 / 300): B has higher recall (0.95 > 0.93) *and* higher QPS (1200 > 300). B dominates D. Drop D.
- **E** (0.88 / 900): A has higher recall (0.90 > 0.88) *and* higher QPS (2000 > 900). A dominates E. Drop E.
- **A, B, C** — none dominates another. A is fastest but least accurate; C is most accurate but slowest; B is in between. That's the frontier: `{A, B, C}`.

Notice the frontier has a *shape*: as you move to higher recall, QPS falls. That downward-sloping staircase is the signature of a real trade. If your plot shows points that are better on both axes than others, you simply haven't cleaned it up yet — those dominated points are noise.

### The third axis: reading RAM off a 2-D scatter

We can only draw two axes cleanly. Standard practice — the one `ann-benchmarks` popularized — is **recall on the Y axis, QPS on a log X axis**, and encode the third variable (RAM, or build time) as **point size and/or color**. Bigger/redder = more RAM.

```
recall@10
  1.00 |                       (o)  <- IVF-Flat: high recall, big RAM, low QPS
  0.98 |                  O          O = large point (fat RAM)
  0.96 |              O        o     o = small point (lean RAM)
  0.94 |         o                 .
  0.92 |     .        .              . = tiny point (quantized, tiny RAM)
  0.90 |  .                    .
       +----------------------------------> QPS (log scale)
        100    300   1000   3000  10000
```

Why **log-scale QPS**? Because index families differ by orders of magnitude, not percentages. A brute-force flat index might do 50 QPS while a tuned HNSW does 8,000. On a linear axis the slow configs bunch into a vertical smear at the left edge and you can't see the frontier's shape. Log scale spreads them out so equal *ratios* get equal visual distance — which is what you actually reason about ("this config is 3× faster") — and turns the frontier into a readable curve.

### Where each index family lives on the plot

This is the payoff of Lectures 7–9: each family occupies a *characteristic region*, and once you internalize the map you can predict where a config will land before you run it.

- **HNSW** — the top-right workhorse. High recall *and* high QPS, because graph traversal is cheap and lands you near the answer in ~log hops. The price is **RAM**: it's fully memory-resident and stores `M×2` neighbor links per vector on top of the raw fp32 vector. On the plot: large points (fat RAM) clustered high and to the right. Sweeping `efSearch` slides a single HNSW build along a short recall-vs-QPS arc.
- **IVF-Flat** — high recall achievable (it stores full vectors, just partitioned into cells), but lower QPS than HNSW at equal recall because you scan every vector in the probed cells. Moderate-to-large RAM (full vectors + centroids). On the plot: upper region, middling-to-left on QPS, medium-large points. Sweeping `nprobe` moves it along its curve.
- **IVF-PQ** — the RAM saver. Product quantization compresses each vector to `m` bytes, so RAM drops by 8–32×. The cost is **recall** (lossy codes) and some CPU for asymmetric distance. On the plot: **tiny points** (lean RAM), sitting lower on recall unless you rescore, spread across a range of QPS. This is the family you reach for when RAM/$ is the binding constraint.
- **Binary + rescore** — the extreme RAM saver with a recovery trick. Store 1 bit/dim (32× smaller than fp32), search fast in Hamming space to get a candidate pool, then re-rank that pool with full-precision vectors. On the plot: very small points (bits are cheap) that can climb *back up* to high recall because the rescore step fixes the ranking of the top candidates. QPS depends on how many you rescore. It occupies a distinctive spot: near-HNSW recall at IVF-PQ-like RAM, if you keep enough raw vectors around for the rescore.

Mental map:

```
        RAM ->  small ................................ large
recall  high |  binary+rescore     IVF-Flat   HNSW
        med  |  IVF-PQ                         (HNSW low efSearch)
        low  |  IVF-PQ (low nprobe)
```

### The ann-benchmarks methodology

`github.com/erikbern/ann-benchmarks` is the reference for *how* to present this honestly, and the Week 2 lab deliberately reproduces its shape. The methodology, distilled:

1. **Standardized recall-vs-QPS curves.** One plot, recall on Y, QPS on log X.
2. **One curve per index family, swept over its runtime knob.** You don't plot a single dot per algorithm — you plot a *curve*, generated by sweeping the query-time parameter that trades recall for speed (`efSearch` for HNSW, `nprobe` for IVF). Each build produces an arc as you vary that knob. This is the fair way to compare: you're comparing the *whole trade-off envelope* of each family, not one arbitrarily-chosen operating point.
3. **Recall measured against exact ground truth.** Same denominator for everyone.
4. **Fixed hardware, warmed index, median over repeats.** QPS is meaningless without stating the machine and warming up (cold-cache and JIT/allocation spikes wreck the first query — take medians over ≥3 runs).

The reason to sweep a knob rather than pick one point: a family with a *better curve* (higher and further right everywhere) is genuinely better for that dataset. A family that only wins at one point is a narrower tool. The curve tells you the whole story.

---

## Worked example

You've run the Week 2 sweep. `results/sweep.csv` looks like this (numbers below are *illustrative* — plausible shapes, not measured; your real sweep will differ):

```csv
family,config,recall@10,qps,build_time_s,rss_mb
hnsw,M16_ef64,0.918,7200,95,1450
hnsw,M16_ef128,0.947,4800,95,1450
hnsw,M32_ef128,0.968,3900,180,2100
hnsw,M32_ef256,0.983,2100,180,2100
hnsw,M64_ef256,0.991,1500,410,3400
ivfpq,nlist1024_pq32_np8,0.872,5600,60,240
ivfpq,nlist1024_pq32_np16,0.913,3800,60,240
ivfpq,nlist1024_pq32_np32,0.951,2200,60,240
ivfpq,nlist1024_pq32_np64,0.974,1200,60,240
binrescore,top200,0.958,2600,70,180
binrescore,top500,0.981,1400,70,180
```

### Step 1 — turn it into a Pareto plot

Load, then draw recall (Y) vs QPS (log X), size/color by `rss_mb`, one series per family. In `plot.py`:

```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results/sweep.csv")
fig, ax = plt.subplots()
for fam, g in df.groupby("family"):
    ax.scatter(g.qps, g["recall@10"], s=g.rss_mb/10, label=fam, alpha=0.7)
ax.set_xscale("log"); ax.set_xlabel("QPS (log)"); ax.set_ylabel("recall@10")
ax.legend()
```

Point size `rss_mb/10` makes HNSW's 1.5–3.4 GB configs visibly fat and IVF-PQ/binary's ~200 MB configs tiny — the RAM story jumps out of the plot immediately.

### Step 2 — extract the frontier

The frontier here is the set of configs where you can't get more recall without losing QPS. A simple, correct extraction over the whole table (across families) — sort by QPS descending, keep a running max recall, a point stays only if it beats every faster point's recall:

```python
d = df.sort_values("qps", ascending=False)
best = -1; frontier = []
for _, r in d.iterrows():
    if r["recall@10"] > best:
        frontier.append(r); best = r["recall@10"]
```

Running that on the table: sorted by QPS we get `hnsw M16_ef64` (7200, 0.918) → keep; `ivfpq np8` (5600, 0.872) → dominated (lower recall, lower QPS) drop; `hnsw M16_ef128` (4800, 0.947) → keep; `ivfpq np16` (3800, 0.913) drop; `hnsw M32_ef128` (3900, 0.968) keep; ... the survivors are the staircase of increasing recall as QPS falls. Everything below the staircase is dominated and gone.

### Step 3 — apply the SLO: `recall@10 ≥ 0.95 at ≥ 500 QPS, minimize RAM`

Filter to **feasible** configs (both constraints met), then pick the **cheapest** on the resource we're told to minimize (RAM):

```python
feasible = df[(df["recall@10"] >= 0.95) & (df.qps >= 500)]
pick = feasible.sort_values("rss_mb").iloc[0]
```

Who's feasible?

```
family      config         recall  qps    rss_mb
hnsw        M32_ef128      0.968    3900   2100
hnsw        M32_ef256      0.983    2100   2100
hnsw        M64_ef256      0.991    1500   3400
ivfpq       np32           0.951    2200    240
ivfpq       np64           0.974    1200    240
binrescore  top200         0.958    2600    180   <-- min RAM
binrescore  top500         0.981    1400    180
```

(`hnsw M16_ef128` at 0.947 misses the recall floor by 0.003 and is excluded — respect the constraint.) Sorting the feasible set by RAM, the winner is **`binrescore top200`**: 0.958 recall, 2600 QPS, **180 MB**. IVF-PQ (`np32`) is a close second at 240 MB.

### Step 4 — justify in 3–5 sentences

> I'd ship **binary-quantization + rescore (top-200)**. It clears the SLO with margin — recall@10 = 0.958 (≥ 0.95) at 2,600 QPS (5× the 500 floor) — while using **180 MB RAM, roughly 12× less than the HNSW configs (2.1–3.4 GB)** that hit similar recall. The RAM saving is the whole point: since the SLO says *minimize RAM* and every feasible option meets recall and QPS comfortably, the tiebreak is memory, and binary+rescore wins decisively. Its build time (~70 s) is also modest, so it's operationally cheap to rebuild. I'd keep IVF-PQ (`np32`, 240 MB) as the fallback if the rescore step's dependence on keeping full vectors around proves awkward, and I'd *reject* `hnsw M64_ef256` outright despite its 0.991 recall — that recall is above SLO (we don't get paid for it) and it costs 19× the RAM for no additional value under this objective.

That last sentence is the mindset the whole lecture is trying to install: **recall above the SLO is waste you pay for in RAM and dollars.** Meeting the bar cheaply beats exceeding it expensively.

---

## How it shows up in production

**The frontier is your negotiation table.** When a PM says "make search more accurate," the honest engineering answer is a frontier plot: "+2 points of recall costs us either 40% of our QPS or 2× the RAM — here's the curve, pick the point and I'll tell you the bill." The trade becomes a business decision instead of an argument, and you stop being asked for physically impossible things.

**Build time is a deployment killer that never shows on the recall/QPS plot.** A config that takes **40 minutes to build** cannot be part of a fast rollback, blocks blue-green re-embedding for 40 minutes per collection, and turns "just rebuild after the churn" into an outage. If your data changes hourly and your index takes 40 minutes to build, you have an architecture problem no amount of recall fixes. Always carry build time as a column even though it doesn't fit on the 2-D plot — encode it as an annotation or a second panel.

**RAM is the silent monthly invoice.** HNSW is memory-resident. A 10M × 768-dim fp32 corpus with `M=32` is roughly `10e6 × (768×4 + 32×2×4)` bytes ≈ `10e6 × 3328` ≈ **33 GB** just for the index — that's a large, expensive instance, per replica, forever. The same corpus under IVF-PQ (`m=48` bytes/vector + overhead) can drop under **1 GB**. On the plot those are a fat point and a tiny point at similar recall; on the invoice they're a `r6i.2xlarge` versus a `t3.large`. The "un-shippable great config" is real: **40-minute build, 40 GB RAM** describes a config that dominates the recall/QPS plot and still gets vetoed in the design review.

**QPS numbers lie if you don't state the machine and warm the index.** A benchmark of "8,000 QPS" means nothing without cores, thread count, batch vs single-query, and a warmup. The first query after load pays JIT, page-fault, and allocation costs — it can be 10× slower. Take medians over repeats, warm the index, and pin the hardware. Comparing a cold single-thread run against a competitor's warm 16-thread run is how vendors win benchmarks and how engineers ship regressions.

**Recall you don't need is money you're burning.** The single most common overspend is targeting 0.99 recall when the product is fine at 0.95. Those last few points sit on the steepest part of the frontier — you pay disproportionately in QPS and RAM for them. Set the SLO from the *product* (does search feel good in an A/B test?), not from a desire to see a big number.

---

## Common misconceptions & failure modes

- **"HNSW is just the best index."** It's the best *region* for high-recall/high-QPS when RAM is available. When RAM or build time is the binding constraint — huge corpora, tight budgets, frequent rebuilds — IVF-PQ or binary+rescore dominate the shippable frontier. "Best" is meaningless without the SLO.
- **Comparing single operating points instead of curves.** Cherry-picking one `efSearch` for HNSW and one `nprobe` for IVF and declaring a winner is how blog posts mislead. Sweep the knob; compare curves.
- **Linear QPS axis.** The slow configs collapse into a vertical smear at the origin and the frontier's shape disappears. Log-scale the QPS axis or you can't read your own plot.
- **Chasing recall past the SLO.** Every point above the required recall is paid-for waste unless the product measurably improves. Meet the bar cheaply.
- **Ignoring the third axis.** A recall/QPS plot with no RAM encoding hides the reason a config is un-shippable. Always encode RAM (size/color) and carry build time as a column.
- **Reporting recall against another ANN index.** Then you've measured *agreement*, not recall. The denominator is always a flat ground-truth index (Lecture 6). A frontier built on the wrong denominator is fiction.
- **Cold / single-query QPS.** No warmup, no medians → the first-query allocation spike poisons the number. Warm, repeat, take the median.
- **Treating a dominated config as a "cheaper option."** If it's dominated, something is better on every axis. It's not a trade-off, it's a mistake. Delete it.

---

## Rules of thumb / cheat sheet

- **Reframe the question.** Not "best index" — "cheapest config that meets the SLO." Filter to feasible, then minimize the constrained resource.
- **Dominance test.** A dominates B iff A ≥ B on every axis and > on at least one. Dominated configs get deleted, not shipped.
- **Plot convention.** Recall on Y, QPS on **log** X, RAM as point size/color, one **curve per family** swept over its runtime knob. This is the ann-benchmarks shape; reproduce it.
- **Family map (approximate):** HNSW = high recall + high QPS + *fat RAM*; IVF-Flat = high recall + medium QPS + medium-large RAM; IVF-PQ = tiny RAM + recall hit; binary+rescore = tiny RAM + recall recovered via re-ranking the candidate pool.
- **Runtime knobs to sweep:** HNSW → `efSearch`; IVF(-PQ) → `nprobe`. Build-time knobs (`M`, `efConstruction`, `nlist`, PQ `m`) change *which curve* you get.
- **Carry four columns always:** recall@k, QPS, RAM (RSS), build_time. The last two don't fit the 2-D plot but veto configs in review.
- **Napkin RAM for HNSW:** `N × (dim×4 + M×2×4)` bytes. 10M × 768d × M32 ≈ 33 GB. Compute this *before* you promise a machine size.
- **Don't buy recall above the SLO.** It sits on the steepest part of the frontier and pays nothing back.
- **Un-shippable smell test:** if build time > your data's change interval, or RAM > your budget instance, the config is out no matter how good the recall/QPS looks.

---

## Connect to the lab

This lecture *is* the frame for the Week 2 lab deliverable. Your `sweep.py` produces `results/sweep.csv` with recall@10, QPS, build_time, and RSS across ≥15 HNSW and ≥6 IVF-PQ configs plus binary+rescore; `plot.py` turns it into `pareto.png` (recall Y, QPS log-X, RAM as size/color, one series per family with the frontier drawn). The written finding — "which config I'd ship for recall@10 ≥ 0.95 at ≥ 500 QPS and why, in 3–5 sentences" — is exactly the Step 3–4 exercise above, run on *your* numbers. Bring build time and RAM into the recommendation explicitly; a finding that names a 40-minute/40-GB config without flagging it as un-shippable misses the whole point of the week.

---

## Going deeper (optional)

- **ann-benchmarks** — `github.com/erikbern/ann-benchmarks`. The canonical methodology and result plots. Read the README and study the shape of the recall-vs-QPS curves; that's the target for your own plot.
- **FAISS wiki** — on the `facebookresearch/faiss` GitHub repo: "Guidelines to choose an index" and "Indexing 1M vectors." The best practical map from dataset size/RAM budget to index choice.
- **Qdrant benchmarks** — the vendor benchmark pages (`qdrant.tech`) are a good example of *how* filtered-ANN results are presented, and a lesson in reading vendor numbers skeptically (check the hardware and whether recall is stated).
- **"Product Quantization for Nearest Neighbor Search" (Jégou, Douze, Schmid, 2011)** — the origin of PQ; skim for intuition on the memory/recall trade, not the proofs.
- Search queries when you want current material: **"ann-benchmarks recall qps tradeoff"**, **"FAISS index selection guidelines"**, **"HNSW memory usage formula"**, **"binary quantization rescore recall recovery vector search"**, **"pareto frontier vector index benchmark"**.

---

## Check yourself

1. Define Pareto dominance in one sentence, and explain why every dominated config should be deleted from your candidate set.
2. Why is the QPS axis plotted on a log scale in ann-benchmarks-style charts? What goes wrong on a linear axis?
3. On a recall-vs-QPS scatter with RAM encoded as point size, describe roughly where you'd expect HNSW, IVF-PQ, and binary+rescore points to sit, and why.
4. Your SLO is "recall@10 ≥ 0.95 at ≥ 500 QPS, minimize RAM." You have a config at 0.991 recall / 1500 QPS / 3.4 GB and another at 0.958 / 2600 QPS / 180 MB. Which do you ship, and what's wrong with picking the first?
5. Why must recall on the plot be measured against a flat index and not against your best HNSW config?
6. Give two reasons a config that dominates the recall/QPS plot might still be un-shippable.

### Answer key

1. A dominates B if A is at least as good as B on every axis (recall, QPS, RAM, build time) and strictly better on at least one. A dominated config is beaten everywhere by something else, so there's never a reason to choose it — keeping it around only adds noise to the decision.
2. Index families differ in QPS by orders of magnitude, not percentages. On a linear axis the slow configs bunch into a vertical smear at the left edge and the frontier's shape is invisible; log scale gives equal ratios equal visual spacing, which matches how you actually reason ("3× faster") and makes the frontier a readable curve.
3. **HNSW**: top-right (high recall, high QPS) with **large** points — it's memory-resident and stores neighbor links per vector, so RAM is fat. **IVF-PQ**: **tiny** points (compressed codes = little RAM), sitting lower on recall unless rescored, spread across a QPS range. **Binary+rescore**: very small points (1 bit/dim) that climb back up to high recall because re-ranking the candidate pool with full vectors fixes the top-k ordering.
4. Ship the **second** (0.958 / 2600 QPS / 180 MB). Both meet the recall and QPS constraints, so under "minimize RAM" the 180 MB config wins by ~19×. The first config's extra recall (0.991) is *above the SLO* — you don't get paid for it, and you'd be spending 19× the RAM to buy accuracy the product doesn't require. Recall above the SLO is waste.
5. Recall means "fraction of the *true* top-k I returned." The true top-k comes from exact/flat search. Measuring against another ANN index only tells you how much the two approximate indexes *agree* — both could be missing the same true neighbors, so you'd report a high number while actually being wrong. The flat index is the only honest denominator.
6. (a) **Build time** — e.g. 40 minutes — blocks fast rollback and blue-green re-embedding, and can't keep up with frequently-changing data. (b) **RAM** — e.g. 40 GB per replica — may exceed the budgeted instance size, making the config too expensive to serve even though its recall/QPS look great. Neither appears on the 2-D recall/QPS plot, which is why you carry them as separate columns.
