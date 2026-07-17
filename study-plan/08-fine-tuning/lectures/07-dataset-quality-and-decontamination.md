# Lecture 7: Dataset Quality, Curation, and Decontamination

> Every fine-tuning tutorial spends 90% of its words on the training loop — the LoRA config, the learning rate, the VRAM math — and about two sentences on the data. That ratio is exactly backwards. In practice, once your training script runs at all, *the dataset is the model*. A clean 500-example set will beat a noisy 50,000-example set on the same task, on the same base model, every time — and it'll train in a tenth of the time and cost. This lecture is the engineering discipline of building that dataset: what "quality" means operationally (not philosophically), why small-and-clean dominates big-and-dirty, how to mechanically prove your eval set never leaked into training (decontamination), when synthetic data is a superpower versus a self-poisoning trap, and the hygiene checklist — dedup, format validation, label balance, a never-trained holdout — that separates a fine-tune you can trust from one that just *looks* trained. Everything is framed around the running task: routing support tickets into strict JSON. After this you can look at a raw data dump and turn it into a training set you'd stake a production deploy on.

**Prerequisites:** Lecture 2 (SFT mechanics, chat templates, the Week-1 disjointness assert), Phase 2 (Pydantic / structured outputs), Phase 7 (you have an eval harness), basic Python (`hashlib`, `json`, sets) · **Reading time:** ~30 min · **Part of:** Phase 08 (Model Adaptation & Fine-Tuning) Week 2

---

## The core idea (plain language)

Fine-tuning is imitation learning. The model does exactly one thing: it looks at your examples and learns to reproduce their patterns — including the patterns you didn't mean to teach it. This is the single fact that governs everything in this lecture. **The model cannot tell the difference between the signal you intended and the noise you left in.** If 3% of your ticket labels say `"priority": "hihg"` (a typo), the model dutifully learns to sometimes emit `"hihg"`. If half your `billing` examples are actually mislabeled `account` tickets, the model learns a blurry, wrong boundary between the two. If your training set is 80% `general` tickets because that's what your queue happened to contain last month, the model learns to answer `general` when in doubt — and quietly tanks on the rare-but-important `security` tickets.

So "quality" is not an aesthetic. It's four concrete, checkable properties:

1. **Clean** — no corrupt rows, no truncated JSON, no HTML boilerplate or signatures bleeding into the label, no duplicate garbage.
2. **Correctly formatted to the exact target schema** — every `assistant` turn is valid, parseable output in the *precise* shape you'll parse at inference. Not "close enough." Byte-for-byte the contract.
3. **Diverse across the input distribution** — the examples span the real range of inputs you'll see in production: short tickets and long ones, angry and polite, every category, edge phrasings, multiple languages if you have them.
4. **Consistent labels** — the same input always maps to the same output. Two near-identical tickets must not be labeled `billing` in one row and `account` in another, or you're teaching the model to flip a coin.

Why does 500 great examples beat 50k noisy ones? Because the model spends its limited fine-tuning capacity (especially under LoRA, where you're training <1% of the weights) fitting *whatever is consistent in the data*. Noise isn't consistent, so mostly it just dilutes the signal and burns steps — but the parts of the noise that *are* consistent (systematic mislabels, a recurring format bug) get learned as if they were the task. You are not "adding more data." On a dirty set you are actively teaching the model to be wrong in reproducible ways. Clean beats big because clean means *the only consistent thing in the data is the thing you want*.

**Decontamination** is the special case of "clean" that will silently destroy your ability to *know* whether the model is any good: if examples from your eval set (or a public benchmark) leak into training, your eval scores go up without the model getting better. You memorized the test. The fix is mechanical and cheap — hash every train row and every eval row, diff the two sets, assert they're disjoint — and it's the same disjointness assert you already wrote in Week 1, just taken seriously.

---

## How it actually works (mechanism, from first principles)

### Why noise is worse than "just less signal"

Naively you might think a noisy dataset is like a clean one with a lower effective sample size: 50k rows at 90% clean ≈ 45k good rows, still way more than 500. That intuition is wrong for two reasons.

**Reason 1: systematic noise is learned as signal.** Random noise (a label wrong in a different, uncorrelated way each time) mostly averages out — it slows learning but doesn't create a stable wrong behavior. But real datasets don't have random noise; they have *systematic* noise. Example: your ticket data came from an old rules-based classifier that always tagged anything containing the word "refund" as `billing`, even when it was a `fraud` report ("someone used my card, I want a refund"). That's not random — it's a consistent rule, and the model will learn it *perfectly*, because from the model's point of view it's indistinguishable from the real task. You've fine-tuned in a bug.

**Reason 2: capacity is finite and cheap to overwhelm.** LoRA on a 7B model trains maybe 20-40M parameters. On 500 examples for 2 epochs that's ~1000 gradient updates. The model does not have the room or the steps to "average out" 5000 mislabeled rows out of 50k — it fits the dominant patterns, and if a wrong pattern is dominant, that's what you get. More data past the point of covering your input distribution has *diminishing* returns; more *noise* has *negative* returns.

A quick numeric illustration. Suppose your true task boundary between `billing` and `account` is clean, but 15% of your `billing` examples are actually mislabeled `account`. The model can't see your intent; it only sees that "these billing-ish inputs map to `account` 15% of the time." The best it can do is reproduce that — so even a perfect learner tops out around **85% agreement with the true labels**, and in practice worse, because the mislabels cluster around the hard boundary cases (exactly where you need accuracy most). No amount of extra data fixes this. Fixing the 15% does.

### What "correctly formatted to the exact target schema" means mechanically

Your inference code will do something like `json.loads(model_output)` and then `TicketRoute(**parsed)` (Pydantic). Every training `assistant` turn must survive *that exact pipeline*. Common ways training data silently violates this:

- Trailing prose: `{"category": "billing", ...}\n\nLet me know if you need anything else!` — the JSON is valid up to `}` but `json.loads` on the whole string fails. The model learns to append chatter.
- Markdown fences: the label is wrapped in ` ```json ... ``` `. Now the model learns to emit fences, and your parser has to strip them forever.
- Key drift: some rows use `"prio"`, others `"priority"`. The model learns both, unpredictably.
- Enum drift: `"priority"` values are `"high"/"med"/"low"` in some rows and `"High"/"Medium"/"Low"` in others. Your downstream `if priority == "high"` check now misses a third of tickets.
- Extra/missing fields: some rows omit `needs_human`. The model learns it's optional; your parser (which requires it) throws at inference.

The rule: **the training label is a contract, and the contract is enforced by the same validator you use in production.** If Pydantic can't parse it, it doesn't go in the training set. No exceptions.

### The mechanism of contamination

Here is the failure, step by step, for the ticket router:

```
1. You have 1000 labeled tickets.
2. You shuffle and split: 900 train, 100 eval.
   -- but the shuffle happened AFTER a dedup pass that missed
      near-duplicates (same customer, same issue, filed twice).
3. Ticket #4471 ("can't log in, reset link expired") is in TRAIN.
   Ticket #4471b (same text, filed 2 min later) is in EVAL.
4. You fine-tune. The model MEMORIZES #4471 -> {"category":"account",...}.
5. At eval, #4471b comes in. The model recognizes it verbatim and
   emits the memorized answer. You score it CORRECT.
6. Your eval accuracy reads 94%. Production accuracy is 81%.
   The 13-point gap is memorized leakage you can't see.
```

The model didn't get better at *routing*; it got better at *recognizing rows it had already seen*. Your eval was supposed to measure generalization to unseen tickets, and you accidentally measured memorization. Every decision you make downstream — ship/no-ship, which base model, how many epochs — is now based on a number that lies.

Benchmark contamination is the same disease at internet scale: if your training data (especially synthetic or scraped data) happens to contain MMLU or IFEval questions, your retention scores in Week 4 are inflated and you'll think a forgetful model is fine.

**The mechanical fix (this is the whole thing):** hash the *content* of each train row and each eval row, put the hashes in two sets, and assert the intersection is empty.

```python
import hashlib, json

def row_fingerprint(row: dict) -> str:
    # Fingerprint the INPUT that determines the label.
    # For the ticket router that's the user message (+ system if it varies).
    user_msg = next(m["content"] for m in row["messages"] if m["role"] == "user")
    norm = " ".join(user_msg.lower().split())      # normalize whitespace/case
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

train_hashes = {row_fingerprint(r) for r in train_rows}
eval_hashes  = {row_fingerprint(r) for r in eval_rows}

overlap = train_hashes & eval_hashes
assert not overlap, f"CONTAMINATION: {len(overlap)} eval rows leaked into train"
```

Two engineering subtleties that matter:

- **Fingerprint the input, not the whole row.** Two tickets with identical text but different labels are a *label consistency* bug, but for contamination you care whether the model has seen this *input* before. Hash the thing the model conditions on.
- **Normalize before hashing**, or trivial differences (a trailing space, different casing) hide exact-duplicate leakage. `" ".join(s.lower().split())` collapses whitespace and case; it's crude but catches the common case. For near-duplicates (paraphrases), exact hashing isn't enough — see the dedup section.

---

## Worked example — cleaning the ticket-router dataset

You export 12,000 support tickets from Zendesk with their historical tags. You want a `{category, priority, needs_human}` JSON router. Here's the curation pass, with numbers.

**Step 0 — baseline.** 12,000 raw rows. Naive approach: dump all 12k into SFT. Let's instead curate.

**Step 1 — format validation.** Run every candidate label through the Pydantic model.

```python
from pydantic import BaseModel
from typing import Literal

class TicketRoute(BaseModel):
    category: Literal["billing","account","technical","fraud","general"]
    priority: Literal["high","medium","low"]
    needs_human: bool

good, bad = [], []
for r in rows:
    try:
        TicketRoute(**json.loads(r["label"]))
        good.append(r)
    except Exception as e:
        bad.append((r, str(e)))
```

Result (illustrative): **1,140 rows fail** — enum casing (`"High"`), a legacy `"acct"` category, missing `needs_human`, three rows with truncated JSON. You now have 10,860. Crucially, you *look at the failures* — the `"acct"` ones reveal an old category you can either remap or drop. Don't silently discard; understand.

**Step 2 — deduplication.** Exact-hash the normalized user text. **2,300 exact duplicates** (customers re-filing, bot-generated tickets). Keep one per hash → 8,560 rows. Then near-dup detection (MinHash or embedding cosine > 0.95) finds another **900 near-duplicates** → 7,660. Dedup matters twice: it stops the model over-weighting whatever gets filed most often, *and* it's a prerequisite for honest decontamination (near-dups across the split are the #1 leakage source).

**Step 3 — label balance.** Count categories:

```
technical  4,900  (64%)
billing    1,500  (20%)
account      800  (10%)
general      380  ( 5%)
fraud         80  ( 1%)   <- the one you MOST need to get right
```

`fraud` is 1% of the data and the highest-stakes label. If you train on this raw distribution, the model learns "fraud is rare, when unsure guess technical," and fraud recall craters. Fixes: **downsample technical** to ~1,500 and **over-sample or actively source more fraud** examples (this is a great place for careful synthetic data — see next section). You don't need perfectly uniform classes, but you cannot let a critical class sit at 1%.

**Step 4 — consistency audit.** Cluster near-identical inputs and check their labels agree. You find 60 cases where nearly the same ticket is tagged `billing` in one row and `account` in another. Pick a rule ("payment-method changes = account, charges/refunds = billing"), fix them, and — importantly — **write the rule into your labeling guide** so it stays fixed.

**Step 5 — the never-trained holdout.** *Before* any more work, carve off a stratified eval split — say 300 rows, sampled to include real proportions of every category *including* `fraud` (over-sample fraud in eval too, so you have enough to measure recall). Lock it. It never enters training, ever, not even for "just one more epoch."

**Step 6 — decontaminate.** Run the hash-and-diff assert between your final train set and the holdout. It catches 4 rows that survived dedup and leaked. Remove them from train.

**Final tally:** from 12,000 raw rows you ship maybe **~2,000 clean, balanced, validated, decontaminated training rows + 300 locked eval rows.** You *threw away 80% of your data* and your model will be *better* for it. That sentence is the whole lecture.

---

## Synthetic data — the superpower and the trap

Synthetic data (using an LLM to generate training examples) is how you fixed the `fraud` shortage above, and in 2025-2026 it's a mainstream, legitimate technique. But it has one catastrophic failure mode you must engineer around.

**When it's fine — even great:**
- **Filling rare classes and edge cases.** You have 80 real `fraud` tickets; you prompt a *stronger* model to generate 400 more realistic, varied fraud scenarios. This is a form of distillation and it's one of the best uses of your budget.
- **Format/schema demonstrations.** Generating many `(ticket → exact JSON)` pairs to nail the output contract.
- **Augmenting phrasing diversity.** Paraphrasing existing tickets into angry/terse/multilingual variants to broaden the input distribution.

**The trap — model collapse.** If you train a model on its *own* outputs, then use that model to generate the next batch, and repeat, you get a feedback loop that quietly destroys diversity: the outputs converge on a few templated phrasings, the tails (rare categories, unusual phrasings) vanish, and real-world accuracy degrades — *while your offline metrics on the equally-collapsed eval set still look fine.* This is documented as "model collapse" / "the curse of recursion." The mechanism is simple: a model's outputs are a *narrower* distribution than its training data (it favors high-probability, typical completions), so each generation-then-train cycle shrinks the distribution further. Diversity is not self-healing; it only leaks out.

**The three lightweight verification patterns that keep synthetic data safe** — apply them *before* any generated row reaches `train.jsonl`:

1. **Schema validation (free, catches ~everything malformed).** Run every generated label through the same Pydantic `TicketRoute`. Rejects: malformed JSON, wrong enums, missing fields. This alone filters a large fraction of bad synthetic output.
2. **Rule checks (task invariants).** Cheap deterministic assertions that encode domain rules: e.g. "a ticket mentioning unauthorized charges must not be `general`," "`needs_human=true` whenever `priority=high` and `category=fraud`." Any generated row that violates an invariant is dropped or flagged. These catch *plausible-but-wrong* outputs that pass schema validation.
3. **Stronger-model review (sample, not all).** Have a stronger model (or a human) review a sample of generated rows for label correctness and realism. You don't need to review all 400 — review 40, measure the accept rate; if it's <~90%, your generation prompt is broken, fix it before trusting the batch.

And the golden rule that prevents collapse outright: **never close the loop on a model's own unfiltered outputs.** Generate with a *stronger* model than the one you're training (or a diverse mix), always verify+filter, and always anchor to *real* data — keep genuine human-labeled examples as the backbone and use synthetic data to fill gaps, not to replace the real distribution.

```
generate (stronger model) --> schema validate --> rule checks
   --> stronger-model review (sample) --> filter --> mix with REAL data --> train
   [never: train_model -> generate -> train_same_model -> repeat, unfiltered]
```

---

## How it shows up in production

- **The "great eval, sad users" gap.** Your fine-tune scored 94% offline and users complain routing is wrong. Root cause nine times out of ten: contamination (memorized eval rows) or an eval set that doesn't match production distribution (all your eval `fraud` cases were easy ones). The fix is upstream, in the data, not in the training config — but people waste days re-tuning hyperparameters first.

- **The schema-drift incident.** Two weeks after deploy, your ticket pipeline starts throwing `json.loads` errors on ~2% of outputs. Cause: a slice of training labels had markdown fences or trailing prose, so the model learned to *sometimes* emit them. Every one of those is a dropped ticket or a crash in production. Format validation at curation time would have caught 100% of it for free.

- **Silent minority-class failure.** The model works great in the demo (which used common tickets) and fails on the 1% that matters. Fraud tickets get routed to `general`, sit in a slow queue, and now it's a security-and-PR problem. Label imbalance you didn't correct becomes an incident.

- **Cost and time.** Training on 50k noisy rows vs 2k clean rows is ~25× the GPU time and cost for a *worse* model. On a rented L4 at ~$0.80/hr, that's the difference between a 15-minute run and a multi-hour one, per experiment, times every iteration you do. Clean data is also *faster to iterate on* — you can re-run the whole loop over lunch.

- **Model collapse from a synthetic-data flywheel.** A team bootstraps training data by generating tickets with their own model, trains on them unfiltered, uses the new model to generate the next batch, and repeats. Diversity quietly collapses — outputs converge to a few templated phrasings, rare categories vanish, real accuracy degrades even though every offline metric on the (equally collapsed) eval set looks fine.

---

## Common misconceptions & failure modes

- **"More data is always better."** False for fine-tuning. Past the point of covering your input distribution, extra *clean* data has diminishing returns and extra *noisy* data has negative returns. Quantity is a vanity metric.

- **"The model will average out the bad labels."** Only for *random* noise. Systematic errors (a legacy rule, a consistent mislabel) are learned as if they were the task. Real datasets are full of systematic noise.

- **"Synthetic data is cheating / synthetic data is free lunch."** Neither. Synthetic data is *fine and often excellent* — for rare classes, hard edge cases, and format demos — **if and only if** it passes verification (schema + rules + optionally a stronger-model review) and you don't feed a model its own unfiltered outputs in a loop. Verified synthetic data from a *stronger* model (distillation) is one of the best tools you have. Unverified self-generated data is how you get model collapse.

- **"I split randomly, so I'm decontaminated."** A random split does *not* protect against near-duplicates crossing the boundary. You must dedup *before* splitting and assert disjointness *after*. Random shuffling and decontamination are different guarantees.

- **"The holdout is for the final check."** No — the holdout must be carved out *first* and touched *never*, until you report the number once. Every time you tune to it, look at it, or accidentally train on it, it stops measuring generalization. Treat it like a sealed evidence bag.

- **"Balanced means exactly uniform classes."** No. It means no critical class is starved and the model isn't dominated by one majority class. Match production proportions where you can, but floor the rare-and-important classes so they have enough signal (and enough eval examples to measure).

- **"Format is close enough; I'll clean it at inference."** Every post-hoc cleanup (stripping fences, regex-extracting JSON) is a permanent tax and a permanent failure mode. Fix the contract in the data.

---

## Rules of thumb / cheat sheet

*(All numbers approximate, task-dependent — calibrate to your own eval.)*

- **Start with ~500-2,000 clean examples**, not 50k noisy ones. Only scale up after clean data plateaus on your eval.
- **Validate every label through your production parser** (Pydantic). Fail-closed: if it doesn't parse, it doesn't train.
- **Dedup before splitting.** Exact hash on normalized text first; then near-dup (embedding cosine > ~0.9-0.95 or MinHash) for paraphrases.
- **Decontaminate with hash-and-diff.** `assert not (train_hashes & eval_hashes)`. Fingerprint the *input*, normalized. Same assert as Week 1 — take it seriously.
- **Carve the eval holdout first, freeze it forever.** Stratify by class; over-sample rare-critical classes so you can actually measure their recall.
- **Floor your minority classes.** Don't let a high-stakes label sit at 1%. Downsample the majority and/or source more minority examples.
- **Synthetic data: generate → verify → filter → *then* train.** Verify with (1) schema validation, (2) rule checks, (3) stronger-model review on a sample. Never train on a model's own unfiltered outputs in a loop; keep real data as the backbone.
- **Keep a labeling guide.** Every consistency fix becomes a written rule so it doesn't regress next batch.
- **Diversity check:** eyeball a random 30 rows. If they feel same-y, your model will be brittle. Ensure length, tone, phrasing, and category spread.
- **When in doubt, read 50 random rows by hand.** No metric replaces looking at your data.

---

## Connect to the lab

This lecture is the discipline behind the Week 2 lab's dataset prep (and it retro-justifies the Week 1 `prepare_dataset.py` disjointness assert). When you build your QLoRA training set: run every `assistant` label through your Pydantic `TicketRoute` model and drop failures; dedup on normalized user text; check and floor your category balance (especially the rare critical class); carve and freeze the stratified eval holdout; and add the `train_hashes & eval_hashes` empty-intersection assert to `prepare_dataset.py` so a contaminated set *fails the build*. If you use synthetic data to fill out rare categories, gate it behind schema + rule + stronger-model verification before it ever reaches `train.jsonl`.

## Going deeper (optional)

- **Hugging Face `datasets` docs** (huggingface.co/docs/datasets) — dedup, filtering, and mapping pipelines you'll actually use.
- **Pydantic docs** (docs.pydantic.dev) — `Literal` enums and validators are your format contract.
- **"Textbooks Are All You Need" (Microsoft, phi models)** — the canonical demonstration that curated, high-quality data lets small models punch far above their weight. Search: *"Textbooks Are All You Need phi paper"*.
- **"The Curse of Recursion: Training on Generated Data Makes Models Forget" (Shumailov et al.)** — the model-collapse paper. Search: *"curse of recursion model collapse"*.
- **LIMA: "Less Is More for Alignment" (Meta)** — the empirical case that a small, high-quality SFT set can align a strong base model. Search: *"LIMA less is more for alignment"*.
- **datasketch (MinHash/LSH) library** — practical near-duplicate detection at scale. Search: *"datasketch MinHash LSH deduplication"*.
- Search queries: *"LLM training data decontamination methodology"*, *"n-gram overlap benchmark contamination detection"*, *"synthetic data verification pipeline LLM"*.

## Check yourself

1. You have 50,000 training rows and a colleague suggests adding 100,000 more scraped ones to "boost accuracy." Under what condition does this *hurt*, and why can't the model just average out the bad rows?
2. Your fine-tune scores 93% offline but ~80% in production. Name the two most likely data-side causes and the one-line check for each.
3. Write (in words or code) the mechanical decontamination check between train and eval. Why do you fingerprint the *input* and normalize it before hashing?
4. When is synthetic data a good idea, and what exactly turns it into "model collapse"? Name two lightweight verification steps that prevent it.
5. Your `fraud` class is 1% of the data. Why is training on that raw distribution dangerous, and what two curation moves fix it?
6. Why must the eval holdout be carved out *first* and never touched, rather than sampled at the end?

### Answer key

1. It hurts whenever the new data is noisier or systematically mislabeled than the existing set, or shifts the distribution away from production. The model can't average out *systematic* noise — a consistent wrong pattern (e.g., a legacy rule that mislabels a class) is indistinguishable from real signal, so it's learned as the task. Random noise averages out; systematic noise gets fit. Extra clean data has diminishing returns; extra noisy data has negative returns.

2. (a) **Contamination** — eval rows (or near-duplicates of them) leaked into train, so the model memorized them and the offline score is inflated. Check: hash-and-diff train vs eval, assert empty intersection, and run near-dup detection across the split. (b) **Distribution mismatch** — the eval set isn't representative of production (e.g., only easy cases). Check: compare class proportions and input characteristics of eval vs a fresh production sample; read 30 real failures by hand.

3. Fingerprint each row's *input* (the user message that determines the label), normalize it (`" ".join(s.lower().split())` to collapse case/whitespace), SHA-256 it, put train and eval hashes in two sets, and `assert not (train_hashes & eval_hashes)`. You hash the *input* because contamination is about whether the model has *seen this input before* (memorization); you normalize so trivial formatting differences don't hide an otherwise-exact duplicate. (Exact hashing still misses paraphrases — pair it with near-dup detection.)

4. Synthetic data is good for covering rare classes, hard edge cases, and demonstrating format — especially when generated by a *stronger* model (distillation). It becomes model collapse when you train a model on its *own unfiltered outputs* in a repeating loop: diversity narrows, rare cases vanish, quality degrades while collapsed metrics still look fine. Prevent it with: (1) schema validation (drop anything your parser rejects), (2) rule checks (reject outputs that violate task invariants), and/or (3) a stronger-model review on a sample. Generate → verify → filter → then train, always anchored to real data.

5. At 1% the model learns "this class is rare; when unsure, guess the majority," so recall on the high-stakes class craters — and you have too few eval examples to even notice. Fixes: **downsample the majority class** and **source or (carefully, verified) synthesize more minority examples**, so the critical class has enough training signal and enough eval examples to measure recall.

6. Because every interaction with the holdout leaks information into your decisions. If you sample it at the end after tuning against it, or peek and adjust, you've implicitly fit to it and it no longer measures generalization to unseen data. Carving it out first and freezing it (stratified, over-sampling rare classes) keeps it a clean, one-shot measurement of real-world performance — a sealed evidence bag, not a scratchpad.
