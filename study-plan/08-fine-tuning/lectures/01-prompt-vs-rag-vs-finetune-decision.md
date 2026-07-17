# Lecture 1: The Adaptation Decision Framework — Prompt vs RAG vs Fine-Tune

> Fine-tuning is the single most over-reached-for tool in the applied-LLM toolbox. An engineer sees "the model got it wrong," reaches for training weights, burns two weeks and a GPU budget, and ships something that a fifteen-minute prompt edit or a retrieval layer would have beaten — cheaper, faster, and without a pipeline to babysit forever. This lecture gives you a *decision procedure*, not a vibe. By the end you will be able to map any task to prompt, RAG, or fine-tune; explain in one breath why fine-tuning teaches *behavior* but not *facts*; run the cost/latency arithmetic that justifies (or kills) a tune; and fill in a one-page decision note that survives review from a skeptical staff engineer.

**Prerequisites:** a working prompt baseline and a RAG baseline (Phase 4), an eval harness with a golden set + LLM judge (Phase 7), comfort reading token counts and per-1k-token pricing · **Reading time:** ~24 min · **Part of:** Phase 08 — Model Adaptation & Fine-Tuning, Week 1

## The core idea (plain language)

There are exactly three levers you can pull to make a general model do your specific job, and they act at three different layers of the system:

1. **Prompt** — you change the *input text* at inference time. Zero infra, instant iteration, no training run. This is the default. You are "programming" the model with words.
2. **RAG (retrieval-augmented generation)** — you change *what facts are in the context* at inference time, by fetching relevant documents and pasting them in. This grounds the model in current, private, or large knowledge it never memorized. Still no weight changes.
3. **Fine-tune** — you change the *weights themselves* by training on examples. This bakes a *style of answering* into the model permanently. It is the only lever that changes the model's default behavior when you give it *no* special instructions.

The single rule that governs the entire framework:

> **Prompt first. RAG for knowledge and facts. Fine-tune for behavior you cannot reliably prompt.**

"Behavior you cannot reliably prompt" is the load-bearing phrase, and it is *enumerable*, not vague. It covers exactly five things:

- **Strict format / schema adherence** — the model must emit valid JSON against your schema every single time, and prompting gets you to 97% not 99.9%.
- **Tone / persona** — a consistent house voice that survives across thousands of calls without a 400-token style preamble on every request.
- **Narrow classification / routing** — a fixed label set or routing decision where a small tuned model beats a big prompted one on both accuracy and cost.
- **Latency + cost reduction via prompt-shrinking** — you have a long, stable instruction block; bake it into weights and delete it from every request.
- **Distillation** — teach a small cheap model to imitate a big expensive one on your narrow task.

Notice what is *not* on that list: "know new facts." Fine-tuning is bad at facts. That is not a tuning bug you can fix with more data — it is what the mechanism *does*, and the next section explains why from first principles.

## How it actually works (mechanism, from first principles)

**Why prompting is the default.** A modern instruct model has already digested trillions of tokens and been instruction-tuned to follow directions. When you write a prompt you are steering an enormous pre-existing capability with a few hundred tokens. The iteration loop is *seconds*: edit text, re-run, observe. No dataset to curate, no GPU to rent, no pipeline to keep alive. The costs are (a) per-request token cost — every instruction token is paid on every call, forever — and (b) reliability that is only as good as your words: the model can drift, ignore a rule under load, or behave differently when the vendor silently ships a new model behind the same API name.

**Why RAG owns facts.** A language model's "knowledge" is a lossy compression of its training corpus into weights, frozen at the training cutoff. It cannot know your 2026 policy doc, a customer's current account balance, or a wiki page edited this morning. RAG fixes this at *inference time*: an embedding model turns the query into a vector, you search a vector index for the most similar chunks of your documents, and you paste those chunks into the prompt. The model answers *from the text in front of it*. The facts live outside the weights, so updating the document store updates the answer instantly — no retraining. And because the source text is right there, you can cite it, which is frequently a hard product/compliance requirement.

**Why fine-tuning changes behavior, not facts.** Supervised fine-tuning (SFT) is next-token prediction on (prompt, completion) pairs. You show the model thousands of examples of the *shape* of a good answer, and gradient descent nudges the weights so that shape becomes the default. Here is the intuition an engineer must internalize:

```
Training signal per example = "given THIS input, the next tokens
                               should look like THIS output"

Repeated over ~1000+ examples with a CONSISTENT output style
  -> the model reliably reproduces the STYLE / FORMAT / TONE
  -> the model does NOT reliably store the specific FACTS in
     any given example (they average into a fuzzy prior)
```

Think about the arithmetic. A 7B model has ~7 billion parameters. One training example is a few hundred tokens. Fine-tune on 1,000 examples for 2 epochs and each individual fact is seen twice and smeared across billions of shared weights via a tiny learning rate. The *pattern* that repeats across all 1,000 examples ("always emit a JSON object with these keys," "always answer in a warm, concise voice") gets reinforced 1,000+ times and sticks. A *fact* that appears in exactly one example ("the refund window is 45 days") gets almost no reinforcement and is easily overridden by the model's far stronger pretraining prior. So at inference the model confidently produces the right *format* filled with a *plausible-but-fabricated* value. This is why **hallucination on facts persists after fine-tuning** — you taught it *how to answer*, not *what is true*.

Contrast the two mechanisms directly:

```
             where the knowledge lives        update cost      hallucinates facts?
Prompt    ->  in the context window            edit text        yes (if not grounded)
RAG       ->  in an external index, injected    reindex docs     no, if retrieval hits
Fine-tune ->  averaged into the weights         full retrain     YES — this is the trap
```

The failure mode to burn into memory: **a fine-tuned model asked a factual question it wasn't grounded on will invent an answer in perfect house style.** The polish makes the fabrication *more* convincing, which is worse than an obviously-generic wrong answer. If your task is "answer questions about knowledge X," fine-tuning is the wrong lever no matter how much data you have — you need RAG so the facts are present at inference.

**Where the two combine.** RAG and fine-tune are not rivals; they operate on different axes. A common mature pattern is *fine-tune for the behavior, RAG for the facts*: tune a model so it always emits your citation-bearing JSON envelope in your house voice, and feed it retrieved documents at inference so the values inside that envelope are true. You almost never fine-tune *instead of* RAG when facts are involved.

## Worked example

Take one concrete task and walk all three levers with numbers. **Task: route inbound support tickets into `{category, priority, needs_human}` as strict JSON, at 200,000 tickets/day.**

**Attempt 1 — Prompt.** A frontier model with a 600-token instruction + 3 few-shot examples hits 94% category accuracy and 98.5% JSON-valid. Cost lens: ~750 input tokens + ~40 output tokens per call. At a representative frontier price of ~$3 / 1M input and ~$15 / 1M output tokens (approximate, 2025-era; check current pricing):

```
per call  = 750 * $3/1e6  + 40 * $15/1e6
          = $0.00225      + $0.0006
          = $0.00285
daily     = 200,000 * $0.00285  ≈  $570/day  ≈  $17,100/month
```

That 1.5% JSON-invalid rate means ~3,000 tickets/day fall through to an error path. Latency p50 ~900 ms on a big model. **Verdict so far: works, but expensive and the invalid rate is a real ops cost.**

**Attempt 2 — RAG.** Retrieve the 5 most-similar past tickets (with their labels) and paste them in. Accuracy nudges to 95%, but you've added ~1,500 retrieved tokens per call *and* a vector-search hop (+80 ms). Cost per call rises to ~$0.0069, roughly **doubling** spend to ~$41k/month. RAG bought ~1 accuracy point at 2.4x the cost — because this task is *behavior* (apply a fixed labeling policy), not *knowledge lookup*. **RAG is the wrong tool here.** (It would be the right tool if the question were "what does our 2026 refund policy say?")

**Attempt 3 — Fine-tune.** You have 5,000 clean, historically-labeled tickets — a stable task definition that hasn't changed in a year. You QLoRA a 7B open model on them. Now:
- Prompt shrinks from 750 tokens to ~60 (the behavior is in the weights). Output ~40 tokens.
- JSON-valid climbs to 99.9%; category accuracy 96%.
- You self-host on a rented L4-class GPU. At a blended self-host cost, per-call token cost is effectively an amortized GPU-hour figure, not a per-token API bill.

```
Rough self-host math (illustrative, verify for your stack):
  1x L4  ≈ $0.60/hr  ,  ~25 tickets/sec sustained on a 7B at int4
  daily volume 200,000  ->  ~2.2 hrs of GPU-seconds of actual work
  but you keep it warm 24h for latency -> ~$14/day  ≈  $430/month
```

Even with a warm GPU 24/7 and generous headroom, self-hosted inference lands near **$430–$900/month vs $17k on the prompted frontier API** — a >20x reduction — *plus* a lower invalid rate and lower p50 latency (~250 ms on a small local model). **Verdict: at this volume and with a stable task + clean data, fine-tune wins decisively.**

The pivot is *scale × stability*. The same fine-tune is a terrible idea at 500 tickets/day (the $17k/month prompt bill is $40/month — the training pipeline costs more to *maintain* than it saves), or if the category taxonomy is redefined quarterly (you retrain every quarter and the old adapter is dead weight).

**Mapping table for quick recall:**

```
"Answer questions about our 2026 policy docs"     -> RAG (facts, changing)
"Always emit this exact JSON envelope"            -> fine-tune (format)
"Summarize in our house voice/persona"            -> fine-tune (tone)
"Route tickets into a fixed 12-label taxonomy"    -> fine-tune (narrow classify, at scale)
"General Q&A quality is a bit weak"               -> prompt, then RAG
"Cut the 1,200-token system prompt we pay for"    -> fine-tune (prompt-shrink) IF stable
"Make GPT-4-class behavior run on a 3B model"     -> fine-tune (distillation)
"Look up a customer's current order status"       -> RAG/tools (live facts) — never fine-tune
```

## How it shows up in production

**The cost/latency lens is the whole business case.** Fine-tuning's payoff is almost always one of two things: (1) you delete a large, stable prompt from every request (prompt-shrinking), or (2) you move work from a big expensive model to a small cheap one (distillation). Both are *per-request savings multiplied by request volume*, weighed against a *fixed* cost: the training run plus the standing cost of owning a pipeline — data curation, retraining cadence, an eval harness, adapter versioning, a serving stack, and an on-call human when the tuned model regresses. The break-even question is blunt: **does (per-request savings × monthly volume) clear the monthly amortized cost of owning the pipeline?** Below ~tens of thousands of daily calls, it usually doesn't, and prompt/RAG wins on TCO even when the fine-tune is technically better.

**Prompt tokens are a recurring tax; weights are a fixed asset.** A 1,000-token instruction block at 1M calls/month is 1B input tokens/month you pay for *forever*. Bake that behavior into weights and the recurring tax drops to near-zero, converted into a one-time training cost plus serving. That conversion is the single most defensible reason to fine-tune, and it's pure arithmetic — you can put it on a slide.

**The maintenance cost is the part engineers forget.** A prompt is a string in a config file; a teammate edits it in a PR. A fine-tune is a *dataset + training config + eval suite + adapter artifact + serving path*, all of which rot. The base model gets deprecated; your data distribution drifts; the taxonomy changes; and every one of those forces a retrain-and-re-eval cycle. Budget for it as an ongoing liability, not a one-off.

**Debugging signature.** When a *prompted* system is wrong, you read the prompt and the output and usually see it. When a *fine-tuned* system is wrong, the failure is baked in and opaque — you can't grep the weights. Your only lens is the eval harness, which is exactly why "you cannot fine-tune responsibly without an eval harness" is a hard prerequisite. If you can't measure task accuracy, JSON-valid rate, *and* capability retention, you cannot tell a good tune from a forgetful one.

**The regression trap.** A fine-tune that gains 15% on your narrow task but silently loses 20% of its general instruction-following (catastrophic forgetting) is often a *net negative* ship. Prompted and RAG systems can't forget general capability because you never touched the weights. This asymmetry is a real reason to prefer the lighter levers when they're close.

## Common misconceptions & failure modes

- **"We'll fine-tune the model on our docs so it knows our product."** The canonical mistake. Fine-tuning on documents teaches the *style* of your docs, not their *facts*; the model will fluently fabricate specifics. Use RAG. If you want both the voice *and* correct facts, fine-tune for voice and RAG the facts.
- **"More training data will fix the hallucinations."** No — the mechanism averages single-occurrence facts into a fuzzy prior regardless of dataset size. Grounding at inference (RAG/tools) fixes factual hallucination; more data does not.
- **"Fine-tuning is more powerful than prompting, so it's the serious solution."** Power isn't the axis; *fit* is. For anything knowledge-shaped or low-volume, fine-tuning is strictly worse on TCO. Reach for the heaviest lever last, not first.
- **"Loss went down, so the fine-tune is good."** Training loss measures fit to *your* data, not task accuracy and not retained general capability. Judge on the three-legged eval (task metrics + judge win-rate + retention), covered in Week 4.
- **"We can skip the prompt/RAG baseline and go straight to tuning."** Then you can't prove the tune was necessary or measure its lift. The baseline *is* your evidence. No baseline, no justified tune.
- **"RAG and fine-tuning are competing choices."** They're orthogonal — facts vs behavior. The mature answer is frequently "both."
- **Under-counting maintenance.** Teams model the training run's cost and ignore the standing pipeline cost, then are surprised when a "cheaper" fine-tune is more expensive in engineer-hours than the API bill it replaced.

## Rules of thumb / cheat sheet

- **Default order: Prompt → RAG → Fine-tune.** Escalate only when the cheaper lever *provably* fails on your eval set.
- **Facts / changing knowledge → RAG.** Behavior / format / tone → fine-tune. Never fine-tune to inject facts.
- **Concrete signals FOR fine-tuning (want most of these):**
  - ≥ ~1,000 clean, consistent examples (quality over quantity).
  - The task definition is *stable* — it won't be redefined next quarter.
  - Cost/latency pressure *at scale* — enough volume that per-request savings clear pipeline cost.
  - A consistent *structured output* is required (strict JSON/schema, every time).
  - You're prompt-shrinking a large stable instruction, or distilling a big model into a small one.
- **Signals AGAINST:** low volume, unstable/changing task, the real need is factual accuracy, no eval harness, no clean labeled data.
- **Break-even (approximate):** if per-request token/model savings × monthly volume < monthly cost of owning the pipeline, don't tune.
- **Always keep the baseline.** The prompt/RAG numbers are the evidence that justifies (or kills) the tune.
- **Fine-tune for voice, RAG for facts** when you need both.

**The one-page decision note (fill this in per task):**

```
DECISION NOTE — <task name>                                    date / author

1. TASK
   - One-sentence task definition:
   - Volume (calls/day) and latency target (p50/p95):
   - Output contract (free text? strict JSON schema? label set?):

2. WHY NOT PROMPT?
   - Baseline prompt result (accuracy / JSON-valid / cost per 1k / p50):
   - Specific way it fails (numbers, not vibes):

3. WHY NOT RAG?
   - Is the failure about FACTS/KNOWLEDGE? (if yes -> RAG, stop here)
   - RAG baseline result if run (accuracy / added tokens / added latency / cost):
   - Why RAG is insufficient or wrong-shaped for this task:

4. EVIDENCE NEEDED TO JUSTIFY TUNING
   - # of clean, consistent examples available (target >= ~1k):
   - Is the task definition stable for >= 6-12 months? (y/n):
   - Projected per-request savings x volume vs monthly pipeline cost:
   - Which fine-tune purpose applies? (format / tone / classify / prompt-shrink / distill):
   - Retention plan: what general-capability suite will I re-run to catch forgetting?

DECISION:  [ ] Prompt   [ ] RAG   [ ] Fine-tune   [ ] Prompt+RAG   [ ] Fine-tune+RAG
```

If sections 2 and 3 can't be filled with *numbers from a real baseline*, you are not ready to fine-tune.

## Connect to the lab

This week's lab has you write exactly this decision note in your repo's `README.md` for one real task (the support-ticket router is the recommended default), *before* you run any training. You'll then debug the SFT pipeline end-to-end on a tiny 0.5B model — proving you can adapt behavior — while the decision note keeps you honest about *whether you should have*. The phase milestone closes the loop: you solve the same task three ways (prompt, RAG, LoRA) and fill the note's "why not" sections with measured accuracy, p50/p95 latency, and cost-per-1k numbers.

## Going deeper (optional)

- **OpenAI fine-tuning guide** (platform.openai.com/docs) — the canonical vendor statement of "prompt/RAG first, fine-tune for behavior," with the standard signals-for-tuning list. Search: "OpenAI fine-tuning best practices when to fine-tune".
- **Hugging Face TRL docs** (huggingface.co/docs/trl) — `SFTTrainer`, the practical SFT entry point you'll use in the lab.
- **Anthropic docs** (docs.anthropic.com) — prompt-engineering and long-context/retrieval guidance; useful for exhausting the prompt lever properly before escalating.
- **"RAG vs fine-tuning" comparisons** — many good practitioner writeups; search: "RAG vs fine-tuning when to use each 2025" and "fine-tuning does not add knowledge". Prefer sources that show measured numbers over assertions.
- **Retrieval-Augmented Generation (Lewis et al., 2020)** — the original RAG paper for the grounding intuition. Search: "Lewis 2020 Retrieval-Augmented Generation paper".
- Label any numbers you carry forward as approximate and re-verify current model pricing before you put a business case in front of anyone.

## Check yourself

1. A product manager says: "Fine-tune the model on our 400-page 2026 benefits handbook so employees can ask it questions." What do you build instead, and what one sentence explains why fine-tuning fails here?
2. Give the five categories of "behavior you cannot reliably prompt" for which fine-tuning is the right tool.
3. Your prompted classifier is 94% accurate and costs $17k/month at 200k calls/day. Under what two conditions does fine-tuning clearly win, and under what one condition does it clearly lose even though it's technically more accurate?
4. Why does adding more training examples *not* fix factual hallucination in a fine-tuned model? Answer from the mechanism.
5. You need answers in your company's house voice *and* grounded in current policy facts. What's the architecture?
6. Name two standing costs of a fine-tune (beyond the training run itself) that a prompt does not incur.

### Answer key

1. Build **RAG** over the handbook. Fine-tuning teaches the *style* of the handbook's prose, not its *facts* — single-occurrence facts get averaged into a fuzzy prior, so the model fluently fabricates specifics. RAG puts the actual policy text in the context at inference and lets you cite it.
2. Strict format/schema adherence; consistent tone/persona; narrow classification/routing; latency+cost reduction via prompt-shrinking; distillation of a big model into a small one.
3. **Wins** when (a) the task definition is stable over 6–12+ months and (b) volume is high enough that per-request savings (prompt-shrink and/or moving to a small self-hosted model) clear the pipeline's monthly maintenance cost. **Loses** when volume is low — e.g. 500 calls/day makes the API bill ~$40/month, less than the cost of owning the training/eval/serving pipeline — even if the tuned model is more accurate.
4. Because the training signal reinforces *patterns that repeat across examples*. A fact appearing in one example is seen a handful of times and smeared across billions of shared weights via a tiny learning rate, easily overridden by the much stronger pretraining prior. Adding examples strengthens the *style* pattern, not any individual fact. Grounding at inference (RAG) is the fix.
5. **Fine-tune for the voice, RAG for the facts:** tune the model so it defaults to your house persona and output envelope, and inject retrieved current-policy documents at inference so the content is true. They're orthogonal levers (behavior vs knowledge).
6. Any two of: dataset curation/refresh as data drifts; a maintained eval harness (task metrics + retention); adapter/model versioning; a serving stack (GPU or FT API); retrain cycles when the base model is deprecated or the task changes; on-call ownership when the tuned model regresses.
