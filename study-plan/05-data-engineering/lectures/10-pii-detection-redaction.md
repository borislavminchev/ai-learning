# Lecture 10: PII Detection and True Redaction with Presidio

> Somewhere in your corpus is a support ticket that reads "call me back at john@acme.com or 415-555-0199, my SSN is 123-45-6789." If that string reaches your embeddings, your vector DB, your fine-tuning set, or an LLM's context window, you have shipped it — and you cannot un-ship it. Redaction is not a nice-to-have cleaning step; it is the last place you can stop personal data before it fans out into a dozen irreversible systems. This lecture exists because the naive version of this job (draw a black box, run a regex, call it done) fails in ways that are silent, expensive, and legally radioactive. After this you will be able to run Microsoft Presidio's Analyzer + Anonymizer to detect and *truly remove* PII, build **consistent hash-based pseudonyms** that preserve referential integrity across a whole corpus, write and test **custom recognizers** for the formats Presidio misses, and — the part nobody teaches — **prove** the raw string is gone from the output bytes rather than merely hidden.

**Prerequisites:** Python 3.11+, `uv`, comfort with regex and Python classes; you've built this week's clean/dedup stage and understand how text flows into embeddings/RAG · **Reading time:** ~30 min · **Part of:** Phase 5 (Data Engineering) Week 2

---

## The core idea (plain language)

PII redaction is two jobs that people constantly conflate:

1. **Detection** — *find* the spans of text that are personal data. This is a recognition problem: some PII has rigid structure (a US SSN is `\d{3}-\d{2}-\d{4}`), some is fuzzy (a person's name is "whatever a named-entity model thinks is a name"), and some only becomes PII from context ("the account is 40291" is meaningless until "account number" sits three words away).
2. **Removal** — *replace or delete* those spans so the original value is unrecoverable from the output. This is a transformation problem, and it is where "I put a black rectangle over it" quietly betrays you.

Microsoft **Presidio** splits itself along exactly this seam. `AnalyzerEngine` does detection: it runs a stack of **recognizers** over your text and returns a list of `RecognizerResult` objects — `(entity_type, start, end, score)`. `AnonymizerEngine` does removal: it takes that list plus the original text and applies an **operator** per entity type (replace, redact, hash, mask, encrypt, keep). The two are decoupled on purpose. You can detect once and anonymize many ways; you can detect with Presidio and remove with your own code.

The single most important distinction in this entire lecture: **cosmetic masking vs. true removal.** Cosmetic masking changes what a human *sees* — a black box in a PDF viewer, a CSS overlay, a "•••••" in a UI — while the original bytes sit untouched underneath, one `Ctrl+A / copy` or `pdftotext` away from exposure. True removal changes the *bytes*: after redaction, the string `123-45-6789` is provably absent from the output file. In an AI pipeline this is not a subtlety. Your downstream consumers — the embedding model, the tokenizer, the chunk stored in Qdrant — read *bytes*, not the rendered view. A black box means nothing to `sentence-transformers`. If the SSN is still in the content stream, it will be embedded, indexed, and retrieved.

---

## How it actually works (mechanism, from first principles)

### The Analyzer: a stack of recognizers voting with confidence

`AnalyzerEngine` is an orchestrator over a **recognizer registry**. When you call `analyzer.analyze(text=..., language="en")`, it runs every recognizer relevant to the requested entities, collects their results, resolves overlaps, and returns a flat list. There are three recognizer *kinds*, and knowing which caught a given span is essential for debugging false positives/negatives.

**1. Pattern (regex) recognizers.** For structured PII. Presidio ships `EmailRecognizer`, `PhoneRecognizer` (backed by Google's `phonenumbers`), `CreditCardRecognizer`, `UsSsnRecognizer`, `IpRecognizer`, `IbanRecognizer`, and more. A pattern recognizer is a regex plus a base score plus optional **validation** and **context**. Crucially, the regex is only the *first* filter:

- `CreditCardRecognizer` matches a 13–19 digit run, then runs the **Luhn checksum**. `4111 1111 1111 1111` passes Luhn (score bumped); `4111 1111 1111 1112` matches the shape but fails Luhn (rejected or scored low). This is why Presidio's credit-card detection has far fewer false positives than a raw regex — the check-digit does real work.
- `UsSsnRecognizer` matches `\d{3}-?\d{2}-?\d{4}` with a base score around 0.5, then leans on context words to climb.

**2. NER (model-backed) recognizers.** For fuzzy PII — `PERSON`, `LOCATION`, `NlP_ORGANIZATION`, `DATE_TIME`, `NRP`. These come from the **NLP engine**, by default **spaCy `en_core_web_lg`** (the large English model, ~560 MB, real word vectors). Presidio's `SpacyRecognizer` runs the spaCy pipeline once per document, reads the `doc.ents`, and maps spaCy labels (`PERSON`, `GPE`, `ORG`) to Presidio entity types. There is no regex for "person" — the model's statistical judgment *is* the detector, which is exactly why names are the least reliable entity type (more below). You can swap the backend to a transformers model (e.g. a fine-tuned BERT NER) via `TransformersNlpEngine` for higher recall at higher latency.

Why `en_core_web_lg` and not `en_core_web_sm`? The small model has no word vectors and materially worse NER; the large model's context sensitivity meaningfully improves `PERSON`/`LOCATION` recall. This is Presidio's documented default and what this week's lab installs (`spacy download en_core_web_lg`). The `md` model is a middle ground; on CPU, `lg` adds tens of milliseconds per document — acceptable for batch corpus cleaning, something to measure if you ever redact in a request path.

**3. Context-aware boosting.** This is the mechanism that turns weak signals into confident ones. Many patterns are ambiguous in isolation: `40291` could be a ZIP code, an order ID, or an account number. Presidio's `LemmaContextAwareEnhancer` scans a window of words (default ±5 tokens, using spaCy lemmas so "accounts" matches "account") around each candidate. If a recognizer declares context words like `["account", "acct", "routing"]` and one appears nearby, the score gets **boosted** (by default +0.35, capped at a minimum enhanced score). So a bare 9-digit number scores low, but "routing number 021000021" clears threshold. In production this is your single biggest lever on the precision/recall curve for custom formats.

### Confidence scores and the threshold you actually ship

Every result carries a `score` in `[0.0, 1.0]`. It is **not** a calibrated probability — treat it as an ordinal confidence knob, not "87% likely to be PII." `analyze(..., score_threshold=0.5)` drops everything below the cut. The tradeoff is the usual one, and for redaction it is asymmetric:

- **False negative** (missed PII) = leaked personal data = the failure that gets you fined. In redaction you fear this most.
- **False positive** (redacted a non-PII token) = over-redaction = degraded corpus quality (you nuked "Washington" the state because NER thought it was a person).

Because the cost is asymmetric, redaction pipelines run a **lower threshold than you'd instinctively pick** (0.3–0.4 is common) and accept some over-redaction, then measure the damage. The right threshold is empirical: seed a labeled fixture, sweep the threshold, and read recall/precision off *your* data.

```python
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()  # loads spaCy en_core_web_lg + default recognizers
results = analyzer.analyze(
    text="Email john@acme.com or call 415-555-0199. SSN 123-45-6789.",
    entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "PERSON"],
    language="en",
    score_threshold=0.35,
)
for r in results:
    print(r.entity_type, r.start, r.end, round(r.score, 2))
# EMAIL_ADDRESS 6 19 1.0
# PHONE_NUMBER 28 40 0.75
# US_SSN 46 57 0.85   (boosted by the nearby word "SSN")
```

### The Anonymizer: operators, and where removal actually happens

`AnonymizerEngine.anonymize(text, analyzer_results, operators)` walks the spans **right-to-left** (so earlier offsets stay valid as lengths change) and rewrites the string. The `operators` dict maps entity type → `OperatorConfig`:

| Operator  | What it does                                  | Reversible? | Referential integrity? |
|-----------|-----------------------------------------------|-------------|------------------------|
| `replace` | swap for a fixed token, e.g. `<EMAIL>`        | no          | no (all emails → same token) |
| `redact`  | delete the span entirely                      | no          | no                     |
| `mask`    | keep N chars, star the rest (`4111********`)   | no          | partial leak           |
| `hash`    | SHA-256 of the value                          | no          | yes (same in → same out) |
| `encrypt` | AES with a key                                | **yes**     | yes                    |
| `keep`    | leave untouched                               | —           | —                      |

The output is a new string. **The original span is gone from that string's bytes** — this is true removal at the text layer, and it's why doing redaction on *extracted text* (not on the rendered PDF) is the robust path for an AI corpus.

```python
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

anon = AnonymizerEngine()
out = anon.anonymize(
    text=text,
    analyzer_results=results,
    operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"})},
)
print(out.text)
```

### Consistent pseudonymization — the referential-integrity problem

Plain `replace` destroys a property you often need: **the same entity should map to the same token everywhere**, so that "John emailed Jane; John then called…" still reads as one John after redaction. If every email becomes `<EMAIL>`, you've collapsed distinct people into one token and destroyed the co-reference structure your RAG answers depend on.

The fix is **deterministic hash-based pseudonymization**: `john@acme.com` → `<EMAIL_7f3a>` *every run, in every document*. Because it's a pure function of the value (plus a secret salt), it is (a) consistent across the corpus, (b) not trivially reversible, and (c) stable across pipeline re-runs so your diffs and Parquet hashes don't churn. Presidio supports custom operators — subclass `Operator`:

```python
import hashlib
from presidio_anonymizer.operators import Operator, OperatorType

SALT = b"corpus-pipeline-v1"  # keep in a secret manager, NOT in git

class ConsistentPseudonym(Operator):
    def operate(self, text: str, params: dict) -> str:
        etype = params["entity_type"]
        digest = hashlib.sha256(SALT + text.encode()).hexdigest()[:4]
        return f"<{etype}_{digest}>"
    def validate(self, params): pass
    def operator_name(self): return "pseudonym"
    def operator_type(self): return OperatorType.Anonymize

anon.add_anonymizer(ConsistentPseudonym)
out = anon.anonymize(
    text=text, analyzer_results=results,
    operators={"DEFAULT": OperatorConfig(
        "pseudonym", {"entity_type": "PII"})},  # entity_type injected per-span
)
```

Two engineering notes that bite people. **Use a salt.** A bare `sha256("john@acme.com")` is instantly reversible by anyone with a dictionary of likely emails/SSNs — the space is tiny. The salt (from a secret manager, rotated per corpus version) makes the rainbow-table attack infeasible. **Watch collisions.** Truncating to 4 hex chars = 65,536 buckets; by the birthday bound you expect a collision around ~300 distinct values (√65536 ≈ 256). For a corpus with thousands of distinct emails, use 8+ hex chars. The lab's `<EMAIL_7f3a>` style is illustrative; size the suffix to `√(2 × distinct_count)` at minimum.

### The masking-vs-removal line, drawn in bytes

```
COSMETIC MASKING (PDF black box)          TRUE REMOVAL (text rewrite)
┌─────────────────────────────┐          ┌─────────────────────────────┐
│ render layer:  ██████████    │          │ render layer:  <SSN_9c1e>    │
│ content stream:"123-45-6789" │  ◄─LEAK  │ content stream:"<SSN_9c1e>"  │
│ pdftotext →   "123-45-6789"  │          │ pdftotext →   "<SSN_9c1e>"   │
└─────────────────────────────┘          └─────────────────────────────┘
 embedding model reads bytes → leaks       embedding model reads bytes → safe
```

For PDFs specifically: a rectangle annotation or a filled black `re` operator draws *over* the glyphs; the text operators (`Tj`/`TJ`) that hold `123-45-6789` are still in the content stream, and `pdftotext`, `pdfplumber`, or any parser recovers them. True PDF redaction requires removing/replacing those content-stream operators (tools like `pikepdf` with true redaction, or Adobe's redact-and-apply). In an AI pipeline you usually sidestep this entirely: **extract to text first (Docling), redact the text, and index the redacted text** — the PDF's content stream never enters your corpus.

---

## Worked example — a support-ticket record, end to end

Input record from the clean stage:

```
Ticket #8842 — Reporter: Maria Gomez (maria.gomez@northwind.co).
Callback 415-555-0199. Card on file 4111 1111 1111 1111.
Internal ref EMP-004521, account 40291.
```

**Detection** with `score_threshold=0.35`, `en_core_web_lg` NER + defaults:

| Span                    | Entity          | Recognizer            | Score | Why |
|-------------------------|-----------------|-----------------------|-------|-----|
| `Maria Gomez`           | PERSON          | SpacyRecognizer (NER) | 0.85  | model + it's a name |
| `maria.gomez@northwind.co` | EMAIL_ADDRESS| EmailRecognizer regex | 1.0   | RFC-shaped, unambiguous |
| `415-555-0199`          | PHONE_NUMBER    | PhoneRecognizer       | 0.75  | `phonenumbers` parse OK |
| `4111 1111 1111 1111`   | CREDIT_CARD     | CreditCardRecognizer  | 1.0   | matches + **passes Luhn** |
| `EMP-004521`            | —               | *(none)*              | —     | **false negative** — no default recognizer |
| `40291`                 | —               | *(none)*              | —     | bare number, no context word nearby |

Two misses. `EMP-004521` (employee ID) and `40291` (account) are exactly the "custom format" false negatives the pitfalls warn about. Add a custom recognizer:

```python
from presidio_analyzer import Pattern, PatternRecognizer

emp_recognizer = PatternRecognizer(
    supported_entity="EMPLOYEE_ID",
    patterns=[Pattern(name="emp_id", regex=r"\bEMP-\d{6}\b", score=0.9)],
)
acct_recognizer = PatternRecognizer(
    supported_entity="ACCOUNT_NUMBER",
    patterns=[Pattern(name="acct", regex=r"\b\d{5}\b", score=0.3)],  # weak alone
    context=["account", "acct", "acc"],  # boosts to ~0.65 near "account"
)
analyzer.registry.add_recognizer(emp_recognizer)
analyzer.registry.add_recognizer(acct_recognizer)
```

Note the deliberate asymmetry: `EMP-\d{6}` is specific enough to score 0.9 outright, but `\d{5}` alone is dangerous (ZIP codes, IDs, quantities), so it starts at 0.3 and relies on **context boosting** to reach threshold only when "account" is nearby. Run again and both are caught.

**Anonymization** with consistent pseudonyms:

```
Ticket #8842 — Reporter: <PERSON_4a1f> (<EMAIL_7f3a>).
Callback <PHONE_2b9c>. Card on file <CREDIT_CARD_e0d2>.
Internal ref <EMPLOYEE_ID_11ac>, account <ACCOUNT_NUMBER_88fe>.
```

**Verification** (the step that makes it real):

```python
SEEDED_PII = ["Maria Gomez", "maria.gomez@northwind.co", "415-555-0199",
              "4111 1111 1111 1111", "EMP-004521", "40291"]
for s in SEEDED_PII:
    assert s not in out.text, f"LEAK: {s!r} still present"
```

If any assertion fires, the record is *not* clean — and in Week 3 that failed assertion becomes a **blocking gate** that halts the whole DAG.

---

## How it shows up in production

- **The leak is irreversible and fan-out amplifies it.** If un-redacted text reaches embeddings, it's now in your vector index, in your embedding provider's logs (if you used an API), in any cached LLM context, and possibly in a fine-tune checkpoint. There is no "delete the SSN" button across all of those. The cost of a false negative is not one bad row — it's a data-subject-access-request nightmare and, under GDPR/CCPA, real fines. This is why redaction is the gate, not the polish.
- **Names are the weakest link, and it's structural.** `PERSON` recall from `en_core_web_lg` on messy real text (lowercase, no punctuation, non-Western names, OCR noise) can be well under what you'd hope. Expect to miss names your regexes never could — and expect over-redaction of capitalized common words. If names matter for compliance, layer a transformers NER backend or a gazetteer, and measure. Don't trust the demo's 100% on your corpus.
- **Latency: spaCy is the cost.** The Analyzer runs the full spaCy pipeline per document. On CPU with `en_core_web_lg`, budget on the order of tens of milliseconds per KB of text (measure yours). For a million-doc corpus that's real wall-clock — batch it, parallelize across processes, and never put it naively in a synchronous request path. Presidio itself is thin; the NLP engine dominates.
- **Consistency churn breaks your hashes.** If your pseudonym isn't deterministic (e.g. you used `random` or an unsalted-but-per-run token), every re-run produces different output bytes, which changes your content hashes, which defeats Week 1's idempotency and Week 3's DVC diffs. Deterministic hashing is what keeps redaction replay-safe.
- **Custom formats are where you actually spend your time.** Every company has employee IDs, internal account numbers, case numbers, non-US phone formats, national IDs. Presidio's defaults catch none of these. The real work of a redaction project is enumerating your organization's PII shapes and writing + testing recognizers for each. Budget for it.

---

## Common misconceptions & failure modes

- **"I drew a black box, it's redacted."** No — the bytes are intact. `pdftotext` recovers them, and so does your embedder. Redact the *text*, verify the *bytes*.
- **"The regex matched, so it's caught."** A regex without validation is a false-positive machine. Presidio's value is regex + Luhn/context/checksum. And a regex you didn't write catches nothing — defaults miss custom formats entirely.
- **"Higher threshold is safer."** Backwards for redaction. A high threshold *increases false negatives* (missed PII = the dangerous error). Lower the threshold, accept over-redaction, measure.
- **"Score is a probability."** It's an ordinal confidence, uncalibrated. `0.85` is not "85% chance." Tune the threshold empirically on labeled data; don't reason about it as probability.
- **"Replace-with-`<PII>` is fine."** It destroys referential integrity — two different people become the same token. Use consistent hash-based pseudonyms when co-reference matters (it usually does for RAG quality).
- **"Unsalted SHA-256 is safe."** For small-cardinality PII (SSNs, phone numbers, likely emails) the preimage space is tiny; an attacker brute-forces it. Salt it, keep the salt secret.
- **"Detection is enough."** Detecting and then not verifying removal is the classic gap. The assertion `raw not in output` is the only thing that turns "I ran Presidio" into "the PII is provably gone."
- **"Presidio anonymizes PDFs."** Presidio operates on text (and images via OCR + the image-redactor module). For PDF byte-level redaction you need PDF tooling; the AI-pipeline answer is to redact extracted text.

---

## Rules of thumb / cheat sheet

- **Redact text, not the rendered doc.** Extract → redact → index. The content stream never enters the corpus.
- **Verify by assertion, always.** For every seeded PII string, `assert raw not in output`. This is your Definition-of-Done, not a nicety.
- **Threshold 0.3–0.4 for redaction** (recall-favoring). Sweep it on a labeled fixture; don't guess.
- **Use `en_core_web_lg`, not `sm`.** Word vectors materially improve name recall. Consider a transformers backend if names are critical.
- **Consistent pseudonyms via salted hash.** Deterministic (replay-safe) + salted (not reversible) + suffix ≥ 8 hex chars for corpora with >1k distinct values.
- **Custom formats need custom recognizers.** Employee IDs, internal accounts, non-US phones → write a `PatternRecognizer`; use `context` words to boost weak patterns instead of raising their base score.
- **Weak pattern + context beats strong pattern alone.** `\d{5}` at score 0.3 + context `["account"]` is safer than `\d{5}` at 0.6 firing on every ZIP.
- **Luhn/checksum validation is free precision.** Prefer recognizers that validate (credit cards, IBANs) over bare regex.
- **Names will leak. Budget for it.** Layer NER + gazetteers, measure recall on *your* text, don't trust demo numbers.
- **Never fabricate the pass.** If the assertion fails, the row is dirty — fail the pipeline (Week 3), don't lower the bar.

---

## Connect to the lab

In this week's lab, `pii.py` wires `AnalyzerEngine` (EMAIL, PHONE, PERSON, CREDIT_CARD, US_SSN, plus your custom `EMPLOYEE_ID`/`ACCOUNT_NUMBER` recognizers) into `AnonymizerEngine` with a **consistent hash-based pseudonym** operator, and emits redacted text into `data/clean/*.jsonl`. The Definition of Done is verification-driven: for **15/15 seeded PII strings**, the raw string must be absent from the output, and the same input entity must map to the same pseudonym every run. In Week 3 this exact check is re-run as a **blocking Dagster asset check / Great Expectations expectation** between the clean and Parquet stages — inject one un-redacted row and the pipeline must halt. That is redaction as a downstream blocking gate: PII removal isn't just cleaning, it's the invariant that guards everything indexed and answered.

---

## Going deeper (optional)

- **Microsoft Presidio docs** — `microsoft.github.io/presidio` (Analyzer, Anonymizer, "Creating a custom recognizer", "Supported entities", the image-redactor module). The single best source; read the custom-recognizer and context-enhancer pages carefully.
- **Presidio GitHub** — `github.com/microsoft/presidio` — the `presidio-analyzer` predefined recognizers directory is worth reading directly to see how `CreditCardRecognizer` (Luhn) and `UsSsnRecognizer` (context) are actually built. Copy their structure for your customs.
- **spaCy models** — `spacy.io/models/en` — details on `en_core_web_lg` vs `md`/`sm`/`trf`, NER labels, and the transformer pipeline if you upgrade the backend.
- **`phonenumbers`** (Google libphonenumber Python port) — search: *python phonenumbers github* — how Presidio validates/normalizes phone numbers across regions; essential when you handle non-US formats.
- **Regulatory grounding** — search: *GDPR pseudonymisation vs anonymisation* and *NIST SP 800-122 PII* — understand that pseudonymized data is still personal data under GDPR (reversible), whereas true anonymization is not; this drives whether you `encrypt` (reversible) or `hash`/`redact` (not).
- **Search queries:** *Presidio custom recognizer context words*, *Presidio consistent anonymization deterministic*, *PDF true redaction content stream pikepdf*, *Luhn algorithm credit card validation*.

---

## Check yourself

1. You put a black rectangle over an SSN in a PDF and feed that PDF into your extraction + embedding pipeline. Is the SSN in your vector index? Why or why not, and how would you check?
2. For a redaction pipeline, should you raise or lower `score_threshold` relative to a "balanced" setting, and which error type does that trade against which?
3. Why does replacing every email with the literal `<EMAIL>` degrade a RAG corpus, and what property does a hash-based pseudonym restore?
4. `sha256("123-45-6789")[:4]` is used as a pseudonym suffix. Name the two distinct problems with this and how you'd fix each.
5. Presidio misses your company's `ACCT-9-digit` format entirely. Walk through what you add, and why you might start its pattern score at 0.3 rather than 0.7.
6. Which entity type is your least reliable detection, why is that structural rather than a config bug, and what do you do about it?

### Answer key

1. **Yes, it's in the index.** The black rectangle is a render-layer annotation; the text operators holding the SSN are still in the PDF content stream. Extraction (`pdftotext`/Docling) recovers the raw string, which then gets embedded and indexed. Check by running the extractor and asserting the SSN string is present in the extracted text — it will be. The fix is to redact the *extracted text*, then assert the raw string is absent from the output bytes.
2. **Lower it** (toward 0.3–0.4). Lower threshold → fewer false negatives (missed PII, the dangerous, irreversible error) at the cost of more false positives (over-redaction, a quality/recoverable cost). Because the costs are asymmetric — a leak is far worse than an over-redaction — you favor recall.
3. Collapsing all emails to one token destroys **referential integrity / co-reference**: two different people become indistinguishable, and "John emailed Jane, then John called" loses that it's one John. A deterministic hash-based pseudonym maps each distinct value to a distinct, stable token (`<EMAIL_7f3a>`), so the same entity reads as the same entity across the corpus while the original value is gone.
4. (a) **Unsalted** — SSN space is small, so an attacker precomputes/brute-forces the hash and reverses it; fix with a secret, rotated salt (`sha256(salt + value)`). (b) **Only 4 hex chars = 65,536 buckets** — birthday collisions appear around ~256–300 distinct values, so distinct SSNs collide to the same token; fix by using ≥8 hex chars sized to `√(2 × distinct_count)`.
5. Add a `PatternRecognizer(supported_entity="ACCOUNT_NUMBER", patterns=[Pattern(regex=r"\bACCT-\d{9}\b", ...)])` and register it (`analyzer.registry.add_recognizer`); write a test with seeded positives and near-miss negatives. You'd start a *generic* 9-digit pattern low (0.3) and add `context=["account","acct"]` so it only clears threshold near account-ish words, avoiding false positives on other 9-digit runs — but note that if the `ACCT-` prefix is distinctive and unambiguous, a high base score is fine; the low-score-plus-context trick is specifically for weak, ambiguous patterns.
6. **`PERSON`.** It's model-backed (spaCy NER), so there's no rigid structure to key on — recall depends on the statistical model and drops on lowercase/noisy/OCR/non-Western text; that's inherent to fuzzy entities, not a misconfiguration. Mitigations: use `en_core_web_lg` or a transformers NER backend, add gazetteers/deny-lists for known names, lower the threshold for `PERSON`, and — critically — measure recall on your own labeled data rather than trusting demo results.
