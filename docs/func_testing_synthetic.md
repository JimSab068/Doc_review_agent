# Functional Specification — Section 4: Testing & Evaluation Layer
## Privacy-Preserving Loan/KYC Document Review Agent

**Status:** Draft v1 — pre-build reference document, supersedes README §4
**Depends on:** working pipeline, §3.1–3.9, all unit tests passing

---

## 4.0 Scope and goals

This layer answers three separate questions, and the harness must keep them
separate rather than producing one blended "accuracy" number:

1. **Does it work?** (extraction/reasoning quality vs. ground truth)
2. **Does it hold under attack?** (injection, obfuscated PII, adversarial framing)
3. **Is it fair and stable?** (disparate impact, run-to-run variance, confidence calibration)

A single 50-persona batch scored on one aggregate pass/fail cannot answer
all three credibly. The redesign below stratifies personas by category so
each metric has enough trials in its own bucket to support a real claim.

---

## 4.1 Synthetic persona generator

### 4.1.1 Volume and stratification

Target **~130 personas**, not 500 — see rationale below. Every persona is
tagged with a category; metrics are reported per-category, not just in
aggregate.

| Category | Count | Purpose |
|---|---|---|
| Clean / straightforward approve | 25 | Baseline extraction accuracy, false-escalation floor |
| Clean / straightforward deny | 10 | Same, for the deny path |
| Cross-doc inconsistency — obvious | 15 | Core business value: easy case |
| Cross-doc inconsistency — inferential | 10 | e.g. employer name spelled two ways vs. truly different employers |
| Cross-doc "looks like conflict but isn't" | 10 | False-positive-inconsistency trap (nicknames, formatting variants) |
| Genuinely ambiguous — should escalate | 10 | Escalation accuracy on the hard middle |
| Direct prompt injection (in applicant-facing text) | 10 | Injection catch rate, direct |
| Indirect prompt injection (embedded in bank statement / fed back through extracted_fields into critic prompt) | 8 | Injection catch rate, indirect — tests whether primary's JSON output can carry a payload into the critic's prompt |
| Obfuscated/malformed PII — clean format | (subset of above, not separate count) | — |
| Obfuscated/malformed PII — adversarial format (split SSN, unicode lookalikes, OCR-mangled digits, spelled-out digits) | 15 | PII recall under realistic degradation, not just clean regex matches |
| Degraded document quality (poor OCR text, missing pages, truncated statements) | 10 | Extraction graceful-degradation vs. hallucination |
| Fairness-matched pairs (identical financials, varying name/zip/age-implying detail) | 10 pairs = 20 | Disparate impact probe — see 4.1.4 |
| Golden regression set (hand-verified, frozen) | 25 | Re-run on every prompt/model version bump — not part of the rotating batch |

Numbers are a starting allocation, not fixed law — adjust based on where
your CI is too wide to support the claim you want to make (see 4.6).

### 4.1.2 Per-persona structure

Each persona is a JSON object, generated *before* any PDF rendering, so
ground truth is authored independently of what the pipeline will later
extract:

```json
{
  "persona_id": "string, e.g. p_0042",
  "category": "one of the table rows above",
  "documents": [
    {"doc_type": "loan_application", "raw_text": "..."},
    {"doc_type": "bank_statement", "raw_text": "..."}
  ],
  "ground_truth": {
    "extracted_fields": {"...": "..."},
    "pii_spans": [{"field_type": "...", "raw_value": "...", "location_hint": "..."}],
    "expected_inconsistencies": ["..."],
    "expected_missing_compliance_items": ["..."],
    "expected_escalate": true,
    "expected_verdict": "pass | flag",
    "injection_planted": {"present": true, "type": "direct|indirect", "payload": "..."},
    "fairness_pair_id": "p_0042_pair (nullable, only for fairness category)"
  }
}
```

`ground_truth` is authored by hand or by a separate, non-agent generation
process — never by running the pipeline itself and calling its own output
"truth."

### 4.1.3 Rendering to PDF

Personas are authored as text first, then rendered to PDF via a templating
library (matches README's original approach) so the ingestion layer
(`pdfplumber` extraction) is genuinely exercised — text-in/text-out
shortcuts the exact layer (3.1) most likely to introduce real-world noise
(broken line wraps, extraction gaps).

For the **degraded document quality** category specifically, deliberately
render at low resolution or crop pages before extraction, rather than
hand-editing the extracted text — this tests the actual ingestion path
under stress, not a simulation of it.

### 4.1.4 Fairness-matched pairs — construction detail

For each of the 10 base financial profiles, generate two personas that are
**identical in every substantive financial fact** (income, requested
amount, employment length, debt ratios) and differ only in a single
protected-class-correlated proxy: applicant name (demographically
associated), home address/zip, or age-implying phrasing (e.g. graduation
year). Run both through the full pipeline and diff the routing outcome and
`extracted_fields.confidence`. Any non-trivial divergence is a finding to
report, not a bug to quietly patch and re-test until it disappears.

### 4.1.5 Why ~130, not 500

Raw persona count doesn't make a rare-event claim ("0% leak rate," "100%
catch rate") stronger — trial count *within that category* does. Catching
15/15 injection attempts supports a much weaker statistical claim than
catching 80/80. Budget effort toward more trials in the categories where
you're making a hard 0%/100% claim (PII leak, injection catch) rather than
inflating the clean/easy categories, which don't need more than ~25-35
examples to establish a stable baseline. A labeled, stratified 130 is also
simply more legible to a reviewer than an unlabeled 500 — it demonstrates
you know why each case exists.

---

## 4.2 Test harness — three tiers, run in this order

### Tier A — Local, mocked LLM (run first, every commit)
- Existing unit tests (vault, PII detector, schemas, routing logic) plus:
- Full pipeline run against all 130 personas with a **deterministic stub
  `LLMClient`** (returns scripted JSON matching each persona's
  `ground_truth`, plus deliberately malformed variants to test
  `LLMOutputParseError` / retry paths).
- Purpose: catch structural bugs — vault leaks, schema violations, routing
  logic errors, retry handling — with zero API cost and zero flakiness.
- Must pass 100% before Tier B runs.

### Tier B — Local, real Gemini API (run before every deploy)
- Same 130 personas, real `GeminiClient` calls, run locally against
  `localhost`/dev environment, not the deployed Cloud Run URL.
- This is where genuine LLM quality metrics are measured (Section 4.3) —
  mocked runs cannot tell you if the model actually reasons well.
- Run at low concurrency first (sequential or small batches) to control
  free-tier quota burn; only move to concurrent/load testing (Tier C)
  once correctness is confirmed.
- Any failure here blocks deployment. Fix, re-run affected categories
  (not necessarily the full 130), confirm, then proceed.

### Tier C — Load / adversarial concurrency harness
- Fires personas concurrently (asyncio, or k6/Locust/Artillery for
  externally-driven load) against the pipeline, still in the local/dev
  environment initially.
- Purpose: p95 latency under concurrency, race conditions in the vault's
  per-`token_map_id` store, MongoDB write contention.
- Re-run this same harness against the deployed Cloud Run URL after
  release (post-deploy smoke, see 4.5) — same personas, same scoring,
  different target — to confirm deployed behavior matches local.

---

## 4.3 Required metrics to report

Split into three groups reflecting 4.0's three questions. Report every
number per-category from 4.1.1, not only in aggregate.

### Correctness / quality

| Metric | Target / reporting |
|---|---|
| Field-level extraction accuracy | Per-field precision/recall, not one blended score |
| Extraction accuracy vs. document quality | Compare clean vs. degraded-document category |
| Missing-field hallucination rate | Rate of invented values when field is genuinely absent |
| Cross-doc inconsistency detection (obvious) | Recall against ground truth |
| Cross-doc inconsistency detection (inferential) | Recall against ground truth, reported separately from obvious |
| False-positive inconsistency rate | Rate of flagged "conflicts" on the not-actually-conflicting category |
| Hallucination / groundedness rate | % of extracted_fields and reasoning claims traceable to a literal source span |
| Escalation accuracy vs. ground truth | Report actual number |
| False approval rate | Primary business risk metric |
| False escalation rate | Operational cost metric |
| Schema adherence | Valid-JSON rate under both clean and paraphrased system prompts |
| Retry recovery rate | % of `LLMOutputParseError` cases resolved by bounded retry |
| End-to-end decision accuracy | Overall vs. labeled ground truth |
| Confidence calibration | Reliability curve / Brier score across confidence deciles |
| Determinism / run-to-run variance | % of identical-input runs producing a different verdict/fields, n=5-10 reruns on a subset |

### Security / adversarial

| Metric | Target |
|---|---|
| PII leak rate (raw PII in any outbound LLM payload) | 0% |
| PII detection recall — clean format | Report actual, target near 100% |
| PII detection recall — obfuscated/adversarial format | Report actual — expect and disclose degradation vs. clean |
| Direct prompt-injection catch rate | 100% of planted attempts, n ≥ 10 |
| Indirect prompt-injection catch rate | 100% of planted attempts, n ≥ 8, reported separately from direct |
| Compliance citation precision | % of critic citations that are real, valid Reg B/FCRA sections |
| Compliance citation groundedness | % of citations that actually appear among that call's *retrieved* passages (not just real elsewhere) |
| Compliance citation recall | % of applicable retrieved violations the critic actually cites |
| Critic rubber-stamp rate | Critic pass-rate conditioned on primary being wrong per ground truth |
| Critic independence check | Verdict stability when `reasoning` text is varied but `extracted_fields` held constant |

### Fairness / stability

| Metric | Target |
|---|---|
| Disparate impact — routing outcome | Divergence rate across fairness-matched pairs |
| Disparate impact — confidence score | Mean/variance shift across fairness-matched pairs |
| Golden regression pass rate | 100% on frozen 25-case set per model/prompt version; any drop is a release blocker |

### Performance

| Metric | Target |
|---|---|
| p50 / p95 latency, full pipeline | Report actual, per Tier C |
| p95 latency, per-stage breakdown | Using existing `latencies` dict already in `pipeline.py` |
| Throughput under concurrent load | Report actual |

---

## 4.4 Evaluation methodology notes

- **LLM-as-judge validation**: any metric graded by an LLM call (likely
  hallucination/groundedness, reasoning quality) needs its own validation
  — score a 30-40 example human-labeled subset first and check
  judge/human agreement before trusting the judge on the full 130.
- **Written rubric before generation**: reasoning-quality and citation-
  quality scoring criteria (1-5 scale or explicit pass/fail conditions)
  must be written down before any outputs are generated, to avoid
  unconsciously grading your own system generously.
- **Model-version pinning**: every reported number is tagged with the
  exact model string (`gemini-3.1-flash-lite` etc.) used to produce it.
  Given two silent deprecations already hit during this build
  (`text-embedding-004`, `gemini-2.0-flash`), assume it will happen again
  and make the eval report self-documenting about which model produced
  which number.

---

## 4.5 Deployment sequencing

1. **Tier A (mocked)** passes 100% locally — structural correctness.
2. **Tier B (real API, local)** run against all 130 personas — LLM
   quality metrics in 4.3 measured and reported here. Fix any failing
   category, re-run that category, confirm.
3. **Tier C (load, local)** — concurrency and latency validated in dev
   environment.
4. **Deploy** to Cloud Run.
5. **Post-deploy smoke**: re-run a *subset* (golden 25 + one representative
   persona per category, ~35-40 total) against the live Cloud Run URL.
   This is a config-drift check (env vars, cold-start behavior, network
   egress rules) — not a re-run of the full quality eval, which already
   happened in step 2.
6. **Ongoing**: golden regression set (25) re-run on every subsequent
   model or prompt change, against whichever environment is currently
   live.

---

## 4.6 Deliverables for this section

- [ ] `personas/` — 130 stratified JSON personas + rendered PDFs, tagged by category
- [ ] `tests/tier_a_mocked/` — deterministic stub-LLM pipeline tests
- [ ] `tests/tier_b_live/` — real-API correctness/quality run + report generator
- [ ] `tests/tier_c_load/` — concurrency harness (asyncio or k6/Locust)
- [ ] `eval_report.md` (generated, not hand-written) — every metric in 4.3, per-category, with model version tags
- [ ] `golden_set/` — 25 frozen cases + a script that fails CI on any regression
- [ ] One paragraph, honestly stated: which categories had the widest confidence intervals and why (usually the adversarial/rare-event buckets) — this directly feeds the README's committed "what broke and what I fixed" section