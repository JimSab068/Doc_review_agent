# Privacy-Preserving Loan/KYC Document Review Agent

A multi-agent (primary + independent critic) document review pipeline for loan/KYC packets, built around a hard tokenization boundary so no raw PII ever reaches a third-party LLM API. Deployed to AWS behind an ALB; evaluated against a 143-persona local harness and a 25-persona live cloud regression set.

---

## 1. Problem statement

Traditional IDP (intelligent document processing) and rules-engine tools (Hyperscience, ABBYY, RPA platforms) handle structured field extraction and rule-based flagging well, but fail at two things: reasoning across multiple documents to catch semantic inconsistencies, and producing an explainable, policy-cited rationale for a decision. This system is a narrow, end-to-end demonstration of an agent architecture that closes that specific gap.

---

## 2. System overview

The system ingests a loan application packet (application form + supporting bank statements), extracts structured fields, cross-checks for inconsistencies and missing compliance items, and routes the result either to auto-resolution or a human review queue — with every step logged for audit.

Two properties distinguish it from a naive "call an LLM on a PDF" prototype:

1. **No raw PII ever reaches the third-party LLM API.** All sensitive fields are tokenized before any external call and detokenized only after the round trip, inside a boundary you control.
2. **No single LLM call is treated as a decision.** A primary agent drafts a decision; a separate critic agent checks that draft against retrieved compliance text (Reg B / FCRA excerpts) before it can proceed.

---

## 3. Architecture components

### 3.1 Ingestion layer
- **Input:** loan application PDF/form + N bank statement PDFs (synthetic personas for testing).
- **Output:** normalized `Document` objects (per-page text), nothing downstream ever touches raw file bytes.


### 3.2 PII detector
- Layered detection: regex for structured formats (SSN, account number, DOB, phone, email, name headers), with obfuscation tolerance for split/newline-broken values, plus a pluggable NER `Protocol` for a future name/address model.
- Output: `PIISpan` objects — `{field_type, raw_value, location, confidence, detector}`.
- Recall, not aggregate accuracy, is the metric that matters here — a missed span is a leak, not a minor miss.

### 3.3 Tokenization vault — the privacy boundary
- Every downstream component operates on tokens only (`[[PII_TYPE_xxxxxxxx]]`); only the vault ever holds the raw-value mapping.
- **Two backends behind one interface (`VaultClient`):**
  - `VAULT_BACKEND=memory` — in-process, Fernet-encrypted dict. Local dev and unit tests.
  - `VAULT_BACKEND=dynamodb` — persistent, KMS-encrypted DynamoDB store. Each token's value is individually KMS-encrypted with an EncryptionContext binding it to the exact table + `token_map_id` + token, so a ciphertext copied into a different request's map fails to decrypt even under the same KMS key.
- **Fail-closed:** if `APP_ENV` is `staging`/`production`, construction refuses to start unless `VAULT_BACKEND=dynamodb` — no silent fallback to an in-memory vault that wouldn't survive a restart or be visible across ECS tasks.
- `assert_no_pii_leak()` is called immediately before every outbound LLM call (primary and critic).
- `discard(token_map_id)` is the real data-deletion operation, called from a `finally` block wrapping the whole pipeline run; DynamoDB TTL is defense-in-depth only, not the primary retention mechanism.

### 3.4 Primary agent
- Given a tokenized document, extracts structured fields and produces a `DraftDecision` (extracted fields, inconsistencies, missing compliance items, confidence 0–1, reasoning).
- **Model:** `gemini-3.1-flash-lite`.
- Hard privacy gate: `vault.assert_no_pii_leak()` runs on the built prompt before every call; the call is aborted, not attempted, if it fails.
- Confidence is calibrated across three explicit tiers (clean / genuinely-uncertain / clear-problem), not a binary clean-vs-flagged split — see §6 for why this mattered.
- Fields that can't be recovered from the source are reported with one consistent sentinel (`[UNRECOVERABLE]`), not five different ad-hoc strings, so recall is actually measurable.

### 3.5 Compliance knowledge base
- ChromaDB, embedded with `gemini-embedding-001`.
- Seeded with real (not fabricated) Reg B (ECOA) and FCRA excerpts — adverse action notice timing, prohibited factors, specificity requirements, consumer report disclosure.
- Retrieves top-k relevant passages against the primary agent's reasoning text for the critic to check against.

### 3.6 Critic agent
- Independent second LLM call, checking the primary's structured output against *retrieved compliance text* — deliberately never given the primary's chain-of-thought, so this is a genuine second opinion, not a rubber stamp.
- Escalation must be grounded in a **specific, concrete** compliance concern tied to the actual extracted fields — not merely "a relevant rule exists" (see §6, Finding 4, for why that distinction had to be made explicit).
- Fails closed: any output that doesn't parse into the `CriticVerdict` schema (including verdict strings outside `pass`/`flag`) is normalized to `flag` + `escalate=True` rather than raising.

### 3.7 Routing / escalation logic
- `escalate = critic.escalate OR critic.verdict == "flag" OR primary.confidence < THRESHOLD`
- Threshold: `0.85`.

### 3.8 Detokenization
- Vault reverses tokens to real values only at final output, only for fields the human-review payload actually needs.

### 3.9 Audit log
- MongoDB Atlas (Motor async). Every request: input hash, PII types detected (never values), primary output, critic verdict + citations, routing decision, per-stage latency, timestamps.
- Append-only by design — no update/delete method exists on the writer.
- Defense-in-depth: a serialized entry is scanned for raw PII shapes (SSN/email patterns) before write; a match raises and blocks the write entirely, on the assumption that if this ever fires, the token boundary was violated upstream.
- Durable: a primary-store write failure falls back to a local append-only log rather than silently dropping the entry, and surfaces the failure to the caller.

---

## 4. Testing / evaluation layer

### 4.1 Synthetic persona corpus
143 stratified, hand-labeled personas across 11 categories (clean approve/deny, cross-doc inconsistency — obvious/inferential/false-positive, ambiguous escalation, direct/indirect injection, obfuscated PII, degraded documents, fairness-matched pairs), rendered to PDF so ingestion is genuinely exercised, not fed clean text.

### 4.2 Three-tier test harness
- **Tier A** — local, deterministic stub LLM. Structural correctness (vault, schemas, routing logic) at zero API cost.
- **Tier B** — real Gemini + Chroma + MongoDB Atlas, rate-limited and checkpointed (see §6). This is where actual model-quality metrics are measured.
- **Tier C** — concurrency/load (internal harness for vault isolation and Mongo write contention; a separate staging load runner against the deployed HTTP endpoint is planned, not yet built — see §8).

---

## 5. Non-functional requirements

- **Deployment shape:** ECS + ECR on AWS, containerized, health-checked, live behind an Application Load Balancer. See §7 for deployment details and §7.3 for live validation results.
- **Security:** encrypted-at-rest vault storage (KMS in production), no PII in any log line outside the vault's own encrypted store, credentials read from environment/Secrets Manager only, never threaded through constructor arguments.
- **Data retention:** vault `discard()` on every request completion; DynamoDB TTL as fallback cleanup for abandoned/crashed requests.

---

## 6. Live evaluation — findings and fixes

This section is the most useful part of this README to actually read. It's a record of what a full live run against real Gemini/Chroma/Mongo actually surfaced, what was wrong, why, and what changed — not a claim that the system was correct on the first try.

### 6.1 Baseline run (143 personas, pre-fix)

| Category | Count | Passed | Pass rate |
|---|---:|---:|---:|
| ambiguous_escalate | 10 | 0 | 0.0% |
| clean_approve | 25 | 24 | 96.0% |
| clean_deny | 10 | 10 | 100.0% |
| degraded_document | 10 | 0 | 0.0% |
| fairness_pair | 20 | 19 | 95.0% |
| inconsistency_inferential | 10 | 10 | 100.0% |
| inconsistency_obvious | 15 | 13 | 86.7% |
| injection_direct | 10 | 10 | 100.0% |
| injection_indirect | 8 | 8 | 100.0% |
| looks_like_conflict_but_isnt | 10 | 9 | 90.0% |
| pii_obfuscated_adversarial | 15 | 15 | 100.0% |

PII leak rate 0%. Routing accuracy 88.7%. False approval 14. Brier score 0.148.

**Root causes identified:**
- **Confidence prompt only defined two tiers** (0.95–1.0 "clean", <0.80 "genuine red flag"), with nothing for the 0.80–0.94 band around the 0.85 threshold — so anything short of a blatant problem got rounded up into high confidence. This drove `ambiguous_escalate`'s 0% pass rate.
- **Critic escalation was gated on the primary's own self-report** (empty inconsistencies + high confidence), not an independent read of the source — a structural rubber-stamp path despite the critic's stated design goal.
- **Fairness pair `fp_008`** diverged in routing (one `auto_resolve`, one `human_queue`, 0.25 confidence gap) on two personas identical in every substantive financial fact — a real disparate-impact finding.

### 6.2 Fixes applied

- Added an explicit three-tier confidence band to the primary agent's prompt, with concrete named triggers for the 0.80–0.94 "genuinely uncertain" tier.
- Rewrote critic instructions so escalation requires an independently-identified, specific concern grounded in the actual data — not the primary's self-reported shape.
- Fixed an event-loop-blocking bug in `pipeline.py`: primary/critic agent calls weren't offloaded via `asyncio.to_thread` the way the KB call was, so the shared rate limiter's blocking sleep could stall the whole event loop (including the async Mongo connection) during throttling.
- Fixed an uncaught `pydantic.ValidationError` crash: the critic occasionally returned `"verdict": "escalate"`, outside the `Literal["pass","flag"]` schema. Now normalized and fails closed instead of raising.
- Added a single `[UNRECOVERABLE]` sentinel convention for fields that can't be extracted, replacing five inconsistent ad-hoc strings (`''`, `'null'`, `'Not stated'`, etc.) that were silently failing exact-match scoring.
- Fixed `ingestion.py`: image-only PDF pages (no text layer) were silently converted to empty strings and misread downstream as a model failure. Now raises a clear, categorized error; the harness records this as its own failure category instead of crashing the batch.
- **Discovered and fixed a second-order regression:** the critic-independence fix initially over-corrected into near-total escalation on clean cases, root-caused via audit-log inspection to the primary agent extracting a schema field (`employment_status`) that most documents never actually contain, reporting it `[UNRECOVERABLE]`, and the critic treating that absence as a due-diligence violation. Fixed by (a) telling the critic to distinguish core fields (income, employer, requested amount) from peripheral ones, and (b) telling the primary agent to omit non-applicable fields entirely rather than reporting them as unrecoverable.

### 6.3 Current results (143 personas, post-fix)

| Category | Count | Passed | Pass rate |
|---|---:|---:|---:|
| ambiguous_escalate | 10 | 3 | 30.0% |
| clean_approve | 25 | 25 | **100.0%** |
| clean_deny | 10 | 10 | 100.0% |
| degraded_document | 10 | 0 | 0.0% |
| fairness_pair | 20 | 19 | 95.0% |
| inconsistency_inferential | 10 | 8 | 80.0% |
| inconsistency_obvious | 15 | 15 | **100.0%** |
| injection_direct | 10 | 7 | 70.0% |
| injection_indirect | 8 | 6 | 75.0% |
| looks_like_conflict_but_isnt | 10 | 10 | **100.0%** |
| pii_obfuscated_adversarial | 15 | 14 | 93.3% |

PII leak rate: **0.0%** (0/143) — unchanged, the one invariant that must never move.
Routing accuracy 87.6%. False approval 12 (down from 14). Brier score 0.1145 (down from 0.148 — meaningfully better calibration). `fp_008` now routes consistently for both personas (disparate-impact finding resolved).

**Net effect: real, targeted improvements on the categories the fixes were aimed at (clean cases, obvious inconsistency, false-positive traps, fairness, calibration), at the cost of new regressions in categories that weren't being tested at the time** (injection catch rate, inferential inconsistency, degraded-document routing accuracy, and 6 new API/parse failures vs. 1 previously). This is the expected shape of iterating against a stratified eval set one category at a time rather than evidence the earlier fixes were wrong — but it means the injection and degraded-document numbers above should be read as **currently regressed and under active investigation**, not as representative of the system's ceiling.

### 6.4 Open items — not yet root-caused

- **`api_error_count`: 1 → 6.** Six personas now hit a live parse/API failure that didn't before. Not yet diagnosed against the new prompt.
- **Injection catch rate regressed** (`injection_direct` 100%→70%, `injection_indirect` 100%→75%). Suspected cause: the more field-specific critic is now escalating for unrelated reasons on some injection personas whose ground truth expects `auto_resolve` (i.e. the system should resist the injection and reach a normal decision) — not yet confirmed against actual critic concern text.
- **`inconsistency_inferential` regressed** (100%→80%), cause not yet investigated.
- **`degraded_document` routing accuracy dropped** (60%→30%) despite the confidence-tier fix targeting this category; pass rate remains 0% either way.
- **New fairness signal on `fp_002`** — diverges in routing with a *zero* confidence gap, meaning the primary agent scored both personas identically but the critic decided differently between them. Structurally different from the resolved `fp_008` finding (which had a confidence gap explaining the divergence) and worth treating as a distinct, still-open finding.
- **Three personas (`p_0114`, `p_0118`, `p_0122`) are image-only PDFs with no text layer** — excluded from scoring pending an explicit OCR/no-OCR decision; not a model or prompt defect.

### 6.5 Reproducibility

Every metric above is computed once, from a saved immutable JSON artifact (`artifacts/evaluations/<timestamp>-<gitsha>.json`) — `metrics.py` never re-calls Gemini to recompute a number. `run_evaluation.py` checkpoints each persona's result as it completes, so an interrupted run can resume without re-spending API quota on already-completed personas.

---

## 7. Cloud deployment & validation

### 7.1 Docker / AWS deployment

The service runs as a FastAPI app (`api.py`) behind a lifespan handler that constructs all external clients (Gemini, DynamoDB+KMS vault, ChromaDB, MongoDB Atlas) once at startup rather than per-request, and fails loudly at boot — not on the first request — if any dependency (e.g. the audit store) can't be reached.

Getting from a working local pipeline to a running container surfaced real integration bugs, not just packaging work:

- **`api.py`/`config.py` didn't match the actual source entry points** — corrected the import paths and settings fields so the container's `Settings` object validates against the env vars the rest of the code (`vault.py`, `audit_log.py`, `compliance_kb.py`, `primary_agent.py`) actually reads, rather than 500ing on the first real request.
- **PII was leaking through the API response boundary.** `extracted_raw_text` (the full untokenized document text) was present on the pipeline's internal `final_payload` and was being returned to the client unfiltered. Fixed by defining an explicit internal-only field allowlist (`_INTERNAL_ONLY_FIELDS`) that's stripped before any response leaves the process — audited specifically for "what shouldn't cross the process boundary," not just for schema shape.
- **The critic agent's PII-leak assertion was silently disabled in the default wiring.** `SecureAuditPipeline`'s default construction builds `CriticAgent` without a `vault_client`, which no-ops the `assert_no_pii_leak` check (it only runs `if self._vault_client is not None`). `api.py`'s lifespan handler now wires the vault client into the critic explicitly, closing that gap without changing the pipeline's default behavior for existing tests.
- **Credential/config env vars were at risk of leaking into logs and stack traces.** Consistent with the vault and audit-log design (`secret_redaction.py`), no secret is ever threaded through a constructor argument — every credential (`GEMINI_API_KEY`, `AUDIT_MONGO_URI`, KMS/DynamoDB config) is read directly from the environment at the point of use.

### 7.2 Image hardening

- Split production and dev dependencies so the shipped image doesn't carry the test/eval toolchain (pytest, load-test harnesses, notebook deps) into production.
- Added a `.dockerignore` so local artifacts, `.env` files, and the eval/test tree never end up in the build context or image layers.
- `tests/` is intentionally never copied into the image (`Dockerfile`'s `COPY src/ ./src/`) — this is why `rate_limiter.py` was moved into `src/` rather than reused from the test tree; the production rate limiter has to actually ship.
- Validated end-to-end with a full smoke test against the built image before promoting: container boot → `/healthz` → `/readyz` (live Atlas + Chroma connectivity) → a real `/review` call.

### 7.3 Live cloud regression run

A 25-case golden regression set (a stratified subset of the 143-persona corpus, covering all 9 categories) was run against the **deployed** endpoint — not a local process — to validate the containerized service end-to-end, not just the pipeline logic.

| Category | Count | Passed | Pass rate |
|---|---:|---:|---:|
| ambiguous_escalate | 4 | 1 | 25.0% |
| clean_approve | 3 | 3 | **100.0%** |
| clean_deny | 2 | 2 | **100.0%** |
| degraded_document | 3 | 0 | 0.0% |
| fairness_pair | 4 | 4 | **100.0%** |
| inconsistency_inferential | 2 | 2 | **100.0%** |
| inconsistency_obvious | 3 | 3 | **100.0%** |
| injection_direct | 2 | 2 | **100.0%** |
| injection_indirect | 2 | 1 | 50.0% |

**PII leak rate: 0.0% (0/25)** against the live deployment — the one number that has held at zero across every run reported in this README, local or cloud. Routing accuracy 75.0% (5 false approvals, 1 false escalation, 1 API/parse failure). Fairness: both matched pairs routed identically. Direction of the remaining gaps (`ambiguous_escalate`, `degraded_document`, `injection_indirect`) is consistent with the open items already tracked in §6.4 — this run is a deployment-correctness check, not evidence those categories have been re-solved.

Run artifact: `golden-cloud-20260818T160344Z.md`, git SHA `068eea8e0faa`, executed against the live ALB endpoint, threshold `0.85`. Per §6.5's reproducibility convention, this is computed once from a saved immutable result — it does not re-call Gemini.

---

## 8. Known limitations

- **No OCR stage.** Image-only PDF pages fail loudly at ingestion rather than silently degrading — but are not recovered. Whether to add OCR is an open decision, not an oversight.
- **Injection resistance and degraded-document handling are currently regressed** relative to the pre-fix baseline (§6.4) and need further root-causing before being called solid.
- **Determinism is not fully characterized.** A dedicated test reruns one known-unstable persona (`p_0075`) 5x to check routing stability; this hasn't been re-run since the latest prompt changes.
- **The live cloud run (§7.3) covers 25 of 143 personas.** It validates the deployed container end-to-end and confirms the zero-PII-leak invariant holds in production, but it is not a substitute for re-running the full 143-persona suite against the deployed endpoint.
- **No load/concurrency test against the live deployed endpoint yet.** Tier C today exercises vault isolation and Mongo write contention internally; a staging load runner against the deployed ALB URL is planned but not built (tracked since §4.2/§8 of the earlier draft spec).

---