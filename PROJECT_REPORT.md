# Silicon Sampling for Indian Public Opinion — Full Project Report

**Purpose of this document:** a single, complete, parameter-by-parameter account of everything done in this project, in the order it was done, with every threshold, every bug found and fixed, every real result, and an honest accounting of what remains. Written so the entire project can be judged and reproduced from this document alone, without needing to read the conversation history it came from.

**Status as of writing (2026-08-28):** ~90% complete against the project's own 7-phase plan (Phase 4/fine-tuning excluded from that figure — deferred by explicit instruction, not incomplete work). Track A inference and subgroup analysis are now fully converged on both models at both item widths (5 and 15 items). See §11 for the full breakdown.

---

## 1. Research Question and Motivation

**"Silicon sampling"** is the practice of conditioning a large language model on a demographic profile and treating its output as a stand-in for that person's answer to a survey question — a proposed cheap substitute for human polling.

Prior validation of this technique (Argyle et al. 2023; WorldValuesBench, LREC-COLING 2024) reports **aggregate** fidelity only: does the LLM reproduce a population's overall opinion distribution. This project asks a different, more decision-relevant question:

> **Does silicon sampling fidelity hold evenly across a population's own internal diversity (urban/rural, income, education, sex, age, region), or does it silently degrade for specific subgroups while looking fine in aggregate?**

We test this for India using World Values Survey Wave 7 (WVS-7) data, and — the methodological core of this project — **require every finding to replicate across two independent, unrelated LLMs and remain stable as the survey-item sample widens** before treating it as a claim about the technique rather than an artifact of one model or one small sample.

---

## 2. Dataset

### 2.1 Source and acquisition

- **Dataset:** World Values Survey Wave 7, cross-national release, **version 6.0** (not v5.0 or earlier).
- **Why v6.0 specifically:** India's WVS-7 fieldwork completed **July 2023**, after v5.0 shipped. v5.0 contains **64 countries and zero India rows**; v6.0 added India, bringing the total to **66 countries**. This was verified two ways: (a) the WVS site's own release notes state India was "the last included survey," and (b) empirically, the WorldValuesBench project's own codebook — built on v5.0 — was checked and confirmed to list 88 country codes with **no India entry at all**.
- **Source URL:** worldvaluessurvey.org → Statistical Data Files → `WVS Cross-National Wave 7 csv v6 0.zip`. Free, gated behind a one-page registration form (no account, no payment), governed by WVSA's Conditions of Use, which **prohibit redistribution of the data files themselves** — this governs several data-handling decisions below (§2.5).
- **Codebook:** `WVS7 Codebook Variables report V6.0.pdf`, the official 404-page variable dictionary, downloaded separately (no registration form required for documentation).
- **File actually used:** `data/raw/WVS_Cross-National_Wave_7_csv_v6_0.csv` — verified on receipt: 97,220 total respondent rows, 66 countries, **exactly 1,692 India rows** (matching the officially published India sample size).

### 2.2 Cleaning

Performed by `src/data/load_wvs.py`:
1. Load raw CSV (613 columns).
2. Recode WVS missing-value sentinel codes **-1 through -5** (don't-know, no-answer, not-applicable, not-asked, missing) to `NaN` across all numeric columns. Real count on this dataset: **6,887,680 cells recoded** across 601 columns.
3. Filter to `B_COUNTRY_ALPHA == "IND"`.
4. Sanity-check the resulting N against the expected 1,692 (warns if it drifts by more than 50).
5. Save to `data/processed/ind_wvs7.parquet`.

**Result:** 1,692 respondents × 613 columns. Overall cell missingness after recoding: **18.9%**.

### 2.3 Item selection — codebook-grounded, not hand-picked

`src/data/parse_codebook.py` parses the official 404-page codebook PDF via regex into structured metadata (question wording, response scale, valid codes, thematic block) for **353 distinct questions**, written to `data/reference/wvs7_codebook.json`.

`src/data/select_items.py` then screens **every** `Q`-prefixed column in the raw data (**373 candidates**) against that metadata, in this exact order (first failure wins):

| Screening step | Rule | Items rejected |
|---|---|---|
| 1. `no_codebook_entry` | Column has no matching codebook entry (technical/country-specific fields not fielded in India) | 20 |
| 2. `demographic_block` | Question is in Q260–Q290 (these condition the prompt; using one as a prediction target would leak the answer) | 40 |
| 3. `derived_variant` | WVS-shipped recode/duplicate of another selected item (e.g. `Q172R` duplicating `Q172`) | 13 |
| 4. `manually_excluded` | Passed automated screening but excluded on methodological grounds (see below) | 9 |
| 5. `scale_too_small` | Fewer than `MIN_RESPONSE_SCALE_SIZE = 4` response options | 83 |
| 6. `non_ordinal_codes` | Response codes not a contiguous integer run | 1 |
| 7. `observed_off_scale` | A value in the data falls outside the codebook's valid codes for that item | 0 |
| 8. `high_missingness` | Missingness exceeds `MAX_MISSINGNESS_PCT = 10` (%) | 63 |
| 9. `low_entropy` | Modal entropy below `MIN_MODAL_ENTROPY = 0.15` (near-unanimous items) | 0 |
| **Passed all screens** | | **144** |

**373 = 144 selected + 229 rejected.** (Ledger confirmed to sum exactly.)

**The 9 manual exclusions**, each checked against actual response labels in the codebook (not assumed from a contiguous-code-range heuristic, which is necessary but not sufficient for true ordinality):

- **Q144** — "Respondent was victim of a crime" — a binary factual/behavioral recall item, not a values/attitude item. Does not belong in a fidelity-of-opinion target set regardless of its passing scale checks.
- **Q152–Q157** (6 items, the WVS postmaterialism battery) — respondents choose from an **unordered menu** of national/personal goals (e.g. "high economic growth" vs. "strong defence forces" vs. "more say in decisions"). The codes name alternatives, not degrees, so MAE/Wasserstein-1 (which assume meaningful distance between adjacent codes) are undefined on them, despite passing the contiguous-code check.
- **Q221, Q222** — vote-frequency items where codes 1–3 form a real frequency scale (Always/Usually/Never) but code 4 means "not allowed to vote" — an eligibility status, not "more extreme than never voting."

**Item metadata is fully auditable:** `data/processed/selected_items.json` contains, for every one of the 144 selected items, its question wording, thematic block, nominal scale size, observed category count, missingness %, and modal entropy.

### 2.4 Fold construction

`src/data/build_folds.py`: **5-fold stratified cross-validation** (`N_FOLDS = 5`), stratified jointly on urban/rural × income tercile, `random_state = RANDOM_SEED = 42`. Verified directly against the output file: no respondent appears in more than one fold's test set; union of all five test sets equals the full 1,692; train/test sizes per fold ≈1,354/338.

### 2.5 What is and is not version-controlled

Per WVSA's non-redistribution clause: the raw CSV, the codebook PDF, and the cleaned respondent-level parquet (`ind_wvs7.parquet`, still individual microdata even after recoding) are **gitignored**. The derived, non-microdata artifacts — `selected_items.json`, `folds.json` (respondent-ID index assignments, no actual survey answers), and the parsed `wvs7_codebook.json` (public documentation, not survey responses) — **are** committed, since they contain no redistributable respondent data.

---

## 3. Baselines (Phase 2)

Established **before any LLM inference**, per the project's own design rule: know the bar an LLM has to clear before spending API calls.

Implemented in `src/eval/baselines.py`, orchestrated in `src/eval/run_baselines.py`, run out-of-fold across **all 144 selected items × 5 folds**:

| Baseline | Mechanism | Accuracy | 95% CI | Mean MAE |
|---|---|---|---|---|
| Gradient boosting | `GradientBoostingClassifier`, demographics → answer, fit per fold | **48.2%** | [45.8, 50.5] | 1.13 |
| Logistic regression | `LogisticRegression`, same features | 46.6% | [44.0, 49.2] | 1.24 |
| Demographic-cell lookup | Empirical answer distribution per demographic cell, falls back to national marginal for unseen cells | 34.5% | [32.3, 36.8] | 1.51 |
| National marginal | Overall answer distribution, **survey-weighted by `W_WEIGHT`** | 32.8% | [30.6, 35.1] | 1.55 |
| Uniform random | Equal probability across all valid options | 19.4% | [18.3, 20.5] | 2.11 |

**Demographic features used:** `H_URBRURAL`, `Q288` (income), `Q275R` (education, 3-band), `Q260` (sex), `Q262` (age), `N_REGION_ISO` (region, cast to categorical string).

**Weighting decision:** the national-marginal baseline is `W_WEIGHT`-weighted because it represents a population-level claim ("what % of India believes X"). Respondent-level predictions (baselines and LLM alike) are deliberately left **unweighted** — weighting would double-count the demographic information already conditioning the prediction.

**95% confidence intervals:** bootstrap, `N_BOOTSTRAP_RESAMPLES = 1000` resamples, `BOOTSTRAP_CI_LEVEL = 0.95`, implemented in `src/eval/bootstrap.py`.

**Fairness caveat, stated explicitly:** the supervised baselines (logistic regression, gradient boosting) are fit *directly* to this population's actual demographic↔answer correlations via cross-validation. An LLM prompted zero-shot has no such access — it reasons from general world knowledge only. Falling short of these two baselines was never a realistic bar for zero-shot prompting and is not, by itself, evidence against silicon sampling.

**Matched-feature baselines, added and completed 2026-08-28:** the table above uses a 6-column demographic subset (`H_URBRURAL`, `Q288`, `Q275R`, `Q260`, `Q262`, `N_REGION_ISO`), which is *not* the same information the LLM's P2 prompt actually sees (14 attributes — see §4.1). `logistic_matched`/`gbm_matched` variants give the supervised baselines the full 14-attribute set instead:

| Baseline | Accuracy | 95% CI | Mean MAE |
|---|---|---|---|
| **Gradient boosting, matched (14 attrs)** | **50.1%** | [47.6, 52.5] | 1.07 |
| Gradient boosting, original (6 attrs) | 48.2% | [45.8, 50.5] | 1.13 |
| Logistic regression, matched (14 attrs) | 46.9% | [44.3, 49.5] | 1.21 |
| Logistic regression, original (6 attrs) | 46.6% | [44.0, 49.2] | 1.24 |

**Effect of matching the feature set is real but modest:** GBM gains 1.9 points (48.2→50.1%) from the extra 8 attributes; logistic regression is essentially flat (+0.3 points) — it was already saturating on the 6-column set and the added attributes (marital status, employment, occupation, social class, religion, town size, language, no. of children) mostly don't add independent linear signal once urban/rural, income, education, sex, age, and region are already in. The zero-shot LLM's gap to the population-fit ceiling is therefore slightly *larger* than the original 6-column comparison suggested, not smaller — the critique that prompted this check was valid, and the correction moves the ceiling further from the LLM's ~24–29% accuracy, not closer.

---

## 4. Prompting and Verbalization

### 4.1 Conditions tested

| Condition | Description |
|---|---|
| **P0** | No demographic information at all |
| **P1** | Minimal (age, sex, region only) — built, not extensively tested |
| **P2** | Full structured profile — **14 attributes**: sex, age, marital status, education, employment, occupation, social class, income decile, religion, urban/rural, region, town size, interview language. **The project's primary/stated condition.** |
| **P3** | The same information rendered as first-person naturalistic backstory prose, not a bulleted list |

### 4.2 Verbalization

`src/prompts/verbalize.py` builds every demographic string and every item's answer-option text **programmatically from the same codebook metadata used for item selection** — no hand-typed labels. A respondent's raw numeric codes (e.g. `Q275 = 6`) are mapped to their official codebook label ("Bachelor or equivalent (ISCED 6)") at verbalization time.

**Region labels — found wrong, now fixed (2026-08-28):** WVS-7's official annex maps 8-digit `N_REGION_ISO` codes to India's 8 sampled states/territory. An earlier version of this table did not locate the annex (it appeared, from the extracted codebook text, to say only "country-specific list of codes in Annex" with no annex included) and substituted a guessed macro-zone scheme (North/South/East/etc.). On locating the actual annex (`data/raw/WVS7_Codebook_Variables_report_V6.0.pdf`, p.227, "INDIA" block), **every one of those 8 guessed labels was wrong** — e.g. code `356028` is **Uttar Pradesh**, not "South zone." The table has been corrected to the verified annex values (356004 Bihar, 356008 Haryana, 356015 Maharashtra, 356021 Punjab, 356025 Telangana, 356028 Uttar Pradesh, 356029 West Bengal, 356034 Delhi).

**Disclosure:** every P1/P2/P3 prediction collected before this fix (all results in §7 below) was prompted with the **wrong** region text for the `region` attribute — e.g. a Uttar Pradesh respondent's prompt said "South zone" instead of "Uttar Pradesh." This does not affect the *subgroup grouping* used in fidelity-gap analysis (that groups on the raw `N_REGION_ISO` code, computed independently of the label text — see `src/eval/subgroups.py`), only what the LLM was actually *told* about the respondent's region as one of 14 demographic attributes. Net effect: the region attribute's contribution to P2/P3 prompts was noise rather than signal throughout this project to date. A fix-and-rerun is listed in §12 as follow-up work, not yet done (would cost fresh API quota on both providers).

**India's language variable is unusable as recorded:** both `S_INTLANGUAGE` and `LNGE_ISO` are constant (`hi`/Hindi) across all 1,692 respondents, with no second language field anywhere in the 613 columns. Verified directly, not assumed. A language-based subgroup axis — relevant given India's linguistic diversity — cannot be tested from this data as released.

### 4.3 Answer parsing

`src/inference/prompting.py` — shared across every model provider (critical for a fair cross-model comparison; using different closing instructions per provider would confound any accuracy difference with wording differences, not model capability). The closing instruction appended to every prompt:

> *"This is anonymized survey-simulation research. Respond with exactly one character: one of {valid options}. No words, no punctuation, no explanation — just that single digit."*

This exact wording was arrived at after an earlier, weaker instruction ("Answer with ONLY the number") produced a **13% refusal/unparsed rate** in an early large run; the current wording drove that to **0.0%** in every subsequent controlled test.

A model's free-text reply is parsed by exact match, then first-token match, then numeric-index interpretation; anything that matches none of these is recorded as an honest **refusal** — never silently coerced into a guess.

---

## 5. Inference Infrastructure

### 5.1 Providers and models

| Provider | Model | Why |
|---|---|---|
| Google AI Studio (free tier) | `gemini-3.1-flash-lite` | Original default (`gemini-2.5-flash`) was retired for new API keys; `3.5-flash-lite`'s daily quota (500/day) was exhausted early, `3.1-flash-lite` chosen as a working alternative with separate quota |
| Groq (free tier) | `openai/gpt-oss-120b` | OpenAI's own open-weight 120B model, hosted free by Groq — chosen after the originally-targeted `llama-3.3-70b-versatile` was found to no longer exist on Groq's live model catalog |

Both integrated via `src/inference/gemini.py` and `src/inference/groq.py`, sharing `CachedInferenceRunner` (`src/inference/base.py`) for caching and `prompting.py` for identical instructions/parsing.

### 5.2 Exact inference parameters

| Parameter | Gemini | Groq |
|---|---|---|
| Temperature | 0.0 | 0.0 |
| Max output tokens | 100 | 150 (low) / 700 (medium) / 3000 (high), by `reasoning_effort` |
| Pacing (requests/min) | 12 | 18 (low) / 10 (medium) / 3 (high) |
| `reasoning_effort` (Groq only) | n/a | low (default; medium/high tested, see §7.4) |
| Real per-token logprobs | Requested, not available at this tier (`real_logprob_rate = 0.0` confirmed on every run) | Not available on chat completions; not requested |

**Why the token budgets are what they are:** both providers' models were found empirically to spend output tokens on internal "thinking" before the visible answer. At low budgets, this truncates the response before the model ever emits the answer character (a hard failure, not a graceful refusal). Measured reasoning-token cost on this task: Groq `low` ≈11 tokens, `medium` ≈82–276, `high` ≈300–2,300+ (highly variable, and impractically expensive to batch-test on free-tier token/minute limits — see §7.4).

**Rate limits, verified empirically via live response headers, not assumed from documentation:**
- Groq `gpt-oss-120b`: 1,000 requests/day, 8,000 tokens/minute.
- Gemini `gemini-3.1-flash-lite`: ~500 requests/day (confirmed via the user's own AI Studio dashboard), 15 requests/minute (empirically triggered a 429 on the 6th call in a burst test before pacing was added).

### 5.3 Caching and resumability

Every prediction is cached to `results/cache/<model>_<respondent>_<item>_<condition>.json` **before** anything else touches it — killing and restarting a run resumes without re-spending API calls on already-answered pairs.

**Real bug found and fixed:** the original caching logic treated *any* cached result — including a failed one (timeout, quota error, network drop) — as final. A resumed run would silently re-serve "no answer" forever for a prediction that failed only because of a transient network blip, never retrying it. Fixed in `run_trackA.py`: a cached result is only treated as final if it either succeeded or failed for a durable reason; failures matching `504`, `deadline exceeded`, `timeout`, `429`, `quota`, `503`, or `connection` are treated as retryable on the next invocation.

---

## 6. Complete Bug List

Every bug found during this project, in the order discovered, all fixed and committed:

1. **`recode_missing_values()` undercount** — logged only the last column's recode count due to a leaked loop variable, not the true total (~6.9M cells).
2. **`filter_by_country()` silent failures** — bare `assert`/silent empty return gave no explanation on a wrong-version file; replaced with actionable `KeyError`/`ValueError` messages.
3. **`is_ordinal_scale()` dtype bug** — the item-selection fallback heuristic required `isinstance(v, int)`, but every numeric column was `float64` after NaN-recoding, silently rejecting 354 of 373 candidates regardless of actual quality. This was the root cause of an earlier (now-superseded) 18-item selection.
4. **Hardcoded scale-type metadata was wrong for 8 of 18 items** in the pre-fix whitelist (e.g. `Q144` labeled a 5-point political item, actually a binary factual item) — resolved by moving to codebook-grounded screening (§2.3).
5. **`src/inference/__init__.py` eager imports** — importing the package pulled in `torch`/`transformers` even when only using the mock or Gemini path, crashing environments without those installed. Fixed with lazy `__getattr__`-based imports.
6. **Report showed raw predicted code instead of resolved label** — "Model predicted: 1" next to "Actual answered: Not at all frequently" — technically correct, visually incoherent. Fixed to resolve both sides through the same label-lookup path.
7. **CSS stat-grid layout bug** — a hairline-border grid trick left a solid color block when tile count didn't evenly fill the last `auto-fit` row; fixed with an explicit column count.
8. **`run_baselines.py` stale return statement** — `NameError` from a leftover variable reference after a refactor.
9. **`pyproject.toml` packaging failure** — `uv sync` failed outright (`hatchling` couldn't determine what to ship, since the declared package name didn't match the `src/` layout). Fixed with an explicit `[tool.hatch.build.targets.wheel] packages = ["src"]`; also declared `requests` as an explicit dependency (used directly by `groq.py`, previously only present transitively).
10. **`format_p0_control()` / `format_p1_minimal()` / `format_p3_naturalistic()` crashed on the full demographics dict** — each accepted only a subset of the ~14 keys `build_prompt()` always forwards; P0 had literally never been run before hitting this. Fixed with `**_ignored` catch-alls on all three.
11. **Groq model name containing `/` broke cache file paths** — `openai/gpt-oss-120b` was used literally in a filename, attempting to write into a nonexistent `openai/` subdirectory. Fixed by sanitizing `/` → `_`.
12. **Unconditional real-logprobs request could have zeroed out every real prediction** — the original design tried `response_logprobs=True` on every call; on a model/tier that doesn't support it, the whole call (including the answer text) failed, not just the probability request. Fixed to detect support once per runner instance and cache the result, so a real answer is never lost to an unsupported optional feature.
13. **Doubled API usage from the above fix's first version** — even after fixing #12, every call still made 2 requests (failed logprobs attempt + retry) until support-detection was cached across the runner's lifetime, not re-probed per call.
14. **Resumable-cache treating transient failures as permanent** — described in §5.3.
15. **Retry-loop script bug (orchestration script, not core codebase)** — an early self-healing retry wrapper's network-down branch called `continue` without incrementing the attempt counter, causing an unbounded (if harmless) retry loop instead of the intended bounded one.

---

## 7. Experiments Run, Chronologically

### 7.1 Mock-model pipeline validation (no API key required)

`src/inference/mock.py` — a network-free fake model, run first specifically to catch integration bugs (#5, #6, #7 above) before spending any real API quota. 20 respondents × 5 items = 99 predictions, fully validated end-to-end including report rendering.

### 7.2 First real run (superseded)

`gemini-3.5-flash-lite`, P2, old (weaker) prompt instruction: 1,122 attempted, 976 answered (13.0% refusal rate), 25.8% accuracy. Superseded by the prompt-instruction fix (§4.3) and the model switch (§5.1); numbers from this run are not used in any current finding.

### 7.3 Six-condition pilot (n=99 each, 20 respondents × 5 items, identical items throughout)

| Technique | Accuracy | MAE | Within ±1 | Refusals |
|---|---|---|---|---|
| Gemini 3.1 Flash Lite — P0 (no demographics) | 31.3% | 1.14 | 65.7% | 0.0% |
| Groq gpt-oss-120b — P3 (naturalistic backstory) | 30.3% | 0.96 | 76.8% | 0.0% |
| Gemini 3.1 Flash Lite — P2 (full demographics) | 29.3% | 1.07 | 70.7% | 0.0% |
| Groq gpt-oss-120b — P0 (no demographics) | 29.3% | 1.07 | 70.7% | 0.0% |
| Groq gpt-oss-120b — P2, `reasoning_effort=medium` | 29.3% | 1.06 | 70.7% | 0.0% |
| Groq gpt-oss-120b — P2, `reasoning_effort=low` | 27.3% | 1.00 | 74.7% | 0.0% |

**Pilot finding:** all six configurations land in a tight 27.3–31.3% band. **P0 and P2 are statistically indistinguishable on both models independently** — i.e., a full 14-attribute demographic profile did not measurably outperform no demographic information at all. This motivated scaling P2 specifically (the condition with a substantive claim to test) rather than continuing to search across prompt conditions.

### 7.4 Reasoning-effort exploration (Groq only)

- `low`: ~11 reasoning tokens/call, reliable, cheap. **Used for all subsequent runs.**
- `medium`: ~82–276 tokens/call, no meaningful accuracy change over `low` (29.3% vs. 27.3%, within noise).
- `high`: ~300–2,300+ tokens/call, highly variable, impractical to batch-test within the 8,000 tokens/minute limit (would cap throughput at ~3 calls/minute). Not pursued further.

### 7.5 Properly-powered two-model run, 5 items (n=457 / n=909)

| Model | n | Accuracy | 95% CI | MAE | Refusal rate |
|---|---|---|---|---|---|
| Gemini 3.1 Flash Lite | 457 | 28.9% | [25.0, 32.9] | 1.15 | 0.2% |
| gpt-oss-120b (Groq) | 909 | 28.7% | [25.9, 31.6] | 0.99 | 0.0% |

**Subgroup fidelity gaps (best category accuracy − worst, restricted to n≥30 categories):**

| Axis | Gemini gap | Groq gap | Replicates? |
|---|---|---|---|
| Region zone | 18.3 pts | 18.4 pts | Yes — near-exact |
| Age band | 23.7 pts | 15.9 pts | Yes, both large |
| Income tercile | 9.1 pts | 5.6 pts | Same direction |
| Education band | 7.1 pts | 5.2 pts | Same direction |
| Urban / Rural | 3.8 pts | 3.2 pts | Yes — smallest in both |
| Sex | 11.0 pts | 0.9 pts | **No** |

### 7.6 Item-widening run, 15 items — both models now fully converged

Both providers re-run on the same 15 items (Groq: n=556, 40 respondents; Gemini: n=421, 30 respondents, a strict subset of Groq's 40) to test whether §7.5's gaps are stable as item coverage widens. **Gemini's run finished converging on 2026-08-28: 421/421 predictions, 0 failures.**

| | Gemini (15 items) | Groq (15 items) |
|---|---|---|
| n | 421 | 556 |
| Accuracy | 24.9% | 23.5% (on the 421-row Gemini-overlap subset; 23.7% on its full n=556) |
| 95% CI | [20.9, 28.7] | [20.3, 27.0] |
| MAE | 1.66 | ~1.6 (comparable) |
| Refusal rate | 0.0% | 0.0% |
| Beats national-marginal baseline | 9/15 items | — |
| Beats demographic-cell baseline | 8/15 items | — |

**Subgroup fidelity gaps, 15 items, both models (region labels are the corrected, actual-annex values from §4.2 — but see the disclosure there: the *prompt text itself* used the old wrong labels for every run below; grouping is unaffected):**

| Axis | 5-item Gemini | 5-item Groq | 15-item Gemini | 15-item Groq | Stable across width? | Replicates across models at 15 items? |
|---|---|---|---|---|---|---|
| Region | 18.3 | 18.4 | **26.4** | **6.3** | No (Gemini grew, Groq shrank) | Direction yes, magnitude no |
| Age band | 23.7 | 15.9 | 9.2 | **16.7** | Groq stable; Gemini shrank | Direction yes, magnitude no |
| Education | 7.1 | 5.2 | 14.3 | 3.4 | No | Direction yes, magnitude no |
| Income | 9.1 | 5.6 | 14.2 | 2.1 | No | Direction yes, magnitude no |
| Sex | 11.0 | 0.9 | 11.1 | 5.6 | Gemini stable; Groq grew | Direction yes, magnitude no |
| Urban/rural | 3.8 | 3.2 | 8.9 | 4.3 | Roughly stable | Direction yes, magnitude no |

**Revised central finding, now that both models are fully converged at 15 items:** the earlier draft of this report (based on Groq alone plus a partial Gemini run) treated age-band as "the one finding that survived a second model and a wider item sample," with region as a false lead. **With Gemini's full 15-item data in hand, that specific claim does not hold up as cleanly** — Gemini's age gap actually *shrank* from 5 to 15 items (23.7 → 9.2), while Groq's grew (15.9 → 16.7); region shows the same pattern in reverse. **No single axis is quantitatively stable across both the model swap and the item-width increase.**

What *does* hold, robustly, across every one of the four (model × item-width) cells: **every axis shows a nonzero, non-trivial gap in every condition tested.** The qualitative claim — silicon sampling fidelity is uneven across demographic subgroups — replicates completely. The quantitative claim — *which* axis has the largest gap, and by how much — does not. This is a more conservative, more defensible finding than the single-axis claim in the earlier draft, and is the one now used in the paper and both published pages.

### 7.7 Cross-model agreement analysis (`src/eval/cross_model_agreement.py`)

On the 421 (respondent, item) pairs both models answered: **raw agreement 41.6%**, Cohen's κ = **0.25** (fair agreement, well above chance but far from consensus). Breaking down the 41.6%: only **10.7%** of all pairs are cases where the models agree *and are both correct*; **30.9%** are cases where they agree *and are both wrong* — i.e., most of the models' agreement is agreeing on the same wrong answer (very likely both defaulting to the same modal response option on hard items), not converging on truth.

Per-item agreement ranges from 14.3% (Q160, "science vs. faith") to 82.1% (Q4, "important in life: politics" — also the item with the highest accuracy for both models, suggesting some items are genuinely easier/harder for *any* LLM, not just one). Per-subgroup agreement is fairly flat (33–52% across every category on every axis) — the two models don't diverge from each other more or less within any particular demographic subgroup, which weakly argues against "one model is subgroup-biased and the other isn't" as an explanation for the fidelity-gap non-replication in §7.6; it looks more like independent noise than a systematic model-specific bias.

### 7.8 Item-level difficulty (`src/eval/item_difficulty.py`)

Hardest items (mean accuracy across both models): **Q160** ("science vs. faith," 10.5% mean), **Q32** ("housewife just as fulfilling," 16.4%), **Q29** ("men make better political leaders," 19.8%), **Q161** ("science breaks down right/wrong," 21.5%), **Q31** ("men make better business executives," 21.7%). Easiest: **Q4** ("important in life: politics," 35.9%), **Q169** ("science vs. religion, religion right," 33.4%), **Q170** ("only acceptable religion is mine," 29.6%). Four of the five hardest items are the gender-role and science/faith attitude items — plausibly items where an LLM's own trained-in priors compete most directly with the actual distribution of Indian survey responses, rather than items where a demographic profile alone should carry much signal.

### 7.9 Sex-gap investigation (`src/eval/sex_gap_investigation.py`)

Checked whether the sex-axis gap (§7.6) concentrates in the three explicit gender-attitude items (Q29/Q31/Q32) — i.e., whether it's really a "the model stereotypes by respondent sex on gender questions" effect rather than a general pattern. **It is not concentrated there.** Mean male-minus-female accuracy gap on the 3 gender-attitude items: Gemini 13.4 pts, Groq 11.7 pts. On the other 12 items: Gemini 10.1 pts (nearly as large), Groq 3.8 pts (much smaller). The gap is broad-based on Gemini (present on religion/science items — Q169, Q170 — as much as on gender items) but more concentrated on Groq. **This rules out the simplest "gender-stereotyping-on-gender-questions" explanation as the sole driver** and points instead toward a more general pattern where male respondents are predicted more accurately across many item types on Gemini specifically — worth a dedicated follow-up but not resolved by this pass.

---

## 8. Free-Tier Accuracy Ceiling — What Was and Wasn't Tried

Every legitimate lever available on free-tier infrastructure was tried and plateaued in the same band:
- **Model size:** Groq's 120B model did not outperform Gemini's smaller "lite" tier.
- **Reasoning effort:** low/medium/high tested; no meaningful gain beyond `low`, and `high` is impractical on free-tier token budgets.
- **Prompt condition:** P0/P2/P3 all within 3 points of each other on both models.

**Not tried, and explicitly why:**
- **Fine-tuning (Track B):** the one lever most likely to close the gap to the supervised baselines. Blocked on Kaggle credentials (requested from the user, not yet received) and deliberately out of scope for this pass by earlier explicit instruction.
- **Full 1,692 × 144 sweep:** at free-tier daily quotas (~500–1,000 requests/day/model), a full sweep is ~243,648 predictions — weeks of continuous free-tier usage, not achievable in this project's timeframe.
- **A larger commercial/frontier model (e.g. GPT-4-class):** none used; every model tested here is free-tier or open-weight.

**Explicit refusal on record:** at one point the accuracy target requested was 70%+. This was not pursued past honest testing because every real lever available plateaued at 27–31%, and manufacturing a higher number (via metric-switching, item cherry-picking, or prompt-tuning against known test answers) was identified and explicitly declined as scientifically illegitimate, per the user's own stated requirement that this project be presentable and defensible for publication.

---

## 9. Published Artifacts

Two live, repeatedly-updated pages (not static — republished at each real milestone, same URLs throughout):

- **Project status/audit:** https://claude.ai/code/artifact/8ee194e7-b11a-495c-ae2f-815aa38c5676 — completion %, phase breakdown, file-by-file pipeline map, live run status when applicable.
- **Interim findings (professor-facing):** https://claude.ai/code/artifact/912449fa-4e97-4260-bfc5-f6e03eeb5301 — thesis, method, full results, the item-widening finding, limitations, next steps.

A working paper draft (`paper/DRAFT.md`, committed to the repo) contains: Abstract, Introduction, Related Work, Data, Method, Results (§7.5–7.6 in full), Discussion, Limitations, Conclusion, Next Steps, References.

---

## 10. Repository Map

| Path | Contents |
|---|---|
| `src/config.py` | Every constant: paths, model registry, CV settings, item-selection thresholds, bootstrap settings |
| `src/data/` | `load_wvs.py`, `parse_codebook.py`, `select_items.py`, `build_folds.py` |
| `src/prompts/` | `verbalize.py` (demographic/item text generation), `templates.py` (P0–P3) |
| `src/inference/` | `base.py` (caching), `mock.py`, `gemini.py`, `groq.py`, `prompting.py` (shared instruction/parsing), `run_trackA.py` (orchestrator), `hf_local.py` (built, untested — local open-weight models) |
| `src/eval/` | `metrics.py`, `baselines.py`, `run_baselines.py`, `bootstrap.py`, `subgroups.py` |
| `src/report/` | `evaluate_run.py`, `comparison_report.py` |
| `data/reference/wvs7_codebook.json` | Parsed codebook, 353 questions, committed (documentation, not microdata) |
| `data/processed/` | `selected_items.json`, `folds.json` — committed. `ind_wvs7.parquet` — gitignored (microdata) |
| `data/raw/` | Raw CSV + codebook PDF — gitignored |
| `results/` | All predictions, evaluations, reports, cache — gitignored entirely (contains real respondent demographics + answers) |
| `paper/DRAFT.md` | Working paper draft |
| `notebooks/kaggle_trackB_finetune.ipynb` | Pre-existing Track B scaffolding, not yet run |
| `DATA_ACQUISITION.md`, `PHASE_A_READY.md`, `PHASE_A_FILE_MAP.md` | Earlier-phase documentation, still accurate for what they cover |

---

## 11. Completion Status — Exact Breakdown

Weighted equally across 6 counted phases (Phase 4 excluded, per explicit instruction to defer fine-tuning without penalizing progress for it):

| Phase | % | Basis |
|---|---|---|
| 0–1: Data + item selection | 100% | Fully verified, auditable, reproducible |
| 2: Baselines | 100% | Full 144-item, 5-fold, out-of-fold sweep, real numbers, plus matched-feature supervised baselines added 2026-08-28 (§3) |
| 3: Track A (zero-shot inference) | 100% | Both models fully converged at both 5 and 15 items, 0 failures |
| 4: Fine-tuning (Track B) | excluded | Kaggle token received 2026-08-28, not yet used; deliberately sequenced after Track A |
| 5: Subgroup analysis | 100% | Full cross-model analysis at 5 and 15 items done; cross-model agreement, item-difficulty, and sex-gap follow-up analyses added 2026-08-28 (§7.6–7.9) |
| 6: Paper / writeup | ~80% | All standard sections drafted with real content; final 15-item cross-model numbers now incorporated; citation/bibliography polish still outstanding |
| 7: Capstone deck / submission | 0% | Explicitly excluded from this assistant's scope by instruction; not started |

**Weighted average (phases 0–6, phase 4 excluded per instruction): ~90%.** The jump from ~70% reflects Track A and the subgroup analysis both reaching full completion, not a redefinition of what "complete" means for either.

---

## 12. What Is Left — Explicit List

1. ~~Finish Gemini's 15-item run~~ — **done 2026-08-28**, 421/421, 0 failures (§7.6).
2. ~~Investigate the non-replicating sex gap~~ — **done 2026-08-28** (§7.9): not concentrated in the 3 explicit gender-attitude items; broader on Gemini than on Groq. Root cause not fully resolved, flagged as a genuine open question rather than a solved one.
3. ~~Verify region-zone labels against the official WVS-7 annex~~ — **done 2026-08-28**, and the original reconstruction was found to be **wrong on every code** (§4.2). Fixed in code; disclosure recorded that all predictions collected to date used the incorrect region text in-prompt.
4. **Re-run P2/P3 inference with the corrected region labels** — new follow-up item created by #3. Not done yet (costs fresh API quota on both providers, ~1,000 requests total to redo the 15-item runs). Worth doing before final submission if quota allows; the corrected labels are already live for any new runs.
5. **Track B fine-tuning** — Kaggle token now on file (received 2026-08-28); not yet used. Would test whether closing the accuracy gap to the supervised baselines also narrows or widens the subgroup fidelity gaps.
6. **Widen beyond 15 items** toward the full 144 — feasible only across many days at free-tier daily quotas, not a single sitting.
7. **Paper polish** — citation formatting, a full bibliography beyond the three sources currently listed, and incorporating the cross-model-agreement, item-difficulty, and sex-gap findings (§7.7–7.9) into the Results/Discussion sections.
8. ~~Matched-feature supervised baselines~~ — **done 2026-08-28** (§3): GBM improves 48.2%→50.1% with the full 14-attribute set; logistic regression essentially unchanged. The LLM's gap to the population-fit ceiling is modestly larger than originally reported, not smaller.
9. **Capstone deck / submission materials** — not started, explicitly outside this assistant's scope per instruction.
