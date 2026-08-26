# Phase A file map — what does what, and where to find it

Scope: **Phase A only** (Phase 0 data prep → Phase 1 item/fold setup → Phase 2
baselines → Phase 3 Track A zero-shot inference → reporting). Phase 4
(fine-tuning) and the paper/deck are out of scope here.

If you just want to run it, see `PHASE_A_READY.md`. This file is the map of
*why each file exists and what it's responsible for* — read it when you want
to change something, not just run something.

---

## The pipeline, in the order data actually flows

```
WVS-7 CSV + codebook PDF
        │
        ▼
 src/data/load_wvs.py ─────────────► data/processed/ind_wvs7.parquet
        │                             (1,692 India respondents, 613 cols)
        ▼
 src/data/parse_codebook.py ───────► data/reference/wvs7_codebook.json
        │                             (353 questions: wording + scale + labels)
        ▼
 src/data/select_items.py ─────────► data/processed/selected_items.json
        │                             (144 items that passed every criterion)
        ▼
 src/data/build_folds.py ──────────► data/processed/folds.json
        │                             (5-fold stratified CV assignment)
        ▼
 src/eval/run_baselines.py ────────► results/baselines_summary.csv
        │                             results/baselines_by_item.csv
        │                             (the bar an LLM has to clear)
        ▼
 src/inference/run_trackA.py ──────► results/predictions/<model>_<condition>.parquet
        │    (uses prompts/verbalize.py + prompts/templates.py           │
        │     + inference/{mock,gemini,hf_local}.py under the hood)      │
        ▼
 src/report/evaluate_run.py ───────► results/evaluated/<run>/*.csv
 src/report/comparison_report.py ──► results/reports/<run>.html
                                      (the report you actually read)
```

---

## `data/` — the datasets and reference material

| Path | What it is |
|---|---|
| `data/raw/WVS_Cross-National_Wave_7_csv_v6_0.csv` | The real WVS-7 microdata download, 97,220 respondents across 66 countries. Gitignored (WVSA terms forbid redistribution). |
| `data/raw/WVS7_Codebook_Variables_report_V6.0.pdf` | The official 404-page WVS-7 variable dictionary. Source of truth for every question's wording and response scale. |
| `data/reference/wvs7_codebook.json` | The PDF above, parsed into structured JSON by `parse_codebook.py`. 353 questions, each with title, wording, valid response codes/labels, and thematic block. This is what every other module reads instead of re-parsing the PDF. |
| `data/processed/ind_wvs7.parquet` | Cleaned India-only respondent table: missing codes (-1 to -5) recoded to NaN, filtered to `B_COUNTRY_ALPHA == "IND"`. 1,692 rows. |
| `data/processed/selected_items.json` | The 144 survey questions chosen as prediction targets, with full audit trail (missingness %, entropy, why each of the 373 candidates that *didn't* make it was rejected). |
| `data/processed/folds.json` | The 5-fold train/test respondent-ID split used for out-of-fold baseline evaluation. |

**If you want to know what a question actually asked or what its answer
options mean**, go to `data/reference/wvs7_codebook.json` — every module in
this project reads labels from there, nothing is hand-typed.

---

## `src/config.py` — every constant in one place

Paths (`DATA_RAW`, `DATA_PROCESSED`, `DATA_REFERENCE`, `RESULTS_DIR`), the
model registry (`MODELS` dict — HF model IDs, quantization settings), the
four prompt condition names (`PROMPT_CONDITIONS`), CV settings (`N_FOLDS=5`,
`RANDOM_SEED=42`), item-selection thresholds (`MAX_MISSINGNESS_PCT`,
`MIN_MODAL_ENTROPY`, `MIN_RESPONSE_SCALE_SIZE`), and bootstrap settings
(`N_BOOTSTRAP_RESAMPLES=1000`, `BOOTSTRAP_CI_LEVEL=0.95`).

Also holds `resolve_wvs_csv()` — finds the right raw CSV in `data/raw/`
regardless of exact filename, and fails with an actionable message (not a
silent wrong-file load) if it can't.

**Change a threshold or add a model here first** — nothing downstream
hardcodes these values independently.

---

## `src/data/` — turning the raw survey into something usable

| File | Responsibility |
|---|---|
| `load_wvs.py` | Loads the raw CSV, recodes WVS missing-value codes to NaN, filters to India, sanity-checks the resulting N against the expected 1,692, saves the parquet. Run via `python -m src.data.load_wvs --country IND`. |
| `parse_codebook.py` | Extracts every question's wording, response labels, and thematic block from the codebook PDF using regex over the extracted text. This is what makes item selection *codebook-grounded* instead of hand-guessed. Run via `python -m src.data.parse_codebook`. |
| `select_items.py` | The item-selection screen. For every `Q*` column: rejects it if there's no codebook entry, if it's in the demographic block (Q260-Q290 — those condition prompts, they can't also be prediction targets), if it's a WVS-shipped derived/recoded duplicate, if it's on the manual exclusion list (nominal or mixed-scale items — see below), if its scale is too small or non-contiguous, if any observed value falls outside the codebook's valid codes, if missingness exceeds 10%, or if entropy is too low. Everything that survives all of that is in the final 144. Run via `python -m src.data.select_items`. |
| `build_folds.py` | Builds the 5-fold stratified CV split, stratifying jointly on urban/rural × income tercile so no fold is accidentally skewed. Verifies no respondent lands in more than one fold's test set. Run via `python -m src.data.build_folds`. |

**The manual exclusion list** (in `select_items.py`, top of file, variable
`EXCLUDED_ITEMS`): `Q144` (crime-victimization — factual recall, not an
attitude), `Q152`–`Q157` (postmaterialism battery — respondents pick from an
*unordered* menu, so there's no valid "distance" between answers), `Q221`/
`Q222` (voting frequency scales where one code means "not eligible to vote,"
not "voted less often than never").

---

## `src/prompts/` — turning a respondent + a question into text

| File | Responsibility |
|---|---|
| `verbalize.py` | Two jobs: `verbalize_demographics(row)` turns one respondent's raw codes (sex, age, education, income, religion, region, etc.) into human-readable strings for the prompt templates below. `verbalize_item(question_id)` turns one survey question into its wording + a numbered list of answer options, pulled straight from the codebook JSON. Also `parse_predicted_code()` — maps a model's free-text answer back onto a valid response code, by exact number match or exact label match; returns `None` (an honest "unparsed," not a guess) if neither matches. |
| `templates.py` | The four prompt-construction functions — `format_p0_control` (no demographics), `format_p1_minimal` (age/sex/region only), `format_p2_structured` (all 14 demographic attributes, bulleted), `format_p3_naturalistic` (first-person backstory prose). `build_prompt(condition, question_text, answer_options, **demographics)` is the one function everything else calls — it picks the right template and assembles the full prompt. |

**If you want to change what a prompt looks like** (wording, format, which
demographics appear), this is the only place to touch. Nothing else builds
prompt text directly.

---

## `src/inference/` — actually asking a model the question

| File | Responsibility |
|---|---|
| `base.py` | `CachedInferenceRunner` — the shared base class. Every prediction is cached to `results/cache/<model>_<respondent>_<item>_<condition>.json` before anything else touches it, so killing a run mid-way and restarting never re-spends an API call on an already-answered question. |
| `mock.py` | `MockInferenceRunner` — makes zero network calls. Returns a plausible-but-meaningless probability distribution so the *rest of the pipeline* (caching, parsing, metrics, reporting) can be proven correct without needing a key. This is what generated `results/reports/MOCK_VALIDATION_mock-demo-model_P2.html`. |
| `gemini.py` | `GeminiInferenceRunner` — the real Gemini 2.5 Flash path. Requests real per-token logprobs from the API (`response_logprobs=True`) and maps them onto the answer options; falls back to a confidence-weighted (not flat-uniform) guess only if the API doesn't return them, and flags which happened (`real_logprobs: true/false`) so the report never silently mixes real and fake confidence. Needs `GOOGLE_API_KEY`. |
| `hf_local.py` | `HFLocalInferenceRunner` — for open-weight models (Llama-3.1-8B, Qwen2.5-7B, Gemma-3-4B) run locally via `transformers`, with 4-bit quantization. Extracts *exact* logprobs from the model's own logits — no API involved, no approximation needed. Needs `torch`/`transformers` installed (not in this environment yet). |
| `run_trackA.py` | **The orchestrator you actually run.** Loads the respondent data, the 144 selected items, and the codebook; for every (respondent, item) pair, verbalizes the demographics + question, builds the prompt, calls the chosen runner, parses the answer, and writes one row per prediction to `results/predictions/<model>_<condition>.parquet`. Also merges in the true subgroup labels (urban/rural, income, etc.) so the report can slice by them. This is the file with the `--model` flag. |

**Picking a model is one flag on `run_trackA.py`** — `get_runner()` at the
top of that file is the only place that branches on model name; adding a
new model means adding one `if` there, nothing else changes.

---

## `src/eval/` — turning predictions into numbers

| File | Responsibility |
|---|---|
| `metrics.py` | The core metric functions: exact-match accuracy, MAE (respects ordinal distance), NLL (needs probabilities), Jensen-Shannon divergence and Wasserstein-1 (distributional metrics — how close is the *distribution* of predicted answers to the real one, not just per-person accuracy). `compute_metrics()` bundles all of these; `fidelity_gap()` and `delta_gap()` are the fairness-across-subgroups numbers the whole study is built around. |
| `baselines.py` | The five things an LLM has to beat: `UniformRandomBaseline`, `NationalMarginalBaseline` (now `W_WEIGHT`-weighted, since it's the one baseline standing in for a population-level claim), `DemographicCellLookup` (empirical answer distribution per demographic cell, falling back to the national marginal for unseen cells), and `SupervisedClassifierBaseline` (logistic regression or gradient boosting on demographics). |
| `run_baselines.py` | **The orchestrator for baselines.** Loops over all 144 items × 5 folds, fits and evaluates all 5 baselines out-of-fold each time, and writes the aggregated results. This produced the numbers in `results/baselines_summary.csv` (the table showing gradient boosting at 48.2% accuracy — the bar to beat). Run via `python -m src.eval.run_baselines`. |
| `bootstrap.py` | `bootstrap_ci()` — resample-with-replacement confidence intervals on any per-row statistic (used everywhere a number is reported, so nothing is presented without its own uncertainty). `bootstrap_diff_ci()` — CI on the *difference* between two paired statistics (e.g., is model A really better than model B, or is that gap noise). |
| `subgroups.py` | `assign_subgroups(df)` — computes the six live subgroup axes (urban/rural, income tercile, education band, sex, age band, WVS region zone) from the raw demographic columns. `metrics_by_subgroup_axis()` — per-category accuracy/MAE/CI, flagging any category under n=30 as underpowered. `fidelity_gap_report()` — best-category-minus-worst-category accuracy per axis, excluding underpowered cells so a tiny group can't manufacture a dramatic-looking gap. |

**Note:** the language subgroup axis doesn't exist here on purpose — every
respondent's interview-language field is constant in the raw data (flagged
in an earlier audit), so there's nothing real to slice on.

---

## `src/report/` — the human-readable output

| File | Responsibility |
|---|---|
| `evaluate_run.py` | Takes a predictions parquet and computes: `overall_metrics()` (headline accuracy + CI, MAE, refusal rate, what fraction of Gemini predictions used real vs. fallback logprobs), `per_item_metrics()` (accuracy per survey question), `versus_baselines()` (lines up the model's per-item accuracy against every Phase 2 baseline's, item by item — this is the number that answers "did silicon sampling actually add anything"). Can be run standalone (`python -m src.report.evaluate_run --predictions ...`) to just dump CSVs, no HTML. |
| `comparison_report.py` | **Builds the actual HTML report.** Headline stat tiles, a bar chart of the model vs. every baseline, a fidelity-gap table by subgroup, full per-category accuracy tables, and — the part built specifically for "convincing, not just a metric" — a gallery of real respondent transcripts: this person's actual WVS answer next to the model's exact predicted answer, color-coded match/miss, with the model's full probability bar chart across every answer option. Self-contained single HTML file, no external dependencies, renders in light or dark mode. Run via `python -m src.report.comparison_report --predictions ... --output ...`. |

**If you want to change what the report looks like or what it shows**, the
whole thing — HTML structure, CSS, the example-card layout — lives in
`comparison_report.py`. `PAGE_TEMPLATE` near the bottom of that file is the
literal page skeleton; the render functions above it (`render_stat_tile`,
`render_example_card`, `render_baseline_comparison`, etc.) each build one
section.

---

## Root-level files (Phase A relevant ones)

| File | What it's for |
|---|---|
| `PHASE_A_READY.md` | The runbook: exact commands to run once you have an API key, what's already proven with the mock model, and design decisions worth knowing before the first real run. |
| `DATA_ACQUISITION.md` | How to get the WVS-7 CSV and codebook PDF from worldvaluessurvey.org — registration steps, exact filenames, version gotchas (why v6.0 and not v5.0). |
| `demo.py` | A synthetic-data walkthrough of Phases 0-2, useful for sanity-checking the pipeline shape without touching real data. Not part of the real run. |
| `pyproject.toml` | Dependency list. `pypdf` was added here for the codebook parser. |

Not covered here (out of Phase A scope): `HANDOFF.md`, `PHASE_3_4_GUIDE.md`,
`PROJECT_STATUS.md` describe the full 15-week plan including fine-tuning
(Phase 4) and the paper — skip them for now per your instruction to leave
fine-tuning aside.

---

## Quick lookup: "I want to change X"

| I want to... | Edit this file |
|---|---|
| Change which survey items are included | `src/data/select_items.py` (thresholds in `src/config.py`) |
| Fix a wrong demographic label in prompts | `src/prompts/verbalize.py` |
| Change what a prompt says | `src/prompts/templates.py` |
| Add a new model to run | `src/inference/run_trackA.py` → `get_runner()` |
| Add a new baseline to compare against | `src/eval/baselines.py` + wire into `run_baselines.py` |
| Add a new metric | `src/eval/metrics.py` |
| Add a new subgroup axis to slice by | `src/eval/subgroups.py` → `assign_subgroups()` |
| Change the report's look or content | `src/report/comparison_report.py` |
| Change a threshold/constant | `src/config.py` |
