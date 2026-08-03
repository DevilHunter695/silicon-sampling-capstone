# Silicon Sampling for Indian Public Opinion: A Fidelity and Failure-Mode Study

**Capstone research + publication pipeline**  
Owner: Bhanuj Bhalla · Timeline: ~15 weeks · Targets: C3NLP, NLP4PI, ACL SRW, arXiv

## Overview

This project measures whether "silicon sampling" — using LLMs conditioned on demographic profiles to simulate survey respondents — works *evenly* across Indian subgroups. Prior validation (Argyle et al. 2023, WorldValuesBench LREC-COLING 2024) has been limited to aggregate figures. We ask: **does fidelity degrade for rural, low-income, low-education, and non-English-interview respondents?**

### Contribution

1. The first **intra-national** fidelity audit of silicon sampling for India, stratified by urban/rural, income, education, region, and language.
2. A **fair-comparison baseline suite** including a demographic-cell lookup table — if an LLM cannot beat a lookup table, silicon sampling adds nothing.
3. The first measurement of whether **fine-tuning narrows or widens** the between-subgroup fidelity gap.
4. A released, reproducible pipeline (code + prompts + eval harness), country-parameterised so others can extend it.

## Data

**Primary:** World Values Survey Wave 7, India. N ≈ 1,692.  
Download from [worldvaluessurvey.org](https://www.worldvaluessurvey.org) → Data & Documentation → Statistical Data Files.

Place the cross-national WVS-7 CSV in `data/raw/`.

**Reuse:** Clone [`demon702/worldvaluesbench`](https://github.com/demon702/worldvaluesbench) for WVS variable↔question mappings and demographic verbalisation.

## Models

| Role | Model | Where |
|---|---|---|
| Track A (closed) | Gemini 2.5 Flash | AI Studio free tier |
| Track A (open) + Track B base | **Llama-3.1-8B-Instruct** | Kaggle T4, QLoRA 4-bit |
| Track A alt | Qwen2.5-7B-Instruct | Kaggle T4 |
| Pipeline debug | Gemma-3-4B-it | Kaggle T4 |

**Design rule:** Track A and Track B use the same base checkpoint (Llama-3.1-8B) to isolate the effect of fine-tuning.

## Method (15-week phased schedule)

### Phase 0: Setup (Wk 1)
- Register + download WVS-7; clone WVB repo
- Scaffold repo; set up `uv` environment
- Literature review on the 3 overlapping papers

**Done when:** WVS CSV loads, India rows = ~1,692 confirmed

### Phase 1: Data (Wk 2–3)
- Parse WVS, recode missing values, select items (§5.1 in handoff)
- Build stratified 5-fold CV split
- Demographic verbalisation

**Done when:** `selected_items.json` frozen (~45 items); fold balance table printed

### Phase 2: Baselines (Wk 3–4) ⭐ *Before any LLM inference*
- Implement all baselines: uniform random, national marginal, demographic-cell lookup, supervised classifier (logistic regression + gradient boosting)
- Compute metrics: exact-match accuracy, MAE, NLL, Jensen–Shannon divergence, Wasserstein-1
- Bootstrap confidence intervals (1,000 resamples)

**Done when:** Cell-lookup and logreg numbers exist; you know what the LLM must beat

### Phase 3: Track A — Zero-shot (Wk 5–7)
- Inference runners with resumable caching
- Logprob extraction over answer-option tokens (logprobs, not sampling)
- Run 3 models × 4 prompt conditions (P0/P1/P2/P3):
  - P0: no demographics (control)
  - P1: minimal (age, sex, region)
  - P2: full structured
  - P3: full naturalistic backstory

**Done when:** Full prediction cache on disk; inference can be resumed after Kaggle session dies

### Phase 4: Track B — Fine-tuning (Wk 8–10)
- QLoRA 4-bit fine-tuning on Kaggle T4
- Training: exactly Track A P2 prompt + respondent's real answer (n ≈ 61k examples per fold)
- 5-fold cross-validation with respondent-level split
- Suggested hyperparams: r=16, alpha=32, lr=2e-4, 2 epochs

**Done when:** Out-of-fold predictions for all N respondents

### Phase 5: Analysis (Wk 11–12)
- Subgroup slicing: urban/rural, income tercile, education band, region, language, age, sex, religion
- Compute **fidelity gap** = (worst slice) − (best slice)
- Compute **Δ_gap = gap(fine-tuned) − gap(zero-shot)**
- Bootstrap CIs on every number
- Contamination probe (§5.7)

**Done when:** Every headline number has a CI and a sample size

### Phase 6: Paper + Release (Wk 13–14)
- 8-page ACL-format write-up
- Release repo with README + reproduction steps

### Phase 7: Capstone + Submit (Wk 15)
- Capstone deck
- arXiv submission
- Workshop submission (C3NLP, NLP4PI, ACL SRW)

## Repository structure

```
CAPSTONE/
├── data/
│   ├── raw/                      # WVS CSV — gitignored
│   └── processed/
│       ├── india_wvs7.parquet    # Parsed, cleaned
│       ├── selected_items.json   # Frozen item list (~45)
│       └── folds.json            # Stratified 5-fold assignment
├── src/
│   ├── config.py                 # Model registry, paths, constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_wvs.py           # Parse, recode missing (-1..-5), filter by country
│   │   ├── select_items.py       # §5.1 item-selection filters
│   │   └── build_folds.py        # Stratified CV split
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── templates.py          # P0/P1/P2/P3 prompt templates
│   │   └── verbalize.py          # Demographics + answer options → text (adapt from WVB)
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── base.py               # Cached, resumable inference interface
│   │   ├── gemini.py             # Gemini 2.5 Flash runner (logprobs)
│   │   └── hf_local.py           # HF Transformers runner (logprobs from logits)
│   ├── finetune/
│   │   ├── __init__.py
│   │   ├── build_dataset.py      # Assemble (prompt, answer) training pairs
│   │   └── train_qlora.py        # QLoRA Kaggle entrypoint
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Accuracy, MAE, NLL, JSD, W1
│   │   ├── baselines.py          # Marginal, cell-lookup, logreg, GBM
│   │   ├── subgroups.py          # Slicing, fidelity gap, Δ_gap
│   │   └── bootstrap.py          # 1000-resample CIs
│   └── report/
│       ├── __init__.py
│       └── make_figures.py       # Matplotlib/Plotly figures for paper
├── notebooks/
│   ├── kaggle_trackA_openweight.ipynb   # Zero-shot inference
│   └── kaggle_trackB_finetune.ipynb     # QLoRA fine-tuning
├── results/                      # Cached predictions + metric tables (gitignored)
├── paper/                        # LaTeX (ACL style)
│   └── main.tex
├── deck/                         # Capstone presentation slides
├── .gitignore
├── README.md                     # This file
├── pyproject.toml               # uv + Poetry
└── REQUIREMENTS.md              # Environment setup
```

## Running the pipeline

### Phase 0–1: Environment + data setup

```bash
# Clone WVB for code reuse
git clone https://github.com/demon702/worldvaluesbench.git wvb_ref

# Activate environment
uv sync

# Download WVS-7 CSV (manual step on worldvaluessurvey.org)
# Place in data/raw/WVS_Cross_Wave_1981_2022_CSV_v5_0.csv

# Parse and select items
python -m src.data.load_wvs --country IND
python -m src.data.select_items
python -m src.data.build_folds
```

### Phase 2: Baselines

```bash
# Compute baseline metrics
python -m src.eval.baselines --output results/baselines.csv
```

### Phase 3: Track A

```bash
# Run zero-shot inference (can resume if interrupted)
python -m src.inference.hf_local --model llama-3.1-8b --condition P2 --resume
```

### Phase 4: Track B

```bash
# On Kaggle: QLoRA fine-tuning
python -m src.finetune.train_qlora --fold 1 --epochs 2
```

### Phase 5: Analysis + figures

```bash
# Compute subgroup metrics and reproduce paper figures
python -m src.eval.subgroups --output results/subgroup_metrics.json
python -m src.report.make_figures --output paper/figures/
```

### Reproduce everything from cached results

```bash
make reproduce
```

## Key design decisions

- **Logprobs, not sampling:** Inference uses log-probabilities over answer tokens, not sampling. Saves 20–40× compute and gives stable distributions for n=1,692.
- **5-fold CV, not 80/20:** Stratified cross-validation ensures subgroup slices stay above n=30. Every respondent gets an out-of-fold prediction.
- **Baselines before LLMs:** Phase 2 before Phase 3. Know the bar the LLM must clear before investing in inference.
- **Track A = Track B base:** Both use Llama-3.1-8B to isolate fine-tuning effect.
- **Resumable + cached inference:** Every result cached to disk. Kaggle sessions die; the pipeline survives.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Hypothesis does not hold (uniform fidelity) | Still publishable as a negative result. Methodology carries the paper. |
| Small subgroup cells (n<30) | 5-fold CV over full N; report n everywhere; flag exploratory vs. primary slices. |
| Gemini free tier lacks logprobs | Fall back to n=20 sampling; document. Open-weight models still give exact logprobs. |
| Kaggle quota exhausted | Drop to 3-fold; use Gemma-3-4B as primary; cut optional ablations. |
| Kaggle session dies | Checkpointing to Kaggle Dataset. All inference already cached and resumable. |
| Data contamination objection | Run §5.7 probe and report regardless of outcome. |

## Publication targets

1. **C3NLP** (Cross-Cultural NLP, ACL workshop) — best fit
2. **NLP4PI** (NLP for Positive Impact)
3. **ACL / EMNLP / NAACL Student Research Workshop**
4. **arXiv** (cs.CL, cs.CY cross-list)

## References

- Argyle et al. (2023). "Out of One, Many: Estimating individual heterogeneity through linguistic evidence." *The SSRN Electronic Journal*.
- WorldValuesBench (LREC-COLING 2024). https://github.com/demon702/worldvaluesbench
- Bhalla, B. (2026). Silicon Sampling for Indian Public Opinion. *Capstone research paper*.

## License

Code: MIT  
Data: See WVS terms (redistribution restricted)

---

**Questions?** See the full implementation plan in `HANDOFF.md` or contact the owner.
