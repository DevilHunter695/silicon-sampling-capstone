# Setup and Installation Guide

## Prerequisites

- Python 3.10+
- pip or uv package manager
- GPU (recommended for local model inference)

## Installation

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements-dev.txt
```

### 2. Prepare Data

#### Download WVS-7

1. Go to [https://www.worldvaluessurvey.org/](https://www.worldvaluessurvey.org/)
2. Sign up (free, takes 2 minutes)
3. Download the **cross-national WVS-7 CSV** (not India-only)
4. Place it in `data/raw/WVS_Cross_Wave_1981_2022_CSV_v5_0.csv`

#### Clean and Prepare

```bash
python -m src.data.load_wvs --country IND
```

This will:
- Load the WVS CSV
- Recode missing values (-1 to -5) as NaN
- Filter to India (IND)
- Save to `data/processed/ind_wvs7.parquet`

### 3. Configure API Keys

#### For Gemini 2.5 Flash (free tier)

1. Get free API key: https://aistudio.google.com/app/apikey
2. Set environment variable:
   ```bash
   export GOOGLE_API_KEY="your-key-here"
   ```

#### For Local Models (HuggingFace)

Models require:
- `transformers>=4.35`
- `torch>=2.1`
- GPU memory (8B model needs ~16GB with 4-bit quantization)

Accept model license:
```bash
huggingface-cli login
# Then accept license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

## Quick Start

### Run Demo (Synthetic Data)

```bash
# Full demo
python demo.py

# Just Phase 0-1 (data + item selection)
python demo.py --mode phase0-1

# Just baselines
python demo.py --mode baselines-only
```

### Run Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

### Run Full Pipeline (Real Data)

```bash
# Phase 0-1: Prepare data
python -m src.data.load_wvs --country IND
python -m src.data.select_items
python -m src.data.build_folds

# Phase 2: Evaluate baselines (creates performance bar)
python -m notebooks.kaggle_trackA_openweight --phase 2

# Phase 3: Zero-shot inference
python -m notebooks.kaggle_trackA_openweight --model gemini-2.5-flash --condition P2

# Phase 4: Fine-tuning (on Kaggle)
# Use notebooks/kaggle_trackB_finetune.ipynb

# Phase 5: Analysis & reporting
python -m src.eval.subgroups
python -m src.report.make_figures
```

## Project Structure

```
src/
├── config.py              # Global configuration
├── data/
│   ├── load_wvs.py       # WVS data loading
│   ├── select_items.py   # Item selection (criteria-based)
│   └── build_folds.py    # Stratified k-fold creation
├── prompts/
│   └── templates.py      # P0/P1/P2/P3 prompt conditions
├── inference/
│   ├── base.py           # Cached, resumable inference base
│   ├── gemini.py         # Gemini inference runner
│   └── hf_local.py       # HuggingFace local inference
├── eval/
│   ├── metrics.py        # Evaluation metrics (accuracy, MAE, NLL, JSD, W1)
│   ├── baselines.py      # Baseline models
│   ├── subgroups.py      # Subgroup analysis (fidelity gap, Δ_gap)
│   └── bootstrap.py      # Bootstrap confidence intervals
└── report/
    └── make_figures.py   # Generate paper figures
```

## Key Concepts

### Prompt Conditions

- **P0**: No demographics (control)
- **P1**: Minimal (age, sex, region)
- **P2**: Full structured (14 attributes) - PRIMARY
- **P3**: Naturalistic backstory

### Metrics

**Individual level:**
- Accuracy: Exact match
- MAE: Mean absolute error (respects ordinal scale)
- NLL: Negative log-likelihood (proper scoring rule)

**Distributional level:**
- JSD: Jensen-Shannon divergence (literature standard)
- W1: Wasserstein-1 distance (respects ordering)

### Baselines

1. **Uniform random** - Random guess across all options
2. **National marginal** - Overall India distribution
3. **Demographic cell lookup** - Empirical distribution per demographic cell
4. **Logistic regression** - Supervised classifier on demographics
5. **Gradient boosting** - Non-linear supervised classifier

LLM must beat at least baselines 2 and 3 for silicon sampling to add value.

### Fairness Metrics

- **Fidelity gap**: (worst slice) - (best slice) per subgroup
- **Δ_gap**: gap(fine-tuned) - gap(zero-shot)
  - Positive Δ_gap = fine-tuning hurt fairness (widened gap)
  - Negative Δ_gap = fine-tuning improved fairness (narrowed gap)

## Computational Resources

### Kaggle Free Tier (Recommended)

- 2x T4 GPUs (16GB each)
- 30 GPU-hours/week
- 12-hour session limit (resumable via cache)

**Estimated costs:**
- Phase 3 (Track A): 8-10 GPU-hours
- Phase 4 (Track B, 5-fold): ~20 GPU-hours

### Local Development

- CPU fine-tuning: Slow but free
- GPU fine-tuning: Fast, requires NVIDIA GPU with 16GB+ VRAM
- Gemini API: Free tier (rate limited, ~15 req/min)

## Troubleshooting

### "WVS CSV not found"

```bash
# Download and place correctly
ls -la data/raw/WVS_Cross_Wave_1981_2022_CSV_v5_0.csv
```

### "No candidate items found"

Adjust thresholds in `src/config.py`:
- `MAX_MISSINGNESS_PCT`: Lower to be more strict
- `MIN_MODAL_ENTROPY`: Lower to allow more near-unanimous items

### "GOOGLE_API_KEY not set"

```bash
export GOOGLE_API_KEY="your-key-from-aistudio"
# Or set in code:
# runner = GeminiInferenceRunner(api_key="your-key")
```

### "Model not found on HuggingFace"

- Accept model license: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Run: `huggingface-cli login`

## Next Steps

1. ✓ Code pipeline (this phase)
2. Get API keys (Gemini free tier or local GPU)
3. Download real WVS-7 India data
4. Run Phase 2 (baselines) first - establishes the bar
5. Run Phase 3 (zero-shot) on Gemini or local model
6. Run Phase 4 (fine-tuning) on Kaggle if GPU available
7. Analyze subgroup fidelity gaps (Phase 5)
8. Write paper (Phase 6)

## References

- [HANDOFF.md](HANDOFF.md) - Detailed implementation plan
- [README.md](README.md) - Project overview
- [WorldValuesBench repo](https://github.com/demon702/worldvaluesbench) - Code reuse reference
