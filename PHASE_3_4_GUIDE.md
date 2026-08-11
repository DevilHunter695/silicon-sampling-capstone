# Phase 3-4: Ready to Execute

**Status:** All infrastructure is prepared. Awaiting real WVS-7 India data download.

## What's Blocking Phase 3-4

The **only** blocker is the real WVS-7 India data, which requires manual download:

1. Go to: https://www.worldvaluessurvey.org/
2. Sign up (free, takes 2 minutes)
3. Download **cross-national WVS-7 CSV** (not India-only, so `--country USA` works later)
4. Place in: `data/raw/WVS_Cross_Wave_1981_2022_CSV_v5_0.csv`

## Once You Have the Data

### Step 1: Prepare Data (Local, ~2 min)
```bash
# This processes WVS CSV → parquet + selects items + builds folds
python -m src.data.load_wvs --country IND

# Verify
ls -lh data/processed/
# Should have: ind_wvs7.parquet, selected_items.json, folds.json
```

### Step 2: Run Baselines Locally (Optional, ~5 min)
Verify the pipeline works end-to-end with your data:
```bash
python demo.py --mode full
# Should complete successfully with real data metrics
```

### Step 3: Phase 3 — Track A Zero-Shot Inference (Kaggle, 8-10 GPU-hours)

**When:** Ready to run inference  
**Where:** Kaggle Notebook  
**How:**
1. Create new Kaggle notebook
2. Copy `notebooks/kaggle_trackA_openweight.ipynb`
3. Upload repo as dataset or mount directly
4. Run cells:
   - Set `model_name = "meta-llama/Llama-3.1-8B-Instruct"` or `"google/Gemma-3-4B-it"` for testing
   - Full inference sweep across all conditions (P0-P3)
   - All results cached → resumable if session crashes

**Outputs:**
- Cached predictions in `results/` directory
- Logprobs over answer tokens (not sampling)
- Refusal logging
- ~1,000+ cached JSON files

### Step 4: Phase 4 — Track B Fine-Tuning (Kaggle, 20 GPU-hours for 5 folds)

**When:** Track A predictions complete  
**Where:** Kaggle Notebook  
**How:**
1. Create new Kaggle notebook
2. Copy `notebooks/kaggle_trackB_finetune.ipynb`
3. Set `demo_folds = 5` to run full 5-fold CV
4. Run cells:
   - QLoRA 4-bit on Llama-3.1-8B
   - 2 epochs, lr=2e-4, r=16, alpha=32
   - Checkpoint after each fold to Kaggle datasets (crash-safe)
   - Generate OOF predictions on test split

**Outputs:**
- 5 fine-tuned model checkpoints (one per fold)
- Out-of-fold predictions for all N respondents
- No train/test leakage verified

### Step 5: Phase 5 — Analysis & Figures (Local or Kaggle, 1-2 GPU-hours)

**Inputs:** Track A and Track B cached predictions  
**Outputs:** Subgroup metrics, fidelity gaps, Δ_gap, figures

```bash
# Run subgroup analysis
python -m src.eval.subgroups --phase 5

# Generate paper figures
python -m src.report.make_figures
```

## Key Infrastructure Built

| Component | Status | Location |
|-----------|--------|----------|
| Phase 0-2 core pipeline | ✅ Complete | `src/data/`, `src/eval/` |
| Inference runners (base class) | ✅ Complete | `src/inference/base.py` |
| Gemini 2.5 Flash runner | ✅ Complete | `src/inference/gemini.py` |
| HuggingFace local runner | ✅ Complete | `src/inference/hf_local.py` |
| Track A Kaggle notebook | ✅ Complete | `notebooks/kaggle_trackA_openweight.ipynb` |
| Track B fine-tuning notebook | ✅ Complete | `notebooks/kaggle_trackB_finetune.ipynb` |
| All tests passing | ✅ Complete | `tests/test_pipeline.py` (10/10) |
| Demo with synthetic data | ✅ Complete | `demo.py` |

## Timeline Estimate

Assuming you download data today:

| Phase | Time | GPU | Notes |
|-------|------|-----|-------|
| Step 1: Data prep | 2 min | CPU | Local |
| Step 2: Baselines | 5 min | CPU | Local, validates pipeline |
| Step 3: Track A | 8-10 hrs | T4 | Kaggle free tier (resumable) |
| Step 4: Track B | 20 hrs | T4 | Kaggle free tier, 5 folds × 4 hrs each |
| Step 5: Analysis | 1-2 hrs | CPU | Local, generates all figures |
| **Total** | **~30 hrs** | **T4** | Fits in Kaggle weekly allowance |

## Important Notes

1. **Data licensing:** WVS-7 forbids redistribution. Do NOT commit `data/raw/*.csv` to git. The `.gitignore` already excludes it.

2. **Model licenses:** Llama-3.1-8B requires accepting license on HuggingFace. Run:
   ```bash
   huggingface-cli login
   # Then accept license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
   ```

3. **Caching & resumability:**
   - Track A: Every prediction cached with key `(respondent_id, item_id, model, condition)`
   - Track B: Checkpoints saved after each epoch to Kaggle datasets
   - If session crashes: re-run the cell, cached results are skipped

4. **Kaggle session limits:**
   - Max 12 hours per session
   - Checkpointing essential for Track B (5 folds × 4 hrs each)
   - Save model after each fold to dataset

5. **Two tracks, one base model:**
   - Track A uses `Llama-3.1-8B-Instruct` zero-shot
   - Track B fine-tunes the same checkpoint
   - This isolates the effect of fine-tuning (no confounding with model choice)

## Troubleshooting

### "Model not found on HuggingFace"
```bash
huggingface-cli login
# Then accept model license
```

### "CUDA out of memory"
- Reduce `per_device_train_batch_size` in Track B (line in notebook)
- Or use smaller model: `google/Gemma-3-4B-it` for pipeline validation

### "Session crashed, lost work"
- All results cached/checkpointed
- Re-run notebook; cached results skipped automatically
- For Track B: checkpoint saved to Kaggle datasets after each fold

### "Selected items wrong"
- `data/processed/selected_items.json` is frozen once Track A starts
- Do NOT change it after Phase 3
- If issues, delete and re-run `python -m src.data.select_items`

## Questions?

Refer to:
- `README.md` — project overview
- `SETUP.md` — detailed setup instructions
- `HANDOFF.md` — full research plan with RQs and metrics
- `PROJECT_STATUS.md` — completion status and structure
