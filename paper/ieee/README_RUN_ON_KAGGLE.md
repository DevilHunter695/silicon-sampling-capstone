# Kaggle run guide: breadth-vs-data fine-tuning (friend's account, 2 x T4)

## Context to give the person / assistant running this
Capstone: "Silicon Sampling for Indian Public Opinion" (LLMs simulating survey respondents; Pew "Religion in India" 2021).
A prior run fine-tuned Qwen2.5-7B-Instruct (QLoRA r=16, a=32, q/k/v/o) on 8 Pew items (fold 0, ~24k respondents) and got 36.1% held-out accuracy vs majority 30.4%, logistic 38.5%, GBM 40.0%.
We showed on a separate Kaggle run that it can be re-scored faithfully (probabilities). Open question for the paper: does fine-tuning transfer to UNSEEN items,
and does it depend on item breadth or only on data size? This notebook answers that with a controlled design:

* **B**: LoRA on items 0:8 (the lab's items), 2,000 respondents, 15,462 examples.
* **A**: LoRA on items 0:24 (3x more items), the SAME 2,000 respondents, the SAME 15,462 examples (randomly drawn from the 24-item pool).
* Both are evaluated on 40 items (0:40) for 500 held-out fold-0 test respondents: trained, mid (8:24) and held-out (24:40) items,
  next to the un-adapted base model and classical baselines trained on identical data (computed separately on CPU).
* P0 (no-profile) control for the lab adapter, A, B and base: 40 prompts.
Everything else (prompt format P2, scoring the digit after prompt+"\n\n", LoRA settings) matches the lab run. Deviations to disclose: fp16 compute on T4 (lab bf16),
answer-only loss on the single answer token (the lab run used a full-sequence causal loss; its later, unreleased v2 script used completion-only loss), fewer respondents (2,000 vs 24k), micro-batch 8 x accum 2.

## Rules
* Do not alter splits, items, seeds, or filter rows. Do not select runs by test accuracy. Report all runs including failures.
* Keep the dataset private. Never put WVS microdata, HF tokens, API keys in the notebook.
* Quota: smoke-test first (cells 1-4). Only then run cell 6. Use the zip in /kaggle/working to download results; the session's files vanish when it ends.

## Steps
1. kaggle.com -> Datasets -> New dataset (Private): upload the CONTENTS of `kaggle_pkg2/` (scripts, adapter/, repo/, lab_predictions_fold0.parquet, notebook). Name e.g. `pew-pkg2`.
2. New notebook -> File -> Import notebook -> `pew_breadth_experiment.ipynb`. Settings: Accelerator GPU T4 x2, Internet On. Add Input -> your dataset.
3. Run cells 1-4. In cell 4 read `s_per_step` from the last record. The full run is 966 optimizer steps per adapter (both adapters train in parallel, one per GPU):
   * s_per_step <= 8  -> ~2.2 h of training. Go.
   * 8 < s_per_step <= 20 -> up to ~5.4 h. Go, but keep `max_hours=5` (training checkpoints and resumes).
   * > 20 -> lower `n_resp` in `run_seed` to 1000 (the matched-budget logic still holds) and tell us.
4. Run cell 6 (`run_seed(0)`). Phases: train A and B in parallel -> evaluate (500 resp x 40 items) -> base zero-shot -> P0 -> zip.
   Expected total ~6-9 h of session time. If the session is interrupted, re-run cells 1-3 and cell 6; training resumes from the last checkpoint and
   evaluation resumes from finished rows.
5. Download `/kaggle/working/seed0_results.zip` (Output panel) and send it back. Optionally run cell 7 (`run_seed(1)`) for a second seed if quota remains.

## What not to run
Do not press "Save & Run All" before the smoke test passes (it re-runs everything and spends the quota).
