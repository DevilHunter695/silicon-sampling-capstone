# Track B fine-tuning — running this at the AI Lab

## What to bring / copy over

Copy the whole repo, or at minimum these paths, onto the lab machine (a USB stick is the
cleanest way to move `data/` — it's under 1MB total, and it avoids sending real respondent
data over any network or third-party service):

```
src/
scripts/
data/processed/ind_wvs7.parquet
data/processed/selected_items.json
data/processed/folds.json
data/reference/wvs7_codebook.json
```

## One-time setup on the lab machine

```bash
pip install -r scripts/trackB_requirements.txt
```

If `openai/gpt-oss-120b` needs a Hugging Face login/token to download, run
`huggingface-cli login` first (only needed once).

## Step 1 — smoke test (do this first, ~5 minutes)

This trains on 5 respondents × 3 items instead of the real run, purely to catch any
environment/path/import problems before spending real GPU time:

```bash
python -m scripts.trackB_finetune --smoke-test
```

If this fails, send me the full output — don't try to debug CUDA/dependency errors
yourself, that's what I'm for.

## Step 2 — real run, fold 0, 15 items (matches Track A's item set exactly)

```bash
python -m scripts.trackB_finetune --fold 0 --n-items 15
```

This will:
1. Build ~15,000 training examples (≈1,353 train respondents × 15 items) using the
   *exact same* prompt-building code Track A used for its zero-shot runs — the only
   difference from Track A is that this model has now seen real WVS-7 answers during
   training, not zero real answers.
2. QLoRA fine-tune `openai/gpt-oss-120b` (4-bit, LoRA rank 16) for 2 epochs.
3. Predict on the held-out 339 test respondents for the same 15 items — genuinely
   out-of-fold, this model never saw these respondents during training.
4. Save predictions to `results/predictions/trackB_openai_gpt-oss-120b_fold0_P2.parquet`
   in the exact same format Track A's predictions use.
5. Print a final summary block with accuracy — copy/paste that back to me.

Runs in the background survive a disconnect if you use `tmux` or `nohup`:

```bash
nohup python -m scripts.trackB_finetune --fold 0 --n-items 15 > trackB.log 2>&1 &
```

Progress and the final summary both land in `trackB_run.log` (and `trackB.log` if you
used the nohup form) — send me either file, or just the tail of it, when it's done.

## If it runs out of memory

`openai/gpt-oss-120b` in 4-bit is a big model — if it doesn't fit even across both H100s
(`device_map="auto"` should split it automatically, but lab GPU availability/VRAM headroom
is genuinely unknown to me), the fallback is a smaller open model. Tell me if this happens
and I'll give you a one-line command swap (e.g. `--model openai/gpt-oss-20b` or a
Llama/Qwen alternative) — don't guess at it yourself.

## What to send back

- The final summary block (accuracy, n predictions, refusal rate)
- The `trackB_run.log` file (or just paste the last ~50 lines)
- The output `.parquet` file if you can get it off the lab machine (same USB stick works)
  — I need this to run the same subgroup fidelity analysis as Track A, not just the
  headline accuracy number.
