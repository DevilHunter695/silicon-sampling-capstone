#!/bin/bash
# Data-efficiency experiment: Qwen2.5-0.5B-Instruct (4-bit MLX), LoRA, answer-only loss, nested respondent subsets.
cd "$(dirname "$0")"
SP=/private/tmp/claude-501/-Users-krishank-Documents-College-Projects----/c07d5fa6-6bd6-4110-b7a8-4e0125580098/scratchpad
PY=$SP/menv/bin/python
M=${LC_MODEL:-$SP/Qwen2.5-0.5B-Instruct-4bit}
TAG=${LC_TAG:-q05}
until [ -f bg2_done.txt ]; do sleep 20; done
# N = 0: zero-shot of the same small model (chat template, same prompts)
$PY zs_mlx.py --condition base_chat --n-resp 200 --out lc_${TAG}_N0.jsonl --model $M > log_lc_${TAG}_N0.txt 2>&1
for N in 25 100 400; do
  ROWS=$(wc -l < lc_N$N/train.jsonl)
  if [ $N -le 100 ]; then EP=2; else EP=1; fi
  ITERS=$(( ROWS * EP / 4 ))
  START=$(date +%s)
  $PY -m mlx_lm lora --model $M --train --data lc_N$N --fine-tune-type lora --num-layers -1 --batch-size 4 --iters $ITERS \
      --learning-rate 1e-4 --mask-prompt --adapter-path lc_adapters_${TAG}/N$N --steps-per-report 25 --steps-per-eval 100000 \
      --val-batches 1 --max-seq-length 512 --grad-checkpoint -c lc_lora.yaml > log_lc_${TAG}_train_N$N.txt 2>&1
  echo "N=$N train_seconds=$(( $(date +%s) - START ))" >> lc_${TAG}_timing.txt
  $PY zs_mlx.py --condition base_chat --n-resp 200 --mlx-adapter lc_adapters_${TAG}/N$N --out lc_${TAG}_N$N.jsonl --model $M > log_lc_${TAG}_eval_N$N.txt 2>&1
done
echo LC_DONE > lc_${TAG}_done.txt
