#!/bin/bash
cd "$(dirname "$0")"
PY=../../menv/bin/python
M=/private/tmp/claude-501/-Users-krishank-Documents-College-Projects----/c07d5fa6-6bd6-4110-b7a8-4e0125580098/scratchpad/qwen4
UN=Q34a,Q13,Q14b,Q69a,Q69b,Q39b,Q14c,Q54a
$PY zs_mlx.py --condition adapter_raw --n-resp 120 --items $UN --out mlx_unseen_adapter_raw.jsonl --model $M > log_un_ad.txt 2>&1
$PY zs_mlx.py --condition base_raw    --n-resp 120 --items $UN --out mlx_unseen_base_raw.jsonl    --model $M > log_un_base.txt 2>&1
$PY zs_mlx.py --condition adapter_raw --n-resp 120 --out mlx_adapter_raw.jsonl --model $M > log_ad.txt 2>&1
echo BG2_DONE > bg2_done.txt
