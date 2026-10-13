#!/bin/bash
cd "$(dirname "$0")"
PY=../../menv/bin/python
M=/private/tmp/claude-501/-Users-krishank-Documents-College-Projects----/c07d5fa6-6bd6-4110-b7a8-4e0125580098/scratchpad/qwen4
$PY zs_mlx.py --condition base_chat --n-resp 200 --out mlx_base_chat.jsonl --model $M > log_base_chat.txt 2>&1
$PY zs_mlx_wvs.py --n-resp 120 --out mlx_wvs_base_chat.jsonl --model $M > log_wvs.txt 2>&1
$PY zs_mlx.py --condition base_raw --n-resp 120 --out mlx_base_raw.jsonl --model $M > log_base_raw.txt 2>&1
echo BG_ALL_DONE > bg_done.txt
