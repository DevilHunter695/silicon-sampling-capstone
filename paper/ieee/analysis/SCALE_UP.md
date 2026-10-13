# Scaling the zero-shot controls

Everything reported from the local runs used a **random, nested subset** of respondents (Pew fold-0 test, and WVS),
because the runs were done on a 16 GB Apple M1 laptop. The same scripts scale to the full sets.
Measured throughput on that machine, with per-respondent prefix caching: about 2.3 s per Pew prompt and 1.7 s per WVS prompt.

## On the Mac
```bash
cd analysis
export PY=/path/to/python3.13-venv/bin/python      # needs: mlx-lm pandas pyarrow numpy scikit-learn
export MODEL=/path/to/Qwen2.5-7B-Instruct-4bit      # mlx-community/Qwen2.5-7B-Instruct-4bit (4.3 GB)
$PY zs_mlx.py     --condition base_chat --n-resp 1000 --out mlx_base_chat.jsonl --model $MODEL   # ~5 h per 1,000 respondents
$PY zs_mlx.py     --condition base_raw  --n-resp 500  --out mlx_base_raw.jsonl  --model $MODEL
$PY zs_mlx_wvs.py --n-resp 400 --out mlx_wvs_base_chat.jsonl --model $MODEL                      # ~3 h
$PY zs_p0.py $MODEL mlx_p0.json
python zs_eval.py && python wvs_qwen_eval.py && python zs_p0_eval.py   # resumable; rerun any time
python gen_tables.py && python gen_tables2.py && python gen_tables3.py && python make_figures.py && python make_fig_dists.py
python ../build_paper.py && pdflatex ../paper_ieee.tex
```
The sampling is prefix-consistent: asking for more respondents extends the previous sample, and every script resumes from its jsonl.

## On a CUDA GPU (a lab slice is fine; 16 GB is enough for 4-bit)
Run from the root of the `pew-dataset-training` branch checkout:
```bash
python analysis/zs_hf.py --condition base_raw    --n-resp 6000 --out mlx_base_raw.jsonl
python analysis/zs_hf.py --condition base_chat   --n-resp 6000 --out mlx_base_chat.jsonl
python analysis/zs_hf.py --condition adapter_raw --n-resp 6000 --out mlx_adapter_raw.jsonl --adapter pewB_run/model_fold0
```
`adapter_raw` is the one run that cannot be done faithfully on the Mac (the adapter was trained on a bitsandbytes NF4 base).
It yields the fine-tuned model's own probabilities (NLL, calibration, ensembling with GBM). Expect 1 to 1.5 h per condition
for all 46,209 rows (inference only).
