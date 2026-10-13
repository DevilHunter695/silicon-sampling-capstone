#!/bin/bash
# Usage: ./morning.sh <folder with the downloaded zips> [python]
# Expects any of: result_unseen.zip result_scale.zip seed0_results.zip seed1_results.zip seed2_results.zip (missing ones are skipped).
set -u
IN="${1:?folder with zips}"; PY="${2:-python3}"; HERE="$(cd "$(dirname "$0")" && pwd)"; W="$IN/_work"; mkdir -p "$W"
REPO="${REPO:-/tmp/ssc_pew}"          # checkout of the pew-dataset-training branch (needs data/processed/*)
for z in result_unseen result_scale seed0_results seed1_results seed2_results; do
  [ -f "$IN/$z.zip" ] && { rm -rf "$W/$z"; mkdir -p "$W/$z"; unzip -oq "$IN/$z.zip" -d "$W/$z"; echo "unzipped $z"; }
done
# baselines for the breadth experiment (500 respondents x 40 items) if not already present
[ -f "$HERE/baselines_probs.parquet" ] || $PY "$HERE/baselines_probs.py" "$REPO" "$HERE/baselines_probs.parquet" 0
[ -f "$HERE/baselines_unseen.parquet" ] || $PY "$HERE/baselines_unseen.py" "$REPO" "$HERE/baselines_unseen.parquet" 1000 8 40
if [ -d "$W/result_unseen" ]; then
  $PY "$HERE/unseen_analysis.py" "$W/result_unseen" "$HERE/baselines_unseen.parquet" "$W/unseen.json" | tee "$W/unseen.txt"
fi
if [ -d "$W/result_scale" ]; then
  F0="${F0_OUT:-$HOME/Downloads/pew_probs/out}"
  $PY "$HERE/scale_analysis.py" "$F0" "$W/result_scale" "$HERE/baselines_probs.parquet" "$W/scale.json" | tee "$W/scale.txt"
fi
SEEDS=""; for s in seed0_results seed1_results seed2_results; do [ -d "$W/$s" ] && SEEDS="$SEEDS $W/$s"; done
if [ -n "$SEEDS" ]; then
  $PY "$HERE/summarize_seeds.py" "$HERE/baselines_probs.parquet" "$W/seeds_summary.json" $SEEDS | tee "$W/seeds.txt"
fi
echo "done. outputs in $W (unseen.json, scale.json, seeds_summary.json, per-seed transfer.json)"
