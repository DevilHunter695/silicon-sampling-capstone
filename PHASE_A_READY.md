# Phase A readiness — what's built, what's left is the API key

Everything through Phase 3 (Track A, zero-shot) is wired and verified end to
end using a mock model that makes zero network calls. The only thing missing
to get real numbers is a model credential.

## What "ready" means here

Every stage below has been run for real, on the real 1,692-respondent India
dataset — not just written and left untested:

| Stage | Status | Evidence |
|---|---|---|
| Data load + clean | ✅ Done | `data/processed/ind_wvs7.parquet`, 1,692 rows |
| Codebook-grounded item selection | ✅ Done | `data/processed/selected_items.json`, 144 items |
| 5-fold CV splits | ✅ Done | `data/processed/folds.json`, no leakage (audited) |
| Demographic + item verbalization | ✅ Done, tested | `src/prompts/verbalize.py` |
| Phase 2 baselines (5 models × 144 items × 5 folds, out-of-fold) | ✅ Run for real | `results/baselines_summary.csv`, `results/baselines_by_item.csv` |
| Inference plumbing (caching, resumability, prompt building) | ✅ Run for real, mock model | `results/predictions/mock-demo-model_P2.parquet` |
| Metrics + subgroup fidelity gaps | ✅ Run for real | `src/report/evaluate_run.py` output |
| Comparison report (HTML) | ✅ Rendered + visually checked | `results/reports/*.html` |
| Real LLM predictions | ⬜ **Needs your API key** | — |

## The one command you'll run

```bash
export GOOGLE_API_KEY="your-key-from-aistudio"
python -m src.inference.run_trackA --model gemini-3.5-flash-lite --condition P2
python -m src.report.comparison_report --predictions results/predictions/gemini-3.5-flash-lite_P2.parquet
```

Open the resulting HTML file in `results/reports/`. That's it — the same
command produces headline accuracy with a bootstrap CI, a bar-chart
comparison against every Phase 2 baseline, a fidelity-gap table per
demographic axis, and a gallery of real transcripts (this respondent's actual
answer vs. what the model predicted, with its full probability distribution
over every option) — all built from your real API results, not placeholders.

Swap `--model gemini-3.5-flash-lite` for `llama-3.1-8b`, `qwen2.5-7b`, or
`gemma-3-4b` to run an open-weight model locally instead (needs
`transformers`/`torch`, not installed in this environment — `uv sync` will
pull them). No other flag or code path changes.

Useful flags while iterating:
- `--n-respondents 50` — quick, cheap run on a random subsample
- `--n-items 10` — same, but fewer questions per respondent
- Predictions are cached per (respondent, item, condition) in
  `results/cache/` — killing and re-running the command resumes, it does not
  re-spend API calls on already-answered pairs.

## What's proven, using the mock model

`MockInferenceRunner` (`src/inference/mock.py`) never touches the network. It
exists solely to exercise every downstream stage — prompt construction,
caching, answer parsing, metrics, subgroup slicing, report rendering — with
real (if meaningless) numbers, so that when a real key lands, the only new
variable is the model's actual judgment, not untested plumbing.

Ran on 20 respondents × 5 items = 99 valid predictions:
- `results/predictions/mock-demo-model_P2.parquet` — full prediction table
- `results/reports/preview.html` — the rendered report, screenshot-checked

Bugs this caught before a real run would have hit them:
1. `src/inference/__init__.py` eagerly imported `torch`/`transformers`, so
   using the mock or Gemini path crashed if those weren't installed. Fixed
   with lazy `__getattr__` imports.
2. The report showed the model's raw answer code ("1") next to the actual
   answer's full text label ("Not at all frequently") — technically correct,
   visually incoherent. Fixed by resolving the predicted code to its label
   text the same way the true answer is resolved.
3. The stat-tile grid's hairline-border CSS trick left a solid color block
   when the tile count didn't evenly fill the last `auto-fit` row. Fixed with
   an explicit 4-column / 2-column-on-mobile grid.

## Phase 2 baseline results (the bar an LLM has to clear)

Out-of-fold, all 144 items, 5 folds each, national-marginal baseline weighted
by `W_WEIGHT` (per the earlier audit — a population-level claim should be
survey-weighted even though respondent-level predictions aren't):

See `results/baselines_summary.csv` for the exact numbers from this run.
Uniform random and the national marginal sit lowest, as expected; gradient
boosting on demographics is the standard the LLM needs to beat by more than
noise for silicon sampling to be worth the API cost.

## Design decisions worth knowing about before the first real run

- **`--condition P2`** (full structured demographics) is the project's stated
  primary condition. Run P0 too if you want the "no demographics" control for
  comparison — same command, one flag change.
- **Gemini logprobs**: `src/inference/gemini.py` now requests real
  `response_logprobs` from the API and only falls back to a confidence-weighted
  guess on the parsed text answer if the API doesn't return them. The report
  flags what fraction of predictions used real vs. fallback probabilities, so
  a reviewer can't mistake one for the other.
- **India's language variable is unusable** (flagged in the earlier audit —
  every respondent is coded `hi` regardless of actual interview language), so
  there is no `sg_language` subgroup axis. Six axes are live: urban/rural,
  income tercile, education band, sex, age band, and WVS region zone.
- **Region zones, not states**: `N_REGION_ISO` gives 8 macro-zones for India,
  not individual states — that's the granularity WVS-7 published, not a
  simplification made here.
- **Small-cell warning is automatic**: any subgroup category under n=30 is
  tagged `n<30` in the report and excluded from the fidelity-gap max/min, so
  a tiny cell can't manufacture a dramatic-looking gap.

## Next after Phase A

Phase 4 (fine-tuning, Track B) and the paper/writeup are out of scope for
this pass, per your instruction to skip fine-tuning for now.
