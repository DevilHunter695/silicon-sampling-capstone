# Pre-specified analysis (written 2026-10-13, before any result of the breadth experiment existed)
Data/splits: Pew fold 0 (pew_folds.json), 2,000 train respondents drawn with seed 0, evaluation on the first 500 respondents of the seed-7 test permutation, items 0:40 of pew_selected_items.json.
Primary comparisons (respondent-clustered bootstrap, 2,000 draws, 95% CI), accuracy first, NLL second:
1. Held-out items 24:40: adapter A (items 0:24) minus adapter B (items 0:8), equal example budget (15,462).
2. Held-out items: each adapter minus the un-adapted base model, and minus the majority class and logistic/GBM trained on the same respondents.
3. Trained items: each adapter minus classical baselines trained on exactly the same (respondent,item) pairs.
4. Difference-in-gains (trained-item gain minus held-out-item gain) for each adapter.
5. Secondary: item-centred directional validity (religion, region, sex), near vs far held-out items (same battery in training or not), P0 control.
Rules: no model/run selection on test accuracy; all seeds and runs are reported; a negative or null result is reported as such; claims are limited to what the CIs support.
Fold-0 full-run analysis (fold0_full_analysis.py): accuracy, NLL, ECE, paired differences, fixed 50/50 FT+GBM average, direction, Q12 behaviour, agreement with lab predictions.
