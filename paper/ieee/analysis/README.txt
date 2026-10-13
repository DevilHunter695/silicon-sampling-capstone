Analysis scripts for the IEEE paper (Python 3: pandas, pyarrow, scikit-learn, matplotlib, scipy).
Run order:
1. wvs_analysis.py  WVS-7 zero-shot: re-verifies accuracy/CIs from saved prediction parquets; permutation tests on subgroup gaps.
2. wvs_extra.py     Cross-model kappa (cluster-bootstrap CI); gender-item split of the sex gap.
3. wvs_null.py      95th percentile of the permutation-null gap per axis (detection thresholds).
4. pew_analysis.py  Pew fold 0: refits baselines with the repo's own pewB_baselines.run_fold, aligns them row-for-row with the
                    fine-tuned predictions; clustered bootstrap CIs, subgroup gaps and permutation tests.
5. pew_extra.py     Concentration for all methods, method agreement, per-item paired differences, subgroup-mean error.
6. gen_tables.py    Writes ../tables/*.tex from the JSON outputs (no hand-typed numbers).
7. make_figures.py  Writes ../figs/*.pdf.
Paths: pew_*.py expect the pew-dataset-training branch cloned at /tmp/ssc_pew and the delivered predictions in
~/Downloads/ishan-cpastone/; wvs_*.py expect the local silicon-sampling-capstone repo. Edit the constants at the top of each script.
