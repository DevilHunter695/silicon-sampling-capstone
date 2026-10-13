"""Distributional (non-argmax) fidelity on Pew fold 0 from the faithful probabilities.
For each (item, subgroup-category) cell with >= n_min respondents: Jensen-Shannon divergence (base 2, in [0,1]) between the mean predicted
distribution and the empirical true distribution, averaged over cells. Respondent-clustered bootstrap CIs.
usage: python distributional_fidelity.py <kaggle_out_dir> <baselines_probs.parquet> <out.json> [data.parquet]"""
import sys, glob, json
import numpy as np, pandas as pd
od, bpp, outp = sys.argv[1:4]
datap = sys.argv[4] if len(sys.argv) > 4 else "/tmp/ssc_pew/data/processed/pew_india_2021.parquet"
rng = np.random.default_rng(31)
def load(c):
    fs = sorted(glob.glob(f"{od}/{c}_s*.jsonl"))
    return pd.concat([pd.read_json(f, lines=True) for f in fs]).drop_duplicates(["respondent_id", "question_id"]) if fs else None
M = {}
for k, c in (("finetuned", "adapter_raw"), ("base_raw", "base_raw"), ("base_chat", "base_chat")):
    x = load(c)
    if x is not None: M[k] = x[["respondent_id", "question_id", "true_idx", "probs"]]
bp = pd.read_parquet(bpp); bp = bp[(bp.regime == "full") & bp.question_id.isin(set(M["finetuned"].question_id))]
for meth in ("majority", "logistic", "gboost"):
    M[meth] = bp[bp.method == meth][["respondent_id", "question_id", "true_idx", "probs"]]
key = lambda x: set(zip(x.respondent_id, x.question_id))
common = set.intersection(*[key(v) for v in M.values()])
for k in M: M[k] = M[k][[(r, q) in common for r, q in zip(M[k].respondent_id, M[k].question_id)]].sort_values(["respondent_id", "question_id"]).reset_index(drop=True)
df = pd.read_parquet(datap); df["respondent_id"] = df.respondent_id.astype(int); df = df.set_index("respondent_id")
def jsd(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float); m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(a / np.maximum(b, 1e-12)), 0.0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
AX = {"religion": "QRELSING", "region": "REGION", "sex": "QGEN", "education": "QEDU", "age": "QAGErec"}
base = M["finetuned"]; ids = np.sort(base.respondent_id.unique()); ix = {r: i for i, r in enumerate(ids)}
def cell_scores(rows_idx_by_cell, P, T, w):
    s = []
    for cell, idx in rows_idx_by_cell.items():
        ww = w[idx]
        if ww.sum() < 1: continue
        K = P.shape[1]
        pred = (P[idx] * ww[:, None]).sum(0) / ww.sum()
        tru = np.bincount(T[idx], weights=ww, minlength=K) / ww.sum()
        s.append(jsd(pred, tru))
    return float(np.mean(s)) if s else np.nan
res = {"n_rows": len(base), "n_resp": len(ids), "axes": {}}
# precompute cells
R = len(ids); riB = base.respondent_id.map(ix).values
for ax, col in AX.items():
    g = df.loc[base.respondent_id.values, col].values
    cells = {}
    for q in base.question_id.unique():
        mq = (base.question_id.values == q)
        for v in pd.unique(g[mq & pd.notna(g)]):
            idx = np.where(mq & (g == v))[0]
            if len(set(riB[idx])) >= 40: cells[(q, v)] = idx
    out = {}
    Wb = rng.multinomial(R, np.ones(R) / R, size=300).astype(float)
    for k, d in M.items():
        K = max(len(p) for p in d.probs); P = np.stack([np.pad(np.asarray(p, float), (0, K - len(p))) for p in d.probs]); T = d.true_idx.values.astype(int)
        pt = cell_scores(cells, P, T, np.ones(len(d)))
        bs = [cell_scores(cells, P, T, W[riB]) for W in Wb]
        out[k] = {"jsd": pt, "ci": np.nanpercentile(bs, [2.5, 97.5]).tolist(), "n_cells": len(cells)}
    # paired difference vs gboost
    res["axes"][ax] = out
    print(ax, {k: round(v["jsd"], 4) for k, v in out.items()}, flush=True)
json.dump(res, open(outp, "w"), indent=1)
