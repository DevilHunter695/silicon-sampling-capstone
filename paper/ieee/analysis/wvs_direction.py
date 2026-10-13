import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/krishank/Documents/College Projects/🧢 /silicon-sampling-capstone"
sys.path.insert(0, REPO); os.chdir(REPO)
from src.config import DATA_PROCESSED
from src.eval.subgroups import assign_subgroups, SUBGROUP_AXES
rng = np.random.default_rng(606)
q = pd.read_json(f"{HERE}/mlx_wvs_base_chat.jsonl", lines=True).drop_duplicates(["respondent_id", "question_id"])
B = pd.read_parquet(f"{HERE}/wvs_baseline_rows.parquet")
p0 = json.load(open(f"{HERE}/mlx_p0_wvs.json"))
m = q.merge(B.drop(columns=["true_idx"]), on=["respondent_id", "question_id"])
m["K"] = m.probs.map(len)
m["p0_pred"] = m.question_id.map(lambda x: p0[x]["pred_idx"])
ids = np.sort(m.respondent_id.unique()); ri = {r: i for i, r in enumerate(ids)}; R = len(ids); m["ri"] = m.respondent_id.map(ri)
W = rng.multinomial(R, np.ones(R) / R, size=3000).astype(float)
def agg(c): return np.bincount(m.ri, weights=c, minlength=R), np.bincount(m.ri, minlength=R).astype(float)
s2, n2 = agg((m.pred_idx == m.true_idx).astype(float)); s0, n0 = agg((m.p0_pred == m.true_idx).astype(float))
out = {"n_rows": int(len(m)), "n_resp": int(R), "acc_p2": float(s2.sum() / n2.sum()), "acc_p0": float(s0.sum() / n0.sum()),
       "p2_minus_p0": float(s2.sum() / n2.sum() - s0.sum() / n0.sum()),
       "p2_minus_p0_ci": np.percentile((W @ s2) / (W @ n2) - (W @ s0) / (W @ n0), [2.5, 97.5]).tolist(),
       "same_pred_rate": float((m.pred_idx == m.p0_pred).mean())}
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
# demographics
df = pd.read_parquet(f"{DATA_PROCESSED}/ind_wvs7.parquet")
if "respondent_id" not in df.columns: df["respondent_id"] = range(len(df))
df["respondent_id"] = df.respondent_id.astype(int)
sub = assign_subgroups(df.set_index("respondent_id", drop=False))
axes = {a: sub[a].astype(object).where(sub[a].notna(), None) for a in SUBGROUP_AXES}
m["soft"] = [float(np.dot(p, np.arange(len(p))) / (len(p) - 1)) for p in m.probs]
m["true"] = m.true_idx / (m.K - 1)
for c in ["pred_idx", "majority", "cell_mode", "logistic_matched", "gbm_matched"]: m[f"{c}_n"] = m[c] / (m.K - 1)
def cells(d, col, n_min, w=None):
    xs, ys = [], []
    for qq, g in d.groupby("question_id"):
        ww = np.ones(len(g)) if w is None else g.respondent_id.map(w).to_numpy()
        if ww.sum() == 0: continue
        mp = np.average(g[col], weights=ww); mt = np.average(g["true"], weights=ww)
        for a, ser in axes.items():
            c = ser.reindex(g.respondent_id).to_numpy()
            for cat in pd.unique(c[pd.notna(c)]):
                mk = (c == cat)
                if len(set(g.respondent_id.to_numpy()[mk])) < n_min or ww[mk].sum() == 0: continue
                xs.append(np.average(g[col].to_numpy()[mk], weights=ww[mk]) - mp); ys.append(np.average(g["true"].to_numpy()[mk], weights=ww[mk]) - mt)
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 5 or xs.std() < 1e-9: return np.nan, np.nan, len(xs)
    return float(np.corrcoef(xs, ys)[0, 1]), float(np.polyfit(ys, xs, 1)[0]), len(xs)
res = {"p0": out}
for name, col in [("qwen_soft", "soft"), ("qwen_hard", "pred_idx_n"), ("gbm", "gbm_matched_n"), ("logistic", "logistic_matched_n"), ("cell_mode", "cell_mode_n")]:
    r, s, n = cells(m, col, 8)
    bs = []
    for _ in range(600):
        cnt = pd.Series(rng.multinomial(R, np.ones(R) / R), index=ids)
        bs.append(cells(m, col, 8, w=cnt)[:2])
    a = np.array(bs, float)
    res[name] = {"r": r, "slope": s, "n_cells": n, "r_ci": np.nanpercentile(a[:, 0], [2.5, 97.5]).tolist() if np.isfinite(a[:, 0]).any() else None}
    print(name, round(r, 3) if r == r else r, round(s, 2) if s == s else s, n)
json.dump(res, open(f"{HERE}/wvs_direction.json", "w"), indent=1, default=float)
