"""Held-out-item transfer test on the FAITHFUL lab adapter (items 8:40 = 32 items never used in training), vs the un-adapted base,
the majority class and classifiers trained with labels on each item. Uncertainty: two-way cluster bootstrap (respondents AND items).
Pre-specified: (1) adapter minus base accuracy, (2) adapter minus majority accuracy, (3) NLL differences, (4) near vs far items.
usage: python unseen_analysis.py <out_unseen_dir> <baselines_unseen.parquet> <out.json>"""
import sys, glob, json
import numpy as np, pandas as pd
od, bpp, outp = sys.argv[1:4]
rng = np.random.default_rng(20261013)
def load(c):
    fs = sorted(glob.glob(f"{od}/{c}_s*.jsonl")); x = pd.concat([pd.read_json(f, lines=True) for f in fs]).drop_duplicates(["respondent_id", "question_id"])
    x["correct"] = (x.pred_idx == x.true_idx).astype(float); x["nll"] = [-np.log(max(p[t], 1e-6)) for p, t in zip(x.probs, x.true_idx)]
    return x[["respondent_id", "question_id", "true_idx", "correct", "nll", "probs"]]
M = {"adapter": load("adapter_raw"), "base": load("base_raw")}
bp = pd.read_parquet(bpp)
for (reg, meth), g in bp.groupby(["regime", "method"]):
    g = g.copy(); g["correct"] = (g.pred_idx == g.true_idx).astype(float); g["nll"] = [-np.log(max(p[t], 1e-6)) for p, t in zip(g.probs, g.true_idx)]
    M[f"{meth}|{reg}"] = g[["respondent_id", "question_id", "true_idx", "correct", "nll", "probs"]]
key = lambda x: set(zip(x.respondent_id, x.question_id)); common = set.intersection(*[key(v) for v in M.values()])
for k in M: M[k] = M[k][[(r, q) in common for r, q in zip(M[k].respondent_id, M[k].question_id)]]
items = sorted(M["adapter"].question_id.unique()); ids = np.sort(M["adapter"].respondent_id.unique())
ri = {r: i for i, r in enumerate(ids)}; ii = {q: i for i, q in enumerate(items)}
def mat(k, col):
    A = np.full((len(ids), len(items)), np.nan); d = M[k]
    A[d.respondent_id.map(ri).values, d.question_id.map(ii).values] = d[col].values; return A
CORR = {k: mat(k, "correct") for k in M}; NLL = {k: mat(k, "nll") for k in M}
def stat(A, B, rs, is_):
    sub = A[np.ix_(rs, is_)] - B[np.ix_(rs, is_)]
    per_item = np.nanmean(sub, axis=0)   # accuracy difference per item
    return float(np.nanmean(per_item))
def boot(A, B, items_idx=None, nb=2000):
    it = np.arange(A.shape[1]) if items_idx is None else np.array(items_idx)
    pt = stat(A, B, np.arange(A.shape[0]), it); bs = []
    for _ in range(nb):
        rs = rng.integers(0, A.shape[0], A.shape[0]); is_ = rng.choice(it, len(it))
        bs.append(stat(A, B, rs, is_))
    return pt, np.percentile(bs, [2.5, 97.5]).tolist()
def fam(q): return q.rstrip("abcdefghijklmnop").replace("rec", "").replace("QM", "Q").replace("QW", "Q")
TRAINED = ["QFERTrec", "Q50", "Q12", "Q3132rec", "ABORT", "Q15", "Q39a", "Q28rec"]; trained_fam = {fam(q) for q in TRAINED}
near = [ii[q] for q in items if fam(q) in trained_fam]; far = [ii[q] for q in items if fam(q) not in trained_fam]
res = {"n_resp": len(ids), "n_items": len(items), "n_rows": len(common), "near_items": [items[i] for i in near], "far_items": [items[i] for i in far], "acc": {}, "paired": {}, "nll": {}, "per_item": {}}
for k in M: res["acc"][k] = float(np.nanmean(np.nanmean(CORR[k], axis=0))); res["nll"][k] = float(np.nanmean(np.nanmean(NLL[k], axis=0)))
for a, b in [("adapter", "base"), ("adapter", "majority|full"), ("adapter", "logistic|full"), ("adapter", "gboost|full"), ("adapter", "gboost|pool2k"), ("gboost|full", "majority|full")]:
    if a in M and b in M:
        res["paired"][f"{a} - {b}"] = {"all": dict(zip(("diff", "ci"), boot(CORR[a], CORR[b])))}
        if len(near) > 1: res["paired"][f"{a} - {b}"]["near"] = dict(zip(("diff", "ci"), boot(CORR[a], CORR[b], near, 1000)))
        if len(far) > 1: res["paired"][f"{a} - {b}"]["far"] = dict(zip(("diff", "ci"), boot(CORR[a], CORR[b], far, 1000)))
        pt, ci = boot(NLL[a], NLL[b], None, 1000); res["paired"][f"{a} - {b}"]["nll"] = {"diff": pt, "ci": ci}
for q in items:
    j = ii[q]; res["per_item"][q] = {k: float(np.nanmean(CORR[k][:, j])) for k in ("adapter", "base", "majority|full", "gboost|full") if k in M}
gains = np.array([res["per_item"][q]["adapter"] - res["per_item"][q]["base"] for q in items])
res["item_gain_summary"] = {"mean": float(gains.mean()), "sd": float(gains.std(ddof=1)), "min": float(gains.min()), "max": float(gains.max()), "n_positive": int((gains > 0).sum())}
vs_major = np.array([res["per_item"][q]["adapter"] - res["per_item"][q]["majority|full"] for q in items])
res["item_vs_majority_summary"] = {"mean": float(vs_major.mean()), "n_positive": int((vs_major > 0).sum())}
json.dump(res, open(outp, "w"), indent=1)
print(f"{res['n_resp']} respondents x {res['n_items']} items ({res['n_rows']} rows)")
for k, v in res["acc"].items(): print(f"  acc {k:18s} {100*v:5.1f}   NLL {res['nll'][k]:.3f}")
for k, v in res["paired"].items():
    print(" ", k, {s: (f"{100*d['diff']:+.1f} [{100*d['ci'][0]:+.1f},{100*d['ci'][1]:+.1f}]" if s != "nll" else f"{d['diff']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]") for s, d in v.items()})
print("item gain summary", res["item_gain_summary"])
