"""Full fold-0 analysis of the faithful Kaggle outputs (adapter_raw, base_raw, base_chat), 6,000 respondents / 46,209 rows.
Pre-specified: accuracy, NLL, ECE (10-bin top-label), paired FT-vs-baseline differences, 50/50 FT+GBM probability average (no tuning),
directional validity of item-centred group differences (religion, region, sex), Q12 behaviour, agreement with the lab's hard predictions.
usage: python fold0_full_analysis.py <out_dir_with_jsonl> <baselines_probs.parquet> <lab_predictions.parquet> <result.json> [<data.parquet>]"""
import sys, glob, json
import numpy as np, pandas as pd
od, bpp, labp, outp = sys.argv[1:5]
datap = sys.argv[5] if len(sys.argv) > 5 else "/tmp/ssc_pew/data/processed/pew_india_2021.parquet"
rng = np.random.default_rng(11)


def load(cond):
    fs = sorted(glob.glob(f"{od}/{cond}_s*.jsonl"))
    if not fs:
        return None
    x = pd.concat([pd.read_json(f, lines=True) for f in fs]).drop_duplicates(["respondent_id", "question_id"])
    return x


def add(x):
    x["correct"] = (x.pred_idx == x.true_idx).astype(float)
    x["nll"] = [-np.log(max(p[t], 1e-6)) for p, t in zip(x.probs, x.true_idx)]
    x["conf"] = [max(p) for p in x.probs]
    return x


def ece(x, bins=10):
    b = np.minimum((x.conf.values * bins).astype(int), bins - 1); e = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            e += m.mean() * abs(x.conf.values[m].mean() - x.correct.values[m].mean())
    return float(e)


ft = load("adapter_raw"); assert ft is not None, "no adapter_raw outputs found"
ft = add(ft)
bp = pd.read_parquet(bpp); bp = bp[bp.regime == "full"]
bp = bp[bp.question_id.isin(set(ft.question_id))]
M = {"finetuned": ft}
for c in ("base_raw", "base_chat"):
    x = load(c)
    if x is not None:
        M[c] = add(x)
for meth in ("majority", "logistic", "gboost"):
    g = bp[bp.method == meth].copy()
    g["correct"] = (g.pred_idx == g.true_idx).astype(float)
    g["nll"] = [-np.log(max(p[t], 1e-6)) for p, t in zip(g.probs, g.true_idx)]
    g["conf"] = [max(p) for p in g.probs]
    M[meth] = g
key = lambda x: set(zip(x.respondent_id, x.question_id))
common = set.intersection(*[key(v) for v in M.values()])
for k in M:
    M[k] = M[k][[(r, q) in common for r, q in zip(M[k].respondent_id, M[k].question_id)]].sort_values(["respondent_id", "question_id"]).reset_index(drop=True)
n = len(M["finetuned"]); ids = np.sort(M["finetuned"].respondent_id.unique()); ix = {r: i for i, r in enumerate(ids)}
print("common rows", n, "respondents", len(ids))
ri = M["finetuned"].respondent_id.map(ix).values
W = rng.multinomial(len(ids), np.ones(len(ids)) / len(ids), size=2000).astype(float)
cnt = np.bincount(ri, minlength=len(ids)).astype(float)
def agg(v): return np.bincount(ri, weights=v, minlength=len(ids))
def ci(v): s = agg(v); r = (W @ s) / (W @ cnt); return float(s.sum() / cnt.sum()), np.percentile(r, [2.5, 97.5]).tolist()
def dci(v): s = agg(v); r = (W @ s) / (W @ cnt); return float(s.sum() / cnt.sum()), r
res = {"n_rows": n, "n_resp": len(ids), "overall": {}, "paired": {}, "per_item": {}}
for k, g in M.items():
    a, ac = ci(g.correct.values); l, lc = ci(g.nll.values)
    res["overall"][k] = {"acc": a, "acc_ci": ac, "nll": l, "nll_ci": lc, "ece": ece(g)}
for k in M:
    if k == "finetuned":
        continue
    for col in ("correct", "nll"):
        a, ra = dci(M["finetuned"][col].values); b, rb = dci(M[k][col].values); d = ra - rb
        res["paired"][f"finetuned - {k} [{col}]"] = {"diff": a - b, "ci": np.percentile(d, [2.5, 97.5]).tolist()}
for q, g in M["finetuned"].groupby("question_id"):
    row = {"finetuned": [float(g.correct.mean()), float(g.nll.mean())]}
    for k in M:
        if k != "finetuned":
            h = M[k][M[k].question_id == q]; row[k] = [float(h.correct.mean()), float(h.nll.mean())]
    res["per_item"][q] = row
# 50/50 probability average with GBM (fixed weight: nothing tuned on test labels)
pe = [(np.array(a) + np.array(b)) / 2 for a, b in zip(M["finetuned"].probs, M["gboost"].probs)]
ens = pd.DataFrame({"correct": [float(np.argmax(p) == t) for p, t in zip(pe, M["finetuned"].true_idx)],
                    "nll": [-np.log(max(p[t], 1e-6)) for p, t in zip(pe, M["finetuned"].true_idx)]})
for col in ("correct", "nll"):
    a, ra = dci(ens[col].values); b, rb = dci(M["gboost"][col].values)
    res["paired"][f"ens(FT,GBM) - gboost [{col}]"] = {"a": a, "b": b, "diff": a - b, "ci": np.percentile(ra - rb, [2.5, 97.5]).tolist()}
# Q12 behaviour
g = M["finetuned"][M["finetuned"].question_id == "Q12"]
res["Q12"] = {"pred_shares": g.pred_idx.value_counts(normalize=True).round(4).to_dict(),
              "mean_probs": np.mean(np.stack(g.probs.values), 0).round(4).tolist(),
              "true_shares": g.true_idx.value_counts(normalize=True).sort_index().round(4).to_dict(),
              "mean_max_prob": float(g.conf.mean())}
# agreement with lab hard predictions
L = pd.read_parquet(labp)[["respondent_id", "question_id", "pred_code_idx"]].rename(columns={"pred_code_idx": "lab"})
m = M["finetuned"].merge(L, on=["respondent_id", "question_id"]); ok = m.lab.notna()
res["agreement_with_lab"] = float((m.pred_idx[ok] == m.lab[ok]).mean())
res["lab_acc_same_rows"] = float((m.lab[ok] == m.true_idx[ok]).mean())
# directional validity: item-centred group differences in expected index (religion, region, sex)
df = pd.read_parquet(datap); df["respondent_id"] = df.respondent_id.astype(int); df = df.set_index("respondent_id")
dire = {}
for ax, col in [("religion", "QRELSING"), ("region", "REGION"), ("sex", "QGEN")]:
    def cell(model, what):
        x = M[model]; ev = np.array([np.dot(np.arange(len(p)), p) for p in x.probs]) if what == "pred" else x.true_idx.values.astype(float)
        t = pd.DataFrame({"q": x.question_id.values, "g": df.loc[x.respondent_id.values, col].values, "v": ev})
        t = t[t.g.notna()]; c = t.groupby(["q", "g"]).v.agg(["mean", "size"]); c = c[c["size"] >= 30]["mean"]
        return c - c.groupby(level=0).transform("mean")
    tr = cell("finetuned", "true")
    row = {}
    for k in M:
        pr = cell(k, "pred"); j = pd.concat([tr, pr], axis=1, keys=["t", "p"]).dropna()
        if len(j) > 5 and j.p.std() > 1e-9 and j.t.std() > 1e-9:
            r = float(np.corrcoef(j.t, j.p)[0, 1]); slope = float(np.polyfit(j.t, j.p, 1)[0]); row[k] = {"r": r, "slope": slope, "n_cells": int(len(j))}
        else:
            row[k] = {"r": None, "slope": None, "n_cells": int(len(j))}
    dire[ax] = row
res["direction"] = dire
json.dump(res, open(outp, "w"), indent=1, default=float)
print("overall:")
for k, v in res["overall"].items():
    print(f"  {k:12s} acc {100*v['acc']:.2f} [{100*v['acc_ci'][0]:.2f},{100*v['acc_ci'][1]:.2f}]  NLL {v['nll']:.3f}  ECE {v['ece']:.3f}")
print("paired:", {k: (round(v['diff'], 4), [round(x, 4) for x in v['ci']]) for k, v in res["paired"].items()})
print("Q12:", res["Q12"]); print("agreement with lab:", round(res["agreement_with_lab"], 4), "lab acc", round(res["lab_acc_same_rows"], 4))
print("direction:", {a: {k: (round(v['r'], 2) if v['r'] is not None else None) for k, v in d.items()} for a, d in dire.items()})
