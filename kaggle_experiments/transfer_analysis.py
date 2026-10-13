"""Analysis of the breadth-vs-data experiment. Pre-specified primary comparisons (written before any results existed):
  P1  held-out items 24:40: accuracy of adapter A (trained on items 0:24) minus adapter B (trained on items 0:8), same examples budget.
  P2  held-out items 24:40: adapter minus zero-shot base (gain from fine-tuning on unseen items), for A and for B.
  P3  trained items: adapter minus classical baselines trained on the SAME examples (matched) and on the full train set.
  P4  difference-in-gains (trained-item gain minus unseen-item gain).
All CIs: respondent-clustered bootstrap (2,000 draws) over the 500 evaluation respondents. No test labels are used for any selection.
usage: python transfer_analysis.py <seed_dir> <baselines_probs.parquet> <out.json> [<base_eval_dir>]"""
import sys, glob, json, os
import numpy as np, pandas as pd

seed_dir, bp_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
base_dir = sys.argv[4] if len(sys.argv) > 4 else f"{seed_dir}/eval_base"
items = json.load(open("/tmp/ssc_pew/data/processed/pew_selected_items.json"))["selected_items"][:40]
GROUPS = {"trained_B(0:8)": items[0:8], "mid(8:24)": items[8:24], "heldout(24:40)": items[24:40], "trained_A(0:24)": items[0:24]}
SIB = lambda q: q.rstrip("abcdefghijrec").replace("QM", "Q").replace("QW", "Q")  # crude family key: Q69a -> Q69, QM21a -> Q21
rng = np.random.default_rng(2026)


def load(d, cond):
    fs = sorted(glob.glob(f"{d}/{cond}_s*.jsonl"))
    if not fs:
        return None
    x = pd.concat([pd.read_json(f, lines=True) for f in fs]).drop_duplicates(["respondent_id", "question_id"])
    x["nll"] = [-np.log(max(p[t], 1e-6)) for p, t in zip(x.probs, x.true_idx)]
    x["correct"] = (x.pred_idx == x.true_idx).astype(float)
    return x[["respondent_id", "question_id", "true_idx", "pred_idx", "probs", "correct", "nll"]]


M = {}
for name, d, c in [("A", f"{seed_dir}/eval_A", "adapter_raw"), ("B", f"{seed_dir}/eval_B", "adapter_raw"), ("base", base_dir, "base_raw")]:
    x = load(d, c)
    if x is not None:
        M[name] = x
bp = pd.read_parquet(bp_path)
bp["correct"] = (bp.pred_idx == bp.true_idx).astype(float)
bp["nll"] = [-np.log(max(p[t], 1e-6)) for p, t in zip(bp.probs, bp.true_idx)]
for (reg, meth), g in bp.groupby(["regime", "method"]):
    if reg == "full" and g.respondent_id.nunique() > 600:   # keep only the 500-respondent subset for this comparison
        pass
    M[f"{meth}|{reg}"] = g[["respondent_id", "question_id", "true_idx", "pred_idx", "probs", "correct", "nll"]]
base_resp = set(M["A"].respondent_id) if "A" in M else set(M["B"].respondent_id)
for k in list(M):
    M[k] = M[k][M[k].respondent_id.isin(base_resp)]


def table(models, qs, col="correct"):
    """respondent-level sums over rows of items qs, aligned across models on the common (respondent,item) rows."""
    common = None
    for m in models:
        s = set(zip(M[m].respondent_id, M[m].question_id))
        common = s if common is None else common & s
    common = {c for c in common if c[1] in qs}
    ids = sorted({r for r, _ in common}); ix = {r: i for i, r in enumerate(ids)}
    out = {}
    for m in models:
        g = M[m]; g = g[[(r, q) in common for r, q in zip(g.respondent_id, g.question_id)]]
        out[m] = (np.bincount(g.respondent_id.map(ix), weights=g[col], minlength=len(ids)),
                  np.bincount(g.respondent_id.map(ix), minlength=len(ids)).astype(float))
    return out, len(ids), len(common)


def boot(a, b, n=2000):
    """a, b = (sum, count) arrays per respondent; returns mean diff a-b and 95% CI under respondent resampling."""
    R = len(a[0]); W = rng.multinomial(R, np.ones(R) / R, size=n).astype(float)
    pa = (W @ a[0]) / (W @ a[1]); pb = (W @ b[0]) / (W @ b[1]); d = pa - pb
    return float(a[0].sum() / a[1].sum()), float(b[0].sum() / b[1].sum()), float(d.mean()), np.percentile(d, [2.5, 97.5]).tolist()


res = {}
def cmp(x, y, gname, col="correct"):
    if x not in M or y not in M:
        return None
    t, nr, nrow = table([x, y], GROUPS[gname], col)
    if nrow == 0:
        return None
    ax, ay, d, ci = boot(t[x], t[y])
    return {"a": ax, "b": ay, "diff": d, "ci": ci, "n_resp": nr, "n_rows": nrow}

pairs = [("A", "B"), ("A", "base"), ("B", "base"), ("A", "gboost|matchedA"), ("A", "logistic|matchedA"), ("A", "majority|matchedA"), ("A", "majority|pool2k"),
         ("B", "gboost|pool2k"), ("B", "logistic|pool2k"), ("B", "majority|pool2k"), ("A", "gboost|pool2k"), ("B", "gboost|full"),
         ("A", "gboost|full"), ("gboost|pool2k", "majority|pool2k"), ("gboost|full", "majority|full")]
for g in GROUPS:
    for x, y in pairs:
        for col in ("correct", "nll"):
            r = cmp(x, y, g, col)
            if r:
                res.setdefault(g, {})[f"{x} - {y} [{col}]"] = r

# difference-in-gains for adapter vs base: (gain on trained items) - (gain on held-out items), bootstrap over respondents
def did(ad, tr_group, un_group):
    if ad not in M or "base" not in M:
        return None
    # bootstrap over respondents common to both tables
    def per_resp(t, m): return pd.Series((t[m][0] / np.maximum(t[m][1], 1)))
    # rebuild with explicit respondent ids to align
    def tab(groups):
        g = M[ad][M[ad].question_id.isin(groups)].merge(M["base"][M["base"].question_id.isin(groups)], on=["respondent_id", "question_id"], suffixes=("_a", "_b"))
        return g.assign(d=g.correct_a - g.correct_b).groupby("respondent_id").d.agg(["sum", "size"])
    a, b = tab(GROUPS[tr_group]), tab(GROUPS[un_group]); ids = a.index.intersection(b.index)
    if len(ids) == 0:
        return None
    a, b = a.loc[ids], b.loc[ids]; R = len(ids); W = rng.multinomial(R, np.ones(R) / R, size=2000).astype(float)
    ga = (W @ a["sum"].values) / (W @ a["size"].values); gb = (W @ b["sum"].values) / (W @ b["size"].values)
    d = ga - gb
    return {"trained_gain": float(a["sum"].sum() / a["size"].sum()), "unseen_gain": float(b["sum"].sum() / b["size"].sum()),
            "did": float(d.mean()), "ci": np.percentile(d, [2.5, 97.5]).tolist(), "n_resp": R}
res["did"] = {f"{ad}: {tg} vs {ug}": did(ad, tg, ug) for ad in ("A", "B") for tg, ug in [("trained_B(0:8)", "heldout(24:40)"), ("trained_A(0:24)", "heldout(24:40)")]}

# near vs far held-out items (does a sibling item from the same battery sit in adapter A's training set?)
fam_tr = {SIB(q) for q in items[0:24]}
near = [q for q in items[24:40] if SIB(q) in fam_tr]; far = [q for q in items[24:40] if SIB(q) not in fam_tr]
GROUPS["heldout_near"], GROUPS["heldout_far"] = near, far
res["heldout_near_items"], res["heldout_far_items"] = near, far
for g in ("heldout_near", "heldout_far"):
    for x, y in [("A", "base"), ("B", "base"), ("A", "B"), ("A", "gboost|pool2k"), ("A", "majority|pool2k")]:
        r = cmp(x, y, g)
        if r:
            res.setdefault(g, {})[f"{x} - {y} [correct]"] = r

# per-item table (accuracy)
per = {}
for q in items:
    row = {}
    for k in ("A", "B", "base", "majority|pool2k", "logistic|pool2k", "gboost|pool2k", "gboost|full"):
        if k in M:
            g = M[k][M[k].question_id == q]
            if len(g):
                row[k] = [round(float(g.correct.mean()), 4), int(len(g)), round(float(g.nll.mean()), 4)]
    per[q] = row
res["per_item"] = per


# per-group accuracy / NLL for every model on the rows common to all of them (for the main table)
res["group_acc"] = {}
core = [m for m in ("base", "B", "A", "majority|pool2k", "logistic|pool2k", "gboost|pool2k", "majority|full", "logistic|full", "gboost|full") if m in M]
for g in ("trained_B(0:8)", "mid(8:24)", "heldout(24:40)", "heldout_near", "heldout_far"):
    if len(core) < 2 or not GROUPS.get(g):
        continue
    out_g = {}
    for col in ("correct", "nll"):
        tt, nr, nrow = table(core, GROUPS[g], col)
        if nrow == 0:
            break
        for m in core:
            R = len(tt[m][0]); Wb = rng.multinomial(R, np.ones(R) / R, size=1000).astype(float)
            v = (Wb @ tt[m][0]) / (Wb @ tt[m][1])
            out_g.setdefault(m, {})[col] = [float(tt[m][0].sum() / tt[m][1].sum()), np.percentile(v, [2.5, 97.5]).tolist()]
            out_g[m]["n_rows"] = nrow
    res["group_acc"][g] = out_g

# directional validity on each item group: item-centred group differences in expected answer index, for religion/region/sex
dfp = pd.read_parquet("/tmp/ssc_pew/data/processed/pew_india_2021.parquet"); dfp["respondent_id"] = dfp.respondent_id.astype(int); dfp = dfp.set_index("respondent_id")
def cells(x, qs, col, what):
    x = x[x.question_id.isin(qs)]
    v = np.array([np.dot(np.arange(len(p)), p) for p in x.probs]) if what == "pred" else x.true_idx.values.astype(float)
    t = pd.DataFrame({"q": x.question_id.values, "g": dfp.loc[x.respondent_id.values, col].values, "v": v}).dropna()
    c = t.groupby(["q", "g"]).v.agg(["mean", "size"]); c = c[c["size"] >= 25]["mean"]
    return c - c.groupby(level=0).transform("mean")
res["direction"] = {}
for gname in ("trained_B(0:8)", "mid(8:24)", "heldout(24:40)"):
    for ax, col in (("religion", "QRELSING"), ("region", "REGION"), ("sex", "QGEN")):
        row = {}
        for k in ("A", "B", "base", "gboost|pool2k", "logistic|pool2k", "gboost|full"):
            if k not in M or "A" not in M:
                continue
            tr = cells(M["A"], GROUPS[gname], col, "true"); pr = cells(M[k], GROUPS[gname], col, "pred")
            j = pd.concat([tr, pr], axis=1, keys=["t", "p"]).dropna()
            if len(j) > 5 and j.p.std() > 1e-9 and j.t.std() > 1e-9:
                row[k] = {"r": float(np.corrcoef(j.t, j.p)[0, 1]), "slope": float(np.polyfit(j.t, j.p, 1)[0]), "n_cells": int(len(j))}
        res["direction"][f"{gname}|{ax}"] = row
json.dump(res, open(out, "w"), indent=1, default=float)
for g in ("trained_B(0:8)", "mid(8:24)", "heldout(24:40)"):
    print("==", g)
    for k, v in res.get(g, {}).items():
        if "[correct]" in k:
            print(f"  {k:40s} {100*v['a']:5.1f} vs {100*v['b']:5.1f}  diff {100*v['diff']:+5.1f} [{100*v['ci'][0]:+5.1f},{100*v['ci'][1]:+5.1f}]  n_resp={v['n_resp']}")
print("== difference in gains")
for k, v in res["did"].items():
    if v: print(f"  {k}: trained {100*v['trained_gain']:+.1f} unseen {100*v['unseen_gain']:+.1f} did {100*v['did']:+.1f} [{100*v['ci'][0]:+.1f},{100*v['ci'][1]:+.1f}]")
