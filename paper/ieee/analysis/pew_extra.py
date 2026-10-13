"""Extra diagnostics on the aligned fold-0 Pew rows: concentration for every method, method agreement,
per-item paired differences, and subgroup-mean estimation error (hard predictions, ordinal items)."""
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(31)
rows = pd.read_parquet("analysis/pew_aligned_rows.parquet")
rows = rows[rows.pred_idx.notna()].copy()
M = ["majority", "logistic", "gboost", "finetuned"]
items = list(rows.question_id.unique())
out = {"concentration": {}, "agreement": {}, "per_item_paired": {}, "mean_error": {}}

# A. concentration and marginal distortion per item and method
for q in items:
    out["concentration"][q] = {}
    for m in M:
        d = rows[(rows.question_id == q) & (rows.method == m)]
        k = int(max(d.true_idx.max(), d.pred_idx.max())) + 1
        p = np.bincount(d.pred_idx.astype(int), minlength=k) / len(d)
        t = np.bincount(d.true_idx.astype(int), minlength=k) / len(d)
        out["concentration"][q][m] = {"top_pred_share": float(p.max()), "tv_marginal": float(0.5 * np.abs(p - t).sum()),
                                      "n_pred_classes_ge5pct": int((p >= 0.05).sum()), "true_top_share": float(t.max())}
out["concentration_mean"] = {m: {k: float(np.mean([out["concentration"][q][m][k] for q in items]))
                                 for k in ("top_pred_share", "tv_marginal")} for m in M}

# B. agreement between methods (row level), respondent-cluster bootstrap
wide = rows.pivot_table(index=["respondent_id", "question_id"], columns="method", values="pred_idx", aggfunc="first").dropna()


def kappa(a, b):
    cats = np.union1d(a, b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe)


rid = wide.index.get_level_values(0).to_numpy()
u = np.unique(rid)
order = np.argsort(rid, kind="stable")
bounds = {r: (np.searchsorted(rid[order], r, "left"), np.searchsorted(rid[order], r, "right")) for r in u}
for a, b in [("finetuned", "gboost"), ("finetuned", "logistic"), ("finetuned", "majority"), ("gboost", "logistic")]:
    x, y = wide[a].to_numpy(), wide[b].to_numpy()
    ag = []
    for _ in range(300):
        pick = rng.choice(u, len(u))
        ix = np.concatenate([order[bounds[r][0]:bounds[r][1]] for r in pick])
        ag.append((x[ix] == y[ix]).mean())
    out["agreement"][f"{a}~{b}"] = {"agree": float((x == y).mean()), "ci": np.percentile(ag, [2.5, 97.5]).tolist(),
                                    "kappa": float(kappa(x, y))}

# C. per-item paired differences (respondent-clustered)
ids = np.sort(rows.respondent_id.unique())
ri = {r: i for i, r in enumerate(ids)}
R = len(ids)
W = rng.multinomial(R, np.ones(R) / R, size=2000).astype(float)
rows["ri"] = rows.respondent_id.map(ri)
rows["correct"] = (rows.pred_idx == rows.true_idx).astype(float)
for q in items:
    s, n = {}, {}
    for m in M:
        d = rows[(rows.question_id == q) & (rows.method == m)]
        s[m] = np.bincount(d.ri, weights=d.correct, minlength=R)
        n[m] = np.bincount(d.ri, minlength=R).astype(float)
    r = {}
    for b in ("gboost", "majority"):
        diff = (W @ s["finetuned"]) / (W @ n["finetuned"]) - (W @ s[b]) / (W @ n[b])
        r["ft_minus_" + b] = {"diff": float(s["finetuned"].sum() / n["finetuned"].sum() - s[b].sum() / n[b].sum()),
                              "ci": np.percentile(diff, [2.5, 97.5]).tolist()}
    out["per_item_paired"][q] = r

# D. subgroup-mean estimation error (ordinal items, hard predictions)
df = pd.read_parquet("/tmp/ssc_pew/data/processed/pew_india_2021.parquet")
df["respondent_id"] = df.respondent_id.astype(int)
df = df.set_index("respondent_id")
cb = json.load(open("/tmp/ssc_pew/data/reference/pew_codebook.json"))["questions"]


def lab(c):
    return {int(k): v for k, v in cb[c]["valid_codes"].items()}


axes = {"Sex": df.QGEN.map(lab("QGEN")), "Urban/rural": df.Urban.map(lab("Urban")),
        "Religion": df.QRELSING.map(lab("QRELSING")), "Region": df.REGION.map(lab("REGION")),
        "Caste": df.QCASTE.map(lab("QCASTE")), "Age": df.QAGErec.map(lab("QAGErec")),
        "Income": df.QINCINDrec.map(lab("QINCINDrec"))}
ordr = rows[rows.question_id != "Q28rec"]
cells = []
for ax, ser in axes.items():
    s_ = ser.reindex(ids)
    for c in s_.dropna().unique():
        idx = np.where((s_ == c).to_numpy())[0]
        if len(idx) >= 100:
            cells.append((ax, c, idx))
# precompute per (method, cell, item): arrays of respondent index, pred, true
pre = {}
for m in M:
    d = ordr[ordr.method == m]
    for q in d.question_id.unique():
        dq = d[d.question_id == q]
        kq = len(cb[q]["valid_codes"]) - 1
        pre[(m, q)] = (dq.ri.to_numpy(), dq.pred_idx.to_numpy() / kq, dq.true_idx.to_numpy() / kq)
member = {}
for ax, c, idx in cells:
    mask = np.zeros(R, bool)
    mask[idx] = True
    member[(ax, c)] = mask


def mean_err(w, m):
    res = []
    for (ax, c), mask in member.items():
        for q in {k[1] for k in pre if k[0] == m}:
            r_, p_, t_ = pre[(m, q)]
            sel = mask[r_]
            if sel.sum() < 100:
                continue
            ww = w[r_[sel]]
            if ww.sum() == 0:
                continue
            res.append(abs((ww * p_[sel]).sum() / ww.sum() - (ww * t_[sel]).sum() / ww.sum()))
    return float(np.mean(res))


est = {m: mean_err(np.ones(R), m) for m in M}
boots = {m: [] for m in M}
for b in range(500):
    for m in M:
        boots[m].append(mean_err(W[b], m))
out["mean_error"] = {m: {"est": est[m], "ci": np.percentile(boots[m], [2.5, 97.5]).tolist()} for m in M}
d_ = np.array(boots["finetuned"]) - np.array(boots["gboost"])
out["mean_error"]["ft_minus_gboost"] = {"diff": est["finetuned"] - est["gboost"], "ci": np.percentile(d_, [2.5, 97.5]).tolist()}
json.dump(out, open("analysis/pew_extra.json", "w"), indent=1)

print(json.dumps({k: out[k] for k in ("concentration_mean", "agreement", "mean_error")}, indent=1))
print({q: {m: round(v[m]["top_pred_share"], 2) for m in M} for q, v in out["concentration"].items()})
print({q: (round(100 * v["ft_minus_gboost"]["diff"], 1), [round(100 * x, 1) for x in v["ft_minus_gboost"]["ci"]])
       for q, v in out["per_item_paired"].items()})
print({q: (round(100 * v["ft_minus_majority"]["diff"], 1), [round(100 * x, 1) for x in v["ft_minus_majority"]["ci"]])
       for q, v in out["per_item_paired"].items()})
