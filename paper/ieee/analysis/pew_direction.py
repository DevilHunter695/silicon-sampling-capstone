"""Directional validity of group differences, and demographic variance decomposition (Pew fold 0; ordinal items only).

For each (item, axis, category) cell: d_pred = group mean prediction - item mean prediction; d_true likewise (indices scaled to [0,1]).
Reports Pearson r(d_pred, d_true) and the OLS slope (slope > 1 means inflation), with respondent-cluster bootstrap CIs.
Also: share of variance in the (soft or hard) prediction vs. in the true answer explained by item, and by item + demographics.
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(404)
cb = json.load(open("/tmp/ssc_pew/data/reference/pew_codebook.json"))["questions"]
K = {q: len(cb[q]["valid_codes"]) for q in cb if "valid_codes" in cb[q]}
rows = pd.read_parquet(f"{HERE}/pew_aligned_rows.parquet"); rows = rows[rows.pred_idx.notna() & (rows.question_id != "Q28rec")]
z = pd.read_json(f"{HERE}/mlx_base_chat.jsonl", lines=True).drop_duplicates(["respondent_id", "question_id"])
z = z[z.question_id != "Q28rec"].copy()
z["soft"] = [float(np.dot(p, np.arange(len(p))) / (len(p) - 1)) for p in z.probs]
z["hard"] = z.pred_idx / (z.probs.map(len) - 1)
z["true"] = z.true_idx / (z.probs.map(len) - 1)
df = pd.read_parquet("/tmp/ssc_pew/data/processed/pew_india_2021.parquet"); df["respondent_id"] = df.respondent_id.astype(int); df = df.set_index("respondent_id")
lab = lambda c: {int(k): v for k, v in cb[c]["valid_codes"].items()}
edu = {1: "a", 2: "a", 3: "b", 4: "b", 5: "c", 6: "c", 7: "c", 8: "c", 9: "d", 10: "d", 11: "d"}
axes = {"Sex": df.QGEN.map(lab("QGEN")), "Urban": df.Urban.map(lab("Urban")), "Religion": df.QRELSING.map(lab("QRELSING")),
        "Region": df.REGION.map(lab("REGION")), "Caste": df.QCASTE.map(lab("QCASTE")), "Age": df.QAGErec.map(lab("QAGErec")),
        "Income": df.QINCINDrec.map(lab("QINCINDrec")), "Education": df.QEDU.map(edu)}

def cells_corr(d, col, truecol, n_min, resp_weights=None):
    """d has respondent_id, question_id, col (pred), truecol. Returns r and slope over centered cells."""
    xs, ys = [], []
    w = d.respondent_id.map(resp_weights).to_numpy() if resp_weights is not None else np.ones(len(d))
    for q, g in d.groupby("question_id"):
        ww = w[d.index.get_indexer(g.index)] if resp_weights is not None else np.ones(len(g))
        mp = np.average(g[col], weights=ww); mt = np.average(g[truecol], weights=ww)
        for ax, ser in axes.items():
            c = ser.reindex(g.respondent_id).to_numpy()
            for cat in pd.unique(c[~pd.isna(c)]):
                m = (c == cat)
                if len(set(g.respondent_id[m])) < n_min or ww[m].sum() == 0: continue
                xs.append(np.average(g[col].to_numpy()[m], weights=ww[m]) - mp); ys.append(np.average(g[truecol].to_numpy()[m], weights=ww[m]) - mt)
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 5 or xs.std() < 1e-9: return np.nan, np.nan, len(xs)
    return float(np.corrcoef(xs, ys)[0, 1]), float(np.polyfit(ys, xs, 1)[0]), len(xs)

def boot(d, col, truecol, n_min, B=1000):
    ids = d.respondent_id.unique(); out = []
    for _ in range(B):
        cnt = pd.Series(rng.multinomial(len(ids), np.ones(len(ids)) / len(ids)), index=ids)
        r, s, _ = cells_corr(d, col, truecol, n_min, resp_weights=cnt)
        out.append((r, s))
    a = np.array(out); return [np.nanpercentile(a[:, 0], [2.5, 97.5]).tolist(), np.nanpercentile(a[:, 1], [2.5, 97.5]).tolist()]

res = {}
# --- parity on the 200-respondent subset
sub = z[["respondent_id", "question_id", "soft", "hard", "true"]].copy()
for m in ["finetuned", "gboost", "logistic"]:
    h = rows[rows.method == m][["respondent_id", "question_id", "pred_idx"]]
    sub = sub.merge(h.rename(columns={"pred_idx": m}), on=["respondent_id", "question_id"], how="left")
    sub[m] = sub[m] / sub.question_id.map(lambda q: K[q] - 1)
sub = sub.dropna().reset_index(drop=True)
for name, col in [("zero_shot_soft", "soft"), ("zero_shot_hard", "hard"), ("finetuned", "finetuned"), ("gboost", "gboost"), ("logistic", "logistic")]:
    r, s, n = cells_corr(sub, col, "true", 15)
    ci = boot(sub, col, "true", 15) if not np.isnan(r) else [[np.nan, np.nan], [np.nan, np.nan]]
    res[f"sub200_{name}"] = {"r": r, "slope": s, "n_cells": n, "r_ci": ci[0], "slope_ci": ci[1]}
    print("sub200", name, round(r, 3), round(s, 2), n, [round(x, 2) for x in ci[0]])
# --- full fold
full = rows[["respondent_id", "question_id", "method", "pred_idx", "true_idx"]].copy()
full["k"] = full.question_id.map(lambda q: K[q] - 1); full["p"] = full.pred_idx / full.k; full["t"] = full.true_idx / full.k
for m in ["finetuned", "gboost", "logistic"]:
    d = full[full.method == m].reset_index(drop=True)
    r, s, n = cells_corr(d, "p", "t", 100); ci = boot(d, "p", "t", 100, B=300)
    res[f"full_{m}"] = {"r": r, "slope": s, "n_cells": n, "r_ci": ci[0], "slope_ci": ci[1]}
    print("full", m, round(r, 3), round(s, 2), n, [round(x, 2) for x in ci[0]])
# --- variance decomposition on the 200 subset
def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
dm = pd.DataFrame({a: ser.reindex(sub.respondent_id).to_numpy() for a, ser in axes.items()})
item_d = pd.get_dummies(sub.question_id, drop_first=True).to_numpy().astype(float)
demo_d = pd.get_dummies(dm.astype(str), drop_first=True).to_numpy().astype(float)
vd = {}
for name, col in [("true", "true"), ("zero_shot_soft", "soft"), ("finetuned", "finetuned"), ("gboost", "gboost")]:
    y = sub[col].to_numpy()
    base = r2(y, item_d); full_ = r2(y, np.column_stack([item_d, demo_d]))
    vd[name] = {"r2_item": base, "r2_item_demo": full_, "incremental_demo": full_ - base}
res["variance_decomposition"] = vd
print(json.dumps(vd, indent=1))
json.dump(res, open(f"{HERE}/pew_direction.json", "w"), indent=1, default=float)
