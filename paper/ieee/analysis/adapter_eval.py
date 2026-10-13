"""(1) Unseen-item transfer: the delivered adapter vs the zero-shot base on 8 Pew items never used in fine-tuning.
(2) Port validation + fine-tuned probabilities on the 8 trained items (MLX re-implementation of the adapter).
All comparisons are on identical (respondent, item) rows; respondent-clustered bootstrap CIs."""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/tmp/ssc_pew"); os.chdir("/tmp/ssc_pew")
from scripts.pewB_baselines import run_fold
from src.config import DATA_PROCESSED
from src.prompts.verbalize_pew import load_pew_codebook
rng = np.random.default_rng(2028)
key = ["respondent_id", "question_id"]
cb = load_pew_codebook()
folds = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"]
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet"); df["respondent_id"] = df.respondent_id.astype(int)
UN = ["Q34a", "Q13", "Q14b", "Q69a", "Q69b", "Q39b", "Q14c", "Q54a"]
TR = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][:8]
K = {q: len(cb[q]["valid_codes"]) for q in UN + TR}
axes_src = {"Sex": ("QGEN", None), "Urban": ("Urban", None), "Religion": ("QRELSING", None), "Region": ("REGION", None),
            "Caste": ("QCASTE", None), "Age": ("QAGErec", None), "Income": ("QINCINDrec", None)}
dfi = df.set_index("respondent_id")


def load(name):
    p = f"{HERE}/{name}"
    if not os.path.exists(p):
        return None
    d = pd.read_json(p, lines=True)
    return d.drop_duplicates(key) if len(d) else None


def boot_acc(S, correct, W):
    s = np.bincount(S.ri, weights=correct, minlength=S.NR); n = np.bincount(S.ri, minlength=S.NR).astype(float)
    return s, n


def _cells(rows, col, true_col, n_min, w=None):
    xs, ys = [], []
    for q, g in rows.groupby("question_id"):
        ww = np.ones(len(g)) if w is None else g.respondent_id.map(w).to_numpy().astype(float)
        if ww.sum() == 0:
            continue
        mp, mt = np.average(g[col], weights=ww), np.average(g[true_col], weights=ww)
        for ax, (c, _) in axes_src.items():
            lab = dfi.loc[g.respondent_id, c].to_numpy()
            for cat in pd.unique(lab[~pd.isna(lab)]):
                m = lab == cat
                if len(set(g.respondent_id.to_numpy()[m])) < n_min or ww[m].sum() == 0:
                    continue
                xs.append(np.average(g[col].to_numpy()[m], weights=ww[m]) - mp); ys.append(np.average(g[true_col].to_numpy()[m], weights=ww[m]) - mt)
    return np.array(xs), np.array(ys)


def direction(rows, col, true_col="true", n_min=15, B=1000):
    xs, ys = _cells(rows, col, true_col, n_min)
    if len(xs) < 5 or xs.std() < 1e-9:
        return None
    out = {"r": float(np.corrcoef(xs, ys)[0, 1]), "slope": float(np.polyfit(ys, xs, 1)[0]), "n_cells": int(len(xs))}
    ids = rows.respondent_id.unique(); rs = []
    for _ in range(B):
        w = pd.Series(rng.multinomial(len(ids), np.ones(len(ids)) / len(ids)), index=ids)
        x, y = _cells(rows, col, true_col, n_min, w)
        if len(x) >= 5 and x.std() > 1e-9:
            rs.append(np.corrcoef(x, y)[0, 1])
    if rs:
        out["r_ci"] = np.percentile(rs, [2.5, 97.5]).tolist()
    return out


def evaluate(runs, items, tag, extra_hard=None):
    """runs: dict name -> dataframe (respondent_id, question_id, true_idx, probs, pred_idx)."""
    S = None
    for d in runs.values():
        k = d[key]; S = k if S is None else S.merge(k, on=key)
    if extra_hard is not None:
        S = S.merge(extra_hard[key], on=key)
    S = S.sort_values(key).reset_index(drop=True)
    idx = S.set_index(key).index
    first = next(iter(runs.values())).drop_duplicates(key).set_index(key).loc[idx]
    true = first.true_idx.to_numpy().astype(int)
    S["true"] = true
    ids = np.sort(S.respondent_id.unique()); ri = {r: i for i, r in enumerate(ids)}
    S["ri"] = S.respondent_id.map(ri); NR = len(ids)
    W = rng.multinomial(NR, np.ones(NR) / NR, size=3000).astype(float)
    preds, probs = {}, {}
    for n, d in runs.items():
        dd = d.set_index(key).loc[idx]
        preds[n] = dd.pred_idx.to_numpy().astype(int); probs[n] = list(dd.probs)
    if extra_hard is not None:
        eh = extra_hard.set_index(key).loc[idx]
        preds["original_ft"] = eh.pred_idx.to_numpy().astype(int)
    # classical baselines trained with labels on fold-0 train (ceilings in the unseen-item case)
    base = run_fold(df, folds[0], items, cb)
    for m, dm in base.items():
        dm = dm.copy(); dm["respondent_id"] = dm.respondent_id.astype(int)
        dd = dm.drop_duplicates(key).set_index(key).loc[idx]
        preds[m] = np.array([int(np.argmax(p)) for p in dd.proba]); probs[m] = list(dd.proba)
    Ks = S.question_id.map(lambda q: K[q]).to_numpy()
    preds["uniform_chance"] = None
    store, out = {}, {"tag": tag, "n_rows": int(len(S)), "n_resp": int(NR), "methods": {}, "paired": {}, "per_item": {}}

    def stats(correct):
        s = np.bincount(S.ri, weights=correct, minlength=NR); n = np.bincount(S.ri, minlength=NR).astype(float)
        return s, n

    for name, p in preds.items():
        if p is None:
            s, n = stats(1.0 / Ks)
        else:
            s, n = stats((p == true).astype(float))
        store[name] = (s, n)
        r = {"acc": float(s.sum() / n.sum()), "ci": np.percentile((W @ s) / (W @ n), [2.5, 97.5]).tolist()}
        if name in probs:
            pr = probs[name]
            r["nll"] = float(np.mean([-np.log(max(pr[i][true[i]], 1e-6)) for i in range(len(true))]))
            k_mean = np.mean(Ks)
        if p is not None:
            tops = [float(np.bincount(p[(S.question_id == q).to_numpy()]).max() / (S.question_id == q).sum()) for q in items if (S.question_id == q).sum()]
            r["mean_top_share"] = float(np.mean(tops))
        out["methods"][name] = r
    out["methods"]["uniform_chance"]["nll"] = float(np.mean(np.log(Ks)))

    def paired(a, b):
        sa, na = store[a]; sb, nb = store[b]
        d = (W @ sa) / (W @ na) - (W @ sb) / (W @ nb)
        return {"diff": float(sa.sum() / na.sum() - sb.sum() / nb.sum()), "ci": np.percentile(d, [2.5, 97.5]).tolist()}

    names = list(runs)
    for a in names:
        for b in names:
            if a != b and a.startswith("adapter") and b.startswith("base"):
                out["paired"][f"{a}-{b}"] = paired(a, b)
        for b in ["uniform_chance", "majority", "logistic", "gboost"]:
            out["paired"][f"{a}-{b}"] = paired(a, b)
    if "original_ft" in preds:
        for a in names:
            if a.startswith("adapter"):
                out["paired"][f"{a}-original_ft"] = paired(a, "original_ft")
                out[f"agree_{a}_original_ft"] = float((preds[a] == preds["original_ft"]).mean())
        out["agree_original_ft_gboost"] = float((preds["original_ft"] == preds["gboost"]).mean())
    for a in names:
        for b in names:
            if a < b:
                out[f"agree_{a}_{b}"] = float((preds[a] == preds[b]).mean())
    for q in items:
        mk = (S.question_id == q).to_numpy()
        if mk.sum():
            out["per_item"][q] = {n: (float((p[mk] == true[mk]).mean()) if p is not None else float(np.mean(1.0 / K[q]))) for n, p in preds.items()}
    # directional validity (soft expected value for probability methods; hard for the rest), scaled to [0,1]
    S2 = S[key].copy(); S2["true"] = true / (Ks - 1)
    out["direction"] = {}
    for n in list(probs) + (["original_ft"] if "original_ft" in preds else []):
        if n in probs:
            S2[n] = [float(np.dot(p_, np.arange(len(p_))) / (len(p_) - 1)) for p_ in probs[n]]
            if n in names:
                S2[n + "_hard"] = preds[n] / (Ks - 1)
        else:
            S2[n] = preds[n] / (Ks - 1)
    for n in [c for c in S2.columns if c not in key + ["true"]]:
        out["direction"][n] = direction(S2.rename(columns={n: "x"}), "x")
    # distribution fidelity of soft predictions
    for n in names:
        tv = []
        for q in items:
            ix = np.where((S.question_id == q).to_numpy())[0]
            if len(ix) == 0: continue
            pr = np.stack([probs[n][i] for i in ix]).mean(0); tm = np.bincount(true[ix], minlength=len(pr)) / len(ix)
            tv.append(0.5 * np.abs(pr - tm).sum())
        out["methods"][n]["soft_tv"] = float(np.mean(tv))
    # ensemble of adapter probabilities with GBM probabilities
    for a in names:
        if a.startswith("adapter"):
            ens = np.array([int(np.argmax(0.5 * np.array(probs[a][i]) + 0.5 * np.array(probs["gboost"][i]))) for i in range(len(true))])
            s, n = stats((ens == true).astype(float)); store["ens"] = (s, n)
            out["ensemble"] = {"acc": float(s.sum() / n.sum()), "ci": np.percentile((W @ s) / (W @ n), [2.5, 97.5]).tolist(),
                               "vs_gboost": paired("ens", "gboost"), "vs_adapter": paired("ens", a),
                               "nll": float(np.mean([-np.log(max(0.5 * probs[a][i][true[i]] + 0.5 * probs["gboost"][i][true[i]], 1e-6)) for i in range(len(true))]))}
    return out


res = {}
ua, ub = load("mlx_unseen_adapter_raw.jsonl"), load("mlx_unseen_base_raw.jsonl")
if ua is not None and ub is not None:
    res["unseen"] = evaluate({"adapter_raw": ua, "base_raw": ub}, UN, "unseen")
elif ua is not None:
    res["unseen_adapter_only"] = evaluate({"adapter_raw": ua}, UN, "unseen_adapter_only")
ta = load("mlx_adapter_raw.jsonl"); tb = load("mlx_base_raw.jsonl")
orig = pd.read_parquet(f"{HERE}/pew_aligned_rows.parquet")
orig = orig[orig.method == "finetuned"][key + ["pred_idx"]].copy()
if ta is not None:
    runs = {"adapter_raw": ta}
    if tb is not None:
        runs["base_raw"] = tb
    res["trained"] = evaluate(runs, TR, "trained", extra_hard=orig)
json.dump(res, open(f"{HERE}/adapter_eval.json", "w"), indent=1)
for tag, r in res.items():
    print(f"== {tag}: {r['n_rows']} rows / {r['n_resp']} resp")
    for n, m in r["methods"].items():
        print(f"  {n:16s} acc={100*m['acc']:.1f} [{100*m['ci'][0]:.1f},{100*m['ci'][1]:.1f}]" + (f" nll={m['nll']:.3f}" if "nll" in m else "") + (f" top={m['mean_top_share']:.2f}" if "mean_top_share" in m else "") + (f" softTV={m['soft_tv']:.3f}" if "soft_tv" in m else ""))
    for k, v in r["paired"].items():
        print(f"  {k:36s} {100*v['diff']:+.1f} [{100*v['ci'][0]:+.1f},{100*v['ci'][1]:+.1f}]")
    print("  direction:", {k: (None if v is None else (round(v['r'], 2), [round(x, 2) for x in v.get('r_ci', [])], round(v['slope'], 2))) for k, v in r["direction"].items()})
    for k in r:
        if k.startswith("agree"):
            print("  ", k, round(r[k], 3))
    if "ensemble" in r:
        print("  ensemble", r["ensemble"])
