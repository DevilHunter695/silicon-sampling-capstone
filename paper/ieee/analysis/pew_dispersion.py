"""Dispersion retention and between-group gap inflation, computed from existing aligned fold-0 rows (hard predictions)."""
import json, numpy as np, pandas as pd
rng = np.random.default_rng(77)
rows = pd.read_parquet("analysis/pew_aligned_rows.parquet"); rows = rows[rows.pred_idx.notna()].copy()
M = ["majority", "logistic", "gboost", "finetuned"]
df = pd.read_parquet("/tmp/ssc_pew/data/processed/pew_india_2021.parquet"); df["respondent_id"] = df.respondent_id.astype(int); df = df.set_index("respondent_id")
cb = json.load(open("/tmp/ssc_pew/data/reference/pew_codebook.json"))["questions"]
lab = lambda c: {int(k): v for k, v in cb[c]["valid_codes"].items()}
axes = {"Sex": df.QGEN.map(lab("QGEN")), "Urban/rural": df.Urban.map(lab("Urban")), "Religion": df.QRELSING.map(lab("QRELSING")),
        "Region": df.REGION.map(lab("REGION")), "Caste": df.QCASTE.map(lab("QCASTE")), "Age": df.QAGErec.map(lab("QAGErec")), "Income": df.QINCINDrec.map(lab("QINCINDrec"))}
ids = np.sort(rows.respondent_id.unique()); R = len(ids); ri = {r: i for i, r in enumerate(ids)}
rows["ri"] = rows.respondent_id.map(ri)
ordr = rows[rows.question_id != "Q28rec"]
items = sorted(ordr.question_id.unique())
K = {q: len(cb[q]["valid_codes"]) for q in items}
# normalise index to [0,1] by (K-1) so items are comparable
ordr = ordr.assign(t=ordr.true_idx / ordr.question_id.map(lambda q: K[q] - 1), p=ordr.pred_idx / ordr.question_id.map(lambda q: K[q] - 1))
cat = {ax: ser.reindex(ids).to_numpy() for ax, ser in axes.items()}
W = rng.multinomial(R, np.ones(R) / R, size=500).astype(float)

def stats(w, method):
    d = ordr[ordr.method == method]
    # dispersion retention: SD(pred)/SD(true) per item, weighted
    disp = []
    for q in items:
        dq = d[d.question_id == q]; ww = w[dq.ri.to_numpy()]
        mt = (ww * dq.t).sum() / ww.sum(); mp = (ww * dq.p).sum() / ww.sum()
        sdt = np.sqrt((ww * (dq.t - mt) ** 2).sum() / ww.sum()); sdp = np.sqrt((ww * (dq.p - mp) ** 2).sum() / ww.sum())
        disp.append(sdp / sdt if sdt > 0 else np.nan)
    # between-group gap inflation: range of group mean pred / range of group mean true, per axis & item
    infl = []
    for ax, c in cat.items():
        cats = [x for x in pd.Series(c).dropna().unique() if (c == x).sum() >= 100]
        for q in items:
            dq = d[d.question_id == q]; r_ = dq.ri.to_numpy(); ww = w[r_]
            mt, mp = [], []
            for x in cats:
                sel = (c[r_] == x)
                if ww[sel].sum() == 0: continue
                mt.append((ww[sel] * dq.t.to_numpy()[sel]).sum() / ww[sel].sum()); mp.append((ww[sel] * dq.p.to_numpy()[sel]).sum() / ww[sel].sum())
            if len(mt) >= 2 and (max(mt) - min(mt)) > 0.01:
                infl.append((max(mp) - min(mp)) / (max(mt) - min(mt)))
    return float(np.nanmean(disp)), float(np.median(infl)), float(np.mean(infl)), disp

out = {}
for m in M:
    est = stats(np.ones(R), m)
    bs = np.array([stats(W[b], m)[:3] for b in range(150)])
    out[m] = {"dispersion_retention": est[0], "dispersion_ci": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
              "gap_inflation_median": est[1], "gap_inflation_mean": est[2], "gap_infl_ci": np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
              "per_item_dispersion": dict(zip(items, [float(x) for x in est[3]]))}
json.dump(out, open("analysis/pew_dispersion.json", "w"), indent=1)
for m in M: print(m, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in out[m].items() if k != "per_item_dispersion"})
