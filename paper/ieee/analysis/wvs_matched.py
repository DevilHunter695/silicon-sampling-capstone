"""WVS-7: score supervised baselines on EXACTLY the (respondent, item) rows the LLMs answered, out-of-fold,
using mode-predicting baselines (majority class, cell-mode) in addition to the repo's sampling-based ones."""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = "/Users/krishank/Documents/College Projects/🧢 /silicon-sampling-capstone"
sys.path.insert(0, REPO); os.chdir(REPO)
from src.config import DATA_PROCESSED
from src.eval.baselines import SupervisedClassifierBaseline
from src.eval.run_baselines import DEMO_COLS, MATCHED_DEMO_COLS, _encode_item
from src.prompts.verbalize import load_codebook

rng = np.random.default_rng(11)
df = pd.read_parquet(f"{DATA_PROCESSED}/ind_wvs7.parquet")
if "respondent_id" not in df.columns:
    df["respondent_id"] = range(len(df))
df["respondent_id"] = df["respondent_id"].astype(int)
folds = json.load(open(f"{DATA_PROCESSED}/folds.json"))["folds"]
codebook = load_codebook()
P = REPO + "/results/predictions/"
llm = {"Gemini": pd.read_parquet(P + "gemini-3.1-flash-lite_P2.parquet"), "gpt-oss": pd.read_parquet(P + "groq-openai_gpt-oss-120b-low_P2.parquet")}
items = sorted(llm["gpt-oss"].question_id.unique())
sample_ids = set(llm["gpt-oss"].respondent_id.astype(int)) | set(llm["Gemini"].respondent_id.astype(int))
cols = sorted(set(DEMO_COLS) | set(MATCHED_DEMO_COLS))
feat = df[["respondent_id"] + cols].copy(); feat["N_REGION_ISO"] = feat["N_REGION_ISO"].astype(str)

preds = []  # respondent, item, method, pred_idx, true_idx
for q in items:
    y = _encode_item(df, q, codebook); valid = y.notna()
    for fold in folds:
        tr = df.respondent_id.isin(fold["train"]) & valid
        te = df.respondent_id.isin(set(fold["test"]) & sample_ids) & valid
        if te.sum() == 0:
            continue
        Xtr = feat[tr].copy(); Xtr["answer"] = y[tr].astype(int).values
        Xte = feat[te].copy()
        ytr = Xtr["answer"].values
        mode = int(np.bincount(ytr).argmax()); marg = np.bincount(ytr) / len(ytr)
        out = pd.DataFrame({"respondent_id": df.loc[te, "respondent_id"].values, "question_id": q, "true_idx": y[te].astype(int).values})
        out["majority"] = mode
        out["marginal_sampled"] = rng.choice(len(marg), size=len(out), p=marg)
        # cell lookup on the 6 original attributes: mode of the cell (fallback: global mode) and sampled
        key = lambda d: d[DEMO_COLS].astype(object).fillna("NA").astype(str).agg("|".join, axis=1)
        ck = key(Xtr); cell_mode = Xtr.groupby(ck)["answer"].agg(lambda s: int(np.bincount(s).argmax())).to_dict()
        cell_dist = {k: np.bincount(g.values, minlength=len(marg)) / len(g) for k, g in Xtr.groupby(ck)["answer"]}
        tk = key(Xte)
        out["cell_mode"] = [cell_mode.get(k, mode) for k in tk]
        out["cell_sampled"] = [rng.choice(len(marg), p=cell_dist.get(k, marg)) for k in tk]
        for name, model_type in [("logistic_matched", "logistic"), ("gbm_matched", "gbm")]:
            try:
                m = SupervisedClassifierBaseline(model_type); m.fit(Xtr, "answer", MATCHED_DEMO_COLS)
                out[name] = m.predict(Xte)
            except Exception as e:
                out[name] = mode
        preds.append(out)
B = pd.concat(preds, ignore_index=True)
methods = ["majority", "marginal_sampled", "cell_mode", "cell_sampled", "logistic_matched", "gbm_matched"]

res = {}
for name, d in llm.items():
    d = d[d.pred_code_idx.notna()][["respondent_id", "question_id", "true_code_idx", "pred_code_idx"]].copy()
    d["respondent_id"] = d.respondent_id.astype(int)
    m = d.merge(B, on=["respondent_id", "question_id"], how="inner")
    assert (m.true_code_idx == m.true_idx).all()
    ids = np.sort(m.respondent_id.unique()); ri = {r: i for i, r in enumerate(ids)}; R = len(ids)
    m["ri"] = m.respondent_id.map(ri)
    W = rng.multinomial(R, np.ones(R) / R, size=4000).astype(float)
    def agg(col):
        c = (m[col] == m.true_idx).astype(float) if col != "llm" else (m.pred_code_idx == m.true_idx).astype(float)
        return np.bincount(m.ri, weights=c, minlength=R), np.bincount(m.ri, minlength=R).astype(float)
    r = {"n_rows": int(len(m)), "n_resp": int(R)}
    sl, nl = agg("llm"); acc_l = (W @ sl) / (W @ nl)
    r["llm"] = {"acc": float(sl.sum() / nl.sum()), "ci": np.percentile(acc_l, [2.5, 97.5]).tolist()}
    for meth in methods:
        s, n = agg(meth); acc = (W @ s) / (W @ n); diff = acc_l - acc
        r[meth] = {"acc": float(s.sum() / n.sum()), "ci": np.percentile(acc, [2.5, 97.5]).tolist(),
                   "llm_minus": float(sl.sum() / nl.sum() - s.sum() / n.sum()), "llm_minus_ci": np.percentile(diff, [2.5, 97.5]).tolist()}
    res[name] = r
# Gemini vs gpt-oss on the common 30 respondents
g, o = llm["Gemini"], llm["gpt-oss"]
common = set(g.respondent_id) & set(o.respondent_id)
cm = {}
for nm, d in [("Gemini", g), ("gpt-oss", o)]:
    d2 = d[d.respondent_id.isin(common) & d.pred_code_idx.notna()]
    cm[nm] = {"acc": float((d2.true_code_idx == d2.pred_code_idx).mean()), "n_rows": int(len(d2)), "n_resp": int(d2.respondent_id.nunique())}
res["common_respondents"] = cm
json.dump(res, open("/private/tmp/claude-501/-Users-krishank-Documents-College-Projects----/c07d5fa6-6bd6-4110-b7a8-4e0125580098/scratchpad/work/analysis/wvs_matched.json", "w"), indent=1)
for nm in ("Gemini", "gpt-oss"):
    r = res[nm]; print(nm, r["n_rows"], r["n_resp"], "LLM", round(100 * r["llm"]["acc"], 1), [round(100 * x, 1) for x in r["llm"]["ci"]])
    for meth in methods:
        print("   ", meth, round(100 * r[meth]["acc"], 1), "LLM-minus", round(100 * r[meth]["llm_minus"], 1), [round(100 * x, 1) for x in r[meth]["llm_minus_ci"]])
print(cm)
