"""Reviewer-proofing: are the supervised baselines under-tuned? Select hyperparameters on a validation split of the TRAIN respondents."""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/tmp/ssc_pew"); os.chdir("/tmp/ssc_pew")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from src.config import DATA_PROCESSED
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook
FEATURES = ["QGEN", "QAGErec", "QEDU", "QRELSING", "QCASTE", "QMARRIEDrec", "Urban", "REGION", "QINCINDrec"]
cb = load_pew_codebook(); items = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][:8]
f0 = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"][0]
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet"); df["respondent_id"] = df.respondent_id.astype(int); df = df.set_index("respondent_id", drop=False)
X = pd.get_dummies(df[FEATURES].fillna(-1).astype(int).astype(str), dtype=float)
rng = np.random.default_rng(3)
tr = np.array(sorted(f0["train"])); te = np.array(sorted(f0["test"]))
perm = rng.permutation(tr); va, fit = perm[: len(tr) // 5], perm[len(tr) // 5:]
def labels(q, ids):
    it = verbalize_item(q, cb); c2i = it["code_to_index"]
    y = df.loc[ids, q].map(lambda v: c2i.get(int(v)) if pd.notna(v) and int(v) in c2i else np.nan)
    return y.to_numpy(), it["n_options"]
def fitpred(model, Xa, ya, Xb, k):
    model.fit(Xa, ya); p = np.zeros((len(Xb), k)); p[:, model.classes_.astype(int)] = model.predict_proba(Xb); return p
res = {"logistic": {"acc": 0, "n": 0, "nll": 0}, "gboost": {"acc": 0, "n": 0, "nll": 0}, "chosen": {}}
LOG = [0.01, 0.1, 1.0, 10.0]
GB = [dict(max_iter=60, learning_rate=0.05, max_depth=3), dict(max_iter=100, learning_rate=0.1, max_depth=4), dict(max_iter=150, learning_rate=0.05, max_depth=None, l2_regularization=1.0), dict(max_iter=100, learning_rate=0.1, max_depth=None, l2_regularization=5.0)]
for q in items:
    yf, k = labels(q, fit); mf = ~np.isnan(yf); yv, _ = labels(q, va); mv = ~np.isnan(yv); yt, _ = labels(q, te); mt = ~np.isnan(yt)
    Xf, Xv, Xt = X.loc[fit][mf], X.loc[va][mv], X.loc[te][mt]
    best = {}
    for name, grid, mk in [("logistic", LOG, lambda c: LogisticRegression(max_iter=400, C=c)), ("gboost", GB, lambda g: HistGradientBoostingClassifier(random_state=0, **g))]:
        scores = []
        for g in grid:
            p = fitpred(mk(g), Xf, yf[mf].astype(int), Xv, k); scores.append(-np.log(np.clip(p[np.arange(len(yv[mv])), yv[mv].astype(int)], 1e-6, 1)).mean())
        g = grid[int(np.argmin(scores))]; best[name] = str(g)
        Xfull = pd.concat([Xf, Xv]); yfull = np.concatenate([yf[mf], yv[mv]]).astype(int)
        p = fitpred(mk(g), Xfull, yfull, Xt, k)
        res[name]["acc"] += (p.argmax(1) == yt[mt].astype(int)).sum(); res[name]["n"] += int(mt.sum())
        res[name]["nll"] += -np.log(np.clip(p[np.arange(mt.sum()), yt[mt].astype(int)], 1e-6, 1)).sum()
    res["chosen"][q] = best; print(q, best, flush=True)
for n in ["logistic", "gboost"]:
    res[n] = {"acc": res[n]["acc"] / res[n]["n"], "nll": res[n]["nll"] / res[n]["n"], "n": res[n]["n"]}
json.dump(res, open(f"{HERE}/tuned_baselines.json", "w"), indent=1); print(res["logistic"], res["gboost"])
