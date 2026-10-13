"""Classical baselines with PROBABILITIES for the Kaggle experiments (runs on CPU).
Regimes: full (all 23,999 fold-0 train respondents), pool2k (the 2,000-respondent pool used for fine-tuning, all its answers),
matchedA (exactly the 15,462 (respondent,item) pairs the 24-item adapter trains on; items 0:24 only).
Test rows: items 0:40 on the first 500 respondents of the seed-7 fold-0 test permutation (same as the Kaggle eval),
plus items 0:8 on all 6,000 test respondents for the 'full' regime.
usage: python baselines_probs.py <repo> <out.parquet> [seed]"""
import sys, os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from kaggle_common import _ctx, train_pairs
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
repo, out = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
df, items, folds, cb, meta = _ctx(repo)
FEATURES = ["QGEN", "QAGErec", "QEDU", "QRELSING", "QCASTE", "QMARRIEDrec", "Urban", "REGION", "QINCINDrec"]
X_all = pd.get_dummies(df[FEATURES].fillna(-1).astype(int).astype(str), dtype=float); X_all.index = df.respondent_id.values
test_all = np.array(sorted(folds[0]["test"]))
test500 = np.random.default_rng(7).permutation(test_all)[:500]
train_ids = np.array(sorted(folds[0]["train"]))
pairsB, _ = train_pairs(repo, 0, 8, 2000, 0, seed)
pairsA, _ = train_pairs(repo, 0, 24, 2000, 0, seed)
pool_resp = sorted({r for r, _ in pairsB})
c2i = {q: meta[q]["code_to_index"] for q in items[:40]}
K = {q: meta[q]["n_options"] for q in items[:40]}


def yvec(q, ids):
    v = df.loc[ids, q]
    return np.array([c2i[q].get(int(x), -1) if pd.notna(x) else -1 for x in v])


def fit_predict(q, tr_ids, te_ids):
    y = yvec(q, tr_ids); m = y >= 0; k = K[q]
    if m.sum() < 5:
        return None
    yt = y[m]; Xt = X_all.loc[np.asarray(tr_ids)[m]]; Xe = X_all.loc[te_ids]
    pi = np.bincount(yt, minlength=k) + 1.0; pi = pi / pi.sum()
    P = {"majority": np.tile(pi, (len(te_ids), 1))}
    for name, mdl in [("logistic", LogisticRegression(max_iter=500, C=1.0)),
                      ("gboost", HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, random_state=0))]:
        if len(set(yt)) < 2:
            P[name] = P["majority"]; continue
        mdl.fit(Xt, yt)
        pr = np.zeros((len(te_ids), k)); pr[:, mdl.classes_.astype(int)] = mdl.predict_proba(Xe)
        P[name] = 0.98 * pr + 0.02 * pi
    return P


rows = []
def emit(regime, q, te_ids, P):
    ytrue = yvec(q, te_ids)
    for meth, pr in P.items():
        for rid, t, p in zip(te_ids, ytrue, pr):
            if t >= 0:
                rows.append((int(rid), q, int(t), meth, regime, int(np.argmax(p)), [float(x) for x in p]))


pa = pd.DataFrame(pairsA, columns=["rid", "q"])
for qi, q in enumerate(items[:40]):
    emit_full = fit_predict(q, train_ids, test500 if qi >= 8 else test_all)
    emit("full", q, test500 if qi >= 8 else test_all, emit_full)
    P = fit_predict(q, np.array(pool_resp), test500); emit("pool2k", q, test500, P)
    if qi < 24:
        tr = pa[pa.q == q].rid.values
        P = fit_predict(q, tr, test500)
        if P is not None:
            emit("matchedA", q, test500, P)
    print(qi, q, flush=True)
d = pd.DataFrame(rows, columns=["respondent_id", "question_id", "true_idx", "method", "regime", "pred_idx", "probs"])
d.to_parquet(out)
print(d.groupby(["regime", "method"]).apply(lambda g: pd.Series({"n": len(g), "acc": (g.pred_idx == g.true_idx).mean()})))
