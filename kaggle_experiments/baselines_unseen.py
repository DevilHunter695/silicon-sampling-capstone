"""Classical baselines with probabilities for the unseen-item test: items 8:40, first N test respondents of the seed-7 permutation.
Regimes: full (all 23,999 fold-0 train respondents with a label on the item) and pool2k (the 2,000-respondent fine-tuning pool).
usage: python baselines_unseen.py <repo> <out.parquet> [n_test=1000] [lo=8] [hi=40]"""
import sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from kaggle_common import _ctx, train_pairs
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
repo, out = sys.argv[1], sys.argv[2]
n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
lo = int(sys.argv[4]) if len(sys.argv) > 4 else 8
hi = int(sys.argv[5]) if len(sys.argv) > 5 else 40
df, items, folds, cb, meta = _ctx(repo)
FEATURES = ["QGEN", "QAGErec", "QEDU", "QRELSING", "QCASTE", "QMARRIEDrec", "Urban", "REGION", "QINCINDrec"]
X_all = pd.get_dummies(df[FEATURES].fillna(-1).astype(int).astype(str), dtype=float); X_all.index = df.respondent_id.values
test = np.random.default_rng(7).permutation(np.array(sorted(folds[0]["test"])))[:n_test]
train_ids = np.array(sorted(folds[0]["train"]))
pairsB, _ = train_pairs(repo, 0, 8, 2000, 0, 0)
pool = np.array(sorted({r for r, _ in pairsB}))
c2i = {q: meta[q]["code_to_index"] for q in items[:44]}; K = {q: meta[q]["n_options"] for q in items[:44]}
def yvec(q, ids):
    v = df.loc[ids, q]; return np.array([c2i[q].get(int(x), -1) if pd.notna(x) else -1 for x in v])
def fit_predict(q, tr, te):
    y = yvec(q, tr); m = y >= 0; k = K[q]
    if m.sum() < 5: return None
    yt = y[m]; Xt = X_all.loc[np.asarray(tr)[m]]; Xe = X_all.loc[te]
    pi = np.bincount(yt, minlength=k) + 1.0; pi /= pi.sum()
    P = {"majority": np.tile(pi, (len(te), 1))}
    for nm, mdl in [("logistic", LogisticRegression(max_iter=500)), ("gboost", HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, random_state=0))]:
        if len(set(yt)) < 2: P[nm] = P["majority"]; continue
        mdl.fit(Xt, yt); pr = np.zeros((len(te), k)); pr[:, mdl.classes_.astype(int)] = mdl.predict_proba(Xe); P[nm] = 0.98 * pr + 0.02 * pi
    return P
rows = []
for qi in range(lo, hi):
    q = items[qi]; yt_ = yvec(q, test)
    for regime, tr in (("full", train_ids), ("pool2k", pool)):
        P = fit_predict(q, tr, test)
        if P is None: continue
        for meth, pr in P.items():
            for rid, t, p in zip(test, yt_, pr):
                if t >= 0: rows.append((int(rid), q, int(t), meth, regime, int(np.argmax(p)), [float(x) for x in p]))
    print(qi, q, flush=True)
d = pd.DataFrame(rows, columns=["respondent_id", "question_id", "true_idx", "method", "regime", "pred_idx", "probs"]); d.to_parquet(out)
print(d.groupby(["regime", "method"]).apply(lambda g: (g.pred_idx == g.true_idx).mean()))
