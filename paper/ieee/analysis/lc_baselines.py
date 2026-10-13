"""Learning curves of classical baselines: train on the first N (nested) respondents, score on the fixed 200-respondent test rows."""
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
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet"); df["respondent_id"] = df.respondent_id.astype(int)
order = json.load(open(f"{HERE}/lc_train_order.json"))
test_rows = pd.read_json(f"{HERE}/mlx_base_chat.jsonl", lines=True).drop_duplicates(["respondent_id", "question_id"])[["respondent_id", "question_id", "true_idx"]]
X_all = pd.get_dummies(df[FEATURES].fillna(-1).astype(int).astype(str), dtype=float); X_all.index = df.respondent_id.values
test_ids = np.sort(test_rows.respondent_id.unique())
rng = np.random.default_rng(88)
all_train = np.array(sorted(f0["train"]))
def idx_of(item, ids):
    it = verbalize_item(item, cb); c2i = it["code_to_index"]; k = it["n_options"]
    sub = df.set_index("respondent_id").loc[ids, item]
    y = sub.map(lambda v: c2i.get(int(v)) if pd.notna(v) and int(v) in c2i else np.nan)
    return y, k
def run(train_ids):
    out = {"majority": [], "logistic": [], "gboost": []}
    for q in items:
        y, k = idx_of(q, train_ids); m = y.notna().to_numpy()
        if m.sum() < 5: continue
        yt = y[m].astype(int).to_numpy(); Xt = X_all.loc[np.array(train_ids)[m]]
        tq = test_rows[test_rows.question_id == q]; Xe = X_all.loc[tq.respondent_id.values]
        pi = (np.bincount(yt, minlength=k) + 1.0); pi = pi / pi.sum()
        P = {"majority": np.tile(pi, (len(tq), 1))}
        for name, mdl in [("logistic", LogisticRegression(max_iter=300, C=1.0)), ("gboost", HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, random_state=0))]:
            if len(set(yt)) < 2: P[name] = P["majority"]; continue
            mdl.fit(Xt, yt); pr = np.zeros((len(tq), k)); pr[:, mdl.classes_.astype(int)] = mdl.predict_proba(Xe); pr = 0.98 * pr + 0.02 * pi; P[name] = pr
        for name, pr in P.items():
            out[name].append(pd.DataFrame({"respondent_id": tq.respondent_id.values, "question_id": q, "true": tq.true_idx.values,
                                           "pred": pr.argmax(1), "nll": -np.log(np.clip(pr[np.arange(len(tq)), tq.true_idx.values.astype(int)], 1e-6, 1))}))
    return {k: pd.concat(v) for k, v in out.items()}
def boot_ci(d, n=1000):
    g = d.assign(c=(d.pred == d.true).astype(float)).groupby("respondent_id").agg(s=("c", "sum"), n=("c", "size"), nll=("nll", "sum"))
    w = rng.multinomial(len(g), np.ones(len(g)) / len(g), size=n).astype(float)
    acc = (w @ g.s.values) / (w @ g.n.values); nl = (w @ g.nll.values) / (w @ g.n.values)
    return np.percentile(acc, [2.5, 97.5]).tolist(), np.percentile(nl, [2.5, 97.5]).tolist()
res = {}
for N in [25, 100, 400, 1600, len(all_train)]:
    reps = []
    n_reps = 1 if N == len(all_train) else 5
    for r in range(n_reps):
        ids = order[:N] if r == 0 and N <= len(order) else (all_train if N == len(all_train) else rng.choice(all_train, N, replace=False))
        reps.append(run(list(ids)))
    entry = {}
    for m in ["majority", "logistic", "gboost"]:
        d0 = reps[0][m]; ci = boot_ci(d0)
        accs = [float((r[m].pred == r[m].true).mean()) for r in reps]; nlls = [float(r[m].nll.mean()) for r in reps]
        entry[m] = {"acc_first_subset": accs[0], "acc_ci": ci[0], "nll_first_subset": nlls[0], "nll_ci": ci[1],
                    "acc_mean_over_subsets": float(np.mean(accs)), "acc_sd_over_subsets": float(np.std(accs)), "nll_mean_over_subsets": float(np.mean(nlls)), "n_subsets": len(reps)}
    res[str(N)] = entry
    print(N, {m: (round(100 * e["acc_first_subset"], 1), round(e["nll_first_subset"], 3)) for m, e in entry.items()}, flush=True)
json.dump(res, open(f"{HERE}/lc_baselines.json", "w"), indent=1)
