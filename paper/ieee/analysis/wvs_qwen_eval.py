"""Evaluate local zero-shot Qwen2.5-7B-Instruct on WVS-7 India (same 15 items, P2 prompt) against out-of-fold baselines scored on the
same (respondent, item) rows, plus subgroup gaps with permutation nulls on this larger respondent sample."""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/krishank/Documents/College Projects/🧢 /silicon-sampling-capstone"
sys.path.insert(0, REPO)
os.chdir(REPO)
from src.config import DATA_PROCESSED
from src.eval.baselines import SupervisedClassifierBaseline
from src.eval.run_baselines import DEMO_COLS, MATCHED_DEMO_COLS, _encode_item
from src.eval.subgroups import assign_subgroups, SUBGROUP_AXES
from src.prompts.verbalize import load_codebook

rng = np.random.default_rng(909)
q = pd.read_json(f"{HERE}/mlx_wvs_base_chat.jsonl", lines=True).drop_duplicates(["respondent_id", "question_id"])
df = pd.read_parquet(f"{DATA_PROCESSED}/ind_wvs7.parquet")
if "respondent_id" not in df.columns:
    df["respondent_id"] = range(len(df))
df["respondent_id"] = df["respondent_id"].astype(int)
folds = json.load(open(f"{DATA_PROCESSED}/folds.json"))["folds"]
cb = load_codebook()
items = sorted(q.question_id.unique())
sample = set(q.respondent_id.astype(int))
cols = sorted(set(DEMO_COLS) | set(MATCHED_DEMO_COLS))
feat = df[["respondent_id"] + cols].copy()
feat["N_REGION_ISO"] = feat["N_REGION_ISO"].astype(str)
key = lambda d: d[DEMO_COLS].astype(object).fillna("NA").astype(str).agg("|".join, axis=1)
fkey = key(feat)

rows = []
for it in items:
    y = _encode_item(df, it, cb)
    valid = y.notna()
    for f in folds:
        tr = df.respondent_id.isin(f["train"]) & valid
        te = df.respondent_id.isin(set(f["test"]) & sample) & valid
        if te.sum() == 0:
            continue
        Xtr = feat[tr].copy()
        Xtr["answer"] = y[tr].astype(int).values
        Xte = feat[te].copy()
        ytr = Xtr["answer"].values
        mode = int(np.bincount(ytr).argmax())
        out = pd.DataFrame({"respondent_id": df.loc[te, "respondent_id"].values, "question_id": it, "true_idx": y[te].astype(int).values})
        out["majority"] = mode
        tab = pd.DataFrame({"k": fkey[tr].values, "a": ytr}).groupby("k")["a"].agg(lambda s: int(np.bincount(s).argmax())).to_dict()
        out["cell_mode"] = [tab.get(k, mode) for k in fkey[te].values]
        for name, mt in [("logistic_matched", "logistic"), ("gbm_matched", "gbm")]:
            try:
                m = SupervisedClassifierBaseline(mt)
                m.fit(Xtr, "answer", MATCHED_DEMO_COLS)
                out[name] = m.predict(Xte)
            except Exception:
                out[name] = mode
        rows.append(out)
B = pd.concat(rows, ignore_index=True)
B.to_parquet(f"{HERE}/wvs_baseline_rows.parquet")
m = q.merge(B, on=["respondent_id", "question_id"], how="inner")
assert (m.true_idx_x == m.true_idx_y).all()
m = m.rename(columns={"true_idx_x": "true_idx"}).drop(columns=["true_idx_y"])
ids = np.sort(m.respondent_id.unique())
ri = {r: i for i, r in enumerate(ids)}
R = len(ids)
m["ri"] = m.respondent_id.map(ri)
W = rng.multinomial(R, np.ones(R) / R, size=4000).astype(float)


def agg(c):
    return np.bincount(m.ri, weights=c, minlength=R), np.bincount(m.ri, minlength=R).astype(float)


res = {"n_rows": int(len(m)), "n_resp": int(R)}
sl, nl = agg((m.pred_idx == m.true_idx).astype(float))
accl = (W @ sl) / (W @ nl)
res["llm"] = {"acc": float(sl.sum() / nl.sum()), "ci": np.percentile(accl, [2.5, 97.5]).tolist()}
res["nll"] = float(np.mean([-np.log(max(p[int(t)], 1e-6)) for p, t in zip(m.probs, m.true_idx)]))
for meth in ["majority", "cell_mode", "logistic_matched", "gbm_matched"]:
    s, n = agg((m[meth] == m.true_idx).astype(float))
    acc = (W @ s) / (W @ n)
    res[meth] = {"acc": float(s.sum() / n.sum()), "ci": np.percentile(acc, [2.5, 97.5]).tolist(),
                 "llm_minus": float(sl.sum() / nl.sum() - s.sum() / n.sum()), "llm_minus_ci": np.percentile(accl - acc, [2.5, 97.5]).tolist()}
# per item
res["per_item"] = {it: {"llm": float((g.pred_idx == g.true_idx).mean()), "majority": float((g.majority == g.true_idx).mean()),
                        "gbm_matched": float((g.gbm_matched == g.true_idx).mean()), "n": int(len(g))} for it, g in m.groupby("question_id")}

# subgroup gaps with permutation null (respondent-level aggregates)
sub = assign_subgroups(df.set_index("respondent_id", drop=False))
gaps = {}
mm = m.groupby("respondent_id").agg(c=("true_idx", lambda s: 0), n=("true_idx", "size"))
corr = m.assign(c=(m.pred_idx == m.true_idx).astype(float)).groupby("respondent_id").c.sum()
mm["c"] = corr
MIN_RESP = 15
for ax in SUBGROUP_AXES:
    lab = sub[ax].reindex(mm.index)
    keep = lab.notna().to_numpy()
    code, cats = pd.factorize(lab[keep].astype(str))
    gc, gn = mm.c.to_numpy()[keep], mm.n.to_numpy()[keep]
    K_ = len(cats)

    def gap(code_):
        s = np.bincount(code_, weights=gc, minlength=K_)
        n = np.bincount(code_, weights=gn, minlength=K_)
        cnt = np.bincount(code_, minlength=K_)
        ok = cnt >= MIN_RESP
        if ok.sum() < 2:
            return np.nan
        a = s[ok] / n[ok]
        return a.max() - a.min()

    obs = gap(code)
    null = np.array([gap(rng.permutation(code)) for _ in range(5000)])
    null = null[~np.isnan(null)]
    cnts = np.bincount(code, minlength=K_)
    gaps[ax] = {"gap": float(obs), "p": float((1 + (null >= obs).sum()) / (1 + len(null))) if len(null) else None,
                "null95": float(np.percentile(null, 95)) if len(null) else None,
                "n_cats_powered": int((cnts >= MIN_RESP).sum())}
res["gaps"] = gaps
json.dump(res, open(f"{HERE}/wvs_qwen_eval.json", "w"), indent=1)
print(res["n_rows"], res["n_resp"], "LLM", round(100 * res["llm"]["acc"], 1), [round(100 * x, 1) for x in res["llm"]["ci"]], "NLL", round(res["nll"], 3))
for k in ["majority", "cell_mode", "logistic_matched", "gbm_matched"]:
    r = res[k]
    print(f"  {k:18s} {100*r['acc']:.1f}  LLM-minus {100*r['llm_minus']:+.1f} [{100*r['llm_minus_ci'][0]:+.1f},{100*r['llm_minus_ci'][1]:+.1f}]")
for ax, g in gaps.items():
    print("  ", ax, {k: (round(100 * v, 1) if isinstance(v, float) and k != "p" else v) for k, v in g.items()})
