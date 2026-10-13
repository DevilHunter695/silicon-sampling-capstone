"""Evaluate local zero-shot Qwen2.5-7B-Instruct runs on Pew fold 0 against the supervised baselines and the ORIGINAL
(lab-run) fine-tuned model's hard predictions, on identical (respondent, item) rows.

Inputs : mlx_base_chat.jsonl, mlx_base_raw.jsonl (optional), pew_aligned_rows.parquet
Output : zs_eval.json
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CLONE = "/tmp/ssc_pew"
sys.path.insert(0, CLONE)
os.chdir(CLONE)
from scripts.pewB_baselines import run_fold
from src.config import DATA_PROCESSED
from src.prompts.verbalize_pew import load_pew_codebook

rng = np.random.default_rng(2027)
key = ["respondent_id", "question_id"]
cb = load_pew_codebook()
items = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][:8]
folds = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"]
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet")
df["respondent_id"] = df["respondent_id"].astype(int)
base = run_fold(df, folds[0], items, cb)
orig = pd.read_parquet(f"{HERE}/pew_aligned_rows.parquet")
ft = orig[orig.method == "finetuned"][key + ["pred_idx", "true_idx"]].rename(columns={"pred_idx": "ft_pred"})

K = {q: len(cb[q]["valid_codes"]) for q in items}


def load_run(name):
    p = f"{HERE}/mlx_{name}.jsonl"
    if not os.path.exists(p):
        return None
    d = pd.read_json(p, lines=True)
    return d if len(d) else None


def evaluate(run_names, tag):
    runs = {n: load_run(n) for n in run_names}
    runs = {n: d for n, d in runs.items() if d is not None}
    if not runs:
        return None
    S = None
    for d in runs.values():
        k = d[key].drop_duplicates()
        S = k if S is None else S.merge(k, on=key)
    S = S.merge(ft, on=key).sort_values(key).reset_index(drop=True)
    idx = S.set_index(key).index
    true = S.true_idx.to_numpy()
    preds, probs = {"finetuned": S.ft_pred.to_numpy()}, {}
    for n, d in runs.items():
        dd = d.drop_duplicates(key).set_index(key).loc[idx]
        assert (dd.true_idx.to_numpy() == true).all()
        preds[f"zs_{n}"] = dd.pred_idx.to_numpy()
        probs[f"zs_{n}"] = list(dd.probs)
    for m, dm in base.items():
        dm = dm.copy()
        dm["respondent_id"] = dm.respondent_id.astype(int)
        dd = dm.drop_duplicates(key).set_index(key).loc[idx]
        preds[m] = np.array([int(np.argmax(p)) for p in dd.proba])
        probs[m] = list(dd.proba)
    ids = np.sort(S.respondent_id.unique())
    ri = {r: i for i, r in enumerate(ids)}
    NR = len(ids)
    S["ri"] = S.respondent_id.map(ri)
    W = rng.multinomial(NR, np.ones(NR) / NR, size=3000).astype(float)

    def agg(correct):
        return (np.bincount(S.ri, weights=correct, minlength=NR), np.bincount(S.ri, minlength=NR).astype(float))

    store, out = {}, {"tag": tag, "n_rows": int(len(S)), "n_resp": int(NR), "methods": {}, "paired": {}}
    for name, p in preds.items():
        s, n = agg((p == true).astype(float))
        store[name] = (s, n)
        r = {"acc": float(s.sum() / n.sum()), "ci": np.percentile((W @ s) / (W @ n), [2.5, 97.5]).tolist()}
        tops, disp = [], []
        for q in items:
            mk = (S.question_id == q).to_numpy()
            if mk.sum() == 0:
                continue
            tops.append(float(np.bincount(p[mk].astype(int)).max() / mk.sum()))
            t_ = true[mk] / (K[q] - 1)
            p_ = p[mk] / (K[q] - 1)
            if q != "Q28rec" and t_.std() > 0:
                disp.append(float(p_.std() / t_.std()))
        r["mean_top_share"] = float(np.mean(tops))
        r["dispersion_retention"] = float(np.mean(disp))
        if name in probs:
            pr = probs[name]
            r["nll"] = float(np.mean([-np.log(max(pr[i][int(true[i])], 1e-6)) for i in range(len(true))]))
            tv = []
            for q in items:
                ix = np.where((S.question_id == q).to_numpy())[0]
                if len(ix) == 0:
                    continue
                mp = np.mean([pr[i] for i in ix], axis=0)
                tm = np.bincount(true[ix].astype(int), minlength=len(mp)) / len(ix)
                tv.append(0.5 * np.abs(mp - tm).sum())
            r["soft_tv"] = float(np.mean(tv))
        out["methods"][name] = r

    def paired(a, b):
        sa, na = store[a]
        sb, nb = store[b]
        d = (W @ sa) / (W @ na) - (W @ sb) / (W @ nb)
        return {"diff": float(sa.sum() / na.sum() - sb.sum() / nb.sum()), "ci": np.percentile(d, [2.5, 97.5]).tolist()}

    zs = [k for k in preds if k.startswith("zs_")]
    for z in zs:
        for b in ["finetuned", "majority", "logistic", "gboost"]:
            out["paired"][f"{z}-{b}"] = paired(z, b)
    for b in ["majority", "logistic", "gboost"]:
        out["paired"][f"finetuned-{b}"] = paired("finetuned", b)
    # per-item accuracies
    out["per_item"] = {}
    for q in items:
        mk = (S.question_id == q).to_numpy()
        if mk.sum():
            out["per_item"][q] = {k: float((v[mk] == true[mk]).mean()) for k, v in preds.items()}
    # agreement of zero-shot with fine-tuned and with GBM
    for z in zs:
        out[f"agree_{z}_ft"] = float((preds[z] == preds["finetuned"]).mean())
        out[f"agree_{z}_gboost"] = float((preds[z] == preds["gboost"]).mean())
    return out


res = {}
for tag, names in [("chat", ["base_chat"]), ("raw", ["base_raw"]), ("both", ["base_chat", "base_raw"])]:
    r = evaluate(names, tag)
    if r:
        res[tag] = r
json.dump(res, open(f"{HERE}/zs_eval.json", "w"), indent=1)
for tag, r in res.items():
    print(f"== {tag}: {r['n_rows']} rows, {r['n_resp']} respondents")
    for n, m in r["methods"].items():
        print(f"  {n:18s} acc={100*m['acc']:.1f} [{100*m['ci'][0]:.1f},{100*m['ci'][1]:.1f}] top={m['mean_top_share']:.2f} disp={m['dispersion_retention']:.2f}"
              + (f" nll={m['nll']:.3f} softTV={m['soft_tv']:.3f}" if "nll" in m else ""))
    for k, v in r["paired"].items():
        print(f"  {k:28s} {100*v['diff']:+.1f} [{100*v['ci'][0]:+.1f},{100*v['ci'][1]:+.1f}]")
