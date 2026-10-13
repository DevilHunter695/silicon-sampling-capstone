"""Within-family zero-shot scale comparison (Qwen2.5 7B / 14B / 32B Instruct) on the same Pew rows and respondents.
usage: python scale_analysis.py <fold0_out_dir(7B)> <out_scale_dir> <baselines_probs.parquet> <out.json>"""
import sys, glob, json, re
import numpy as np, pandas as pd
f0, osd, bpp, outp = sys.argv[1:5]
rng = np.random.default_rng(55)
def rd(files):
    return pd.concat([pd.read_json(f, lines=True) for f in files]).drop_duplicates(["respondent_id", "question_id"]) if files else None
M = {}
for cond in ("base_raw", "base_chat"):
    M[f"Qwen2.5-7B|{cond}"] = rd(sorted(glob.glob(f"{f0}/{cond}_s*.jsonl")))
for f in sorted(glob.glob(f"{osd}/*__*_s*.jsonl")):
    m = re.match(r".*/(.+?)__(base_raw|base_chat)_s\d+of\d+\.jsonl", f); M.setdefault(f"{m.group(1).replace('-Instruct','')}|{m.group(2)}", []).append(f)
for k, v in list(M.items()):
    if isinstance(v, list): M[k] = rd(v)
bp = pd.read_parquet(bpp); bp = bp[bp.regime == "full"]
key = lambda x: set(zip(x.respondent_id, x.question_id))
mj = bp[bp.method == "majority"]
common = key(mj)
for k, v in M.items(): common &= key(v)
rows = {}
for k, v in M.items():
    d = v[[(r, q) in common for r, q in zip(v.respondent_id, v.question_id)]].copy()
    d["correct"] = (d.pred_idx == d.true_idx).astype(float); d["nll"] = [-np.log(max(p[t], 1e-6)) for p, t in zip(d.probs, d.true_idx)]; rows[k] = d
mj = mj[[(r, q) in common for r, q in zip(mj.respondent_id, mj.question_id)]].copy(); mj["correct"] = (mj.pred_idx == mj.true_idx).astype(float)
ids = np.sort(mj.respondent_id.unique()); ix = {r: i for i, r in enumerate(ids)}; R = len(ids)
W = rng.multinomial(R, np.ones(R) / R, size=2000).astype(float)
def agg(d, col="correct"): return np.bincount(d.respondent_id.map(ix), weights=d[col], minlength=R), np.bincount(d.respondent_id.map(ix), minlength=R).astype(float)
sm, nm = agg(mj); out = {"n_resp": int(R), "n_rows": int(len(mj)), "majority": float(sm.sum() / nm.sum()), "models": {}}
for k, d in rows.items():
    s, n = agg(d); a = (W @ s) / (W @ n); diff = a - (W @ sm) / (W @ nm)
    out["models"][k] = {"acc": float(s.sum() / n.sum()), "ci": np.percentile(a, [2.5, 97.5]).tolist(), "nll": float(d.nll.mean()),
                        "minus_majority": float(s.sum() / n.sum() - sm.sum() / nm.sum()), "minus_majority_ci": np.percentile(diff, [2.5, 97.5]).tolist()}
json.dump(out, open(outp, "w"), indent=1)
print(f"{R} respondents, {len(mj)} rows; majority {100*out['majority']:.1f}")
for k, v in sorted(out["models"].items()): print(f"  {k:26s} acc {100*v['acc']:5.1f} [{100*v['ci'][0]:.1f},{100*v['ci'][1]:.1f}]  vs majority {100*v['minus_majority']:+.1f} [{100*v['minus_majority_ci'][0]:+.1f},{100*v['minus_majority_ci'][1]:+.1f}]  NLL {v['nll']:.2f}")
