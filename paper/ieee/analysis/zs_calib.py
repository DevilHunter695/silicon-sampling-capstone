"""Prior-corrected zero-shot control on Pew: p'(k) ~ p_zs(k) * pi_train(k) / mean_zs(k), per item (label-free for the test rows)."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/tmp/ssc_pew"); os.chdir("/tmp/ssc_pew")
from src.config import DATA_PROCESSED
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook
rng = np.random.default_rng(5)
cb = load_pew_codebook()
items = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][:8]
f0 = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"][0]
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet"); df["respondent_id"] = df.respondent_id.astype(int)
tr = df[df.respondent_id.isin(f0["train"])]
z = pd.read_json(f"{HERE}/mlx_base_chat.jsonl", lines=True).drop_duplicates(["respondent_id", "question_id"])
orig = pd.read_parquet(f"{HERE}/pew_aligned_rows.parquet")
ft = orig[orig.method == "finetuned"][["respondent_id", "question_id", "pred_idx"]].rename(columns={"pred_idx": "ft"})
z = z.merge(ft, on=["respondent_id", "question_id"])
pred = np.zeros(len(z), int)
for q in items:
    it = verbalize_item(q, cb); c2i = it["code_to_index"]
    y = tr[q].map(lambda v: c2i.get(int(v)) if pd.notna(v) and int(v) in c2i else np.nan).dropna().astype(int)
    k = len(it["ordinal_values"]); pi = np.bincount(y, minlength=k) / len(y)
    ix = np.where(z.question_id == q)[0]
    P = np.stack(z.probs.iloc[ix].to_numpy())
    pbar = P.mean(0)
    Pc = P * (pi / np.maximum(pbar, 1e-6)); Pc /= Pc.sum(1, keepdims=True)
    pred[ix] = Pc.argmax(1)
z["cal"] = pred
ids = np.sort(z.respondent_id.unique()); ri = {r: i for i, r in enumerate(ids)}; R = len(ids); z["ri"] = z.respondent_id.map(ri)
W = rng.multinomial(R, np.ones(R) / R, size=3000).astype(float)
def agg(c): return np.bincount(z.ri, weights=c, minlength=R), np.bincount(z.ri, minlength=R).astype(float)
out = {}
for name, p in [("zs_raw", z.pred_idx.to_numpy()), ("zs_prior_corrected", z.cal.to_numpy()), ("finetuned", z.ft.to_numpy())]:
    s, n = agg((p == z.true_idx.to_numpy()).astype(float)); out[name] = {"acc": float(s.sum() / n.sum()), "ci": np.percentile((W @ s) / (W @ n), [2.5, 97.5]).tolist(), "_s": s, "_n": n}
def pdiff(a, b):
    sa, na = out[a]["_s"], out[a]["_n"]; sb, nb = out[b]["_s"], out[b]["_n"]
    d = (W @ sa) / (W @ na) - (W @ sb) / (W @ nb); return {"diff": float(sa.sum() / na.sum() - sb.sum() / nb.sum()), "ci": np.percentile(d, [2.5, 97.5]).tolist()}
res = {k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in out.items()}
res["paired"] = {"finetuned-zs_prior_corrected": pdiff("finetuned", "zs_prior_corrected"), "zs_prior_corrected-zs_raw": pdiff("zs_prior_corrected", "zs_raw")}
json.dump(res, open(f"{HERE}/zs_calib.json", "w"), indent=1)
print(json.dumps(res, indent=1))
