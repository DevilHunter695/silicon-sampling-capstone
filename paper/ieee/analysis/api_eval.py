"""Zero-shot gpt-oss-120b (Groq, low reasoning) on the Pew items: nested 100-respondent subset, identical rows with baselines, FT and local zero-shot Qwen."""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/tmp/ssc_pew"); os.chdir("/tmp/ssc_pew")
from scripts.pewB_baselines import run_fold
from src.config import DATA_PROCESSED
from src.prompts.verbalize_pew import load_pew_codebook
rng = np.random.default_rng(777); key = ["respondent_id", "question_id"]
cb = load_pew_codebook(); items = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][:8]
f0 = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"][0]
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet"); df["respondent_id"] = df.respondent_id.astype(int)
g = pd.read_json(f"{HERE}/pew_api_groq.jsonl", lines=True).drop_duplicates(key)
orig = pd.read_parquet(f"{HERE}/pew_aligned_rows.parquet"); ft = orig[orig.method == "finetuned"][key + ["pred_idx"]].rename(columns={"pred_idx": "ft"})
zq = pd.read_json(f"{HERE}/mlx_base_chat.jsonl", lines=True).drop_duplicates(key)[key + ["pred_idx"]].rename(columns={"pred_idx": "qwen_zs"})
S = g.merge(ft, on=key).merge(zq, on=key, how="left").sort_values(key).reset_index(drop=True)
base = run_fold(df, f0, items, cb)
for m, d in base.items():
    d = d.copy(); d["respondent_id"] = d.respondent_id.astype(int)
    S = S.merge(d.drop_duplicates(key).assign(**{m: lambda x: x.proba.map(lambda p: int(np.argmax(p)))})[key + [m]], on=key, how="left")
S["groq"] = S.pred_idx.where(S.pred_idx.notna(), -1).astype(int)
ids = np.sort(S.respondent_id.unique()); ri = {r: i for i, r in enumerate(ids)}; R = len(ids); S["ri"] = S.respondent_id.map(ri)
W = rng.multinomial(R, np.ones(R) / R, size=4000).astype(float)
def agg(mask_col):
    c = (S[mask_col] == S.true_idx).astype(float).to_numpy(); return np.bincount(S.ri, weights=c, minlength=R), np.bincount(S.ri, minlength=R).astype(float)
out = {"n_rows": int(len(S)), "n_resp": int(R), "n_unparsed": int((S.pred_idx.isna()).sum()), "methods": {}, "paired": {}}
st = {}
for m in ["groq", "majority", "logistic", "gboost", "ft"]:
    s, n = agg(m); st[m] = (s, n); out["methods"][m] = {"acc": float(s.sum() / n.sum()), "ci": np.percentile((W @ s) / (W @ n), [2.5, 97.5]).tolist()}
sub = S[S.qwen_zs.notna()]
out["qwen_zs_on_same_rows"] = float((sub.qwen_zs == sub.true_idx).mean()); out["n_rows_with_qwen"] = int(len(sub))
for b in ["majority", "logistic", "gboost", "ft"]:
    d = (W @ st["groq"][0]) / (W @ st["groq"][1]) - (W @ st[b][0]) / (W @ st[b][1])
    out["paired"][f"groq-{b}"] = {"diff": float(st["groq"][0].sum() / st["groq"][1].sum() - st[b][0].sum() / st[b][1].sum()), "ci": np.percentile(d, [2.5, 97.5]).tolist()}
tops = {q: float(np.bincount(S[S.question_id == q].groq.clip(0)).max() / (S.question_id == q).sum()) for q in items}
out["mean_top_share"] = float(np.mean(list(tops.values())))
json.dump(out, open(f"{HERE}/api_eval.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["n_rows", "n_resp", "n_unparsed", "methods", "paired", "qwen_zs_on_same_rows", "mean_top_share"]}, indent=1))
