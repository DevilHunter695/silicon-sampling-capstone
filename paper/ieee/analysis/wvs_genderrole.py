"""Provenance for the exploratory gender-role paragraph: female-male difference in mean answer index, predicted (soft, zero-shot Qwen) vs true, per WVS item."""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/krishank/Documents/College Projects/🧢 /silicon-sampling-capstone"
sys.path.insert(0, REPO); os.chdir(REPO)
from src.config import DATA_PROCESSED
from src.eval.subgroups import assign_subgroups
q = pd.read_json(f"{HERE}/mlx_wvs_base_chat.jsonl", lines=True).drop_duplicates(["respondent_id", "question_id"])
df = pd.read_parquet(f"{DATA_PROCESSED}/ind_wvs7.parquet")
if "respondent_id" not in df.columns:
    df["respondent_id"] = range(len(df))
df["respondent_id"] = df["respondent_id"].astype(int)
sub = assign_subgroups(df.set_index("respondent_id", drop=False))
sexcol = "sg_sex"
q["sex"] = q.respondent_id.map(sub[sexcol])
q["pred_soft"] = [float(np.dot(np.arange(len(p)), p) / max(len(p) - 1, 1)) for p in q.probs]
q["true_n"] = [t / max(len(p) - 1, 1) for t, p in zip(q.true_idx, q.probs)]
rows = []
for it, g in q.groupby("question_id"):
    f, m = g[g.sex.astype(str).str.lower().str.startswith("f")], g[g.sex.astype(str).str.lower().str.startswith("m")]
    if len(f) < 10 or len(m) < 10:
        continue
    dp, dt = f.pred_soft.mean() - m.pred_soft.mean(), f.true_n.mean() - m.true_n.mean()
    rows.append({"item": it, "n_f": int(len(f)), "n_m": int(len(m)), "pred_diff": round(dp, 4), "true_diff": round(dt, 4),
                 "same_sign": bool(np.sign(dp) == np.sign(dt)), "ratio": round(dp / dt, 2) if abs(dt) > 1e-6 else None})
out = pd.DataFrame(rows); print(out.to_string(index=False)); out.to_csv(f"{HERE}/wvs_genderrole.csv", index=False)
