import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = "/Users/krishank/Documents/College Projects/🧢 /silicon-sampling-capstone"
sys.path.insert(0, REPO); os.chdir(REPO)
from src.config import DATA_PROCESSED
from src.eval.run_baselines import DEMO_COLS, _encode_item
from src.prompts.verbalize import load_codebook
df = pd.read_parquet(f"{DATA_PROCESSED}/ind_wvs7.parquet")
if "respondent_id" not in df.columns: df["respondent_id"] = range(len(df))
folds = json.load(open(f"{DATA_PROCESSED}/folds.json"))["folds"]; items = json.load(open(f"{DATA_PROCESSED}/selected_items.json"))["selected_items"]
cb = load_codebook()
feat = df[["respondent_id"] + DEMO_COLS].copy(); feat["N_REGION_ISO"] = feat["N_REGION_ISO"].astype(str)
key = lambda d: d[DEMO_COLS].astype(object).fillna("NA").astype(str).agg("|".join, axis=1)
fkey = key(feat)
per_item = {"majority": [], "cell_mode": [], "uniform": []}
for q in items:
    y = _encode_item(df, q, cb); valid = y.notna()
    if valid.sum() < 100: continue
    K = int(y.max()) + 1
    maj, cell, n_ = 0, 0, 0
    for f in folds:
        tr = df.respondent_id.isin(f["train"]) & valid; te = df.respondent_id.isin(f["test"]) & valid
        ytr = y[tr].astype(int).values; mode = np.bincount(ytr).argmax()
        yte = y[te].astype(int).values
        maj += (yte == mode).sum(); n_ += len(yte)
        tab = pd.DataFrame({"k": fkey[tr].values, "a": ytr}).groupby("k")["a"].agg(lambda s: int(np.bincount(s).argmax())).to_dict()
        pred = np.array([tab.get(k, mode) for k in fkey[te].values]); cell += (pred == yte).sum()
    per_item["majority"].append(maj / n_); per_item["cell_mode"].append(cell / n_); per_item["uniform"].append(1.0 / K)
out = {k: float(np.mean(v)) for k, v in per_item.items()}; out["n_items"] = len(per_item["majority"])
json.dump(out, open("/private/tmp/claude-501/-Users-krishank-Documents-College-Projects----/c07d5fa6-6bd6-4110-b7a8-4e0125580098/scratchpad/work/analysis/wvs_mode144.json", "w"), indent=1); print(out)
