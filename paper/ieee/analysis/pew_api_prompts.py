"""Step A: build Pew P2 prompts (no closing instruction; the API runners add it) for a nested respondent subset."""
import json, os, sys
import numpy as np, pandas as pd
ORIG = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/tmp/ssc_pew"); os.chdir("/tmp/ssc_pew")
from src.config import DATA_PROCESSED
from src.prompts.templates import build_prompt
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook, verbalize_demographics
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
cb = load_pew_codebook(); items = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][:8]
f0 = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"][0]
ids = np.random.default_rng(7).permutation(np.array(sorted(f0["test"])))[:N]   # same prefix-consistent sample as zs_mlx.py
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet"); df["respondent_id"] = df.respondent_id.astype(int); df = df.set_index("respondent_id", drop=False)
meta = {q: verbalize_item(q, cb) for q in items}
n = 0
with open(f"{HERE}/pew_api_prompts.jsonl", "w") as f:
    for rid in ids:
        r = df.loc[int(rid)]; demo = verbalize_demographics(r, cb)
        for q in items:
            tv = r.get(q); it = meta[q]
            if pd.isna(tv) or int(tv) not in it["code_to_index"]: continue
            f.write(json.dumps({"respondent_id": int(rid), "question_id": q, "prompt": build_prompt("P2", it["question_text"], it["options_text"], **demo),
                                "labels": [str(c) for c in it["ordinal_values"]], "true_idx": it["code_to_index"][int(tv)]}) + "\n"); n += 1
print("prompts", n)
