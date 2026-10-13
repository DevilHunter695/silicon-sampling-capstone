"""Build nested training sets for the data-efficiency experiment (Pew fold-0 TRAIN respondents) in mlx_lm {prompt, completion} format."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ORIG = os.getcwd()
sys.path.insert(0, "/tmp/ssc_pew"); os.chdir("/tmp/ssc_pew")
from src.config import DATA_PROCESSED
from src.inference.prompting import build_answer_instruction
from src.prompts.templates import build_prompt
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook, verbalize_demographics
cb = load_pew_codebook(); items = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][:8]
f0 = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"][0]
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet"); df["respondent_id"] = df.respondent_id.astype(int); df = df.set_index("respondent_id", drop=False)
order = np.random.default_rng(1234).permutation(np.array(sorted(f0["train"])))
json.dump([int(x) for x in order[:3200]], open(f"{HERE}/lc_train_order.json", "w"))
meta = {q: verbalize_item(q, cb) for q in items}
def rows_for(ids):
    out = []
    for rid in ids:
        r = df.loc[int(rid)]; demo = verbalize_demographics(r, cb)
        for q in items:
            tv = r.get(q); it = meta[q]
            if pd.isna(tv) or int(tv) not in it["code_to_index"]: continue
            labels = [str(c) for c in it["ordinal_values"]]
            out.append({"prompt": build_prompt("P2", it["question_text"], it["options_text"], **demo) + build_answer_instruction(labels), "completion": str(int(tv))})
    return out
valid = rows_for(order[-60:])[:200]
for N in [25, 100, 400, 1600]:
    d = f"{HERE}/lc_N{N}"; os.makedirs(d, exist_ok=True)
    tr = rows_for(order[:N])
    with open(f"{d}/train.jsonl", "w") as f: [f.write(json.dumps(x) + "\n") for x in tr]
    with open(f"{d}/valid.jsonl", "w") as f: [f.write(json.dumps(x) + "\n") for x in valid[:40]]
    with open(f"{d}/test.jsonl", "w") as f: [f.write(json.dumps(x) + "\n") for x in valid[:10]]
    print(N, len(tr))
