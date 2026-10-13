"""WVS-7 no-demographics (P0) zero-shot Qwen: one prompt per item (15), scored with chained multi-token options."""
import json, os, sys
import numpy as np, pandas as pd
ORIG = os.getcwd()
REPO = "/Users/krishank/Documents/College Projects/🧢 /silicon-sampling-capstone"
sys.path.insert(0, REPO); os.chdir(REPO)
import mlx.core as mx
from mlx_lm import load
from src.inference.prompting import build_answer_instruction
from src.prompts.templates import build_prompt
from src.prompts.verbalize import load_codebook, verbalize_item
MODEL = sys.argv[1]; OUT = os.path.join(ORIG, sys.argv[2])
model, tok = load(MODEL); cb = load_codebook()
items = sorted(pd.read_parquet(f"{REPO}/results/predictions/groq-openai_gpt-oss-120b-low_P2.parquet").question_id.unique())
def lp(ids):
    lg = model(mx.array([ids]))[:, -1, :].astype(mx.float32)[0]; return lg - mx.logsumexp(lg)
res = {}
for q in items:
    it = verbalize_item(q, cb); labels = [str(c) for c in it["ordinal_values"]]
    prompt = build_prompt("P0", it["question_text"], it["options_text"]) + build_answer_instruction(labels)
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False), add_special_tokens=False)
    base = lp(ids); cand = [tok.encode(l, add_special_tokens=False) for l in labels]; first = {}
    for c in cand:
        if len(c) > 1 and c[0] not in first: first[c[0]] = lp(ids + [c[0]])
    sc = []
    for c in cand:
        if len(c) == 1:
            s = float(base[c[0]])
            if c[0] in first:
                cont = float(mx.logsumexp(mx.array([float(first[c[0]][d[1]]) for d in cand if len(d) > 1 and d[0] == c[0]]))); s += float(mx.log(1 - mx.exp(mx.array(cont))))
            sc.append(s)
        else: sc.append(float(base[c[0]]) + float(first[c[0]][c[1]]))
    sc = np.array(sc); p = np.exp(sc - sc.max()); p = (p / p.sum()).tolist()
    res[q] = {"probs": p, "pred_idx": int(np.argmax(p))}; print(q, res[q]["pred_idx"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
