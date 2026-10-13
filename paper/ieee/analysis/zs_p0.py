"""No-demographics (P0) zero-shot Qwen: one prompt per item, identical for every respondent (8 forward passes for Pew, 15 for WVS)."""
import json, os, sys
import numpy as np
ORIG = os.getcwd()
sys.path.insert(0, "/tmp/ssc_pew"); os.chdir("/tmp/ssc_pew")
import mlx.core as mx
from mlx_lm import load
from src.config import DATA_PROCESSED
from src.inference.prompting import build_answer_instruction
from src.prompts.templates import build_prompt
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook
MODEL = sys.argv[1]; OUT = os.path.join(ORIG, sys.argv[2])
model, tok = load(MODEL)
cb = load_pew_codebook()
items = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][:8]
res = {}
for q in items:
    it = verbalize_item(q, cb); labels = [str(c) for c in it["ordinal_values"]]
    prompt = build_prompt("P0", it["question_text"], it["options_text"]) + build_answer_instruction(labels)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text, add_special_tokens=False)
    lg = model(mx.array([ids]))[:, -1, :].astype(mx.float32)[0]
    dig = [tok.encode(l, add_special_tokens=False)[0] for l in labels]
    p = np.array(mx.softmax(lg[mx.array(dig)], axis=-1)).tolist()
    res[q] = {"probs": p, "pred_idx": int(np.argmax(p)), "n_tokens": len(ids)}
    print(q, res[q]["pred_idx"], [round(x, 2) for x in p], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
