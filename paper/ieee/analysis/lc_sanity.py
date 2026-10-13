import json, os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(f"{HERE}/lc_N100/train.jsonl")][:160]
import mlx.core as mx
from mlx_lm import load
M = "/private/tmp/claude-501/-Users-krishank-Documents-College-Projects----/c07d5fa6-6bd6-4110-b7a8-4e0125580098/scratchpad/Qwen2.5-0.5B-Instruct-4bit"
for name, ad in [("base", None), ("N100", f"{HERE}/lc_adapters_q05/N100"), ("N25", f"{HERE}/lc_adapters_q05/N25")]:
    model, tok = load(M, adapter_path=ad) if ad else load(M)
    ok = 0; pred = []
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=False)
        ids = tok.encode(text, add_special_tokens=False)
        lg = model(mx.array([ids]))[:, -1, :][0]
        # restrict to digits 1..9
        dig = [tok.encode(str(c), add_special_tokens=False)[0] for c in range(1, 9)]
        p = int(np.argmax(np.array(lg[mx.array(dig)]).astype(float))) + 1
        pred.append(p); ok += (str(p) == r["completion"])
    print(name, "in-sample acc", round(ok / len(rows), 3), "pred dist", np.bincount(pred, minlength=9)[1:].tolist(), "true dist", np.bincount([int(r["completion"]) for r in rows], minlength=9)[1:].tolist())
