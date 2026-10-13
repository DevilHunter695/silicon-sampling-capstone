"""Local (Apple-silicon, MLX) scoring of Pew fold-0 test rows with Qwen2.5-7B-Instruct.

Conditions
  base_raw     base model, raw prompt (+ "\n\n"), same input format the fine-tuned model was trained/tested on
  base_chat    base model, chat template (the natural zero-shot way to use an instruct model)
  adapter_raw  base + the delivered LoRA adapter, raw prompt (validates the port against the original predictions)

For every (respondent, item) the next-token distribution is read over the item's option digits, so we obtain both a hard
prediction (argmax) and probabilities. Results are appended to a jsonl so the run can be resumed.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

ORIG = os.getcwd()
CLONE = "/tmp/ssc_pew"
sys.path.insert(0, CLONE)
os.chdir(CLONE)

import mlx.core as mx
try:
    mx.set_cache_limit(768 * 1024 * 1024)
except Exception:
    pass
import mlx.nn as nn
from mlx_lm import load

from src.config import DATA_PROCESSED
from src.inference.prompting import build_answer_instruction
from src.prompts.templates import build_prompt
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook, verbalize_demographics

ap = argparse.ArgumentParser()
ap.add_argument("--condition", required=True, choices=["base_raw", "base_chat", "adapter_raw"])
ap.add_argument("--n-resp", type=int, default=100)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="/private/tmp/claude-501/-Users-krishank-Documents-College-Projects----/c07d5fa6-6bd6-4110-b7a8-4e0125580098/scratchpad/qwen4")
ap.add_argument("--adapter", default="/Users/krishank/Downloads/ishan-cpastone/adapter_model.safetensors")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--items", default=None, help="comma-separated item ids (default: the 8 used for fine-tuning)")
ap.add_argument("--max-rows", type=int, default=None)
ap.add_argument("--mlx-adapter", default=None, help="directory of an MLX-native LoRA adapter (from mlx_lm lora)")
args = ap.parse_args()
if not os.path.isabs(args.out):
    args.out = os.path.join(ORIG, args.out)
if args.mlx_adapter and not os.path.isabs(args.mlx_adapter):
    args.mlx_adapter = os.path.join(ORIG, args.mlx_adapter)


class LoRAWrap(nn.Module):
    def __init__(self, base, A, B, scale):
        super().__init__()
        self.base, self.A, self.B, self.scale = base, A, B, scale

    def __call__(self, x):
        return self.base(x) + self.scale * ((x @ self.A.T) @ self.B.T)


def attach_adapter(model, path, scale=2.0):
    w = mx.load(path)
    n = 0
    for k, v in w.items():
        if not k.endswith("lora_A.weight"):
            continue
        base_key = k[: -len(".lora_A.weight")]
        parts = base_key.replace("base_model.model.", "", 1).split(".")  # model.layers.N.self_attn.q_proj
        layer = model.model.layers[int(parts[2])]
        mod = getattr(layer.self_attn, parts[-1])
        A = w[k].astype(mx.bfloat16)
        B = w[base_key + ".lora_B.weight"].astype(mx.bfloat16)
        setattr(layer.self_attn, parts[-1], LoRAWrap(mod, A, B, scale))
        n += 1
    return n


codebook = load_pew_codebook()
items = args.items.split(",") if args.items else json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][:8]
folds = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"]
test_ids = np.array(sorted(folds[0]["test"]))
rng = np.random.default_rng(args.seed)
sample_ids = rng.permutation(test_ids)[: min(args.n_resp, len(test_ids))]  # prefix-consistent across --n-resp values
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet")
df["respondent_id"] = df["respondent_id"].astype(int)
df = df.set_index("respondent_id", drop=False)

model, tok = load(args.model, adapter_path=args.mlx_adapter) if args.mlx_adapter else load(args.model)
if args.condition == "adapter_raw":
    print("adapter modules attached:", attach_adapter(model, args.adapter), flush=True)

done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        r = json.loads(line)
        done.add((r["respondent_id"], r["question_id"]))

item_meta = {q: verbalize_item(q, codebook) for q in items}
digit_ids = {}
for q, it in item_meta.items():
    ids = []
    for c in it["ordinal_values"]:
        t = tok.encode(str(c), add_special_tokens=False)
        assert len(t) == 1, (c, t)
        ids.append(t[0])
    digit_ids[q] = ids

import copy
from mlx_lm.models.cache import make_prompt_cache

n_done = 0
t0 = time.time()


def build_ids(row, demo, q):
    it = item_meta[q]
    labels = [str(c) for c in it["ordinal_values"]]
    prompt = build_prompt("P2", it["question_text"], it["options_text"], **demo) + build_answer_instruction(labels)
    if args.condition == "base_chat":
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    else:
        text = prompt + "\n\n"
    return tok.encode(text, add_special_tokens=False)


with open(args.out, "a") as fh:
    for rid in sample_ids:
        row = df.loc[int(rid)]
        demo = verbalize_demographics(row, codebook)
        todo = []
        for q in items:
            if (int(rid), q) in done:
                continue
            it = item_meta[q]
            tv = row.get(q)
            if pd.isna(tv) or int(tv) not in it["code_to_index"]:
                continue
            todo.append((q, int(tv), build_ids(row, demo, q)))
        if not todo:
            continue
        # longest common token prefix across this respondent's prompts, computed once and reused
        L = min(len(t[2]) for t in todo) - 1
        for i in range(L):
            if any(t[2][i] != todo[0][2][i] for t in todo):
                L = i
                break
        cache = make_prompt_cache(model)
        if L > 0:
            model(mx.array([todo[0][2][:L]]), cache=cache)
            mx.eval([c.state for c in cache])
        for q, tv, ids in todo:
            it = item_meta[q]
            c = copy.deepcopy(cache)
            logits = model(mx.array([ids[L:]]), cache=c)[:, -1, :].astype(mx.float32)
            full = mx.softmax(logits, axis=-1)
            sel = logits[0, mx.array(digit_ids[q])]
            p = np.array(mx.softmax(sel, axis=-1)).tolist()
            valid_mass = float(full[0, mx.array(digit_ids[q])].sum())
            fh.write(json.dumps({"respondent_id": int(rid), "question_id": q, "true_idx": it["code_to_index"][tv],
                                 "probs": p, "pred_idx": int(np.argmax(p)), "valid_mass": valid_mass, "n_tokens": len(ids),
                                 "prefix_tokens": L, "condition": args.condition}) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 50 == 0:
                mx.clear_cache()
            if n_done % 24 == 0:
                print(f"{n_done} rows, {(time.time() - t0) / n_done:.2f}s/row", flush=True)
            if args.max_rows and n_done >= args.max_rows:
                sys.exit(0)
print("finished", n_done, "rows in", round(time.time() - t0), "s")
