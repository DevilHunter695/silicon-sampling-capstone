"""CUDA / Hugging Face twin of zs_mlx.py, for scaling the same experiment on a GPU server (4-bit bitsandbytes, optional PEFT adapter).

Run from the root of the pew-dataset-training branch:
    python analysis/zs_hf.py --condition base_raw    --n-resp 6000 --out mlx_base_raw.jsonl
    python analysis/zs_hf.py --condition base_chat   --n-resp 6000 --out mlx_base_chat.jsonl
    python analysis/zs_hf.py --condition adapter_raw --n-resp 6000 --out mlx_adapter_raw.jsonl --adapter pewB_run/model_fold0
Output format is identical to zs_mlx.py, so zs_eval.py consumes it unchanged. The run is resumable.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.getcwd())
from src.config import DATA_PROCESSED
from src.inference.prompting import build_answer_instruction
from src.prompts.templates import build_prompt
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook, verbalize_demographics

ap = argparse.ArgumentParser()
ap.add_argument("--condition", required=True, choices=["base_raw", "base_chat", "adapter_raw"])
ap.add_argument("--n-resp", type=int, default=6000)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--adapter", default=None)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
if args.condition == "adapter_raw":
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

cb = load_pew_codebook()
items = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][:8]
folds = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"]
test_ids = np.array(sorted(folds[0]["test"]))
sample = np.random.default_rng(args.seed).permutation(test_ids)[: min(args.n_resp, len(test_ids))]
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet")
df["respondent_id"] = df["respondent_id"].astype(int)
df = df.set_index("respondent_id", drop=False)
meta = {q: verbalize_item(q, cb) for q in items}
digit = {q: [tok.encode(str(c), add_special_tokens=False)[0] for c in meta[q]["ordinal_values"]] for q in items}

done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        r = json.loads(line)
        done.add((r["respondent_id"], r["question_id"]))
t0, n = time.time(), 0
with open(args.out, "a") as fh, torch.no_grad():
    for rid in sample:
        row = df.loc[int(rid)]
        demo = verbalize_demographics(row, cb)
        for q in items:
            if (int(rid), q) in done:
                continue
            it = meta[q]
            tv = row.get(q)
            if pd.isna(tv) or int(tv) not in it["code_to_index"]:
                continue
            labels = [str(c) for c in it["ordinal_values"]]
            prompt = build_prompt("P2", it["question_text"], it["options_text"], **demo) + build_answer_instruction(labels)
            text = (tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
                    if args.condition == "base_chat" else prompt + "\n\n")
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            logits = model(**ids).logits[0, -1].float()
            full = torch.softmax(logits, -1)
            p = torch.softmax(logits[digit[q]], -1).cpu().tolist()
            fh.write(json.dumps({"respondent_id": int(rid), "question_id": q, "true_idx": it["code_to_index"][int(tv)], "probs": p,
                                 "pred_idx": int(np.argmax(p)), "valid_mass": float(full[digit[q]].sum()),
                                 "n_tokens": int(ids["input_ids"].shape[1]), "condition": args.condition}) + "\n")
            n += 1
            if n % 200 == 0:
                fh.flush()
                print(n, "rows", round((time.time() - t0) / n, 3), "s/row", flush=True)
print("done", n)
