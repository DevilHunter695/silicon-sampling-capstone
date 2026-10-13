"""Faithful fold-0 inference for the Pew fine-tuned adapter on a CUDA GPU (Kaggle T4/P100).

One process = one GPU = one shard of respondents. It loads the 4-bit (NF4) base model once, then scores
  adapter_raw : base + LoRA adapter (the lab's fine-tuned model)
  base_raw    : same base, adapter disabled, same prompt (zero-shot control)
  base_chat   : same base, adapter disabled, chat template (zero-shot control)
Output per (condition, shard): JSONL rows {respondent_id, question_id, true_idx, probs, pred_idx, valid_mass, n_tokens, condition}.
Safe to kill and restart: finished rows are skipped; a time budget stops cleanly; non-finite logits abort loudly.
"""
import argparse, json, os, sys, time, glob
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="root of the pew-dataset-training checkout")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--prev-dir", default=None, help="directory with jsonl files from an earlier run to resume from")
ap.add_argument("--conditions", default="adapter_raw,base_raw,base_chat")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--n-resp", type=int, default=6000)
ap.add_argument("--item-range", default="0:8", help="slice of pew_selected_items to score, e.g. 0:40")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max-hours", type=float, default=10.0)
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--dry-run", action="store_true", help="build prompts and tokenise only; no model")
args = ap.parse_args()

import pandas as pd
os.chdir(args.repo)
sys.path.insert(0, args.repo)
from transformers import AutoTokenizer
from src.config import DATA_PROCESSED
from src.inference.prompting import build_answer_instruction
from src.prompts.templates import build_prompt
from src.prompts.verbalize import verbalize_item
from src.prompts.verbalize_pew import load_pew_codebook, verbalize_demographics

T0 = time.time()
os.makedirs(args.out_dir, exist_ok=True)
tok = AutoTokenizer.from_pretrained(args.adapter if args.dry_run else args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

cb = load_pew_codebook()
_lo, _hi = [int(x) for x in args.item_range.split(":")]
items = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][_lo:_hi]
folds = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"]
test_ids = np.array(sorted(folds[0]["test"]))
order = np.random.default_rng(args.seed).permutation(test_ids)[: min(args.n_resp, len(test_ids))]
mine = order[args.shard:: args.nshards]
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet")
df["respondent_id"] = df["respondent_id"].astype(int)
df = df.set_index("respondent_id", drop=False)
meta = {q: verbalize_item(q, cb) for q in items}
digit = {}
for q in items:
    ids = [tok.encode(str(c), add_special_tokens=False) for c in meta[q]["ordinal_values"]]
    assert all(len(i) == 1 for i in ids), f"option code is not a single token for {q}: {ids}"
    digit[q] = [i[0] for i in ids]


def build_rows(condition):
    rows = []
    for rid in mine:
        row = df.loc[int(rid)]
        demo = verbalize_demographics(row, cb)
        for q in items:
            it = meta[q]
            tv = row.get(q)
            if pd.isna(tv) or int(tv) not in it["code_to_index"]:
                continue
            labels = [str(c) for c in it["ordinal_values"]]
            prompt = build_prompt("P2", it["question_text"], it["options_text"], **demo) + build_answer_instruction(labels)
            text = (tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
                    if condition == "base_chat" else prompt + "\n\n")
            rows.append({"rid": int(rid), "q": q, "true_idx": it["code_to_index"][int(tv)], "text": text})
    return rows


conds = args.conditions.split(",")
if args.dry_run:
    for c in conds:
        r = build_rows(c)
        lens = [len(tok(x["text"], add_special_tokens=False)["input_ids"]) for x in r[:2000]]
        print(f"[dry] {c}: shard {args.shard}/{args.nshards} respondents={len(mine)} rows={len(r)} "
              f"tokens mean={np.mean(lens):.0f} max={max(lens)}")
    print("[dry] example tail:", repr(r[0]["text"][-160:]))
    print("[dry] digit tokens:", {q: digit[q] for q in items})
    sys.exit(0)

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

dt = getattr(torch, args.dtype)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt)
base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, torch_dtype=dt, device_map={"": 0})
model = PeftModel.from_pretrained(base, args.adapter)
model.eval()
print(f"[load] model ready after {(time.time()-T0)/60:.1f} min on {torch.cuda.get_device_name(0)}", flush=True)

for cond in conds:
    out = os.path.join(args.out_dir, f"{cond}_s{args.shard}of{args.nshards}.jsonl")
    if args.prev_dir:
        prev = os.path.join(args.prev_dir, os.path.basename(out))
        if os.path.exists(prev) and not os.path.exists(out):
            import shutil
            shutil.copy(prev, out)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
                done.add((r["respondent_id"], r["question_id"]))
            except Exception:
                pass  # a truncated last line from a killed run
    todo = [r for r in build_rows(cond) if (r["rid"], r["q"]) not in done]
    print(f"[{cond}] shard {args.shard}: {len(done)} done, {len(todo)} to do", flush=True)
    n, t1, stop = 0, time.time(), False
    CH = 256
    with open(out, "a") as fh:
        for c0 in range(0, len(todo), CH):
            chunk = todo[c0:c0 + CH]
            for r in chunk:
                r["n"] = len(tok(r["text"], add_special_tokens=False)["input_ids"])
            chunk.sort(key=lambda r: r["n"])
            for b0 in range(0, len(chunk), args.batch):
                if (time.time() - T0) / 3600 > args.max_hours:
                    stop = True
                    break
                bt = chunk[b0:b0 + args.batch]
                enc = tok([r["text"] for r in bt], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
                with torch.no_grad():
                    if cond == "adapter_raw":
                        lg = model(**enc).logits[:, -1].float()
                    else:
                        with model.disable_adapter():
                            lg = model(**enc).logits[:, -1].float()
                if not torch.isfinite(lg).all():
                    raise RuntimeError("non-finite logits; rerun with --dtype bfloat16 or float32 compute")
                full = torch.softmax(lg, -1)
                for i, r in enumerate(bt):
                    d = digit[r["q"]]
                    p = torch.softmax(lg[i, d], -1).cpu().tolist()
                    fh.write(json.dumps({"respondent_id": r["rid"], "question_id": r["q"], "true_idx": r["true_idx"], "probs": p,
                                         "pred_idx": int(np.argmax(p)), "valid_mass": float(full[i, d].sum()),
                                         "n_tokens": int(enc["input_ids"].shape[1]), "condition": cond}) + "\n")
                n += len(bt)
                if n % (args.batch * 25) < args.batch:
                    fh.flush()
                    print(f"[{cond}] s{args.shard} {n}/{len(todo)} rows, {(time.time()-t1)/max(n,1):.3f} s/row, "
                          f"elapsed {(time.time()-T0)/3600:.2f} h", flush=True)
            fh.flush()
            if stop:
                break
    if stop:
        print(f"[{cond}] time budget reached; stopping cleanly. Re-run with --prev-dir to resume.", flush=True)
        break
    print(f"[{cond}] shard {args.shard} complete", flush=True)
print("finished", flush=True)
