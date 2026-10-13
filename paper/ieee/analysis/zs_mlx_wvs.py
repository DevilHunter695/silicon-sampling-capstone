"""Local zero-shot Qwen2.5-7B-Instruct (4-bit MLX) on WVS-7 India, the same 15 items and P2 prompt as the API runs.

Respondents are drawn in a fixed random order (seed 42) so the run is resumable and can be extended simply by raising --n-resp.
Options whose code is multi-token (the two-digit "10" on 10-point scales) are scored exactly by chaining token log-probabilities.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

ORIG = os.getcwd()
REPO = "/Users/krishank/Documents/College Projects/🧢 /silicon-sampling-capstone"
sys.path.insert(0, REPO)
os.chdir(REPO)
import mlx.core as mx
from mlx_lm import load
from src.config import DATA_PROCESSED
from src.inference.prompting import build_answer_instruction
from src.prompts.templates import build_prompt
from src.prompts.verbalize import load_codebook, verbalize_demographics, verbalize_item

ap = argparse.ArgumentParser()
ap.add_argument("--n-resp", type=int, default=200)
ap.add_argument("--out", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-rows", type=int, default=None)
args = ap.parse_args()
if not os.path.isabs(args.out):
    args.out = os.path.join(ORIG, args.out)

df = pd.read_parquet(f"{DATA_PROCESSED}/ind_wvs7.parquet")
if "respondent_id" not in df.columns:
    df["respondent_id"] = range(len(df))
df["respondent_id"] = df["respondent_id"].astype(int)
items = sorted(pd.read_parquet(f"{REPO}/results/predictions/groq-openai_gpt-oss-120b-low_P2.parquet").question_id.unique())
codebook = load_codebook()
order = np.random.default_rng(args.seed).permutation(len(df))
resp = df.iloc[order[: args.n_resp]]

model, tok = load(args.model)
done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        r = json.loads(line)
        done.add((r["respondent_id"], r["question_id"]))
meta = {q: verbalize_item(q, codebook) for q in items}


import copy
from mlx_lm.models.cache import make_prompt_cache


def logp(logits):
    lg = logits[:, -1, :].astype(mx.float32)[0]
    return lg - mx.logsumexp(lg)


def build_ids(demo, q):
    it = meta[q]
    labels = [str(c) for c in it["ordinal_values"]]
    prompt = build_prompt("P2", it["question_text"], it["options_text"], **demo) + build_answer_instruction(labels)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


t0, n = time.time(), 0
with open(args.out, "a") as fh:
    for _, row in resp.iterrows():
        demo = verbalize_demographics(row, codebook)
        todo = []
        for q in items:
            if (int(row.respondent_id), q) in done:
                continue
            it = meta[q]
            tv = row.get(q)
            if pd.isna(tv) or int(tv) not in it["code_to_index"]:
                continue
            todo.append((q, int(tv), build_ids(demo, q)))
        if not todo:
            continue
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
            it = meta[q]
            labels = [str(c) for c in it["ordinal_values"]]
            c = copy.deepcopy(cache)
            lp = logp(model(mx.array([ids[L:]]), cache=c))
            cand = [tok.encode(l, add_special_tokens=False) for l in labels]
            first = {}
            for cd in cand:
                if len(cd) > 1 and cd[0] not in first:
                    c2 = copy.deepcopy(c)
                    first[cd[0]] = logp(model(mx.array([[cd[0]]]), cache=c2))
            scores = []
            for cd in cand:
                if len(cd) == 1:
                    sc_ = float(lp[cd[0]])
                    if cd[0] in first:
                        cont = float(mx.logsumexp(mx.array([float(first[cd[0]][d[1]]) for d in cand if len(d) > 1 and d[0] == cd[0]])))
                        sc_ += float(mx.log(1 - mx.exp(mx.array(cont))))
                    scores.append(sc_)
                else:
                    scores.append(float(lp[cd[0]]) + float(first[cd[0]][cd[1]]))
            sc = np.array(scores)
            pr = np.exp(sc - sc.max())
            pr = (pr / pr.sum()).tolist()
            fh.write(json.dumps({"respondent_id": int(row.respondent_id), "question_id": q, "true_idx": it["code_to_index"][tv],
                                 "probs": pr, "pred_idx": int(np.argmax(pr)), "n_tokens": len(ids), "prefix_tokens": L}) + "\n")
            fh.flush()
            n += 1
            if n % 24 == 0:
                print(f"{n} rows, {(time.time() - t0) / n:.2f}s/row", flush=True)
            if args.max_rows and n >= args.max_rows:
                sys.exit(0)
print("finished", n, "rows", round(time.time() - t0), "s")
