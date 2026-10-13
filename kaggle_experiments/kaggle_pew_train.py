"""QLoRA fine-tuning of Qwen2.5-7B-Instruct on Pew (fold-0 train respondents) for a Kaggle T4. Answer-only loss, fp16, resumable.

One process = one GPU = one adapter. Loss is cross-entropy on the single answer-digit token that follows prompt + "\\n\\n"
(the same format the lab's adapter was trained and is scored in). LoRA: r=16, alpha=32, dropout 0.05, q/k/v/o (the lab's setting).
"""
import argparse, json, os, sys, time, math
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--item-range", default="0:8", help="python-style slice into pew_selected_items, e.g. 0:8 or 0:24")
ap.add_argument("--n-resp", type=int, default=2000)
ap.add_argument("--n-ex", type=int, default=0, help="<=0: match the 8-item pool size (same examples for 0:8 and 0:24)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--micro-batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (smoke test)")
ap.add_argument("--save-every", type=int, default=100)
ap.add_argument("--max-hours", type=float, default=10.0)
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

T0 = time.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_common import train_pairs, _ctx
lo, hi = [int(x) for x in args.item_range.split(":")]
repo = args.repo
pairs, ref_n = train_pairs(repo, lo, hi, args.n_resp, args.n_ex, args.seed)
df, items, folds, cb, meta = _ctx(repo)
from transformers import AutoTokenizer
from src.inference.prompting import build_answer_instruction
from src.prompts.templates import build_prompt
from src.prompts.verbalize_pew import verbalize_demographics
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
os.makedirs(args.out_dir, exist_ok=True)
json.dump({"args": vars(args), "n_pairs": len(pairs), "ref_pool": ref_n,
           "n_respondents": len({r for r, _ in pairs}), "items": items[lo:hi]}, open(f"{args.out_dir}/train_meta.json", "w"), indent=1)
print(f"[data] items {lo}:{hi} -> {len(pairs)} examples from {len({r for r,_ in pairs})} respondents (8-item pool {ref_n})", flush=True)

demo_cache = {}


def example(rid, q):
    if rid not in demo_cache:
        demo_cache[rid] = verbalize_demographics(df.loc[rid], cb)
    it = meta[q]
    labels = [str(c) for c in it["ordinal_values"]]
    prompt = build_prompt("P2", it["question_text"], it["options_text"], **demo_cache[rid]) + build_answer_instruction(labels)
    ans = str(int(df.loc[rid].get(q)))
    t = tok.encode(ans, add_special_tokens=False)
    assert len(t) == 1
    return prompt + "\n\n", t[0]


if args.dry_run:
    ls = [len(tok(example(*p)[0], add_special_tokens=False)["input_ids"]) for p in pairs[:500]]
    print(f"[dry] tokens mean={np.mean(ls):.0f} max={max(ls)}; steps/epoch={len(pairs)//(args.micro_batch*args.accum)}")
    print("[dry] example tail:", repr(example(*pairs[0])[0][-120:]), "target token", example(*pairs[0])[1])
    sys.exit(0)

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, torch_dtype=torch.float16, device_map={"": 0})
base.config.use_cache = False
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
base.enable_input_require_grads()
model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
for _p in model.parameters():
    if _p.requires_grad:
        _p.data = _p.data.float()  # GradScaler needs fp32 trainable weights
model.print_trainable_parameters()
inner = model.base_model.model  # Qwen2ForCausalLM with LoRA layers injected in place
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
eff = args.micro_batch * args.accum
steps_total = args.epochs * (len(pairs) // eff)
warm = 20
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(s, steps_total) / max(steps_total, 1)))) )
scaler = torch.amp.GradScaler("cuda")

ck = f"{args.out_dir}/ckpt"
step = 0
if os.path.exists(f"{ck}/state.pt"):
    st = torch.load(f"{ck}/state.pt", map_location="cpu")
    set_peft_model_state_dict(model, load_file(f"{ck}/adapter_model.safetensors"))
    opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"]); sched.load_state_dict(st["sched"]); step = st["step"]
    print(f"[resume] from step {step}", flush=True)


def save(path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)


logf = open(f"{args.out_dir}/train_log.jsonl", "a")
model.train()
t1 = time.time()
done_steps_this_run = 0
stopped = False
while step < steps_total and not stopped:
    ep, within = divmod(step, max(len(pairs) // eff, 1))
    order = np.random.default_rng(args.seed + 17 + ep).permutation(len(pairs))
    chunk = [pairs[i] for i in order[within * eff:(within + 1) * eff]]
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for m0 in range(0, eff, args.micro_batch):
        ex = [example(*p) for p in chunk[m0:m0 + args.micro_batch]]
        enc = tok([e[0] for e in ex], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        tgt = torch.tensor([e[1] for e in ex], device="cuda:0")
        with torch.autocast("cuda", dtype=torch.float16):
            h = inner.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False).last_hidden_state[:, -1]
            logits = inner.lm_head(h).float()
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss; reduce lr or switch precision")
        scaler.scale(loss * len(ex) / eff).backward()
        tot += loss.item() * len(ex) / eff
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    step += 1; done_steps_this_run += 1
    if step % 10 == 0 or step == 1:
        rec = {"step": step, "loss": round(tot, 4), "lr": sched.get_last_lr()[0], "s_per_step": round((time.time() - t1) / done_steps_this_run, 2),
               "elapsed_h": round((time.time() - T0) / 3600, 2)}
        logf.write(json.dumps(rec) + "\n"); logf.flush(); print(rec, flush=True)
    if step % args.save_every == 0:
        save(ck)
        torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict(), "sched": sched.state_dict(), "step": step}, f"{ck}/state.pt")
    if args.max_steps and done_steps_this_run >= args.max_steps:
        stopped = True
    if (time.time() - T0) / 3600 > args.max_hours:
        print("[time] budget reached; checkpointing and exiting", flush=True)
        save(ck); torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict(), "sched": sched.state_dict(), "step": step}, f"{ck}/state.pt")
        stopped = True
if step >= steps_total:
    save(f"{args.out_dir}/adapter_final")
    print("[done] final adapter saved", flush=True)
else:
    print(f"[stop] at step {step}/{steps_total}; re-run the same command to resume", flush=True)
