"""No-profile (P0) control: one prompt per item, identical for every respondent. Scores each named adapter and the base model.
usage: python kaggle_p0.py --repo R --adapters lab=/path/a,B=/path/b --out p0.json [--item-range 0:40]"""
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--adapters", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--item-range", default="0:40"); ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_common import _ctx
df, items, folds, cb, meta = _ctx(a.repo)
from src.inference.prompting import build_answer_instruction
from src.prompts.templates import build_prompt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
lo, hi = [int(x) for x in a.item_range.split(":")]
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, torch_dtype=torch.float16, device_map={"": 0})
ad = dict(kv.split("=", 1) for kv in a.adapters.split(","))
names = list(ad)
model = PeftModel.from_pretrained(base, ad[names[0]], adapter_name=names[0])
for n in names[1:]:
    model.load_adapter(ad[n], adapter_name=n)
model.eval()
res = {}
for q in items[lo:hi]:
    it = meta[q]; labels = [str(c) for c in it["ordinal_values"]]
    text = build_prompt("P0", it["question_text"], it["options_text"]) + build_answer_instruction(labels) + "\n\n"
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    dig = [tok.encode(l, add_special_tokens=False)[0] for l in labels]
    res[q] = {}
    with torch.no_grad():
        for n in names:
            model.set_adapter(n)
            lg = model(**enc).logits[0, -1].float()
            res[q][n] = torch.softmax(lg[dig], -1).tolist()
        with model.disable_adapter():
            lg = model(**enc).logits[0, -1].float()
            res[q]["base"] = torch.softmax(lg[dig], -1).tolist()
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out, len(res), "items")
