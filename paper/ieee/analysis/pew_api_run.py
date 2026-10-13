"""Step B: run the project's existing API runners (resumable cache) on the Pew prompts. Usage: pew_api_run.py {groq|gemini}"""
import json, os, sys, time
REPO = "/Users/krishank/Documents/College Projects/🧢 /silicon-sampling-capstone"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO); os.chdir(REPO)
for line in open(f"{REPO}/.env"):
    line = line.strip()
    if line.startswith("export "): line = line[7:]
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
which = sys.argv[1]
from src.inference.run_trackA import get_runner
runner = get_runner("gemini-3.1-flash-lite") if which == "gemini" else get_runner("groq-openai/gpt-oss-120b")
out_path = f"{HERE}/pew_api_{which}.jsonl"
done = set()
if os.path.exists(out_path):
    for l in open(out_path):
        r = json.loads(l); done.add((r["respondent_id"], r["question_id"]))
n = 0
with open(out_path, "a") as fh:
    for l in open(f"{HERE}/pew_api_prompts.jsonl"):
        r = json.loads(l)
        if (r["respondent_id"], r["question_id"]) in done: continue
        try:
            pred, probs, meta = runner.infer_single(r["prompt"], r["labels"])
        except Exception as e:
            pred, meta = None, {"error": str(e)[:200]}
        idx = r["labels"].index(str(pred)) if pred is not None and str(pred) in r["labels"] else None
        fh.write(json.dumps({"respondent_id": r["respondent_id"], "question_id": r["question_id"], "true_idx": r["true_idx"], "pred_idx": idx,
                             "raw": str(pred)[:40], "error": (meta or {}).get("error")}) + "\n"); fh.flush(); n += 1
        if n % 25 == 0: print(n, flush=True)
print("done", n)
