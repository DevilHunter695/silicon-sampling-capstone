"""Run transfer_analysis.py for every seed directory and summarise A-B and adapter-base differences across seeds.
usage: python summarize_seeds.py <baselines_probs.parquet> <out.json> <seed_dir> [<seed_dir> ...]   (each seed_dir has eval_A, eval_B, optionally eval_base)"""
import sys, subprocess, json, os
bp, outp, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
here = os.path.dirname(os.path.abspath(__file__)); res = {}
base_dir = next((f"{d}/eval_base" for d in dirs if os.path.isdir(f"{d}/eval_base")), None)
for d in dirs:
    o = f"{d}/transfer.json"
    cmd = [sys.executable, f"{here}/transfer_analysis.py", d, bp, o] + ([base_dir] if base_dir else [])
    r = subprocess.run(cmd, capture_output=True, text=True); print(r.stdout[-1500:]); print(r.stderr[-300:])
    try: res[os.path.basename(d.rstrip('/'))] = json.load(open(o))
    except Exception as e: res[os.path.basename(d.rstrip('/'))] = {"error": str(e)}
summ = {}
for g in ("trained_B(0:8)", "mid(8:24)", "heldout(24:40)"):
    for k in ("A - B [correct]", "A - base [correct]", "B - base [correct]"):
        vals = [r[g][k]["diff"] for r in res.values() if isinstance(r, dict) and g in r and k in r[g]]
        if vals: summ[f"{g} | {k}"] = {"per_seed": [round(100 * v, 2) for v in vals], "mean": round(100 * sum(vals) / len(vals), 2)}
json.dump({"seeds": list(res), "summary": summ}, open(outp, "w"), indent=1); print(json.dumps(summ, indent=1))
