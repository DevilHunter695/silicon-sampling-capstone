"""Render paper_template.tex -> paper_ieee.tex, substituting <<path|fmt>> with numbers read from the analysis JSON files."""
import json, re, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "analysis")


def J(name):
    p = os.path.join(A, name)
    return json.load(open(p)) if os.path.exists(p) else {}


D = {
    "wm": J("wvs_matched.json"), "pe": J("pew_results.json"), "px": J("pew_extra.json"), "pd": J("pew_dispersion.json"),
    "zs": J("zs_eval.json"), "wq": J("wvs_qwen_eval.json"), "wn": J("wvs_null.json"), "w144": J("wvs_mode144.json"),
    "gc": J("pew_gap_closed.json"), "wr": J("wvs_results.json"), "we": J("wvs_extra.json"), "mo": J("pew_mae_ordinal.json"), "zc": J("zs_calib.json"), "zp": J("zs_p0.json"), "dr": J("pew_direction.json"), "wd": J("wvs_direction.json"), "ae": J("adapter_eval.json"), "did": J("transfer_did.json"),
    "api": J("api_eval.json"), "ff": J("fold0_full.json"), "tb": J("tuned_baselines.json"), "lcb": J("lc_baselines.json"), "lce": J("lc_eval.json"),
}


def _derived():
    d = {}
    try:
        me = D["px"]["mean_error"]
        d["red_ft"] = 1 - me["finetuned"]["est"] / me["majority"]["est"]
        d["red_gb"] = 1 - me["gboost"]["est"] / me["majority"]["est"]
    except Exception:
        pass
    try:
        import numpy as np
        d["uniform_pew"] = float(np.mean([1 / 4, 1 / 7, 1 / 4, 1 / 7, 1 / 4, 1 / 4, 1 / 7, 1 / 8]))
    except Exception:
        pass
    try:
        lcb = D["lcb"]
        d["lc_gap400"] = lcb["400"]["logistic"]["acc_mean_over_subsets"] - lcb["400"]["majority"]["acc_mean_over_subsets"]
        d["lc_gap1600"] = lcb["1600"]["logistic"]["acc_mean_over_subsets"] - lcb["1600"]["majority"]["acc_mean_over_subsets"]
    except Exception:
        pass
    return d


D["calc"] = _derived()


def get(path):
    cur = D
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def fmt(v, f):
    if f == "p1":
        return f"{100 * v:.1f}"
    if f == "p0":
        return f"{100 * v:.0f}"
    if f == "a1":  # absolute value in points
        return f"{abs(100 * v):.1f}"
    if f == "s1":  # signed points with typographic minus
        x = 100 * v
        return "0.0" if abs(x) < 0.05 else (f"+{x:.1f}" if x > 0 else f"$-${abs(x):.1f}")
    if f == "d2":
        return f"{v:.2f}"
    if f == "d3":
        return f"{v:.3f}"
    if f == "d1":
        return f"{v:.1f}"
    if f == "x1":
        return f"{v:.1f}$\\times$"
    if f == "int":
        return f"{int(v):,}".replace(",", "{,}")
    raise ValueError(f)


def sub(m):
    path, f = m.group(1), m.group(2)
    try:
        return fmt(get(path), f)
    except Exception as e:  # leave a visible marker rather than silently inventing a number
        sys.stderr.write(f"MISSING {path}: {e}\n")
        return "\\textbf{??}"


tmpl = sys.argv[1] if len(sys.argv) > 1 else "paper_template.tex"
outname = sys.argv[2] if len(sys.argv) > 2 else "paper_ieee.tex"
src = open(os.path.join(HERE, tmpl)).read()
out = re.sub(r"<<([A-Za-z0-9_.\-]+)\|(\w+)>>", sub, src)
open(os.path.join(HERE, outname), "w").write(out)
print("rendered", len(out), "chars; missing:", out.count("??"))
