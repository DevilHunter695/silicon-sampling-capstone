"""Tables for the second paper draft (profile controls, transfer, data efficiency)."""
import json, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(os.path.dirname(HERE), "tables"); os.makedirs(T, exist_ok=True)
J = lambda n: json.load(open(os.path.join(HERE, n))) if os.path.exists(os.path.join(HERE, n)) else None
p1 = lambda x: f"{100*x:.1f}"
def s1(x):
    v = 100 * x
    return "0.0" if abs(v) < 0.05 else (f"+{v:.1f}" if v > 0 else f"$-${abs(v):.1f}")
Z, ZP, ZC, DR, AE = J("zs_eval.json"), J("zs_p0.json"), J("zs_calib.json"), J("pew_direction.json"), J("adapter_eval.json")
c = Z["chat"]
# P0 NLL on the same rows
p0 = json.load(open(os.path.join(HERE, "mlx_p0.json")))
zr = pd.read_json(os.path.join(HERE, "mlx_base_chat.jsonl"), lines=True).drop_duplicates(["respondent_id", "question_id"])
nll_p0 = float(np.mean([-np.log(max(p0[q]["probs"][t], 1e-6)) for q, t in zip(zr.question_id, zr.true_idx)]))
vd = DR["variance_decomposition"]
rows = [("True answers", None, None, None, None, f"{100*vd['true']['incremental_demo']:.1f}\\%"),
        ("Majority class (uses labels)", c["methods"]["majority"], c["methods"]["majority"].get("nll"), None, None, "---"),
        ("Zero-shot, no profile (P0)", {"acc": ZP["acc_p0"], "ci": ZP["acc_p0_ci"]}, nll_p0, None, None, "0.0\\%"),
        ("Zero-shot, profile (chat)", c["methods"]["zs_base_chat"], c["methods"]["zs_base_chat"]["nll"], DR["sub200_zero_shot_soft"], None, f"{100*vd['zero_shot_soft']['incremental_demo']:.1f}\\%"),
        ("Zero-shot, prior-corrected", ZC["zs_prior_corrected"], None, None, None, "---"),
        ("Fine-tuned (lab run, hard)", c["methods"]["finetuned"], None, DR["sub200_finetuned"], None, f"{100*vd['finetuned']['incremental_demo']:.1f}\\%"),
        ("Gradient boosting (uses labels)", c["methods"]["gboost"], c["methods"]["gboost"].get("nll"), DR["sub200_gboost"], None, f"{100*vd['gboost']['incremental_demo']:.1f}\\%")]
t = ["\\setlength{\\tabcolsep}{2.6pt}", "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
     "\\textbf{Predictor (200 respondents)} & \\textbf{Acc. (\\%)} [95\\% CI] & \\textbf{NLL} & \\textbf{Dir. $r$} & \\textbf{Profile var.} \\\\", "\\midrule"]
for name, m, nll, d, _, vs in rows:
    acc = f"{p1(m['acc'])} [{p1(m['ci'][0])}, {p1(m['ci'][1])}]" if m else "---"
    dd = f"{d['r']:.2f}" if d else "---"
    t.append(f"{name} & {acc} & {('%.2f' % nll) if nll is not None else '---'} & {dd} & {vs} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(T, "tab_controls.tex"), "w").write("\n".join(t))
# transfer table
U = AE["unseen"]; names = [("uniform_chance", "Chance (1/K)"), ("base_raw", "Zero-shot Qwen2.5-7B"), ("adapter_raw", "Fine-tuned Qwen2.5-7B (port)"),
                           ("majority", "Majority (uses labels)"), ("logistic", "Logistic (uses labels)"), ("gboost", "GBM (uses labels)")]
t = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lrrr@{}}", "\\toprule", "\\textbf{Method (8 unseen items)} & \\textbf{Acc. (\\%)} [95\\% CI] & \\textbf{NLL} & \\textbf{Dir. $r$} \\\\", "\\midrule"]
for k, lab in names:
    m = U["methods"][k]; d = U["direction"].get(k)
    r = (f"{d['r']:.2f}".replace("-", "$-$")) if d else "---"
    t.append(f"{lab} & {p1(m['acc'])} [{p1(m['ci'][0])}, {p1(m['ci'][1])}] & {('%.2f' % m['nll']) if 'nll' in m else '---'} & {r} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(T, "tab_transfer.tex"), "w").write("\n".join(t))
print("ok", round(nll_p0, 3))
