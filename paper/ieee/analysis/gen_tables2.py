"""Tables for the revised paper (matched-row WVS comparison, distributional diagnostics)."""
import json
M = json.load(open("analysis/wvs_matched.json")); D144 = json.load(open("analysis/wvs_mode144.json"))
DISP = json.load(open("analysis/pew_dispersion.json")); EX = json.load(open("analysis/pew_extra.json"))
f1 = lambda x: f"{100*x:.1f}"
def sgn(x):
    v = 100 * x
    return ("0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "$-$"))
# --- WVS matched rows
names = [("majority", "Majority class"), ("cell_mode", "Demographic-cell mode (6 attr.)"), ("logistic_matched", "Logistic regression (14 attr.)"), ("gbm_matched", "Gradient boosting (14 attr.)")]
t = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lrr@{}}", "\\toprule",
     " & \\multicolumn{1}{c}{\\textbf{Gemini rows}} & \\multicolumn{1}{c}{\\textbf{gpt-oss rows}} \\\\",
     " & \\multicolumn{1}{c}{($n{=}421$, 30 resp.)} & \\multicolumn{1}{c}{($n{=}556$, 40 resp.)} \\\\", "\\midrule"]
g, o = M["Gemini"], M["gpt-oss"]
t.append(f"LLM accuracy (\\%) & {f1(g['llm']['acc'])} [{f1(g['llm']['ci'][0])}, {f1(g['llm']['ci'][1])}] & {f1(o['llm']['acc'])} [{f1(o['llm']['ci'][0])}, {f1(o['llm']['ci'][1])}] \\\\")
t.append("\\midrule")
t.append("\\multicolumn{3}{@{}l}{\\textit{Baseline accuracy (\\%); LLM minus baseline [95\\% CI]}} \\\\")
for k, lab in names:
    a, b = g[k], o[k]
    t.append(f"{lab} & {f1(a['acc'])}; {sgn(a['llm_minus'])} [{sgn(a['llm_minus_ci'][0])}, {sgn(a['llm_minus_ci'][1])}] & "
             f"{f1(b['acc'])}; {sgn(b['llm_minus'])} [{sgn(b['llm_minus_ci'][0])}, {sgn(b['llm_minus_ci'][1])}] \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open("tables/tab_wvs_matched.tex", "w").write("\n".join(t))
# --- distributional diagnostics (Pew)
C = EX["concentration_mean"]; A = EX["agreement"]
names2 = {"majority": "Majority class", "logistic": "Logistic regression", "gboost": "Gradient boosting", "finetuned": "Fine-tuned Qwen2.5-7B"}
t = ["\\setlength{\\tabcolsep}{2.6pt}", "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
     "\\textbf{Predictor} & \\textbf{Top share} & \\textbf{TV} & \\textbf{SD ret.} & \\textbf{Gap infl.} \\\\", "\\midrule",
     "True answers & 0.31 & 0.00 & 1.00 & 1.00 \\\\"]
for m in ["majority", "logistic", "gboost", "finetuned"]:
    d = DISP[m]
    t.append(f"{names2[m]} & {C[m]['top_pred_share']:.2f} & {C[m]['tv_marginal']:.2f} & {d['dispersion_retention']:.2f} & {d['gap_inflation_median']:.1f}$\\times$ \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open("tables/tab_pew_conc.tex", "w").write("\n".join(t))
# --- WVS 144 context
print(json.dumps({"144": D144}, indent=1))
print(open("tables/tab_wvs_matched.tex").read()); print(open("tables/tab_pew_conc.tex").read())
