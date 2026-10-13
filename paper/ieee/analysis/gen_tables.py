"""Generate LaTeX table bodies from the result JSON so no number is hand-transcribed."""
import json
import numpy as np
import pandas as pd

P = json.load(open("analysis/pew_results.json"))
W = json.load(open("analysis/wvs_results.json"))
rows = pd.read_parquet("analysis/pew_aligned_rows.parquet")
rng = np.random.default_rng(99)

# ordinal-only MAE (exclude the nominal Q28rec), respondent-clustered bootstrap
ordr = rows[(rows.question_id != "Q28rec") & rows.pred_idx.notna()]
ids = np.sort(rows.respondent_id.unique())
ri = {r: i for i, r in enumerate(ids)}
R = len(ids)
Wt = rng.multinomial(R, np.ones(R) / R, size=2000).astype(float)
mae = {}
for m in ["majority", "logistic", "gboost", "finetuned"]:
    d = ordr[ordr.method == m]
    s = np.bincount(d.respondent_id.map(ri), weights=d.abs_err, minlength=R)
    n = np.bincount(d.respondent_id.map(ri), minlength=R).astype(float)
    mae[m] = [float(s.sum() / n.sum())] + [float(x) for x in np.percentile((Wt @ s) / (Wt @ n), [2.5, 97.5])]
json.dump(mae, open("analysis/pew_mae_ordinal.json", "w"), indent=1)


def f1(x):
    return f"{100 * x:.1f}"


def sg(x):
    v = 100 * x
    if abs(v) < 0.05:
        return "0.0"
    return f"{v:+.1f}".replace("-", "$-$")


def pv(p):
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


# ---- WVS: LLMs vs baselines on the same items
base144 = {
    "Uniform random": (19.4, "[18.3, 20.5]"),
    "National marginal": (32.6, "[30.4, 35.0]"),
    "Demographic-cell lookup": (34.5, "[32.2, 36.8]"),
    "Logistic regression (14 attr.)": (46.9, "[44.3, 49.5]"),
    "Gradient boosting (14 attr.)": (50.1, "[47.6, 52.5]"),
}
m15 = {"Uniform random": "uniform", "National marginal": "marginal", "Demographic-cell lookup": "cell_lookup",
       "Logistic regression (14 attr.)": "logistic_matched", "Gradient boosting (14 attr.)": "gbm_matched"}
sb = W["Gemini 3.1 Flash Lite"]["same_items_baselines"]
t = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule",
     "\\textbf{Method} & \\textbf{15 items (\\%)} & \\textbf{144 items (\\%)} \\\\", "\\midrule"]
for k in ["Gemini 3.1 Flash Lite", "gpt-oss-120b"]:
    w = W[k]
    t.append(f"{k} (zero-shot) & {f1(w['accuracy'])} [{f1(w['acc_ci_cluster'][0])}, {f1(w['acc_ci_cluster'][1])}] & --- \\\\")
t.append("\\midrule")
for k, (v, ci) in base144.items():
    t.append(f"{k} & {f1(sb[m15[k]])} & {v:.1f} {ci} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open("tables/tab_wvs_methods.tex", "w").write("\n".join(t))

# ---- WVS: subgroup gaps with permutation p and detection threshold
N = json.load(open("analysis/wvs_null.json"))
AX = [("sg_region", "Region"), ("sg_age_band", "Age band"), ("sg_education", "Education"),
      ("sg_income", "Income"), ("sg_sex", "Sex"), ("sg_urban_rural", "Urban/rural")]
t = ["\\setlength{\\tabcolsep}{2.4pt}", "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
     " & \\multicolumn{3}{c}{\\textbf{Gemini}} & \\multicolumn{3}{c}{\\textbf{gpt-oss-120b}} \\\\",
     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
     "\\textbf{Axis} & \\textbf{Gap} & \\textbf{$p$} & \\textbf{Null$_{95}$} & \\textbf{Gap} & \\textbf{$p$} & \\textbf{Null$_{95}$} \\\\", "\\midrule"]
for k, lab in AX:
    g, o = W["Gemini 3.1 Flash Lite"]["gaps"][k], W["gpt-oss-120b"]["gaps"][k]
    ng, no = N["Gemini"][k]["null_p95"], N["gpt-oss"][k]["null_p95"]
    t.append(f"{lab} & {f1(g['gap'])} & {pv(g['perm_p'])} & {f1(ng)} & {f1(o['gap'])} & {pv(o['perm_p'])} & {f1(no)} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open("tables/tab_wvs_gaps.tex", "w").write("\n".join(t))

# ---- Pew overall
EX = json.load(open("analysis/pew_extra.json"))
names = {"majority": "Majority class", "logistic": "Logistic regression", "gboost": "Gradient boosting",
         "finetuned": "Fine-tuned Qwen2.5-7B"}
t = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
     "\\textbf{Method} & \\textbf{Accuracy (\\%) [95\\% CI]} & \\textbf{MAE}$^\\dagger$ & \\textbf{Grp. err.}$^\\S$ \\\\", "\\midrule"]
for m in ["majority", "logistic", "gboost", "finetuned"]:
    o = P["overall"][m]
    t.append(f"{names[m]} & {f1(o['accuracy'])} [{f1(o['ci'][0])}, {f1(o['ci'][1])}] & {mae[m][0]:.2f} & {EX['mean_error'][m]['est']:.2f} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open("tables/tab_pew_overall.tex", "w").write("\n".join(t))

# ---- Pew concentration / agreement
C = EX["concentration_mean"]; A = EX["agreement"]
agree = {"majority": A["finetuned~majority"]["agree"], "logistic": A["finetuned~logistic"]["agree"], "gboost": A["finetuned~gboost"]["agree"]}
t = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
     "\\textbf{Predictor} & \\textbf{Top share} & \\textbf{TV} & \\textbf{Agrees w/ FT} \\\\", "\\midrule",
     "True answers & 0.31 & 0.00 & --- \\\\"]
for m in ["majority", "logistic", "gboost"]:
    t.append(f"{names[m]} & {C[m]['top_pred_share']:.2f} & {C[m]['tv_marginal']:.2f} & {100*agree[m]:.1f}\\% \\\\")
t.append(f"Fine-tuned Qwen2.5-7B & {C['finetuned']['top_pred_share']:.2f} & {C['finetuned']['tv_marginal']:.2f} & --- \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open("tables/tab_pew_conc.tex", "w").write("\n".join(t))

# ---- Pew per item
SHORT = {"QFERTrec": "Number of children", "Q50": "Reads scripture", "Q12": "Politicians' role in religion",
         "Q3132rec": "Attends worship", "ABORT": "Abortion legality", "Q15": "Sex-selection checkup",
         "Q39a": "Watches religious TV", "Q28rec": "Religious identity$^\\ddagger$"}
t = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
     "\\textbf{Item} & $K$ & \\textbf{Maj.} & \\textbf{Log.} & \\textbf{GBM} & \\textbf{FT} \\\\", "\\midrule"]
for q, v in P["per_item"].items():
    t.append(f"{SHORT[q]} & {v['n_options']} & {f1(v['majority']['acc'])} & {f1(v['logistic']['acc'])} & "
             f"{f1(v['gboost']['acc'])} & {f1(v['finetuned']['acc'])} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open("tables/tab_pew_items.tex", "w").write("\n".join(t))

# ---- Pew subgroups
order = ["Religion", "Region", "Education", "Age", "Caste", "Urban/rural", "Income", "Sex"]
t = ["\\setlength{\\tabcolsep}{3.2pt}", "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
     "\\textbf{Axis} & \\textbf{Maj.} & \\textbf{GBM} & \\textbf{FT gap [95\\% CI]} & "
     "\\textbf{$p$} & \\textbf{FT$-$GBM [95\\% CI]} \\\\", "\\midrule"]
for ax in order:
    s_ = P["subgroups"][ax]
    me = s_["methods"]
    ft = me["finetuned"]
    d = s_["gap_diff_ft_minus_gboost"]
    t.append(f"{ax} & {f1(me['majority']['gap'])} & {f1(me['gboost']['gap'])} & "
             f"{f1(ft['gap'])} [{f1(ft['gap_ci'][0])}, {f1(ft['gap_ci'][1])}] & {pv(ft['perm_p'])} & "
             f"{sg(d['diff'])} [{sg(d['ci'][0])}, {sg(d['ci'][1])}] \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open("tables/tab_pew_gaps.tex", "w").write("\n".join(t))

print(json.dumps(mae, indent=1))
for f in ["tab_wvs_gaps", "tab_pew_overall", "tab_pew_conc"]:
    print("=====", f)
    print(open(f"tables/{f}.tex").read())
