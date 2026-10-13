"""Tables that depend on the local zero-shot runs. Run after zs_eval.py and wvs_qwen_eval.py."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(os.path.dirname(HERE), "tables")
os.makedirs(T, exist_ok=True)


def J(n):
    p = os.path.join(HERE, n)
    return json.load(open(p)) if os.path.exists(p) else None


def p1(x):
    return f"{100 * x:.1f}"


def s1(x):
    v = 100 * x
    return "0.0" if abs(v) < 0.05 else (f"+{v:.1f}" if v > 0 else f"$-${abs(v):.1f}")


M, Q, Z, W = J("wvs_matched.json"), J("wvs_qwen_eval.json"), J("zs_eval.json"), J("wvs_null.json")
WR = J("wvs_results.json")

# ---------- WVS: matched rows, three LLMs
if M:
    cols = [("Gemini", M["Gemini"]), ("gpt-oss-120b", M["gpt-oss"])]
    if Q:
        cols.append(("Qwen2.5-7B", Q))
    names = [("majority", "Majority class"), ("cell_mode", "Cell mode (6 attr.)"), ("logistic_matched", "Logistic (14 attr.)"), ("gbm_matched", "GBM (14 attr.)")]
    t = ["\\setlength{\\tabcolsep}{2.4pt}", "\\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", "\\toprule",
         "\\textbf{Row sample} & " + " & ".join(f"\\textbf{{{c[0]}}}" for c in cols) + " \\\\",
         "Respondents / rows & " + " & ".join(f"{c[1]['n_resp']} / {c[1]['n_rows']}" for c in cols) + " \\\\", "\\midrule",
         "LLM accuracy (\\%) & " + " & ".join(f"{p1(c[1]['llm']['acc'])} [{p1(c[1]['llm']['ci'][0])}, {p1(c[1]['llm']['ci'][1])}]" for c in cols) + " \\\\",
         "\\midrule", "\\multicolumn{%d}{@{}l}{\\textit{Baseline accuracy (\\%%); LLM $-$ baseline [95\\%% CI]}} \\\\" % (len(cols) + 1)]
    for k, lab in names:
        cells = []
        for _, c in cols:
            r = c[k]
            cells.append(f"{p1(r['acc'])}; {s1(r['llm_minus'])} [{s1(r['llm_minus_ci'][0])}, {s1(r['llm_minus_ci'][1])}]")
        t.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(T, "tab_wvs_matched.tex"), "w").write("\n".join(t))

# ---------- WVS subgroup gaps: API LLMs + Qwen
if WR and W:
    AX = [("sg_region", "Region"), ("sg_age_band", "Age band"), ("sg_education", "Education"), ("sg_income", "Income"),
          ("sg_sex", "Sex"), ("sg_urban_rural", "Urban/rural")]

    def pv(p):
        return "$<$.001" if p < 0.001 else f"{p:.2f}".lstrip("0")

    hdr = ["Gemini (30)", "gpt-oss (40)"] + (["Qwen (%d)" % Q["n_resp"]] if Q else [])
    t = ["\\setlength{\\tabcolsep}{2.4pt}", "\\begin{tabular}{@{}l" + "r" * len(hdr) + "@{}}", "\\toprule",
         "\\textbf{Axis} & " + " & ".join(f"\\textbf{{{h}}}" for h in hdr) + " \\\\", "\\midrule"]
    for k, lab in AX:
        cells = []
        for nm in ("Gemini 3.1 Flash Lite", "gpt-oss-120b"):
            g = WR[nm]["gaps"][k]
            nk = "Gemini" if nm.startswith("Gemini") else "gpt-oss"
            cells.append(f"{p1(g['gap'])} ({pv(g['perm_p'])}) [{p1(W[nk][k]['null_p95'])}]")
        if Q:
            g = Q["gaps"][k]
            cells.append(f"{p1(g['gap'])} ({pv(g['p'])}) [{p1(g['null95'])}]")
        t.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(T, "tab_wvs_gaps.tex"), "w").write("\n".join(t))

# ---------- Pew: zero-shot control on identical rows
if Z and "chat" in Z:
    r = Z["chat"]
    r_chat = Z["chat"]
    names = [("majority", "Majority class"), ("logistic", "Logistic regression"), ("gboost", "Gradient boosting"),
             ("zs_base_chat", "Zero-shot Qwen2.5-7B (chat)"), ("finetuned", "Fine-tuned Qwen2.5-7B")]
    t = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
         "\\textbf{Method} & \\textbf{Accuracy (\\%) [95\\% CI]} & \\textbf{NLL} & \\textbf{FT $-$ method} \\\\", "\\midrule"]
    src = r
    for k, lab in names:
        if k not in src["methods"]:
            continue
        m = src["methods"][k]
        nll = f"{m['nll']:.2f}" if "nll" in m else "---"
        pd_ = src["paired"].get(f"finetuned-{k}") if k != "finetuned" else None
        if pd_ is None and k.startswith("zs_"):
            q_ = src["paired"].get(f"{k}-finetuned")
            if q_:
                pd_ = {"diff": -q_["diff"], "ci": [-q_["ci"][1], -q_["ci"][0]]}
        dcell = f"{s1(pd_['diff'])} [{s1(pd_['ci'][0])}, {s1(pd_['ci'][1])}]" if pd_ else "---"
        t.append(f"{lab} & {p1(m['acc'])} [{p1(m['ci'][0])}, {p1(m['ci'][1])}] & {nll} & {dcell} \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(T, "tab_pew_zs.tex"), "w").write("\n".join(t))
print("tables written")
