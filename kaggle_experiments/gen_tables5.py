"""LaTeX tables for the final paper from fold0_full.json (fold0_full_analysis.py) and transfer.json (transfer_analysis.py).
usage: python gen_tables5.py <fold0_full.json> <transfer.json> <out_dir>"""
import json, sys, os
f0p, trp, od = sys.argv[1:4]
os.makedirs(od, exist_ok=True)
pc = lambda v: f"{100*v:.1f}"
sg = lambda v: ("$-$" if v < 0 else "+") + f"{abs(100*v):.1f}"
NAMES = {"majority": "Majority class", "logistic": "Logistic regression", "gboost": "Gradient boosting", "finetuned": "Fine-tuned Qwen2.5-7B",
         "base_raw": "Zero-shot Qwen2.5-7B (raw)", "base_chat": "Zero-shot Qwen2.5-7B (chat)"}
if os.path.exists(f0p):
    r = json.load(open(f0p)); o = r["overall"]
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Method & Acc.\ (\%) [95\% CI] & NLL & ECE \\", r"\midrule"]
    for k in ("majority", "logistic", "gboost", "finetuned", "base_raw", "base_chat"):
        if k in o:
            v = o[k]
            L.append(f"{NAMES[k]} & {pc(v['acc'])} [{pc(v['acc_ci'][0])}, {pc(v['acc_ci'][1])}] & {v['nll']:.2f} & {v['ece']:.3f} \\\\")
    pr = r["paired"]
    L.append(r"\midrule")
    for key, lab in [("finetuned - gboost [correct]", "FT $-$ GBM, accuracy (pts)"), ("finetuned - majority [correct]", "FT $-$ majority, accuracy (pts)"),
                     ("finetuned - base_raw [correct]", "FT $-$ zero-shot (raw), accuracy (pts)"), ("ens(FT,GBM) - gboost [correct]", "FT+GBM average $-$ GBM, accuracy (pts)")]:
        if key in pr:
            v = pr[key]; L.append(f"{lab} & \\multicolumn{{3}}{{l}}{{{sg(v['diff'])} [{sg(v['ci'][0])}, {sg(v['ci'][1])}]}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{od}/tab_fold0_full.tex", "w").write("\n".join(L) + "\n")
if os.path.exists(trp):
    t = json.load(open(trp)); ga = t.get("group_acc", {})
    cols = [("base", "Base"), ("B", "FT-8"), ("A", "FT-24"), ("majority|pool2k", "Maj."), ("logistic|pool2k", "Logit"), ("gboost|pool2k", "GBM")]
    L = [r"\begin{tabular}{@{}l" + "c" * len(cols) + r"@{}}", r"\toprule", "Items & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for g, lab in [("trained_B(0:8)", "Trained by both (1--8)"), ("mid(8:24)", "Trained by FT-24 only (9--24)"), ("heldout(24:40)", "Held out for both (25--40)"),
                   ("heldout_near", "\\quad held out, sibling in training"), ("heldout_far", "\\quad held out, no sibling")]:
        if g in ga and ga[g]:
            row = [pc(ga[g][m]["correct"][0]) if m in ga[g] else "--" for m, _ in cols]
            L.append(f"{lab} & " + " & ".join(row) + r" \\")
    L += [r"\midrule", r"\multicolumn{" + str(len(cols) + 1) + r"}{@{}l}{Paired differences, accuracy points [95\% CI]} \\"]
    for g, lab in [("trained_B(0:8)", "items 1--8"), ("mid(8:24)", "items 9--24"), ("heldout(24:40)", "items 25--40")]:
        for key, nm in [("A - B [correct]", "FT-24 $-$ FT-8"), ("A - base [correct]", "FT-24 $-$ base"), ("B - base [correct]", "FT-8 $-$ base")]:
            v = t.get(g, {}).get(key)
            if v:
                L.append(f"{nm}, {lab} & \\multicolumn{{{len(cols)}}}{{l}}{{{sg(v['diff'])} [{sg(v['ci'][0])}, {sg(v['ci'][1])}]}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{od}/tab_breadth.tex", "w").write("\n".join(L) + "\n")
print("tables written to", od, os.listdir(od))
