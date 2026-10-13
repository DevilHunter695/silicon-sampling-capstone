import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm
names = {f.name for f in fm.fontManager.ttflist}
serif = "Times New Roman" if "Times New Roman" in names else "DejaVu Serif"
plt.rcParams.update({"font.family": "serif", "font.serif": [serif], "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "stix"})
R = json.load(open("analysis/pew_results.json"))
SHORT = {"QFERTrec": "Number of children", "Q50": "Reads scripture", "Q12": "Politicians' role in religion", "Q3132rec": "Attends worship",
         "ABORT": "Abortion legality", "Q15": "Sex-selection checkup", "Q39a": "Watches religious TV", "Q28rec": "Religious identity"}
MK = {"majority": ("o", "white", "Majority"), "logistic": ("^", "0.55", "Logistic"), "gboost": ("s", "0.25", "GBM"), "finetuned": ("D", "black", "Fine-tuned Qwen2.5-7B")}

# ---------- Fig 2: two panels
items = sorted(R["per_item"], key=lambda q: R["per_item"][q]["finetuned"]["acc"] - R["per_item"][q]["majority"]["acc"])
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"width_ratios": [1.15, 1]})
y = np.arange(len(items))
a = ax[0]
for m, (mk, fc, lab) in MK.items():
    xs = [100 * R["per_item"][q][m]["acc"] for q in items]
    a.scatter(xs, y, marker=mk, s=22, facecolor=fc, edgecolor="black", linewidth=0.7, label=lab, zorder=3)
for i, q in enumerate(items):
    lo, hi = R["per_item"][q]["finetuned"]["ci"]; a.plot([100 * lo, 100 * hi], [i, i], color="black", lw=1.0, zorder=2)
a.set_yticks(y); a.set_yticklabels([SHORT[q] for q in items]); a.set_xlabel("Exact-match accuracy (%)"); a.set_title("(a) Accuracy by item (fold 0, 6,000 held-out respondents)", loc="left")
a.set_xlim(18, 70); a.grid(axis="x", lw=0.3, color="0.85"); a.legend(loc="center right", frameon=True, framealpha=0.95, edgecolor="0.7", handletextpad=0.2, borderpad=0.3)
b = ax[1]; h = 0.26
ex = json.load(open("analysis/pew_extra.json"))["concentration"]
tm = [100 * ex[q]["finetuned"]["true_top_share"] for q in items]
gb = [100 * ex[q]["gboost"]["top_pred_share"] for q in items]
pm = [100 * ex[q]["finetuned"]["top_pred_share"] for q in items]
b.barh(y + h, tm, height=h, color="0.85", edgecolor="black", linewidth=0.5, label="True answers")
b.barh(y, gb, height=h, color="0.55", edgecolor="black", linewidth=0.5, label="GBM predictions")
b.barh(y - h, pm, height=h, color="0.1", edgecolor="black", linewidth=0.5, label="Fine-tuned predictions")
b.set_yticks(y); b.set_yticklabels([]); b.set_xlim(0, 135); b.set_xticks([0, 20, 40, 60, 80, 100]); b.set_xlabel("Share of responses in the single most common class (%)")
b.set_title("(b) Concentration of answers and predictions", loc="left"); b.grid(axis="x", lw=0.3, color="0.85")
b.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="0.7", borderpad=0.3)
plt.tight_layout(w_pad=0.6); plt.savefig("figs/fig_pew_items.pdf"); plt.close()

# ---------- Fig 3: subgroup gaps (Pew)
axes_order = sorted(R["subgroups"], key=lambda k: -R["subgroups"][k]["methods"]["finetuned"]["gap"])
fig, a = plt.subplots(figsize=(3.45, 2.75)); y = np.arange(len(axes_order))
off = {"majority": 0.24, "logistic": 0.08, "gboost": -0.08, "finetuned": -0.24}
for m, (mk, fc, lab) in MK.items():
    xs = [100 * R["subgroups"][k]["methods"][m]["gap"] for k in axes_order]
    a.scatter(xs, y + off[m], marker=mk, s=17, facecolor=fc, edgecolor="black", linewidth=0.6, label=lab.replace("Fine-tuned Qwen2.5-7B", "Fine-tuned"), zorder=3)
for i, k in enumerate(axes_order):
    lo, hi = R["subgroups"][k]["methods"]["finetuned"]["gap_ci"]; a.plot([100 * lo, 100 * hi], [i + off["finetuned"]] * 2, color="black", lw=0.9, zorder=2)
a.set_yticks(y); a.set_yticklabels(axes_order); a.invert_yaxis(); a.set_xlabel("Fidelity gap: best $-$ worst category accuracy (pts)")
a.grid(axis="x", lw=0.3, color="0.85"); a.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.7", handletextpad=0.2, borderpad=0.3)
plt.tight_layout(); plt.savefig("figs/fig_pew_gaps.pdf"); plt.close()

# ---------- Fig 1: pipeline
fig, ax = plt.subplots(figsize=(7.1, 2.35)); ax.set_xlim(0, 100); ax.set_ylim(2, 36); ax.axis("off")
def box(x, y, w, h, txt, fc="white", bold=False, fs=6.6):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=fc, ec="black", lw=0.7); ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.15)
def arr(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="black"))
for x, lab in [(1, "DATA"), (24, "PREPARATION"), (51, "PREDICTORS"), (79, "EVALUATION")]:
    ax.text(x, 34.3, lab, fontsize=6.8, fontweight="bold", color="0.3")
box(1, 21, 19, 10, "WVS-7 India\n$N$ = 1,692\n144 screened items", "0.93"); box(1, 4.5, 19, 10, "Pew Religion in India\n$N$ = 29,999\n8 items (73 screened)", "0.93")
box(24, 21, 22, 10, "Codebook-driven item\nscreening; 5-fold\nrespondent-level split"); box(24, 4.5, 22, 10, "Codebook verbalization:\ndemographics + question\n+ options $\\rightarrow$ prompt (P2)")
box(51, 25.5, 24, 6.2, "Zero-shot LLMs (WVS-7; Pew control)\nGemini, gpt-oss-120b, Qwen2.5-7B", "0.93", fs=6.2)
box(51, 14.7, 24, 6.2, "Supervised baselines (both)\nmajority, cell mode, logistic, GBM", "0.93", fs=6.2)
box(51, 3.9, 24, 6.2, "QLoRA fine-tuned Qwen2.5-7B\n(Pew, fold 0)", "0.93", fs=6.2)
box(79, 17.5, 20, 13.5, "Identical-row scoring:\naccuracy, NLL, dispersion,\ngap inflation;\nclustered bootstrap CIs"); box(79, 3.9, 20, 10.6, "Subgroup fidelity gap\n+ permutation test", "0.85", bold=False)
arr(20.5, 26, 23.5, 26); arr(20.5, 9.5, 23.5, 9.5); arr(46.5, 26, 50.5, 28.5); arr(46.5, 26, 50.5, 18); arr(46.5, 9.5, 50.5, 7)
arr(46.5, 9.5, 50.5, 17); arr(75.5, 28.5, 78.5, 26); arr(75.5, 18, 78.5, 23); arr(75.5, 7, 78.5, 20); arr(89, 17, 89, 14.8)
plt.savefig("figs/fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.03); plt.close()
print("figs ok", serif)
