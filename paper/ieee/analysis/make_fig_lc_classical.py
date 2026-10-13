import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = os.path.dirname(os.path.abspath(__file__))
names = {f.name for f in fm.fontManager.ttflist}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman" if "Times New Roman" in names else "DejaVu Serif"], "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
lcb = json.load(open(f"{HERE}/lc_baselines.json")); Z = json.load(open(f"{HERE}/zs_eval.json"))["chat"]["methods"]
Ns = [25, 100, 400, 1600, 23999]
fig, ax = plt.subplots(1, 2, figsize=(3.45, 2.2))
for m, lab, mk, col in [("majority", "Majority", "o", "0.6"), ("logistic", "Logistic", "^", "0.35"), ("gboost", "GBM", "s", "0.0")]:
    ax[0].errorbar(Ns, [100 * lcb[str(N)][m]["acc_mean_over_subsets"] for N in Ns], yerr=[100 * lcb[str(N)][m]["acc_sd_over_subsets"] for N in Ns], marker=mk, color=col, lw=1, ms=3.5, capsize=1.2, label=lab)
    ax[1].plot(Ns, [lcb[str(N)][m]["nll_mean_over_subsets"] for N in Ns], marker=mk, color=col, lw=1, ms=3.5)
ax[0].scatter([23999], [100 * Z["finetuned"]["acc"]], marker="D", s=22, color="black", zorder=5, label="Fine-tuned 7B")
ax[0].axhline(100 * Z["zs_base_chat"]["acc"], color="black", ls=":", lw=0.8); ax[0].text(28, 100 * Z["zs_base_chat"]["acc"] + 0.7, "zero-shot 7B", fontsize=6)
for a in ax: a.set_xscale("log"); a.set_xlabel("Training respondents")
ax[0].set_ylabel("Accuracy (%)"); ax[1].set_ylabel("NLL"); ax[0].legend(fontsize=5.8, frameon=False, loc="lower right", handlelength=1.2)
ax[1].set_ylim(1.35, 2.1); ax[0].set_ylim(14, 42)
ax[0].set_title("(a)", loc="left", fontsize=7.5); ax[1].set_title("(b)", loc="left", fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{HERE}/../figs/fig_lc_classical.pdf"); print("ok")
