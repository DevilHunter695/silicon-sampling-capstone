import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = os.path.dirname(os.path.abspath(__file__))
names = {f.name for f in fm.fontManager.ttflist}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman" if "Times New Roman" in names else "DejaVu Serif"], "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
A = json.load(open(f"{HERE}/adapter_eval.json"))
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
# (a) accuracy: trained items vs unseen items
rows = [("Chance (1/K)", "uniform_chance", "0.9"), ("Zero-shot Qwen (raw)", "base_raw", "0.6"), ("Fine-tuned Qwen (MLX port)", "adapter_raw", "0.15"),
        ("Majority (with labels)", "majority", "0.8"), ("GBM (with labels)", "gboost", "0.4")]
w = 0.36
for j, (tag, title) in enumerate([("trained", "8 trained items"), ("unseen", "8 unseen items")]):
    r = A[tag]["methods"]; x = np.arange(len(rows)) + (j - 0.5) * w
    ax[0].bar(x, [100 * r[k]["acc"] for _, k, _ in rows], w, color=["0.9", "0.6", "0.15", "0.8", "0.4"] if j == 0 else ["0.95", "0.75", "0.35", "0.9", "0.55"],
              edgecolor="black", lw=0.5, hatch="" if j == 0 else "///", label=title)
    ax[0].errorbar(x, [100 * r[k]["acc"] for _, k, _ in rows], yerr=[[100 * (r[k]["acc"] - r[k]["ci"][0]) for _, k, _ in rows], [100 * (r[k]["ci"][1] - r[k]["acc"]) for _, k, _ in rows]],
                   fmt="none", ecolor="black", lw=0.7, capsize=1.5)
ax[0].set_xticks(np.arange(len(rows))); ax[0].set_xticklabels([n for n, _, _ in rows], rotation=22, ha="right", fontsize=6.6)
ax[0].set_ylabel("Exact-match accuracy (%)"); ax[0].set_title("(a) Accuracy: trained vs. unseen items", loc="left", fontsize=7.5)
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(fc="0.6", ec="black", label="trained items"), Patch(fc="0.75", ec="black", hatch="///", label="unseen items")], fontsize=6.5, frameon=False, loc="upper left")
ax[0].set_ylim(0, 60)
# (b) gain of adapter over base, per item set, with paired CI
g = [A["trained"]["paired"]["adapter_raw-base_raw"], A["unseen"]["paired"]["adapter_raw-base_raw"]]
xs = [0, 1]
ax[1].bar(xs, [100 * v["diff"] for v in g], 0.5, color=["0.3", "0.7"], edgecolor="black", lw=0.5)
ax[1].errorbar(xs, [100 * v["diff"] for v in g], yerr=[[100 * (v["diff"] - v["ci"][0]) for v in g], [100 * (v["ci"][1] - v["diff"]) for v in g]], fmt="none", ecolor="black", lw=0.8, capsize=2)
ax[1].axhline(0, color="black", lw=0.6)
ax[1].set_xticks(xs); ax[1].set_xticklabels(["trained items", "unseen items"]); ax[1].set_ylabel("Fine-tuned minus zero-shot (points)")
ax[1].set_title("(b) Gain from fine-tuning", loc="left", fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{HERE}/../figs/fig_transfer.pdf"); print("ok")
