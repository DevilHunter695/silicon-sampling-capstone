import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, json
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
NAVY, ORG, GRY, LGT = "#001d3e", "#ffb020", "#8a94a8", "#c9d3ea"
def bars(path, labels, vals, lo, hi, colors, ylab="Accuracy (%)", ref=None, size=(6.4, 3.6)):
    fig, ax = plt.subplots(figsize=size, dpi=200)
    x = range(len(labels))
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5, yerr=[[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]], capsize=3, error_kw={"lw": 1})
    for i, v in enumerate(vals): ax.text(i, hi[i] + 0.8, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9); ax.set_ylabel(ylab); ax.set_ylim(0, max(hi) + 8)
    if ref: ax.axhline(ref[0], ls="--", color="black", lw=0.8); ax.text(len(labels) - 0.5, ref[0] + 0.6, ref[1], ha="right", fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, facecolor="white"); plt.close(fig)
bars("c_subset.png", ["Majority\nclass", "Zero-shot\n(profile)", "Zero-shot\n(no profile)", "Zero-shot\nprior-corrected", "Fine-tuned\n7B", "Gradient\nboosting"],
     [30.2, 17.5, 21.1, 26.3, 37.7, 40.4], [28.1, 15.7, 19.1, 24.0, 35.1, 37.3], [32.4, 19.4, 23.1, 28.6, 40.3, 43.4],
     [GRY, ORG, ORG, ORG, NAVY, GRY], ref=(19.4, "chance 19.4%"))
bars("c_fold0.png", ["Majority", "Logistic", "Gradient\nboosting", "Fine-tuned\n7B", "Zero-shot\n(raw)", "Zero-shot\n(chat)"],
     [30.4, 38.5, 40.0, 38.0, 18.0, 16.1], [29.9, 38.0, 39.6, 37.5, 17.6, 15.7], [30.8, 39.0, 40.5, 38.4, 18.4, 16.4],
     [GRY, GRY, GRY, NAVY, ORG, ORG], ref=(19.4, "chance 19.4%"))
d = json.load(open("demo_row.json"))
fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
opts = ["1. A large\ninfluence", "2. Some\ninfluence", "3. Not too much\ninfluence", "4. No influence\nat all"]
w = 0.26
for k, (key, lab, col) in enumerate((("zs", "Zero-shot Qwen2.5-7B", ORG), ("ft", "Fine-tuned Qwen2.5-7B", NAVY), ("gbm", "Gradient boosting", GRY))):
    ax.bar([i + (k - 1) * w for i in range(4)], d[key], w, label=lab, color=col, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(4)); ax.set_xticklabels(opts, fontsize=8.5); ax.set_ylabel("Predicted probability"); ax.set_ylim(0, 0.95)
ax.annotate("TRUE ANSWER", xy=(0, 0.55), xytext=(0, 0.72), ha="center", fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=1.3))
ax.legend(fontsize=8.5, frameon=False, loc="upper right"); fig.tight_layout(); fig.savefig("c_demo.png", facecolor="white"); plt.close(fig)
print("charts ok")
