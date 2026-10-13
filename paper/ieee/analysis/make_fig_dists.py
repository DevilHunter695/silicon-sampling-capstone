import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = os.path.dirname(os.path.abspath(__file__))
names = {f.name for f in fm.fontManager.ttflist}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman" if "Times New Roman" in names else "DejaVu Serif"],
                     "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
z = pd.read_json(f"{HERE}/mlx_base_chat.jsonl", lines=True).drop_duplicates(["respondent_id", "question_id"])
o = pd.read_parquet(f"{HERE}/pew_aligned_rows.parquet")
key = ["respondent_id", "question_id"]
ft = o[o.method == "finetuned"][key + ["pred_idx"]].rename(columns={"pred_idx": "ft"})
gb = o[o.method == "gboost"][key + ["pred_idx"]].rename(columns={"pred_idx": "gb"})
d = z.merge(ft, on=key).merge(gb, on=key)
cb = json.load(open("/tmp/ssc_pew/data/reference/pew_codebook.json"))["questions"]
short = {"Q12": "Politicians' role in religion", "Q28rec": "Religious identity", "Q3132rec": "Attends worship", "QFERTrec": "Number of children", "Q50": "Reads scripture", "ABORT": "Abortion legality"}
items = ["Q12", "Q28rec", "Q3132rec", "QFERTrec"]
fig, axs = plt.subplots(1, 4, figsize=(7.1, 2.05), sharey=True)
shade = {"true": "0.82", "zs": "0.55", "ft": "0.12"}
for ax, q in zip(axs, items):
    g = d[d.question_id == q]; k = len(g.probs.iloc[0]); x = np.arange(k); w = 0.27
    tr = np.bincount(g.true_idx, minlength=k) / len(g); zs = np.bincount(g.pred_idx, minlength=k) / len(g); f_ = np.bincount(g.ft.astype(int), minlength=k) / len(g)
    ax.bar(x - w, 100 * tr, w, color=shade["true"], edgecolor="black", lw=0.4, label="True answers")
    ax.bar(x, 100 * zs, w, color=shade["zs"], edgecolor="black", lw=0.4, label="Zero-shot Qwen")
    ax.bar(x + w, 100 * f_, w, color=shade["ft"], edgecolor="black", lw=0.4, label="Fine-tuned Qwen")
    ax.set_title(short[q], fontsize=7.5); ax.set_xticks(x); ax.set_xticklabels([str(i + 1) for i in x], fontsize=6.5); ax.set_xlabel("Answer option", fontsize=7)
    ax.grid(axis="y", lw=0.3, color="0.88")
axs[0].set_ylabel("Share of respondents (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.02))
plt.tight_layout(w_pad=0.4, rect=(0, 0, 1, 0.9))
os.makedirs(f"{HERE}/../figs", exist_ok=True)
plt.savefig(f"{HERE}/../figs/fig_pew_dists.pdf"); print("ok", len(d))
