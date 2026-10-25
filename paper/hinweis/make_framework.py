import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"] = "serif"; plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
fig, ax = plt.subplots(figsize=(6.3, 2.35), dpi=300); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, text, fc="white", fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec="black", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="black"))
for x, t in ((1, "1. DATA"), (24, "2. PROMPT"), (47, "3. PREDICTORS"), (72, "4. READOUT, DIAGNOSTICS")):
    ax.text(x, 38.3, t, fontsize=6.2, fontweight="bold", va="center")
box(1, 25, 19, 10, "Pew Religion in India\nN = 29,999; 73 screened\nitems; fold 0:\n23,999 train / 6,000 test", "0.92")
box(1, 5, 19, 14, "WVS-7 India\nN = 1,692; 144 screened\nitems; 15 used for\nzero-shot replication", "0.92")
box(24, 15, 19, 20, "Profile verbalised from\ncodebook labels\n+ question text\n+ answer options\n= prompt P2\n(no-profile prompt P0)", "white")
box(47, 29, 21, 7, "Zero-shot Qwen2.5-7B\n(chat, raw, no profile)", "0.92")
box(47, 20, 21, 7, "QLoRA fine-tuned\nQwen2.5-7B (8 items)", "0.92")
box(47, 11.5, 21, 6.5, "Larger zero-shot LLMs\n(gpt-oss-120b, Gemini)", "0.92")
box(47, 2.5, 21, 7, "Matched baselines\n(majority, logistic, GBM)", "0.92")
box(72, 21, 27, 14, "Next-token probabilities over\nthe option digits; argmax =\nprediction. Accuracy, NLL, ECE,\nvariance share, direction, JSD", "white")
box(72, 3, 27, 15, "Tests: subgroup gaps (Holm),\nheld-out-item transfer,\nlabelled-data requirement,\ncompute and scaling", "0.82", bold=False)
arrow(20, 30, 24, 27); arrow(20, 12, 24, 20)
arrow(43, 28, 47, 32); arrow(43, 25, 47, 23.5); arrow(43, 22, 47, 14.5); arrow(43, 18, 47, 6)
arrow(68, 32, 72, 31); arrow(68, 23.5, 72, 26); arrow(68, 14.5, 72, 14); arrow(68, 6, 72, 8)
arrow(85.5, 21, 85.5, 18.2)
fig.savefig("fig_framework.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
print("ok")
