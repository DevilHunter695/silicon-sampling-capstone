import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = os.path.dirname(os.path.abspath(__file__))
names = {f.name for f in fm.fontManager.ttflist}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman" if "Times New Roman" in names else "DejaVu Serif"], "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
cb = json.load(open("/tmp/ssc_pew/data/reference/pew_codebook.json"))["questions"]
K = {q: len(cb[q]["valid_codes"]) for q in cb if "valid_codes" in cb[q]}
rows = pd.read_parquet(f"{HERE}/pew_aligned_rows.parquet"); rows = rows[rows.pred_idx.notna() & (rows.question_id != "Q28rec")]
z = pd.read_json(f"{HERE}/mlx_base_chat.jsonl", lines=True).drop_duplicates(["respondent_id", "question_id"]); z = z[z.question_id != "Q28rec"].copy()
z["soft"] = [float(np.dot(p, np.arange(len(p))) / (len(p) - 1)) for p in z.probs]; z["true"] = z.true_idx / (z.probs.map(len) - 1)
sub = z[["respondent_id", "question_id", "soft", "true"]].copy()
for m in ["finetuned", "gboost"]:
    h = rows[rows.method == m][["respondent_id", "question_id", "pred_idx"]].rename(columns={"pred_idx": m})
    sub = sub.merge(h, on=["respondent_id", "question_id"]); sub[m] = sub[m] / sub.question_id.map(lambda q: K[q] - 1)
df = pd.read_parquet("/tmp/ssc_pew/data/processed/pew_india_2021.parquet"); df["respondent_id"] = df.respondent_id.astype(int); df = df.set_index("respondent_id")
lab = lambda c: {int(k): v for k, v in cb[c]["valid_codes"].items()}
edu = {1: "a", 2: "a", 3: "b", 4: "b", 5: "c", 6: "c", 7: "c", 8: "c", 9: "d", 10: "d", 11: "d"}
axes = {"Sex": df.QGEN.map(lab("QGEN")), "Urban": df.Urban.map(lab("Urban")), "Religion": df.QRELSING.map(lab("QRELSING")), "Region": df.REGION.map(lab("REGION")),
        "Caste": df.QCASTE.map(lab("QCASTE")), "Age": df.QAGErec.map(lab("QAGErec")), "Income": df.QINCINDrec.map(lab("QINCINDrec")), "Education": df.QEDU.map(edu)}
cells = {c: [] for c in ["x", "zs", "ft", "gb"]}
for q, g in sub.groupby("question_id"):
    mu = {c: g[c].mean() for c in ["true", "soft", "finetuned", "gboost"]}
    for ax, ser in axes.items():
        lbl = ser.reindex(g.respondent_id).to_numpy()
        for cat in pd.unique(lbl[~pd.isna(lbl)]):
            m = lbl == cat
            if len(set(g.respondent_id.to_numpy()[m])) < 15: continue
            cells["x"].append(g.true.to_numpy()[m].mean() - mu["true"]); cells["zs"].append(g.soft.to_numpy()[m].mean() - mu["soft"])
            cells["ft"].append(g.finetuned.to_numpy()[m].mean() - mu["finetuned"]); cells["gb"].append(g.gboost.to_numpy()[m].mean() - mu["gboost"])
x = np.array(cells["x"])
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.35), sharex=True, sharey=True)
for a, key, title in zip(axs, ["zs", "ft", "gb"], ["Zero-shot Qwen2.5-7B", "Fine-tuned Qwen2.5-7B", "Gradient boosting"]):
    y = np.array(cells[key]); r = np.corrcoef(x, y)[0, 1]; b = np.polyfit(x, y, 1)
    a.scatter(x, y, s=9, color="0.15", alpha=0.65, linewidths=0)
    xx = np.linspace(x.min(), x.max(), 20); a.plot(xx, b[0] * xx + b[1], color="black", lw=0.9)
    a.plot(xx, xx, color="0.6", lw=0.7, ls="--")
    a.axhline(0, color="0.85", lw=0.5); a.axvline(0, color="0.85", lw=0.5)
    a.set_title(f"{title}\n$r$={r:.2f}, slope={b[0]:.2f}", fontsize=7.5); a.set_xlabel("True group difference")
axs[0].set_ylabel("Predicted group difference")
plt.tight_layout(); os.makedirs(f"{HERE}/../figs", exist_ok=True); plt.savefig(f"{HERE}/../figs/fig_direction.pdf"); print("cells", len(x))
