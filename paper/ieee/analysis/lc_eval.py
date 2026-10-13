"""Data-efficiency curve: small LLM (LoRA, answer-only loss) vs classical models on nested respondent subsets; fixed 200 test respondents."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = os.path.dirname(os.path.abspath(__file__)); key = ["respondent_id", "question_id"]; rng = np.random.default_rng(321)
TAG = os.environ.get("LC_TAG", "q05")
res = {}
for N in [0, 25, 100, 400, 1600]:
    p = f"{HERE}/lc_{TAG}_N{N}.jsonl"
    if not os.path.exists(p): continue
    d = pd.read_json(p, lines=True).drop_duplicates(key)
    if len(d) < 1000: continue
    ids = np.sort(d.respondent_id.unique()); ri = {r: i for i, r in enumerate(ids)}; R = len(ids)
    d["ri"] = d.respondent_id.map(ri)
    c = (d.pred_idx == d.true_idx).astype(float); nll = [-np.log(max(p_[int(t)], 1e-6)) for p_, t in zip(d.probs, d.true_idx)]
    s = np.bincount(d.ri, weights=c, minlength=R); n = np.bincount(d.ri, minlength=R).astype(float); sn = np.bincount(d.ri, weights=np.array(nll), minlength=R)
    W = rng.multinomial(R, np.ones(R) / R, size=2000).astype(float)
    res[N] = {"acc": float(s.sum() / n.sum()), "ci": np.percentile((W @ s) / (W @ n), [2.5, 97.5]).tolist(), "nll": float(sn.sum() / n.sum()),
              "nll_ci": np.percentile((W @ sn) / (W @ n), [2.5, 97.5]).tolist(), "n_rows": int(len(d))}
    print(N, res[N]["n_rows"], round(100 * res[N]["acc"], 1), [round(100 * x, 1) for x in res[N]["ci"]], round(res[N]["nll"], 3))
json.dump(res, open(f"{HERE}/lc_eval.json", "w"), indent=1)
lcb = json.load(open(f"{HERE}/lc_baselines.json"))
names = {f.name for f in fm.fontManager.ttflist}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman" if "Times New Roman" in names else "DejaVu Serif"], "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.4))
Ns = [25, 100, 400, 1600, 23999]
for k, (m, lab, mk, col) in enumerate([("majority", "Majority class", "o", "0.6"), ("logistic", "Logistic regression", "^", "0.35"), ("gboost", "GBM", "s", "0.0")]):
    xs = [N for N in Ns if str(N) in lcb]
    ax[0].errorbar(xs, [100 * lcb[str(N)][m]["acc_mean_over_subsets"] for N in xs], yerr=[100 * lcb[str(N)][m]["acc_sd_over_subsets"] for N in xs], marker=mk, color=col, lw=1, ms=4, capsize=1.5, label=lab)
    ax[1].plot(xs, [lcb[str(N)][m]["nll_mean_over_subsets"] for N in xs], marker=mk, color=col, lw=1, ms=4)
xs = [N for N in [25, 100, 400, 1600] if N in res]
if xs:
    ax[0].errorbar(xs, [100 * res[N]["acc"] for N in xs], yerr=[[100 * (res[N]["acc"] - res[N]["ci"][0]) for N in xs], [100 * (res[N]["ci"][1] - res[N]["acc"]) for N in xs]], marker="D", color="black", mfc="white", lw=1.2, ms=4.5, capsize=1.5, label="Qwen2.5-0.5B, LoRA")
    ax[1].plot(xs, [res[N]["nll"] for N in xs], marker="D", color="black", mfc="white", lw=1.2, ms=4.5)
if 0 in res:
    ax[0].axhline(100 * res[0]["acc"], color="black", ls=":", lw=0.8); ax[0].text(26, 100 * res[0]["acc"] + 0.6, "0.5B zero-shot", fontsize=6.3)
    ax[1].axhline(res[0]["nll"], color="black", ls=":", lw=0.8)
for a in ax: a.set_xscale("log"); a.set_xlabel("Training respondents")
ax[0].set_ylabel("Exact-match accuracy (%)"); ax[1].set_ylabel("NLL"); ax[0].legend(fontsize=6.3, frameon=False, loc="lower right")
ax[0].set_title("(a) Accuracy", loc="left", fontsize=7.5); ax[1].set_title("(b) Negative log-likelihood", loc="left", fontsize=7.5)
ax[1].set_ylim(1.3, 2.2)
plt.tight_layout(); plt.savefig(f"{HERE}/../figs/fig_lc.pdf"); print("fig ok")
