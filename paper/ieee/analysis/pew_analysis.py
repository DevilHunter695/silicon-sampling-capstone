"""Independent re-analysis of the fold-0 Pew run (Qwen2.5-7B-Instruct QLoRA).
Refits the three supervised baselines on fold 0 with the repo's own code, aligns them
row-for-row with the fine-tuned model's predictions, and computes respondent-clustered
bootstrap CIs, mode-collapse diagnostics, and subgroup gaps with permutation tests."""
import json, sys
import numpy as np, pandas as pd

CLONE = "/tmp/ssc_pew"
PRED = "/Users/krishank/Downloads/ishan-cpastone/pewB_Qwen_Qwen2.5-7B-Instruct_fold0_P2.parquet"
sys.path.insert(0, CLONE)
import os; os.chdir(CLONE); OUT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
from scripts.pewB_baselines import run_fold
from src.config import DATA_PROCESSED
from src.prompts.verbalize_pew import load_pew_codebook

rng = np.random.default_rng(2026)
cb = load_pew_codebook()
items = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"][:8]
folds = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"]
f0 = folds[0]
df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet")
df["respondent_id"] = df["respondent_id"].astype(int)

# ---- leakage / alignment checks
ft = pd.read_parquet(PRED)
test_ids, train_ids = set(f0["test"]), set(f0["train"])
chk = {"n_train_resp": len(train_ids), "n_test_resp": len(test_ids),
       "pred_resp": int(ft.respondent_id.nunique()),
       "pred_resp_in_test": bool(set(ft.respondent_id) <= test_ids),
       "pred_resp_overlap_train": len(set(ft.respondent_id) & train_ids),
       "n_rows": len(ft), "n_items": int(ft.question_id.nunique())}
print("CHECKS", chk, flush=True)

# ---- baselines (repo code, fold 0, identical features to the LLM prompt)
base = run_fold(df, f0, items, cb)
rows = []
for name, d in base.items():
    d = d.copy(); d["pred_idx"] = d["proba"].apply(lambda p: int(np.argmax(p))); d["method"] = name
    rows.append(d[["respondent_id", "question_id", "true_idx", "pred_idx", "method"]])
b = pd.concat(rows, ignore_index=True); b["respondent_id"] = b["respondent_id"].astype(int)
f = ft.rename(columns={"true_code_idx": "true_idx", "pred_code_idx": "pred_idx"})[["respondent_id", "question_id", "true_idx", "pred_idx"]].copy()
f["method"] = "finetuned"
allm = pd.concat([b, f], ignore_index=True)
# common rows
key = ["respondent_id", "question_id"]
common = f[key].merge(b[b.method == "majority"][key], on=key)
allm = allm.merge(common, on=key)
# alignment check on ground truth
tt = allm.pivot_table(index=key, columns="method", values="true_idx", aggfunc="first")
assert (tt.nunique(axis=1) == 1).all(), "true labels disagree across methods"
chk["n_common_rows"] = len(common); chk["n_baseline_rows"] = int((b.method == "majority").sum())
allm["correct"] = (allm["pred_idx"] == allm["true_idx"]).astype(float)           # NaN pred -> False -> counted wrong
allm["abs_err"] = (allm["pred_idx"] - allm["true_idx"]).abs()
M = ["majority", "logistic", "gboost", "finetuned"]

resp_ids = np.array(sorted(test_ids & set(allm.respondent_id)))
R = len(resp_ids); ridx = {r: i for i, r in enumerate(resp_ids)}
allm["ri"] = allm["respondent_id"].map(ridx)

def agg(method, mask=None, col="correct"):
    d = allm[allm.method == method] if mask is None else allm[(allm.method == method) & mask]
    s = np.bincount(d["ri"], weights=d[col].fillna(0), minlength=R); n = np.bincount(d["ri"], weights=d[col].notna().astype(float), minlength=R)
    return s, n

B = 4000
W = rng.multinomial(R, np.ones(R) / R, size=B).astype(float)         # bootstrap respondent weights
def boot_ratio(s, n):
    return (W @ s) / (W @ n)

res = {"checks": chk, "overall": {}, "paired": {}, "per_item": {}}
cache = {}
for m in M:
    s, n = agg(m); cache[m] = (s, n)
    est = s.sum() / n.sum(); ci = np.percentile(boot_ratio(s, n), [2.5, 97.5])
    sm, nm = agg(m, mask=allm["pred_idx"].notna(), col="abs_err")
    res["overall"][m] = {"accuracy": est, "ci": ci.tolist(), "mae": sm.sum() / nm.sum(),
                         "mae_ci": np.percentile(boot_ratio(sm, nm), [2.5, 97.5]).tolist(), "n_rows": int(n.sum())}
for a, bm in [("finetuned", "majority"), ("finetuned", "logistic"), ("finetuned", "gboost"), ("gboost", "majority"), ("logistic", "majority")]:
    sa, na = cache[a]; sb, nb = cache[bm]
    d = boot_ratio(sa, na) - boot_ratio(sb, nb)
    res["paired"][f"{a}-{bm}"] = {"diff": sa.sum() / na.sum() - sb.sum() / nb.sum(), "ci": np.percentile(d, [2.5, 97.5]).tolist()}

# ---- per item
for q in items:
    mask = allm["question_id"] == q
    r = {}
    for m in M:
        s, n = agg(m, mask=mask); r[m] = {"acc": s.sum() / n.sum(), "ci": np.percentile(boot_ratio(s, n), [2.5, 97.5]).tolist()}
    d = allm[(allm.method == "finetuned") & mask]
    vc = d["pred_idx"].value_counts(normalize=True)
    truth = d["true_idx"].value_counts(normalize=True)
    nopt = len(cb[q]["valid_codes"])
    pm = np.zeros(nopt); tm = np.zeros(nopt)
    for k, v in vc.items(): pm[int(k)] = v
    for k, v in truth.items(): tm[int(k)] = v
    r["ft_top_pred_share"] = float(vc.max()); r["ft_n_distinct_pred"] = int(len(vc)); r["ft_n_classes_ge5pct"] = int((vc >= 0.05).sum())
    r["true_modal_share"] = float(truth.max()); r["n_options"] = nopt
    r["ft_tv_marginal"] = float(0.5 * np.abs(pm - tm).sum())
    r["ft_top_pred_equals_true_mode"] = bool(vc.idxmax() == truth.idxmax())
    r["n_rows"] = int(len(d)); r["title"] = cb[q]["title"][:80]
    res["per_item"][q] = r

# ---- does FT ever deviate from the majority class? agreement with the majority prediction
mj = allm[allm.method == "majority"].set_index(key)["pred_idx"]
fd = allm[allm.method == "finetuned"].set_index(key)
fd["maj_pred"] = mj.reindex(fd.index).values
res["ft_agrees_with_majority_pred"] = float((fd["pred_idx"] == fd["maj_pred"]).mean())
dev = fd[fd["pred_idx"] != fd["maj_pred"]]
gb = allm[allm.method == "gboost"].set_index(key)["correct"].reindex(dev.index)
mjc = allm[allm.method == "majority"].set_index(key)["correct"].reindex(dev.index)
res["deviation_rows"] = {"n": int(len(dev)), "share": float(len(dev) / len(fd)), "ft_acc": float(dev["correct"].mean()),
                         "gboost_acc_same_rows": float(gb.mean()), "majority_acc_same_rows": float(mjc.mean())}
# within-item respondent-level variation of FT prediction vs demographics: does prediction depend on respondent at all?
res["ft_items_with_constant_prediction"] = [q for q, r in res["per_item"].items() if r["ft_n_distinct_pred"] == 1]

# ---- subgroup analysis
dem = df[["respondent_id", "QGEN", "Urban", "QRELSING", "REGION", "QCASTE", "QINCINDrec", "QAGErec", "QEDU"]].copy()
def lab(col): return {int(k): v for k, v in cb[col]["valid_codes"].items()}
edu_map = {1: "Illiterate/none", 2: "Illiterate/none", 3: "Primary", 4: "Primary", 5: "Secondary", 6: "Secondary", 7: "Secondary", 8: "Secondary", 9: "Degree+", 10: "Degree+", 11: "Degree+"}
axes = {"Sex": dem["QGEN"].map(lab("QGEN")), "Urban/rural": dem["Urban"].map(lab("Urban")),
        "Religion": dem["QRELSING"].map(lab("QRELSING")), "Region": dem["REGION"].map(lab("REGION")),
        "Caste": dem["QCASTE"].map(lab("QCASTE")), "Income": dem["QINCINDrec"].map(lab("QINCINDrec")),
        "Age": dem["QAGErec"].map(lab("QAGErec")), "Education": dem["QEDU"].map(edu_map)}
axes_df = pd.DataFrame(axes); axes_df["respondent_id"] = dem["respondent_id"].values
axes_df = axes_df.set_index("respondent_id").reindex(resp_ids)
MIN_RESP = 100
sub = {}
for ax in axes:
    lbl = axes_df[ax].to_numpy()
    cats = [c for c in pd.Series(lbl).dropna().unique() if (lbl == c).sum() >= MIN_RESP]
    if len(cats) < 2: continue
    cat_idx = {c: np.where(lbl == c)[0] for c in cats}
    sub[ax] = {"cats": {str(c): int(len(i)) for c, i in cat_idx.items()}, "methods": {}}
    for m in M:
        s, n = cache[m]
        accs = {str(c): float(s[i].sum() / n[i].sum()) for c, i in cat_idx.items()}
        def gap_from_weights(w):
            a = [(w[i] @ s[i]) / (w[i] @ n[i]) for i in cat_idx.values()]; return max(a) - min(a)
        obs = gap_from_weights(np.ones(R))
        bg = np.array([gap_from_weights(W[b_]) for b_ in range(0, B, 4)])
        # permutation: shuffle category labels among respondents in powered categories
        in_pow = np.concatenate(list(cat_idx.values())); labs_pow = np.concatenate([[k] * len(i) for k, i in enumerate(cat_idx.values())])
        cnt = 0; nul = []
        for _ in range(2000):
            p = rng.permutation(labs_pow)
            a = [s[in_pow[p == k]].sum() / n[in_pow[p == k]].sum() for k in range(len(cat_idx))]
            g = max(a) - min(a); nul.append(g); cnt += g >= obs
        sub[ax]["methods"][m] = {"acc_by_cat": accs, "gap": float(obs), "gap_ci": np.percentile(bg, [2.5, 97.5]).tolist(),
                                 "perm_p": float((cnt + 1) / 2001), "null_median": float(np.median(nul))}
    # paired difference in gap, FT vs gboost
    dg = []
    for b_ in range(0, B, 4):
        w = W[b_]
        gg = {}
        for m in ("finetuned", "gboost", "majority"):
            s, n = cache[m]; a = [(w[i] @ s[i]) / (w[i] @ n[i]) for i in cat_idx.values()]; gg[m] = max(a) - min(a)
        dg.append((gg["finetuned"] - gg["gboost"], gg["finetuned"] - gg["majority"]))
    dg = np.array(dg)
    sub[ax]["gap_diff_ft_minus_gboost"] = {"diff": sub[ax]["methods"]["finetuned"]["gap"] - sub[ax]["methods"]["gboost"]["gap"], "ci": np.percentile(dg[:, 0], [2.5, 97.5]).tolist()}
    sub[ax]["gap_diff_ft_minus_majority"] = {"diff": sub[ax]["methods"]["finetuned"]["gap"] - sub[ax]["methods"]["majority"]["gap"], "ci": np.percentile(dg[:, 1], [2.5, 97.5]).tolist()}
    print(ax, {m: round(sub[ax]["methods"][m]["gap"], 3) for m in M}, flush=True)
res["subgroups"] = sub

# training-loss curve for figure
steps = [json.loads(l) for l in open("/Users/krishank/Downloads/ishan-cpastone/train_progress.jsonl")]
steps = [s for s in steps if "loss" in s]
res["train"] = {"n_log_points": len(steps), "loss_first": steps[0]["loss"], "loss_last": steps[-1]["loss"], "step_last": steps[-1]["step"],
                "loss_mean_after_100": float(np.mean([s["loss"] for s in steps if s["step"] >= 100]))}
pd.DataFrame(steps)[["step", "loss", "learning_rate"]].to_csv("train_loss.csv", index=False)
allm.to_parquet("pew_aligned_rows.parquet")
json.dump(res, open("pew_results.json", "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ("checks", "overall", "paired", "ft_agrees_with_majority_pred", "deviation_rows", "ft_items_with_constant_prediction", "train")}, indent=1, default=float))
