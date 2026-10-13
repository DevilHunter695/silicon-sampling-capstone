"""Re-verify WVS-India zero-shot numbers from the saved prediction files and test
whether subgroup fidelity gaps exceed what random group assignment produces."""
import json, sys
import numpy as np, pandas as pd

REPO = "/Users/krishank/Documents/College Projects/🧢 /silicon-sampling-capstone"
FILES = {"Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite_P2", "gpt-oss-120b": "groq-openai_gpt-oss-120b-low_P2"}
AXES = ["sg_region", "sg_age_band", "sg_education", "sg_income", "sg_sex", "sg_urban_rural"]
MIN_ROWS = 30
rng = np.random.default_rng(12345)

def cluster_boot_ci(df, stat, n=2000):
    ids = df["respondent_id"].unique(); groups = {r: g for r, g in df.groupby("respondent_id")}
    vals = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids), replace=True)
        vals.append(stat(pd.concat([groups[r] for r in pick])))
    return np.percentile(vals, [2.5, 97.5])

def gap(df, axis):
    g = df.groupby(axis, observed=True)["correct"].agg(["mean", "size"])
    g = g[g["size"] >= MIN_ROWS]
    return (g["mean"].max() - g["mean"].min()) if len(g) >= 2 else np.nan

def perm_p(df, axis, n=5000):
    obs = gap(df, axis)
    resp = df.groupby("respondent_id")[axis].first()
    ids = resp.index.to_numpy(); labels = resp.to_numpy()
    cnt = 0; valid = 0; null = []
    for _ in range(n):
        shuf = dict(zip(ids, rng.permutation(labels)))
        d = df.assign(**{axis: df["respondent_id"].map(shuf)})
        gg = gap(d, axis)
        if not np.isnan(gg):
            valid += 1; null.append(gg); cnt += gg >= obs
    return obs, (cnt + 1) / (valid + 1), float(np.median(null))

out = {}
for name, f in FILES.items():
    d = pd.read_parquet(f"{REPO}/results/predictions/{f}.parquet")
    d["correct"] = (d["true_code_idx"] == d["pred_code_idx"]).astype(float)
    ans = d[d["pred_code_idx"].notna()].copy()
    ans["abs_err"] = (ans["true_code_idx"] - ans["pred_code_idx"]).abs()
    res = {"n_rows": len(d), "n_respondents": int(d.respondent_id.nunique()),
           "accuracy": ans["correct"].mean(), "mae": ans["abs_err"].mean(),
           "refusal_rate": float(d["refusal"].mean())}
    lo, hi = cluster_boot_ci(ans, lambda x: x["correct"].mean())
    res["acc_ci_cluster"] = [lo, hi]
    # same-sample baselines: mean over the 15 items (per-item baselines were fit on all 1692 respondents, out-of-fold)
    bl = pd.read_csv(f"{REPO}/results/evaluated/{f}/versus_baselines.csv")
    res["same_items_baselines"] = {c: float(bl[c].mean()) for c in bl.columns if c not in ("question_id",)}
    res["resp_per_category"] = {}
    gaps = {}
    for ax in AXES:
        obs, p, nullmed = perm_p(ans, ax)
        cats = ans.groupby(ax, observed=True)["respondent_id"].nunique()
        tab = ans.groupby(ax, observed=True)["correct"].agg(["mean", "size"])
        powered = tab[tab["size"] >= MIN_ROWS]
        gaps[ax] = {"gap": float(obs), "perm_p": float(p), "null_median_gap": nullmed,
                    "n_cats_powered": int(len(powered)),
                    "min_resp_in_powered_cat": int(cats[powered.index].min()),
                    "max_resp_in_powered_cat": int(cats[powered.index].max())}
        print(name, ax, gaps[ax], flush=True)
    res["gaps"] = gaps
    out[name] = res
json.dump(out, open("wvs_results.json", "w"), indent=1, default=float)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "gaps"} for k, v in out.items()}, indent=1, default=float))
