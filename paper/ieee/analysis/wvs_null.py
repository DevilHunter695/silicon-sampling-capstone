import json, numpy as np, pandas as pd
REPO = "/Users/krishank/Documents/College Projects/🧢 /silicon-sampling-capstone/results/predictions/"
FILES = {"Gemini": "gemini-3.1-flash-lite_P2", "gpt-oss": "groq-openai_gpt-oss-120b-low_P2"}
AX = ["sg_region", "sg_age_band", "sg_education", "sg_income", "sg_sex", "sg_urban_rural"]
rng = np.random.default_rng(5); out = {}
for name, f in FILES.items():
    d = pd.read_parquet(REPO + f + ".parquet"); d = d[d.pred_code_idx.notna()].copy()
    d["c"] = (d.true_code_idx == d.pred_code_idx).astype(float)
    g = d.groupby("respondent_id").agg(c=("c", "sum"), n=("c", "size"))
    out[name] = {}
    for ax in AX:
        lab = d.groupby("respondent_id")[ax].first().reindex(g.index)
        keep = lab.notna().to_numpy()
        code, cats = pd.factorize(lab[keep].astype(str)); K = len(cats)
        gc, gn = g.c.to_numpy()[keep], g.n.to_numpy()[keep]
        def gap(code_):
            s = np.bincount(code_, weights=gc, minlength=K); n = np.bincount(code_, weights=gn, minlength=K)
            ok = n >= 30
            if ok.sum() < 2: return np.nan
            a = s[ok] / n[ok]; return a.max() - a.min()
        obs = gap(code); null = np.array([gap(rng.permutation(code)) for _ in range(5000)]); null = null[~np.isnan(null)]
        out[name][ax] = {"obs": float(obs), "null_median": float(np.median(null)), "null_p95": float(np.percentile(null, 95)),
                         "p": float((1 + (null >= obs).sum()) / (1 + len(null)))}
json.dump(out, open("analysis/wvs_null.json", "w"), indent=1)
for n, v in out.items():
    for ax, r in v.items(): print(n, ax, {k: round(100 * x, 1) if k != "p" else round(x, 3) for k, x in r.items()})
