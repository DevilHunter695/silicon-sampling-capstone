import json, numpy as np, pandas as pd
REPO = "/Users/krishank/Documents/College Projects/🧢 /silicon-sampling-capstone/results/predictions/"
rng = np.random.default_rng(7)
g = pd.read_parquet(REPO + "gemini-3.1-flash-lite_P2.parquet"); q = pd.read_parquet(REPO + "groq-openai_gpt-oss-120b-low_P2.parquet")
def kappa(a, b):
    cats = np.union1d(a, b); n = len(a)
    po = (a == b).mean(); pe = sum(((a == c).mean()) * ((b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe)
m = g.merge(q, on=["respondent_id", "question_id"], suffixes=("_g", "_q"))
m = m[m.pred_code_idx_g.notna() & m.pred_code_idx_q.notna()]
a, b, t = m.pred_code_idx_g.to_numpy(), m.pred_code_idx_q.to_numpy(), m.true_code_idx_g.to_numpy()
out = {"n_pairs": len(m), "n_resp": int(m.respondent_id.nunique()), "raw_agree": float((a == b).mean()), "kappa": float(kappa(a, b)),
       "both_correct": float(((a == t) & (b == t)).mean()), "both_wrong_agree": float(((a == b) & (a != t)).mean())}
groups = {r: i.to_numpy() for r, i in m.reset_index(drop=True).groupby("respondent_id").groups.items()}
ids = np.array(list(groups)); ks, ag = [], []
for _ in range(4000):
    pick = rng.choice(ids, len(ids), replace=True); ix = np.concatenate([groups[r] for r in pick])
    ks.append(kappa(a[ix], b[ix])); ag.append((a[ix] == b[ix]).mean())
out["kappa_ci"] = np.percentile(ks, [2.5, 97.5]).tolist(); out["agree_ci"] = np.percentile(ag, [2.5, 97.5]).tolist()
# chance-level: agreement expected if models were independent given their marginals
# sex gap decomposition with cluster CIs
GEN = {"Q29", "Q31", "Q32"}
def sexgap(d, items_mask):
    d = d[items_mask(d) & d.pred_code_idx.notna() & d.sg_sex.notna()]
    c = (d.true_code_idx == d.pred_code_idx)
    return c[d.sg_sex == "Male"].mean() - c[d.sg_sex == "Female"].mean()
for name, d in [("gemini", g), ("gptoss", q)]:
    res = {}
    for lab, mask in [("gender_items", lambda x: x.question_id.isin(GEN)), ("other_items", lambda x: ~x.question_id.isin(GEN))]:
        est = sexgap(d, mask); grp = {r: x for r, x in d.groupby("respondent_id")}; ids2 = np.array(list(grp)); bs = []
        for _ in range(3000):
            pick = rng.choice(ids2, len(ids2), replace=True); bs.append(sexgap(pd.concat([grp[r] for r in pick]), mask))
        bs = np.array([x for x in bs if not np.isnan(x)])
        res[lab] = {"male_minus_female": float(est), "ci": np.percentile(bs, [2.5, 97.5]).tolist(), "n_rows": int(mask(d).sum())}
    out["sexgap_" + name] = res
# pilot P0 vs P2 files
pil = {}
for f in ["groq-openai_gpt-oss-120b_P0", "groq-openai_gpt-oss-120b_P2", "gemini-3.1-flash-lite_P0"]:
    d = pd.read_parquet(REPO + f + ".parquet"); pil[f] = {"n": len(d), "resp": int(d.respondent_id.nunique()), "items": sorted(d.question_id.unique().tolist()),
        "acc": float((d.true_code_idx == d.pred_code_idx).mean())}
out["pilot_files"] = pil
json.dump(out, open("wvs_extra.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
