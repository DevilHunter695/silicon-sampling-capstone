"""How much do demographics change the zero-shot model? P0 (no profile) vs P2 (profile) on the same rows."""
import json, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3)
p0 = json.load(open(f"{HERE}/mlx_p0.json"))
z = pd.read_json(f"{HERE}/mlx_base_chat.jsonl", lines=True).drop_duplicates(["respondent_id", "question_id"])
z["p0_pred"] = z.question_id.map(lambda q: p0[q]["pred_idx"])
z["p0_probs"] = z.question_id.map(lambda q: p0[q]["probs"])
z["tv_shift"] = [0.5 * np.abs(np.array(a) - np.array(b)).sum() for a, b in zip(z.probs, z.p0_probs)]
ids = np.sort(z.respondent_id.unique()); ri = {r: i for i, r in enumerate(ids)}; R = len(ids); z["ri"] = z.respondent_id.map(ri)
W = rng.multinomial(R, np.ones(R) / R, size=3000).astype(float)
def acc(col):
    c = (z[col] == z.true_idx).astype(float).to_numpy()
    s = np.bincount(z.ri, weights=c, minlength=R); n = np.bincount(z.ri, minlength=R).astype(float)
    return float(s.sum() / n.sum()), np.percentile((W @ s) / (W @ n), [2.5, 97.5]).tolist(), s, n
a0, ci0, s0, n0 = acc("p0_pred"); a2, ci2, s2, n2 = acc("pred_idx")
d = (W @ s2) / (W @ n2) - (W @ s0) / (W @ n0)
out = {"n_rows": int(len(z)), "n_resp": int(R), "acc_p0": a0, "acc_p0_ci": ci0, "acc_p2": a2, "acc_p2_ci": ci2,
       "p2_minus_p0": a2 - a0, "p2_minus_p0_ci": np.percentile(d, [2.5, 97.5]).tolist(),
       "same_prediction_rate": float((z.pred_idx == z.p0_pred).mean()), "mean_tv_shift": float(z.tv_shift.mean()),
       "p0_pred_by_item": {q: p0[q]["pred_idx"] for q in p0}}
json.dump(out, open(f"{HERE}/zs_p0.json", "w"), indent=1); print(json.dumps(out, indent=1))
