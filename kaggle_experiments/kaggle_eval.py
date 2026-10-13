"""Sanity-check and summarise the Kaggle outputs. Usage: python kaggle_eval.py <out_dir> <lab_predictions.parquet>"""
import sys, glob, json, numpy as np, pandas as pd
out, lab = sys.argv[1], sys.argv[2]
L = pd.read_parquet(lab)[["respondent_id", "question_id", "pred_code_idx", "true_code_idx"]].rename(columns={"pred_code_idx": "lab_pred"})
for cond in ["adapter_raw", "base_raw", "base_chat"]:
    fs = sorted(glob.glob(f"{out}/{cond}_s*.jsonl"))
    if not fs:
        continue
    d = pd.concat([pd.read_json(f, lines=True) for f in fs]).drop_duplicates(["respondent_id", "question_id"])
    d["correct"] = d.pred_idx == d.true_idx
    d["nll"] = [-np.log(max(p[t], 1e-6)) for p, t in zip(d.probs, d.true_idx)]
    print(f"\n== {cond}: {len(d)} rows, {d.respondent_id.nunique()} respondents (of 46,209 / 6,000)")
    print(f"accuracy {100*d.correct.mean():.2f}  NLL {d.nll.mean():.3f}  mean valid digit mass {d.valid_mass.mean():.3f}")
    print((100 * d.groupby("question_id").correct.mean()).round(1).to_dict())
    if cond == "adapter_raw":
        m = d.merge(L, on=["respondent_id", "question_id"])
        ok = m.lab_pred.notna()
        print(f"agreement with lab hard predictions: {100*(m.pred_idx[ok]==m.lab_pred[ok]).mean():.1f}%  "
              f"(lab acc on these rows {100*(m.lab_pred[ok]==m.true_idx[ok]).mean():.2f})")
        q12 = d[d.question_id == "Q12"].pred_idx.value_counts(normalize=True).round(3).to_dict()
        print("Q12 predicted-answer shares:", q12)
