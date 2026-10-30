"""Build demo.html: one offline interactive page over the stored Pew fold-0 model outputs.
Input: demo_rows.parquet from build_data.py. Output: demo.html (no server, no internet)."""
import sys, json, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
d = pd.read_parquet(src)
for c in ("ft", "zs", "gboost", "logistic", "majority", "options", "profile"):
    d[c] = d[c].apply(lambda x: None if x is None else json.loads(x))
NAMES = {"QFERTrec": "Number of children", "Q12": "Politicians in religion", "Q15": "Sex-selective checkups", "ABORT": "Abortion",
         "Q28rec": "What being [religion] means", "Q3132rec": "Worship attendance", "Q39a": "Watching religious TV", "Q50": "Reading religious books"}
ATTRS = [("sex", "Sex"), ("age", "Age"), ("marital_status", "Marital status"), ("education", "Education"), ("religion", "Religion"),
         ("caste", "Caste"), ("urban_rural", "Urban / rural"), ("region", "Region")]
def short(v):
    v = str(v).split(" (")[0].split(";")[0].replace(" [OR]", "").replace(" (DO NOT READ)", "")
    return {"Divorced, separated, partner is no longer alive": "Divorced / separated / widowed", "Married or living with a partner": "Married / partner"}.get(v, v)
cats = {a: sorted({short(p.get(a, "Unknown")) for p in d.profile}) for a, _ in ATTRS}
qs = sorted(d.qid.unique(), key=lambda q: list(NAMES).index(q))
qi = {q: i for i, q in enumerate(qs)}
rids = sorted(d.rid.unique()); ri = {r: i for i, r in enumerate(rids)}
resp = {}
for r, p in zip(d.rid, d.profile):
    if r not in resp: resp[r] = [cats[a].index(short(p.get(a, "Unknown"))) for a, _ in ATTRS]
R = [[int(r)] + resp[r] for r in rids]
Q = [dict(id=q, name=NAMES[q], text=d[d.qid == q].question.iloc[0], opts=d[d.qid == q].options.iloc[0]) for q in qs]
def pad(x, n=8): return [int(round(v * 1000)) for v in x] + [0] * (n - len(x)) if x is not None else [0] * n
rows = [[ri[r.rid], qi[r.qid], int(r.true_idx)] + pad(r.ft) + pad(r.gboost) + pad(r.zs) + pad(r.majority) for r in d.itertuples()]
d["conf"] = d.ft.apply(max); d["ok"] = d.ft.apply(lambda x: int(np.argmax(x))) == d.true_idx
d["pred"] = d.ft.apply(lambda x: int(np.argmax(x)))
d = d.reset_index(drop=True)
curated = []
hi = d[d.ok & (d.conf > .7)]
for lab, g in (("Unmarried, no children", hi[hi.pred == 0]), ("Large family (3+ children)", hi[hi.pred == 3])):
    for i in g.sample(min(25, len(g)), random_state=3).index: curated.append([int(i), lab])
for q, g in d[(d.qid != "QFERTrec") & d.ok].groupby("qid"):
    for i in g.sort_values("conf", ascending=False).head(3).index: curated.append([int(i), "Most confident correct: " + NAMES[q]])
tot = dict(n=len(d), acc=float(d.ok.mean()), mj=0.3036, gb=0.4004, zs=0.1608)  # gb/mj from paper tables (pew_aligned_rows.parquet)
data = json.dumps(dict(cats=cats, attrs=ATTRS, R=R, Q=Q, rows=rows, curated=curated, tot=tot), separators=(",", ":"), ensure_ascii=False)
html = open(__file__.replace("make_demo.py", "template.html")).read().replace("__DATA__", data)
open(out, "w").write(html); print("wrote", out, len(html) // 1024, "KB", tot)
