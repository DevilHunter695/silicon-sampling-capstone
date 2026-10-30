"""Build an offline interactive demo over the stored Pew fold-0 model outputs.
Usage: python make_demo.py demo_rows.parquet out.html [full|selected]
  full     all 46,209 test answers, nothing picked.
  selected only the answers where the fine-tuned model was >=70% confident, plus our best correct examples.
Input: demo_rows.parquet from build_data.py. Output needs no server and no internet."""
import sys, json, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "full"
assert mode in ("full", "selected")
d = pd.read_parquet(src)
for c in ("ft", "zs", "gboost", "logistic", "majority", "options", "profile"):
    d[c] = d[c].apply(lambda x: None if x is None else json.loads(x))
NAMES = {"QFERTrec": "Number of children", "Q12": "Politicians in religion", "Q15": "Sex-selective checkups", "ABORT": "Abortion",
         "Q28rec": "What being [religion] means", "Q3132rec": "Worship attendance", "Q39a": "Watching religious TV", "Q50": "Reading religious books"}
ATTRS = [("sex", "Sex"), ("age", "Age"), ("marital_status", "Marital status"), ("education", "Education"), ("religion", "Religion"),
         ("caste", "Caste"), ("urban_rural", "Urban / rural"), ("region", "Region"), ("income_decile", "Income")]
def short(v):
    v = str(v).split(" (")[0].split(";")[0].replace(" [OR]", "").replace(" (DO NOT READ)", "")
    return {"Divorced, separated, partner is no longer alive": "Divorced / separated / widowed", "Married or living with a partner": "Married / partner"}.get(v, v)
cats = {a: sorted({str(p.get(a, "Unknown")) for p in d.profile}) for a, _ in ATTRS}   # original labels (sent to the live model)
disp = {a: [short(v) for v in cats[a]] for a, _ in ATTRS}                                 # shorter labels for display
qs = sorted(d.qid.unique(), key=lambda q: list(NAMES).index(q))
qi = {q: i for i, q in enumerate(qs)}
rids = sorted(d.rid.unique()); ri = {r: i for i, r in enumerate(rids)}
resp = {}
for r, p in zip(d.rid, d.profile):
    if r not in resp: resp[r] = [cats[a].index(str(p.get(a, "Unknown"))) for a, _ in ATTRS]
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
acc = lambda g, col: float((g[col].apply(lambda x: int(np.argmax(x))) == g.true_idx).mean())
byq = [dict(name=NAMES[q], n=int(len(g)), ft=acc(g, "ft"), gb=acc(g, "gboost"), zs=acc(g, "zs"), mj=acc(g, "majority")) for q, g in ((q, d[d.qid == q]) for q in qs)]
ch = d[d.qid == "QFERTrec"]
marital = []
for lab, v in (("Unmarried", "Unmarried"), ("Married / partner", "Married or living with a partner"), ("Divorced / widowed", "Divorced, separated, partner is no longer alive")):
    g = ch[ch.profile.apply(lambda p: str(p.get("marital_status")) == v)]
    marital.append(dict(l=lab, n=int(len(g)), a=float(g.ok.mean()), h=float((g.conf >= .7).mean())))
summ = dict(byq=byq, marital=marital, hi=int((d.conf >= .7).sum()), hiacc=float(d[d.conf >= .7].ok.mean()))
if mode == "selected":
    keep = sorted(set(d.index[d.conf >= .7]) | {i for i, _ in curated})
    newi = {o: n for n, o in enumerate(keep)}
    used = sorted({rows[i][0] for i in keep}); rmap = {o: n for n, o in enumerate(used)}
    R = [R[o] for o in used]
    rows = [[rmap[rows[i][0]]] + rows[i][1:] for i in keep]
    curated = [[newi[i], lab] for i, lab in curated]
data = json.dumps(dict(mode=mode, cats=cats, disp=disp, attrs=ATTRS, R=R, Q=Q, rows=rows, curated=curated if mode == "selected" else [], tot=tot, sum=summ), separators=(",", ":"), ensure_ascii=False)
html = open(__file__.replace("make_demo.py", "template.html")).read().replace("__DATA__", data)
html = html.replace("<title>Group 55 Interactive Demo</title>", "<title>Group 55 Demo %s</title>" % ("Selected Data" if mode == "selected" else "Full Data"))
open(out, "w").write(html); print("wrote", out, len(html) // 1024, "KB", tot)
