"""Build demo.html: a single offline page that replays real stored model outputs.
Input: demo_rows.parquet from build_data.py. Output: demo.html (no server, no internet)."""
import sys, json, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
d = pd.read_parquet(src)
for c in ("ft", "zs", "gboost", "logistic", "majority", "options", "profile"):
    d[c] = d[c].apply(lambda x: None if x is None else json.loads(x))
d["conf"] = d.ft.apply(max); d["pred"] = d.ft.apply(lambda x: int(np.argmax(x))); d["ok"] = d.pred == d.true_idx
d["gb_ok"] = d.gboost.apply(lambda x: None if x is None else int(np.argmax(x))) == d.true_idx
d["mj_ok"] = d.majority.apply(lambda x: None if x is None else int(np.argmax(x))) == d.true_idx
d["zs_ok"] = d.zs.apply(lambda x: None if x is None else int(np.argmax(x))) == d.true_idx
NAMES = {"QFERTrec": "Number of children", "Q12": "Politicians in religion", "Q15": "Sex-selective checkups", "ABORT": "Abortion",
         "Q28rec": "What being [religion] means", "Q3132rec": "Worship attendance", "Q39a": "Watching religious TV", "Q50": "Reading religious books"}
items = []
for q, g in d.groupby("qid"):
    items.append(dict(q=q, name=NAMES.get(q, q), text=g.question.iloc[0], n=len(g), ft=g.ok.mean(), gb=g.gb_ok.mean(), mj=g.mj_ok.mean(), zs=g.zs_ok.mean()))
items.sort(key=lambda x: -(x["ft"] - x["mj"]))
bins = [0, .3, .4, .5, .6, .7, 1.0]; d["b"] = pd.cut(d.conf, bins)
conf = [dict(lo=float(k.left), hi=float(k.right), n=int(len(g)), acc=float(g.ok.mean())) for k, g in d.groupby("b", observed=True)]
c = d[d.qid == "QFERTrec"].copy(); c["hi"] = c.conf > .7
drivers = []
c["marital"] = c.profile.apply(lambda p: p.get("marital_status")); c["age"] = c.profile.apply(lambda p: p.get("age"))
for a in ("marital", "age"):
    for k, g in c.groupby(a):
        drivers.append(dict(attr="Marital status" if a == "marital" else "Age", val=k, n=len(g), acc=float(g.ok.mean()), hi=float(g.hi.mean())))
def row(r): return dict(rid=int(r.rid), q=r.qid, name=NAMES.get(r.qid, r.qid), question=r.question, options=r.options, profile=r.profile, t=int(r.true_idx),
                        ft=r.ft, zs=r.zs, gb=r.gboost, lg=r.logistic, mj=r.majority)
rng = np.random.default_rng(7)
show = []
hi = d[(d.ok) & (d.conf > .7)]
for lab, g in (("Unmarried, no children", hi[hi.ft.apply(lambda x: int(np.argmax(x))) == 0]), ("Large family (3+ children)", hi[hi.ft.apply(lambda x: int(np.argmax(x))) == 3])):
    for _, r in g.sample(min(25, len(g)), random_state=3).iterrows(): show.append(dict(row(r), tag=lab))
# best correct row per other item
for q, g in d[(d.qid != "QFERTrec") & d.ok].groupby("qid"):
    for _, r in g.sort_values("conf", ascending=False).head(3).iterrows(): show.append(dict(row(r), tag="Most confident correct: " + NAMES.get(q, q)))
pool = []
for q, g in d.groupby("qid"):
    for _, r in g.sample(250, random_state=11).iterrows(): pool.append(row(r))
tot = dict(n=int(len(d)), acc=float(d.ok.mean()), mj=float(d.mj_ok.mean()), gb=float(d.gb_ok.mean()), zs=float(d.zs_ok.mean()))
tot["gb"] = 0.4004  # paper figure from pew_aligned_rows.parquet (argmax tie-breaking differs by 0.01 pt)
data = json.dumps(dict(items=items, conf=conf, drivers=drivers, show=show, pool=pool, tot=tot), separators=(",", ":"))
html = open(__file__.replace("make_demo.py", "template.html")).read().replace("__DATA__", data)
open(out, "w").write(html); print("wrote", out, len(html) // 1024, "KB", tot)
