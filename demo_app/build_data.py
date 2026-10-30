import sys, json, pandas as pd, numpy as np
sys.path.insert(0, ".")
from src.prompts.verbalize_pew import load_pew_codebook, verbalize_demographics
from src.prompts.verbalize import verbalize_item
OUT="/Users/krishank/Downloads/pew_probs/out"; BL="/Users/krishank/kaggle_pkg2/baselines_probs.parquet"
cb=load_pew_codebook()
df=pd.read_parquet("data/processed/pew_india_2021.parquet"); df["respondent_id"]=df.respondent_id.astype(int); df=df.set_index("respondent_id",drop=False)
def load(cond): 
    d=pd.concat([pd.read_json(f"{OUT}/{cond}_s{i}of2.jsonl",lines=True) for i in (0,1)])
    return d.set_index(["respondent_id","question_id"])
ft=load("adapter_raw"); zs=load("base_chat")
b=pd.read_parquet(BL); print(b.method.unique(), b.regime.unique())
b=b[b.regime=="full"]
bl={m:b[b.method==m].set_index(["respondent_id","question_id"]) for m in ("majority","logistic","gboost")}
meta={q:verbalize_item(q,cb) for q in ft.index.get_level_values(1).unique()}
rows=[]
for (rid,q),r in ft.iterrows():
    prof=verbalize_demographics(df.loc[rid],cb); prof={k:v for k,v in prof.items() if v is not None}
    it=meta[q]; opts=[l.split(". ",1)[1] for l in it["options_text"].splitlines()]
    def p(t,key=(rid,q)):
        try: return list(map(float,t.loc[key].probs if not isinstance(t.loc[key],pd.DataFrame) else t.loc[key].iloc[0].probs))
        except KeyError: return None
    rows.append(dict(rid=int(rid),qid=q,question=it["question_text"],options=opts,profile=prof,true_idx=int(r.true_idx),
        ft=list(map(float,r.probs)),zs=p(zs),gboost=p(bl["gboost"]),logistic=p(bl["logistic"]),majority=p(bl["majority"])))
o=pd.DataFrame(rows); o["options"]=o.options.apply(json.dumps); o["profile"]=o.profile.apply(json.dumps)
for c in ("ft","zs","gboost","logistic","majority"): o[c]=o[c].apply(lambda x: None if x is None else json.dumps([round(v,4) for v in x]))
o.to_parquet("/private/tmp/claude-501/-Users-krishank-Documents-College-Projects----/c07d5fa6-6bd6-4110-b7a8-4e0125580098/scratchpad/demo/demo_rows.parquet"); print(o.shape); print(o.iloc[0].to_dict())
