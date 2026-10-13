"""Shared, deterministic training-set construction for the breadth-vs-data experiment (used on Kaggle AND locally for matched classical baselines)."""
import json, os, sys
import numpy as np, pandas as pd


def _ctx(repo):
    os.chdir(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.config import DATA_PROCESSED
    from src.prompts.verbalize import verbalize_item
    from src.prompts.verbalize_pew import load_pew_codebook
    cb = load_pew_codebook()
    items = json.load(open(f"{DATA_PROCESSED}/pew_selected_items.json"))["selected_items"]
    folds = json.load(open(f"{DATA_PROCESSED}/pew_folds.json"))["folds"]
    df = pd.read_parquet(f"{DATA_PROCESSED}/pew_india_2021.parquet")
    df["respondent_id"] = df["respondent_id"].astype(int)
    df = df.set_index("respondent_id", drop=False)
    meta = {q: verbalize_item(q, cb) for q in items[:44]}
    return df, items, folds, cb, meta


def train_pairs(repo, lo, hi, n_resp=2000, n_ex=0, seed=0):
    """(respondent_id, question_id) training pairs: a fixed pool of n_resp fold-0 TRAIN respondents (same pool for any item range),
    all valid answers on items[lo:hi], randomly subsampled to n_ex examples. n_ex<=0 means 'as many as the 8-item pool has',
    so the 8-item and 24-item runs see exactly the same number of examples from the same respondents."""
    df, items, folds, cb, meta = _ctx(repo)
    train_ids = np.array(sorted(folds[0]["train"]))
    resp = np.random.default_rng(seed).permutation(train_ids)[:n_resp]

    def pool(a, b):
        out = []
        for r in resp:
            row = df.loc[int(r)]
            for q in items[a:b]:
                v = row.get(q)
                if pd.notna(v) and int(v) in meta[q]["code_to_index"]:
                    out.append((int(r), q))
        return out

    ref = pool(0, 8)
    n_ex = len(ref) if n_ex <= 0 else n_ex
    p = pool(lo, hi)
    sel = np.random.default_rng(seed + 1000).permutation(len(p))[:n_ex]
    return [p[i] for i in sel], len(ref)
