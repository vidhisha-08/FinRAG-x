import pandas as pd
from load_data import ds, split
from config import QUESTION_COL
from pipeline import R

res = pd.read_csv("../data/eval_results.csv")
refused = res[res["D_refused"] == True].head(10)
by_id = {r["id"]: r for r in ds[split]}

for _, row in refused.iterrows():
    r = by_id[row["id"]]
    ev = R.search(r[QUESTION_COL], k=5, mode="hybrid_rerank")
    from_gold = [c for c in ev if c["context_id"] == r["context_id"]]
    print("-" * 70)
    print("Q:", r[QUESTION_COL][:120])
    print("retrieved types:", [(c["type"], c["context_id"] == r["context_id"]) for c in ev])
    print("chunks from gold page:", len(from_gold), "/ 5")
    print("gold table chunk retrieved:", any(c["type"] == "table" for c in from_gold))