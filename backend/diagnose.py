import json
import pandas as pd
from load_data import ds, split
from config import QUESTION_COL, ANSWER_COL
from pipeline import answer_question

res = pd.read_csv("../data/eval_results.csv")
refused = res[res["D_refused"] == True].head(8)
by_id = {r["id"]: r for r in ds[split]}

for _, row in refused.iterrows():
    r = by_id[row["id"]]
    out = answer_question(r[QUESTION_COL])
    failed = [k for k, v in out["checks"].items() if not v]
    print("-" * 70)
    print("Q:", r[QUESTION_COL][:160])
    print("GOLD:", r[ANSWER_COL])
    print("CALC:", out["program"], "->", out["result"])
    print("FAILED CHECKS:", failed)
    print("MODEL SAID:", out["raw_answer"][:250].replace("\n", " "))