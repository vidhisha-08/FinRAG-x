import pandas as pd
from pipeline import answer_question

df = pd.read_csv("../data/eval_results_v2.csv")
ids = ["finqa_test_192", "finqa_test_251", "finqa_test_556", "finqa_dev_129",
       "finqa_test_16", "finqa_test_834", "finqa_dev_498", "finqa_dev_718"]

for i in ids:
    r = df[df.id == i].iloc[0]
    out = answer_question(r.question, mode="hybrid_rerank")
    print("\n", i, "| gold:", r.gold)
    print("  passed:", out["passed"], "| result:", out["result"])
    print("  raw:", out["raw_answer"][:160].replace("\n", " "))
    print("  checks:", out["checks"])