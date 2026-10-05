import json
from pipeline import answer_question
from load_data import ds, split
from config import QUESTION_COL, ANSWER_COL

chunks = json.load(open("../data/chunks.json"))
pool_ids = {c["context_id"] for c in chunks}
rows = [r for r in ds[split] if r["context_id"] in pool_ids]

r = rows[0]
out = answer_question(r[QUESTION_COL])
print("QUESTION:", r[QUESTION_COL])
print("GOLD ANSWER:", r[ANSWER_COL])
print("MODEL ANSWER:", out["answer"])
print("CALCULATION:", out["program"], "->", out["result"])
print("CHECKS:", out["checks"])
print("EVIDENCE COVERAGE:", out["evidence_coverage"], "%")
print("RAW ANSWER:", out["raw_answer"])