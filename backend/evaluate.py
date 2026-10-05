import json
import os
import time
import pandas as pd
from load_data import ds, split
from config import QUESTION_COL, ANSWER_COL
from engine import llm, generate, numbers_in
from pipeline import R, answer_question

N = 30                      # questions for the answer comparison
K = 5
PARTIAL = "../data/eval_partial.jsonl"

chunks = json.load(open("../data/chunks.json"))
pool_ids = {c["context_id"] for c in chunks}
rows = [r for r in ds[split] if r["context_id"] in pool_ids]


def to_float(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").replace("$", ""))
    except ValueError:
        return None


def has_answer(text, gold):
    targets = {abs(gold), abs(gold) * 100}          # 0.0436 can also be written as 4.36
    for n in numbers_in(text):
        for t in targets:
            if abs(n - t) <= max(0.01 * t, 1e-4):
                return True
    return False


def is_yes_no(q, gold):
    first = q.strip().split()[0].lower()
    return gold in (0.0, 1.0) and first in ("did", "was", "is", "were", "does", "has", "had", "are", "do", "can")


sample = [r for r in rows if to_float(r[ANSWER_COL]) is not None
          and not is_yes_no(r[QUESTION_COL], to_float(r[ANSWER_COL]))][:N]

done = {}
if os.path.exists(PARTIAL):
    for line in open(PARTIAL):
        item = json.loads(line)
        done[item["id"]] = item
print(len(done), "already done,", len(sample) - len(done), "to go")

for i, r in enumerate(sample, 1):
    if r["id"] in done:
        continue
    q, gold = r[QUESTION_COL], to_float(r[ANSWER_COL])
    try:
        b_text = generate(q, R.search(q, k=K, mode="dense"), "none")
        d = answer_question(q, mode="dense")
    except Exception as e:
        print("Stopped at question", i, "->", str(e)[:100])
        print("Fix the connection, then run this file again. Progress is saved.")
        break

    item = {
        "id": r["id"], "question": q, "gold": gold,
        "A_correct": False,
        "B_correct": has_answer(b_text, gold),
        "C_correct": False,
        "D_correct": bool(d["passed"] and has_answer(d["answer"], gold)),
        "D_refused": not d["passed"],
        "D_calc_correct": d["result"] is not None and has_answer(str(d["result"]), gold),
    }
    with open(PARTIAL, "a") as f:
        f.write(json.dumps(item) + "\n")
    done[r["id"]] = item
    print(f"{i}/{len(sample)} done")
    time.sleep(2)

if done:
    res = pd.DataFrame(list(done.values()))
    res.to_csv("../data/eval_results.csv", index=False)
    print("\nQuestions evaluated:", len(res))
    print("ANSWER QUALITY (share of questions)")
    print(res[["A_correct", "B_correct", "C_correct", "D_correct",
               "D_refused", "D_calc_correct"]].mean().round(3))