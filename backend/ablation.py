import json
import os
import time
import pandas as pd
from load_data import ds, split
from config import QUESTION_COL, ANSWER_COL
from engine import numbers_in
from pipeline import answer_question

N = 3
PARTIAL = "../data/ablation_partial.jsonl"

chunks = json.load(open("../data/chunks.json"))
pool_ids = {c["context_id"] for c in chunks}
rows = [r for r in ds[split] if r["context_id"] in pool_ids]

def to_float(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").replace("$", ""))
    except ValueError:
        return None

def has_answer(text, gold):
    targets = {abs(gold), abs(gold) * 100}
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
        full = answer_question(q, mode="dense")
        no_expand = answer_question(q, mode="dense", expand=False)
    except Exception as e:
        print("Stopped at question", i, "->", str(e)[:100])
        print("Run this file again later. Progress is saved.")
        break

    item = {
        "id": r["id"], "question": q, "gold": gold,
        "full_correct": bool(full["passed"] and has_answer(full["answer"], gold)),
        "full_refused": not full["passed"],
        "no_checker_correct": has_answer(full["raw_answer"], gold),     # same answer, refusal ignored
        "no_expand_correct": bool(no_expand["passed"] and has_answer(no_expand["answer"], gold)),
        "no_expand_refused": not no_expand["passed"],
    }
    with open(PARTIAL, "a") as f:
        f.write(json.dumps(item) + "\n")
    done[r["id"]] = item
    print(f"{i}/{len(sample)} done")
    time.sleep(10)

if done:
    res = pd.DataFrame(list(done.values()))
    res = res.drop(columns=[c for c in res.columns if c.startswith("no_calc")], errors="ignore")
    res.to_csv("../data/ablation_results.csv", index=False)
    print("\nQuestions:", len(res))
    print("ABLATION (share of questions)")
    print(res.drop(columns=["id", "question", "gold"]).mean().round(3))