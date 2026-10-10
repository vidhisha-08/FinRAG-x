"""Evaluate systems A-D on a seeded, extendable sample. Resumable.

    python evaluate.py --n 30      # then --n 100, --n 200: earlier answers are reused

Ladder (each step changes one thing, so differences can be attributed):
  A  direct LLM, no retrieval
  B  basic RAG: dense top-5, no page expansion, model may calculate itself
  C  hybrid + rerank + page expansion, model may calculate itself
  D  C + calculation engine + verification   (= FinRAG-X)
"""
import argparse
import json
import logging
import random
import time
from pathlib import Path

import pandas as pd

from config import ANSWER_COL, QUESTION_COL
from engine import direct_answer, generate
from datasets import load_dataset
from metrics import has_answer, is_yes_no, to_float, wilson
from pipeline import R, answer_question

ds = load_dataset("G4KMU/t2-ragbench", "FinQA")
split = "dev"

K = 5
DATA = Path(__file__).resolve().parent.parent / "data"
PARTIAL = DATA / "eval_partial_v2.jsonl"       # new files: old A/C columns were hard-coded False
OUT = DATA / "eval_results_v2.csv"


def build_sample(n, seed):
    pool_ids = {c["context_id"] for c in R.chunks}
    eligible = [r for r in ds[split]
                if r["context_id"] in pool_ids
                and to_float(r[ANSWER_COL]) is not None
                and not is_yes_no(r[QUESTION_COL], to_float(r[ANSWER_COL]))]
    random.Random(seed).shuffle(eligible)      # was: first N rows (biased to the start of the file)
    return eligible[:n]                        # prefix-stable: raising n keeps earlier questions


def run_item(r):
    q, gold = r[QUESTION_COL], to_float(r[ANSWER_COL])
    page = r["context_id"]

    a = direct_answer(q)
    b_ev = R.search(q, k=K, mode="dense")
    b = generate(q, b_ev, None)
    c = answer_question(q, mode="hybrid_rerank", use_calc=False, use_checker=False)
    d = answer_question(q, mode="hybrid_rerank")

    return {
        "id": r["id"], "question": q, "gold": gold,
        "A_correct": has_answer(a, gold), "A_answer": a[:600],
        "B_correct": has_answer(b, gold), "B_answer": b[:600],
        "B_ctx_hit": any(x["context_id"] == page for x in b_ev),
        "C_correct": has_answer(c["raw_answer"], gold), "C_answer": c["raw_answer"][:600],
        "D_raw_correct": has_answer(d["raw_answer"], gold),       # before the verifier decides
        "D_refused": not d["passed"],
        "D_correct": bool(d["passed"] and has_answer(d["raw_answer"], gold)),
        "D_ctx_hit": any(x["context_id"] == page for x in d["evidence"]),
        "D_calc_correct": d["result"] is not None and has_answer(str(d["result"]), gold),
        "D_calc_error": d["calc_error"],
        "D_checks": json.dumps(d["checks"]),
        "D_program": json.dumps(d["program"]),
        "D_coverage": d["evidence_coverage"],
        "D_answer": d["raw_answer"][:600],
    }


def summarize(df):
    n = len(df)
    print(f"\nQuestions evaluated: {n}   (95% Wilson intervals; overlapping = not distinguishable)")
    for label, col in [("A  direct LLM", "A_correct"), ("B  basic RAG", "B_correct"),
                       ("C  hybrid+rerank+page", "C_correct"),
                       ("D  FinRAG-X (released & correct)", "D_correct")]:
        k = int(df[col].sum())
        lo, hi = wilson(k, n)
        print(f"  {label:34s} {k:3d}/{n}  {k / n:5.1%}   [{lo:.0%} - {hi:.0%}]")

    answered = df[~df["D_refused"]]
    print(f"\nD released {len(answered)}/{n} answers; correct among released: "
          f"{int(answered['D_correct'].sum())}/{len(answered)}")
    print("What the verifier did:")
    print("  kept a correct answer        :", int((~df["D_refused"] & df["D_raw_correct"]).sum()))
    print("  FALSE REFUSAL (was correct)  :", int((df["D_refused"] & df["D_raw_correct"]).sum()))
    print("  caught a wrong answer        :", int((df["D_refused"] & ~df["D_raw_correct"]).sum()))
    print("  MISSED a wrong answer        :", int((~df["D_refused"] & ~df["D_raw_correct"]).sum()))
    print(f"\nRetrieval (gold page among evidence): B {df['B_ctx_hit'].mean():.0%}, "
          f"D {df['D_ctx_hit'].mean():.0%}")
    print(f"Calculator exact on {int(df['D_calc_correct'].sum())}/{n}; "
          f"program errors on {int(df['D_calc_error'].notna().sum())}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    sample = build_sample(args.n, args.seed)
    done = {}
    if PARTIAL.exists():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            done[item["id"]] = item
    todo = [r for r in sample if r["id"] not in done]
    print(f"{len(sample) - len(todo)} already done, {len(todo)} to go")

    for i, r in enumerate(todo, 1):
        try:
            item = run_item(r)
        except Exception as e:
            print("Stopped at", r["id"], "->", str(e)[:200])
            print("Fix the problem, then run again. Progress is saved.")
            break
        with open(PARTIAL, "a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")
        done[r["id"]] = item
        print(f"{i}/{len(todo)} done")
        time.sleep(1)

    results = [done[r["id"]] for r in sample if r["id"] in done]
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUT, index=False)
        summarize(df)


if __name__ == "__main__":
    main()