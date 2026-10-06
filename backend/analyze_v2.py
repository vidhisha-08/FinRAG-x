"""Print the cases that explain the evaluate.py summary. Paste the output back for review.

    python analyze_v2.py
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "eval_results_v2.csv"
df = pd.read_csv(CSV)
for col in ("A_correct", "B_correct", "C_correct", "D_raw_correct", "D_refused", "D_correct"):
    df[col] = df[col].astype(bool)


def short(x, n=160):
    return str(x).replace("\n", " ")[:n]


def show(title, rows, cols):
    print(f"\n=== {title}: {len(rows)} ===")
    for _, r in rows.iterrows():
        print(f"- {r['id']}  gold={r['gold']:.6g}")
        for c in cols:
            print(f"    {c}: {short(r[c])}")


show("B correct but C wrong (what did hybrid + page expansion break?)",
     df[df.B_correct & ~df.C_correct], ["B_answer", "C_answer", "D_ctx_hit"])

show("MISSED wrong answers (verifier released them)",
     df[~df.D_refused & ~df.D_raw_correct], ["D_program", "D_checks", "D_answer"])

show("FALSE REFUSALS (answer was right, verifier blocked it)",
     df[df.D_refused & df.D_raw_correct], ["D_checks", "D_answer"])

fails = Counter()
for s in df[df.D_refused]["D_checks"]:
    for name, ok in json.loads(s).items():
        if not ok:
            fails[name] += 1
print("\n=== Which check failed on refused answers ===")
for name, n in fails.most_common():
    print(f"  {name}: {n}")

print("\n=== Calculator ===")
prog = df.D_program.astype(str)
empty = df.D_program.isna() | prog.str.contains('"steps": []', regex=False)
print(f"  calc exact: {int(df.D_calc_correct.sum())}/{len(df)};  no program produced: {int(empty.sum())}")