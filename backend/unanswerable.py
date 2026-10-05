import json
import time
from load_data import ds, split
from config import QUESTION_COL, ANSWER_COL
from engine import generate, extract_program, execute, verify
from pipeline import R, NUMERIC_WORDS, expand_with_page

N = 10
chunks = json.load(open("../data/chunks.json"))
pool_ids = {c["context_id"] for c in chunks}
rows = [r for r in ds[split] if r["context_id"] in pool_ids][:N]


def wrong_evidence(q, gold_page):
    found = R.search(q, k=30, mode="hybrid_rerank")
    return [c for c in found if c["context_id"] != gold_page][:5]    # gold page removed


b_refuse = d_refuse = 0
for i, r in enumerate(rows, 1):
    q = r[QUESTION_COL]
    ev = wrong_evidence(q, r["context_id"])

    b_text = generate(q, ev, "none")                       # B: basic RAG, no checker
    b_refuse += "INSUFFICIENT EVIDENCE" in b_text.upper()

    ev_d = expand_with_page(ev)                            # D: with checker
    program, result = None, None
    if any(w in q.lower() for w in NUMERIC_WORDS):
        try:
            program = extract_program(q, "\n\n".join(c["text"][:1500] for c in ev_d))
            result = execute(program) if program.get("op") != "none" else None
        except Exception:
            program, result = None, None
    d_text = generate(q, ev_d, f"{result:.6f}" if result is not None else "none")
    d_refuse += not all(verify(d_text, ev_d, program, result).values())
    if "INSUFFICIENT EVIDENCE" not in b_text.upper() or all(verify(d_text, ev_d, program, result).values()):
        print("  NOT REFUSED ->", q[:100])
        print("  B said:", b_text[:150].replace("\n", " "))
        print("  D said:", d_text[:150].replace("\n", " "))

    print(f"{i}/{N} done")
    time.sleep(2)

print("\nUNANSWERABLE TEST (gold page removed; higher refusal is better)")
print("B refused:", b_refuse, "/", N)
print("D refused:", d_refuse, "/", N)