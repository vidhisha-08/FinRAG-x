import json
import time
from load_data import ds, split
from config import QUESTION_COL
from engine import generate, extract_program, execute, verify
from pipeline import R, NUMERIC_WORDS, expand_with_page

chunks = json.load(open("../data/chunks.json"))
pool_ids = {c["context_id"] for c in chunks}
rows = [r for r in ds[split] if r["context_id"] in pool_ids]

b_refuse = d_refuse = used = 0
for r in rows:
    if used >= 6:
        break
    q = r[QUESTION_COL]
    company = next(c["company"] for c in chunks if c["context_id"] == r["context_id"])
    # near miss: same company, different report page
    ev = [c for c in R.search(q, k=60, mode="hybrid_rerank")
          if c["company"] == company and c["context_id"] != r["context_id"]][:5]
    if len(ev) < 3:
        continue                                   # company has no other page in the index
    used += 1

    b_text = generate(q, ev, "none")
    b_refuse += "INSUFFICIENT EVIDENCE" in b_text.upper()

    ev_d = ev
    program, result = None, None
    if any(w in q.lower() for w in NUMERIC_WORDS):
        try:
            program = extract_program(q, "\n\n".join(c["text"][:1500] for c in ev_d))
            result = execute(program) if program.get("op") != "none" else None
        except Exception:
            program, result = None, None
    d_text = generate(q, ev_d, f"{result:.6f}" if result is not None else "none")
    d_ok = all(verify(d_text, ev_d, program, result).values())
    d_refuse += not d_ok
    print(f"{used}/10 {company}: B {'refused' if 'INSUFFICIENT EVIDENCE' in b_text.upper() else 'ANSWERED'}"
          f" | D {'ANSWERED' if d_ok else 'refused'}")
    time.sleep(2)

print("\nNEAR-MISS TEST (same company, wrong page; higher refusal is better)")
print("Questions used:", used)
print("B refused:", b_refuse, "/", used)
print("D refused:", d_refuse, "/", used)