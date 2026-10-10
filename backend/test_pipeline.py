import json
from pipeline import answer_question
from datasets import load_dataset
from config import QUESTION_COL, ANSWER_COL

ds = load_dataset("G4KMU/t2-ragbench", "FinQA")
split = "dev"
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
print("\nRETRIEVED EVIDENCE DETAILS:")
for item in out["evidence"]:
    print(f"\nID: {item['id']}")
    print(f"Context: {item['context_id']}")
    print(f"Type: {item['type']}")
    print(f"Text: {item['text']}")


from retrieval import Retriever

retriever = Retriever()
question = r[QUESTION_COL]
gold_context = r["context_id"]

print("\nGOLD CONTEXT:", gold_context)

for mode in ["dense", "bm25", "dense_rerank", "hybrid", "hybrid_rerank"]:
    results = retriever.search(question, k=5, pool=50, mode=mode)

    print(f"\n--- {mode} ---")
    seen = set()

    for item in results:
        context_id = item["context_id"]

        if context_id in seen:
            continue

        seen.add(context_id)

        marker = " <-- GOLD CONTEXT" if context_id == gold_context else ""
        print(
            f"{context_id}{marker} | "
            f"{item['type']} | "
            f"{item['text'][:180]}"
        )