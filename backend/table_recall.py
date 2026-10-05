import json
from load_data import ds, split
from pipeline import R, expand_with_page

N = 50

chunks = json.load(open("../data/chunks.json"))
table_ids = {}
for c in chunks:
    if c["type"] == "table":
        table_ids.setdefault(c["context_id"], set()).add(c["chunk_id"])

rows = [r for r in ds[split] if r["context_id"] in table_ids][:N]
print(len(rows), "questions whose page has a table")

print("\nGOLD TABLE FOUND IN TOP RESULTS")
print(f"{'search':20s} {'plain':>8s} {'+ page expansion':>18s}")
for mode in ("dense", "dense_rerank", "hybrid_rerank"):
    plain = expanded = 0
    for r in rows:
        found = R.search(r["question"], k=5, mode=mode)
        gold = table_ids[r["context_id"]]
        plain += any(c["chunk_id"] in gold for c in found)
        expanded += any(c["chunk_id"] in gold for c in expand_with_page(found))
    print(f"{mode:20s} {plain / len(rows):8.2f} {expanded / len(rows):18.2f}")