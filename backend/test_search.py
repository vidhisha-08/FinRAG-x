from retrieval import Retriever
from load_data import ds, split
import json

chunks = json.load(open("../data/chunks.json"))
pool_ids = {c["context_id"] for c in chunks}
rows = [r for r in ds[split] if r["context_id"] in pool_ids]
r = rows[0]
print("QUESTION:", r["question"])
print("GOLD PAGE:", r["context_id"], "\n")

R = Retriever()
for c in R.search(r["question"]):
    mark = "<-- GOLD" if c["context_id"] == r["context_id"] else ""
    print(c["chunk_id"], c["type"], mark, "|", c["text"][:120].replace("\n", " "))