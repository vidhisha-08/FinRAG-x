import json
import os
from load_data import ds, split
from config import MAX_DOCS

# one row per report page
df = ds[split].to_pandas().drop_duplicates("context_id").head(MAX_DOCS)


def group_lines(text, limit=900):
    """Join short lines into chunks of about 900 characters."""
    out, cur = [], ""
    for line in str(text).split("\n"):
        line = line.strip()
        if not line:
            continue
        if cur and len(cur) + len(line) > limit:
            out.append(cur)
            cur = line
        else:
            cur = (cur + " " + line).strip()
    if cur:
        out.append(cur)
    return out


chunks = []
for d, row in enumerate(df.itertuples()):
    tag = f"{row.company_name} {row.report_year} page {row.page_number}"
    pieces = [("text", t) for t in group_lines(row.pre_text)]
    if str(row.table).strip():
        pieces.append(("table", str(row.table)))
    pieces += [("text", t) for t in group_lines(row.post_text)]

    for i, (kind, t) in enumerate(pieces):
        chunks.append({
            "chunk_id": f"D{d}_C{i}",
            "context_id": row.context_id,
            "company": row.company_name,
            "year": int(row.report_year),
            "page": int(row.page_number),
            "type": kind,
            "text": f"[{tag}] {t}",
        })

os.makedirs("../data", exist_ok=True)
json.dump(chunks, open("../data/chunks.json", "w"))
print(len(df), "pages,", len(chunks), "chunks")