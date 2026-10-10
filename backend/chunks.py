
import json
from pathlib import Path

from datasets import load_dataset

from config import MAX_DOCS

DATASET_NAME = "G4KMU/t2-ragbench"
CONFIG_NAME = "FinQA"

# Paths are based on this file's location, not the terminal's directory.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
OUTPUT_FILE = PROJECT_DIR / "data" / "chunks.json"

CHUNK_SIZE = 900


def group_lines(text, limit=CHUNK_SIZE):
    """Combine text lines into chunks of approximately limit characters."""
    if text is None:
        return []

    text = str(text)
    output = []
    current = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if current and len(current) + len(line) + 1 > limit:
            output.append(current)
            current = line
        else:
            current = (current + " " + line).strip()

    if current:
        output.append(current)

    return output


def safe_value(value, default="Unknown"):
    """Convert missing metadata into a readable value."""
    if value is None:
        return default

    try:
        if value != value:  # Handles floating-point NaN
            return default
    except (TypeError, ValueError):
        pass

    return str(value)


def build_chunks():
    print(f"Loading {DATASET_NAME}, configuration {CONFIG_NAME}...")
    dataset = load_dataset(DATASET_NAME, CONFIG_NAME)

    # Combine the available splits so the retrieval corpus includes
    # all unique contexts represented in this benchmark configuration.
    records = []
    for split_name, split_data in dataset.items():
        print(f"{split_name}: {len(split_data)} examples")
        for record in split_data:
            records.append(record)

    # Keep one record per document context.
    unique_records = {}
    for record in records:
        context_id = record.get("context_id")

        if context_id is None:
            continue

        unique_records.setdefault(str(context_id), record)

    pages = list(unique_records.values())

    # MAX_DOCS is retained for small smoke tests.
    # Set MAX_DOCS = None in config.py to index all unique contexts.
    if MAX_DOCS is not None:
        pages = pages[:MAX_DOCS]

    chunks = []

    for page_index, row in enumerate(pages):
        company = safe_value(row.get("company_name"))
        year = safe_value(row.get("report_year"))
        page = safe_value(row.get("page_number"))
        context_id = str(row["context_id"])

        source = safe_value(row.get("file_name"))
        label = f"{company} | {year} | page {page}"

        pieces = []

        for text in group_lines(row.get("pre_text")):
            pieces.append(("text", text))

        table = row.get("table")
        if table is not None and str(table).strip():
            pieces.append(("table", str(table)))

        for text in group_lines(row.get("post_text")):
            pieces.append(("text", text))

        # Fallback when the structured fields are empty.
        if not pieces:
            for text in group_lines(row.get("context")):
                pieces.append(("text", text))

        for piece_index, (kind, text) in enumerate(pieces):
            chunks.append({
                "chunk_id": f"D{page_index}_C{piece_index}",
                "context_id": context_id,
                "company": company,
                "year": year,
                "page": page,
                "source": source,
                "type": kind,
                "text": f"[{label}] {text}",
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print("\nChunking completed.")
    print("Unique contexts processed:", len(pages))
    print("Total chunks created:", len(chunks))
    print("Output file:", OUTPUT_FILE)

    if chunks:
        print("\nFirst chunk preview:")
        print(json.dumps(chunks[0], ensure_ascii=False, indent=2))

    return chunks


if __name__ == "__main__":
    build_chunks()