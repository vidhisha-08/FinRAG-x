
from datasets import load_dataset
from retrieval import Retriever
from engine import extract_program, execute


def main():
    # 1. Load one example from the FinQA development dataset
    dataset = load_dataset("G4KMU/t2-ragbench", "FinQA")
    example = dataset["dev"][0]

    # 2. Display original dataset metadata
    print("\nORIGINAL DATASET METADATA:")

    for field in [
        "id",
        "context_id",
        "company_name",
        "report_year",
        "page_number",
        "file_name",
        "question",
        "program_answer",
        "original_answer",
    ]:
        print(f"{field}: {example.get(field)}")

    # 3. Display a preview of the original context
    print("\nORIGINAL CONTEXT PREVIEW:")
    print(str(example.get("context", ""))[:1200])

    print("\nQUESTION:", example["question"])
    print("GOLD CONTEXT:", example["context_id"])
    print("GOLD ANSWER:", example["program_answer"])

    # 4. Load the retriever
    retriever = Retriever()

    # 5. Find all chunks belonging to the gold context
    matching = [
        chunk
        for chunk in retriever.chunks
        if chunk["context_id"] == str(example["context_id"])
    ]

    print("\nCHUNKS FOR GOLD CONTEXT:", len(matching))

    # 6. Display previews of the first five chunks
    for chunk in matching[:5]:
        print("\nID:", chunk["chunk_id"])
        print("Company:", chunk["company"])
        print("Year:", chunk["year"])
        print("Page:", chunk["page"])
        print("Type:", chunk["type"])
        print("Text:", chunk["text"][:350])

    # 7. Display the complete table chunk
    print("\nCOMPLETE TABLE CHUNK:")

    table_chunk = None

    for chunk in matching:
        if chunk["type"] == "table":
            table_chunk = chunk
            print("\nID:", chunk["chunk_id"])
            print(chunk["text"])

    if table_chunk is None:
        print("No table chunk found for this context.")
        return

    # 8. Test calculation extraction using the original context
    # and the complete table chunk
    print("\nCALCULATION EXTRACTION TEST:")

    question = example["question"]
    evidence = (
        str(example.get("context", ""))
        + "\n"
        + table_chunk["text"]
    )

    try:
        program = extract_program(question, evidence)

        print("Extracted program:", program)

        if program is not None:
            result = execute(program)
            print("Calculated result:", result)
        else:
            print("No calculation program was extracted.")

    except Exception as error:
        print("Calculation test failed:")
        print(type(error).__name__, str(error))

    # 9. Show the expected answer for comparison
    print("\nEXPECTED ANSWER:", example["program_answer"])
    print("EXPECTED CALCULATION: 637 / 5 = 127.4")


if __name__ == "__main__":
    main()