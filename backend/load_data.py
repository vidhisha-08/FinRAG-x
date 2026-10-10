
from datasets import load_dataset, get_dataset_config_names

DATASET_NAME = "G4KMU/t2-ragbench"
CONFIG_NAME = "FinQA"


def main():
    # 1. Check which dataset configurations are available
    print("Checking available dataset configurations...")
    configs = get_dataset_config_names(DATASET_NAME)
    print("Available configurations:", configs)

    if CONFIG_NAME not in configs:
        raise ValueError(
            f"Configuration '{CONFIG_NAME}' not found. "
            f"Available configurations: {configs}"
        )

    # 2. Load the selected configuration
    print(f"\nLoading configuration: {CONFIG_NAME}")
    ds = load_dataset(DATASET_NAME, CONFIG_NAME)

    # 3. Inspect every available split
    print("\nAvailable dataset splits:")

    for split_name, split_data in ds.items():
        print(f"\nSplit: {split_name}")
        print("Number of examples:", len(split_data))
        print("Columns:", split_data.column_names)

        if len(split_data) == 0:
            print("This split is empty.")
            continue

        # 4. Inspect one example without printing a potentially
        # very large financial document
        example = split_data[0]

        print("Example field names:", list(example.keys()))

        if "id" in example:
            print("Example ID:", example["id"])

        if "question" in example:
            print("Example question:", example["question"])

        if "context_id" in example:
            print("Context ID:", example["context_id"])

        if "file_name" in example:
            print("Source file:", example["file_name"])

        if "context" in example:
            context = example["context"]
            print("Context type:", type(context).__name__)
            print(
                "Context preview:",
                str(context)[:300].replace("\n", " ")
            )


if __name__ == "__main__":
    main()