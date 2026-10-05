from datasets import load_dataset, get_dataset_config_names

print(get_dataset_config_names("G4KMU/t2-ragbench"))
ds = load_dataset("G4KMU/t2-ragbench", "FinQA")
split = list(ds.keys())[-1]

if __name__ == "__main__":
    print(split, ds[split].column_names)
    print(ds[split][0])