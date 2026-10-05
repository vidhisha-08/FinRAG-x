import pandas as pd

res = pd.read_csv("../data/eval_results.csv")
print("Total:", len(res))
print("\nB right, D wrong:")
print(res[(res.B_correct) & (~res.D_correct)][["question", "gold", "D_refused", "D_calc_correct"]].to_string())
print("\nD right, B wrong:")
print(res[(~res.B_correct) & (res.D_correct)][["question", "gold"]].to_string())
print("\nD refused:")
print(res[res.D_refused][["question", "gold"]].to_string())