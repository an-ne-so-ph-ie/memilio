import json
import os
import pandas as pd
import testing_different_goals_framework as tdgf

screening = "output/screening_kappa_2.2_12_seeds"
test_folder = "output/history_check"
os.makedirs(test_folder, exist_ok=True)
with open(os.path.join(screening, "1_strategy.json")) as file:
    stored = json.load(file)
seed_index = stored.pop("seed_index")
result = tdgf.simulate_strategy(stored, seed_index, 1, test_folder)
reference = pd.read_csv(os.path.join(screening, "per_seed.csv")).query("number_to_save == 1").iloc[0]
for objective in ["f_I", "f_ED", "f_W", "f_S_A", "peak_icu"]:
    print(objective, result[objective], reference[objective])