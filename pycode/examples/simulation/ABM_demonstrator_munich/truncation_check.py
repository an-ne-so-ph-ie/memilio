
import os
import json
from multiprocessing import Pool
import numpy as np
import pandas as pd
import testing_different_goals_framework as tdgf
import abm_demonstrator_munich

generation_0_folder = "/storage/users/anne_sophie/screening_kappa_2.2_12_seeds_h77_d19_notest"
check_folder = "/storage/users/anne_sophie/truncation_check"
strategy_file_number = 1            # Strategie aus Generation 0 (Datei 1_strategy.json)
short_days, long_days = 63, 77
compared_hours = short_days * 24

with open(os.path.join(generation_0_folder, f"{strategy_file_number}_strategy.json")) as file:
    stored_strategy = json.load(file)
seed_index = stored_strategy["seed_index"]
change_days, periods = tdgf.decode_strategy(stored_strategy)
print(f"Seed {seed_index}, Wechseltage {change_days}, Perioden {periods}")


def simulate(simulation_days):
    # eigener Ordner pro Länge, damit sich die Dateien nicht überschreiben
    output_folder = os.path.join(check_folder, f"h{simulation_days}")
    os.makedirs(output_folder, exist_ok=True)
    abm_demonstrator_munich.run_abm_simulation(seed_index, 1, output_folder, simulation_days=simulation_days,
                                               change_days=change_days, periods=periods)
    return output_folder


def read_run(output_folder):
    prefix = os.path.join(output_folder, "1")
    compartments = pd.read_csv(f"{prefix}_comps_damping_compact_version.csv", sep=" ")
    presence = np.load(f"{prefix}_presence.npy")
    infection_paths = pd.read_csv(f"{prefix}_infection_paths_damping.txt", sep=r"\s+")
    return compartments, presence, infection_paths


def infections_before(infection_paths, hour):
    # S = Stunden im Zustand S = Ansteckungszeitpunkt (wer bei 0 schon infiziert war, hat S = 0)
    infected = infection_paths[infection_paths["S"] < hour]
    return infected[["Agent_id", "S"]].sort_values(["Agent_id", "S"]).to_numpy()


if __name__ == "__main__":
    with Pool(2) as pool:
        short_folder, long_folder = pool.map(simulate, [short_days, long_days])

    short_compartments, short_presence, short_paths = read_run(short_folder)
    long_compartments, long_presence, long_paths = read_run(long_folder)

    # 1. Zustände pro Stunde bis Tag 63
    short_part = short_compartments[short_compartments["t"] <= compared_hours].reset_index(drop=True)
    long_part = long_compartments[long_compartments["t"] <= compared_hours].reset_index(drop=True)
    compartments_equal = short_part.shape == long_part.shape and (short_part.to_numpy() == long_part.to_numpy()).all()
    print(f"Zustände bis Stunde {compared_hours}: {'gleich' if compartments_equal else 'VERSCHIEDEN'} "
          f"({len(short_part)} Zeilen)")

    # 2. Anwesenheit pro Ortstyp und Altersgruppe bis Tag 63 (Zeile h = Stunde h)
    rows = compared_hours + 1
    presence_equal = (short_presence[:rows].shape == long_presence[:rows].shape
                      and (short_presence[:rows] == long_presence[:rows]).all())
    print(f"Anwesenheit bis Stunde {compared_hours}: {'gleich' if presence_equal else 'VERSCHIEDEN'}")

    # 3. Wer sich wann angesteckt hat, bis Tag 63
    short_infections = infections_before(short_paths, compared_hours)
    long_infections = infections_before(long_paths, compared_hours)
    infections_equal = short_infections.shape == long_infections.shape and (short_infections == long_infections).all()
    print(f"Ansteckungen vor Stunde {compared_hours}: {'gleich' if infections_equal else 'VERSCHIEDEN'} "
          f"({len(short_infections)} gegen {len(long_infections)})")

    # 4. Ziele bei Auswertung bis Tag 63, einmal aus jedem Lauf
    tdgf.evaluation_end_day = short_days
    tdgf.start_hour, tdgf.end_hour = tdgf.evaluation_start_day * 24, short_days * 24
    tdgf.evaluated_days = np.arange(tdgf.evaluation_start_day, short_days)
    death_probability = tdgf.death_probability_per_age_group(seed_index)
    objectives = pd.DataFrame({
        f"simuliert {short_days} Tage": tdgf.objectives_of_run(short_folder, 1, death_probability),
        f"simuliert {long_days} Tage": tdgf.objectives_of_run(long_folder, 1, death_probability)})
    objectives["gleich"] = np.isclose(objectives.iloc[:, 0], objectives.iloc[:, 1])
    print(f"\nZiele, ausgewertet bis Tag {short_days}:")
    print(objectives.to_string())
    print("(f_D und f_YLL sind nur gleich, wenn die Korrektur der Todeszeit in compute_objectives schon eingebaut ist)")

    if compartments_equal and presence_equal and infections_equal:
        print("\nErgebnis: Die ersten 63 Tage sind identisch. Generation 0 darf für Horizont 63 wiederverwendet werden.")
    else:
        print("\nErgebnis: Unterschiede in den ersten 63 Tagen. Generation 0 NICHT für Horizont 63 wiederverwenden.")