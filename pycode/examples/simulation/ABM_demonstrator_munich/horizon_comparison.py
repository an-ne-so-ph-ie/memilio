# # Horizont 63 / 70 / 77 und Wahl der Ziele (aus den Dateien von Generation 0, ohne neue Simulation)

import os
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import testing_different_goals_framework as tdgf

output_folder = "/storage/users/anne_sophie/screening_kappa_2.2_12_seeds_h77_d19_notest"
plot_folder = os.path.join(os.path.dirname(os.path.abspath(tdgf.__file__)), "plots_horizon")
os.makedirs(plot_folder, exist_ok=True)
horizons = [63, 70, 77]
reference_horizon = 77                       # so lange wurde simuliert
health_objectives = ["f_I", "f_ED", "f_EYLL"]
cost_objectives = ["f_W", "f_S_A"]
icu_beds = 387                               # Quelle noch prüfen
icu_shares = [0.15, 0.2, 0.25]
end_days_for_incidence = 3                   # letzte Tage vor dem Horizont für "ist die Welle vorbei"
recompute = False                            # True: Ziele neu aus den Dateien rechnen, auch wenn by_horizon.csv existiert
number_of_processes = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

per_seed = pd.read_csv(os.path.join(output_folder, "per_seed.csv"))
per_seed = per_seed[per_seed["generation"] == 0]
cache_file = os.path.join(output_folder, "by_horizon.csv")
death_probability = {seed: tdgf.death_probability_per_age_group(seed) for seed in per_seed["seed_index"].unique()}


def set_horizon(end_day):
    # alle Größen, die das Framework beim Import aus evaluation_end_day ableitet
    tdgf.evaluation_end_day = end_day
    tdgf.start_hour, tdgf.end_hour = tdgf.evaluation_start_day * 24, end_day * 24
    tdgf.evaluated_days = np.arange(tdgf.evaluation_start_day, end_day)


def objectives_for_all_horizons(job):
    # liest die Dateien eines Laufs einmal und rechnet die Ziele für jeden Horizont
    strategy, seed_index, number_to_save = job
    output_prefix = os.path.join(output_folder, str(number_to_save))
    compartments = pd.read_csv(f"{output_prefix}_comps_damping_compact_version.csv", sep=" ")
    presence = np.load(f"{output_prefix}_presence.npy")
    infection_paths = pd.read_csv(f"{output_prefix}_infection_paths_damping.txt", sep=r"\s+")
    infection_hours = infection_paths["S"].to_numpy()
    daily_new_infections = np.bincount((infection_hours[infection_hours > 0] // 24).astype(int), minlength=reference_horizon + 1)
    rows = []
    for horizon in horizons:
        set_horizon(horizon)
        values = tdgf.compute_objectives(infection_paths, compartments, presence, None, death_probability[seed_index])
        end_incidence = daily_new_infections[horizon - end_days_for_incidence:horizon].mean()
        rows.append({"horizon": horizon, "strategy": strategy, "seed_index": seed_index, **values,
                     "end_incidence_share": end_incidence / daily_new_infections[:reference_horizon].max()})
    return rows


def non_dominated(frame, health, cost):
    points = frame[[health, cost]].to_numpy()
    dominated = [((points <= point).all(axis=1) & (points < point).any(axis=1)).any() for point in points]
    return frame.index[~np.array(dominated)]


def paired_noise_and_signal(frame, objective):
    # pro Strategiepaar: log-Verhältnis über dieselben Partikel; Rauschen = Standardfehler, Signal = |Mittel|
    table = np.log(frame.pivot(index="strategy", columns="seed_index", values=objective).to_numpy())
    first, second = np.triu_indices(len(table), k=1)
    differences = table[first] - table[second]
    noise = differences.std(axis=1, ddof=1) / np.sqrt(differences.shape[1])
    signal = np.abs(differences.mean(axis=1))
    return np.median(noise), np.median(signal)

if os.path.exists(cache_file) and not recompute:
    by_horizon = pd.read_csv(cache_file)
    print("geladen:", cache_file)
else:
    jobs = list(per_seed[["strategy", "seed_index", "number_to_save"]].itertuples(index=False, name=None))
    start = time.time()
    with Pool(number_of_processes) as pool:
        results = pool.map(objectives_for_all_horizons, jobs)
    by_horizon = pd.DataFrame([row for rows in results for row in rows])
    by_horizon.to_csv(cache_file, index=False)
    print(f"{len(jobs)} Läufe in {time.time() - start:.0f} s mit {number_of_processes} Prozessen, gespeichert in {cache_file}")

objectives = [name for name in tdgf.objective_candidates if name in by_horizon.columns]
means = by_horizon.groupby(["horizon", "strategy"])[objectives + ["end_incidence_share"]].mean()

# Kontrolle: der Simulationshorizont muss per_seed.csv exakt reproduzieren
reproduced = by_horizon[by_horizon["horizon"] == reference_horizon].sort_values(["strategy", "seed_index"])
original = per_seed.sort_values(["strategy", "seed_index"])
for objective in objectives:
    assert np.allclose(reproduced[objective].to_numpy(), original[objective].to_numpy(), equal_nan=True), objective
print(f"Horizont {reference_horizon} reproduziert per_seed.csv für alle Ziele.")

print("\n1) Spannweite über die Strategien (Mittel über Seeds)")
for horizon in horizons:
    print(f"\nHorizont {horizon}")
    print(means.loc[horizon, objectives].describe().loc[["mean", "min", "max"]].T.round(3).to_string())

print("\n2) Ist die Welle vorbei? Neue Infektionen pro Tag in den letzten "
      f"{end_days_for_incidence} Tagen relativ zum Maximum (über Strategien)")
print(means["end_incidence_share"].unstack(level=0).describe().loc[["mean", "min", "max"]].round(3).to_string())

print("\n3) Nachlauf: was nach dem kürzeren Horizont noch dazukommt, relativ zur Spannweite bei diesem Horizont")
for objective in health_objectives + cost_objectives:
    values = means[objective].unstack(level=0)
    tail = pd.DataFrame({f"{horizon}->{reference_horizon}": (values[reference_horizon] - values[horizon])
                         / (values[horizon].max() - values[horizon].min())
                         for horizon in horizons if horizon != reference_horizon})
    print(f"\n{objective}")
    print(tail.describe().loc[["mean", "std", "min", "max"]].round(3).to_string())

print(f"\n4) Rangfolge: Spearman jedes Ziels zwischen Horizont und {reference_horizon}")
rank_rows = {horizon: {objective: means.loc[horizon, objective].corr(means.loc[reference_horizon, objective], method="spearman")
                       for objective in objectives} for horizon in horizons if horizon != reference_horizon}
print(pd.DataFrame(rank_rows).round(3).to_string())

print("\n5) Rauschen gegen Signal (gepaart über dieselben Partikel, Median über Strategiepaare)")
noise_rows = []
for horizon in horizons:
    for objective in objectives:
        noise, signal = paired_noise_and_signal(by_horizon[by_horizon["horizon"] == horizon], objective)
        noise_rows.append({"horizon": horizon, "objective": objective, "noise": noise, "signal": signal,
                           "signal_to_noise": signal / noise})
print(pd.DataFrame(noise_rows).pivot(index="objective", columns="horizon", values="signal_to_noise").round(2).to_string())

print("\n6) Zielpaare: Spearman, Zahl der nicht dominierten Strategien, Überschneidung mit der Front bei "
      f"{reference_horizon}")
pair_rows = []
for health in health_objectives:
    for cost in cost_objectives:
        reference_front = set(non_dominated(means.loc[reference_horizon], health, cost))
        for horizon in horizons:
            front = set(non_dominated(means.loc[horizon], health, cost))
            pair_rows.append({"pair": f"{health} / {cost}", "horizon": horizon,
                              "spearman": means.loc[horizon, health].corr(means.loc[horizon, cost], method="spearman"),
                              "non_dominated": len(front), "shared_with_77": len(front & reference_front)})
print(pd.DataFrame(pair_rows).round(2).to_string(index=False))

print(f"\n7) ICU-Grenze als Anteil von {icu_beds} Betten: Zahl der zulässigen Strategien (von {per_seed['strategy'].nunique()})")
print("   je nach Statistik über die 12 Seeds; nur 'mean' entspricht dem Constraint in der Main (Mittelwert - Grenze <= 0)")
icu_rows = []
for horizon in horizons:
    icu = by_horizon[by_horizon["horizon"] == horizon].groupby("strategy")["peak_icu"]
    statistics = {"mean": icu.mean(), "q75": icu.quantile(0.75), "q90": icu.quantile(0.9), "max": icu.max()}
    for share in icu_shares:
        limit = share * icu_beds
        icu_rows.append({"horizon": horizon, "share": share, "limit": round(limit, 1),
                         **{name: int((values <= limit).sum()) for name, values in statistics.items()}})
print(pd.DataFrame(icu_rows).to_string(index=False))
print(f"   peak_icu über die Strategien (Mittel über Seeds): {means.loc[reference_horizon, 'peak_icu'].min():.1f} "
      f"bis {means.loc[reference_horizon, 'peak_icu'].max():.1f}")

print("\n8) Wie stark wirken die Strategien pro Partikel? Spannweite max/min von f_I über die Strategien")
particles = pd.read_csv(os.path.join(tdgf.input_directory, "pop8_sel_particles.csv"), index_col=0)
reference_runs = by_horizon[by_horizon["horizon"] == reference_horizon]
spread = reference_runs.groupby("seed_index")["f_I"].agg(lambda values: values.max() / values.min())
particle_check = particles.loc[spread.index, ["kappa", "damp_time", "damp_lvl"]].assign(
    kappa_x_damp_lvl=lambda frame: frame["kappa"] * frame["damp_lvl"], spread=spread)
print(particle_check.sort_values("damp_lvl").round(3).to_string())
print(particle_check.corr(method="spearman")["spread"].round(2).to_string())

print("\n9) Toleranz fürs Abbruchkriterium: Rauschen eines Strategieunterschieds in Einheiten der Frontbreite")
print("   (MultiObjectiveSpaceTermination misst Verschiebungen normiert auf die Frontbreite; delta = Maximum über die Ziele)")
tolerance_rows = []
for health in health_objectives:
    for cost in cost_objectives:
        for horizon in horizons:
            horizon_means = means.loc[horizon]
            front = horizon_means.loc[non_dominated(horizon_means, health, cost)]
            row = {"pair": f"{health} / {cost}", "horizon": horizon}
            for role, objective in [("health", health), ("cost", cost)]:
                relative_noise, _ = paired_noise_and_signal(by_horizon[by_horizon["horizon"] == horizon], objective)
                absolute_noise = relative_noise * front[objective].mean()          # log-Verhältnis -> absolute Einheiten
                front_width = front[objective].max() - front[objective].min()
                row[f"noise_{role}"] = absolute_noise
                row[f"front_width_{role}"] = front_width
                row[f"normalized_{role}"] = absolute_noise / front_width if front_width > 0 else np.nan
            row["normalized_noise_max"] = np.nanmax([row["normalized_health"], row["normalized_cost"]])
            tolerance_rows.append(row)
tolerance_table = pd.DataFrame(tolerance_rows)
print(tolerance_table.round(4).to_string(index=False))

for health in health_objectives:
    for cost in cost_objectives:
        figure, axis = plt.subplots(figsize=(6, 4.5))
        for horizon, marker in zip(horizons, ["o", "s", "^"]):
            front = means.loc[horizon].loc[non_dominated(means.loc[horizon], health, cost)].sort_values(cost)
            axis.plot(front[cost], front[health], marker=marker, label=f"horizon {horizon} days")
        axis.set_xlabel(cost)
        axis.set_ylabel(health)
        axis.legend()
        figure.tight_layout()
        figure.savefig(os.path.join(plot_folder, f"fronts_{health}_{cost}.png"), dpi=150)
        plt.close(figure)
print("Plots in", plot_folder)