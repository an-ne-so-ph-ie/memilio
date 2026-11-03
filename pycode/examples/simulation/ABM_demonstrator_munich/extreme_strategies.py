
import os
import pandas as pd
import testing_different_goals_framework as tdgf
from testing_different_goals_main import run_jobs


if __name__ == "__main__":  
    output_folder = os.environ.get("OUTPUT_FOLDER", os.path.join(os.getcwd(), "output", "extremes"))
    os.makedirs(output_folder, exist_ok=True)
    available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    nothing = {"home_office": 0.0, "school_closure": 0.0, "social_events": 0.0,
               "work_test": "none", "school_test": "none", "probability_test_work": 0.0, "probability_test_school": 0.0}
    settings = {
        "no_measures":      nothing,
        "home_office_only": {**nothing, "home_office": tdgf.maximum_home_office},
        "closures_only":    {**nothing, "home_office": tdgf.maximum_home_office, "school_closure": 1.0, "social_events": 1.0},
        "tests_only":       {**nothing, "work_test": "pcr", "school_test": "pcr", "probability_test_work": 1.0, "probability_test_school": 1.0},
    }
    settings["all_maximal"] = {**settings["closures_only"], **{key: value for key, value in settings["tests_only"].items() if "test" in key}}

    strategies = {name: {**{f"{key}_{period}": value for period in (1, 2) for key, value in values.items()},
                         "change_day_1": 25, "change_day_2": 45} for name, values in settings.items()}

    jobs = []
    for strategy_index, (name, strategy) in enumerate(strategies.items()):
        for seed_index in tdgf.selected_seed_indices():
            jobs.append((strategy_index, seed_index, len(jobs) + 1, strategy, output_folder))

    results = run_jobs(jobs, available_cpus)
    rows = [{"strategy": list(strategies)[job[0]], "seed_index": job[1], "number_to_save": job[2], **result}
            for job, result in zip(jobs, results)]
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_folder, "extremes_per_seed.csv"), index=False)
    print(table.groupby("strategy")[["f_I", "f_ED", "f_W", "f_S_A"]].mean().round(3).to_string())
    print("failed runs:", sum(1 for result in results if not result))