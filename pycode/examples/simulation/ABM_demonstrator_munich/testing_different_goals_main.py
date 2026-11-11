#############################################################################
# Copyright (C) 2020-2021 German Aerospace Center (DLR-SC)
#
# Authors: Martin J. Kuehn, Wadim Koslow, Daniel Abele
#
# Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#############################################################################

import datetime
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
 
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.mixed import MixedVariableDuplicateElimination, MixedVariableGA, MixedVariableMating
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.variable import Choice, Integer, Real
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.core.population import Population
 
import testing_different_goals_framework as tdgf
 
objectives = ["f_I", "f_S_A"]              
constraints = {}                         # TODO AS
population_size = 48
#constraints = {"test_capacity_ratio": 1.0}    # an keinem Tag mehr Tests als freie Kapa
new_generations = int(os.environ.get("NEW_GENERATIONS", 5))
number_of_generations = 1                # 1 = only the start population (generation 0)
pymoo_seed = 1
generation_0_folder = os.environ["GENERATION_0_FOLDER"]          # Screening, 77 Tage simuliert
output_folder = os.environ["OUTPUT_FOLDER"]

 
def run_one_abm_process(job):
    
    strategy_index, seed_index, number_to_save, strategy, path_output_folder = job
    print(f"[START] simulation={number_to_save}, strategy={strategy_index}, seed={seed_index}", flush=True)
    try:
        result = tdgf.simulate_strategy(strategy, seed_index, number_to_save, path_output_folder)
    except Exception as error:
        print(f"[FAILED] simulation={number_to_save}, strategy={strategy_index}, seed={seed_index}: {error!r}",
              flush=True)
        return {}
    print(f"[DONE] simulation={number_to_save}, strategy={strategy_index}, seed={seed_index}, "
          f"f_I={result['f_I']:.0f}", flush=True)
    return result


def run_jobs(jobs, available_cpus): # failed jobs run again
    results = [{} for _ in jobs]
    multiprocessing_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=available_cpus, mp_context=multiprocessing_context,
                             max_tasks_per_child=1) as executor:
        jobs_to_execute = {executor.submit(run_one_abm_process, job): job_index for job_index, job in enumerate(jobs)}
        for job_to_execute in as_completed(jobs_to_execute):
            job_index = jobs_to_execute[job_to_execute]
            try:
                results[job_index] = job_to_execute.result()
            except Exception as error:      # the process itself died, e.g. out of memory
                print(f"[PROCESS FAILED] simulation={jobs[job_index][2]}: {error!r}", flush=True)
    return results

def objectives_of_stored_run(job):
    # Generation 0: Ziele neu berechnen weil für 77 Tage simuliert
    seed_index, number_to_save = job
    return tdgf.objectives_of_run(generation_0_folder, number_to_save, tdgf.death_probability_per_age_group(seed_index))

 
 
def pymoo_variables():
    variables = {}
    for name, (variable_type, *settings) in tdgf.decision_variables.items():
        if variable_type == "real":
            variables[name] = Real(bounds=tuple(settings))
        elif variable_type == "integer":
            variables[name] = Integer(bounds=tuple(settings))
        else:
            variables[name] = Choice(options=settings[0])
    return variables
 
 
def strategy_key(strategy):
    return tuple(str(strategy[name]) for name in tdgf.variable_names)

def load_generation_0(available_cpus):
    # Strategien aus x_strategy.json, Ziele neu berechnen
    screening = pd.read_csv(os.path.join(generation_0_folder, "per_seed.csv"))
    screening = screening[screening["generation"] == 0].sort_values(["strategy", "seed_index"]).reset_index(drop=True)
    strategies = []
    for strategy_index, runs in screening.groupby("strategy"):
        with open(os.path.join(generation_0_folder, f"{runs['number_to_save'].iloc[0]}_strategy.json")) as file:
            stored = json.load(file)
        strategies.append({name: stored[name] for name in tdgf.variable_names})
    jobs = list(zip(screening["seed_index"], screening["number_to_save"]))
    with mp.get_context("spawn").Pool(available_cpus) as pool:
        results = pool.map(objectives_of_stored_run, jobs)
    per_seed = pd.concat([screening[["strategy", "seed_index", "number_to_save"]], pd.DataFrame(results)], axis=1)
    per_seed.insert(0, "generation", 0)
    return strategies, per_seed.reindex(columns=["generation", "strategy", "seed_index", "number_to_save"] + tdgf.objective_candidates)

 
class ChangeDayRepair(Repair):
    """ check first and second period """
    def _do(self, problem, population, **kwargs):
        for strategy in population:
            first_day, second_day = sorted((strategy["change_day_1"], strategy["change_day_2"]))
            if second_day - first_day < tdgf.minimum_period_days:
                second_day = min(first_day + tdgf.minimum_period_days, tdgf.latest_change_day)
                first_day = second_day - tdgf.minimum_period_days
            strategy["change_day_1"], strategy["change_day_2"] = first_day, second_day
        return population
 
 
class AbmProblem(Problem):
    def __init__(self, cached_means):
        super().__init__(vars=pymoo_variables(), n_obj=len(objectives), n_ieq_constr=len(constraints))
        self.seed_indices = tdgf.selected_seed_indices()
        self.candidate_means = cached_means        # strategy_key -> Mittel über Seeds; bekannte Strategien werden nicht neu simuliert
        self.generation = 0
        self.number_to_save_counter = 0
        self.path_output_folder = output_folder
        self.log_file = os.path.join(output_folder, "per_seed.csv")
        self.available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

 
    def _evaluate(self, population, output, *args, **kwargs):
        jobs = []
        for strategy_index, strategy in enumerate(population):     
            for seed_index in self.seed_indices:
                self.number_to_save_counter += 1
                jobs.append((strategy_index, seed_index, self.number_to_save_counter, strategy, self.path_output_folder))
        print(f"Generation {self.generation}: {len(population)} strategies, {len(jobs)} jobs.", flush=True)

        if jobs: 
            results = run_jobs(jobs, self.available_cpus)
            failed = [job_index for job_index, result in enumerate(results) if not result]
            if failed:
                print(f"{len(failed)} simulations failed, running them once more.", flush=True)
                results_again = run_jobs([jobs[job_index] for job_index in failed], min(self.available_cpus, len(failed)))
                for job_index, result in zip(failed, results_again):
                    results[job_index] = result
                failed = [job_index for job_index, result in enumerate(results) if not result]
            if failed:
                failed_file = os.path.join(self.path_output_folder, f"failed_generation_{self.generation}.json")
                with open(failed_file, "w") as file:
                    json.dump([{"strategy": jobs[job_index][0], "seed_index": int(jobs[job_index][1]),
                                "number_to_save": jobs[job_index][2],
                                **{name: tdgf.to_python_value(value) for name, value in jobs[job_index][3].items()}}
                            for job_index in failed], file, indent=1)
                raise RuntimeError(f"{len(failed)} simulations failed in generation {self.generation}, see {failed_file}")

    
            rows = [{"generation": self.generation, "strategy": job[0], "seed_index": job[1], "number_to_save": job[2], **result} for job, result in zip(jobs, results)]
            per_seed = pd.DataFrame(rows).reindex(columns=["generation", "strategy", "seed_index", "number_to_save"] + tdgf.objective_candidates)
            per_seed.to_csv(self.log_file, mode="a", header=not os.path.exists(self.log_file), index=False)
            number_failed = per_seed[objectives[0]].isna().sum()
            if number_failed:
                print(f"There are {number_failed} failed simulations.", flush=True)
    
            mean_per_strategy = per_seed.groupby("strategy")[tdgf.objective_candidates].mean()
            for strategy_index, means in mean_per_strategy.iterrows():
                self.candidate_means[strategy_key(population[strategy_index])] = means

        mean_per_strategy = mean_per_strategy.reindex(range(len(population)))
        if mean_per_strategy[objectives].isna().any().any():
            raise RuntimeError(f"NaN in objectives in generation {self.generation}")
        #for strategy, (_, means) in zip(population, mean_per_strategy.iterrows()):
        #    self.candidate_means[strategy_key(strategy)] = means
        output["F"] = mean_per_strategy[objectives].to_numpy()
        if constraints:
            output["G"] = np.column_stack([mean_per_strategy[objective] - limit for objective, limit in constraints.items()])

        print(f"Finished generation {self.generation}.", flush=True)
        self.generation += 1

def save_population(algorithm):
    # Speicher population nach der Auswahl: Strategie, alle Ziele (Mittel über Seeds) und Rang
    problem = algorithm.problem
    generation = problem.generation - 1
    population = algorithm.pop
    rows = []
    for strategy, rank in zip(population.get("X"), population.get("rank")):
        means = problem.candidate_means[strategy_key(strategy)]
        rows.append({**{name: tdgf.to_python_value(strategy[name]) for name in tdgf.variable_names},
                     **means.to_dict(), "rank": int(rank)})
    pd.DataFrame(rows).to_csv(os.path.join(output_folder, f"population_generation_{generation}.csv"), index=False)
    return generation

def latest_saved_generation():
    numbers = [int(name[len("population_generation_"):-len(".csv")]) for name in os.listdir(output_folder)
               if name.startswith("population_generation_") and name.endswith(".csv")]
    return max(numbers) if numbers else None

def start_from_generation_0():
    if os.path.exists(os.path.join(output_folder, "per_seed.csv")):
        raise RuntimeError(f"per_seed.csv exists in {output_folder} but no saved population, choose another OUTPUT_FOLDER")
    os.makedirs(output_folder, exist_ok=True)
    available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    strategies, per_seed_generation_0 = load_generation_0(available_cpus)
    per_seed_generation_0.to_csv(os.path.join(output_folder, "per_seed.csv"), index=False)
    means = per_seed_generation_0.groupby("strategy")[tdgf.objective_candidates].mean()
    cached_means = {strategy_key(strategy): means.loc[strategy_index] for strategy_index, strategy in enumerate(strategies)}
    print(f"Generation 0 loaded from {generation_0_folder}: {len(strategies)} strategies, "
          f"evaluated until day {tdgf.evaluation_end_day}.", flush=True)
    return strategies, cached_means, 0, 0

def continue_from_saved_generation(generation):
    # letzte gespeicherte Population einlesen, Läufe einer abgebrochenen späteren Generation aus per_seed.csv entfernen
    population = pd.read_csv(os.path.join(output_folder, f"population_generation_{generation}.csv"), float_precision="round_trip")
    strategies = [{name: tdgf.to_python_value(row[name]) for name in tdgf.variable_names} for _, row in population.iterrows()]
    for strategy in strategies:
        for name, (variable_type, *settings) in tdgf.decision_variables.items():
            if variable_type == "integer":
                strategy[name] = int(strategy[name])
    cached_means = {strategy_key(strategy): population.loc[index, tdgf.objective_candidates]
                    for index, strategy in enumerate(strategies)}
    log_file = os.path.join(output_folder, "per_seed.csv")
    per_seed = pd.read_csv(log_file)
    unfinished = per_seed["generation"] > generation
    if unfinished.any():
        print(f"Removing {unfinished.sum()} rows of unfinished generations from per_seed.csv", flush=True)
        per_seed[~unfinished].to_csv(log_file, index=False)
    simulated_here = per_seed.loc[~unfinished & (per_seed["generation"] > 0), "number_to_save"]
    last_number_to_save = int(simulated_here.max()) if len(simulated_here) else 0
    print(f"Continue from generation {generation} ({len(strategies)} strategies).", flush=True)
    return strategies, cached_means, generation, last_number_to_save

def build_algorithm(strategies, cached_means, generation, last_number_to_save):
    problem = AbmProblem(cached_means)
    problem.generation = generation                       # die eingelesene Population wird als diese Generation gezählt
    problem.number_to_save_counter = last_number_to_save  # keine Dateien überschreiben
    repair = ChangeDayRepair()
    algorithm = MixedVariableGA(pop_size=population_size, sampling=Population.new(X=np.array(strategies, dtype=object)),
                                survival=RankAndCrowding(), repair=repair,
                                mating=MixedVariableMating(selection=TournamentSelection(func_comp=binary_tournament),
                                                           repair=repair,
                                                           eliminate_duplicates=MixedVariableDuplicateElimination()),
                                output=MultiObjectiveOutput(), advance_after_initial_infill=True)
    algorithm.tournament_type = "comp_by_dom_and_crowding"     # für binary_tournament
    remaining_generations = new_generations - generation
    algorithm.setup(problem, termination=MaximumGenerationTermination(1 + remaining_generations),
                    seed=pymoo_seed + generation, verbose=True)   # neuer Zufallsstrom pro Fortsetzung
    return algorithm, remaining_generations

def write_run_settings(problem):
    run_settings = {"objectives": objectives, "constraints": constraints, "population_size": population_size,
                    "pymoo_seed": pymoo_seed, "new_generations": new_generations,
                    "generation_0_folder": generation_0_folder, "seed_indices": problem.seed_indices,
                    "kappa_factor": tdgf.abm_demonstrator_munich.kappa_conversion_factor,
                    "particle_file": tdgf.abm_demonstrator_munich.particle_file,
                    "simulation_days": tdgf.simulation_days, "evaluation_end_day": tdgf.evaluation_end_day,
                    "earliest_change_day": tdgf.earliest_change_day, "latest_change_day": tdgf.latest_change_day,
                    "minimum_period_days": tdgf.minimum_period_days,
                    "contact_files": {location_type.name: list(entry) for location_type, entry
                                      in tdgf.abm_demonstrator_munich.contact_files.items()}}
    settings_file = os.path.join(output_folder, f"run_settings_from_generation_{problem.generation}.json")
    with open(settings_file, "w") as file:
        json.dump(run_settings, file, indent=1)

 
 
if __name__ == "__main__":
    start_time = time.time()
    os.makedirs(output_folder, exist_ok=True)
    saved_generation = latest_saved_generation()
    if saved_generation is None:
        start_state = start_from_generation_0()
    else:
        start_state = continue_from_saved_generation(saved_generation)
    algorithm, remaining_generations = build_algorithm(*start_state)
    if remaining_generations <= 0:
        raise SystemExit(f"Budget of {new_generations} generations already reached, increase NEW_GENERATIONS.")
    write_run_settings(algorithm.problem)
 
    while algorithm.has_next():
        algorithm.next()
        generation = save_population(algorithm)
        print(f"Saved population of generation {generation}, time {time.time() - start_time:.0f} s", flush=True)
 
    result = algorithm.result()
    front_strategies = list(np.atleast_1d(result.X))
    front = pd.DataFrame([algorithm.problem.candidate_means[strategy_key(strategy)] for strategy in front_strategies])
    front = pd.concat([front.reset_index(drop=True),
                       pd.DataFrame([{name: tdgf.to_python_value(strategy[name]) for name in tdgf.variable_names}
                                     for strategy in front_strategies])], axis=1)
    front.to_csv(os.path.join(output_folder, "front.csv"), index=False)
    print("Pareto front:\n", front[objectives].sort_values(objectives[1]).to_string(), flush=True)
    print("Time:", time.time() - start_time)
