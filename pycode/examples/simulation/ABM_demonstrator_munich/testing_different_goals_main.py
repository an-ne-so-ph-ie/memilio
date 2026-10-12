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
 
import testing_different_goals_framework as tdgf
 
objectives = ["f_I", "f_W"]              
constraints = {}                         # TODO AS
population_size = 48
number_of_generations = 1                # 1 = only the start population (generation 0)
 
 
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
    def __init__(self):

        super().__init__(vars=pymoo_variables(), n_obj=len(objectives), n_ieq_constr=len(constraints))
        self.seed_indices = tdgf.selected_seed_indices()
        self.candidate_means = {}              
        self.generation = 0
        self.number_to_save_counter = 0
        self.path_output_folder = os.path.join(os.getcwd(), "output", f"nsgaii_testing_goals{datetime.date.today()}")
        os.makedirs(self.path_output_folder, exist_ok=True)
        self.log_file = os.path.join(self.path_output_folder, "per_seed.csv")
        self.available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        print(f"{self.available_cpus} available CPUs, seeds {self.seed_indices}, "
              f"output in {self.path_output_folder}", flush=True)
 
    def _evaluate(self, population, output, *args, **kwargs):
        jobs = []
        for strategy_index, strategy in enumerate(population):     
            for seed_index in self.seed_indices:
                self.number_to_save_counter += 1
                jobs.append((strategy_index, seed_index, self.number_to_save_counter, strategy, self.path_output_folder))
        print(f"Generation {self.generation}: {len(population)} strategies, {len(jobs)} jobs.", flush=True)
 
        results = [{} for _ in jobs]
        multiprocessing_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.available_cpus, mp_context=multiprocessing_context,  max_tasks_per_child=1) as executor:
            jobs_to_execute = {executor.submit(run_one_abm_process, job): job_index for job_index, job in enumerate(jobs)}
            for job_to_execute in as_completed(jobs_to_execute):
                job_index = jobs_to_execute[job_to_execute]
                try:
                    results[job_index] = job_to_execute.result()
                except Exception as error:      # the process itself died, e.g. out of memory
                    print(f"[PROCESS FAILED] simulation={jobs[job_index][2]}: {error!r}", flush=True)
 
        rows = [{"generation": self.generation, "strategy": job[0], "seed_index": job[1], "number_to_save": job[2], **result} for job, result in zip(jobs, results)]
        per_seed = pd.DataFrame(rows).reindex(columns=["generation", "strategy", "seed_index", "number_to_save"] + tdgf.objective_candidates)
        per_seed.to_csv(self.log_file, mode="a", header=not os.path.exists(self.log_file), index=False)
        number_failed = per_seed[objectives[0]].isna().sum()
        if number_failed:
            print(f"There are {number_failed} failed simulations.", flush=True)
 
        mean_per_strategy = per_seed.groupby("strategy")[tdgf.objective_candidates].mean()   
        mean_per_strategy = mean_per_strategy.reindex(range(len(population)))
        for strategy, (_, means) in zip(population, mean_per_strategy.iterrows()):
            self.candidate_means[strategy_key(strategy)] = means
        output["F"] = np.nan_to_num(mean_per_strategy[objectives].to_numpy(), nan=1e12)   
        if constraints:
            violations = np.column_stack([mean_per_strategy[objective] - limit for objective, limit in constraints.items()])
            output["G"] = np.nan_to_num(violations, nan=1e12)
        print(f"Finished generation {self.generation}.", flush=True)
        self.generation += 1
 
 
if __name__ == "__main__":
    start_nsgaii = time.time()
    abm_problem = AbmProblem()
    run_settings = {"objectives": objectives, "constraints": constraints, "population_size": population_size,
                    "number_of_generations": number_of_generations, "seed_indices": abm_problem.seed_indices}
    with open(os.path.join(abm_problem.path_output_folder, "run_settings.json"), "w") as file:
        json.dump(run_settings, file, indent=1)
    repair = ChangeDayRepair()
    algorithm = MixedVariableGA(pop_size=population_size, survival=RankAndCrowding(), repair=repair,
                                mating=MixedVariableMating(selection=TournamentSelection(func_comp=binary_tournament),
                                                           repair=repair,
                                                           eliminate_duplicates=MixedVariableDuplicateElimination()),
                                output=MultiObjectiveOutput(), advance_after_initial_infill=True)
    algorithm.tournament_type = "comp_by_dom_and_crowding"     
    result_abm = minimize(abm_problem, algorithm, ("n_gen", number_of_generations), seed=1, verbose=True)
    print("Time:", time.time() - start_nsgaii)
 
    
    front_strategies = list(np.atleast_1d(result_abm.X))
    pareto_front = pd.DataFrame(front_strategies)[tdgf.variable_names]
    candidates = pd.DataFrame([abm_problem.candidate_means[strategy_key(strategy)] for strategy in front_strategies])
    pareto_front = pd.concat([candidates.reset_index(drop=True), pareto_front], axis=1)
    pareto_front.to_csv(os.path.join(abm_problem.path_output_folder, "front.csv"), index=False)
    print("Pareto front:\n", pareto_front[objectives].to_string(), flush=True)
 
