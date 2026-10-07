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
import argparse
import numpy as np
import os
import sys
import time

import pandas as pd
import argparse
import time
import sys
import matplotlib.pyplot as plt

import memilio.simulation as mio
import memilio.simulation.abm as mabm
dir(mabm)

import os
from concurrent.futures import ProcessPoolExecutor

import abm_demonstrator_munich as adm


import pymoo.gradient.toolbox as anp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import multiprocessing as mp

import datetime

def run_one_abm_process(job):
    """
    Executes one ABM simulation in a separate process.
    """
    sim_seed_index, number_to_save, parameter_for_closure, path_output_folder = job
    try:
        work_closure = parameter_for_closure[0]
        school_closure = parameter_for_closure[1]
        print(
            f"[START] simulation={number_to_save}, "
            f"seed={sim_seed_index}, "
            f"work={work_closure:.4f}, "
            f"school={school_closure:.4f}",
            flush=True,
        )
        print(parameter_for_closure)

        
        n_persons_abm = adm.run_abm_simulation(sim_seed_index, number_to_save, path_output_folder, work_closure, school_closure)
        df = pd.read_csv(os.path.join(path_output_folder, f"{number_to_save}_comps_damping_compact_version.csv"), sep=r"\s+")
        # TODO population size noch veränderbar machen
        infected_people = n_persons_abm - df["S"].iloc[-1]

        print(
            f"[DONE] simulation={number_to_save}, "
            f"seed={sim_seed_index}, "
            f"infected={infected_people}",
            flush=True,
        )

        return infected_people
    
    except Exception as e:
        print(f"Simulation {number_to_save}, seed {sim_seed_index}, x={parameter_for_closure} failed: {e!r}", flush=True)
        print(f"This is the exception thrown: {e}")
        return np.nan 

class nsgaii__on_abm_parallelized(Problem):
    def __init__(self):
        # initializing nsgaii number of goals and space
        super().__init__(n_var=2, n_obj=2, xl=(0.0,0.0), xu=(1.0,1.0))

        #initializing parameters for abm simulation 
        self.number_to_save_counter = 0
        self.number_process_repetition = 6
        print(f"Using {self.number_process_repetition} different seeds for ABM simulations and returning mean for nsgaii.")
        self.path_output_folder = os.path.join(os.getcwd(), f"output/output_workingclosure_nsgaii_with_processpoolexecutor_{datetime.date.today()}")
        os.makedirs(self.path_output_folder, exist_ok=True)
        mio.abm.set_log_level_warn()

        #initializing our parallelization
        self.available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        #self.max_workers = min(self.available_cpus, 1)
        print(f"{self.available_cpus} available CPUs")
        #print(f"Using {self.max_workers} Workers.")
        
    def _evaluate(self, x, out):
        number_individuals = len(x)
        print(f"We have {number_individuals} individuals (=parameters to test to find the pareto front).")
        
        jobs = []
        for individual_index in range(number_individuals):
            for sim_seed_index in range(1, self.number_process_repetition + 1):
                self.number_to_save_counter += 1
                number_to_save = self.number_to_save_counter
                jobs.append((sim_seed_index, number_to_save, x[individual_index], self.path_output_folder))

        print(f"We created {len(jobs)} jobs. We can now run the ABM simulations parallelized.")

        #TODO
        multiprocessing_context = mp.get_context("spawn")
        results = [np.nan] * len(jobs)

        with ProcessPoolExecutor(max_workers=self.available_cpus, mp_context = multiprocessing_context, max_tasks_per_child=1) as executor:
                jobs_to_execute = {executor.submit(run_one_abm_process, job): job for job in jobs}
                for job_to_execute in as_completed(jobs_to_execute):
                    job = jobs_to_execute[job_to_execute]
                    job_index = jobs.index(job)
                    try:
                        result = job_to_execute.result()
                        results[job_index] = result
                        # TO DO besser vermutlich: bissle andere programmierung mit i in der menge drin, sollte ich mir noch einmal anschauen
                    except Exception as e:
                        sim_seed_index_f, number_to_save_f, x_f, _ = job
                        print(f"[PROCESS FAILED] job_index={job_index}, simulation={number_to_save_f}, "
                        f"PID={os.getpid()}",
                        f"seed={sim_seed_index_f}, x={x_f}: {e!r}", flush=True)
                
        results_np = np.asarray(results, dtype=float)
        if np.isnan(results_np).any():
            failed_indices = np.where(np.isnan(results_np))[0]
            print(f"There are {len(failed_indices)} nan values in the result.")
            results_np = np.where(np.isnan(results_np), 1e6, results_np)
        
        infected_people = results_np.reshape(number_individuals, self.number_process_repetition)
        
        mean_infected_people = infected_people.mean(axis=1)
        mean_closures = np.asarray(x).mean(axis=1)
        
        out["F"] = anp.column_stack([mean_infected_people, mean_closures]) #anp.column_stack([mean_infected_people, mean_infected_people]) #
        print("Finished evaluation of all ABM simulations.", flush=True)
        

        


if __name__ == "__main__":

    print("Experiment: school_closure vs 100*work_closure")
    
    start_nsgaii = time.time()
    abm_problem = nsgaii__on_abm_parallelized()
    abm_algorithm = NSGA2(pop_size=24)
    result_abm = minimize(
        abm_problem,
        abm_algorithm,
        ("n_gen", 6),
        seed=1,
        verbose=False,
    )
    end_nsgaii = time.time()

    print("Time:", end_nsgaii - start_nsgaii)
    print(result_abm.success)
    print(result_abm.message)
    print("X:", result_abm.pop.get("X"))
    print(result_abm.X)
    print("Result F:", result_abm.F)

    plot = Scatter()
    plot.add(result_abm.F, facecolor="none", edgecolor="red",)
    plot.show()
    figure_path = os.path.join(sys.path[0],"output", "figures", f"testing_multithreadprocessing_{datetime.date.today()}_100000.png",)
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    fig = plt.gcf()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")










    