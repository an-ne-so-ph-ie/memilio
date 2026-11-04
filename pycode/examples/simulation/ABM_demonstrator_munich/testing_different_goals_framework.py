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

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import datetime

import memilio.simulation.abm as abm
from memilio.simulation import AgeGroup
import abm_demonstrator_munich

base_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_directory)   
input_directory = os.path.join(base_directory, "input")
output_directory = os.path.join(base_directory, "nsga2_results")

simulation_end_day = 70 #60                
earliest_change_day = 19 #25  # erster lockdown             
latest_change_day = 49
reference_week_days = 7                
#number_of_seeds = 15         
maximum_home_office = 0.7   
#seed_selection_random_seed = 2020
minimum_period_days = 7    

simulation_start_date = datetime.date(2020, 3, 2)
#TODO need to check numbers again
free_test_capacity_per_week = {12: 490, 13: 1130, 14: 1270, 15: 1541, 16: 1648, 17: 1666, 18: 1867, 19: 1818, 20: 1754}

# interventions 
test_types = ["none", "pcr"] #"antigen"
interventions = ["home_office", "school_closure", "social_events", "work_test", "school_test",
                 "probability_test_work", "probability_test_school"]
variable_names = [f"{intervention}_{period_number}" for period_number in (1, 2) for intervention in interventions]
variable_names += ["change_day_1", "change_day_2"]
objective_candidates = ["f_I", "f_D", "f_ED", "f_YLL", "f_EYLL", "peak_hosp", "peak_icu",   # health
                        "f_W", "f_K_work", "f_S_A", "f_S_B", "f_K_social", "f_K_all"]   # work, social

# order
location_type_home, location_type_school, location_type_work, location_type_social_event = range(4)
number_of_location_types, number_of_age_groups = 11, 6

evaluation_start_day = 0
start_hour, end_hour = evaluation_start_day * 24, simulation_end_day * 24
evaluated_days = np.arange(evaluation_start_day, simulation_end_day)

decision_variables = {}
for period_number in (1, 2):
    decision_variables[f"home_office_{period_number}"] = ("real", 0.0, maximum_home_office)   
    decision_variables[f"school_closure_{period_number}"] = ("real", 0.0, 1.0)
    decision_variables[f"social_events_{period_number}"] = ("real", 0.0, 1.0)
    decision_variables[f"work_test_{period_number}"] = ("choice", test_types)
    decision_variables[f"school_test_{period_number}"] = ("choice", test_types)
    decision_variables[f"probability_test_work_{period_number}"] = ("real", 0.0, 1.0)
    decision_variables[f"probability_test_school_{period_number}"] = ("real", 0.0, 1.0)
decision_variables["change_day_1"] = ("integer", earliest_change_day, latest_change_day)
decision_variables["change_day_2"] = ("integer", earliest_change_day, latest_change_day)
variable_names = list(decision_variables)



def selected_seed_indices():
    selection_file = os.environ.get("SELECTED_PARTICLES", "selected_particles.csv")
    return pd.read_csv(os.path.join(input_directory, selection_file))["particle"].astype(int).tolist()

 

def to_python_value(value):
    return value.item() if hasattr(value, "item") else value

def test_capacity_of_day(day):
    """ getfree PCR capacity of the calendar week the simulation day belongs to (rki)"""
    calendar_week = (simulation_start_date + datetime.timedelta(days=int(day))).isocalendar()[1]
    return free_test_capacity_per_week[calendar_week]


def test_values(tests_hourly):
    """daily tests from the hourly logger and the worst ratio to the capacity"""
    tests_daily = tests_hourly[1:, 0].reshape(-1, 24).sum(axis=1)[:simulation_end_day]   # erster Eintrag = Startzeitpunkt
    positive_daily = tests_hourly[1:, 1].reshape(-1, 24).sum(axis=1)[:simulation_end_day]
    ratios = [tests_daily[day] / test_capacity_of_day(day) for day in range(earliest_change_day, simulation_end_day)]
    return {"tests_max_daily": float(tests_daily.max()), "tests_total": float(tests_daily.sum()),
            "tests_positive_total": float(positive_daily.sum()), "test_capacity_ratio": float(max(ratios))}
 
def decode_strategy(strategy):
    """pymoo strategy converting into abm strategy"""
    change_days = [int(strategy["change_day_1"]), int(strategy["change_day_2"])]
    periods = [{intervention: to_python_value(strategy[f"{intervention}_{period_number}"]) for intervention in interventions} for period_number in (1, 2)]
    return change_days, periods
 
 
def objectives_of_run(path_output_folder, number_to_save, death_probability):
    """Objectives of one finished run from output files only"""
    output_prefix = os.path.join(path_output_folder, str(number_to_save))
    compartments = pd.read_csv(f"{output_prefix}_comps_damping_compact_version.csv", sep=" ")
    presence = np.load(f"{output_prefix}_presence.npy")                   
    contacts_file = f"{output_prefix}_contacts_daily.npy"
    daily_contacts = np.load(contacts_file).sum(axis=(2, 3)) if os.path.exists(contacts_file) else None              
    #daily_contacts = np.load(f"{output_prefix}_contacts_daily.npy").sum(axis=(2, 3))    
    infection_paths = pd.read_csv(f"{output_prefix}_infection_paths_damping.txt", sep=r"\s+")
    objective_values = compute_objectives(infection_paths, compartments, presence, daily_contacts, death_probability)
    tests_file = f"{output_prefix}_tests.npy"
    if os.path.exists(tests_file):
        objective_values.update(test_values(np.load(tests_file)))
    return objective_values


def simulate_strategy(strategy, seed_index, number_to_save, path_output_folder):
    change_days, periods = decode_strategy(strategy)
    abm_demonstrator_munich.run_abm_simulation(seed_index, number_to_save, path_output_folder,
                                               simulation_days=simulation_end_day,
                                               change_days=change_days, periods=periods)
    with open(os.path.join(path_output_folder, f"{number_to_save}_strategy.json"), "w") as file:
        json.dump({"seed_index": int(seed_index), **{k: to_python_value(v) for k, v in strategy.items()}}, file)
    return objectives_of_run(path_output_folder, number_to_save, death_probability_per_age_group(seed_index))
 
 
def death_probability_per_age_group(seed_index):
    """P(symptomatic) * P(severe | symptomatic) * (P(death | severe) + P(critical | severe) * P(death | critical))"""
    import abm_demonstrator_munich
    parameter_table = pd.read_csv(os.path.join(input_directory, "parameter_table.csv"), index_col=0)
    particle_table = pd.read_csv(os.path.join(input_directory, "pop8_sel_particles.csv"), index_col=0)
    parameters = abm_demonstrator_munich.set_infection_parameters(parameter_table,
                                                                  particle_table.loc[seed_index, "kappa"])
    return (parameter_per_age_group(parameters, "SymptomaticPerInfectedNoSymptoms")
            * parameter_per_age_group(parameters, "SeverePerInfectedSymptoms")
            * (parameter_per_age_group(parameters, "DeathsPerInfectedSevere")
               + parameter_per_age_group(parameters, "CriticalPerInfectedSevere")
               * parameter_per_age_group(parameters, "DeathsPerInfectedCritical")))
 
 
def parameter_per_age_group(parameters, name):
    """Value of an ABM parameter (virus variant Wildtype) for the 6 age groups."""
    return np.array([getattr(parameters, name)[abm.VirusVariant.Wildtype, AgeGroup(age_group)].value for age_group in range(number_of_age_groups)])
 
 
def daily_presence(presence, times, location_type):
    """Hourly to daily presence (hour x location type x age group)"""
    day_of_time_point = (times // 24).astype(int)
    values = presence[:, location_type].sum(axis=1)
    simulated = day_of_time_point < simulation_end_day
    return np.bincount(day_of_time_point[simulated], values[simulated], minlength=simulation_end_day)
 
 
def weekly_sum(daily_values):
    """sum for full week after the reference week."""
    number_of_weeks = len(evaluated_days) // 7
    return daily_values[evaluation_start_day:evaluation_start_day + 7 * number_of_weeks].reshape(number_of_weeks, 7).sum(axis=1)
 
 
def contact_reduction(daily_contacts, location_types):
    contacts = daily_contacts[:, location_types].sum(axis=1)
    return 1 - contacts[evaluated_days].sum() / contacts[evaluated_days % 7].sum()
 
 
def compute_objectives(infection_paths, compartments, presence, daily_contacts, death_probability):
    ages = pd.read_csv(os.path.join(input_directory, "persons_scaled.csv"), usecols=["age"])["age"].to_numpy()
    age_group_of_agent = np.digitize(ages, [5, 16, 35, 60, 80])[infection_paths["Agent_id"].to_numpy()]

    infection_hour, hours_dead = infection_paths["S"].to_numpy(), infection_paths["D"].to_numpy()
    newly_infected = (infection_hour > start_hour) & (infection_hour < end_hour)
    newly_dead = (hours_dead > 0) & (end_hour - hours_dead > start_hour)
    new_infections = np.bincount(age_group_of_agent[newly_infected], minlength=number_of_age_groups)
    new_deaths = np.bincount(age_group_of_agent[newly_dead], minlength=number_of_age_groups)

    with open(os.path.join(base_directory, "choosing_parameters", "yll_2020.json")) as file:
        years_of_life_lost = json.load(file)
    mean_years_of_life_lost = np.array([years_of_life_lost[age_group]["mean_yll"]  for age_group in ["0-4", "5-15", "16-34", "35-59", "60-79", "80+"]])
    times = compartments["t"].to_numpy()
    in_evaluation_window = (times >= start_hour) & (times <= end_hour)
    hospitalized = compartments["Isev"] + compartments["Icri"]
 
    
    reference_day = evaluated_days % 7                   # same weekday in the reference week so  frist week
    work_presence = daily_presence(presence, times, location_type_work)
    school_presence = daily_presence(presence, times, location_type_school)
    social_event_presence = daily_presence(presence, times, location_type_social_event)
    working_days = work_presence[reference_day] > 0
    reference_school = school_presence[:reference_week_days].sum()
    reference_social_events = social_event_presence[:reference_week_days].sum()
    weekly_school, weekly_social_events = weekly_sum(school_presence), weekly_sum(social_event_presence)
 
    objective_values = {
        "f_I": new_infections.sum(),
        "f_D": new_deaths.sum(),
        "f_ED": new_infections @ death_probability,
        "f_YLL": new_deaths @ mean_years_of_life_lost,
        "f_EYLL": new_infections @ (death_probability * mean_years_of_life_lost),
        "peak_hosp": hospitalized[in_evaluation_window].max(),
        "peak_icu": compartments["Icri"][in_evaluation_window].max(),
        "f_W": 1 - np.mean(work_presence[evaluated_days][working_days] / work_presence[reference_day][working_days]),
        "f_S_A": 1 - np.mean((weekly_school + weekly_social_events) / (reference_school + reference_social_events)),
        "f_S_B": 1 - np.mean((weekly_school / reference_school + weekly_social_events / reference_social_events) / 2),
        #"f_K_all": contact_reduction(daily_contacts, list(range(number_of_location_types))),
        #"f_K_work": contact_reduction(daily_contacts, [location_type_work]),
        #"f_K_social": contact_reduction(daily_contacts, [location_type_school, location_type_social_event]),
    }
    return {name: float(value) for name, value in objective_values.items()}
 
