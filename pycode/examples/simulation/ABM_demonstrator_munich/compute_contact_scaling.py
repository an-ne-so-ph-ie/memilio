"""
Calculate from *_presence.npy-Dateien (Hours x 11 loc types x 6 age groups) the mean of time spend there per day 
based on Kerkman et al

Command: python compute_contact_scaling.py <folder_with_presence_files> <contact_scaling> [nbr days]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

location_type_names = ["Home", "School", "Work", "SocialEvent", "BasicsShop", "Hospital", "ICU",
                       "Car", "PublicTransport", "TransportWithoutContact", "Cemetery"]
age_group_names = ["Age0to4", "Age5to14", "Age15to34", "Age35to59", "Age60to79", "Age80plus"]
location_types_with_contacts = ["Home", "School", "Work", "SocialEvent", "BasicsShop"]


def compute_hours_per_day(presence, number_of_days):
    """ 
    Mean of hours spend per day of one age group (over the whole week) spend at that loc type
    """
    presence_window = presence[:24 * number_of_days]
    population_per_age_group = presence[0].sum(axis=0)
    person_hours = presence_window.sum(axis=0)
    return person_hours / (number_of_days * population_per_age_group)


def compute_contact_scaling(hours_per_day, minimum_hours):
    """Calculate scale factor  24 / hours
    """
    contact_scaling = np.zeros_like(hours_per_day)
    is_present = hours_per_day >= minimum_hours
    contact_scaling[is_present] = 24.0 / hours_per_day[is_present]
    return contact_scaling


if __name__ == "__main__":
    presence_folder = sys.argv[1]
    output_file = sys.argv[2]
    # normally 14 days, because then the interventions are starting
    number_of_days = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    minimum_hours = 0.05

    presence_files = sorted(glob.glob(os.path.join(presence_folder, "*_presence.npy")))

    hours_per_seed_and_file = np.array([compute_hours_per_day(np.load(presence_file), number_of_days)
                               for presence_file in presence_files])
    hours_mean = hours_per_seed_and_file.mean(axis=0)
    hours_std = hours_per_seed_and_file.std(axis=0)

    hours_table = pd.DataFrame(hours_mean, index=location_type_names, columns=age_group_names)
    hours_std_table = pd.DataFrame(hours_std, index=location_type_names, columns=age_group_names)
    
    contact_scaling = compute_contact_scaling(hours_mean, minimum_hours)
    contact_scaling_table = pd.DataFrame(contact_scaling, index=location_type_names, columns=age_group_names)

    contact_scaling_table.loc[location_types_with_contacts].to_csv(output_file)
    print(f"\nSaved in: {output_file}")