"""
Purpose:
- Generate data files the contain the proportion of headlines that particpants
    said they believed or would share, accounting for congruency.

Inputs:
- None

Outputs:
- chatgpt-fact-checker/results/SI_5way_with_congruency_by_participant.csv
- chatgpt-fact-checker/results/SI_5way_with_congruency_mean_and_ci.csv

Author: Matthew DeVerna
"""
import os
import sys

import pandas as pd

ROOT_DIR = "data_analysis"
# Ensure we are in the data_cleaning directory for paths to work
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

from boot import bootstrap_ci

# Constants
RESULTS_DIR = "../../results"

# Set up loading function
DATA_CLEANING_DIR = "../data_cleaning"
sys.path.insert(0, DATA_CLEANING_DIR)
from db import find_file

# Get file version and load
DATA_DIR = "../../data/cleaned_data/"
FNAME = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")
all_data = pd.read_parquet(FNAME)

### Calculate proportion of headlines participants said yes to by congruency ###
# Congruency as a concept only makes sense for participants coded as "Dem" or "Rep,"
# this drops independents
partisans = all_data[all_data.congruency.notna()]

# In order to calcualte the proportion of headlines a participant said they believed
# or would share for EACH TYPE OF HEADLINE, we need the total number of headlines for
# each type. In this case, the total will differ by Reps and Dems but be the same for
# each participant within those groups. This is because "congruent" is based on ones own
# partisanship relative to the headline.
single_rep_id = partisans[partisans.party_recoded == "Rep"].iloc[0].ResponseId
single_rep = partisans[partisans.ResponseId == single_rep_id]

single_dem_id = partisans[partisans.party_recoded == "Dem"].iloc[0].ResponseId
single_dem = partisans[partisans.ResponseId == single_dem_id]

# Create headline counts dict for republicans
# {('congruent', False, 'false'): 10,
#  ('congruent', True, 'unsure'): 8,
#  ('congruent', True, 'false'): 2,
#  ('incongruent', False, 'false'): 8,
#  ('incongruent', False, 'unsure'): 2,
#  ('incongruent', True, 'unsure'): 5,
#  ('incongruent', True, 'true'): 3,
#  ('incongruent', True, 'false'): 2}
rep_num_headlines_map = (
    single_rep.groupby(["congruency", "veracity"])["ano_true_false_unsure"]
    .value_counts()
    .to_dict()
)

# Create headline counts dict for democrats
# {('congruent', False, 'false'): 8,
#  ('congruent', False, 'unsure'): 2,
#  ('congruent', True, 'unsure'): 5,
#  ('congruent', True, 'true'): 3,
#  ('congruent', True, 'false'): 2,
#  ('incongruent', False, 'false'): 10,
#  ('incongruent', True, 'unsure'): 8,
#  ('incongruent', True, 'false'): 2}
dem_num_headlines_map = (
    single_dem.groupby(["congruency", "veracity"])["ano_true_false_unsure"]
    .value_counts()
    .to_dict()
)

# Calculate the proportion of each type of headline participants said they believed/would share
temp_dfs = []
for participant in partisans.ResponseId.unique():
    # Get responses for all headlines for a single participant
    single_df = all_data[all_data.ResponseId == participant][
        [
            "Group",
            "Condition",
            "congruency",
            "veracity",
            "ano_true_false_unsure",
            "exp_response",
            "party_recoded",
        ]
    ]

    # Select the headline type count map based on their ideology
    ideo = single_df.iloc[0].party_recoded
    num_headlines_map = (
        rep_num_headlines_map if ideo == "Rep" else dem_num_headlines_map
    )

    # Count the number of times they said yes to each type of headline
    said_yes = (
        single_df.groupby(["congruency", "veracity", "ano_true_false_unsure"])[
            "exp_response"
        ]
        .sum()
        .to_frame("said_yes")
        .reset_index()
    )

    # Calculate the proportion based on the headline type
    prop_yes = []
    for idx, row_data in said_yes.iterrows():
        total_headlines_by_type = num_headlines_map[
            (
                row_data.congruency,
                row_data.veracity,
                row_data.ano_true_false_unsure,
            )
        ]
        prop_yes.append(row_data.said_yes / total_headlines_by_type)

    # Update the frame with new columns and store
    said_yes["prop_yes"] = prop_yes
    said_yes["ResponseId"] = participant
    said_yes["Group"] = single_df.Group.unique().item()
    said_yes["Condition"] = single_df.Condition.unique().item()

    # Each df represents a single participant and contains multiple rows for each type
    # of headline scenario (e.g., congruent and incongruent, true and false, etc.) with
    # a column that contains the proportion of those headlines they said they believed
    # or would share
    temp_dfs.append(said_yes)

assert (
    len(temp_dfs) == partisans.ResponseId.nunique()
), "Error! Number of frames should match number of participants!"

by_participant_df = pd.concat(temp_dfs)

# Calculate the mean and CI for each headline scenario, group, and condition
grouped_results = (
    by_participant_df.groupby(
        ["Group", "Condition", "congruency", "veracity", "ano_true_false_unsure"]
    )["prop_yes"]
    .mean()
    .to_frame("mean")
    .reset_index()
)
cis = (
    by_participant_df.groupby(
        ["Group", "Condition", "congruency", "veracity", "ano_true_false_unsure"]
    )["prop_yes"]
    .apply(bootstrap_ci)
    .to_frame("ci")
    .reset_index()
)
grouped_results = grouped_results.merge(
    cis, on=["Group", "Condition", "congruency", "veracity", "ano_true_false_unsure"]
)

### Save results ###
by_participant_df.to_csv(
    f"{RESULTS_DIR}/SI_5way_with_congruency_by_participant.csv", index=False
)
grouped_results.to_csv(
    f"{RESULTS_DIR}/SI_5way_with_congruency_mean_and_ci.csv", index=False
)
