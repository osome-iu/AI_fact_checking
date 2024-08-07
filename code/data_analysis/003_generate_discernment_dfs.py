"""
Purpose:
    Create discernment dataframes.

Inputs:
    None

Outputs:
    New files:
        - discernment_df_main_groups_only.csv: Subject-level discernment data
        - discernment_df_main_groups_only_w_veracity.csv: Similar but with veracity
            (so two rows per subject: True and False).

Author: Harry Yaojun Yan & Matthew DeVerna
"""

import os
import sys

import pandas as pd

NUM_HEADLINES = 20  # 20 per category (T/F)
OUTPUT_DIR = "../../results/"
OUTPUT_FNAME = "discernment_df_main_groups_only.csv"
OUTPUT_FNAME_W_VERACITY = "discernment_df_main_groups_only_w_veracity.csv"
ROOT_DIR = "data_analysis"

# Ensure we are in the data_analysis directory for paths to work
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception("Must run this script from the `code/data_analysis/` directory!")

# Set up loading function
DATA_CLEANING_DIR = "../data_cleaning"
sys.path.insert(0, DATA_CLEANING_DIR)
from db import find_file

# Get file version and load
DATA_DIR = "../../data/cleaned_data/"
FNAME = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")
df = pd.read_parquet(FNAME)

# Calculate the number of yes responses per participant (ResponseId)
yes_by_participant = (
    df.groupby(["Group", "Condition", "ResponseId", "veracity"])["exp_response"]
    .sum()
    .to_frame("num_yes")
    .reset_index()
)

# Calcualte the proportion of total headlines they said believed/would share
yes_by_participant["prop_yes"] = yes_by_participant["num_yes"] / NUM_HEADLINES

# Wrangle for easier calculation
yes_by_participant_pivot = yes_by_participant.pivot(
    index=["Group", "Condition", "ResponseId"],
    columns=["veracity"],
    values=["prop_yes"],
).reset_index()

# After the pivot, we have a multi-level column names.
# The below combines them for simplicity
yes_by_participant_pivot.columns = [
    "_".join(map(str, col)).rstrip("_")
    for col in yes_by_participant_pivot.columns.values
]

# Calculate the discernment
yes_by_participant_pivot["discernment"] = (
    yes_by_participant_pivot["prop_yes_True"]
    - yes_by_participant_pivot["prop_yes_False"]
)

# Save dfs
yes_by_participant.to_csv(
    os.path.join(OUTPUT_DIR, OUTPUT_FNAME_W_VERACITY), index=False
)
yes_by_participant_pivot.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FNAME), index=False)
