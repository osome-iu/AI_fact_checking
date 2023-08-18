"""
Purpose:
- Generate the means and 95% confidence interval for the control and forced conditions
    for both groups (belief vs. sharing), accounting for the five headline types:
         1. Actual True and ChatGPT thinks its True
         2. Actual True and ChatGPT is Unsure
         3. Actual True and ChatGPT thinks it is False
         4. Actual False and ChatGPT thinks its True
         5. Actual False and ChatGPT is Unsure
         6. NEVER OCCURS in our data: Actual False and ChatGPT thinks it is True

Inputs:
- None
- Paths hardcoded as constants below

Outputs:
- five_headline_type_mean_ci.csv : mean and bootstrap confidence intervals as described above
- five_headline_type_by_participant.csv : by-participant data used to generate these results

Author: Matthew DeVerna
"""
import os
import sys

import pandas as pd

ROOT_DIR = "data_analysis"
# Ensure we are in the data_cleaning directory for paths to work
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

# Local imports
from boot import bootstrap_ci

OUTPUT_DIR = "../../results"

# Set up loading function
DATA_CLEANING_DIR = "../data_cleaning"
sys.path.insert(0, DATA_CLEANING_DIR)
from db import find_file

# Get file version and load
DATA_DIR = "../../data/cleaned_data/"
FNAME = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")
all_data = pd.read_parquet(FNAME)

# Combine the ground truth with the ChatGPT guesses to create a new coding column
gtruth_cgpt = [
    f"{gtruth}_{cgpt}"
    for gtruth, cgpt in zip(
        all_data["veracity"].astype(str),
        all_data["ano_true_false_unsure"].str.capitalize(),
    )
]
all_data["v_split_by_cgpt"] = gtruth_cgpt

# Calculate the number of yes responses per participant (ResponseId)
yes_by_participant = (
    all_data.groupby(["Group", "Condition", "ResponseId", "v_split_by_cgpt"])[
        "exp_response"
    ]
    .sum()
    .to_frame("num_yes")
    .reset_index()
)

# To correctly calculate the proportion of headlines that folks said "Yes" to,
# we need to count the number of headlines in each of the five categories
single_participant = list(all_data.ResponseId)[:1]
num_headlines_map = (
    all_data[all_data.ResponseId.isin(single_participant)]["v_split_by_cgpt"]
    .value_counts()
    .to_dict()
)

# Calculates the proportion of each type of headline that a participant got correct
# E.g., What proportion of the "True_False" headlines, did participant X answer "Yes" to?
prop_yes = []
for idx, row_data in yes_by_participant.iterrows():
    prop_yes.append(row_data.num_yes / num_headlines_map[row_data.v_split_by_cgpt])
yes_by_participant["prop_yes"] = prop_yes

# Calculate the mean proportion of headlines that participants answer "Yes" to
# in each condition.
mean_said_yes_by_group = (
    yes_by_participant.groupby(["Group", "Condition", "v_split_by_cgpt"])["prop_yes"]
    .mean()
    .to_frame("mean_said_yes")
    .reset_index()
)

# Use the bootstrap function to calculate 95% confidence intervals for each group (see boot.py for details)
# - Confidence used = 0.95
# - Number of bootstrapped samples = 5_000
# - The function uses the `d_only=True` argument by default to return only half the distance
#    between the upper and lower bounds, convenient for plotting later
ci95_said_yes_by_group = (
    yes_by_participant.groupby(["Group", "Condition", "v_split_by_cgpt"])["prop_yes"]
    .apply(bootstrap_ci)
    .to_frame("ci_said_yes")
    .reset_index()
)

# Merge the group means and the 95% confidence intervals
mean_said_yes_by_group = mean_said_yes_by_group.merge(
    ci95_said_yes_by_group, on=["Group", "Condition", "v_split_by_cgpt"]
)

# Split the "v_split_by_cgpt" column into two columns again
mean_said_yes_by_group[["ground_truth", "cGPT_thinks"]] = mean_said_yes_by_group[
    "v_split_by_cgpt"
].str.split("_", 1, expand=True)

yes_by_participant[["ground_truth", "cGPT_thinks"]] = yes_by_participant[
    "v_split_by_cgpt"
].str.split("_", 1, expand=True)

# Save the results
mean_results_fname = os.path.join(OUTPUT_DIR, "five_headline_type_mean_ci.csv")
by_part_fname = os.path.join(OUTPUT_DIR, "five_headline_type_by_participant.csv")
mean_said_yes_by_group.to_csv(mean_results_fname, index=False)
yes_by_participant.to_csv(by_part_fname, index=False)
