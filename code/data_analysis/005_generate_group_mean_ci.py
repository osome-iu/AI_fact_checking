"""
Purpose:
- Generate the group means and 95% confidence interval for the Control, Forced,
    Optional conditions (belief and sharing groups).

Inputs:
- None
- Paths hardcoded as constants below

Outputs:
- group_mean_ci.csv : group means and bootstrap confidence intervals

Author: Matthew DeVerna
"""

import os

import pandas as pd

ROOT_DIR = "data_analysis"
# Ensure we are in the data_cleaning directory for paths to work
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

# Local imports
from boot import bootstrap_ci

OUTPUT_DIR = "../../results"
DATA_PATH = "../../results/discernment_df_main_groups_only_w_veracity.csv"
data = pd.read_csv(DATA_PATH)

# Calculate the mean proportion of headlines that participants answer "Yes" to
# in each condition.
mean_said_yes_by_group = (
    data.groupby(["Group", "Condition", "veracity"])["prop_yes"]
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
    data.groupby(["Group", "Condition", "veracity"])["prop_yes"]
    .apply(bootstrap_ci)
    .to_frame("ci_said_yes")
    .reset_index()
)

# Merge the group means and the 95% confidence intervals
mean_said_yes_by_group = mean_said_yes_by_group.merge(
    ci95_said_yes_by_group, on=["Group", "Condition", "veracity"]
)

# Save output
output_fname = os.path.join(OUTPUT_DIR, "group_mean_ci.csv")
mean_said_yes_by_group.to_csv(output_fname, index=False)
