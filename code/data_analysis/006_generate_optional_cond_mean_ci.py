"""
Purpose:
- Generate the weighted group means and 95% confidence interval for the optional
    condition for both groups (belief vs. sharing). Also return the by-participant
    data used to generate these results.

Inputs:
- None
- Paths hardcoded as constants below

Outputs:
- optional_results_mean_ci.csv : group means and bootstrap confidence intervals
- optional_results_by_participant.csv : by-participant data used to generate these results

Author: Matthew DeVerna
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = "data_analysis"
# Ensure we are in the data_cleaning directory for paths to work
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

# Local imports
from boot import bootstrap_wci

OUTPUT_DIR = "../../results"

# Set up loading function
DATA_CLEANING_DIR = "../data_cleaning"
sys.path.insert(0, DATA_CLEANING_DIR)
from db import find_file

# Get file version and load
DATA_DIR = "../../data/cleaned_data/"
FNAME = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")


def check_rows(df):
    """
    A function to check that the data frame has the right number of rows and participants
    """
    # Check we've got the right number of participants
    total_participants = option_df.ResponseId.nunique()
    wrong_num_err = (
        "Error! There is an incorrect number of participants in this data frame!"
    )
    num_parts_in_frame = df.ResponseId.nunique()
    assert total_participants == num_parts_in_frame, wrong_num_err

    # Check we've got the right number of rows
    row_num_err = "Error! There is an incorrect number of rows in this data frame!"
    rows_per_part = 2  # Opt-in and opt-out
    assert num_parts_in_frame * rows_per_part == len(df), row_num_err


if __name__ == "__main__":
    # Select only the option condition data
    df = pd.read_parquet(FNAME)
    option_df = df[df.option_cond.str.contains("Opt")].copy()

    # Count the number of times each participant opt'd into each condition
    # by group and headline veracity.
    by_condition_counts = (
        option_df.groupby(["Group", "ResponseId", "veracity"])["option_cond"]
        .value_counts()
        .to_frame("count")
        .reset_index()
        .melt(
            id_vars=["Group", "ResponseId", "veracity", "option_cond"],
            value_vars="count",
            value_name="num",
        )
        .pivot(
            index=["Group", "ResponseId", "option_cond"],
            columns=["veracity"],
            values=["num"],
        )
        .reset_index()
    )

    # Update the columns, which are tuples, and sort
    by_condition_counts.columns = [
        f"{str(x[0])}_{str(x[1])}".rstrip("_") for x in by_condition_counts.columns
    ]
    by_condition_counts = by_condition_counts.sort_values("ResponseId").reset_index(
        drop=True
    )

    # If any of these cell totals are zero, they will be NaN so we fill them with zeros
    by_condition_counts.fillna(0, inplace=True)

    # The above code counts only events that occured. If a participant never opt'd into
    # a specific option, then we have to add those zeros into the data frame.
    options = set(["Opt_in", "Opt_out"])
    missing_records = []
    for participant in by_condition_counts.ResponseId.unique():
        one_p_df = by_condition_counts[by_condition_counts.ResponseId == participant]

        # If one of the rows does not exist...
        curr_options = set(one_p_df.option_cond.unique())
        if options != curr_options:
            missing_option_cond = list(options.difference(curr_options))[0]
            p_group = one_p_df.Group.unique().item()

            # ... create that record
            missing_records.append(
                {
                    "Group": p_group,
                    "ResponseId": participant,
                    "option_cond": missing_option_cond,
                    "num_False": 0,
                    "num_True": 0,
                }
            )

    # Combine the existing df with a new one populated by the missing records
    by_condition_counts = (
        pd.concat([by_condition_counts, pd.DataFrame.from_records(missing_records)])
        .sort_values(
            by=["Group", "ResponseId", "option_cond"],
        )
        .reset_index(drop=True)
    )

    # Check we've got the right number of rows
    check_rows(by_condition_counts)

    # Calculate the total number of times that participants said "Yes"
    # This means that they either believed the headline or were willing to share it
    said_yes_counts = (
        option_df.groupby(["Group", "ResponseId", "veracity", "option_cond"])[
            "exp_response"
        ]
        .sum()
        .to_frame("num_yes")
        .reset_index()
        .sort_values("ResponseId")
        .reset_index(drop=True)
    )

    said_yes_counts = said_yes_counts.pivot(
        index=["Group", "ResponseId", "option_cond"],
        columns=["veracity"],
        values=["num_yes"],
    ).reset_index()

    # Update the columns and sort
    said_yes_counts.columns = [
        f"{str(x[0])}_{str(x[1])}".rstrip("_") for x in said_yes_counts.columns
    ]
    said_yes_counts.sort_values(by=["Group", "ResponseId", "option_cond"], inplace=True)

    # Merge the two frames.
    #   Note that the "said_yes_counts" frame will again be missing rows
    #   for conditions that we manually added in the `by_condition_counts` frame.
    #   After the merge, these cells are given NaN values, which are filled with zeros.
    by_condition_all = (
        by_condition_counts.merge(
            said_yes_counts, on=["Group", "ResponseId", "option_cond"], how="left"
        )
        .fillna(0)
        .copy()
    )

    # Calculate the proportion of headlines opt'd into to which participants said "Yes"
    by_condition_all["prop_yes_False"] = by_condition_all["num_yes_False"].div(
        by_condition_all["num_False"]
    )
    by_condition_all["prop_yes_True"] = by_condition_all["num_yes_True"].div(
        by_condition_all["num_True"]
    )

    # We will later need to scale the weighted values by the number
    # of participants per group so we first get these totals.
    # Looks like the below:
    # {('Belief', 'Opt_in', False): 219,
    #  ('Belief', 'Opt_in', True): 231,
    #  ...}
    tot_part_by_group = (
        option_df.groupby(["Group", "option_cond", "veracity"])["ResponseId"]
        .nunique()
        .to_dict()
    )

    # Calculate the weighted means and bootstrapped CIs
    temp_frames = []
    mean_value_records = []
    grouped_data = by_condition_all.groupby(["Group", "option_cond"])
    for group_optcond, temp_data in grouped_data:
        # Group and condition identifiers for temp_data
        temp_group = group_optcond[0]
        temp_optcond = group_optcond[1]

        for bool_obj in [True, False]:
            # Calculate the weights
            num_col_name = f"num_{bool_obj}"
            weight_col_name = f"w_{bool_obj}"
            temp_data[weight_col_name] = temp_data[num_col_name].div(
                temp_data[num_col_name].sum()
            )

            ### Now use them to calculate the weighted mean and bootstrapped CI
            prop_col_name = f"prop_yes_{bool_obj}"
            temp_data_no_nan = temp_data.dropna(subset=prop_col_name)
            observations = list(temp_data_no_nan[prop_col_name])

            # Here we use the raw number of headlines that they contributed instead of
            # the actual weights because it will allow the weighted mean calculation
            # to incorporate the different number of total headlines.
            weights = list(temp_data_no_nan[num_col_name])

            weighted_mean = np.average(a=observations, weights=weights)
            # See boot.py for details on below function
            w_95_ci = bootstrap_wci(obs=observations, weights=weights)

            mean_value_records.append(
                {
                    "Group": temp_group,
                    "option_cond": temp_optcond,
                    "veracity": bool_obj,
                    "w_mean": weighted_mean,
                    "w_95_ci": w_95_ci,
                }
            )

        temp_frames.append(temp_data)

    by_condition_all = pd.concat(temp_frames)
    by_condition_all.sort_values(
        by=["Group", "ResponseId", "option_cond"], inplace=True
    )
    by_condition_all.reset_index(drop=True, inplace=True)
    check_rows(by_condition_all)
    group_mean_ci_df = pd.DataFrame.from_records(mean_value_records)

    # Save output
    by_part_fname = os.path.join(OUTPUT_DIR, "optional_results_by_participant.csv")
    mean_results_fname = os.path.join(OUTPUT_DIR, "optional_results_mean_ci.csv")
    by_condition_all.to_csv(by_part_fname, index=False)
    group_mean_ci_df.to_csv(mean_results_fname, index=False)
