"""
Purpose: 
    Generate the statistics for the different group comparisons related to the
    Optional Condition by annotation.

    Run via:
    python generate_optional_stats.py > /path/to/output_dir/optional_comparisons_stats_by_annotation.txt

Inputs:
    None

Outputs:
    New file: optional_comparisons_stats_by_annotation.txt

Author: Matthew DeVerna
"""

import os

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu

from effect_size import cohen_d
from boot import mean_diff_bootstrap_ci
from bonferroni import bonferroni_correction

NUM_COMPARISONS = 4
ROOT_DIR = "data_analysis"
if __name__ == "__main__":
    # Ensure we are in the data_analysis directory for paths to work
    if os.path.basename(os.getcwd()) != ROOT_DIR:
        raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

# Paths and filenames
DATA_DIR = "../../results/"
OPTIONAL_RESULTS_FNAME = "optional_results_by_participant_by_annotation.csv"

# Read in the data
df = pd.read_csv(os.path.join(DATA_DIR, OPTIONAL_RESULTS_FNAME))

# This will store the proportion of headlines that participants said they believed
# or would be willing to share (values). Importantly, they must be weighted and then
# scaled by the number of participants to allow for proper calculations.
data_dict = {}

grouped_data = df.groupby(["Group", "option_cond", "ano_true_false_unsure"])
for group_optcond, temp_data in grouped_data:
    # Group and condition identifiers for temp_data
    temp_group = group_optcond[0]
    temp_optcond = group_optcond[1]
    temp_annotation = group_optcond[2]

    for bool_obj in [True, False]:

        if bool_obj == False and temp_annotation == "true":
            continue

        # Set needed column names
        prop_col_name = f"prop_yes_{bool_obj}"
        num_col_name = f"num_{bool_obj}"
        weight_col_name = f"w_{bool_obj}"

        # Drop NaNs and get necessary data
        no_nan = temp_data.dropna(subset=prop_col_name)
        num_participants = no_nan["ResponseId"].nunique()
        observations = no_nan[prop_col_name]
        weights = no_nan[weight_col_name]

        # Weight observations and then scale those values by the number of participants
        # so that individual values are on the proper scale
        weighted_obs_scaled = (observations * weights) * num_participants

        # Check we did this right by ensuring the mean of these values equals the weighted avg.
        raw_weight_cnt = no_nan[num_col_name]
        weighted_mean = np.average(a=observations, weights=raw_weight_cnt)
        err_msg = "Error! Weighted mean (np.average) != scaled mean (np.mean)!"
        assert np.isclose(np.mean(weighted_obs_scaled), weighted_mean), err_msg

        data_dict[temp_group, temp_annotation, temp_optcond, bool_obj] = (
            weighted_obs_scaled
        )

print("Running tests comparing the opt-in and opt-out groups...")
print("\nMEAN DIFFERENCES")
print("Note: All results compare the opt-in versus opt-out groups.")
print("-" * 50)
results_dict = dict()
for group in ["Belief", "Share"]:
    for veracity in [True, False]:
        for annotation in ["true", "false", "unsure"]:
            if veracity == False and annotation == "true":
                continue
            group1vals = data_dict[(group, annotation, "Opt_in", veracity)]
            group2vals = data_dict[(group, annotation, "Opt_out", veracity)]

            mean_diff = np.mean(group1vals) - np.mean(group2vals)
            print(f"\t- {group}: Veracity: {veracity} | Judged: {annotation}")
            print(f"\t\t- Mean difference (opt in - opt out): {mean_diff:.4f}")
            print(f"\t\t- Mean difference (opt in - opt out; %): {mean_diff:.2%}")

            mwu_results = mannwhitneyu(group1vals, group2vals)
            cohensd = cohen_d(group1vals, group2vals)
            ci = mean_diff_bootstrap_ci(
                group1vals, group2vals, confidence=0.95, d_only=False
            )
            results_dict[(group, veracity, annotation)] = {
                "mwu": mwu_results,
                "cohensd": cohensd,
                "ci": ci,
            }

print("\nSTATS")
print("-" * 50)
for group_veracity, results_dict in results_dict.items():
    pval = results_dict["mwu"].pvalue
    pval_corrected = bonferroni_correction(pval, NUM_COMPARISONS)
    ustat = results_dict["mwu"].statistic
    cohensd = results_dict["cohensd"]
    lowci, highci = results_dict["ci"]

    print(
        f"\t{group_veracity[0]} | Veracity: {group_veracity[1]} | Judged: {group_veracity[2]}"
    )
    print(f"\t\t- pval             : {pval}")
    print(f"\t\t- pval (corrected) : {pval_corrected:}")
    print(f"\t\t- pval (corr, rnd) : {pval_corrected:.4f}")
    print(f"\t\t- U                : {ustat}")
    print(f"\t\t- Cohen's d        : {cohensd:.4f}")
    print(f"\t\t- 95% CI           : [{lowci:.4f}, {highci:.4f}]")
    print(f"\t\t- 95% CI (%)       : [{lowci:.2%}, {highci:.2%}]")
