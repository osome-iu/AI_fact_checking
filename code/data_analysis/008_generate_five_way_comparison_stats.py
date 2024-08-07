"""
Purpose: 
    Generate the statistics for the different group comparisons when accounting for ChatGPT's accuracy.
    Handles various multiple test correction/adjustment methods.

    Run via:
    python 008_generate_five_way_comparison_stats.py > /path/to/output_dir/five_way_comparison_stats.txt

Inputs:
    None

Outputs:
    New file: five_way_comparison_stats.txt

Author: Matthew DeVerna
"""

import os

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from effect_size import cohen_d
from boot import mean_diff_bootstrap_ci

ROOT_DIR = "data_analysis"
if __name__ == "__main__":
    # Ensure we are in the data_analysis directory for paths to work
    if os.path.basename(os.getcwd()) != ROOT_DIR:
        raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

# Paths and filenames
DATA_DIR = "../../results/"
FIVE_WAY_FNAME = "five_headline_type_by_participant.csv"

# Lists for comparisons
GROUPS = ["Belief", "Share"]
# Control is excluded from the list below because it is included in all comparisons
# Human-FC is excluded from the list below because it doesn't make sense for this comparison
CONDITIONS = ["Forced", "Optional"]
VERACITY_LIST = [True, False]
cGPT_THINK_LIST = ["True", "Unsure", "False"]

CORRECTION_METHOD = "bonferroni"
AVAILABLE_METHODS = [
    "bonferroni",  # one-step correction
    "sidak",  # one-step correction
    "holm-sidak",  # step down method using Sidak adjustments
    "holm",  # step down method using bonferroni adjustments (aka holm-bonferroni)
    "simes-hochberg",  # step-up method (independent)
    "hommel",  # closed method based on Simes tests (non-negative)
    "fdr_bh",  # Benjamini/Hochberg (non-negative)
    "fdr_by",  # Benjamini/Yekutieli (negative)
    "fdr_tsbh",  # two stage fdr correction (non-negative)
    "fdr_tsbky",  # two stage fdr correction (non-negative)
]
if CORRECTION_METHOD not in AVAILABLE_METHODS:
    raise Exception(f"`{CORRECTION_METHOD}` not in {AVAILABLE_METHODS}")


def pvalue_adjustment(comparisons_dic, alpha=0.5, method="bonferroni"):
    """
    Take a dictionary of comparisons (keys) and scipy.stats results objects (values) and
    return only significant comparisons, with the adjusted p-values and statistic
    (based on the chosen method).

    Note: This is a convenience function that applies `statsmodels.stats.multitest.multipletests`.

    Parameters:
    -----------
    - comparisons_dic (dict): A dictionary of comparisons and scipy.stats results objects
    - alpha (float): The family-wise error rate, e.g. 0.1
    - method (str): The method used to adjust the p-values.
        - See available options above
        - Ref: https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    """
    # Extract the comparison names and p-values from the dictionary
    keys = list(comparisons_dic.keys())
    # Keys here are tuples, so we combine them into a string
    comparisons = ["-".join(key) for key in keys]
    # Each object is a result object returned by scipy.stats.mannwhitneyu
    result_dicts = list(comparisons_dic.values())
    p_vals = [dic.pvalue for dic in result_dicts]
    statistics = [dic.statistic for dic in result_dicts]

    # Perform the Holm-Bonferroni adjustment
    rejected_bools, adjusted_pvals, _, _ = multipletests(
        p_vals, alpha=alpha, method=method
    )

    # Separate significant and non-significant comparisons
    significant_comparisons = [
        {comparison: {"pvalue_adjusted": adjusted_pval, "statistic": statistic}}
        for comparison, adjusted_pval, statistic, is_rejected in zip(
            comparisons, adjusted_pvals, statistics, rejected_bools
        )
        if is_rejected
    ]
    non_significant_comparisons = [
        {comparison: {"pvalue_adjusted": adjusted_pval, "statistic": statistic}}
        for comparison, adjusted_pval, statistic, is_rejected in zip(
            comparisons, adjusted_pvals, statistics, rejected_bools
        )
        if not is_rejected
    ]

    return significant_comparisons, non_significant_comparisons


# Load data
df = pd.read_csv(os.path.join(DATA_DIR, FIVE_WAY_FNAME))

# Put data into a dictionary for convenience
data_dict = {}
for keys, data in df.groupby(["Group", "Condition", "ground_truth", "cGPT_thinks"]):
    data_dict[keys] = data["prop_yes"].values

print("\nPerform Mann Whitney U-tests comparing each group to the control group")
print("-" * 50)
for group in GROUPS:
    for condition in CONDITIONS:
        print("\n")
        print(f"{group}-{condition}")
        print("#" * 50)
        group_pvals = dict()
        for veracity in VERACITY_LIST:
            for cGPT_think in cGPT_THINK_LIST:
                # This condition never happens in our data
                if (veracity == False) and (cGPT_think == "True"):
                    continue

                print(f"\t- {group}: {condition} {veracity} x {cGPT_think}")
                group1vals = data_dict[(group, "Control", veracity, cGPT_think)]
                group2vals = data_dict[(group, condition, veracity, cGPT_think)]
                mean_diff = np.mean(group1vals) - np.mean(group2vals)
                print(f"\t\t- Mean difference (control - treated): {mean_diff:.4f}")
                print(f"\t\t- Mean difference (control - treated; %): {mean_diff:.2%}")
                cohensd = cohen_d(group1vals, group2vals)
                low95, high95 = mean_diff_bootstrap_ci(
                    group1vals,
                    group2vals,
                    confidence=0.95,
                    d_only=False,
                )
                results = mannwhitneyu(
                    group1vals,
                    group2vals,
                )
                print(f"\t\t- U statistic  : {results.statistic}")
                print(f"\t\t- p-value      : {results.pvalue:}")
                print(f"\t\t- p-value (rnd): {results.pvalue:.4f}")
                print(f"\t\t- Cohens d(rnd): {cohensd:.4f}")
                print(f"\t\t- 95% CI       : [{low95:.4f}, {high95:.4f}]")
                print(f"\t\t- 95% CI       : [{low95:.2%}, {high95:.2%}]")

                group_pvals[
                    (group, "Control", condition, str(veracity), cGPT_think)
                ] = results

        print(f"\nAdjusting p-values (via {CORRECTION_METHOD} method)...\n")
        significant_comparisons, non_significant_comparisons = pvalue_adjustment(
            group_pvals, alpha=0.05, method="holm"
        )
        print("Significant Comparisons:")
        print("-" * 50)
        if len(significant_comparisons) == 0:
            print("\t No significant comparisons")

        # significant_comparisons is a list of nested dictionaries
        for results in significant_comparisons:
            for comparison, data in results.items():
                print(f"\t {comparison}")
                for key, val in data.items():
                    print(f"\t\t {key}      : {val:}")
                    if key == "pvalue_adjusted":
                        print(f"\t\t {key} (rnd): {val:.4f}")

        print("\nNon-significant Comparisons:")
        print("-" * 50)
        if len(non_significant_comparisons) == 0:
            print("\t All comparisons significant")

        # non_significant_comparisons is a list of nested dictionaries
        for results in non_significant_comparisons:
            for comparison, data in results.items():
                print(f"\t {comparison}")
                for key, val in data.items():
                    print(f"\t\t {key}      : {val:}")
                    if key == "pvalue_adjusted":
                        print(f"\t\t {key} (rnd): {val:.4f}")
