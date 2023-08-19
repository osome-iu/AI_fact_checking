"""
Purpose: 
    Run discernment-related statistics.

    Run via: python 004_generate_group_differences_main_groups_only.py > /path/to/output_dir/group_differences.txt

Inputs:
    None

Outputs:
    New file: group_differences.txt

Author: Harry Yaojun Yan & Matthew DeVerna
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import kruskal, mannwhitneyu

from statsmodels.formula.api import ols

from effect_size import cohen_d
from boot import mean_diff_bootstrap_ci

DATA_CLEANING_DIR = "../data_cleaning"
sys.path.insert(0, DATA_CLEANING_DIR)
# Imports the names of columns in the loaded data by each section of the study
from db import find_file
from bonferroni import bonferroni_correction

DATA_DIR = "../../data/cleaned_data/"
# FNAME = find_file(DATA_DIR, "*long_form.parquet")
FNAME = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")

DATA_DIR = "../../results/"
DISCERNMENT_FNAME = "discernment_df_main_groups_only.csv"
ROOT_DIR = "data_analysis"

GROUP_NAMES = ["Control", "Forced", "Optional"]


if __name__ == "__main__":
    # Ensure we are in the data_analysis directory for paths to work
    if os.path.basename(os.getcwd()) != ROOT_DIR:
        raise Exception(
            "Must run this script from the `code/data_analysis/` directory!"
        )

    # Load data
    df_all = pd.read_parquet(FNAME)
    df_discern = pd.read_csv(os.path.join(DATA_DIR, DISCERNMENT_FNAME))

    print(
        "\n########################################################################\n",
        "                       GENERAL SHARING INCREASE\n",
        "########################################################################\n",
    )
    prop_shared = (
        df_all.groupby(["Group", "Condition", "ResponseId"])["exp_response"]
        .mean()
        .to_frame("prop_shared")
        .reset_index()
    )
    share = prop_shared[prop_shared.Group == "Share"]

    forced = share[share.Condition == "Forced"]["prop_shared"]
    control = share[share.Condition == "Control"]["prop_shared"]

    prop_change = np.mean(forced) - np.mean(control)
    manu_stats = mannwhitneyu(forced, control)
    cohensd = cohen_d(forced, control)
    low95, high95 = mean_diff_bootstrap_ci(forced, control, d_only=False)

    print("Amount sharing increased between Forced and Control:")
    print("-" * 50)
    print(f"\t- Mean prop. change: {prop_change:.4f}")
    print(f"\t- Mean perc. change: {prop_change:.2%}")
    print("-" * 50)
    print("Mann Whitney U Results:")
    print(f"\t- U   : {manu_stats.statistic}")
    print(f"\t- pval: {manu_stats.pvalue}")
    print(f"\t- Cohen's d: {cohensd}")
    print(f"\t- 95% CI: [{low95:.4f}, {high95:.4f}]")
    print(f"\t- 95% CI (%): [{low95:.2%}, {high95:.2%}]")

    print(
        "\n########################################################################\n",
        "                       CHANGES IN  DISCERNMENT\n",
        "########################################################################\n",
    )

    mean_discernment = df_discern.groupby(["Group", "Condition"])["discernment"].mean()

    b_forced_v_control = (
        mean_discernment.Belief.Forced - mean_discernment.Belief.Control
    )
    b_optional_v_control = (
        mean_discernment.Belief.Optional - mean_discernment.Belief.Control
    )

    s_forced_v_control = mean_discernment.Share.Forced - mean_discernment.Share.Control
    s_optional_v_control = (
        mean_discernment.Share.Optional - mean_discernment.Share.Control
    )

    print("Belief:")
    print(
        f"\t- Forced - Control  : {b_forced_v_control:.4f} ",
        f"({b_forced_v_control:.2%})",
    )
    print(
        f"\t- Optional - Control: {b_optional_v_control:.4f} ",
        f"({b_optional_v_control:.2%})",
    )
    print("Share:")
    print(
        f"\t- Forced - Control  : {s_forced_v_control:.4f} ",
        f"({s_forced_v_control:.2%})",
    )
    print(
        f"\t- Optional - Control: {s_optional_v_control:.4f} ",
        f"({s_optional_v_control:.2%})",
    )
    print("-" * 10)
    print(
        "Note: Raw and percentage values rounded to "
        "the fourth and second decimal, respectively."
    )
    print("Note: See below for statistical tests.")

    print(
        "\n########################################################################\n",
        "                  GROUP COMPARISONS FOR DISCERNMENT\n",
        "########################################################################\n",
    )

    belief_data = df_discern[df_discern.Group == "Belief"]
    share_data = df_discern[df_discern.Group == "Share"]

    print("Belief ANOVA")
    print("-" * 50)
    print("\t- Model: discernment ~ C(Condition)")
    model_belief = ols("discernment ~ C(Condition)", data=belief_data).fit()
    anova_belief = sm.stats.anova_lm(model_belief, typ=2)
    print(anova_belief, "\n")

    print("Share ANOVA")
    print("-" * 50)
    print("\t- Model: discernment ~ C(Condition)")
    model_share = ols("discernment ~ C(Condition)", data=share_data).fit()
    anova_share = sm.stats.anova_lm(model_share, typ=2)
    print(anova_share, "\n")

    print("CHECK ASSUMPTIONS")
    print("-" * 50)

    print("Checking for normality...")
    print("-" * 25, "\n")

    print("Belief:")
    stat, p_value = shapiro(model_belief.resid)
    print(f"\t- Shapiro-Wilk Test Statistic: {stat}")
    print(f"\t- Shapiro-Wilk Test p-value: {p_value}\n")

    print("Share:")
    stat, p_value = shapiro(model_share.resid)
    print(f"\t- Shapiro-Wilk Test Statistic: {stat}")
    print(f"\t- Shapiro-Wilk Test p-value: {p_value}\n")

    print("Checking for homoscedasticity...")
    print("-" * 25, "\n")

    b_control = df_discern[
        (df_discern.Condition == "Control") & (df_discern.Group == "Belief")
    ].discernment
    b_forced = df_discern[
        (df_discern.Condition == "Forced") & (df_discern.Group == "Belief")
    ].discernment
    b_optional = df_discern[
        (df_discern.Condition == "Optional") & (df_discern.Group == "Belief")
    ].discernment

    s_control = df_discern[
        (df_discern.Condition == "Control") & (df_discern.Group == "Share")
    ].discernment
    s_forced = df_discern[
        (df_discern.Condition == "Forced") & (df_discern.Group == "Share")
    ].discernment
    s_optional = df_discern[
        (df_discern.Condition == "Optional") & (df_discern.Group == "Share")
    ].discernment

    print("Belief:")
    stat, p = levene(b_control, b_forced, b_optional)
    print(f"\t- Levene test statistic: {stat}")
    print(f"\t- Levene test p-value: {p}\n")

    print("Share:")
    stat, p = levene(s_control, s_forced, s_optional)
    print(f"\t- Levene test statistic: {stat}")
    print(f"\t- Levene test p-value: {p}\n")

    print("NON-PARAMETRIC ALTERNATIVES")
    print("-" * 50)

    print("Belief:")
    print("-" * 25)
    stat, p = kruskal(b_control, b_forced, b_optional)
    print(f"Kruskal-Wallis statistic: {stat}")
    print(f"Kruskal-Wallis test p-value: {p}\n")

    # Perform Mann-Whitney U tests
    n_groups = len(GROUP_NAMES)
    mwu_results = {}  # Dictionary to hold results
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            # Get names and values
            group1name = GROUP_NAMES[i]
            group2name = GROUP_NAMES[j]
            group1vals = eval(f"b_{group1name.lower()}")
            group2vals = eval(f"b_{group2name.lower()}")

            # Calculate effect sizes and 95% CIs
            cohensd = cohen_d(group1vals, group2vals)
            low95, high95 = mean_diff_bootstrap_ci(
                group1vals, group2vals, confidence=0.95, d_only=False
            )
            mwu = mannwhitneyu(
                group1vals,
                group2vals,
                alternative="two-sided",
            )
            mwu_results[(group1name, group2name)] = {
                "statistic": mwu.statistic,
                "p_value": mwu.pvalue,
                "p_value_corrected": bonferroni_correction(mwu.pvalue, n_groups),
                "cohens_d": cohensd,
                "95_ci": (low95, high95),
            }

    # Print results
    print("Mann-Whitney U Tests")
    for (group1, group2), result in mwu_results.items():
        print(f"\t- {group1} vs {group2}:")
        print(f'\t\t- Statistic: {result["statistic"]:.3f}')
        print(f'\t\t- P-value: {result["p_value"]:.3f}')
        print(
            f'\t\t- P-value (Bonferroni corrected): {result["p_value_corrected"]:.3f}'
        )
        print(f'\t\t- Cohens d: {result["cohens_d"]:.3f}')
        low_ci, high_ci = result["95_ci"]
        print(f"\t\t- 95% CI: [{low_ci:.4f}, {high_ci:.4f}]")
        print(f"\t\t- 95% CI (%): [{low_ci:.2%}, {high_ci:.2%}]")

    print("Share:")
    print("-" * 25)
    stat, p = kruskal(s_control, s_forced, s_optional)
    print(f"Kruskal-Wallis test statistic: {stat}")
    print(f"Kruskal-Wallis test p-value: {p}\n")

    # Perform Mann-Whitney U tests
    n_groups = len(GROUP_NAMES)
    mwu_results = {}  # Dictionary to hold results
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            # Get names and values
            group1name = GROUP_NAMES[i]
            group2name = GROUP_NAMES[j]
            group1vals = eval(f"s_{group1name.lower()}")
            group2vals = eval(f"s_{group2name.lower()}")

            # Calculate effect sizes and 95% CIs
            cohensd = cohen_d(group1vals, group2vals)
            low95, high95 = mean_diff_bootstrap_ci(
                group1vals, group2vals, confidence=0.95, d_only=False
            )
            mwu = mannwhitneyu(
                group1vals,
                group2vals,
                alternative="two-sided",
            )
            mwu_results[(group1name, group2name)] = {
                "statistic": mwu.statistic,
                "p_value": mwu.pvalue,
                "p_value_corrected": bonferroni_correction(mwu.pvalue, n_groups),
                "cohens_d": cohensd,
                "95_ci": (low95, high95),
            }

    # Print results
    print("Mann-Whitney U Tests")
    for (group1, group2), result in mwu_results.items():
        print(f"\t- {group1} vs {group2}:")
        print(f'\t\t- Statistic: {result["statistic"]:.3f}')
        print(f'\t\t- P-value: {result["p_value"]:.3f}')
        print(
            f'\t\t- P-value (Bonferroni corrected): {result["p_value_corrected"]:.3f}'
        )
        print(f'\t\t- Cohens d: {result["cohens_d"]:.3f}')
        low_ci, high_ci = result["95_ci"]
        print(f"\t\t- 95% CI: [{low_ci:.4f}, {high_ci:.4f}]")
        print(f"\t\t- 95% CI (%): [{low_ci:.2%}, {high_ci:.2%}]")

    print(
        "\n########################################################################\n",
        "               GROUP COMPARISONS FOR TRUE HEADLINES\n",
        "########################################################################\n",
    )

    print("Belief ANOVA")
    print("-" * 50)
    print("\t- Model: prop_yes_True ~ C(Condition)")
    model_belief = ols("prop_yes_True ~ C(Condition)", data=belief_data).fit()
    anova_belief = sm.stats.anova_lm(model_belief, typ=2)
    print(anova_belief, "\n")

    print("Share ANOVA")
    print("-" * 50)
    print("\t- Model: prop_yes_True ~ C(Condition)")
    model_share = ols("prop_yes_True ~ C(Condition)", data=share_data).fit()
    anova_share = sm.stats.anova_lm(model_share, typ=2)
    print(anova_share, "\n")

    print("CHECK ASSUMPTIONS")
    print("-" * 50)

    print("Checking for normality...")
    print("-" * 25, "\n")

    print("Belief:")
    stat, p_value = shapiro(model_belief.resid)
    print(f"\t- Shapiro-Wilk Test Statistic: {stat}")
    print(f"\t- Shapiro-Wilk Test p-value: {p_value}\n")

    print("Share:")
    stat, p_value = shapiro(model_share.resid)
    print(f"\t- Shapiro-Wilk Test Statistic: {stat}")
    print(f"\t- Shapiro-Wilk Test p-value: {p_value}\n")

    print("Checking for homoscedasticity...")
    print("-" * 25, "\n")

    b_control = df_discern[
        (df_discern.Condition == "Control") & (df_discern.Group == "Belief")
    ].prop_yes_True
    b_forced = df_discern[
        (df_discern.Condition == "Forced") & (df_discern.Group == "Belief")
    ].prop_yes_True
    b_optional = df_discern[
        (df_discern.Condition == "Optional") & (df_discern.Group == "Belief")
    ].prop_yes_True

    s_control = df_discern[
        (df_discern.Condition == "Control") & (df_discern.Group == "Share")
    ].prop_yes_True
    s_forced = df_discern[
        (df_discern.Condition == "Forced") & (df_discern.Group == "Share")
    ].prop_yes_True
    s_optional = df_discern[
        (df_discern.Condition == "Optional") & (df_discern.Group == "Share")
    ].prop_yes_True

    print("Belief:")
    stat, p = levene(b_control, b_forced, b_optional)
    print(f"\t- Levene test statistic: {stat}")
    print(f"\t- Levene test p-value: {p}\n")

    print("Share:")
    stat, p = levene(s_control, s_forced, s_optional)
    print(f"\t- Levene test statistic: {stat}")
    print(f"\t- Levene test p-value: {p}\n")

    print("NON-PARAMETRIC ALTERNATIVES")
    print("-" * 50)

    print("Belief:")
    print("-" * 25)
    stat, p = kruskal(b_control, b_forced, b_optional)
    print(f"Kruskal-Wallis statistic: {stat}")
    print(f"Kruskal-Wallis test p-value: {p}\n")

    # Perform Mann-Whitney U tests
    n_groups = len(GROUP_NAMES)
    mwu_results = {}  # Dictionary to hold results
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            # Get names and values
            group1name = GROUP_NAMES[i]
            group2name = GROUP_NAMES[j]
            group1vals = eval(f"b_{group1name.lower()}")
            group2vals = eval(f"b_{group2name.lower()}")

            # Calculate effect sizes and 95% CIs
            cohensd = cohen_d(group1vals, group2vals)
            low95, high95 = mean_diff_bootstrap_ci(
                group1vals, group2vals, confidence=0.95, d_only=False
            )
            mwu = mannwhitneyu(
                group1vals,
                group2vals,
                alternative="two-sided",
            )
            mwu_results[(group1name, group2name)] = {
                "statistic": mwu.statistic,
                "p_value": mwu.pvalue,
                "p_value_corrected": bonferroni_correction(mwu.pvalue, n_groups),
                "cohens_d": cohensd,
                "95_ci": (low95, high95),
            }

    # Print results
    print("Mann-Whitney U Tests")
    for (group1, group2), result in mwu_results.items():
        print(f"\t- {group1} vs {group2}:")
        print(f'\t\t- Statistic: {result["statistic"]:.3f}')
        print(f'\t\t- P-value: {result["p_value"]:.3f}')
        print(
            f'\t\t- P-value (Bonferroni corrected): {result["p_value_corrected"]:.3f}'
        )
        print(f'\t\t- Cohens d: {result["cohens_d"]:.3f}')
        low_ci, high_ci = result["95_ci"]
        print(f"\t\t- 95% CI: [{low_ci:.4f}, {high_ci:.4f}]")
        print(f"\t\t- 95% CI (%): [{low_ci:.2%}, {high_ci:.2%}]")

    print("Share:")
    print("-" * 25)
    stat, p = kruskal(s_control, s_forced, s_optional)
    print(f"Kruskal-Wallis test statistic: {stat}")
    print(f"Kruskal-Wallis test p-value: {p}\n")

    # Perform Mann-Whitney U tests
    n_groups = len(GROUP_NAMES)
    mwu_results = {}  # Dictionary to hold results
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            # Get names and values
            group1name = GROUP_NAMES[i]
            group2name = GROUP_NAMES[j]
            group1vals = eval(f"s_{group1name.lower()}")
            group2vals = eval(f"s_{group2name.lower()}")

            # Calculate effect sizes and 95% CIs
            cohensd = cohen_d(group1vals, group2vals)
            low95, high95 = mean_diff_bootstrap_ci(
                group1vals, group2vals, confidence=0.95, d_only=False
            )
            mwu = mannwhitneyu(
                group1vals,
                group2vals,
                alternative="two-sided",
            )
            mwu_results[(group1name, group2name)] = {
                "statistic": mwu.statistic,
                "p_value": mwu.pvalue,
                "p_value_corrected": bonferroni_correction(mwu.pvalue, n_groups),
                "cohens_d": cohensd,
                "95_ci": (low95, high95),
            }

    # Print results
    print("Mann-Whitney U Tests")
    for (group1, group2), result in mwu_results.items():
        print(f"\t- {group1} vs {group2}:")
        print(f'\t\t- Statistic: {result["statistic"]:.3f}')
        print(f'\t\t- P-value: {result["p_value"]:.3f}')
        print(
            f'\t\t- P-value (Bonferroni corrected): {result["p_value_corrected"]:.3f}'
        )
        print(f'\t\t- Cohens d: {result["cohens_d"]:.3f}')
        low_ci, high_ci = result["95_ci"]
        print(f"\t\t- 95% CI: [{low_ci:.4f}, {high_ci:.4f}]")
        print(f"\t\t- 95% CI (%): [{low_ci:.2%}, {high_ci:.2%}]")

    print(
        "\n########################################################################\n",
        "               GROUP COMPARISONS FOR FALSE HEADLINES\n",
        "########################################################################\n",
    )

    print("Belief ANOVA")
    print("-" * 50)
    print("\t- Model: prop_yes_False ~ C(Condition)")
    model_belief = ols("prop_yes_False ~ C(Condition)", data=belief_data).fit()
    anova_belief = sm.stats.anova_lm(model_belief, typ=2)
    print(anova_belief, "\n")

    print("Share ANOVA")
    print("-" * 50)
    print("\t- Model: prop_yes_False ~ C(Condition)")
    model_share = ols("prop_yes_False ~ C(Condition)", data=share_data).fit()
    anova_share = sm.stats.anova_lm(model_share, typ=2)
    print(anova_share, "\n")

    print("CHECK ASSUMPTIONS")
    print("-" * 50)

    print("Checking for normality...")
    print("-" * 25, "\n")

    print("Belief:")
    stat, p_value = shapiro(model_belief.resid)
    print(f"\t- Shapiro-Wilk Test Statistic: {stat}")
    print(f"\t- Shapiro-Wilk Test p-value: {p_value}\n")

    print("Share:")
    stat, p_value = shapiro(model_share.resid)
    print(f"\t- Shapiro-Wilk Test Statistic: {stat}")
    print(f"\t- Shapiro-Wilk Test p-value: {p_value}\n")

    print("Checking for homoscedasticity...")
    print("-" * 25, "\n")

    b_control = df_discern[
        (df_discern.Condition == "Control") & (df_discern.Group == "Belief")
    ].prop_yes_False
    b_forced = df_discern[
        (df_discern.Condition == "Forced") & (df_discern.Group == "Belief")
    ].prop_yes_False
    b_optional = df_discern[
        (df_discern.Condition == "Optional") & (df_discern.Group == "Belief")
    ].prop_yes_False

    s_control = df_discern[
        (df_discern.Condition == "Control") & (df_discern.Group == "Share")
    ].prop_yes_False
    s_forced = df_discern[
        (df_discern.Condition == "Forced") & (df_discern.Group == "Share")
    ].prop_yes_False
    s_optional = df_discern[
        (df_discern.Condition == "Optional") & (df_discern.Group == "Share")
    ].prop_yes_False

    print("Belief:")
    stat, p = levene(b_control, b_forced, b_optional)
    print(f"\t- Levene test statistic: {stat}")
    print(f"\t- Levene test p-value: {p}\n")

    print("Share:")
    stat, p = levene(s_control, s_forced, s_optional)
    print(f"\t- Levene test statistic: {stat}")
    print(f"\t- Levene test p-value: {p}\n")

    print("NON-PARAMETRIC ALTERNATIVES")
    print("-" * 50)

    print("Belief:")
    print("-" * 25)
    stat, p = kruskal(b_control, b_forced, b_optional)
    print(f"Kruskal-Wallis statistic: {stat}")
    print(f"Kruskal-Wallis test p-value: {p}\n")

    # Perform Mann-Whitney U tests
    n_groups = len(GROUP_NAMES)
    mwu_results = {}  # Dictionary to hold results
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            # Get names and values
            group1name = GROUP_NAMES[i]
            group2name = GROUP_NAMES[j]
            group1vals = eval(f"b_{group1name.lower()}")
            group2vals = eval(f"b_{group2name.lower()}")

            # Calculate effect sizes and 95% CIs
            cohensd = cohen_d(group1vals, group2vals)
            low95, high95 = mean_diff_bootstrap_ci(
                group1vals, group2vals, confidence=0.95, d_only=False
            )
            mwu = mannwhitneyu(
                group1vals,
                group2vals,
                alternative="two-sided",
            )
            mwu_results[(group1name, group2name)] = {
                "statistic": mwu.statistic,
                "p_value": mwu.pvalue,
                "p_value_corrected": bonferroni_correction(mwu.pvalue, n_groups),
                "cohens_d": cohensd,
                "95_ci": (low95, high95),
            }

    # Print results
    print("Mann-Whitney U Tests")
    for (group1, group2), result in mwu_results.items():
        print(f"\t- {group1} vs {group2}:")
        print(f'\t\t- Statistic: {result["statistic"]:.3f}')
        print(f'\t\t- P-value: {result["p_value"]:.3f}')
        print(
            f'\t\t- P-value (Bonferroni corrected): {result["p_value_corrected"]:.3f}'
        )
        print(f'\t\t- Cohens d: {result["cohens_d"]:.3f}')
        low_ci, high_ci = result["95_ci"]
        print(f"\t\t- 95% CI: [{low_ci:.4f}, {high_ci:.4f}]")
        print(f"\t\t- 95% CI (%): [{low_ci:.2%}, {high_ci:.2%}]")

    print("Share:")
    print("-" * 25)
    stat, p = kruskal(s_control, s_forced, s_optional)
    print(f"Kruskal-Wallis test statistic: {stat}")
    print(f"Kruskal-Wallis test p-value: {p}\n")

    # Perform Mann-Whitney U tests
    n_groups = len(GROUP_NAMES)
    mwu_results = {}  # Dictionary to hold results
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            # Get names and values
            group1name = GROUP_NAMES[i]
            group2name = GROUP_NAMES[j]
            group1vals = eval(f"s_{group1name.lower()}")
            group2vals = eval(f"s_{group2name.lower()}")

            # Calculate effect sizes and 95% CIs
            cohensd = cohen_d(group1vals, group2vals)
            low95, high95 = mean_diff_bootstrap_ci(
                group1vals, group2vals, confidence=0.95, d_only=False
            )
            mwu = mannwhitneyu(
                group1vals,
                group2vals,
                alternative="two-sided",
            )
            mwu_results[(group1name, group2name)] = {
                "statistic": mwu.statistic,
                "p_value": mwu.pvalue,
                "p_value_corrected": bonferroni_correction(mwu.pvalue, n_groups),
                "cohens_d": cohensd,
                "95_ci": (low95, high95),
            }

    # Print results
    print("Mann-Whitney U Tests")
    for (group1, group2), result in mwu_results.items():
        print(f"\t- {group1} vs {group2}:")
        print(f'\t\t- Statistic: {result["statistic"]:.3f}')
        print(f'\t\t- P-value: {result["p_value"]:.3f}')
        print(
            f'\t\t- P-value (Bonferroni corrected): {result["p_value_corrected"]:.3f}'
        )
        print(f'\t\t- Cohens d: {result["cohens_d"]:.3f}')
        low_ci, high_ci = result["95_ci"]
        print(f"\t\t- 95% CI: [{low_ci:.4f}, {high_ci:.4f}]")
        print(f"\t\t- 95% CI (%): [{low_ci:.2%}, {high_ci:.2%}]")
