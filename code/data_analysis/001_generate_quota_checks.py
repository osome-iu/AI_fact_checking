"""
Purpose: 
    Record basic statistics about the final sample.
    Statistically compare experimental groups to one another and record the results.

    Run via: python 001_generate_quota_checks.py > /path/to/output_dir/quota_checks.txt

Inputs:
    None

Outputs:
    New file: quota_checks.txt

Author: Matthew DeVerna
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

from itertools import combinations

DATA_CLEANING_DIR = "../data_cleaning"
sys.path.insert(0, DATA_CLEANING_DIR)
# Imports the names of columns in the loaded data by each section of the study
from db import find_file

DATA_DIR = "../../data/cleaned_data/"
# FNAME = find_file(DATA_DIR, "*long_form.parquet")
FNAME = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")
ROOT_DIR = "data_analysis"


# Function to perform pairwise chi-squared tests
def pairwise_chi2_test(data, group_col, condition_col, count_col, alpha=0.05):
    results = []
    # Get unique conditions
    conditions = data[condition_col].unique()
    # Perform pairwise comparisons
    for cond1, cond2 in combinations(conditions, 2):
        sub_data = data[data[condition_col].isin([cond1, cond2])]
        contingency_table = (
            sub_data.pivot(index=group_col, columns=condition_col, values=count_col)
            .fillna(0)
            .values
        )
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        results.append((cond1, cond2, chi2_stat, p_val))

    # Adjust p-values using Bonferroni correction
    results_df = pd.DataFrame(
        results,
        columns=["Condition 1", "Condition 2", "Chi-squared Statistic", "p-value"],
    )
    results_df["Adjusted p-value"] = results_df["p-value"] * len(results)

    return results_df


def get_participant_race(response):
    """
    Return the race of the respondent based on all of their response.

    Parameters:
    -----------
    - response (list of str): All options selected by the respondent. See preregistration
        for details.
    """
    if len(response) == 1:
        if response[0] in [
            "American Indian or Alaska Native",
            "Native Hawaiian or Pacific Islander",
        ]:
            return "Other"
        else:
            return response[0]

    elif any("Hispanic or Latino/a" in i for i in response):
        return "Hispanic or Latino/a"

    elif any("Black or African American" in i for i in response):
        return "Black or African American"

    elif any("Asian" in i for i in response):
        return "Asian"

    else:
        all_selected = " / ".join(response)
        print(f"Classified as 'Other' based on: [{all_selected}]")
        return "Other"


if __name__ == "__main__":
    # Ensure we are in the data_analysis directory for paths to work
    if os.path.basename(os.getcwd()) != ROOT_DIR:
        raise Exception(
            "Must run this script from the `code/data_analysis/` directory!"
        )

    # Load data files and create a column for each experimental group
    df = pd.read_parquet(FNAME)
    df.drop_duplicates(subset=["ResponseId"], inplace=True)

    ### SAMPLE DEMOGRAPHICS ###
    ###########################
    print("SAMPLE DEMOGRAPHICS")
    print("-" * 50)
    total_participants = len(df)
    print(f"Total num participants: {total_participants}\n")

    ### GENDER ###
    # --------------------------#
    print("Gender:")
    print("-" * 50)
    print("COUNTS:")
    print("-" * 10)
    print(df["gender"].value_counts(), "\n")

    print("PERCENTAGES:")
    print("-" * 10)
    print((df["gender"].value_counts() / total_participants) * 100, "\n")

    ### AGE ###
    # --------------------------#

    # Calculate participant age. Human-FC groups subtracted from 2024 vs. the others
    # from 2023 given that the study was conducted at different times.
    human_fc_mask = df["Condition"] == "Human-FC"
    df.loc[human_fc_mask, "age"] = 2024 - df.loc[human_fc_mask, "year_birth"]
    df.loc[~human_fc_mask, "age"] = 2023 - df.loc[~human_fc_mask, "year_birth"]

    bins = [18, 25, 35, 45, 55, 65, np.inf]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["age_bracket"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels,
        right=False,  # Makes sure to NOT include right edge of bucket
    )

    print("Age:")
    print("-" * 50)
    print("COUNTS:")
    print("-" * 10)
    print(df["age_bracket"].value_counts(), "\n")

    print("PERCENTAGES:")
    print("-" * 10)
    print((df["age_bracket"].value_counts() / total_participants) * 100, "\n")

    ### RACE ###
    # --------------------------#

    # Categorize participants based on race
    df["race_split"] = df.race.str.split(",")
    df["race_final"] = df.race_split.map(get_participant_race)

    print("Race:")
    print("-" * 50)
    print("COUNTS:")
    print("-" * 10)
    print(df["race_final"].value_counts(), "\n")

    print("PERCENTAGES:")
    print("-" * 10)
    print((df["race_final"].value_counts() / total_participants) * 100, "\n")

    ### EDUCATION ###
    # --------------------------#
    print("Education:")
    print("-" * 50)
    print("COUNTS:")
    print("-" * 10)
    education_counts = df["edu"].value_counts()
    print(education_counts, "\n")

    less_than_college = (
        education_counts[
            "High school graduate (high school diploma or equivalent including GED)"
        ]
        + education_counts["Some college but no degree"]
        + education_counts["Less than high school degree"]
    )

    college_and_beyond = (
        education_counts["Bachelor's degree in college (4-year)"]
        + education_counts["Master's degree"]
        + education_counts["Associate degree in college (2-year)"]
        + education_counts["Professional degree (JD, MD)"]
        + education_counts["Doctoral degree"]
    )

    print("COUNTS COMBINED:")
    print("-" * 10)
    print(f"Counts less than college : {less_than_college}")
    print(f"Counts college and beyond: {college_and_beyond}\n")

    percent_less_than_college = (less_than_college / total_participants) * 100
    percent_college_and_beyond = (college_and_beyond / total_participants) * 100

    print("PERCENTAGES COMBINED:")
    print("-" * 10)
    print(f"Percent less than college : {percent_less_than_college}%")
    print(f"Percent college and beyond: {percent_college_and_beyond}%")

    ### Party ID ###
    # --------------------------#
    print("Party ID:")
    print("-" * 50)
    print("COUNTS:")
    print("-" * 10)
    print(df["party_recoded"].value_counts(), "\n")

    print("PERCENTAGES:")
    print("-" * 10)
    print((df["party_recoded"].value_counts() / total_participants) * 100, "\n")

    ### CHECKING FOR GROUP DIFFERENCES ###
    # Note on stats:
    # - The chi2_contingency is a convenience function here since we do not know what the
    #       expected frequencies should be and we are comparing more than two groups.
    #       By running the correct = False (vs. True) we are not using the Yates' correction.
    #           Ref: https://en.wikipedia.org/wiki/Yates%27s_correction_for_continuity
    # - The distributions are not statistically different whether we include the correction or not.
    # - See the function documention for more details
    ######################################
    print("CHECKING FOR GROUP DIFFERENCES")
    print("#" * 50)
    print("#" * 50)

    ### Gender ###
    # --------------------------#
    print("Gender:")
    print("-" * 50)
    print("COUNTS:")
    print("-" * 10)

    counts_by_group_and_gender = (
        df.groupby(["Group", "Condition", "gender"])["gender"]
        .count()
        .to_frame("count")
        .reset_index()
    )
    print(counts_by_group_and_gender, "\n")

    num_other = counts_by_group_and_gender[
        counts_by_group_and_gender.gender == "Other"
    ]["count"].sum()
    print(f"Drop {num_other} participants who report as 'Other / Non-binary'\n")
    counts_by_group_and_gender = counts_by_group_and_gender[
        counts_by_group_and_gender.gender != "Other"
    ]

    print("STATISTICAL DIFFERENCE BASED ON GROUP:")
    print("-" * 10)
    for group in ["Belief", "Share"]:
        group_counts = counts_by_group_and_gender[
            counts_by_group_and_gender.Group == group
        ]

        chi2_stat, p_val, dof, expected_freq = stats.chi2_contingency(
            group_counts.pivot(
                index="gender", columns=["Group", "Condition"], values="count"
            ).values,
            correction=False,
        )

        print("\t- Group:", group)
        print("\t- Chi-squared statistic:", chi2_stat)
        print("\t- p-value:", p_val)
        print("\t- Degrees of freedom:", dof)
        print("-" * 50, "\n")

    ### RACE ###
    # --------------------------#
    print("Race:")
    print("-" * 50)
    print("COUNTS:")
    print("-" * 10)

    counts_by_group_and_race = (
        df.groupby(["Group", "Condition", "race_final"])["race_final"]
        .count()
        .to_frame("count")
        .reset_index()
    )

    print(counts_by_group_and_race, "\n")

    print("STATISTICAL DIFFERENCE BASED ON GROUP:")
    print("-" * 10)
    for group in ["Belief", "Share"]:
        group_counts = counts_by_group_and_race[counts_by_group_and_race.Group == group]

        chi2_stat, p_val, dof, expected_freq = stats.chi2_contingency(
            group_counts.pivot(
                index="race_final", columns=["Group", "Condition"], values="count"
            ).values,
            correction=False,
        )

        print("Group:", group)
        print("Chi-squared statistic:", chi2_stat)
        print("p-value:", p_val)
        print("Degrees of freedom:", dof)
        print("-" * 50, "\n")

    ### AGE ###
    # --------------------------#
    print("Age:")
    print("-" * 50)
    print("COUNTS:")
    print("-" * 10)

    counts_by_group_and_age = (
        df.groupby(["Group", "Condition", "age_bracket"])["age_bracket"]
        .count()
        .to_frame("count")
        .reset_index()
    )

    print(counts_by_group_and_age, "\n")

    print("STATISTICAL DIFFERENCE BASED ON GROUP:")
    print("-" * 10)
    for group in ["Belief", "Share"]:
        group_counts = counts_by_group_and_age[counts_by_group_and_age.Group == group]

        chi2_stat, p_val, dof, expected_freq = stats.chi2_contingency(
            group_counts.pivot(
                index="age_bracket", columns=["Group", "Condition"], values="count"
            ).values,
            correction=False,
        )

        print("Group:", group)
        print("Chi-squared statistic:", chi2_stat)
        print("p-value:", p_val)
        print("Degrees of freedom:", dof)
        print("-" * 50, "\n")

    ### EDUCATION ###
    # --------------------------#
    print("Education:")
    print("-" * 50)
    print("COUNTS:")
    print("-" * 10)

    edu_simp_map = {
        "Associate degree in college (2-year)": "degree",
        "Bachelor's degree in college (4-year)": "degree",
        "Doctoral degree": "degree",
        "Professional degree (JD, MD)": "degree",
        "Master's degree": "degree",
        "Some college but no degree": "no degree",
        "High school graduate (high school diploma or equivalent including GED)": "no degree",
        "Less than high school degree": "no degree",
    }

    df["edu_simple"] = df.edu.map(edu_simp_map)

    counts_by_group_and_edu = (
        df.groupby(["Group", "Condition", "edu_simple"])["edu_simple"]
        .count()
        .to_frame("count")
        .reset_index()
    )

    print(counts_by_group_and_edu, "\n")

    print("STATISTICAL DIFFERENCE BASED ON GROUP:")
    print("-" * 10)
    for group in ["Belief", "Share"]:
        group_counts = counts_by_group_and_edu[counts_by_group_and_edu.Group == group]

        chi2_stat, p_val, dof, expected_freq = stats.chi2_contingency(
            group_counts.pivot(
                index="edu_simple", columns=["Group", "Condition"], values="count"
            ).values,
            correction=False,
        )

        print("Group:", group)
        print("Chi-squared statistic:", chi2_stat)
        print("p-value:", p_val)
        print("Degrees of freedom:", dof)
        print("-" * 50, "\n")

        if p_val < 0.05:
            print("Significant difference between groups found")
            print("Performing pairwise chi-squared tests for group:", group)
            pairwise_results = pairwise_chi2_test(
                group_counts, "edu_simple", "Condition", "count"
            )
            print(pairwise_results)
            print("-" * 50, "\n")

    ### PARTY ###
    # --------------------------#
    print("Party ID:")
    print("-" * 50)
    print("COUNTS:")
    print("-" * 10)

    counts_by_group_and_party_id = (
        df.groupby(["Group", "Condition", "party_recoded"])["party_recoded"]
        .count()
        .to_frame("count")
        .reset_index()
    )

    print(counts_by_group_and_party_id, "\n")

    print("STATISTICAL DIFFERENCE BASED ON GROUP:")
    print("-" * 10)
    for group in ["Belief", "Share"]:
        group_counts = counts_by_group_and_party_id[
            counts_by_group_and_party_id.Group == group
        ]

        chi2_stat, p_val, dof, expected_freq = stats.chi2_contingency(
            group_counts.pivot(
                index="party_recoded", columns=["Group", "Condition"], values="count"
            )
            .dropna()
            .values,
            correction=False,
        )

        print("Group:", group)
        print("Chi-squared statistic:", chi2_stat)
        print("p-value:", p_val)
        print("Degrees of freedom:", dof)
        print("-" * 50, "\n")

        if p_val < 0.05:
            print("Significant difference between groups found")
            print("Performing pairwise chi-squared tests for group:", group)
            pairwise_results = pairwise_chi2_test(
                group_counts, "party_recoded", "Condition", "count"
            )
            print(pairwise_results)
            print("-" * 50, "\n")
