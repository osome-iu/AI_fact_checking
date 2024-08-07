"""
Purpose: 
    Calculate attrition stats.

    Run via: python 015_generate_attrition_stats.py > /path/to/output_dir/attrition_stats.txt

Notes:
    This script is excluded from the data analysis pipeline as it operates on uncleaned data.

Inputs:
    None

Outputs:
    New file: attrition_stats.txt

Author: Matthew DeVerna
"""

import itertools
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

INPUT_DATA_DIR = "../../data/raw_response_data"
INPUT_FILE_NAME1 = "ChatGPT_Intervention_misinfo_April_7_2023.csv"
INPUT_FILE_NAME2 = "ChatGPT_Intervention_Additional_June_29_2024.csv"


# Define the function to perform pairwise chi-squared tests
def pairwise_chi_squared(df, group_col, count_cols):
    pairs = list(itertools.combinations(df[group_col].unique(), 2))
    results = []

    for group1, group2 in pairs:
        # Only make comparisons within group and versus the control group
        if group1.split("-")[0] != group2.split("-")[0]:
            continue

        if group1.split("-")[1] != "Control":
            continue

        data1 = df[df[group_col] == group1][count_cols].values.flatten()
        data2 = df[df[group_col] == group2][count_cols].values.flatten()

        obs = pd.DataFrame(
            [data1, data2], columns=["tot_attrition_cnt", "good_comp_cnt"]
        )
        chi2, p, dof, ex = stats.chi2_contingency(obs)

        results.append({"Group1": group1, "Group2": group2, "Chi2": chi2, "p-value": p})

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Load the control + LLM conditions data
    old_data = pd.read_csv(
        os.path.join(INPUT_DATA_DIR, INPUT_FILE_NAME1),
        skiprows=[1, 2],
        low_memory=False,
    )

    # Load the human fact check conditions data
    new_data = pd.read_csv(
        os.path.join(INPUT_DATA_DIR, INPUT_FILE_NAME2),
        skiprows=[1, 2],
        low_memory=False,
    )

    # Get rid of preview
    old_data = old_data[old_data.Status == "IP Address"]
    new_data = new_data[new_data.Status == "IP Address"]

    # Find final data collection time
    ## B/c after data collection finished, people may trickle in.
    ## These should not count.

    old_data["EndDate"] = pd.to_datetime(old_data.EndDate)
    old_data["StartDate"] = pd.to_datetime(old_data.StartDate)

    new_data["EndDate"] = pd.to_datetime(new_data.EndDate)
    new_data["StartDate"] = pd.to_datetime(new_data.StartDate)

    new_data_end_date = new_data[new_data.gc == 1].EndDate.max()
    old_data_end_date = old_data[old_data.gc == 1].EndDate.max()

    old_data = old_data[old_data.EndDate < old_data_end_date]
    new_data = new_data[new_data.EndDate < new_data_end_date]

    # The number of good participants in each group
    cond_gc_counts = {
        "Belief-Control": 241,
        "Share-Control": 267,
        "Belief-Forced": 247,
        "Share-Forced": 269,
        "Belief-Optional": 261,
        "Share-Optional": 263,
        "Belief-HumanFC": 300,
        "Share-HumanFC": 311,
    }

    # Create a dictionary of the group/condtion and suffix of qualtrics question names
    cond_att_check_q_dict = {
        "Belief-Control": "_ctrl_belief",
        "Share-Control": "_ctrl_share",
        "Belief-Forced": "_td_belief",
        "Share-Forced": "_td_share",
        "Belief-Optional": "_to_belief",
        "Share-Optional": "_to_share",
        "Belief-HumanFC": "_td_belief",
        "Share-HumanFC": "_td_share",
    }

    records = []

    for group, question_suffix in cond_att_check_q_dict.items():
        record = {"group_cond": group}

        # Select the right data to work with
        if "HumanFC" in group:
            temp_df = new_data
        else:
            temp_df = old_data

        # Select only the experimental question columns
        exp_stim_qs_cols = temp_df.columns[
            temp_df.columns.str.endswith(question_suffix)
        ]

        # Select experimental column data
        temp_exp_data = temp_df[exp_stim_qs_cols]

        # Drop rows that are all NaNs, meaning they represent participants in other groups
        temp_exp_data = temp_exp_data.dropna(how="all")

        # Build the column name for the attention check question
        atten_check_question_col = f"41{question_suffix}"

        # Count attention checks and drop outs
        counts = temp_exp_data[atten_check_question_col].value_counts(dropna=False)
        counts = {
            "atten_check_fail_cnt": counts["No"],
            "drop_out_cnt": counts[np.nan],
            "good_comp_cnt": cond_gc_counts[group],
        }
        counts["tot_attrition_cnt"] = (
            counts["atten_check_fail_cnt"] + counts["drop_out_cnt"]
        )
        counts["total_entered"] = counts["tot_attrition_cnt"] + counts["good_comp_cnt"]

        record.update(counts)
        records.append(record)

    # Create a data frame with counts
    final_df = pd.DataFrame.from_records(records)

    # Iterate over count columns of each type and calculate the
    # proportion, relative to the total number of participants who entered that group
    cnt_columns = final_df.columns[final_df.columns.str.endswith("cnt")]
    for col in cnt_columns:
        proportions = final_df[col] / final_df["total_entered"]
        final_df[col.replace("cnt", "prop")] = proportions

    final_df = final_df.sort_values("group_cond")

    print("Group \t\t\t| Att. Check Fail % \t| Drop out % \t| Total %")
    print("-" * 65)

    for i, row in final_df.iterrows():
        group_name = row.group_cond
        atten_check_fail_prop = f"{row.atten_check_fail_prop:.2%}"
        drop_out_prop = row.drop_out_prop
        tot_attrition_prop = row.tot_attrition_prop

        if group_name == "Share-Forced":
            atten_check_fail_prop = "n/a"
        print(
            f"{group_name} |"
            + f" {atten_check_fail_prop} |"
            + f" {drop_out_prop:.2%} |"
            + f" {tot_attrition_prop:.2%}"
        )
    print(
        "Note: We experienced Qualtrics data collection error with respect to the Shared-Forced group's attention check question."
    )

    # Calculate other types of attrition for subjects who were not assigned to a group
    new_counts = new_data["term"].value_counts()
    new_consent = new_counts.consent
    new_age = new_counts.age
    new_vouching = new_counts.quality
    new_non_us_resident = new_counts.reside

    old_counts = old_data["term"].value_counts()
    old_consent = old_counts.Consent
    old_age = old_counts.Birth_year
    old_vouching = old_counts.Self_vouching
    old_non_us_resident = old_counts.Residence + old_counts.State

    total_consent = new_consent + old_consent
    total_age = new_age + old_age
    total_vouching = new_vouching + old_vouching
    total_non_us_resident = new_non_us_resident + old_non_us_resident

    print("\n\nNon-experimental attrition*:")
    print(f"\t- Did not consent: {total_consent}")
    print(f"\t- Too young: {total_age}")
    print(f"\t- Would not vouch for good answers: {total_vouching}")
    print(f"\t- Not US resident: {total_non_us_resident}")
    print("*Subjects were never assigned to an experimental group")

    # Test differential attrition for each group
    print("\n\nDifferential attrition:")
    print("-" * 50)
    belief_df = final_df[final_df["group_cond"].str.startswith("Belief")]
    share_df = final_df[final_df["group_cond"].str.startswith("Share")]
    # We exclude this group because there was an error with data collection
    share_df = share_df[share_df["group_cond"] != "Share-Forced"]

    print("Comparing Belief groups (Chi-squared contingency):")
    results = stats.chi2_contingency(belief_df[["good_comp_cnt", "tot_attrition_cnt"]])

    print(f"Statistic: {results.statistic}")
    print(f"pvalue   : {results.pvalue}\n")

    print("Comparing Share groups (Chi-squared contingency):")
    results = stats.chi2_contingency(share_df[["good_comp_cnt", "tot_attrition_cnt"]])
    print(f"Statistic: {results.statistic}")
    print(f"pvalue   : {results.pvalue}\n")

    # We find differential attrition amongst the groups, now we isolate
    # the groups that have differential attrition
    # Perform pairwise chi-squared tests
    print("\n\nPairwise chi-squared tests (Belief):")
    pairwise_results = pairwise_chi_squared(
        belief_df, "group_cond", ["tot_attrition_cnt", "good_comp_cnt"]
    )
    pairwise_results["p-value-adj"] = pairwise_results["p-value"] * len(
        pairwise_results
    )

    # Determine significant pairs
    pairwise_results["Significant"] = pairwise_results["p-value-adj"] < 0.05
    print(pairwise_results[["Group1", "Group2", "p-value-adj", "Significant"]])

    print("\n\nPairwise chi-squared tests (Share):")
    pairwise_results = pairwise_chi_squared(
        share_df, "group_cond", ["tot_attrition_cnt", "good_comp_cnt"]
    )
    pairwise_results["p-value-adj"] = pairwise_results["p-value"] * len(
        pairwise_results
    )

    # Determine significant pairs
    pairwise_results["Significant"] = pairwise_results["p-value-adj"] < 0.05
    print(pairwise_results[["Group1", "Group2", "p-value-adj", "Significant"]])
