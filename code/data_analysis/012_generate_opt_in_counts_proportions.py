"""
Purpose: 
    Generate the counts and proportions for when participants opted into seeing fact checks.

    Run via:
    python 012_generate_opt_in_counts_proportions.py > /path/to/output_dir/opt_in_counts_proportions_stats.txt

Inputs:
    None

Outputs:
    1. New file: opt_in_counts_proportions_stats.txt containing counts and proportions and statistical comparisons
    2. The different data frames generated in the script
        - opt_in_counts_proportions_by_veracity.csv: number and proportion of headlines opted into
            and out of, by group and veracity
        - opt_in_counts_proportions.csv: number and proportion of headlines opted into and out of, by group

Author: Matthew DeVerna
"""

import os
import sys

import pandas as pd
import scipy.stats as stats

from itertools import combinations

# Ensure we are in the script's directory for paths to work
if os.getcwd() != os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

NUM_HEADLINES = 40
OUTPUT_DIR = "../../results/"
DATA_CLEANING_DIR = "../data_cleaning"
sys.path.insert(0, DATA_CLEANING_DIR)
from db import find_file

# Get file version and load
DATA_DIR = "../../data/cleaned_data/"
# FNAME = find_file(DATA_DIR, "*long_form.parquet")
FNAME = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")

# Load data, select only what we need
all_data = pd.read_parquet(FNAME)
optional_df = all_data[all_data["Condition"] == "Optional"]

### Generate counts for each group by veracity
# --------------------------------------------
opt_in_counts = (
    optional_df.groupby(["Group", "ResponseId", "veracity"])["option_cond"]
    .value_counts()
    .to_frame("count")
    .reset_index()
)
opt_in_props = (
    optional_df.groupby(["Group", "ResponseId", "veracity"])["option_cond"]
    .value_counts(normalize=True)
    .to_frame("proportion")
    .reset_index()
)

opt_in_freq = pd.merge(
    left=opt_in_counts,
    right=opt_in_props,
    on=["Group", "ResponseId", "option_cond", "veracity"],
)

# Lookup dict for each participants group
participant_group_lookup = {
    data[0]: data[1]
    for idx, data in all_data[["ResponseId", "Group"]].drop_duplicates().iterrows()
}


# Create a MultiIndex with all possible combinations
response_ids = opt_in_freq["ResponseId"].unique()
veracities = [True, False]
option_conds = ["Opt_in", "Opt_out"]

multi_index = pd.MultiIndex.from_product(
    [response_ids, veracities, option_conds],
    names=["ResponseId", "veracity", "option_cond"],
)

# Use it to create a DataFrame with the complete set of combinations, then add the Group
complete_df = pd.DataFrame(index=multi_index).reset_index()
complete_df["Group"] = complete_df.ResponseId.map(participant_group_lookup)

# Merge the complete DataFrame with the original data
merged_df = pd.merge(
    complete_df,
    opt_in_freq,
    on=["ResponseId", "veracity", "option_cond", "Group"],
    how="left",
)

# Fill missing values with zeros
merged_df["count"] = merged_df["count"].fillna(0)
merged_df["proportion"] = merged_df["proportion"].fillna(0)

# # Sort the DataFrame by ResponseId to maintain order
by_veracity_df = merged_df.sort_values(
    by=["Group", "ResponseId", "veracity", "option_cond"]
)

# Save the results
by_veracity_df.to_csv(
    os.path.join(OUTPUT_DIR, "opt_in_counts_proportions_by_veracity.csv"), index=False
)

### Aggregate counts and proportions by group
# --------------------------------------------
by_group_df = (
    by_veracity_df.groupby(["Group", "ResponseId", "option_cond"])["count"]
    .sum()
    .to_frame("count")
    .reset_index()
)
by_group_df["proportion"] = by_group_df["count"] / NUM_HEADLINES

# A couple of sanity checks...
counts_equal_40 = all(
    val == 40
    for val in by_group_df.groupby(["Group", "ResponseId"])["count"].sum().values
)
props_equal_one = all(
    round(val, 5) == 1.0
    for val in by_group_df.groupby(["Group", "ResponseId"])["proportion"].sum().values
)
assert counts_equal_40, "Error: At least one count does not add up to 40!"
assert props_equal_one, "Error: At least one proportions does not add up to 1!"

# Save the results
by_group_df.to_csv(
    os.path.join(OUTPUT_DIR, "opt_in_counts_proportions.csv"), index=False
)

### Generate statistical comparisons
# --------------------------------------------

print("By group".upper())
print("#" * 50, "\n")

by_group_opt_in_df = by_group_df[by_group_df["option_cond"] == "Opt_in"]

print("Mann-Whitney U test: Belief vs. Share Proportion of Fact Checks Opt'd Into")
print("~" * 50)
U, p = stats.mannwhitneyu(
    by_group_opt_in_df[by_group_opt_in_df["Group"] == "Belief"]["proportion"],
    by_group_opt_in_df[by_group_opt_in_df["Group"] == "Share"]["proportion"],
)
print(f"U = {U}, p = {p}\n")

print("Opt In Counts (mean)")
print("-" * 50)
print(by_group_opt_in_df.groupby("Group")["count"].mean(), "\n")

print("Opt In Counts (standard deviation)")
print("-" * 50)
print(by_group_opt_in_df.groupby("Group")["count"].std(), "\n")

print("Opt In Counts (median)")
print("-" * 50)
print(by_group_opt_in_df.groupby("Group")["count"].median(), "\n")

print("Opt In Proportions (mean)")
print("-" * 50)
print(by_group_opt_in_df.groupby("Group")["proportion"].mean(), "\n")

print("Opt In Proportions (standard deviation)")
print("-" * 50)
print(by_group_opt_in_df.groupby("Group")["proportion"].std(), "\n")

print("Opt In Proportions (median)")
print("-" * 50)
print(by_group_opt_in_df.groupby("Group")["proportion"].median(), "\n")

print("Number and proportion of participants who opt into > 20 fact checks")
print("-" * 50)
more_than_20 = by_group_opt_in_df[by_group_opt_in_df["count"] > 20]
less_than_or_equal_20 = by_group_opt_in_df[by_group_opt_in_df["count"] <= 20]
num_greater_than_20 = len(more_than_20)
proportion_greater_than_20 = num_greater_than_20 / len(by_group_opt_in_df)
print(f"{num_greater_than_20} ({proportion_greater_than_20:.2%})\n")

print(
    "Mean (std) and median of counts for participants who opt into > 20 and <= 20 fact checks"
)
print("-" * 50)
mean_greater_than_20 = more_than_20["count"].mean()
median_greater_than_20 = more_than_20["count"].median()
std_greater_than_20 = more_than_20["count"].std()
mean_less_than_20 = less_than_or_equal_20["count"].mean()
median_less_than_20 = less_than_or_equal_20["count"].median()
std_less_than_20 = less_than_or_equal_20["count"].std()
print(f"Mean (std) for > 20: {mean_greater_than_20:.2f} ({std_greater_than_20:.2f})")
print(f"Median: {median_greater_than_20}")
print(f"Mean (std) for <= 20: {mean_less_than_20:.2f} ({std_less_than_20:.2f})")
print(f"Median: {median_less_than_20}\n")


print("\n\nBy veracity".upper())
print("#" * 50, "\n")

by_group_opt_in_df = by_group_df[by_group_df["option_cond"] == "Opt_in"]

opt_in_by_veracity = by_veracity_df[by_veracity_df["option_cond"] == "Opt_in"]
belief_df = opt_in_by_veracity[opt_in_by_veracity["Group"] == "Belief"]
share_df = opt_in_by_veracity[opt_in_by_veracity["Group"] == "Share"]

# List of groups
groups = [
    belief_df[belief_df["veracity"] == True]["proportion"],
    belief_df[belief_df["veracity"] == False]["proportion"],
    share_df[share_df["veracity"] == True]["proportion"],
    share_df[share_df["veracity"] == False]["proportion"],
]
group_names = ["belief-true", "belief-false", "share-true", "share-false"]


print("Kruskal-Wallis test: Proportion of Fact Checks Opt'd Into by Veracity")
print("~" * 50)

# Perform Kruskal-Wallis test
stat, p = stats.kruskal(*groups)
print(f"Kruskal-Wallis statistic: {stat}")
print(f"Kruskal-Wallis test p-value: {p}\n")

print("Mann-Whitney U tests (Bonferroni corrected)")
print("~" * 50)

# Perform pairwise comparisons
comparisons = list(combinations(range(len(groups)), 2))
p_values = []

for i, j in comparisons:
    groupname1, groupname2 = group_names[i], group_names[j]
    # Skip comparisons involving different groups
    if groupname1.split("-")[0] != groupname2.split("-")[0]:
        continue

    stat, p = stats.mannwhitneyu(groups[i], groups[j])
    p_values.append((groupname1, groupname2, p))

# Apply Bonferroni correction
corrected_p_values = [(g1, g2, p * len(p_values)) for (g1, g2, p) in p_values]

# Display the results
print(f"Bonferroni corrected p-values (num. groups = {len(p_values)}):")
for g1, g2, p in corrected_p_values:
    print(f"\t- Comparison: {g1} vs {g2}, p-value: {p}")

print("\nInterpretation with Bonferroni correction:")
alpha = 0.05
for g1, g2, p in corrected_p_values:
    if p < alpha:
        print(f"\t\t- Significant difference between {g1} and {g2}")
    else:
        print(f"\t- No significant difference between {g1} and {g2}")

print("\nOpt In Counts  (mean)")
print("-" * 50)
print(opt_in_by_veracity.groupby(["Group", "option_cond", "veracity"])["count"].mean())

print("\nOpt In Counts  (standard deviation)")
print("-" * 50)
print(opt_in_by_veracity.groupby(["Group", "option_cond", "veracity"])["count"].std())

print("\nOpt In Counts (median)")
print("-" * 50)
print(
    opt_in_by_veracity.groupby(["Group", "option_cond", "veracity"])["count"].median()
)

print("\nOpt In Proportions (mean)")
print("-" * 50)
print(
    opt_in_by_veracity.groupby(["Group", "option_cond", "veracity"])[
        "proportion"
    ].mean()
)

print("\nOpt In Proportions (standard deviation)")
print("-" * 50)
print(
    opt_in_by_veracity.groupby(["Group", "option_cond", "veracity"])["proportion"].std()
)

print("\nOpt In Proportions (median)")
print("-" * 50)
print(
    opt_in_by_veracity.groupby(["Group", "option_cond", "veracity"])[
        "proportion"
    ].median()
)
