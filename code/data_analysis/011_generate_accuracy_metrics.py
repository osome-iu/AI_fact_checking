"""
Purpose: 
    Generate precision, recall, and F1 for ChatGPTs judgments.
    Note: Since some judgments are coded as "unsure" we calculate these results
        after coding the "unsure" items as BOTH True and False separately.

    Run via:
    python 011_generate_accuracy_metrics.py > /path/to/output_dir/accuracy_metrics.txt

Inputs:
    None

Outputs:
    New file: accuracy_metrics.txt

Author: Matthew DeVerna
"""

import os
import sys

import pandas as pd

from metrics import df_precision, df_recall

UNSURE_TRUE_MAP = {"true": True, "false": False, "unsure": True}
UNSURE_FALSE_MAP = {"true": True, "false": False, "unsure": False}

# Ensure we are in the script's directory for paths to work
if os.getcwd() != os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

DATA_CLEANING_DIR = "../data_cleaning"
sys.path.insert(0, DATA_CLEANING_DIR)
from db import find_file

# Get file version and load
DATA_DIR = "../../data/cleaned_data/"
# FNAME = find_file(DATA_DIR, "*long_form.parquet")
FNAME = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")

# Load data, select only what we need, and rename column to work with the metric functions
all_data = pd.read_parquet(FNAME)
headlines_df = all_data.drop_duplicates(subset="qualtrics_question_num").reset_index(
    drop=True
)
headlines_df = headlines_df[
    ["qualtrics_question_num", "veracity", "ano_true_false_unsure"]
]
headlines_df = headlines_df.rename(columns={"ano_true_false_unsure": "judgment"})

print("Model Accuracy".upper())
print("#" * 50)
print(f"\nCondition: Raw".upper())
print("%" * 50, "\n")

print("Counts")
print("~" * 50)
print(headlines_df.groupby("veracity")["judgment"].value_counts())
print("\n")

print("Proportions")
print("~" * 50)
print(headlines_df.groupby("veracity")["judgment"].value_counts(normalize=True))
print("\n")

# Calculate the metrics
for condition, map in [
    ("unsure=true", UNSURE_TRUE_MAP),
    ("unsure=false", UNSURE_FALSE_MAP),
]:
    print(f"\nCondition: {condition}".upper())
    print("%" * 50, "\n")
    temp_df = headlines_df.copy()
    temp_df.judgment = temp_df.judgment.map(map)

    print("Counts")
    print("~" * 50)
    print(temp_df.groupby("veracity")["judgment"].value_counts())
    print("\n")

    print("Proportions")
    print("~" * 50)
    print(temp_df.groupby("veracity")["judgment"].value_counts(normalize=True))
    print("\n")

    print("Metrics")
    print("~" * 50)
    for verdict in [True, False]:

        print(f"Veracity: {verdict}")
        print("-" * 25)
        precision = df_precision(temp_df, verdict)
        recall = df_recall(temp_df, verdict)
        numerator = 2 * precision * recall
        denominator = precision + recall
        f1 = numerator / denominator if denominator != 0 else 0.0

        print(f"\t- Precision: {precision:.2f}")
        print(f"\t- Recall   : {recall:.2f}")
        print(f"\t- F1       : {f1:.2f}")
