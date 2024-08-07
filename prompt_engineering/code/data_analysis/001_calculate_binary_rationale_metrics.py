"""
Purpose:
- Calculate precision, recall, and F1 for the the 'binary' and 'rationale' prompts.
- Print counts and proportions of each headline type

Inputs:
- None. Files and paths are set as constants.

Outputs:
- None. Results are printed to stdout.
- Generate a record of the results by running:
    - python 001_calculate_binary_rationale_metrics.py > output.txt
"""

import pandas as pd

from metrics import *

CLEAN_JUDGMENTS_FP = "../../results/clean_api_judgments_binary_and_rationale.csv"

# Load and split the file
df = pd.read_csv(CLEAN_JUDGMENTS_FP)
binary_df = df[df["prompt_type"] == "prompt_json_binary"].copy()
rationale_df = df[df["prompt_type"] == "prompt_json_rationale"].copy()

for name, df in [("binary", binary_df), ("rationale", rationale_df)]:
    print(f"Prompt Type: {name}".upper())
    print("-" * 75, "\n")

    print("Counts")
    print("~" * 50)
    print(df.groupby("veracity")["judgment"].value_counts())
    print("\n")

    print("Proportions")
    print("~" * 50)
    print(df.groupby("veracity")["judgment"].value_counts(normalize=True))
    print("\n")

    print("Metrics")
    print("~" * 50)
    for verdict in [True, False]:
        print(f"Veracity: {verdict}")
        print("-" * 25)
        precision = df_precision(df, verdict)
        recall = df_recall(df, verdict)
        f1 = (2 * precision * recall) / (precision + recall)
        accuracy = df_accuracy(df)
        false_negative_rate = df_false_negative_rate(df, verdict)
        false_positive_rate = df_false_positive_rate(df, verdict)

        print(f"\t- Precision: {precision:.2f}")
        print(f"\t- Recall   : {recall:.2f}")
        print(f"\t- F1       : {f1:.2f}")
        print(f"\t- Accuracy : {accuracy:.2f}")
        print(f"\t- FN Rate  : {false_negative_rate:.2f}")
        print(f"\t- FP Rate  : {false_positive_rate:.2f}")

    print("#" * 100, "\n\n")
