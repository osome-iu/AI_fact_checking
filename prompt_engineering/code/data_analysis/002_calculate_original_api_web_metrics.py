"""
Purpose:
- Calculate precision, recall, and F1 for the the 'original' prompt.
    - 'web' results are based on getting responses via chat.openai.com (the second time around)
    - 'api' results are based on getting responses via the OpenAI API
- Print counts and proportions of each headline type

Inputs:
- None. Files and paths are set as constants.

Outputs:
- None. Results are printed to stdout.
- Generate a record of the results by running:
    - python 002_calculate_original_api_web_metrics.py > output.txt
"""

import os

import pandas as pd

from metrics import *

API_DIR = "../../data/manual_annotation"
WEB_DIR = os.path.join(API_DIR, "web_results")

UNSURE_TRUE_MAP = {"true": True, "false": False, "unsure": True}
UNSURE_FALSE_MAP = {"true": True, "false": False, "unsure": False}
DEFAULT_MAP = {"true": True, "false": False}

# Ensure we are in the scripts directory for paths to work
if os.getcwd() != os.path.dirname(os.path.realpath(__file__)):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

api_df = pd.read_csv(os.path.join(API_DIR, "majority_api_judgments.csv"))
web_df = pd.read_csv(os.path.join(WEB_DIR, "majority_web_judgments.csv"))


for name, df in [("api", api_df), ("web", web_df)]:
    print(f"Method: {name}".upper())
    print("-" * 75, "\n")

    print("Raw Counts")
    print("~" * 50)
    print(df.groupby("veracity")["judgment"].value_counts())
    print("\n")

    print("Raw Proportions")
    print("~" * 50)
    print(df.groupby("veracity")["judgment"].value_counts(normalize=True))
    print("\n")

    for condition, map in [
        ("ignore unsure", DEFAULT_MAP),
        ("unsure=true", UNSURE_TRUE_MAP),
        ("unsure=false", UNSURE_FALSE_MAP),
    ]:
        print(f"\nCondition: {condition}".upper())
        print("%" * 50, "\n")
        if condition == "ignore unsure":
            temp_df = df.query("judgment != 'unsure'").copy()
        else:
            temp_df = df.copy()
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
            accuracy = df_accuracy(temp_df)
            false_negative_rate = df_false_negative_rate(temp_df, verdict)
            false_positive_rate = df_false_positive_rate(temp_df, verdict)

            print(f"\t- Precision: {precision:.2f}")
            print(f"\t- Recall   : {recall:.2f}")
            print(f"\t- F1       : {f1:.2f}")
            print(f"\t- Accuracy : {accuracy:.2f}")
            print(f"\t- FN Rate  : {false_negative_rate:.2f}")
            print(f"\t- FP Rate  : {false_positive_rate:.2f}")

    print("#" * 100, "\n\n")
