"""
Purpose: Create three different versions of the fact_checks_prompt_original.jsonl
    document for manual annotation.

Input: The .jsonl file containing the chatgpt responses for the replicated original prompt.
    - File: `prompt_engineering/data/intermediate_files/fact_checks_prompt_original.jsonl`

Output: Shuffled versions of the headline and cgpt_response saved in csv format.

Author: Matthew DeVerna
"""

import datetime
import json
import os
import random

import pandas as pd

from copy import deepcopy

DATA_DIR = "../data/intermediate_files"
OUTPUT_DIR = "../data/manual_annotation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_VERSIONS_TO_CREATE = 3

PROMPT_PREFIX = "I saw something today that claimed "
PROMPT_SUFFIX = ". Do you think that this is likely to be true?"

# Make sure we are in the proper directory for the relative output dirs/files
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != CURR_DIR:
    os.chdir(CURR_DIR)

# Load data, convert to dataframe
fp_original = os.path.join(DATA_DIR, "fact_checks_prompt_original.jsonl")
with open(fp_original, "r") as f:
    original_results = [json.loads(line.rstrip()) for line in f]
original_df = pd.DataFrame.from_records(original_results)

# Extract the text from the user_content
items_records_list = []
for idx, row in original_df.iterrows():
    items_records_list.append(
        {
            "headline": row["user_content"]
            .replace(PROMPT_PREFIX, "")
            .replace(PROMPT_SUFFIX, ""),
            "cgpt_response": row["response_text"],
        }
    )

for i, item in enumerate(range(NUM_VERSIONS_TO_CREATE)):
    records_copy = deepcopy(items_records_list)

    # Set the randomizer seed for reproducibility and shuffle the list
    random.seed(i)
    random.shuffle(records_copy)

    # Save the list as a csv
    temp_df = pd.DataFrame.from_records(records_copy)
    now = datetime.datetime.now().strftime("%F_%X")
    file_out_path = os.path.join(OUTPUT_DIR, f"{now}__manual_annotation_v{i}.csv")
    temp_df.to_csv(file_out_path, index=False)

print("--- Script complete ---")
