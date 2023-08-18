"""
Purpose: Create three different versions of the headlines_text_w_responses.csv
    document for manual annotation.

Input: The .csv file containing the headlines and the chatgpt responses.
    - File: `data/headlines/headlines_text_w_responses.csv`

Output: Shuffled versions of the filename, headline, cgpt_response saved in csv format.

Author: Matthew DeVerna
"""
import datetime
import os
import random
import sys

import pandas as pd

from copy import deepcopy

REPO_ROOT = "chatgpt-fact-checker"
DATA_FILE_PATH = "./data/stimuli/headlines_text_w_responses.csv"
OUTPUT_DIR = "./data/manual_annotation"
NUM_VERSIONS_TO_CREATE = 3

# Ensure we only run this from the repository root directory
if os.path.basename(os.getcwd()) != REPO_ROOT:
    sys.exit(
        "ALL SCRIPTS MUST BE RUN FROM THE REPO ROOT!!\n"
        f"\tCurrent directory  : {os.getcwd()}\n"
        f"\tRepo root directory: {REPO_ROOT}\n"
    )

headlines_df = pd.read_csv(DATA_FILE_PATH)

items_records_list = []
for item in headlines_df.itertuples():
    items_records_list.append(
        {
            "filename": item.filename,
            "headline": item.text,
            "cgpt_response": item.cgpt_response,
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
