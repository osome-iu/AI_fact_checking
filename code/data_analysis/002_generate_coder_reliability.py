"""
Purpose: 
    Merged manual annotations and save output.
    Calculate and record intercoder reliability (Krippendorf's alpha).

    Run via: python 002_generate_coder_reliability.py > /path/to/output_dir/intercoder_reliability.txt

Inputs:
    None

Outputs:
    New files:
    - intercoder_reliability.txt (created by this script print messages)
    - {%Y-%m-%d}_annotations_merged.csv : the merged df of all coders responses (i.e., each coders
        response to each question is included)

Author: Matthew DeVerna
"""
import os
import datetime
import pprint

import pandas as pd

from krippendorff import alpha as kalpha

DATA_DIR = "../../data/manual_annotation/"
ROOT_DIR = "data_analysis"
# Ensure we are in the data_analysis directory for paths to work
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception("Must run this script from the `code/data_analysis/` directory!")

# Columns we want from the data
INCLUDE_COLS = [
    "filename",
    "headline",
    "cgpt_response",
    "true_false_unsure",
    # "mentions_cutoff_date", # Not included in this study!
    # "is_debunk",
    # "suggests_lateral_read",
    # "presents_media_literacy_tips",
    # "rebutts_science_denialism",
    # "mention_social_norms",
]

CODER1FNAME = "2023-04-10_manual_annotation_v0.csv"
CODER2FNAME = "2023-04-10_manual_annotation_v1.csv"
CODER3FNAME = "2023-04-10_manual_annotation_v2.csv"

# Load data
coder1 = pd.read_csv(os.path.join(DATA_DIR, CODER1FNAME), usecols=INCLUDE_COLS)
coder2 = pd.read_csv(os.path.join(DATA_DIR, CODER2FNAME), usecols=INCLUDE_COLS)
coder3 = pd.read_csv(os.path.join(DATA_DIR, CODER3FNAME), usecols=INCLUDE_COLS)

# Mark each coder and combine frames
coder1["coder"] = 1
coder2["coder"] = 2
coder3["coder"] = 3
all_coded = pd.concat([coder1, coder2, coder3])

# Extract the question number from the filename
all_coded["q_num"] = all_coded["filename"].apply(lambda x: int(x.split("_")[0]))

# Save the data
today = datetime.datetime.now().strftime("%Y-%m-%d")
all_coded.to_csv(os.path.join(DATA_DIR, f"{today}_annotations_merged.csv"), index=False)

# Extract only the columns that we coded
coded_cols = all_coded.columns[3:-2]

print("Create a dictionary to map the values to numeric values...")
coded_value_map = dict()
for col in coded_cols:
    unique_vals = all_coded[col].unique()
    temp_map = dict()
    for idx, val in enumerate(unique_vals):
        temp_map[val] = idx
    coded_value_map[col] = temp_map
pprint.pprint(coded_value_map, indent=4)

print("Recoding the values for each column...")
for col in coded_cols:
    all_coded[col] = all_coded[col].map(coded_value_map[col])

print("Calculating Krippendorff's alpha...")
for col in coded_value_map:
    kripp_alpha = kalpha(
        all_coded.pivot(index="coder", columns=["q_num"], values=col),
        level_of_measurement="nominal",
    )
    print(f"Dimension: {col}")
    print(f"\t- Krippendorff Alpha: {kripp_alpha}")

print("See codebook for details on each dimension.")

print("Script complete.")
