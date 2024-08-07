"""
Purpose:
- Select the majority judgments for the different manual annotations for the
    results of the original prompt with the API and repeated in the web browser.

Inputs:
- None. Files and paths are set as constants.

Outputs:
- A .csv file where each line represents the majority label for a single headline, 
    for the original prompt across three coders.
"""

import os

import pandas as pd

from collections import Counter, defaultdict

API_DIR = "../../data/manual_annotation"
WEB_DIR = os.path.join(API_DIR, "web_results")

API_CODER1FNAME = "completed-2024-03-11_13 28 19__manual_annotation_v0.csv"
API_CODER2FNAME = "completed-2024-03-11_13 28 19__manual_annotation_v1.csv"
API_CODER3FNAME = "completed-2024-03-11_13 28 19__manual_annotation_v2.csv"

# Same filenames, saved in a different directory
WEB_CODER1FNAME = "completed-2024-03-11_13_28_19__manual_annotation_v0.csv"
WEB_CODER2FNAME = "completed-2024-03-11_13_28_19__manual_annotation_v1.csv"
WEB_CODER3FNAME = "completed-2024-03-11_13_28_19__manual_annotation_v2.csv"

HEADLINES_FILE = "../../../data/stimuli/all_experimental_stimuli_meta_data.csv"

# Ensure we are in the scripts directory for paths to work
if os.getcwd() != os.path.dirname(os.path.realpath(__file__)):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Load the headlines data
headlines_df = pd.read_csv(HEADLINES_FILE)
headlines_df.text = headlines_df.text.str.replace("'", "")  # For matching later

# -------------------------------- #
# Handle the API results
# -------------------------------- #

# Load the annotations for the API results
api_coder1 = pd.read_csv(os.path.join(API_DIR, API_CODER1FNAME))
api_coder2 = pd.read_csv(os.path.join(API_DIR, API_CODER2FNAME))
api_coder3 = pd.read_csv(os.path.join(API_DIR, API_CODER3FNAME))

# Mark each coder and combine frames
# -------------------------------- #
api_coder1["coder"] = 1
api_coder2["coder"] = 2
api_coder3["coder"] = 3
api_all_coded = pd.concat([api_coder1, api_coder2, api_coder3])

# Get majority label for each headline
count_dict = defaultdict(dict)
options = ["true", "false", "unsure"]
for headline, data in api_all_coded.groupby("headline"):
    counts = Counter(data["true_false_unsure"])

    # Results look like this: [('label1', count), ('label2', count), ...]
    most_common_list = counts.most_common()

    # Three items in the list means each person labeled the item differently
    if len(most_common_list) == 3:
        raise Exception("We have a split majority!")

    count_dict[headline] = counts.most_common()[0][0]


# Convert to dataframe
api_judgments_df = pd.DataFrame.from_records(
    [
        {"headline": headline, "judgment": judgment}
        for headline, judgment in count_dict.items()
    ]
)
# To match the headlines in the headlines_df
api_judgments_df.headline = api_judgments_df.headline.str.replace("'", "")

# Add other helpful columns
merged_df = pd.merge(
    headlines_df[["qualtrics_question_num", "text", "ideo_lean", "veracity"]],
    api_judgments_df,
    left_on="text",
    right_on="headline",
)
merged_df = merged_df.drop(columns=["text"])  # Duplicate column of different name
merged_df.to_csv(os.path.join(API_DIR, "majority_api_judgments.csv"), index=False)

# -------------------------------- #
# Handle the web results
# -------------------------------- #

# Load the annotations for the API results
web_coder1 = pd.read_csv(os.path.join(WEB_DIR, WEB_CODER1FNAME))
web_coder2 = pd.read_csv(os.path.join(WEB_DIR, WEB_CODER2FNAME))
web_coder3 = pd.read_csv(os.path.join(WEB_DIR, WEB_CODER3FNAME))

# Mark each coder and combine frames
# -------------------------------- #
web_coder1["coder"] = 1
web_coder2["coder"] = 2
web_coder3["coder"] = 3
web_all_coded = pd.concat(
    [
        web_coder1,
        web_coder2,
        web_coder3,
    ]
)


# Get majority label for each headline
count_dict = defaultdict(dict)
options = ["true", "false", "unsure"]
for headline, data in web_all_coded.groupby("headline"):
    counts = Counter(data["true_false_unsure"])

    # Results look like this: [('label1', count), ('label2', count), ...]
    most_common_list = counts.most_common()

    # Three items in the list means each person labeled the item differently
    if len(most_common_list) == 3:
        raise Exception("We have a split majority!")

    count_dict[headline] = counts.most_common()[0][0]


# Convert to dataframe
web_judgments_df = pd.DataFrame.from_records(
    [
        {"headline": headline, "judgment": judgment}
        for headline, judgment in count_dict.items()
    ]
)
# To match the headlines in the headlines_df
web_judgments_df.headline = web_judgments_df.headline.str.replace("'", "")

# Add other helpful columns
merged_df = pd.merge(
    headlines_df[["qualtrics_question_num", "text", "ideo_lean", "veracity"]],
    web_judgments_df,
    left_on="text",
    right_on="headline",
)
merged_df = merged_df.drop(columns=["text"])  # Duplicate column of different name
merged_df.to_csv(os.path.join(WEB_DIR, "majority_web_judgments.csv"), index=False)
