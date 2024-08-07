"""
Purpose:
- Clean the results of the two binary prompts tested with the API.

Inputs:
- None. Files and paths are set as constants.

Outputs:
- A .csv file where each row represents one judgment. Columns are:
    - 'prompt_type' (str): The type of prompt used. Options: ['binary', 'rationale']
    - 'judgment' (bool) : The model's judgment. Options: [True, False]
    - 'veracity' (bool) : The veracity of the judgment. Options: [True, False]
    - 'rationale' (str, None) : The rationale for the judgment provided by the model. Only
        provided for the 'rationale' prompt. Otherwise, None
    - 'time_created' (int): The timestamp of when the query was called
    - 'model' (str): The name of the model used
"""

import json

import os

import pandas as pd

# Ensure we are in the same directory as the script for paths to work
if os.path.join(os.path.dirname(os.path.realpath(__file__))) != os.getcwd():
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__))))

INTERMEDIATE_DIR = "../../data/intermediate_files"
RESULTS_DIR = "../../results"

RATIONALE_FP = os.path.join(INTERMEDIATE_DIR, "fact_checks_prompt_json_rationale.jsonl")
BINARY_FP = os.path.join(INTERMEDIATE_DIR, "fact_checks_prompt_json_binary.jsonl")


def return_clean_record(record, is_rationale=False):
    """
    Return a clean dictionary record with only what we want.

    Parameters:
    -----------
    - record (dict): A nested dictionary with keys:
        - 'prompt_type', 'response_text', 'veracity', 'time_created', 'model'

    Returns:
    -----------
    - clean_record (dict): A dictionary with keys:
        - 'prompt_type', 'judgment', 'veracity', 'rationale', 'time_created', 'model'
    """
    v_map = {"True": True, "False": False}
    return {
        "prompt_type": record["prompt_type"],
        "judgment": v_map[record["response_text"]["judgment"]],
        "veracity": record["veracity"],
        "rationale": record["response_text"]["rationale"] if is_rationale else None,
        "time_created": record["time_created"],
        "model": record["model"],
    }


if __name__ == "__main__":

    # Load and clean the API results
    with open(RATIONALE_FP, "r") as f:
        rationale_records = [
            return_clean_record(json.loads(line), is_rationale=True) for line in f
        ]

    with open(BINARY_FP, "r") as f:
        binary_records = [return_clean_record(json.loads(line)) for line in f]

    # Convert to dataframes
    rationale_df = pd.DataFrame.from_records(rationale_records)
    binary_df = pd.DataFrame.from_records(binary_records)

    # Combined the two dataframes
    combined_df = pd.concat([rationale_df, binary_df])

    # Save as csv
    combined_df.to_csv(
        os.path.join(RESULTS_DIR, "clean_api_judgments_binary_and_rationale.csv"),
        index=False,
    )
