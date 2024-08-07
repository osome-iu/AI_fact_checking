#!/bin/bash

# Purpose:
#   Run the prompt engineering analysis pipeline. See each individual script to understand what they do.
#
# Note:
#   This script does not run the data collection portion, as this may be temporally sensitive
#   and likely to break as environment variables need to be set up correctly. If you want to run
#   it, please make sure you know what you're doing and are willing to spend money for OpenAI API
#   costs. You can always reach out to Matt for more information.
#
# Inputs:
#   None
#
# Output:
#   See individual scripts for information about their respective outputs.
#
# How to call:
#   ```
#   bash run_pipeline.sh
#   ```
#
# Author: Matthew DeVerna

# Throw error if not in project root
EXPECTED_DIR="code"
CURRENT_DIR=$(basename "$PWD")

if [ "$CURRENT_DIR" != "$EXPECTED_DIR" ]; then
  echo "Error: This script must be run from the $EXPECTED_DIR directory."
  exit 1  # Exit with an error code of 1
fi

# # Data collection
# cd data_collection
# python3 collect_fact_checks.py

# echo "##########################################"
# echo "--------- Data collection complete ---------"
# echo "##########################################"
# echo ""
# echo ""

# Data cleaning
# Note: the 'cd' line must be updated to the one directly below to run the data collection portion)
cd data_cleaning
# cd ../data_cleaning
python3 001_select_majority_original_api_coding.py
python3 002_clean_binary_api_results.py

echo "##########################################"
echo "--------- Data cleaning complete ---------"
echo "##########################################"
echo ""
echo ""

# Data analysis
cd ../data_analysis
python3 001_calculate_binary_rationale_metrics.py > ../../results/api_judgments_binary_rationale_metrics.txt
python3 002_calculate_original_api_web_metrics.py > ../../results/original_api_web_metrics.txt

echo "##########################################"
echo "--------- Data analysis complete ---------"
echo "##########################################"
echo ""
echo "See the `prompt_engineering/results` directory for results."
