#!/bin/bash

# Purpose:
#   Run the entire data analysis pipeline and generate all results.
#   Each individual script generates different files in the `results` directory.
#   See individual scripts for details.
#
# Inputs:
#   None
#
#
# Output:
#   See individual scripts for information about their respective outputs.
#
# How to call:
#   ```
#   bash 000_run_data_cleaning_pipeline.sh
#   ```
#
# Author: Matthew DeVerna

# Throw error if not in project root
EXPECTED_DIR="data_analysis"
CURRENT_DIR=$(basename "$PWD")

if [ "$CURRENT_DIR" != "$EXPECTED_DIR" ]; then
  echo "Error: This script must be run from the $EXPECTED_DIR directory."
  exit 1  # Exit with an error code of 1
fi

echo "Running data analysis pipeline..."

python 001_generate_quota_checks.py > ../../results/quota_checks.txt ; echo "001_generate_quota_checks.py completed."
python 002_generate_coder_reliability.py > ../../results/intercoder_reliability.txt ; echo "002_generate_coder_reliability.py completed."
python 003_generate_discernment_dfs.py ; echo "003_generate_discernment_dfs.py completed."
python 004_generate_group_differences_main_groups_only.py > ../../results/group_differences.txt ; echo "004_generate_group_differences_main_groups_only.py completed."
python 005_generate_group_mean_ci.py ; echo "005_generate_group_mean_ci.py completed."
python 006_generate_optional_cond_mean_ci.py ; echo "006_generate_optional_cond_mean_ci.py completed."
python 007_generate_results_by_five_headline_types.py ; echo "007_generate_results_by_five_headline_types.py completed."
python 008_generate_five_way_comparison_stats.py > ../../results/five_way_comparison_stats.txt ; echo "008_generate_five_way_comparison_stats.py completed."
python 009_generate_optional_stats.py > ../../results/optional_comparisons_stats.txt ; echo "009_generate_optional_stats.py completed."
python 010_generate_SI_5way_congruency.py ; echo "010_generate_SI_5way_congruency.py completed."