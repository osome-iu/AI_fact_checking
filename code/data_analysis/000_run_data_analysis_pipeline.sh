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

python 001_generate_quota_checks.py > ../../results/quota_checks.txt
python 002_generate_coder_reliability.py > ../../results/intercoder_reliability.txt
python 003_generate_discernment_dfs.py
python 004_generate_group_differences_main_groups_only.py > ../../results/group_differences.txt
python 005_generate_group_mean_ci.py
python 006_generate_optional_cond_mean_ci.py
python 007_generate_results_by_five_headline_types.py
python 008_generate_five_way_comparison_stats.py > ../../results/five_way_comparison_stats.txt
python 009_generate_optional_stats.py > ../../results/optional_comparisons_stats.txt
python 010_generate_SI_5way_congruency.py
python 011_generate_accuracy_metrics.py >  ../../results/accuracy_metrics.txt
python 012_generate_opt_in_counts_proportions.py > ../../results/opt_in_counts_proportions_stats.txt
python 013_generate_optional_cond_mean_ci_by_annotation.py
python 014_generate_optional_stats_by_annotation.py > ../../results/optional_comparisons_stats_by_annotation.txt