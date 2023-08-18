#!/bin/bash

# Purpose:
#   Run the entire project pipeline. See individual scripts for details.
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

# Throw error if not in correct directory
EXPECTED_DIR="code"
CURRENT_DIR=$(basename "$PWD")

if [ "$CURRENT_DIR" != "$EXPECTED_DIR" ]; then
  echo "Error: This script must be run from the $EXPECTED_DIR directory."
  exit 1  # Exit with an error code of 1
fi


# Analyze data
cd data_analysis
bash 000_run_data_analysis_pipeline.sh

echo "##########################################"
echo "--------- Data analysis complete ---------"
echo "##########################################"
echo ""
echo ""

# Generate figures
cd ../figure_creation
bash 000_generate_all_figures.sh

echo "######################################"
echo "----- Figure generation complete -----"
echo "######################################"