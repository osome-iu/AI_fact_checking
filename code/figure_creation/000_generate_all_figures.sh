#!/bin/bash

# Purpose:
#   Run the entire figure creation pipeline and generate all figures.
#   Each individual script generates different files in the `figures` directory.
#   See individual scripts for details.
#
# Inputs:
#   None
#
# Output:
#   See individual scripts for information about their respective outputs.
#
# How to call:
#   ```
#   bash 000_generate_all_figures.sh
#   ```
#
# Author: Matthew DeVerna

# Throw error if not in project root
EXPECTED_DIR="figure_creation"
CURRENT_DIR=$(basename "$PWD")

if [ "$CURRENT_DIR" != "$EXPECTED_DIR" ]; then
  echo "Error: This script must be run from the $EXPECTED_DIR directory."
  exit 1  # Exit with an error code of 1
fi

echo "Generating figures..."

# Generate main text figures
python generate_main_effects_fig.py; echo "main effects figure generated"
python generate_five_way_fig.py; echo "five-way figure generated"
python generate_opt-in_vs_opt-out_fig_by_annotation.py; echo "optional figure generated"

# Generate SI figures
python generate_SI_main_effects_atai_fig.py; echo "SI: main effects (ATAI) figure generated"
python generate_SI_main_effects_congruency_fig.py; echo "SI: main effects (congruency) figure generated"
python generate_SI_5way_atai_fig.py; echo "SI: fiveway (ATAI) figure generated"
python generate_SI_5way_congruency_figs.py; echo "SI: fiveway (congruency) figure generated"
python generate_SI_opt_in_distributions.py; echo "SI: opt-in distributions figures generated"