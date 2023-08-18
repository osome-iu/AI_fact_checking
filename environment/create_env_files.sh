#!/bin/bash

# Create different yaml files in all the ways the conda
#   allows, since it can be finicky depending on the system.
# Our hope is that giving you all the versions make it easier to trouble shoot
#   should you run into any issues.
# 
# Note that the environment was activated prior to running this script!
# 
# Author: Matthew DeVerna

conda env export --from-history > env_from_history_cgpt_intervention.yml
conda env export > env_cgpt_intervention.yml
conda list --explicit > env_explicit_cgpt_intervention.txt

echo ""
echo "Environment files created."