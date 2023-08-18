#!/bin/bash

# This script that was utilized to generate the python virtual environment.
# 
# Note: conda (V 23.3.1) was installed already
# 
# Author: Matthew DeVerna

# Install dependencies that are not built into Python with conda
conda install -c conda-forge numpy
conda install -c conda-forge scipy
conda install -c conda-forge statsmodels
conda install -c conda-forge matplotlib
conda install -c conda-forge seaborn
conda install -c conda-forge pip
pip install krippendorff
pip install pyarrow
conda install -c conda-forge 'pandas=1.4.4'

echo ""
echo "Environment creation completed."