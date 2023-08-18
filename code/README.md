# `code/`

Project scripts are saved in this directory.

### Directories
- `data_analysis/`: scripts for analyzing data.
- `data_cleaning/`: includes some convenience functions/objects imported by other scripts.
- `figure_creation/`: scripts for creating figures.
- `r_code/`: R code for the supplementary analysis.

### Files
- `manual_annotation_randomizer.py`: randomly creates different versions of the manual annotation document. The randomization puts the ChatGPT responses in different orders.
    - Headline stimuli were chunked into groups of True/False pro-Dem/pro-Rep, etc in the original file. To make sure that our coders did not know anything about the headline in the coding file, we shuffled the order in which they were reviewed.
- `run_pipeline.sh`: `bash`` script to run the entire project pipeline. All scripts in `data_analysis/` are run and then all scripts in `figure_creation/`