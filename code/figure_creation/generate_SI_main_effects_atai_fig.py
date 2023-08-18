"""
Purpose:
- Generate the SI figure showing how attitudes toward AI is related to the
    main experiment effects.

Inputs:
- None

Outputs:
- chatgpt-fact-checker/figures/main_effects_SI_atai.pdf
- chatgpt-fact-checker/figures/main_effects_SI_atai.png

Author: Matthew DeVerna
"""
import os
import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

plt.rcParams.update({"font.size": 16})

ROOT_DIR = "figure_creation"
# Ensure we are in the data_cleaning directory for paths to work
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

# Set up loading function
DATA_CLEANING_DIR = "../data_cleaning"
sys.path.insert(0, DATA_CLEANING_DIR)
from db import find_file

# Constants
# ---------------------
### Paths ###
FIGURES_DIR = "../../figures"
DISCERNMENT_DATA_FILE = "../../results/discernment_df_main_groups_only_w_veracity.csv"
DATA_DIR = "../../data/cleaned_data/"
ALL_DATA_FILE = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")

### For plotting ###
# Data selectors
GROUPS = ["Belief", "Share"]
CONDITIONS = ["Control", "Forced", "Optional"]

# Colors to match the main figure
COLOR_MAP = {True: "#aad2df", False: "#ff7171"}

# These define the CENTER of the bins — does not affect regression.
#    Ref: https://seaborn.pydata.org/generated/seaborn.regplot.html
#        - See `x_bins` description for more info.
XBINS = np.arange(1, 7.5, 0.5)

# Convert proportions to percentages
YTICKS = mtick.PercentFormatter(xmax=1, decimals=0, symbol="%")

# Data
# ---------------------

# Load
all_data = pd.read_parquet(ALL_DATA_FILE)
df = pd.read_csv(DISCERNMENT_DATA_FILE)

# Add attitudes towards AI to the main df
atai = all_data.drop_duplicates(subset=["ResponseId"])[
    ["ResponseId", "Group", "Condition", "AI_att_mean"]
]
df_w_atai = df.merge(atai, on=["ResponseId", "Group", "Condition"])


### Generate figure
# ---------------------
fig, axs = plt.subplots(figsize=(14, 8), ncols=3, nrows=2)
for row_idx, group in enumerate(GROUPS):
    for col_idx, cond in enumerate(CONDITIONS):
        for veracity in [True, False]:
            selected_data = df_w_atai[
                (df_w_atai.Group == group)
                & (df_w_atai.Condition == cond)
                & (df_w_atai.veracity == veracity)
            ]

            loc_tuple = (row_idx, col_idx)
            sns.regplot(
                selected_data,
                x="AI_att_mean",
                y="prop_yes",
                x_estimator=np.mean,
                x_bins=XBINS,
                color=COLOR_MAP[veracity],
                label=str(veracity),
                n_boot=1_000,
                robust=True,
                ax=axs[loc_tuple],
            )

            # Title
            axs[loc_tuple].set_title(cond)

            # Y-axis label, ticks, and spines
            axs[loc_tuple].set_ylim((0, 1))
            axs[loc_tuple].yaxis.set_major_formatter(YTICKS)
            if col_idx != 0:
                axs[loc_tuple].set_ylabel(None)
                axs[loc_tuple].yaxis.set_ticklabels([])
                axs[loc_tuple].yaxis.set_tick_params(length=0)
                axs[loc_tuple].spines["left"].set_visible(False)
            else:
                name = "Believed" if group == "Belief" else "Willing to share"
                axs[loc_tuple].set_ylabel(name)

            # X-axis label and ticks
            axs[loc_tuple].set_xlabel("Attitude towards AI")
            axs[loc_tuple].xaxis.set_ticks(np.arange(1, 8, 1), np.arange(1, 8, 1))

# Handle spines
for ax in axs.flatten():
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True)

# Subplot spacing
plt.tight_layout()
plt.subplots_adjust(hspace=0.45)

# Create the legend
handle1 = plt.Line2D(
    [], [], marker="o", color=COLOR_MAP[True], markersize=8, label="True"
)
handle2 = plt.Line2D(
    [], [], marker="o", color=COLOR_MAP[False], markersize=8, label="False"
)
axs[(0, 1)].legend(handles=[handle1, handle2], ncol=2, frameon=True, loc="upper center")

# Add figure annotations
axs[(0, 0)].annotate(
    "(a)", xy=(-0.3, 0.98), xycoords=axs[(0, 0)].transAxes, fontsize=16
)
axs[(1, 0)].annotate(
    "(b)", xy=(-0.3, 0.98), xycoords=axs[(1, 0)].transAxes, fontsize=16
)

# Save figure
plt.savefig(os.path.join(FIGURES_DIR, "SI_main_effects_atai.pdf"), dpi=800)
plt.savefig(
    os.path.join(FIGURES_DIR, "SI_main_effects_atai.png"), transparent=True, dpi=800
)
