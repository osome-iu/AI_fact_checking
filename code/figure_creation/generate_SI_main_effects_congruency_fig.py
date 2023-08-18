"""
Purpose:
- Generate Figure showing congruency relationship with belief and sharing behavior
    for the main effects.

Inputs:
- None

Outputs:
- chatgpt-fact-checker/figures/SI_main_effects_congruency.pdf
- chatgpt-fact-checker/figures/SI_main_effects_congruency.png

Author: Matthew DeVerna
"""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

plt.rcParams.update({"font.size": 12})

# Constants
ROOT_DIR = "figure_creation"
DATA_DIR = "../../data/cleaned_data"
DATA_CLEANING_DIR = "../data_cleaning"
DATA_ANALYSIS_DIR = "../data_analysis"
FIGURES_DIR = "../../figures"

# Ensure we are in the data_cleaning directory for paths to work
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

# Get the latest long form data
sys.path.insert(0, DATA_CLEANING_DIR)
from db import find_file

ALL_DATA_FILE = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")

sys.path.insert(0, DATA_ANALYSIS_DIR)
from boot import bootstrap_ci


# Convert proportions to percentages
YTICKS = mtick.PercentFormatter(xmax=1, decimals=0, symbol="%")

# Colors to match the main figure
COLOR_MAP = {True: "#aad2df", False: "#ff7171"}

# Markers for each type of headline
MARKER_MAP = {"congruent": "o", "incongruent": "x"}

# Map for the x-axis locations
x_map = {
    "congruent": 1,
    "incongruent": 2,
}


### Wrangle the data ###
# ----------------------
all_data = pd.read_parquet(ALL_DATA_FILE)

# Calculate the number of yes responses per participant (ResponseId)
yes_by_participant = (
    all_data.groupby(["Group", "Condition", "ResponseId", "veracity", "congruency"])[
        "exp_response"
    ]
    .sum()
    .to_frame("num_yes")
    .reset_index()
)

# There are 20 true and 20 false headlines, half are pro-Rep the other half pro-Dem,
# so we calcualte the proportion of headlines they said yes to by dividing by 10
yes_by_participant["prop_yes"] = yes_by_participant["num_yes"] / 10

# Calculate the mean proportion of yes responses and the 95% CI
mean_df = (
    yes_by_participant.groupby(["Group", "Condition", "veracity", "congruency"])[
        "prop_yes"
    ]
    .mean()
    .to_frame("mean_prop")
    .reset_index()
)

cis = (
    yes_by_participant.groupby(["Group", "Condition", "veracity", "congruency"])[
        "prop_yes"
    ]
    .apply(bootstrap_ci)
    .to_frame("ci")
    .reset_index()
)
mean_df = mean_df.merge(cis, on=["Group", "Condition", "veracity", "congruency"])


### Make the damn figure! ###
# ---------------------------

fig, axes = plt.subplots(figsize=(8, 6), nrows=2, ncols=3)

for row_idx, group in enumerate(["Belief", "Share"]):
    for col_idx, cond in enumerate(["Control", "Forced", "Optional"]):
        for veracity in [True, False]:
            for congru in ["congruent", "incongruent"]:
                selected_df = mean_df[
                    (mean_df.Group == group)
                    & (mean_df.Condition == cond)
                    & (mean_df.veracity == veracity)
                    & (mean_df.congruency == congru)
                ]

                # Snag the values we need
                x_buff = 0.02 if veracity else -0.02

                loc = (row_idx, col_idx)
                x = x_map[congru] + x_buff
                y = selected_df["mean_prop"]
                y_low = selected_df["mean_prop"] - selected_df["ci"]
                y_high = selected_df["mean_prop"] + selected_df["ci"]

                # Plot the points
                axes[loc].scatter(
                    x=x,
                    y=y,
                    color=COLOR_MAP[veracity],
                    marker=MARKER_MAP[congru],
                    zorder=3,
                    label=f"{veracity}" if congru == "congruent" else None,
                )

                # Plot the error bars (95% CI)
                axes[loc].plot(
                    [x, x], [y_low, y_high], color=COLOR_MAP[veracity], zorder=3
                )
                axes[loc].grid(True, axis="y", zorder=0)

                # Make it pretty!
                axes[loc].set_title(f"{cond}")
                axes[loc].yaxis.set_major_formatter(YTICKS)

                axes[loc].set_ylim((0, 1))

                axes[loc].set_xticks(sorted(x_map.values()))
                axes[loc].set_xticklabels(["Congruent", "Incongruent"])
                axes[loc].set_xlim(0.7, 2.3)

                axes[loc].spines["top"].set_visible(False)
                axes[loc].spines["right"].set_visible(False)

                if col_idx != 0:
                    axes[loc].yaxis.set_ticklabels([])
                    axes[loc].yaxis.set_tick_params(length=0)
                if loc == (0, 0):
                    axes[loc].set_ylabel("Believed")
                if loc == (1, 0):
                    axes[loc].set_ylabel("Willing to share")

axes[(0, 2)].legend()

plt.tight_layout()

# Add figure annotations
axes[(0, 0)].annotate(
    "(a)", xy=(-0.5, 0.98), xycoords=axes[(0, 0)].transAxes, fontsize=16
)
axes[(1, 0)].annotate(
    "(b)", xy=(-0.5, 0.98), xycoords=axes[(1, 0)].transAxes, fontsize=16
)

fig.savefig(
    os.path.join(FIGURES_DIR, "SI_main_effects_congruency.pdf"),
    dpi=800,
)
fig.savefig(
    os.path.join(FIGURES_DIR, "SI_main_effects_congruency.png"),
    dpi=800,
    transparent=True,
)
