"""
Purpose:
- Generate the main effects figure in the manuscript.

Inputs:
- None

Outputs:
- chatgpt-fact-checker/figures/main_effects.pdf
- chatgpt-fact-checker/figures/main_effects.png

Author: Matthew DeVerna
"""

import os
import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec

plt.rcParams.update({"font.size": 14})

ROOT_DIR = "figure_creation"
# Ensure we are in the data_cleaning directory for paths to work
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

# Local imports
from plot_utils import darken_hexcode

# Constants
FONT_SIZE = 14
FIGURES_DIR = "../../figures"
DATA_PATH = "../../results/discernment_df_main_groups_only_w_veracity.csv"

MEANS_DATA_FILE = "../../results/group_mean_ci.csv"

# Set up loading function
DATA_CLEANING_DIR = "../data_cleaning"
sys.path.insert(0, DATA_CLEANING_DIR)
from db import find_file

# Get file version and load
DATA_DIR = "../../data/cleaned_data/"
FNAME = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")

# Load data
mg_discern = pd.read_csv(MEANS_DATA_FILE)
all_data = pd.read_parquet(FNAME)

# Calculate the proportion of True and False (ground truth) headlines that
# ChatGPT thinks are true/false/unsure
question_data = all_data[~all_data.qualtrics_question_num.duplicated()]
question_data = question_data[
    ["qualtrics_question_num", "veracity", "ano_true_false_unsure"]
].reset_index(drop=True)
true_props = (
    question_data[question_data.veracity == True]
    .ano_true_false_unsure.value_counts(normalize=True)
    .to_dict()
)
false_props = (
    question_data[question_data.veracity == False]
    .ano_true_false_unsure.value_counts(normalize=True)
    .to_dict()
)


# Create dictionaries that are slightly easier to work with when plotting
belief_mask = mg_discern["Group"] == "Belief"
optional_mask = mg_discern["Condition"] == "Optional"

belief_group = mg_discern[belief_mask]
sharing_group = mg_discern[~belief_mask]

mg_belief_dict = {
    "Control": {True: {"mean": 0, "ci": 0}, False: {"mean": 0, "ci": 0}},
    "Forced": {True: {"mean": 0, "ci": 0}, False: {"mean": 0, "ci": 0}},
    "Optional": {True: {"mean": 0, "ci": 0}, False: {"mean": 0, "ci": 0}},
    "Human-FC": {True: {"mean": 0, "ci": 0}, False: {"mean": 0, "ci": 0}},
}

for v_opt in belief_group.veracity.unique():
    for con in belief_group.Condition.unique():
        # Add mean value
        mg_belief_dict[con][v_opt]["mean"] = belief_group.loc[
            (belief_group.Condition == con) & (belief_group.veracity == v_opt),
            "mean_said_yes",
        ].item()

        # Add ci distance
        mg_belief_dict[con][v_opt]["ci"] = belief_group.loc[
            (belief_group.Condition == con) & (belief_group.veracity == v_opt),
            "ci_said_yes",
        ].item()

mg_sharing_dict = {
    "Control": {True: {"mean": 0, "ci": 0}, False: {"mean": 0, "ci": 0}},
    "Forced": {True: {"mean": 0, "ci": 0}, False: {"mean": 0, "ci": 0}},
    "Optional": {True: {"mean": 0, "ci": 0}, False: {"mean": 0, "ci": 0}},
    "Human-FC": {True: {"mean": 0, "ci": 0}, False: {"mean": 0, "ci": 0}},
}

for v_opt in sharing_group.veracity.unique():
    for con in sharing_group.Condition.unique():
        # Add mean value
        mg_sharing_dict[con][v_opt]["mean"] = sharing_group.loc[
            (sharing_group.Condition == con) & (sharing_group.veracity == v_opt),
            "mean_said_yes",
        ].item()

        # Add ci distance
        mg_sharing_dict[con][v_opt]["ci"] = sharing_group.loc[
            (sharing_group.Condition == con) & (sharing_group.veracity == v_opt),
            "ci_said_yes",
        ].item()


fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[0.2, 1])

# Create subplots
ax_map = {
    1: fig.add_subplot(gs[0, :]),
    2: fig.add_subplot(gs[1, 0]),
    3: fig.add_subplot(gs[1, 1]),
}

# Set the y-axis tick formatter to show percentage values
fmt = "%.0f"  # This format string will display values as percentages without decimal places
xticks = mtick.PercentFormatter(xmax=1, decimals=0, symbol="%")


ax_map[1].spines["top"].set_visible(True)
ax_map[1].spines["bottom"].set_visible(False)
ax_map[1].spines["right"].set_visible(True)
ax_map[1].xaxis.set_major_formatter(xticks)
ax_map[1].xaxis.tick_top()

# Define the labels for the chart
bar_thickness = 0.8
anno_fontsize = 16

color_map = {
    "true": "#f0f0f0",
    "unsure": "#bdbdbd",
    "false": "#636363",
}

# For presentation
# color_map = {
#     'true' : "#FFEBB4",
#     'unsure' : "#FFBFA9",
#     'false' : "#FFACAC",
# }

# For vertical alignment of bar annotations (so annoying)
v_adjust_t = 0.065
v_adjust_f = 0.045

### True bar
# ----------------
ax_map[1].barh(
    #     "True",
    1,
    true_props["true"],
    height=bar_thickness,
    label="True",
    color=color_map["true"],
)
ax_map[1].annotate(
    f"{true_props['true']:.0%}",
    #     xy=((true_props['true']/2), "True"),
    xy=((true_props["true"] / 2), 1 - v_adjust_t),
    #     xytext=((true_props['true']/2), "True"),
    xytext=((true_props["true"] / 2), 1 - v_adjust_t),
    textcoords="offset points",
    ha="center",
    va="center",
    fontsize=anno_fontsize,
)
ax_map[1].barh(
    #     "True",
    1,
    true_props["unsure"],
    left=true_props["true"],
    height=bar_thickness,
    label="Unsure",
    color=color_map["unsure"],
)
ax_map[1].annotate(
    f"{true_props['unsure']:.0%}",
    #     xy=(true_props['true']+(true_props['unsure']/2), "True"),
    xy=(true_props["true"] + (true_props["unsure"] / 2), 1 - v_adjust_t),
    #     xytext=(true_props['true']+(true_props['unsure']/2), "True"),
    xytext=(true_props["true"] + (true_props["unsure"] / 2), 1 - v_adjust_t),
    textcoords="offset points",
    ha="center",
    va="center",
    fontsize=anno_fontsize,
)
ax_map[1].barh(
    #     "True",
    1,
    true_props["false"],
    left=true_props["true"] + true_props["unsure"],
    height=bar_thickness,
    label="False",
    color=color_map["false"],
)
ax_map[1].annotate(
    f"{true_props['false']:.0%}",
    #     xy=(true_props['true']+true_props['unsure']+(true_props['false']/2), "True"),
    xy=(
        true_props["true"] + true_props["unsure"] + (true_props["false"] / 2),
        1 - v_adjust_t,
    ),
    #     xytext=(true_props['true']+true_props['unsure']+(true_props['false']/2), "True"),
    xytext=(
        true_props["true"] + true_props["unsure"] + (true_props["false"] / 2),
        1 - v_adjust_t,
    ),
    textcoords="offset points",
    ha="center",
    va="center",
    fontsize=anno_fontsize,
    color="white",
)

### False bar
# ----------------
ax_map[1].barh(
    #     "False",
    0,
    false_props["unsure"],
    height=bar_thickness,
    color=color_map["unsure"],
)
ax_map[1].annotate(
    f"{false_props['unsure']:.0%}",
    #     xy=(false_props['unsure']/2, "False"),
    xy=(false_props["unsure"] / 2, 0 - v_adjust_f),
    #     xytext=(false_props['unsure']/2, "False"),
    xytext=(false_props["unsure"] / 2, 0 - v_adjust_f),
    textcoords="offset points",
    ha="center",
    va="center",
    fontsize=anno_fontsize,
)
ax_map[1].barh(
    #     "False",
    0,
    false_props["false"],
    left=false_props["unsure"],
    height=bar_thickness,
    color=color_map["false"],
)
ax_map[1].annotate(
    f"{false_props['false']:.0%}",
    #     xy=(false_props['unsure']+(false_props['false']/2), "False"),
    xy=(false_props["unsure"] + (false_props["false"] / 2), 0 - v_adjust_f),
    #     xytext=(false_props['unsure']+(false_props['false']/2), "False"),
    xytext=(false_props["unsure"] + (false_props["false"] / 2), 0 - v_adjust_f),
    textcoords="offset points",
    ha="center",
    va="center",
    fontsize=anno_fontsize,
    color="white",
)

ax_map[1].set_yticks([0, 1])
ax_map[1].set_xticks([])
ax_map[1].set_yticklabels(["False", "True"])

ax_map[1].set_ylabel("Veracity")

ax_map[1].set_xlim(0, 1)

ax_map[1].spines["top"].set_visible(False)
ax_map[1].spines["right"].set_visible(False)

legend = fig.legend(
    title="",
    ncol=3,
    loc="upper center",  # For tall
    bbox_to_anchor=(0.74, 0.94),
    labelspacing=0.1,  # for tall
    frameon=False,
    fontsize=14,
    title_fontsize=15,
)


################################################################################################
################################################################################################
################################################################################################

my_palette = ["#A6D0DD", "#FF6969"]
darker_palette = [
    darken_hexcode(my_palette[0], 0.2),
    darken_hexcode(my_palette[1], 0.2),
]


# Set the y-axis tick formatter to show percentage values
fmt = "%.0f"  # This format string will display values as percentages without decimal places
yticks = mtick.PercentFormatter(xmax=1, decimals=0, symbol="%")

# Handle axes formating
for x in range(2, 4):
    ax_map[x].grid(
        axis="y",
        which="major",
        linestyle="-",
        linewidth="0.5",
        color="lightgray",
        zorder=0,
    )
    ax_map[x].spines["top"].set_visible(False)
    ax_map[x].spines["right"].set_visible(False)
    ax_map[x].yaxis.set_major_formatter(yticks)


# Figure wide parameters
alpha = 0.95
offset = 0.2
base_x_vals = np.array([0, 0.7, 1.4, 2.1])
anno_buffer = 0.15
anno_width = 0.02
capsize = 0
y_reduce = 0.2

# Slices
groups = ["mg_belief", "mg_sharing"]
conditions = ["Control", "Forced", "Optional", "Human-FC"]
veracity_options = [True, False]

for plot_idx, group in enumerate(groups, start=2):
    ### Create bars
    # --------------------
    for veracity in veracity_options:
        ax_map[plot_idx].bar(
            x=base_x_vals if veracity else (base_x_vals + offset),
            height=[
                eval(f"{group}_dict['Control'][veracity]['mean']"),
                eval(f"{group}_dict['Forced'][veracity]['mean']"),
                eval(f"{group}_dict['Optional'][veracity]['mean']"),
                eval(f"{group}_dict['Human-FC'][veracity]['mean']"),
            ],
            yerr=[
                eval(f"{group}_dict['Control'][veracity]['ci']"),
                eval(f"{group}_dict['Forced'][veracity]['ci']"),
                eval(f"{group}_dict['Optional'][veracity]['ci']"),
                eval(f"{group}_dict['Human-FC'][veracity]['ci']"),
            ],
            capsize=capsize,
            ecolor="k",
            color=my_palette[0] if veracity else my_palette[1],
            edgecolor=darker_palette[0] if veracity else darker_palette[1],
            label=veracity if group == "mg_belief" else None,
            width=offset,
            zorder=3,
            alpha=alpha,
        )

    ax_map[plot_idx].set_xticks((base_x_vals + offset) - offset / 2)
    y_label = "Believed" if group == "mg_belief" else "Willing to share"
    ax_map[plot_idx].set_ylabel(y_label)

    ### Create annotations
    # --------------------
    for idx, cond in enumerate(conditions):
        ymin = eval(f"{group}_dict")[cond][False]["mean"]
        ymax = eval(f"{group}_dict")[cond][True]["mean"]
        vdist = ymax - ymin
        xmin = base_x_vals[idx] + offset + anno_buffer - anno_width
        xmax = base_x_vals[idx] + offset + anno_buffer

        # V-lines
        ax_map[plot_idx].vlines(
            x=base_x_vals[idx] + offset + anno_buffer,
            ymin=ymin,
            ymax=ymax,
            color="k",
            lw=1,
        )

        # H-lines (top)
        ax_map[plot_idx].hlines(y=ymax, xmin=xmin, xmax=xmax, color="k", lw=1)
        # H-lines (bottom)
        ax_map[plot_idx].hlines(
            y=ymin,
            xmin=xmin,
            xmax=xmax,
            color="k",
            lw=1,
        )
        # Add text distance (discernment)
        dist = vdist
        num_str = f"{dist:.0%}"
        ax_map[plot_idx].annotate(
            num_str,
            xy=(xmax, ymax),
            xytext=(xmax, ymax),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
        )

# Applies to both subplots because sharex/y=True when subplots declared
conditions = ["Control", "LLM\nForced", "LLM\nOptional", "Human\nFact Check"]
ax_map[2].set_xticklabels(conditions, fontsize=12)
ax_map[3].set_xticklabels(conditions, fontsize=12)
ax_map[2].set_ylim((0, 1))
ax_map[3].set_ylim((0, 1))

ax_map[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_map[3].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

### Set legend details
# --------------------

legend2 = fig.legend(
    handles=[
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=my_palette[0],
            edgecolor=darken_hexcode(my_palette[0]),
            label="True",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=my_palette[1],
            edgecolor=darken_hexcode(my_palette[1]),
            label="False",
        ),
    ],
    #     title = "ChatGPT\nfact-checks",
    frameon=False,
    bbox_to_anchor=(0.95, 0.7),
    fontsize=14,
)

# fig.tight_layout()

ax_map[1].text(0.82, 1.34, "LLM judgment", transform=ax_map[1].transAxes)


plt.subplots_adjust(hspace=0.2, wspace=0.38, right=0.95)

### Add the A and B (Must be after .tight_layout() or it messes with things)
# --------------------
ax_map[1].annotate("(b)", xy=(-0.125, 1.3), xycoords=ax_map[1].transAxes, fontsize=16)
ax_map[2].annotate("(c)", xy=(-0.3, 0.98), xycoords=ax_map[2].transAxes, fontsize=16)
ax_map[3].annotate("(d)", xy=(-0.3, 0.98), xycoords=ax_map[3].transAxes, fontsize=16)

fig.savefig(os.path.join(FIGURES_DIR, "main_effects.pdf"), transparent=True, dpi=1000)
fig.savefig(os.path.join(FIGURES_DIR, "main_effects.png"), transparent=True, dpi=1000)
fig.savefig(os.path.join(FIGURES_DIR, "main_effects.svg"), transparent=True, dpi=1000)
