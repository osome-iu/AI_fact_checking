"""
Purpose:
- Generate Figure 1 in the manuscript.

Inputs:
- None

Outputs:
- chatgpt-fact-checker/figures/five_way.pdf
- chatgpt-fact-checker/figures/five_way.png

Author: Matthew DeVerna
"""
import os

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from collections import defaultdict
from matplotlib.legend_handler import HandlerTuple

plt.rcParams.update({"font.size": 16})

ROOT_DIR = "figure_creation"
# Ensure we are in the data_cleaning directory for paths to work
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

# Local imports
from plot_utils import darken_hexcode

# Constants
FONT_SIZE = 16
FIGURES_DIR = "../../figures"

# Load data and remove Optional condition
MEANS_DATA_FILE = "../../results/five_headline_type_mean_ci.csv"
mean_said_yes_by_group = pd.read_csv(MEANS_DATA_FILE, dtype={"ground_truth": str})
mean_said_yes_by_group = mean_said_yes_by_group[
    mean_said_yes_by_group.Condition != "Optional"
]

# Split by group and sort
belief_only = mean_said_yes_by_group[mean_said_yes_by_group.Group == "Belief"].copy()
share_only = mean_said_yes_by_group[mean_said_yes_by_group.Group == "Share"].copy()
belief_only.sort_values(by=["Condition", "v_split_by_cgpt"], inplace=True)
share_only.sort_values(by=["Condition", "v_split_by_cgpt"], inplace=True)

# Create dictionaries that are easier to work with
share_dict = defaultdict(dict)
for idx, r_data in share_only.iterrows():
    share_dict[r_data.Condition].update(
        {
            (r_data.ground_truth, r_data.cGPT_thinks): {
                "mean": r_data.mean_said_yes,
                "ci": r_data.ci_said_yes,
            }
        }
    )

belief_dict = defaultdict(dict)
for idx, r_data in belief_only.iterrows():
    belief_dict[r_data.Condition].update(
        {
            (r_data.ground_truth, r_data.cGPT_thinks): {
                "mean": r_data.mean_said_yes,
                "ci": r_data.ci_said_yes,
            }
        }
    )


mpl.rcParams["font.family"] = "Arial"
plt.rcParams.update(
    {
        "text.color": "k",
        "axes.labelcolor": "k",
        "xtick.color": "k",
        "ytick.color": "k",
        "grid.color": "k",
        "axes.edgecolor": "k",
        "grid.color": "k",
    }
)

# Create the figure
##########################################
##########################################
rows, cols = 4, 3
fig, ax = plt.subplots(
    nrows=rows,
    ncols=cols,
    figsize=(8, 8),
    sharey=False,
    sharex=True,
    constrained_layout=True,
)


# Set colors
my_palette = ["#9ec6d2", "#f26464"]
control_color = my_palette[0]
darker_control = darken_hexcode(control_color, factor=0.2)
forced_color = my_palette[1]
darker_forced = darken_hexcode(forced_color, factor=0.2)

# Percent ticks and annotation specs
yticks = mtick.PercentFormatter(xmax=1, decimals=0, symbol="%")
offset = bar_width = 0.3
anno_buffer = 0.15
anno_width = 0.02

# Handle axes formating
for x in range(rows):
    for y in range(cols):
        ax[(x, y)].grid(
            axis="y",
            which="major",
            linestyle="-",
            linewidth="0.5",
            color="lightgray",
            zorder=0,
        )
        ax[(x, y)].set_ylim((0.2, 0.85))
        ax[(x, y)].spines["top"].set_visible(False)
        ax[(x, y)].spines["right"].set_visible(False)
        ax[(x, y)].yaxis.set_major_formatter(yticks)
        ax[(x, y)].tick_params(axis="x", which="both", length=0)
        if y > 0:
            ax[(x, y)].set_yticklabels([])
        if y != 0:
            ax[(x, y)].spines["left"].set_visible(False)
            ax[(x, y)].tick_params(axis="y", which="both", length=0)


# Where each plot will be placed
loc_map = {
    ("Belief", "False", "False"): (1, 0),
    ("Belief", "False", "Unsure"): (1, 1),
    ("Belief", "False", "True"): (1, 2),
    ("Belief", "True", "False"): (0, 0),
    ("Belief", "True", "Unsure"): (0, 1),
    ("Belief", "True", "True"): (0, 2),
    ("Share", "False", "False"): (3, 0),
    ("Share", "False", "Unsure"): (3, 1),
    ("Share", "False", "True"): (3, 2),
    ("Share", "True", "False"): (2, 0),
    ("Share", "True", "Unsure"): (2, 1),
    ("Share", "True", "True"): (2, 2),
}

base_x = np.array([0.1, 0.2])
for group in ["Belief", "Share"]:
    # Grab the right data dictionary based on the condition
    temp_dict = belief_dict if group == "Belief" else share_dict

    # Loop over the groundtruth vs. cgpt combinations
    for gt in ["True", "False"]:
        for cgpt_think in ["True", "Unsure", "False"]:
            # Get the location of the plot based on the setting
            plot_idx = loc_map[(group, gt, cgpt_think)]

            # This condition never happens, so we alter the axes and then skip
            if (gt, cgpt_think) == ("False", "True"):
                ax[plot_idx].spines["top"].set_visible(False)
                ax[plot_idx].spines["bottom"].set_visible(False)
                ax[plot_idx].spines["right"].set_visible(False)
                ax[plot_idx].spines["left"].set_visible(False)
                ax[plot_idx].set_yticks([])

                # if group == "Share":
                ax[plot_idx].set_xlabel("True\n")

                continue

            # Loop over each condition to plot both points
            for cond in ["Control", "Forced"]:
                # Grab the values needed
                mean_value = temp_dict[cond][((gt, cgpt_think))]["mean"]
                err = temp_dict[cond][((gt, cgpt_think))]["ci"]
                x = base_x[0] if cond == "Control" else base_x[1]
                c = control_color if gt == "True" else forced_color
                ec = darker_control if gt == "True" else darker_forced
                mstyle = "o" if cond == "Control" else ">"

                # Add the point
                ax[plot_idx].scatter(
                    x=x,
                    y=mean_value,
                    color=c,
                    marker=mstyle,
                    s=70,
                    edgecolor=ec,
                    zorder=3,
                )
                # Add the error bar
                ax[plot_idx].plot(
                    [x, x],
                    [mean_value - err, mean_value + err],
                    color="k",
                    zorder=2,
                )

                # Remove x-ticks/labels (add the ones we want later)
                if plot_idx[0] in [1, 3]:
                    temp_title = cgpt_think
                    ax[plot_idx].set_xlabel(temp_title, fontsize=16)
                else:
                    ax[plot_idx].set_xticks(base_x)
                    ax[plot_idx].set_xticklabels(["", ""], rotation=90)

# Set y-labels (LEFT)
# ax[(1, 0)].set_ylabel("the", fontsize=15)
# ax[(2, 0)].set_ylabel("", fontsize=15)

ylabel_believed = fig.text(
    s="Believed", x=0.02, y=0.76, rotation=90, ha="left", va="center"
)
ylabel_share = fig.text(
    s="Willing to share",
    x=0.02,
    y=0.3,
    rotation=90,
    ha="left",
    va="center",
)

# Set y-labels (RIGHT)
ax[(0, 2)].set_ylabel("\n\nTrue", rotation=-90, labelpad=50)
ax[(0, 2)].yaxis.set_label_position("right")

ax[(1, 2)].set_ylabel("\n\nFalse", rotation=-90, labelpad=50)
ax[(1, 2)].yaxis.set_label_position("right")


ax[(2, 2)].set_ylabel("\n\nTrue", rotation=-90, labelpad=50)
ax[(2, 2)].yaxis.set_label_position("right")

ax[(3, 2)].set_ylabel("\n\nFalse", rotation=-90, labelpad=50)
ax[(3, 2)].yaxis.set_label_position("right")

# Controls spacing of points
ax[(1, 0)].set_xlim((0, 0.3))

# Add supxlabel
supx = fig.supxlabel(
    "      LLM judgment", fontweight="bold", ha="center", va="bottom", fontsize=16
)
fig.suptitle(" ")

v1 = fig.text(
    s="Veracity",
    x=0.99,
    y=0.76,
    rotation=-90,
    ha="right",
    va="center",
    fontweight="bold",
)
fig.text(
    s="Veracity",
    x=0.99,
    y=0.30,
    rotation=-90,
    ha="right",
    va="center",
    fontweight="bold",
)


#### Annotations ####
# ---- ---- ---- ---- ----
anno_buffer = anno_width = 0.01
offset = 0.03

# Belief True/False
# ---- ---- ---- ---- ----
ymin = belief_dict["Forced"][("True", "False")]["mean"]
ymax = belief_dict["Control"][("True", "False")]["mean"]

# Vline
ax[(0, 0)].vlines(
    x=base_x[1] + offset + anno_buffer, ymin=ymin, ymax=ymax, color="black", lw=1
)
# H-lines (top)
ax[(0, 0)].hlines(
    y=ymax,
    xmin=base_x[1] + offset + anno_buffer - anno_width,
    xmax=base_x[1] + offset + anno_buffer,
    color="black",
    lw=1,
)
# H-lines (bottom)
ax[(0, 0)].hlines(
    y=ymin,
    xmin=base_x[1] + offset + anno_buffer - anno_width,
    xmax=base_x[1] + offset + anno_buffer,
    color="black",
    lw=1,
)
# Add text distance (discernment)
dist = ymin - ymax
num_str = f"{dist:.0%}"
ax[(0, 0)].annotate(
    num_str,
    xy=(base_x[1] + offset + anno_buffer, ymax),
    xytext=(base_x[1] + offset + anno_buffer, ymax),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=14,
)
# ax[(0, 0)].text(base_x[1] + 0.05, ymin + 0.03, "***", va="center")

# Belief False/Unsure
# ---- ---- ---- ---- ----
ymax = belief_dict["Forced"][("False", "Unsure")]["mean"]
ymin = belief_dict["Control"][("False", "Unsure")]["mean"]

# Vline
ax[(1, 1)].vlines(
    x=base_x[1] + offset + anno_buffer, ymin=ymin, ymax=ymax, color="black", lw=1
)
# H-lines (top)
ax[(1, 1)].hlines(
    y=ymax,
    xmin=base_x[1] + offset + anno_buffer - anno_width,
    xmax=base_x[1] + offset + anno_buffer,
    color="black",
    lw=1,
)
# H-lines (bottom)
ax[(1, 1)].hlines(
    y=ymin,
    xmin=base_x[1] + offset + anno_buffer - anno_width,
    xmax=base_x[1] + offset + anno_buffer,
    color="black",
    lw=1,
)
# Add text distance (discernment)
dist = ymin - ymax
num_str = f"{dist:.0%}"
ax[(1, 1)].annotate(
    num_str,
    xy=(base_x[1] + offset + anno_buffer * 2, ymax),
    xytext=(base_x[1] + offset + anno_buffer * 2, ymax),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=14,
)
# ax[(1, 1)].text(base_x[1] + 0.05, ymin + 0.02, "*", va="center")

# Share False/Unsure
# ---- ---- ---- ---- ----
ymax = share_dict["Forced"][("False", "Unsure")]["mean"]
ymin = share_dict["Control"][("False", "Unsure")]["mean"]

# Vline
ax[(3, 1)].vlines(
    x=base_x[1] + offset + anno_buffer, ymin=ymin, ymax=ymax, color="black", lw=1
)
# H-lines (top)
ax[(3, 1)].hlines(
    y=ymax,
    xmin=base_x[1] + offset + anno_buffer - anno_width,
    xmax=base_x[1] + offset + anno_buffer,
    color="black",
    lw=1,
)
# H-lines (bottom)
ax[(3, 1)].hlines(
    y=ymin,
    xmin=base_x[1] + offset + anno_buffer - anno_width,
    xmax=base_x[1] + offset + anno_buffer,
    color="black",
    lw=1,
)
# Add text distance (discernment)
dist = ymin - ymax
num_str = f"{dist:.0%}"
ax[(3, 1)].annotate(
    num_str,
    xy=(base_x[1] + offset + anno_buffer * 3, ymax),
    xytext=(base_x[1] + offset + anno_buffer * 3, ymax),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=14,
)
# ax[(3, 1)].text(base_x[1] + 0.05, ymin + 0.02, "*", va="center")

# Share True/True
# ---- ---- ---- ---- ----
ymax = share_dict["Forced"][("True", "True")]["mean"]
ymin = share_dict["Control"][("True", "True")]["mean"]

# Vline
ax[(2, 2)].vlines(
    x=base_x[1] + offset + anno_buffer, ymin=ymin, ymax=ymax, color="black", lw=1
)
# H-lines (top)
ax[(2, 2)].hlines(
    y=ymax,
    xmin=base_x[1] + offset + anno_buffer - anno_width,
    xmax=base_x[1] + offset + anno_buffer,
    color="black",
    lw=1,
)
# H-lines (bottom)
ax[(2, 2)].hlines(
    y=ymin,
    xmin=base_x[1] + offset + anno_buffer - anno_width,
    xmax=base_x[1] + offset + anno_buffer,
    color="black",
    lw=1,
)
# Add text distance (discernment)
dist = ymax - ymin
num_str = f"{dist:.0%}"
ax[(2, 2)].annotate(
    num_str,
    xy=(base_x[1] + offset + anno_buffer * 2, ymax),
    xytext=(base_x[1] + offset + anno_buffer * 2, ymax),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=14,
)
# ax[(2, 2)].text(base_x[1] + 0.05, ymin + 0.02, "*", va="center")


### Annotation to guide y-axis ###

### BELIEF TRUE
# Draw the arrow
ax[(0, 0)].annotate(
    "",
    xy=(0.05, 0.75),
    xytext=(0.05, 0.25),
    arrowprops=dict(
        mutation_scale=10,
        arrowstyle="<|-|>",
        linestyle=(0, (5, 10)),
        lw=0.5,
        color="black",
    ),
)

# Add the text labels
ax[(0, 0)].text(0.038, 0.71, "Good", rotation=90, ha="center", va="top", fontsize=12)
ax[(0, 0)].text(0.038, 0.31, "Bad", rotation=90, ha="center", va="bottom", fontsize=12)


### BELIEF False
# Draw the arrow
ax[(1, 0)].annotate(
    "",
    xy=(0.05, 0.75),
    xytext=(0.05, 0.25),
    arrowprops=dict(
        mutation_scale=10,
        arrowstyle="<|-|>",
        linestyle=(0, (5, 10)),
        lw=0.5,
        color="black",
    ),
)

# Add the text labels
ax[(1, 0)].text(0.038, 0.71, "Bad", rotation=90, ha="center", va="top", fontsize=12)
ax[(1, 0)].text(0.038, 0.31, "Good", rotation=90, ha="center", va="bottom", fontsize=12)


### SHARE TRUE
# Draw the arrow
ax[(2, 0)].annotate(
    "",
    xy=(0.05, 0.75),
    xytext=(0.05, 0.25),
    arrowprops=dict(
        mutation_scale=10,
        arrowstyle="<|-|>",
        linestyle=(0, (5, 10)),
        lw=0.5,
        color="black",
    ),
)

# Add the text labels
ax[(2, 0)].text(0.038, 0.71, "Good", rotation=90, ha="center", va="top", fontsize=12)
ax[(2, 0)].text(0.038, 0.31, "Bad", rotation=90, ha="center", va="bottom", fontsize=12)

### SHARE False
# Draw the arrow
ax[(3, 0)].annotate(
    "",
    xy=(0.05, 0.75),
    xytext=(0.05, 0.25),
    arrowprops=dict(
        mutation_scale=10,
        arrowstyle="<|-|>",
        linestyle=(0, (5, 10)),
        lw=0.5,
        color="black",
    ),
)

# Add the text labels
ax[(3, 0)].text(0.038, 0.71, "Bad", rotation=90, ha="center", va="top", fontsize=12)
ax[(3, 0)].text(0.038, 0.31, "Good", rotation=90, ha="center", va="bottom", fontsize=12)


#### LEGEND ####
# Create the legend handles and labels
handles = [
    plt.Line2D([], [], marker="o", linestyle="None", color="black", markersize=7),
    plt.Line2D([], [], marker=">", linestyle="None", color="black", markersize=7),
]
labels = ["Control", "Forced"]

# Create a handler for the handles
handler_map = {tuple(handles): HandlerTuple(ndivide=None)}

# Call tight layout before adding legend and midpoint line because it alters
# the coordinate system
# plt.tight_layout()

# Place the legend at the top of the figure
bbox = {"alpha": 0.5, "pad": 10}
legend = fig.legend(
    handles,
    labels,
    handler_map=handler_map,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,
    prop={"size": 16},
    frameon=False,
)

# Add a dashed line at the midpoint of the figure
# ax[(0, 0)].plot(
#     [0.03, 0.95],
#     [0.495, 0.495],
#     color="black",
#     linestyle=(0, (5, 10)),  # More control with line style
#     lw=1,
#     transform=plt.gcf().transFigure,
#     clip_on=False,
# )

ax[(0, 0)].annotate("(a)", xy=(-0.5, 0.8), xycoords=ax[(0, 0)].transAxes, fontsize=16)
ax[(2, 0)].annotate("(b)", xy=(-0.5, 0.8), xycoords=ax[(2, 0)].transAxes, fontsize=16)


fig.savefig(
    os.path.join(FIGURES_DIR, "five_way.pdf"),
    transparent=True,
    dpi=800,
)
fig.savefig(
    os.path.join(FIGURES_DIR, "five_way.png"),
    transparent=True,
    dpi=800,
)
