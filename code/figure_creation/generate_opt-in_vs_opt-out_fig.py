"""
Purpose:
- Generate opt-in vs. opt-out figure.

Inputs:
- None

Outputs:
- chatgpt-fact-checker/figures/opt-in_vs_opt-out.pdf
- chatgpt-fact-checker/figures/opt-in_vs_opt-out.png

Author: Matthew DeVerna
"""
import os

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict

FIGURES_DIR = "../../figures"

mpl.rcParams.update({"font.size": 16})
mpl.rcParams["hatch.linewidth"] = 3.0

# Ensure we are in the data_cleaning directory for paths to work
ROOT_DIR = "figure_creation"
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

# Local imports
from plot_utils import darken_hexcode

# Load optional means
OPTIONAL_MEANS = "../../results/optional_results_mean_ci.csv"
opt_means = pd.read_csv(OPTIONAL_MEANS)

# Convert to dictionaries for easier plotting
means = defaultdict(dict)
for _, row in opt_means.iterrows():
    group = row["Group"]
    key = f"{row['option_cond']}_{'true' if row['veracity'] else 'false'}".lower()
    val = row["w_mean"]
    means[group].update({key: val})
means = dict(means)

cis = defaultdict(dict)
for _, row in opt_means.iterrows():
    group = row["Group"]
    key = f"{row['option_cond']}_{'true' if row['veracity'] else 'false'}".lower()
    val = row["w_95_ci"]
    cis[group].update({key: val})
cis = dict(cis)

# Set the color palette
my_palette = ["#A6D0DD", "#FF6969"]
darker_palette = [darken_hexcode(hex, factor=0.3) for hex in my_palette]

### Generate the figure
####################################
# Figure wide parameters
alpha = 1  # 0.95
offset = 0.2  # used for placing bars
offset_l = 0.15  # used for left-set annotations
base_x_vals = np.array([0, 0.7])
anno_buffer = 0.15
anno_width = 0.02
capsize = 3

# Create figure with subplots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True, sharex=True)

# Set the y-axis tick formatter to show percentage values
fmt = "%.0f"  # This format string will display values as percentages without decimal places
yticks = mtick.PercentFormatter(xmax=1, decimals=0, symbol="%")

# Handle axes formating
for x in range(2):
    ax[x].grid(
        axis="y",
        which="major",
        linestyle="-",
        linewidth="0.5",
        color="lightgray",
        zorder=0,
    )
    for loc in ["top", "right"]:
        ax[x].spines[loc].set_visible(False)
    ax[x].yaxis.set_major_formatter(yticks)
    ax[x].set_xlim(-0.3, 1.1)

#### Belief Figure ####
## ----------------- ##

# Opt-out
ax[0].bar(
    x=base_x_vals,
    height=[
        means["Belief"]["opt_out_true"],
        means["Belief"]["opt_out_false"],
    ],
    yerr=[
        cis["Belief"]["opt_out_true"],
        cis["Belief"]["opt_out_false"],
    ],
    ecolor="black",
    capsize=capsize,
    color=[my_palette[0], my_palette[1]],
    edgecolor=[darker_palette[0], darker_palette[1]],
    hatch=["", ""],
    width=offset,
    zorder=3,
    alpha=alpha,
)

# Opt-in
ax[0].bar(
    x=base_x_vals + offset,
    height=[
        means["Belief"]["opt_in_true"],
        means["Belief"]["opt_in_false"],
    ],
    yerr=[
        cis["Belief"]["opt_in_true"],
        cis["Belief"]["opt_in_false"],
    ],
    ecolor="black",
    capsize=capsize,
    color=[my_palette[0], my_palette[1]],
    edgecolor=[darker_palette[0], darker_palette[1]],
    hatch=["//", "//"],
    width=offset,
    zorder=3,
    alpha=alpha,
)

ax[0].set_ylabel("Believed")

# -------- -------- True annotation -------- --------

ymin_b_true = means["Belief"]["opt_out_true"]
ymax_b_true = means["Belief"]["opt_in_true"]


ax[0].vlines(
    x=base_x_vals[0] - offset_l, ymin=ymin_b_true, ymax=ymax_b_true, color="black", lw=1
)
# H-lines (top)
ax[0].hlines(
    y=ymax_b_true,
    xmin=base_x_vals[0] - offset_l,
    xmax=base_x_vals[0] - offset_l + anno_width,
    color="black",
    lw=1,
)
# H-lines (bottom)
ax[0].hlines(
    y=ymin_b_true,
    xmin=base_x_vals[0] - offset_l,
    xmax=base_x_vals[0] - offset_l + anno_width,
    color="black",
    lw=1,
)
# Add text distance (discernment)
dist = ymax_b_true - ymin_b_true
num_str = f"{dist:.0%}"
ax[0].annotate(
    num_str,
    xy=(base_x_vals[0] - offset_l, ymax_b_true),
    xytext=(base_x_vals[0] - offset_l, ymax_b_true),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=14,
)

# -------- -------- False annotation -------- --------

ymin_b_false = means["Belief"]["opt_out_false"]
ymax_b_false = means["Belief"]["opt_in_false"]

ax[0].vlines(
    x=base_x_vals[1] - offset_l,
    ymin=ymin_b_false,
    ymax=ymax_b_false,
    color="black",
    lw=1,
)
# H-lines (top)
ax[0].hlines(
    y=ymax_b_false,
    xmin=base_x_vals[1] - offset_l,
    xmax=base_x_vals[1] - offset_l + anno_width,
    color="black",
    lw=1,
)
# H-lines (bottom)
ax[0].hlines(
    y=ymin_b_false,
    xmin=base_x_vals[1] - offset_l,
    xmax=base_x_vals[1] - offset_l + anno_width,
    color="black",
    lw=1,
)
# Add text distance (discernment)
dist = ymax_b_false - ymin_b_false
num_str = f"{dist:.0%}"
ax[0].annotate(
    num_str,
    xy=(base_x_vals[1] - offset_l, ymax_b_false),
    xytext=(base_x_vals[1] - offset_l, ymax_b_false),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=14,
)


#### Sharing Figure ####
## ------------------ ##

# Opt-out
ax[1].bar(
    x=base_x_vals,
    height=[
        means["Share"]["opt_out_true"],
        means["Share"]["opt_out_false"],
    ],
    yerr=[
        cis["Share"]["opt_out_true"],
        cis["Share"]["opt_out_false"],
    ],
    ecolor="black",
    capsize=capsize,
    color=[my_palette[0], my_palette[1]],
    edgecolor=[darker_palette[0], darker_palette[1]],
    hatch=["", ""],
    width=offset,
    zorder=3,
    alpha=alpha,
    label="Opt-out",
)

# Opt-in
ax[1].bar(
    x=base_x_vals + offset,
    height=[
        means["Share"]["opt_in_true"],
        means["Share"]["opt_in_false"],
    ],
    yerr=[
        cis["Share"]["opt_in_true"],
        cis["Share"]["opt_in_false"],
    ],
    ecolor="black",
    capsize=capsize,
    color=[my_palette[0], my_palette[1]],
    edgecolor=[darker_palette[0], darker_palette[1]],
    hatch=["//", "//"],
    width=offset,
    zorder=3,
    alpha=alpha,
    label="Opt-in",
)

# -------- -------- True annotation -------- --------

ymin_b_true = means["Share"]["opt_out_true"]
ymax_b_true = means["Share"]["opt_in_true"]

ax[1].vlines(
    x=base_x_vals[0] - offset_l, ymin=ymin_b_true, ymax=ymax_b_true, color="black", lw=1
)
# H-lines (top)
ax[1].hlines(
    y=ymax_b_true,
    xmin=base_x_vals[0] - offset_l,  # +offset+anno_buffer-anno_width,
    xmax=base_x_vals[0] - offset_l + anno_width,  # +offset+anno_buffer,
    color="black",
    lw=1,
)
# H-lines (bottom)
ax[1].hlines(
    y=ymin_b_true,
    xmin=base_x_vals[0] - offset_l,  # +offset+anno_buffer-anno_width,
    xmax=base_x_vals[0] - offset_l + anno_width,  # +offset+anno_buffer,
    color="black",
    lw=1,
)
# Add text distance (discernment)
dist = ymax_b_true - ymin_b_true
num_str = f"{dist:.0%}"
ax[1].annotate(
    num_str,
    xy=(base_x_vals[0] - offset_l, ymax_b_true),
    xytext=(base_x_vals[0] - offset_l, ymax_b_true),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=14,
)

# -------- -------- False annotation -------- --------

ymin_b_false = means["Share"]["opt_out_false"]
ymax_b_false = means["Share"]["opt_in_false"]

ax[1].vlines(
    x=base_x_vals[0] - offset_l, ymin=ymin_b_true, ymax=ymax_b_true, color="black", lw=1
)
# H-lines (top)
ax[1].vlines(
    x=base_x_vals[1] - offset_l,
    ymin=ymin_b_false,
    ymax=ymax_b_false,
    color="black",
    lw=1,
)
# H-lines (top)
ax[1].hlines(
    y=ymax_b_false,
    xmin=base_x_vals[1] - offset_l,  # +offset+anno_buffer-anno_width,
    xmax=base_x_vals[1] - offset_l + anno_width,  # +offset+anno_buffer,
    color="black",
    lw=1,
)
# H-lines (bottom)
ax[1].hlines(
    y=ymin_b_false,
    xmin=base_x_vals[1] - offset_l,  # +offset+anno_buffer-anno_width,
    xmax=base_x_vals[1] - offset_l + anno_width,  # +offset+anno_buffer,
    color="black",
    lw=1,
)
# Add text distance (discernment)
dist = ymax_b_false - ymin_b_false
num_str = f"{dist:.0%}"
ax[1].annotate(
    num_str,
    xy=(base_x_vals[1] - offset_l, ymax_b_false),
    xytext=(base_x_vals[1] - offset_l, ymax_b_false),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=14,
)

veracity_options = [True, False]
ax[1].set_ylabel("Willing to share")
ax[0].set_xticks((base_x_vals + offset / 2), veracity_options)

# Specify the desired order of legend item labels
desired_order = ["Opt out", "Opt in"]

# Create custom legend handles with desired colors
sorted_handles = [
    mpatches.Patch(facecolor="white", edgecolor="black", label="Opt out"),
    mpatches.Patch(facecolor="white", edgecolor="black", label="Opt in", hatch="//"),
]

fig.legend(
    sorted_handles,
    desired_order,
    ncol=1,  # For wide
    loc="upper right",  # For wide
    bbox_to_anchor=(0.93, 0.93),  # For wide
    labelspacing=0,
    frameon=False,
)

# Add annotations
ax[0].annotate("(a)", xy=(0, 1.05), xycoords=ax[0].transAxes, ha="center", fontsize=16)
ax[1].annotate("(b)", xy=(0, 1.05), xycoords=ax[1].transAxes, ha="center", fontsize=16)


fig.savefig(
    os.path.join(FIGURES_DIR, "opt-in_vs_opt-out.pdf"),
    transparent=True,
    dpi=800,
)
fig.savefig(
    os.path.join(FIGURES_DIR, "opt-in_vs_opt-out.png"),
    transparent=True,
    dpi=800,
)
