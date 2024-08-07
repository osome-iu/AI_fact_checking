"""
Purpose:
- Generate opt-in vs. opt-out figure by annotation.

Inputs:
- None

Outputs:
- chatgpt-fact-checker/figures/opt-in_vs_opt-out_by_annotation.pdf
- chatgpt-fact-checker/figures/opt-in_vs_opt-out_by_annotation.png

Author: Matthew DeVerna
"""

import os

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = "../../figures"

mpl.rcParams.update({"font.size": 12})
mpl.rcParams["hatch.linewidth"] = 3.0

# Ensure we are in the data_cleaning directory for paths to work
ROOT_DIR = "figure_creation"
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

# Local imports
from plot_utils import darken_hexcode

# Load optional means
OPTIONAL_MEANS = "../../results/optional_results_mean_ci_by_annotation.csv"
opt_means = pd.read_csv(OPTIONAL_MEANS)

# Set the color palette
my_palette = ["#A6D0DD", "#FF6969"]
darker_palette = [darken_hexcode(hex, factor=0.3) for hex in my_palette]

### Generate the figure
####################################
# Figure wide parameters
alpha = 1  # 0.95
offset = 0.2  # used for placing bars
offset_l = 0.15  # used for left-set annotations
offset_r = 0.35  # used for left-set annotations
base_x_vals = np.array([0, 0.7])
anno_buffer = 0.15
anno_width = 0.02
capsize = 0

# Location mapping for each scenario
idxloc_map = {
    ("Belief", "true"): (0, 0),
    ("Share", "true"): (1, 0),
    ("Belief", "unsure"): (0, 1),
    ("Share", "unsure"): (1, 1),
    ("Belief", "false"): (0, 2),
    ("Share", "false"): (1, 2),
}

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5), sharey=True)

# Set the y-axis tick formatter to show percentage values
fmt = "%.0f"  # This format string will display values as percentages without decimal places
yticks = mtick.PercentFormatter(xmax=1, decimals=0, symbol="%")
ax[(0, 0)].set_ylim(0, 1)

for selectors, data in opt_means.groupby(["Group", "ano_true_false_unsure"]):
    group, annotation = selectors
    loc = idxloc_map[(group, annotation)]

    is_opt_in = data["option_cond"] == "Opt_in"
    is_true = data["veracity"] == True

    # Select the mean values
    opt_out_true_mean = data[~is_opt_in & is_true]["w_mean"].item()
    opt_out_false_mean = data[~is_opt_in & ~is_true]["w_mean"].item()

    opt_in_true_mean = data[is_opt_in & is_true]["w_mean"].item()
    opt_in_false_mean = data[is_opt_in & ~is_true]["w_mean"].item()

    # Select the CI values
    opt_out_true_ci = data[~is_opt_in & is_true]["w_95_ci"].item()
    opt_out_false_ci = data[~is_opt_in & ~is_true]["w_95_ci"].item()

    opt_in_true_ci = data[is_opt_in & is_true]["w_95_ci"].item()
    opt_in_false_ci = data[is_opt_in & ~is_true]["w_95_ci"].item()

    # True
    ax[loc].bar(
        x=[base_x_vals[0], base_x_vals[0] + offset],
        height=[opt_out_true_mean, opt_in_true_mean],
        yerr=[opt_out_true_ci, opt_in_true_ci],
        ecolor="black",
        capsize=capsize,
        color=my_palette[0],
        edgecolor=darker_palette[0],
        hatch=["", "//"],
        width=offset,
        zorder=3,
        alpha=alpha,
    )

    # False
    # Skip the scenario where false items don't occur. Otherwise, lines are added at zero.
    if loc[1] != 0:
        ax[loc].bar(
            x=[base_x_vals[1], base_x_vals[1] + offset],
            height=[opt_out_false_mean, opt_in_false_mean],
            yerr=[opt_out_false_ci, opt_in_false_ci],
            ecolor="black",
            capsize=capsize,
            color=my_palette[1],
            edgecolor=darker_palette[1],
            hatch=["", "//"],
            width=offset,
            zorder=3,
            alpha=alpha,
        )

    # -------- -------- True annotation -------- --------

    ymin_b_true = opt_out_true_mean
    ymax_b_true = opt_in_true_mean
    dist = ymax_b_true - ymin_b_true

    ax[loc].vlines(
        x=base_x_vals[0] - offset_l if dist > 0 else base_x_vals[0] + offset_r,
        ymin=ymin_b_true,
        ymax=ymax_b_true,
        color="black",
        lw=1,
    )
    # H-lines (top)
    ax[loc].hlines(
        y=ymax_b_true,
        xmin=base_x_vals[0] - offset_l if dist > 0 else base_x_vals[0] + offset_r,
        xmax=(
            base_x_vals[0] - offset_l + anno_width
            if dist > 0
            else base_x_vals[0] + offset_r - anno_width
        ),
        color="black",
        lw=1,
    )
    # H-lines (bottom)
    ax[loc].hlines(
        y=ymin_b_true,
        xmin=base_x_vals[0] - offset_l if dist > 0 else base_x_vals[0] + offset_r,
        xmax=(
            base_x_vals[0] - offset_l + anno_width
            if dist > 0
            else base_x_vals[0] + offset_r - anno_width
        ),
        color="black",
        lw=1,
    )
    # Add text distance (discernment)
    num_str = f"{dist:.0%}" if dist > 0 else f"{dist*-1:.0%}"
    ax[loc].annotate(
        num_str,
        xy=(
            (base_x_vals[0] - offset_l, ymax_b_true)
            if dist > 0
            else (base_x_vals[0] + offset_r, ymin_b_true)
        ),
        xytext=(
            (base_x_vals[0] - offset_l, ymax_b_true)
            if dist > 0
            else (base_x_vals[0] + offset_r, ymin_b_true)
        ),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=12,
    )

    # -------- -------- False annotation -------- --------

    ymin_b_false = opt_out_false_mean
    ymax_b_false = opt_in_false_mean
    dist = ymax_b_false - ymin_b_false
    if dist == 0:
        continue

    ax[loc].vlines(
        x=base_x_vals[1] - offset_l,
        ymin=ymin_b_false,
        ymax=ymax_b_false,
        color="black",
        lw=1,
    )
    # H-lines (top)
    ax[loc].hlines(
        y=ymax_b_false,
        xmin=base_x_vals[1] - offset_l,
        xmax=base_x_vals[1] - offset_l + anno_width,
        color="black",
        lw=1,
    )
    # H-lines (bottom)
    ax[loc].hlines(
        y=ymin_b_false,
        xmin=base_x_vals[1] - offset_l,
        xmax=base_x_vals[1] - offset_l + anno_width,
        color="black",
        lw=1,
    )
    # Add text distance (discernment)
    num_str = f"{dist:.0%}"
    ax[loc].annotate(
        num_str,
        xy=(base_x_vals[1] - offset_l, ymax_b_false),
        xytext=(base_x_vals[1] - offset_l, ymax_b_false),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=12,
    )

ax[(0, 0)].set_ylabel("Believe")
ax[(1, 0)].set_ylabel("Willing to share")

ax[(0, 0)].set_title("LLM judged True", pad=15, fontsize=12)
ax[(0, 1)].set_title("LLM judged Unsure", pad=15, fontsize=12)
ax[(0, 2)].set_title("LLM judged False", pad=15, fontsize=12)

ax[(1, 0)].set_title("LLM judged True", pad=15, fontsize=12)
ax[(1, 1)].set_title("LLM judged Unsure", pad=15, fontsize=12)
ax[(1, 2)].set_title("LLM judged False", pad=15, fontsize=12)

plt.tight_layout()

desired_order = ["Opt out", "Opt in"]

# Create custom legend handles with desired colors
sorted_handles = [
    mpatches.Patch(facecolor="white", edgecolor="black", label="Opt out"),
    mpatches.Patch(facecolor="white", edgecolor="black", label="Opt in", hatch="//"),
]

fig.legend(
    sorted_handles,
    desired_order,
    ncol=1,
    loc="upper right",
    bbox_to_anchor=(1, 0.91),
    labelspacing=0,
    frameon=False,
    fontsize=10,
)


# Set the y-axis tick formatter to show percentage values
fmt = "%.0f"  # This format string will display values as percentages without decimal places
yticks = mtick.PercentFormatter(xmax=1, decimals=0, symbol="%")

for loc_tuple in idxloc_map.values():
    ax[loc_tuple].set_xticks((base_x_vals + offset / 2), [True, False])

    ax[loc_tuple].grid(
        axis="y",
        which="major",
        linestyle="-",
        linewidth="0.5",
        color="lightgray",
        zorder=0,
    )
    for loc in ["top", "right"]:
        ax[loc_tuple].spines[loc].set_visible(False)
    ax[loc_tuple].yaxis.set_major_formatter(yticks)
    ax[loc_tuple].set_xlim(-0.3, 1.1)

plt.subplots_adjust(hspace=0.6)

# Add annotations
ax[(0, 0)].annotate(
    "(a)", xy=(-0.25, 1.25), xycoords=ax[(0, 0)].transAxes, ha="center", fontsize=16
)
ax[(0, 0)].annotate(
    "(b)", xy=(-0.25, 1.25), xycoords=ax[(1, 0)].transAxes, ha="center", fontsize=16
)


fig.savefig(
    os.path.join(FIGURES_DIR, "opt-in_vs_opt-out_by_annotation.pdf"),
    transparent=True,
    dpi=800,
    bbox_inches="tight",
)
fig.savefig(
    os.path.join(FIGURES_DIR, "opt-in_vs_opt-out_by_annotation.png"),
    transparent=True,
    dpi=800,
    bbox_inches="tight",
)
fig.savefig(
    os.path.join(FIGURES_DIR, "opt-in_vs_opt-out_by_annotation.svg"),
    transparent=True,
    dpi=800,
    bbox_inches="tight",
)
