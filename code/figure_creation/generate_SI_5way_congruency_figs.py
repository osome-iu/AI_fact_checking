"""
Purpose:
- Generate congruency SI figure for the five-way main text figure.

Inputs:
- None

Outputs:
- chatgpt-fact-checker/figures/SI_5way_congruency_belief.pdf
- chatgpt-fact-checker/figures/SI_5way_congruency_belief.png
- chatgpt-fact-checker/figures/SI_5way_congruency_share.pdf
- chatgpt-fact-checker/figures/SI_5way_congruency_share.png

Author: Matthew DeVerna
"""

import os

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 16})

ROOT_DIR = "figure_creation"
# Ensure we are in the data_cleaning directory for paths to work
if os.path.basename(os.getcwd()) != ROOT_DIR:
    raise Exception(f"Must run this script from the `{ROOT_DIR}` directory!")

GROUPS = ["Belief", "Share"]
DATA_DIR = "../../results"
DATA_FNAME = "SI_5way_with_congruency_mean_and_ci"
FIGURES_DIR = "../../figures"

# Load data
grouped_results = pd.read_csv(f"{DATA_DIR}/{DATA_FNAME}.csv")

for temp_group in GROUPS:
    # Create a grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # Define the combinations of temp_cGPT_think and temp_veracity
    combinations = [
        ("false", True),
        ("unsure", True),
        ("true", True),
        ("false", False),
        ("unsure", False),
    ]

    color_map = {
        True: "#9ec6d2",
        False: "#f26464",
    }
    mark_dict = {"Control": "o", "Forced": "x", "Optional": ">", "Human-FC": "d"}
    xbuff_map = {"Control": -0.2, "Forced": -0.05, "Optional": 0.1, "Human-FC": 0.25}

    x_base = np.array([-0.5, 0.5])

    # Iterate over the combinations and plot each panel
    for i, (temp_cGPT_think, temp_veracity) in enumerate(combinations):
        # Filter the data
        b_t_f_avgs = grouped_results[
            (grouped_results.Group == temp_group)
            & (grouped_results.veracity == temp_veracity)
            & (grouped_results.ano_true_false_unsure == temp_cGPT_think)
        ]

        # Determine the subplot position
        row = i // 3
        col = i % 3

        # Plot the panel
        ax = axs[row, col]

        for cond in b_t_f_avgs.Condition.unique():
            sliced_df = b_t_f_avgs[b_t_f_avgs.Condition == cond]

            ax.set_title(cond)

            # Scatter plot
            xbuff = xbuff_map[cond]
            ax.scatter(
                x_base + xbuff,
                sliced_df["mean"],
                label="mean",
                color=color_map[temp_veracity],
                marker=mark_dict[cond],
                zorder=3,
                s=100,
            )

            # Error bars
            ax.errorbar(
                x_base + xbuff,
                sliced_df["mean"],
                yerr=sliced_df["ci"],
                fmt="none",
                color="grey",
                zorder=1,
            )

            # Axes labels, limits, and grid
            if col == 0:
                prefix = "True" if row == 0 else "False"
                suffix = "believed" if temp_group == "Belief" else "willing to share"
                ax.set_ylabel(f"{prefix} {suffix}")
            if col in [1, 2]:
                ax.yaxis.set_ticklabels([])
                ax.yaxis.set_tick_params(length=0)
                ax.spines["left"].set_visible(False)
            ax.set_ylim([0, 1])
            ax.grid(True, axis="y", zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(-1, 1)
            ax.xaxis.set_ticks(x_base, ["congruent", "incongruent"])

    # Add legend to final subplot and then remove everything else
    handles = [
        plt.Line2D([], [], color="k", marker="o", linestyle="None", label="Control"),
        plt.Line2D([], [], color="k", marker="x", linestyle="None", label="Forced"),
        plt.Line2D([], [], color="k", marker=">", linestyle="None", label="Optional"),
        plt.Line2D([], [], color="k", marker="d", linestyle="None", label="Human FC"),
    ]

    labels = [handle.get_label() for handle in handles]

    # Create the legend
    axs[1, 2].legend(
        title="Condition", handles=handles, labels=labels, loc="upper left"
    )

    # Remove False x True
    axs[1, 2].spines["top"].set_visible(False)
    axs[1, 2].spines["bottom"].set_visible(False)
    axs[1, 2].spines["left"].set_visible(False)
    axs[1, 2].spines["right"].set_visible(False)
    axs[1, 2].xaxis.set_ticks([])
    axs[1, 2].yaxis.set_ticks([])

    # Add labels
    cGPT_labels = ["False", "Unsure", "True", "False", "Unsure"]
    for i in range(5):
        axs[i // 3, i % 3].set_title(f"{cGPT_labels[i]}")

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    # Save the plot
    plt.savefig(f"{FIGURES_DIR}/SI_5way_congruency_{temp_group.lower()}.pdf", dpi=800)
    plt.savefig(
        f"{FIGURES_DIR}/SI_5way_congruency_{temp_group.lower()}.png",
        transparent=True,
        dpi=800,
    )
