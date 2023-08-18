"""
Purpose:
- Generate Figure showing attitudes of AI relationship with belief and sharing behavior
    for each headline type (veracity x cGPT judgement).

Inputs:
- None

Outputs:
- chatgpt-fact-checker/figures/SI_belief_at-ai.pdf
- chatgpt-fact-checker/figures/SI_belief_at-ai.png
- chatgpt-fact-checker/figures/SI_share_at-ai.pdf
- chatgpt-fact-checker/figures/SI_share_at-ai.png

Author: Matthew DeVerna
"""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
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
FIGURES_DIR = "../../figures"
DATA_DIR = "../../data/cleaned_data/"
ALL_DATA_FILE = find_file(DATA_DIR, "*long_form_cgpt_fc_paper.parquet")
all_data = pd.read_parquet(ALL_DATA_FILE)

# Add a new coding that captures veracity x cGPT judgement
new_coding = [
    f"{truth}_{cgpt}"
    for truth, cgpt in zip(
        all_data["veracity"].astype(str),
        all_data["ano_true_false_unsure"].str.capitalize(),
    )
]
all_data["v_split_by_cgpt"] = new_coding

# Calculate the number of yes responses per participant (ResponseId)
yes_by_participant = (
    all_data.groupby(
        [
            "Group",
            "Condition",
            "ResponseId",
            "v_split_by_cgpt",
        ]
    )["exp_response"]
    .sum()
    .to_frame("num_yes")
    .reset_index()
)

# Calculate the number of headlines for each type of headline
single_participant = list(all_data.ResponseId)[:1]
num_headlines_map = (
    all_data[all_data.ResponseId.isin(single_participant)]["v_split_by_cgpt"]
    .value_counts()
    .to_dict()
)

# Calculates the proportion of each type of headline that a participant got correct
# E.g., What proportion of the "True_False" headlines, did participant X answer "Yes" to?
prop_yes = []
for idx, row_data in yes_by_participant.iterrows():
    prop_yes.append(row_data.num_yes / num_headlines_map[row_data.v_split_by_cgpt])
yes_by_participant["prop_yes"] = prop_yes

# Split this column into two again
yes_by_participant[["ground_truth", "cGPT_thinks"]] = yes_by_participant[
    "v_split_by_cgpt"
].str.split("_", 1, expand=True)

# Merge participants attitudes about AI value
yes_by_participant = yes_by_participant.merge(
    all_data[["ResponseId", "AI_att_mean"]].drop_duplicates(subset="ResponseId"),
    on="ResponseId",
    how="left",
)

# Select the data for later
b_no_opt = yes_by_participant[
    (yes_by_participant.Group == "Belief")
    & (yes_by_participant.Condition != "Optional")
].copy()
s_no_opt = yes_by_participant[
    (yes_by_participant.Group == "Share") & (yes_by_participant.Condition != "Optional")
].copy()


######### BELIEF FIGURE #########
#################################
rows, cols = 2, 3
fig, axs = plt.subplots(rows, cols, figsize=(10, 8), sharex=False, sharey=True)

loc_map = {
    ("False", "False"): (1, 0),
    ("False", "Unsure"): (1, 1),
    ("False", "True"): (1, 2),
    ("True", "False"): (0, 0),
    ("True", "Unsure"): (0, 1),
    ("True", "True"): (0, 2),
}

yticks = mtick.PercentFormatter(xmax=1, decimals=0, symbol="%")
for x in range(rows):
    for y in range(cols):
        axs[(x, y)].grid(
            axis="y",
            which="major",
            linestyle="-",
            linewidth="0.5",
            color="lightgray",
            zorder=0,
        )
        axs[(x, y)].spines["top"].set_visible(False)
        axs[(x, y)].spines["right"].set_visible(False)
        axs[(x, y)].yaxis.set_major_formatter(yticks)

        axs[(x, y)].xaxis.set_ticks(range(8), range(8))

        if y == 0:
            axs[(x, y)].set_ylabel("Proportion believed")
        else:
            axs[(x, y)].set_ylabel(None)
        if x == 1:
            axs[(x, x)].set_xlabel("Attitude towards AI")
        else:
            axs[(x, x)].set_xlabel(None)

# These define the CENTER of the bins — does not affect regression.
#    Ref: https://seaborn.pydata.org/generated/seaborn.regplot.html
#        - See `x_bins` description for more info.
xbins = np.arange(1, 7.5, 0.5)

for veracity in ["False", "True"]:
    for cgpt_think in ["False", "Unsure", "True"]:
        selected_df = b_no_opt[
            (b_no_opt.ground_truth == veracity) & (b_no_opt.cGPT_thinks == cgpt_think)
        ]
        loc_tuple = loc_map[(veracity, cgpt_think)]

        for cond in ["Forced", "Control"]:
            sns.regplot(
                data=selected_df[selected_df.Condition == cond],
                x="AI_att_mean",
                y="prop_yes",
                x_estimator=np.mean,
                x_bins=xbins,
                label=cond,
                n_boot=1_000,
                robust=True,
                ax=axs[loc_tuple],
            )

        axs[loc_tuple].set_title(f"{cgpt_think}")
        axs[loc_tuple].grid(zorder=0)

for x in range(rows):
    for y in range(cols):
        if (y == 0) and (x == 0):
            axs[(x, y)].set_ylabel("True Believed")
        elif (y == 0) and (x == 1):
            axs[(x, y)].set_ylabel("False Believed")
        else:
            axs[(x, y)].set_ylabel("")

        axs[(x, y)].set_xlabel("Attitude towards AI")

axs[(0, 2)].legend()
axs[(1, 2)].remove()
fig.tight_layout()

fig.savefig(
    os.path.join(FIGURES_DIR, "SI_5way_belief_at-ai.pdf"),
    dpi=800,
)
fig.savefig(
    os.path.join(FIGURES_DIR, "SI_5way_belief_at-ai.png"), dpi=800, transparent=True
)

######### SHARE FIGURE #########
################################

rows, cols = 2, 3
fig, axs = plt.subplots(rows, cols, figsize=(10, 8), sharex=False, sharey=True)

# Set the axes
yticks = mtick.PercentFormatter(xmax=1, decimals=0, symbol="%")
for x in range(rows):
    for y in range(cols):
        axs[(x, y)].grid(
            axis="y",
            which="major",
            linestyle="-",
            linewidth="0.5",
            color="lightgray",
            zorder=0,
        )
        axs[(x, y)].spines["top"].set_visible(False)
        axs[(x, y)].spines["right"].set_visible(False)
        axs[(x, y)].yaxis.set_major_formatter(yticks)

        axs[(x, y)].xaxis.set_ticks(range(8), range(8))

        if y == 0:
            axs[(x, y)].set_ylabel("Proportion believed")
        else:
            axs[(x, y)].set_ylabel(None)
        if x == 1:
            axs[(x, x)].set_xlabel("Attitude towards AI")
        else:
            axs[(x, x)].set_xlabel(None)

# These define the CENTER of the bins — does not affect regression.
xbins = np.arange(1, 7.5, 0.5)

for veracity in ["False", "True"]:
    for cgpt_think in ["False", "Unsure", "True"]:
        selected_df = s_no_opt[
            (s_no_opt.ground_truth == veracity) & (s_no_opt.cGPT_thinks == cgpt_think)
        ]
        loc_tuple = loc_map[(veracity, cgpt_think)]

        for cond in ["Forced", "Control"]:
            sns.regplot(
                data=selected_df[selected_df.Condition == cond],
                x="AI_att_mean",
                y="prop_yes",
                x_estimator=np.mean,
                x_bins=xbins,
                label=cond,
                n_boot=1_000,
                robust=True,
                ax=axs[loc_tuple],
            )

        axs[loc_tuple].set_title(f"{cgpt_think}")
        axs[loc_tuple].grid(zorder=0)

for x in range(rows):
    for y in range(cols):
        if (y == 0) and (x == 0):
            axs[(x, y)].set_ylabel("True willing\nto share")
        elif (y == 0) and (x == 1):
            axs[(x, y)].set_ylabel("False willing\nto share")
        else:
            axs[(x, y)].set_ylabel("")

        axs[(x, y)].set_xlabel("Attitude towards AI")

axs[(0, 2)].legend(fontsize=12)
axs[(1, 2)].remove()
fig.tight_layout()

fig.savefig(
    os.path.join(FIGURES_DIR, "SI_5way_share_at-ai.pdf"),
    dpi=800,
)
fig.savefig(
    os.path.join(FIGURES_DIR, "SI_5way_share_at-ai.png"), dpi=800, transparent=True
)
