"""
Purpose:
- Generate figures that show the proportion of headlines that people opt'd into in the
    optional condition. One figure is for the different groups ("Belief", "Share") and
    another breaks it out by veracity and group.

Inputs:
- None

Outputs:
- chatgpt-fact-checker/figures/SI_opt_in_distributions_by_group.pdf
- chatgpt-fact-checker/figures/SI_opt_in_distributions_by_group.png
- chatgpt-fact-checker/figures/SI_opt_in_distributions_by_veracity.pdf
- chatgpt-fact-checker/figures/SI_opt_in_distributions_by_veracity.png

Author: Matthew DeVerna
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd

plt.rcParams.update({"font.size": 12})

# Constants
BY_GROUP_FILE = "../../results/opt_in_counts_proportions.csv"
BY_VERACITY_FILE = "../../results/opt_in_counts_proportions_by_veracity.csv"
FIGURES_DIR = "../../figures"

# Ensure we are in the script's directory for paths to work
if os.getcwd() != os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))


# Colors to match the main figure
COLOR_MAP = {True: "#aad2df", False: "#ff7171"}

# Load the data files
by_group_df = pd.read_csv(BY_GROUP_FILE)
by_veracity_df = pd.read_csv(BY_VERACITY_FILE)

# By group
# ---------------------

by_group_opt_in = by_group_df[by_group_df["option_cond"] == "Opt_in"]

fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(
    data=by_group_opt_in,
    x="proportion",
    hue="Group",
    kde=True,
    bins=10,
)

ax.set_xlabel("proportion of headlines")
ax.set_ylabel("number of participants")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

fig.savefig(
    os.path.join(FIGURES_DIR, "SI_opt_in_distributions_by_group.pdf"),
    dpi=800,
)
fig.savefig(
    os.path.join(FIGURES_DIR, "SI_opt_in_distributions_by_group.png"),
    dpi=800,
    transparent=True,
)


# By veracity
# ---------------------

by_veracity_opt_in = by_veracity_df[by_veracity_df["option_cond"] == "Opt_in"]

belief_df = by_veracity_opt_in[by_veracity_opt_in.Group == "Belief"]
share_df = by_veracity_opt_in[by_veracity_opt_in.Group == "Share"]

fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

sns.histplot(
    data=belief_df, x="proportion", hue="veracity", kde=True, bins=10, ax=ax[0]
)

sns.histplot(data=share_df, x="proportion", hue="veracity", kde=True, bins=10, ax=ax[1])

ax[0].set_title("Belief")
ax[1].set_title("Share")

for a in ax:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.set_ylim(0, 130)
    a.set_xlabel("proportion of headlines")
    a.set_ylabel("number of participants")

# Add figure annotations
ax[0].annotate("(a)", xy=(-0.2, 1.1), xycoords=ax[0].transAxes, fontsize=16)
ax[1].annotate("(b)", xy=(-0.2, 1.1), xycoords=ax[1].transAxes, fontsize=16)

fig.savefig(
    os.path.join(FIGURES_DIR, "SI_opt_in_distributions_by_veracity.pdf"),
    dpi=800,
)
fig.savefig(
    os.path.join(FIGURES_DIR, "SI_opt_in_distributions_by_veracity.png"),
    dpi=800,
    transparent=True,
)
