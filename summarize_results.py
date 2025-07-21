import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from torchvision.models.quantization import resnet18


def filter_by_pretrained(df: pd.DataFrame):
    pretrained = df[df["pretrained"] == True]
    random = df[df["pretrained"] == False]
    return pretrained, random

def filter_by_augmented(df: pd.DataFrame):
    augmented = df[df["augmentations"] != "No Augmentations"]
    not_augmented = df[df["augmentations"] == "No Augmentations"]
    return augmented, not_augmented

def filter_by_architecture(df: pd.DataFrame):
    resnet18 = df[df["architecture"] == "ResNet18"]
    resnet50 = df[df["architecture"] == "ResNet50"]
    return resnet18, resnet50


def plot_auc(df: pd.DataFrame, group):
    architecture = "ResNet18"
    # Sample data

    target_domains = df["taget_domain"].unique()

    augmentations = df[group].unique()
    combinations = product(target_domains, augmentations)

    auc_scores = []
    labels = []
    for target_domain, augmentation in combinations:
        row = df[(df["taget_domain"] == target_domain) & (df[group] == augmentation)]
        auc_score = row["auc-score"].iloc[0]
        auc_scores.append(auc_score)
        labels.append(augmentation)
    # # Settings
    n_groups = len(target_domains)
    n_bars =  len(auc_scores) // n_groups # number of bars in each group
    bar_width = 0.3
    inner_gap = 0.1  # gap between bars inside a group
    outer_gap = 0.8   # gap between bar groups
    #
    # # Compute total width for one group (including internal gaps)
    group_width = n_bars * bar_width + (n_bars - 1) * inner_gap
    #
    # # x positions for each group center
    x = np.arange(n_groups) * (group_width + outer_gap)
    #
    fig, ax = plt.subplots()

    target_domains = [f"Domain {x}" for x in target_domains]
    # Bar positions within each group
    for i, (data, label) in enumerate(zip(auc_scores, augmentations)):
        offset = -group_width / 2 + i * (bar_width + inner_gap) + bar_width / 2
        ax.bar(x + offset, data, width=bar_width, label=label)

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(target_domains)
    ax.set_ylabel('Score')
    ax.legend()

    y_min, y_max = ax.get_ylim()
    y_ticks = np.linspace(y_min, 1, 11)  # 10 intervals = 11 ticks
    ax.set_yticks(y_ticks)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    # Move legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=group)

    plt.tight_layout()
    plt.show()


dir = "results"

df_unbalanced = pd.read_csv(os.path.join(dir, "results_camelyon_unbalanced.csv"))
df_balanced = pd.read_csv(os.path.join(dir, "results_camelyon.csv"))
group_by = ["taget_domain", "augmentations", "architecture"]
avg_balanced = df_balanced.groupby(group_by).mean(numeric_only=True).reset_index().drop(columns=["Unnamed: 0", "i", "pretrained"])
avg_unbalanced = df_unbalanced.groupby(group_by).mean(numeric_only=True).reset_index().drop(columns=["Unnamed: 0", "i", "pretrained"])

joined = pd.merge(avg_balanced, avg_unbalanced, on=["taget_domain", "augmentations", "architecture"], suffixes=("_balanced", "_unbalanced"))
joined.to_csv(os.path.join(dir, "joined.csv"))

resnet18, resnet50 = filter_by_architecture(joined)

resnet18.to_csv(os.path.join(dir, "joined_resnet18.csv"))
resnet50.to_csv(os.path.join(dir, "joined_resnet50.csv"))

_, not_augmented_18 = filter_by_augmented(resnet18)
_, not_augmented_50 = filter_by_augmented(resnet50)
not_augmented = pd.merge(not_augmented_18, not_augmented_50, on=["taget_domain"], suffixes=("_18", "_50"))
not_augmented.to_csv(os.path.join(dir, "not_augmented.csv"))