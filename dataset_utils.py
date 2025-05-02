"""
This module creates the DataFrames, import them from here.
"""

import os
from typing import Union
from PIL import Image
import pandas as pd


# Build the Pacs Dataframe
builder = {"label": [], "image": [], "domain": []}

# iterate over all files, the labels can be obained from
# the directory structure.
for domain in (
    "sketch",
    "photo",
    "cartoon",
    "art_painting",
):
    domain_path = os.path.join("pacs-dataset/pacs_data/pacs_data", domain)
    for label in os.listdir(domain_path):
        label_path = os.path.join(domain_path, label)
        for filename in os.listdir(label_path):
            builder["label"].append(label)
            builder["domain"].append(domain)
            builder["image"].append(Image.open(os.path.join(label_path, filename)))


pacs_df = pd.DataFrame(builder)


def split_domains(
    df: pd.DataFrame, target_domain: str
) -> Union[pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame into two subsets based on a target domain.

    This function takes a DataFrame containing domain information and splits it into
    two separate DataFrames: one containing rows for the specified target domain
    and another containing all other domains.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'domain' column.
        target_domain (str): The domain to be used as the target for splitting.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - The first DataFrame contains all rows with the source domain.
            - The second DataFrame contains only the rows with the target domain.

    Raises:
        AssertionError: If the target_domain is not found in the unique domains of the DataFrame.
    """
    assert target_domain in df["domain"].unique()
    source_domains = list(df["domain"].unique())
    source_domains.remove(target_domain)

    df_target = df[df["domain"] == target_domain]
    df_source = df[df["domain"] != target_domain]
    return df_source, df_target
