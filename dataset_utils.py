"""
This module creates the DataFrames, import them from here.
"""
import itertools
import random
import os
from typing import Union
from PIL import Image
import numpy as np
import pandas as pd
from datasets import Dataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
from cuda import device

seed = 42  # random.randint(1, 100_000)
print(f"RANDOM SEED: {seed}")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

transform_to_tensor = transforms.ToTensor()
transform_to_pil = transforms.ToPILImage()


def to_tensor(image: Image, move_to_device=True) -> torch.Tensor:
    t = transform_to_tensor(image).unsqueeze(0)
    if move_to_device:
        return t.to(device)
    else:
        return t


def to_pil(tensor: torch.Tensor) -> Image:
    return transform_to_pil(torch.Tensor(tensor).squeeze(0))


def load_image(path, move_to_device=True):
    image = Image.open(path).convert("RGB")
    return to_tensor(image, move_to_device=move_to_device)


# Build the Pacs Dataframe
builder = {"labels": [], "image": [], "domain": []}

# iterate over all files, the labels can be obained from
# the directory structure.
dataset = "camelyon"
if dataset == "pacs":
    for domain in (
        "sketch",
        "photo",
        "cartoon",
        "art_painting",
    ):
        domain_path = os.path.join("../pacs-dataset/pacs_data/pacs_data", domain)
        for label in os.listdir(domain_path):
            label_path = os.path.join(domain_path, label)
            for filename in os.listdir(label_path):
                builder["labels"].append(label)
                builder["domain"].append(domain)
                path = os.path.join(label_path, filename)
                builder["image"].append(path)

elif dataset == "camelyon":
    df = pd.read_csv("camelyon17/data/camelyon17_v1.0/metadata.csv")
    total_images = 10000
    TUMOR, NO_TUMOR = 1, 0
    categories = [TUMOR, NO_TUMOR]
    nodes = [0, 1, 2, 3]
    combinations = list(itertools.product(categories, nodes))
    imgs_per_combination = total_images // len(combinations)
    selected_rows = []
    pre_path = "camelyon17/data/camelyon17_v1.0/patches"

    for category, node in combinations:
        tmp = df[(df["tumor"] == category) & (df["node"] == node)].head(imgs_per_combination)
        for index, row in tmp.iterrows():
            patient = row["patient"]
            x = row["x_coord"]
            y = row["y_coord"]
            num = row["Unnamed: 0"]
            img_path = os.path.join(
                pre_path,
                f"patient_{str(patient).zfill(3)}_node_{node}",
                f"patch_patient_{str(patient).zfill(3)}_node_{node}_x_{x}_y_{y}.png"
            )
            builder["labels"].append(category)
            builder["domain"].append(node)
            builder["image"].append(img_path)



pacs_df = pd.DataFrame(builder)
pacs_classes = pacs_df["labels"].unique().tolist()
pacs_df["labels"] = pacs_df["labels"].map(lambda x: pacs_classes.index(x))


all_datasets = {
    "pacs": {
        "df": pacs_df,
        "classes": pacs_classes,
        "domains": list(pacs_df["domain"].unique()),
        "shape": (3, 227, 227),
    }
}


def get_dataloader(
    df: pd.DataFrame, batch_size: int = 32
) -> Union[DataLoader, DataLoader]:

    # Load images and labels into tensors
    images = []
    labels = []
    for index, row in df.iterrows():
        images.append(
            load_image(row["image"], move_to_device=False)
        )  # Do not move to device yet
        labels.append(row["labels"])

    # Convert lists to tensors
    images_tensor = torch.cat(images).to(device)

    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    # Create Tensor Datasets
    dataset = TensorDataset(images_tensor, labels_tensor)

    # Create DataLoaders
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


def split_df(df: pd.DataFrame, test_size: float = 0.2):
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    train_size = 1 - test_size

    # Calculate the split index
    split_index = int(len(df) * train_size)

    # Split the DataFrame into training and testing sets
    train_df = df_shuffled[:split_index].reset_index(drop=True)
    test_df = df_shuffled[split_index:].reset_index(drop=True)
    return train_df, test_df


def create_dataset(df: pd.DataFrame, test_size: float = None) -> Dataset:
    dataset = Dataset.from_pandas(df)

    def load_image(example):
        image = Image.open(example["image"])
        return {"features": to_tensor(image)}

    dataset = dataset.map(load_image)

    if isinstance(test_size, float):
        dataset = dataset.train_test_split(test_size=test_size)
    dataset = dataset.remove_columns(("image", "domain"))
    return dataset


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
