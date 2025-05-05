"""
This module creates the DataFrames, import them from here.
"""

import os
from typing import Union
from PIL import Image
import pandas as pd
from datasets import Dataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
from cuda import device

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
    image = Image.open(path)
    return to_tensor(image, move_to_device=move_to_device)


# Build the Pacs Dataframe
builder = {"labels": [], "image": [], "domain": []}

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
            builder["labels"].append(label)
            builder["domain"].append(domain)
            path = os.path.join(label_path, filename)
            builder["image"].append(path)


pacs_df = pd.DataFrame(builder)
pacs_classes = pacs_df["labels"].unique().tolist()
pacs_df["labels"] = pacs_df["labels"].map(lambda x: pacs_classes.index(x))


def get_dataloaders(
    df: pd.DataFrame, test_size: float = 0.2, batch_size: int = 32
) -> Union[DataLoader, DataLoader]:

    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    train_size = 1 - test_size

    # Calculate the split index
    split_index = int(len(df) * train_size)

    # Split the DataFrame into training and testing sets
    train_df = df_shuffled[:split_index].reset_index(drop=True)
    test_df = df_shuffled[split_index:].reset_index(drop=True)

    # train_df["image"] = train_df["image"].apply(load_image)
    # test_df["image"] = test_df["image"].apply(load_image)

    # Load images and labels into tensors
    train_images = []
    train_labels = []
    for index, row in train_df.iterrows():
        train_images.append(
            load_image(row["image"], move_to_device=False)
        )  # Do not move to device yet
        train_labels.append(row["labels"])

    test_images = []
    test_labels = []
    for index, row in test_df.iterrows():
        test_images.append(
            load_image(row["image"], move_to_device=False)
        )  # Do not move to device yet
        test_labels.append(row["labels"])

    # Convert lists to tensors
    train_images_tensor = torch.cat(
        train_images
    )  # Concatenate tensors along the first dimension
    test_images_tensor = torch.cat(test_images)

    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(device)

    # Create Tensor Datasets
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


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
