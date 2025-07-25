import os
import json
import re
import gc
from copy import deepcopy
import argparse
from itertools import product
import optuna
from tqdm import tqdm
import pandas as pd
import torch
from config import MAX_EPOCHS, PATIENCE, BATCH_SIZE, DEFAULT_PARAMS
from dataset_utils import all_datasets
from models import get_resnet_18, get_resnet_50, calculate_val_loss
from utils import split_df_into_loaders

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name", required=False, type=str, choices=["pacs", "camelyon", "camelyon_unbalanced"]
)

args = parser.parse_args()
dataset_name = args.dataset_name


def start_training(target_domain: str, model_name: str, pretrained: bool, params):
    dataset = all_datasets[dataset_name]

    train_loader, val_loader, test_loader, target_domain = split_df_into_loaders(
        dataset["df"], target_domain=target_domain
    )

    if model_name == "ResNet18":
        model = get_resnet_18(pretrained=pretrained, dataset_name=dataset_name)
    elif model_name == "ResNet50":
        model = get_resnet_50(pretrained=pretrained, dataset_name=dataset_name)

    return calculate_val_loss(
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        model=model,
        HYPERPARAMS=params,
    )


model_names = ["ResNet18", "ResNet50"]
if dataset_name == "pacs":
    target_domains = ["art_painting", "cartoon", "photo", "sketch"]
elif dataset_name == "camelyon" or dataset_name == "camelyon_unbalanced":
    target_domains = ["0", "1", "2", "3"]
else:
    raise ValueError(f"invalid dataset_name: {dataset_name}")

params_list = [
    ("No Augmentations", {}),
    (
        "Augmix",
        {
            "USE_AUGMIX": True,
            "SEVERITY": 3,
            "MIXTURE_WIDTH": 3,
            "CHAIN_DEPTH": -1,
            "ALPHA": 0.73,
            "ALL_OPS": True,
            "INTERPOLATION": "BILINEAR",
        },
    ),
    (
        "Fourier",
        {
            "USE_FOURIER": True,
            "SQUARE_SIZE": 72,
            "ETA": 0.4767002143366463,
        },
    ),
    (
        "Jigsaw",
        {"USE_JIGSAW": True, "MIN_GRID_SIZE": 2, "MAX_GRID_SIZE": 5},
    ),
    (
        "Dlow",
        {"USE_DLOW": True, "target_domain": None},
    ),
    (
        "Augmix and Fourier",
        {
            # Augmix
            "USE_AUGMIX": True,
            "SEVERITY": 3,
            "MIXTURE_WIDTH": 3,
            "CHAIN_DEPTH": -1,
            "ALPHA": 0.73,
            "ALL_OPS": True,
            "INTERPOLATION": "BILINEAR",
            # Fourier
            "USE_FOURIER": True,
            "SQUARE_SIZE": 72,
            "ETA": 0.4767002143366463,
        },
    ),
]

if dataset_name == "camelyon":
    params_list = [x for x in params_list if x[0] != "Dlow"]

indices = tuple(range(4))

os.makedirs("weights", exist_ok=True)

if __name__ == "__main__":
    for i, model_name, (augmentation_desc, augmentation_params), target_domain in tqdm(
        product(indices, model_names, params_list, target_domains),
        total=(
            len(indices) * len(model_names) * len(params_list) * len(target_domains)
        ),
        desc="Running combinations",
    ):
        SAVE_PATH = (
            f"{dataset_name}_{model_name}_{augmentation_desc}_{target_domain}_{i}.pth"
        )
        if SAVE_PATH in os.listdir("weights"):
            print("skip", SAVE_PATH)
            continue
        print(SAVE_PATH)

        params = deepcopy(DEFAULT_PARAMS)
        params.update(augmentation_params)
        params["TARGET_DOMAIN"] = target_domain

        loss, accuracy, weights = start_training(
            target_domain=target_domain,
            model_name=model_name,
            pretrained=False,
            params=params,
        )
        torch.save(weights, os.path.join("weights", SAVE_PATH))
        print("loss", loss, "accuracy", accuracy)

        del weights
        gc.collect()
        torch.cuda.empty_cache()

    print("done")
