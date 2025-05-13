import os
from tqdm import tqdm
from itertools import product
from torch import save
from models import get_resnet_18, get_resnet_50, calculate_val_loss
from dataset_utils import all_datasets, seed
from utils import split_df_into_loaders


filename = "final_results.txt"

if os.path.exists(filename):
    response = input(
        f"File '{filename}' already exists. Do you want to delete it? (y/n): "
    ).lower()
    if response == "y":
        os.remove(filename)
    else:
        print("File was not deleted. Exiting.")
        exit()


with open(filename, "w") as f:
    f.write(f"RANDOM SEED: {seed}")


params = {
    "EPOCHS": 300,
    "PATIENCE": 50,
    "LEARNING_RATE": 0.001,
    "BETA_1": 0.9,
    "BETA_2": 0.999,
    "BATCH_SIZE": 32,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "CosineAnnealingLR",
    "MOMENTUM": 0.53,
    "DAMPENING": 0.0145,
    "WEIGHT_DECAY": 0.0,
}

dataset_name = "pacs"
dataset = all_datasets[dataset_name]


model_names = ("ResNet18", "ResNet50")
pretrained_options = (True, False)
target_domains = dataset["domains"]

for model_name, pretrained, target_domain in tqdm(
    product(model_names, pretrained_options, target_domains),
    total=len(model_names) * len(pretrained_options) * len(target_domains),
    desc="Running combinations",
):
    train_loader, val_loader, test_loader, target_domain = split_df_into_loaders(
        df=dataset["df"], target_domain=target_domain
    )

    losses = []
    for _ in range(4):
        if model_name == "ResNet18":
            model = get_resnet_18(pretrained=pretrained)
        elif model_name == "ResNet50":
            model = get_resnet_50(pretrained=pretrained)

        loss, weights = calculate_val_loss(
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            model=model,
            HYPERPARAMS=params,
            return_best_weights=True,
        )
        losses.append(loss)
        if losses[-1] == min(losses):
            save(weights, f"weights/{model_name}_{target_domain}_{pretrained}.pth")

    msg = f"{model_name} model with target_domain = {target_domain}, pretrained = {pretrained}: {losses}"
    # print(msg)
    with open(filename, "w") as f:
        f.write(msg)
