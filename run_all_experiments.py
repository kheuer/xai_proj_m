import os
import gc
from copy import deepcopy
from tqdm import tqdm
from itertools import product
from torch import save
from torch.cuda import empty_cache
from models import (
    get_resnet_18,
    get_resnet_50,
    calculate_val_loss,
    _do_eval,
    get_criterion,
)
from dataset_utils import all_datasets, seed
from utils import split_df_into_loaders
from transformers.transformation_utils import get_transform_pipeline


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

if not os.path.isdir("weights"):
    os.makedirs("weights")


with open(filename, "w") as f:
    f.write(f"RANDOM SEED: {seed}")


params_resnet_18_random = {
    "EPOCHS": 300,
    "PATIENCE": 50,
    "LEARNING_RATE": 0.001,
    "BETA_1": 0.9,
    "BETA_2": 0.999,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "CosineAnnealingLR",
    "MOMENTUM": 0.53,
    "DAMPENING": 0.0145,
    "WEIGHT_DECAY": 0.0,
    # augmentation params
    "TRANSFORMATIONS_ORDER": [],
}

params_resnet_18_pretrained = {
    "EPOCHS": 300,
    "PATIENCE": 50,
    "LEARNING_RATE": 0.0037,
    "BETA_1": 0.95,
    "BETA_2": 0.9999,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "CosineAnnealingLR",
    "MOMENTUM": 0.736,
    "DAMPENING": 0.0327,
    "WEIGHT_DECAY": 0.0068,
    # augmentation params
    "TRANSFORMATIONS_ORDER": [],
}


params_resnet_50_pretrained = {
    "EPOCHS": 300,
    "PATIENCE": 50,
    "LEARNING_RATE": 0.0059,
    "BETA_1": 0.8,
    "BETA_2": 0.99,
    "WEIGHT_DECAY": 0.0068,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "StepLR",
    "MOMENTUM": 0.6991,
    "DAMPENING": 0.0713,
    "GAMMA": 0.40,
    "STEP_SIZE": 29,
    # augmentation params
    "TRANSFORMATIONS_ORDER": [],
}

params_resnet_50_random = {
    "EPOCHS": 300,
    "PATIENCE": 50,
    "LEARNING_RATE": 0.0085,
    "BETA_1": 0.95,
    "BETA_2": 0.999,
    "WEIGHT_DECAY": 0.08,
    "OPTIMIZER": "SGD",
    "SCHEDULER": "LinearLR",
    "MOMENTUM": 0.81,
    "DAMPENING": 0.17,
    "GAMMA": 0.17,
    "STEP_SIZE": 47,
    # augmentation params
    "TRANSFORMATIONS_ORDER": [],
}


augmented = bool(params_resnet_18_pretrained["TRANSFORMATIONS_ORDER"])
assert (
    bool(params_resnet_18_pretrained["TRANSFORMATIONS_ORDER"])
    == bool(params_resnet_18_random["TRANSFORMATIONS_ORDER"])
    == bool(params_resnet_50_pretrained["TRANSFORMATIONS_ORDER"])
    == bool(params_resnet_50_random["TRANSFORMATIONS_ORDER"])
)

dataset_name = "pacs"
dataset = all_datasets[dataset_name]


model_names = ("ResNet18", "ResNet50")
pretrained_options = (True, False)
target_domains = dataset["domains"]


def log(msg):
    print(msg)
    with open(filename, "a") as f:
        f.write(msg)


N_RUNS = 4

for model_name, pretrained, target_domain in tqdm(
    product(model_names, pretrained_options, target_domains),
    total=len(model_names) * len(pretrained_options) * len(target_domains),
    desc="Running combinations",
):
    train_loader, val_loader, test_loader, _ = split_df_into_loaders(
        df=dataset["df"], target_domain=target_domain
    )

    losses = []
    accuracies = []
    for i in range(N_RUNS):
        if model_name == "ResNet18":
            model = get_resnet_18(pretrained=pretrained)
            if pretrained:
                params = deepcopy(params_resnet_18_pretrained)
            else:
                params = deepcopy(params_resnet_18_random)
        elif model_name == "ResNet50":
            model = get_resnet_50(pretrained=pretrained)
            if pretrained:
                params = deepcopy(params_resnet_50_pretrained)
            else:
                params = deepcopy(params_resnet_50_random)

        _, weights = calculate_val_loss(
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            model=model,
            HYPERPARAMS=params,
            return_best_weights=True,
        )

        loss, accuracy = _do_eval(
            model=model,
            criterion=get_criterion(train_loader),
            dataloader=test_loader,
            transformation_pipeline=get_transform_pipeline(params),
        )
        accuracies.append(accuracy)
        losses.append(loss)

        save(
            weights,
            f"weights/{model_name}_{target_domain}_{pretrained}_{augmented}_{i}.pth",
        )
        log(
            f"\n{model_name} model with target_domain = {target_domain}, pretrained = {pretrained}, augmented = {augmented}. Single accuracy: {accuracy}. Single loss: {loss}"
        )
    # manual garbage collection to avoid cuda OOM errors
    del weights
    del train_loader
    del val_loader
    del test_loader
    model.to("cpu")
    del model
    gc.collect()
    empty_cache()
    log(
        f"\n{model_name} model with target_domain = {target_domain}, pretrained = {pretrained}, augmented = {augmented}. Mean accuracy: {sum(accuracies)/N_RUNS}. Mean loss: {sum(losses)/N_RUNS}"
    )

# create the csv file with results
import validate_results
