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
from config import (
    params_resnet_18_random_augmented_art_painting,
    params_resnet_18_random_augmented_cartoon,
    params_resnet_18_random_augmented_sketch,
    params_resnet_18_random_augmented_photo,
)


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

if not os.path.isdir("weights_resnet_18_experiment"):
    os.makedirs("weights_resnet_18_experiment")


with open(filename, "w") as f:
    f.write(f"RANDOM SEED: {seed}")


dataset_name = "pacs"
dataset = all_datasets[dataset_name]


target_domains = dataset["domains"]
pretrained = False
augmented = True


def log(msg):
    print(msg)
    with open(filename, "a") as f:
        f.write(msg)


N_RUNS = 4

for target_domain in tqdm(
    target_domains,
    desc="Running combinations",
):
    print("target-domain:", target_domain)
    train_loader, val_loader, test_loader, _ = split_df_into_loaders(
        df=dataset["df"], target_domain=target_domain
    )
    losses = []
    accuracies = []
    for i in range(N_RUNS):
        save_path = f"weights_resnet_18_experiment/{target_domain}_{pretrained}_{augmented}_{i}.pth"
        if os.path.exists(save_path):
            print("skipping", save_path)
            continue

        model = get_resnet_18(pretrained=pretrained)

        variable_name = f"params_resnet_18_{'pretrained' if pretrained else 'random'}"
        if augmented:
            variable_name += "_augmented"
        variable_name += f"_{target_domain}"

        params = deepcopy(globals().get(variable_name))
        params["TARGET_DOMAIN"] = target_domain

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

        save(weights, save_path)
        log(
            f"\nResNet18 model with target_domain = {target_domain}, pretrained = {pretrained}, augmented = {augmented}. Single accuracy: {accuracy}. Single loss: {loss}"
        )
    # manual garbage collection to avoid cuda OOM errors
    if globals().get("weights"):
        del weights
        del train_loader
        del val_loader
        del test_loader
        model.to("cpu")
        del model
    gc.collect()
    empty_cache()
    log(
        f"\nResNet18 model with target_domain = {target_domain}, pretrained = {pretrained}, augmented = {augmented}. Mean accuracy: {sum(accuracies)/N_RUNS}. Mean loss: {sum(losses)/N_RUNS}"
    )
