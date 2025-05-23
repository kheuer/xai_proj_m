"""this module optimizes the hyperparameters"""

import os
from datetime import datetime
import optuna
from utils import get_expected_input, split_df_into_loaders
from dataset_utils import split_domains, get_dataloader, split_df
from models import (
    get_resnet_18,
    get_resnet_50,
    all_datasets,
    calculate_val_loss,
)
from config import MAX_EPOCHS, PATIENCE, BATCH_SIZE

model_name = get_expected_input("Please choose a model:", ("ResNet18", "ResNet50"))
pretrained = {"Yes": True, "No": False}[
    get_expected_input("Use pre-trained weights? ", ("Yes", "No"))
]

# TODO: ask the user for input when we obtain another dataset
dataset_name = "pacs"
dataset = all_datasets[dataset_name]

# take a random sample to speed up training
_, df_sampled = split_df(dataset["df"], test_size=0.2)

train_loader, val_loader, test_loader, target_domain = split_df_into_loaders(df_sampled)

STUDY_NAME = f"STUDY_{model_name}_{target_domain}_{pretrained}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"


def objective(trial):
    """
    Objective function for Optuna to optimize hyperparameters.

    Args:
    - trial: optuna.trial.Trial object

    Returns:
    - Loss value to minimize
    """

    params = {
        "EPOCHS": MAX_EPOCHS,
        "PATIENCE": PATIENCE,
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 0.000001, 0.01),
        "BETAS": (
            trial.suggest_categorical("BETA_1", [0.8, 0.9, 0.95]),
            trial.suggest_categorical("BETA_2", [0.99, 0.999, 0.9999]),
        ),
        "WEIGHT_DECAY": trial.suggest_float("WEIGHT_DECAY", 0.0, 0.1),
        "OPTIMIZER": trial.suggest_categorical("OPTIMIZER", ["AdamW", "SGD"]),
        "SCHEDULER": trial.suggest_categorical(
            "SCHEDULER",
            ["CosineAnnealingLR", "ReduceLROnPlateau", "LinearLR", "StepLR", "None"],
        ),
        "MOMENTUM": trial.suggest_float("MOMENTUM", 0.5, 0.9),
        "DAMPENING": trial.suggest_float("DAMPENING", 0, 0.2),
        "GAMMA": trial.suggest_float("GAMMA", 0.1, 0.9),
        "STEP_SIZE": trial.suggest_int("STEP_SIZE", 5, 50),
    }

    if model_name == "ResNet18":
        model = get_resnet_18(pretrained=pretrained)
    else:
        model = get_resnet_50(pretrained=pretrained)

    loss = calculate_val_loss(
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        model=model,
        HYPERPARAMS=params,
    )

    return loss


def print_callback(study, trial):
    """Print trial information to the console"""
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(
        f"     Best value: {study.best_trial.value}, Best params: {study.best_trial.params}"
    )
    print("#####################################################")


def write_callback(study, trial):
    """Write trial information to a file"""
    # Directory where the log file will be stored
    log_directory = "optuna_logs"
    # Create the directory if it does not exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Log file path
    log_file_path = f"{log_directory}/{STUDY_NAME}.txt"

    # Open the file in append mode ('a'), which creates the file if it does not exist
    with open(log_file_path, "a") as file:
        file.write(f"Current value: {trial.value}, Current params: {trial.params}\n")
        file.write(
            f"     Best value: {study.best_trial.value}, Best params: {study.best_trial.params}\n\n"
        )
        file.write("#####################################################\n")


if __name__ == "__main__":
    # Optimize the study using objective function
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name=STUDY_NAME,
    )
    study.optimize(
        objective, gc_after_trial=True, callbacks=[print_callback, write_callback]
    )
