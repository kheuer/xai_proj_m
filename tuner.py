"""this module optimizes the hyperparameters"""

import argparse
import os
import shutil
from datetime import datetime
import optuna
from torchvision.transforms import InterpolationMode
from utils import get_expected_input, split_df_into_loaders
from dataset_utils import split_domains, get_dataloader, split_df
from models import (
    get_resnet_18,
    get_resnet_50,
    all_datasets,
    calculate_val_loss,
)
from config import MAX_EPOCHS, PATIENCE, BATCH_SIZE, NUM_TRIALS, SAVE_FREQ

model_name = get_expected_input("Please choose a model:", ("ResNet18", "ResNet50"))
pretrained = {"Yes": True, "No": False}[
    get_expected_input("Use pre-trained weights? ", ("Yes", "No"))
]

dataset_name = get_expected_input(
    "Which dataset should be used? ", ("pacs", "camelyon")
)
dataset = all_datasets[dataset_name]

transformations = {"Yes": True, "No": False}[
    get_expected_input("Apply transformations during training? ", ("Yes", "No"))
]

# take a random sample to speed up training
_, df_sampled = split_df(dataset["df"], test_size=0.2)

train_loader, val_loader, test_loader, target_domain = split_df_into_loaders(df_sampled)

STUDY_NAME = f"STUDY_{dataset_name}_{model_name}_{target_domain}_pretrained_{pretrained}_transformations_{transformations}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"


def objective_simple(trial: optuna.trial.Trial):
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
        # disable tranformations
        "TRANSFORMATIONS_ORDER": [],
        "USE_AUGMIX": False,
        "USE_FOURIER": False,
        "USE_JIGSAW": False,
        "USE_DLOW": False,
    }

    if model_name == "ResNet18":
        model = get_resnet_18(pretrained=pretrained, dataset_name=dataset_name)
    else:
        model = get_resnet_50(pretrained=pretrained, dataset_name=dataset_name)

    loss = calculate_val_loss(
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        model=model,
        HYPERPARAMS=params,
        trial=trial,
    )

    if trial.number % SAVE_FREQ == 0:
        save_path = os.path.join("trials", f"{STUDY_NAME}.db")
        shutil.copy2(f"{STUDY_NAME}.db", save_path)

    return loss


def objective_transformations(trial: optuna.trial.Trial):
    params = {
        # LEARNING PARAMS
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
        # TRANSFORMATION PARAMS
        # Augmix params
        "USE_AUGMIX": trial.suggest_categorical("USE_AUGMIX", [True, False]),
        "SEVERITY": trial.suggest_int("SEVERITY", 1, 10),
        "MIXTURE_WIDTH": trial.suggest_int("MIXTURE_WIDTH", 1, 10),
        "CHAIN_DEPTH": trial.suggest_int("CHAIN_DEPTH", 1, 10),
        "ALPHA": trial.suggest_float("ALPHA", 0.0, 1.0),
        "ALL_OPS": trial.suggest_categorical("ALL_OPS", [True, False]),
        "INTERPOLATION": trial.suggest_categorical(
            "INTERPOLATION", ["NEAREST", "BILINEAR"]
        ),
        # Fourier params
        "USE_FOURIER": trial.suggest_categorical("USE_FOURIER", [True, False]),
        "SQUARE_SIZE": trial.suggest_int(
            "SQUARE_SIZE_SINGLE_SIDE", 2, dataset["shape"][-1]
        ),
        "ETA": trial.suggest_float("ETA", 0, 1),
        # Jigsaw params
        "USE_JIGSAW": trial.suggest_categorical("USE_JIGSAW", [True, False]),
        "MIN_GRID_SIZE": trial.suggest_int("MIN_GRID_SIZE", 2, 6),
        "MAX_GRID_SIZE": trial.suggest_int("MAX_GRID_SIZE", 6, 15),
        # Dlow params
        "USE_DLOW": False,
        # Order params
        "TRANSFORMATIONS_ORDER": trial.suggest_categorical(
            "TRANSFORMATIONS_ORDER",
            [
                "Augmix,Dlow,Fourier,Jigsaw",
                "Augmix,Fourier,Dlow,Jigsaw",
                "Fourier,Augmix,Dlow,Jigsaw",
                "Fourier,Dlow,Augmix,Jigsaw",
                "Dlow,Augmix,Fourier,Jigsaw",
                "Dlow,Fourier,Augmix,Jigsaw",
            ],
        ),
    }
    if dataset_name == "pacs":
        params["USE_DLOW"] = trial.suggest_categorical("USE_DLOW", [True, False])
        params["TARGET_DOMAIN"] = target_domain

    if model_name == "ResNet18":
        model = get_resnet_18(pretrained=pretrained, dataset_name=dataset_name)
    else:
        model = get_resnet_50(pretrained=pretrained, dataset_name=dataset_name)

    loss = calculate_val_loss(
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        model=model,
        HYPERPARAMS=params,
        trial=trial,
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
    os.makedirs("trials", exist_ok=True)
    storage_uri = f"sqlite:///trials/{STUDY_NAME}.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name=STUDY_NAME,
        storage=storage_uri,
        load_if_exists=True,
    )

    done_trials = len(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            or t.state == optuna.trial.TrialState.PRUNED
        ]
    )
    remaining_trails = max(0, NUM_TRIALS - done_trials)

    objective = {True: objective_transformations, False: objective_simple}[
        transformations
    ]

    study.optimize(
        objective,
        gc_after_trial=True,
        callbacks=[print_callback, write_callback],
        n_trials=NUM_TRIALS,
    )
