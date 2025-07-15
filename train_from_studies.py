import os.path
import json
import re

import optuna
import torch

from config import MAX_EPOCHS, PATIENCE, BATCH_SIZE
from dataset_utils import all_datasets
from models import get_resnet_18, get_resnet_50, calculate_val_loss
from utils import split_df_into_loaders


def start_training(target_domain: str, model_name: str, pretrained: bool, params):
    dataset_name = "pacs"
    dataset = all_datasets[dataset_name]

    train_loader, val_loader, test_loader, target_domain = split_df_into_loaders(
        dataset["df"], target_domain=target_domain
    )

    if model_name == "ResNet18":
        model = get_resnet_18(pretrained=pretrained, dataset_name=dataset_name)
    else:
        model = get_resnet_50(pretrained=pretrained, dataset_name=dataset_name)

    return calculate_val_loss(
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        model=model,
        HYPERPARAMS=params,
        return_best_weights=True,
    )


study_dir = os.path.join(os.environ.get("TMPDIR"), "studies")
studies = os.listdir(study_dir)
pattern = (
    r"STUDY_(ResNet\d+)_([0-3])_pretrained_(True|False)_transformations_(True|False)"
)

for study_file in studies:
    # == studyname

    study_name = study_file.split(".")[0]

    current_study_dir = os.path.join(study_dir, study_name)
    # training on this study has at least started
    if os.path.exists(current_study_dir):
        continue

    os.makedirs(current_study_dir)

    study = optuna.load_study(
        storage=f"sqlite:///{os.path.join(study_dir, study_file)}",
        study_name=study_name,
    )

    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)

    match = re.search(pattern, study_name)
    if match:
        model_name = match.group(1)
        target_domain = match.group(2)
        pretrained = match.group(3) == "True"
        transformations = match.group(4) == "True"

        print(f"start training study: {study_name}")
        for rank in range(3):

            params = sorted_trials[rank].params
            params["EPOCHS"] = MAX_EPOCHS
            params["PATIENCE"] = PATIENCE
            params["BATCH_SIZE"] = BATCH_SIZE
            if not transformations:
                params["TRANSFORMATIONS_ORDER"] = []
                params["USE_AUGMIX"] = []
                params["USE_FOURIER"] = []
                params["USE_JIGSAW"] = []
                params["USE_DLOW"] = []
            params["USE_DLOW"] = False
            print(params)
            loss, weights = start_training(
                target_domain=target_domain,
                model_name=model_name,
                pretrained=pretrained,
                params=params,
            )
            print(f"finished training for {rank}. best trial with loss: {loss}")
            weights_path = os.path.join(
                current_study_dir,
                f"studynr_{rank}_loss_{str(f'{loss:.2f}').replace('.', ',')}.pth",
            )
            torch.save(weights, weights_path)
