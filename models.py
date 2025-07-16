import copy
import os
from typing import Union, Tuple, Callable

import optuna
from IPython.display import clear_output
import torch
from torch import nn
import torchvision
from dataset_utils import all_datasets
from cuda import device
from utils import plot_loss
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
import numpy as np
from transformers.transformation_utils import get_transform_pipeline


default_caching_behaviour = True


def get_resnet_18(pretrained: bool, dataset_name: str) -> torchvision.models.resnet18:
    if default_caching_behaviour:
        resnet18 = torchvision.models.resnet18(
            weights="IMAGENET1K_V1" if pretrained else None
        )
    else:
        resnet18 = torchvision.models.resnet18(weights=None)
        if pretrained:
            resnet18.load_state_dict(
                torch.load(os.path.join(os.environ.get("TMPDIR"), "resnet18.pth"))
            )

    resnet18.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(
            resnet18.fc.in_features,
            len(all_datasets[dataset_name]["classes"]),
        ),
    )
    resnet18.to(device)
    return resnet18


def get_resnet_50(pretrained: bool, dataset_name: str) -> torchvision.models.resnet50:
    resnet50 = torchvision.models.resnet50(
        weights="IMAGENET1K_V1" if pretrained else None
    )
    resnet50.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(
            resnet50.fc.in_features,
            len(all_datasets[dataset_name]["classes"]),
        ),
    )
    resnet50.to(device)
    return resnet50


class NoScheduler:
    def __init__(self):
        pass

    def step(self):
        pass


def get_criterion(train_loader: DataLoader) -> nn.CrossEntropyLoss:
    # create the loss function and correct for inballances in the training dataset by using weights
    y = train_loader.dataset.tensors[1].cpu().tolist()
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    weights = torch.Tensor(weights).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def calculate_val_loss(
    train_loader: DataLoader,
    test_loader: DataLoader,
    val_loader: DataLoader,
    model: torchvision.models,
    HYPERPARAMS: dict,
    trial: optuna.trial.Trial = None,
) -> Union[float, Tuple[float, torch.Tensor]]:

    if "BETAS" in HYPERPARAMS:
        betas = HYPERPARAMS["BETAS"]
    elif "BETA_1" in HYPERPARAMS and "BETA_2" in HYPERPARAMS:
        betas = (HYPERPARAMS["BETA_1"], HYPERPARAMS["BETA_2"])
    else:
        betas = None

    match HYPERPARAMS["OPTIMIZER"]:
        case "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=HYPERPARAMS["LEARNING_RATE"],
                betas=betas,
                weight_decay=HYPERPARAMS["WEIGHT_DECAY"],
                maximize=False,
            )
        case "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=HYPERPARAMS["LEARNING_RATE"],
                momentum=HYPERPARAMS["MOMENTUM"],
                dampening=HYPERPARAMS["DAMPENING"],
                weight_decay=HYPERPARAMS["WEIGHT_DECAY"],
                maximize=False,
            )

    match HYPERPARAMS["SCHEDULER"]:
        case "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, HYPERPARAMS["EPOCHS"], eta_min=0.0, last_epoch=-1
            )
        case "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                HYPERPARAMS["STEP_SIZE"],
                gamma=HYPERPARAMS["GAMMA"],
                last_epoch=-1,
            )
        case "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=HYPERPARAMS["GAMMA"],
                patience=HYPERPARAMS["PATIENCE"],
                threshold=0.0001,  # default
                threshold_mode="rel",
                cooldown=0,  # default
                min_lr=0,  # default
            )
        case "LinearLR":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1 / 3,  # default
                end_factor=1.0,  # default
                total_iters=8,
                last_epoch=-1,  # default
            )
        case "None":
            scheduler = NoScheduler()

    transformation_pipeline = get_transform_pipeline(params=HYPERPARAMS)

    criterion = get_criterion(train_loader)

    best_loss = float("inf")
    epochs_without_improvement = 0
    best_model_weights = None

    train_losses = []
    val_losses = []
    test_losses = []
    test_accuracies = []

    # Training Loop
    for epoch in range(HYPERPARAMS["EPOCHS"]):

        # Train the model
        train_losses.append(
            _do_train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                dataloader=train_loader,
                transformation_pipeline=transformation_pipeline,
            )
        )

        # Validate the model
        val_avg_loss, _, _, _ = _do_eval(
            model=model,
            criterion=criterion,
            dataloader=val_loader,
            transformation_pipeline=transformation_pipeline,
        )

        val_losses.append(val_avg_loss)

        test_avg_loss, test_avg_accuracy, _, _ = _do_eval(
            model=model,
            criterion=criterion,
            dataloader=test_loader,
            transformation_pipeline=transformation_pipeline,
        )

        test_losses.append(test_avg_loss)
        test_accuracies.append(test_avg_accuracy)

        match HYPERPARAMS["SCHEDULER"]:
            case "ReduceLROnPlateau":
                scheduler.step(val_losses[-1])
            case _:
                scheduler.step()

        # clear cell in case this is run in a jupyter notebook
        clear_output(wait=True)
        plot_loss(train_losses, val_losses, test_losses)
        # Early Stopping Check
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            epochs_without_improvement = 0
            # Save the best model
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= HYPERPARAMS["PATIENCE"]:
                break

        if trial:
            trial.report(val_losses[-1], step=epoch)

    # Load the best model weights
    # model.load_state_dict(best_model_weights) # saved from variable
    best_index = val_losses.index(best_loss)
    test_loss = test_losses[best_index]
    corresponding_accuracy = test_accuracies[best_index]

    return test_loss, corresponding_accuracy, best_model_weights


def _do_train(
    model: torchvision.models,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim,
    dataloader: torch.utils.data.DataLoader,
    transformation_pipeline: Callable,
) -> float:
    model.train(True)
    running_loss = 0.0
    for features, labels in dataloader:
        optimizer.zero_grad()

        features_transformed = transformation_pipeline(
            features
        )  # the transformation makes no changes if specified such
        outputs = model(features_transformed)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def _do_eval(
    model: torchvision.models,
    criterion: torch.nn.CrossEntropyLoss,
    dataloader: torch.utils.data.DataLoader,
    transformation_pipeline: Callable,
) -> tuple[float, float, np.array, np.array]:
    model.eval()  # Switch to evaluation mode
    validation_loss = 0.0
    correct = 0
    total = 0
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features_transformed = transformation_pipeline(
                features
            )  # the transformation makes no changes if specified such
            outputs = model(features_transformed)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            all_labels.append(np.array(labels.detach().cpu()))
            all_probabilities.append(np.array(outputs.detach().cpu()))

            # Calculate predictions and compare to labels
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = validation_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return (
        avg_loss,
        accuracy,
        np.concatenate(all_probabilities, axis=0),
        np.concatenate(all_labels, axis=0),
    )
