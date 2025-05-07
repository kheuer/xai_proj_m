from typing import Union
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


MAX_EPOCHS = 100
PATIENCE = 20


def get_resnet_18(
    weights: Union[str, None] = "ResNet18_Weights.DEFAULT",
) -> torchvision.models.resnet18:
    resnet18 = torchvision.models.resnet18(weights=weights)
    resnet18.fc = nn.Linear(
        resnet18.fc.in_features,
        len(all_datasets["pacs"]["classes"]),
    )
    resnet18.to(device)
    return resnet18


def get_resnet_50(
    weights: Union[str, None] = "ResNet50_Weights.DEFAULT",
) -> torchvision.models.resnet50:
    resnet50 = torchvision.models.resnet50(weights=weights)
    resnet50.fc = nn.Linear(
        resnet50.fc.in_features,
        len(all_datasets["pacs"]["classes"]),
    )
    resnet50.to(device)
    return resnet50


class NoScheduler:
    def __init__(self):
        pass

    def step(self):
        pass


def calculate_val_loss(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: torchvision.models,
    HYPERPARAMS: dict,
) -> float:

    match HYPERPARAMS["OPTIMIZER"]:
        case "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=HYPERPARAMS["LEARNING_RATE"],
                betas=HYPERPARAMS["BETAS"],
                # weight_decay=HYPERPARAMS["WEIGHT_DECAY"],
                maximize=False,
            )
        case "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=HYPERPARAMS["LEARNING_RATE"],
                momentum=HYPERPARAMS["MOMENTUM"],
                dampening=HYPERPARAMS["DAMPENING"],
                # weight_decay=HYPERPARAMS["WEIGHT_DECAY"],
                maximize=False,
            )

    match HYPERPARAMS["SCHEDULER"]:
        case "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, MAX_EPOCHS, eta_min=0.0, last_epoch=-1
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
                patience=PATIENCE,
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

    y = train_loader.dataset.tensors[1].cpu().tolist()
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    weights = torch.Tensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    best_loss = float("inf")
    epochs_without_improvement = 0
    best_model_weights = None

    # Lists to store training and validation loss for plotting
    train_losses = []
    val_losses = []

    # Training Loop
    for epoch in range(HYPERPARAMS["EPOCHS"]):
        model.train(True)
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        # Validate the model
        model.eval()  # Switch to evaluation mode
        validation_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(validation_loss / len(test_loader))

        clear_output(wait=True)
        plot_loss(train_losses, val_losses)

        # Early Stopping Check
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
            best_model_weights = model.state_dict()  # Save the best model
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= HYPERPARAMS["PATIENCE"]:
                break

    # Load the best model weights
    # MODEL.load_state_dict(best_model_weights)
    return best_loss / len(test_loader)
