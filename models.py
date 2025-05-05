import torch
import torch
from torch import nn
import torchvision
from dataset_utils import (
    pacs_classes,
)
from cuda import device


class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, len(pacs_classes))
        self.to(device)

    def forward(self, x):
        x = self.resnet18(x)
        x = nn.Softmax(dim=1)(x)
        return x
