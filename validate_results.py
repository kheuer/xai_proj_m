import os
from tqdm import tqdm
import pandas as pd
from torch import load, nn
from models import get_resnet_18, get_resnet_50, _do_eval, get_criterion
from dataset_utils import all_datasets
from utils import split_df_into_loaders
from cuda import device

dataset = all_datasets["pacs"]

builder = {"taget_domain": [], "architecture": [], "pretrained": [], "test_loss": []}

for filename in tqdm(os.listdir("weights")):
    architecture, target_domain, remainder = filename.replace(
        "art_painting", "art-painting"
    ).split("_")
    if target_domain == "art-painting":
        target_domain = "art_painting"
    pretrained = remainder.startswith("True")
    if architecture == "ResNet18":
        model = get_resnet_18(pretrained=False)
    else:
        model = get_resnet_50(pretrained=False)

    state_dict = load(os.path.join("weights", filename), map_location=device)
    model.load_state_dict(state_dict)

    train_loader, _, test_loader, _ = split_df_into_loaders(
        df=dataset["df"], target_domain=target_domain
    )

    builder["taget_domain"].append(target_domain)
    builder["architecture"].append(architecture)
    builder["pretrained"].append(pretrained)
    builder["test_loss"].append(
        _do_eval(
            model=model, criterion=get_criterion(train_loader), dataloader=test_loader
        )
    )

df = pd.DataFrame(builder)
df.to_csv("results.csv")
print(df)
