import os
from tqdm import tqdm
import pandas as pd
from torch import load, nn
from models import get_resnet_18, get_resnet_50, _do_eval, get_criterion
from transformers.transformation_utils import get_transform_pipeline
from dataset_utils import all_datasets, split_df
from utils import split_df_into_loaders
from cuda import device


dataset = all_datasets["pacs"]

builder = {
    "taget_domain": [],
    "architecture": [],
    "pretrained": [],
    "test_loss": [],
    "test_accuracy": [],
    "i": [],
}

for filename in tqdm(os.listdir("weights")):
    architecture, target_domain, pretrained, augmented, i = filename.replace(
        "art_painting", "art-painting"
    ).split("_")
    i = i.removesuffix(".pth")
    if target_domain == "art-painting":
        target_domain = "art_painting"
    pretrained = pretrained.startswith("True")
    augmented = augmented.startswith("True")
    if architecture == "ResNet18":
        model = get_resnet_18(pretrained=False)
    else:
        model = get_resnet_50(pretrained=False)

    state_dict = load(os.path.join("weights", filename), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    train_loader, _, test_loader, _ = split_df_into_loaders(
        df=dataset["df"], target_domain=target_domain
    )

    transformation_pipeline = get_transform_pipeline(
        params={
            "TRANSFORMATIONS_ORDER": "",
        }
    )

    builder["taget_domain"].append(target_domain)
    builder["architecture"].append(architecture)
    builder["pretrained"].append(pretrained)
    builder["i"].append(i)

    loss, accuracy = _do_eval(
        model=model,
        criterion=get_criterion(train_loader),
        dataloader=test_loader,
        transformation_pipeline=transformation_pipeline,
    )
    builder["test_loss"].append(loss)
    builder["test_accuracy"].append(accuracy)

    del model

df = pd.DataFrame(builder)
df.to_csv("results.csv")
print(df)
