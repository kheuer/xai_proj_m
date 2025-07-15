import os
import gc
from tqdm import tqdm
import pandas as pd
from torch import load, nn
from models import get_resnet_18, get_resnet_50, _do_eval, get_criterion
from transformers.transformation_utils import get_transform_pipeline
from dataset_utils import all_datasets, split_df
from utils import split_df_into_loaders
from cuda import device
from torch.cuda import empty_cache

dataset_name = "pacs"
dataset = all_datasets[dataset_name]

builder = {
    "taget_domain": [],
    "architecture": [],
    "pretrained": [],
    "augmented": [],
    "test_loss": [],
    "test_accuracy": [],
    "i": [],
}


def main():
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
            model = get_resnet_18(pretrained=False, dataset_name=dataset_name)
        else:
            model = get_resnet_50(pretrained=False, dataset_name=dataset_name)

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
        builder["augmented"].append(augmented)
        builder["i"].append(i)

        loss, accuracy = _do_eval(
            model=model,
            criterion=get_criterion(train_loader),
            dataloader=test_loader,
            transformation_pipeline=transformation_pipeline,
        )
        builder["test_loss"].append(loss)
        builder["test_accuracy"].append(accuracy)

        model.to("cpu")
        del model
        del state_dict
        del train_loader
        del _
        del test_loader
        del transformation_pipeline
    gc.collect()
    empty_cache()
    df = pd.DataFrame(builder)
    df.to_csv("results.csv")
    print(df)
    return df


if __name__ == "__main__":
    main()
