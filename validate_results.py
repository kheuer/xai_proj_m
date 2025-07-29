import argparse
import os
import gc
from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm
import pandas as pd
from torch import load, nn
from models import get_resnet_18, get_resnet_50, _do_eval, get_criterion
from transformers.transformation_utils import get_transform_pipeline
from dataset_utils import all_datasets, split_df
from utils import split_df_into_loaders
from cuda import device
from torch.cuda import empty_cache
from train_for_comparision_study import params_list
from config import DEFAULT_PARAMS

dataset_name = "camelyon_unbalanced"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    required=False,
    choices=["pacs", "camelyon", "camelyon_unbalanced"],
)
args = parser.parse_args()

if args.dataset_name is not None:
    dataset_name = args.dataset_name
print(dataset_name)
dataset = all_datasets[dataset_name]
label_mean = dataset["df"]["labels"].mean()
builder = {
    "dataset_name": [],
    "taget_domain": [],
    "architecture": [],
    "augmentations": [],
    "test_loss": [],
    "test_accuracy": [],
    "i": [],
    "tp": [],
    "tn": [],
    "fp": [],
    "fn": [],
    "auc-score": [],
}


def main():
    for filename in tqdm(os.listdir("weights")):
        trained_dataset_name, architecture, augmentations, target_domain, i = (
            filename.replace("art_painting", "art-painting").split("_")
        )
        if trained_dataset_name not in dataset_name:
            # make sure the dataset matches but weights trained on camelyon_unbalanced should still match camelyon
            continue

        i = int(i.removesuffix(".pth"))
        if target_domain == "art-painting":
            target_domain = "art_painting"
        if architecture == "ResNet18":
            model = get_resnet_18(pretrained=False, dataset_name=trained_dataset_name)
        elif architecture == "ResNet50":
            model = get_resnet_50(pretrained=False, dataset_name=trained_dataset_name)

        state_dict = load(os.path.join("weights", filename), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        train_loader, _, test_loader, _ = split_df_into_loaders(
            df=dataset["df"], target_domain=target_domain
        )

        found = False
        for desc, params_candidate in params_list:
            if desc == augmentations:
                params = deepcopy(DEFAULT_PARAMS)
                params.update(params_candidate)
                params["TARGET_DOMAIN"] = target_domain
                found = True
                break
        assert found
        print(filename)
        transformation_pipeline = get_transform_pipeline(params=params)

        builder["dataset_name"].append(dataset_name)
        builder["architecture"].append(architecture)
        builder["augmentations"].append(augmentations)
        builder["taget_domain"].append(target_domain)
        builder["i"].append(i)

        loss, accuracy, all_logits, all_labels = _do_eval(
            model=model,
            criterion=get_criterion(train_loader),
            dataloader=test_loader,
            transformation_pipeline=transformation_pipeline,
        )
        if "camelyon" in dataset_name:
            probabilities = torch.softmax(torch.tensor(all_logits), dim=1)
            # model learned to classify tumor as 0 and no tumor as 1
            # swap labels for tumor = 1 and no tumor = 0
            inverted_labels = np.array([1 - y for y in all_labels])
            auc_value = roc_auc_score(inverted_labels, probabilities[:, 0])

            predicted = np.argmax(probabilities.numpy(), axis=1)
            predicted_inverted = np.array([1 - pred for pred in predicted])
            # model learned to classify tumor as 0 and no tumor as 1
            # swap values
            tn, fp, fn, tp = confusion_matrix(
                inverted_labels, predicted_inverted
            ).ravel()
            builder["tp"].append(tp)
            builder["tn"].append(tn)
            builder["fp"].append(fp)
            builder["fn"].append(fn)
            builder["auc-score"].append(auc_value)
        else:
            builder["tp"].append(None)
            builder["tn"].append(None)
            builder["fp"].append(None)
            builder["fn"].append(None)
            builder["auc-score"].append(None)

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
    df["pretrained"] = False
    df.to_csv(f"results_{dataset_name}.csv")
    print(df)
    return df


if __name__ == "__main__":
    main()
