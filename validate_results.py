import os
import gc
import re

import numpy as np
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

dataset = all_datasets["pacs"]

builder = {
    "target_domain": [],
    "test_domain": [],
    "architecture": [],
    "pretrained": [],
    "augmented": [],
    "test_loss": [],
    "test_accuracy": [],
    "auc-score": [],
    "i": [],
    "tp": [],
    "tn": [],
    "fp": [],
    "fn": [],

}



def get_config(filename, dataset="pacs"):
    if dataset == "pacs":
        architecture, target_domain, pretrained, augmented, i = filename.replace(
            "art_painting", "art-painting"
        ).split("_")
        i = i.removesuffix(".pth")
        if target_domain == "art-painting":
            target_domain = "art_painting"
        pretrained = pretrained.startswith("True")
        augmented = augmented.startswith("True")
    elif dataset == "camelyon":
        pattern = r"STUDY_(ResNet\d+)_([0-3])_pretrained_(True|False)_transformations_(True|False)_[\d\-_:]+_([0-3])\.pth"
        match = re.search(pattern, filename)
        if match:
            architecture = match.group(1)
            target_domain = match.group(2)
            pretrained = match.group(3) == "True"
            augmented = match.group(4) == "True"
            i = match.group(5)
    else:
        raise NotImplementedError("This dataset is not supported yet.")

    return architecture, target_domain, pretrained, augmented, i

def run_evaluation(architecture, target_domain, filename):
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
    loss, accuracy, probabilities, labels = _do_eval(
        model=model,
        criterion=get_criterion(train_loader),
        dataloader=test_loader,
        transformation_pipeline=transformation_pipeline,
    )

    model.to("cpu")
    del model
    del state_dict
    del train_loader
    del _
    del test_loader
    del transformation_pipeline

    return loss, accuracy, probabilities, labels

def main():
    for filename in tqdm(os.listdir("weights")):
        architecture, target_domain, pretrained, augmented, i = get_config(filename, "camelyon")
        if not pretrained or augmented:
            continue
        for test_domain in range(1):

            loss, accuracy, probabilities, labels, = run_evaluation(architecture, target_domain, filename)
            print(np.mean(labels))
            auc_value = roc_auc_score(labels, probabilities[:, 1])
            predicted = np.argmax(probabilities, axis=1)

            #model learned to classify tumor as 0 and no tumor as 1
            #swap values later
            tn, fp, fn, tp = confusion_matrix(labels, predicted).ravel()

            builder["target_domain"].append(target_domain)
            builder["test_domain"].append(test_domain)
            builder["architecture"].append(architecture)
            builder["pretrained"].append(pretrained)
            builder["augmented"].append(augmented)
            builder["i"].append(i)
            builder["auc-score"].append(auc_value)
            builder["test_loss"].append(loss)
            builder["test_accuracy"].append(accuracy)
            builder["tp"].append(tn)
            builder["tn"].append(tp)
            builder["fp"].append(fn)
            builder["fn"].append(fp)




    gc.collect()
    empty_cache()
    df = pd.DataFrame(builder)
    df.to_csv("results.csv")
    print(df)
    return df


if __name__ == "__main__":
    main()
