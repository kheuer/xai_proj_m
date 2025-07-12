import os
import gc
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
from torch import load, nn
from models import get_resnet_18, get_resnet_50, _do_eval, get_criterion
from transformers.transformation_utils import get_transform_pipeline
from dataset_utils import all_datasets, split_df
from utils import split_df_into_loaders
from cuda import device
from torch.cuda import empty_cache
from config import (
    params_resnet_18_random_augmented_art_painting,
    params_resnet_18_random_augmented_cartoon,
    params_resnet_18_random_augmented_sketch,
    params_resnet_18_random_augmented_photo,
)


dataset = all_datasets["pacs"]

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
    for filename in tqdm(os.listdir("weights_resnet_18_experiment")):
        print(filename)
        target_domain, _, _, i = filename.replace("art_painting", "art-painting").split(
            "_"
        )
        i = int(i.removesuffix(".pth"))
        if target_domain == "art-painting":
            target_domain = "art_painting"

        model = get_resnet_18(pretrained=False)

        state_dict = load(
            os.path.join("weights_resnet_18_experiment", filename), map_location=device
        )
        model.load_state_dict(state_dict)
        model.to(device)

        train_loader, _, test_loader, _ = split_df_into_loaders(
            df=dataset["df"], target_domain=target_domain
        )

        if target_domain == "art_painting":
            params = deepcopy(params_resnet_18_random_augmented_art_painting)
        elif target_domain == "sketch":
            params = deepcopy(params_resnet_18_random_augmented_sketch)
        elif target_domain == "photo":
            params = deepcopy(params_resnet_18_random_augmented_photo)
        elif target_domain == "cartoon":
            params = deepcopy(params_resnet_18_random_augmented_cartoon)
        else:
            raise RuntimeError("unknown target domain")
        params["TARGET_DOMAIN"] = target_domain
        transformation_pipeline = get_transform_pipeline(params=params)

        builder["target_domain"].append(target_domain)
        builder["architecture"].append("ResNet18")
        builder["pretrained"].append(False)
        builder["augmented"].append(True)
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
    df.to_csv("results_resnet18_experiment.csv")
    print(df)
    return df


if __name__ == "__main__":
    main()
