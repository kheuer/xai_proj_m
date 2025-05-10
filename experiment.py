from models import get_resnet_18, get_resnet_50, calculate_val_loss
from dataset_utils import get_dataloader, all_datasets, split_df, split_domains
from utils import (
    get_expected_input,
    get_params_from_user,
    split_df_into_loaders,
)

model_name = get_expected_input("Please choose a model:", ("ResNet18", "ResNet50"))
if model_name == "ResNet18":
    model = get_resnet_18()
elif model_name == "ResNet50":
    model = get_resnet_50()

# TODO: ask the user for input when we obtain another dataset
dataset_name = "pacs"
dataset = all_datasets[dataset_name]
params = get_params_from_user()

train_loader, val_loader, test_loader, target_domain = split_df_into_loaders(
    dataset["df"]
)


print(
    f"\n\n\nTrain {model_name} model with target_domain = {target_domain} and Hyperparameters = {params}"
)

loss = calculate_val_loss(
    train_loader=train_loader,
    test_loader=test_loader,
    val_loader=val_loader,
    model=model,
    HYPERPARAMS=params,
)

print("Final validation loss:", loss)
