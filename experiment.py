from models import get_resnet_18, calculate_val_loss
from dataset_utils import get_dataloader, all_datasets, split_df, split_domains
from utils import get_expected_input, get_params_from_user

model_name = get_expected_input("Please choose a model:", ("ResNet18", "ResNet50"))

# TODO: ask the user for input when we obtain another dataset
dataset_name = "pacs"
dataset = all_datasets[dataset_name]

target_domain = get_expected_input(
    "Please choose te target domain:", dataset["domains"]
)

params = get_params_from_user()

train_df, test_df = split_domains(dataset["df"], target_domain)


# train_df, test_df = split_domains(DATAFRAME, TARGET_DOMAIN)
train_df, test_df = split_df(all_datasets["pacs"]["df"], test_size=0.2)
train_loader = get_dataloader(train_df, batch_size=params["BATCH_SIZE"])
test_loader = get_dataloader(test_df, batch_size=params["BATCH_SIZE"])

model = get_resnet_18()

loss = calculate_val_loss(
    train_loader=train_loader, test_loader=test_loader, model=model, HYPERPARAMS=params
)
