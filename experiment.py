from models import get_resnet_18, calculate_val_loss
from dataset_utils import (
    get_dataloader,
    all_datasets,
    split_df,
)

params = {
    "EPOCHS": 3,
    "PATIENCE": 20,
    "LEARNING_RATE": 0.001,
    "BETAS": (0.9, 0.999),
    "WEIGHT_DECAY": 0,
    "BATCH_SIZE": 32,
}

# train_df, test_df = split_domains(DATAFRAME, TARGET_DOMAIN)
train_df, test_df = split_df(all_datasets["pacs"]["df"], test_size=0.2)
train_loader = get_dataloader(train_df, batch_size=params["BATCH_SIZE"])
test_loader = get_dataloader(test_df, batch_size=params["BATCH_SIZE"])

model = get_resnet_18()

loss = calculate_val_loss(
    train_loader=train_loader, test_loader=test_loader, model=model, HYPERPARAMS=params
)
