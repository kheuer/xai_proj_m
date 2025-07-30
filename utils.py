from ast import literal_eval
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from config import MAX_EPOCHS, PATIENCE, BATCH_SIZE
from dataset_utils import get_dataloader, all_datasets, split_df, split_domains
from torch.utils.data import DataLoader


def show_tensor(tensor):
    image_tensor = tensor["pixel_values"]
    image_tensor = image_tensor.squeeze(0)
    image_np = image_tensor.permute(1, 2, 0).detach().numpy()
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()


def plot_loss(train_losses, val_losses, test_losses):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))  # Create a 1x2 subplot

    # Standard Loss Plot
    ax[0].plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label=f"Training Loss (last={round(train_losses[-1], 2)})",
        marker="o",
    )
    ax[0].plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label=f"Validation Loss (last={round(val_losses[-1], 2)})",
        marker="o",
    )
    ax[0].plot(
        range(1, len(test_losses) + 1),
        test_losses,
        label=f"Test Loss (last={round(test_losses[-1], 2)})",
        marker="o",
    )
    ax[0].set_title("Training and Validation Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid()

    # Logarithmic Loss Plot
    ax[1].plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label=f"Training Loss (last={round(train_losses[-1], 2)})",
        marker="o",
    )
    ax[1].plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label=f"Validation Loss (last={round(val_losses[-1], 2)})",
        marker="o",
    )
    ax[1].plot(
        range(1, len(test_losses) + 1),
        test_losses,
        label=f"Test Loss (last={round(test_losses[-1], 2)})",
        marker="o",
    )
    ax[1].set_title("Training and Validation Loss (Log Scale)")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].set_yscale("log")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.savefig("loss_comparison.png")
    plt.close()
    # plt.show(block=False)


def show_label_distribution(dataloader, classes) -> None:
    df = pd.DataFrame({"labels": dataloader.dataset.tensors[1].cpu()})
    df["labels"] = df["labels"].apply(lambda x: f"{classes[x]}: ({x})")

    label_counts = df["labels"].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Label distribution")
    plt.axis("equal")
    plt.show()


def get_expected_input(prompt: str, options: List[str]) -> str:
    choice = -1
    while int(choice) not in range(1, len(options) + 1, 1):
        choice = input(
            prompt
            + " "
            + " ".join([f"[{i+1}] {x}" for i, x in enumerate(options)])
            + " "
        ).strip()
        try:
            int(choice)
        except ValueError:
            choice = -1
    return options[int(choice) - 1]


def parse_params(data):
    try:
        return literal_eval(data[data.index("{") :].strip())
    except SyntaxError:
        raise ValueError("The data contains no well formated key-value pair")


def get_params_from_user():
    try:
        parsed = parse_params(input("Please insert the paramers dictionary: "))
        params = {}
        if "EPOCHS" in parsed:
            assert isinstance(parsed["EPOCHS"], int)
            params["EPOCHS"] = parsed["EPOCHS"]
        else:
            params["EPOCHS"] = MAX_EPOCHS

        if "PATIENCE" in parsed:
            assert isinstance(parsed["PATIENCE"], int)
            params["PATIENCE"] = parsed["PATIENCE"]
        else:
            params["PATIENCE"] = PATIENCE

        assert isinstance(parsed["LEARNING_RATE"], float)
        params["LEARNING_RATE"] = parsed["LEARNING_RATE"]

        assert isinstance(parsed["BETA_1"], float)
        assert isinstance(parsed["BETA_2"], float)
        params["BETAS"] = (parsed["BETA_1"], parsed["BETA_2"])

        assert isinstance(parsed["WEIGHT_DECAY"], float)
        params["WEIGHT_DECAY"] = parsed["WEIGHT_DECAY"]

        assert parsed["OPTIMIZER"] in ("AdamW", "SGD")
        params["OPTIMIZER"] = parsed["OPTIMIZER"]

        assert parsed["SCHEDULER"] in (
            "CosineAnnealingLR",
            "ReduceLROnPlateau",
            "LinearLR",
            "StepLR",
            "None",
        )
        params["SCHEDULER"] = parsed["SCHEDULER"]

        if params["OPTIMIZER"] == "SGD":
            assert isinstance(parsed["MOMENTUM"], float)
            params["MOMENTUM"] = parsed["MOMENTUM"]

            assert isinstance(parsed["DAMPENING"], float)
            params["DAMPENING"] = parsed["DAMPENING"]

        if parsed["SCHEDULER"] in ("StepLR", "ReduceLROnPlateau"):
            assert isinstance(parsed["GAMMA"], float)
            params["GAMMA"] = parsed["GAMMA"]

        if parsed["SCHEDULER"] == "StepLR":
            assert isinstance(parsed["STEP_SIZE"], int)
            params["STEP_SIZE"] = parsed["STEP_SIZE"]

        if "USE_JIGSAW" in parsed:
            params["USE_JIGSAW"] = parsed["USE_JIGSAW"]
            if params["USE_JIGSAW"]:
                params["MIN_GRID_SIZE"] = parsed["MIN_GRID_SIZE"]
                params["MAX_GRID_SIZE"] = parsed["MAX_GRID_SIZE"]

        if "USE_DLOW" in parsed:
            params["USE_DLOW"] = parsed["USE_DLOW"]
            if "target_domain" in parsed:
                params["target_domain"] = parsed["target_domain"]

        if "USE_FOURIER" in parsed:
            params["USE_FOURIER"] = parsed["USE_FOURIER"]
            if params["USE_FOURIER"]:
                params["SQUARE_SIZE"] = parsed["SQUARE_SIZE"]
                params["ETA"] = parsed["ETA"]

        if "USE_AUGMIX" in parsed:
            params["USE_AUGMIX"] = parsed["USE_AUGMIX"]
            if params["USE_AUGMIX"]:
                params["SEVERITY"] = parsed["SEVERITY"]
                params["MIXTURE_WIDTH"] = parsed["MIXTURE_WIDTH"]
                params["CHAIN_DEPTH"] = parsed["CHAIN_DEPTH"]
                params["ALPHA"] = parsed["ALPHA"]
                params["ALL_OPS"] = parsed["ALL_OPS"]
                params["INTERPOLATION"] = parsed["INTERPOLATION"]

        params["TRANSFORMATIONS_ORDER"] = parsed.get(
            "TRANSFORMATIONS_ORDER", "Augmix,Dlow,Fourier,Jigsaw"
        )

        return params

    except ValueError:
        raise ValueError("This dictionary is not well formated")


# this has to be here to avoid circular imports
def split_df_into_loaders(
    df: pd.DataFrame, target_domain: Union[str, None] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, str]:
    if target_domain is None:
        target_domain = get_expected_input(
            "Please choose the target domain:", sorted(df["domain"].unique())
        )

    df_source, df_target = split_domains(df, target_domain)

    train_df, val_df = split_df(df_source, test_size=0.15)
    test_df = df_target

    train_loader = get_dataloader(train_df, batch_size=BATCH_SIZE)
    val_loader = get_dataloader(val_df, batch_size=BATCH_SIZE)
    test_loader = get_dataloader(test_df, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader, target_domain


def split_df_into_loaders_val_alt_domain(
    df: pd.DataFrame, target_domain: Union[str, None] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, str]:
    if target_domain is None:
        target_domain = get_expected_input(
            "Please choose the target domain:", sorted(df["domain"].unique())
        )

    df_train_and_val, df_test = split_domains(df, target_domain)

    val_domain = {
        "sketch": "cartoon",
        "photos": "art_painting",
        "cartoon": "sketch",
        "art_painting": "photo",
    }[target_domain]

    df_train, df_val = split_domains(df_train_and_val, val_domain)

    train_loader = get_dataloader(df_train, batch_size=BATCH_SIZE)
    val_loader = get_dataloader(df_val, batch_size=BATCH_SIZE)
    test_loader = get_dataloader(df_test, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader, target_domain
