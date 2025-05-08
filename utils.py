from ast import literal_eval
from typing import Union, List
import matplotlib.pyplot as plt
import pandas as pd


def show_tensor(tensor):
    image_tensor = tensor["pixel_values"]
    image_tensor = image_tensor.squeeze(0)
    image_np = image_tensor.permute(1, 2, 0).detach().numpy()
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()


def plot_loss(train_losses, val_losses):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))  # Create a 1x2 subplot

    # Standard Loss Plot
    ax[0].plot(
        range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o"
    )
    ax[0].plot(
        range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o"
    )
    ax[0].set_title("Training and Validation Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid()

    # Logarithmic Loss Plot
    ax[1].plot(
        range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o"
    )
    ax[1].plot(
        range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o"
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
        return literal_eval(data[data.index("{") :])
    except SyntaxError:
        raise ValueError("The data contains no well formated key-value pair")


def get_params_from_user():
    while True:
        try:
            params = parse_params(input("Please insert the paramers dictionary: "))
            assert isinstance(params["EPOCHS"], float)
            assert isinstance(params["PATIENCE"], int)
            assert isinstance(params["LEARNING_RATE"], float)
            assert isinstance(params["BETA_1"], float)
            assert isinstance(params["BETA_2"], float)
            assert isinstance(params["BATCH_SIZE"], int)
            assert params["OPTIMIZER"] in ("AdamW", "SGD")
            assert params["SCHEDULER"] in (
                "CosineAnnealingLR",
                "ReduceLROnPlateau",
                "LinearLR",
                "StepLR",
                "None",
            )

            if params["OPTIMIZER"] == "SGD":
                assert isinstance(params["MOMENTUM"], float)
                assert isinstance(params["DAMPENING"], float)

            if params["SCHEDULER"] in ("StepLR", "ReduceLROnPlateau"):
                assert isinstance(params["GAMMA"], float)
            if params["SCHEDULER"] == "StepLR":
                assert isinstance(params["STEP_SIZE"], int)
            params["BETAS"] = (params["BETA_1"], params["BETA_2"])
            return params

        except (ValueError, AssertionError, KeyError):
            continue
