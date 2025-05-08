from ast import literal_eval
from typing import Union, List
import matplotlib.pyplot as plt
import pandas as pd
from config import MAX_EPOCHS, PATIENCE


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

        assert isinstance(parsed["BATCH_SIZE"], int)
        params["BATCH_SIZE"] = parsed["BATCH_SIZE"]

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

        return params

    except ValueError:
        raise ValueError("This dictionary is not well formated")
