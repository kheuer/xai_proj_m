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
