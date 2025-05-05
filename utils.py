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
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o"
    )
    plt.plot(
        range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o"
    )
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss.png")
    plt.show()


def show_label_distribution(dataloader, classes) -> None:
    df = pd.DataFrame({"labels": dataloader.dataset.tensors[1].cpu()})
    df["labels"] = df["labels"].apply(lambda x: classes[x])

    label_counts = df["labels"].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Label distribution")
    plt.axis("equal")
    plt.show()
