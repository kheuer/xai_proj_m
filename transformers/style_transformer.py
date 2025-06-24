import os
import torch
import torchvision.transforms
from torchvision.utils import save_image

from utils import split_df_into_loaders
from dataset_utils import all_datasets, split_df
from dlow.models.test_model import TestModel


class StyleTransformer(object):
    """
    Transformer class to interpolate the style of images from source domains
    while explicitly excluding the target domain. This is typically used in
    domain adaptation tasks where the model applies learned visual styles from
    multiple domains to transform input images.
    """

    def __init__(self, ckpt_dir: str, target_domain: str, img_size=(227, 227)):
        """
        Initializes the transformer with two source-domain models.

        Args:
            ckpt_dir (str): Directory containing all model checkpoints.
            target_domain (str): Name of the target domain to exclude when selecting source models.
        """
        super().__init__()
        ckpt_path1, ckpt_path2 = self.get_ckpt_path(ckpt_dir, target_domain)

        self.model1 = TestModel(ckpt_path1)
        self.model2 = TestModel(ckpt_path2)
        self.resizer = torchvision.transforms.Resize(img_size)

    def __call__(self, img):
        """
        Applies random style transformation to each image in a batch.

        Args:
            img (torch.Tensor): Batch of images, shape (N, C, H, W)

        Returns:
            torch.Tensor: Transformed batch of images.
        """
        with torch.no_grad():
            for i in range(img.shape[0]):
                img_i = img[i].unsqueeze(0)  # Shape: (1, C, H, W)
                weights = [
                    torch.rand(1).item() for _ in range(4)
                ]  # Generate 4 random weights

                # Randomly choose between model1 and model2
                model = self.model1 if torch.rand(1).item() < 0.5 else self.model2

                model.set_input({"A": img_i}, weights)
                model.test()

                # Get the generated image and resize
                generated_img = model.get_current_visuals()["fake_B"]
                img[i] = self.resizer(generated_img).squeeze(0)

        return img

    def get_ckpt_path(self, ckpt_dir, target_domain):
        """
        Selects two source domain checkpoints excluding the target domain.

        Args:
            ckpt_dir (str): Directory containing model checkpoints directories.
            target_domain (str): Domain to be excluded from source checkpoints.

        Returns:
            tuple: Paths to two usable source domain model checkpoints.
        """
        ckpts_dirs = os.listdir(ckpt_dir)
        usable_ckpt_dirs = [
            ckpt_dir for ckpt_dir in ckpts_dirs if target_domain not in ckpt_dir
        ]

        return os.path.join(ckpt_dir, usable_ckpt_dirs[0]), os.path.join(
            ckpt_dir, usable_ckpt_dirs[1]
        )


# example usage
if __name__ == "__main__":

    dataset_name = "pacs"
    dataset = all_datasets[dataset_name]

    # take a random sample to speed up training
    _, df_sampled = split_df(dataset["df"], test_size=0.2)
    train_loader, val_loader, test_loader, target_domain = split_df_into_loaders(
        df_sampled, target_domain="sketch"
    )
    import torchvision.transforms as transforms

    transforms = transforms.Compose([StyleTransformer("dlow/checkpoints/", "sketch")])

    for data in train_loader:
        imgs = data[0]
        for i, image in enumerate(imgs):
            save_image(image, f"output_images/image_{i}.png")

        out = transforms(imgs)
        os.makedirs("output_images", exist_ok=True)

        # Save each image
        for i, image in enumerate(out):
            save_image(image, f"output_images/out_{i}.png")
        break
