import torch
import torch.nn.functional as F

class JigsawTransform(object):
    """
        Applies a block-wise random permutation to each image in a batch.

        This transformation splits each input image into a grid of non-overlapping square blocks,
        randomly permutes the blocks, and reconstructs the image. It can operate in two modes:

        1. **Fixed grid size**: All images are divided into blocks using the same `grid_size`.
        2. **Random grid size**: Each image is assigned a random grid size sampled uniformly from
           the range [`min_grid_size`, `max_grid_size`].

        This transformation preserves the original image dimensions and can be useful for tasks
        like self-supervised learning or data augmentation by disrupting local spatial structure.

        Args:
            grid_size (int, optional): Fixed grid size to use for all images. If `None`, random
                grid sizes will be sampled per image.
            min_grid_size (int, optional): Minimum grid size to sample when using random grid mode.
            max_grid_size (int, optional): Maximum grid size to sample when using random grid mode.
    """
    def __init__(self, grid_size=None, min_grid_size=2, max_grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size

    def get_grid_sizes(self, batch_size):
        return torch.randint(low=self.min_grid_size, high=self.max_grid_size, size=(batch_size,))

    def permute_random_grid(self, images):
        batch_size, channels, height, width = images.shape
        assert height == width  # Currently only supports square images
        device = images.device

        # Get a different random grid size for each image in the batch
        grid_sizes = self.get_grid_sizes(batch_size)
        output_images = []

        # Process each image in the batch independently
        for b in range(batch_size):
            grid_size = grid_sizes[b].item()

            # Compute the size to pad the image so it divides evenly into grid blocks
            target_size = ((height + grid_size - 1) // grid_size) * grid_size
            padding = target_size - height
            block_size = target_size // grid_size  # Size of each block

            # Pad the image using reflection padding to avoid border artifacts
            img = F.pad(images[b:b + 1], (0, padding, 0, padding), mode='reflect')  # (1, C, T, T)
            img = img[0]  # Remove batch dimension -> (C, T, T)

            # Divide the image into grid blocks:
            # Shape becomes (C, G, bs, G, bs) where G = grid_size, bs = block_size
            blocks = img.view(channels, grid_size, block_size, grid_size, block_size)

            # Rearrange dimensions to group blocks spatially: (C, G, G, bs, bs)
            blocks = blocks.permute(0, 1, 3, 2, 4).contiguous()

            # Flatten spatial grid into a sequence of blocks: (C, G², bs, bs)
            blocks = blocks.view(channels, grid_size * grid_size, block_size, block_size)

            # Apply a random permutation to the blocks (same for all channels of image `b`)
            perm = torch.randperm(grid_size * grid_size, device=device)
            blocks = blocks[:, perm]

            # Reshape blocks back into grid layout
            blocks = blocks.view(channels, grid_size, grid_size, block_size, block_size)

            # Reorder dimensions back to spatial layout: (C, T, T)
            blocks = blocks.permute(0, 1, 3, 2, 4).contiguous()
            img_permuted = blocks.view(channels, target_size, target_size)

            # Remove any padding to return to original size
            img_permuted = img_permuted[..., :height, :width]  # Shape: (C, H, W)

            # Collect permuted image
            output_images.append(img_permuted)

        # Stack the list of permuted images back into a batch: (B, C, H, W)
        return torch.stack(output_images, dim=0)

    def permute_fixed_grid(self, images):
        # Get image dimensions
        batch_size, channels, height, width = images.shape
        assert height == width  # Ensure the image is square
        device = images.device
        grid_size = self.grid_size

        # Calculate size to pad image so it is divisible by the grid size
        target_size = ((height + grid_size - 1) // grid_size) * grid_size
        padding = target_size - height
        block_size = target_size // grid_size  # Size of each block

        # Pad images using reflection to make them evenly divisible by grid
        images = F.pad(images, (0, padding, 0, padding), mode='reflect')  # Shape: (B, C, T, T)

        # Split images into non-overlapping grid blocks
        # Reshape to (B, C, G, bs, G, bs): grid rows/columns with block size
        blocks = images.view(batch_size, channels, grid_size, block_size, grid_size, block_size)

        # Rearrange to group blocks: (B, C, G, G, bs, bs)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5)

        # Flatten grid blocks: (B, C, G², bs, bs)
        blocks = blocks.contiguous().view(batch_size, channels, grid_size * grid_size, block_size, block_size)

        # Create a new tensor for the permuted blocks
        permuted_blocks = torch.empty_like(blocks)

        # Apply a different random permutation to the blocks of each image
        for b in range(batch_size):
            perm = torch.randperm(grid_size * grid_size, device=device)  # Random block permutation
            permuted_blocks[b] = blocks[b, :, perm]  # Apply permutation to all channels of image b

        # Reshape permuted blocks back to grid layout
        blocks = permuted_blocks.view(batch_size, channels, grid_size, grid_size, block_size, block_size)

        # Rearrange back to spatial layout: (B, C, T, T)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        images_permuted = blocks.view(batch_size, channels, target_size, target_size)

        # Crop back to original size by removing padding
        return images_permuted[..., :height, :width]

    def __call__(self, images):
        if self.grid_size:
            return self.permute_fixed_grid(images)
        else:
            return self.permute_random_grid(images)




# example usage
if __name__ == '__main__':

    import os
    from torchvision.utils import save_image

    from dataset_utils import all_datasets, split_df
    from utils import split_df_into_loaders

    dataset_name = "pacs"
    dataset = all_datasets[dataset_name]

    # take a random sample to speed up training
    _, df_sampled = split_df(dataset["df"], test_size=0.2)

    train_loader, val_loader, test_loader, target_domain = split_df_into_loaders(df_sampled,
                                                                                 target_domain='sketch')
    import torchvision.transforms as transforms

    transforms = transforms.Compose([
      JigsawTransform()
    ])
    for data in train_loader:
        imgs = data[0]
        for i, image in enumerate(imgs):
            save_image(image, f'output_images/image_{i}.png')

        out = transforms(imgs)
        os.makedirs('output_images', exist_ok=True)

        # Save each image
        for i, image in enumerate(out):
            save_image(image, f'output_images/out_{i}.png')
        break

