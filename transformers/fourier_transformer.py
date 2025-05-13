import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift

class FourierTransformer(object):
    def __init__(self, square_size=None, img_size=227, eta=None):
        """
               Initializes the FourierTransformer.

               This transformer supports three augmentation modes depending on the arguments provided:

               Mode 0: Spatial-only swapping
                   - Only `square_size` is provided.
                   - Swaps a central square region in the amplitude spectrum between shuffled image pairs.

               Mode 1: Amplitude mixing
                   - Only `eta` is provided.
                   - Mixes the entire amplitude spectrum with shuffled pairs using a random λ ∈ [0, η].

               Mode 2: Combined
                   - Both `square_size` and `eta` are provided.
                   - Applies both local swapping of the central amplitude region and global mixing.

               Args:
                   square_size (int, optional): Size (in pixels) of the central square region to swap.
                   img_size (int, optional): Spatial size of the input images. Default is 227.
                   eta (float, optional): Upper bound of λ for amplitude mixing (λ ∼ U(0, η)).

               Raises:
                   ValueError: If neither `square_size` nor `eta` is provided.
        """

        super().__init__()

        if square_size is not None and eta is None:
            self.square_size = square_size
            center = img_size // 2
            self.square_start = center - (square_size // 2)
            self.square_stop = center + (square_size // 2)
            self.augmentation_type = 0

        elif square_size is None and eta is not None:
            self.eta = eta
            self.augmentation_type = 1

        elif square_size is not None and eta is not None:
            self.square_size = square_size
            center = img_size // 2
            self.square_start = center - (square_size // 2)
            self.square_stop = center + (square_size // 2)
            self.eta = eta
            self.augmentation_type = 2

        else:
            raise ValueError("Invalid arguments: provide at least square_size or eta")

    def swap_fourier_centers_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        """
        Swaps the central amplitude region in the frequency domain between shuffled image pairs.
        Phase is kept intact, only the magnitude (amplitude) is exchanged locally.

        Args:
            images (torch.Tensor): Tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Tensor of shape (B, C, H, W) after inverse FFT
        """
        B, C, H, W = images.shape

        # Define central square coordinates
        half = self.square_size // 2
        cy, cx = H // 2, W // 2
        y_slice = slice(cy - half, cy + half)
        x_slice = slice(cx - half, cx + half)

        # Compute FFT and shift the zero frequency to center
        fft_img = fftshift(fft2(images), dim=(-2, -1))

        # Separate amplitude (magnitude) and phase
        amplitude = torch.abs(fft_img)
        phase = torch.angle(fft_img)

        # Shuffle the batch and extract the shuffled amplitudes
        perm = torch.randperm(B)
        amplitude_shuffled = amplitude[perm]

        # Swap central square in amplitude
        swapped_amplitude = amplitude.clone()
        swapped_amplitude[:, :, y_slice, x_slice] = amplitude_shuffled[:, :, y_slice, x_slice]

        # Recombine amplitude and phase into complex tensor
        fft_result = swapped_amplitude * torch.exp(1j * phase)

        # Inverse FFT to get back real-space image
        ifft = ifft2(ifftshift(fft_result, dim=(-2, -1))).real

        return ifft

    def mix_fourier_centers_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        """
        Mixes the amplitude of each image with a randomly shuffled image in the batch using
        a random interpolation coefficient λ ∼ U(0, η), while keeping the phase unchanged.

        Args:
            images (torch.Tensor): Tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Tensor of shape (B, C, H, W) after inverse FFT
        """
        B, C, H, W = images.shape

        # FFT and shift zero frequency to the center
        fft_img = fftshift(fft2(images), dim=(-2, -1))

        # Extract amplitude and phase
        amplitude = torch.abs(fft_img)
        phase = torch.angle(fft_img)

        # Shuffle the batch and get corresponding amplitudes
        perm = torch.randperm(B)
        amplitude_shuffled = amplitude[perm]

        # Sample lambda ∼ U(0, eta) per image and reshape for broadcasting
        lambdas = torch.empty((B, 1, 1, 1), device=amplitude.device).uniform_(0, self.eta)

        # Interpolate amplitudes (mix with shuffled version)
        amplitude = (1 - lambdas) * amplitude + lambdas * amplitude_shuffled

        # Combine mixed amplitude with original phase
        fft_result = amplitude * torch.exp(1j * phase)

        # Inverse FFT to reconstruct real images
        ifft = ifft2(ifftshift(fft_result, dim=(-2, -1))).real

        return ifft

    def __call__(self, img):

        if self.augmentation_type == 0:
            return self.swap_fourier_centers_amplitude(img)
        elif self.augmentation_type == 1:
            return self.mix_fourier_centers_amplitude(img)
        else:
            if torch.rand(1).item() < 0.5:
                return self.swap_fourier_centers_amplitude(img)
            else:
                return self.mix_fourier_centers_amplitude(img)

#example usage
if __name__ == '__main__':

    import os
    from torchvision.utils import save_image

    from dataset_utils import all_datasets, split_df
    from utils import split_df_into_loaders

    dataset_name = "pacs"
    dataset = all_datasets[dataset_name]

    # take a random sample to speed up training
    _, df_sampled = split_df(dataset["df"], test_size=0.2)

    train_loader, val_loader, test_loader, target_domain = split_df_into_loaders(df_sampled, target_domain='sketch')
    import torchvision.transforms as transforms

    transforms = transforms.Compose([
        #use either amplitude swap or amplitude mix, random choice
        FourierTransformer(square_size=10, eta=1.0)
        #use amplitude swap
        #FourierTransformer(square_size=10)
        #use amplitude mix
        #FourierTransformer(eta=1.0)
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
