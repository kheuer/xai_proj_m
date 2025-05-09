import numpy as np
from math import sqrt
import torchvision.transforms.v2 as T


def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """copied from the GitHub repository at https://github.com/MediaBrain-SJTU/FACT"""

    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start : h_start + h_crop, w_start : w_start + w_crop] = (
        lam * img2_abs_[h_start : h_start + h_crop, w_start : w_start + w_crop]
        + (1 - lam) * img1_abs_[h_start : h_start + h_crop, w_start : w_start + w_crop]
    )
    img2_abs[h_start : h_start + h_crop, w_start : w_start + w_crop] = (
        lam * img1_abs_[h_start : h_start + h_crop, w_start : w_start + w_crop]
        + (1 - lam) * img2_abs_[h_start : h_start + h_crop, w_start : w_start + w_crop]
    )

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12


transform_pipeline = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
