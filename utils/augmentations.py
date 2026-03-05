"""
augmentations.py — Forensic-specific augmentations (blurs, noise, JPEG artefacts).

All transforms are designed to simulate real-world post-processing that
forgers or social-media pipelines apply.  Built on top of albumentations
for speed, with paired image + mask support.
"""

from typing import Dict, Tuple

import albumentations as A
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_train_augmentations(
    image_size: int = 512,
    jpeg_quality_range: Tuple[int, int] = (50, 95),
    blur_limit: Tuple[int, int] = (3, 7),
    noise_var_limit: Tuple[float, float] = (10.0, 50.0),
    downscale_range: Tuple[float, float] = (0.25, 0.75),
) -> A.Compose:
    """Return the training augmentation pipeline.

    The pipeline includes:
    - Random crop + resize to ``image_size``
    - Horizontal / vertical flips
    - Gaussian blur (simulates anti-aliasing used to mask splicing)
    - Additive Gaussian noise (sensor noise simulation)
    - JPEG compression artefacts
    - Downscale → upscale (anti-forensic resizing attack)
    - Normalisation to [0, 1]

    Parameters
    ----------
    image_size : int
        Target spatial size (H = W).
    jpeg_quality_range : tuple of int
        (min, max) JPEG quality factor.
    blur_limit : tuple of int
        Kernel size range for Gaussian blur.
    noise_var_limit : tuple of float
        Variance range for additive Gaussian noise.
    downscale_range : tuple of float
        Scale range for the downscale–upscale attack.

    Returns
    -------
    A.Compose
        Albumentations pipeline; call with ``pipeline(image=img, mask=msk)``.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            # --- forensic-specific ---
            A.GaussianBlur(blur_limit=blur_limit, sigma_limit=(0.5, 2.0), p=0.4),
            A.GaussNoise(std_range=(noise_var_limit[0] / 255.0, noise_var_limit[1] / 255.0), p=0.4),
            A.ImageCompression(
                quality_range=jpeg_quality_range,
                p=0.5,
            ),
            A.Downscale(
                scale_range=(downscale_range[0], downscale_range[1]),
                p=0.3,
            ),
        ],
        additional_targets={},
    )


def get_val_augmentations(image_size: int = 512) -> A.Compose:
    """Return the validation / test augmentation pipeline.

    Only deterministic resizing and normalisation — no stochastic transforms.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
        ],
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def apply_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    transform: A.Compose,
) -> Dict[str, np.ndarray]:
    """Apply an albumentations transform to an image + mask pair.

    Parameters
    ----------
    image : np.ndarray
        BGR/RGB uint8 image (H, W, 3).
    mask : np.ndarray
        Binary uint8 mask (H, W) — 0 = authentic, 255 = tampered.
    transform : A.Compose
        Albumentations pipeline.

    Returns
    -------
    dict
        ``{"image": augmented_image, "mask": augmented_mask}``
    """
    return transform(image=image, mask=mask)
