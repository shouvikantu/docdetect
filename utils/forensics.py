"""
forensics.py — Math utilities for SRM, ELA, and Gradient forensic features.

Provides functions to extract forensic-grade signal residuals from document
images, producing the 3-channel "forensic" companion that is stacked with
the RGB image to form a 6-channel input tensor.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  SRM (Spatial Rich Model) High-Pass Filters
# ---------------------------------------------------------------------------

# Canonical 5×5 SRM kernels (subset of the 30 Fridrich filters).
# Three representative kernels are used here; extend as needed.

SRM_KERNELS = np.array(
    [
        # 1st-order edge kernel
        [
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0],
            [0,  1, -2,  1,  0],
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0],
        ],
        # 2nd-order square kernel
        [
            [0,  0,  0,  0,  0],
            [0, -1,  2, -1,  0],
            [0,  2, -4,  2,  0],
            [0, -1,  2, -1,  0],
            [0,  0,  0,  0,  0],
        ],
        # 3rd-order edge kernel
        [
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0],
            [0, -1,  3, -3,  1],
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0],
        ],
    ],
    dtype=np.float32,
)


class SRMFilterLayer(nn.Module):
    """Fixed-weight convolutional layer that applies SRM high-pass filters.

    Input : (B, 3, H, W) RGB image tensor  (values in [0, 1])
    Output: (B, C_srm, H, W) noise residual maps  (C_srm = n_kernels × 3)
    """

    def __init__(self) -> None:
        super().__init__()
        kernels = torch.from_numpy(SRM_KERNELS)  # (K, 5, 5)
        n_kernels = kernels.shape[0]
        # Expand to per-channel: each kernel applied independently to R, G, B
        # Shape: (K*3, 1, 5, 5) used with groups=3
        weight = kernels.unsqueeze(1).repeat(3, 1, 1, 1)  # (K*3, 1, 5, 5)
        self.register_buffer("weight", weight)
        self.groups = 3
        self.n_out = n_kernels * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        return F.conv2d(x, self.weight, padding=2, groups=self.groups)


def compute_srm(image: np.ndarray) -> np.ndarray:
    """Compute a single-channel SRM noise residual (mean of abs residuals).

    Parameters
    ----------
    image : np.ndarray
        BGR or RGB uint8 image, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Single-channel float32 residual, shape (H, W), normalised to [0, 1].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    residuals = []
    for kernel in SRM_KERNELS:
        res = cv2.filter2D(gray, -1, kernel)
        residuals.append(np.abs(res))
    srm_map = np.mean(residuals, axis=0)
    srm_map = np.clip(srm_map / (srm_map.max() + 1e-8), 0, 1)
    return srm_map


# ---------------------------------------------------------------------------
# 2.  ELA (Error Level Analysis)
# ---------------------------------------------------------------------------

def compute_ela(
    image: np.ndarray,
    quality: int = 90,
    scale: float = 15.0,
) -> np.ndarray:
    """Compute Error Level Analysis map.

    Re-compresses the image at a given JPEG quality and measures the
    pixel-wise absolute difference from the original. Tampered regions
    typically show higher error levels.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image, shape (H, W, 3).
    quality : int
        JPEG re-compression quality (1-100).
    scale : float
        Multiplier for visualisation (clamped to [0, 255]).

    Returns
    -------
    np.ndarray
        Single-channel float32 ELA map, shape (H, W), normalised to [0, 1].
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", image, encode_param)
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    diff = cv2.absdiff(image, recompressed).astype(np.float32)
    ela_map = np.mean(diff, axis=2) * scale
    ela_map = np.clip(ela_map / 255.0, 0, 1)
    return ela_map


# ---------------------------------------------------------------------------
# 3.  Gradient Magnitude
# ---------------------------------------------------------------------------

def compute_gradient(image: np.ndarray) -> np.ndarray:
    """Compute edge / gradient magnitude using Sobel filters.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Single-channel float32 gradient map, shape (H, W), normalised to [0, 1].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = np.clip(magnitude / (magnitude.max() + 1e-8), 0, 1)
    return magnitude


# ---------------------------------------------------------------------------
# 4.  Combined 3-Channel Forensic Feature Map
# ---------------------------------------------------------------------------

def build_forensic_channels(image: np.ndarray) -> np.ndarray:
    """Stack SRM, ELA, and Gradient into a 3-channel forensic tensor.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Float32 array of shape (H, W, 3) with channels [SRM, ELA, Gradient],
        each normalised to [0, 1].
    """
    srm = compute_srm(image)
    ela = compute_ela(image)
    grad = compute_gradient(image)
    return np.stack([srm, ela, grad], axis=-1).astype(np.float32)
