"""
gpu_forensics.py — GPU-accelerated forensic feature extraction.

Moves SRM, ELA (approximated), and Gradient computation to GPU as
nn.Module layers, eliminating the CPU bottleneck during training.

This replaces the CPU-based forensic computation in forensics.py with
fully GPU-resident, batched operations that run as part of the model
forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPUForensicFeatures(nn.Module):
    """Compute 3-channel forensic features entirely on GPU.

    Channels:
        0. SRM — Spatial Rich Model noise residuals (high-pass filters)
        1. ELA — Error Level Analysis approximation (blur-difference)
        2. Gradient — Sobel edge magnitude

    Input : (B, 3, H, W) RGB float32 tensor in [0, 1]
    Output: (B, 3, H, W) forensic feature maps in [0, 1]
    """

    def __init__(self) -> None:
        super().__init__()
        self._build_srm_filters()
        self._build_sobel_filters()
        self._build_ela_blur()

    def _build_srm_filters(self) -> None:
        """Register SRM high-pass filter kernels."""
        kernels = torch.tensor([
            # 1st-order edge
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 1, -2, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            # 2nd-order square
            [[0, 0, 0, 0, 0],
             [0, -1, 2, -1, 0],
             [0, 2, -4, 2, 0],
             [0, -1, 2, -1, 0],
             [0, 0, 0, 0, 0]],
            # 3rd-order edge
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, -1, 3, -3, 1],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
        ], dtype=torch.float32)  # (3, 5, 5)

        # Convert to grayscale weights: average over RGB
        # Weight shape: (3, 1, 5, 5) — 3 filters, 1 input channel
        weight = kernels.unsqueeze(1)  # (3, 1, 5, 5)
        self.register_buffer("srm_weight", weight)

        # RGB to grayscale weights
        gray_weight = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
        self.register_buffer("gray_weight", gray_weight.view(1, 3, 1, 1))

    def _build_sobel_filters(self) -> None:
        """Register Sobel gradient kernels."""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _build_ela_blur(self) -> None:
        """Build Gaussian blur kernel for ELA approximation.

        Real ELA does JPEG encode→decode→diff which is non-differentiable.
        We approximate it with: |image - blur(image)| which captures similar
        high-frequency residuals that indicate compression artifacts.
        """
        size = 7
        sigma = 2.0
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = (g.unsqueeze(0) * g.unsqueeze(1))
        kernel = kernel / kernel.sum()
        # Shape: (1, 1, 7, 7) for depthwise conv
        self.register_buffer("ela_kernel", kernel.unsqueeze(0).unsqueeze(0))
        self.ela_pad = size // 2

    def _compute_srm(self, x: torch.Tensor) -> torch.Tensor:
        """SRM noise residuals. Input: (B, 3, H, W), Output: (B, 1, H, W)."""
        # Convert to grayscale
        gray = (x * self.gray_weight).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        # Apply SRM filters
        residuals = F.conv2d(gray, self.srm_weight, padding=2)  # (B, 3, H, W)
        # Mean absolute residual
        srm = residuals.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        # Normalize to [0, 1]
        B = srm.shape[0]
        srm_flat = srm.view(B, -1)
        srm_max = srm_flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        srm = srm / (srm_max + 1e-8)
        return srm

    def _compute_ela(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate ELA. Input: (B, 3, H, W), Output: (B, 1, H, W)."""
        # Apply Gaussian blur per channel (depthwise)
        B, C, H, W = x.shape
        x_flat = x.view(B * C, 1, H, W)
        blurred = F.conv2d(x_flat, self.ela_kernel, padding=self.ela_pad)
        blurred = blurred.view(B, C, H, W)
        # ELA = mean absolute difference across channels
        ela = (x - blurred).abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        # Scale and clamp
        ela = (ela * 15.0).clamp(0, 1)
        return ela

    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Sobel gradient magnitude. Input: (B, 3, H, W), Output: (B, 1, H, W)."""
        gray = (x * self.gray_weight).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2)
        # Normalize to [0, 1]
        B = mag.shape[0]
        mag_flat = mag.view(B, -1)
        mag_max = mag_flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        mag = mag / (mag_max + 1e-8)
        return mag

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all 3 forensic channels.

        Parameters
        ----------
        x : Tensor (B, 3, H, W), float32 in [0, 1]

        Returns
        -------
        Tensor (B, 3, H, W), forensic features in [0, 1]
        """
        srm = self._compute_srm(x)
        ela = self._compute_ela(x)
        grad = self._compute_gradient(x)
        return torch.cat([srm, ela, grad], dim=1)
