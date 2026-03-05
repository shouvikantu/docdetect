"""
swin_forensic.py — Swin Transformer V2 adapted for 6-channel forensic input.

Takes a 6-channel tensor (3 RGB + 3 forensic: SRM, ELA, Gradient) and
produces a pixel-level binary segmentation mask predicting tampered regions.

Architecture
------------
1.  A custom patch-embedding layer that accepts 6 input channels.
2.  A pretrained Swin-V2 backbone (with the first projection layer replaced).
3.  A UPerNet-style FPN decoder that fuses multi-scale features.
4.  A lightweight segmentation head (1×1 conv → sigmoid).

The model is designed so that the 3 RGB channels can be initialised from
ImageNet-pretrained weights (by repeating-then-averaging), while the 3
forensic channels start from scratch.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from utils.gpu_forensics import GPUForensicFeatures


# ---------------------------------------------------------------------------
# Helper: Lateral + FPN decoder (UPerNet-style)
# ---------------------------------------------------------------------------

class FPNDecoder(nn.Module):
    """Simple Feature Pyramid Network decoder for semantic segmentation.

    Accepts multi-scale feature maps from the backbone and produces a single
    fused feature map at 1/4 resolution, then upsamples to full resolution.
    """

    def __init__(
        self,
        in_channels: List[int],
        fpn_dim: int = 256,
    ) -> None:
        super().__init__()
        # Lateral 1×1 convs to project each stage to fpn_dim
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_dim, 1) for c in in_channels
        ])
        # Smooth 3×3 convs after addition
        self.smooths = nn.ModuleList([
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) for _ in in_channels
        ])
        # Final fusion: concatenate all levels → 1×1 conv
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * len(in_channels), fpn_dim, 1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        features : list of Tensor
            Multi-scale features from the backbone, ordered from high-res
            (stage 1) to low-res (stage 4).

        Returns
        -------
        Tensor
            Fused feature map, shape (B, fpn_dim, H/4, W/4).
        """
        # Lateral projections
        laterals = [l(f) for l, f in zip(self.laterals, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Smooth
        outs = [s(l) for s, l in zip(self.smooths, laterals)]

        # Upsample all to the largest resolution (stage 1)
        target_size = outs[0].shape[2:]
        upsampled = [
            F.interpolate(o, size=target_size, mode="bilinear", align_corners=False)
            for o in outs
        ]

        # Fuse
        return self.fuse(torch.cat(upsampled, dim=1))


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class SwinForensic(nn.Module):
    """Swin-V2 forensic document tampering detector.

    Parameters
    ----------
    backbone_name : str
        ``timm`` model name, e.g. ``"swinv2_tiny_window8_256"``.
    pretrained : bool
        Whether to load ImageNet-pretrained weights for the backbone.
    in_channels : int
        Number of input channels (default 6: RGB + forensic).
    fpn_dim : int
        Hidden dimension of the FPN decoder.
    num_classes : int
        Number of output segmentation classes (1 for binary).
    """

    def __init__(
        self,
        backbone_name: str = "swinv2_tiny_window8_256",
        pretrained: bool = True,
        in_channels: int = 6,
        fpn_dim: int = 256,
        num_classes: int = 1,
        gpu_forensics: bool = False,
    ) -> None:
        super().__init__()
        self.gpu_forensics = gpu_forensics

        # When gpu_forensics=True, input is 3ch RGB and we compute
        # forensic features on GPU, then concat to get 6ch internally.
        if gpu_forensics:
            self.forensic_module = GPUForensicFeatures()
            in_channels = 6  # Always 6ch to the backbone

        # --- Backbone (feature extractor) ---
        # When pretrained, load with 3 channels first so we get proper ImageNet weights,
        # then adapt the first conv to accept 6 channels.
        load_in_chans = 3 if (pretrained and in_channels != 3) else in_channels
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=load_in_chans,
        )

        # Get channel dims for each stage from timm
        feature_info = self.backbone.feature_info.channels()

        # --- Adapt first conv if pretrained ---
        if pretrained and in_channels != 3:
            self._adapt_first_conv(in_channels)

        # --- FPN Decoder ---
        self.decoder = FPNDecoder(in_channels=feature_info, fpn_dim=fpn_dim)

        # --- Segmentation Head ---
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 4, 3, padding=1),
            nn.BatchNorm2d(fpn_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim // 4, num_classes, 1),
        )

    # ------------------------------------------------------------------
    # Weight adaptation
    # ------------------------------------------------------------------

    def _adapt_first_conv(self, in_channels: int) -> None:
        """Adapt the first convolutional layer for >3 input channels.

        Strategy: copy the pretrained 3-channel weights into the first 3
        channels and initialise the remaining channels with the mean of
        the pretrained weights (a reasonable starting point).
        """
        # Find the first conv layer in the backbone
        first_conv = None
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break

        if first_conv is None:
            return

        old_weight = first_conv.weight.data  # (out, 3, kH, kW)
        out_ch, _, kh, kw = old_weight.shape

        new_weight = torch.zeros(out_ch, in_channels, kh, kw)
        # Copy pretrained weights for RGB channels
        new_weight[:, :3, :, :] = old_weight
        # Init extra channels with mean of RGB weights
        mean_weight = old_weight.mean(dim=1, keepdim=True)
        for c in range(3, in_channels):
            new_weight[:, c : c + 1, :, :] = mean_weight

        first_conv.weight = nn.Parameter(new_weight)
        first_conv.in_channels = in_channels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, 6, H, W).
        return_features : bool
            If True, also return intermediate FPN features.

        Returns
        -------
        dict
            ``"seg"`` : (B, num_classes, H, W) — segmentation logits.
            ``"features"`` : Tensor (optional) — FPN fused features.
        """
        B, C, H, W = x.shape

        # If using GPU forensics, compute forensic channels on the fly
        if self.gpu_forensics and C == 3:
            with torch.no_grad():
                forensic = self.forensic_module(x)  # (B, 3, H, W)
            x = torch.cat([x, forensic], dim=1)     # (B, 6, H, W)
            C = 6

        # Multi-scale features from backbone (timm Swin outputs NHWC)
        features: List[torch.Tensor] = self.backbone(x)
        # Permute from NHWC to NCHW for conv layers
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        # Decode
        fpn_out = self.decoder(features)  # (B, fpn_dim, H', W')

        # Segmentation logits → upsample to input resolution
        seg_logits = self.seg_head(fpn_out)
        seg_logits = F.interpolate(
            seg_logits, size=(H, W), mode="bilinear", align_corners=False,
        )

        out = {"seg": seg_logits}
        if return_features:
            out["features"] = fpn_out

        return out


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test standard 6-channel mode
    model = SwinForensic(pretrained=False)
    dummy = torch.randn(2, 6, 256, 256)
    result = model(dummy)
    print(f"[6ch] Input: {dummy.shape} -> Output: {result['seg'].shape}")

    # Test GPU forensics mode (3-channel RGB input)
    model_fast = SwinForensic(pretrained=False, gpu_forensics=True)
    dummy_rgb = torch.randn(2, 3, 256, 256)
    result_fast = model_fast(dummy_rgb)
    print(f"[3ch+GPU] Input: {dummy_rgb.shape} -> Output: {result_fast['seg'].shape}")
    print(f"Parameters: {sum(p.numel() for p in model_fast.parameters()):,}")
