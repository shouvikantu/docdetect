"""
preprocess.py — Convert raw images to 6-channel tensors (RGB + Forensic).

Reads raw image files and their corresponding masks, computes the 3-channel
forensic feature map (SRM, ELA, Gradient), stacks it with the RGB channels
to form a 6-channel tensor, and saves the result as ``.npy`` or ``.pth``
for fast loading during training.

Usage
-----
    python -m data.preprocess \
        --image_dir /path/to/images \
        --mask_dir  /path/to/masks  \
        --output_dir /path/to/processed \
        --image_size 512 \
        --format npy
"""

import argparse
import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Relative import when running as part of the package
from utils.forensics import build_forensic_channels


def preprocess_single(
    image_path: str,
    mask_path: str,
    image_size: int = 512,
) -> dict:
    """Load one image + mask and produce the 6-channel tensor.

    Returns
    -------
    dict
        ``{"tensor": np.ndarray (6, H, W), "mask": np.ndarray (1, H, W)}``
    """
    # Load image (BGR) and mask
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    # Resize
    image = cv2.resize(image, (image_size, image_size))
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    # Compute forensic channels (H, W, 3)
    forensic = build_forensic_channels(image)

    # Convert RGB to float [0, 1] and channel-first
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))  # (3, H, W)

    # Forensic channels to channel-first
    forensic = np.transpose(forensic, (2, 0, 1))  # (3, H, W)

    # Stack → (6, H, W)
    tensor_6ch = np.concatenate([rgb, forensic], axis=0)

    # Mask → (1, H, W), binarise
    mask_bin = (mask > 127).astype(np.float32)[np.newaxis, ...]

    return {"tensor": tensor_6ch, "mask": mask_bin}


def preprocess_dataset(
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    image_size: int = 512,
    fmt: Literal["npy", "pth"] = "npy",
) -> None:
    """Batch-preprocess an entire directory of images + masks.

    The output directory mirrors the input structure. Each sample is saved as
    either a ``.npy`` file (two arrays: tensor & mask) or a ``.pth`` file
    (dict of torch tensors).

    Parameters
    ----------
    image_dir : str
        Directory containing raw images.
    mask_dir : str
        Directory containing corresponding binary masks (same basenames).
    output_dir : str
        Where to write processed tensors.
    image_size : int
        Spatial size to resize images to.
    fmt : str
        ``"npy"`` or ``"pth"``.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(Path(image_dir).glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}]

    print(f"Found {len(image_paths)} images in {image_dir}")

    for img_path in tqdm(image_paths, desc="Preprocessing"):
        # Find matching mask
        mask_candidates = [
            Path(mask_dir) / img_path.name,
            Path(mask_dir) / img_path.with_suffix(".png").name,
        ]
        mask_path = None
        for mc in mask_candidates:
            if mc.exists():
                mask_path = mc
                break

        if mask_path is None:
            print(f"  [SKIP] No mask found for {img_path.name}")
            continue

        result = preprocess_single(str(img_path), str(mask_path), image_size)

        stem = img_path.stem
        if fmt == "npy":
            out_file = Path(output_dir) / f"{stem}.npz"
            np.savez_compressed(
                str(out_file),
                tensor=result["tensor"],
                mask=result["mask"],
            )
        else:
            out_file = Path(output_dir) / f"{stem}.pth"
            torch.save(
                {
                    "tensor": torch.from_numpy(result["tensor"]),
                    "mask": torch.from_numpy(result["mask"]),
                },
                str(out_file),
            )

    print(f"Done. Processed files saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess raw images into 6-channel forensic tensors.",
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Path to raw images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to GT masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--image_size", type=int, default=512, help="Resize target (default: 512)")
    parser.add_argument("--format", type=str, default="npy", choices=["npy", "pth"], help="Output format")
    args = parser.parse_args()

    preprocess_dataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
