"""
lmdb_dataset.py — PyTorch Dataset that reads directly from DocTamper LMDB files.

Decodes images and masks on-the-fly, applies augmentations, computes
forensic channels (SRM, ELA, Gradient), and returns 6-channel tensors
ready for training.

Key design choices:
    - LMDB env is opened lazily (on first __getitem__) so the dataset
      is safely picklable for DataLoader with num_workers > 0.
    - Augmentations are applied to the raw image + mask BEFORE
      computing forensic features, so the forensic channels reflect
      the augmented (realistic) image rather than augmented noise maps.
"""

import io
from typing import Dict, Optional

import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.augmentations import get_train_augmentations, get_val_augmentations
from utils.forensics import build_forensic_channels


class DocTamperLMDBDataset(Dataset):
    """Dataset that reads image + mask pairs from a DocTamper LMDB database.

    Parameters
    ----------
    lmdb_path : str
        Path to the LMDB directory (containing ``data.mdb``).
    image_size : int
        Target spatial size (H = W) for resizing.
    split : str
        ``"train"`` or ``"val"`` — controls which augmentation pipeline is used.
    transform : callable, optional
        Override the default augmentation with a custom albumentations pipeline.
    """

    def __init__(
        self,
        lmdb_path: str,
        image_size: int = 512,
        split: str = "train",
        transform=None,
    ) -> None:
        super().__init__()
        self.lmdb_path = lmdb_path
        self.image_size = image_size
        self.split = split

        # Set up augmentation pipeline
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_augmentations(image_size)
        else:
            self.transform = get_val_augmentations(image_size)

        # Read num-samples from the LMDB *once* at init time
        # (this is a lightweight read that doesn't hold the env open)
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            num_bytes = txn.get(b"num-samples")
            if num_bytes is None:
                raise RuntimeError(
                    f"LMDB at {lmdb_path} has no 'num-samples' key. "
                    "Is this a valid DocTamper dataset?"
                )
            self.num_samples = int(num_bytes)
        env.close()

        # Env will be opened lazily per worker
        self._env = None

    def _get_env(self) -> lmdb.Environment:
        """Lazily open the LMDB env (fork-safe for DataLoader workers)."""
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._env

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        env = self._get_env()

        with env.begin(write=False) as txn:
            # --- Read image ---
            img_key = f"image-{idx:09d}".encode("utf-8")
            img_buf = txn.get(img_key)
            if img_buf is None:
                raise KeyError(f"Image key {img_key} not found in LMDB")

            img_arr = np.frombuffer(img_buf, dtype=np.uint8)
            image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)  # BGR, (H, W, 3)

            if image is None:
                raise RuntimeError(f"Failed to decode image at index {idx}")

            # --- Read mask ---
            lbl_key = f"label-{idx:09d}".encode("utf-8")
            lbl_buf = txn.get(lbl_key)

            if lbl_buf is not None:
                mask_arr = np.frombuffer(lbl_buf, dtype=np.uint8)
                mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)  # (H, W)
                if mask is None:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                # No mask → all-authentic (zeros)
                mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # --- Augment (on raw BGR image + mask) ---
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]  # (H, W, 3), uint8
        mask = augmented["mask"]    # (H, W), uint8

        # Ensure consistent size after augmentation
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(
                mask, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

        # --- Compute forensic channels ---
        forensic = build_forensic_channels(image)  # (H, W, 3), float32 [0, 1]

        # --- Convert to tensors ---
        # RGB: (H, W, 3) BGR→RGB, float32 [0, 1], channel-first
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))  # (3, H, W)

        forensic = np.transpose(forensic, (2, 0, 1))  # (3, H, W)

        # Stack → (6, H, W)
        tensor_6ch = np.concatenate([rgb, forensic], axis=0)

        # Mask → (1, H, W), binarise
        mask_bin = (mask > 127).astype(np.float32)[np.newaxis, ...]

        return {
            "image": torch.from_numpy(tensor_6ch),
            "mask": torch.from_numpy(mask_bin),
            "index": idx,
        }
