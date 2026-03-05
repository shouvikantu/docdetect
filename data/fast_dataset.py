"""
fast_dataset.py — Lightweight LMDB Dataset for GPU-forensics training.

Skips all CPU-side forensic feature computation. Returns only RGB (3ch)
images + masks. Forensic features are computed on GPU inside the model.

This is ~5-10x faster than the standard lmdb_dataset.py pipeline.
"""

from typing import Dict, Optional

import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.augmentations import get_train_augmentations, get_val_augmentations


class FastLMDBDataset(Dataset):
    """Lightweight LMDB dataset — returns 3ch RGB + mask only.

    Parameters
    ----------
    lmdb_path : str
        Path to the LMDB directory.
    image_size : int
        Target spatial size (H = W).
    split : str
        ``"train"`` or ``"val"``.
    transform : callable, optional
        Override default augmentation pipeline.
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

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_augmentations(image_size)
        else:
            self.transform = get_val_augmentations(image_size)

        # Read num-samples once
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            num_bytes = txn.get(b"num-samples")
            if num_bytes is None:
                raise RuntimeError(f"LMDB at {lmdb_path} has no 'num-samples' key.")
            self.num_samples = int(num_bytes)
        env.close()
        self._env = None

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False,
                readahead=False, meminit=False,
            )
        return self._env

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        env = self._get_env()

        with env.begin(write=False) as txn:
            img_key = f"image-{idx:09d}".encode("utf-8")
            img_buf = txn.get(img_key)
            if img_buf is None:
                raise KeyError(f"Image key {img_key} not found")

            img_arr = np.frombuffer(img_buf, dtype=np.uint8)
            image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to decode image at index {idx}")

            lbl_key = f"label-{idx:09d}".encode("utf-8")
            lbl_buf = txn.get(lbl_key)
            if lbl_buf is not None:
                mask_arr = np.frombuffer(lbl_buf, dtype=np.uint8)
                mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Augment
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        # Ensure size
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_NEAREST)

        # RGB float32 tensor — NO forensic computation
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))  # (3, H, W)
        mask_bin = (mask > 127).astype(np.float32)[np.newaxis, ...]  # (1, H, W)

        return {
            "image": torch.from_numpy(rgb),
            "mask": torch.from_numpy(mask_bin),
            "index": idx,
        }
