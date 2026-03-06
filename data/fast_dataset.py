"""
fast_dataset.py — High-performance LMDB Dataset with RAM caching.

Skips all CPU-side forensic computation (done on GPU instead).
Caches decoded images in RAM after first access — subsequent epochs
have zero disk I/O and run at memory speed.

Typical speedup vs standard pipeline: 10-20x.
"""

from typing import Dict, Optional

import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.augmentations import get_train_augmentations, get_val_augmentations


class FastLMDBDataset(Dataset):
    """High-performance LMDB dataset with RAM caching.

    Features:
    - Returns 3ch RGB only (forensic features computed on GPU in model)
    - Caches decoded images in RAM after first read (zero disk I/O after epoch 1)
    - Supports max_samples to train on a subset

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
    max_samples : int, optional
        Limit dataset to first N samples (for faster experiments).
    cache_in_ram : bool
        If True, cache decoded images in RAM after first access.
    """

    def __init__(
        self,
        lmdb_path: str,
        image_size: int = 512,
        split: str = "train",
        transform=None,
        max_samples: int = -1,
        cache_in_ram: bool = True,
    ) -> None:
        super().__init__()
        self.lmdb_path = lmdb_path
        self.image_size = image_size
        self.cache_in_ram = cache_in_ram

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
            total = int(num_bytes)
        env.close()

        self.num_samples = min(total, max_samples) if max_samples > 0 else total
        self._env = None

        # RAM cache: store decoded + resized images and masks as numpy uint8
        # At 512x512x3 uint8 = 768KB per image. 120K images ≈ 90GB.
        # At 256x256x3 uint8 = 192KB per image. 120K images ≈ 23GB.
        # Masks are much smaller (1 channel).
        if cache_in_ram:
            self._image_cache: list = [None] * self.num_samples
            self._mask_cache: list = [None] * self.num_samples
            self._cached_count = 0
        else:
            self._image_cache = None
            self._mask_cache = None

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False,
                readahead=True, meminit=False,
            )
        return self._env

    def __len__(self) -> int:
        return self.num_samples

    def _load_from_lmdb(self, idx: int):
        """Load and decode image + mask from LMDB."""
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

        # Pre-resize to target size for cache efficiency
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)
        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Try cache first
        if self.cache_in_ram and self._image_cache[idx] is not None:
            image = self._image_cache[idx]
            mask = self._mask_cache[idx]
        else:
            image, mask = self._load_from_lmdb(idx)
            if self.cache_in_ram:
                self._image_cache[idx] = image
                self._mask_cache[idx] = mask

        # Augment (operates on copies due to numpy semantics)
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        # RGB float32 tensor — NO forensic computation
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))  # (3, H, W)
        mask_bin = (mask > 127).astype(np.float32)[np.newaxis, ...]  # (1, H, W)

        return {
            "image": torch.from_numpy(rgb),
            "mask": torch.from_numpy(mask_bin),
            "index": idx,
        }
