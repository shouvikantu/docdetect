"""
dataset.py — PyTorch Dataset that loads pre-computed 6-channel tensors.

Expects data produced by ``preprocess.py`` — either ``.npz`` (NumPy) or
``.pth`` (PyTorch) files, each containing:
    - ``tensor``: float32 array of shape (6, H, W)
    - ``mask``:   float32 array of shape (1, H, W)
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class DocDetectDataset(Dataset):
    """Dataset for loading pre-processed 6-channel forensic tensors.

    Parameters
    ----------
    data_dir : str
        Directory containing ``.npz`` or ``.pth`` files.
    split_file : str, optional
        Path to a text file listing basenames (without extension) to include.
        If ``None``, all files in ``data_dir`` are used.
    transform : callable, optional
        Additional tensor-level transform applied after loading
        (e.g., normalisation, random erasing).
    """

    SUPPORTED_EXTENSIONS = {".npz", ".pth"}

    def __init__(
        self,
        data_dir: str,
        split_file: Optional[str] = None,
        transform=None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: List[Path] = self._discover_samples(split_file)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No .npz or .pth files found in {data_dir}. "
                "Run preprocess.py first."
            )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_samples(self, split_file: Optional[str]) -> List[Path]:
        """Find all valid sample files, optionally filtered by a split list."""
        all_files = sorted(
            p for p in self.data_dir.iterdir()
            if p.suffix in self.SUPPORTED_EXTENSIONS
        )

        if split_file is not None:
            allowed = set(Path(split_file).read_text().strip().splitlines())
            all_files = [p for p in all_files if p.stem in allowed]

        return all_files

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_npz(path: Path) -> Dict[str, torch.Tensor]:
        data = np.load(str(path))
        return {
            "tensor": torch.from_numpy(data["tensor"]),
            "mask": torch.from_numpy(data["mask"]),
        }

    @staticmethod
    def _load_pth(path: Path) -> Dict[str, torch.Tensor]:
        return torch.load(str(path), map_location="cpu", weights_only=True)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.samples[idx]

        if path.suffix == ".npz":
            sample = self._load_npz(path)
        else:
            sample = self._load_pth(path)

        tensor_6ch: torch.Tensor = sample["tensor"]  # (6, H, W)
        mask: torch.Tensor = sample["mask"]            # (1, H, W)

        if self.transform is not None:
            tensor_6ch = self.transform(tensor_6ch)

        return {"image": tensor_6ch, "mask": mask, "name": path.stem}
