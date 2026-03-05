"""
train.py — Training script for the DocTamper forensic detection model.

Usage
-----
    python train.py \
        --train_lmdb dataset/DocTamperV1-TrainingSet \
        --val_lmdb   dataset/DocTamperV1-TestingSet \
        --epochs 50 \
        --batch_size 8 \
        --image_size 512 \
        --lr 1e-4
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from data.lmdb_dataset import DocTamperLMDBDataset
from models.swin_forensic import SwinForensic


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice loss with configurable weighting."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict:
    """Compute segmentation metrics for a batch."""
    preds = (torch.sigmoid(logits) > threshold).float()
    targets_bin = (targets > 0.5).float()

    preds_flat = preds.view(-1)
    targets_flat = targets_bin.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-8)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item(),
    }


# ---------------------------------------------------------------------------
# Training & Validation Loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    max_steps: int = -1,
    grad_clip: float = 1.0,
    amp_dtype: torch.dtype = torch.float16,
) -> dict:
    """Train for one epoch. Returns average loss & metrics."""
    model.train()
    running_loss = 0.0
    running_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, batch in enumerate(pbar):
        if max_steps > 0 and step >= max_steps:
            break

        images = batch["image"].to(device)  # (B, 6, H, W)
        masks = batch["mask"].to(device)    # (B, 1, H, W)

        optimizer.zero_grad()

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            output = model(images)
            loss = criterion(output["seg"], masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        metrics = compute_metrics(output["seg"], masks)
        running_loss += loss.item()
        for k in running_metrics:
            running_metrics[k] += metrics[k]
        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", f1=f"{metrics['f1']:.4f}")

    avg_loss = running_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in running_metrics.items()}
    avg_metrics["loss"] = avg_loss
    return avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_steps: int = -1,
    amp_dtype: torch.dtype = torch.float16,
) -> dict:
    """Validate the model. Returns average loss & metrics."""
    model.eval()
    running_loss = 0.0
    running_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)
    for step, batch in enumerate(pbar):
        if max_steps > 0 and step >= max_steps:
            break

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            output = model(images)
            loss = criterion(output["seg"], masks)

        metrics = compute_metrics(output["seg"], masks)
        running_loss += loss.item()
        for k in running_metrics:
            running_metrics[k] += metrics[k]
        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", f1=f"{metrics['f1']:.4f}")

    avg_loss = running_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in running_metrics.items()}
    avg_metrics["loss"] = avg_loss
    return avg_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DocTamper detection model")

    # Data
    parser.add_argument("--train_lmdb", type=str, required=True, help="Path to training LMDB")
    parser.add_argument("--val_lmdb", type=str, required=True, help="Path to validation LMDB")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Model
    parser.add_argument(
        "--backbone", type=str, default="swinv2_tiny_window8_256",
        help="timm backbone name",
    )
    parser.add_argument("--fpn_dim", type=int, default=256, help="FPN hidden dimension")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet pretrain")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--bce_weight", type=float, default=0.5, help="BCE loss weight")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="Dice loss weight")

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Checkpoint dir")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Performance
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.0+, ~20-30%% speedup)")

    # Debug
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps per epoch (debug)")

    args = parser.parse_args()

    # --- Device & Precision ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Auto-detect BF16 support (Ampere+ GPUs: A100, RTX 30xx, etc.)
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print(f"Using device: {device} (BF16 — Ampere+ detected)")
        else:
            amp_dtype = torch.float16
            print(f"Using device: {device} (FP16)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        amp_dtype = torch.float32  # MPS doesn't support AMP well
        print(f"Using device: {device} (FP32)")
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32
        print(f"Using device: {device} (FP32)")

    # --- Datasets ---
    print("Loading training dataset...")
    train_ds = DocTamperLMDBDataset(
        lmdb_path=args.train_lmdb,
        image_size=args.image_size,
        split="train",
    )
    print(f"  Training samples: {len(train_ds)}")

    print("Loading validation dataset...")
    val_ds = DocTamperLMDBDataset(
        lmdb_path=args.val_lmdb,
        image_size=args.image_size,
        split="val",
    )
    print(f"  Validation samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    print(f"Creating model: {args.backbone}")
    model = SwinForensic(
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        fpn_dim=args.fpn_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {num_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # --- torch.compile ---
    if args.compile:
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)
        print("  ✓ Model compiled (first batch will be slow due to compilation)")

    # --- Loss, Optimizer, Scheduler ---
    criterion = CombinedLoss(
        bce_weight=args.bce_weight, dice_weight=args.dice_weight
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    # GradScaler only needed for FP16; BF16 and FP32 don't need it
    use_scaler = (device.type == "cuda" and amp_dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    start_epoch = 0
    best_f1 = 0.0

    # --- Resume ---
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best F1 = {best_f1:.4f}")

    # --- TensorBoard ---
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_log_dir = os.path.join(args.output_dir, "tb_logs")
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"  TensorBoard logs: {tb_log_dir}")
    except ImportError:
        print("  TensorBoard not available, skipping.")

    # --- Output dir ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Training Loop ---
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, max_steps=args.max_steps, grad_clip=args.grad_clip,
            amp_dtype=amp_dtype,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch,
            max_steps=args.max_steps, amp_dtype=amp_dtype,
        )

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0

        # Log
        print(
            f"Epoch {epoch:3d} | "
            f"train loss: {train_metrics['loss']:.4f}  F1: {train_metrics['f1']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f}  F1: {val_metrics['f1']:.4f}  "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"lr: {current_lr:.2e} | {elapsed:.0f}s"
        )

        if tb_writer is not None:
            for k, v in train_metrics.items():
                tb_writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                tb_writer.add_scalar(f"val/{k}", v, epoch)
            tb_writer.add_scalar("lr", current_lr, epoch)

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_f1": best_f1,
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        # Save latest
        torch.save(ckpt, os.path.join(args.output_dir, "latest.pth"))

        # Save best
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            ckpt["best_f1"] = best_f1
            torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
            print(f"  ✓ New best F1: {best_f1:.4f}")

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
