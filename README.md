# 🔍 DocDetect — Document Tampering Detection

A deep learning pipeline for detecting tampered regions in document images using a **dual-stream Swin Transformer V2** architecture with forensic frequency analysis.

Built on the [DocTamper](https://github.com/ayumiymk/DocTamperV1) dataset and inspired by state-of-the-art document forensics research.

---

## 🏗️ Architecture

```
Input Image (H × W × 3)
       │
       ├──► RGB Stream ──────────────────┐
       │                                 │
       └──► Forensic Feature Extraction  │
            ├── SRM (Noise Residuals)    ├──► 6-Channel Tensor (H × W × 6)
            ├── ELA (Error Levels)       │         │
            └── Gradient Magnitude ──────┘         ▼
                                          Swin Transformer V2
                                          (Pretrained Backbone)
                                                   │
                                              FPN Decoder
                                            (Multi-Scale Fusion)
                                                   │
                                            Segmentation Head
                                                   │
                                              Binary Mask
                                         (Tampered vs Authentic)
```

**Key components:**
- **Forensic Features** — SRM high-pass filters, Error Level Analysis, and Sobel gradient magnitude capture manipulation artifacts invisible to the naked eye
- **Swin-V2 Backbone** — Hierarchical vision transformer with shifted windows, initialized from ImageNet weights
- **FPN Decoder** — UPerNet-style feature pyramid that fuses multi-scale representations
- **Combined Loss** — BCE + Dice loss handles the severe class imbalance in tampering detection

---

## 📁 Project Structure

```
docdetect/
├── data/
│   ├── dataset.py          # Dataset for pre-computed .npz/.pth tensors
│   ├── lmdb_dataset.py     # LMDB-native dataset (reads DocTamper directly)
│   └── preprocess.py       # Offline preprocessing: raw images → 6-ch tensors
├── models/
│   └── swin_forensic.py    # SwinV2 + FPN decoder + segmentation head
├── utils/
│   ├── forensics.py        # SRM, ELA, Gradient feature extraction
│   └── augmentations.py    # Training & validation augmentation pipelines
├── train.py                # Full training script
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Setup Environment

```bash
conda create -n docdetect python=3.12 -y
conda activate docdetect
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the [DocTamperV1](https://github.com/ayumiymk/DocTamperV1) dataset and place the LMDB directories:

```
dataset/
├── DocTamperV1-TrainingSet/   # ~16 GB
│   ├── data.mdb
│   └── lock.mdb
└── DocTamperV1-TestingSet/    # ~4 GB
    ├── data.mdb
    └── lock.mdb
```

### 3. Train

```bash
python train.py \
  --train_lmdb dataset/DocTamperV1-TrainingSet \
  --val_lmdb dataset/DocTamperV1-TestingSet \
  --epochs 50 \
  --batch_size 8 \
  --image_size 512 \
  --lr 1e-4
```

**Smoke test** (2 steps, no pretrained weights):
```bash
python train.py \
  --train_lmdb dataset/DocTamperV1-TrainingSet \
  --val_lmdb dataset/DocTamperV1-TestingSet \
  --epochs 1 --batch_size 2 --image_size 256 \
  --max_steps 2 --no_pretrained --num_workers 0
```

---

## ⚙️ Training Options

| Argument | Default | Description |
|---|---|---|
| `--train_lmdb` | *required* | Path to training LMDB |
| `--val_lmdb` | *required* | Path to validation LMDB |
| `--image_size` | `512` | Input resolution (H=W) |
| `--batch_size` | `8` | Batch size |
| `--epochs` | `50` | Number of epochs |
| `--lr` | `1e-4` | Learning rate |
| `--backbone` | `swinv2_tiny_window8_256` | timm backbone name |
| `--fpn_dim` | `256` | FPN hidden dimension |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--grad_clip` | `1.0` | Gradient clipping norm |
| `--bce_weight` | `0.5` | BCE loss weight |
| `--dice_weight` | `0.5` | Dice loss weight |
| `--output_dir` | `checkpoints` | Checkpoint save directory |
| `--resume` | — | Resume from checkpoint path |
| `--no_pretrained` | `false` | Disable ImageNet pretrained weights |
| `--max_steps` | `-1` | Limit steps per epoch (for debugging) |

---

## 🧪 Forensic Features

Each input image is enriched with three forensic channels:

| Channel | Method | What it Detects |
|---|---|---|
| **SRM** | Spatial Rich Model (5×5 high-pass filters) | Noise inconsistencies from splicing/copy-move |
| **ELA** | Error Level Analysis (JPEG re-compression) | Regions saved at different compression levels |
| **Gradient** | Sobel edge magnitude | Boundary artifacts from cut-paste operations |

---

## 📊 Metrics

Training tracks the following per-epoch metrics:

- **F1 Score** — Primary metric, balances precision/recall
- **IoU** (Intersection over Union) — Measures overlap with ground truth
- **Precision** — Fraction of detected pixels that are truly tampered
- **Recall** — Fraction of tampered pixels that are detected

Best model is saved by validation F1. TensorBoard logs are written to `checkpoints/tb_logs/`.

---

## 🚀 Google Colab

To train with GPU on Colab Pro:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/shouvikantu/docdetect.git /content/docdetect
!ln -s /content/drive/MyDrive/docdetect/dataset /content/docdetect/dataset
!pip install -r /content/docdetect/requirements.txt

%cd /content/docdetect
!python train.py \
  --train_lmdb dataset/DocTamperV1-TrainingSet \
  --val_lmdb dataset/DocTamperV1-TestingSet \
  --epochs 50 --batch_size 8 --image_size 512 --lr 1e-4 \
  --output_dir /content/drive/MyDrive/docdetect/checkpoints
```

---

## 📄 License

MIT

---

## 🙏 Acknowledgments

- [DocTamperV1](https://github.com/ayumiymk/DocTamperV1) — Dataset
- [Swin Transformer V2](https://github.com/microsoft/Swin-Transformer) — Backbone architecture
- [timm](https://github.com/huggingface/pytorch-image-models) — Pretrained model hub
