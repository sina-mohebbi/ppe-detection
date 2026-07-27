"""
Train a YOLO11 PPE detector — tuned for an 8 GB laptop GPU (RTX 4060).

The defaults here are deliberately chosen for a memory-constrained GPU:

  * model = yolo11s   -> small model, fits 8 GB comfortably with room to spare
  * amp   = True      -> mixed precision (FP16); roughly halves VRAM use + faster
  * batch = 16        -> safe for yolo11s @ 640px on 8 GB (drop to 8 if you OOM)
  * imgsz = 640       -> standard; lower to 512/416 is the biggest VRAM lever
  * cache = 'disk'    -> cache resized images to DISK, not RAM (protects 16 GB RAM)
  * workers = 8       -> dataloader workers; kept modest for Windows
  * patience = 20     -> early-stop a plateaued run so we don't waste GPU time

Usage:
    python src/train.py                      # sensible defaults
    python src/train.py --epochs 60 --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data" / "ACTIVE_DATA_YAML.txt"


def get_data_yaml() -> str:
    if not POINTER.exists():
        raise FileNotFoundError(
            "data/ACTIVE_DATA_YAML.txt missing. Run: python src/prepare_data.py"
        )
    return POINTER.read_text().strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO11 PPE detector")
    p.add_argument("--model", default="yolo11s.pt", help="base weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="lower to 8 if you hit OOM")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--name", default="ppe_yolo11s", help="run name under runs/")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Confirm we're actually on the GPU — training on CPU by accident is the
    # classic "why is this so slow" trap.
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Training on GPU: {gpu} ({vram:.1f} GB VRAM)")
        device = 0
    else:
        print("WARNING: CUDA not available — falling back to CPU (very slow).")
        device = "cpu"

    data_yaml = get_data_yaml()
    print(f"Dataset: {data_yaml}\n")

    model = YOLO(args.model)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=device,
        amp=True,              # mixed precision — key VRAM saver
        cache="disk",          # cache to disk, not RAM
        project=str(ROOT / "runs"),
        name=args.name,
        plots=True,            # saves confusion matrix, PR curves, etc.
        seed=42,               # reproducible runs
    )

    # Copy the best weights somewhere predictable for the rest of the pipeline.
    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    if best.exists():
        dest = ROOT / "models" / "best.pt"
        dest.write_bytes(best.read_bytes())
        print(f"\nBest weights -> {dest}")
    print("\nTraining complete. Next: python src/evaluate.py")


if __name__ == "__main__":
    main()
