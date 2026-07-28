"""
Evaluate the trained PPE model and collect artefacts for the report.

Produces:
  * Console table of mAP@0.5, mAP@0.5:0.95, precision, recall (overall + per class)
  * Confusion matrix + PR/F1 curves (Ultralytics saves these automatically)
  * A grid of predictions on unseen test images, copied into reports/

Usage:
    python src/evaluate.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "best.pt"
POINTER = ROOT / "data" / "ACTIVE_DATA_YAML.txt"
REPORTS = ROOT / "reports"


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS} not found. Train first: python src/train.py")

    data_yaml = POINTER.read_text().strip()
    model = YOLO(str(WEIGHTS))

    # Validate on the test split (falls back to val if no test defined).
    print("Running validation ...\n")
    metrics = model.val(data=data_yaml, split="test", plots=True, device=0)

    # Headline numbers.
    print("\n" + "=" * 50)
    print("OVERALL METRICS")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  precision    : {metrics.box.mp:.4f}")
    print(f"  recall       : {metrics.box.mr:.4f}")
    print("=" * 50)

    # Per-class breakdown — where the interesting failure story usually hides.
    names = model.names
    print("\nPER-CLASS mAP@0.5:")
    for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
        print(f"  {names[i]:14s}: {ap:.4f}")

    # Copy the plots Ultralytics generated into reports/ for the writeup.
    val_dir = Path(metrics.save_dir)
    REPORTS.mkdir(exist_ok=True)
    for plot in ("confusion_matrix.png", "PR_curve.png", "F1_curve.png",
                 "confusion_matrix_normalized.png"):
        src = val_dir / plot
        if src.exists():
            shutil.copy(src, REPORTS / plot)
    print(f"\nPlots copied to {REPORTS}")
    print(f"(full validation output in {val_dir})")


if __name__ == "__main__":
    main()
