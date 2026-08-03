"""
Per-class confidence-threshold tuning.

A detector gives every box a confidence score, and you keep boxes above a
threshold. Using one global threshold for all classes is suboptimal: an
easy/large class and a small/rare class have very different ideal operating
points. This script finds the threshold that maximises F1 **per class** on the
validation set, then measures the effect on the held-out test set.

Methodology (no data leakage):
  1. Run the model on val at a very low conf to gather every candidate box.
  2. Match predictions to ground truth (IoU >= 0.5, greedy by confidence).
  3. For each class, sweep thresholds and pick the one with the best F1.
  4. Apply those val-tuned thresholds to the *test* set and compare against a
     single global threshold.

Usage:
    python src/tune_thresholds.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "best.pt"
NAMES = ["Gloves", "Vest", "goggles", "helmet", "mask", "safety_shoe"]
IMGSZ = 640
GLOBAL_THR = 0.35              # baseline: one threshold for everything
GRID = np.round(np.arange(0.05, 0.90, 0.01), 2)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(stem: str, lbl_dir: Path, w: int, h: int):
    p = lbl_dir / f"{stem}.txt"
    boxes = []
    if p.exists():
        for line in p.read_text().strip().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            c, xc, yc, bw, bh = int(float(t[0])), *map(float, t[1:5])
            boxes.append((c, [(xc - bw / 2) * w, (yc - bh / 2) * h,
                              (xc + bw / 2) * w, (yc + bh / 2) * h]))
    return boxes


def collect(model, img_dir: Path, lbl_dir: Path):
    """Return per-class list of (confidence, is_true_positive) and GT counts."""
    per = {i: [] for i in range(len(NAMES))}
    gt_counts = {i: 0 for i in range(len(NAMES))}
    for r in model.predict(source=str(img_dir), stream=True, conf=0.001,
                           imgsz=IMGSZ, device=0, verbose=False):
        h, w = r.orig_shape
        gt = load_gt(Path(r.path).stem, lbl_dir, w, h)
        for c, _ in gt:
            gt_counts[c] += 1
        preds = sorted(
            [(int(b.cls), float(b.conf), b.xyxy[0].tolist()) for b in r.boxes],
            key=lambda x: -x[1],
        )
        used = set()
        for pc, conf, pb in preds:
            best_j, best_iou = -1, 0.5
            for j, (gc, gb) in enumerate(gt):
                if j in used or gc != pc:
                    continue
                v = iou(pb, gb)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                used.add(best_j)
                per[pc].append((conf, True))
            else:
                per[pc].append((conf, False))
    return per, gt_counts


def prf(per_c, n_gt, thr):
    tp = sum(1 for conf, is_tp in per_c if conf >= thr and is_tp)
    fp = sum(1 for conf, is_tp in per_c if conf >= thr and not is_tp)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def best_thresholds(per, gt_counts):
    out = {}
    for c in range(len(NAMES)):
        best = (GLOBAL_THR, 0.0)
        for thr in GRID:
            _, _, f1 = prf(per[c], gt_counts[c], thr)
            if f1 > best[1]:
                best = (float(thr), f1)
        out[c] = best[0]
    return out


def macro_f1(per, gt_counts, thr_fn):
    f1s = []
    for c in range(len(NAMES)):
        _, _, f1 = prf(per[c], gt_counts[c], thr_fn(c))
        f1s.append(f1)
    return float(np.mean(f1s)), f1s


def main():
    model = YOLO(str(WEIGHTS))

    print("Gathering validation predictions ...")
    val_per, val_gt = collect(model, ROOT / "data" / "valid" / "images",
                              ROOT / "data" / "valid" / "labels")
    thr = best_thresholds(val_per, val_gt)

    print("Gathering test predictions ...")
    test_per, test_gt = collect(model, ROOT / "data" / "test" / "images",
                                ROOT / "data" / "test" / "labels")

    print("\n== Tuned per-class thresholds (from val) & test F1 ==")
    print(f"{'class':12s} | {'thr':>4s} | {'F1@0.35':>7s} | {'F1@tuned':>8s}")
    for c in range(len(NAMES)):
        _, _, f1_global = prf(test_per[c], test_gt[c], GLOBAL_THR)
        _, _, f1_tuned = prf(test_per[c], test_gt[c], thr[c])
        print(f"{NAMES[c]:12s} | {thr[c]:4.2f} | {f1_global:7.3f} | {f1_tuned:8.3f}")

    m_global, _ = macro_f1(test_per, test_gt, lambda c: GLOBAL_THR)
    m_tuned, _ = macro_f1(test_per, test_gt, lambda c: thr[c])
    print(f"\nMacro-F1 on test:  global 0.35 = {m_global:.3f}   tuned = {m_tuned:.3f}")


if __name__ == "__main__":
    main()
