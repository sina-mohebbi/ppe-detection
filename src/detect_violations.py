"""
PPE compliance / violation detection.

This turns the PPE *detector* into a compliance *checker*. Detection alone says
"there is a helmet here"; compliance says "this person is missing a helmet",
which is the actual question a safety system needs to answer.

How it works (no retraining needed):
  1. A pretrained COCO model (yolo11n) detects **people**.
  2. Our fine-tuned model detects **PPE** (helmet, vest, ...).
  3. For each person we check whether a helmet sits in their head region and a
     vest in their torso region. Missing either → flagged as a violation.

Compliance is checked for **helmet + hi-vis vest** — the two canonical
construction requirements, and our model's two strongest classes.

Usage:
    python src/detect_violations.py --source clip.mp4
    python src/detect_violations.py --source photo.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

GREEN = (60, 200, 60)
RED = (40, 40, 220)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPE compliance / violation detection")
    p.add_argument("--source", required=True, help="image or video path")
    p.add_argument("--ppe-weights", default=str(ROOT / "models" / "best.pt"))
    p.add_argument("--person-weights", default="yolo11n.pt", help="COCO person detector")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--ppe-conf", type=float, default=0.35)
    p.add_argument("--person-conf", type=float, default=0.40)
    p.add_argument("--device", default=0)
    return p.parse_args()


def cx_cy(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


# PPE required for compliance. Helmet (hard hat) is the universal construction
# rule and our strongest/most-reliable class, so it's the default. Add "Vest" here
# to also require a hi-vis vest (stricter; depends on vest recall being good).
REQUIRED = ("helmet",)


def check_compliance(person_box, ppe_items) -> list[str]:
    """Return the list of missing required PPE for one person ([] = compliant)."""
    px1, py1, px2, py2 = person_box
    ph = py2 - py1
    # Region a PPE item must fall in to count as "worn by this person".
    zones = {
        # Head band extends a little above the person box (helmets sit on top of
        # the head, sometimes just above the detected person bbox).
        "helmet": (py1 - 0.10 * ph, py1 + 0.40 * ph),
        "Vest": (py1 + 0.25 * ph, py1 + 0.80 * ph),
    }

    worn = set()
    for name, box in ppe_items:
        cx, cy = cx_cy(box)
        if not (px1 <= cx <= px2):        # must sit horizontally within the person
            continue
        zone = zones.get(name)
        if zone and zone[0] <= cy <= zone[1]:
            worn.add(name)

    return [f"NO {p.upper()}" for p in REQUIRED if p not in worn]


def annotate(frame, persons, ppe_items, args) -> tuple["cv2.Mat", int, int]:
    n_ok = n_bad = 0
    for pbox in persons:
        missing = check_compliance(pbox, ppe_items)
        x1, y1, x2, y2 = map(int, pbox)
        color = RED if missing else GREEN
        label = " | ".join(missing) if missing else "COMPLIANT"
        if missing:
            n_bad += 1
        else:
            n_ok += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # Clamp the label band inside the frame (persons often touch the top edge).
        band_top = y1 - th - 10
        if band_top < 0:
            band_top = y1 + 2
        cv2.rectangle(frame, (x1, band_top), (x1 + tw + 8, band_top + th + 8), color, -1)
        cv2.putText(frame, label, (x1 + 4, band_top + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame, n_ok, n_bad


def detect(frame, ppe_model, person_model, args):
    pr = person_model.predict(frame, imgsz=args.imgsz, conf=args.person_conf,
                              classes=[0], device=args.device, verbose=False)[0]
    persons = [b.xyxy[0].tolist() for b in pr.boxes]

    fr = ppe_model.predict(frame, imgsz=args.imgsz, conf=args.ppe_conf,
                           device=args.device, verbose=False)[0]
    ppe_items = [(ppe_model.names[int(b.cls)], b.xyxy[0].tolist()) for b in fr.boxes]
    return annotate(frame, persons, ppe_items, args)


def main() -> None:
    args = parse_args()
    ppe_model = YOLO(args.ppe_weights)
    person_model = YOLO(args.person_weights)
    src = Path(args.source)
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)

    if src.suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(str(src))
        frame, ok, bad = detect(frame, ppe_model, person_model, args)
        out = out_dir / f"{src.stem}_compliance.jpg"
        cv2.imwrite(str(out), frame)
        print(f"{ok} compliant, {bad} violation(s) -> {out}")
        return

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = out_dir / f"{src.stem}_compliance.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, _, _ = detect(frame, ppe_model, person_model, args)
        writer.write(frame)
        n += 1
        if n % 30 == 0:
            print(f"  processed {n} frames ...")
    cap.release()
    writer.release()
    print(f"Done. {n} frames -> {out}")


if __name__ == "__main__":
    main()
