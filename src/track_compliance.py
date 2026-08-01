"""
Temporal PPE-compliance monitoring with multi-object tracking.

Per-frame compliance (src/detect_violations.py) can only say "someone is missing
a helmet right now". A real monitoring system needs to follow each worker over
time: *which* worker, and for *how long*. This adds that using ByteTrack.

Pipeline (per frame):
  1. Track people across frames with ByteTrack -> each person gets a stable ID.
  2. Detect PPE with our model.
  3. Check helmet compliance per tracked person (reuses check_compliance()).
  4. Accumulate per-ID history -> report the ongoing violation duration.

The output video labels each worker "ID{n}" with COMPLIANT / NO HELMET and, for
violations, how many seconds they've been non-compliant. A HUD shows the live
head-count and current violations.

Usage:
    python src/track_compliance.py --source clip.mp4
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from detect_violations import check_compliance   # reuse the compliance logic

ROOT = Path(__file__).resolve().parents[1]
GREEN = (60, 200, 60)
RED = (40, 40, 220)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal PPE compliance monitoring")
    p.add_argument("--source", required=True, help="video path")
    p.add_argument("--ppe-weights", default=str(ROOT / "models" / "best.pt"))
    p.add_argument("--person-weights", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--ppe-conf", type=float, default=0.30)
    p.add_argument("--person-conf", type=float, default=0.45)
    p.add_argument("--device", default=0)
    return p.parse_args()


def draw_label(frame, x, y, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = y - th - 8 if y - th - 8 > 0 else y + 2
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + 8), color, -1)
    cv2.putText(frame, text, (x + 4, top + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main() -> None:
    args = parse_args()
    ppe_model = YOLO(args.ppe_weights)
    person_model = YOLO(args.person_weights)

    src = Path(args.source)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = ROOT / "reports" / f"{src.stem}_tracked.mp4"
    out.parent.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Per-track history.
    streak = defaultdict(int)            # consecutive violation frames
    violation_frames = defaultdict(int)  # total violation frames
    seen = defaultdict(int)
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1

        # 1. Track people (persist keeps IDs stable across frames).
        pres = person_model.track(
            frame, persist=True, classes=[0], conf=args.person_conf,
            imgsz=args.imgsz, tracker="bytetrack.yaml", device=args.device,
            verbose=False,
        )[0]
        # 2. Detect PPE.
        fres = ppe_model.predict(frame, imgsz=args.imgsz, conf=args.ppe_conf,
                                 device=args.device, verbose=False)[0]
        ppe_items = [(ppe_model.names[int(b.cls)], b.xyxy[0].tolist())
                     for b in fres.boxes]

        current_violations = 0
        if pres.boxes.id is not None:
            for box in pres.boxes:
                tid = int(box.id)
                pbox = box.xyxy[0].tolist()
                missing = check_compliance(pbox, ppe_items)
                seen[tid] += 1
                if missing:
                    violation_frames[tid] += 1
                    streak[tid] += 1
                    current_violations += 1
                else:
                    streak[tid] = 0

                x1, y1, x2, y2 = map(int, pbox)
                color = RED if missing else GREEN
                if missing:
                    dur = streak[tid] / fps
                    label = f"ID{tid} NO HELMET {dur:.1f}s"
                else:
                    label = f"ID{tid} COMPLIANT"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                draw_label(frame, x1, y1, label, color)

        # 3. HUD.
        hud = f"Workers: {len(seen)}   Violations now: {current_violations}"
        cv2.rectangle(frame, (0, 0), (430, 34), (0, 0, 0), -1)
        cv2.putText(frame, hud, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        writer.write(frame)
        if n % 30 == 0:
            print(f"  processed {n} frames ...")

    cap.release()
    writer.release()

    # Summary.
    print(f"\nDone. {n} frames -> {out}")
    print("Per-worker compliance summary:")
    for tid in sorted(seen):
        bad = violation_frames[tid]
        pct = 100 * bad / seen[tid]
        print(f"  ID{tid}: seen {seen[tid]} frames, "
              f"non-compliant {bad} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
