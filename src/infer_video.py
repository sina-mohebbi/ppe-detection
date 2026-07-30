"""
Run PPE detection on a video or image and save an annotated copy.

Written for CPU-first, edge-style inference (this is the "does it run on modest
hardware" demonstration), with a few deliberate optimizations:

  * cv2 / torch thread counts pinned  -> predictable CPU load, no oversubscription
  * frames streamed one at a time      -> constant memory, never loads whole video
  * optional --frame-skip              -> process every Nth frame for throughput
  * optional --imgsz downscale         -> big CPU saving on 4K phone footage

Usage:
    python src/infer_video.py --source my_clip.mp4
    python src/infer_video.py --source my_clip.mp4 --frame-skip 2 --imgsz 512
    python src/infer_video.py --source photo.jpg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# Pin threads BEFORE heavy work so we don't thrash all 16 logical cores.
cv2.setNumThreads(4)
torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPE detection on video/image")
    p.add_argument("--source", required=True, help="path to video or image")
    p.add_argument("--weights", default=str(ROOT / "models" / "best.onnx"),
                   help="best.onnx (CPU) or best.pt (GPU)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--frame-skip", type=int, default=1,
                   help="process every Nth frame (video only); 1 = every frame")
    p.add_argument("--device", default="cpu", help="'cpu' or 0 for GPU")
    return p.parse_args()


def run_image(model: YOLO, args) -> None:
    src = Path(args.source)
    res = model.predict(str(src), imgsz=args.imgsz, conf=args.conf,
                        device=args.device, verbose=False)[0]
    out = ROOT / "reports" / f"{src.stem}_pred.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), res.plot())
    print(f"Saved {out}")


def run_video(model: YOLO, args) -> None:
    src = Path(args.source)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {src} (try converting .mov -> .mp4)")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = ROOT / "reports" / f"{src.stem}_pred.mp4"
    out_path.parent.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in / args.frame_skip, (w, h)
    )

    n, t0 = 0, time.perf_counter()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.frame_skip:            # cheaply skip frames
            continue
        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)[0]
        writer.write(res.plot())
        n += 1
        if n % 30 == 0:
            print(f"  processed {n} frames ...")

    cap.release()
    writer.release()
    dt = time.perf_counter() - t0
    print(f"\nDone. {n} frames in {dt:.1f}s -> {n/dt:.1f} FPS effective")
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_image(model, args)
    else:
        run_video(model, args)


if __name__ == "__main__":
    main()
