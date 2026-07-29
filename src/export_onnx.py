"""
Optimize the trained model for edge / CPU deployment, and benchmark it.

This is the "engineering" part of the story: we take the trained PyTorch model
and produce lighter, faster artefacts, then MEASURE the trade-off honestly.

Steps:
  1. Export best.pt -> ONNX (FP32)
  2. Quantize the ONNX to INT8 (dynamic quantization via onnxruntime)
  3. Benchmark all three on a fixed input:
        - best.pt   on GPU   (the training format)
        - ONNX FP32 on CPU   (edge baseline)
        - ONNX INT8 on CPU   (optimized edge)
     reporting model size, mean latency, and FPS.
  4. Write a Markdown table to reports/benchmark.md

Usage:
    python src/export_onnx.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
IMGSZ = 640
WARMUP, RUNS = 5, 30


def mb(path: Path) -> float:
    return path.stat().st_size / 1024**2


def bench_onnx(model_path: Path, providers: list[str]) -> float:
    """Return mean latency (ms) for an ONNX model over RUNS iterations."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4  # keep CPU usage predictable
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), so, providers=providers)
    inp = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32)

    for _ in range(WARMUP):
        sess.run(None, {inp: x})
    t0 = time.perf_counter()
    for _ in range(RUNS):
        sess.run(None, {inp: x})
    return (time.perf_counter() - t0) / RUNS * 1000


def bench_torch_gpu(pt_path: Path) -> float | None:
    """Return mean GPU latency (ms) for the .pt model, or None if no GPU."""
    import torch
    if not torch.cuda.is_available():
        return None
    model = YOLO(str(pt_path))
    dummy = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)
    for _ in range(WARMUP):
        model.predict(dummy, device=0, imgsz=IMGSZ, verbose=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        model.predict(dummy, device=0, imgsz=IMGSZ, verbose=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / RUNS * 1000


def main() -> None:
    pt = MODELS / "best.pt"
    if not pt.exists():
        raise FileNotFoundError(f"{pt} not found. Train first.")

    # 1. Export to ONNX (FP32).
    print("Exporting to ONNX ...")
    onnx_fp32 = Path(
        YOLO(str(pt)).export(format="onnx", imgsz=IMGSZ, simplify=True, dynamic=False)
    )
    onnx_fp32 = onnx_fp32.rename(MODELS / "best.onnx")

    # 2. Quantize to INT8.
    print("Quantizing to INT8 ...")
    onnx_int8 = MODELS / "best_int8.onnx"
    quantize_dynamic(str(onnx_fp32), str(onnx_int8), weight_type=QuantType.QInt8)

    # 3. Benchmark.
    print("Benchmarking (this takes a moment) ...\n")
    gpu_ms = bench_torch_gpu(pt)
    cpu_fp32_ms = bench_onnx(onnx_fp32, ["CPUExecutionProvider"])
    cpu_int8_ms = bench_onnx(onnx_int8, ["CPUExecutionProvider"])

    rows = [
        ("PyTorch .pt", "GPU (RTX 4060)", mb(pt), gpu_ms),
        ("ONNX FP32", "CPU", mb(onnx_fp32), cpu_fp32_ms),
        ("ONNX INT8", "CPU", mb(onnx_int8), cpu_int8_ms),
    ]

    # 4. Write the table.
    lines = [
        "# Optimization benchmark\n",
        f"Input size: {IMGSZ}x{IMGSZ}, averaged over {RUNS} runs "
        f"({WARMUP} warmup).\n",
        "| Format | Device | Size (MB) | Latency (ms) | FPS |",
        "|--------|--------|-----------|--------------|-----|",
    ]
    print(f"{'Format':14s}{'Device':16s}{'Size(MB)':10s}{'Latency(ms)':13s}FPS")
    for name, dev, size, ms in rows:
        fps = 1000 / ms if ms else 0
        ms_s = f"{ms:.1f}" if ms else "n/a"
        fps_s = f"{fps:.1f}" if ms else "n/a"
        print(f"{name:14s}{dev:16s}{size:<10.1f}{ms_s:13s}{fps_s}")
        lines.append(f"| {name} | {dev} | {size:.1f} | {ms_s} | {fps_s} |")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "benchmark.md").write_text("\n".join(lines))
    print(f"\nWrote {REPORTS / 'benchmark.md'}")
    print("Next: python src/infer_video.py --source your_clip.mp4")


if __name__ == "__main__":
    main()
