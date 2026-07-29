# Optimization & deployment benchmark

Input size 640×640, averaged over 30 runs (5 warmup). GPU = RTX 4060 Laptop,
CPU = Intel i7-13620H with 4 inference threads.

| Format | Device | Size (MB) | Latency (ms) | FPS |
|--------|--------|-----------|--------------|-----|
| PyTorch `.pt` | GPU | 18.3 | 7.6 | 131.5 |
| ONNX FP32 | CPU | 36.2 | 60.9 | 16.4 |
| ONNX INT8 (dynamic) | CPU | 9.4 | 740.4 | 1.4 |

## What this shows

**1. The model is edge-deployable.** Exported to ONNX, it runs at **~16 FPS on
CPU alone** — no GPU required. For a safety-monitoring use case (which doesn't
need 100+ FPS), that's comfortably real-time on commodity hardware.

**2. Dynamic INT8 quantization was the wrong tool here — and I measured that
rather than assuming.** It shrank the model 4× (36 → 9 MB) but made CPU inference
**~12× slower**. The reason is architectural: `quantize_dynamic` quantizes
weights and computes activation scales on the fly, which pays off for
weight-bound operators (transformer/RNN matmuls). YOLO is convolution-bound, and
ONNXRuntime's CPU backend ends up wrapping each conv in quantize/dequantize steps
whose overhead dwarfs any saving. The correct approach for a CNN would be
**static (calibration-based) quantization** or quantization-aware training —
noted as future work.

**Conclusion:** for CPU deployment, **FP32 ONNX is the right target.** Further CPU
speedups come cheaply from lowering the input resolution (`--imgsz 512/416`) and
frame-skipping on video, both supported in `src/infer_video.py`.
