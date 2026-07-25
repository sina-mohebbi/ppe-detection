# PPE Detection 🦺

Detecting **personal protective equipment** — helmet, safety vest, mask, gloves,
goggles, safety shoes — in images and video with **YOLO11**, built to run on a
CPU for edge-style deployment.

> 🚧 **Work in progress** — building this in the open, step by step.

## Why this project

Safety-compliance monitoring is a real, deployed use of computer vision on
construction sites, in factories, and in warehouses. The plan is to take it
end-to-end rather than stop at a notebook: **data → training → evaluation →
optimization → demo → deployment.**

## Roadmap

- [x] Project scaffold
- [ ] Dataset preparation
- [ ] Training pipeline (YOLO11, GPU)
- [ ] Evaluation & failure analysis
- [ ] Optimization (ONNX export + benchmark)
- [ ] Demo (Streamlit)
- [ ] Deployment

## Planned stack

**Ultralytics YOLO11** · **PyTorch** · **ONNX / onnxruntime** · **OpenCV** ·
**Streamlit**

## Author

Sina Mohebbi
