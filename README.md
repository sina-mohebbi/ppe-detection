<p align="center">
  <img src="assets/banner.svg" alt="PPE Detection — Real-time Computer Vision with YOLO11" width="100%">
</p>

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

## Dataset

The base dataset is a public PPE collection in YOLO format
([51ddhesh/PPE_Detection](https://huggingface.co/datasets/51ddhesh/PPE_Detection),
CC-BY-4.0), covering six classes: `Gloves, Vest, goggles, helmet, mask, safety_shoe`.

| Split | Images |
|-------|--------|
| Train | 8,774 |
| Val   | 2,070 |
| Test  | 1,234 |

`src/prepare_data.py` gets it ready for training: it extracts the archive,
rewrites the dataset's `data.yaml` to absolute paths (so training works no matter
which directory it's launched from — a common cause of "0 images found" errors),
and prints a quick sanity summary of image counts and class names.

```bash
curl.exe -L -o data/PPE.zip https://huggingface.co/datasets/51ddhesh/PPE_Detection/resolve/main/PPE.zip
python src/prepare_data.py
```

## Roadmap

- [x] Project scaffold
- [x] Dataset preparation
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
