# PPE Detection — Project Report

## What I built

An end-to-end computer-vision system for **personal protective equipment (PPE)
detection and compliance monitoring** — helmet, safety vest, mask, gloves,
goggles, safety shoes. I wanted it to go the whole distance rather than stop at a
notebook: **data → training → evaluation → generalization → optimization →
compliance logic → tracking → a runnable demo → deployment.**

A self-imposed constraint shaped many decisions: everything trains on a single
**8 GB laptop GPU (RTX 4060)** and deploys on a **plain CPU** — no cloud, no
dedicated hardware.

---

## The data

Base dataset: the public
[`51ddhesh/PPE_Detection`](https://huggingface.co/datasets/51ddhesh/PPE_Detection)
(Hugging Face, CC-BY-4.0), YOLO format, six classes.

| Split | Images |
|-------|--------|
| Train | 8,774 |
| Val   | 2,070 |
| Test  | 1,234 |

**An honest note on the data:** this is a *general-domain* PPE dataset with broad
class definitions — "goggles" covers any eyewear (including prescription glasses),
"mask" covers any face covering, "safety_shoe" any footwear. That breadth helps
generalization but means the model isn't strictly construction-specific. Curating
a tighter, construction-only subset is the clearest path to better weak-class
accuracy (see *Limitations*).

`src/prepare_data.py` handles setup — it rewrites the dataset's `data.yaml` to
absolute paths (a classic source of "0 images found" errors) and prints a sanity
summary before any training runs.

---

## Training

I fine-tuned **YOLO11s** (the small variant) — a deliberate choice: it fits 8 GB
comfortably and stays light enough for CPU deployment, while still being accurate
enough. The training config is tuned for the memory budget: mixed precision (AMP,
~halves VRAM), batch 16 at 640px, disk caching (not RAM), controlled dataloader
workers, and early stopping. A run is 50 epochs, ~98 minutes on the RTX 4060.

---

## Results

Evaluated on the **held-out test set** (1,234 images never seen in training):

| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.80** |
| mAP@0.5:0.95 | 0.52 |
| Precision | 0.85 |
| Recall | 0.74 |

Per class (final model):

| Class | mAP@0.5 |
|-------|---------|
| Vest | 0.94 |
| goggles | 0.94 |
| helmet | 0.89 |
| mask | 0.72 |
| safety_shoe | 0.68 |
| Gloves | 0.66 |

**How this compares:** an independent published model on the *same six classes*
([`Tanishjain9/yolov8n-ppe-detection-6classes`](https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes))
reports ~0.81 overall, with gloves ~0.69 and safety_shoe ~0.64. My model matches
it overall and beats it on 3 of 6 classes (goggles, vest, safety_shoe) — so it
sits **at the published frontier for this dataset.**

---

## Improving generalization (the merge)

The results above are on the same distribution the model trained on. Testing on a
*second* public dataset (Ultralytics Construction-PPE) exposed severe overfitting:

| Baseline model tested on | mAP@0.5 |
|--------------------------|---------|
| Its own test set | 0.812 |
| An unseen dataset | **0.118** |

The fix was to **merge the second dataset in**, harmonizing its class taxonomy
onto ours (`src/merge_datasets.py`). After retraining:

| Test set | Baseline | Merged (final) |
|----------|----------|----------------|
| Original | 0.812 | 0.803 |
| Cross-dataset | 0.118 | **0.741** |

**Cross-dataset accuracy jumped ~6×** for under a point of regression on the
original test — a large real-world robustness gain at almost no cost. The merged
model is the one used for all demos and deployment.

---

## Trying to lift the weak classes — five measured experiments

Gloves, masks, and safety shoes trail the other classes. I ran five controlled
experiments (same seed, one variable at a time) to try to improve them:

| Experiment | Result |
|------------|--------|
| Merge more data (Construction-PPE) | generalization ↑, weak classes flat |
| Heavier augmentation (copy-paste, mixup) | mask +0.05, but gloves/shoes ↓ |
| High-resolution training (960px) | goggles/helmet ↑, gloves/shoes flat |
| Per-class threshold tuning | safety_shoe F1 +0.04 |
| +4,663 more glove examples | gloves ↓ (distribution mismatch) |

**Conclusion:** no method reliably improved gloves/safety_shoe. The bottleneck is
**data quality and class definition, not model capacity** — confirmed both by
these experiments and by the published benchmark (which caps gloves/shoes in the
same band). The weak classes are limited by *recall*: when the object is clearly
visible the model detects it well (e.g. safety shoes at 0.79–0.84 confidence); it
misses small, occluded, or awkward-angle instances. The real fix is cleaner,
distribution-matched labels — not more training tricks.

---

## Optimization for the edge

Exported to ONNX and benchmarked (30 runs, 640×640):

| Format | Device | Size (MB) | Latency (ms) | FPS |
|--------|--------|-----------|--------------|-----|
| PyTorch `.pt` | GPU | 18.3 | 7.4 | 136 |
| ONNX FP32 | CPU | 36.2 | 58.4 | 17 |
| ONNX INT8 (dynamic) | CPU | 9.4 | 727 | 1.4 |

**The model runs ~17 FPS on CPU alone** — zero GPU dependency at deployment.
I also tried **INT8 dynamic quantization**, expecting smaller-and-faster; it got
4× smaller but **12× slower** — a useful negative result. Dynamic quantization
suits weight-bound (transformer/RNN) operators; YOLO is convolution-bound, so
ONNXRuntime wraps every conv in quantize/dequantize steps that cost more than they
save. FP32 ONNX stays the deployment target; static quantization is the correct
future fix.

---

## From detection to compliance

Detection says "there is a helmet here"; a safety system needs "is this person
compliant?" `src/detect_violations.py` adds that layer — it pairs a pretrained
COCO person detector with the PPE model and checks whether each person has a
helmet in their head region, flagging anyone who doesn't. No retraining needed.
It's a deliberately simple spatial heuristic, so it inherits the detector's limits
(a missed helmet reads as a false violation), but it turns a bounding-box model
into a usable compliance monitor.

---

## Temporal monitoring with tracking

`src/track_compliance.py` adds **ByteTrack** multi-object tracking on top: each
worker gets a persistent ID, and the system accumulates a per-worker compliance
history — reporting *sustained* violations rather than per-frame noise, with a
live HUD counting workers and current violations. It scales from a close 2-worker
view to a busy site with ~10 workers tracked and evaluated at once.

---

## The demo & deployment

`demo/app.py` is a Streamlit app: upload an image or short video and watch the
model annotate it, running on the CPU ONNX model so it works anywhere. The
`deploy/` folder is a self-contained package (app, CPU-only requirements, model)
ready for a free Hugging Face Space — no GPU required.

---

## Honest limitations

- **General-domain data.** Loose class definitions (goggles = any eyewear, etc.)
  mean the model isn't strictly construction-specific.
- **Weak classes at the dataset ceiling.** Gloves/mask/safety_shoe are
  recall-limited and can't be pushed further without better data (proven above).
- **Compliance is a heuristic.** Its accuracy is bounded by the detector's — the
  tracking layer aggregates, it doesn't fix missed detections.

---

## What I'd do next

1. **Curate a construction-specific subset** with tightened labels — the
   experiments show this is the real lever for the weak classes.
2. **Static (calibration-based) INT8** for a genuine CPU speedup.
3. **Add self-recorded footage** to close the deployment-distribution gap.
4. **Deploy the Streamlit app** to a public URL for a clickable demo.

---

## Reproducing this

```bash
python src/prepare_data.py                 # unzip + fix paths
python src/merge_datasets.py               # merge second dataset (needs it downloaded)
python src/train.py --epochs 50            # train (GPU)
python src/evaluate.py                     # metrics + confusion matrix
python src/export_onnx.py                  # ONNX + benchmark
python src/detect_violations.py --source photo.jpg   # compliance
python src/track_compliance.py --source clip.mp4     # tracking
streamlit run demo/app.py                  # interactive demo
```
