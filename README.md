<p align="center">
  <img src="assets/banner.svg" alt="PPE Detection — Real-time Computer Vision with YOLO11" width="100%">
</p>

PPE Detection 🦺👷

A computer vision system for detecting personal protective equipment such as
helmets, safety vests, masks, gloves, goggles, and safety shoes in images and
videos.

The model uses YOLO11 and can run on a CPU, making it suitable for edge devices
and systems without a dedicated GPU.

<p align="center">
  <img src="assets/tracking_group.gif" alt="Real-time PPE compliance monitoring" width="90%">
</p>
<p align="center"><em>Real-time helmet compliance monitoring: workers are detected,
tracked with stable IDs, and checked for helmet use.</em></p>

Why this project

PPE monitoring is a practical use of computer vision in construction sites,
factories, and warehouses.

The goal of this project is to build the complete system, not just a model-training
notebook:

data → training → evaluation → optimization → demo → deployment

Dataset

The base dataset is a public PPE dataset in YOLO format:

51ddhesh/PPE_Detection
under the CC-BY-4.0 licence.

It contains six classes:

Gloves, Vest, goggles, helmet, mask, and safety_shoe.

Split	Images
Train	8,774
Val	2,070
Test	1,234

src/prepare_data.py prepares the dataset for training. It:

* extracts the downloaded archive
* changes the paths in data.yaml to absolute paths
* checks the number of images in each split
* prints the class names

Using absolute paths allows training to work correctly regardless of the directory
from which the command is launched. This also prevents common "0 images found"
errors.

curl.exe -L -o data/PPE.zip https://huggingface.co/datasets/51ddhesh/PPE_Detection/resolve/main/PPE.zip
python src/prepare_data.py

Training

I fine-tuned YOLO11s using transfer learning.

I chose the s small model because it runs within 8 GB of VRAM while remaining
light enough for CPU deployment. It also provides enough accuracy for this task.

src/train.py is configured for my RTX 4060 Laptop GPU with 8 GB of VRAM:

* Mixed precision (AMP) — reduces VRAM use and speeds up training
* Batch size 16 at 640px — uses the available GPU memory without running out
* Disk caching — stores the dataset cache on disk instead of system RAM
* Early stopping — stops training when the model is no longer improving

python src/train.py --epochs 50

Before training starts, the script checks that PyTorch is using the GPU. After
training, it copies the best model weights to:

models/best.pt

Results

The model was trained for 50 epochs and evaluated on the held-out test set.

The test set contains 1,234 images that were not used during training.

Metric	Value
mAP@0.5	0.812
mAP@0.5:0.95	0.520
Precision	0.847
Recall	0.745

Results by class

Class	mAP@0.5
goggles	0.947
Vest	0.940
helmet	0.891
safety_shoe	0.726
mask	0.718
Gloves	0.652

What the results show

Performance varies between classes.

Vests, goggles, and helmets have the highest scores. These objects are usually
larger, more visually distinctive, and well represented in the dataset.

Gloves and masks are more difficult to detect because they are smaller and can
look very different depending on the image. They also have fewer training
examples. Masks, for example, have only around 86 instances in the test set, so
their result is less stable.

Precision is about 0.85, while recall is about 0.75. This means that when the
model detects an object, it is usually correct, but it still misses some difficult
or small objects.

For a safety-monitoring system, recall can be increased by lowering the confidence
threshold, although this may also produce more false detections.

src/evaluate.py generates these metrics together with:

* the confusion matrix
* precision-recall curves
* F1-score curves

<p align="center">
  <img src="reports/confusion_matrix_normalized.png" alt="Normalized confusion matrix" width="70%">
</p>

Improving generalization

The first results were measured on the same dataset used for training.

To check how well the model performs on completely new data, I also tested it on
a second public dataset: Ultralytics Construction-PPE.

The first model performed well on its own test set but poorly on the new dataset:

Baseline model tested on	mAP@0.5
Its own test set	0.812
An unseen dataset	0.118

The drop from 0.812 to 0.118 showed that the model had learned the original
dataset too closely and did not generalize well to different images.

To improve this, I merged the second dataset with the original one.

src/merge_datasets.py:

* matches the class names from the second dataset to the original classes
* removes classes that are not used in this project
* creates a combined dataset for retraining

After retraining on the merged dataset:

Test set	Baseline	Merged
Original	0.812	0.803
Unseen cross-dataset set	0.118	0.741

Cross-dataset mAP increased from 0.118 to 0.741, while performance on the
original test set decreased only slightly from 0.812 to 0.803.

This shows that the merged model performs much better on new data with almost no
loss on the original dataset.

The merged model is used for the demos and deployment.

Limitations

A full explanation is available in
reports/dataset_merge.md.

The dataset merge mainly improves generalization. It does not fully solve the lower
performance of gloves, masks, and safety shoes on the original dataset.

The mapping from boots in the second dataset to safety_shoe is also not
perfect. I tested removing this mapping. It slightly improved the original
safety_shoe result, but reduced overall performance and cross-dataset
generalization, so I kept it.

CPU and edge optimization

The goal of this step was to answer two questions:

1. Can the model run without a GPU?
2. How fast can it run?

src/export_onnx.py exports the trained model to ONNX, creates a dynamically
quantized INT8 version, and benchmarks all three model formats.

| Format | Device | Size (MB) | Latency (ms) | FPS |
|––––|––––|———–|—–|
| PyTorch .pt | GPU | 18.3 | 7.6 | 131 |
| ONNX FP32 | CPU | 36.2 | 60.9 | 16 |
| ONNX INT8 dynamic | CPU | 9.4 | 740 | 1.4 |

Main findings

* The model can run in real time on a CPU.
    The ONNX FP32 model runs at around 16 FPS without a GPU. This is fast enough
    for many safety-monitoring applications.
* Dynamic INT8 quantization reduced performance.
    It made the model around four times smaller, but CPU inference became about
    twelve times slower.

Dynamic quantization often works better for transformer and RNN models. YOLO
mainly uses convolution operations. In this case, ONNX Runtime adds extra
quantization and dequantization steps around the convolution layers, and those
steps take more time than they save.

A better option for this type of CNN would be static quantization using a
calibration dataset. This is left as future work.

For now, the ONNX FP32 model is the main CPU deployment format.

A full benchmark report is available in:

reports/benchmark.md

Demo

The Streamlit application in demo/app.py allows users to upload an image or a
short video and view the detections.

The application uses the ONNX model on the CPU, so no GPU is required.

Detection on a construction-site video that was not used during training. Video
footage from Mixkit under its free licence.

src/infer_video.py provides the same video detection from the command line.

It includes CPU-friendly options such as frame skipping and input resizing, which
can improve performance on lower-end hardware.

streamlit run demo/app.py
python src/infer_video.py --source clip.mp4 --frame-skip 2

Predictions on test images

From detection to compliance

Basic object detection can show that a helmet exists somewhere in an image.

A safety-monitoring system needs to answer a more useful question:

Is each detected person wearing a helmet?

src/detect_violations.py adds this functionality.

It combines:

* a pretrained COCO person detector
* the trained PPE detector
* a simple position-based helmet check

For each detected person, the script checks whether a helmet appears inside the
person’s head region. People without a detected helmet are marked as violations.

No additional model training is required.

Each person is marked as COMPLIANT in green or NO HELMET in red on held-out test
images.

This method uses a simple position-based rule and therefore has the same
limitations as the object detector. For example, a helmet missed by the detector
may incorrectly be reported as a violation.

However, it allows the system to check helmet compliance for each detected person
instead of only showing PPE bounding boxes.

python src/detect_violations.py --source photo.jpg

Tracking workers over time

Checking each video frame separately only shows who appears non-compliant at that
moment.

A monitoring system should also follow each worker over time.

src/track_compliance.py adds ByteTrack multi-object tracking. Each worker is
given a stable ID, and the system stores a helmet-compliance history for that
worker.

This reduces unstable frame-by-frame results and helps identify violations that
continue over several frames.

Each worker is tracked with a stable ID. The live display shows the number of
workers, the current number of violations, and a COMPLIANT or NO HELMET label for
each worker. Video footage from Pexels and not used
during training.

The same system working on a busier site, with up to 10 workers tracked and
checked at the same time.

python src/track_compliance.py --source clip.mp4

The tracking system still depends on the quality of the PPE detector. A missed
helmet may still be reported as a violation.

However, tracking allows the system to monitor each worker’s helmet compliance
over time instead of making independent decisions for every frame.

Stack

Ultralytics YOLO11 · PyTorch · ONNX / ONNX Runtime · OpenCV ·
Streamlit

Author

Sina Mohebbi