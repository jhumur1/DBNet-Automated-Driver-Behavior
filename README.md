# DBNet-Automated-Driver-Behavior-Analysis
RAAICON 2025

This project implements YOLOv12 for detecting driver distractions using a single-camera dataset. It leverages the Ultralytics YOLO framework (github.com) and Roboflow for dataset management.

## 📌 Features
Custom YOLOv12 model (yolo12n_new.yaml) with 115 layers and ~1.7M parameters.

Trained on the Driver Distraction Single-Camera dataset (9 classes).

Uses Tesla T4 GPU with CUDA 12.4 for acceleration.

Training pipeline with 100 epochs, batch size 16, and automatic optimizer selection (AdamW).

Augmentations: blur, grayscale, CLAHE, flips, perspective, etc.

## ⚙️ Setup
bash
# Check GPU
!nvidia-smi

# Install dependencies
pip install ultralytics roboflow
📂 Dataset
Dataset is hosted on Roboflow:

python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("mtech-yi2na").project("driver-distraction_single-camera")
version = project.version(5)
dataset = version.download("yolov12")
🏗️ Model
python
from ultralytics import YOLO

# Load YOLOv12 model
model = YOLO('/content/yolo12n_new.yaml')
model.info(verbose=True)
🚀 Training
python
model.train(
    data="/content/Driver-Distraction_Single-Camera-5/data.yaml",
    epochs=100,
    batch=16
)
Training logs show steady improvement in mAP50 and mAP50-95:

Epoch	Precision	Recall	mAP50	mAP50-95
1	0.571	0.196	0.161	0.0745
5	0.684	0.547	0.584	0.319
10	0.763	0.688	0.781	0.428
20	0.822	0.809	0.859	0.515
25	0.844	0.843	0.880	0.528
30	0.809	0.849	0.880	0.537
📊 Results
Best mAP50: ~0.88

Best mAP50-95: ~0.54

Model converges around epoch 25–30 with strong detection performance.

## 📈 Next Steps
Fine-tune with larger batch size or longer training.

Experiment with YOLOv12-Large for higher accuracy.

Deploy model for real-time driver monitoring.

## 🤝 Acknowledgements
Ultralytics YOLO (github.com in Bing)

Roboflow

Dataset: Driver Distraction Single-Camera
