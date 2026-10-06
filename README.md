# 🍃 Leaf Authenticity AI: Real vs. Fake Leaf Detector

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-005CED.svg?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![MobileNetV3](https://img.shields.io/badge/Model-MobileNetV3--Small-brightgreen.svg?style=flat)](https://pytorch.org/vision/stable/models/mobilenetv3.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, edge-optimized computer vision system that classifies and validates **living organic leaves vs. fake/synthetic leaves** (printed paper, cardboard, screen displays, cloth, plastic) in real time using your webcam or external USB camera.

Powered by a fine-tuned **MobileNetV3-Small** deep learning model exported to **ONNX Runtime**, running at high framerates (30+ FPS) on standard CPUs with zero GPU requirements.

---

## ✨ Key Features

- **🎯 Region-of-Interest (ROI) Targeting Reticle**:
  - Focuses inference specifically on the leaf inside the center 280×280 target box.
  - Ignores surrounding hands, face, room background, and variable ambient lighting.
- **🛡️ Anti-Fluctuation & Precision Stabilization**:
  - **25-Frame Temporal Smoothing**: Rolling probability buffer eliminates frame-by-frame jitter.
  - **65% Confidence Gate**: Stays in an amber `SCANNING...` state until the model is confident.
  - **8-Frame Stability Lock**: Requires consecutive consensus before confirming the final label.
- **📷 Dual Camera & Hot-Switching**:
  - Automatically detects and defaults to external USB webcams (`Camera #1`) or falls back to built-in laptop cameras (`Camera #0`).
  - Press **`c`** to hot-switch between cameras live without restarting the program.
- **⚡ Ultra-Lightweight Edge Inference**:
  - Deployed via a pre-trained **5.8 MB ONNX model** (`models/mobilenet_v3_small.onnx`).
  - Runs in real-time on standard laptop CPUs.
- **📸 Snapshot Tool**:
  - Press **`s`** to save timestamped high-resolution photos with the detection HUD overlay.

---

## 📁 Repository Structure

```text
Leaf/
├── camera_test.py                # Main real-time camera detector & HUD
├── run_camera.bat                # 1-Click launcher for Windows
├── requirements.txt              # Minimal project dependencies
├── README.md                     # Project documentation
│
├── models/
│   ├── mobilenet_v3_small.onnx   # Deployed 5.8MB ONNX model (Ready out-of-the-box)
│   ├── train.py                  # PyTorch training pipeline with webcam augmentations
│   └── export_onnx.py            # Script to convert PyTorch weights to ONNX
│
└── utils/
    └── synthetic_generator.py    # Synthetic leaf scene generator for training
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/deepak7-shadow/Leaf-.git
cd Leaf-
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows:
.\.venv\Scripts\activate

# Activate on Linux/macOS:
source .venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

### 3. Run the Detector

#### On Windows (Easiest — 1 Click):
Double-click `run_camera.bat` or run:
```powershell
.\run_camera.bat
```

#### Via Command Line:
```powershell
# External USB Webcam (Default recommended)
python camera_test.py --camera 1

# Built-in Laptop Camera
python camera_test.py --camera 0

# Auto-detect camera
python camera_test.py
```

---

## 🎮 Interactive Controls

| Key | Action |
|:---:|:---|
| **`c`** | **Hot-switch camera** between External USB Webcam (`#1`) and Device Cam (`#0`) |
| **`s`** | **Save snapshot** of current frame with HUD overlay to disk |
| **`q`** / **`ESC`** | **Quit** and close camera safely |

---

## 🧠 Model Architecture & Training

The classifier uses **MobileNetV3-Small** fine-tuned on real living leaves and spoofed/artificial substrates:

- **Classes**:
  - `0`: **REAL LEAF** (Living, chlorophyll-rich botanical plant foliage)
  - `1`: **FAKE LEAF** (Printed paper, fabric, screen projections, plastic artificial foliage)
- **Webcam Noise Augmentations**:
  - Sensor Gaussian noise simulation
  - Motion and dot-gain Gaussian blur
  - Color jitter (gamut and lighting variations)
  - Random perspective and rotation transforms
- **Loss**: Class-weighted Cross-Entropy loss for balanced accuracy across datasets.

### To Retrain the Model:
```bash
python models/train.py --data-dir path/to/dataset --epochs 20 --batch-size 16
```
*(The training script automatically evaluates validation accuracy and exports the updated ONNX model to `models/mobilenet_v3_small.onnx`).*

---

## 📄 License
This project is licensed under the MIT License — feel free to use and adapt for your agricultural and computer vision projects.
