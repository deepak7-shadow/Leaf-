# 🍃 AI Nozzle: Intelligent Crop Disease Scanner & Precision Spraying System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flutter](https://img.shields.io/badge/Flutter-3.x-02569B.svg?style=flat&logo=flutter&logoColor=white)](https://flutter.dev/)
[![Spring Boot](https://img.shields.io/badge/Spring_Boot-3.x-6DB33F.svg?style=flat&logo=springboot&logoColor=white)](https://spring.io/projects/spring-boot)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-005CED.svg?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![MobileNetV3](https://img.shields.io/badge/Model-MobileNetV3--Small-brightgreen.svg?style=flat)](https://pytorch.org/vision/stable/models/mobilenetv3.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end precision agriculture platform combining **10-Class Crop Disease & Authenticity AI**, **real-time edge vision**, **automated solenoid spray valve controllers**, a **Flutter mobile/web app**, and a **Spring Boot microservice**.

---

## 🏗️ Architecture Overview

The repository unifies all layers of the precision spraying ecosystem:

1. **AI Vision & Edge Pipeline (`core/`, `models/`, `camera_test.py`)**:
   - 10-Class Deep Learning Scanner (MobileNetV3-Small in ONNX & TFLite formats).
   - Distinguishes **8 plant diseases** (*Anthracnose, Bacterial Leaf Spot, Black Rot, Downy Mildew, Leaf Spot/Blight, Mosaic Virus, Powdery Mildew, Rust*), **Healthy living foliage**, and **Not_Leaf / Synthetic distractors**.
   - Solenoid valve actuation controller with debounce cooldown, master safety lockout, and automated pesticide/fertilizer prescription matching.

2. **Cross-Platform Mobile & Web Client (`flutter_app/`)**:
   - Live camera scanner & gallery image analyzer.
   - Built-in TFLite offline inference, damage estimation, and language localizations.
   - Cloud telemetry integration via Blynk IoT.

3. **Backend Service (`spring_backend/`)**:
   - High-throughput REST API for crop analysis, telemetry logging, and treatment recommendations.

4. **Hardware Driver (`ardunio/`)**:
   - Firmware for Arduino / ESP32 microcontroller relays actuating high-pressure solenoid spray nozzles.

---

## 📁 Repository Structure

```text
├── config/
│   ├── fertilizer.json          # Disease-specific fertilizer & fungicide treatment prescriptions
│   └── settings.yaml            # Pipeline camera, detector, and valve timings
├── core/
│   ├── nozzle_controller.py     # Solenoid valve actuation and safety lockout logic
│   ├── pipeline.py              # End-to-end 2-stage vision pipeline
│   ├── stage1_detector.py       # Fast OpenCV vegetation candidate detection
│   ├── stage2_classifier.py     # MobileNetV3 ONNX multi-class classifier
│   └── vegetation_indices.py    # ExG / vegetation index calculations
├── models/
│   ├── class_names.json         # 10 target classes
│   ├── crop_model_10class.onnx  # Trained 10-class ONNX edge model
│   ├── crop_model_float16.tflite# Quantized TFLite edge model
│   └── train_10class.py         # PyTorch 10-class training pipeline
├── flutter_app/                 # Complete Flutter mobile/web client application
├── spring_backend/              # Spring Boot backend API microservice
├── ardunio/                     # Microcontroller relay & valve control firmware
├── tests/                       # Unit and integration test suite
├── camera_test.py               # Real-time desktop camera scanner with HUD
├── run_camera.bat               # 1-Click desktop launcher (Windows)
└── requirements.txt             # Edge pipeline Python dependencies
```

---

## 🚀 Quick Start

### 1. Run the Real-Time Camera Scanner (Desktop)
```bash
# Run 1-click launcher
run_camera.bat

# Or launch directly with Python
python camera_test.py
```
**Controls**:
- `c` = Hot-switch between USB webcam and integrated laptop camera
- `d` = Toggle vegetation mask overlay
- `s` = Capture snapshot with HUD telemetry
- `q` = Quit

### 2. Run the Test Suite
```bash
python -m unittest discover tests
```

### 3. Launch Flutter App
```bash
cd flutter_app
flutter pub get
flutter run
```

### 4. Launch Spring Boot Backend
```bash
cd spring_backend/spring_backend_fresh/spring_backend
./mvnw spring-boot:run
```

---

## 🛡️ License
MIT License.
