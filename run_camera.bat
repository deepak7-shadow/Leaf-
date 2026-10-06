@echo off
title Leaf Authenticity Detector
cd /d "%~dp0"
echo ========================================================
echo   LEAF AUTHENTICITY DETECTOR (REAL vs FAKE)
echo ========================================================
echo.
echo [INFO] Detected cameras:
echo        Camera 1: External USB Webcam (Lenovo FHD)
echo        Camera 0: Integrated Device Camera
echo.
echo [INFO] Starting detector on USB Webcam (Camera #1)...
echo [INFO] Press 'c' to switch cameras live!
echo [INFO] Press 'q' to quit, 's' to save a snapshot.
echo.
.\.venv\Scripts\python.exe camera_test.py --camera 1
if errorlevel 1 (
    echo.
    echo [WARNING] Camera 1 failed, trying Camera 0...
    .\.venv\Scripts\python.exe camera_test.py --camera 0
)
