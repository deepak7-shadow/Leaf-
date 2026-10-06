@echo off
title Leaf Authenticity Detector
cd /d "%~dp0"
echo ========================================================
echo   LEAF AUTHENTICITY DETECTOR (REAL vs FAKE)
echo ========================================================
echo.
echo [INFO] Starting detector on webcam...
echo [INFO] Hold the leaf inside the on-screen target box.
echo [INFO] Press 'q' to quit, 's' to save a snapshot.
echo.
.\.venv\Scripts\python.exe camera_test.py --camera 0
if errorlevel 1 (
    echo.
    echo [ERROR] Camera failed to open or crashed.
    pause
)
