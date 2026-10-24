@echo off
title AI Nozzle - Crop Leaf Disease & Health Scanner
cd /d "%~dp0"
echo ========================================================
echo   AI NOZZLE - CROP LEAF DISEASE & AUTHENTICITY SCANNER
echo ========================================================
echo.
echo [INFO] Controls:
echo        'c' = Switch camera (USB Webcam vs Laptop Camera)
echo        'd' = Toggle vegetation mask
echo        's' = Save snapshot
echo        'q' = Quit
echo.
.\.venv\Scripts\python.exe camera_test.py --camera 0
if errorlevel 1 (
    echo.
    echo [WARNING] Camera 0 failed, trying Camera 1...
    .\.venv\Scripts\python.exe camera_test.py --camera 1
)
pause
