@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ================================================================
echo                    CLONE-ME-AI Launcher
echo        Identity-Preserved AI Image Generation Application
echo ================================================================

REM 1. Check Python Interpreter
if exist "python_embeded\python.exe" (
    set "PYTHON_BIN=%~dp0python_embeded\python.exe"
) else (
    set "PYTHON_BIN=python"
)

echo [1/3] Using Python interpreter: !PYTHON_BIN!

REM 2. Check ComfyUI Service
echo [2/3] Checking ComfyUI server status at http://127.0.0.1:8188 ...
curl -s --connect-timeout 2 --max-time 3 http://127.0.0.1:8188/system_stats >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] ComfyUI is not currently responding.
    echo [*] Starting ComfyUI server in a separate process...
    start "ComfyUI Server" cmd /c "run_nvidia_gpu.bat"
    echo [*] Waiting 8 seconds for ComfyUI to initialize...
    timeout /t 8 /nobreak >nul
) else (
    echo [*] ComfyUI server is already online!
)

REM 3. Launch Backend & Web Interface
echo [3/3] Starting CLONE-ME-AI Backend at http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000

cd /d "%~dp0backend"
!PYTHON_BIN! -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

pause
