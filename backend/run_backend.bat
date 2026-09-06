@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ================================================================
echo                   CLONE-ME-AI Backend Server
echo ================================================================

REM Check if embedded python exists in parent directory
if exist "..\python_embeded\python.exe" (
    set "PYTHON_BIN=..\python_embeded\python.exe"
) else (
    set "PYTHON_BIN=python"
)

echo Using Python interpreter: %PYTHON_BIN%
echo Starting Uvicorn server on http://127.0.0.1:8000 ...
echo Press Ctrl+C to terminate.
echo ================================================================

%PYTHON_BIN% -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

pause
