@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] PIXFORM is not installed yet.
    echo Please run install.ps1 first.
    pause
    exit /b 1
)

echo Starting PIXFORM...
start "" "venv\Scripts\python.exe" "backend\app.py"
timeout /t 3 /nobreak >nul
start "" "http://localhost:8000"
