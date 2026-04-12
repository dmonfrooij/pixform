@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] PIXFORMWIN is not installed yet.
    echo Please run install.ps1 first.
    pause
    exit /b 1
)

set "PROFILE=%~1"
if /I "%PROFILE%"=="nvidia" set "PIXFORM_DEVICE=cuda"
if /I "%PROFILE%"=="cuda"   set "PIXFORM_DEVICE=cuda"
if /I "%PROFILE%"=="mac"    set "PIXFORM_DEVICE=mps"
if /I "%PROFILE%"=="mps"    set "PIXFORM_DEVICE=mps"
if /I "%PROFILE%"=="cpu"    set "PIXFORM_DEVICE=cpu"

if "%PIXFORM_DEVICE%"=="" (
    if exist ".pixform_device" set /p PIXFORM_DEVICE=<.pixform_device
)
if "%PIXFORM_DEVICE%"=="" set "PIXFORM_DEVICE=auto"

echo Starting PIXFORMWIN (PIXFORM_DEVICE=%PIXFORM_DEVICE%)...
"venv\Scripts\python.exe" app.py %PROFILE%
