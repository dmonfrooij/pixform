@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] PIXFORM is not installed yet.
    echo Please run install.ps1 first.
    pause
    exit /b 1
)

set "PROFILE=%~1"
set "ACTIVE_MODELS=%~2"
if /I "%PROFILE%"=="nvidia" set "PIXFORM_DEVICE=cuda"
if /I "%PROFILE%"=="cuda" set "PIXFORM_DEVICE=cuda"
if /I "%PROFILE%"=="mac" set "PIXFORM_DEVICE=mps"
if /I "%PROFILE%"=="mps" set "PIXFORM_DEVICE=mps"
if /I "%PROFILE%"=="cpu" set "PIXFORM_DEVICE=cpu"

if "%PIXFORM_DEVICE%"=="" (
    if exist ".pixform_device" set /p PIXFORM_DEVICE=<.pixform_device
)
if "%PIXFORM_DEVICE%"=="" set "PIXFORM_DEVICE=auto"

if "%ACTIVE_MODELS%"=="" (
    echo.
    echo Choose the active model for this session:
    echo   1 = TripoSR
    echo   2 = Hunyuan3D-2
    echo   3 = TRELLIS
    echo   4 = TRELLIS.2
    echo   A = All installed models ^(higher memory use^)
    set /p ACTIVE_MODELS=Active model [1 ^| 2 ^| 3 ^| 4 ^| A] ^(Enter = A^):
)
if "%ACTIVE_MODELS%"=="" set "ACTIVE_MODELS=A"
set "PIXFORM_ACTIVE_MODELS=%ACTIVE_MODELS%"
if "%PIXFORM_TRELLIS2_OPEN_ONLY%"=="" set "PIXFORM_TRELLIS2_OPEN_ONLY=1"

echo Starting PIXFORM (PIXFORM_DEVICE=%PIXFORM_DEVICE%, PIXFORM_ACTIVE_MODELS=%PIXFORM_ACTIVE_MODELS%)...
start "" "venv\Scripts\python.exe" "backend\app.py"
timeout /t 3 /nobreak >nul
start "" "http://localhost:8000"
