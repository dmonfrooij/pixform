# PIXFORM - Installation Script
# Right-click > "Run with PowerShell"
param(
    [ValidateSet("auto", "nvidia", "cpu")]
    [string]$Profile = "auto"
)

Set-Location $PSScriptRoot
$ErrorActionPreference = "Continue"

function Step($m)  { Write-Host "`n== $m ==" -ForegroundColor Cyan }
function OK($m)    { Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn($m)  { Write-Host "  [!]  $m" -ForegroundColor Yellow }
function Fail($m)  { Write-Host "`n  [ERROR] $m" -ForegroundColor Red; Read-Host "Press Enter to exit"; exit 1 }
function Info($m)  { Write-Host "  $m" -ForegroundColor Gray }

Clear-Host
Write-Host "  PIXFORM - Image to 3D Pipeline" -ForegroundColor Yellow
Write-Host "  Installing..." -ForegroundColor Gray

# ── Prerequisites ──────────────────────────────────────────────────────────────
Step "Checking prerequisites"
try { git --version | Out-Null; OK "Git found" }
catch { Fail "Git not found. Install from: https://git-scm.com/download/win" }

$py = $null
$pyCmd = $null
$pyArgs = @()
@(
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "C:\Python310\python.exe",
    "C:\Program Files\Python310\python.exe"
) | ForEach-Object { if ((Test-Path -LiteralPath $_) -and -not $py) { $py = $_ } }
if (-not $py) {
    try {
        $v = py -3.10 --version 2>&1
        if ($v -match "3\.10") {
            $pyCmd = "py"
            $pyArgs = @("-3.10")
        }
    } catch {}
} else {
    $pyCmd = $py
}
if (-not $pyCmd) { Fail "Python 3.10 not found. Download: https://www.python.org/downloads/release/python-31011/" }
OK "Python 3.10 ready"

try {
    $nvOut = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
    OK "GPU: $nvOut"
} catch { Warn "Could not detect GPU" }

$runtimeDevice = "cpu"
if ($Profile -eq "nvidia") {
    $runtimeDevice = "cuda"
} elseif ($Profile -eq "auto") {
    try { nvidia-smi | Out-Null; $runtimeDevice = "cuda" } catch { $runtimeDevice = "cpu" }
}
Info "Install profile: $Profile (runtime device target: $runtimeDevice)"

# ── Virtual environment ────────────────────────────────────────────────────────
Step "Creating virtual environment"
$venvPath = Join-Path $PSScriptRoot "venv"
if (Test-Path -LiteralPath $venvPath) { Remove-Item -Recurse -Force -LiteralPath $venvPath }
& $pyCmd @pyArgs -m venv "$venvPath"
$PY = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PY)) { Fail "Failed to create venv" }
& "$PY" -m pip install --upgrade pip setuptools wheel -q
OK "Virtual environment ready"

# ── PyTorch ────────────────────────────────────────────────────────────────────
if ($runtimeDevice -eq "cuda") {
    Step "Installing PyTorch 2.5.1 + CUDA 12.4 (~2.5 GB)"
    Info "This is the largest download, please wait..."
    & "$PY" -m pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" --index-url https://download.pytorch.org/whl/cu124 -q
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch CUDA installation failed" }
    OK "PyTorch CUDA installed"
} else {
    Step "Installing PyTorch 2.5.1 (CPU)"
    & "$PY" -m pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" -q
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch CPU installation failed" }
    OK "PyTorch CPU installed"
}

# ── Core dependencies ──────────────────────────────────────────────────────────
Step "Installing core dependencies"
if ($runtimeDevice -eq "cuda") {
    $rembgPkg = "rembg[gpu]"
    $onnxPkg = "onnxruntime-gpu"
} else {
    $rembgPkg = "rembg"
    $onnxPkg = "onnxruntime"
}
& "$PY" -m pip install `
    "numpy==1.26.4" "Pillow>=10.0" "trimesh[easy]" "scipy" "imageio" `
    "einops" "omegaconf>=2.3" "huggingface_hub" "transformers>=4.40" `
    "accelerate>=0.30" "diffusers>=0.27" "fastapi==0.115.5" `
    "uvicorn[standard]==0.32.1" "python-multipart==0.0.12" "pydantic>=2.0" `
    "httpx" "$rembgPkg" "$onnxPkg" "open3d" "PyMCubes" "pyrender" -q
if ($LASTEXITCODE -ne 0) { Fail "Core dependencies failed" }
OK "Core dependencies installed"

# ── Hunyuan3D-2 ────────────────────────────────────────────────────────────────
Step "Cloning Hunyuan3D-2 (quality model)"
$hunyuanRepo = Join-Path $PSScriptRoot "hunyuan3d_repo"
if (Test-Path -LiteralPath $hunyuanRepo) { Remove-Item -Recurse -Force -LiteralPath $hunyuanRepo }
git clone --quiet --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$hunyuanRepo"
if (-not (Test-Path -LiteralPath $hunyuanRepo)) { Fail "Failed to clone Hunyuan3D-2" }
OK "Hunyuan3D-2 cloned"

# Show what was cloned
Info "Repo top-level contents:"
Get-ChildItem -LiteralPath $hunyuanRepo | ForEach-Object { Info "  $($_.Name)" }

# Install Hunyuan3D-2 requirements if present
$hunyuanReq = Join-Path $hunyuanRepo "requirements.txt"
if (Test-Path -LiteralPath $hunyuanReq) {
    & "$PY" -m pip install -r "$hunyuanReq" -q
    OK "Hunyuan3D-2 requirements installed"
} else {
    Warn "No requirements.txt found in repo"
}

# Find and copy hy3dshape folder
$backendPath  = Join-Path $PSScriptRoot "backend"
# The module is called hy3dgen (not hy3dshape)
$hy3dDest = Join-Path $backendPath "hy3dgen"
if (Test-Path -LiteralPath $hy3dDest) { Remove-Item -Recurse -Force -LiteralPath $hy3dDest }

$hy3dSrc = Join-Path $hunyuanRepo "hy3dgen"
if (Test-Path -LiteralPath $hy3dSrc) {
    Copy-Item -Recurse -LiteralPath $hy3dSrc -Destination $hy3dDest
    OK "hy3dgen copied to backend"
    # Rename hy3dgen/rembg.py to avoid conflict with the real rembg package
    $conflictFile = Join-Path $hy3dDest "rembg.py"
    if (Test-Path -LiteralPath $conflictFile) {
        Rename-Item -LiteralPath $conflictFile -NewName "rembg_hy3d.py"
        OK "Renamed conflicting rembg.py -> rembg_hy3d.py"
    }
} else {
    Warn "hy3dgen folder not found"
}

# ── TRELLIS ────────────────────────────────────────────────────────────────────
if ($runtimeDevice -eq "cuda") {
    Step "Installing TRELLIS (best-quality model)"

    # Clone TRELLIS repo
    $trellisRepo = Join-Path $PSScriptRoot "trellis_repo"
    if (Test-Path -LiteralPath $trellisRepo) { Remove-Item -Recurse -Force -LiteralPath $trellisRepo }
    git clone --quiet --depth 1 https://github.com/microsoft/TRELLIS.git "$trellisRepo"
    if (Test-Path -LiteralPath $trellisRepo) {
        OK "TRELLIS repo cloned"
    } else {
        Warn "Failed to clone TRELLIS repo — skipping TRELLIS install"
    }

    if (Test-Path -LiteralPath $trellisRepo) {
        # TRELLIS runtime dependencies
        Info "Installing xformers (attention backend)..."
        # xformers uses the cu124 index (matching PyTorch 2.5.1+cu124 installed above).
        # spconv packages are named by CUDA major version: spconv-cu120 supports all CUDA 12.x including 12.4.
        & "$PY" -m pip install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124 -q
        if ($LASTEXITCODE -eq 0) { OK "xformers installed" } else { Warn "xformers failed" }

        Info "Installing spconv (sparse convolutions)..."
        & "$PY" -m pip install spconv-cu120 -q
        if ($LASTEXITCODE -eq 0) { OK "spconv installed" } else { Warn "spconv failed" }

        Info "Installing utils3d and mesh tools..."
        & "$PY" -m pip install `
            "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" `
            xatlas pyvista pymeshfix igraph -q
        if ($LASTEXITCODE -eq 0) { OK "TRELLIS mesh tools installed" } else { Warn "Some TRELLIS mesh tools failed" }

        # nvdiffrast — optional, enables textured GLB export
        Info "Trying nvdiffrast (optional — textured GLB)..."
        $nvdiffPath = Join-Path $env:TEMP "pixform_nvdiffrast"
        if (Test-Path -LiteralPath $nvdiffPath) { Remove-Item -Recurse -Force -LiteralPath $nvdiffPath }
        git clone --quiet https://github.com/NVlabs/nvdiffrast.git "$nvdiffPath" 2>$null
        if (Test-Path -LiteralPath $nvdiffPath) {
            & "$PY" -m pip install "$nvdiffPath" -q
            if ($LASTEXITCODE -eq 0) { OK "nvdiffrast installed (textured GLB enabled)" } else { Warn "nvdiffrast failed — textured GLB will use plain mesh fallback" }
        } else {
            Warn "nvdiffrast clone failed — textured GLB will use plain mesh fallback"
        }

        # Copy trellis Python package to backend
        $trellisSrc  = Join-Path $trellisRepo "trellis"
        $trellisDest = Join-Path $backendPath "trellis"
        if (Test-Path -LiteralPath $trellisDest) { Remove-Item -Recurse -Force -LiteralPath $trellisDest }
        if (Test-Path -LiteralPath $trellisSrc) {
            Copy-Item -Recurse -LiteralPath $trellisSrc -Destination $trellisDest
            OK "TRELLIS package copied to backend"
        } else {
            Warn "TRELLIS trellis/ folder not found in repo"
        }
    }
} else {
    Step "Skipping TRELLIS (CUDA/NVIDIA GPU required)"
    Warn "TRELLIS requires an NVIDIA GPU. Re-run install with -Profile nvidia to enable it."
}

# ── TripoSR ────────────────────────────────────────────────────────────────────
Step "Cloning TripoSR (fast model)"
$triposrRepo = Join-Path $PSScriptRoot "triposr_repo"
if (Test-Path -LiteralPath $triposrRepo) { Remove-Item -Recurse -Force -LiteralPath $triposrRepo }
git clone --quiet --depth 1 https://github.com/VAST-AI-Research/TripoSR.git "$triposrRepo"
$triposrTsr = Join-Path $triposrRepo "tsr"
if (-not (Test-Path -LiteralPath $triposrTsr)) { Fail "Failed to clone TripoSR" }

$backendTsr = Join-Path $backendPath "tsr"
if (Test-Path -LiteralPath $backendTsr) { Remove-Item -Recurse -Force -LiteralPath $backendTsr }
Copy-Item -Recurse -LiteralPath $triposrTsr -Destination $backendTsr
OK "TripoSR copied"

Info "Patching TripoSR source..."
& "$PY" -c @"
import re, ast

p = 'backend/tsr/utils.py'
txt = open(p, encoding='utf-8').read()
txt = txt.replace(
    'image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)',
    'image = torch.tensor(np.array(image).astype(np.float32) / 255.0)'
)
txt = re.sub(r'^import rembg\s*$', '# rembg removed', txt, flags=re.MULTILINE)
txt = re.sub(r'from rembg import remove', 'try:\n    from rembg import remove\nexcept ImportError:\n    def remove(img, **kw): return img', txt)
txt = re.sub(r'rembg\.remove\(([^)]+)\)', r'remove(\1)', txt)
ast.parse(txt)
open(p, 'w', encoding='utf-8').write(txt)
print('  utils.py patched')

p2 = 'backend/tsr/models/isosurface.py'
txt2 = open(p2, encoding='utf-8').read()
old = 'from torchmcubes import marching_cubes'
new = 'try:\n    from torchmcubes import marching_cubes\nexcept ImportError:\n    import mcubes as _mc\n    import torch as _t\n    import numpy as _np\n    def marching_cubes(vol, threshold):\n        v, f = _mc.marching_cubes(vol.cpu().numpy(), float(threshold))\n        return _t.tensor(v.astype(_np.float32)), _t.tensor(f.astype(_np.int64))\n'
if old in txt2:
    txt2 = txt2.replace(old, new)
    ast.parse(txt2)
    open(p2, 'w', encoding='utf-8').write(txt2)
    print('  isosurface.py patched')
"@
OK "TripoSR patched"

# ── Pin NumPy last ─────────────────────────────────────────────────────────────
Step "Pinning NumPy/OpenCV compatibility"
& "$PY" -m pip install "numpy==1.26.4" "opencv-python==4.10.0.84" --force-reinstall -q
OK "NumPy/OpenCV pinned (1.26.4 / 4.10.0.84)"

# ── Validate ───────────────────────────────────────────────────────────────────
Step "Validating installation"
& "$PY" -c @"
import sys, pathlib
sys.path.insert(0, 'backend')

import torch
print(f'  PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available(): print(f'  GPU: {torch.cuda.get_device_name(0)}')

import numpy as np
print(f'  NumPy {np.__version__}')

t = torch.tensor(np.zeros((4,4), dtype=np.float32))
print(f'  numpy->torch: OK {t.shape}')

try:
    from tsr.system import TSR
    print('  TripoSR: OK')
except Exception as e:
    print(f'  TripoSR: FAILED - {e}')

try:
    hy3d = pathlib.Path('backend/hy3dgen')
    if hy3d.exists():
        sys.path.insert(0, str(hy3d))
        print(f'  hy3dgen folder: found ({len(list(hy3d.iterdir()))} items)')
    else:
        print('  hy3dgen folder: NOT FOUND')
except Exception as e:
    print(f'  hy3dgen: FAILED - {e}')

try:
    trellis_pkg = pathlib.Path('backend/trellis')
    if trellis_pkg.exists():
        print(f'  TRELLIS package: found ({len(list(trellis_pkg.iterdir()))} items)')
    else:
        print('  TRELLIS package: not found (CUDA-only feature)')
except Exception as e:
    print(f'  TRELLIS: FAILED - {e}')

try:
    from rembg import remove
    print('  rembg: OK')
except Exception as e:
    print(f'  rembg: FAILED - {e}')

try:
    import open3d
    print(f'  open3d {open3d.__version__}: OK')
except Exception as e:
    print(f'  open3d: FAILED - {e}')

try:
    import cv2
    print(f'  opencv-python {cv2.__version__}: OK')
except Exception as e:
    print(f'  opencv-python: FAILED - {e}')
"@

Write-Host "`n  ========================================" -ForegroundColor Green
Write-Host "   PIXFORM installed!" -ForegroundColor Green
Write-Host "   Run PIXFORM.bat to start." -ForegroundColor Green
Write-Host "  ========================================" -ForegroundColor Green

Set-Content -LiteralPath (Join-Path $PSScriptRoot ".pixform_device") -Value $runtimeDevice -Encoding ASCII
OK "Saved runtime device preference to .pixform_device"

Read-Host "Press Enter to exit"
