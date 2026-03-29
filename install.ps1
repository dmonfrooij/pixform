# PIXFORM - Installation Script
# Right-click > "Run with PowerShell"
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
@(
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "C:\Python310\python.exe",
    "C:\Program Files\Python310\python.exe"
) | ForEach-Object { if ((Test-Path -LiteralPath $_) -and -not $py) { $py = $_ } }
if (-not $py) {
    try { $v = py -3.10 --version 2>&1; if ($v -match "3\.10") { $py = "py -3.10" } } catch {}
}
if (-not $py) { Fail "Python 3.10 not found. Download: https://www.python.org/downloads/release/python-31011/" }
OK "Python 3.10: $py"

try {
    $nvOut = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
    OK "GPU: $nvOut"
} catch { Warn "Could not detect GPU" }

# ── Virtual environment ────────────────────────────────────────────────────────
Step "Creating virtual environment"
$venvPath = Join-Path $PSScriptRoot "venv"
if (Test-Path -LiteralPath $venvPath) { Remove-Item -Recurse -Force -LiteralPath $venvPath }
& $py -m venv "$venvPath"
$PY = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PY)) { Fail "Failed to create venv" }
& "$PY" -m pip install --upgrade pip setuptools wheel -q
OK "Virtual environment ready"

# ── PyTorch ────────────────────────────────────────────────────────────────────
Step "Installing PyTorch 2.5.1 + CUDA 12.4 (~2.5 GB)"
Info "This is the largest download, please wait..."
& "$PY" -m pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" --index-url https://download.pytorch.org/whl/cu124 -q
if ($LASTEXITCODE -ne 0) { Fail "PyTorch installation failed" }
OK "PyTorch installed"

# ── Core dependencies ──────────────────────────────────────────────────────────
Step "Installing core dependencies"
& "$PY" -m pip install `
    "numpy==1.26.4" "Pillow>=10.0" "trimesh[easy]" "scipy" "imageio" `
    "einops" "omegaconf>=2.3" "huggingface_hub" "transformers>=4.40" `
    "accelerate>=0.30" "diffusers>=0.27" "fastapi==0.115.5" `
    "uvicorn[standard]==0.32.1" "python-multipart==0.0.12" "pydantic>=2.0" `
    "httpx" "rembg[gpu]" "onnxruntime-gpu" "open3d" "PyMCubes" "pyrender" -q
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
Step "Pinning NumPy 1.26.4"
& "$PY" -m pip install "numpy==1.26.4" --force-reinstall -q
OK "NumPy pinned at 1.26.4"

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
    from rembg import remove
    print('  rembg: OK')
except Exception as e:
    print(f'  rembg: FAILED - {e}')

try:
    import open3d
    print(f'  open3d {open3d.__version__}: OK')
except Exception as e:
    print(f'  open3d: FAILED - {e}')
"@

Write-Host "`n  ========================================" -ForegroundColor Green
Write-Host "   PIXFORM installed!" -ForegroundColor Green
Write-Host "   Run PIXFORM.bat to start." -ForegroundColor Green
Write-Host "  ========================================" -ForegroundColor Green
Read-Host "Press Enter to exit"
