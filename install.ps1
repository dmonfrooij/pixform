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

function Get-CudaToolkitVersion {
    $hasNvcc = $null -ne (Get-Command nvcc -ErrorAction SilentlyContinue)
    if (-not $hasNvcc) { return "" }
    $nvccOut = (nvcc --version 2>&1 | Out-String)
    if ($nvccOut -match "release\s+([0-9]+\.[0-9]+)") { return $matches[1] }
    return ""
}

function Get-TorchCudaVersion($pythonExe) {
    return (& "$pythonExe" -c "import torch; print(torch.version.cuda or '')" 2>$null | Select-Object -First 1).Trim()
}

function Try-EnableMSVCFromVSBuildTools {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) { return $false }
    try {
        $installPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null | Select-Object -First 1).Trim()
        if (-not $installPath) { return $false }

        $vsDevCmd = Join-Path $installPath "Common7\Tools\VsDevCmd.bat"
        if (-not (Test-Path -LiteralPath $vsDevCmd)) { return $false }

        $cmdLine = ('call "{0}" -no_logo -arch=x64 -host_arch=x64 >nul && set' -f $vsDevCmd)
        $dump = & cmd /d /s /c $cmdLine
        foreach ($line in $dump) {
            if ($line -match "^([^=]+)=(.*)$") {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
            }
        }
        return ($null -ne (Get-Command cl -ErrorAction SilentlyContinue))
    } catch {
        Warn "Could not import VS Build Tools environment: $($_.Exception.Message)"
        return $false
    }
}

function Patch-TrellisFlexicubes($pythonExe, $targetPath) {
    if (-not (Test-Path -LiteralPath $targetPath)) { return $false }
    & "$pythonExe" -c @'
from pathlib import Path
import sys

p = Path(sys.argv[1])
txt = p.read_text(encoding="utf-8")
if "from kaolin.utils.testing import check_tensor" not in txt:
    print("  flexicubes.py already patched")
    raise SystemExit(0)

new_import = '''try:
    from kaolin.utils.testing import check_tensor
except ImportError:
    try:
        from kaolin.testing import check_tensor
    except ImportError:
        def check_tensor(tensor, shape, throw=False):
            ok = torch.is_tensor(tensor)
            if ok and shape is not None:
                if tensor.dim() != len(shape):
                    ok = False
                else:
                    for actual, expected in zip(tensor.shape, shape):
                        if expected is not None and actual != expected:
                            ok = False
                            break
            if throw and not ok:
                raise ValueError("Tensor does not match expected shape")
            return ok
'''

txt = txt.replace("from kaolin.utils.testing import check_tensor", new_import, 1)
p.write_text(txt, encoding="utf-8")
print("  flexicubes.py patched (kaolin fallback)")
'@ "$targetPath"
    return ($LASTEXITCODE -eq 0)
}

function Patch-OVoxelSetupForWindows($targetPath) {
    if (-not (Test-Path -LiteralPath $targetPath)) { return $false }
    $content = @'
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")
IS_WINDOWS = platform.system() == "Windows"

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_HIP = True
    else:
        IS_HIP = False
else:
    if BUILD_TARGET == "cuda":
        IS_HIP = False
    elif BUILD_TARGET == "rocm":
        IS_HIP = True
    else:
        raise ValueError(f"Invalid BUILD_TARGET={BUILD_TARGET}")

if not IS_HIP:
    cc_flag = ["-allow-unsupported-compiler"] if IS_WINDOWS else []
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

if IS_WINDOWS:
    extra_compile_args = {
        "cxx": ["/O2", "/std:c++17", "/EHsc", "/permissive-", "/Zc:__cplusplus"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "--extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xcompiler=/std:c++17",
            "-Xcompiler=/EHsc",
            "-Xcompiler=/permissive-",
            "-Xcompiler=/Zc:__cplusplus",
        ] + cc_flag,
    }
else:
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17"] + cc_flag,
    }

setup(
    name="o_voxel",
    packages=[
        'o_voxel',
        'o_voxel.convert',
        'o_voxel.io',
    ],
    ext_modules=[
        CUDAExtension(
            name="o_voxel._C",
            sources=[
                "src/hash/hash.cu",
                "src/convert/flexible_dual_grid.cpp",
                "src/convert/volumetic_attr.cpp",
                "src/serialize/api.cu",
                "src/serialize/hilbert.cu",
                "src/serialize/z_order.cu",
                "src/io/svo.cpp",
                "src/io/filter_parent.cpp",
                "src/io/filter_neighbor.cpp",
                "src/rasterize/rasterize.cu",
                "src/ext.cpp",
            ],
            include_dirs=[
                os.path.join(ROOT, "third_party/eigen"),
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
'@
    Set-Content -LiteralPath $targetPath -Value $content -Encoding UTF8
    return $true
}

function Install-GitCudaPackage($displayName, $repoUrl, $pythonExe, $tempRoot) {
    if (-not (Test-Path -LiteralPath $tempRoot)) {
        New-Item -ItemType Directory -Path $tempRoot | Out-Null
    }
    $srcDir = Join-Path $tempRoot ($displayName -replace '[^A-Za-z0-9_.-]', '_')
    if (Test-Path -LiteralPath $srcDir) { Remove-Item -Recurse -Force -LiteralPath $srcDir }

    Info "Installing $displayName..."
    git clone --quiet --depth 1 --recurse-submodules "$repoUrl" "$srcDir"
    if (Test-Path -LiteralPath $srcDir) {
        git -C "$srcDir" submodule update --init --recursive --depth 1 2>$null | Out-Null
    }
    if (-not (Test-Path -LiteralPath $srcDir)) {
        Warn "$displayName clone failed"
        return $false
    }

    & "$pythonExe" -m pip install --no-build-isolation "$srcDir" -q
    if ($LASTEXITCODE -eq 0) {
        OK "$displayName installed"
        return $true
    }

    Warn "$displayName build failed - see output above."
    return $false
}

function Stop-VenvPythonProcesses($venvPath) {
    $venvPython = Join-Path $venvPath "Scripts\python.exe"
    $venvPrefix = ($venvPath.TrimEnd('\') + '\').ToLowerInvariant()
    try {
        $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction Stop |
            Where-Object {
                $_.ExecutablePath -and (
                    $_.ExecutablePath.ToLowerInvariant() -eq $venvPython.ToLowerInvariant() -or
                    $_.ExecutablePath.ToLowerInvariant().StartsWith($venvPrefix)
                )
            }
        foreach ($p in $procs) {
            Warn "Stopping running venv Python process (PID $($p.ProcessId))"
            try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {}
        }
        if ($procs) { Start-Sleep -Seconds 1 }
    } catch {
        Warn "Could not inspect running Python processes: $($_.Exception.Message)"
    }
}

function Remove-PathSafely($path, $label) {
    if (-not (Test-Path -LiteralPath $path)) { return }

    for ($i = 0; $i -lt 3; $i++) {
        try {
            Remove-Item -Recurse -Force -LiteralPath $path -ErrorAction Stop
            return
        } catch {
            Warn "Failed to remove $label (attempt $($i + 1)/3): $($_.Exception.Message)"
            Start-Sleep -Seconds 1
        }
    }

    $stalePath = "$path.stale.$([DateTime]::Now.ToString('yyyyMMddHHmmss'))"
    try {
        Rename-Item -LiteralPath $path -NewName (Split-Path -Leaf $stalePath) -ErrorAction Stop
        Warn "$label was locked; moved to $(Split-Path -Leaf $stalePath)"
    } catch {
        Fail "Could not clean locked $label. Close running PIXFORM/Python processes and retry. Details: $($_.Exception.Message)"
    }
}

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
if (Test-Path -LiteralPath $venvPath) {
    Stop-VenvPythonProcesses $venvPath
    Remove-PathSafely $venvPath "existing venv"
}
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
    git clone --quiet --depth 1 --recurse-submodules https://github.com/microsoft/TRELLIS.git "$trellisRepo"
    if (Test-Path -LiteralPath $trellisRepo) {
        git -C "$trellisRepo" submodule update --init --recursive --depth 1 2>$null | Out-Null
    }
    if (Test-Path -LiteralPath $trellisRepo) {
        OK "TRELLIS repo cloned"
    } else {
        Warn "Failed to clone TRELLIS repo - skipping TRELLIS install"
    }

    if (Test-Path -LiteralPath $trellisRepo) {
        # TRELLIS runtime dependencies
        Info "Installing xformers (attention backend)..."
        # xformers 0.0.28.post3 matches torch 2.5.1 on the cu124 index.
        # spconv packages are named by CUDA major version: spconv-cu120 supports all CUDA 12.x including 12.4.
        & "$PY" -m pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124 -q
        if ($LASTEXITCODE -eq 0) { OK "xformers installed" } else { Warn "xformers failed" }

        Info "Installing spconv (sparse convolutions)..."
        & "$PY" -m pip install spconv-cu120 -q
        if ($LASTEXITCODE -eq 0) { OK "spconv installed" } else { Warn "spconv failed" }

        Info "Installing utils3d and mesh tools..."
        & "$PY" -m pip install `
            "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" `
            easydict xatlas pyvista pymeshfix igraph -q
        if ($LASTEXITCODE -eq 0) { OK "TRELLIS mesh tools installed" } else { Warn "Some TRELLIS mesh tools failed" }

        # nvdiffrast - optional, enables textured GLB export
        Info "Trying nvdiffrast (optional - textured GLB)..."
        $nvdiffPath = Join-Path $env:TEMP "pixform_nvdiffrast"
        $hasNvcc = $null -ne (Get-Command nvcc -ErrorAction SilentlyContinue)
        if (-not $hasNvcc) {
            Warn "CUDA toolkit (nvcc) not found - skipping nvdiffrast; textured GLB will use plain mesh fallback"
        } else {
            $torchCuda = Get-TorchCudaVersion "$PY"
            $nvccVersion = Get-CudaToolkitVersion
            Info "nvdiffrast toolchain check: torch CUDA=$torchCuda ; nvcc=$nvccVersion"

            if ($torchCuda -and $nvccVersion -and ($torchCuda -ne $nvccVersion)) {
                Warn "CUDA mismatch (torch=$torchCuda, nvcc=$nvccVersion) - skipping nvdiffrast build; textured GLB will use plain mesh fallback"
                Warn "Tip: set CUDA toolkit for this shell to match torch before install."
                Warn "  `$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$torchCuda'"
                Warn "  `$env:CUDA_PATH = `$env:CUDA_HOME"
                Warn "  `$env:Path = `"`$env:CUDA_HOME\bin;`$env:CUDA_HOME\libnvvp;`$env:Path`""
            } else {
                if (Test-Path -LiteralPath $nvdiffPath) { Remove-Item -Recurse -Force -LiteralPath $nvdiffPath }
                git clone --quiet https://github.com/NVlabs/nvdiffrast.git "$nvdiffPath" 2>$null
                if (Test-Path -LiteralPath $nvdiffPath) {
                    & "$PY" -m pip install ninja -q
                    & "$PY" -m pip install --no-build-isolation "$nvdiffPath" -q
                    if ($LASTEXITCODE -eq 0) { OK "nvdiffrast installed (textured GLB enabled)" } else { Warn "nvdiffrast failed - textured GLB will use plain mesh fallback" }
                } else {
                    Warn "nvdiffrast clone failed - textured GLB will use plain mesh fallback"
                }
            }
        }

        # Copy trellis Python package to backend
        $trellisSrc  = Join-Path $trellisRepo "trellis"
        $trellisDest = Join-Path $backendPath "trellis"
        if (Test-Path -LiteralPath $trellisDest) { Remove-Item -Recurse -Force -LiteralPath $trellisDest }
        if (Test-Path -LiteralPath $trellisSrc) {
            Copy-Item -Recurse -LiteralPath $trellisSrc -Destination $trellisDest
            OK "TRELLIS package copied to backend"
            $flexicubesPath = Join-Path $trellisDest "representations\mesh\flexicubes\flexicubes.py"
            if (Patch-TrellisFlexicubes "$PY" "$flexicubesPath") {
                OK "TRELLIS kaolin fallback applied"
            } else {
                Warn "Could not patch TRELLIS flexicubes.py for kaolin fallback"
            }
        } else {
            Warn "TRELLIS trellis/ folder not found in repo"
        }
    }
} else {
    Step "Skipping TRELLIS (CUDA/NVIDIA GPU required)"
    Warn "TRELLIS requires an NVIDIA GPU. Re-run install with -Profile nvidia to enable it."
}

# ── TRELLIS.2 (experimental on Windows) ──────────────────────────────────────
if ($runtimeDevice -eq "cuda") {
    Step "Installing TRELLIS.2 (experimental Windows support)"
    $trellis2Repo = Join-Path $PSScriptRoot "trellis2_repo"
    if (Test-Path -LiteralPath $trellis2Repo) { Remove-Item -Recurse -Force -LiteralPath $trellis2Repo }
    git clone --quiet --depth 1 --recurse-submodules https://github.com/microsoft/TRELLIS.2.git "$trellis2Repo"
    if (Test-Path -LiteralPath $trellis2Repo) {
        git -C "$trellis2Repo" submodule update --init --recursive --depth 1 2>$null | Out-Null
        OK "TRELLIS.2 repo cloned"

        Info "Installing TRELLIS.2 core dependencies..."
        & "$PY" -m pip install `
            imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh `
            transformers "gradio==6.0.1" tensorboard pandas lpips zstandard kornia timm `
            "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" -q
        if ($LASTEXITCODE -eq 0) { OK "TRELLIS.2 core dependencies installed" } else { Warn "Some TRELLIS.2 core dependencies failed" }

        $oVoxelPath = Join-Path $trellis2Repo "o-voxel"
        if (Test-Path -LiteralPath $oVoxelPath) {
            Info "Installing TRELLIS.2 native packages (CuMesh, FlexGEMM, o-voxel)..."

            # Put venv\Scripts first so the pip-installed ninja.exe is found by torch's BuildExtension
            $origPath = $env:PATH
            $origDistutilsUseSdk = $env:DISTUTILS_USE_SDK
            $origMsSdk = $env:MSSdk
            $env:PATH = "$venvPath\Scripts;$env:PATH"
            $origBuildTarget = $env:BUILD_TARGET
            $env:BUILD_TARGET = "cuda"
            $env:DISTUTILS_USE_SDK = "1"
            $env:MSSdk = "1"

            # Pre-flight: check for MSVC (cl.exe) and nvcc (CUDA Toolkit developer headers)
            $hasCl   = $null -ne (Get-Command cl   -ErrorAction SilentlyContinue)
            $hasNvcc = $null -ne (Get-Command nvcc -ErrorAction SilentlyContinue)
            $torchCuda = Get-TorchCudaVersion "$PY"
            $nvccVersion = Get-CudaToolkitVersion

            if (-not $hasCl) {
                Info "cl.exe not found in current shell; trying VS Build Tools environment bootstrap..."
                $hasCl = Try-EnableMSVCFromVSBuildTools
            }

            $clCmd = Get-Command cl -ErrorAction SilentlyContinue
            if ($clCmd) { Info "MSVC compiler: $($clCmd.Source)" }

            if (-not $hasCl) {
                Warn "MSVC compiler (cl.exe) not found - o-voxel needs Visual Studio Build Tools."
                Warn "Install from: https://aka.ms/vs/17/release/vs_BuildTools.exe"
                Warn "If already installed, re-run install.ps1 from 'x64 Native Tools Command Prompt for VS 2022'"
                Warn "Skipping o-voxel - TRELLIS.2 will be disabled."
            } elseif (-not $hasNvcc) {
                Warn "CUDA Toolkit nvcc not found - o-voxel needs the full CUDA Toolkit (not just the runtime)."
                Warn "Install the CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
                Warn "Skipping o-voxel - TRELLIS.2 will be disabled."
            } elseif ($torchCuda -and $nvccVersion -and ($torchCuda -ne $nvccVersion)) {
                Warn "CUDA mismatch for o-voxel build (torch=$torchCuda, nvcc=$nvccVersion)."
                Warn "Set toolkit path to v$torchCuda in this shell, then rerun install:"
                Warn "  `$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$torchCuda'"
                Warn "  `$env:CUDA_PATH = `$env:CUDA_HOME"
                Warn "  `$env:Path = `"`$env:CUDA_HOME\bin;`$env:CUDA_HOME\libnvvp;`$env:Path`""
                Warn "Skipping TRELLIS.2 native deps - TRELLIS.2 will be disabled."
            } else {
                $nativeTempRoot = Join-Path $env:TEMP "pixform_trellis2_native"
                $null = Install-GitCudaPackage "CuMesh" "https://github.com/JeffreyXiang/CuMesh.git" "$PY" "$nativeTempRoot"
                $null = Install-GitCudaPackage "FlexGEMM" "https://github.com/JeffreyXiang/FlexGEMM.git" "$PY" "$nativeTempRoot"

                $oVoxelSetup = Join-Path $oVoxelPath "setup.py"
                if (Patch-OVoxelSetupForWindows "$oVoxelSetup") {
                    OK "o-voxel setup.py patched for Windows"
                } else {
                    Warn "Could not patch o-voxel setup.py for Windows"
                }

                & "$PY" -m pip install --no-build-isolation "$oVoxelPath" -q
                if ($LASTEXITCODE -eq 0) { OK "o-voxel installed" } else {
                    Warn "o-voxel build failed - see output above."
                    Warn "TRELLIS.2 may remain disabled until MSVC/CUDA build issues are resolved."
                }
            }

            if ($null -eq $origBuildTarget) {
                Remove-Item Env:BUILD_TARGET -ErrorAction SilentlyContinue
            } else {
                $env:BUILD_TARGET = $origBuildTarget
            }
            if ($null -eq $origDistutilsUseSdk) {
                Remove-Item Env:DISTUTILS_USE_SDK -ErrorAction SilentlyContinue
            } else {
                $env:DISTUTILS_USE_SDK = $origDistutilsUseSdk
            }
            if ($null -eq $origMsSdk) {
                Remove-Item Env:MSSdk -ErrorAction SilentlyContinue
            } else {
                $env:MSSdk = $origMsSdk
            }
            $env:PATH = $origPath
        } else {
            Warn "TRELLIS.2 o-voxel folder not found"
        }

        $trellis2Src  = Join-Path $trellis2Repo "trellis2"
        $trellis2Dest = Join-Path $backendPath "trellis2"
        if (Test-Path -LiteralPath $trellis2Dest) { Remove-Item -Recurse -Force -LiteralPath $trellis2Dest }
        if (Test-Path -LiteralPath $trellis2Src) {
            Copy-Item -Recurse -LiteralPath $trellis2Src -Destination $trellis2Dest
            OK "TRELLIS.2 package copied to backend"
        } else {
            Warn "TRELLIS.2 package not found in repo"
        }
    } else {
        Warn "Failed to clone TRELLIS.2 repo - skipping TRELLIS.2 integration"
    }
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
txt = re.sub(r'^import rembg\s*`$', '# rembg removed', txt, flags=re.MULTILINE)
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
        import os
        os.environ.setdefault('ATTN_BACKEND', 'xformers')
        os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')
        print(f'  TRELLIS package: found ({len(list(trellis_pkg.iterdir()))} items)')
        try:
            import spconv
            import easydict
            from trellis.pipelines import TrellisImageTo3DPipeline
            print('[SPARSE] Backend: spconv, Attention: xformers')
            print('  TRELLIS: OK')
        except ImportError as ie:
            print(f'  TRELLIS: FAILED - {ie}')
    else:
        print('  TRELLIS package: not found (CUDA-only feature)')
except Exception as e:
    print(f'  TRELLIS: FAILED - {e}')

try:
    trellis2_pkg = pathlib.Path('backend/trellis2')
    if trellis2_pkg.exists():
        print(f'  TRELLIS.2 package: found ({len(list(trellis2_pkg.iterdir()))} items)')
        import importlib.util
        import os
        missing = [d for d in ('cumesh', 'flex_gemm', 'o_voxel') if importlib.util.find_spec(d) is None]
        if missing:
            missing_str = ', '.join(missing)
            print(f'  TRELLIS.2 runtime deps missing: {missing_str} (TRELLIS.2 disabled)')
        else:
            os.environ.setdefault('SPARSE_CONV_BACKEND', 'spconv')
            os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')
            os.environ.setdefault('ATTN_BACKEND', 'xformers')
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            print('  TRELLIS.2: OK')
    else:
        print('  TRELLIS.2 package: not found (optional CUDA feature)')
except Exception as e:
    print(f'  TRELLIS.2: FAILED - {e}')

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

