# PIXFORM - Installation Script
# Right-click > "Run with PowerShell"
param(
    [ValidateSet("auto", "nvidia", "cpu")]
    [string]$Profile = "auto",
    [string]$Models = ""
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

function Get-NvidiaGpuInfo {
    $result = [ordered]@{
        Found = $false
        Lines = @()
        Source = ""
    }

    $nvidiaSmiCandidates = @(
        "nvidia-smi",
        "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        "$env:ProgramW6432\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    )

    foreach ($candidate in $nvidiaSmiCandidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        $isExePath = $candidate -like "*.exe"
        if ($isExePath -and -not (Test-Path -LiteralPath $candidate)) { continue }
        try {
            $out = & $candidate --query-gpu=name,memory.total --format=csv,noheader 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $lines = @($out | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ })
                if ($lines.Count -gt 0) {
                    $result.Found = $true
                    $result.Lines = $lines
                    $result.Source = "nvidia-smi"
                    return $result
                }
            }
        } catch {}
    }

    # Fallback for systems where nvidia-smi is unavailable but NVIDIA adapters exist.
    try {
        $adapters = Get-CimInstance Win32_VideoController -ErrorAction Stop |
            Where-Object { $_.Name -match "NVIDIA" }
        if ($adapters) {
            $result.Found = $true
            $result.Lines = @(
                $adapters | ForEach-Object {
                    $name = ($_.Name | Out-String).Trim()
                    if ($_.AdapterRAM) {
                        $gb = [math]::Round(([double]$_.AdapterRAM / 1GB), 1)
                        "$name, ${gb} GB (WMI)"
                    } else {
                        "$name (WMI)"
                    }
                }
            )
            $result.Source = "wmi"
        }
    } catch {}

    return $result
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
    $patchScript = Join-Path $env:TEMP "pixform_patch_flexicubes.py"
    @'
from pathlib import Path
import ast
import sys

p = Path(sys.argv[1])
txt = p.read_text(encoding="utf-8")
needle = "from kaolin.utils.testing import check_tensor"

if needle not in txt:
    if "from kaolin.testing import check_tensor" in txt or "def check_tensor(tensor, shape, throw=False):" in txt:
        print("  flexicubes.py already patched")
        raise SystemExit(0)
    print("  flexicubes.py patch skipped: expected kaolin import not found")
    raise SystemExit(2)

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

patched = txt.replace(needle, new_import, 1)
ast.parse(patched)
p.write_text(patched, encoding="utf-8")
print("  flexicubes.py patched (kaolin fallback)")
'@ | Set-Content -LiteralPath $patchScript -Encoding UTF8

    & "$pythonExe" "$patchScript" "$targetPath"
    $exitCode = $LASTEXITCODE
    Remove-Item -LiteralPath $patchScript -Force -ErrorAction SilentlyContinue
    return ($exitCode -eq 0)
}

function Patch-Trellis2ImageExtractorForOSS($pythonExe, $targetPath) {
    if (-not (Test-Path -LiteralPath $targetPath)) { return $false }
    $patchScript = Join-Path $env:TEMP "pixform_patch_trellis2_image_extractor.py"
    @'
from pathlib import Path
import ast
import sys

p = Path(sys.argv[1])
txt = p.read_text(encoding="utf-8")

if "self.backend = \"dinov3\"" in txt and "Falling back to open DINOv2" in txt:
    print("  trellis2 image_feature_extractor.py already patched")
    raise SystemExit(0)

if "from transformers import DINOv3ViTModel" in txt and "import logging" not in txt:
    txt = txt.replace("from transformers import DINOv3ViTModel\n", "from transformers import DINOv3ViTModel\nimport logging\n", 1)

if "logger = logging.getLogger(\"pixform\")" not in txt:
    marker = "from PIL import Image\n\n"
    if marker in txt:
        txt = txt.replace(marker, "from PIL import Image\n\n\nlogger = logging.getLogger(\"pixform\")\n\n", 1)

old_init = """    def __init__(self, model_name: str, image_size=512):\n        self.model_name = model_name\n        self.model = DINOv3ViTModel.from_pretrained(model_name)\n        self.model.eval()\n"""
new_init = """    def __init__(self, model_name: str, image_size=512):\n        self.model_name = model_name\n        self.backend = \"dinov3\"\n        try:\n            self.model = DINOv3ViTModel.from_pretrained(model_name)\n        except Exception as e:\n            # Fallback to public DINOv2 when DINOv3 weights are gated/unavailable.\n            self.backend = \"dinov2\"\n            fallback_name = \"dinov2_vitl14_reg\"\n            logger.warning(\n                \"DINOv3 load failed (%s). Falling back to open DINOv2 (%s).\",\n                e,\n                fallback_name,\n            )\n            self.model = torch.hub.load(\"facebookresearch/dinov2\", fallback_name, pretrained=True)\n        self.model.eval()\n"""
if old_init in txt:
    txt = txt.replace(old_init, new_init, 1)

old_extract = """    def extract_features(self, image: torch.Tensor) -> torch.Tensor:\n        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)\n"""
new_extract = """    def extract_features(self, image: torch.Tensor) -> torch.Tensor:\n        if self.backend == \"dinov2\":\n            features = self.model(image, is_training=True)[\"x_prenorm\"]\n            return F.layer_norm(features, features.shape[-1:])\n        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)\n"""
if old_extract in txt:
    txt = txt.replace(old_extract, new_extract, 1)

ast.parse(txt)
p.write_text(txt, encoding="utf-8")
print("  trellis2 image_feature_extractor.py patched (open-source fallback)")
'@ | Set-Content -LiteralPath $patchScript -Encoding UTF8

    & "$pythonExe" "$patchScript" "$targetPath"
    $exitCode = $LASTEXITCODE
    Remove-Item -LiteralPath $patchScript -Force -ErrorAction SilentlyContinue
    return ($exitCode -eq 0)
}

function Patch-Trellis2PipelineAliases($pythonExe, $targetPath) {
    if (-not (Test-Path -LiteralPath $targetPath)) { return $false }
    $patchScript = Join-Path $env:TEMP "pixform_patch_trellis2_aliases.py"
    @'
from pathlib import Path
import ast
import sys

p = Path(sys.argv[1])
txt = p.read_text(encoding="utf-8")

alias_line = "Trellis2ImageTo3DPipeline = TrellisImageTo3DPipeline"
if alias_line in txt:
    print("  trellis2 pipelines alias already patched")
    raise SystemExit(0)

anchor = "from .trellis_text_to_3d import TrellisTextTo3DPipeline"
if anchor not in txt:
    print("  trellis2 pipelines alias patch skipped: import anchor not found")
    raise SystemExit(2)

txt = txt.replace(anchor, anchor + "\n\n# Compatibility alias expected by PIXFORM runtime.\n" + alias_line, 1)
ast.parse(txt)
p.write_text(txt, encoding="utf-8")
print("  trellis2 pipelines alias patched")
'@ | Set-Content -LiteralPath $patchScript -Encoding UTF8

    & "$pythonExe" "$patchScript" "$targetPath"
    $exitCode = $LASTEXITCODE
    Remove-Item -LiteralPath $patchScript -Force -ErrorAction SilentlyContinue
    return ($exitCode -eq 0)
}

function Patch-Trellis2ImagePipelineCompat($pythonExe, $targetPath) {
    if (-not (Test-Path -LiteralPath $targetPath)) { return $false }
    $patchScript = Join-Path $env:TEMP "pixform_patch_trellis2_image_pipeline.py"
    @'
from pathlib import Path
import ast
import sys

p = Path(sys.argv[1])
txt = p.read_text(encoding="utf-8")
changed = False

if "cond_resolution = max(patch_size, (int(resolution) // patch_size) * patch_size)" not in txt:
    old = "        self.image_cond_model.image_size = resolution\n"
    new = (
        "        # Patch-based backbones (e.g. DINO) require image size divisible by patch size (14).\n"
        "        patch_size = 14\n"
        "        cond_resolution = max(patch_size, (int(resolution) // patch_size) * patch_size)\n"
        "        self.image_cond_model.image_size = cond_resolution\n"
    )
    if old in txt:
        txt = txt.replace(old, new, 1)
        changed = True

if "Windows fallback path may not have optional native mesh repair extensions." not in txt:
    old = "            m.fill_holes()\n"
    new = (
        "            try:\n"
        "                m.fill_holes()\n"
        "            except Exception:\n"
        "                # Windows fallback path may not have optional native mesh repair extensions.\n"
        "                pass\n"
    )
    if old in txt:
        txt = txt.replace(old, new, 1)
        changed = True

if not changed:
    print("  trellis2 image_to_3d pipeline already patched")
    raise SystemExit(0)

ast.parse(txt)
p.write_text(txt, encoding="utf-8")
print("  trellis2 image_to_3d pipeline patched (patch14 + mesh fallback guards)")
'@ | Set-Content -LiteralPath $patchScript -Encoding UTF8

    & "$pythonExe" "$patchScript" "$targetPath"
    $exitCode = $LASTEXITCODE
    Remove-Item -LiteralPath $patchScript -Force -ErrorAction SilentlyContinue
    return ($exitCode -eq 0)
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

$AllModelKeys = @("triposr", "hunyuan", "trellis", "trellis2")
$ModelSelectionHelp = "all | triposr,hunyuan,trellis,trellis2 | 1,2,3,4"

function Parse-ModelSelection([string]$rawSelection) {
    $selected = [ordered]@{
        triposr = $false
        hunyuan = $false
        trellis = $false
        trellis2 = $false
    }
    $aliases = @{
        "1" = "triposr"
        "2" = "hunyuan"
        "3" = "trellis"
        "4" = "trellis2"
        "hunyuan3d-2" = "hunyuan"
        "hunyuan3d2" = "hunyuan"
        "trellis.2" = "trellis2"
    }

    if ([string]::IsNullOrWhiteSpace($rawSelection) -or $rawSelection.Trim().ToLowerInvariant() -eq "all") {
        foreach ($name in $AllModelKeys) { $selected[$name] = $true }
        return $selected
    }

    $tokens = $rawSelection -split "[\s,;]+" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    foreach ($token in $tokens) {
        $name = $token.Trim().ToLowerInvariant()
        if ($aliases.ContainsKey($name)) { $name = $aliases[$name] }
        if (-not $selected.Contains($name)) {
            throw "Unknown model '$token'. Use: $ModelSelectionHelp"
        }
        $selected[$name] = $true
    }

    if (-not (($selected.Values | Where-Object { $_ }).Count)) {
        throw "No valid models selected. Use: $ModelSelectionHelp"
    }

    return $selected
}

function Prompt-ModelSelection {
    Info "Choose which models to install:"
    Info "  1 = TripoSR      (fast, CPU/MPS/CUDA)"
    Info "  2 = Hunyuan3D-2  (quality, CUDA only)"
    Info "  3 = TRELLIS      (best quality, CUDA only)"
    Info "  4 = TRELLIS.2    (experimental, CUDA only)"
    $reply = Read-Host "Install models [$ModelSelectionHelp] (Enter = all)"
    return Parse-ModelSelection $reply
}

function Get-SelectedModelNames($selectedModels) {
    return @($AllModelKeys | Where-Object { $selectedModels[$_] })
}

function Remove-ModelArtifacts($label, [string[]]$paths) {
    foreach ($path in $paths) {
        if (Test-Path -LiteralPath $path) {
            Remove-PathSafely $path $label
        }
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

$gpuInfo = Get-NvidiaGpuInfo
if ($gpuInfo.Found) {
    foreach ($gpuLine in $gpuInfo.Lines) {
        OK "GPU: $gpuLine"
    }
} else {
    Warn "Could not detect NVIDIA GPU"
}

$runtimeDevice = "cpu"
if ($Profile -eq "nvidia") {
    $runtimeDevice = "cuda"
} elseif ($Profile -eq "auto") {
    # Auto profile prefers NVIDIA/CUDA whenever any NVIDIA GPU is detectable.
    if ($gpuInfo.Found) { $runtimeDevice = "cuda" } else { $runtimeDevice = "cpu" }
}
Info "Install profile: $Profile (runtime device target: $runtimeDevice)"

try {
    if ([string]::IsNullOrWhiteSpace($Models)) {
        $SelectedModels = Prompt-ModelSelection
    } else {
        $SelectedModels = Parse-ModelSelection $Models
    }
} catch {
    Fail $_.Exception.Message
}

if ($runtimeDevice -ne "cuda") {
     foreach ($gpuOnlyModel in @("hunyuan", "trellis", "trellis2")) {
         if ($SelectedModels[$gpuOnlyModel]) {
             Warn "$gpuOnlyModel requires CUDA/NVIDIA and will be skipped for profile '$Profile'."
             $SelectedModels[$gpuOnlyModel] = $false
         }
     }
 }

 # On Windows, TRELLIS.2 C++ extensions can be tricky; ask user to confirm before proceeding.
 if ([Environment]::OSVersion.Platform -eq "Win32NT" -and $SelectedModels["trellis2"]) {
     Warn "TRELLIS.2 requires C++ extensions (CuMesh, FlexGEMM, o-voxel) that may be hard to build on Windows."
     Warn "The installer will attempt to build them automatically. Failures are non-fatal: TRELLIS.2 will fall back to pure-Python mode."
     $t2confirm = Read-Host "  Continue with TRELLIS.2 installation? [Y/n]"
     if ($t2confirm.Trim().ToLowerInvariant() -eq "n") {
         Warn "TRELLIS.2 skipped at user request."
         $SelectedModels["trellis2"] = $false
     } else {
         Info "Proceeding with TRELLIS.2 installation..."
     }
 }

$selectedModelNames = Get-SelectedModelNames $SelectedModels
if (-not $selectedModelNames.Count) {
    Fail "No installable models remain for the selected profile. Choose at least TripoSR or rerun with -Profile nvidia."
}
Info ("Selected models: " + ($selectedModelNames -join ", "))

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

$backendPath = Join-Path $PSScriptRoot "backend"
$hunyuanRepo = Join-Path $PSScriptRoot "hunyuan3d_repo"
$trellisRepo = Join-Path $PSScriptRoot "trellis_repo"
$trellis2Repo = Join-Path $PSScriptRoot "trellis2_repo"
$triposrRepo = Join-Path $PSScriptRoot "triposr_repo"
$hy3dDest = Join-Path $backendPath "hy3dgen"
$trellisDest = Join-Path $backendPath "trellis"
$trellis2Dest = Join-Path $backendPath "trellis2"
$backendTsr = Join-Path $backendPath "tsr"

if (-not $SelectedModels["hunyuan"]) { Remove-ModelArtifacts "Hunyuan3D-2 files" @($hunyuanRepo, $hy3dDest) }
if (-not $SelectedModels["trellis"]) { Remove-ModelArtifacts "TRELLIS files" @($trellisRepo, $trellisDest) }
if (-not $SelectedModels["trellis2"]) { Remove-ModelArtifacts "TRELLIS.2 files" @($trellis2Repo, $trellis2Dest) }
if (-not $SelectedModels["triposr"]) { Remove-ModelArtifacts "TripoSR files" @($triposrRepo, $backendTsr) }

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
    "httpx" "$rembgPkg" "$onnxPkg" "open3d" "PyMCubes" "pyrender" "kornia" "timm" -q
if ($LASTEXITCODE -ne 0) { Fail "Core dependencies failed" }
OK "Core dependencies installed"

if ($SelectedModels["hunyuan"]) {
    # ── Hunyuan3D-2 ────────────────────────────────────────────────────────────
    Step "Cloning Hunyuan3D-2 (quality model)"
    if (Test-Path -LiteralPath $hunyuanRepo) { Remove-Item -Recurse -Force -LiteralPath $hunyuanRepo }
    git clone --quiet --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$hunyuanRepo"
    if (-not (Test-Path -LiteralPath $hunyuanRepo)) { Fail "Failed to clone Hunyuan3D-2" }
    OK "Hunyuan3D-2 cloned"

    Info "Repo top-level contents:"
    Get-ChildItem -LiteralPath $hunyuanRepo | ForEach-Object { Info "  $($_.Name)" }

    $hunyuanReq = Join-Path $hunyuanRepo "requirements.txt"
    if (Test-Path -LiteralPath $hunyuanReq) {
        & "$PY" -m pip install -r "$hunyuanReq" -q
        OK "Hunyuan3D-2 requirements installed"
    } else {
        Warn "No requirements.txt found in repo"
    }

    if (Test-Path -LiteralPath $hy3dDest) { Remove-Item -Recurse -Force -LiteralPath $hy3dDest }

    $hy3dSrc = Join-Path $hunyuanRepo "hy3dgen"
    if (Test-Path -LiteralPath $hy3dSrc) {
        Copy-Item -Recurse -LiteralPath $hy3dSrc -Destination $hy3dDest
        OK "hy3dgen copied to backend"
        $conflictFile = Join-Path $hy3dDest "rembg.py"
        if (Test-Path -LiteralPath $conflictFile) {
            Rename-Item -LiteralPath $conflictFile -NewName "rembg_hy3d.py"
            OK "Renamed conflicting rembg.py -> rembg_hy3d.py"
        }
    } else {
        Warn "hy3dgen folder not found"
    }
} else {
    Step "Skipping Hunyuan3D-2 (not selected)"
}

# ── TRELLIS ────────────────────────────────────────────────────────────────────
if ($SelectedModels["trellis"] -and $runtimeDevice -eq "cuda") {
    Step "Installing TRELLIS (best-quality model)"
    Info "TRELLIS uses open DINOv2 dependencies. TRELLIS.2 can fall back to open DINOv2 if DINOv3 access is unavailable."

    # Clone TRELLIS repo
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
} elseif (-not $SelectedModels["trellis"]) {
    Step "Skipping TRELLIS (not selected)"
} else {
    Step "Skipping TRELLIS (CUDA/NVIDIA GPU required)"
    Warn "TRELLIS requires an NVIDIA GPU. Re-run install with -Profile nvidia to enable it."
}

# ── TRELLIS.2 ──────────────────────────────────────────────────────────────────
if ($SelectedModels["trellis2"] -and $runtimeDevice -eq "cuda") {
    Step "Installing TRELLIS.2 (experimental, CUDA only)"
    Warn "TRELLIS.2 native extensions may fail to build; failures are non-fatal."

    # ── Clone official TRELLIS.2 repo ──
    $trellis2RepoUrl = "https://github.com/microsoft/TRELLIS.2.git"
    if (Test-Path -LiteralPath $trellis2Repo) { Remove-Item -Recurse -Force -LiteralPath $trellis2Repo }
    Info "Cloning TRELLIS.2 repo..."
    git clone --quiet --depth 1 --recurse-submodules "$trellis2RepoUrl" "$trellis2Repo" 2>$null
    if (-not (Test-Path -LiteralPath $trellis2Repo)) {
        Warn "TRELLIS.2 repo clone failed - will try to use existing backend/trellis2 if present"
    }
    if (Test-Path -LiteralPath $trellis2Repo) {
        git -C "$trellis2Repo" submodule update --init --recursive --depth 1 2>$null | Out-Null
        OK "TRELLIS.2 repo cloned"
    }

    # ── Copy trellis2 Python package to backend ──
    $t2PkgSrc = Join-Path $trellis2Repo "trellis2"
    if (-not (Test-Path -LiteralPath $t2PkgSrc)) {
        # Some repos may ship it as 'trellis' inside the trellis2 branch
        $t2PkgSrc = Join-Path $trellis2Repo "trellis"
    }
    if (Test-Path -LiteralPath $trellis2Dest) { Remove-Item -Recurse -Force -LiteralPath $trellis2Dest }
    if (Test-Path -LiteralPath $t2PkgSrc) {
        Copy-Item -Recurse -LiteralPath $t2PkgSrc -Destination $trellis2Dest
        OK "TRELLIS.2 package copied to backend"
        # Ensure PIXFORM runtime import compatibility alias exists.
        $t2PipelinesInit = Join-Path $trellis2Dest "pipelines\__init__.py"
        if (Patch-Trellis2PipelineAliases "$PY" "$t2PipelinesInit") {
            OK "TRELLIS.2 pipeline alias patched"
        }
        # Patch flexicubes kaolin fallback
        $t2FlexPath = Join-Path $trellis2Dest "representations\mesh\flexicubes\flexicubes.py"
        if (Patch-TrellisFlexicubes "$PY" "$t2FlexPath") {
            OK "TRELLIS.2 kaolin fallback applied"
        }
        # Patch image feature extractor for open-source DINOv2 fallback
        $t2IFEPath = Join-Path $trellis2Dest "modules\sparse\attention\image_feature_extractor.py"
        if (-not (Test-Path -LiteralPath $t2IFEPath)) {
            $t2IFEPath = Join-Path $trellis2Dest "modules\attention\image_feature_extractor.py"
        }
        if (Patch-Trellis2ImageExtractorForOSS "$PY" "$t2IFEPath") {
            OK "TRELLIS.2 image extractor patched (DINOv2 fallback)"
        }
        $t2PipelinePath = Join-Path $trellis2Dest "pipelines\trellis2_image_to_3d.py"
        if (-not (Test-Path -LiteralPath $t2PipelinePath)) {
            $t2PipelinePath = Join-Path $trellis2Dest "pipelines\trellis_image_to_3d.py"
        }
        if (Patch-Trellis2ImagePipelineCompat "$PY" "$t2PipelinePath") {
            OK "TRELLIS.2 image-to-3D pipeline patched (patch14 + mesh fallback guards)"
        }
    } else {
        Warn "trellis2/ package folder not found in cloned repo - TRELLIS.2 Python package unavailable"
    }

    # ── xformers / spconv (shared with TRELLIS, skip if already done) ──
    $xformersCheck = & "$PY" -c "import xformers" 2>$null; $xformersOk = ($LASTEXITCODE -eq 0)
    if (-not $xformersOk) {
        Info "Installing xformers for TRELLIS.2..."
        & "$PY" -m pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124 -q
        if ($LASTEXITCODE -eq 0) { OK "xformers installed" } else { Warn "xformers failed" }
        & "$PY" -m pip install spconv-cu120 -q
        if ($LASTEXITCODE -eq 0) { OK "spconv installed" } else { Warn "spconv failed" }
    }

    Info "Installing TRELLIS.2 Python helpers..."
    & "$PY" -m pip install easydict plyfile xatlas pyvista pymeshfix igraph -q
    if ($LASTEXITCODE -eq 0) { OK "TRELLIS.2 helper packages installed" } else { Warn "TRELLIS.2 helper package install had failures" }
    & "$PY" -m pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" -q
    if ($LASTEXITCODE -eq 0) { OK "utils3d installed" } else { Warn "utils3d install failed" }

    # ── flex_gemm: already present as pure Python in backend/flex_gemm ──
    $flexGemmDest = Join-Path $backendPath "flex_gemm"
    if (Test-Path -LiteralPath $flexGemmDest) {
        OK "flex_gemm already present in backend (pure Python)"
    } else {
        # Try to clone flex_gemm from known location
        $fgOk = Install-GitCudaPackage "flex_gemm" "https://github.com/microsoft/FlexGEMM.git" "$PY" "$env:TEMP\pixform_ext"
        if (-not $fgOk) {
            Warn "flex_gemm build failed - TRELLIS.2 may not load"
        }
    }

    # ── o_voxel: already present as pure Python in backend/o_voxel ──
    $oVoxelDest = Join-Path $backendPath "o_voxel"
    if (Test-Path -LiteralPath $oVoxelDest) {
        OK "o_voxel already present in backend (pure Python)"
    } else {
        # Try to build o_voxel C++ extension
        $oVoxelRepoPath = Join-Path $env:TEMP "pixform_o_voxel"
        $ovOk = Install-GitCudaPackage "o_voxel" "https://github.com/microsoft/o_voxel.git" "$PY" "$env:TEMP\pixform_ext"
        if (-not $ovOk) {
            Warn "o_voxel build failed - TRELLIS.2 may run slower (pure Python fallback)"
        }
    }

    # ── cumesh: use bundled stub (no public repo available) ──
    Info "Setting up cumesh (stub fallback - native build not available publicly)..."
    $cumeshDest = Join-Path $backendPath "cumesh"
    $cumeshCheck = & "$PY" -c "import cumesh; print(cumesh.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        OK "cumesh already importable ($($cumeshCheck.Trim()))"
    } else {
        if (Test-Path -LiteralPath $cumeshDest) {
            OK "cumesh stub present in backend/"
        } else {
            Warn "cumesh not found - TRELLIS.2 will run in degraded mode (mesh ops via fallback)"
            Warn "If you have the CuMesh source, place it in backend/cumesh/ and reinstall."
        }
    }
} elseif (-not $SelectedModels["trellis2"]) {
    Step "Skipping TRELLIS.2 (not selected)"
} else {
    Step "Skipping TRELLIS.2 (CUDA/NVIDIA GPU required)"
    Warn "TRELLIS.2 requires an NVIDIA GPU. Re-run install with -Profile nvidia to enable it."
}

# ── TripoSR ────────────────────────────────────────────────────────────────────
if ($SelectedModels["triposr"]) {
    Step "Cloning TripoSR (fast model)"
    if (Test-Path -LiteralPath $triposrRepo) { Remove-Item -Recurse -Force -LiteralPath $triposrRepo }
    git clone --quiet --depth 1 https://github.com/VAST-AI-Research/TripoSR.git "$triposrRepo"
    $triposrTsr = Join-Path $triposrRepo "tsr"
    if (-not (Test-Path -LiteralPath $triposrTsr)) { Fail "Failed to clone TripoSR" }

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
} else {
    Step "Skipping TripoSR (not selected)"
}

# ── Pin NumPy last ─────────────────────────────────────────────────────────────
Step "Pinning NumPy/OpenCV compatibility"
& "$PY" -m pip install "numpy==1.26.4" "opencv-python==4.10.0.84" --force-reinstall -q
OK "NumPy/OpenCV pinned (1.26.4 / 4.10.0.84)"

$modelsConfigPath = Join-Path $PSScriptRoot ".pixform_models.json"
$selectedModelConfig = [ordered]@{}
foreach ($name in $AllModelKeys) {
    $selectedModelConfig[$name] = [bool]$SelectedModels[$name]
}
$modelJson = $selectedModelConfig | ConvertTo-Json
[System.IO.File]::WriteAllText($modelsConfigPath, $modelJson, (New-Object System.Text.UTF8Encoding($false)))
OK "Saved model selection to .pixform_models.json"

# ── Validate ───────────────────────────────────────────────────────────────────
Step "Validating installation"
& "$PY" -c @"
import sys, pathlib, json
sys.path.insert(0, 'backend')

cfg_path = pathlib.Path('.pixform_models.json')
selected = {k: True for k in ('triposr', 'hunyuan', 'trellis', 'trellis2')}
if cfg_path.exists():
    try:
        data = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
        if isinstance(data, dict):
            for key in selected:
                if key in data:
                    selected[key] = bool(data[key])
    except Exception as e:
        print(f'  model selection config: FAILED - {e}')

import torch
print(f'  PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available(): print(f'  GPU: {torch.cuda.get_device_name(0)}')

import numpy as np
print(f'  NumPy {np.__version__}')

t = torch.tensor(np.zeros((4,4), dtype=np.float32))
print(f'  numpy->torch: OK {t.shape}')

if not selected.get('triposr', True):
    print('  TripoSR: skipped (not selected)')
else:
    try:
        from tsr.system import TSR
        print('  TripoSR: OK')
    except Exception as e:
        print(f'  TripoSR: FAILED - {e}')

if not selected.get('hunyuan', True):
    print('  Hunyuan3D-2: skipped (not selected)')
else:
    try:
        hy3d = pathlib.Path('backend/hy3dgen')
        if hy3d.exists():
            sys.path.insert(0, str(hy3d))
            print(f'  hy3dgen folder: found ({len(list(hy3d.iterdir()))} items)')
        else:
            print('  hy3dgen folder: NOT FOUND')
    except Exception as e:
        print(f'  hy3dgen: FAILED - {e}')

if not selected.get('trellis', True):
    print('  TRELLIS: skipped (not selected)')
else:
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

if not selected.get('trellis2', True):
    print('  TRELLIS.2: skipped (not selected)')
else:
    try:
        trellis2_pkg = pathlib.Path('backend/trellis2')
        if trellis2_pkg.exists():
            print(f'  TRELLIS.2 package: found ({len(list(trellis2_pkg.iterdir()))} items)')
            import importlib.util
            import os
            # flex_gemm is the only hard requirement; cumesh and o_voxel are soft-optional
            missing_hard = [d for d in ('flex_gemm',) if importlib.util.find_spec(d) is None]
            missing_soft = [d for d in ('cumesh', 'o_voxel') if importlib.util.find_spec(d) is None]
            if missing_hard:
                missing_str = ', '.join(missing_hard)
                print(f'  TRELLIS.2 required deps missing: {missing_str} (TRELLIS.2 disabled)')
            else:
                if missing_soft:
                    missing_soft_str = ', '.join(missing_soft)
                    print(f'  TRELLIS.2 optional deps missing (degraded mode): {missing_soft_str}')
                os.environ.setdefault('SPARSE_CONV_BACKEND', 'spconv')
                os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')
                os.environ.setdefault('ATTN_BACKEND', 'xformers')
                sys.path.insert(0, str(trellis2_pkg.parent))
                from trellis2.pipelines import Trellis2ImageTo3DPipeline
                print('  TRELLIS.2: OK (Trellis2ImageTo3DPipeline available)')
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

