#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PROFILE="${1:-mac}" # mac | nvidia | cpu | auto
case "$PROFILE" in
  mac|nvidia|cpu|auto) ;;
  *)
    echo "[ERROR] Unknown profile: $PROFILE"
    echo "Usage: ./install_mac.sh [mac|nvidia|cpu|auto]"
    exit 1
    ;;
esac

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git is required"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 is required"
  exit 1
fi

python3 - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit("[ERROR] Python 3.10 is required")
print("[OK] Python 3.10 detected")
PY

RUNTIME_DEVICE="cpu"
if [[ "$PROFILE" == "mac" ]]; then
  RUNTIME_DEVICE="mps"
elif [[ "$PROFILE" == "nvidia" ]]; then
  RUNTIME_DEVICE="cuda"
elif [[ "$PROFILE" == "auto" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    RUNTIME_DEVICE="mps"
  fi
fi

echo "== Creating virtual environment =="
rm -rf venv
python3 -m venv venv
PY="venv/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel

echo "== Installing PyTorch =="
if [[ "$RUNTIME_DEVICE" == "cuda" ]]; then
  "$PY" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
else
  "$PY" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
fi

echo "== Installing core dependencies =="
if [[ "$RUNTIME_DEVICE" == "cuda" ]]; then
  REMBG_PKG="rembg[gpu]"
  ONNX_PKG="onnxruntime-gpu"
else
  REMBG_PKG="rembg"
  ONNX_PKG="onnxruntime"
fi

"$PY" -m pip install \
  "numpy==1.26.4" "opencv-python==4.10.0.84" "Pillow>=10.0" "trimesh[easy]" "scipy" "imageio" \
  "einops" "omegaconf>=2.3" "huggingface_hub" "transformers>=4.40" \
  "accelerate>=0.30" "diffusers>=0.27" "fastapi==0.115.5" "uvicorn[standard]==0.32.1" \
  "python-multipart==0.0.12" "pydantic>=2.0" "httpx" "$REMBG_PKG" "$ONNX_PKG" \
  PyMCubes pyrender

# Open3D is optional on macOS arm; keep install best-effort.
"$PY" -m pip install open3d || echo "[WARN] open3d install failed (continuing)"

echo "== Cloning TripoSR =="
rm -rf triposr_repo
rm -rf backend/tsr
git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git triposr_repo
cp -R triposr_repo/tsr backend/tsr

echo "== Patching TripoSR =="
"$PY" - <<'PY'
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

p2 = 'backend/tsr/models/isosurface.py'
txt2 = open(p2, encoding='utf-8').read()
old = 'from torchmcubes import marching_cubes'
new = 'try:\n    from torchmcubes import marching_cubes\nexcept ImportError:\n    import mcubes as _mc\n    import torch as _t\n    import numpy as _np\n    def marching_cubes(vol, threshold):\n        v, f = _mc.marching_cubes(vol.cpu().numpy(), float(threshold))\n        return _t.tensor(v.astype(_np.float32)), _t.tensor(f.astype(_np.int64))\n'
if old in txt2:
    txt2 = txt2.replace(old, new)
    ast.parse(txt2)
    open(p2, 'w', encoding='utf-8').write(txt2)

print('[OK] TripoSR patch applied')
PY

if [[ "$RUNTIME_DEVICE" == "cuda" ]]; then
  echo "== Cloning Hunyuan3D-2 (CUDA profile) =="
  rm -rf hunyuan3d_repo
  rm -rf backend/hy3dgen
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git hunyuan3d_repo
  if [[ -f hunyuan3d_repo/requirements.txt ]]; then
    "$PY" -m pip install -r hunyuan3d_repo/requirements.txt || echo "[WARN] Some Hunyuan dependencies failed"
  fi
  cp -R hunyuan3d_repo/hy3dgen backend/hy3dgen
  if [[ -f backend/hy3dgen/rembg.py ]]; then
    mv backend/hy3dgen/rembg.py backend/hy3dgen/rembg_hy3d.py
  fi

  echo "== Installing TRELLIS (best-quality model, CUDA profile) =="
  rm -rf trellis_repo
  git clone --depth 1 --recurse-submodules https://github.com/microsoft/TRELLIS.git trellis_repo
  if [[ -d trellis_repo ]]; then
    git -C trellis_repo submodule update --init --recursive --depth 1 >/dev/null 2>&1 || true
  fi

  # TRELLIS core runtime dependencies
  # xformers 0.0.28.post3 matches torch 2.5.1 on the cu124 index.
  # spconv packages are named by CUDA major version: spconv-cu120 supports all CUDA 12.x including 12.4.
  "$PY" -m pip install \
    xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124 || \
    echo "[WARN] xformers install failed"
  "$PY" -m pip install spconv-cu120 || echo "[WARN] spconv install failed"
  "$PY" -m pip install \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" \
    easydict xatlas pyvista pymeshfix igraph || echo "[WARN] Some TRELLIS deps failed"

  # nvdiffrast — enables textured GLB export (optional)
  git clone https://github.com/NVlabs/nvdiffrast.git /tmp/pixform_nvdiffrast 2>/dev/null || true
  if [[ -d /tmp/pixform_nvdiffrast ]]; then
    "$PY" -m pip install --no-build-isolation /tmp/pixform_nvdiffrast || echo "[WARN] nvdiffrast failed — textured GLB will use fallback"
  fi

  # mip-splatting — enables Gaussian rendering for textured GLB (optional)
  git clone https://github.com/autonomousvision/mip-splatting.git /tmp/pixform_mipsplat 2>/dev/null || true
  if [[ -d /tmp/pixform_mipsplat/submodules/diff-gaussian-rasterization ]]; then
    "$PY" -m pip install /tmp/pixform_mipsplat/submodules/diff-gaussian-rasterization/ || \
      echo "[WARN] mip-splatting failed — textured GLB will use fallback"
  fi

  # Copy trellis Python package to backend
  rm -rf backend/trellis
  if [[ -d trellis_repo/trellis ]]; then
    cp -R trellis_repo/trellis backend/trellis
    echo "[OK] TRELLIS copied to backend"
    "$PY" - <<'PY'
from pathlib import Path

p = Path('backend/trellis/representations/mesh/flexicubes/flexicubes.py')
if p.exists():
    txt = p.read_text(encoding='utf-8')
    old = 'from kaolin.utils.testing import check_tensor\n'
    new = '''try:
    from kaolin.utils.testing import check_tensor
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
    if old in txt and 'except ImportError:' not in txt:
        p.write_text(txt.replace(old, new), encoding='utf-8')
        print('[OK] flexicubes.py patched (kaolin fallback)')
PY
  else
    echo "[WARN] TRELLIS trellis/ folder not found in repo"
  fi
else
  echo "== Skipping Hunyuan3D-2 clone for non-CUDA profile =="
  echo "== Skipping TRELLIS clone for non-CUDA profile =="
fi

echo "== Final pinning =="
"$PY" -m pip install --force-reinstall numpy==1.26.4 opencv-python==4.10.0.84

printf '%s\n' "$RUNTIME_DEVICE" > .pixform_device

echo "== Validation =="
"$PY" - <<'PY'
import sys
sys.path.insert(0, 'backend')
import torch
import numpy as np
import cv2
print('PyTorch', torch.__version__)
print('NumPy', np.__version__)
print('OpenCV', cv2.__version__)
print('CUDA available:', torch.cuda.is_available())
print('MPS available:', bool(getattr(torch.backends, 'mps', None)) and torch.backends.mps.is_available())
import pathlib
trellis_path = pathlib.Path('backend/trellis')
if trellis_path.exists():
    print('TRELLIS package: found')
    from trellis.pipelines import TrellisImageTo3DPipeline
    print('TRELLIS import: OK')
else:
    print('TRELLIS package: not found (CUDA-only feature)')
PY

echo ""
echo "PIXFORM installed."
echo "Start with: ./PIXFORM.sh"


