# PIXFORM — Image to 3D

Convert a single photo into a local 3D model and export it as a printable or preview-friendly mesh.

**Included models:**
- **TripoSR** — fastest option, works on CUDA / MPS / CPU
- **Hunyuan3D-2** — higher-quality diffusion model, CUDA only
- **TRELLIS** — best quality, textured GLB support, CUDA only

**Export formats:** STL · 3MF · GLB · OBJ

---

## Requirements

- Windows 10/11 (64-bit) or macOS (Apple Silicon recommended)
- [Python 3.10](https://www.python.org/downloads/release/python-31011/)
- [Git](https://git-scm.com/download/win)
- For best performance: NVIDIA GPU with up-to-date drivers

### Hardware notes

- **TripoSR** can run on:
  - NVIDIA (`cuda`)
  - Apple Silicon (`mps`)
  - CPU
- **Hunyuan3D-2** currently loads only on **CUDA / NVIDIA**
- **TRELLIS** currently loads only on **CUDA / NVIDIA**

Rough VRAM guidance:

- TripoSR: **6+ GB** recommended
- Hunyuan3D-2: **10+ GB** recommended
- TRELLIS: **12+ GB** recommended

---

## Installation

### Windows

```powershell
git clone https://github.com/dmonfrooij/pixform.git
cd pixform
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1 -Profile auto
```

Profiles for `install.ps1`:

- `auto` (default): uses CUDA when an NVIDIA GPU is detected, otherwise CPU
- `nvidia`: forces CUDA packages and installs CUDA-only model support
- `cpu`: installs CPU runtime only

### macOS (Apple Silicon)

```bash
git clone https://github.com/dmonfrooij/pixform.git
cd pixform
chmod +x install_mac.sh PIXFORM.sh
./install_mac.sh mac
```

Profiles for `install_mac.sh`:

- `mac` (default): uses Apple Metal (`mps`)
- `auto`: selects `mps` on macOS
- `cpu`: forces CPU runtime
- `nvidia`: CUDA profile for external CUDA setups only

### What the installer does

- Creates a fresh `venv`
- Installs PyTorch and runtime dependencies
- Clones **TripoSR** support files
- For CUDA installs, also clones **Hunyuan3D-2** and **TRELLIS** support files
- Saves the preferred runtime device in `.pixform_device`

On first app launch, model weights are downloaded and cached automatically. The amount depends on which models are available for your selected device.

---

## Usage

### Windows

```powershell
.\PIXFORM.bat
```

Optional startup overrides:

```powershell
.\PIXFORM.bat nvidia
.\PIXFORM.bat cpu
```

### macOS

```bash
./PIXFORM.sh
```

Optional startup overrides:

```bash
./PIXFORM.sh mac
./PIXFORM.sh cpu
./PIXFORM.sh nvidia
```

The browser opens at `http://localhost:8000`.

Basic flow:

1. Drop an image
2. Choose a model
3. Pick a quality preset
4. Click **Generate 3D Model**
5. Download STL / 3MF / GLB / OBJ

---

## Models

| Model | Best for | Device support | Notes |
|------|------|------|------|
| TripoSR | Fast previews and lighter GPUs | CUDA / MPS / CPU | Default fast single-image reconstruction |
| Hunyuan3D-2 | Better geometry quality | CUDA only | Loaded only when runtime device resolves to CUDA |
| TRELLIS | Highest quality and textured GLB | CUDA only | Best-quality option; CUDA-only dependencies required |

---

## Quality presets

These presets are defined in the UI and apply different values depending on the selected model.

| Preset | TripoSR res | Hunyuan / TRELLIS steps | Post-processing | Approx time (TripoSR / Hunyuan / TRELLIS) |
|--------|-------------|-------------------------|-----------------|-------------------------------------------|
| ⚡ Draft | 128 | 10 | `none` | ~10 sec / ~1 min / ~3 min |
| 🔹 Low | 192 | 20 | `light` | ~30 sec / ~2 min / ~4 min |
| 🔷 Medium | 256 | 30 | `light` | ~1 min / ~3 min / ~5 min |
| ⭐ High | 512 | 50 | `standard` | ~15 min / ~5 min / ~8 min |
| 🔶 Ultra | 640 | 75 | `standard` | ~30 min / ~8 min / ~10 min |
| 💎 Extreme | 768 | 100 | `heavy` | ~45 min / ~12 min / ~12 min |
| 🔥 Maximum | 1024 | 100 | `heavy` | ~60+ min / ~15 min / ~15 min |

There is also a **Custom** preset in the UI for manual control.

---

## Device behavior

- The backend resolves `PIXFORM_DEVICE` with safe fallback logic:
  - `auto` → `cuda` → `mps` → `cpu`
- The launch scripts also reuse the device saved in `.pixform_device`
- If you request `cuda` but CUDA is unavailable, PIXFORM falls back automatically
- If you request `mps` but MPS is unavailable, PIXFORM falls back automatically

### Manual override examples

**Windows PowerShell**

```powershell
$env:PIXFORM_DEVICE = 'cuda'
.\PIXFORM.bat
```

```powershell
$env:PIXFORM_DEVICE = 'cpu'
.\PIXFORM.bat
```

**macOS / bash**

```bash
PIXFORM_DEVICE=mps ./PIXFORM.sh
```

```bash
PIXFORM_DEVICE=cpu ./PIXFORM.sh
```

---

## Tips for best results

| ✅ Do | ❌ Don't |
|-------|---------|
| Use a plain white or grey background | Use a cluttered background |
| Use even, diffuse lighting | Use harsh shadows or reflections |
| Keep the object fully in frame | Crop the object |
| Use a front or 3/4 view | Use extreme top-down angles |
| Use one clear subject | Use multiple objects |
| Start with at least 512×512 px | Use tiny or blurry images |

---

## Project structure

```text
pixform/
├── backend/
│   └── app.py          # FastAPI server + model loading + export pipeline
├── frontend/
│   └── index.html      # Web UI + quality presets
├── install.ps1         # Windows installer script
├── install_mac.sh      # macOS installer script
├── PIXFORM.bat         # Windows launcher
├── PIXFORM.sh          # macOS launcher
├── pixform.iss         # Inno Setup script
├── triposr_repo/       # Cloned during install
├── hunyuan3d_repo/     # Cloned for CUDA profiles
├── trellis_repo/       # Cloned for CUDA profiles
└── README.md
```

---

## Building the Windows installer

1. Install [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Open `pixform.iss`
3. Build the script
4. The installer is written to `dist/PIXFORM_Setup_v1.0.exe`

---

## Credits

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) — VAST AI Research & Stability AI
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) — Tencent
- [TRELLIS](https://github.com/microsoft/TRELLIS) — Microsoft
- [rembg](https://github.com/danielgatis/rembg) — background removal
- [Open3D](http://www.open3d.org/) — mesh processing

## License

Public
