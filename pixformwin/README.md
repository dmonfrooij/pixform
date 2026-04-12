# PIXFORMWIN — Image to 3D (Native Desktop App)

Convert a single photo into a print-ready 3D model using state-of-the-art AI,
running fully locally on your GPU — as a **native desktop window** (no browser needed).

Built on [PIXFORM](https://github.com/dmonfrooij/pixform) · wrapped with
[pywebview](https://pywebview.flowrl.com/) (Edge WebView2 on Windows).

**Two models:**
- **TripoSR** — Fast (~1–3 min), 6 GB VRAM
- **Hunyuan3D-2** — High quality (~3–6 min), 10 GB VRAM

**Export formats:** STL · 3MF · GLB · OBJ

---

## Requirements

- Windows 10/11 (64-bit) — primary target
- macOS (Apple Silicon M1/M2/M3) supported via MPS profile
- [Python 3.10](https://www.python.org/downloads/release/python-31011/) — during install: **do NOT check** "Add Python to PATH"
- [Git](https://git-scm.com/download/win)
- NVIDIA GPU with 6+ GB VRAM + up-to-date drivers (recommended)

---

## Installation

### Windows (NVIDIA or CPU)

```powershell
git clone https://github.com/dmonfrooij/pixformwin.git
cd pixformwin
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1 -Profile auto
```

Profiles for `install.ps1`:
- `auto` (default): uses NVIDIA CUDA when detected, otherwise CPU
- `nvidia`: force CUDA packages
- `cpu`: force CPU packages

### macOS (Apple Silicon)

```bash
git clone https://github.com/dmonfrooij/pixformwin.git
cd pixformwin
chmod +x install_mac.sh PIXFORMWIN.sh
./install_mac.sh mac
```

The installer downloads ~5–8 GB on first run (PyTorch, TripoSR, Hunyuan3D-2).

When you first **start** the app, HuggingFace model weights are downloaded automatically
(~additional 8 GB). This only happens once.

---

## Usage

### Windows

```powershell
.\PIXFORMWIN.bat
```

Optional profile override:

```powershell
.\PIXFORMWIN.bat nvidia
.\PIXFORMWIN.bat cpu
```

### macOS

```bash
./PIXFORMWIN.sh
```

The app opens as a **native desktop window** (no browser, no address bar).

1. Drop an image
2. Choose model (TripoSR = fast, Hunyuan3D-2 = quality)
3. Pick a quality preset
4. Click **Generate 3D Model**
5. Download STL / 3MF / GLB / OBJ

---

## Quality presets

| Preset | TripoSR res | Hunyuan steps | Time (approx) |
|--------|-------------|---------------|---------------|
| ⚡ Draft | 128 | 10 | ~10 sec |
| 🔹 Low | 192 | 20 | ~30 sec |
| 🔷 Medium | 256 | 30 | ~1 min |
| ⭐ High ★ | 512 | 50 | ~15 min |
| 🔶 Ultra | 640 | 75 | ~30 min |
| 💎 Extreme | 768 | 100 | ~45 min |
| 🔥 Maximum | 1024 | 100 | ~60+ min |

Times based on RTX 3080 Ti. If VRAM runs out, resolution automatically steps down.

---

## Tips for best results

| ✅ Do | ❌ Don't |
|-------|---------|
| Plain white/grey background | Cluttered background |
| Even, diffuse lighting | Harsh shadows or reflections |
| Object fully in frame | Cropped or partial objects |
| Front or 3/4 view | Top-down or extreme angles |
| Single object | Multiple objects |
| Min 512×512 px | Tiny or blurry images |

### Device notes

- `Hunyuan3D-2` currently loads only on CUDA/NVIDIA.
- On MacBook Pro (`mps`), `TripoSR` works; Hunyuan is disabled automatically.
- Override device manually with env var `PIXFORM_DEVICE=auto|cuda|mps|cpu` before launch.

---

## Project structure

```
pixformwin/
├── app.py              # Main entry: launches backend + opens pywebview window
├── backend/
│   └── app.py          # FastAPI server + TripoSR + Hunyuan3D-2 pipeline
├── frontend/
│   └── index.html      # Web UI (served by FastAPI, shown in native window)
├── install.ps1         # Windows dependency installer
├── install_mac.sh      # macOS dependency installer
├── PIXFORMWIN.bat      # Windows launcher
├── PIXFORMWIN.sh       # macOS/Linux launcher
├── pixformwin.iss      # Inno Setup script for .exe installer
└── README.md
```

## Building the .exe installer

1. Install [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Open `pixformwin.iss`
3. Click Build → creates `dist/PIXFORMWIN_Setup_v1.0.exe`

---

## Differences from PIXFORM

| Feature | PIXFORM | PIXFORMWIN |
|---------|---------|------------|
| Interface | Browser tab | Native desktop window |
| Launcher | `PIXFORM.bat` | `PIXFORMWIN.bat` |
| Requires browser | Yes | No |
| Backend | FastAPI/uvicorn | Same |
| Models | TripoSR + Hunyuan3D-2 | Same |
| Entry point | `backend/app.py` | `app.py` (pywebview wrapper) |

---

## Credits

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) — VAST AI Research & Stability AI
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) — Tencent
- [rembg](https://github.com/danielgatis/rembg) — background removal
- [Open3D](http://www.open3d.org/) — mesh processing
- [pywebview](https://pywebview.flowrl.com/) — native window wrapper

## License

Public
