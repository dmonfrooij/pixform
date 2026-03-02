# PIXFORM — Image to 3D

Convert a single photo into a print-ready 3D model using state-of-the-art AI, running fully locally on your GPU.

**Two models:**
- **TripoSR** — Fast (~1–3 min), 6 GB VRAM
- **Hunyuan3D-2** — High quality (~3–6 min), 10 GB VRAM

**Export formats:** STL · 3MF · GLB · OBJ

---

## Requirements

- Windows 10/11 (64-bit)
- [Python 3.10](https://www.python.org/downloads/release/python-31011/) — during install: **do NOT check** "Add Python to PATH"
- [Git](https://git-scm.com/download/win)
- NVIDIA GPU with 6+ GB VRAM + up-to-date drivers

---

## Installation

```powershell
git clone https://github.com/YOUR_USERNAME/pixform.git
cd pixform
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1
```

The installer downloads ~5–8 GB on first run (PyTorch, TripoSR, Hunyuan3D-2).

When you first **start** the app, Huggingface model weights are downloaded automatically (~additional 8 GB). This only happens once.

---

## Usage

```powershell
PIXFORM.bat
```

Browser opens at `http://localhost:8000`

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

---

## Project structure

```
pixform/
├── backend/
│   └── app.py          # FastAPI server + TripoSR + Hunyuan3D-2 pipeline
├── frontend/
│   └── index.html      # Web UI
├── installer/
│   └── pixform.iss     # Inno Setup script for .exe installer
├── install.ps1         # Dependency installer
├── PIXFORM.bat         # App launcher
└── README.md
```

## Building the .exe installer

1. Install [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Open `installer/pixform.iss`
3. Click Build → creates `installer/dist/PIXFORM_Setup_v1.0.exe`

---

## Credits

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) — VAST AI Research & Stability AI
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) — Tencent
- [rembg](https://github.com/danielgatis/rembg) — background removal
- [Open3D](http://www.open3d.org/) — mesh processing

## License

Public
