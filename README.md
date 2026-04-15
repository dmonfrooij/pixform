# PIXFORM — Afbeelding naar 3D

PIXFORM zet één invoerafbeelding om naar een lokaal 3D-model met focus op zo hoog mogelijke outputkwaliteit, bruikbare exports en eenvoudige lokale installatie.

In deze repository zitten drie generatiemodi:

- **TripoSR** — snelste optie, werkt op `cuda`, `mps` en `cpu`
- **Hunyuan3D-2** — hogere shape-kwaliteit, momenteel alleen bruikbaar op `cuda`
- **TRELLIS** — hoogste kwaliteit, met GLB-export en waar mogelijk getextureerde GLB, alleen op `cuda`

**Exportformaten:** `STL`, `3MF`, `GLB`, `OBJ`

---

## Wat PIXFORM doet

- lokale webapp op `http://localhost:8000`
- upload van één afbeelding
- optionele AI-achtergrondverwijdering
- keuze uit snel, kwaliteit of maximale kwaliteit
- preview-render van het resultaat
- download van meerdere 3D-formaten

De backend draait vanuit `backend/app.py` en de webinterface staat in `frontend/index.html`.

---

## Ondersteunde platformen

### Windows + NVIDIA
Aanbevolen voor de beste prestaties en toegang tot alle modellen.

Ondersteund via:
- `install.ps1`
- `PIXFORM.bat`

### macOS / MacBook Pro (Apple Silicon)
Geschikt voor snelle lokale workflow met `mps` / Metal.

Ondersteund via:
- `install_mac.sh`
- `PIXFORM.sh`

### CPU fallback
Werkt ook zonder ondersteunde GPU, maar duidelijk trager. In dat geval is vooral **TripoSR** praktisch bruikbaar.

---

## Vereisten

### Algemeen
- Git
- Python **3.10**

### Windows
- Windows 10 of 11
- PowerShell
- Voor beste prestaties: NVIDIA GPU met recente driver

Voor CUDA-profielen (`-Profile nvidia`) gelden extra vereisten:

- NVIDIA driver + `nvidia-smi` beschikbaar
- Visual Studio Build Tools (C++ toolchain, `cl.exe`) voor native builds
- CUDA Toolkit met `nvcc` beschikbaar op `PATH`
- Aanbevolen: toolkitversie gelijk aan de CUDA-versie van geïnstalleerde PyTorch (standaard in dit script: `12.4`)

### macOS
- `python3` beschikbaar
- macOS shell met bash
- Apple Silicon wordt impliciet ondersteund via `mps`

> Op basis van de scripts in deze repository verwacht zowel de Windows- als macOS-installatie expliciet **Python 3.10**.

---

## Modellen

| Model | Beste voor | Device support | Opmerking |
|---|---|---|---|
| TripoSR | Snel testen, previews, bredere compatibiliteit | CUDA / MPS / CPU | Laadt als primaire algemene fallback |
| Hunyuan3D-2 | Betere geometrie dan TripoSR | CUDA | Wordt alleen geladen als runtime device `cuda` is |
| TRELLIS | Hoogste kwaliteit | CUDA | Probeert getextureerde GLB te maken; valt anders terug op gewone GLB |

De backend kiest het runtime-device via `PIXFORM_DEVICE` met veilige fallback-logica.

---

## Device-keuze

PIXFORM ondersteunt deze device-modi:

- `cuda`
- `mps`
- `cpu`
- `auto`

De backend-resolutievolgorde in `backend/app.py` is:

1. `cuda`
2. `mps`
3. `cpu`

Aliasnamen die ook werken:
- `nvidia` → `cuda`
- `mac` → `mps`

Als een gevraagd device niet beschikbaar is, valt PIXFORM automatisch terug naar een bruikbaar alternatief.

---

## Installatie

Gebruik bij voorkeur een schone virtual environment; de installatiescripts maken zelf een nieuwe `venv` aan.

### Windows

Voer in PowerShell uit:

```powershell
cd C:\Users\Eiboer\PycharmProjects\pixform
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1 -Profile auto
```

Voor `-Profile nvidia` kun je vooraf expliciet de toolkit in de huidige shell zetten (voorbeeld voor CUDA 12.4):

```powershell
$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
$env:CUDA_PATH = $env:CUDA_HOME
$env:Path = "$env:CUDA_HOME\bin;$env:CUDA_HOME\libnvvp;$env:Path"
nvcc --version
```

Controleer daarna desnoods ook welke CUDA-versie PyTorch in de venv gebruikt:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Als `torch.version.cuda` en `nvcc --version` niet matchen, kunnen optionele native builds (zoals `nvdiffrast` en `o-voxel` voor TRELLIS.2) worden overgeslagen.

Beschikbare profielen voor `install.ps1`:

- `auto` — gebruikt `cuda` als NVIDIA beschikbaar is, anders `cpu`
- `nvidia` — forceert CUDA-installatie en installeert CUDA-only modelondersteuning
- `cpu` — installeert alleen CPU-runtime

Wat `install.ps1` doet:

- maakt `venv` opnieuw aan
- installeert PyTorch 2.5.1
- installeert kernpakketten zoals FastAPI, rembg, OpenCV en trimesh
- clone't **Hunyuan3D-2**
- clone't **TripoSR**
- clone't **TRELLIS** alleen bij CUDA-profiel
- schrijft het gekozen runtime-device naar `.pixform_device`

### macOS / MacBook Pro

Voer uit:

```bash
cd /pad/naar/pixform
chmod +x install_mac.sh PIXFORM.sh
./install_mac.sh mac
```

Beschikbare profielen voor `install_mac.sh`:

- `mac` — gebruikt `mps`
- `auto` — kiest op macOS ook `mps`
- `cpu` — forceert CPU
- `nvidia` — CUDA-profiel voor systemen waar dat expliciet gewenst is

Wat `install_mac.sh` doet:

- maakt `venv` opnieuw aan
- installeert PyTorch 2.5.1
- installeert de kernafhankelijkheden
- installeert `open3d` best-effort
- clone't en patcht **TripoSR**
- clone't **Hunyuan3D-2** en **TRELLIS** alleen in CUDA-profiel
- schrijft het runtime-device naar `.pixform_device`

### Eerste start

Bij de eerste echte run kunnen modelgewichten nog extra worden gedownload en lokaal gecachet. Dat kost tijd en schijfruimte, vooral voor Hunyuan3D-2 en TRELLIS.

---

## Starten

### Windows

```bat
PIXFORM.bat
```

Optionele overrides:

```bat
PIXFORM.bat nvidia
PIXFORM.bat cuda
PIXFORM.bat cpu
PIXFORM.bat mac
PIXFORM.bat mps
```

### macOS

```bash
./PIXFORM.sh
```

Optionele overrides:

```bash
./PIXFORM.sh mac
./PIXFORM.sh mps
./PIXFORM.sh cpu
./PIXFORM.sh nvidia
```

De launcher start `backend/app.py` en opent daarna de browser op:

```text
http://localhost:8000
```

---

## Handmatige device-override

Je kunt ook direct de environment variable zetten.

### Windows PowerShell

```powershell
$env:PIXFORM_DEVICE = 'cuda'
.\PIXFORM.bat
```

```powershell
$env:PIXFORM_DEVICE = 'cpu'
.\PIXFORM.bat
```

### macOS / bash

```bash
PIXFORM_DEVICE=mps ./PIXFORM.sh
```

```bash
PIXFORM_DEVICE=cpu ./PIXFORM.sh
```

De launchers lezen anders automatisch `.pixform_device`; als dat bestand niet bestaat, gebruiken ze `auto`.

---

## Gebruik in de app

Basisworkflow:

1. sleep of selecteer een afbeelding
2. kies een model
3. kies een quality preset
4. kies of de achtergrond verwijderd moet worden
5. klik op **Generate**
6. download `STL`, `3MF`, `GLB` of `OBJ`

De backend heeft hiervoor onder andere deze routes:

- `GET /health`
- `POST /convert`
- `GET /status/{job_id}`
- `DELETE /jobs/{job_id}`

---

## Kwaliteitspresets

De presets komen rechtstreeks uit de UI in `frontend/index.html`.

| Preset | TripoSR resolutie | Hunyuan/TRELLIS steps | Post-processing | Richttijd (TripoSR / Hunyuan / TRELLIS) |
|---|---:|---:|---|---|
| ⚡ Draft | 128 | 10 | `none` | ~10 sec / ~1 min / ~3 min |
| 🔹 Low | 192 | 20 | `light` | ~30 sec / ~2 min / ~4 min |
| 🔷 Medium | 256 | 30 | `light` | ~1 min / ~3 min / ~5 min |
| ⭐ High | 512 | 50 | `standard` | ~15 min / ~5 min / ~8 min |
| 🔶 Ultra | 640 | 75 | `standard` | ~30 min / ~8 min / ~10 min |
| 💎 Extreme | 768 | 100 | `heavy` | ~45 min / ~12 min / ~12 min |
| 🔥 Maximum | 1024 | 100 | `heavy` | ~60+ min / ~15 min / ~15 min |
| ✏️ Custom | handmatig | handmatig | handmatig | afhankelijk van instellingen |

Beschikbare post-processing niveaus:

- `none`
- `light`
- `standard`
- `heavy`

---

## Beste 3D-kwaliteit: aanbevolen keuzes

Als je de best mogelijke output wilt:

- gebruik een scherpe, goed belichte afbeelding
- houd het onderwerp volledig in beeld
- gebruik één duidelijk object
- vermijd drukke achtergrond en motion blur
- gebruik bij voorkeur `Hunyuan3D-2` of `TRELLIS` op NVIDIA/CUDA
- kies `High`, `Ultra`, `Extreme` of `Maximum`
- laat post-processing op `standard` of `heavy` staan

Praktische keuze per situatie:

- **Snel testen** → TripoSR + Draft/Low/Medium
- **Goede mesh-kwaliteit** → Hunyuan3D-2 + High/Ultra
- **Best mogelijke kwaliteit** → TRELLIS + High/Ultra/Extreme

---

## Outputformaten

PIXFORM exporteert vanuit de backend deze formaten:

- `model.stl`
- `model.3mf`
- `model.glb`
- `model.obj`
- `preview.png`

Belangrijk:

- `STL` is de meest printvriendelijke export
- `3MF` wordt ook aangeboden en heeft een handmatige fallback-export in de backend
- `OBJ` is breed compatibel
- `GLB` is beschikbaar voor compacte distributie en preview
- bij **TRELLIS** probeert PIXFORM een **getextureerde GLB** te maken als optionele afhankelijkheden beschikbaar zijn; lukt dat niet, dan wordt een gewone mesh-GLB weggeschreven

Outputs worden opgeslagen in:

```text
backend/outputs/<job-id>/
```

Uploads komen terecht in:

```text
backend/uploads/
```

---

## Modelbeschikbaarheid en `/health`

De endpoint `GET /health` geeft onder andere terug:

- of `triposr` geladen is
- of `hunyuan` geladen is
- of `trellis` geladen is
- of `rembg` actief is
- of `cuda` of `mps` beschikbaar is
- welk runtime-device werkelijk gebruikt wordt

Dit is handig als een CUDA-only model niet selecteerbaar is of bij start niet correct laadt.

---

## Probleemoplossing

### 1. CUDA-model laadt niet
Controleer:
- NVIDIA-driver
- of `torch.cuda.is_available()` waar is
- of je met `-Profile nvidia` hebt geïnstalleerd
- of `backend/trellis` en `backend/hy3dgen` aanwezig zijn

### 2. Op Mac gebruikt de app geen MPS
Controleer:
- of `PIXFORM_DEVICE` niet op `cpu` staat
- of `.pixform_device` niet per ongeluk `cpu` bevat
- of PyTorch `mps` ziet

### 3. TRELLIS geeft geen getextureerde GLB
Dat is niet altijd een harde fout. In de backend zit expliciet fallback-logica:
- eerst poging tot textured GLB
- daarna fallback naar gewone GLB-export

### 4. TRELLIS laadt niet op Windows
Controleer:
- draai installatie met `-Profile nvidia`
- controleer `/health` op `trellis_status` en `trellis_error`
- gebruik `xformers` als attention-backend (de backend zet dit standaard op Windows)

Extra context:
- een melding over ontbrekende `triton` is meestal een performance-waarschuwing en niet per se een load-fout
- de `kaolin` import in `flexicubes.py` heeft een fallback in deze codebase

### 5. Installatie duurt lang
Dat is normaal voor:
- PyTorch
- Hunyuan3D-2 dependencies
- TRELLIS dependencies
- eerste modeldownload bij runtime

---

## Projectstructuur

```text
pixform/
├── backend/
│   ├── app.py            # FastAPI backend en modelpipeline
│   ├── outputs/          # gegenereerde resultaten
│   ├── uploads/          # geüploade bronafbeeldingen
│   ├── tsr/              # TripoSR runtimebestanden
│   ├── hy3dgen/          # Hunyuan3D runtimebestanden
│   └── trellis/          # TRELLIS runtimebestanden (na CUDA-installatie)
├── frontend/
│   └── index.html        # webinterface
├── install.ps1           # Windows installer
├── install_mac.sh        # macOS installer
├── PIXFORM.bat           # Windows launcher
├── PIXFORM.sh            # macOS launcher
├── pixform.iss           # Inno Setup script
├── triposr_repo/         # gekloonde bron tijdens/na installatie
├── hunyuan3d_repo/       # gekloonde bron tijdens/na installatie
├── trellis_repo/         # gekloonde bron tijdens/na CUDA-installatie
└── README.md
```

---

## Windows installer bouwen

Op basis van de repository staat er een Inno Setup-script in `pixform.iss`.

Globaal proces:

1. installeer Inno Setup
2. open `pixform.iss`
3. build het script

---

## Credits

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [TRELLIS](https://github.com/microsoft/TRELLIS)
- [rembg](https://github.com/danielgatis/rembg)
- [Open3D](http://www.open3d.org/)

## Licentie

Zie `LICENSE`.

