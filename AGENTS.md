# AGENTS.md

## Big picture
- PIXFORM is a local FastAPI app (`backend/app.py`) serving both API and UI (`GET /` reads `frontend/index.html`).
- Core runtime is orchestrated in one file: model loading, job queue, preprocessing, inference, postprocess, export, and preview rendering all live in `backend/app.py`.
- There is no persistent DB/queue: `jobs` is an in-memory dict, and artifacts are file-based in `backend/uploads/` and `backend/outputs/<job_id>/`.

## Backend flow to understand first
- Startup (`lifespan`) calls `load_all_models()` once; device comes from `resolve_runtime_device()` using `PIXFORM_DEVICE` with aliases `nvidia->cuda`, `mac->mps`.
- `/convert` writes upload to disk, normalizes settings (`_normalize_triposr_resolution`, step clamp, post level), then dispatches a background task.
- Model runners are separate async functions: `run_triposr`, `run_hunyuan`, `run_trellis`; each updates progress via `upd(...)` and writes the same output contract.
- Export contract is stable: `model.stl`, `model.3mf`, `model.glb`, `model.obj`, `preview.png`; 3MF has a manual XML/ZIP fallback (`_export_3mf_manual`).
- Postprocess pipeline is opinionated for watertight print meshes (`postprocess_mesh`): repair -> smoothing -> Poisson -> hole fill -> voxel fallback -> scale to 100 mm.

## Frontend/API coupling
- Frontend is a single static file with inline JS (`frontend/index.html`) that calls `/health`, `/convert`, `/status/{job_id}`.
- Polling cadence is hardcoded: health every 20s, status every 1.8s.
- Model availability in UI is driven by `/health` flags and `*_status` fields.
- Important mismatch: UI includes `trellis2` card and sends `model=trellis2`, but backend `/convert` currently accepts only `triposr|hunyuan|trellis`.

## Install/build workflows (project-specific)
- Primary Windows flow: run `install.ps1` then `PIXFORM.bat`; launcher reads `.pixform_device` or CLI profile override.
- Primary macOS flow: run `install_mac.sh` then `PIXFORM.sh`.
- Installers clone upstream repos (`triposr_repo`, `hunyuan3d_repo`, `trellis_repo`, `trellis2_repo`) and copy runtime packages into `backend/` (`tsr`, `hy3dgen`, `trellis`, `trellis2`).
- Installers patch third-party code in place (TripoSR rembg/marching cubes, TRELLIS kaolin fallback). Re-running installer may overwrite manual edits in generated runtime folders.

## Conventions and guardrails for edits
- Treat `backend/app.py` as the integration hub; keep route response fields and status messages stable because UI renders them directly.
- Keep device logic aligned across `resolve_runtime_device()`, launchers (`PIXFORM.bat`, `PIXFORM.sh`), and installer-written `.pixform_device`.
- Generated/runtime directories are intentionally ignored (`.gitignore`: `backend/tsr`, `backend/hy3dgen`, `backend/uploads`, `backend/outputs`, cloned repos).
- Encoding is mixed in repo: some files are UTF-16 (notably `frontend/index.html`, `install_mac.sh`); preserve existing encoding unless intentionally migrating.

## External integrations
- Heavy runtime dependencies are imported dynamically in `backend/app.py`: TripoSR (`tsr`), Hunyuan3D-2 (`hy3dgen`), TRELLIS (`trellis`), rembg, open3d, trimesh.
- Optional CUDA-only enhancements are expected to fail gracefully (for example textured GLB via `nvdiffrast` in TRELLIS path).
- Packaging for Windows desktop distribution uses `pixform.iss` (Inno Setup) and assumes installer+launcher workflow.
