"""
PIXFORM Backend
Image to 3D pipeline - TripoSR (fast) + Hunyuan3D-2 (quality) + TRELLIS (best)
"""
import os, sys, uuid, shutil, asyncio, logging, zipfile, time, json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pixform")

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads";  UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = BASE_DIR / "outputs";  OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_SELECTION_FILE = BASE_DIR.parent / ".pixform_models.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# hy3dgen is path-based, not a package - add it explicitly
_hy3d_path = BASE_DIR / "hy3dgen"
if _hy3d_path.exists() and str(_hy3d_path) not in sys.path:
    sys.path.insert(0, str(_hy3d_path))

# Global model state

models = {
    "triposr":   None,
    "hunyuan":   None,
    "trellis":   None,
    "trellis2":  None,
    "rembg_sess": None,
    "installed_models": {},
    "runtime_device": "cpu",
}
model_health = {
    "triposr":  {"status": "pending", "error": None},
    "hunyuan":  {"status": "pending", "error": None},
    "trellis":  {"status": "pending", "error": None},
    "trellis2": {"status": "pending", "error": None},
    "rembg":    {"status": "pending", "error": None},
}
jobs: dict = {}

VALID_POST_LEVELS = {"none", "light", "standard", "heavy"}
TRIPOSR_RES_LEVELS = [1024, 896, 768, 640, 512, 448, 384, 320, 256, 192, 128]


def resolve_runtime_device() -> str:
    """Resolve runtime device from env with safe fallback order."""
    import torch

    pref = os.getenv("PIXFORM_DEVICE", "auto").strip().lower()
    alias = {
        "nvidia": "cuda",
        "mac": "mps",
    }
    pref = alias.get(pref, pref)

    cuda_ok = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps_backend) and mps_backend.is_available()

    if pref in {"", "auto"}:
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    if pref == "cuda":
        if cuda_ok:
            return "cuda"
        logger.warning("PIXFORM_DEVICE=cuda requested but CUDA is unavailable, falling back")
        return "mps" if mps_ok else "cpu"

    if pref == "mps":
        if mps_ok:
            return "mps"
        logger.warning("PIXFORM_DEVICE=mps requested but MPS is unavailable, falling back")
        return "cuda" if cuda_ok else "cpu"

    if pref == "cpu":
        return "cpu"

    logger.warning(f"Unknown PIXFORM_DEVICE '{pref}', using auto")
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"


def set_model_health(name: str, status: str, error: Optional[str] = None):
    if name in model_health:
        model_health[name]["status"] = status
        model_health[name]["error"] = error


def _format_model_load_error(model_name: str, exc: Exception) -> str:
    """Return concise, actionable load errors for UI health cards."""
    raw = str(exc)
    low = raw.lower()
    hf_access_markers = (
        "cannot access gated repo",
        "gated repo",
        "access to model",
        "please log in",
        "401",
        "unauthorized",
    )
    if any(marker in low for marker in hf_access_markers):
        return (
            f"{model_name} could not access a gated model dependency. "
            "Open-source fallback should be used automatically; if loading still fails, "
            "check /health details for the remaining missing dependency."
        )
    return raw


def _resolve_installed_models() -> dict:
    detected = {
        "triposr": (BASE_DIR / "tsr").exists(),
        "hunyuan": (BASE_DIR / "hy3dgen").exists(),
        "trellis": (BASE_DIR / "trellis").exists(),
        "trellis2": (BASE_DIR / "trellis2").exists(),
    }
    if not MODEL_SELECTION_FILE.exists():
        return detected
    try:
        data = json.loads(MODEL_SELECTION_FILE.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            for name in detected:
                if name in data:
                    detected[name] = bool(data[name]) and detected[name]
    except Exception as e:
        logger.warning(f"Could not read model selection file {MODEL_SELECTION_FILE}: {e}")
    return detected


def _foreground_profile(img) -> dict:
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] < 4:
        h, w = arr.shape[:2]
        return {
            "aspect_ratio": float(w / max(h, 1)),
            "subject_width": int(w),
            "subject_height": int(h),
            "mode": "image_bounds",
        }

    alpha = arr[:, :, 3]
    mask = alpha > 24
    if mask.any():
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        mode = "alpha_bounds"
    else:
        h, w = alpha.shape
        x0, x1, y0, y1 = 0, w, 0, h
        mode = "image_bounds"

    subject_w = max(1, x1 - x0)
    subject_h = max(1, y1 - y0)
    return {
        "aspect_ratio": float(subject_w / subject_h),
        "subject_width": subject_w,
        "subject_height": subject_h,
        "mode": mode,
    }


def _apply_input_aspect_correction(mesh, target_profile: Optional[dict]):
    if not target_profile:
        return mesh

    try:
        target_ratio = float(target_profile.get("aspect_ratio") or 0.0)
    except Exception:
        return mesh

    if not np.isfinite(target_ratio) or target_ratio <= 0:
        return mesh

    extents = np.asarray(mesh.extents, dtype=np.float64)
    if extents.shape[0] < 3 or extents[1] <= 1e-6:
        return mesh

    candidates = []
    for axis, vertical_axis in ((0, 1), (2, 1), (0, 2)):
        if extents[axis] <= 1e-6 or extents[vertical_axis] <= 1e-6:
            continue
        current_ratio = float(extents[axis] / extents[vertical_axis])
        candidates.append((abs(current_ratio - target_ratio), axis, vertical_axis, current_ratio))

    if not candidates:
        return mesh

    _, axis, vertical_axis, current_ratio = min(candidates, key=lambda item: item[0])
    if current_ratio <= 0:
        return mesh

    scale_factor = float(np.clip(target_ratio / current_ratio, 0.2, 5.0))
    if not np.isfinite(scale_factor) or abs(scale_factor - 1.0) < 0.03:
        return mesh

    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    scale = np.ones(3, dtype=np.float64)
    scale[axis] = scale_factor

    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    mesh.apply_translation(center)

    axis_name = "X" if axis == 0 else "Z"
    vertical_name = "X" if vertical_axis == 0 else ("Y" if vertical_axis == 1 else "Z")
    logger.info(
        "Applied input aspect correction on %s/%s axes: target %.3f, current %.3f, scale %.3f (%s)",
        axis_name,
        vertical_name,
        target_ratio,
        current_ratio,
        scale_factor,
        target_profile.get("mode", "unknown"),
    )
    return mesh


def _model_unavailable_message(model_key: str, label: str) -> str:
    status = model_health[model_key]["status"]
    detail = model_health[model_key]["error"]
    if status == "not_installed":
        return f"{label} is not installed. Re-run install.ps1 and include {label}."
    if status == "skipped" and detail:
        return f"{label} is unavailable on this runtime: {detail}."
    if detail:
        return f"{label} is not loaded: {detail}"
    return f"{label} is not loaded. Check /health and setup."


# Model loading

def load_all_models():
    import torch

    runtime_device = resolve_runtime_device()
    models["runtime_device"] = runtime_device
    installed_models = _resolve_installed_models()
    models["installed_models"] = installed_models
    logger.info(f"Runtime device: {runtime_device}")

    # rembg session (RMBG-1.4 - best background removal quality)
    set_model_health("rembg", "loading")
    try:
        from rembg import new_session
        models["rembg_sess"] = new_session("isnet-general-use")
        set_model_health("rembg", "loaded")
        logger.info("rembg (ISNet) background remover loaded")
    except Exception as e:
        logger.warning(f"rembg failed: {e}")
        try:
            from rembg import new_session
            models["rembg_sess"] = new_session("u2net")
            set_model_health("rembg", "loaded")
            logger.info("rembg (u2net) background remover loaded")
        except Exception as e2:
            set_model_health("rembg", "failed", str(e2))
            logger.warning(f"rembg fallback failed: {e2}")

    if runtime_device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    elif runtime_device == "mps":
        logger.info("Apple Metal (MPS) detected")
    else:
        logger.warning("No GPU backend available - running on CPU")

    # TripoSR
    if not installed_models.get("triposr"):
        set_model_health("triposr", "not_installed", "Not installed by installer selection")
    else:
        set_model_health("triposr", "loading")
        try:
            from tsr.system import TSR
            logger.info("Loading TripoSR model (~1 GB)...")
            m = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            m.renderer.set_chunk_size(131072 if runtime_device == "cuda" else 65536)
            m.to(runtime_device)
            models["triposr"] = m
            set_model_health("triposr", "loaded")
            logger.info(f"TripoSR loaded on {runtime_device}")
        except Exception as e:
            set_model_health("triposr", "failed", str(e))
            logger.warning(f"TripoSR failed to load: {e}")

    # Hunyuan3D-2 shape
    if runtime_device != "cuda":
        if not installed_models.get("hunyuan"):
            set_model_health("hunyuan", "not_installed", "Not installed by installer selection")
        else:
            set_model_health("hunyuan", "skipped", "CUDA/NVIDIA required")
        if not installed_models.get("trellis"):
            set_model_health("trellis", "not_installed", "Not installed by installer selection")
        else:
            set_model_health("trellis", "skipped", "CUDA/NVIDIA required")
        if not installed_models.get("trellis2"):
            set_model_health("trellis2", "not_installed", "Not installed by installer selection")
        else:
            set_model_health("trellis2", "skipped", "CUDA/NVIDIA required")
        logger.warning("Hunyuan3D-2 skipped: currently supported only on CUDA/NVIDIA")
        return

    if not installed_models.get("hunyuan"):
        set_model_health("hunyuan", "not_installed", "Not installed by installer selection")
    else:
        try:
            set_model_health("hunyuan", "loading")
            # Ensure hy3dgen is on path (it's a folder, not an installed package)
            _p = BASE_DIR / "hy3dgen"
            if _p.exists() and str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
            from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            logger.info("Loading Hunyuan3D-2 model (~8 GB first time, cached after)...")
            pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2",
                use_safetensors=True,
                device="cuda",
            )
            models["hunyuan"] = pipe
            set_model_health("hunyuan", "loaded")
            logger.info("Hunyuan3D-2 loaded")
        except Exception as e:
            set_model_health("hunyuan", "failed", str(e))
            logger.warning(f"Hunyuan3D-2 failed to load: {e}")

    # TRELLIS (CUDA-only, 12+ GB VRAM recommended)
    if not installed_models.get("trellis"):
        set_model_health("trellis", "not_installed", "Not installed by installer selection")
    else:
        try:
            set_model_health("trellis", "loading")
            os.environ.setdefault("SPCONV_ALGO", "native")
            os.environ.setdefault("ATTN_BACKEND", "xformers")
            os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

            # Pre-check dependencies
            import spconv
            import easydict

            from trellis.pipelines import TrellisImageTo3DPipeline
            logger.info("Loading TRELLIS model (~16 GB first time, cached after)...")
            pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
            pipe.cuda()
            models["trellis"] = pipe
            set_model_health("trellis", "loaded")
            logger.info("TRELLIS loaded")
        except ImportError as e:
            dep_name = str(e)
            if "spconv" in dep_name:
                set_model_health("trellis", "failed", "Missing: spconv (sparse convolutions)")
                logger.warning("TRELLIS failed to load: Missing spconv - ensure it was installed during setup")
            elif "kaolin" in dep_name:
                set_model_health(
                    "trellis",
                    "failed",
                    "Missing: kaolin (or failed flexicubes fallback patch in install.ps1)",
                )
                logger.warning("TRELLIS failed to load: Missing kaolin and no fallback patch detected")
            elif "easydict" in dep_name:
                set_model_health("trellis", "failed", "Missing: easydict")
                logger.warning("TRELLIS failed to load: Missing easydict")
            else:
                set_model_health("trellis", "failed", str(e))
                logger.warning(f"TRELLIS failed to load: {e}")
        except Exception as e:
            set_model_health("trellis", "failed", _format_model_load_error("TRELLIS", e))
            logger.warning(f"TRELLIS failed to load: {e}")

    # TRELLIS.2 (CUDA-only, requires cumesh + flex_gemm + o_voxel + spconv)
    if not installed_models.get("trellis2"):
        set_model_health("trellis2", "not_installed", "Not installed by installer selection")
    else:
        try:
            import importlib as _il
            set_model_health("trellis2", "loading")
            _trellis2_pkg = BASE_DIR / "trellis2"
            if not _trellis2_pkg.exists():
                raise ImportError("trellis2 package not found in backend/")

            # Probe required native deps before attempting a full load
            _missing = [d for d in ("cumesh", "flex_gemm") if _il.util.find_spec(d) is None]
            if _missing:
                raise ImportError(f"Missing TRELLIS.2 runtime deps: {', '.join(_missing)}")

            os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")
            os.environ.setdefault("ATTN_BACKEND", "xformers")
            os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            logger.info("Loading TRELLIS.2 model (~20 GB first time, cached after)...")
            pipe2 = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
            pipe2.cuda()
            models["trellis2"] = pipe2
            set_model_health("trellis2", "loaded")
            logger.info("TRELLIS.2 loaded")
        except ImportError as e:
            set_model_health("trellis2", "failed", _format_model_load_error("TRELLIS.2", e))
            logger.warning(f"TRELLIS.2 failed to load: {e}")
        except Exception as e:
            set_model_health("trellis2", "failed", _format_model_load_error("TRELLIS.2", e))
            logger.warning(f"TRELLIS.2 failed to load: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_all_models)
    yield


app = FastAPI(title="PIXFORM", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Data models

class JobStatus(BaseModel):
    job_id:      str
    status:      str            # queued | processing | cancelling | cancelled | done | error
    progress:    int = 0
    message:     str = ""
    model_used:  str = ""
    time_taken:  Optional[float] = None
    stl_url:     Optional[str] = None
    tmf_url:     Optional[str] = None
    glb_url:     Optional[str] = None
    obj_url:     Optional[str] = None
    preview_url: Optional[str] = None
    poly_count:  Optional[int] = None
    cancel_requested: Optional[bool] = None


# Utilities

def upd(job_id, **kw):
    if job_id in jobs:
        jobs[job_id].update(kw)


def _job_terminal(status: str) -> bool:
    return status in {"done", "error", "cancelled"}


def _cancel_requested(job_id: str) -> bool:
    j = jobs.get(job_id)
    return bool(j and j.get("cancel_requested"))


def _assert_not_cancelled(job_id: str):
    if _cancel_requested(job_id):
        raise RuntimeError("Job cancelled by user")


def _resolve_timeout_seconds(env_name: str, default_seconds: int) -> Optional[int]:
    raw = os.getenv(env_name, str(default_seconds)).strip()
    try:
        seconds = int(raw)
    except Exception:
        seconds = default_seconds
    # 0 (or negative) disables timeout for users that prefer waiting indefinitely.
    return None if seconds <= 0 else seconds


def remove_background(img, job_id):
    """Remove background using rembg. Returns RGBA PIL Image."""
    from PIL import Image
    sess = models.get("rembg_sess")
    if sess is None:
        return img.convert("RGBA")
    try:
        from rembg import remove as rembg_remove
        result = rembg_remove(img.convert("RGB"), session=sess)
        upd(job_id, progress=20, message="Background removed")
        return result.convert("RGBA")
    except Exception as e:
        logger.warning(f"Background removal failed: {e}")
        return img.convert("RGBA")


def _normalize_triposr_resolution(value: int) -> int:
    """Snap requested resolution to the closest supported TripoSR extraction level."""
    try:
        v = int(value)
    except Exception:
        return 512
    v = max(128, min(1024, v))
    return min(TRIPOSR_RES_LEVELS, key=lambda x: abs(x - v))


def _fast_finalize_mesh(mesh, target_profile: Optional[dict] = None):
    """Fast fallback mesh finalize path used when heavy postprocess times out."""
    import trimesh

    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    mesh = _apply_input_aspect_correction(mesh, target_profile)

    bounds = mesh.bounds
    size = max(bounds[1] - bounds[0])
    if size > 0:
        mesh.apply_scale(100.0 / size)

    return mesh


def postprocess_mesh(mesh, job_id, level="standard", target_profile: Optional[dict] = None):
    """High quality mesh post-processing for 3D printing.
    level: none | light | standard | heavy
    Goal: always produce a 100% watertight, slicer-ready mesh.
    """
    import trimesh
    import trimesh.smoothing
    import numpy as _np

    smooth_iters   = {"none": 0, "light": 5,  "standard": 15, "heavy": 25}.get(level, 15)
    poisson_depth  = {"none": 9, "light": 10, "standard": 11, "heavy": 12}.get(level, 11)
    poisson_points = {"none": 60000, "light": 100000, "standard": 200000, "heavy": 350000}.get(level, 200000)
    use_poisson = level in {"standard", "heavy"}

    logger.info(f"Post-processing [{level}]: {len(mesh.faces):,} faces")
    upd(job_id, progress=86, message="Cleaning mesh...")

    # ÔöÇÔöÇ 1. Keep largest component ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    components = mesh.split(only_watertight=False)
    if components:
        mesh = max(components, key=lambda c: len(c.faces))

    # ÔöÇÔöÇ 2. Basic repair ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # ÔöÇÔöÇ 3. Smoothing ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    if smooth_iters > 0:
        upd(job_id, progress=89, message="Smoothing mesh...")
        try:
            mesh = trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
        except Exception:
            try:
                mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters, lamb=0.2)
            except Exception:
                pass

    # Use Poisson only for standard/heavy to keep draft/low presets responsive.
    if use_poisson:
        upd(job_id, progress=91, message="Poisson reconstruction (watertight)...")
        try:
            import open3d as o3d

            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices  = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.compute_vertex_normals()

            pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=poisson_points)
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(30)

            poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
            )

            # Trim low-density outlier surface fragments based on quality level
            density_trim_pct = {"none": 1, "light": 3, "standard": 5, "heavy": 8}.get(level, 5)
            dens   = _np.asarray(densities)
            thresh = _np.percentile(dens, density_trim_pct)
            poisson_mesh.remove_vertices_by_mask(dens < thresh)
            poisson_mesh.compute_vertex_normals()

            v = _np.asarray(poisson_mesh.vertices)
            f = _np.asarray(poisson_mesh.triangles)
            if len(v) > 100 and len(f) > 100:
                candidate = trimesh.Trimesh(vertices=v, faces=f, process=True)
                if len(candidate.faces) > 100:
                    mesh = candidate
                    logger.info(f"Poisson OK: {len(mesh.faces):,} faces, watertight: {mesh.is_watertight}")

        except Exception as e:
            logger.warning(f"Poisson failed: {e} - continuing with trimesh repair")
    else:
        upd(job_id, progress=91, message="Skipping Poisson for fast preset...")

    # ÔöÇÔöÇ 5. Hole filling ÔÇö multiple passes ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    upd(job_id, progress=94, message="Filling holes...")
    for _ in range(5):
        if mesh.is_watertight:
            break
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

    # ÔöÇÔöÇ 6. Voxel remesh fallback ÔÇö guaranteed closed if still open ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    if not mesh.is_watertight:
        upd(job_id, progress=96, message="Voxel remesh (closing remaining holes)...")
        try:
            pitch    = mesh.extents.max() / 128
            vox      = mesh.voxelized(pitch=pitch)
            remeshed = vox.marching_cubes
            if remeshed.is_watertight and len(remeshed.faces) > 100:
                remeshed = trimesh.smoothing.filter_taubin(remeshed, iterations=5)
                mesh = remeshed
                logger.info(f"Voxel remesh: {len(mesh.faces):,} faces, watertight: {mesh.is_watertight}")
        except Exception as e:
            logger.warning(f"Voxel remesh failed: {e}")

    # ÔöÇÔöÇ 7. Final cleanup ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    # ÔöÇÔöÇ 8. Match final proportions to the input image silhouette ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    mesh = _apply_input_aspect_correction(mesh, target_profile)

    # ÔöÇÔöÇ 9. Scale to 100mm ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    bounds = mesh.bounds
    size = max(bounds[1] - bounds[0])
    if size > 0:
        mesh.apply_scale(100.0 / size)

    logger.info(f"Post-processing done: {len(mesh.faces):,} faces | watertight: {mesh.is_watertight}")
    return mesh


def to_trimesh(raw):
    import trimesh
    if isinstance(raw, trimesh.Trimesh):
        return raw
    if isinstance(raw, trimesh.Scene):
        parts = [g for g in raw.geometry.values() if len(g.faces) > 0]
        return trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()
    if hasattr(raw, "verts_packed"):
        return trimesh.Trimesh(
            vertices=raw.verts_packed().cpu().numpy(),
            faces=raw.faces_packed().cpu().numpy()
        )
    return trimesh.Trimesh(vertices=raw.vertices, faces=raw.faces)


def export_all(mesh, out_dir: Path, job_id: str):
    """Export mesh to STL, 3MF, GLB and OBJ."""
    stl = out_dir / "model.stl"
    tmf = out_dir / "model.3mf"
    glb = out_dir / "model.glb"
    obj = out_dir / "model.obj"

    try:
        mesh.export(str(stl))
    except Exception as e:
        raise RuntimeError(f"STL export failed: {e}") from e

    try:
        mesh.export(str(tmf))
    except Exception:
        _export_3mf_manual(mesh, tmf)

    for fmt_path in (glb, obj):
        try:
            mesh.export(str(fmt_path))
        except Exception as e:
            logger.warning(f"{fmt_path.suffix.upper()} export failed: {e}")

    upd(
        job_id,
        stl_url=f"/outputs/{job_id}/model.stl" if stl.exists() else None,
        tmf_url=f"/outputs/{job_id}/model.3mf" if tmf.exists() else None,
        glb_url=f"/outputs/{job_id}/model.glb" if glb.exists() else None,
        obj_url=f"/outputs/{job_id}/model.obj" if obj.exists() else None,
        poly_count=len(mesh.faces),
    )


def _export_3mf_manual(mesh, path: Path):
    model = ET.Element("model", {
        "unit": "millimeter",
        "xmlns": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
    })
    res = ET.SubElement(model, "resources")
    obj = ET.SubElement(res, "object", {"id": "1", "type": "model"})
    me  = ET.SubElement(obj, "mesh")
    ve  = ET.SubElement(me, "vertices")
    for v in mesh.vertices:
        ET.SubElement(ve, "vertex", {"x": f"{v[0]:.6f}", "y": f"{v[1]:.6f}", "z": f"{v[2]:.6f}"})
    te = ET.SubElement(me, "triangles")
    for f in mesh.faces:
        ET.SubElement(te, "triangle", {"v1": str(f[0]), "v2": str(f[1]), "v3": str(f[2])})
    build = ET.SubElement(model, "build")
    ET.SubElement(build, "item", {"objectid": "1"})
    xml_str = ET.tostring(model, encoding="unicode", xml_declaration=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        zf.writestr("_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        zf.writestr("3D/3dmodel.model", xml_str)


def render_preview(mesh, path: Path):
    from PIL import Image as _Img, ImageDraw as _Draw, ImageEnhance as _Enh
    import io as _io

    # ÔöÇÔöÇ Attempt 1: pyrender (high quality, multi-light) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    try:
        import pyrender
        import trimesh
        import numpy as _np

        scene = pyrender.Scene(bg_color=[20, 20, 26, 255], ambient_light=[0.15, 0.15, 0.15])

        mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(mesh_pr)

        # Camera ÔÇö isometric-ish view from upper-front-right
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        extent = (bounds[1] - bounds[0]).max()
        dist   = extent * 2.2

        camera = pyrender.PerspectiveCamera(yfov=_np.pi / 4.0, aspectRatio=1.0)
        cam_pose = _np.eye(4)
        cam_pose[:3, 3] = center + _np.array([dist * 0.6, dist * 0.4, dist * 0.8])
        # Look at center
        fwd = center - cam_pose[:3, 3]
        fwd /= _np.linalg.norm(fwd)
        right = _np.cross([0, 1, 0], fwd); right /= _np.linalg.norm(right)
        up    = _np.cross(fwd, right)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -fwd
        scene.add(camera, pose=cam_pose)

        # Key light (warm, front-right)
        key = pyrender.DirectionalLight(color=[1.0, 0.97, 0.90], intensity=4.5)
        scene.add(key, pose=cam_pose)

        # Fill light (cool, left)
        fill_pose = _np.eye(4)
        fill_pose[:3, 3] = center + _np.array([-dist * 0.8, dist * 0.2, dist * 0.4])
        fill = pyrender.DirectionalLight(color=[0.75, 0.85, 1.0], intensity=1.8)
        scene.add(fill, pose=fill_pose)

        # Rim light (back-top)
        rim_pose = _np.eye(4)
        rim_pose[:3, 3] = center + _np.array([0, dist * 1.2, -dist * 0.5])
        rim = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.2)
        scene.add(rim, pose=rim_pose)

        r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)
        color, _ = r.render(scene)
        r.delete()

        pil = _Img.fromarray(color)
        pil = _Enh.Contrast(pil).enhance(1.25)
        pil = _Enh.Sharpness(pil).enhance(1.2)
        pil.save(str(path))
        return
    except Exception:
        pass

    # ÔöÇÔöÇ Attempt 2: trimesh scene renderer ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    try:
        import trimesh
        scene = trimesh.Scene(mesh)
        png = scene.save_image(resolution=(800, 800), background=[20, 20, 26, 255])
        if png:
            pil = _Img.open(_io.BytesIO(png))
            pil = _Enh.Contrast(pil).enhance(1.25)
            pil = _Enh.Sharpness(pil).enhance(1.2)
            pil.save(str(path))
            return
    except Exception:
        pass

    # ÔöÇÔöÇ Fallback: PIL placeholder ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    try:
        img = _Img.new("RGB", (800, 800), (20, 20, 26))
        d = _Draw.Draw(img)
        d.text((310, 380), "3D Model", fill=(200, 255, 80))
        d.text((285, 410), f"{len(mesh.faces):,} faces", fill=(120, 120, 140))
        img.save(str(path))
    except Exception:
        pass


# ÔöÇÔöÇ TripoSR pipeline ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

async def run_triposr(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    try:
        import torch
        from PIL import Image
        from tsr.utils import resize_foreground

        t_start = time.time()
        upd(job_id, status="processing", progress=5, message="Loading image...")

        img = Image.open(image_path).convert("RGBA")
        _assert_not_cancelled(job_id)

        # Background removal
        if settings.get("remove_bg", True):
            upd(job_id, progress=10, message="Removing background...")
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(None, remove_background, img, job_id)
            _assert_not_cancelled(job_id)

        input_profile = _foreground_profile(img)

        # Preprocessing
        upd(job_id, progress=22, message="Preprocessing image...")
        img = resize_foreground(img, 0.85)
        img_np = np.array(img, dtype=np.float32) / 255.0
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            a = img_np[:, :, 3:4]
            img_np = img_np[:, :, :3] * a + 0.5 * (1.0 - a)
        img = Image.fromarray((img_np * 255.0).astype(np.uint8))
        # Enhance contrast and sharpness so the model sees clearer edges and detail
        from PIL import ImageEnhance
        img = ImageEnhance.Contrast(img).enhance(1.35)
        img = ImageEnhance.Sharpness(img).enhance(1.4)
        img = img.resize((512, 512), Image.LANCZOS)

        # Inference
        upd(job_id, progress=28, message="Generating 3D structure (TripoSR)...")
        loop = asyncio.get_event_loop()

        runtime_device = models.get("runtime_device", "cpu")

        def infer():
            with torch.no_grad():
                return models["triposr"]([img], device=runtime_device)

        scene_codes = await loop.run_in_executor(None, infer)
        _assert_not_cancelled(job_id)

        # Mesh extraction ÔÇö try highest resolution that fits in VRAM
        target_res = settings.get("resolution", 512)
        fallbacks = [r for r in TRIPOSR_RES_LEVELS if r <= target_res]
        if not fallbacks:
            fallbacks = [128]

        def extract():
            for res in fallbacks:
                try:
                    upd(job_id, progress=35, message=f"Extracting mesh at resolution {res}...")
                    result = models["triposr"].extract_mesh(
                        scene_codes, resolution=res, has_vertex_color=False
                    )[0]
                    logger.info(f"TripoSR mesh extracted at resolution {res}")
                    return result, res
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    logger.warning(f"Resolution {res} failed: {e}")
                    if runtime_device == "cuda":
                        torch.cuda.empty_cache()
            raise RuntimeError("All resolutions failed")

        raw, used_res = await loop.run_in_executor(None, extract)
        _assert_not_cancelled(job_id)

        upd(job_id, progress=75, message=f"Mesh extracted at {used_res}")

        # Post-processing
        mesh = to_trimesh(raw)
        if len(mesh.faces) == 0:
            raise RuntimeError("TripoSR produced an empty mesh")
        post_level = settings.get("post", "standard")
        upd(job_id, progress=80, message=f"Post-processing [{post_level}]...")
        loop = asyncio.get_event_loop()
        mesh = await loop.run_in_executor(None, lambda: postprocess_mesh(mesh, job_id, post_level, input_profile))
        _assert_not_cancelled(job_id)

        # Export
        upd(job_id, progress=95, message="Exporting files...")
        export_all(mesh, out_dir, job_id)
        render_preview(mesh, out_dir / "preview.png")
        _assert_not_cancelled(job_id)

        elapsed = round(time.time() - t_start, 1)
        upd(job_id,
            status="done", progress=100,
            message=f"Done in {elapsed}s - {len(mesh.faces):,} polygons",
            model_used=f"TripoSR (res {used_res})",
            time_taken=elapsed,
            preview_url=f"/outputs/{job_id}/preview.png",
        )

    except Exception as e:
        if str(e) == "Job cancelled by user":
            upd(job_id, status="cancelled", message="Cancelled by user")
            return
        logger.error(f"TripoSR error: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))


# ÔöÇÔöÇ Hunyuan3D-2 pipeline ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

async def run_hunyuan(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    try:
        import torch
        from PIL import Image

        t_start = time.time()
        upd(job_id, status="processing", progress=5, message="Loading image...")

        img = Image.open(image_path).convert("RGBA")
        _assert_not_cancelled(job_id)

        # Background removal
        if settings.get("remove_bg", True):
            upd(job_id, progress=10, message="Removing background...")
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(None, remove_background, img, job_id)
            _assert_not_cancelled(job_id)

        input_profile = _foreground_profile(img)

        # Convert to RGB with white background for Hunyuan3D
        img_rgb = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            img_rgb.paste(img, mask=img.split()[3])
        else:
            img_rgb = img.convert("RGB")

        # Enhance contrast and sharpness so the model sees clearer edges and detail
        from PIL import ImageEnhance
        img_rgb = ImageEnhance.Contrast(img_rgb).enhance(1.35)
        img_rgb = ImageEnhance.Sharpness(img_rgb).enhance(1.4)

        # Resize to 1024x1024 for higher quality input
        img_rgb = img_rgb.resize((1024, 1024), Image.LANCZOS)

        upd(job_id, progress=22, message="Generating 3D shape with Hunyuan3D-2...")
        upd(job_id, progress=25, message="This takes 2-5 minutes, please wait...")

        loop = asyncio.get_event_loop()

        def infer():
            with torch.no_grad():
                # Save temp image for pipeline
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    tmp = tmp_file.name
                img_rgb.save(tmp)
                try:
                    result = models["hunyuan"](image=tmp, num_inference_steps=settings.get('steps', 50))
                    return result[0]
                finally:
                    import os
                    if os.path.exists(tmp):
                        os.unlink(tmp)

        upd(job_id, progress=30, message="Running Hunyuan3D-2 diffusion model...")
        raw_mesh = await loop.run_in_executor(None, infer)
        _assert_not_cancelled(job_id)

        upd(job_id, progress=78, message="Mesh generated")

        # Post-processing
        mesh = to_trimesh(raw_mesh)
        if len(mesh.faces) == 0:
            raise RuntimeError("Hunyuan3D-2 produced an empty mesh")
        post_level = settings.get("post", "standard")
        upd(job_id, progress=80, message=f"Post-processing [{post_level}]...")
        mesh = await loop.run_in_executor(None, lambda: postprocess_mesh(mesh, job_id, post_level, input_profile))
        _assert_not_cancelled(job_id)

        # Export
        upd(job_id, progress=95, message="Exporting files...")
        export_all(mesh, out_dir, job_id)
        render_preview(mesh, out_dir / "preview.png")
        _assert_not_cancelled(job_id)

        elapsed = round(time.time() - t_start, 1)
        upd(job_id,
            status="done", progress=100,
            message=f"Done in {elapsed}s - {len(mesh.faces):,} polygons",
            model_used="Hunyuan3D-2",
            time_taken=elapsed,
            preview_url=f"/outputs/{job_id}/preview.png",
        )

    except Exception as e:
        if str(e) == "Job cancelled by user":
            upd(job_id, status="cancelled", message="Cancelled by user")
            return
        logger.error(f"Hunyuan3D-2 error: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))


# ÔöÇÔöÇ TRELLIS pipeline ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

async def run_trellis(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    try:
        import torch
        from PIL import Image

        t_start = time.time()
        upd(job_id, status="processing", progress=5, message="Loading image...")

        img = Image.open(image_path).convert("RGBA")
        _assert_not_cancelled(job_id)

        # Background removal (our rembg session; TRELLIS will skip its own if RGBA alpha is set)
        if settings.get("remove_bg", True):
            upd(job_id, progress=10, message="Removing background...")
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(None, remove_background, img, job_id)
            _assert_not_cancelled(job_id)

        input_profile = _foreground_profile(img)

        upd(job_id, progress=22, message="Generating 3D shape with TRELLIS...")
        upd(job_id, progress=25, message="This takes 5 to 10 minutes, please wait...")

        loop = asyncio.get_event_loop()
        steps = settings.get("steps", 50)
        ss_steps   = max(12, min(50, steps // 3))
        slat_steps = max(12, min(50, steps))

        def infer():
            with torch.no_grad():
                outputs = models["trellis"].run(
                    img,
                    seed=42,
                    sparse_structure_sampler_params={"steps": ss_steps},
                    slat_sampler_params={"steps": slat_steps},
                    formats=["mesh", "gaussian"],
                    preprocess_image=True,
                )
            return outputs

        upd(job_id, progress=30, message="Running TRELLIS diffusion model...")
        outputs = await loop.run_in_executor(None, infer)
        _assert_not_cancelled(job_id)

        upd(job_id, progress=78, message="3D structure generated")

        glb_textured = False
        textured_enabled = str(os.getenv("PIXFORM_TRELLIS_TEXTURED", "0")).strip().lower() in {"1", "true", "yes", "on"}
        texture_timeout = _resolve_timeout_seconds("PIXFORM_TRELLIS_TEXTURE_TIMEOUT_SEC", 480)

        def make_textured_glb():
            try:
                from trellis.utils import postprocessing_utils
                glb = postprocessing_utils.to_glb(
                    outputs["gaussian"][0],
                    outputs["mesh"][0],
                    simplify=0.95,
                    texture_size=1024,
                )
                glb.export(str(out_dir / "model.glb"))
                return True
            except Exception as e:
                logger.warning(f"TRELLIS textured GLB failed (nvdiffrast/mip-splatting may not be installed): {e}")
                return False

        if textured_enabled:
            upd(job_id, progress=79, message="Generating textured GLB (optional)...")
            try:
                tex_task = loop.run_in_executor(None, make_textured_glb)
                glb_textured = await asyncio.wait_for(tex_task, timeout=texture_timeout) if texture_timeout else await tex_task
            except asyncio.TimeoutError:
                logger.warning("TRELLIS textured GLB stage timed out; falling back to plain GLB")
                glb_textured = False
        else:
            logger.info("TRELLIS textured GLB disabled by default (set PIXFORM_TRELLIS_TEXTURED=1 to enable)")
        _assert_not_cancelled(job_id)

        def extract_trimesh():
            import trimesh as _trimesh
            m = outputs["mesh"][0]
            return _trimesh.Trimesh(
                vertices=m.vertices.cpu().numpy(),
                faces=m.faces.cpu().numpy(),
                process=True,
            )

        raw_mesh = await loop.run_in_executor(None, extract_trimesh)
        _assert_not_cancelled(job_id)

        if len(raw_mesh.faces) == 0:
            raise RuntimeError("TRELLIS produced an empty mesh")

        post_level = settings.get("post", "standard")
        upd(job_id, progress=80, message=f"Post-processing [{post_level}]...")
        post_default = {"none": 600, "light": 900, "standard": 1800, "heavy": 2700}.get(post_level, 1800)
        post_timeout = _resolve_timeout_seconds("PIXFORM_TRELLIS_POST_TIMEOUT_SEC", post_default)
        try:
            post_task = loop.run_in_executor(None, lambda: postprocess_mesh(raw_mesh, job_id, post_level, input_profile))
            mesh = await asyncio.wait_for(post_task, timeout=post_timeout) if post_timeout else await post_task
        except asyncio.TimeoutError:
            logger.warning(f"TRELLIS postprocess timed out after {post_timeout}s; using fast fallback mesh finalize")
            upd(job_id, progress=88, message="Postprocess timeout, using fast finalize...")
            mesh = _fast_finalize_mesh(raw_mesh.copy(), input_profile)
        _assert_not_cancelled(job_id)

        upd(job_id, progress=95, message="Exporting files...")

        stl      = out_dir / "model.stl"
        tmf      = out_dir / "model.3mf"
        glb_path = out_dir / "model.glb"
        obj      = out_dir / "model.obj"

        try:
            mesh.export(str(stl))
        except Exception as e:
            raise RuntimeError(f"STL export failed: {e}") from e

        try:
            mesh.export(str(tmf))
        except Exception:
            _export_3mf_manual(mesh, tmf)

        try:
            mesh.export(str(obj))
        except Exception as e:
            logger.warning(f"OBJ export failed: {e}")

        # If textured GLB was not produced, write plain GLB from mesh
        if not glb_textured:
            try:
                mesh.export(str(glb_path))
            except Exception as e:
                logger.warning(f"Plain GLB export failed: {e}")

        upd(
            job_id,
            stl_url=f"/outputs/{job_id}/model.stl"  if stl.exists()      else None,
            tmf_url=f"/outputs/{job_id}/model.3mf"  if tmf.exists()      else None,
            glb_url=f"/outputs/{job_id}/model.glb"  if glb_path.exists() else None,
            obj_url=f"/outputs/{job_id}/model.obj"  if obj.exists()      else None,
            poly_count=len(mesh.faces),
        )

        render_preview(mesh, out_dir / "preview.png")
        _assert_not_cancelled(job_id)

        elapsed = round(time.time() - t_start, 1)
        glb_note = " (textured)" if glb_textured else ""
        upd(job_id,
            status="done", progress=100,
            message=f"Done in {elapsed}s - {len(mesh.faces):,} polygons{glb_note}",
            model_used="TRELLIS",
            time_taken=elapsed,
            preview_url=f"/outputs/{job_id}/preview.png",
        )

    except Exception as e:
        if str(e) == "Job cancelled by user":
            upd(job_id, status="cancelled", message="Cancelled by user")
            return
        logger.error(f"TRELLIS error: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))


# ── TRELLIS.2 pipeline ───────────────────────────────────────────────────────

async def run_trellis2(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    try:
        import torch
        from PIL import Image

        t_start = time.time()
        upd(job_id, status="processing", progress=5, message="Loading image...")

        img = Image.open(image_path).convert("RGBA")
        _assert_not_cancelled(job_id)

        # Background removal (trellis2 honours RGBA alpha - no double-processing)
        if settings.get("remove_bg", True):
            upd(job_id, progress=10, message="Removing background...")
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(None, remove_background, img, job_id)
            _assert_not_cancelled(job_id)

        input_profile = _foreground_profile(img)

        upd(job_id, progress=22, message="Generating 3D shape with TRELLIS.2...")
        upd(job_id, progress=25, message="This takes 6 to 12 minutes, please wait...")

        loop = asyncio.get_event_loop()
        steps = settings.get("steps", 50)

        def infer():
            with torch.no_grad():
                return models["trellis2"].run(
                    img,
                    seed=42,
                    shape_slat_sampler_params={"steps": steps},
                    tex_slat_sampler_params={"steps": steps},
                    preprocess_image=True,
                )

        upd(job_id, progress=30, message="Running TRELLIS.2 diffusion model...")
        outputs = await loop.run_in_executor(None, infer)
        _assert_not_cancelled(job_id)

        upd(job_id, progress=78, message="3D structure generated")

        # outputs is List[MeshWithVoxel]; take first sample
        m = outputs[0]

        # ── Textured GLB export (requires cumesh + nvdiffrast) ────────────────
        glb_textured = False

        def make_textured_glb():
            try:
                import o_voxel
                glb = o_voxel.postprocess.to_glb(
                    vertices=m.vertices,
                    faces=m.faces,
                    attr_volume=m.attrs,
                    coords=m.coords,
                    attr_layout=m.layout,
                    voxel_size=m.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=1_000_000,
                    texture_size=2048,
                )
                glb.export(str(out_dir / "model.glb"))
                return True
            except Exception as e:
                logger.warning(
                    f"TRELLIS.2 textured GLB failed (cumesh/nvdiffrast may not be installed): {e}"
                )
                return False

        glb_textured = await loop.run_in_executor(None, make_textured_glb)
        _assert_not_cancelled(job_id)

        # ── Extract trimesh for STL / 3MF / OBJ (and plain GLB fallback) ─────
        def extract_trimesh():
            import trimesh as _trimesh
            return _trimesh.Trimesh(
                vertices=m.vertices.cpu().numpy(),
                faces=m.faces.cpu().numpy(),
                process=True,
            )

        raw_mesh = await loop.run_in_executor(None, extract_trimesh)
        _assert_not_cancelled(job_id)

        if len(raw_mesh.faces) == 0:
            raise RuntimeError("TRELLIS.2 produced an empty mesh")

        post_level = settings.get("post", "standard")
        upd(job_id, progress=80, message=f"Post-processing [{post_level}]...")
        mesh = await loop.run_in_executor(None, lambda: postprocess_mesh(raw_mesh, job_id, post_level, input_profile))
        _assert_not_cancelled(job_id)

        # ── Export ─────────────────────────────────────────────────────────────
        upd(job_id, progress=95, message="Exporting files...")

        stl      = out_dir / "model.stl"
        tmf      = out_dir / "model.3mf"
        glb_path = out_dir / "model.glb"
        obj      = out_dir / "model.obj"

        try:
            mesh.export(str(stl))
        except Exception as e:
            raise RuntimeError(f"STL export failed: {e}") from e

        try:
            mesh.export(str(tmf))
        except Exception:
            _export_3mf_manual(mesh, tmf)

        try:
            mesh.export(str(obj))
        except Exception as e:
            logger.warning(f"OBJ export failed: {e}")

        if not glb_textured:
            try:
                mesh.export(str(glb_path))
            except Exception as e:
                logger.warning(f"Plain GLB export failed: {e}")

        upd(
            job_id,
            stl_url=f"/outputs/{job_id}/model.stl"  if stl.exists()      else None,
            tmf_url=f"/outputs/{job_id}/model.3mf"  if tmf.exists()      else None,
            glb_url=f"/outputs/{job_id}/model.glb"  if glb_path.exists() else None,
            obj_url=f"/outputs/{job_id}/model.obj"  if obj.exists()      else None,
            poly_count=len(mesh.faces),
        )

        render_preview(mesh, out_dir / "preview.png")
        _assert_not_cancelled(job_id)

        elapsed = round(time.time() - t_start, 1)
        glb_note = " (textured)" if glb_textured else ""
        upd(job_id,
            status="done", progress=100,
            message=f"Done in {elapsed}s - {len(mesh.faces):,} polygons{glb_note}",
            model_used="TRELLIS.2",
            time_taken=elapsed,
            preview_url=f"/outputs/{job_id}/preview.png",
        )

    except Exception as e:
        if str(e) == "Job cancelled by user":
            upd(job_id, status="cancelled", message="Cancelled by user")
            return
        logger.error(f"TRELLIS.2 error: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))


# ÔöÇÔöÇ Demo pipeline (no GPU) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

async def run_demo(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    import trimesh
    for p, m in [(20, "Analyzing..."), (50, "Building mesh..."), (80, "Exporting...")]:
        upd(job_id, status="processing", progress=p, message=m)
        await asyncio.sleep(0.8)
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    export_all(mesh, out_dir, job_id)
    render_preview(mesh, out_dir / "preview.png")
    upd(job_id, status="done", progress=100,
        message="Demo mode - no GPU available",
        model_used="Demo",
        preview_url=f"/outputs/{job_id}/preview.png")


# ÔöÇÔöÇ Routes ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

@app.get("/health")
async def health():
    import torch
    mps_backend = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps_backend) and mps_backend.is_available()
    return {
        "triposr":    models["triposr"] is not None,
        "hunyuan":    models["hunyuan"] is not None,
        "trellis":    models["trellis"] is not None,
        "trellis2":   models["trellis2"] is not None,
        "triposr_installed": models.get("installed_models", {}).get("triposr", False),
        "hunyuan_installed": models.get("installed_models", {}).get("hunyuan", False),
        "trellis_installed": models.get("installed_models", {}).get("trellis", False),
        "trellis2_installed": models.get("installed_models", {}).get("trellis2", False),
        "rembg":      models["rembg_sess"] is not None,
        "cuda":       torch.cuda.is_available(),
        "mps":        mps_ok,
        "runtime_device": models.get("runtime_device", "cpu"),
        "gpu":        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "triposr_status": model_health["triposr"]["status"],
        "triposr_error": model_health["triposr"]["error"],
        "hunyuan_status": model_health["hunyuan"]["status"],
        "hunyuan_error": model_health["hunyuan"]["error"],
        "trellis_status": model_health["trellis"]["status"],
        "trellis_error": model_health["trellis"]["error"],
        "trellis2_status": model_health["trellis2"]["status"],
        "trellis2_error": model_health["trellis2"]["error"],
        "rembg_status": model_health["rembg"]["status"],
        "rembg_error": model_health["rembg"]["error"],
    }


@app.post("/convert", response_model=JobStatus)
async def convert(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model:      str  = "triposr",    # triposr | hunyuan | trellis
    remove_bg:  str  = "true",
    resolution: int  = 512,
    steps:      int  = 50,           # hunyuan/trellis diffusion steps
    post:       str  = "standard",   # none | light | standard | heavy
):
    model = (model or "triposr").lower().strip()
    if model not in {"triposr", "hunyuan", "trellis", "trellis2"}:
        raise HTTPException(400, "Invalid model. Use: triposr, hunyuan, trellis, or trellis2")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Images only")

    if post not in VALID_POST_LEVELS:
        raise HTTPException(400, "Invalid post level. Use: none, light, standard, heavy")

    job_id   = str(uuid.uuid4())
    ext      = Path(file.filename or "image.jpg").suffix or ".jpg"
    img_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    settings = {
        "remove_bg":  remove_bg.lower() == "true",
        "resolution": _normalize_triposr_resolution(resolution),
        "steps":      max(5, min(100, steps)),
        "post":       post,
    }

    # Determine which model to use
    if model == "triposr":
        if models["triposr"] is None:
            raise HTTPException(503, _model_unavailable_message("triposr", "TripoSR"))
        run_fn = run_triposr
        model_label = "TripoSR"
    elif model == "hunyuan":
        if models["hunyuan"] is None:
            raise HTTPException(503, _model_unavailable_message("hunyuan", "Hunyuan3D-2"))
        run_fn = run_hunyuan
        model_label = "Hunyuan3D-2"
    elif model == "trellis":
        if models["trellis"] is None:
            raise HTTPException(503, _model_unavailable_message("trellis", "TRELLIS"))
        run_fn = run_trellis
        model_label = "TRELLIS"
    elif model == "trellis2":
        if models["trellis2"] is None:
            raise HTTPException(503, _model_unavailable_message("trellis2", "TRELLIS.2"))
        run_fn = run_trellis2
        model_label = "TRELLIS.2"
    else:
        run_fn = run_demo
        model_label = "Demo"

    jobs[job_id] = dict(
        job_id=job_id, status="queued", progress=0,
        message="Queued...",
        model_used=model_label,
        time_taken=None,
        stl_url=None, tmf_url=None, glb_url=None, obj_url=None,
        preview_url=None, poly_count=None,
        cancel_requested=False,
    )

    out_dir = OUTPUT_DIR / job_id
    out_dir.mkdir(exist_ok=True)

    background_tasks.add_task(run_fn, job_id, img_path, out_dir, settings)
    return JobStatus(**jobs[job_id])


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return JobStatus(**jobs[job_id])


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if _job_terminal(job.get("status", "")):
        return {"job_id": job_id, "status": job.get("status"), "cancel_requested": False}

    job["cancel_requested"] = True
    if job.get("status") in {"queued", "processing"}:
        job["status"] = "cancelling"
        job["message"] = "Cancel requested, waiting for safe stop..."

    return {"job_id": job_id, "status": job.get("status"), "cancel_requested": True}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id in jobs and not _job_terminal(jobs[job_id].get("status", "")):
        jobs[job_id]["cancel_requested"] = True
        jobs[job_id]["status"] = "cancelling"
        jobs[job_id]["message"] = "Cancel requested, waiting for safe stop..."
        return {"job_id": job_id, "status": "cancelling", "deleted": False}

    jobs.pop(job_id, None)
    shutil.rmtree(OUTPUT_DIR / job_id, ignore_errors=True)
    return {"deleted": job_id}


app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    encodings = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")
    for candidate in [
        BASE_DIR.parent / "frontend" / "index.html",
        BASE_DIR.parent / "index.html",
        Path.cwd() / "frontend" / "index.html",
    ]:
        if candidate.exists():
            for enc in encodings:
                try:
                    return candidate.read_text(encoding=enc)
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Failed reading frontend file {candidate} with {enc}: {e}")
                    break
    return HTMLResponse("<h1>PIXFORM</h1><p><a href='/docs'>API docs</a></p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
