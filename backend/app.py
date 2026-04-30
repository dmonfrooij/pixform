"""
PIXFORM Backend
Image to 3D pipeline - TripoSR (fast) + Hunyuan3D-2 (quality) + TRELLIS (best)
"""
import os, sys, uuid, shutil, asyncio, logging, zipfile, time, json, gc, base64
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from urllib import request as urlrequest, error as urlerror
from urllib.parse import urlparse

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
MODEL_KEYS = ("triposr", "hunyuan", "trellis", "trellis2")

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
    "active_models": {},
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
trellis2_infer_lock = asyncio.Lock()

VALID_POST_LEVELS = {"none", "light", "standard", "heavy"}
TRIPOSR_RES_LEVELS = [1024, 896, 768, 640, 512, 448, 384, 320, 256, 192, 128]
TRELLIS2_RES_LEVELS = [512, 1024, 1536]
TRELLIS2_STAGE_DEFAULTS = {
    "sparse": {"guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0},
    "shape": {"guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0},
    "tex": {"guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0},
}
TRELLIS2_STEP_PROFILES = {
    # Slider=20 -> sparse/shape/tex = 12/20/16
    "conservative": {"sparse_ratio": 0.60, "tex_ratio": 0.80},
    # Slider=20 -> sparse/shape/tex = 12/20/20
    "balanced": {"sparse_ratio": 0.60, "tex_ratio": 1.00},
    # Slider=20 -> sparse/shape/tex = 16/20/20
    "aggressive": {"sparse_ratio": 0.80, "tex_ratio": 1.00},
}


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


def _resolve_cleanup_seconds(env_name: str, default_seconds: int) -> int:
    try:
        value = int(str(os.getenv(env_name, default_seconds)).strip())
    except Exception:
        value = default_seconds
    return max(5, value)


def _format_model_load_error(model_name: str, exc: Exception) -> str:
    """Return concise, actionable load errors for UI health cards."""
    raw = str(exc)
    low = raw.lower()
    if _is_hf_auth_error_text(low):
        dinov3_hint = ""
        if "dinov3" in low or "facebook/dinov3" in low:
            dinov3_hint = (
                " TRELLIS.2 requires 'facebook/dinov3-vitl16-pretrain-lvd1689m' (gated HuggingFace model). "
                "Run 'download_dinov3.py' once to save it locally - after that TRELLIS.2 runs fully offline. "
                "See README section 7 or run: python download_dinov3.py"
            )
        return (
            f"{model_name} could not access a gated model dependency.{dinov3_hint}"
        )
    if "no attribute flexidualgridvaedecoder" in low or "no attribute sparseunetvaedecoder" in low:
        return (
            f"{model_name} runtime package is missing required TRELLIS.2 classes "
            "(FlexiDualGridVaeDecoder/SparseUnetVaeDecoder). "
            "Update backend/trellis2 to a TRELLIS.2-compatible code release; reinstall alone won't fix this."
        )
    return raw


def _is_hf_auth_error_text(low_text: str) -> bool:
    hf_access_markers = (
        "cannot access gated repo",
        "gated repo",
        "access to model",
        "please log in",
        "401",
        "403",
        "unauthorized",
        "repository not found",
        "invalid username or password",
    )
    return any(marker in low_text for marker in hf_access_markers)


def _resolve_trellis2_model_source() -> str:
    """
    Resolve TRELLIS.2 model source.

    Priority:
      1) PIXFORM_TRELLIS2_MODEL
      2) backend/models/trellis2_4b (local/offline default)
      3) microsoft/TRELLIS.2-4B (remote, may require auth)
    """
    explicit = os.getenv("PIXFORM_TRELLIS2_MODEL", "").strip()
    if explicit:
        return explicit

    local_default = BASE_DIR / "models" / "trellis2_4b"
    if local_default.exists():
        return str(local_default)

    return "microsoft/TRELLIS.2-4B"


def _resolve_hf_token() -> Optional[str]:
    for env_name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.getenv(env_name, "").strip()
        if token:
            return token
    return None


def _load_trellis2_pipeline_with_auth_compat(Trellis2ImageTo3DPipeline, source: str, hf_token: Optional[str]):
    if not hf_token:
        return Trellis2ImageTo3DPipeline.from_pretrained(source)

    # Make token available for libraries that read auth from env only.
    os.environ.setdefault("HF_TOKEN", hf_token)

    for kw_name in ("token", "use_auth_token"):
        try:
            return Trellis2ImageTo3DPipeline.from_pretrained(source, **{kw_name: hf_token})
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" in msg and kw_name in msg:
                logger.info("TRELLIS.2 loader does not support '%s'; trying next auth method", kw_name)
                continue
            raise

    logger.info("TRELLIS.2 loader uses env-based auth fallback")
    return Trellis2ImageTo3DPipeline.from_pretrained(source)


def _resolve_trellis2_endpoint_url() -> str:
    return os.getenv("PIXFORM_TRELLIS2_ENDPOINT_URL", "").strip()


def _resolve_trellis2_endpoint_timeout_sec() -> int:
    try:
        v = int(str(os.getenv("PIXFORM_TRELLIS2_ENDPOINT_TIMEOUT_SEC", "90")).strip())
    except Exception:
        v = 90
    return max(10, v)


def _redacted_endpoint_for_log(endpoint_url: str) -> str:
    try:
        p = urlparse(endpoint_url)
        host = p.netloc or endpoint_url
        path = p.path or ""
        return f"{host}{path}"
    except Exception:
        return endpoint_url


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


def _resolve_active_models(installed_models: dict) -> dict:
    active = {name: bool(installed_models.get(name, False)) for name in MODEL_KEYS}
    raw = os.getenv("PIXFORM_ACTIVE_MODELS", "").strip().lower()
    if not raw or raw in {"all", "installed", "*", "a"}:
        return active

    aliases = {
        "1": "triposr",
        "2": "hunyuan",
        "3": "trellis",
        "4": "trellis2",
        "triposr": "triposr",
        "hunyuan": "hunyuan",
        "hunyuan3d": "hunyuan",
        "hunyuan3d-2": "hunyuan",
        "trellis": "trellis",
        "trellis2": "trellis2",
        "trellis.2": "trellis2",
    }
    selected = set()
    unknown = []
    for part in raw.replace(";", ",").split(","):
        token = part.strip().lower()
        if not token:
            continue
        model_name = aliases.get(token)
        if model_name:
            selected.add(model_name)
        else:
            unknown.append(token)

    if unknown:
        logger.warning(f"Ignoring unknown PIXFORM_ACTIVE_MODELS entries: {', '.join(unknown)}")
    if not selected:
        logger.warning("PIXFORM_ACTIVE_MODELS resolved to no known models; falling back to all installed models")
        return active

    return {name: bool(installed_models.get(name, False)) and name in selected for name in MODEL_KEYS}


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
    if status == "inactive":
        return f"{label} is installed but inactive for this session. Restart PIXFORM and choose {label} as the active model."
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
    active_models = _resolve_active_models(installed_models)
    models["installed_models"] = installed_models
    models["active_models"] = active_models
    logger.info(f"Runtime device: {runtime_device}")
    logger.info(
        "Active models this session: %s",
        ", ".join([name for name in MODEL_KEYS if active_models.get(name)]) or "none",
    )

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
    elif not active_models.get("triposr"):
        set_model_health("triposr", "inactive", "Disabled at startup by active model selection")
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
        elif not active_models.get("hunyuan"):
            set_model_health("hunyuan", "inactive", "Disabled at startup by active model selection")
        else:
            set_model_health("hunyuan", "skipped", "CUDA/NVIDIA required")
        if not installed_models.get("trellis"):
            set_model_health("trellis", "not_installed", "Not installed by installer selection")
        elif not active_models.get("trellis"):
            set_model_health("trellis", "inactive", "Disabled at startup by active model selection")
        else:
            set_model_health("trellis", "skipped", "CUDA/NVIDIA required")
        if not installed_models.get("trellis2"):
            set_model_health("trellis2", "not_installed", "Not installed by installer selection")
        elif not active_models.get("trellis2"):
            set_model_health("trellis2", "inactive", "Disabled at startup by active model selection")
        else:
            set_model_health("trellis2", "skipped", "CUDA/NVIDIA required")
        logger.warning("Hunyuan3D-2 skipped: currently supported only on CUDA/NVIDIA")
        return

    if not installed_models.get("hunyuan"):
        set_model_health("hunyuan", "not_installed", "Not installed by installer selection")
    elif not active_models.get("hunyuan"):
        set_model_health("hunyuan", "inactive", "Disabled at startup by active model selection")
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
    elif not active_models.get("trellis"):
        set_model_health("trellis", "inactive", "Disabled at startup by active model selection")
    else:
        try:
            set_model_health("trellis", "loading")
            # native is slower but more predictable on Windows; allow opt-in auto via env override.
            spconv_algo = os.getenv("PIXFORM_SPCONV_ALGO", "native").strip().lower()
            if spconv_algo not in {"auto", "native"}:
                spconv_algo = "native"
            os.environ["SPCONV_ALGO"] = spconv_algo
            os.environ.setdefault("ATTN_BACKEND", "xformers")
            os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
            logger.info(
                "TRELLIS backends: SPCONV_ALGO=%s, ATTN_BACKEND=%s, SPARSE_ATTN_BACKEND=%s",
                os.environ.get("SPCONV_ALGO"),
                os.environ.get("ATTN_BACKEND"),
                os.environ.get("SPARSE_ATTN_BACKEND"),
            )

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
    elif not active_models.get("trellis2"):
        set_model_health("trellis2", "inactive", "Disabled at startup by active model selection")
    else:
        try:
            import importlib as _il
            set_model_health("trellis2", "loading")
            _trellis2_pkg = BASE_DIR / "trellis2"
            if not _trellis2_pkg.exists():
                raise ImportError("trellis2 package not found in backend/")

            # Probe required native deps before attempting a full load
            _missing_hard = [d for d in ("flex_gemm",) if _il.util.find_spec(d) is None]
            if _missing_hard:
                raise ImportError(f"Missing TRELLIS.2 runtime deps: {', '.join(_missing_hard)}")
            _missing_soft = [d for d in ("cumesh", "o_voxel") if _il.util.find_spec(d) is None]
            if _missing_soft:
                logger.warning(f"TRELLIS.2: optional native deps missing ({', '.join(_missing_soft)}); running in degraded mode")

            os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")
            os.environ.setdefault("ATTN_BACKEND", "xformers")
            os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            trellis2_source = _resolve_trellis2_model_source()
            logger.info("Loading TRELLIS.2 model (~20 GB first time, cached after)...")
            logger.info("TRELLIS.2 source: %s", trellis2_source)
            hf_token = _resolve_hf_token()
            pipe2 = _load_trellis2_pipeline_with_auth_compat(Trellis2ImageTo3DPipeline, trellis2_source, hf_token)
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
    cleanup_stop = asyncio.Event()
    cleanup_task = asyncio.create_task(_job_cleanup_loop(cleanup_stop))
    try:
        yield
    finally:
        cleanup_stop.set()
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


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
    stage: Optional[str] = None
    last_update_ts: Optional[float] = None
    endpoint_attempted: Optional[bool] = None
    endpoint_used: Optional[bool] = None
    endpoint_fail_reason: Optional[str] = None


# Utilities

def upd(job_id, **kw):
    if job_id in jobs:
        kw.setdefault("last_update_ts", time.time())
        if "status" in kw and _job_terminal(str(kw["status"])):
            kw.setdefault("completed_at", time.time())
            kw.setdefault(
                "cleanup_after_ts",
                time.time() + _resolve_cleanup_seconds("PIXFORM_JOB_RETENTION_SEC", 3600),
            )
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


def _safe_unlink(path_like) -> bool:
    try:
        path = Path(path_like)
        if path.exists():
            path.unlink()
            return True
    except Exception as e:
        logger.warning(f"Cleanup unlink failed for {path_like}: {e}")
    return False


def _safe_rmtree(path_like) -> bool:
    try:
        path = Path(path_like)
        if path.exists():
            shutil.rmtree(path, ignore_errors=False)
            return True
    except Exception as e:
        logger.warning(f"Cleanup rmtree failed for {path_like}: {e}")
    return False


def _release_runtime_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _cleanup_job_artifacts(job: dict, remove_upload: bool = True, remove_output: bool = True) -> dict:
    removed = {"upload": False, "output": False}
    if remove_upload:
        upload_path = job.get("upload_path")
        if upload_path:
            removed["upload"] = _safe_unlink(upload_path)
    if remove_output:
        output_dir = job.get("output_dir")
        if output_dir:
            removed["output"] = _safe_rmtree(output_dir)
    return removed


def _finalize_job_resources(job_id: str):
    job = jobs.get(job_id)
    if not job:
        _release_runtime_memory()
        return

    # Uploads are only needed until preprocessing/inference starts; keep outputs until manual or timed cleanup.
    if not job.get("upload_deleted"):
        job["upload_deleted"] = _safe_unlink(job.get("upload_path")) if job.get("upload_path") else True

    if job.get("delete_requested") and _job_terminal(str(job.get("status", ""))):
        _cleanup_job_artifacts(job, remove_upload=not job.get("upload_deleted"), remove_output=True)
        jobs.pop(job_id, None)

    _release_runtime_memory()


def _prune_old_jobs(force_terminal_cleanup: bool = False) -> dict:
    now = time.time()
    removed_jobs = 0
    removed_outputs = 0
    removed_uploads = 0
    for job_id, job in list(jobs.items()):
        status = str(job.get("status", ""))
        if not _job_terminal(status):
            continue

        cleanup_after_ts = float(job.get("cleanup_after_ts") or 0.0)
        should_remove = force_terminal_cleanup or bool(job.get("delete_requested")) or (cleanup_after_ts and now >= cleanup_after_ts)
        if not should_remove:
            continue

        removed = _cleanup_job_artifacts(job, remove_upload=not job.get("upload_deleted"), remove_output=True)
        removed_uploads += int(bool(removed.get("upload")))
        removed_outputs += int(bool(removed.get("output")))
        jobs.pop(job_id, None)
        removed_jobs += 1

    if removed_jobs:
        logger.info(
            "Cleaned up %s old job(s) (uploads=%s, outputs=%s)",
            removed_jobs,
            removed_uploads,
            removed_outputs,
        )
    return {"jobs": removed_jobs, "uploads": removed_uploads, "outputs": removed_outputs}


async def _job_cleanup_loop(stop_event: asyncio.Event):
    interval = _resolve_cleanup_seconds("PIXFORM_CLEANUP_INTERVAL_SEC", 120)
    while not stop_event.is_set():
        try:
            _prune_old_jobs()
        except Exception as e:
            logger.warning(f"Background cleanup loop failed: {e}")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


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


def _image_has_meaningful_alpha(img) -> bool:
    """True when the image already carries a usable foreground alpha mask."""
    try:
        if img is None or "A" not in img.getbands():
            return False
        alpha = np.asarray(img.getchannel("A"), dtype=np.uint8)
        return bool(alpha.size) and int(alpha.max()) > 0 and int(alpha.min()) < 250
    except Exception:
        return False


def _resolve_trellis2_seed(settings: dict) -> int:
    """Mirror official app behavior: randomize by default, allow fixed override for debugging."""
    max_seed = (1 << 31) - 1

    for raw in (settings.get("seed"), os.getenv("PIXFORM_TRELLIS2_FIXED_SEED", "")):
        if raw in (None, ""):
            continue
        try:
            return max(0, min(int(raw), max_seed))
        except Exception:
            continue

    return int.from_bytes(os.urandom(4), "big") & max_seed


def _normalize_triposr_resolution(value: int) -> int:
    """Snap requested resolution to the closest supported TripoSR extraction level."""
    try:
        v = int(value)
    except Exception:
        return 512
    v = max(128, min(1024, v))
    return min(TRIPOSR_RES_LEVELS, key=lambda x: abs(x - v))


def _normalize_trellis2_resolution(value: int) -> int:
    """Snap TRELLIS.2 resolution to supported pipelines: 512, 1024, 1536."""
    try:
        v = int(value)
    except Exception:
        return 1024
    v = max(512, min(1536, v))
    return min(TRELLIS2_RES_LEVELS, key=lambda x: abs(x - v))


def _resolve_trellis2_steps_cap() -> int:
    """Default cap is official-like (50), with optional env override for expert tuning."""
    try:
        cap = int(str(os.getenv("PIXFORM_TRELLIS2_MAX_STAGE_STEPS", "50")).strip())
    except Exception:
        cap = 50
    return max(8, min(cap, 300))


def _normalize_trellis2_steps(value: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = 36
    return max(8, min(v, _resolve_trellis2_steps_cap()))


def _resolve_trellis2_sampler_profile() -> str:
    profile = str(os.getenv("PIXFORM_TRELLIS2_SAMPLER_PROFILE", "balanced")).strip().lower()
    if profile not in TRELLIS2_STEP_PROFILES:
        return "balanced"
    return profile


def _resolve_trellis2_stage_steps(ui_steps: int) -> tuple[int, int, int]:
    requested = _normalize_trellis2_steps(ui_steps)
    stage_max = _resolve_trellis2_steps_cap()

    shape_steps = min(requested, stage_max)
    profile = TRELLIS2_STEP_PROFILES[_resolve_trellis2_sampler_profile()]
    sparse_steps = int(round(shape_steps * profile["sparse_ratio"]))
    tex_steps = int(round(shape_steps * profile["tex_ratio"]))
    sparse_steps = max(8, min(sparse_steps, stage_max))
    tex_steps = max(8, min(tex_steps, stage_max))
    return sparse_steps, shape_steps, tex_steps


def _resolve_trellis2_pipeline_type(resolution: int) -> str:
    req_resolution = _normalize_trellis2_resolution(resolution)
    if req_resolution == 1536:
        return "1536_cascade"
    if req_resolution == 1024:
        return "1024_cascade"
    return "512"


def _build_trellis2_sampler_params(ui_steps: int) -> dict:
    sparse_steps, shape_steps, tex_steps = _resolve_trellis2_stage_steps(ui_steps)
    sparse = {"steps": sparse_steps, **TRELLIS2_STAGE_DEFAULTS["sparse"]}
    shape = {"steps": shape_steps, **TRELLIS2_STAGE_DEFAULTS["shape"]}
    tex = {"steps": tex_steps, **TRELLIS2_STAGE_DEFAULTS["tex"]}
    return {
        "sparse_structure_sampler_params": sparse,
        "shape_slat_sampler_params": shape,
        "tex_slat_sampler_params": tex,
        "_steps_tuple": (sparse_steps, shape_steps, tex_steps),
    }


def _fast_finalize_mesh(mesh, target_profile: Optional[dict] = None):
    """Fast fallback mesh finalize path used when heavy postprocess times out.
    Applies basic repair + a low-resolution voxel remesh to guarantee a solid, closed mesh.
    """
    import trimesh
    import trimesh.smoothing

    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    # If still not watertight, do a fast low-res voxel remesh to guarantee closure.
    # This is much faster than the full Poisson pipeline but still produces a solid mesh.
    if not mesh.is_watertight:
        try:
            pitch = max(float(mesh.extents.max()) / 96, 1e-6)
            vox = mesh.voxelized(pitch=pitch)
            vox.fill()
            remeshed = vox.marching_cubes
            if len(remeshed.faces) > 100:
                smoothed = trimesh.smoothing.filter_taubin(remeshed, iterations=3)
                mesh = smoothed if smoothed.is_watertight else remeshed
        except Exception:
            pass

    mesh = _apply_input_aspect_correction(mesh, target_profile)

    bounds = mesh.bounds
    size = max(bounds[1] - bounds[0])
    if size > 0:
        mesh.apply_scale(100.0 / size)

    return mesh


def _try_meshfix_watertight(mesh):
    """Try to make mesh watertight with MeshFix while preserving geometry detail.
    Returns original mesh if pymeshfix is unavailable or repair fails.
    """
    import trimesh
    import numpy as _np

    try:
        import pymeshfix

        mf = pymeshfix.MeshFix(_np.asarray(mesh.vertices), _np.asarray(mesh.faces))
        mf.repair(verbose=False, joincomp=True, remove_smallest_components=False)
        v, f = mf.v, mf.f
        if len(v) > 100 and len(f) > 100:
            repaired = trimesh.Trimesh(vertices=v, faces=f, process=False)
            return repaired
    except Exception as e:
        logger.info(f"MeshFix skipped: {e}")

    return mesh


def postprocess_mesh(mesh, job_id, level="standard", target_profile: Optional[dict] = None, model_key: Optional[str] = None):
    """High quality mesh post-processing for 3D printing.
    level: none | light | standard | heavy
    Goal: always produce a 100% watertight, slicer-ready mesh.
    """
    import trimesh
    import trimesh.smoothing
    import numpy as _np

    smooth_iters        = {"none": 0, "light": 5,  "standard": 10, "heavy": 15}.get(level, 10)
    poisson_depth       = {"none": 9, "light": 10, "standard": 12, "heavy": 13}.get(level, 12)
    poisson_points      = {"none": 60000, "light": 150000, "standard": 350000, "heavy": 600000}.get(level, 350000)
    post_smooth_iters   = {"none": 0, "light": 3,  "standard": 8,  "heavy": 12}.get(level, 8)
    use_poisson = level in {"standard", "heavy"}
    is_trellis2 = model_key == "trellis2"
    if is_trellis2 and level == "light":
        # TRELLIS.2 Windows fallback tends to produce sparse/open shells; run a lighter
        # Poisson pass for "light" to close surfaces without the full "standard" cost.
        use_poisson = True
        poisson_depth = 10
        poisson_points = 150000

    logger.info(f"Post-processing [{level}]: {len(mesh.faces):,} faces")
    upd(job_id, progress=86, message="Cleaning mesh...")

    # ÔöÇÔöÇ 1. Keep largest component ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    try:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            mesh = max(parts, key=lambda g: len(g.faces))
            logger.info(f"Keeping largest connected component: {len(mesh.faces):,} faces")
    except Exception:
        pass

    # ÔöÇÔöÇ 2. Basic repair ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # ÔöÇÔöÇ 3. Smoothing ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
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

            # Build Open3D mesh from the FULL mesh (no pre-decimation – uniform sampling
            # is O(n) and fast regardless of face count, so we keep all detail).
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices  = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.compute_vertex_normals()

            # Step 1: oversample uniformly using INTERPOLATED VERTEX normals (not face normals).
            # Vertex normals are area-weighted averages of adjacent face normals → much smoother
            # than raw per-face normals. This prevents the "small bumps" artifact where Poisson
            # reconstructs each noisy face normal as a physical bump on the surface.
            oversample_n = poisson_points * 5   # 5x is enough; 10x was overkill and slow
            pcd = o3d_mesh.sample_points_uniformly(
                number_of_points=oversample_n,
                use_triangle_normal=False,   # False = interpolate smooth vertex normals
            )

            # Step 2: voxel_down_sample → evenly distributed points (mimics poisson_disk spacing).
            # Complex meshes often collapse to far fewer points than requested with a naive
            # surface-area estimate, so retry with smaller voxels until we are near target.
            surface_area = max(float(o3d_mesh.get_surface_area()), 1e-8)
            voxel_size = float(_np.sqrt(surface_area / poisson_points))
            sampled_pcd = pcd
            pcd = sampled_pcd.voxel_down_sample(voxel_size=voxel_size)
            retry_count = 0
            while len(pcd.points) < int(poisson_points * 0.7) and retry_count < 4:
                voxel_size *= 0.8
                pcd = sampled_pcd.voxel_down_sample(voxel_size=voxel_size)
                retry_count += 1
            logger.info(
                f"Poisson input: {len(pcd.points):,} pts (target {poisson_points:,}), depth={poisson_depth}, voxel_retry={retry_count}"
            )

            # Step 3: re-orient normals consistently after voxel averaging.
            # Keep interpolated mesh normals as-is; re-orienting toward a camera point can flip
            # valid normals on concave or wrapped geometry. Only estimate/orient when normals are missing.
            if not pcd.has_normals():
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 3, max_nn=50))
                pcd.orient_normals_consistent_tangent_plane(24)

            poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth, width=0, scale=1.05, linear_fit=True
            )

            # Trim only the absolute lowest-density outlier "outer envelope" fragments.
            # Poisson reconstruction is inherently watertight BEFORE this trim; keeping
            # the trim percentage near zero preserves that guarantee.
            density_trim_pct = 0.0 if is_trellis2 else {"none": 0.1, "light": 0.25, "standard": 0.25, "heavy": 0.5}.get(level, 0.25)
            dens = _np.asarray(densities)
            if density_trim_pct > 0:
                thresh = _np.percentile(dens, density_trim_pct)
                poisson_mesh.remove_vertices_by_mask(dens < thresh)
            # Clean up any non-manifold edges before converting.
            poisson_mesh.remove_non_manifold_edges()
            poisson_mesh.remove_degenerate_triangles()
            poisson_mesh.remove_duplicated_triangles()
            poisson_mesh.remove_duplicated_vertices()
            poisson_mesh.compute_vertex_normals()

            v = _np.asarray(poisson_mesh.vertices)
            f = _np.asarray(poisson_mesh.triangles)
            if len(v) > 100 and len(f) > 100:
                candidate = trimesh.Trimesh(vertices=v, faces=f, process=True)
                if len(candidate.faces) > 100:
                    mesh = candidate
                    logger.info(f"Poisson OK: {len(mesh.faces):,} faces, watertight: {mesh.is_watertight}")

                    # Post-Poisson smoothing: removes jagged trim-boundary artifacts and
                    # gives the final mesh a clean, smooth surface ready for printing.
                    if post_smooth_iters > 0:
                        upd(job_id, progress=93, message="Smoothing reconstructed surface...")
                        try:
                            mesh = trimesh.smoothing.filter_taubin(mesh, iterations=post_smooth_iters)
                        except Exception:
                            pass

        except Exception as e:
            logger.warning(f"Poisson failed: {e} - continuing with trimesh repair")
    else:
        upd(job_id, progress=91, message="Skipping Poisson for fast preset...")

    # ── 5. Hole filling – multiple passes ──────────────────────────────────────
    upd(job_id, progress=94, message="Filling holes...")
    for _ in range(10):
        if mesh.is_watertight:
            break
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

    # MeshFix can sometimes over-close / inflate TRELLIS.2 geometry into a large blob.
    # Keep it opt-in for manual experiments instead of part of the default path.
    if is_trellis2 and not mesh.is_watertight and os.environ.get("PIXFORM_TRELLIS2_USE_MESHFIX", "0") == "1":
        upd(job_id, progress=95, message="Repairing manifold (MeshFix)...")
        mesh = _try_meshfix_watertight(mesh)
        logger.info(f"MeshFix result: {len(mesh.faces):,} faces, watertight: {mesh.is_watertight}")

    # ── 6. Voxel remesh fallback – guaranteed closed if still open ─────────────
    if not mesh.is_watertight:
        upd(job_id, progress=96, message="Voxel remesh (closing remaining holes)...")
        try:
            voxel_divisions = 160 if is_trellis2 else 128
            pitch    = max(float(mesh.extents.max()) / voxel_divisions, 1e-6)
            vox      = mesh.voxelized(pitch=pitch)
            vox.fill()
            remeshed = vox.marching_cubes
            if len(remeshed.faces) > 100:
                # Smooth BEFORE accepting – Taubin can open micro-holes, so smooth first,
                # then fill any opened holes, then verify.
                remesh_smooth_iters = 3 if is_trellis2 else 5
                smoothed = trimesh.smoothing.filter_taubin(remeshed, iterations=remesh_smooth_iters)
                for _ in range(5):
                    if smoothed.is_watertight:
                        break
                    trimesh.repair.fill_holes(smoothed)
                    trimesh.repair.fix_normals(smoothed)
                mesh = smoothed if smoothed.is_watertight else remeshed
                logger.info(f"Voxel remesh: {len(mesh.faces):,} faces, watertight: {mesh.is_watertight}")
        except Exception as e:
            logger.warning(f"Voxel remesh failed: {e}")

    # ── 7. Absolute watertight guarantee ──────────────────────────────────────
    # If still not watertight after everything above, do a final bare voxel remesh
    # at low resolution with NO smoothing. Marching cubes is always closed by definition.
    if not mesh.is_watertight:
        upd(job_id, progress=98, message="Enforcing watertight (final pass)...")
        try:
            pitch_final = max(float(mesh.extents.max()) / 96, 1e-6)
            vox_final = mesh.voxelized(pitch=pitch_final)
            vox_final.fill()
            mesh_final  = vox_final.marching_cubes
            if mesh_final.is_watertight and len(mesh_final.faces) > 100:
                logger.info(f"Watertight enforced via low-res voxel: {len(mesh_final.faces):,} faces")
                mesh = mesh_final
            else:
                logger.warning("Could not enforce watertight; delivering best-effort mesh")
        except Exception as e:
            logger.warning(f"Final watertight pass failed: {e}")

    # ÔöÇÔöÇ 7. Final cleanup ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    # ÔöÇÔöÇ 8. Match final proportions to the input image silhouette ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    mesh = _apply_input_aspect_correction(mesh, target_profile)

    # ÔöÇÔöÇ 9. Scale to 100mm ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
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

    # ÔöÇÔöÇ Fallback: PIL placeholder ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
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
        upd(job_id, status="processing", progress=5, message="Loading image...", stage="load_image")

        img = Image.open(image_path).convert("RGBA")
        _assert_not_cancelled(job_id)

        # Background removal
        if settings.get("remove_bg", True):
            upd(job_id, progress=10, message="Removing background...", stage="remove_background")
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(None, remove_background, img, job_id)
            _assert_not_cancelled(job_id)

        input_profile = _foreground_profile(img)

        # Preprocessing
        upd(job_id, progress=22, message="Preprocessing image...", stage="preprocess")
        img = resize_foreground(img, 0.85)
        img_np = np.array(img, dtype=np.float32) / 255.0
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            a = img_np[:, :, 3:4]
            img_np = img_np[:, :, :3] * a + 0.5 * (1.0 - a)
        img = Image.fromarray((img_np * 255.0).astype(np.uint8))
        from PIL import ImageEnhance
        img = ImageEnhance.Contrast(img).enhance(1.35)
        img = ImageEnhance.Sharpness(img).enhance(1.4)
        img = img.resize((512, 512), Image.LANCZOS)

        # Inference
        upd(job_id, progress=28, message="Generating 3D structure (TripoSR)...", stage="inference")
        loop = asyncio.get_event_loop()

        runtime_device = models.get("runtime_device", "cpu")

        def infer():
            with torch.no_grad():
                return models["triposr"]([img], device=runtime_device)

        scene_codes = await loop.run_in_executor(None, infer)
        _assert_not_cancelled(job_id)

        # Mesh extraction — try highest resolution that fits in VRAM
        target_res = settings.get("resolution", 512)
        fallbacks = [r for r in TRIPOSR_RES_LEVELS if r <= target_res]
        if not fallbacks:
            fallbacks = [128]

        def extract():
            for res in fallbacks:
                try:
                    upd(job_id, progress=35, message=f"Extracting mesh at resolution {res}...", stage="mesh_extract")
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

        upd(job_id, progress=75, message=f"Mesh extracted at {used_res}", stage="mesh_ready")

        mesh = to_trimesh(raw)
        if len(mesh.faces) == 0:
            raise RuntimeError("TripoSR produced an empty mesh")
        post_level = settings.get("post", "standard")
        upd(job_id, progress=80, message=f"Post-processing [{post_level}]...", stage="postprocess")
        mesh = await loop.run_in_executor(None, lambda: postprocess_mesh(mesh, job_id, post_level, input_profile))
        _assert_not_cancelled(job_id)

        upd(job_id, progress=95, message="Exporting files...", stage="export")
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
            stage="done",
        )

    except Exception as e:
        if str(e) == "Job cancelled by user":
            upd(job_id, status="cancelled", message="Cancelled by user")
            return
        logger.error(f"TripoSR error: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))
    finally:
        _finalize_job_resources(job_id)


# ÔöÇÔöÇ Hunyuan3D-2 pipeline ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

async def run_hunyuan(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    try:
        import torch
        from PIL import Image

        t_start = time.time()
        upd(job_id, status="processing", progress=5, message="Loading image...", stage="load_image")

        img = Image.open(image_path).convert("RGBA")
        _assert_not_cancelled(job_id)

        # Background removal
        if settings.get("remove_bg", True):
            upd(job_id, progress=10, message="Removing background...", stage="remove_background")
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

        upd(job_id, progress=22, message="Generating 3D shape with Hunyuan3D-2...", stage="preprocess")
        upd(job_id, progress=25, message="This takes 2-5 minutes, please wait...", stage="inference")

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

        upd(job_id, progress=30, message="Running Hunyuan3D-2 diffusion model...", stage="inference")
        raw_mesh = await loop.run_in_executor(None, infer)
        _assert_not_cancelled(job_id)

        upd(job_id, progress=78, message="Mesh generated", stage="mesh_ready")

        # Post-processing
        mesh = to_trimesh(raw_mesh)
        if len(mesh.faces) == 0:
            raise RuntimeError("Hunyuan3D-2 produced an empty mesh")
        post_level = settings.get("post", "standard")
        upd(job_id, progress=80, message=f"Post-processing [{post_level}]...", stage="postprocess")
        post_default = {"none": 300, "light": 600, "standard": 900, "heavy": 1800}.get(post_level, 900)
        post_timeout = _resolve_timeout_seconds("PIXFORM_POST_TIMEOUT_SEC", post_default)
        mesh_for_fallback = mesh.copy()
        try:
            post_task = loop.run_in_executor(None, lambda: postprocess_mesh(mesh, job_id, post_level, input_profile))
            post_started = time.time()
            while not post_task.done():
                await asyncio.sleep(8)
                _assert_not_cancelled(job_id)
                elapsed_post = int(time.time() - post_started)
                if post_timeout and elapsed_post >= post_timeout:
                    raise asyncio.TimeoutError
                job_st = jobs.get(job_id, {})
                stale_for = time.time() - float(job_st.get("last_update_ts") or post_started)
                if stale_for >= 16:
                    heartbeat_progress = min(93, 80 + elapsed_post // 30)
                    upd(job_id, progress=int(heartbeat_progress), message=f"Post-processing [{post_level}]... {elapsed_post}s elapsed", stage="postprocess")
            mesh = await post_task
        except asyncio.TimeoutError:
            logger.warning(f"Hunyuan3D-2 postprocess timed out after {post_timeout}s; using fast fallback")
            upd(job_id, progress=88, message="Postprocess timeout, using fast finalize...", stage="postprocess_fallback")
            mesh = _fast_finalize_mesh(mesh_for_fallback, input_profile)
        _assert_not_cancelled(job_id)

        # Export
        upd(job_id, progress=95, message="Exporting files...", stage="export")
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
            stage="done",
        )

    except Exception as e:
        if str(e) == "Job cancelled by user":
            upd(job_id, status="cancelled", message="Cancelled by user")
            return
        logger.error(f"Hunyuan3D-2 error: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))
    finally:
        _finalize_job_resources(job_id)


# ÔöÇÔöÇ TRELLIS pipeline ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

async def run_trellis(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    try:
        import torch
        from PIL import Image

        t_start = time.time()
        upd(job_id, status="processing", progress=5, message="Loading image...", stage="load_image")

        img = Image.open(image_path).convert("RGBA")
        _assert_not_cancelled(job_id)

        # Background removal (our rembg session; TRELLIS will skip its own if RGBA alpha is set)
        if settings.get("remove_bg", True):
            upd(job_id, progress=10, message="Removing background...", stage="remove_background")
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(None, remove_background, img, job_id)
            _assert_not_cancelled(job_id)

        input_profile = _foreground_profile(img)

        upd(job_id, progress=22, message="Generating 3D shape with TRELLIS...", stage="preprocess")
        upd(job_id, progress=25, message="This takes 5 to 10 minutes, please wait...", stage="inference")

        loop = asyncio.get_event_loop()
        steps = settings.get("steps", 50)
        ss_steps   = max(12, min(steps, steps // 3))
        slat_steps = max(12, steps)

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

        upd(job_id, progress=30, message="Running TRELLIS diffusion model...", stage="inference")
        infer_started = time.time()
        infer_future = loop.run_in_executor(None, infer)
        # TRELLIS inference runs in a blocking executor call; emit a heartbeat so UI users can see it's alive.
        while not infer_future.done():
            await asyncio.sleep(8)
            elapsed_inf = int(time.time() - infer_started)
            if _cancel_requested(job_id):
                upd(
                    job_id,
                    progress=30,
                    message=f"Cancel requested; waiting for a safe stop... ({elapsed_inf}s elapsed)",
                    stage="inference",
                )
            else:
                # Slow ramp during inference to avoid a frozen-looking status page on long runs.
                heartbeat_progress = min(75, 30 + elapsed_inf // 20)
                upd(
                    job_id,
                    progress=int(heartbeat_progress),
                    message=f"Running TRELLIS diffusion model... {elapsed_inf}s elapsed",
                    stage="inference",
                )
        outputs = await infer_future
        _assert_not_cancelled(job_id)

        upd(job_id, progress=78, message="Diffusion complete, decoding 3D outputs...", stage="decode")

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
            upd(job_id, progress=79, message="Generating textured GLB (optional)...", stage="texture")
            try:
                tex_task = loop.run_in_executor(None, make_textured_glb)
                glb_textured = await asyncio.wait_for(tex_task, timeout=texture_timeout) if texture_timeout else await tex_task
            except asyncio.TimeoutError:
                logger.warning("TRELLIS textured GLB stage timed out; falling back to plain GLB")
                glb_textured = False
        else:
            logger.info("TRELLIS textured GLB disabled by default (set PIXFORM_TRELLIS_TEXTURED=1 to enable)")
        _assert_not_cancelled(job_id)

        upd(job_id, progress=79, message="Converting decoded mesh...", stage="extract_mesh")

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
        upd(job_id, progress=80, message=f"Post-processing [{post_level}]...", stage="postprocess")
        post_default = {"none": 600, "light": 900, "standard": 1800, "heavy": 2700}.get(post_level, 1800)
        post_timeout = _resolve_timeout_seconds("PIXFORM_TRELLIS_POST_TIMEOUT_SEC", post_default)
        try:
            post_task = loop.run_in_executor(None, lambda: postprocess_mesh(raw_mesh, job_id, post_level, input_profile))
            post_started = time.time()
            while not post_task.done():
                await asyncio.sleep(8)
                _assert_not_cancelled(job_id)
                elapsed_post = int(time.time() - post_started)
                if post_timeout and elapsed_post >= post_timeout:
                    raise asyncio.TimeoutError

                # Keep status alive if postprocess internals have not updated recently.
                job = jobs.get(job_id, {})
                last_update_ts = float(job.get("last_update_ts") or post_started)
                stale_for = time.time() - last_update_ts
                if stale_for >= 16:
                    heartbeat_progress = min(93, 80 + elapsed_post // 30)
                    upd(
                        job_id,
                        progress=int(heartbeat_progress),
                        message=f"Post-processing [{post_level}]... {elapsed_post}s elapsed",
                        stage="postprocess",
                    )

            mesh = await post_task
        except asyncio.TimeoutError:
            logger.warning(f"TRELLIS postprocess timed out after {post_timeout}s; using fast fallback mesh finalize")
            upd(job_id, progress=88, message="Postprocess timeout, using fast finalize...", stage="postprocess_fallback")
            mesh = _fast_finalize_mesh(raw_mesh.copy(), input_profile)
        _assert_not_cancelled(job_id)

        upd(job_id, progress=95, message="Exporting files...", stage="export")

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

        upd(job_id, progress=97, message="Rendering preview image...", stage="preview")
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
            stage="done",
        )

    except Exception as e:
        if str(e) == "Job cancelled by user":
            upd(job_id, status="cancelled", message="Cancelled by user")
            return
        logger.error(f"TRELLIS error: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))
    finally:
        _finalize_job_resources(job_id)


# ── TRELLIS.2 pipeline ───────────────────────────────────────────────────────

async def run_trellis2(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    try:
        import torch
        from PIL import Image

        t_start = time.time()
        upd(job_id, status="processing", progress=5, message="Loading image...", stage="load_image")

        img = Image.open(image_path).convert("RGBA")
        _assert_not_cancelled(job_id)

        # TRELLIS.2 works best with a clean RGBA foreground; preserve existing alpha instead of re-matting it.
        input_has_alpha_mask = _image_has_meaningful_alpha(img)
        if settings.get("remove_bg", True):
            if input_has_alpha_mask:
                upd(job_id, progress=10, message="Using embedded alpha mask...", stage="remove_background")
            else:
                upd(job_id, progress=10, message="Removing background...", stage="remove_background")
                loop = asyncio.get_event_loop()
                img = await loop.run_in_executor(None, remove_background, img, job_id)
                _assert_not_cancelled(job_id)

        input_profile = _foreground_profile(img)

        upd(job_id, progress=22, message="Generating 3D shape with TRELLIS.2...", stage="preprocess")
        upd(job_id, progress=25, message="This takes 6 to 12 minutes, please wait...", stage="inference")

        loop = asyncio.get_event_loop()
        t2_params = _build_trellis2_sampler_params(int(settings.get("steps", 36)))
        sparse_steps, shape_steps, tex_steps = t2_params.pop("_steps_tuple")
        req_resolution = _normalize_trellis2_resolution(int(settings.get("resolution", 1024)))
        pipeline_type = _resolve_trellis2_pipeline_type(req_resolution)
        seed = _resolve_trellis2_seed(settings)
        logger.info(
            "TRELLIS.2 seed=%s sampler profile=%s steps(sparse/shape/tex)=%s/%s/%s resolution=%s pipeline=%s alpha_mask=%s",
            seed,
            _resolve_trellis2_sampler_profile(),
            sparse_steps,
            shape_steps,
            tex_steps,
            req_resolution,
            pipeline_type,
            input_has_alpha_mask,
        )

        # Match official TRELLIS.2 behavior: preprocess once, then run with preprocess_image=False.
        upd(job_id, progress=28, message="Preparing TRELLIS.2 conditioning image...", stage="preprocess")
        img = await loop.run_in_executor(None, lambda: models["trellis2"].preprocess_image(img))
        _assert_not_cancelled(job_id)

        def infer():
            with torch.no_grad():
                return models["trellis2"].run(
                    img,
                    seed=seed,
                    sparse_structure_sampler_params=t2_params["sparse_structure_sampler_params"],
                    shape_slat_sampler_params=t2_params["shape_slat_sampler_params"],
                    tex_slat_sampler_params=t2_params["tex_slat_sampler_params"],
                    pipeline_type=pipeline_type,
                    preprocess_image=False,
                )

        wait_seconds = 0
        while trellis2_infer_lock.locked():
            _assert_not_cancelled(job_id)
            wait_seconds += 2
            upd(
                job_id,
                progress=30,
                message=f"Another TRELLIS.2 job is running; queued ({wait_seconds}s)...",
                stage="queue",
            )
            await asyncio.sleep(2)

        async with trellis2_infer_lock:
            _assert_not_cancelled(job_id)
            upd(job_id, progress=30, message="Running TRELLIS.2 diffusion model...", stage="inference")
            outputs = await loop.run_in_executor(None, infer)
        _assert_not_cancelled(job_id)

        upd(job_id, progress=78, message="3D structure generated", stage="mesh_ready")

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
        upd(job_id, progress=79, message="Extracting mesh from TRELLIS.2 output...", stage="extract_mesh")

        def extract_trimesh():
            import trimesh as _trimesh
            import numpy as _np
            # process=False: skip slow auto-repair/dedup on raw dense mesh;
            # postprocess_mesh handles cleanup later.
            verts = m.vertices.cpu().numpy() if hasattr(m.vertices, 'cpu') else _np.array(m.vertices)
            faces = m.faces.cpu().numpy() if hasattr(m.faces, 'cpu') else _np.array(m.faces)
            return _trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        raw_mesh = await loop.run_in_executor(None, extract_trimesh)
        _assert_not_cancelled(job_id)
        logger.info("TRELLIS.2 raw mesh: %s faces", len(raw_mesh.faces))

        if len(raw_mesh.faces) == 0:
            raise RuntimeError("TRELLIS.2 produced an empty mesh")

        # ── Pre-decimate TRELLIS.2 raw mesh before post-processing ───────────
        # Target scales with post level: less post = fewer faces needed.
        _post_level_now = settings.get("post", "standard")
        # Pre-decim target: Poisson reconstruction works from point samples, not faces.
        # Keeping 1.5M+ faces is wasteful and makes Taubin smoothing + sampling very slow.
        # 400-600k is more than enough detail for Poisson to reconstruct at depth 11-12.
        _predecim_defaults = {"none": 100_000, "light": 200_000, "standard": 400_000, "heavy": 600_000}
        _TRELLIS2_PREDECIM_TARGET = int(os.environ.get(
            "PIXFORM_TRELLIS2_PREDECIM_FACES",
            str(_predecim_defaults.get(_post_level_now, 1_000_000))
        ))
        if len(raw_mesh.faces) > _TRELLIS2_PREDECIM_TARGET:
            upd(job_id, progress=80, message=f"Decimating {len(raw_mesh.faces):,} → {_TRELLIS2_PREDECIM_TARGET:,} faces...", stage="predecimate")
            logger.info("TRELLIS.2 pre-decimating %s → %s faces (post=%s)...", len(raw_mesh.faces), _TRELLIS2_PREDECIM_TARGET, _post_level_now)
            def _predecimate():
                try:
                    import open3d as o3d
                    import numpy as _np
                    o3d_mesh = o3d.geometry.TriangleMesh()
                    o3d_mesh.vertices = o3d.utility.Vector3dVector(_np.array(raw_mesh.vertices, dtype=_np.float64))
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(_np.array(raw_mesh.faces, dtype=_np.int32))
                    o3d_mesh.remove_degenerate_triangles()
                    o3d_mesh.remove_duplicated_triangles()
                    o3d_mesh.remove_duplicated_vertices()
                    o3d_mesh.remove_non_manifold_edges()
                    decimated_o3d = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=_TRELLIS2_PREDECIM_TARGET)
                    if len(decimated_o3d.triangles) > int(_TRELLIS2_PREDECIM_TARGET * 1.35):
                        decimated_o3d = decimated_o3d.simplify_quadric_decimation(target_number_of_triangles=_TRELLIS2_PREDECIM_TARGET)
                    import trimesh as _trimesh
                    result = _trimesh.Trimesh(
                        vertices=_np.asarray(decimated_o3d.vertices),
                        faces=_np.asarray(decimated_o3d.triangles),
                        process=False,
                    )
                    try:
                        parts = result.split(only_watertight=False)
                        if len(parts) > 1:
                            result = max(parts, key=lambda g: len(g.faces))
                    except Exception:
                        pass
                    return result if len(result.faces) > 0 else raw_mesh
                except Exception as e:
                    logger.warning(f"TRELLIS.2 pre-decimation failed ({e}), using raw mesh")
                    return raw_mesh
            raw_mesh = await loop.run_in_executor(None, _predecimate)
            _assert_not_cancelled(job_id)
            logger.info("TRELLIS.2 pre-decimated to %s faces", len(raw_mesh.faces))


        # ── Post-process TRELLIS.2 mesh via the shared pipeline ──────────────
        # This runs the full Poisson reconstruction + hole filling + voxel remesh
        # fallback, identical to what TripoSR and Hunyuan use.
        post_level = settings.get("post", "standard")
        upd(job_id, progress=85, message="Post-processing mesh (Poisson + repair)...")
        post_default = {"none": 300, "light": 600, "standard": 1200, "heavy": 2400}.get(post_level, 1200)
        post_timeout = _resolve_timeout_seconds("PIXFORM_POST_TIMEOUT_SEC", post_default)
        raw_mesh_copy = raw_mesh.copy()
        try:
            post_task = loop.run_in_executor(
                None,
                lambda: postprocess_mesh(raw_mesh, job_id, post_level, input_profile, model_key="trellis2"),
            )
            post_started = time.time()
            while not post_task.done():
                await asyncio.sleep(8)
                _assert_not_cancelled(job_id)
                elapsed_post = int(time.time() - post_started)
                if post_timeout and elapsed_post >= post_timeout:
                    raise asyncio.TimeoutError
                job_st = jobs.get(job_id, {})
                stale_for = time.time() - float(job_st.get("last_update_ts") or post_started)
                if stale_for >= 16:
                    heartbeat_progress = min(93, 85 + elapsed_post // 30)
                    upd(job_id, progress=int(heartbeat_progress), message=f"Post-processing [{post_level}]... {elapsed_post}s elapsed", stage="postprocess")
            mesh = await post_task
        except asyncio.TimeoutError:
            logger.warning(f"TRELLIS.2 postprocess timed out after {post_timeout}s; using fast fallback")
            upd(job_id, progress=88, message="Postprocess timeout, using fast finalize...", stage="postprocess_fallback")
            mesh = _fast_finalize_mesh(raw_mesh_copy, input_profile)
        _assert_not_cancelled(job_id)

        logger.info(f"TRELLIS.2 post-processing complete: {len(mesh.faces):,} faces | watertight: {mesh.is_watertight}")

        # ── Export ─────────────────────────────────────────────────────────────
        upd(job_id, progress=95, message="Exporting files...", stage="export")

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
            stage="done",
        )

    except Exception as e:
        if str(e) == "Job cancelled by user":
            upd(job_id, status="cancelled", message="Cancelled by user")
            return
        logger.error(f"TRELLIS.2 error: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))
    finally:
        _finalize_job_resources(job_id)


def _decode_data_url_or_b64(value: str) -> bytes:
    payload = value.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def _write_optional_b64_file(payload: Optional[str], out_path: Path) -> bool:
    if not payload or not isinstance(payload, str):
        return False
    try:
        out_path.write_bytes(_decode_data_url_or_b64(payload))
        return True
    except Exception:
        return False


def _write_optional_url_file(file_url: Optional[str], out_path: Path, auth_token: Optional[str]) -> bool:
    if not file_url or not isinstance(file_url, str):
        return False
    try:
        headers = {"Accept": "application/octet-stream"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        req = urlrequest.Request(file_url, headers=headers, method="GET")
        with urlrequest.urlopen(req, timeout=45) as resp:
            out_path.write_bytes(resp.read())
        return True
    except Exception:
        return False


def _extract_endpoint_outputs_to_dir(endpoint_payload: dict, out_dir: Path, auth_token: Optional[str]) -> dict:
    blob = endpoint_payload if isinstance(endpoint_payload, dict) else {}
    nested = blob.get("outputs") if isinstance(blob.get("outputs"), dict) else {}

    def _pick(*keys):
        for k in keys:
            if k in blob:
                return blob.get(k)
            if k in nested:
                return nested.get(k)
        return None

    _write_optional_b64_file(_pick("glb_base64", "model_glb_base64"), out_dir / "model.glb")
    _write_optional_b64_file(_pick("obj_base64", "model_obj_base64"), out_dir / "model.obj")
    _write_optional_b64_file(_pick("stl_base64", "model_stl_base64"), out_dir / "model.stl")
    _write_optional_b64_file(_pick("tmf_base64", "3mf_base64", "model_3mf_base64"), out_dir / "model.3mf")
    _write_optional_b64_file(_pick("preview_base64", "preview_png_base64"), out_dir / "preview.png")

    _write_optional_url_file(_pick("glb_url", "model_glb_url"), out_dir / "model.glb", auth_token)
    _write_optional_url_file(_pick("obj_url", "model_obj_url"), out_dir / "model.obj", auth_token)
    _write_optional_url_file(_pick("stl_url", "model_stl_url"), out_dir / "model.stl", auth_token)
    _write_optional_url_file(_pick("tmf_url", "3mf_url", "model_3mf_url"), out_dir / "model.3mf", auth_token)
    _write_optional_url_file(_pick("preview_url", "preview_png_url"), out_dir / "preview.png", auth_token)

    return {
        "glb_url": "model.glb" if (out_dir / "model.glb").exists() else None,
        "obj_url": "model.obj" if (out_dir / "model.obj").exists() else None,
        "stl_url": "model.stl" if (out_dir / "model.stl").exists() else None,
        "tmf_url": "model.3mf" if (out_dir / "model.3mf").exists() else None,
        "preview_url": "preview.png" if (out_dir / "preview.png").exists() else None,
        "poly_count": _pick("poly_count"),
    }


def _post_trellis2_endpoint_sync(image_path: Path, settings: dict, out_dir: Path) -> dict:
    endpoint_url = _resolve_trellis2_endpoint_url()
    if not endpoint_url:
        return {"ok": False, "reason": "endpoint disabled"}

    hf_token = _resolve_hf_token()
    if not hf_token:
        return {"ok": False, "reason": "HF token missing"}

    request_body = {
        "inputs": {
            "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        },
        "parameters": {
            "remove_bg": bool(settings.get("remove_bg", True)),
            "resolution": int(settings.get("resolution", 1024)),
            "steps": int(settings.get("steps", 36)),
            "post": str(settings.get("post", "standard")),
            "preserve_proportions": bool(settings.get("preserve_proportions", True)),
        },
        "options": {
            "wait_for_model": True,
        },
    }

    timeout_sec = _resolve_trellis2_endpoint_timeout_sec()
    endpoint_log_name = _redacted_endpoint_for_log(endpoint_url)
    img_size = 0
    try:
        img_size = image_path.stat().st_size
    except Exception:
        pass
    logger.info(
        "TRELLIS.2 endpoint request: endpoint=%s timeout=%ss image_bytes=%s steps=%s res=%s post=%s",
        endpoint_log_name,
        timeout_sec,
        img_size,
        settings.get("steps"),
        settings.get("resolution"),
        settings.get("post"),
    )

    started = time.monotonic()
    poll_round = 0
    while True:
        poll_round += 1
        logger.info("TRELLIS.2 endpoint poll #%s -> %s", poll_round, endpoint_log_name)
        req = urlrequest.Request(
            endpoint_url,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json",
                "Accept": "application/json, application/octet-stream, model/gltf-binary",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=min(60, timeout_sec)) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                raw = resp.read()
                logger.info(
                    "TRELLIS.2 endpoint response: status=%s content_type=%s bytes=%s",
                    getattr(resp, "status", "?"),
                    ctype or "unknown",
                    len(raw),
                )
        except urlerror.HTTPError as e:
            try:
                err_raw = e.read().decode("utf-8", errors="ignore")
                err_json = json.loads(err_raw) if err_raw else {}
            except Exception:
                err_json = {}
                err_raw = str(e)

            if e.code in {503, 504}:
                est = err_json.get("estimated_time") if isinstance(err_json, dict) else None
                sleep_s = int(est) if isinstance(est, (int, float)) else 5
                logger.warning(
                    "TRELLIS.2 endpoint transient HTTP %s on poll #%s (estimated_time=%s)",
                    e.code,
                    poll_round,
                    est,
                )
                if time.monotonic() - started + sleep_s >= timeout_sec:
                    return {"ok": False, "reason": f"endpoint timeout after {timeout_sec}s"}
                time.sleep(max(2, min(sleep_s, 20)))
                continue

            return {"ok": False, "reason": f"http {e.code}: {err_raw}"}
        except Exception as e:
            return {"ok": False, "reason": f"endpoint request failed: {e}"}

        if "application/octet-stream" in ctype or "model/gltf-binary" in ctype:
            (out_dir / "model.glb").write_bytes(raw)
            logger.info("TRELLIS.2 endpoint returned binary GLB payload")
            return {
                "ok": True,
                "glb_url": "model.glb",
                "obj_url": None,
                "stl_url": None,
                "tmf_url": None,
                "preview_url": None,
                "poly_count": None,
            }

        try:
            payload = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            return {"ok": False, "reason": "endpoint returned unsupported response"}

        if isinstance(payload, dict) and payload.get("error"):
            err_text = str(payload.get("error"))
            est = payload.get("estimated_time")
            logger.warning(
                "TRELLIS.2 endpoint JSON error on poll #%s: %s (estimated_time=%s)",
                poll_round,
                err_text,
                est,
            )
            if est is not None and time.monotonic() - started < timeout_sec:
                sleep_s = max(2, min(int(est), 20)) if isinstance(est, (int, float)) else 5
                if time.monotonic() - started + sleep_s >= timeout_sec:
                    return {"ok": False, "reason": f"endpoint timeout after {timeout_sec}s"}
                time.sleep(sleep_s)
                continue
            return {"ok": False, "reason": err_text}

        files = _extract_endpoint_outputs_to_dir(payload if isinstance(payload, dict) else {}, out_dir, hf_token)
        has_mesh = any(files.get(k) for k in ("glb_url", "obj_url", "stl_url", "tmf_url"))
        logger.info(
            "TRELLIS.2 endpoint artifacts: glb=%s obj=%s stl=%s 3mf=%s preview=%s",
            bool(files.get("glb_url")),
            bool(files.get("obj_url")),
            bool(files.get("stl_url")),
            bool(files.get("tmf_url")),
            bool(files.get("preview_url")),
        )
        if has_mesh:
            return {"ok": True, **files}

        if time.monotonic() - started >= timeout_sec:
            return {"ok": False, "reason": f"endpoint timeout after {timeout_sec}s"}
        if poll_round >= 2:
            return {"ok": False, "reason": "endpoint returned no mesh artifacts"}
        time.sleep(3)


async def run_trellis2_endpoint_first(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    try:
        endpoint_url = _resolve_trellis2_endpoint_url()
        upd(job_id, endpoint_attempted=False, endpoint_used=False, endpoint_fail_reason=None)
        if endpoint_url:
            upd(job_id, endpoint_attempted=True)
            upd(job_id, status="processing", progress=6, message="Trying TRELLIS.2 endpoint...", stage="endpoint")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: _post_trellis2_endpoint_sync(image_path, settings, out_dir))
            _assert_not_cancelled(job_id)
            if result.get("ok"):
                elapsed = round(time.time() - float(jobs.get(job_id, {}).get("created_at", time.time())), 1)
                upd(
                    job_id,
                    status="done",
                    progress=100,
                    message=f"Done in {elapsed}s (endpoint)",
                    model_used="TRELLIS.2 (endpoint)",
                    endpoint_used=True,
                    time_taken=elapsed,
                    stl_url=f"/outputs/{job_id}/{result['stl_url']}" if result.get("stl_url") else None,
                    tmf_url=f"/outputs/{job_id}/{result['tmf_url']}" if result.get("tmf_url") else None,
                    glb_url=f"/outputs/{job_id}/{result['glb_url']}" if result.get("glb_url") else None,
                    obj_url=f"/outputs/{job_id}/{result['obj_url']}" if result.get("obj_url") else None,
                    preview_url=f"/outputs/{job_id}/{result['preview_url']}" if result.get("preview_url") else None,
                    poly_count=result.get("poly_count") if isinstance(result.get("poly_count"), int) else None,
                    stage="done",
                )
                return
            fail_reason = str(result.get("reason", "unknown endpoint failure"))
            logger.warning("TRELLIS.2 endpoint failed, falling back to local: %s", fail_reason)
            upd(
                job_id,
                endpoint_fail_reason=fail_reason,
                progress=8,
                message=f"Endpoint failed ({fail_reason}), falling back to local...",
                stage="endpoint_fallback",
            )

        if models.get("trellis2") is None:
            upd(job_id, status="error", message="TRELLIS.2 local model unavailable and endpoint failed")
            return

        logger.info("TRELLIS.2 local fallback started for job %s", job_id)
        await run_trellis2(job_id, image_path, out_dir, settings)
    except Exception as e:
        if str(e) == "Job cancelled by user":
            upd(job_id, status="cancelled", message="Cancelled by user")
            return
        logger.error(f"TRELLIS.2 endpoint/local orchestration failed: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))
        _finalize_job_resources(job_id)


# ÔöÇÔöÇ Demo pipeline (no GPU) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

async def run_demo(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    import trimesh
    for p, m in [(20, "Analyzing..."), (50, "Building mesh..."), (80, "Exporting...")]:
        stage = "analyze" if p == 20 else ("build" if p == 50 else "export")
        upd(job_id, status="processing", progress=p, message=m, stage=stage)
        await asyncio.sleep(0.8)
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    export_all(mesh, out_dir, job_id)
    render_preview(mesh, out_dir / "preview.png")
    upd(job_id, status="done", progress=100,
        message="Demo mode - no GPU available",
        model_used="Demo",
        preview_url=f"/outputs/{job_id}/preview.png",
        stage="done")


# ÔöÇÔöÇ Routes ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
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
        "triposr_active": models.get("active_models", {}).get("triposr", False),
        "hunyuan_active": models.get("active_models", {}).get("hunyuan", False),
        "trellis_active": models.get("active_models", {}).get("trellis", False),
        "trellis2_active": models.get("active_models", {}).get("trellis2", False),
        "active_models": [name for name in MODEL_KEYS if models.get("active_models", {}).get(name, False)],
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
        "trellis2_endpoint_enabled": bool(_resolve_trellis2_endpoint_url()),
        "trellis2_endpoint_host": _redacted_endpoint_for_log(_resolve_trellis2_endpoint_url()) if _resolve_trellis2_endpoint_url() else None,
        "hf_token_present": bool(_resolve_hf_token()),
        "rembg_status": model_health["rembg"]["status"],
        "rembg_error": model_health["rembg"]["error"],
    }


@app.post("/convert", response_model=JobStatus)
async def convert(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model:      str  = "triposr",    # triposr | hunyuan | trellis | trellis2
    remove_bg:  str  = "true",
    resolution: int  = 512,
    steps:      int  = 50,           # hunyuan/trellis diffusion steps
    post:       str  = "standard",   # none | light | standard | heavy
    preserve_proportions: str = "auto",
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

    preserve_raw = (preserve_proportions or "auto").strip().lower()
    if preserve_raw in {"", "auto"}:
        preserve_flag = model == "trellis2"
    else:
        preserve_flag = preserve_raw == "true"

    normalized_resolution = _normalize_triposr_resolution(resolution)
    normalized_steps = max(5, min(1000, steps))
    if model == "trellis2":
        normalized_resolution = _normalize_trellis2_resolution(resolution)
        normalized_steps = _normalize_trellis2_steps(steps)

    settings = {
        "remove_bg":  remove_bg.lower() == "true",
        "resolution": normalized_resolution,
        "steps":      normalized_steps,
        "post":       post,
        "preserve_proportions": preserve_flag,
    }

    # Determine which model to use
    if model == "triposr":
        if models["triposr"] is None:
            _safe_unlink(img_path)
            raise HTTPException(503, _model_unavailable_message("triposr", "TripoSR"))
        run_fn = run_triposr
        model_label = "TripoSR"
    elif model == "hunyuan":
        if models["hunyuan"] is None:
            _safe_unlink(img_path)
            raise HTTPException(503, _model_unavailable_message("hunyuan", "Hunyuan3D-2"))
        run_fn = run_hunyuan
        model_label = "Hunyuan3D-2"
    elif model == "trellis":
        if models["trellis"] is None:
            _safe_unlink(img_path)
            raise HTTPException(503, _model_unavailable_message("trellis", "TRELLIS"))
        run_fn = run_trellis
        model_label = "TRELLIS"
    elif model == "trellis2":
        endpoint_enabled = bool(_resolve_trellis2_endpoint_url())
        if models["trellis2"] is None and not endpoint_enabled:
            _safe_unlink(img_path)
            raise HTTPException(503, _model_unavailable_message("trellis2", "TRELLIS.2"))
        run_fn = run_trellis2_endpoint_first if endpoint_enabled else run_trellis2
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
        delete_requested=False,
        stage="queued",
        last_update_ts=time.time(),
        created_at=time.time(),
        completed_at=None,
        cleanup_after_ts=None,
        upload_path=str(img_path),
        output_dir=str(OUTPUT_DIR / job_id),
        upload_deleted=False,
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
        upd(job_id, status="cancelling", message="Cancel requested, waiting for safe stop...", stage="cancelling")
    else:
        upd(job_id)

    return {"job_id": job_id, "status": job.get("status"), "cancel_requested": True}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id in jobs and not _job_terminal(jobs[job_id].get("status", "")):
        jobs[job_id]["cancel_requested"] = True
        jobs[job_id]["delete_requested"] = True
        upd(job_id, status="cancelling", message="Cancel requested, waiting for safe stop...", stage="cancelling")
        return {"job_id": job_id, "status": "cancelling", "deleted": False}

    job = jobs.pop(job_id, None)
    if job:
        _cleanup_job_artifacts(job, remove_upload=not job.get("upload_deleted"), remove_output=True)
    else:
        shutil.rmtree(OUTPUT_DIR / job_id, ignore_errors=True)
    return {"deleted": job_id}


@app.post("/jobs/cleanup")
async def cleanup_jobs(all: bool = False):
    result = _prune_old_jobs(force_terminal_cleanup=bool(all))
    return {"ok": True, "all": bool(all), **result}


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
