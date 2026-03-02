"""
PIXFORM Backend
Image to 3D pipeline — TripoSR (fast) + Hunyuan3D-2 (quality)
"""
import os, sys, uuid, shutil, asyncio, logging, zipfile, time
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

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# hy3dgen is path-based, not a package - add it explicitly
_hy3d_path = BASE_DIR / "hy3dgen"
if _hy3d_path.exists() and str(_hy3d_path) not in sys.path:
    sys.path.insert(0, str(_hy3d_path))

# ── Global model state ────────────────────────────────────────────────────────
models = {
    "triposr":   None,
    "hunyuan":   None,
    "rembg_sess": None,
}
jobs: dict = {}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_all_models():
    import torch

    # rembg session (RMBG-1.4 — best background removal quality)
    try:
        from rembg import new_session
        models["rembg_sess"] = new_session("isnet-general-use")
        logger.info("✅ rembg (ISNet) background remover loaded")
    except Exception as e:
        logger.warning(f"rembg failed: {e}")
        try:
            from rembg import new_session
            models["rembg_sess"] = new_session("u2net")
            logger.info("✅ rembg (u2net) background remover loaded")
        except Exception as e2:
            logger.warning(f"rembg fallback failed: {e2}")

    if not torch.cuda.is_available():
        logger.warning("⚠️  No CUDA GPU — running in CPU demo mode")
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    # TripoSR
    try:
        from tsr.system import TSR
        logger.info("Loading TripoSR model (~1 GB)...")
        m = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        m.renderer.set_chunk_size(131072)
        m.to("cuda")
        models["triposr"] = m
        logger.info("✅ TripoSR loaded")
    except Exception as e:
        logger.warning(f"TripoSR failed to load: {e}")

    # Hunyuan3D-2 shape
    try:
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
        logger.info("✅ Hunyuan3D-2 loaded")
    except Exception as e:
        logger.warning(f"Hunyuan3D-2 failed to load: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_all_models)
    yield


app = FastAPI(title="PIXFORM", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Data models ───────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id:      str
    status:      str            # queued | processing | done | error
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


# ── Utilities ─────────────────────────────────────────────────────────────────

def upd(job_id, **kw):
    if job_id in jobs:
        jobs[job_id].update(kw)


def remove_background(img, job_id):
    """Remove background using rembg. Returns RGBA PIL Image."""
    from PIL import Image
    sess = models.get("rembg_sess")
    if sess is None:
        return img.convert("RGBA")
    try:
        from rembg import remove as rembg_remove
        result = rembg_remove(img.convert("RGB"), session=sess)
        upd(job_id, progress=20, message="Background removed ✓")
        return result.convert("RGBA")
    except Exception as e:
        logger.warning(f"Background removal failed: {e}")
        return img.convert("RGBA")


def postprocess_mesh(mesh, job_id, level="standard"):
    """High quality mesh post-processing for 3D printing.
    level: none | light | standard | heavy
    """
    if level == "none":
        # Just scale, no processing
        bounds = mesh.bounds
        size = max(bounds[1] - bounds[0])
        if size > 0:
            mesh.apply_scale(100.0 / size)
        return mesh

    import trimesh
    import trimesh.smoothing

    # Smoothing iterations per level
    smooth_iters = {"light": 3, "standard": 15, "heavy": 30}.get(level, 15)
    poisson_depth = {"light": 8, "standard": 10, "heavy": 11}.get(level, 10)
    poisson_points = {"light": 50000, "standard": 100000, "heavy": 200000}.get(level, 100000)

    logger.info(f"Post-processing [{level}]: {len(mesh.faces):,} faces")
    upd(job_id, progress=88, message=f"Cleaning mesh [{level}]...")

    # 1. Keep only largest component
    components = mesh.split(only_watertight=False)
    if components:
        mesh = max(components, key=lambda c: len(c.faces))

    # 2. Fix mesh issues
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)

    upd(job_id, progress=90, message="Smoothing mesh...")

    # 3. Taubin smoothing — best for 3D printing (preserves volume)
    try:
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
    except Exception as e:
        logger.warning(f"Taubin smoothing failed: {e}")
        try:
            mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters//2, lamb=0.2)
        except Exception:
            pass

    # 4. Open3D Poisson reconstruction for watertight, print-ready mesh
    if level != "light":
        upd(job_id, progress=92, message="Poisson reconstruction (print-ready)...")
        try:
            import open3d as o3d
            import numpy as _np

            # Convert trimesh -> open3d
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices  = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.compute_vertex_normals()

            # Convert to point cloud for Poisson
            pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=poisson_points)
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(30)

            # Poisson surface reconstruction
            poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
            )

            # Remove low-density vertices - use conservative threshold to avoid holes
            dens = _np.asarray(densities)
            thresh = _np.percentile(dens, 2)  # was 5 - lower = fewer holes
            poisson_mesh.remove_vertices_by_mask(_np.asarray(densities) < thresh)
            poisson_mesh.compute_vertex_normals()

            v = _np.asarray(poisson_mesh.vertices)
            f = _np.asarray(poisson_mesh.triangles)
            if len(v) > 0 and len(f) > 0:
                mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
                logger.info(f"Poisson reconstruction: {len(mesh.faces):,} faces")
            else:
                logger.warning("Poisson produced empty mesh, keeping smoothed version")

        except Exception as e:
            logger.warning(f"Open3D Poisson failed: {e}, using smoothed mesh")

    # 5. Final cleanup
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    trimesh.repair.fill_holes(mesh)

    # 6. Aggressive hole filling passes
    for _ in range(3):
        trimesh.repair.fill_holes(mesh)
        if mesh.is_watertight:
            break

    # 7. If still not watertight, try convex hull as last resort for heavy mode
    if not mesh.is_watertight and level == "heavy":
        try:
            hull = mesh.convex_hull
            # Only use hull if it's not too different in volume
            if hull.volume > 0 and abs(hull.volume - mesh.volume) / hull.volume < 0.3:
                mesh = hull
                logger.info("Used convex hull to achieve watertight mesh")
        except Exception:
            pass

    # 8. Scale to 100mm longest axis (standard 3D print size)
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
    # GLB
    glb = out_dir / "model.glb"
    mesh.export(str(glb))

    # STL
    stl = out_dir / "model.stl"
    mesh.export(str(stl))

    # OBJ
    obj = out_dir / "model.obj"
    mesh.export(str(obj))

    # 3MF
    tmf = out_dir / "model.3mf"
    try:
        mesh.export(str(tmf))
    except Exception:
        _export_3mf_manual(mesh, tmf)

    upd(job_id,
        stl_url=f"/outputs/{job_id}/model.stl",
        tmf_url=f"/outputs/{job_id}/model.3mf",
        glb_url=f"/outputs/{job_id}/model.glb",
        obj_url=f"/outputs/{job_id}/model.obj",
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
    try:
        import trimesh
        scene = trimesh.Scene(mesh)
        png = scene.save_image(resolution=(600, 600), background=[20, 20, 26, 255])
        if png:
            path.write_bytes(png)
            return
    except Exception:
        pass
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (600, 600), (20, 20, 26))
        d = ImageDraw.Draw(img)
        d.text((230, 280), "3D Model", fill=(200, 255, 80))
        d.text((210, 305), f"{len(mesh.faces):,} faces", fill=(100, 100, 120))
        img.save(str(path))
    except Exception:
        pass


# ── TripoSR pipeline ──────────────────────────────────────────────────────────

async def run_triposr(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    try:
        import torch
        from PIL import Image
        from tsr.utils import resize_foreground

        t_start = time.time()
        upd(job_id, status="processing", progress=5, message="Loading image...")

        img = Image.open(image_path).convert("RGBA")

        # Background removal
        if settings.get("remove_bg", True):
            upd(job_id, progress=10, message="Removing background...")
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(None, remove_background, img, job_id)

        # Preprocessing
        upd(job_id, progress=22, message="Preprocessing image...")
        img = resize_foreground(img, 0.85)
        img_np = np.array(img, dtype=np.float32) / 255.0
        if img_np.ndim == 3 and img_np.shape[2] == 4:
            a = img_np[:, :, 3:4]
            img_np = img_np[:, :, :3] * a + 0.5 * (1.0 - a)
        img = Image.fromarray((img_np * 255.0).astype(np.uint8))
        img = img.resize((512, 512), Image.LANCZOS)

        # Inference
        upd(job_id, progress=28, message="Generating 3D structure (TripoSR)...")
        loop = asyncio.get_event_loop()

        def infer():
            with torch.no_grad():
                return models["triposr"]([img], device="cuda")

        scene_codes = await loop.run_in_executor(None, infer)

        # Mesh extraction — try highest resolution that fits in VRAM
        target_res = settings.get("resolution", 512)
        fallbacks  = [r for r in [512, 448, 384, 320, 256] if r <= target_res]

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
                    torch.cuda.empty_cache()
            raise RuntimeError("All resolutions failed")

        raw, used_res = await loop.run_in_executor(None, extract)

        upd(job_id, progress=75, message=f"Mesh extracted at {used_res} ✓")

        # Post-processing
        mesh = to_trimesh(raw)
        post_level = settings.get("post", "standard")
        upd(job_id, progress=80, message=f"Post-processing [{post_level}]...")
        loop = asyncio.get_event_loop()
        mesh = await loop.run_in_executor(None, postprocess_mesh, mesh, job_id, post_level)

        # Export
        upd(job_id, progress=95, message="Exporting files...")
        export_all(mesh, out_dir, job_id)
        render_preview(mesh, out_dir / "preview.png")

        elapsed = round(time.time() - t_start, 1)
        upd(job_id,
            status="done", progress=100,
            message=f"Done in {elapsed}s — {len(mesh.faces):,} polygons",
            model_used=f"TripoSR (res {used_res})",
            time_taken=elapsed,
            preview_url=f"/outputs/{job_id}/preview.png",
        )

    except Exception as e:
        logger.error(f"TripoSR error: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))


# ── Hunyuan3D-2 pipeline ──────────────────────────────────────────────────────

async def run_hunyuan(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    try:
        import torch
        from PIL import Image

        t_start = time.time()
        upd(job_id, status="processing", progress=5, message="Loading image...")

        img = Image.open(image_path).convert("RGBA")

        # Background removal
        if settings.get("remove_bg", True):
            upd(job_id, progress=10, message="Removing background...")
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(None, remove_background, img, job_id)

        # Convert to RGB with white background for Hunyuan3D
        img_rgb = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            img_rgb.paste(img, mask=img.split()[3])
        else:
            img_rgb = img.convert("RGB")

        # Resize to 512x512
        img_rgb = img_rgb.resize((512, 512), Image.LANCZOS)

        upd(job_id, progress=22, message="Generating 3D shape with Hunyuan3D-2...")
        upd(job_id, progress=25, message="This takes 2-5 minutes, please wait...")

        loop = asyncio.get_event_loop()

        def infer():
            with torch.no_grad():
                # Save temp image for pipeline
                import tempfile
                tmp = tempfile.mktemp(suffix=".png")
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

        upd(job_id, progress=78, message="Mesh generated ✓")

        # Post-processing
        mesh = to_trimesh(raw_mesh)
        post_level = settings.get("post", "standard")
        upd(job_id, progress=80, message=f"Post-processing [{post_level}]...")
        mesh = await loop.run_in_executor(None, postprocess_mesh, mesh, job_id, post_level)

        # Export
        upd(job_id, progress=95, message="Exporting files...")
        export_all(mesh, out_dir, job_id)
        render_preview(mesh, out_dir / "preview.png")

        elapsed = round(time.time() - t_start, 1)
        upd(job_id,
            status="done", progress=100,
            message=f"Done in {elapsed}s — {len(mesh.faces):,} polygons",
            model_used="Hunyuan3D-2",
            time_taken=elapsed,
            preview_url=f"/outputs/{job_id}/preview.png",
        )

    except Exception as e:
        logger.error(f"Hunyuan3D-2 error: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))


# ── Demo pipeline (no GPU) ────────────────────────────────────────────────────

async def run_demo(job_id: str, image_path: Path, out_dir: Path, settings: dict):
    import trimesh
    for p, m in [(20, "Analyzing..."), (50, "Building mesh..."), (80, "Exporting...")]:
        upd(job_id, status="processing", progress=p, message=m)
        await asyncio.sleep(0.8)
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    export_all(mesh, out_dir, job_id)
    render_preview(mesh, out_dir / "preview.png")
    upd(job_id, status="done", progress=100,
        message="Demo mode — no GPU available",
        model_used="Demo",
        preview_url=f"/outputs/{job_id}/preview.png")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    import torch
    return {
        "triposr":    models["triposr"] is not None,
        "hunyuan":    models["hunyuan"] is not None,
        "rembg":      models["rembg_sess"] is not None,
        "cuda":       torch.cuda.is_available(),
        "gpu":        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/convert", response_model=JobStatus)
async def convert(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model:      str  = "triposr",    # triposr | hunyuan
    remove_bg:  str  = "true",
    resolution: int  = 512,
    steps:      int  = 50,           # hunyuan diffusion steps
    post:       str  = "standard",   # none | light | standard | heavy
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Images only")

    job_id   = str(uuid.uuid4())
    ext      = Path(file.filename or "image.jpg").suffix or ".jpg"
    img_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    settings = {
        "remove_bg":  remove_bg.lower() == "true",
        "resolution": resolution,
        "steps":      max(5, min(100, steps)),
        "post":       post,
    }

    # Determine which model to use
    if model == "hunyuan" and models["hunyuan"] is not None:
        run_fn = run_hunyuan
        model_label = "Hunyuan3D-2"
    elif models["triposr"] is not None:
        run_fn = run_triposr
        model_label = "TripoSR"
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


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    jobs.pop(job_id, None)
    shutil.rmtree(OUTPUT_DIR / job_id, ignore_errors=True)
    return {"deleted": job_id}


app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    for candidate in [
        BASE_DIR.parent / "frontend" / "index.html",
        BASE_DIR.parent / "index.html",
        Path.cwd() / "frontend" / "index.html",
    ]:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return HTMLResponse("<h1>PIXFORM</h1><p><a href='/docs'>API docs</a></p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
