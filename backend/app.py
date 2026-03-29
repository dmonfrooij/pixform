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

VALID_POST_LEVELS = {"none", "light", "standard", "heavy"}
TRIPOSR_RES_LEVELS = [1024, 896, 768, 640, 512, 448, 384, 320, 256, 192, 128]


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
    Goal: always produce a 100% watertight, slicer-ready mesh.
    """
    import trimesh
    import trimesh.smoothing
    import numpy as _np

    smooth_iters   = {"none": 0, "light": 5,  "standard": 15, "heavy": 25}.get(level, 15)
    poisson_depth  = {"none": 9, "light": 10, "standard": 11, "heavy": 12}.get(level, 11)
    poisson_points = {"none": 60000, "light": 100000, "standard": 200000, "heavy": 350000}.get(level, 200000)

    logger.info(f"Post-processing [{level}]: {len(mesh.faces):,} faces")
    upd(job_id, progress=86, message="Cleaning mesh...")

    # ── 1. Keep largest component ──────────────────────────────────────────────
    components = mesh.split(only_watertight=False)
    if components:
        mesh = max(components, key=lambda c: len(c.faces))

    # ── 2. Basic repair ────────────────────────────────────────────────────────
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # ── 3. Smoothing ───────────────────────────────────────────────────────────
    if smooth_iters > 0:
        upd(job_id, progress=89, message="Smoothing mesh...")
        try:
            mesh = trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)
        except Exception:
            try:
                mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters, lamb=0.2)
            except Exception:
                pass

    # ── 4. Poisson reconstruction → mathematically watertight surface ─────────
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
        logger.warning(f"Poisson failed: {e} — continuing with trimesh repair")

    # ── 5. Hole filling — multiple passes ──────────────────────────────────────
    upd(job_id, progress=94, message="Filling holes...")
    for _ in range(5):
        if mesh.is_watertight:
            break
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

    # ── 6. Voxel remesh fallback — guaranteed closed if still open ─────────────
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

    # ── 7. Final cleanup ───────────────────────────────────────────────────────
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    # ── 8. Scale to 100mm ──────────────────────────────────────────────────────
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


def _normalize_triposr_resolution(value: int) -> int:
    """Snap requested resolution to the closest supported TripoSR extraction level."""
    try:
        v = int(value)
    except Exception:
        return 512
    v = max(128, min(1024, v))
    return min(TRIPOSR_RES_LEVELS, key=lambda x: abs(x - v))


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

    # ── Attempt 1: pyrender (high quality, multi-light) ─────────────────────
    try:
        import pyrender
        import trimesh
        import numpy as _np

        scene = pyrender.Scene(bg_color=[20, 20, 26, 255], ambient_light=[0.15, 0.15, 0.15])

        mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(mesh_pr)

        # Camera — isometric-ish view from upper-front-right
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

    # ── Attempt 2: trimesh scene renderer ────────────────────────────────────
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

    # ── Fallback: PIL placeholder ─────────────────────────────────────────────
    try:
        img = _Img.new("RGB", (800, 800), (20, 20, 26))
        d = _Draw.Draw(img)
        d.text((310, 380), "3D Model", fill=(200, 255, 80))
        d.text((285, 410), f"{len(mesh.faces):,} faces", fill=(120, 120, 140))
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
        # Enhance contrast and sharpness so the model sees clearer edges and detail
        from PIL import ImageEnhance
        img = ImageEnhance.Contrast(img).enhance(1.35)
        img = ImageEnhance.Sharpness(img).enhance(1.4)
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
                    torch.cuda.empty_cache()
            raise RuntimeError("All resolutions failed")

        raw, used_res = await loop.run_in_executor(None, extract)

        upd(job_id, progress=75, message=f"Mesh extracted at {used_res} ✓")

        # Post-processing
        mesh = to_trimesh(raw)
        if len(mesh.faces) == 0:
            raise RuntimeError("TripoSR produced an empty mesh")
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

        # Enhance contrast and sharpness so the model sees clearer edges and detail
        from PIL import ImageEnhance
        img_rgb = ImageEnhance.Contrast(img_rgb).enhance(1.35)
        img_rgb = ImageEnhance.Sharpness(img_rgb).enhance(1.4)

        # Resize to 512x512
        img_rgb = img_rgb.resize((512, 512), Image.LANCZOS)

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

        upd(job_id, progress=78, message="Mesh generated ✓")

        # Post-processing
        mesh = to_trimesh(raw_mesh)
        if len(mesh.faces) == 0:
            raise RuntimeError("Hunyuan3D-2 produced an empty mesh")
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
    model = (model or "triposr").lower().strip()
    if model not in {"triposr", "hunyuan"}:
        raise HTTPException(400, "Invalid model. Use: triposr or hunyuan")

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
    if model == "hunyuan":
        if models["hunyuan"] is None:
            raise HTTPException(503, "Hunyuan3D-2 model is not loaded. Check /health and GPU setup.")
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
