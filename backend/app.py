
# --- PIXFORM: mesh paint extension (appended) ---
# Added by Copilot feature/3mf-first: lightweight mesh painting and /paint endpoint

import colorsys


def _kmeans_numpy(X, k=6, max_iters=60):
    """Simple KMeans implementation using numpy (no sklearn dependency)."""
    import numpy as _np
    n, d = X.shape
    rng = _np.random.default_rng(seed=int.from_bytes(_np.random.bytes(4), "big"))
    centers = _np.empty((k, d), dtype=float)
    centers[0] = X[rng.integers(n)]
    # kmeans++ init
    dist = ((X - centers[0])**2).sum(axis=1)
    for i in range(1, k):
        probs = dist / (dist.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
        newdist = ((X - centers[i])**2).sum(axis=1)
        dist = _np.minimum(dist, newdist)
    labels = _np.zeros(n, dtype=int)
    for _ in range(max_iters):
        # assign
        dists = _np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
        new_labels = _np.argmin(dists, axis=1)
        if _np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update
        for j in range(k):
            sel = X[labels == j]
            if len(sel) > 0:
                centers[j] = sel.mean(axis=0)
    return labels


def _smooth_face_labels(mesh, labels, iterations=2):
    """Smooth face labels across mesh adjacency to avoid speckle regions."""
    import numpy as _np
    face_to_verts = _np.asarray(mesh.faces)
    vertex_to_faces = [[] for _ in range(len(mesh.vertices))]
    for fi, f in enumerate(face_to_verts):
        for v in f:
            vertex_to_faces[v].append(fi)
    neighbors = [set() for _ in range(len(face_to_verts))]
    for vfaces in vertex_to_faces:
        for a in vfaces:
            for b in vfaces:
                if a != b:
                    neighbors[a].add(b)
    labels = labels.copy()
    for _ in range(iterations):
        new_labels = labels.copy()
        for fi in range(len(labels)):
            nbs = list(neighbors[fi])
            if not nbs:
                continue
            arr = labels[nbs]
            vals, counts = _np.unique(arr, return_counts=True)
            if len(vals) == 0:
                continue
            majority = vals[_np.argmax(counts)]
            if (counts.max() / max(1, len(nbs))) > 0.5:
                new_labels[fi] = majority
        labels = new_labels
    return labels


def _build_palette(k):
    """Return k RGBA tuples (uint8) using HSV spacing for visually distinct colors."""
    out = []
    for i in range(k):
        h = (i / float(k)) % 1.0
        s = 0.65 + 0.25 * ((i % 2))
        v = 0.85
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((int(r * 255), int(g * 255), int(b * 255), 255))
    return out


def _face_labels_to_vertex_colors(mesh, face_labels, palette):
    """Convert per-face labels + palette into per-vertex RGBA uint8 colors by averaging adjacent faces."""
    import numpy as _np
    vcol = _np.zeros((len(mesh.vertices), 4), dtype=_np.uint32)
    counts = _np.zeros((len(mesh.vertices),), dtype=_np.uint32)
    faces = _np.asarray(mesh.faces)
    for fi, f in enumerate(faces):
        c = palette[int(face_labels[fi])]
        for v in f:
            vcol[v, :3] += (c[0], c[1], c[2])
            vcol[v, 3] += c[3]
            counts[v] += 1
    nonzero = counts > 0
    # avoid divide-by-zero
    vcolors = _np.zeros((len(mesh.vertices), 4), dtype=_np.uint8)
    if nonzero.any():
        vcolors[nonzero, :3] = (_np.round(vcol[nonzero, :3].astype(float) / counts[nonzero].reshape(-1, 1))).astype(_np.uint8)
        vcolors[nonzero, 3] = 255
    return vcolors


def paint_mesh_sync(job_id: str, mesh_path, out_dir, options: dict):
    """Synchronous mesh painting worker. Intended to be run via BackgroundTasks.
    options: {'color_count': int, 'post': str}
    """
    import time
    try:
        import numpy as _np
        import trimesh
        from PIL import Image

        t_start = time.time()
        upd(job_id, status="processing", progress=5, message="Loading mesh...", stage="load_mesh")
        _assert_not_cancelled(job_id)

        mesh_path = Path(mesh_path)
        mesh = trimesh.load(mesh_path, process=False)
        if mesh is None or len(mesh.faces) == 0:
            raise RuntimeError("Uploaded mesh is empty or unsupported")

        upd(job_id, progress=12, message="Analyzing geometry...", stage="analyze")
        _assert_not_cancelled(job_id)

        face_centroids = _np.asarray(mesh.triangles_center)
        face_normals = _np.asarray(mesh.face_normals)
        face_vert_ids = _np.asarray(mesh.faces)

        # approximate curvature: std of adjacent vertex normals
        vert_normals = _np.asarray(mesh.vertex_normals)
        curvature = _np.zeros(len(face_centroids))
        for i, f in enumerate(face_vert_ids):
            nv = vert_normals[f]
            curvature[i] = float(_np.std(nv, axis=0).sum())

        height = face_centroids[:, 2:3].astype(float)

        features = _np.hstack([
            face_centroids.astype(float),
            face_normals.astype(float),
            curvature.reshape(-1, 1),
            height,
        ])
        fmean = features.mean(axis=0, keepdims=True)
        fstd = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - fmean) / fstd

        k = int(max(2, min(12, int(options.get('color_count', 6)))))
        upd(job_id, progress=25, message=f"Clustering into {k} regions...", stage="cluster")
        labels = _kmeans_numpy(features, k=k, max_iters=80)
        _assert_not_cancelled(job_id)

        upd(job_id, progress=45, message="Smoothing color regions...", stage="segment")
        labels = _smooth_face_labels(mesh, labels, iterations=2)
        _assert_not_cancelled(job_id)

        upd(job_id, progress=55, message="Assigning palette...", stage="color")
        palette = _build_palette(k)
        face_colors = [_np.array(palette[int(l)], dtype=_np.uint8) for l in labels]
        vcolors = _face_labels_to_vertex_colors(mesh, labels, palette)
        mesh.visual.vertex_colors = vcolors
        _assert_not_cancelled(job_id)

        upd(job_id, progress=70, message="Post-processing mesh for printability...", stage="postprocess")
        # try meshfix if available
        try:
            mesh = _try_meshfix_watertight(mesh)
        except Exception:
            pass
        mesh = postprocess_mesh(mesh, job_id, level=options.get('post', 'standard'), target_profile=None)
        _assert_not_cancelled(job_id)

        upd(job_id, progress=90, message="Exporting colored models...", stage="export")
        # Save GLB (vertex colors included)
        try:
            mesh.export(str(Path(out_dir) / "model.glb"))
        except Exception as e:
            logger.warning(f"Colored GLB export failed: {e}")

        # Attempt 3MF export (best-effort). If trimesh exports with color, fine; otherwise fallback
        try:
            mesh.export(str(Path(out_dir) / "model.3mf"))
        except Exception:
            try:
                _export_3mf_manual(mesh, Path(out_dir) / "model.3mf")
            except Exception as e:
                logger.warning(f"3MF export (fallback) failed: {e}")

        # Also export STL/OBJ where appropriate (color will not be present in STL)
        try:
            mesh.export(str(Path(out_dir) / "model.stl"))
        except Exception:
            pass
        try:
            mesh.export(str(Path(out_dir) / "model.obj"))
        except Exception:
            pass

        # Render preview; render_preview respects vertex colors when possible
        try:
            render_preview(mesh, Path(out_dir) / "preview.png")
        except Exception:
            pass

        elapsed = round(time.time() - t_start, 1)
        upd(job_id,
            status="done", progress=100,
            message=f"Painted in {elapsed}s - {len(mesh.faces):,} polygons",
            model_used="Painted (upload)",
            time_taken=elapsed,
            preview_url=f"/outputs/{job_id}/preview.png" if (Path(out_dir) / "preview.png").exists() else None,
            stl_url=f"/outputs/{job_id}/model.stl" if (Path(out_dir) / "model.stl").exists() else None,
            tmf_url=f"/outputs/{job_id}/model.3mf" if (Path(out_dir) / "model.3mf").exists() else None,
            glb_url=f"/outputs/{job_id}/model.glb" if (Path(out_dir) / "model.glb").exists() else None,
            obj_url=f"/outputs/{job_id}/model.obj" if (Path(out_dir) / "model.obj").exists() else None,
            poly_count=len(mesh.faces),
            stage="done",
        )

    except Exception as e:
        if str(e) == "Job cancelled by user":
            upd(job_id, status="cancelled", message="Cancelled by user")
            return
        logger.error(f"Paint error: {e}", exc_info=True)
        upd(job_id, status="error", message=str(e))
    finally:
        _finalize_job_resources(job_id)


@app.post("/paint", response_model=JobStatus)
async def paint(
    background_tasks: BackgroundTasks,
    mesh_file: UploadFile = File(...),
    color_count: int = 6,
    post: str = "standard",
):
    """Upload a mesh (STL/3MF/OBJ/PLY/GLB) and automatically paint it into N colors for multi-filament printing.
    Returns the usual JobStatus and writes outputs to /outputs/<job_id>/
    """
    # basic validation
    if post not in VALID_POST_LEVELS:
        raise HTTPException(400, "Invalid post level. Use: none, light, standard, heavy")

    job_id = str(uuid.uuid4())
    ext = Path(mesh_file.filename or "mesh.stl").suffix or ".stl"
    if ext.lower() not in {".stl", ".3mf", ".obj", ".ply", ".glb", ".gltf"}:
        raise HTTPException(400, "Supported mesh formats: .stl, .3mf, .obj, .ply, .glb, .gltf")

    mesh_path = UPLOAD_DIR / f"{job_id}_mesh{ext}"
    with open(mesh_path, "wb") as f:
        shutil.copyfileobj(mesh_file.file, f)

    # clamp color_count
    try:
        color_count = int(color_count)
    except Exception:
        color_count = 6
    color_count = max(2, min(12, color_count))

    jobs[job_id] = dict(
        job_id=job_id, status="queued", progress=0,
        message="Queued...",
        model_used="Painted",
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
        upload_path=str(mesh_path),
        output_dir=str(OUTPUT_DIR / job_id),
        upload_deleted=False,
    )

    out_dir = OUTPUT_DIR / job_id
    out_dir.mkdir(exist_ok=True)

    background_tasks.add_task(paint_mesh_sync, job_id, str(mesh_path), str(out_dir), {"color_count": color_count, "post": post})
    return JobStatus(**jobs[job_id])

# --- end appended mesh paint extension
