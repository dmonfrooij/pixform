"""
o_voxel.postprocess – Windows stub.

The full implementation (to_glb) requires cumesh, nvdiffrast, and flex_gemm
which are native CUDA extensions. When those are present the real code runs;
otherwise a clear RuntimeError is raised so callers can catch and fall back.
"""


def to_glb(
    vertices,
    faces,
    attr_volume,
    coords,
    attr_layout,
    aabb,
    voxel_size=None,
    grid_size=None,
    decimation_target=1_000_000,
    texture_size=2048,
    remesh=False,
    remesh_band=1,
    remesh_project=0.9,
    verbose=False,
    use_tqdm=False,
    **kwargs,
):
    """
    Convert a TRELLIS.2 MeshWithVoxel to a textured GLB trimesh.

    Requires: cumesh, nvdiffrast, flex_gemm (all CUDA-only).
    Raises RuntimeError when those dependencies are missing.
    """
    try:
        import cumesh          # noqa: F401 – probe only
        import nvdiffrast.torch  # noqa: F401
        from flex_gemm.ops.grid_sample import grid_sample_3d  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"o_voxel.postprocess.to_glb requires cumesh, nvdiffrast, and "
            f"flex_gemm (CUDA-only). Missing dependency: {exc}"
        ) from exc

    # All dependencies are present – run the real implementation.
    import sys
    from pathlib import Path

    # The real postprocess lives in the o-voxel repo next to the project root.
    _repo = Path(__file__).parent.parent.parent / "trellis2_repo" / "o-voxel" / "o_voxel"
    if _repo.exists() and str(_repo.parent) not in sys.path:
        sys.path.insert(0, str(_repo.parent))

    from o_voxel.postprocess import to_glb as _real_to_glb  # type: ignore
    return _real_to_glb(
        vertices=vertices,
        faces=faces,
        attr_volume=attr_volume,
        coords=coords,
        attr_layout=attr_layout,
        aabb=aabb,
        voxel_size=voxel_size,
        grid_size=grid_size,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=remesh_band,
        remesh_project=remesh_project,
        verbose=verbose,
        use_tqdm=use_tqdm,
        **kwargs,
    )

