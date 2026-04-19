from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def _unsupported(name: str):
    raise RuntimeError(
        f"o_voxel native extension is not available on this Windows install; {name} is unsupported."
    )


def _to_numpy_aabb(aabb) -> np.ndarray:
    if isinstance(aabb, torch.Tensor):
        aabb = aabb.detach().cpu().numpy()
    return np.asarray(aabb, dtype=np.float32)


def _to_grid_size(grid_size) -> np.ndarray:
    if isinstance(grid_size, torch.Tensor):
        grid_size = grid_size.detach().cpu().numpy()
    if isinstance(grid_size, int):
        return np.array([grid_size, grid_size, grid_size], dtype=np.int32)
    return np.asarray(grid_size, dtype=np.int32)


def _cube_face_fallback(coords_np: np.ndarray, aabb_np: np.ndarray, grid_size_np: np.ndarray):
    step = (aabb_np[1] - aabb_np[0]) / np.maximum(grid_size_np, 1)
    occupied = {tuple(v) for v in coords_np.tolist()}
    face_defs: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]], ...] = (
        ((-1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)),
        ((1, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)),
        ((0, -1, 0), (0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)),
        ((0, 1, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)),
        ((0, 0, -1), (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
        ((0, 0, 1), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)),
    )

    vertex_to_index: dict[Tuple[int, int, int], int] = {}
    vertices_grid: list[Tuple[int, int, int]] = []
    faces: list[Tuple[int, int, int]] = []

    def _v_idx(v: Tuple[int, int, int]) -> int:
        idx = vertex_to_index.get(v)
        if idx is None:
            idx = len(vertices_grid)
            vertex_to_index[v] = idx
            vertices_grid.append(v)
        return idx

    for x, y, z in occupied:
        for neigh, c0, c1, c2, c3 in face_defs:
            nx, ny, nz = x + neigh[0], y + neigh[1], z + neigh[2]
            if (nx, ny, nz) in occupied:
                continue
            i0 = _v_idx((x + c0[0], y + c0[1], z + c0[2]))
            i1 = _v_idx((x + c1[0], y + c1[1], z + c1[2]))
            i2 = _v_idx((x + c2[0], y + c2[1], z + c2[2]))
            i3 = _v_idx((x + c3[0], y + c3[1], z + c3[2]))
            faces.append((i0, i1, i2))
            faces.append((i0, i2, i3))

    if not vertices_grid or not faces:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    vg = np.asarray(vertices_grid, dtype=np.float32)
    vertices_world = aabb_np[0][None, :] + vg * step[None, :]
    return vertices_world.astype(np.float32), np.asarray(faces, dtype=np.int64)


def flexible_dual_grid_to_mesh(
    coords: torch.Tensor,
    dual_vertices: torch.Tensor,
    intersected_flag: torch.Tensor,
    split_weight: torch.Tensor | None,
    aabb,
    voxel_size=None,
    grid_size=None,
    train: bool = False,
):
    """Fallback mesh extraction for Windows without native o_voxel extension.

    Prefer marching-cubes on sparse occupancy for smoother surfaces, then fall back
    to exposed voxel faces when marching cubes is unavailable.
    """
    del dual_vertices, intersected_flag, split_weight, voxel_size, train

    device = coords.device
    coords_np = coords.detach().to("cpu").numpy().astype(np.int32, copy=False)
    if coords_np.size == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0, 3), dtype=torch.long, device=device),
        )

    if grid_size is None:
        max_xyz = coords_np.max(axis=0)
        grid_size_np = (max_xyz + 1).astype(np.int32)
    else:
        grid_size_np = _to_grid_size(grid_size)

    aabb_np = _to_numpy_aabb(aabb)
    step = (aabb_np[1] - aabb_np[0]) / np.maximum(grid_size_np, 1)

    verts_world = None
    faces_np = None

    try:
        min_corner = coords_np.min(axis=0)
        max_corner = coords_np.max(axis=0)
        local_shape = (max_corner - min_corner + 3).astype(np.int32)
        # Guard memory blowups on extreme sparse ranges.
        if int(np.prod(local_shape, dtype=np.int64)) <= 32_000_000:
            occ = np.zeros(tuple(local_shape.tolist()), dtype=np.uint8)
            local = coords_np - min_corner[None, :] + 1
            occ[local[:, 0], local[:, 1], local[:, 2]] = 1

            try:
                import trimesh

                mc_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(occ, pitch=1.0)
                if mc_mesh is not None and len(mc_mesh.faces) > 0:
                    verts_grid = np.asarray(mc_mesh.vertices, dtype=np.float32) + (min_corner.astype(np.float32) - 1.0)
                    verts_world = aabb_np[0][None, :] + verts_grid * step[None, :]
                    faces_np = np.asarray(mc_mesh.faces, dtype=np.int64)
            except Exception:
                try:
                    import mcubes

                    verts_local, faces_local = mcubes.marching_cubes(occ.astype(np.float32), 0.5)
                    if len(faces_local) > 0:
                        verts_grid = np.asarray(verts_local, dtype=np.float32) + (min_corner.astype(np.float32) - 1.0)
                        verts_world = aabb_np[0][None, :] + verts_grid * step[None, :]
                        faces_np = np.asarray(faces_local, dtype=np.int64)
                except Exception:
                    pass
    except Exception:
        pass

    if verts_world is None or faces_np is None or len(faces_np) == 0:
        verts_world, faces_np = _cube_face_fallback(coords_np, aabb_np, grid_size_np)

    return (
        torch.from_numpy(verts_world).to(device=device, dtype=torch.float32),
        torch.from_numpy(faces_np).to(device=device, dtype=torch.long),
    )


def mesh_to_flexible_dual_grid(*args, **kwargs):
    _unsupported("mesh_to_flexible_dual_grid")

