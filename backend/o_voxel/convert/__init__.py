def _unsupported(name: str):
    raise RuntimeError(
        f"o_voxel native extension is not available on this Windows install; {name} is unsupported."
    )


def flexible_dual_grid_to_mesh(*args, **kwargs):
    _unsupported("flexible_dual_grid_to_mesh")


def mesh_to_flexible_dual_grid(*args, **kwargs):
    _unsupported("mesh_to_flexible_dual_grid")

