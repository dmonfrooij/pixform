from __future__ import annotations

import torch


def _dense_from_sparse(feats: torch.Tensor, coords: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if feats.dim() != 2:
        raise ValueError(f"Expected feats [N, C], got {tuple(feats.shape)}")
    if coords.dim() != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected coords [N, 4], got {tuple(coords.shape)}")

    channels = feats.shape[1]
    if len(shape) >= 4:
        _, width, height, depth = [int(x) for x in shape[-4:]]
    elif len(shape) == 3:
        width, height, depth = [int(x) for x in shape]
    else:
        raise ValueError(f"Unsupported voxel shape: {shape}")

    batch = int(coords[:, 0].max().item()) + 1 if coords.numel() else 1
    dense = feats.new_zeros((batch, channels, depth, height, width))

    b = coords[:, 0].long()
    x = coords[:, 1].long()
    y = coords[:, 2].long()
    z = coords[:, 3].long()
    valid = (
        (b >= 0) & (b < batch) &
        (x >= 0) & (x < width) &
        (y >= 0) & (y < height) &
        (z >= 0) & (z < depth)
    )
    if valid.any():
        dense[b[valid], :, z[valid], y[valid], x[valid]] = feats[valid]
    return dense


def grid_sample_3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: torch.Size,
    grid: torch.Tensor,
    mode: str = "trilinear",
) -> torch.Tensor:
    """Small Windows fallback for TRELLIS.2 imports.

    This mirrors the public `flex_gemm.ops.grid_sample.grid_sample_3d` symbol with a
    dense PyTorch implementation so the TRELLIS.2 Python package can import even when
    the native FlexGEMM extension is unavailable.
    """
    if mode not in {"nearest", "trilinear"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if grid.dim() != 3 or grid.shape[-1] != 3:
        raise ValueError(f"Expected grid [B, L, 3], got {tuple(grid.shape)}")

    dense = _dense_from_sparse(feats, coords, shape)
    batch = dense.shape[0]
    if grid.shape[0] != batch:
        if grid.shape[0] == 1:
            grid = grid.expand(batch, -1, -1)
        else:
            raise ValueError(f"Grid batch {grid.shape[0]} does not match sparse batch {batch}")

    if len(shape) >= 4:
        _, width, height, depth = [int(x) for x in shape[-4:]]
    else:
        width, height, depth = [int(x) for x in shape]

    norm = grid.to(dense.dtype).clone()
    def _scale(v: torch.Tensor, size: int) -> torch.Tensor:
        if size <= 1:
            return torch.zeros_like(v)
        return (v / (size - 1)) * 2 - 1

    norm[..., 0] = _scale(norm[..., 0], width)
    norm[..., 1] = _scale(norm[..., 1], height)
    norm[..., 2] = _scale(norm[..., 2], depth)
    sample_grid = norm.view(batch, -1, 1, 1, 3)

    sampled = torch.nn.functional.grid_sample(
        dense,
        sample_grid,
        mode="bilinear" if mode == "trilinear" else "nearest",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(-1).squeeze(-1).transpose(1, 2)

