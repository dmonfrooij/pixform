"""
o_voxel.serialize – Windows stub.

The full implementation uses CUDA / CPU native extensions (_C) for Z-order /
Hilbert curve encoding. A pure-Python fallback is provided for the Z-order
case; Hilbert raises RuntimeError when the native extension is absent.
"""
from __future__ import annotations
from typing import List, Literal
import torch


# ---------------------------------------------------------------------------
# Pure-Python / PyTorch Z-order (Morton) encode / decode
# (handles up to 10 bits per axis → 30-bit code, matching the native impl)
# ---------------------------------------------------------------------------

def _spread_bits(x: torch.Tensor) -> torch.Tensor:
    """Interleave 10-bit integers into a 30-bit Morton code (per axis)."""
    x = x & 0x3FF
    x = (x | (x << 16)) & 0x30000FF
    x = (x | (x <<  8)) & 0x300F00F
    x = (x | (x <<  4)) & 0x30C30C3
    x = (x | (x <<  2)) & 0x9249249
    return x


def _compact_bits(x: torch.Tensor) -> torch.Tensor:
    """Reverse of _spread_bits: extract every third bit."""
    x = x & 0x9249249
    x = (x | (x >>  2)) & 0x30C30C3
    x = (x | (x >>  4)) & 0x300F00F
    x = (x | (x >>  8)) & 0x300F00F  # keep 10 bits per axis
    x = (x | (x >> 16)) & 0x3FF
    return x


@torch.no_grad()
def encode_seq(
    coords: torch.Tensor,
    permute: List[int] = [0, 1, 2],
    mode: Literal["z_order", "hilbert"] = "z_order",
) -> torch.Tensor:
    """Encode (N, 3) integer coordinates into a 30-bit Morton / Hilbert code."""
    assert coords.shape[-1] == 3 and coords.ndim == 2
    if mode == "hilbert":
        _require_native("encode_seq (hilbert mode)")
    x = coords[:, permute[0]].long()
    y = coords[:, permute[1]].long()
    z = coords[:, permute[2]].long()
    return _spread_bits(x) | (_spread_bits(y) << 1) | (_spread_bits(z) << 2)


@torch.no_grad()
def decode_seq(
    code: torch.Tensor,
    permute: List[int] = [0, 1, 2],
    mode: Literal["z_order", "hilbert"] = "z_order",
) -> torch.Tensor:
    """Decode a 30-bit Morton code back to (N, 3) integer coordinates."""
    assert code.ndim == 1
    if mode == "hilbert":
        _require_native("decode_seq (hilbert mode)")
    cx = _compact_bits(code)
    cy = _compact_bits(code >> 1)
    cz = _compact_bits(code >> 2)
    coords = torch.stack([cx, cy, cz], dim=-1)
    # Un-permute
    inv = [0, 0, 0]
    for i, p in enumerate(permute):
        inv[p] = i
    return coords[:, inv]


def _require_native(name: str):
    raise RuntimeError(
        f"o_voxel.serialize.{name} requires the native CUDA extension (_C) "
        "which is not available on this install."
    )

