from __future__ import annotations

from pathlib import Path
import torch


def _path(path) -> Path:
    return Path(path)


def write(path, coords, attrs=None, **_kwargs):
    target = _path(path)
    payload = {"coords": coords.detach().cpu()}
    if attrs is not None:
        payload["attrs"] = attrs.detach().cpu()
    torch.save(payload, target)


def read(path, **_kwargs):
    payload = torch.load(_path(path), map_location="cpu")
    return payload["coords"], payload.get("attrs")


def read_vxz(path, **kwargs):
    return read(path, **kwargs)


def write_vxz(path, coords, attrs=None, **kwargs):
    return write(path, coords, attrs, **kwargs)

