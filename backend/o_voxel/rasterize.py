import torch
import torch.nn.functional as F
try:
    from easydict import EasyDict as edict
except ImportError:
    class edict(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)
        def __setattr__(self, k, v):
            self[k] = v


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


class VoxelRenderer:
    """
    CPU-based software fallback renderer for the Voxel representation.
    Replaces the unavailable CUDA native extension with a pure-PyTorch
    point-splatting rasterizer (z-buffer, back-to-front).

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.rendering_options = edict({
            "resolution": None,
            "near": 0.1,
            "far": 10.0,
            "ssaa": 1,
        })
        self.rendering_options.update(rendering_options)

    def render(
            self,
            position: torch.Tensor,
            attrs: torch.Tensor,
            voxel_size: float,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
        ) -> edict:
        """
        CPU software-rasterize the voxels (point-splatting with z-buffer).

        Args:
            position (torch.Tensor): (N, 3) voxel centres in world space
            attrs (torch.Tensor): (N, C) per-voxel attributes
            voxel_size (float): edge length of each voxel
            extrinsics (torch.Tensor): (4, 4) world-to-camera transform
            intrinsics (torch.Tensor): (3, 3) OpenCV camera intrinsics
                (fx, fy normalised to [0,1]; cx, cy in [0,1])

        Returns:
            edict with keys:
                attr  (torch.Tensor): (C, H, W) rendered attributes
                depth (torch.Tensor): (H, W)    rendered depth
                alpha (torch.Tensor): (H, W)    rendered alpha (binary)
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far  = self.rendering_options["far"]
        ssaa = int(self.rendering_options.get("ssaa", 1))

        H = W = resolution * ssaa
        N, C = attrs.shape[0], attrs.shape[1]
        device = position.device
        dtype  = position.dtype

        # --- Build MVP ---
        perspective = intrinsics_to_projection(intrinsics, near, far)
        mvp = perspective @ extrinsics            # (4,4)

        # Homogeneous coords
        ones  = torch.ones(N, 1, device=device, dtype=dtype)
        pos_h = torch.cat([position, ones], dim=1)   # (N, 4)

        # Clip space
        pos_clip = (mvp @ pos_h.T).T                 # (N, 4)
        w_clip   = pos_clip[:, 3].clamp(min=1e-6)

        # Camera-space depth (for z-buffer value)
        pos_cam = (extrinsics @ pos_h.T).T           # (N, 4)
        z       = pos_cam[:, 2]

        # NDC
        x_ndc = pos_clip[:, 0] / w_clip
        y_ndc = pos_clip[:, 1] / w_clip
        z_ndc = pos_clip[:, 2] / w_clip

        # Pixel coords (origin top-left; y increases downward)
        px = ((x_ndc + 1.0) * 0.5 * W).long()
        py = ((1.0 - y_ndc) * 0.5 * H).long()

        # Validity / frustum culling
        valid = (
            (z      >  near) & (z      < far) &
            (z_ndc  >= -1.0) & (z_ndc  <= 1.0) &
            (px     >= 0)    & (px     < W)  &
            (py     >= 0)    & (py     < H)
        )

        # Output buffers
        color = torch.zeros(C, H, W, dtype=dtype, device=device)
        depth = torch.zeros(H, W,    dtype=dtype, device=device)
        alpha = torch.zeros(H, W,    dtype=dtype, device=device)

        if valid.any():
            vidx = valid.nonzero(as_tuple=True)[0]

            # Sort back-to-front (largest z first) so the closest voxel
            # is written last and wins for duplicate pixels.
            order = torch.argsort(z[vidx], descending=True)
            vidx  = vidx[order]

            px_s    = px[vidx]
            py_s    = py[vidx]
            z_s     = z[vidx]
            attrs_s = attrs[vidx]       # (K, C)

            flat = (py_s * W + px_s).long()   # (K,)

            # Scattered write – last (closest) value wins for duplicate pixels
            depth.view(-1).scatter_(0, flat, z_s)
            alpha.view(-1).scatter_(0, flat, torch.ones_like(z_s))
            color_flat = color.reshape(C, H * W)
            color_flat.scatter_(1, flat.unsqueeze(0).expand(C, -1), attrs_s.T)

        if ssaa > 1:
            color = F.interpolate(
                color[None], size=(resolution, resolution),
                mode='bilinear', align_corners=False, antialias=True
            ).squeeze(0)
            depth = F.interpolate(
                depth[None, None], size=(resolution, resolution),
                mode='bilinear', align_corners=False, antialias=True
            ).squeeze()
            alpha = F.interpolate(
                alpha[None, None], size=(resolution, resolution),
                mode='bilinear', align_corners=False, antialias=True
            ).squeeze()

        return edict({'attr': color, 'depth': depth, 'alpha': alpha})

