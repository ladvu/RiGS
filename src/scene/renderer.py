import torch
from typing import *
from gsplat.rendering import rasterization_2dgs, rasterization
from einops import repeat

class GaussianRenderer:
    def __init__(
        self,
        use_2dgs: bool = True
    ):
        self.use_2dgs = use_2dgs

    def render(
        self, 
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        width:int,
        height:int,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
        backgrounds: Optional[torch.Tensor] = None,
        render_mode: str = "RGB",
        **kwargs
    ):
        if backgrounds is not None:
            color_dim = colors.shape[-1]
            pad_size = color_dim - len(backgrounds)
            backgrounds = torch.tensor(backgrounds + [0.0] * pad_size, device=means.device, dtype=torch.float32)
            backgrounds = repeat(backgrounds, "c -> t k c", t=extrinsics.shape[0], k=1)

        inputs = {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors.unsqueeze(1), # T (N+M) 6
            "viewmats": extrinsics.unsqueeze(1),  # [T, 1, 4, 4]
            "Ks": intrinsics.unsqueeze(1),  # [T, 1, 3, 3]
            "width": width,
            "height": height,
            "near_plane": near_plane,
            "far_plane": far_plane,
            "radius_clip": radius_clip,
            "eps2d": eps2d,
            "render_mode": render_mode,
            "backgrounds": backgrounds,
        }
        outputs = {}
        if self.use_2dgs:
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                distort, median_depth,
                info,
            ) = rasterization_2dgs(**inputs)
            outputs["normals"] = render_normals
            outputs["median_depth"] = median_depth
            outputs["normals_from_depth"] = normals_from_depth
        else:
            (
                render_colors,
                render_alphas,
                info,
            ) = rasterization(**inputs)

        outputs["colors"] = render_colors
        outputs["alphas"] = render_alphas
        outputs["info"] = info

        return outputs
    
    def __call__(self, *args, **kwds):
        return self.render(*args, **kwds)