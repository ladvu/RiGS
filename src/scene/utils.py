import torch
from typing import *
from .dataset import Parser
import math
import os
from utils.misc_utils import knn, PesudoTensor
from utils.gs_utils import RGB2SH
from flow3d.data.utils import to_device
from flow3d.init_utils import (
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    vis_init_params,
    init_trainable_poses,
)
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from loguru import logger as guru
from utils.pcd_utils import export_pointcloud

import torch.nn.functional as F
from torch import Tensor
from gsplat.utils import normalized_quat_to_rotmat

# from torch_cluster import fps

def create_dynamic_splats_with_optimizers(
    parser: Parser,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    init_lifespan: float = 3.0,
    lifespan_range: float = 10.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    batch_size: int = 1,
    device: str = "cuda",
    model_type: str = "2dgs",
    color_activation: Literal["sigmoid", "relu"] = "sigmoid",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    dynamic_points = parser.dynamic_points
    dynamic_points_rgb = parser.dynamic_points_rgb

    M = dynamic_points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(dynamic_points, 4)[:, 1:] ** 2).mean(dim=-1)  # [M,]
    dist_avg = torch.sqrt(dist2_avg)
    # scale_dim = 2 if model_type == "2dgs" else 3
    scale_dim = 3
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, scale_dim)  # [M, 3]
    quats = torch.rand((M, 4))  # [M, 4]
    opacities = torch.logit(torch.full((M,), init_opacity))  # [M,]
    S = len(parser.images)
    lifespan = torch.logit(torch.full((M, ), init_lifespan / lifespan_range))  # [M, ]
    # nonrigidity = torch.full((M, ), 1.0)
    temporal_center = parser.temporal_center.float()
    fwd_velocity = parser.fwd_velocity_gs
    bwd_velocity = parser.bwd_velocity_gs
    
    # need to search the params 
    dynamic_gaussian_params = [
        ("means", torch.nn.Parameter(dynamic_points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-4),

        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),

        ("temporal_center", torch.nn.Parameter(temporal_center), 1e-4),
        ("lifespan_p", torch.nn.Parameter(lifespan), 5e-2), # very important !!!
        ("lifespan_m", torch.nn.Parameter(lifespan), 5e-2), 
        ("velocity_p", torch.nn.Parameter(fwd_velocity), 1e-5),
        ("velocity_m", torch.nn.Parameter(bwd_velocity), 1e-5),
        # ("colors", torch.nn.Parameter(torch.logit(dynamic_points_rgb)), 1e-2),
        # ("colors", torch.nn.Parameter(RGB2SH(dynamic_points_rgb)), 2.5e-3),
    ]
    if color_activation == "sigmoid":
        dynamic_gaussian_params.append(("colors", torch.nn.Parameter(torch.logit(dynamic_points_rgb)), 1e-2))
    elif color_activation == "sh_relu":
        dynamic_gaussian_params.append(("colors", torch.nn.Parameter(RGB2SH(dynamic_points_rgb)), 2.5e-3))
    
    dynamic_splats = torch.nn.ParameterDict({n: v for n, v, _ in dynamic_gaussian_params}).to(device)
    dynamic_gs_optimizers = {
        name: torch.optim.Adam(
            [{"params": dynamic_splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in dynamic_gaussian_params
    }
    return dynamic_splats, dynamic_gs_optimizers

def create_static_splats_with_optimizers(
    parser: Parser,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    batch_size: int = 1,
    device: str = "cuda",
    model_type:str = "2dgs",
    color_activation: Literal["sigmoid", "sh_relu"] = "sigmoid",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    static_points = parser.static_points    
    static_points_rgb = parser.static_points_rgb
    # static points 
    N = static_points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(static_points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    # scale_dim = 2 if model_type == "2dgs" else 3
    scale_dim = 3
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, scale_dim)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    
    static_gaussian_params = [
        # name, value, lr
        ("means", torch.nn.Parameter(static_points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3), # important !!!
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]
    if sh_degree > 0:
        static_colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        static_colors[:, 0, :] = RGB2SH(static_points_rgb)
        static_gaussian_params.append(("sh0", torch.nn.Parameter(static_colors[:, :1, :]), 2.5e-3))
        static_gaussian_params.append(("shN", torch.nn.Parameter(static_colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # static_gaussian_params.append(("colors", torch.nn.Parameter(torch.logit(static_points_rgb)), 1e-2))
        # static_gaussian_params.append(("colors", torch.nn.Parameter(RGB2SH(static_points_rgb)), 2.5e-3))
        if color_activation == "sigmoid":
            static_gaussian_params.append(("colors", torch.nn.Parameter(torch.logit(static_points_rgb)), 1e-2))
        elif color_activation == "sh_relu":
            static_gaussian_params.append(("colors", torch.nn.Parameter(RGB2SH(static_points_rgb)), 2.5e-3))

    static_splats = torch.nn.ParameterDict({n: v for n, v, _ in static_gaussian_params}).to(device)
    static_gs_optimizers = {
        name: torch.optim.Adam(
            [{"params": static_splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in static_gaussian_params
    }
    return static_splats, static_gs_optimizers

def create_splats_with_optimizers(
    parser: Parser,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    init_lifespan: float = 3.0,
    lifespan_range: float = 10.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    batch_size: int = 1,
    device: str = "cuda",
    model_type: str = "2dgs",
    color_activation: Literal["sigmoid", "relu"] = "sigmoid",
) -> Tuple[torch.nn.ParameterDict, torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer], Dict[str, torch.optim.Optimizer]]:
    dynamic_splats, dynamic_gs_optimizers = create_dynamic_splats_with_optimizers(
        parser,
        init_opacity,
        init_scale,
        init_lifespan,
        lifespan_range,
        scene_scale,
        sh_degree,
        batch_size,
        device,
        model_type,
        color_activation
    )
    static_splats, static_gs_optimizers = create_static_splats_with_optimizers(
        parser,
        init_opacity,
        init_scale,
        scene_scale,
        sh_degree,
        batch_size,
        device,
        model_type,
        color_activation
    )
    return dynamic_splats, static_splats, dynamic_gs_optimizers, static_gs_optimizers

# def create_anchor_points(
#     parser: Parser,
#     ratio: float = 0.1,
#     device: str = "cuda",
# ):
#     xyz = parser.dynamic_points
#     t = parser.temporal_center
#     velocity = parser.velocity_gs
#     batch = torch.zeros(xyz.shape[0], dtype=torch.long)
#     index = fps(xyz, batch, ratio=ratio, random_start=False)
#     anchor_points = xyz[index]
#     anchor_t = t[index]
#     anchor_velocity = velocity[index]
#     anchor_points = torch.cat([anchor_points, anchor_t.unsqueeze(-1)], dim=-1)
#     return anchor_points.to(device), anchor_velocity.to(device)

def squeeze_info(info, key_for_gradient="gradient_2dgs"):
    info_new = {}
    ignore_keys = set(['isect_ids', 'flatten_ids', 'isect_offsets'])
    keys_to_copy = set(info.keys()) - ignore_keys
    gradient_2dgs = PesudoTensor(info[key_for_gradient][:, 0], info[key_for_gradient].grad[:, 0])
    for key in keys_to_copy:
        if isinstance(info[key], torch.Tensor):
            info_new[key] = info[key].squeeze(1)
        else:
            info_new[key] = info[key]
    info_new[key_for_gradient] = gradient_2dgs
    return info_new

def split_info(info, splits, dim, key_for_gradient="gradient_2dgs"):
    info_new = [{} for _ in range(len(splits))]
    for key in info.keys():
        if isinstance(info[key], torch.Tensor) and key not in [key_for_gradient, "render_distort"]:
            # import pdb
            # pdb.set_trace()
            res = torch.split(info[key], splits, dim=dim)
            for i in range(len(splits)):
                info_new[i][key] = res[i]
        else:
            for i in range(len(splits)):
                info_new[i][key] = info[key]
    g_data, g_grad = info[key_for_gradient].data, info[key_for_gradient].grad
    g_data_split = torch.split(g_data, splits, dim=dim)
    g_grad_split = torch.split(g_grad, splits, dim=dim)
    for i in range(len(splits)):
        info_new[i][key_for_gradient] = PesudoTensor(g_data_split[i], g_grad_split[i])
    return info_new


def filter_means2d(means2d: torch.Tensor, mask: torch.Tensor):
    """Filter 2D Gaussian means by a mask. 

    Args:
        means2d (torch.Tensor): Float Tensor of Shape [b n 2]
        mask (torch.Tensor): Bool Tensor of Shape [b h w]
    Returns:
        means2d_mask  (torch.Tensor): bool Tensor of Shape [b n], True if means2d is in the mask else false
    """    
    b, n, _ = means2d.shape
    b_mask, h, w = mask.shape
    
    assert b == b_mask, f"Batch size mismatch: means2d has {b}, mask has {b_mask}"
    
    # Extract x, y coordinates
    x_coords_float = means2d[..., 0]  # [b, n]
    y_coords_float = means2d[..., 1]  # [b, n]
    
    # Check if coordinates are within bounds
    x_valid = (x_coords_float >= 0) & (x_coords_float < w)  # [b, n]
    y_valid = (y_coords_float >= 0) & (y_coords_float < h)  # [b, n]
    coords_valid = x_valid & y_valid  # [b, n]
    
    # Convert to integer coordinates for valid points only
    x_coords = x_coords_float.long()  # [b, n]
    y_coords = y_coords_float.long()  # [b, n]
    
    # Clamp to valid range (for safe indexing, even though we'll mask invalid ones)
    x_coords = torch.clamp(x_coords, 0, w - 1)
    y_coords = torch.clamp(y_coords, 0, h - 1)
    
    # Create batch indices for advanced indexing
    batch_indices = torch.arange(b, device=means2d.device).unsqueeze(1).expand(b, n)  # [b, n]
    
    # Sample mask values at the 2D coordinates
    mask_values = mask[batch_indices, y_coords, x_coords]  # [b, n]
    
    # Only consider points that are within bounds AND in the mask
    means2d_mask = coords_valid & mask_values  # [b, n]
    
    return means2d_mask

# def make_static(gaussians: Dict[str, torch.Tensor], info):
#     pass

# def make_dynamic(gaussians: Dict[str, torch.Tensor], info):
#     pass


def create_motion_base_splats(
    parser: Parser,
    num_fg: int,
    num_motion_bases: int,
    vis: bool = False,
    init_scale: float = 1.0,
    color_activation: Literal["sigmoid", "sh_relu"] = "sigmoid",
    port:int = 8881
):
    tracks_3d = TrackObservations(*parser.get_tracks_3d(num_fg))
    print(
        f"{tracks_3d.xyz.shape=} {tracks_3d.visibles.shape=} "
        f"{tracks_3d.invisibles.shape=} {tracks_3d.confidences.shape} "
        f"{tracks_3d.colors.shape}"
    )
    if not tracks_3d.check_sizes():
        import ipdb

        ipdb.set_trace()

    rot_type = "6d"
    cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item())
    valid_mask = tracks_3d.valid[:, cano_t]
    tracks_3d = tracks_3d.filter_valid(valid_mask)
    num_fg = tracks_3d.xyz.shape[0]

    guru.info(f"{cano_t=} {num_fg=} {num_motion_bases=}")
    # export canonical points 
    export_pointcloud(
        tracks_3d.xyz[:, cano_t],
        tracks_3d.colors,
        os.path.join(parser.cache_data_path, f"canonical_points_{num_fg}_pts.ply"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port
    )
    motion_bases = motion_bases.to(device)

    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs, init_scale, color_activation)
    fg_params = fg_params.to(device)

    tracks_3d = tracks_3d.to(device)
    return fg_params, motion_bases, tracks_3d

@torch.no_grad()
def add_new_gs(
    new_gaussians: Dict[str, Tensor],
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
):
    num_new_gs = len(new_gaussians["means"])
    device = params["means"].device

    def param_fn(name: str, p: Tensor) -> Tensor:
        p_new = torch.cat([p, new_gaussians[name]], dim=0)
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_add = torch.zeros((num_new_gs, *v.shape[1:]), device=device, dtype=v.dtype)
        return torch.cat([v, v_add], dim=0)

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v_add = torch.zeros((num_new_gs, *v.shape[1:]), device=device, dtype=v.dtype)
            state[k] = torch.cat((v, v_add), dim=0)

@torch.no_grad()
def _update_param_with_optimizer(
    param_fn: Callable[[str, Tensor], Tensor],
    optimizer_fn: Callable[[str, Tensor], Tensor],
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    names: Union[List[str], None] = None,
):
    """Update the parameters and the state in the optimizers with defined functions.

    Args:
        param_fn: A function that takes the name of the parameter and the parameter itself,
            and returns the new parameter.
        optimizer_fn: A function that takes the key of the optimizer state and the state value,
            and returns the new state value.
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        names: A list of key names to update. If None, update all. Default: None.
    """
    if names is None:
        # If names is not provided, update all parameters
        names = list(params.keys())

    for name in names:
        param = params[name]
        new_param = param_fn(name, param)
        params[name] = new_param
        if name not in optimizers:
            assert not param.requires_grad, (
                f"Optimizer for {name} is not found, but the parameter is trainable."
                f"Got requires_grad={param.requires_grad}"
            )
            continue
        optimizer = optimizers[name]
        for i in range(len(optimizer.param_groups)):
            param_state = optimizer.state[param]
            del optimizer.state[param]
            for key in param_state.keys():
                if key != "step":
                    v = param_state[key]
                    param_state[key] = optimizer_fn(key, v)
            optimizer.param_groups[i]["params"] = [new_param]
            optimizer.state[new_param] = param_state