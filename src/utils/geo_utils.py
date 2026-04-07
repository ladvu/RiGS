import torch
from utils.cam_utils import closed_form_inverse_se3
from einops import rearrange, einsum
import cv2
from torch.nn import functional  as F
import numpy as np


def depth_map_to_cam_coords(depth_map: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (torch.Tensor): Depth map of shape (..., H, W).
        intrinsics (torch.Tensor): Camera intrinsics matrix of shape (..., 3, 3).

    Returns:
        torch.Tensor: Camera coordinates of shape (..., H, W, 3).
    """
    H, W = depth_map.shape[-2:]
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    v = v.to(depth_map.device).float().unsqueeze(0).expand(depth_map.shape)
    u = u.to(depth_map.device).float().unsqueeze(0).expand(depth_map.shape)

    fx = rearrange(intrinsics[..., 0, 0], "... -> ... 1 1")
    fy = rearrange(intrinsics[..., 1, 1], "... -> ... 1 1")
    cx = rearrange(intrinsics[..., 0, 2], "... -> ... 1 1")
    cy = rearrange(intrinsics[..., 1, 2], "... -> ... 1 1")

    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    cam_coords = torch.stack((x, y, z), dim=-1)  # (H, W, 3)
    return cam_coords

def depth_map_to_world_coords(depth_map:torch.Tensor, intrinsics:torch.Tensor, extrinsics:torch.Tensor) -> torch.Tensor:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (torch.Tensor): Depth map of shape (..., H, W).
        intrinsics (torch.Tensor): Camera intrinsics matrix of shape (..., 3, 3).
        extrinsics (torch.Tensor): Camera extrinsics matrix of shape (..., 4, 4).

    Returns:
        torch.Tensor: World coordinates of shape (..., H, W, 3).
    """
    cam_coords = depth_map_to_cam_coords(depth_map, intrinsics)  # (..., H, W, 3)
    H, W = depth_map.shape[-2:]

    # Convert cam_coords to homogeneous coordinates
    ones = torch.ones((*cam_coords.shape[:-1], 1), device=cam_coords.device)
    cam_coords_hom = torch.cat((cam_coords, ones), dim=-1)  # (..., H, W, 4)

    c2w_mats = closed_form_inverse_se3(extrinsics)  # (..., 4, 4)
    # Apply extrinsics to get world coordinates
    world_coords_hom = einsum(c2w_mats, cam_coords_hom, "... i j, ... h w j -> ... h w i")

    # Convert back to Cartesian coordinates
    world_coords = world_coords_hom[..., :3] / world_coords_hom[..., 3:]

    return world_coords


def warp_np(flow: np.ndarray, image: np.ndarray):
    """
    Warp an image using optical flow with OpenCV's remap function.
    
    Args:
        flow: optical flow array of shape (H, W, 2) containing (dx, dy) displacements
        image: input image of shape (H, W, C) or (H, W) for grayscale
        
    Returns:
        warped image with same shape as input image
    """
    h, w = flow.shape[:2]
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Add flow to get destination coordinates
    map_x = (x_coords + flow[..., 0]).astype(np.float32)
    map_y = (y_coords + flow[..., 1]).astype(np.float32)
    
    # Use cv2.remap to warp the image
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return warped


def warp_tensor(flow: torch.Tensor, image: torch.Tensor):
    """
    Warp an image using optical flow with PyTorch's grid_sample function.
    
    Args:
        flow: optical flow tensor of shape (B, H, W, 2) containing (dx, dy) displacements
        image: input image tensor of shape (B, H, W, C)
        
    Returns:
        warped image tensor with same shape as input image
    """
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    B, H, W, _ = image.shape
    
    # Create normalized coordinate grids
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    uv = torch.stack((u, v), dim=-1).float().to(flow.device)  # (H, W, 2)
    uv = uv.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    grid = uv + flow

    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0  # Normalize to [-1, 1]
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0  # Normalize to [-1, 1]

    warped = torch.nn.functional.grid_sample(
        image.permute(0, 3, 1, 2),  # (B, C, H, W)
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).permute(0, 2, 3, 1)  # (B, H, W, C)

    return warped

def project(world_points: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Project 3D world points to 2D pixel coordinates.

    Args:
        world_points (torch.Tensor): 3D world points of shape (..., 3).
        intrinsics (torch.Tensor): Camera intrinsics matrix of shape (..., 3, 3).
        extrinsics (torch.Tensor): Camera extrinsics matrix of shape (..., 4, 4).

    Returns:
        torch.Tensor: 2D pixel coordinates of shape (..., 2).
    """
    # Convert world points to homogeneous coordinates
    ones = torch.ones((*world_points.shape[:-1], 1), device=world_points.device)
    world_points_hom = torch.cat((world_points, ones), dim=-1)  # (..., N, 4)

    cam_points_hom = einsum(extrinsics, world_points_hom, "... i j, ... n j -> ... n i")  # (..., N, 4)
    cam_points = cam_points_hom[..., :3] / cam_points_hom[..., 3:] 

    # Project camera coordinates to pixel coordinates
    pixel_coords_hom = einsum(intrinsics, cam_points, "... i j, ... n j -> ... n i")  # (..., N, 3)
    pixel_coords = pixel_coords_hom[..., :2] / pixel_coords_hom[..., 2:]  # (..., N, 2)

    return pixel_coords

def get_uv_grid(H:int, W:int):
    v, u = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing='ij'
    )
    uv = torch.stack([u, v], axis=-1).float()
    return uv  

def compute_pesudo_flow(
    world_points: torch.Tensor, 
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
):
    B, S, H, W, C = world_points.shape
    world_points = rearrange(world_points, "B S H W C -> B S (H W) C")
    uv_reproj = project(
        world_points,
        intrinsics,
        extrinsics
    )
    uv_reproj = rearrange(uv_reproj, "B S (H W) C -> B S H W C", H=H, W=W)
    uv = get_uv_grid(H, W).to(uv_reproj.device)
    return uv_reproj - uv

def compute_flow_masks(
    fwd_flows: torch.Tensor, 
    bwd_flows: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5
):

    bwd2fwd_flow = warp_tensor(fwd_flows, bwd_flows)
    fwd_lr_error = torch.norm(fwd_flows + bwd2fwd_flow, dim=-1)
    fwd_mask = (
        fwd_lr_error
        < alpha
        * (torch.norm(fwd_flows, dim=-1) + torch.norm(bwd2fwd_flow, dim=-1))
        + beta
    )

    fwd2bwd_flow = warp_tensor(bwd_flows, fwd_flows)
    bwd_lr_error = torch.norm(bwd_flows + fwd2bwd_flow, dim=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha
        * (torch.norm(bwd_flows, dim=-1) + torch.norm(fwd2bwd_flow, dim=-1))
        + beta
    )
    return fwd_mask, bwd_mask

def compute_dynamic_masks(
    fwd_flows: torch.Tensor, 
    bwd_flows: torch.Tensor,
    world_points: torch.Tensor, 
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    alpha: float = 0.25,
    dynamic_thres: float = 0.5,
    static_thres: float = 0.1
    # scaling: torch.Tensor = None,
):
    # Add batch dimension if missing
    if fwd_flows.dim() == 4:
        fwd_flows = fwd_flows.unsqueeze(0)
    if bwd_flows.dim() == 4:
        bwd_flows = bwd_flows.unsqueeze(0)
    if world_points.dim() == 4:
        world_points = world_points.unsqueeze(0)
    if intrinsics.dim() == 3:
        intrinsics = intrinsics.unsqueeze(0)
    if extrinsics.dim() == 3:
        extrinsics = extrinsics.unsqueeze(0)

    pesudo_fwd_flow = compute_pesudo_flow(
        world_points[:,:-1],
        intrinsics[:, 1:],
        extrinsics[:, 1:],
    )
    pesudo_bwd_flow = compute_pesudo_flow(
        world_points[:, 1:],
        intrinsics[:, :-1],
        extrinsics[:, :-1],
    )
    fwd_diff = torch.norm(fwd_flows[:, :-1] - pesudo_fwd_flow, dim=-1)
    fwd_rel = alpha * (torch.norm(fwd_flows[:, :-1], dim=-1) + torch.norm(pesudo_fwd_flow, dim=-1))
    fwd_dyn_mask = fwd_diff > fwd_rel + dynamic_thres
    fwd_stat_mask = fwd_diff < fwd_rel + static_thres

    bwd_diff = torch.norm(bwd_flows[:, 1:] - pesudo_bwd_flow, dim=-1)
    bwd_rel = alpha * (torch.norm(bwd_flows[:, 1:], dim=-1) + torch.norm(pesudo_bwd_flow, dim=-1))
    bwd_dyn_mask = bwd_diff > bwd_rel + dynamic_thres
    bwd_stat_mask = bwd_diff < bwd_rel + static_thres

    dyn_mask = torch.cat([fwd_dyn_mask[:, :1], fwd_dyn_mask[:, 1:] | bwd_dyn_mask[:, :-1], bwd_dyn_mask[:, -1:]], dim=1)
    stat_mask = torch.cat([fwd_stat_mask[:, :1], fwd_stat_mask[:, 1:] & bwd_stat_mask[:, :-1], bwd_stat_mask[:, -1:]], dim=1)

    return dyn_mask, stat_mask

def compute_velocity(
    world_points: torch.Tensor,
    fwd_flows: torch.Tensor, fwd_flow_masks: torch.Tensor,
    bwd_flows: torch.Tensor, bwd_flow_masks: torch.Tensor,
    point_masks: torch.Tensor,
    scale: float = None,
):
    if fwd_flows.dim() == 4:
        fwd_flows = fwd_flows.unsqueeze(0)
    if bwd_flows.dim() == 4:
        bwd_flows = bwd_flows.unsqueeze(0)
    if world_points.dim() == 4:
        world_points = world_points.unsqueeze(0)
    if point_masks.dim() == 3:
        point_masks = point_masks.unsqueeze(0)

    b = world_points.shape[0]
    world_points = torch.cat([world_points, point_masks[..., None]], dim=-1) # (B, S, H, W, 4) # warp once
    points_p1 = world_points[:, 1:].flatten(0, 1)
    points_m1 = world_points[:, :-1].flatten(0, 1)

    fwd_points = warp_tensor(fwd_flows[:, :-1].flatten(0, 1), points_p1)
    fwd_points, fwd_masks = torch.split(fwd_points, (3, 1), dim=-1)
    fwd_masks = fwd_masks.squeeze(-1) > 0.5
    fwd_velocity = fwd_points - points_m1[..., :3]
    bwd_points = warp_tensor(bwd_flows[:, 1:].flatten(0, 1), points_m1)
    bwd_points, bwd_masks = torch.split(bwd_points, (3, 1), dim=-1)
    bwd_masks = bwd_masks.squeeze(-1) > 0.5
    bwd_velocity = points_p1[..., :3] - bwd_points

    fwd_velocity = rearrange(fwd_velocity, "(b s) h w c -> b s h w c", b=b)
    bwd_velocity = rearrange(bwd_velocity, "(b s) h w c -> b s h w c", b=b)
    fwd_masks = rearrange(fwd_masks, "(b s) h w -> b s h w", b=b) 
    bwd_masks = rearrange(bwd_masks, "(b s) h w -> b s h w", b=b) 
    

    fwd_velocity = F.pad(fwd_velocity, (0,0, 0,0, 0,0, 0,1), mode='constant', value=0.0)
    bwd_velocity = F.pad(bwd_velocity, (0,0, 0,0, 0,0, 1,0), mode='constant', value=0.0)
    fwd_masks = F.pad(fwd_masks, (0,0, 0,0, 0,1), mode='constant', value=False) & fwd_flow_masks
    bwd_masks = F.pad(bwd_masks, (0,0, 0,0, 1,0), mode='constant', value=False) & bwd_flow_masks
    # for init
    fwd_velocity[~fwd_masks] = 0.0
    bwd_velocity[~bwd_masks] = 0.0


    return fwd_velocity, fwd_masks, bwd_velocity, bwd_masks

def compute_sampson_error(x1, x2, F):
    """
    :param x1 (*, N, 2)
    :param x2 (*, N, 2)
    :param F (*, 3, 3)
    """
    h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)
    h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)
    d1 = torch.matmul(h1, F.transpose(-1, -2))  # (B, N, 3)
    d2 = torch.matmul(h2, F)  # (B, N, 3)
    z = (h2 * d1).sum(dim=-1)  # (B, N)
    err = torch.sqrt(z**2 / (
        d1[..., 0] ** 2 + d1[..., 1] ** 2 + d2[..., 0] ** 2 + d2[..., 1] ** 2
    ))
    return err

