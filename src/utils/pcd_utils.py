import torch
import trimesh
import numpy as np
import open3d as o3d
from torch_scatter import scatter_add, scatter_mean

def preprocess_pointcloud_3d(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    voxel_size: float = 0.01,
    max_points: int = 50_000
):
    """
    Preprocess point cloud by downsampling and removing outliers using Open3D.
    
    Args:
        xyz (torch.Tensor): Point cloud coordinates of shape (N, 3).
        rgb (torch.Tensor): Point cloud colors of shape (N, 3), values in [0, 1].
        voxel_size (float): Voxel size for downsampling.
        
    Returns:
        tuple: (downsampled_xyz, downsampled_rgb) as torch tensors.
    """
    assert xyz.shape[0] == rgb.shape[0], "Number of points and colors must match."
    assert xyz.shape[1] == 3, "Point cloud must have shape (N, 3)."
    assert rgb.shape[1] == 3, "Colors must have shape (N, 3)."
    
    # Convert to numpy
    device = xyz.device
    xyz_np = xyz.cpu().numpy().astype(np.float64)
    rgb_np = rgb.cpu().numpy().astype(np.float64)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np)
    pcd.colors = o3d.utility.Vector3dVector(rgb_np)
    
    # Voxel downsampling first
    print(f"3D Point cloud before downsampling: {len(pcd.points)} points")
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"3D Point cloud after downsampling: {len(pcd_downsampled.points)} points")

    # Remove statistical outliers after downsampling
    pcd_clean, _ = pcd_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"3D Point cloud after outlier removal: {len(pcd_clean.points)} points")
    
    # Convert back to torch tensors
    xyz_processed = torch.from_numpy(np.asarray(pcd_clean.points)).float().to(device)
    rgb_processed = torch.from_numpy(np.asarray(pcd_clean.colors)).float().to(device)

    # Limit to max_points if necessary
    if len(xyz_processed) > max_points:
        indices = torch.randperm(len(xyz_processed))[:max_points]
        xyz_processed = xyz_processed[indices]
        rgb_processed = rgb_processed[indices]
        print(f"3D Point cloud after limiting to max points: {len(xyz_processed)} points")
    
    return xyz_processed, rgb_processed

def voxelization_4d(xyz, t, feat, spatial_voxel_size, temporal_voxel_size):
    """
    Voxelization of 4D point cloud (3D + time) with feature aggregation
    use softmax weights based on distance to voxel center(4D).
    Args:
        xyz (torch.Tensor): Point cloud coordinates of shape (N, 3).
        t (torch.Tensor): Temporal indices of shape (N,).
        feat (torch.Tensor): Point features of shape (N, feat_dim).
        spatial_voxel_size (float): Voxel size for spatial dimensions.
        temporal_voxel_size (float): Voxel size for temporal dimension.
    Returns:
        tuple: (voxelized_xyz, voxelized_t, voxelized_feat) as torch tensors.
    """
    voxel_size = torch.tensor([
        spatial_voxel_size,
        spatial_voxel_size,
        spatial_voxel_size,
        temporal_voxel_size
    ], device=xyz.device)
    xyzt = torch.cat([xyz, t.unsqueeze(-1).float()], dim=-1)  # [N, 4]

    voxel_indices = (xyzt / voxel_size).round().int()  # [N, 4]
    unique_voxels, inverse_indices, counts = torch.unique(
        voxel_indices, dim=0, return_inverse=True, return_counts=True
    )
    # compute weights 
    voxel_xyzt_mean = scatter_mean(xyzt, inverse_indices, dim=0)
    xyzt_remap = voxel_xyzt_mean[inverse_indices]
    xyzt_var = (xyzt - xyzt_remap) ** 2
    voxel_xyzt_var = scatter_mean(xyzt_var, inverse_indices, dim=0)
    exp_weights = - torch.sum((xyzt - xyzt_remap)**2 / (voxel_xyzt_var[inverse_indices] + 1e-6), dim=-1)
    xyzt_weights = torch.exp(exp_weights) # [N, 4]

    voxel_weights = scatter_add(
        xyzt_weights, inverse_indices, dim=0
    )  # [num_unique_voxels]
    weights = (xyzt_weights / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(
        -1
    )  # [N, 1]
    # Compute weighted average of positions and features
    xyz_t_feat = torch.cat([xyzt, feat], dim=-1)  # [N, 4 + feat_dim]
    weighted_feat = xyz_t_feat * weights  # [N, feat_dim]
    # Aggregate per voxel
    voxel_feat = scatter_add(
       weighted_feat, inverse_indices, dim=0
    )  # [num_unique_voxels, feat_dim]
    feat_dim = feat.shape[-1]
    xyz, t, feat = torch.split(
        voxel_feat, [3, 1, feat_dim], dim=-1
    )
    return xyz, t.squeeze(-1), feat


def preprocess_pointcloud_4d(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    fwd_v: torch.Tensor,
    bwd_v: torch.Tensor,
    t: torch.Tensor,
    spatial_voxel_size: float = 0.01,
    temporal_voxel_size: float = 2.0
):
    """
    Preprocess 4D point cloud (3D + time) by downsampling and removing outliers using Open3D.
    
    Args:
        xyz (torch.Tensor): Point cloud coordinates of shape (N, 3).
        rgb (torch.Tensor): Point cloud colors of shape (N, 3), values in [0, 1].
        v (torch.Tensor): Velocity vectors of shape (N, 3).
        t (torch.Tensor): Temporal indices of shape (N,).
        voxel_size (float): Voxel size for downsampling.

    Returns:
        tuple: (downsampled_xyz, downsampled_rgb, downsampled_v, downsampled_t) as torch tensors.
    """
    assert xyz.shape[0] == rgb.shape[0] == t.shape[0], "Number of points, colors, velocities, and times must match."
    assert xyz.shape[1] == 3, "Point cloud must have shape (N, 3)."
    assert rgb.shape[1] == 3, "Colors must have shape (N, 3)."
    feat = torch.cat([rgb, fwd_v, bwd_v], dim=-1)  # [N, 6]
    print(f"4D Point cloud before voxelization: {len(xyz)} points")
    xyz, t, feat = voxelization_4d(xyz, t, feat, spatial_voxel_size, temporal_voxel_size)
    print(f"4D Point cloud after voxelization: {len(xyz)} points")
    rgb, fwd_v, bwd_v = torch.split(feat, [3, 3, 3], dim=-1)
    xyz_np = xyz.cpu().numpy().astype(np.float64)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"4D Point cloud after outlier removal: {len(ind)} points")
    ind = torch.from_numpy(np.asarray(ind)).long().to(xyz.device)
    xyz = xyz[ind]
    rgb = rgb[ind]
    fwd_v = fwd_v[ind]
    bwd_v = bwd_v[ind]
    t = t[ind]
    return xyz, rgb, fwd_v, bwd_v, t

    
def export_pointcloud(xyz:torch.Tensor, rgb:torch.Tensor, filename:str):
    """
    Export point cloud to .ply file.

    Args:
        xyz (torch.Tensor): Point cloud coordinates of shape (N, 3).
        rgb (torch.Tensor): Point cloud colors of shape (N, 3), values in [0, 1].
        filename (str): Output .ply file path.
    """
    assert xyz.shape[0] == rgb.shape[0], "Number of points and colors must match."
    assert xyz.shape[1] == 3, "Point cloud must have shape (N, 3)."
    assert rgb.shape[1] == 3, "Colors must have shape (N, 3)."
    
    # Convert to numpy
    xyz_np = xyz.cpu().numpy()
    rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)

    # Create a Trimesh PointCloud object
    point_cloud = trimesh.points.PointCloud(vertices=xyz_np, colors=rgb_np)

    # Export to .ply file
    point_cloud.export(file_obj=filename, file_type='ply')
