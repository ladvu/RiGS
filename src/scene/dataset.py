import json
import os
from typing import *
import os.path as osp
import gc
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from utils import (
    load_vipe_data,
    read_pose_artifacts,
    read_intrinsics_artifacts,
    closed_form_inverse_se3,
    preprocess_pointcloud_3d,
    preprocess_pointcloud_4d,
    depth_map_to_world_coords,
    compute_velocity,
    load_cache_data,
    save_cache_data,
    find_existing_cache,
    generate_cache_path,
    erode,
    laplacian_filter_depth,
    parse_tapir_track_info,
    normalize_coords,
    export_pointcloud,
    rt_to_mat4,
    torch_quantile
)
import torch.nn.functional as F
from torch.nn.functional import interpolate
from einops import reduce, repeat, einsum
from tqdm import tqdm
import matplotlib.pyplot as plt
import roma

class SceneNormDict(TypedDict):
    scale: float
    transfm: torch.Tensor


class Parser:

    def __init__(
        self,
        data_dir: str,
        name: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 1,
        chunk_size: int = 100,
        use_gpu: bool = True,
        cache_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        depth_thres: float = 0.5,
        depth_min: float = 1e-3,
        depth_max: float = 1000.0,
        cache_force_reload: bool = False,
        depth_noise_scale: float = 0.0,
    ):
        self.data_dir = data_dir
        self.data_name = name
        # val_dir = val_dir if val_dir is not None else data_dir
        self.val_dir = val_dir
        self.test_every = test_every
        self.factor = factor

        # Check if cache exists and load from cache
        cache_data_path = None
        if cache_dir is not None:
            cache_data_path = find_existing_cache(cache_dir, name, None)
        if not cache_force_reload and cache_data_path is not None:
            # First try to find existing cache with matching data_dir
            print(f"Loading data from existing cache: {cache_data_path}")
            self.cache_data_path = cache_data_path
            cached_data = load_cache_data(cache_data_path)
            if cached_data is not None:
                self._load_from_cache(cached_data)
                self.loaded = True
                S, H, W, _ = self.images.shape
                self.num_frames = S
                self.original_H = H * factor
                self.original_W = W * factor
                return
        # Save to cache if cache_dir is provided
        self.loaded = False
        if cache_data_path is None:
            cache_data_path = generate_cache_path(cache_dir, name, data_dir)
        self.cache_data_path = cache_data_path
        
        print(f"Cache not found or invalid, processing from source data...")
        data = load_vipe_data(data_dir, name)
        (
            intrinsics,
            poses,
            images,
            depths,
            fwd_flows, fwd_masks,
            bwd_flows, bwd_masks,
            dynamic_masks,
            query_tracks_2d,
        ) = (
            data["intrinsics"],
            data["poses"],
            data["rgb"],
            data["depths"],
            data["fwd_flows"], data["fwd_masks"],
            data["bwd_flows"], data["bwd_masks"],
            data["foreground_masks"],
            data["query_tracks_2d"],
        )
        del data
        noise_scale = torch.std(depths) * depth_noise_scale
        should_add_noise = torch.rand_like(depths) < 0.25
        depths = depths + torch.randn_like(depths) * noise_scale * should_add_noise.float()
        S, H, W, _ = images.shape
        self.num_frames = S
        self.original_H = H
        self.original_W = W 
        H, W = H // factor, W // factor
        time_indices = torch.arange(0, S)
        self.time_indices = time_indices
        self.query_tracks_2d = query_tracks_2d
        
        # data
        self.camtoworlds = torch.empty(S, 4, 4) 
        self.intrinsics = torch.empty(S, 3, 3)
        self.extrinsics = torch.empty(S, 4, 4)

        # supervision
        self.images = torch.empty(S, H, W, 3)
        self.depths = torch.empty(S, H, W)

        self.depths_valid = torch.empty(S, H, W, dtype=torch.bool)
        self.fwd_velocity_masks = torch.empty(S, H, W, dtype=torch.bool)
        self.bwd_velocity_masks = torch.empty(S, H, W, dtype=torch.bool)

        self.fwd_velocity_map = torch.empty(S, H, W, 3)
        self.bwd_velocity_map = torch.empty(S, H, W, 3)
        self.points_map = torch.empty(S, H, W, 3)

        self.dynamic_masks = torch.full((S, H, W), False, dtype=torch.bool) # (N, H, W)
        self.static_masks = torch.full((S, H, W), True, dtype=torch.bool)

        device = "cuda" if use_gpu else "cpu"
        for i in tqdm(range(0, S, chunk_size)):
            start_idx = i
            end_idx = min(i + chunk_size, S)
            processed_batch = self.process_batch(
                intrinsics=intrinsics[start_idx:end_idx].to(device),
                poses=poses[start_idx:end_idx].to(device),
                images=images[start_idx:end_idx].to(device),
                depths=depths[start_idx:end_idx].to(device),
                fwd_flows=fwd_flows[start_idx:end_idx].to(device), fwd_masks=fwd_masks[start_idx:end_idx].to(device),
                bwd_flows=bwd_flows[start_idx:end_idx].to(device), bwd_masks=bwd_masks[start_idx:end_idx].to(device),
                dynamic_masks=dynamic_masks[start_idx:end_idx].to(device),
                factor=factor,
                depth_thres=depth_thres,
                depth_min=depth_min,
                depth_max=depth_max
            )
            self.camtoworlds[start_idx:end_idx] = processed_batch["camtoworlds"].cpu()
            self.intrinsics[start_idx:end_idx] = processed_batch["intrinsics"].cpu()
            self.extrinsics[start_idx:end_idx] = processed_batch["extrinsics"].cpu()

            self.images[start_idx:end_idx] = processed_batch["images"].cpu()
            self.depths[start_idx:end_idx] = processed_batch["depths"].cpu()

            self.fwd_velocity_map[start_idx:end_idx] = processed_batch["fwd_velocity_map"].cpu()
            self.bwd_velocity_map[start_idx:end_idx] = processed_batch["bwd_velocity_map"].cpu()
            self.fwd_velocity_masks[start_idx:end_idx] = processed_batch["fwd_velocity_masks"].cpu()
            self.bwd_velocity_masks[start_idx:end_idx] = processed_batch["bwd_velocity_masks"].cpu()

            self.dynamic_masks[start_idx:end_idx] = processed_batch["dynamic_masks"].cpu()
            self.static_masks[start_idx:end_idx] = processed_batch["static_masks"].cpu()
            self.depths_valid[start_idx:end_idx] = processed_batch["depths_valid"].cpu()

            self.points_map[start_idx:end_idx] = processed_batch["points_map"].cpu()

        self.normalize_transform = np.eye(4)
        self.normalize_scale = 1.0

        if val_dir is not None:
            self.load_val_data(val_dir, name)
        else:
            print("no val data provided. default to sampling from training data.")
            if self.test_every > 1:
                self.val_images = self.images[::self.test_every]
                self.val_camtoworlds = self.camtoworlds[::self.test_every]
                self.val_intrinsics = self.intrinsics[::self.test_every]
                self.val_time_indices = self.time_indices[::self.test_every]
                self.val_masks = torch.ones_like(self.dynamic_masks[::self.test_every])
            else:
                print("test_every is 1, using all frames as val set.")
                self.val_images = self.images
                self.val_camtoworlds = self.camtoworlds
                self.val_intrinsics = self.intrinsics
                self.val_time_indices = self.time_indices
                self.val_masks = torch.ones_like(self.dynamic_masks)
            
            self.val_cam_ids = torch.arange(1, len(self.val_time_indices)+1)
            self.num_val_cams = len(self.val_cam_ids)

        gc.collect()

    def _load_from_cache(self, cached_data: Dict[str, Any]):
        """Load parser attributes from cached data."""
        for key in cached_data:
            setattr(self, key, cached_data[key])

    def _save_to_cache(self, cache_path: str):
        """Save parser attributes to cache."""
        extra_keys = ["data_dir", "val_dir", "test_every", "scene_scale", "d_scale", "query_tracks_2d", "factor", "num_val_cams", "normalize_transform", "normalize_scale"]
        cache_data = {}
        for key in self.__dict__:
            if key in extra_keys or isinstance(getattr(self, key), torch.Tensor):
                cache_data[key] = getattr(self, key)
        
        save_cache_data(cache_path, cache_data, meta_keys=["data_dir", "val_dir", "test_every", "scene_scale", "d_scale", "factor", "num_val_cams"])

    def load_val_data(self, val_dir:str, name:str):
        # load images, intrinsics, camtoworlds
        # load images
        rgb_dir = os.path.join(val_dir, "val_images", name)
        image_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        num_val = len(image_files)
        assert num_val > 0, f"No images found in {rgb_dir}"
        images = []


        for i, image_file in enumerate(image_files):
            image = Image.open(image_file).convert("RGB")
            # image = image.resize((self.images.shape[2], self.images.shape[1]), Image.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            images.append(torch.from_numpy(image))
        self.val_images = torch.stack(images, dim=0)  # (N, H, W, 3)
        self.val_images = interpolate(
            self.val_images.permute(0, 3, 1, 2), 
            size=(self.original_H // self.factor, self.original_W // self.factor),
            mode="bilinear",
            align_corners=False
        ).permute(0, 2, 3, 1)

        mask_dir = os.path.join(val_dir, "val_masks", name)
        if os.path.exists(mask_dir):
            mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
            val_masks = []
            for i, mask_file in enumerate(mask_files):
                mask = Image.open(mask_file).convert("L")
                mask = mask.resize((self.images.shape[2], self.images.shape[1]), Image.NEAREST)
                mask = np.array(mask).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask) > 0.5
                val_masks.append(mask)
            self.val_masks = torch.stack(val_masks, dim=0)  # (N, H, W)
        else:
            self.val_masks = torch.ones((num_val, self.images.shape[1], self.images.shape[2]), dtype=torch.bool)

        # load pose
        pose_dir = os.path.join(val_dir, "val_pose", name + ".npz")
        if os.path.exists(pose_dir):
            self.val_camtoworlds = read_pose_artifacts(pose_dir)
        else:
            self.val_camtoworlds = self.camtoworlds[:1].repeat(num_val, 1, 1)
        # load intrinsics
        intri_dir = os.path.join(val_dir, "val_intrinsics", name + ".npz")
        if os.path.exists(intri_dir):
            self.val_intrinsics = read_intrinsics_artifacts(intri_dir)
            self.val_intrinsics[:, 0] = self.val_intrinsics[:, 0] / self.factor
            self.val_intrinsics[:, 1] = self.val_intrinsics[:, 1] / self.factor
        else:
            self.val_intrinsics = self.intrinsics[:1].repeat(num_val, 1, 1)

        time_indices_path = os.path.join(val_dir, "val_time", name + ".npy")
        if os.path.exists(time_indices_path):
            self.val_time_indices = torch.from_numpy(np.load(time_indices_path))
        else:
            self.val_time_indices = torch.arange(0, num_val)

        cam_ids_path = os.path.join(val_dir, "val_cam_ids", name + '.npy')
        if os.path.exists(cam_ids_path):
            self.val_cam_ids = torch.from_numpy(np.load(cam_ids_path))
            self.num_val_cams = self.val_cam_ids.max().item()
        else:
            self.val_cam_ids = torch.ones_like(self.val_time_indices)
            self.num_val_cams = 1


    def process_batch(
        self,
        intrinsics: torch.Tensor,
        poses: torch.Tensor,
        images: torch.Tensor,
        depths: torch.Tensor,
        fwd_flows: torch.Tensor, fwd_masks: torch.Tensor,
        bwd_flows: torch.Tensor, bwd_masks: torch.Tensor,
        dynamic_masks:torch.Tensor,
        factor: int = 1,
        depth_thres: float = 0.5,
        depth_min: float = 1e-3,
        depth_max: float = 1000.0,
    ):
        extrinsics = closed_form_inverse_se3(poses)
        S, H, W = images.shape[:3]
        depths_valid = laplacian_filter_depth(depths, depth_thres, 5) & (depths > depth_min) & (depths < depth_max)
        if factor != 1:
            target_shape = (H // factor, W // factor)
            scale = (H / target_shape[0], W / target_shape[1])
            # downsample
            images = interpolate(
                images.permute(0, 3, 1, 2), 
                size=target_shape,
                mode="bilinear",
            ).permute(0, 2, 3, 1)
            depths = interpolate(
                depths.unsqueeze(1),
                size=target_shape,
                mode="nearest",
            )[:, 0]
            depths_valid = interpolate(
                depths_valid.unsqueeze(1).float(),
                size=target_shape,
                mode="nearest"
            )[:, 0].bool() 
            fwd_flows = interpolate(
                fwd_flows.permute(0, 3, 1, 2),
                size=target_shape,
                mode="bilinear",
            ).permute(0, 2, 3, 1)
            bwd_flows = interpolate(
                bwd_flows.permute(0, 3, 1, 2),
                size=target_shape,
                mode="bilinear",
            ).permute(0, 2, 3, 1)
            dynamic_masks = interpolate(
                dynamic_masks.unsqueeze(1).float(),
                size=target_shape,
                mode="nearest"
            )[:, 0].bool() 
            fwd_masks = interpolate(
                fwd_masks.unsqueeze(1).float(),
                size=target_shape,
                mode="nearest"
            )[:, 0].bool() 
            bwd_masks = interpolate(
                bwd_masks.unsqueeze(1).float(),
                size=target_shape,
                mode="nearest"
            )[:, 0].bool() 
            fwd_flows[..., 0] = fwd_flows[..., 0] / scale[1]
            bwd_flows[..., 0] = bwd_flows[..., 0] / scale[1]
            fwd_flows[..., 1] = fwd_flows[..., 1] / scale[0]
            bwd_flows[..., 1] = bwd_flows[..., 1] / scale[0]
            intrinsics[:, 0] /= scale[1]  # scale fx, fy, cx, cy
            intrinsics[:, 1] /= scale[0]

        static_masks = ~dynamic_masks
        # depths = fix_sky_depth(depths)
        # process point cloud
        points = depth_map_to_world_coords(depths, intrinsics, extrinsics)
        # compute velocity
        (
            fwd_velocity_map,
            fwd_velocity_masks,
            bwd_velocity_map,
            bwd_velocity_masks
        ) = compute_velocity(
            points,
            fwd_flows, fwd_masks, 
            bwd_flows, bwd_masks,
            depths_valid,
        )
        fwd_velocity_map = fwd_velocity_map[0]
        bwd_velocity_map = bwd_velocity_map[0]
        fwd_velocity_masks = fwd_velocity_masks[0] & dynamic_masks
        bwd_velocity_masks = bwd_velocity_masks[0] & dynamic_masks

        return dict(
            camtoworlds = poses,  #  (S-2, 4, 4)
            intrinsics = intrinsics,
            extrinsics = extrinsics,
            images = images,
            depths = depths,
            fwd_velocity_map = fwd_velocity_map,
            bwd_velocity_map = bwd_velocity_map,
            dynamic_masks = dynamic_masks, # (N, H, W)
            static_masks = static_masks,
            depths_valid = depths_valid,
            fwd_velocity_masks = fwd_velocity_masks,
            bwd_velocity_masks = bwd_velocity_masks,
            points_map = points,
        )

    def init_points(self, sample_rate: int, voxel_size: float, use_gpu: bool = True, mask_kernel_size: int = 3, maximum: int = 50_000):
        """
        Initialize static and dynamic point clouds after all batches are processed.
        
        Args:
            sample_rate: Sample rate for point cloud extraction
            voxel_size: Voxel size for downsampling
            use_gpu: Whether to use GPU for computation
        """
        device = "cuda" if use_gpu else "cpu"
        
        # Move data to device for processing
        images = self.images.to(device)
        dynamic_masks = self.dynamic_masks.to(device)
        static_masks = self.static_masks.to(device)
        time_indices = self.time_indices.to(device)
        fwd_velocity_map = self.fwd_velocity_map.to(device)
        bwd_velocity_map = self.bwd_velocity_map.to(device)
        depths_valid = self.depths_valid.to(device)
        
        points = self.points_map.to(device)
        
        # Sample frames using sample_rate
        sampled_indices = torch.arange(0, len(time_indices), sample_rate, device=device)
        sampled_points = points[sampled_indices]  # (N, H, W, 3)
        sampled_images = images[sampled_indices]  # (N, H, W, 3)
        sampled_depths_valid = depths_valid[sampled_indices]  # (N, H, W)

        sampled_dynamic_masks = dynamic_masks[sampled_indices]  # (N, H, W)
        sampled_static_masks = static_masks[sampled_indices]  # (N, H, W)
        sampled_dynamic_masks = erode(sampled_dynamic_masks, kernel_size=mask_kernel_size) & sampled_depths_valid
        sampled_static_masks = erode(sampled_static_masks, kernel_size=mask_kernel_size) & sampled_depths_valid

        sampled_dynamic_masks = sampled_dynamic_masks & sampled_depths_valid
        sampled_static_masks = sampled_static_masks & sampled_depths_valid

        sampled_time_indices = time_indices[sampled_indices]  # (N,)
        sampled_fwd_velocity = fwd_velocity_map[sampled_indices]  # (N, H, W, 3)
        sampled_bwd_velocity = bwd_velocity_map[sampled_indices]  # (N, H, W, 3)
        
        # Process dynamic points
        H, W = sampled_dynamic_masks.shape[1:3]
        temporal_center = repeat(sampled_time_indices, "s -> s h w", h=H, w=W)  # (N, H, W)
        
        # Flatten and extract dynamic points
        dynamic_mask_flat = sampled_dynamic_masks.reshape(-1)  # (N*H*W,)
        if dynamic_mask_flat.any():
            self.dynamic_points = sampled_points.reshape(-1, 3)[dynamic_mask_flat].cpu()  # (M, 3)
            self.dynamic_points_rgb = sampled_images.reshape(-1, 3)[dynamic_mask_flat].cpu()  # (M, 3)
            self.temporal_center = temporal_center.reshape(-1)[dynamic_mask_flat].cpu()  # (M,)
            self.fwd_velocity_gs = sampled_fwd_velocity.reshape(-1, 3)[dynamic_mask_flat].cpu()  # (M, 3)
            self.bwd_velocity_gs = sampled_bwd_velocity.reshape(-1, 3)[dynamic_mask_flat].cpu()  # (M, 3)
        else:
            self.dynamic_points = torch.empty(0, 3)
            self.dynamic_points_rgb = torch.empty(0, 3)
            self.temporal_center = torch.empty(0)
            self.fwd_velocity_gs = torch.empty(0, 3)
            self.bwd_velocity_gs = torch.empty(0, 3)
        
        # Flatten and extract static points
        static_mask_flat = sampled_static_masks.reshape(-1)  # (N*H*W,)
        if static_mask_flat.any():
            self.static_points = sampled_points.reshape(-1, 3)[static_mask_flat].cpu()  # (K, 3)
            self.static_points_rgb = sampled_images.reshape(-1, 3)[static_mask_flat].cpu()  # (K, 3)
        else:
            self.static_points = torch.empty(0, 3)
            self.static_points_rgb = torch.empty(0, 3)
        
        # Calculate scene scale from static points
        if len(self.static_points) > 0:
            static_mean = torch.mean(self.static_points, dim=0)
            # scale = torch.norm(self.static_points - static_mean, dim=1).mean().item()
            bg_points_centered = self.static_points - static_mean
            bg_min_scale = torch_quantile(bg_points_centered, 0.05, dim=0)
            bg_max_scale = torch_quantile(bg_points_centered, 0.95, dim=0)
            scale = torch.max(bg_max_scale - bg_min_scale).item() / 2.0
        else:
            scale = 1.0
            
        self.scene_scale = scale
        if len(self.dynamic_points) > 0:
            scene_center = self.dynamic_points.mean(0)
            tracks_3d_centered = self.dynamic_points - scene_center
            min_scale = torch_quantile(tracks_3d_centered, 0.05, dim=0)
            max_scale = torch_quantile(tracks_3d_centered, 0.95, dim=0)
            # import pdb; pdb.set_trace()
            self.d_scale = torch.max(max_scale - min_scale).item() / 2.0
        else:
            self.d_scale = 1.0
        
        voxel_size = voxel_size * scale
        
        # Preprocess point clouds with voxel downsampling
        if len(self.static_points) > 0:
            (
                self.static_points,
                self.static_points_rgb
            ) = preprocess_pointcloud_3d(
                self.static_points, 
                self.static_points_rgb,
                voxel_size,
                maximum
            )
        
        if len(self.dynamic_points) > 0:
            (
                self.dynamic_points,
                self.dynamic_points_rgb,
                self.fwd_velocity_gs,
                self.bwd_velocity_gs,
                self.temporal_center
            ) = preprocess_pointcloud_4d(
                self.dynamic_points,
                self.dynamic_points_rgb,
                self.fwd_velocity_gs,
                self.bwd_velocity_gs,
                self.temporal_center,
                voxel_size,
            )

        # fwd_v_prune = torch.norm(self.fwd_velocity_map, dim=-1) > 0.1 * scale
        # bwd_v_prune = torch.norm(self.bwd_velocity_map, dim=-1) > 0.1 * scale

        # self.fwd_velocity_map[fwd_v_prune] = 0.0
        # self.bwd_velocity_map[bwd_v_prune] = 0.0
        
        # self.fwd_velocity_gs = torch.where(
        #     torch.norm(self.fwd_velocity_gs, dim=-1, keepdim=True) > 0.1 * scale,
        #     torch.zeros_like(self.fwd_velocity_gs),
        #     self.fwd_velocity_gs
        # )
        # self.bwd_velocity_gs = torch.where(
        #     torch.norm(self.bwd_velocity_gs, dim=-1, keepdim=True) > 0.1 * scale,
        #     torch.zeros_like(self.bwd_velocity_gs),
        #     self.bwd_velocity_gs
        # )
        # self.fwd_velocity_masks = self.fwd_velocity_masks & fwd_v_prune
        # self.bwd_velocity_masks = self.bwd_velocity_masks & bwd_v_prune

    def get_tracks_3d(
        self, num_samples: int, step: int = 1, show_pbar: bool = True, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get 3D tracks from the dataset.

        Args:
            num_samples (int | None): The number of samples to fetch. If None,
                fetch all samples. If not None, fetch roughly a same number of
                samples across each frame. Note that this might result in
                number of samples less than what is specified.
            step (int): The step to temporally subsample the track.
        """
        cached_track_3d_path = osp.join(self.cache_data_path, f"tracks_3d_{num_samples}.pth")
        if osp.exists(cached_track_3d_path) and step == 1:
            print("loading cached 3d tracks data...")
            cached_track_3d_data = torch.load(cached_track_3d_path)
            tracks_3d, visibles, invisibles, confidences, tracks_valid, track_colors = (
                cached_track_3d_data["tracks_3d"],
                cached_track_3d_data["visibles"],
                cached_track_3d_data["invisibles"],
                cached_track_3d_data["confidences"],
                cached_track_3d_data["tracks_valid"],
                cached_track_3d_data["track_colors"],
            )
            return tracks_3d, visibles, invisibles, confidences, tracks_valid, track_colors

        # Load 2D tracks.
        raw_tracks_2d = []
        candidate_frames = list(range(0, self.num_frames, step))
        num_sampled_frames = len(candidate_frames)
        for i in (
            tqdm(candidate_frames, desc="Loading 2D tracks", leave=False)
            if show_pbar
            else candidate_frames
        ):
            curr_num_samples = self.query_tracks_2d[i].shape[0]
            num_samples_per_frame = (
                int(np.floor(num_samples / num_sampled_frames))
                if i != candidate_frames[-1]
                else num_samples
                - (num_sampled_frames - 1)
                * int(np.floor(num_samples / num_sampled_frames))
            )
            if num_samples_per_frame < curr_num_samples:
                track_sels = np.random.choice(
                    curr_num_samples, (num_samples_per_frame,), replace=False
                )
            else:
                track_sels = np.arange(0, curr_num_samples)
            curr_tracks_2d = []
            for j in range(0, self.num_frames, step):
                if i == j:
                    target_tracks_2d = self.query_tracks_2d[i]
                else:
                    target_tracks_2d = torch.from_numpy(
                        np.load(
                            osp.join(
                                self.data_dir,
                                "tapnet",
                                self.data_name,
                                f"{i:05d}_{j:05d}.npy",
                            )
                        ).astype(np.float32)
                    )
                curr_tracks_2d.append(target_tracks_2d[track_sels])
            raw_tracks_2d.append(torch.stack(curr_tracks_2d, dim=1))
        # guru.info(f"{step=} {len(raw_tracks_2d)=} {raw_tracks_2d[0].shape=}")

        # Process 3D tracks.
        inv_Ks = torch.linalg.inv(self.intrinsics)[::step]
        c2ws = self.camtoworlds[::step]
        H, W = self.images.shape[1:3]
        filtered_tracks_3d, filtered_visibles, filtered_track_colors = [], [], []
        filtered_invisibles, filtered_confidences = [], []
        filtered_tracks_valid = []

        # masks = erode(self.dynamic_masks & self.depths_valid, kernel_size=5)
        masks = self.dynamic_masks
        masks = masks.float()
        for i, tracks_2d in enumerate(raw_tracks_2d):
            tracks_2d = tracks_2d.swapdims(0, 1)
            tracks_2d, occs, dists = (
                tracks_2d[..., :2],
                tracks_2d[..., 2],
                tracks_2d[..., 3],
            )
            # visibles = postprocess_occlusions(occs, dists)
            visibles, invisibles, confidences = parse_tapir_track_info(occs, dists)
            # Unproject 2D tracks to 3D.
            track_depths = F.grid_sample(
                self.depths[::step, None],
                normalize_coords(tracks_2d[..., None, :], self.original_H, self.original_W),
                align_corners=True,
                padding_mode="border",
            )[:, 0]
            tracks_3d = (
                torch.einsum(
                    "nij,npj->npi",
                    inv_Ks,
                    F.pad(tracks_2d / self.factor, (0, 1), value=1.0),
                )
                * track_depths
            )
            tracks_3d = torch.einsum(
                "nij,npj->npi", c2ws, F.pad(tracks_3d, (0, 1), value=1.0)
            )[..., :3]
            # Filter out out-of-mask tracks.
            is_in_masks = (
                F.grid_sample(
                    masks[::step, None],
                    normalize_coords(tracks_2d[..., None, :], self.original_H, self.original_W),
                    align_corners=True,
                    mode="nearest",
                    padding_mode="zeros"
                ).squeeze()
                > 0.5
            )
            visibles *= is_in_masks
            invisibles *= is_in_masks
            confidences *= is_in_masks.float()
            # Get track's color from the query frame.
            track_colors = (
                F.grid_sample(
                    self.images[i * step : i * step + 1].permute(0, 3, 1, 2),
                    normalize_coords(tracks_2d[i : i + 1, None, :], self.original_H, self.original_W),
                    align_corners=True,
                    padding_mode="border",
                )
                .squeeze()
                .T
            )
            # at least visible 5% of the time, otherwise discard
            visible_counts = visibles.sum(0)
            valid = (visible_counts >= min(
                int(0.05 * self.num_frames),
                visible_counts.float().quantile(0.1).item(),
            )) # & is_in_masks.all(dim=0)

            filtered_tracks_3d.append(tracks_3d[:, valid])
            filtered_visibles.append(visibles[:, valid])
            filtered_invisibles.append(invisibles[:, valid])
            filtered_confidences.append(confidences[:, valid])
            filtered_track_colors.append(track_colors[valid])
            filtered_tracks_valid.append(is_in_masks[:, valid])

        filtered_tracks_3d = torch.cat(filtered_tracks_3d, dim=1).swapdims(0, 1)
        filtered_visibles = torch.cat(filtered_visibles, dim=1).swapdims(0, 1)
        filtered_invisibles = torch.cat(filtered_invisibles, dim=1).swapdims(0, 1)
        filtered_confidences = torch.cat(filtered_confidences, dim=1).swapdims(0, 1)
        filtered_tracks_valid = torch.cat(filtered_tracks_valid, dim=1).swapdims(0, 1)
        filtered_track_colors = torch.cat(filtered_track_colors, dim=0)
        # filter outlier 
        scene_center = self.dynamic_points.mean(0)
        tracks_3d_centered = self.dynamic_points - scene_center
        min_scale = tracks_3d_centered.quantile(0.05, dim=0)
        max_scale = tracks_3d_centered.quantile(0.95, dim=0)

        # import pdb; pdb.set_trace()
        for i in range(self.num_frames):
            pts = filtered_tracks_3d[:, i]
            valid_mask = filtered_tracks_valid[:, i] & filtered_visibles[:, i]
            mean = pts[valid_mask].mean(dim=0, keepdim=True)
            offset = (pts - mean) * valid_mask[:, None].float()
            outlier = (offset < min_scale) | (offset > max_scale)
            filtered_tracks_valid[:, i] = filtered_tracks_valid[:, i] & (~outlier.any(dim=-1))
            filtered_visibles[:, i] = filtered_visibles[:, i] & (~outlier.any(dim=-1))
            filtered_invisibles[:, i] = filtered_invisibles[:, i] & (~outlier.any(dim=-1))
            filtered_confidences[:, i] = filtered_confidences[:, i] * (~outlier.any(dim=-1)).float()

        if step == 1:
            torch.save(
                {
                    "tracks_3d": filtered_tracks_3d,
                    "visibles": filtered_visibles,
                    "invisibles": filtered_invisibles,
                    "confidences": filtered_confidences,
                    "tracks_valid": filtered_tracks_valid,
                    "track_colors": filtered_track_colors,
                },
                cached_track_3d_path,
            )
            all_pts = []
            all_clrs = []
            for i in range(self.num_frames):
                pts = filtered_tracks_3d[:, i]
                valid_mask = filtered_tracks_valid[:, i] & filtered_visibles[:, i]
                pts = pts[valid_mask].reshape(-1, 3)
                all_pts.append(pts)
                clrs = filtered_track_colors[valid_mask].reshape(-1, 3)
                all_clrs.append(clrs)
                export_pointcloud(
                    pts,
                    clrs,
                    cached_track_3d_path.replace(".pth", f"_{i}.ply"),
                )
            all_pts = torch.cat(all_pts, dim=0)
            all_clrs = torch.cat(all_clrs, dim=0)
            export_pointcloud(
                all_pts,
                all_clrs,
                cached_track_3d_path.replace(".pth", f"_all.ply"),
            )

        return (
            filtered_tracks_3d,
            filtered_visibles,
            filtered_invisibles,
            filtered_confidences,
            filtered_tracks_valid,
            filtered_track_colors,
        )
    
    def normalize(self):
        cached_scene_norm_dict_path = osp.join(
            self.cache_data_path, "scene_norm_dict.pth"
        )
        if osp.exists(cached_scene_norm_dict_path):
            print("loading cached scene norm dict...")
            self.scene_norm_dict = torch.load(
                osp.join(self.cache_data_path, "scene_norm_dict.pth")
            )
        else:
            scene_center = self.dynamic_points.mean(0) 
            scale = self.d_scale
            original_up = -F.normalize(self.extrinsics[:, 1, :3].mean(0), dim=-1)
            target_up = original_up.new_tensor([0.0, 0.0, 1.0])
            R = roma.rotvec_to_rotmat(
                F.normalize(original_up.cross(target_up, dim=-1), dim=-1)
                * original_up.dot(target_up).acos_()
            )
            transfm = rt_to_mat4(R, torch.einsum("ij,j->i", -R, scene_center))
            self.scene_norm_dict = SceneNormDict(scale=scale, transfm=transfm)
            os.makedirs(self.cache_data_path, exist_ok=True)
            torch.save(self.scene_norm_dict, cached_scene_norm_dict_path)
       

        # Normalize the scene.
        scale = self.scene_norm_dict["scale"]
        transfm = self.scene_norm_dict["transfm"]
        self.scene_scale = self.scene_scale / scale
        self.d_scale = 1.0
        self.extrinsics = self.extrinsics @ torch.linalg.inv(transfm)
        self.extrinsics[:, :3, 3] /= scale
        self.depths /= scale
        # import pdb; pdb.set_trace()
        def transform_and_scale(transform, scale, points, rotation_only=False):
            # Handle different input shapes by reshaping to 2D, transforming, then reshaping back
            original_shape = points.shape
            points_2d = points.reshape(-1, points.shape[-1])
            
            # Apply transformation: (4,4) @ (N,4) -> (N,4), then take first 3 components
            if not rotation_only:
                points_homogeneous = F.pad(points_2d, (0, 1), value=1.0)  # Add homogeneous coordinate
                transformed_points = torch.einsum("ij,nj->ni", transform, points_homogeneous)
                transformed_points = transformed_points[..., :3] 
            else:
                transformed_points = torch.einsum("ij,nj->ni", transform[:3, :3], points_2d)

            transformed_points /= scale  # Take only xyz and scale
            # Reshape back to original shape
            return transformed_points.reshape(original_shape)

        self.bwd_velocity_gs = transform_and_scale(transfm,  scale, self.bwd_velocity_gs, rotation_only=True)
        self.fwd_velocity_gs = transform_and_scale(transfm,  scale, self.fwd_velocity_gs, rotation_only=True)
        self.fwd_velocity_map = transform_and_scale(transfm, scale, self.fwd_velocity_map, rotation_only=True)
        self.bwd_velocity_map = transform_and_scale(transfm, scale, self.bwd_velocity_map, rotation_only=True)

        self.static_points = transform_and_scale(transfm, scale, self.static_points)
        self.dynamic_points = transform_and_scale(transfm, scale, self.dynamic_points)
        self.camtoworlds = torch.einsum("ij,tjk->tik", transfm, self.camtoworlds)
        self.camtoworlds[:, :3, 3] /= scale

        self.val_camtoworlds = torch.einsum("ij,tjk->tik", transfm, self.val_camtoworlds)
        self.val_camtoworlds[:, :3, 3] /= scale
        # for future reverse transform
        self.normalize_transform = transfm
        self.normalize_scale = scale
    
    def get_w2cs(self) -> torch.Tensor:
        return self.extrinsics

    def get_Ks(self) -> torch.Tensor:
        return self.intrinsics

    def get_img_wh(self) -> Tuple[int, int]:
        return self.original_W // self.factor, self.original_H // self.factor

        

class Dataset:

    def __init__(
        self,
        parser: Parser,
        split: Literal["train", "val"] = "train",
        num_targets_per_frame: int = 3,
    ):
        self.parser = parser
        self.split = split
        self.num_targets_per_frame = num_targets_per_frame
        if split == "train":
            indices = np.arange(len(self.parser.images))
            self.num_frames = len(self.parser.images)
        else:
            indices = np.arange(len(self.parser.val_images))
            self.num_frames = len(self.parser.val_images)
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    

    def __getitem__(self, item: int) -> Dict[str, Any]:
        if self.split == "train":
            index = self.indices[item]
            K = self.parser.intrinsics[index].clone() 
            camtoworlds = self.parser.camtoworlds[index].clone()
            image = self.parser.images[index].clone()
            depth = self.parser.depths[index].clone()
            dynamic_mask = self.parser.dynamic_masks[index].clone()
            static_mask = self.parser.static_masks[index].clone()
            fwd_velocity = self.parser.fwd_velocity_map[index].clone()
            bwd_velocity = self.parser.bwd_velocity_map[index].clone()
            time = self.parser.time_indices[index].clone()
            depth_valid = self.parser.depths_valid[index].clone()
            fwd_velocity_mask = self.parser.fwd_velocity_masks[index].clone()
            bwd_velocity_mask = self.parser.bwd_velocity_masks[index].clone()
            query_tracks_2d = self.parser.query_tracks_2d[index][:, :2].clone() / self.parser.factor

            data = {
                "Ks": K.float(),
                "camtoworlds": camtoworlds.float(),
                "pixels": image.float(),
                "depths_gt": depth.float(),
                "dynamic_mask": dynamic_mask.bool(),
                "static_mask": static_mask.bool(),
                "fwd_velocity": fwd_velocity.float(),
                "bwd_velocity": bwd_velocity.float(),
                "query_time": time.float(),
                "image_ids": torch.tensor(index, dtype=torch.long),
                "depth_valid": depth_valid,
                "fwd_velocity_mask": fwd_velocity_mask,
                "bwd_velocity_mask": bwd_velocity_mask,
                "query_tracks_2d": query_tracks_2d.float(),
            }

            target_inds = torch.from_numpy(
                np.random.choice(
                    self.num_frames, (self.num_targets_per_frame,), replace=False
                )
            )
            # (N, P, 4).
            target_tracks_2d = torch.stack(
                [
                    torch.from_numpy(
                        np.load(
                            osp.join(
                                self.parser.data_dir,
                                "tapnet",
                                self.parser.data_name,
                                f"{index:05d}_{target_index:05d}.npy",
                            )
                        ).astype(np.float32)
                    )
                    for target_index in target_inds
                ],
                dim=0,
            )
            # (N,).
            target_ts = torch.from_numpy(self.indices[target_inds])
            data["target_ts"] = target_ts
            # (N, 4, 4).
            data["target_w2cs"] = self.parser.extrinsics[target_ts]
            # (N, 3, 3).
            data["target_Ks"] = self.parser.intrinsics[target_ts]
            # (N, P, 2).
            data["target_tracks_2d"] = target_tracks_2d[..., :2] / self.parser.factor
            (
                data["target_visibles"],
                data["target_invisibles"],
                data["target_confidences"],
            ) = parse_tapir_track_info(
                target_tracks_2d[..., 2], target_tracks_2d[..., 3]
            )
            # (N, P).
            data["target_track_depths"] = F.grid_sample(
                self.parser.depths[target_inds, None],
                normalize_coords(
                    target_tracks_2d[..., None, :2],
                    self.parser.original_H,
                    self.parser.original_W,
                ),
                align_corners=True,
                padding_mode="border",
            )[:, 0, :, 0]
            data["target_track_depth_valid"] = F.grid_sample(
                (erode(self.parser.depths_valid[target_inds] & self.parser.dynamic_masks[target_inds], 5))[:, None].float(),
                normalize_coords(
                    target_tracks_2d[..., None, :2],
                    self.parser.original_H,
                    self.parser.original_W,
                ),
                align_corners=True,
                mode="nearest",
                padding_mode="zeros",
            )[:, 0, :, 0] > 0.5

            
        else:
            index = self.indices[item]
            K = self.parser.val_intrinsics[index].clone()
            camtoworlds = self.parser.val_camtoworlds[index].clone()
            image = self.parser.val_images[index].clone()
            time = self.parser.val_time_indices[index].clone()
            mask = self.parser.val_masks[index].clone()
            cam_ids = self.parser.val_cam_ids[index].clone()

            data = {
                "Ks": K.float(),
                "camtoworlds": camtoworlds.float(),
                "pixels": image.float(),
                "query_time": time.float(),
                "masks": mask.bool(),
                "image_ids": torch.tensor(index, dtype=torch.long),
                "cam_ids": cam_ids
            }

        return data
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated_batch = {}
        for key in batch[0]:
            if key in ["query_tracks_2d", "target_tracks_2d", "target_visibles", "target_invisibles", "target_confidences", "target_track_depths", "target_track_depth_valid"]: 
                collated_batch[key] = list([b[key] for b in batch])
            else:
                collated_batch[key] = torch.stack([b[key] for b in batch], dim=0)
        return collated_batch


if __name__ == "__main__":
    pass
