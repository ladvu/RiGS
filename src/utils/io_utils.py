from pathlib import Path
import zipfile
import numpy as np
from typing import *
import torch
import OpenEXR
import os.path as osp
import imageio
import os
import cv2
import uuid
from torchvision.io import write_video
from PIL import Image
import math
import json


def read_depth_artifacts(zip_file_path: Path):
    """
    Read metric depth from zipped exr files.
    """
    valid_width, valid_height = 0, 0
    depths = []
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                exr = OpenEXR.InputFile(f)
                header = exr.header()
                dw = header["dataWindow"]
                valid_width = width = dw.max.x - dw.min.x + 1
                valid_height = height = dw.max.y - dw.min.y + 1
                channels = exr.channels(["Z"])
                depth_data = np.frombuffer(channels[0], dtype=np.float16).reshape((height, width))
                depths.append(torch.from_numpy(depth_data.copy()).float())
    return torch.stack(depths, dim=0)  # (N, H, W)


def read_rgb_artifacts(rgb_file_path: Path):
    """
    Read RGB from H264-encoded video.
    """
    rgbs = []
    reader = imageio.get_reader(rgb_file_path, "ffmpeg")
    for frame_idx, rgb in enumerate(reader):
        rgb = torch.from_numpy(rgb) / 255.0
        rgbs.append(rgb)
    return torch.stack(rgbs, dim=0)  # (N, H, W, C)

def read_pose_artifacts(pose_file_path: Path):
    """
    Read camera pose from txt file.
    """
    poses = []
    poses = np.load(pose_file_path)["data"]
    return torch.from_numpy(poses).float()

def read_intrinsics_artifacts(intrinsics_file_path: Path):
    """
    Read camera intrinsics from txt file.
    """
    intrinsics = np.load(intrinsics_file_path)["data"]
    intrinsics_mat = np.zeros((len(intrinsics), 3, 3))
    intrinsics_mat[:, 0, 0] = intrinsics[:, 0]
    intrinsics_mat[:, 1, 1] = intrinsics[:, 1]
    intrinsics_mat[:, 0, 2] = intrinsics[:, 2]
    intrinsics_mat[:, 1, 2] = intrinsics[:, 3]
    intrinsics_mat[:, 2, 2] = 1.0
    return torch.from_numpy(intrinsics_mat).float()


def read_flow_artifacts(zip_file_path:str):
    flows= []
    fwd_weights = []
    bwd_weights = []
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            with z.open(file_name) as f:
                if file_name.endswith('.npz'):
                    data = np.load(f)
                    fwd_weights.append(data['fwd_weight'])
                    bwd_weights.append(data['bwd_weight'])
                elif file_name.endswith('.png'):
                    file_bytes = f.read()
                    np_arr = np.frombuffer(file_bytes, np.uint8)
                    flow = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    flows.append(flow)
    flows = np.stack(flows, axis=0)
    fwd_weights = np.stack(fwd_weights, axis=0)
    bwd_weights = np.stack(bwd_weights, axis=0)
    H, W = flows.shape[1:3]
    
    # Convert from uint16 back to flow values
    # flows shape: (N, H, W, 4) where channels are [fwd_x, fwd_y, bwd_x, bwd_y]
    flows = flows.astype(np.float32)
    # Denormalize: reverse the process from infer_flow.py
    flows = flows / 65535.0 * 2.0 - 1.0  # Back to normalized [-1, 1] range
    
    # Scale back to pixel coordinates
    flows[..., [0, 2]] = flows[..., [0, 2]] * (W - 1)  # x components
    flows[..., [1, 3]] = flows[..., [1, 3]] * (H - 1)  # y components
    
    # Split into forward and backward flows
    forward_flow = flows[..., :2]   # channels 0, 1: forward flow (x, y)
    backward_flow = flows[..., 2:]  # channels 2, 3: backward flow (x, y)

    return torch.from_numpy(forward_flow), torch.from_numpy(backward_flow), torch.from_numpy(fwd_weights), torch.from_numpy(bwd_weights)

def read_image_artifacts(image_dir: str):
    image_paths = sorted(list(os.listdir(image_dir)))
    images = []
    for p in image_paths:
        full_path = os.path.join(image_dir, p)
        img = np.array(Image.open(full_path))
        images.append(img)
    return torch.from_numpy(np.stack(images, axis=0)).float() / 255.0  # (N, H, W, C)

def read_tapnet_artifacts(data_dir):
    num_frames = int(math.sqrt(len(os.listdir(data_dir))))
    tracks = [torch.from_numpy(
        np.load(
            osp.join(
                data_dir,
                f"{i:05d}_{i:05d}.npy",
            )
        ).astype(np.float32)
    ) for i in range(num_frames)]
    return tracks

def read_static_mask_artifacts(zip_file_path: Path) -> torch.Tensor:
    """
    Read static-valid binary masks from zipped PNG files.
    """
    masks = []
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            if not file_name.endswith(".png"):
                continue
            with z.open(file_name) as f:
                mask_buffer = np.frombuffer(f.read(), dtype=np.uint8)
                mask = cv2.imdecode(mask_buffer, cv2.IMREAD_UNCHANGED)
                assert mask is not None, f"Failed to decode static mask {file_name}"
                masks.append(torch.from_numpy(mask.copy() > 0))
    return torch.stack(masks, dim=0)


def read_flow_consistency_artifacts(zip_file_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read forward/backward optical-flow consistency masks from zipped PNG files.
    """
    masks_by_frame = {}
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            if not file_name.endswith(".png"):
                continue
            frame_idx = int(file_name.split("_")[0])
            direction = file_name.split("_")[1].split(".")[0]
            with z.open(file_name) as f:
                mask_buffer = np.frombuffer(f.read(), dtype=np.uint8)
                mask = cv2.imdecode(mask_buffer, cv2.IMREAD_UNCHANGED)
                assert mask is not None, f"Failed to decode flow consistency mask {file_name}"
                direction_idx = 0 if direction == "fwd" else 1
                masks_by_frame.setdefault(frame_idx, [None, None])[direction_idx] = torch.from_numpy(mask.copy() > 0)

    fwd_masks, bwd_masks = []
    for frame_idx in sorted(masks_by_frame):
        fwd_mask, bwd_mask = masks_by_frame[frame_idx]
        assert fwd_mask is not None and bwd_mask is not None, f"Missing flow consistency pair for frame {frame_idx}"
        fwd_masks.append(fwd_mask)
        bwd_masks.append(bwd_mask)
    return torch.stack(fwd_masks, dim=0), torch.stack(bwd_masks, dim=0)


def load_vipe_data(dataroot, artifact_name:str) -> Mapping:
    depths = read_depth_artifacts(os.path.join(dataroot,  'depth', f"{artifact_name}.zip"))
    fwd_flows, bwd_flows, _, _ = read_flow_artifacts(os.path.join(dataroot,  'flow', f"{artifact_name}.zip"))
    fwd_masks, bwd_masks = read_flow_consistency_artifacts(os.path.join(dataroot,  'flow_consistency', f"{artifact_name}.zip"))
    foreground_masks = ~read_static_mask_artifacts(os.path.join(dataroot, "static_mask", f"{artifact_name}.zip"))
    rgbs = read_image_artifacts(os.path.join(dataroot, "images", artifact_name))
    poses = read_pose_artifacts(os.path.join(dataroot, "pose", f"{artifact_name}.npz"))
    # extrinsics = closed_form_inverse_se3(poses)
    intrinsics = read_intrinsics_artifacts(os.path.join(dataroot, "intrinsics", f"{artifact_name}.npz"))
    query_tracks_2d = read_tapnet_artifacts(os.path.join(dataroot, "tapnet", artifact_name))

    # cam_points = depth_map_to_cam_coords(depths, intrinsics)
    # world_points = depth_map_to_world_coords(depths, intrinsics, extrinsics)
    return {
        "intrinsics": intrinsics,
        "poses": poses,
        # "extrinsics": extrinsics,
        "rgb": rgbs,
        "fwd_flows": fwd_flows,
        "bwd_flows": bwd_flows,
        # "fwd_weights": fwd_weights,
        # "bwd_weights": bwd_weights,
        "depths": depths,
        "fwd_masks" : fwd_masks,
        "bwd_masks" : bwd_masks,
        "foreground_masks": foreground_masks,
        "query_tracks_2d": query_tracks_2d,
        # "obj_ids_masks": obj_ids_masks
        # "cam_points": cam_points,
        # "world_points": world_points,
    }

def generate_cache_path(cache_dir: str, name: str, data_dir: str) -> str:
    """
    Generate a unique cache path using scene name, data_dir hash, and UUID.
    
    Args:
        cache_dir: Base cache directory
        name: Scene name
        data_dir: Data directory path
        
    Returns:
        Unique cache path
    """
    # Create a hash of the data_dir for uniqueness
    import hashlib
    data_dir_hash = hashlib.md5(data_dir.encode()).hexdigest()[:8]
    
    # Generate a short UUID for additional uniqueness
    cache_uuid = str(uuid.uuid4())[:8]
    
    # Combine them: cache_dir/name_datahash_uuid
    cache_name = f"{name}_{data_dir_hash}_{cache_uuid}"
    return os.path.join(cache_dir, cache_name)

def find_existing_cache(cache_dir: str, name: str, data_dir: str) -> Optional[str]:
    """
    Find existing cache directory for the given name and data_dir.
    
    Args:
        cache_dir: Base cache directory
        name: Scene name
        data_dir: Data directory path
        
    Returns:
        Path to existing cache directory, or None if not found
    """
    if not os.path.exists(cache_dir):
        return None
        
    # Look for directories that start with the scene name
    for dirname in os.listdir(cache_dir):
        if dirname.startswith(f"{name}_"):
            cache_path = os.path.join(cache_dir, dirname)
            if os.path.isdir(cache_path):
                # Try to load and check if data_dir matches
                cached_data = load_cache_data(cache_path, data_dir)
                if cached_data is not None:
                    return cache_path
    return None

def load_cache_data(cache_path: str, expected_data_dir: str = None):
    """
    Load cached parser data from disk.
    
    Args:
        cache_path: Path to the cache directory containing the data
        expected_data_dir: Expected data directory path for validation
        
    Returns:
        Dictionary containing all cached parser attributes, or None if cache invalid
    """
    if not os.path.exists(cache_path):
        return None
        
    try:
        # Load metadata first to check data_dir
        # metadata = np.load(os.path.join(cache_path, "metadata.npz"))
        with open(os.path.join(cache_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        cached_data_dir = str(metadata["data_dir"])
        
        # Check if data_dir matches (if expected_data_dir is provided)
        if expected_data_dir is not None and cached_data_dir != expected_data_dir:
            print(f"Cache data_dir mismatch: cached='{cached_data_dir}', expected='{expected_data_dir}'")
            return None
        
        # Load all cached data
        data = dict(metadata)
            
        # Load tensor data
        pt_data = torch.load(os.path.join(cache_path, "tensor_data.pt"), map_location='cpu', weights_only=False)
        data.update(pt_data)
        
        return data
        
    except Exception as e:
        print(f"Error loading cache data: {e}")
        return None

def save_cache_data(cache_path: str, data: Mapping, meta_keys: List[str]):
    """
    Save parser data to cache directory.
    
    Args:
        cache_path: Path to save cache data
        data: Dictionary containing parser attributes to cache
    """
    os.makedirs(cache_path, exist_ok=True)
    meta_dict = {}
    pt_dict = {} 
    try:
        # Save tensor data
        for attr in data:
            if attr in meta_keys:
                meta_dict[attr] = data[attr]
            else:
                pt_dict[attr] = data[attr]
                
        # Save metadata (including data_dir) to npz file
        # np.savez(
            # os.path.join(cache_path, "metadata.npz"),
            # **meta_dict
        # )
        with open(os.path.join(cache_path, "metadata.json"), 'w') as f:
            json.dump(meta_dict, f)

        torch.save(pt_dict, os.path.join(cache_path, f"tensor_data.pt"))
        
        print(f"Cache data saved to {cache_path}")
        
    except Exception as e:
        print(f"Error saving cache data: {e}")
        raise


def ensure_even_dim_tensor(video_tensor:torch.Tensor):
    H, W = video_tensor.shape[1:3]
    if H % 2 != 0:
        video_tensor = video_tensor[:, :H-1, :, :]
    if W % 2 != 0:
        video_tensor = video_tensor[:, :, :W-1, :]
    return video_tensor

def export_video(file_name:str, video_tensor:torch.Tensor, fps:int):
    video_tensor = ensure_even_dim_tensor(video_tensor)
    write_video(file_name, video_tensor, fps=fps)
