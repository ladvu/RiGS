
# This script infers object-wise motion mask (also motion scores)
# Usually works well for scenes with limited motion, eg. Nvidia Dynamic Scene Dataset.
# May not work so well for scenes with large motion. Currently not verified.
# This script relies on optical flow and autoseg. Please run optical flow and autoseg first.

import sys
from pathlib import Path
import os
from typing import *
import torch
from torch.nn import functional as F
from multiprocessing import Pool
import functools
from numba import jit
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
sys.path.append(SRC_DIR)
from utils import (
    read_flow_artifacts,
    read_autoseg_artifacts,
    compute_flow_masks,
)
from argparse import ArgumentParser
from tqdm import tqdm
import skimage.morphology
import cv2
import numpy as np
from PIL import Image
from einops import reduce
from scipy.signal import find_peaks
# from numba import jit
import time


@torch.no_grad()
def infer_flow_mask(
    fwd_flows: torch.Tensor, bwd_flows: torch.Tensor,
    save_dir:str,
    alpha:float = 0.5, beta: float=0.5,
):
    fwd_masks, bwd_masks = compute_flow_masks(
        fwd_flows[:-1],
        bwd_flows[1:],
        alpha=alpha,
        beta=beta
    )
    fwd_masks = F.pad(fwd_masks, (0,0,0,0,0,1), mode="constant",value=True)
    bwd_masks = F.pad(bwd_masks, (0,0,0,0,1,0), mode="constant",value=True)

    for i, (fwd_mask, bwd_mask) in enumerate(zip(fwd_masks, bwd_masks)):
        
        Image.fromarray(
            (fwd_mask.cpu().numpy() * 255).astype(np.uint8),
        ).save(
            os.path.join(save_dir, f"{i:05d}_fwd.png"),
        )

        Image.fromarray(
            (bwd_mask.cpu().numpy() * 255).astype(np.uint8),
        ).save(
            os.path.join(save_dir, f"{i:05d}_bwd.png"),
        )
    return fwd_masks, bwd_masks

# @jit
def compute_sampson_error(
    x1: np.ndarray,
    x2: np.ndarray,
    F: np.ndarray,
    mask: np.ndarray,
    H: int, 
    W: int,
):
    # Ensure consistent data types
    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    F = F.astype(np.float32)
    
    # Create homogeneous coordinates manually to avoid concatenate issues with numba
    ones_col = np.ones((x1.shape[0], 1), dtype=np.float32)
    x1_homo = np.column_stack((x1, ones_col))
    x2_homo = np.column_stack((x2, ones_col))
    
    Fx1 = x1_homo @ F.T
    Fx2 = x2_homo @ F
    err = (np.sum(x2_homo * Fx1, axis=1) ** 2) / (
        Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Fx2[:, 0] ** 2 + Fx2[:, 1] ** 2 + 1e-8
    )
    err = err.reshape(H, W)
    err = err * mask.astype(np.float32)
    return err

def process_single_frame(
    idx: int,
    fwd_flows_np: np.ndarray,
    bwd_flows_np: np.ndarray,
    fwd_masks_np: np.ndarray,
    bwd_masks_np: np.ndarray,
    H: int,
    W: int,
    num_frames: int,
    save_dir: str = None,
    max_points: int = 5000,
    use_ransac: bool = True,
    ransac_threshold: float = 0.01,
    ransac_confidence: float = 0.99,
    ransac_max_iters: int = 1000,
) -> np.ndarray:
    """
    Process a single frame to compute sampson error and mask.
    Works with numpy arrays to avoid CUDA/multiprocessing issues.
    """
    # Create UV grid
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing='ij'
    )
    xx = 2 * (xx + 0.5) / W - 1
    yy = 2 * (yy + 0.5) / H - 1
    x1 = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (H*W, 2)
    
    err_list = []
    
    for step in [1]:
        # Backward flow
        if idx - step >= 0:
            bwd_flow = bwd_flows_np[idx].astype(np.float32)
            bwd_mask = bwd_masks_np[idx]
            flow = np.stack([
                2.0 * bwd_flow[..., 0] / (W - 1),
                2.0 * bwd_flow[..., 1] / (H - 1),
            ], axis=-1).astype(np.float32)
            x2 = (x1 + flow.reshape(-1, 2)).astype(np.float32)  # (H*W, 2)
            x_mask = bwd_mask.reshape(-1)
            
            if x_mask.sum() > 8:  # Need at least 8 points for fundamental matrix
                x1_valid = x1[x_mask]
                x2_valid = x2[x_mask]
                
                if len(x1_valid) > max_points:
                    indices = np.random.choice(len(x1_valid), max_points, replace=False)
                    x1_valid = x1_valid[indices]
                    x2_valid = x2_valid[indices]
                
                if use_ransac:
                    F, _ = cv2.findFundamentalMat(
                        x1_valid, x2_valid, 
                        cv2.FM_RANSAC,
                        ransacReprojThreshold=ransac_threshold,
                        confidence=ransac_confidence,
                        maxIters=ransac_max_iters
                    )
                else:
                    F, _ = cv2.findFundamentalMat(x1_valid, x2_valid, cv2.FM_LMEDS)
                if F is not None:
                    F = F.astype(np.float32)
                    # Compute Sampson error
                    err = compute_sampson_error(
                        x1,
                        x2, 
                        F,
                        bwd_mask,
                        H, 
                        W
                    )
                    # x1_homo = np.concatenate([x1, np.ones((x1.shape[0], 1))], axis=1)
                    # x2_homo = np.concatenate([x2, np.ones((x2.shape[0], 1))], axis=1)
                    # Fx1 = x1_homo @ F.T
                    # Fx2 = x2_homo @ F
                    # err = (np.sum(x2_homo * Fx1, axis=1) ** 2) / (
                    #     Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Fx2[:, 0] ** 2 + Fx2[:, 1] ** 2 + 1e-8
                    # )
                    # err = err.reshape(H, W)
                    # err = err * bwd_mask.astype(np.float32)
                    err_list.append(err)
        
        # Forward flow
        if idx + step < num_frames:
            fwd_flow = fwd_flows_np[idx].astype(np.float32)
            fwd_mask = fwd_masks_np[idx]
            flow = np.stack([
                2.0 * fwd_flow[..., 0] / (W - 1),
                2.0 * fwd_flow[..., 1] / (H - 1),
            ], axis=-1).astype(np.float32)
            x2 = (x1 + flow.reshape(-1, 2)).astype(np.float32)  # (H*W, 2)
            x_mask = fwd_mask.reshape(-1)
            
            if x_mask.sum() > 8:  # Need at least 8 points for fundamental matrix
                x1_valid = x1[x_mask]
                x2_valid = x2[x_mask]
                if len(x1_valid) > max_points:
                    indices = np.random.choice(len(x1_valid), max_points, replace=False)
                    x1_valid = x1_valid[indices]
                    x2_valid = x2_valid[indices]
                
                if use_ransac:
                    F, _ = cv2.findFundamentalMat(
                        x1_valid, x2_valid, 
                        cv2.FM_RANSAC,
                        ransacReprojThreshold=ransac_threshold,
                        confidence=ransac_confidence,
                        maxIters=ransac_max_iters
                    )
                else:
                    F, _ = cv2.findFundamentalMat(x1_valid, x2_valid, cv2.FM_LMEDS)
                if F is not None:
                    F = F.astype(np.float32)
                    # Compute Sampson error
                    err = compute_sampson_error(
                        x1,
                        x2, 
                        F,
                        fwd_mask,
                        H, 
                        W
                    )
                    # x1_homo = np.concatenate([x1, np.ones((x1.shape[0], 1))], axis=1)
                    # x2_homo = np.concatenate([x2, np.ones((x2.shape[0], 1))], axis=1)
                    # Fx1 = x1_homo @ F.T
                    # Fx2 = x2_homo @ F
                    # err = (np.sum(x2_homo * Fx1, axis=1) ** 2) / (
                    #     Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Fx2[:, 0] ** 2 + Fx2[:, 1] ** 2 + 1e-8
                    # )
                    # err = err.reshape(H, W)
                    # err = err * fwd_mask.astype(np.float32)
                    err_list.append(err)
    
    if len(err_list) > 0:
        err = np.maximum.reduce(err_list)
        # sqrt may be have better result
        err = np.sqrt(err)
    else:
        err = np.zeros((H, W), dtype=np.float32)
    
    # if vis_dir is not None:
    #     mask = skimage.morphology.binary_opening(
    #         err > 0.005, 
    #         skimage.morphology.disk(1)
    #     )
    #     mask = skimage.morphology.dilation(mask, skimage.morphology.disk(2))
    #     Image.fromarray((mask.astype(np.uint8) * 255)).save(
    #         os.path.join(vis_dir, f"{idx:05d}.png")
    #     )
    #     err_vis = err / err.max()
    #     Image.fromarray(
    #         (err_vis * 255.0).astype(np.uint8),
    #         os.path.join(vis_dir, f"{idx:05d}_error.png"),
    #     )
    np.save(
        os.path.join(save_dir, f"{idx:05d}.npy"),
        err.astype(np.float32)
    )
    
    return err


@torch.no_grad()
def infer_mask_sampson(
    fwd_flows:torch.Tensor, fwd_masks: torch.Tensor,
    bwd_flows:torch.Tensor, bwd_masks: torch.Tensor,
    save_dir:str,
    num_processes: int = None,
    max_points: int = 5000,
    use_ransac: bool = True,
    ransac_threshold: float = 0.01,
    ransac_confidence: float = 0.99,
    ransac_max_iters: int = 1000,
    debug: bool = False,
):
    """
    Infer masks using Sampson error with multiprocessing acceleration.
    
    Args:
        fwd_flows: Forward optical flows (S, H, W, 2)
        fwd_masks: Forward flow masks (S, H, W)
        bwd_flows: Backward optical flows (S, H, W, 2)
        bwd_masks: Backward flow masks (S, H, W)
        save_dir: Directory to save output masks
        num_processes: Number of processes to use (default: CPU count)
        debug: If True, use serial processing instead of multiprocessing
    """
    S, H, W, _ = fwd_flows.shape
    num_frames = fwd_flows.shape[0]
    
    # Convert to numpy for processing (avoid CUDA issues)
    fwd_flows_np = fwd_flows.cpu().numpy()
    bwd_flows_np = bwd_flows.cpu().numpy()
    fwd_masks_np = fwd_masks.cpu().numpy()
    bwd_masks_np = bwd_masks.cpu().numpy()
    
    # Create partial function with fixed parameters

    params = []
    for i in range(num_frames):
        params.append((
            i,
            fwd_flows_np,
            bwd_flows_np,
            fwd_masks_np,
            bwd_masks_np,
            H,
            W,
            num_frames,
            save_dir,
            max_points,
            use_ransac,
            ransac_threshold,
            ransac_confidence,
            ransac_max_iters,
        ))
    
    if debug:
        # Serial processing for debugging
        print(f"Processing {num_frames} frames serially (debug mode)...")
        for i in tqdm(range(num_frames), desc="Processing frames (serial)"):
            process_single_frame(*params[i])
    else:
        # Parallel processing
        if num_processes is None:
            num_processes = min(os.cpu_count(), num_frames)
        
        print(f"Processing {num_frames} frames using {num_processes} processes...")
        start_time = time.time()
        
        with Pool(num_processes) as p:
            p.starmap(process_single_frame, params)

        print(f"Finish Sampson in {time.time() - start_time}s")
            
    
    errors = []
    for i in range(num_frames):
        errors.append(np.load(os.path.join(save_dir, f"{i:05d}.npy")))
    
    errors = torch.from_numpy(np.stack(errors, axis=0)).float().to(fwd_flows.device)
    return errors

@torch.no_grad()
def compute_motion_scores_chunked(
    errors: torch.Tensor,
    fwd_masks: torch.Tensor,
    bwd_masks: torch.Tensor,
    obj_ids_masks: torch.Tensor,
    fwd_weights: torch.Tensor,
    bwd_weights: torch.Tensor,
    chunk_size: int = 10,
    filter_time_thres: float = 1e-4
):
    """
    Compute object motion scores in chunks to reduce memory usage.
    
    Args:
        errors: Error tensor (S, H, W) - should be on CPU
        fwd_masks: Forward masks (S, H, W) - should be on CPU
        bwd_masks: Backward masks (S, H, W) - should be on CPU
        obj_ids_masks: Object ID masks (S, O, H, W) - should be on CPU
        fwd_weights: Forward weights (S, H, W) - should be on CPU
        bwd_weights: Backward weights (S, H, W) - should be on CPU
        chunk_size: Number of objects to process at once
        filter_time_thres: Threshold for filtering static frames
    
    Returns:
        obj_motions: Motion scores for each object (O,) - on CPU
    """
    s, o, H, W = obj_ids_masks.shape
    obj_motions = torch.zeros(o, dtype=torch.float32)
    
    # Get device for computation (prefer CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process objects in chunks
    error_objs = []
    for chunk_start in range(0, s, chunk_size):
        chunk_end = min(chunk_start + chunk_size, s)
        
        # Move chunk data to device
        obj_masks_chunk = obj_ids_masks[chunk_start:chunk_end].to(device)  # (S, chunk_size, H, W)
        errors_chunk = errors[chunk_start:chunk_end].to(device)
        fwd_masks_chunk = fwd_masks[chunk_start:chunk_end].to(device)
        bwd_masks_chunk = bwd_masks[chunk_start:chunk_end].to(device)
        fwd_weights_chunk = fwd_weights[chunk_start:chunk_end].to(device)
        bwd_weights_chunk = bwd_weights[chunk_start:chunk_end].to(device)
        
        # Compute flow masks and weights for this chunk (do processing on GPU)
        flow_masks_chunk = fwd_masks_chunk & bwd_masks_chunk
        flow_weights_chunk = torch.amax(torch.stack([fwd_weights_chunk, bwd_weights_chunk], 0), 0)
        weights_chunk = 1 / ((flow_weights_chunk + 1)**2).unsqueeze(1)
        
        # Compute flow object masks for this chunk
        flow_obj_masks_chunk = flow_masks_chunk.unsqueeze(1) & obj_masks_chunk  # (chunk_size, o, H, W)
        
        # Compute error for this chunk
        error_obj_chunk = errors_chunk.unsqueeze(1) * flow_obj_masks_chunk.float() * weights_chunk  # (chunk_size,o, H, W)
        
        # Reduce to get per-frame per-object errors
        error_sum = reduce(error_obj_chunk, "s o h w -> s o", "sum")
        weight_sum = reduce(obj_masks_chunk.float() * weights_chunk, "s o h w -> s o", "sum").clamp(min=1.0)
        error_obj_chunk = error_sum / weight_sum  # (chunk_size, o)
        error_objs.append(error_obj_chunk)

        # Clear intermediate tensors to free GPU memory
        # del obj_masks_chunk, errors_chunk, fwd_masks_chunk, bwd_masks_chunk, fwd_weights_chunk, bwd_weights_chunk
        # del flow_masks_chunk, flow_weights_chunk, weights_chunk, flow_obj_masks_chunk
        # del error_sum, weight_sum
        torch.cuda.empty_cache()
        
    # Filter static frames
    error_objs = torch.cat(error_objs, dim=0)
    static_frames = error_objs < filter_time_thres
    error_objs[static_frames] = 0.0
    
    # Compute motion scores for this chunk
    motion_sum = reduce(error_objs, "s o -> o", "sum")
    valid_frame_count = reduce((~static_frames).float(), "s o -> o", "sum").clamp(min=1.0)
    
    # Store results (move back to CPU)
    obj_motions = (motion_sum / valid_frame_count).cpu()
    
    return obj_motions

@torch.no_grad()
def main(args):
    dataroot = args.data_root
    artifact_name = args.data_name
    image_dir = os.path.join(dataroot, "images", f"{artifact_name}")

    # dyn_mask_reproj_outdir = os.path.join(args.output_dir, "foreground_masks_reproj", f"{artifact_name}")
    dyn_mask_sampson_outdir = os.path.join(args.output_dir, "foreground_masks_sampson", f"{artifact_name}")
    # comb_mask_outdir = os.path.join(args.output_dir, "foreground_masks_combined", f"{artifact_name}")
    final_mask_outdir = os.path.join(args.output_dir, "foreground_masks_final", f"{artifact_name}")
    flow_mask_outdir = os.path.join(args.output_dir, "flow_mask", f"{artifact_name}")
    flow_weight_outdir = os.path.join(args.output_dir, "flow_weight_vis", f"{artifact_name}")
    # obj_vis_outdir = os.path.join(args.output_dir, "object_vis", f"{artifact_name}")
    motion_outdir = os.path.join(args.output_dir, "object_motion")

    # os.makedirs(dyn_mask_reproj_outdir, exist_ok=True)
    os.makedirs(dyn_mask_sampson_outdir, exist_ok=True)
    os.makedirs(flow_mask_outdir, exist_ok=True)
    os.makedirs(flow_weight_outdir, exist_ok=True)
    # os.makedirs(comb_mask_outdir, exist_ok=True)
    os.makedirs(final_mask_outdir, exist_ok=True)
    # os.makedirs(obj_vis_outdir, exist_ok=True)
    os.makedirs(motion_outdir, exist_ok=True)


    # depths = read_depth_artifacts(os.path.join(dataroot,  "depth", f"{artifact_name}.zip")).to(args.device)
    # poses = read_pose_artifacts(os.path.join(dataroot, "pose", f"{artifact_name}.npz")).to(args.device)
    # intrinsics = read_intrinsics_artifacts(os.path.join(dataroot, "intrinsics", f"{artifact_name}.npz")).to(args.device)
    fwd_flows, bwd_flows, fwd_weights, bwd_weights = read_flow_artifacts(os.path.join(dataroot, "flow", f"{artifact_name}.zip"))

    fwd_weights_vis = fwd_weights / (reduce(fwd_weights, "s h w -> s 1 1", "max") + 1e-4)
    bwd_weights_vis = bwd_weights / (reduce(bwd_weights, "s h w -> s 1 1", "max") + 1e-4)

    for i, (fwd_w, bwd_w) in enumerate(zip(fwd_weights_vis, bwd_weights_vis)):
        cv2.imwrite(
            os.path.join(flow_weight_outdir, f"{i:05d}_fwd.png"),
            (fwd_w.cpu().numpy() * 255.0).astype(np.uint8)
        )
        cv2.imwrite(
            os.path.join(flow_weight_outdir, f"{i:05d}_bwd.png"),
            (bwd_w.cpu().numpy() * 255.0).astype(np.uint8)
        )
    for i, (fwd_w, bwd_w) in enumerate(zip(fwd_weights, bwd_weights)):
        cv2.imwrite(
            os.path.join(flow_weight_outdir, f"{i:05d}_fwd_mask.png"),
            ((fwd_w < args.flow_weight_thres).float().numpy()* 255.0).astype(np.uint8)
        )
        cv2.imwrite(
            os.path.join(flow_weight_outdir, f"{i:05d}_bwd_mask.png"),
            ((bwd_w < args.flow_weight_thres).float().numpy()* 255.0).astype(np.uint8)
        )

    fwd_masks, bwd_masks = infer_flow_mask(
        fwd_flows, bwd_flows,
        alpha=args.flow_alpha,
        beta=args.flow_beta,
        save_dir=flow_mask_outdir
    )
    errors = infer_mask_sampson(
        fwd_flows, fwd_masks,
        bwd_flows, bwd_masks,
        save_dir=dyn_mask_sampson_outdir,
        num_processes=args.num_processes,
        max_points=args.max_points,
        use_ransac=args.use_ransac,
        ransac_threshold=args.ransac_threshold,
        ransac_confidence=args.ransac_confidence,
        ransac_max_iters=args.ransac_max_iters,
        debug=args.debug
    )
        

    obj_ids_masks = read_autoseg_artifacts(os.path.join(dataroot, "autoseg", f"{artifact_name}.zip"))
    if obj_ids_masks.ndim == 5:
        obj_ids_masks = obj_ids_masks.squeeze(2)
    s, o, H, W = obj_ids_masks.shape
    obj_area = reduce(obj_ids_masks.float(), "s o h w -> o", "sum") / s
    # min_area = H * W * args.min_area_ratio
    keep_objs = obj_area > args.min_area
    obj_ids_masks = obj_ids_masks[:, keep_objs]
    
    # Use chunked computation to reduce memory usage
    print(f"Computing motion scores for {obj_ids_masks.shape[1]} objects with chunk size {args.chunk_size}...")
    obj_motions = compute_motion_scores_chunked(
        errors.cpu(), 
        fwd_masks.cpu(), 
        bwd_masks.cpu(),
        obj_ids_masks.cpu(), 
        fwd_weights.cpu(),
        bwd_weights.cpu(),
        chunk_size=args.chunk_size,
        filter_time_thres=args.filter_time_thres
    )

    all_obj_motions = np.zeros(shape=(o, ), dtype=np.float32)
    all_obj_motions[keep_objs.cpu().numpy()] = obj_motions.numpy()

    np.savez(
        os.path.join(motion_outdir, f"{artifact_name}.npz"),
        motion=np.sort(all_obj_motions)[::-1],
        idx=np.argsort(all_obj_motions)[::-1]
    )
    # Here we use thresholding to select dynamic objects
    # But this may not be robust enough when motion scores differ a lot, eg. one object moves very fast, while others move slowly

    order = np.argsort(all_obj_motions)[::-1].copy()  # Make a copy to avoid negative stride
    motion_thres = all_obj_motions.max() * args.keep_motion_ratio
    obj_selected_mask = all_obj_motions > motion_thres

    # Find objects that meet the motion threshold
    selected_indices = np.where(obj_selected_mask)[0]
    if len(selected_indices) > 0:
        # Find the position of the last selected object in the sorted order
        selected_positions_in_order = []
        for idx in selected_indices:
            pos = np.where(order == idx)[0]
            if len(pos) > 0:
                selected_positions_in_order.append(pos[0])
        
        if len(selected_positions_in_order) > 0:
            # Get the last position of selected objects
            last_selected_pos = max(selected_positions_in_order)
            # Add dilate parameter to include more objects
            final_pos = min(last_selected_pos + args.dilate, len(order) - 1)
            obj_selected = order[:final_pos + 1]
        else:
            # If no objects found, just take the first few based on dilate
            obj_selected = order[:max(1, args.dilate)]
    else:
        # If no objects meet threshold, take top objects based on dilate parameter
        obj_selected = order[:max(1, args.dilate)]

    # Filter selected objects to only include those that were kept after area filtering
    obj_selected = obj_selected[obj_selected < obj_ids_masks.shape[1]]
    
    if len(obj_selected) > 0:
        final_masks = torch.any(obj_ids_masks[:, obj_selected], dim=1)
    else:
        # If no objects selected, create empty masks
        final_masks = torch.zeros((obj_ids_masks.shape[0], obj_ids_masks.shape[2], obj_ids_masks.shape[3]), dtype=torch.bool)

    for i, mask in enumerate(final_masks):
        Image.fromarray((mask.numpy() * 255.0).astype(np.uint8)).save(
            os.path.join(final_mask_outdir, f"{i:05d}.png")
        )
    




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--output_dir", type=str)
    # filter time thres depends on how large the motion is (fps, abs of flow ...)
    # Why filter time thres? Some object moves only for a few frames and remains static for most of the time.
    # If the motion is continous in your dataset, set this to zero.
    parser.add_argument("--filter_time_thres", type=float, default=1e-4) 
    parser.add_argument("--keep_motion_ratio", type=float, default=0.25)
    parser.add_argument("--flow_alpha", type=float, default=0.5)
    parser.add_argument("--flow_beta", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes for multiprocessing (default: CPU count)")
    parser.add_argument("--flow_weight_thres", type=float, default=1.0)
    parser.add_argument("--min_area", type=float, default=1.0)
    parser.add_argument("--vis_error", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=10, help="Chunk size for object processing to reduce memory usage")
    parser.add_argument("--max_points", type=int, default=10000, help="Maximum number of points for fundamental matrix estimation")
    parser.add_argument("--use_ransac", action="store_true")
    parser.add_argument("--ransac_threshold", type=float, default=0.01, help="RANSAC reprojection threshold")
    parser.add_argument("--ransac_confidence", type=float, default=0.99, help="RANSAC confidence")
    parser.add_argument("--ransac_max_iters", type=int, default=1000, help="RANSAC maximum iterations")
    parser.add_argument("--debug", action="store_true", help="Use serial processing instead of multiprocessing for debugging")
    parser.add_argument("--dilate", type=int, default=0)
    # parser.add_argument("--dynamic_thres", type=float, default=2.0)
    # parser.add_argument("--static_thres", type=float, default=2.0)
    # parser.add_argument("--alpha", type=float, default=0.1)
     # parser.add_argument("--add_points_every", type=int, default=10)
    # parser.add_argument("--num_iters", type=int, default=5)
    args = parser.parse_args()
    main(args)
   