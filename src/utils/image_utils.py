import torch
import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, medial_axis
from scipy.ndimage import distance_transform_edt

def mask_morphology(masks: torch.Tensor, op="open", kernel_size: int = 3, ) -> torch.Tensor:
    """Apply morphological operation to a batch of masks.

    Args:
        masks (torch.Tensor): A batch of masks with shape (B, H, W) and pixel values in [0, 1].
        kernel_size (int): Size of the structuring element used for opening. Default is 3.

    Returns:
        torch.Tensor: The batch of masks after applying morphological opening.
    """
    B, H, W = masks.shape
    device = masks.device
    operation = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "dilate": cv2.MORPH_DILATE,
        "erode": cv2.MORPH_ERODE
    }[op]
    assert masks.dtype == torch.bool, "Input masks must be of boolean type"
    # Create a structuring element (kernel) for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    morphed_masks = torch.empty_like(masks)
    for i in range(B):
        mask_np = masks[i].cpu().numpy().astype('uint8')
        mask_np = cv2.morphologyEx(mask_np, operation, kernel)
        morphed_masks[i] = torch.from_numpy(mask_np.astype(bool)).to(device)
    return morphed_masks    

def find_connected_components_and_sample_points(mask, min_area=100, points_per_component=5, sampling_strategy="stratified"):
    """
    Find connected components in a binary mask and sample points from components larger than min_area.
    Uses various sampling strategies to better represent the entire region.
    
    Args:
        mask: Binary mask (numpy array)
        min_area: Minimum area threshold for components
        points_per_component: Number of points to sample per component
        sampling_strategy: Strategy for sampling points
            - "stratified": Sample from different distance layers
            - "skeleton": Sample along morphological skeleton
            - "grid": Regular grid-based sampling
            - "centroid_spread": Sample around centroid with spatial spread
    
    Returns:
        query_points: Array of sampled points (N, 2) in (y, x) format
        labels: Array of component labels for each point
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    # Ensure mask is binary
    binary_mask = mask > 0
    
    # Apply morphological closing to fill small holes
    binary_mask = binary_closing(binary_mask, disk(3))
    
    # Find connected components
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)
    
    query_points = []
    labels = []
    
    for region in regions:
        # Check if component area is above threshold
        if region.area >= min_area:
            # Create a mask for this specific component
            component_mask = (labeled_mask == region.label)
            
            if sampling_strategy == "stratified":
                sampled_coords = _sample_stratified_by_distance(component_mask, points_per_component)
            elif sampling_strategy == "skeleton":
                sampled_coords = _sample_from_skeleton(component_mask, points_per_component)
            elif sampling_strategy == "grid":
                sampled_coords = _sample_grid_based(component_mask, points_per_component)
            elif sampling_strategy == "centroid_spread":
                sampled_coords = _sample_centroid_spread(component_mask, region.centroid, points_per_component)
            else:
                # Fallback to original method
                distance_map = distance_transform_edt(component_mask)
                coords = region.coords
                distances = distance_map[coords[:, 0], coords[:, 1]]
                if len(coords) >= points_per_component:
                    indices = np.argsort(distances)[-points_per_component:]
                    sampled_coords = coords[indices]
                else:
                    sampled_coords = coords
            
            # Add to results
            query_points.extend(sampled_coords.tolist())
            labels.extend([region.label] * len(sampled_coords))
    
    query_points = np.array(query_points) if query_points else np.empty((0, 2))
    labels = np.array(labels) if labels else np.empty((0,))
    
    return query_points, labels


def _sample_stratified_by_distance(component_mask, points_per_component):
    """
    Sample points from different distance layers to better represent the region.
    """
    from skimage.morphology import medial_axis
    
    # Compute distance transform
    distance_map = distance_transform_edt(component_mask)
    
    # Get all coordinates in the component
    coords = np.column_stack(np.where(component_mask))
    distances = distance_map[coords[:, 0], coords[:, 1]]
    
    if len(coords) <= points_per_component:
        return coords
    
    # Create distance-based strata
    max_dist = distances.max()
    min_dist = distances.min()
    
    # Divide into layers based on distance percentiles
    num_layers = min(points_per_component, 5)  # At most 5 layers
    percentiles = np.linspace(20, 100, num_layers)  # Start from 20th percentile to avoid edge
    
    sampled_coords = []
    points_per_layer = max(1, points_per_component // num_layers)
    
    for i, percentile in enumerate(percentiles):
        threshold = np.percentile(distances, percentile)
        layer_mask = distances >= threshold
        layer_coords = coords[layer_mask]
        
        if len(layer_coords) > 0:
            # Sample randomly from this layer
            num_samples = min(points_per_layer, len(layer_coords))
            if i == len(percentiles) - 1:  # Last layer gets remaining points
                num_samples = min(points_per_component - len(sampled_coords), len(layer_coords))
            
            indices = np.random.choice(len(layer_coords), num_samples, replace=False)
            sampled_coords.extend(layer_coords[indices])
            
            if len(sampled_coords) >= points_per_component:
                break
    
    return np.array(sampled_coords[:points_per_component])


def _sample_from_skeleton(component_mask, points_per_component):
    """
    Sample points along the morphological skeleton of the component.
    """
    from skimage.morphology import medial_axis
    
    # Compute medial axis (skeleton)
    skeleton = medial_axis(component_mask)
    skeleton_coords = np.column_stack(np.where(skeleton))
    
    if len(skeleton_coords) == 0:
        # Fallback to centroid if no skeleton
        coords = np.column_stack(np.where(component_mask))
        return coords[np.random.choice(len(coords), min(points_per_component, len(coords)), replace=False)]
    
    if len(skeleton_coords) <= points_per_component:
        return skeleton_coords
    
    # Sample evenly along skeleton
    indices = np.linspace(0, len(skeleton_coords) - 1, points_per_component, dtype=int)
    return skeleton_coords[indices]


def _sample_grid_based(component_mask, points_per_component):
    """
    Sample points using a regular grid within the component.
    """
    # Get bounding box of the component
    coords = np.column_stack(np.where(component_mask))
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    
    # Calculate grid size
    area = max_row - min_row + 1, max_col - min_col + 1
    grid_size = int(np.ceil(np.sqrt(points_per_component)))
    
    # Create grid points
    row_step = max(1, (max_row - min_row) // grid_size)
    col_step = max(1, (max_col - min_col) // grid_size)
    
    sampled_coords = []
    for r in range(min_row, max_row + 1, row_step):
        for c in range(min_col, max_col + 1, col_step):
            if component_mask[r, c]:
                sampled_coords.append([r, c])
                if len(sampled_coords) >= points_per_component:
                    break
        if len(sampled_coords) >= points_per_component:
            break
    
    # If not enough points from grid, add some random interior points
    if len(sampled_coords) < points_per_component:
        remaining = points_per_component - len(sampled_coords)
        available_coords = coords[~np.isin(coords, sampled_coords).all(axis=1)]
        if len(available_coords) > 0:
            indices = np.random.choice(len(available_coords), min(remaining, len(available_coords)), replace=False)
            sampled_coords.extend(available_coords[indices])
    
    return np.array(sampled_coords[:points_per_component])


def _sample_centroid_spread(component_mask, centroid, points_per_component):
    """
    Sample points around the centroid with spatial spread to represent the region.
    """
    coords = np.column_stack(np.where(component_mask))
    
    if len(coords) <= points_per_component:
        return coords
    
    centroid = np.array([centroid[0], centroid[1]])  # (row, col) format
    
    # Calculate distances from centroid
    distances_from_centroid = np.linalg.norm(coords - centroid, axis=1)
    
    # Sample points with diverse distances from centroid
    # Sort by distance and take points at regular intervals
    sorted_indices = np.argsort(distances_from_centroid)
    
    # Select points at regular intervals to ensure spatial spread
    if points_per_component == 1:
        # Just take the point closest to centroid
        selected_indices = [sorted_indices[len(sorted_indices)//3]]  # Not exactly center to avoid edge
    else:
        # Sample from different distance ranges
        interval = len(sorted_indices) // points_per_component
        selected_indices = []
        for i in range(points_per_component):
            idx = min(i * interval + interval // 2, len(sorted_indices) - 1)
            selected_indices.append(sorted_indices[idx])
    
    return coords[selected_indices]

def look_up_ids(obj_ids_mask: np.ndarray, point: np.ndarray):
    y, x = point
    one_hot = obj_ids_mask[:, y, x]
    id = one_hot.argmax().item()
    if not one_hot[id]:
        return None
    return id


def densify_mask_sam(
    predictor,
    inference_state,
    mask,
    frame_idx,
    num_iters,
    positive:bool,
    obj_ids_mask: np.ndarray = None,
    **kwargs
):
    current_mask = mask.copy() 
    min_area = kwargs.get('min_area', 1) 
    vis_points = kwargs.pop("vis_points", True)
    seg_mask_np = np.zeros_like(mask, dtype=bool)
    sampled_points = [np.array([[0, 0]])]
    for i in range(num_iters):
        if current_mask.sum() < min_area:
            print(f"Mask completely segmented at iteration {i}")
            break
            
        # Use stratified sampling by default for better region representation
        sampling_strategy = kwargs.pop('sampling_strategy', 'stratified')
        points, _ = find_connected_components_and_sample_points(current_mask, sampling_strategy=sampling_strategy, **kwargs)
        if len(points) == 0:
            print(f"No more points to sample at iteration {i}")
            break
        sampled_points.append(points)
        for point in points:
            obj_id = look_up_ids(point=point, obj_ids_mask=obj_ids_mask) if obj_ids_mask is not None else None
            if obj_id is None:
                print(f"Point not on any object at iteration {i}, skip")
                continue

            input_points = point[np.newaxis, [1, 0]].astype(np.float32)
            if positive:
                labels = np.ones(1, np.int32)  # all positive clicks
            else:
                labels = np.zeros(1, np.int32)  # all negative clicks

            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=input_points,
                labels=labels,
                clear_old_points=False
            )
            seg_mask = out_mask_logits[:, 0] > 0.0
            seg_mask = seg_mask.any(dim=0)  # (H, W)
            seg_mask_np = seg_mask.cpu().numpy()
            current_mask = mask & (~seg_mask_np)
    sampled_points = np.concatenate(sampled_points, axis=0) # y x
    if vis_points:
        vis_points_mask = draw_points_on_mask(seg_mask_np, sampled_points)
    else:
        vis_points_mask = None

    return seg_mask_np, sampled_points, vis_points_mask

def draw_points_on_mask(mask: np.ndarray, points:np.ndarray):
    """Draw sampled points on the mask for visualization.

    Args:
        mask (np.ndarray): Binary mask (H, W)
        points (np.ndarray): Sampled points (N, 2) in (y, x) format
    """
    vis_mask = np.stack([mask.astype(np.uint8)*255]*3, axis=-1)  # (H, W, 3)
    for point in points:
        y, x = point
        cv2.circle(vis_mask, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
    return vis_mask


@torch.autocast(device_type="cuda", dtype=torch.bfloat16) 
def track_mask_sam(predictor, inference_state, num_frames, H, W):
    masks = np.zeros((num_frames, H, W), dtype=bool)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=False):
        mask = np.stack([(out_mask_logits[i, 0] > 0.0).cpu().numpy() for i in range(len(out_obj_ids))], axis=0) # (num_obj, H, W)
        masks[out_frame_idx] = np.any(mask, axis=0)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        mask = np.stack([(out_mask_logits[i, 0] > 0.0).cpu().numpy() for i in range(len(out_obj_ids))], axis=0) # (num_obj, H, W)
        masks[out_frame_idx] = np.any(mask, axis=0)
    return masks

# copied from mosca

def laplacian_filter_depth(depths, threshold_ratio=0.5, ksize=5, open_ksize=3):
    # logging.info("Filtering depth maps...")
    # filter the depth changing boundary, they are not reliable
    device = depths.device
    if isinstance(depths, torch.Tensor):
        depths = depths.cpu().numpy()
    dep_valid_masks = []
    ellip_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
    )
    for dep in depths:
        # detect the edge boundary of depth
        dep = dep.astype(np.float32)
        # ! to handle different scale, the threshold should be adaptive
        threshold = np.median(dep) * threshold_ratio
        mask, _ = detect_depth_occlusion_boundaries(dep, threshold, ksize)
        mask = mask > 0.5
        mask = ~mask  # valid mask
        # ! do a morph operator to remove outliers
        mask_opened = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, ellip_kernel
        )
        mask_opened = mask_opened > 0
        # mask_opened = mask
        dep_valid_masks.append(mask_opened)
    dep_valid_masks = np.stack(dep_valid_masks, axis=0)
    dep_valid_masks = torch.from_numpy(dep_valid_masks).to(device)
    return dep_valid_masks


def detect_depth_occlusion_boundaries(depth_map, threshold=10, ksize=5):
    error = cv2.Laplacian(depth_map, cv2.CV_64F, ksize=ksize)
    error = np.abs(error)
    _, occlusion_boundaries = cv2.threshold(error, threshold, 255, cv2.THRESH_BINARY)
    return occlusion_boundaries.astype(np.uint8), error


def sky_mask_from_depth(depths: torch.Tensor, thres:float = 3.0):
    """ 
    use ostu's method to find two different regions in depth map, one of them is sky
    normally the mean of sky region should be way larger than the other region, here use a threshold to separate them

    Args:
        depths (torch.Tensor): Depth maps of shape (N, H, W).
        thres (float): Threshold to separate sky region based on depth.
        
    Returns:
        torch.Tensor: Sky masks of shape (N, H, W) where True indicates sky region.
    """
    if depths.dim() == 2:
        depths = depths.unsqueeze(0)
    
    N, H, W = depths.shape
    sky_masks = torch.zeros((N, H, W), dtype=torch.bool, device=depths.device)
    
    for i in range(N):
        depth_map = depths[i].cpu().numpy().astype(np.float32)
        
        # Normalize depth map to 0-255 for better threshold processing
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Use Otsu's method to automatically find the optimal threshold
        otsu_threshold, binary_mask = cv2.threshold(
            depth_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Create two masks for the two regions
        mask_high = binary_mask == 255  # High depth values (potential sky)
        mask_low = binary_mask == 0     # Low depth values (foreground objects)
        
        mean_high = np.mean(depth_map[mask_high])
        mean_low = np.mean(depth_map[mask_low])
        if mean_high > mean_low * thres:
            sky_mask = mask_high
            # Apply morphological closing to fill small gaps in sky region
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            sky_mask = cv2.morphologyEx(sky_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            sky_mask = sky_mask.astype(bool)
        else:
            sky_mask = np.zeros((H, W), dtype=bool)
        sky_masks[i] = torch.from_numpy(sky_mask).to(depths.device)
    
    return sky_masks

def fix_sky_depth(depths: torch.Tensor, thres:float = 3.0):
    sky_masks = sky_mask_from_depth(depths, thres).to(depths.device)

    if torch.any(sky_masks):
        for depth, sky_mask in zip(depths, sky_masks):
            if torch.any(sky_mask):
                non_sky_depths = depth[~sky_mask]
                if non_sky_depths.numel() > 0:
                    max_non_sky_depth = torch.max(non_sky_depths)
                    depth[sky_mask] = max_non_sky_depth
    return depths

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch


class MorphOp:
    """Morphological dilation / erosion operator for 4D input tensors of shape (B,C,H,W) with variable output channel count (single morphological operation and step and square kernels only).

    Makes use of unfolded tensors to replace kernel convolution with matrix multiplication

    Ref: [1] Tensorflow reference implementation: "Morphological Networks for Image De-raining (Ranjan Mondal et al. (2019))" https://github.com/ranjanZ/2D-Morphological-Network
         [2] Vector-based pytorch implementation: "Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019))" https://github.com/jlebensold/iclr_2019_buffalo-3
         [3] Torch-model-based implementation: pytorch morphological dilation2d and erosion2d https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d
    """

    def __init__(
        self,
        c_out: int,
        type_str: str,
        device: torch.device,
        kernel_size: int = 5,
        use_soft_max: bool = False,
        soft_max_beta=20,
    ):
        """
        c_out: the number of target output channels after applying the operation [int]
        type: either "dilation2d" or "erosion2d" [str]
        kernel_size: the spatial size of the morphological operation [int]
        use_soft_max: using the soft max rather the torch.max(), ref: [2] [bool]
        soft_max_beta: used by soft_max [float]
        """
        self.c_out = c_out
        self.type_str = type_str

        self.kernel_size = kernel_size
        self.use_soft_max = use_soft_max
        self.soft_max_beta = soft_max_beta

        assert self.type_str in [
            "dilation2d",
            "erosion2d",
        ], f"MorphOp: invalid type {self.type_str}"

        # Unfold operator to replace convolution with matrix multiplication
        self.unfold = torch.nn.Unfold(kernel_size).to(device)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply morphological operation on input tensor of shape (B,C,H,W)
        """

        # add padding to inputs depending on kernel sizes (pad last two H/W dimensions of input)
        h, w = input.shape[-2:]

        pad_N = self.kernel_size - 1
        pad_start = pad_N // 2
        pad_end = pad_N - pad_start
        x = torch.nn.functional.pad(input, (pad_start, pad_end, pad_start, pad_end), mode="replicate")

        # perform unfold to apply morphological operation via
        # simple patch-based operation instead of convolution
        x = self.unfold(x).unsqueeze(1)  # (B, 1, Cin*k, L), with number of patches L

        # apply actual morphological operation
        if self.type_str == "erosion2d":
            x = -x  # (B, Cout, Cin*k, L)

        # combine internal dimensions to output channel number
        if self.use_soft_max:
            x = torch.logsumexp(x * self.soft_max_beta, dim=2, keepdim=False) / self.soft_max_beta  # (B, Cout, L)
        else:
            x, _ = torch.max(x, dim=2, keepdim=False)  # (B, Cout, L)

        if self.type_str == "erosion2d":
            x = -1 * x  # (B, Cout, L)

        # use view instead of fold to avoid creating a copy
        return x.view(-1, self.c_out, h, w)  # (B, Cout, sqrt(L), sqrt(L))


def dilate(input: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Dilate the input tensor using a square kernel of size kernel_size
    input and output are [(B), H, W], either float 0-1 or bool
    """
    batch_dim = input.dim() == 3
    if not batch_dim:
        input = input.unsqueeze(0)

    is_bool = input.dtype == torch.bool
    if is_bool:
        input = input.float()

    op = MorphOp(1, "dilation2d", input.device, kernel_size)
    res = op(input.unsqueeze(1)).squeeze(1)

    res = (res > 0.5) if is_bool else res

    return res if batch_dim else res.squeeze(0)


def erode(input: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Erode the input tensor using a square kernel of size kernel_size
    input and output are [(B), H, W], either float 0-1 or bool
    """
    batch_dim = input.dim() == 3
    if not batch_dim:
        input = input.unsqueeze(0)

    is_bool = input.dtype == torch.bool
    if is_bool:
        input = input.float()

    op = MorphOp(1, "erosion2d", input.device, kernel_size)
    res = op(input.unsqueeze(1)).squeeze(1)

    res = (res > 0.5) if is_bool else res

    return res if batch_dim else res.squeeze(0)
