import argparse
import fnmatch
import os
import os.path as osp
from glob import glob
from typing import Literal

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Pipeline, pipeline
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
import tempfile
import Imath


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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UINT16_MAX = 65535


models = {
    "depth-anything": "LiheYoung/depth-anything-large-hf",
    "depth-anything-v2": "depth-anything/Depth-Anything-V2-Large-hf",
    "depthpro": "apple/DepthPro-hf",
    "zoedepth": "Intel/zoedepth-nyu-kitti",
}
def get_unidepth_model(device):
    version = "v2"
    backbone = "vitl14"
    model = torch.hub.load(
        "lpiccinelli-eth/UniDepth",
        "UniDepth",
        version=version,
        backbone=backbone,
        pretrained=True,
        trust_repo=True,
        force_reload=True,
    )

    model.to(device)
    model.eval()
    return model


def get_pipeline(model_name: str):
    if model_name != "unidepth":
        pipe = pipeline(task="depth-estimation", model=models[model_name], device=DEVICE)
        return pipe
    else:
        model = get_unidepth_model(DEVICE)
        pipe = lambda x: {"predicted_depth": 1.0 / model.infer(x)["depth"]}
    print(f"{model_name} model loaded.")
    return model

def to_uint16(disp: np.ndarray):
    disp_min = disp.min()
    disp_max = disp.max()

    if disp_max - disp_min > np.finfo("float").eps:
        disp_uint16 = UINT16_MAX * (disp - disp_min) / (disp_max - disp_min)
    else:
        disp_uint16 = np.zeros(disp.shape, dtype=disp.dtype)
    disp_uint16 = disp_uint16.astype(np.uint16)
    return disp_uint16


def get_disp(
    pipe: Pipeline,
    img_file: str,
    ret_type: Literal["uint16", "float"] = "float",
    model_name: str = "depth-anything",
    return_disp: bool = True,
):
    image = Image.open(img_file)
    disp = pipe(image)["predicted_depth"]
    if return_disp and model_name in ["zoedepth", "depthpro"]:
        disp = 1.0 / disp
        # normalize to [0, 1]
        # disp_min = disp.min()
        # disp_max = disp.max()
        # disp = (disp - disp_min) / (disp_max - disp_min)
    # import pdb; pdb.set_trace()
    # disp = torch.nn.functional.interpolate(
        # disp.unsqueeze(0).unsqueeze(0), size=image.size[::-1], mode="bicubic", align_corners=False
    # )
    disp = disp.squeeze().cpu().numpy()
    if ret_type == "uint16":
        return to_uint16(disp)
    elif ret_type == "float":
        return disp
    else:
        raise ValueError(f"Unknown return type {ret_type}")


def save_disp_from_dir(
    model_name: str,
    img_dir: str,
    out_dir: str,
    matching_pattern: str = "*",
):
    img_files = sorted(glob(osp.join(img_dir, "*.jpg"))) + sorted(
        glob(osp.join(img_dir, "*.png"))
    )
    img_files = [
        f for f in img_files if fnmatch.fnmatch(osp.basename(f), matching_pattern)
    ]
    if osp.exists(out_dir) and len(glob(osp.join(out_dir, "*.npy"))) == len(img_files):
        print(f"Raw {model_name} depth maps already computed for {img_dir}")
        return

    pipe = get_pipeline(model_name)
    os.makedirs(out_dir, exist_ok=True)
    for img_file in tqdm(img_files, f"computing {model_name} depth maps"):
        disp = get_disp(pipe, img_file, ret_type="float", model_name=model_name)
        out_file = osp.join(out_dir, osp.splitext(osp.basename(img_file))[0] + ".npy")
        np.save(out_file, disp)


def align_monodepth_with_metric_depth(
    metric_depth_dir: str,
    input_monodepth_dir: str,
    output_monodepth_dir: str,
    matching_pattern: str = "*",
):
    print(
        f"Aligning monodepth in {input_monodepth_dir} with metric depth in {metric_depth_dir}"
    )
    img_files = sorted(os.listdir(input_monodepth_dir))
    # os.makedirs(output_monodepth_dir, exist_ok=True)
    metric_depths = read_depth_artifacts(metric_depth_dir)
    output_depths = []
    for i, f in tqdm(enumerate(img_files)):
        # metric_path = osp.join(metric_depth_dir, imname + ".npy")
        mono_path = osp.join(input_monodepth_dir, f)
        mono_disp_map = np.load(mono_path)
        metric_depth = metric_depths[i]
        valid_mask = metric_depth > 2e-3

        metric_disp_map = 1.0 / metric_depth[valid_mask]
        mono_disp = mono_disp_map[valid_mask]
        ms_colmap_disp = metric_disp_map - np.median(metric_disp_map) + 1e-8
        ms_mono_disp = mono_disp - np.median(mono_disp) + 1e-8

        scale = np.median(ms_colmap_disp / ms_mono_disp)
        shift = np.median(metric_disp_map - scale * mono_disp)
        aligned_disp = scale * mono_disp_map + shift

        min_thre = max(1e-4, np.quantile(aligned_disp, 0.05))
        # set depth values that are too small to invalid (0)
        out_mask = aligned_disp < min_thre
        # aligned_disp[aligned_disp < min_thre] = 0.0
        aligned_depth = 1.0 / aligned_disp
        aligned_depth[out_mask] = 0.0

        output_depths.append(aligned_depth)
    
    
    with zipfile.ZipFile(output_monodepth_dir, "w", zipfile.ZIP_DEFLATED) as z:
        for frame_idx, metric_depth in enumerate(output_depths):
            height, width = metric_depth.shape
            header = OpenEXR.Header(width, height)
            header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
            with tempfile.NamedTemporaryFile(suffix=".exr") as f:
                exr = OpenEXR.OutputFile(f.name, header)
                exr.writePixels({"Z": metric_depth.astype(np.float16).tobytes()})
                exr.close()
                z.write(f.name, f"{frame_idx:05d}.exr")



def align_monodepth_with_colmap(
    sparse_dir: str,
    input_monodepth_dir: str,
    output_monodepth_dir: str,
    matching_pattern: str = "*",
):
    from pycolmap import SceneManager

    manager = SceneManager(sparse_dir)
    manager.load()

    cameras = manager.cameras
    images = manager.images
    points3D = manager.points3D
    point3D_id_to_point3D_idx = manager.point3D_id_to_point3D_idx

    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    os.makedirs(output_monodepth_dir, exist_ok=True)
    images = [
        image
        for _, image in images.items()
        if fnmatch.fnmatch(image.name, matching_pattern)
    ]
    for image in tqdm(images, "Aligning monodepth with colmap point cloud"):

        point3D_ids = image.point3D_ids
        point3D_ids = point3D_ids[point3D_ids != manager.INVALID_POINT3D]
        pts3d_valid = points3D[[point3D_id_to_point3D_idx[id] for id in point3D_ids]]  # type: ignore
        K = cameras[image.camera_id].get_camera_matrix()
        rot = image.R()
        trans = image.tvec.reshape(3, 1)
        extrinsics = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)

        pts3d_valid_homo = np.concatenate(
            [pts3d_valid, np.ones_like(pts3d_valid[..., :1])], axis=-1
        )
        pts3d_valid_cam_homo = extrinsics.dot(pts3d_valid_homo.T).T
        pts2d_valid_cam = K.dot(pts3d_valid_cam_homo[..., :3].T).T
        pts2d_valid_cam = pts2d_valid_cam[..., :2] / pts2d_valid_cam[..., 2:3]
        colmap_depth = pts3d_valid_cam_homo[..., 2]

        monodepth_path = osp.join(
            input_monodepth_dir, osp.splitext(image.name)[0] + ".png"
        )
        mono_disp_map = iio.imread(monodepth_path) / UINT16_MAX

        colmap_disp = 1.0 / np.clip(colmap_depth, a_min=1e-6, a_max=1e6)
        mono_disp = cv2.remap(
            mono_disp_map,  # type: ignore
            pts2d_valid_cam[None, ...].astype(np.float32),
            None,  # type: ignore
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        ms_colmap_disp = colmap_disp - np.median(colmap_disp) + 1e-8
        ms_mono_disp = mono_disp - np.median(mono_disp) + 1e-8

        scale = np.median(ms_colmap_disp / ms_mono_disp)
        shift = np.median(colmap_disp - scale * mono_disp)

        mono_disp_aligned = scale * mono_disp_map + shift

        min_thre = min(1e-6, np.quantile(mono_disp_aligned, 0.01))
        # set depth values that are too small to invalid (0)
        mono_disp_aligned[mono_disp_aligned < min_thre] = 0.0
        np.save(
            osp.join(output_monodepth_dir, image.name.split(".")[0] + ".npy"),
            mono_disp_aligned,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything-v2",
        help="depth model to use, one of [depth-anything, depth-anything-v2]",
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    # parser.add_argument("--img_dir", type=str, required=True)
    # parser.add_argument("--out_raw_dir", type=str, required=True)
    # parser.add_argument("--out_aligned_dir", type=str, default=None)
    # parser.add_argument("--sparse_dir", type=str, default=None)
    # parser.add_argument("--metric_dir", type=str, default=None)
    parser.add_argument("--matching_pattern", type=str, default="*")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # assert args.model in [
        # "depth-anything",
        # "depth-anything-v2",
    # ], f"Unknown model {args.model}"
    img_dir = os.path.join(args.data_root,  "images", args.scene_name)
    out_raw_dir = os.path.join(args.data_root, args.model, args.scene_name)
    metric_dir = os.path.join(args.data_root, "depth", args.scene_name + ".zip")
    out_aligned_dir = os.path.join(args.data_root, "depth_" + args.model, args.scene_name + ".zip")

    os.makedirs(out_raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_aligned_dir), exist_ok=True)
    save_disp_from_dir(
        args.model, img_dir, out_raw_dir, args.matching_pattern
    )

    # elif metric_dir is not None and out_aligned_dir is not None:
    align_monodepth_with_metric_depth(
        metric_dir,
        out_raw_dir,
        out_aligned_dir,
        args.matching_pattern,
    )


if __name__ == "__main__":
    """ example usage for iphone dataset:
    python infer_depth_anything_align.py \
        --data_root /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/optimize-gaussian/data/iphone_processed \
        --scene_name paper-windmill 
    """
    main()