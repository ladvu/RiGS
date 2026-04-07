import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "dependencies/RAFT"))
import torch
import argparse
import numpy as np
import os
import os.path as osp
from torchvision import transforms
from PIL import Image
from dependencies.raft import load_RAFT
from tqdm import tqdm
import cv2
import math
import zipfile
import tempfile
from torch.nn.functional import interpolate

def load_images_as_tensor(path='data/truck', interval=1, PIXEL_LIMIT=255000):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    
    # --- 1. Load image paths or video frames ---
    if osp.isdir(path):
        print(f"Loading images from directory: {path}")
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for i in range(0, len(filenames), interval):
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert('RGB'))
            except Exception as e:
                print(f"Could not load image {filenames[i]}: {e}")
    elif path.lower().endswith('.mp4'):
        print(f"Loading frames from video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): raise IOError(f"Cannot open video file: {path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sources.append(Image.fromarray(rgb_frame))
            frame_idx += 1
        cap.release()
    else:
        raise ValueError(f"Unsupported path. Must be a directory or a .mp4 file: {path}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    print(f"Found {len(sources)} images/frames. Processing...")


    tensor_list = []
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            # Convert to tensor
            img_tensor = to_tensor_transform(img_pil)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)

def info2weight(info:torch.Tensor):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=10)
    log_b[:, 1] = 0
    heatmap = (log_b * weight).sum(dim=1)
    return heatmap

def infer_optical_flow(args, model, imgs:torch.Tensor, save_path:str, chunk_size:int = 10, downsample_factor:int = 1):
    
    with torch.no_grad():
        # Process pairs of images in chunks
        fwd_flows = []
        bwd_flows = []
        fwd_weights = []
        bwd_weights = []
        for i in tqdm(range(0, len(imgs)-1, chunk_size)):
            end_idx = min(i + chunk_size, len(imgs) - 1)
            img1_batch = imgs[i:end_idx]
            img2_batch = imgs[i+1:end_idx+1]
            forward_flow, forward_info = model(img1_batch, img2_batch)
            fwd_flows.append(forward_flow.cpu().numpy())
            fwd_weights.append(info2weight(forward_info).cpu().numpy())
            backward_flow, backward_info = model(img2_batch, img1_batch)
            bwd_flows.append(backward_flow.cpu().numpy())
            bwd_weights.append(info2weight(backward_info).cpu().numpy())
        fwd_flows = np.concatenate(fwd_flows, axis=0)
        fwd_weights = np.concatenate(fwd_weights, axis=0)
        bwd_flows = np.concatenate(bwd_flows, axis=0)
        bwd_weights = np.concatenate(bwd_weights, axis=0)
    flows = np.concatenate([
        np.pad(fwd_flows, ((0, 1), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0), 
        np.pad(bwd_flows, ((1, 0), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    ], axis=1).transpose(0, 2, 3, 1)  # (N, H, W, 4)
    

    fwd_weights= np.pad(fwd_weights, ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    bwd_weights = np.pad(bwd_weights, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    # flows = np.concatenate([
    #     fwd_flows,
    #     bwd_flows,
    # ], axis=1).transpose(0, 2, 3, 1)  # (N, H, W, 4)
    N, H, W, _ = flows.shape
    flows[..., [0 ,2]] = flows[..., [0, 2]] / (W - 1)
    flows[..., [1 ,3]] = flows[..., [1, 3]] / (H - 1)
    flows = (flows + 1) * 0.5 * 65535
    flows = flows.astype(np.uint16)
    with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as z:
        for i, flow in enumerate(flows):
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                cv2.imwrite(f.name, flow)
                z.write(f.name, f"{i:05d}.png")
            with tempfile.NamedTemporaryFile(suffix=".npz") as f:
                np.savez(
                    f.name,
                    fwd_weight=fwd_weights[i],
                    bwd_weight=bwd_weights[i],
                )
                z.write(f.name, f"{i:05d}.npz")

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the RAFT model.")

    parser.add_argument("--data_path", type=str, default='examples/skating.mp4', help="Path to the input image directory or a video file.")
    parser.add_argument("--name", type=str)
    parser.add_argument("--save_path", type=str, help="Path to save the output flow file.")
    parser.add_argument("--raft_cfg", type=str, default='../dependencies/RAFT/core/configs/config_spring_M.json', help="Path to the configuration JSON file.")
    parser.add_argument("--raft_ckpt", type=str, default='../dependencies/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth', help="Path to the RAFT checkpoint file.")
    args = parser.parse_args()
    device = "cuda:0"
    imgs = load_images_as_tensor(args.data_path, interval=1).to(device) # (N, 3, H, W)
    model = load_RAFT(
        args.raft_ckpt,
        args.raft_cfg
    ).to(device).eval()
    os.makedirs(args.save_path, exist_ok=True)
    infer_optical_flow(args, model, imgs, os.path.join(args.save_path, f"{args.name}.zip"), chunk_size=10)
