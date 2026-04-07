import os

os.environ["PATH"] += os.pathsep + "/sbin"

import torch
from PIL import Image
import numpy as np
import os, os.path as osp
from tqdm import tqdm
import cv2
from matplotlib import cm
import sys
import imageio.v2 as iio
import glob

sys.path.append(osp.abspath(osp.dirname(__file__)))



def make_video(src_dir, dst_fn):
    import imageio

    print(f"export video to {dst_fn}...")
    # print(os.listdir(src_dir))
    img_fn = [
        f for f in os.listdir(src_dir) if f.endswith(".png") or f.endswith(".jpg")
    ]
    img_fn.sort()
    frames = []
    for fn in tqdm(img_fn):
        frames.append(imageio.imread(osp.join(src_dir, fn)))
    imageio.mimsave(dst_fn, frames)
    return


def load_image(fn):
    rgb = torch.from_numpy(np.array(Image.open(fn))).permute(2, 0, 1)  # C, H, W
    h, w = rgb.shape[-2:]
    return rgb, h, w


@torch.no_grad()
def process(image, out_fn, model):
    pred = model.infer(image)
    dep = pred["depth"]
    dep = dep.cpu()[0, 0].numpy()
    disp = 1 / dep
    np.save(out_fn, disp)

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
    )

    model.to(device)
    model.eval()
    return model


def unidepth_process_folder(
    model,
    fn_list,
    dst,
    invalid_mask_list=None,
):
    print("Generating UniDepth...")
    os.makedirs(dst, exist_ok=True)
    dep_list = []
    device = next(model.parameters()).device
    for i in tqdm(range(len(fn_list))):
        fn = fn_list[i]
        img = torch.from_numpy(np.array(Image.open(fn_list[i]))).permute(2, 0, 1)  # C, H, W
        save_fn = osp.basename(fn).replace(".jpg", ".npy")
        out_fn = os.path.join(dst, save_fn)
        process(img.to(device), out_fn, model)
        


if __name__ == "__main__":
    device = "cuda"
    # src = "../../data/nvidia_dev_N/Playground/"
    # src = "../../data/nvidia_dev_H/Playground/"
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="source folder")
    parser.add_argument("--scene_name", type=str, required=True, help="source folder")
    args = parser.parse_args()
    unidepth_model = get_unidepth_model(device=device)

    dst=osp.join(args.data_root, "unidepth", args.scene_name)
    os.makedirs(dst, exist_ok=True)
    image_dir = osp.join(args.data_root, "images", args.scene_name)
    unidepth_process_folder(
        unidepth_model,
        fn_list=glob.glob(osp.join(image_dir, "*.png")) + glob.glob(osp.join(image_dir, "*.jpg")),
        dst=dst
    )