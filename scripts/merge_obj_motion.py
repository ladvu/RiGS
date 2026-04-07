
# use this script to manually merge object masks,
# steps: 
# 1. check the current segmentation results in foreground_masks_final
# 2. if not satisfied, check object_motions, which is already sorted by motion score. use numpy-viewer plugin to preview the npz if you use vscode.
# 3. find a good set of object IDs to merge, and modify the obj_ids below. you may need to refer to autoseg for visualization of each object.
# 4. run this script to generate new foreground masks.

# This is an example to merge object mask for "wheel" scene. The original masks are not good enough.

import os
import numpy as np
import sys
from pathlib import Path
from PIL import  Image 
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from utils import read_autoseg_artifacts


data_root = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/optimize-gaussian/data/iphone_processed"
scene_name = "wheel"

zip_path = os.path.join(data_root, "autoseg", scene_name + ".zip")
auto_seg = read_autoseg_artifacts(zip_path).numpy()

obj_ids = np.array([47, 21, 10, 36, 39, 48, 52, 30, 11, 29])
result_mask = auto_seg[:, obj_ids].any(axis=1)

for i, mask in enumerate(result_mask):
    Image.fromarray((mask * 255.0).astype(np.uint8)).save(os.path.join(data_root, "foreground_masks_final", scene_name, f"{i:05d}.png"))