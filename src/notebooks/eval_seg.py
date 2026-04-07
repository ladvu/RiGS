import numpy as np
import os
import pandas as pd
from PIL import Image
def iou(annotation, segmentation):
    """Compute region similarity as the Jaccard Index.

    Args:
      annotation   (ndarray): binary annotation   map.
      segmentation (ndarray): binary segmentation map.

    Returns:
      jaccard (float): region similarity
    """
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)
    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / np.sum(
            (annotation | segmentation), dtype=np.float32
        )
def eval_segmentation(preds, gts):
    scores = []
    for i in range(len(preds)):
        pred = preds[i]
        gt = gts[i]
        iou_score = iou(gt, pred)
        scores.append(iou_score)
    return np.mean(scores)

if __name__ == "__main__":
    romo_dir = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/davis_2016_val/romo"
    our_dir = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/davis_2016_val/foreground_masks"
    seganymo_dir = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/davis_2016_val/moseg_sam2/initial_preds"
    gt_dir = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/davis_2016_val/foreground_masks_gt"
    
    scene_names = sorted(os.listdir(gt_dir))
    romo_profile = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/davis_2016_val/profile_romo.csv"
    our_profile = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/davis_2016_val/profile.csv"
    seganymo_profile = "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/chenyuwu/project/TPGS/data/davis_2016_val/profile_seganymo.csv"
    romo_df = pd.read_csv(romo_profile)
    our_df = pd.read_csv(our_profile)
    seganymo_df = pd.read_csv(seganymo_profile)

    seg_results = {
        "romo": {},
        "ours": {},
        "seganymo": {}
    }
    avg_results = {
        "romo": {},
        "ours": {},
        "seganymo": {}
    }
    for scene_name in scene_names:

        gt_seg_dir = os.path.join(gt_dir, scene_name)
        gts = []
        for f in sorted(os.listdir(gt_seg_dir)):
            x = np.array(Image.open(os.path.join(gt_seg_dir, f)))
            if x.ndim == 3:
                x = x.any(axis=-1)
            elif x.dtype == np.uint8:
                x = x > 0.5
            gts.append(x)
        romo_seg_dir = os.path.join(romo_dir, scene_name, 'masks_sam')
        our_seg_dir = os.path.join(our_dir, scene_name)
        seganymo_seg_dir = os.path.join(seganymo_dir, scene_name)
        romo_preds = [np.array(Image.open(os.path.join(romo_seg_dir, f))) < 127.5 for f in sorted(os.listdir(romo_seg_dir))]
        romo_iou_score = eval_segmentation(romo_preds, gts)

        our_preds = [np.array(Image.open(os.path.join(our_seg_dir, f))) > 127.5 for f in sorted(os.listdir(our_seg_dir))]
        our_iou_score = eval_segmentation(our_preds, gts)

        seganymo_preds = [np.array(Image.open(os.path.join(seganymo_seg_dir, f))) > 0.5 for f in sorted(os.listdir(seganymo_seg_dir))]
        seganymo_iou_score = eval_segmentation(seganymo_preds, gts)


        romo_ellapsed = romo_df[romo_df['scene_name'] == scene_name]['time'].sum().item()
        seganymo_ellapsed = seganymo_df[seganymo_df['scene_name'] == scene_name]['time'].sum().item()
        our_ellapsed = our_df[our_df['scene_name'] == scene_name]['time'].sum().item()

        seg_results["romo"][scene_name] = { "iou":romo_iou_score, "time": romo_ellapsed }
        seg_results["ours"][scene_name] = { "iou":our_iou_score, "time": our_ellapsed }
        seg_results['seganymo'][scene_name] = {"iou":seganymo_iou_score, "time": seganymo_ellapsed}
        

    for key in avg_results.keys():
        avg_results[key]["avg_iou"] = np.mean([seg_results[key][scene_name]["iou"] for scene_name in scene_names])
        avg_results[key]["time"] = np.mean([seg_results[key][scene_name]["time"] for scene_name in scene_names])


    df_seg = pd.DataFrame(avg_results).T
    print(df_seg.to_string())