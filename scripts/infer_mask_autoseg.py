# code from autoseg-sam2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import argparse
from loguru import logger
import zipfile
import tempfile
from functools import partial

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    # iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    # inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    # for i in range(num_masks):
    #     for j in range(i, num_masks):
    #         intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
    #         union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
    #         iou = intersection / union
    #         iou_matrix[i, j] = iou
    #         # select mask pairs that may have a severe internal relationship
    #         if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
    #             inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
    #             inner_iou_matrix[i, j] = inner_iou

    #         if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
    #             inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
    #             inner_iou_matrix[j, i] = inner_iou

    # https://github.com/minghanqin/LangSplat/issues/15
    masks_area = masks_area.float().cuda()
    masks_flat = masks_ord.reshape(num_masks, -1).float().cuda()
    intersection = masks_flat @ masks_flat.T
    union = masks_area[:, None] + masks_area[None, :] - intersection
    iou_matrix = intersection / union
    iou_matrix = torch.triu(iou_matrix)
    intersection_over_i = intersection / masks_area[:, None]
    intersection_over_j = intersection / masks_area[None, :]
    cond1 = (intersection_over_i < 0.5) & (intersection_over_j >= 0.85)
    cond2 = (intersection_over_i >= 0.85) & (intersection_over_j < 0.5)
    inner_iou_matrix = torch.zeros_like(iou_matrix)
    inner_iou = 1 - intersection_over_i * intersection_over_j
    inner_iou_matrix[cond1] = inner_iou[cond1]
    i, j = torch.where(cond2)
    inner_iou_matrix[j, i] = inner_iou[i, j]
    iou_matrix = iou_matrix.cpu()
    inner_iou_matrix = inner_iou_matrix.cpu()

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    # import ipdb; ipdb.set_trace()
    return selected_idx

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def save_masks_to_zip(all_mask_data, save_path):
    """
    Save all mask data to a single zip file
    Args:
        all_mask_data: list of tuples (frame_idx, mask_list)
        save_path: path to save the zip file
    """
    with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as z:
        for frame_idx, mask_list in all_mask_data:
            with tempfile.NamedTemporaryFile(suffix=".npy") as f:
                np.save(f.name, mask_list)
                z.write(f.name, f"{frame_idx:05d}.npy")
    

def search_new_obj(masks_from_prev, mask_list,other_masks_list=None,mask_ratio_thresh=0,ratio=0.5, area_threash = 5000):
    new_mask_list = []
    mask_none = ~masks_from_prev[0]
    if isinstance(mask_none, np.ndarray):
        fn = lambda x: x.copy()
        # mask_none = mask_none.copy()[0]
    else:
        fn = lambda x: x.clone().cpu().numpy()
        # mask_none = mask_none.clone()[0].cpu().numpy()
    mask_none = fn(mask_none)[0]

    for prev_mask in masks_from_prev[1:]:
        mask_none &= ~(fn(prev_mask)[0])

    for mask in mask_list:
        seg = mask['segmentation']
        if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
            new_mask_list.append(mask)
    
    for mask in new_mask_list:
        mask_none &= ~mask['segmentation']
    logger.info(len(new_mask_list))
    # import ipdb; ipdb.set_trace()
    logger.info("now ratio:",mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) )
    logger.info("expected ratios:",mask_ratio_thresh)
    if other_masks_list is not None:
        for mask in other_masks_list:
            if mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) > mask_ratio_thresh: 
                seg = mask['segmentation']
                if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
                    new_mask_list.append(mask)
                    mask_none &= ~seg
            else:
                break
    logger.info(len(new_mask_list))

    return new_mask_list


def cal_no_mask_area_ratio(out_mask_list):
    h = out_mask_list[0].shape[1]
    w = out_mask_list[0].shape[2]
    mask_none = ~out_mask_list[0]
    mask_none = mask_none.copy() if isinstance(mask_none, np.ndarray) else mask_none.clone()
    for prev_mask in out_mask_list[1:]:
        mask_none &= ~prev_mask
    return(mask_none.sum() / (h * w))


class Prompts:
    def __init__(self,bs:int):
        self.batch_size = bs
        self.prompts = {}
        self.obj_list = []
        self.key_frame_list = []
        self.key_frame_obj_begin_list = []

    def add(self,obj_id,frame_id,mask):
        if obj_id not in self.obj_list:
            new_obj = True
            self.prompts[obj_id] = []
            self.obj_list.append(obj_id)
        else:
            new_obj = False
        self.prompts[obj_id].append((frame_id,mask))
        if frame_id not in self.key_frame_list and new_obj:
            # import ipdb; ipdb.set_trace()
            self.key_frame_list.append(frame_id)
            self.key_frame_obj_begin_list.append(obj_id)
            logger.info("key_frame_obj_begin_list:",self.key_frame_obj_begin_list)
    
    def get_obj_num(self):
        return len(self.obj_list)
    
    def __len__(self):
        if self.obj_list % self.batch_size == 0:
            return len(self.obj_list) // self.batch_size
        else:
            return len(self.obj_list) // self.batch_size +1
    
    def __iter__(self):
        # self.batch_index = 0
        self.start_idx = 0
        self.iter_frameindex = 0
        return self

    def __next__(self):
        if self.start_idx < len(self.obj_list):
            if self.iter_frameindex == len(self.key_frame_list)-1:
                end_idx = min(self.start_idx+self.batch_size, len(self.obj_list))
            else:
                if self.start_idx+self.batch_size < self.key_frame_obj_begin_list[self.iter_frameindex+1]:
                    end_idx = self.start_idx+self.batch_size
                else:
                    end_idx =  self.key_frame_obj_begin_list[self.iter_frameindex+1]
                    self.iter_frameindex+=1
                # end_idx = min(self.start_idx+self.batch_size, self.key_frame_obj_begin_list[self.iter_frameindex+1])
            batch_keys = self.obj_list[self.start_idx:end_idx]
            batch_prompts = {key: self.prompts[key] for key in batch_keys}
            self.start_idx = end_idx
            return batch_prompts
        # if self.batch_index * self.batch_size < len(self.obj_list):
        #     start_idx = self.batch_index * self.batch_size
        #     end_idx = min(start_idx + self.batch_size, len(self.obj_list))
        #     batch_keys = self.obj_list[start_idx:end_idx]
        #     batch_prompts = {key: self.prompts[key] for key in batch_keys}
        #     self.batch_index += 1
        #     return batch_prompts
        else:
            raise StopIteration
        
def get_video_segments(prompts_loader,predictor,inference_state,final_output=False, to_numpy=True):
    fn = (lambda x: x.cpu().numpy()) if to_numpy else (lambda x: x)
    video_segments = {}
    for batch_prompts in tqdm(prompts_loader,desc="processing prompts\n"):
        predictor.reset_state(inference_state)
        for id, prompt_list in batch_prompts.items():
            for prompt in prompt_list:
                # import ipdb; ipdb.set_trace()
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=prompt[0],
                    obj_id=id,
                    mask=prompt[1]
                )
        # start_frame_idx = 0 if final_output else None
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = { }
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments[out_frame_idx][out_obj_id]= fn(out_mask_logits[i] > 0.0)
        
        if final_output:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx][out_obj_id]= fn(out_mask_logits[i] > 0.0)
    return video_segments

def ensure_no_overlap(masks):
    """
    Ensure that the masks do not overlap.
    if there is an overlaping, assign the overlapping area to the mask with the smallest area.

    Args:
        masks (np.ndarray): list of tuples (frame_idx, mask_list)
    """
    result_masks = []
    # import pdb; pdb.set_trace()
    for n, mask in masks:
        area = np.sum(mask, axis=(1, 2))  # Calculate area for each object
        sorted_indices = np.argsort(area)  # Sort objects by area
        combined_mask = np.zeros_like(mask[0]) 
        obj_masks = np.zeros_like(mask)
        for idx in sorted_indices:
            current_mask = mask[idx]
            overlap = combined_mask & current_mask  # Find overlapping areas
            current_mask = current_mask & ~overlap  # Remove overlapping areas from current mask
            combined_mask |= current_mask  # Update combined mask
            obj_masks[idx] = current_mask
        result_masks.append((n, obj_masks))
    return result_masks


def choose_level(
    masks_default, masks_s, masks_m, masks_l,
    level_key: str = "large",
    use_other_level: bool = False,
    post_nms: bool = True,
    iou_thr=0.8, score_thr=0.7, inner_thr=0.5
):
    fn = partial(masks_update, iou_thr=iou_thr, score_thr=score_thr, inner_thr=inner_thr) if post_nms else (lambda *args, **kwargs: args)
    masks_level = {
        'default': masks_default,
        'small': masks_s,
        'middle': masks_m,
        'large': masks_l
    }
    masks = fn(masks_level[level_key])[0]
    if use_other_level:
        other_key = {
            "default": "small+middle+large",
            "small": "",
            "middle": "small",
            "large": "small+middle"
        }[level_key]
        other_masks = []
        for key in other_key.split("+"):
            masks_k = fn(masks_level[key])[0]
            other_masks = other_masks + [mask for mask in masks_k]
    else:
        other_masks = None
    return masks, other_masks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True)
    parser.add_argument("--output_path",type=str,required=True)
    parser.add_argument("--vis_path", type=str, default=None)
    parser.add_argument("--level",choices=['default','small','middle','large'])
    parser.add_argument("--min_area_ratio", type=float, default=1e-3)
    parser.add_argument("--batch_size",type=int,default=20)
    parser.add_argument("--detect_stride",type=int,default=10)
    parser.add_argument("--use_other_level",type=int,default=1)
    parser.add_argument("--postnms",type=int,default=1)
    parser.add_argument("--pred_iou_thresh",type=float,default=0.7)
    parser.add_argument("--box_nms_thresh",type=float,default=0.7)
    parser.add_argument("--stability_score_thresh",type=float,default=0.85)
    parser.add_argument("--no_overlap", action="store_true")
    args = parser.parse_args()
    logger.info(args)
    video_dir = args.video_path
    level = args.level
    output_path = args.output_path

    sam2_checkpoint = "./checkpoints/sam2/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)

    sam_ckpt_path="checkpoints/sam1/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    
    
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    now_frame = 0

    image_path = os.path.join(video_dir,frame_names[now_frame])
    image = np.array(cv2.imread(image_path))
    H, W = image.shape[:2]
    inference_state = predictor.init_state(video_path=video_dir)
    masks_from_prev = []
    sum_id = 0
    min_area = int(args.min_area_ratio * H * W)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=args.pred_iou_thresh, 
        box_nms_thresh=args.box_nms_thresh, 
        stability_score_thresh=args.stability_score_thresh, 
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=min_area,
    )

    prompts_loader = Prompts(bs=args.batch_size)  
    while True:
        logger.info(f"frame: {now_frame}")

        sum_id = prompts_loader.get_obj_num()
        image_path = os.path.join(video_dir,frame_names[now_frame])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
        masks, other_masks = choose_level(
            masks_default, masks_s, masks_m, masks_l,
            level_key=level,
            use_other_level=bool(args.use_other_level),
            post_nms=bool(args.postnms),
        )
        if now_frame == 0: # first frame
            ann_obj_id_list = range(len(masks))
            for ann_obj_id in tqdm(ann_obj_id_list):
                seg = masks[ann_obj_id]['segmentation']
                prompts_loader.add(ann_obj_id,0,seg)
        else:  
            new_mask_list = search_new_obj(masks_from_prev, masks, other_masks,mask_ratio_thresh)
            logger.info(f"number of new obj: {len(new_mask_list)}")
            for id,mask in enumerate(masks_from_prev):
                if mask.sum() == 0:
                    continue
                prompts_loader.add(id,now_frame,mask[0])
            for i in range(len(new_mask_list)):
                new_mask = new_mask_list[i]['segmentation']
                prompts_loader.add(sum_id+i,now_frame,new_mask)

        logger.info(f"obj num: {prompts_loader.get_obj_num()}")

        if now_frame==0 or len(new_mask_list)!=0:
            video_segments = get_video_segments(prompts_loader,predictor,inference_state)
        # video_segments contains the per-frame segmentation results
        
        vis_frame_stride = args.detect_stride
        max_area_no_mask = (0,-1)
        for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
            if out_frame_idx < now_frame:
                continue
            out_mask_list = []
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                out_mask_list.append(out_mask)
            
            no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
            if now_frame == out_frame_idx:
                mask_ratio_thresh = no_mask_ratio
                logger.info(f"mask_ratio_thresh: {mask_ratio_thresh}")

            if no_mask_ratio > mask_ratio_thresh + 0.01 and out_frame_idx > now_frame:
                masks_from_prev = out_mask_list
                max_area_no_mask = (no_mask_ratio, out_frame_idx)
                logger.info(max_area_no_mask)
                # mask_ratio_thresh = no_mask_ratio
                break
        if max_area_no_mask[1] == -1:
            break
        logger.info("max_area_no_mask:", max_area_no_mask)
        now_frame = max_area_no_mask[1]


    ###### Final output ######
    video_segments = get_video_segments(prompts_loader,predictor,inference_state,final_output=True)
    
    # Collect all mask data for saving to zip
    all_mask_data = []
    
    for out_frame_idx in tqdm(range(0, len(frame_names), 1)):
        out_mask_list = []
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_mask_list.append(out_mask)
        out_mask_list = np.concatenate(out_mask_list, axis=0)
        # no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
        # logger.info(no_mask_ratio)
        # save_masks(out_mask_list, out_frame_idx, save_dir)
        # Collect mask data for zip saving
        all_mask_data.append((out_frame_idx, out_mask_list))
    
    # Save all masks to a single zip file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if args.no_overlap:
        all_mask_data = ensure_no_overlap(all_mask_data)
    save_masks_to_zip(all_mask_data, output_path)
    if args.vis_path is not None:
        os.makedirs(args.vis_path, exist_ok=True)
        num_objects = all_mask_data[0][1].shape[0]
        for i in range(num_objects):
            obj_dir = os.path.join(args.vis_path, f"{i}")
            os.makedirs(obj_dir, exist_ok=True)
            for frame_idx, mask in all_mask_data:
                Image.fromarray((mask[i] * 255.0).astype(np.uint8)).save(
                    os.path.join(obj_dir, f"{frame_idx:05d}.png")
                )
    logger.info(f"All masks saved to {output_path}")
    
