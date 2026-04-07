import os
import sys
from pathlib import Path
from typing import *
from dataclasses import dataclass
from functools import partial
import tyro
import torch
import torch.multiprocessing as mp
from torch.nn import functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger
from einops import reduce
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import time
import csv

# Add paths
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "dependencies/RAFT"))

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from dependencies.raft import load_RAFT
from utils import compute_flow_masks
from infer_mask_autoseg import Prompts, choose_level, search_new_obj, get_video_segments, cal_no_mask_area_ratio
from infer_flow import info2weight
from infer_dynamic_mask import process_single_frame, compute_motion_scores_chunked
import time

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

@dataclass
class DynMaskPipelineConfig:
    # flow config
    raft_cfg: str = '../dependencies/RAFT/core/configs/config_spring_M.json'
    raft_ckpt: str = '../dependencies/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth'
    flow_batch_size:int = 10
    # auto seg config
    sam_min_area_ratio: float = 1e-4
    level: Literal['default', 'large', 'middle', 'small'] = 'large'
    search_new_obj: bool = False
    sam_batch_size: int = 40
    detect_stride: int = 100
    use_other_level: bool = False
    postnms: bool = True
    pred_iou_thresh: float = 0.7
    box_nms_thresh: float = 0.7
    stability_score_thresh: float = 0.85
    sam2_ckpt_path:str = "./checkpoints/sam2/sam2_hiera_large.pt"
    sam2_model_cfg:str = "sam2_hiera_l.yaml"
    sam_ckpt_path:str ="checkpoints/sam1/sam_vit_h_4b8939.pth"
    # dynamic mask config
    filter_time_thres:float = 1e-4
    keep_motion_ratio:float = 0.25
    flow_alpha:float = 0.5
    flow_beta:float = 0.5
    num_processes:int = 24
    flow_weight_thres:float = 1.0
    dyn_min_area_ratio:float = 1e-4
    max_points: int = 5000
    use_ransac: bool = False
    ransac_threshold: float = 0.01
    ransac_confidence: float = 0.99
    ransac_max_iters: int = 1000
    dyn_batch_size:int = 10
    debug: bool = False
    sampson_dir:str = "foreground_masks_sampson"
    # io config
    data_root:str = "../data/davis_2016_val"
    save_dir:str = "../data/davis_2016_val"
    profile_csv:str = "../data/davis_2016_val/profile.csv"
    

class DynMaskPipeline:
    def __init__(self, cfg: DynMaskPipelineConfig):
        self.cfg = cfg
        self.predictor = build_sam2_video_predictor(cfg.sam2_model_cfg, cfg.sam2_ckpt_path)
        self.sam2 = build_sam2(cfg.sam2_model_cfg, cfg.sam2_ckpt_path, device ='cuda', apply_postprocessing=False)
        self.sam1 = sam_model_registry["vit_h"](checkpoint=cfg.sam_ckpt_path).to('cuda')
        self.inference_state = None
        self.prompts_loader = None

        self.raft = load_RAFT(
            cfg.raft_ckpt,
            cfg.raft_cfg
        ).to("cuda").eval()
        
        # Profile data storage
        self.profile_data = []
        
    def set_images(self, image_dir:str):
        logger.info(f"Loading images from {image_dir}...")
        image_names = sorted(os.listdir(image_dir))
        image_tensors = []
        image_pils = []
        for name in tqdm(image_names):
            x_pil = np.array(Image.open(os.path.join(image_dir, name)))
            x = torch.from_numpy(x_pil)
            x = x / 255.0
            image_tensors.append(x)
            image_pils.append(x_pil)
        self.image_tensors = torch.stack(image_tensors, dim=0).cuda()
        self.image_pils = image_pils
        self.inference_state = self.predictor.init_state(video_path=image_dir)
        T, H, W, C = self.image_tensors.shape
        min_area = int(self.cfg.sam_min_area_ratio * H * W)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam1,
            points_per_side=32,
            pred_iou_thresh=self.cfg.pred_iou_thresh, 
            box_nms_thresh=self.cfg.box_nms_thresh, 
            stability_score_thresh=self.cfg.stability_score_thresh, 
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=min_area,
        )
        self.prompts_loader = Prompts(bs=self.cfg.sam_batch_size)

    @torch.no_grad() 
    @torch.amp.autocast(enabled=False, device_type="cuda")
    def process_flow(self):
        logger.info("Processing optical flow...")
        # Process pairs of images in chunks
        fwd_flows = []
        bwd_flows = []
        fwd_weights = []
        bwd_weights = []
        chunk_size = self.cfg.flow_batch_size
        imgs = self.image_tensors.permute(0, 3, 1, 2)
        for i in tqdm(range(0, len(imgs)-1, chunk_size)):
            end_idx = min(i + chunk_size, len(imgs) - 1)
            img1_batch = imgs[i:end_idx]
            img2_batch = imgs[i+1:end_idx+1]
            forward_flow, forward_info = self.raft(img1_batch, img2_batch)

            fwd_flows.append(forward_flow)
            fwd_weights.append(info2weight(forward_info))
            backward_flow, backward_info = self.raft(img2_batch, img1_batch)
            bwd_flows.append(backward_flow)
            bwd_weights.append(info2weight(backward_info))

        fwd_flows = torch.cat(fwd_flows, dim=0)
        fwd_weights = torch.cat(fwd_weights, dim=0)
        bwd_flows = torch.cat(bwd_flows, dim=0)
        bwd_weights = torch.cat(bwd_weights, dim=0)

        fwd_flows = F.pad(fwd_flows, (0,0, 0,0, 0,0, 0,1), mode='constant', value=0).permute(0, 2, 3, 1)
        bwd_flows = F.pad(bwd_flows, (0,0, 0,0, 0,0, 1,0), mode='constant', value=0).permute(0, 2, 3, 1)


        fwd_weights = F.pad(fwd_weights, (0,0, 0,0, 0,1), mode='constant', value=0)
        bwd_weights = F.pad(bwd_weights, (0,0, 0,0, 1,0), mode='constant', value=0)
        # import ipdb; ipdb.set_trace() 
        fwd_flow_masks, bwd_flow_masks = compute_flow_masks(
            fwd_flows, bwd_flows,
            self.cfg.flow_alpha,
            self.cfg.flow_beta,
        )
        return (fwd_flows.cpu(), bwd_flows.cpu(), fwd_weights.cpu(), bwd_weights.cpu(), fwd_flow_masks.cpu(), bwd_flow_masks.cpu())

    @torch.no_grad() 
    def process_autoseg(self,):
        logger.info("Processing auto segmentation...")
        masks_from_prev = []
        now_frame = 0
        while True:
            logger.info(f"frame: {now_frame}")
            sum_id = self.prompts_loader.get_obj_num()
            image = self.image_pils[now_frame]
            masks_default, masks_s, masks_m, masks_l = self.mask_generator.generate(image)
            masks, other_masks = choose_level(
                masks_default, masks_s, masks_m, masks_l,
                level_key=self.cfg.level,
                use_other_level=bool(self.cfg.use_other_level),
                post_nms=bool(self.cfg.postnms),
            )
            if now_frame == 0: # first frame
                ann_obj_id_list = range(len(masks))
                for ann_obj_id in tqdm(ann_obj_id_list):
                    seg = masks[ann_obj_id]['segmentation']
                    self.prompts_loader.add(ann_obj_id,0,seg)
            else:  
                new_mask_list = search_new_obj(masks_from_prev, masks, other_masks,mask_ratio_thresh)
                logger.info(f"number of new obj: {len(new_mask_list)}")
                for id,mask in enumerate(masks_from_prev):
                    if mask.sum() == 0:
                        continue
                    self.prompts_loader.add(id,now_frame,mask[0])
                for i in range(len(new_mask_list)):
                    new_mask = new_mask_list[i]['segmentation']
                    self.prompts_loader.add(sum_id+i,now_frame,new_mask)

            logger.info(f"obj num: {self.prompts_loader.get_obj_num()}")

            if now_frame==0 or len(new_mask_list)!=0:
                video_segments = get_video_segments(self.prompts_loader,self.predictor,self.inference_state, to_numpy=False)
            # video_segments contains the per-frame segmentation results
            if not self.cfg.search_new_obj:
                break
            vis_frame_stride = self.cfg.detect_stride
            max_area_no_mask = (0,-1)
            for out_frame_idx in tqdm(range(0, len(self.image_tensors), vis_frame_stride)):
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
        if now_frame != 0:
            video_segments = get_video_segments(
                self.prompts_loader,
                self.predictor,
                self.inference_state,final_output=True, to_numpy=False)
        all_masks = []
        for out_frame_idx in tqdm(range(0, len(self.image_tensors), 1)):
            out_mask_i = []
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                out_mask_i.append(out_mask)
            out_mask_i = torch.cat(out_mask_i, dim=0)
            all_masks.append(out_mask_i)
        masks = torch.stack(all_masks, dim=0)  # (T, N, H, W)

        return masks.cpu()
    
    def process_sampson(
        self, 
        fwd_flows:torch.Tensor, fwd_masks: torch.Tensor,
        bwd_flows:torch.Tensor, bwd_masks: torch.Tensor,
        save_dir:str,
        debug:bool = False
    ):
        logger.info("Processing Sampson error...")
        fwd_flows_np = fwd_flows.cpu().numpy()
        bwd_flows_np = bwd_flows.cpu().numpy()
        fwd_masks_np = fwd_masks.cpu().numpy()
        bwd_masks_np = bwd_masks.cpu().numpy()
        S, H, W, _ = fwd_flows.shape
        num_frames = fwd_flows.shape[0]
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
                self.cfg.max_points,
                self.cfg.use_ransac,
                self.cfg.ransac_threshold,
                self.cfg.ransac_confidence,
                self.cfg.ransac_max_iters,
            ))

        if debug or self.cfg.num_processes < 0:
            # Serial processing for debugging
            print(f"Processing {num_frames} frames serially (debug mode)...")
            for i in tqdm(range(num_frames), desc="Processing frames (serial)"):
                process_single_frame(*params[i])
        else:
            # Parallel processing
            if self.cfg.num_processes == 0:
                num_processes = min(os.cpu_count(), num_frames)
            else:
                num_processes = self.cfg.num_processes

            print(f"Processing {num_frames} frames using {num_processes} processes...")
            start_time = time.time()

            with mp.Pool(num_processes) as p:
                p.starmap(process_single_frame, params)

            print(f"Finish Sampson in {time.time() - start_time}s")


        errors = []
        for i in range(num_frames):
            errors.append(np.load(os.path.join(save_dir, f"{i:05d}.npy")))

        errors = torch.from_numpy(np.stack(errors, axis=0)).float().to(fwd_flows.device)
        return errors

    @torch.no_grad() 
    def process_dyn_mask(
        self, 
        errors,
        fwd_weights,
        bwd_weights,
        fwd_flow_masks,
        bwd_flow_masks,
        obj_ids_masks
    ):
        logger.info("Processing dynamic masks...") 
        s, o, H, W = obj_ids_masks.shape
        obj_area = reduce(obj_ids_masks.float(), "s o h w -> o", "sum") / s
        min_area = int(H * W * self.cfg.dyn_min_area_ratio)
        keep_objs = obj_area > min_area
        obj_ids_masks = obj_ids_masks[:, keep_objs]
        obj_motions = compute_motion_scores_chunked(
            errors, 
            fwd_flow_masks, 
            bwd_flow_masks,
            obj_ids_masks, 
            fwd_weights,
            bwd_weights,
            chunk_size=self.cfg.dyn_batch_size,
            filter_time_thres=self.cfg.filter_time_thres 
        )
        obj_motions = obj_motions.cpu()
        all_obj_motions = torch.zeros(size=(o, ), dtype=torch.float32, device='cpu')
        all_obj_motions[keep_objs.cpu()] = obj_motions

        motion_thres = all_obj_motions.max() * self.cfg.keep_motion_ratio
        obj_selected = obj_motions > motion_thres

        final_masks = torch.any(obj_ids_masks[:, obj_selected], dim=1)
        return final_masks, all_obj_motions, obj_ids_masks


    def run(self, scene_name, return_intermediate=False):
        # Step 1: Process flow
        start_time = time.time()
        fwd_flows, bwd_flows, fwd_weights, bwd_weights, fwd_flow_masks, bwd_flow_masks = self.process_flow()
        flow_time = time.time() - start_time
        self.profile_data.append([scene_name, "process_flow", flow_time])
        
        # Step 2: Process autoseg
        start_time = time.time()
        obj_ids_masks = self.process_autoseg()
        autoseg_time = time.time() - start_time
        self.profile_data.append([scene_name, "process_autoseg", autoseg_time])
        
        # Step 3: Process sampson
        start_time = time.time()
        save_dir = os.path.join(self.cfg.save_dir,self.cfg.sampson_dir, scene_name)
        os.makedirs(save_dir, exist_ok=True)
        errors = self.process_sampson(
            fwd_flows, fwd_flow_masks,
            bwd_flows, bwd_flow_masks,
            save_dir=save_dir,
            debug=self.cfg.debug
        )
        sampson_time = time.time() - start_time
        self.profile_data.append([scene_name, "process_sampson", sampson_time])
        
        # Step 4: Process dynamic mask
        start_time = time.time()
        final_masks, all_obj_motions, obj_ids_masks = self.process_dyn_mask(
            errors,
            fwd_weights,
            bwd_weights,
            fwd_flow_masks,
            bwd_flow_masks,
            obj_ids_masks
        )
        dyn_mask_time = time.time() - start_time
        if scene_name is not None:
            self.profile_data.append([scene_name, "process_dyn_mask", dyn_mask_time])
        
        if return_intermediate:
            return final_masks, all_obj_motions, dict(fwd_flows=fwd_flows, bwd_flows=bwd_flows, errors=errors, obj_ids_masks=obj_ids_masks)
        
        return final_masks, all_obj_motions
    
    def save_profile_to_csv(self, csv_path):
        """Save profile data to CSV file"""
        if not self.profile_data:
            logger.warning("No profile data to save")
            return
            
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['scene_name', 'step', 'time'])
            writer.writerows(self.profile_data)
        logger.info(f"Profile data saved to {csv_path}")
            


if __name__ == "__main__":
    cfg = tyro.cli(DynMaskPipelineConfig)
    pipeline = DynMaskPipeline(cfg)
    
    image_dir = os.path.join(cfg.data_root, "images")
    mask_save_dir = os.path.join(cfg.save_dir, "foreground_masks")
    motion_save_dir = os.path.join(cfg.save_dir, "object_motion")
    obj_mask_save_dir = os.path.join(cfg.save_dir, "object_mask")
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(motion_save_dir, exist_ok=True)
    scene_names = os.listdir(image_dir)
    for scene_name in scene_names:
        logger.info(f"Processing scene: {scene_name}...")
        torch.cuda.synchronize()
        pipeline.set_images(os.path.join(image_dir, scene_name))
        final_masks, all_obj_motions, meta = pipeline.run(scene_name, return_intermediate=True)
        torch.cuda.synchronize()
        pipeline.save_profile_to_csv(cfg.profile_csv)
        T, H, W = final_masks.shape
        obj_ids_masks = meta["obj_ids_masks"]
        O = obj_ids_masks.shape[1]
        for t in range(T):
            mask_t = final_masks[t].cpu().numpy().astype(np.uint8) * 255
            mask_save_scene_dir = os.path.join(mask_save_dir, scene_name)
            os.makedirs(mask_save_scene_dir, exist_ok=True)
            Image.fromarray(mask_t).save(os.path.join(mask_save_scene_dir, f"{t:05d}.png"))
            obj_ids_masks_t = obj_ids_masks[t].cpu().numpy().astype(np.uint8) * 255
            for o in range(O):
                obj_mask_save_scene_dir = os.path.join(obj_mask_save_dir, scene_name, f"{o}")
                os.makedirs(obj_mask_save_scene_dir, exist_ok=True)
                Image.fromarray(obj_ids_masks_t[o]).save(os.path.join(obj_mask_save_scene_dir, f"{t:05d}.png"))

        all_obj_motions = all_obj_motions.cpu().numpy()
        np.savez(
            os.path.join(motion_save_dir, f"{scene_name}_motion.npz"),
            motion=np.sort(all_obj_motions)[::-1],
            idx=np.argsort(all_obj_motions)[::-1]
        )
    
