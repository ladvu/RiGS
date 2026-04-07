import json
import os
import time
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid
from utils import (
    query_gaussian_t,
    activate_gaussians,
    batch_normalize,
    export_video,
    export_pointcloud
)
from einops import rearrange, repeat
from scene.strategy import DynamicStrategy
from config import Config
from scene.utils import *
from runner.base import BaseRunner
# from torch_cluster import nearest

class DynamicRunner(BaseRunner):
    def __init__(
        self, cfg: Config
    ):

        cfg.pose_opt = False
        super().__init__(cfg)
    
    def setup_splats(self, cfg):
        # Model
        (
            self.splats,
            self.optimizers
        ) = create_dynamic_splats_with_optimizers(
            self.parser,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            batch_size=cfg.batch_size,
            velocity_degree=cfg.velocity_degree,
            device=self.device,
        )

        print(f"Model initialized. Number of Dynamic GS: {len(self.splats['means'])}")
        
        self.dynamic_strategy = DynamicStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            prune_lifespan=cfg.prune_lifespan,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            key_for_gradient=self.key_for_gradient,
        )

        self.dynamic_strategy.check_sanity(self.splats, self.optimizers)
        self.dynamic_strategy_state = self.dynamic_strategy.initialize_state(self.scene_scale)
        max_steps = cfg.max_steps
        self.schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["velocity"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["lifespan"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        
        self.static_masks = self.parser.static_masks
        self.dynamic_masks = self.parser.dynamic_masks
    
    def train_step(self, data, step, pbar):
        cfg = self.cfg
        image_ids = data["image_ids"]
        static_mask = self.static_masks[image_ids]
        dynamic_mask = self.dynamic_masks[image_ids]
        if static_mask.sum() <= 10 or dynamic_mask.sum() <= 10:
            print("Skip this frame with too few static or dynamic pixels")
            return
        pixels = data["pixels"]
        camtoworlds = data["camtoworlds"]
        height, width = pixels.shape[1:3]
        Ks = data["Ks"]
        query_time = data["query_time"]
        t = query_time.shape[0]
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        gaussians = self.get_splats(
            query_time=query_time,
            sh_degree=sh_degree_to_use,
            camtoworlds=camtoworlds,
        )
        # add velocity to color for supervision and visualization
        gaussians["colors"] = torch.cat([
            gaussians["colors"],
            gaussians["velocity"],
        ], dim=-1) 
        renders = self.rasterize_splats(
            gaussians=gaussians,
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            render_mode="RGB+D",
        )
        loss, loss_dict, info = self.compute_loss(data, renders, gaussians, step)
        loss.backward()
        info = squeeze_info(info, key_for_gradient=self.key_for_gradient)
        self.dynamic_strategy.step_post_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.dynamic_strategy_state,
            step=step,
            info=info,
        )
        # optimize
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in self.schedulers:
            scheduler.step()
        
        self.pbar_log(pbar, loss_dict, step)
        self.tb_log(step, data, renders, loss_dict, stage="train")
    
    def val_step(self, data, step, pbar):
        cfg = self.cfg
        pixels = data["pixels"]
        camtoworlds = data["camtoworlds"]
        height, width = pixels.shape[1:3]
        Ks = data["Ks"]
        query_time = data["query_time"]
        t = query_time.shape[0]
        torch.cuda.synchronize()
        tic = time.time()
        gaussians = self.get_splats(
            query_time=query_time,
            sh_degree=cfg.sh_degree,
            camtoworlds=camtoworlds,
        )
        # add velocity to color for supervision and visualization
        gaussians["colors"] = torch.cat([
            gaussians["colors"],
            repeat(self.splats["velocity"], "n c -> t n c", t=t)
        ], dim=-1) 
        renders = self.rasterize_splats(
            gaussians=gaussians,
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            render_mode="RGB+D",
        )
        torch.cuda.synchronize()
        ellipse_time = max(time.time() - tic, 1e-10)
        self.tb_log(step, data, renders, None, stage="val")
        pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
        colors = renders[0][..., :3].permute(0, 3, 1, 2)  # [1, 3, H, W]
        colors = torch.clamp(colors, 0.0, 1.0)
        metrics = {
            "psnr" : self.psnr(colors, pixels),
            "ssim" : self.ssim(colors, pixels),
            "lpips" : self.lpips(colors, pixels),
            "ellipse_time": ellipse_time
        }
        
        return metrics

    def compute_loss(self, data:Dict[str, torch.Tensor], renders:Dict[str, torch.Tensor], gaussians:Dict[str, torch.Tensor], step:int):
        image_ids = data["image_ids"]
        dynamic_mask = self.dynamic_masks[image_ids].to(self.device)
        static_mask = self.static_masks[image_ids].to(self.device)
        pixels = data["pixels"]
        depths_gt = data["depths_gt"]
        velocity = data["velocity"]
        cfg = self.cfg
        (
            feature,
            alphas,
            normals,
            normals_from_depth,
            render_distort,
            render_median,
            info
        ) = renders
        (
            colors,
            velocity_rendered,
            depths
        ) = feature[..., 0:3], feature[..., 3:-1], feature[..., -1]
        alphas_mask = alphas[dynamic_mask].detach()
        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=self.device)
            colors = colors + bkgd * (1.0 - alphas)
        info[self.key_for_gradient].retain_grad()

        loss = 0.0
        rgb_loss = F.l1_loss(colors[dynamic_mask], pixels[dynamic_mask])
        loss = loss + rgb_loss
        # velocity_loss = self.compute_velocity_loss(gaussians)
        # loss = loss + velocity_loss
        velocity_loss = F.l1_loss(velocity_rendered[dynamic_mask], velocity[dynamic_mask]) * cfg.velocity_lambda
        loss = loss + velocity_loss
        alpha_loss = torch.mean(alphas[static_mask]) * cfg.alpha_lambda
        loss = loss + alpha_loss

        depthloss = F.l1_loss(depths[dynamic_mask], depths_gt[dynamic_mask], reduction="none") / depths_gt[dynamic_mask]
        depthloss = torch.mean(depthloss) * cfg.depth_lambda
        loss = loss + depthloss 
        # reg_loss = F.relu(1 - self.dynamic_splats["lifespan"]).mean() * cfg.reg_lambda
        # loss = loss + reg_loss
        # normal consistency loss
        if cfg.model_type == "2dgs" and step > cfg.normal_start_iter:
            curr_normal_lambda = cfg.normal_lambda
            normals_from_depth_ = normals_from_depth[dynamic_mask] * alphas_mask
            normal_error = (1 - (normals[dynamic_mask] * normals_from_depth_).sum(dim=-1))
            normal_loss = curr_normal_lambda * normal_error.mean()
        else:
            normal_loss = torch.tensor(0.0, device=depthloss.device)
        loss = loss + normal_loss
        loss_dict = {
            "total_loss": loss.item(),
            "rgb_loss": rgb_loss.item(),
            "velocity_loss": velocity_loss.item(),
            "depth_loss": depthloss.item(),
            "alpha_loss": alpha_loss.item(),
            # "reg_loss": reg_loss.item(),
            "normal_loss": normal_loss.item(),
        }
        return loss, loss_dict, info
    
    def pbar_log(self, pbar, loss_dict, step):
        desc = f"loss={loss_dict['total_loss']:.3f}| "
        desc += f"depth loss={loss_dict['depth_loss']:.6f}| "
        desc += f"vel loss={loss_dict['velocity_loss']:.3f}| "
        pbar.set_description(desc)
    
    def refine_mask(self, dataloader, step:int):
        raise NotImplementedError("Refine mask not implemented yet for dynamic runner")
    
    def save_checkpoint(self, step, meta = None):
        meta["num_GS"] = len(self.splats["means"])
        print("Step: ", step, meta)
        with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
            json.dump(meta, f)
        torch.save(
            {
               "dynamic_splats" : self.splats.state_dict(),
            },
            f"{self.ckpt_dir}/ckpt_{step}.pt",
        )
    
    def load_checkpoint(self, checkpoint):
        if not hasattr(self, "dynamic_splats"):
            self.splats = torch.nn.ParameterDict({
                "means": torch.nn.Parameter(torch.empty(0, 3)),
                "quats": torch.nn.Parameter(torch.empty(0, 4)),
                "scales": torch.nn.Parameter(torch.empty(0, 3)),
                "opacities": torch.nn.Parameter(torch.empty(0)),
                "colors": torch.nn.Parameter(torch.empty(0, 3)),
                "velocity": torch.nn.Parameter(torch.empty(0, 3)),
                "lifespan": torch.nn.Parameter(torch.empty(0)),
                "temporal_center": torch.nn.Parameter(torch.empty(0)),
            })
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=self.device)
        if "dynamic_splats" in checkpoint:
            checkpoint = checkpoint["dynamic_splats"]
        matched_keys = []
        for k in self.splats.keys():
            if k in checkpoint:
                self.splats[k].data = checkpoint[k]
                matched_keys.append(k)
        unexpected_keys = set(checkpoint.keys()).difference(set(matched_keys))
        missed_keys = set(self.splats.keys()).difference(set(matched_keys))
        return missed_keys, unexpected_keys

    def tb_log(self, step, data, renders, loss_dict, stage):
        cfg = self.cfg
        if cfg.tb_every > 0 and step % cfg.tb_every == 0:
            feature, alphas, normals, normals_from_depth, render_distort, render_median, info = renders
            (
                colors,
                velocity_rendered,
                depths
            ) = feature[..., 0:3], feature[..., 3:-1], feature[..., -1]
            pixels = data["pixels"]
            depths_gt = data["depths_gt"]
            velocity = data["velocity"]
            mem = torch.cuda.max_memory_allocated() / 1024**3
            self.writer.add_scalar(f"{stage}/mem", mem, step)
            self.writer.add_scalar(f"{stage}/dynamic_num_GS", len(self.splats["means"]), step)
            if loss_dict is not None:
                self.writer.add_scalar(f"{stage}/loss", loss_dict["total_loss"], step)
                self.writer.add_scalar(f"{stage}/rgbloss", loss_dict["rgb_loss"], step)
                self.writer.add_scalar(f"{stage}/velocityloss", loss_dict["velocity_loss"], step)
                self.writer.add_scalar(f"{stage}/depthloss", loss_dict["depth_loss"], step)
                self.writer.add_scalar(f"{stage}/normalloss", loss_dict["normal_loss"], step)
            image_ids = data["image_ids"]
            dynamic_mask = self.dynamic_masks[image_ids].to(self.device)

            rgb_vis = torch.cat([pixels * dynamic_mask.unsqueeze(-1), colors], dim=1).permute(0, 3, 1, 2).detach().cpu() # [B, C, H, W]
            depth_gt_vis = repeat(batch_normalize(depths_gt * dynamic_mask), "b h w -> b c h w", c=3)

            depth_render_vis = repeat(batch_normalize(depths), "b h w -> b c h w", c=3)
            depth_vis = torch.cat([depth_gt_vis, depth_render_vis], dim=-2).detach().cpu()
            
            velocity_render_vis = rearrange(velocity_rendered / (torch.norm(velocity_rendered, dim=-1, keepdim=True) + 1e-5), "b h w c -> b c h w")
            velocity_gt_vis = rearrange(velocity / (torch.norm(velocity, dim=-1, keepdim=True) + 1e-5), "b h w c -> b c h w")
            velocity_vis = (torch.cat([velocity_gt_vis * dynamic_mask.unsqueeze(1), velocity_render_vis], dim=-2) / 2 + 0.5).detach().cpu()

            canvas = torch.cat([rgb_vis, depth_vis, velocity_vis], dim=-1)
            if normals is not None:
                normals_vis = rearrange(normals, "b h w c -> b c h w")
                normals_from_depth_vis = rearrange(normals_from_depth, "b h w c -> b c h w")
                normals_vis = (torch.cat([normals_vis, normals_from_depth_vis], dim=-2) / 2 + 0.5).detach().cpu()
                canvas = torch.cat([canvas, normals_vis], dim=-1)

            canvas = make_grid(canvas, nrow=1)
            canvas = torch.clamp(canvas, 0.0, 1.0)
            self.writer.add_image(f"{stage}/render", canvas, step)
            Image.fromarray(
                (canvas.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
            ).save(os.path.join(self.render_dir, f"{stage}_{step}.png"))
            self.writer.flush()

            export_pointcloud(
                self.splats["means"].detach(),
                torch.sigmoid(self.splats["colors"]).detach(),
                os.path.join(self.ply_dir, f"{step}.ply")
            )
    

    