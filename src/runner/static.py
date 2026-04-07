import json
import os
import time
from typing import Dict, Union
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
)
from einops import rearrange, repeat
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from config import Config
from scene.utils import *
from utils.pcd_utils import export_pointcloud
from utils.gs_utils import SH2RGB
from runner.base import BaseRunner

class StaticRunner(BaseRunner):
    def __init__(
        self, cfg: Config
    ):
        super().__init__(cfg)
    
    def setup_splats(self, cfg):
        # Model
        (
            self.splats,
            self.optimizers
        ) = create_static_splats_with_optimizers(
            self.parser,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            batch_size=cfg.batch_size,
            device=self.device,
        )
        print(f"Model initialized. Number of Static GS: {len(self.splats['means'])}")
        # Densification Strategy
        if cfg.strategy == "default":
            self.static_strategy = DefaultStrategy(
                verbose=True,
                prune_opa=cfg.prune_opa,
                grow_grad2d=cfg.grow_grad2d,
                grow_scale3d=cfg.grow_scale3d,
                prune_scale3d=cfg.prune_scale3d,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                reset_every=cfg.reset_every,
                refine_every=cfg.refine_every,
                key_for_gradient=self.key_for_gradient,
            )
            self.static_strategy.check_sanity(self.splats, self.optimizers)
            self.static_strategy_state = self.static_strategy.initialize_state(self.scene_scale)
        else:
            self.static_strategy = MCMCStrategy(
                verbose=True,
                min_opacity=cfg.prune_opa,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                refine_every=cfg.refine_every,
            ) 
            self.static_strategy.check_sanity(self.splats, self.optimizers)
            self.static_strategy_state = self.static_strategy.initialize_state()

        
        max_steps = cfg.max_steps
        self.schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            )
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            self.schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        self.static_masks = self.parser.static_masks
        self.dynamic_masks = self.parser.dynamic_masks
    
    def train_step(self, data, step, pbar):
        cfg = self.cfg
        static_mask = self.static_masks[image_ids]
        dynamic_mask = self.dynamic_masks[image_ids]
        if static_mask.sum() <= 10 or dynamic_mask.sum() <= 10:
            print("Skip this frame with too few static or dynamic pixels")
            return
        pixels = data["pixels"]
        image_ids = data["image_ids"]
        camtoworlds = data["camtoworlds"]
        height, width = pixels.shape[1:3]
        Ks = data["Ks"]
        query_time = data["query_time"]
        if cfg.pose_noise:
            camtoworlds = self.pose_perturb(camtoworlds, image_ids)
        if cfg.pose_opt:
            camtoworlds = self.pose_adjust(camtoworlds, image_ids)
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        gaussians = self.get_splats(
            query_time=query_time,
            sh_degree=sh_degree_to_use,
            camtoworlds=camtoworlds,
        )
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
        info = squeeze_info(info, self.key_for_gradient)
        lr = self.schedulers[0].get_last_lr()[0]
        if cfg.strategy == "default":
            self.static_strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.static_strategy_state,
                step=step,
                info=info,
            )
        else:
            self.static_strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.static_strategy_state,
                step=step,
                info=info,
                lr=lr
            )
        # optimize
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.pose_optimizers:
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
        torch.cuda.synchronize()
        tic = time.time()
        gaussians = self.get_splats(
            query_time=query_time,
            sh_degree=cfg.sh_degree,
            camtoworlds=camtoworlds,
        )
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

    def refine_mask(self, dataloader, step:int):
        raise NotImplementedError("Mask refinement not implemented for static runner")
    
   
    def compute_loss(self, data:Dict[str, torch.Tensor], renders:Dict[str, torch.Tensor], gaussians:Dict[str, torch.Tensor], step:int):
        ids = data["image_ids"]
        static_mask = self.static_masks[ids]

        pixels = data["pixels"]
        depths_gt = data["depths_gt"]
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
        colors, depths = feature[..., 0:3], feature[..., -1]
        alphas_mask = alphas[static_mask].detach()
        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=self.device)
            colors = colors + bkgd * (1.0 - alphas)
        info[self.key_for_gradient].retain_grad()

        loss = 0.0
        rgb_loss = F.l1_loss(colors[static_mask], pixels[static_mask])
        loss = loss + rgb_loss
        # alpha_loss = torch.mean(alphas[dynamic_mask]) * cfg.alpha_lambda
        # loss = loss + alpha_loss
        depth_gt_ = depths_gt[static_mask]
        depthloss = F.l1_loss(depths[static_mask], depth_gt_, reduction="none") / depth_gt_
        depthloss = torch.mean(depthloss) * cfg.depth_lambda 
        loss = loss + depthloss 

        # iso_loss = F.mse_loss(self.static_splats["scales"][:, 0], self.static_splats["scales"][:, 1]) * cfg.iso_lambda
        # loss = loss + iso_loss
        # scale_gs = torch.exp(self.static_splats["scales"])
        # scale_limit = cfg.prune_scale3d * self.scene_scale
        # scale_reg_loss = F.relu(scale_gs - scale_limit).mean() * cfg.scale_reg_lambda
        # loss = loss + scale_reg_loss
        # reg_loss = F.relu(1 - self.static_splats["lifespan"]).mean() * cfg.reg_lambda
        # loss = loss + reg_loss
        # normal consistency loss
        if cfg.model_type == "2dgs" and step > cfg.normal_start_iter:
            curr_normal_lambda = cfg.normal_lambda
            normals_from_depth_ = normals_from_depth[static_mask] * alphas_mask
            normal_error = (1 - (normals[static_mask] * normals_from_depth_).sum(dim=-1))
            normal_loss = curr_normal_lambda * normal_error.mean()
        else:
            normal_loss = torch.tensor(0.0, device=depthloss.device)
        loss = loss + normal_loss

        loss_dict = {
            "total_loss": loss.item(),
            "rgb_loss": rgb_loss.item(),
            "depth_loss": depthloss.item(),
            # "alpha_loss": alpha_loss.item(),
            # "reg_loss": reg_loss.item(),
            "normal_loss": normal_loss.item(),
        }
        return loss, loss_dict, info

    def tb_log(self, step, data, renders, loss_dict, stage):
        cfg = self.cfg
        if cfg.tb_every > 0 and step % cfg.tb_every == 0:
            feature, alphas, normals, normals_from_depth, render_distort, render_median, info = renders
            colors, depths = feature[..., 0:3], feature[..., -1]
            pixels = data["pixels"]
            depths_gt = data["depths_gt"]
            mem = torch.cuda.max_memory_allocated() / 1024**3
            self.writer.add_scalar(f"{stage}/mem", mem, step)
            self.writer.add_scalar(f"{stage}/static_num_GS", len(self.splats["means"]), step)
            if loss_dict is not None:
                self.writer.add_scalar(f"{stage}/loss", loss_dict["total_loss"], step)
                self.writer.add_scalar(f"{stage}/rgbloss", loss_dict["rgb_loss"], step)
                self.writer.add_scalar(f"{stage}/depthloss", loss_dict["depth_loss"], step)
                self.writer.add_scalar(f"{stage}/normalloss", loss_dict["normal_loss"], step)

            ids = data["image_ids"]
            static_mask = self.static_masks[ids].to(self.device)

            rgb_vis = torch.cat([pixels * static_mask.unsqueeze(-1), colors], dim=1).permute(0, 3, 1, 2).detach().cpu() # [B, C, H, W]
            depth_gt_vis = repeat(batch_normalize(depths_gt * static_mask), "b h w -> b c h w", c=3)
            depth_render_vis = repeat(batch_normalize(depths), "b h w -> b c h w", c=3)
            depth_vis = torch.cat([depth_gt_vis, depth_render_vis], dim=-2).detach().cpu()

            canvas = torch.cat([rgb_vis, depth_vis], dim=-1)
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

            if "colors" in self.splats:
                colors = torch.sigmoid(self.splats["colors"]).detach()
            else:
                colors = SH2RGB(self.splats["sh0"][:, 0]).detach()

            export_pointcloud(
                self.splats["means"].detach(),
                colors,
                os.path.join(self.ply_dir, f"{step}.ply")
            )
    
    def pbar_log(self, pbar, loss_dict, step):
        desc = f"loss={loss_dict['total_loss']:.3f}| "
        desc += f"depth loss={loss_dict['depth_loss']:.6f}| "
        # cfg = self.cfg
        # if cfg.pose_opt and cfg.pose_noise:
            # monitor the pose error if we inject noise
            # pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
            # desc += f"pose err={pose_err.item():.6f}| "
        pbar.set_description(desc)
    
    def save_checkpoint(self, step, meta = None):
        meta["num_GS"] = len(self.splats["means"])
        print("Step: ", step, meta)
        with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
            json.dump(meta, f)
        torch.save(
            {
                "static_splats": self.splats.state_dict(),
            },
            f"{self.ckpt_dir}/ckpt_{step}.pt",
        )

    def load_checkpoint(self, checkpoint):
        if not hasattr(self, "static_splats"):
            sh_degree = self.cfg.sh_degree
            self.splats = torch.nn.ParameterDict({
                "means": torch.nn.Parameter(torch.empty(0, 3)),
                "quats": torch.nn.Parameter(torch.empty(0, 4)),
                "scales": torch.nn.Parameter(torch.empty(0, 3)),
                "opacities": torch.nn.Parameter(torch.empty(0)),
                # "sh0": torch.nn.Parameter(torch.empty(0, 1, 3)),
                # "shN": torch.nn.Parameter(torch.empty(0, (sh_degree+1)**2 - 1, 3)),
                # "velocity": torch.nn.Parameter(torch.empty(0, 3)),
            })
            if sh_degree > 0:
                self.splats["sh0"] = torch.nn.Parameter(torch.empty(0, 1, 3))
                self.splats["shN"] = torch.nn.Parameter(torch.empty(0, (sh_degree+1)**2 - 1, 3))
            else:
                self.splats["colors"] = torch.nn.Parameter(torch.empty(0, 3))
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=self.device)
        if "static_splats" in checkpoint:
            checkpoint = checkpoint["static_splats"]
        matched_keys = []
        for k in self.splats.keys():
            if k in checkpoint:
                self.splats[k].data = checkpoint[k]
                matched_keys.append(k)
        unexpected_keys = set(checkpoint.keys()).difference(set(matched_keys))
        missed_keys = set(self.splats.keys()).difference(set(matched_keys))
        return missed_keys, unexpected_keys

    