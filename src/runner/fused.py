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
    export_pointcloud,
    SH2RGB,
    compute_normal_consistency_loss,
    compute_depth_loss,
    compute_ssim,
    quantile_loss,
    # get_edge_aware_weight,
)
from einops import rearrange, repeat
from gsplat.strategy import DefaultStrategy
from scene.strategy import DynamicStrategy
from config import Config
from scene.utils import *
from runner.base import BaseRunner
from torch import Tensor
from tqdm import tqdm
import math

class FusedRunner(BaseRunner):
    def __init__(
        self, cfg: Config
    ):
        super().__init__(cfg)
    
    def setup_splats(self, cfg):
        # Model
        (
            self.dynamic_splats,
            self.static_splats,
            self.dynamic_optimizers,
            self.static_optimizers
        ) = create_splats_with_optimizers(
            self.parser,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            init_lifespan=cfg.init_lifespan,
            lifespan_range=cfg.lifespan_range, 
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            batch_size=cfg.batch_size,
            device=self.device,
            model_type=cfg.model_type,
            
        )
        print(f"Model initialized.")
        print(f"Number of Dynamic GS: {len(self.dynamic_splats['means'])}")
        print(f"Number of Static GS: {len(self.static_splats['means'])}")
        # Densification Strategy
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
        self.dynamic_strategy = DynamicStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            prune_lifespan=cfg.prune_lifespan / cfg.lifespan_range,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            key_for_gradient=self.key_for_gradient
        )
        self.static_strategy.check_sanity(self.static_splats, self.static_optimizers)
        self.static_strategy_state = self.static_strategy.initialize_state(self.scene_scale)
        self.dynamic_strategy.check_sanity(self.dynamic_splats, self.dynamic_optimizers)
        self.dynamic_strategy_state = self.dynamic_strategy.initialize_state(self.scene_scale)

        max_steps = cfg.max_steps
        self.schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.static_optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.dynamic_optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.dynamic_optimizers["velocity_p"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.dynamic_optimizers["velocity_m"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.dynamic_optimizers["lifespan_p"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.dynamic_optimizers["lifespan_m"], gamma=0.01 ** (1.0 / max_steps)
            )
        ]
        
        self.static_masks = self.parser.static_masks
        self.dynamic_masks = self.parser.dynamic_masks

    def train_step(self, data, step, pbar):
        cfg = self.cfg
        pixels = data["pixels"]
        camtoworlds = data["camtoworlds"]
        height, width = pixels.shape[1:3]
        Ks = data["Ks"]
        image_ids = data["image_ids"].to(self.device)
        query_time = data["query_time"]
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        if cfg.pose_opt and step > cfg.pose_opt_start_iter:
            camtoworlds = self.pose_adjust(camtoworlds, image_ids)
        inputs = {
            "camtoworlds": camtoworlds,
            "Ks": Ks,
            "width": width,
            "height": height,
            "near_plane": cfg.near_plane,
            "far_plane": cfg.far_plane,
        }
        if step < cfg.init_steps:
            static_gaussians = self.get_splats(
                camtoworlds=camtoworlds,
                query_time=query_time,
                sh_degree=sh_degree_to_use,
                mode="static"
            )
            dynamic_gaussians = self.get_splats(
                camtoworlds=camtoworlds,
                query_time=query_time,
                sh_degree=sh_degree_to_use,
                mode="dynamic",
                features=["velocity_p", "velocity_m"]
            )
            renders_static = self.rasterize_splats(
                gaussians=static_gaussians,
                render_mode="RGB+ED",
                **inputs,
            )
            renders_dynamic = self.rasterize_splats(
                gaussians=dynamic_gaussians,
                render_mode="RGB+ED",
                **inputs
            )
            loss_static, loss_dict_static, info_static = self.compute_static_loss(data, renders_static, step)
            loss_dynamic, loss_dict_dynamic, info_dynamic = self.compute_dynamic_loss(data, renders_dynamic, step)
            loss = loss_static + loss_dynamic 
            loss.backward()
            info_static = squeeze_info(info_static, self.key_for_gradient)
            info_dynamic = squeeze_info(info_dynamic, self.key_for_gradient)
            loss_dict = {**loss_dict_static, **loss_dict_dynamic}
            self.pbar_log(pbar, loss_dict, step)
            self.tb_log(step, data, renders_static, loss_dict_static, stage="train/static")
            self.tb_log(step, data, renders_dynamic, loss_dict_dynamic, stage="train/dynamic")
        else:
            M = len(self.dynamic_splats["means"])
            N = len(self.static_splats["means"])
            gaussians = self.get_splats(
                camtoworlds=camtoworlds,
                query_time=query_time,
                sh_degree=sh_degree_to_use,
                mode="fused",
                features=["velocity_p", "velocity_m", "motion"]
            )
            renders = self.rasterize_splats(
                gaussians=gaussians,
                render_mode="RGB+ED",
                **inputs
            )
            loss, loss_dict, info = self.compute_loss(data, renders, gaussians, step)
            loss.backward()
            info = squeeze_info(info, self.key_for_gradient)
            info_static, info_dynamic = split_info(info, [N, M], 1, self.key_for_gradient)
            self.pbar_log(pbar, loss_dict, step)
            self.tb_log(step, data, renders, loss_dict, stage="train/fused")
           
        self.static_strategy.step_post_backward(
            params=self.static_splats,
            optimizers=self.static_optimizers,
            state=self.static_strategy_state,
            step=step,
            info=info_static,
        )
        self.dynamic_strategy.step_post_backward(
            params=self.dynamic_splats,
            optimizers=self.dynamic_optimizers,
            state=self.dynamic_strategy_state,
            step=step,
            info=info_dynamic,
        )
        # optimize
        for optimizer in self.static_optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.dynamic_optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in self.schedulers:
            scheduler.step()
        if cfg.pose_opt and step > cfg.pose_opt_start_iter:
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in self.pose_schedulers:
                scheduler.step()

    def val_step(self, data, step, pbar):
        cfg = self.cfg
        pixels = data["pixels"]
        camtoworlds = data["camtoworlds"]
        masks = data["masks"]
        height, width = pixels.shape[1:3]
        Ks = data["Ks"]
        query_time = data["query_time"]
        image_ids = data["image_ids"]
        assert len(image_ids) == 1, "Only support batch size 1 for test-time pose alignment."
        image_ids = image_ids[0]
        if cfg.test_time_pose_opt:
            camtoworlds = self.tt_pose_align[image_ids](camtoworlds, torch.tensor([0], device=self.device, dtype=torch.long))
        torch.cuda.synchronize()
        tic = time.time()
        gaussians = self.get_splats(
            camtoworlds=camtoworlds,
            query_time=query_time,
            sh_degree=cfg.sh_degree,
            mode="fused"
        )
        renders = self.rasterize_splats(
            gaussians=gaussians,
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            render_mode="RGB+ED",
        )
        torch.cuda.synchronize()
        ellipse_time = max(time.time() - tic, 1e-10)
        self.tb_log(step, data, renders, None, stage="val")
        pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
        colors = renders[0][..., :3].permute(0, 3, 1, 2)  # [1, 3, H, W]
        # colors = torch.clamp(colors, 0.0, 1.0)
        metrics = {
            "psnr" : self.psnr(colors, pixels),
            "ssim" : self.ssim(colors, pixels),
            "lpips" : self.lpips(
                colors * 2.0 - 1.0,
                pixels * 2.0 - 1.0),
            "mpsnr": self.mpsnr(colors, pixels, masks),
            "mssim": self.mssim(colors, pixels, masks),
            "mlpips": self.mlpips(colors, pixels, masks),

            "inv_mpsnr": self.mpsnr(colors, pixels, ~masks),
            "inv_mssim": self.mssim(colors, pixels, ~masks),
            "inv_mlpips": self.mlpips(colors, pixels, ~masks),

            "ellipse_time": ellipse_time
        }
        
        return metrics

    def test_time_pose_align(self, dataloader, step:int):
        cfg = self.cfg
        # freeze all gaussians
        self.static_splats.requires_grad_(False)
        self.dynamic_splats.requires_grad_(False)
        print("Start Test-Time Pose Alignment...")
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
            data = self.copy_data_to_device(data)
            pixels = data["pixels"]
            camtoworlds = data["camtoworlds"]
            height, width = pixels.shape[1:3]
            Ks = data["Ks"]
            image_ids = data["image_ids"]
            assert len(image_ids) == 1, "Only support batch size 1 for test-time pose alignment."
            image_ids = image_ids[0]
            query_time = data["query_time"]
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            inputs = {
                # "camtoworlds": camtoworlds,
                "Ks": Ks,
                "width": width,
                "height": height,
                "near_plane": cfg.near_plane,
                "far_plane": cfg.far_plane,
            }
            pose_optimizer = torch.optim.Adam( 
                self.tt_pose_align[image_ids].parameters(),
                lr=cfg.test_time_pose_opt_lr,
                # weight_decay=cfg.test_time_pose_opt_reg,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                pose_optimizer,
                T_max=self.cfg.test_time_pose_steps_each - self.cfg.test_time_decay_start,
                eta_min=self.cfg.test_time_post_opt_lr_final,
            )
            with torch.no_grad():
                gaussians = self.get_splats(
                    camtoworlds=camtoworlds, # sh??
                    query_time=query_time,
                    sh_degree=sh_degree_to_use,
                    mode="fused",
                    features=[]
                )
            # early stop
            eps = 2e-3
            last_psnr = 0.0
            for j in range(cfg.test_time_pose_steps_each):
                renders = self.rasterize_splats(
                    gaussians=gaussians,
                    render_mode="RGB",
                    camtoworlds=self.tt_pose_align[image_ids](camtoworlds, torch.tensor([0], device=self.device, dtype=torch.long)),
                    **inputs
                )
                rgb_pred = renders[0]
                mse = F.mse_loss(rgb_pred, pixels)
                loss = - 10 * torch.log10(1 / mse)
                psnr = - loss.item()
                loss.backward()
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)  
                if j >= cfg.test_time_decay_start:
                    scheduler.step()
                if abs(psnr - last_psnr) < eps:
                    break
                last_psnr = psnr
                # if j % 10 == 0 or j == cfg.test_time_pose_steps_each - 1:
                    # print(f"{i}th frame Test-Time Pose Align Step {j}: PSNR: {psnr.item():.2f}")
        self.static_splats.requires_grad_(True)
        self.dynamic_splats.requires_grad_(True)
    
    def refine_mask(self, dataloader, step:int):
        cfg = self.cfg
        for i, data in enumerate(dataloader):
            data = self.copy_data_to_device(data)
            depths_gt = data["depths_gt"]
            image_ids = data["image_ids"]
            camtoworlds = data["camtoworlds"]
            dynamic_mask = self.dynamic_masks[image_ids]
            height, width = depths_gt.shape[1:3]
            Ks = data["Ks"]
            query_time = data["query_time"]
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            inputs = {
                "camtoworlds": camtoworlds,
                "Ks": Ks,
                "width": width,
                "height": height,
                "near_plane": cfg.near_plane,
                "far_plane": cfg.far_plane,
                "render_mode": "ED",
            }
            gaussians = self.get_splats(
                camtoworlds=camtoworlds,
                query_time=query_time,
                sh_degree=sh_degree_to_use,
                mode="dynamic"
            )
            renders= self.rasterize_splats(
                gaussians=gaussians,
                **inputs
            )
            alpha = renders[1].squeeze(-1)
            _, H, W = alpha.shape
            valid = (alpha > 0.5).cpu() # b h w, bool
            frame_index = image_ids[0].item()
            choose_mask = valid & (~dynamic_mask)
            unchoose_mask = (~valid) & dynamic_mask

        export_video(
            os.path.join(self.visual_dir, f"refine_dynamic_{step}.mp4"),
            (self.dynamic_masks[..., None].float() * self.parser.images * 255.0).to(torch.uint8),
            fps=24
        )
        export_video(
            os.path.join(self.visual_dir, f"refine_static_{step}.mp4"),
            (self.static_masks[..., None].float() * self.parser.images * 255.0).to(torch.uint8),
            fps=24
        )


   
    def get_splats(
        self,
        query_time: Tensor,
        sh_degree:int,
        *args,
        **kwargs
    ):
        # dynamic_gaussians = kwargs.pop("dynamic_gaussians")
        # static_gaussians = kwargs.pop("static_gaussians")
        mode = kwargs.pop("mode")
        features = kwargs.pop("features", [])
        assert mode in ["fused", "dynamic", "static"]
        if mode in ["fused", "static"]:
            static_gaussians = activate_gaussians(self.static_splats, self.cfg.lifespan_range)
            static_gaussians = query_gaussian_t(
                static_gaussians,
                query_time=query_time,
                sh_degree=sh_degree,
                camera_pose=kwargs.get("camtoworlds", None),
                opacity_multiplier=self.cfg.opacity_multiplier,
            )
            static_gaussians["motion"] = torch.zeros_like(static_gaussians["opacities"]).unsqueeze(-1)
        if mode in ["fused", "dynamic"]:
            dynamic_gaussians = activate_gaussians(self.dynamic_splats, self.cfg.lifespan_range)
            dynamic_gaussians = query_gaussian_t(
                dynamic_gaussians,
                query_time=query_time,
                sh_degree=sh_degree,
                camera_pose=kwargs.get("camtoworlds", None),
                opacity_multiplier=self.cfg.opacity_multiplier,
            )
            dynamic_gaussians["motion"] = torch.ones_like(dynamic_gaussians["opacities"]).unsqueeze(-1) # the same as alpha
        gaussians = {}
        if mode == "fused":
            keys = ["means", "quats", "scales", "opacities", "colors", "velocity_p", "velocity_m", "motion"]
            for k in keys:
                gaussians[k] = torch.cat([static_gaussians[k], dynamic_gaussians[k]], dim=1)
        elif mode == "dynamic":
            gaussians = dynamic_gaussians
        else:
            gaussians = static_gaussians
        f = gaussians["colors"]
        for feature in features:
            f = torch.cat([f, gaussians[feature]], dim=-1)
        gaussians["colors"] = f
        return gaussians

    
    def compute_dynamic_loss(self, data:Dict[str, torch.Tensor], renders:Dict[str, torch.Tensor], step:int):
        (
            ids,
            depth_valid,
            fwd_velocity, fwd_velocity_mask,
            bwd_velocity, bwd_velocity_mask,
            pixels,
            depths_gt,
        ) = (
            data["image_ids"],
            data["depth_valid"],
            data["fwd_velocity"], data["fwd_velocity_mask"],
            data["bwd_velocity"], data["bwd_velocity_mask"],
            data["pixels"],
            data["depths_gt"],
        )

        dynamic_mask = self.dynamic_masks[ids].to(self.device)
        
        cfg = self.cfg
        (
            feature,
            alphas,
            normals,
            normals_from_depth,
            _,
            _,
            info
        ) = renders
        (
            colors,
            velocity_p_rendered,
            velocity_m_rendered,
            depths
        ) = feature[..., 0:3], feature[..., 3:6], feature[..., 6:9], feature[..., -1]
        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=self.device)
            colors = colors + bkgd * (1.0 - alphas)
        info[self.key_for_gradient].retain_grad()

        loss = 0.0
        rgb_loss = F.l1_loss(colors[dynamic_mask], pixels[dynamic_mask]) * (1.0 - cfg.ssim_lambda)
        loss = loss + rgb_loss

        fwd_velocity_loss = quantile_loss(
            velocity_p_rendered,
            fwd_velocity,
            fwd_velocity_mask,
            0.8
        )
        bwd_velocity_loss = quantile_loss(
            velocity_m_rendered,
            bwd_velocity,
            bwd_velocity_mask,
            0.8
        )
        velocity_loss = fwd_velocity_loss + bwd_velocity_loss
        velocity_loss = velocity_loss * cfg.velocity_lambda
        loss = loss + velocity_loss

        velocity_reg_loss = (
            torch.norm(self.dynamic_splats["velocity_p"], dim=-1) + \
            torch.norm(self.dynamic_splats["velocity_m"], dim=-1)
        ) / 2

        velocity_reg_thres = self.parser.scene_scale * cfg.velocity_reg_thres
        velocity_reg_loss = velocity_reg_loss[velocity_reg_loss > velocity_reg_thres]
        velocity_reg_loss = torch.mean(velocity_reg_loss) * cfg.velocity_reg_lambda
        loss = loss + velocity_reg_loss

        point_mask = dynamic_mask & depth_valid
        depthloss, _, _, _ = compute_depth_loss(depths, depths_gt, point_mask)
        depthloss = depthloss * cfg.depth_lambda
        loss = loss + depthloss 

        alphas = alphas.squeeze(-1)
        alphas = torch.clamp(alphas, 1e-6, 1.0 - 1e-6)
        alpha_loss = - (dynamic_mask.float() * torch.log(alphas) + (1.0 - dynamic_mask.float()) * torch.log(1 - alphas)).mean() * cfg.alpha_lambda
        loss = loss + alpha_loss
  
        # normal consistency loss
        if cfg.model_type == "2dgs" and step > cfg.normal_start_iter:
            normal_loss = compute_normal_consistency_loss(normals_from_depth, normals, alphas, point_mask) * cfg.normal_lambda
        else:
            normal_loss = torch.tensor(0.0, device=depthloss.device)
        
        loss = loss + normal_loss
        loss_dict = {
            "dynamic/loss": loss.item(),
            "dynamic/rgb_loss": rgb_loss.item(),
            "dynamic/velocity_loss": velocity_loss.item(),
            "dynamic/depth_loss": depthloss.item(),
            "dynamic/normal_loss": normal_loss.item(),
            "dynamic/velocity_reg_loss": velocity_reg_loss.item(),
            # "dynamic/alpha_loss": alpha_loss.item(),
        }
        return loss, loss_dict, info

    def compute_static_loss(self, data:Dict[str, torch.Tensor], renders:Dict[str, torch.Tensor], step:int):

        (
            ids,
            pixels,
            depths_gt,
            depth_valid
        ) = (
            data["image_ids"],
            data["pixels"],
            data["depths_gt"],
            data["depth_valid"]
        )
        static_mask = self.static_masks[ids].to(self.device)
        cfg = self.cfg
        (
            feature,
            alphas,
            normals,
            normals_from_depth,
            _,
            _,
            info
        ) = renders
        colors, depths = feature[..., 0:3], feature[..., -1]
        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=self.device)
            colors = colors + bkgd * (1.0 - alphas)
        info[self.key_for_gradient].retain_grad()

        loss = 0.0
        rgb_loss = F.l1_loss(colors[static_mask], pixels[static_mask]) * (1.0 - cfg.ssim_lambda)
        loss = loss + rgb_loss

        # edge_weight, edge_valid = get_edge_aware_weight(depths_gt / (self.parser.scene_scale / 2), mask=static_mask)
        # depthloss = F.l1_loss(depths[static_mask], depths_gt[static_mask], reduction="none") / depths_gt[static_mask]
        point_mask = static_mask  & depth_valid
        depthloss, _, _, _ = compute_depth_loss(depths, depths_gt, point_mask)
        depthloss = depthloss * cfg.depth_lambda
        loss = loss + depthloss 

        # normal consistency loss
        if cfg.model_type == "2dgs" and step > cfg.normal_start_iter:
            normal_loss = compute_normal_consistency_loss(normals_from_depth, normals, alphas, point_mask) * cfg.normal_lambda
        else:
            normal_loss = torch.tensor(0.0, device=depthloss.device)

        loss = loss + normal_loss
        if torch.any(torch.isnan(loss)):
            import pdb; pdb.set_trace()
        loss_dict = {
            "static/loss": loss.item(),
            "static/rgb_loss": rgb_loss.item(),
            "static/depth_loss": depthloss.item(),
            "static/normal_loss": normal_loss.item(),
            # "static/alpha_loss": alpha_loss.item(),
        }
        return loss, loss_dict, info
    
    def compute_loss(self, data:Dict[str, torch.Tensor], renders:Dict[str, torch.Tensor], gaussians:Dict[str, torch.Tensor], step:int):
        (
            ids,
            depth_valid,
            fwd_velocity, fwd_velocity_mask,
            bwd_velocity, bwd_velocity_mask,
            pixels,
            depths_gt,
        ) = (
            data["image_ids"],
            data["depth_valid"],
            data["fwd_velocity"], data["fwd_velocity_mask"],
            data["bwd_velocity"], data["bwd_velocity_mask"],
            data["pixels"],
            data["depths_gt"],
        )
        dynamic_mask = self.dynamic_masks[ids].to(self.device)
        cfg = self.cfg
        (
            feature,
            alphas,
            normals,
            normals_from_depth,
            _,
            _,
            info
        ) = renders
        (
            colors,
            velocity_p_rendered,
            velocity_m_rendered,
            motion,
            depths
        ) = feature[..., 0:3], feature[..., 3:6], feature[..., 6:9], feature[..., -2], feature[..., -1]
        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=self.device)
            colors = colors + bkgd * (1.0 - alphas)
        info[self.key_for_gradient].retain_grad()

        loss = 0.0
        rgb_loss = F.l1_loss(colors, pixels, reduction='mean') * (1.0 - cfg.ssim_lambda)
        loss = loss + rgb_loss

        # ssimloss = (1.0 - self.ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))) * cfg.ssim_lambda
        ssimloss = (1.0 - compute_ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))) * cfg.ssim_lambda
        loss = loss + ssimloss

        # depthloss = F.l1_loss(depths, depths_gt, reduction='none') / depths_gt
        depthloss, _, _, _ = compute_depth_loss(depths, depths_gt, depth_valid)
        depthloss = depthloss * cfg.depth_lambda
        loss = loss + depthloss 

        fwd_velocity_loss = quantile_loss(
            velocity_p_rendered,
            fwd_velocity,
            fwd_velocity_mask,
            0.8
        )
        bwd_velocity_loss = quantile_loss(
            velocity_m_rendered,
            bwd_velocity,
            bwd_velocity_mask,
            0.8
        )
        velocity_loss = fwd_velocity_loss + bwd_velocity_loss
        velocity_loss = velocity_loss * cfg.velocity_lambda
        loss = loss + velocity_loss


        velocity_reg_loss = (torch.norm(self.dynamic_splats["velocity_p"], dim=-1) + torch.norm(self.dynamic_splats["velocity_m"], dim=-1)) / 2
        velocity_reg_thres = self.parser.scene_scale * cfg.velocity_reg_thres
        velocity_reg_loss = velocity_reg_loss[velocity_reg_loss > velocity_reg_thres]
        velocity_reg_loss = torch.mean(velocity_reg_loss) * cfg.velocity_reg_lambda
        loss = loss + velocity_reg_loss

        motion = torch.clamp(motion, 1e-6, 1.0 - 1e-6)
        alpha_loss = - (dynamic_mask.float() * torch.log(motion) + (1.0 - dynamic_mask.float()) * torch.log(1 - motion)).mean() * cfg.alpha_lambda
        loss = loss + alpha_loss

        if cfg.model_type == "2dgs" and step > cfg.normal_start_iter:
            normal_loss = compute_normal_consistency_loss(normals_from_depth, normals, alphas, depth_valid) * cfg.normal_lambda
        else:
            normal_loss = torch.tensor(0.0, device=depthloss.device)
        loss = loss + normal_loss

        loss_dict = {
            "fused/loss": loss.item(),
            "fused/rgb_loss": rgb_loss.item(),
            "fused/ssim_loss": ssimloss.item(),
            "fused/depth_loss": depthloss.item(),
            "fused/normal_loss": normal_loss.item(),
            "fused/alpha_loss": alpha_loss.item(),
            "fused/velocity_loss": velocity_loss.item(),
            "fused/velocity_reg_loss": velocity_reg_loss.item(),
        }
        return loss, loss_dict, info

    
    def pbar_log(self, pbar, loss_dict, step):
        desc = ""
        for key in ["dynamic/loss", "static/loss", "fused/loss"]:
            if key in loss_dict:
                desc += f"{key}={loss_dict[key]:.3f}| "
        pbar.set_description(desc)
    
    def save_checkpoint(self, step, meta = None):
        meta["num_static_GS"] = len(self.static_splats["means"])
        meta["num_dynamic_GS"] = len(self.dynamic_splats["means"])
        print("Step: ", step, meta)
        with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
            json.dump(meta, f)
        torch.save(
            {
                "static_splats": self.static_splats.state_dict(),
                "dynamic_splats": self.dynamic_splats.state_dict()
            },
            f"{self.ckpt_dir}/ckpt_{step}.pt",
        )
        
        # Save static_mask and dynamic_mask as images
        mask_dir = os.path.join(self.ckpt_dir, f"mask_{step}")
        os.makedirs(mask_dir, exist_ok=True)
        
        # Save each frame's mask as an image
        for i in range(len(self.static_masks)):
            static_mask_img = (self.static_masks[i].float() * 255).cpu().numpy().astype(np.uint8)
            dynamic_mask_img = (self.dynamic_masks[i].float() * 255).cpu().numpy().astype(np.uint8)
            
            # Save static mask
            Image.fromarray(static_mask_img).save(
                os.path.join(mask_dir, f"static_mask_{i:04d}.png")
            )
            # Save dynamic mask  
            Image.fromarray(dynamic_mask_img).save(
                os.path.join(mask_dir, f"dynamic_mask_{i:04d}.png")
            )

    def load_checkpoint(self, checkpoint):
        if not hasattr(self, "static_splats"):
            sh_degree = self.cfg.sh_degree
            self.static_splats = torch.nn.ParameterDict({
                "means": torch.nn.Parameter(torch.empty(0, 3)),
                "quats": torch.nn.Parameter(torch.empty(0, 4)),
                "scales": torch.nn.Parameter(torch.empty(0, 3)),
                "opacities": torch.nn.Parameter(torch.empty(0)),
                "sh0": torch.nn.Parameter(torch.empty(0, 1, 3)),
                "shN": torch.nn.Parameter(torch.empty(0, (sh_degree+1)**2 - 1, 3)),
            })
        if not hasattr(self, "dynamic_splats"):
            self.dynamic_splats = torch.nn.ParameterDict({
                "means": torch.nn.Parameter(torch.empty(0, 3)),
                "quats": torch.nn.Parameter(torch.empty(0, 4)),
                "scales": torch.nn.Parameter(torch.empty(0, 3)),
                "opacities": torch.nn.Parameter(torch.empty(0)),
                "colors": torch.nn.Parameter(torch.empty(0, 3)),
                "velocity_p": torch.nn.Parameter(torch.empty(0, 3)),
                "velocity_m": torch.nn.Parameter(torch.empty(0, 3)),
                "lifespan_p": torch.nn.Parameter(torch.empty(0)),
                "lifespan_m": torch.nn.Parameter(torch.empty(0)),
                "temporal_center": torch.nn.Parameter(torch.empty(0)),
            })
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=self.device)
        static_matched_keys = []
        for k in self.static_splats.keys():
            if k in checkpoint:
                self.static_splats[k].data = checkpoint["static_splats"][k]
                static_matched_keys.append(k)
        static_unexpected_keys = set(checkpoint["static_splats"].keys()).difference(set(static_matched_keys))
        static_missed_keys = set(self.static_splats.keys()).difference(set(static_matched_keys))

        dynamic_matched_keys = []
        for k in self.dynamic_splats.keys():
            if k in checkpoint:
                self.dynamic_splats[k].data = checkpoint["dynamic_splats"][k]
                dynamic_matched_keys.append(k)
        dynamic_unexpected_keys = set(checkpoint["dynamic_splats"].keys()).difference(set(dynamic_matched_keys))
        dynamic_missed_keys = set(self.dynamic_splats.keys()).difference(set(dynamic_matched_keys))
        return static_missed_keys, static_unexpected_keys, dynamic_missed_keys, dynamic_unexpected_keys

    def tb_log(self, step, data, renders, loss_dict, stage):
        cfg = self.cfg
        if cfg.tb_every > 0 and (step % cfg.tb_every == 0 or stage=="val"):
            feature, _, normals, normals_from_depth, _, _, info = renders
            if feature.shape[-1] == 4:
                colors, depths = feature[..., 0:3], feature[..., -1]
                velocity_p_rendered, velocity_m_rendered, motion = None, None, None
            elif feature.shape[-1] == 10:
                colors, velocity_p_rendered, velocity_m_rendered, depths = feature[..., 0:3], feature[..., 3:6], feature[..., 6:9], feature[..., -1]
                motion = None
            else:
                (
                    colors,
                    velocity_p_rendered,
                    velocity_m_rendered,
                    motion,
                    depths
                ) = feature[..., 0:3], feature[..., 3:6], feature[..., 6:9], feature[..., -2], feature[..., -1]

            pixels = data["pixels"]
            mem = torch.cuda.max_memory_allocated() / 1024**3
            self.writer.add_scalar(f"{stage}/mem", mem, step)
            self.writer.add_scalar(f"{stage}/static_num_GS", len(self.static_splats["means"]), step)
            self.writer.add_scalar(f"{stage}/dynamic_num_GS", len(self.dynamic_splats["means"]), step)
            if loss_dict is not None:
                for key in loss_dict:
                    self.writer.add_scalar(f"{stage}/{key}", loss_dict[key], step)

            rgb_vis = torch.cat([pixels, colors], dim=1).permute(0, 3, 1, 2).detach().cpu() # [B, C, H, W]
            canvas = rgb_vis

            if "depths_gt" in data:
                depths_gt = data["depths_gt"]
                depth_gt_vis = repeat(batch_normalize(depths_gt), "b h w -> b c h w", c=3)
                depth_render_vis = repeat(batch_normalize(depths), "b h w -> b c h w", c=3)
                depth_vis = torch.cat([depth_gt_vis, depth_render_vis], dim=-2).detach().cpu()
                canvas = torch.cat([canvas, depth_vis], dim=-1)

            if normals_from_depth is not None:
                normals_vis = rearrange(normals, "b h w c -> b c h w")
                normals_from_depth_vis = rearrange(normals_from_depth, "b h w c -> b c h w")
                normals_vis = (torch.cat([normals_vis, normals_from_depth_vis], dim=-2) / 2 + 0.5).detach().cpu()
                canvas = torch.cat([canvas, normals_vis], dim=-1)

            if velocity_p_rendered is not None:
                image_ids = data["image_ids"]
                dynamic_mask = self.dynamic_masks[image_ids].to(self.device)
                velocity = data["fwd_velocity"]
                velocity_render_vis = velocity_p_rendered / (torch.norm(velocity_p_rendered, dim=-1, keepdim=True) + 1e-5)
                velocity_gt_vis = velocity / (torch.norm(velocity, dim=-1, keepdim=True) + 1e-5)
                velocity_vis = torch.cat([velocity_gt_vis * dynamic_mask.unsqueeze(-1), velocity_render_vis * dynamic_mask.unsqueeze(-1)], dim=1) / 2 + 0.5
                velocity_vis = rearrange(velocity_vis, "b h w c -> b c h w").detach().cpu()
                canvas = torch.cat([canvas, velocity_vis], dim=-1)
            if velocity_m_rendered is not None:
                image_ids = data["image_ids"]
                dynamic_mask = self.dynamic_masks[image_ids].to(self.device)
                velocity = data["bwd_velocity"]
                velocity_render_vis = velocity_m_rendered / (torch.norm(velocity_m_rendered, dim=-1, keepdim=True) + 1e-5)
                velocity_gt_vis = velocity / (torch.norm(velocity, dim=-1, keepdim=True) + 1e-5)
                velocity_vis = torch.cat([velocity_gt_vis * dynamic_mask.unsqueeze(-1), velocity_render_vis * dynamic_mask.unsqueeze(-1)], dim=1) / 2 + 0.5
                velocity_vis = rearrange(velocity_vis, "b h w c -> b c h w").detach().cpu()
                canvas = torch.cat([canvas, velocity_vis], dim=-1)
            if motion is not None:
                motion = repeat(motion, "b h w -> b c h w", c=3)
                dyn_mask = repeat(dynamic_mask, "b h w -> b c h w", c=3)
                motion_vis = torch.cat([dyn_mask, motion], dim=-2).detach().cpu()
                canvas = torch.cat([canvas, motion_vis], dim=-1)

            canvas = make_grid(canvas, nrow=1)
            canvas = torch.clamp(canvas, 0.0, 1.0)
            self.writer.add_image(f"{stage}/render", canvas, step)
            Image.fromarray(
                (canvas.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
            ).save(os.path.join(self.render_dir, f"{stage.replace('/', '_')}_{step}.png"))
            self.writer.flush()
            # if "colors" in self.static_splats:
            #     colors_static = torch.sigmoid(self.static_splats["colors"]).detach()
            #     # colors_static = self.static_splats["colors"].detach()
            # else:
            #     colors_static = SH2RGB(self.static_splats["sh0"][:, 0]).detach()

            # export_pointcloud(
            #     self.static_splats["means"].detach(),
            #     colors_static,
            #     os.path.join(self.ply_dir, f"static_{step}.ply")
            # )
            # export_pointcloud(
            #     self.dynamic_splats["means"].detach(),
            #     torch.sigmoid(self.dynamic_splats["colors"]).detach(),
            #     # self.dynamic_splats["colors"].detach(),
            #     os.path.join(self.ply_dir, f"dynamic_{step}.ply")
            # )

    