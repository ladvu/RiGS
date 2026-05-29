import json
import os
import time
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid
import math
import imageio.v3 as iio
from utils import (
    query_gaussian_t,
    activate_gaussians,
    batch_normalize,
    export_video,
    export_pointcloud,
    SH2RGB,
    transform_gaussians,
    decay_gaussians,
    compute_tracking_feature,
    compute_velocity_feature,
    compute_normal_consistency_loss,
    compute_depth_loss,
    compute_ssim,
    # compute_grad_loss,
    quantile_loss,
    masked_l1_loss,
    compute_se3_smoothness_loss,
    closed_form_inverse_se3,
    erode,
    make_transient,
    fit_gaussian_mixture,
    find_local_minimum_between_means
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
from flow3d.params import MotionBases
from gsplat.strategy.ops import remove, reset_opa

class FusedRunner(BaseRunner):
    def __init__(
        self, cfg: Config
    ):
        super().__init__(cfg)
    
    def setup_splats(self, cfg):
        if cfg.rescale_gaussian:
            self.scale_factor = self.parser.scene_scale
        else:
            self.scale_factor = 1.0
        # Model
        (
            self.static_splats,
            self.static_optimizers
        ) = create_static_splats_with_optimizers(
            self.parser,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale / self.scale_factor,
            scene_scale=self.parser.d_scale,
            sh_degree=cfg.sh_degree,
            batch_size=cfg.batch_size,
            device=self.device,
            model_type=cfg.model_type,
            color_activation=cfg.color_activation
        )
        print(f"Model initialized.")
        print(f"Number of Static GS: {len(self.static_splats['means'])}")
        # Densification Strategy
        self.static_strategy = DefaultStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d * self.parser.scene_scale,
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            key_for_gradient=self.key_for_gradient,
        )
        self.static_strategy.check_sanity(self.static_splats, self.static_optimizers)
        self.static_strategy_state = self.static_strategy.initialize_state(self.parser.d_scale)

        max_steps = cfg.max_steps
        self.schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            # torch.optim.lr_scheduler.ExponentialLR(
            #     self.static_optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            # ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.static_optimizers["scales"], gamma=0.1 ** (1.0 / max_steps)
            ),
        ]
        
        self.static_masks = erode(self.parser.static_masks, cfg.mask_kernel_size)
        self.dynamic_masks = self.parser.dynamic_masks
        self.setup_motion_base_splats(cfg)

        self.transient_splats = None
        self.transient_optimizers = None
        self.transient_strategy = None
    
    def setup_motion_base_splats(self, cfg:Config):
        motion_base_path = os.path.join(self.parser.cache_data_path, "motion_base.pth")
        fg_params_path = os.path.join(self.parser.cache_data_path, "fg_params.pth")
        t_l_path = os.path.join(self.parser.cache_data_path, "temporal_lifespan.pth")

        if os.path.exists(motion_base_path) and os.path.exists(fg_params_path) and os.path.exists(t_l_path):
            fg_params = torch.load(fg_params_path, map_location="cpu")
            motion_bases = torch.load(motion_base_path, map_location="cpu")
            t_l = torch.load(t_l_path, map_location="cpu")
            temporal_centers = t_l["temporal_center"]
            lifespans = t_l["lifespan"]
            self.motion_bases = MotionBases(motion_bases["params.rots"], motion_bases["params.transls"]).to(self.device)
            self.dynamic_splats = torch.nn.ParameterDict(
                {n.replace("params.", ""): torch.nn.Parameter(v) for n, v in fg_params.items() if n.startswith("params.")}
            ).to(self.device)
        else:
            fg_params, motion_bases, tracks_3d = create_motion_base_splats(
                self.parser,
                cfg.num_fg,
                cfg.num_motion_bases,
                vis=cfg.vis,
                port=cfg.port,
                init_scale= cfg.init_scale/self.scale_factor,
                color_activation=cfg.color_activation,
            )
            # run initial optimization
            Ks = self.parser.get_Ks().to(self.device)
            w2cs = self.parser.get_w2cs().to(self.device)
            run_initial_optim(fg_params, motion_bases, tracks_3d, Ks, w2cs, scene_scale=self.parser.d_scale)

            # temporal_center and lifespan from track3d. 
            # temporal_center of a gaussian will be the mean of the observed frames
            # lifespan will be (max_observed_frame - min_observed_frame + 1)
            # Ideally tracking should be continuous. Need to extract the longest observed segment? really nessessary?
            # For fast processing, use first and last visible frames directly
            visibles = tracks_3d.visibles # G, T
            M, T = visibles.shape
            first_visible = torch.argmax(visibles.long(), dim=1)  # (num_visible,)
            # Find last visible frame (search from the end)
            flipped = torch.flip(visibles, dims=[1])
            last_visible_from_end = torch.argmax(flipped.long(), dim=1)
            last_visible = T - 1 - last_visible_from_end  # (num_visible,)
            # Calculate temporal center and lifespan
            span_center = (first_visible + last_visible).float() / 2.0
            span_length = (last_visible - first_visible).float() / 2.0
            
            temporal_centers = torch.nan_to_num(span_center, nan=T/2.0, posinf=T/2.0, neginf=T/2.0)
            lifespans = torch.nan_to_num(span_length, nan=T/2.0, posinf=T/2.0, neginf=T/2.0)
                    
            torch.save(motion_bases.state_dict(), motion_base_path)
            torch.save(fg_params.state_dict(), fg_params_path)
            torch.save({"temporal_center": temporal_centers, "lifespan": lifespans}, t_l_path)

            self.dynamic_splats = fg_params.params
            self.motion_bases = motion_bases
        if cfg.vis and cfg.port is not None:
            from flow3d.vis.utils import  get_server
            from flow3d.init_utils import vis_tracks_3d
            server = get_server(cfg.port)
            # try:
            #     means = self.dynamic_splats["means"]
            #     coefs = F.softmax(self.dynamic_splats["motion_coefs"], dim=-1)
            #     clrs = torch.sigmoid(self.dynamic_splats["colors"]).detach().cpu().numpy()
            #     while True:
            #         for t in range(self.parser.num_frames):
            #             with torch.no_grad():
            #                 transform = self.motion_bases.compute_transforms(torch.tensor([t], device=self.device, dtype=torch.long), coefs)
            #                 pts = torch.einsum(
            #                     "pnij,pj->pni",
            #                     transform,
            #                     F.pad(means, (0, 1), value=1.0),
            #                 )
            #             server.scene.add_point_cloud("points", pts[:, 0].detach().cpu().numpy(), clrs, point_size=0.01 * self.parser.scene_scale)
            #             time.sleep(0.3)
            # except KeyboardInterrupt:
            #     pass
            idcs = np.random.choice(len(self.dynamic_splats["means"]), 100)
            labels = np.linspace(0, 1, 100)
            ts = torch.arange(self.motion_bases.num_frames, device=self.device)
            with torch.no_grad():
                coefs = F.softmax(self.dynamic_splats["motion_coefs"], dim=-1)
                transfms = self.motion_bases.compute_transforms(ts, coefs)
                pred_means = torch.einsum(
                    "pnij,pj->pni",
                    transfms,
                    F.pad(self.dynamic_splats["means"], (0, 1), value=1.0),
                )
                vis_means = pred_means[idcs].detach().cpu().numpy()
            vis_tracks_3d(server, vis_means, labels, name="init_params")

        # add lifespan, temporal_center parameter
        M = len(self.dynamic_splats["means"])
        T = self.parser.num_frames 
        
        self.dynamic_splats["lifespan"] = torch.nn.Parameter(lifespans.clone().cuda())
        self.dynamic_splats["temporal_center"] = torch.nn.Parameter(temporal_centers.clone().cuda())

        splats_lr_config = {
            "means": 1.6e-4 * self.parser.d_scale,
            "opacities": 1e-2,
            "scales": 5e-3,
            "quats": 1e-3,
            "colors": 1e-2,
            "motion_coefs": 1e-2,
            "lifespan": 1e-3,
            "temporal_center": 1e-3,
        }

        self.dynamic_optimizers = {
            name: torch.optim.Adam(
                [{"params": self.dynamic_splats[name], "lr": lr * math.sqrt(cfg.batch_size)}],
                eps=1e-15 / math.sqrt(cfg.batch_size),
                betas=(1 - cfg.batch_size * (1 - 0.9), 1 - cfg.batch_size * (1 - 0.999)),
            )
            for name, lr in splats_lr_config.items() 
        }
        motion_base_lr_config = {
            "rots": 1e-4,
            "transls":  1.6e-4 * self.parser.d_scale,
        }
        self.motion_base_optimizers = {
            name: torch.optim.Adam(
                [{"params": self.motion_bases.params[name], "lr": lr * math.sqrt(cfg.batch_size)}],
                eps=1e-15 / math.sqrt(cfg.batch_size),
                betas=(1 - cfg.batch_size * (1 - 0.9), 1 - cfg.batch_size * (1 - 0.999)),
            )
            for name, lr in motion_base_lr_config.items()
        }
        self.dynamic_strategy = DefaultStrategy(
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
        self.dynamic_strategy.check_sanity(self.dynamic_splats, self.dynamic_optimizers)
        self.dynamic_strategy_state = self.dynamic_strategy.initialize_state(self.parser.d_scale)
        self.schedulers.append(
            # torch.optim.lr_scheduler.ExponentialLR(
            #     self.dynamic_optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps)
            # )
            torch.optim.lr_scheduler.ExponentialLR(
                self.dynamic_optimizers["scales"], gamma=0.1 ** (1.0 / cfg.max_steps)
            )
        )

        # self.schedulers.append(
        #     torch.optim.lr_scheduler.ExponentialLR(
        #         self.motion_base_optimizers["transls"], gamma=0.1 ** (1.0 / cfg.max_steps)
        #     )
        # )
        # self.schedulers.append(
        #     torch.optim.lr_scheduler.ExponentialLR(
        #         self.motion_base_optimizers["rots"], gamma=0.1 ** (1.0 / cfg.max_steps)
        #     )
        # )
        
    def train_step(self, data, step, pbar):
        cfg = self.cfg
        pixels = data["pixels"]
        camtoworlds = data["camtoworlds"]
        height, width = pixels.shape[1:3]
        Ks = data["Ks"]
        image_ids = data["image_ids"].to(self.device)
        query_time = data["query_time"]
        target_time = data["target_ts"]
        target_w2cs = data["target_w2cs"]
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        if cfg.pose_opt and step > cfg.pose_opt_start_iter:
            camtoworlds = self.pose_adjust(camtoworlds, image_ids)
            b, t = target_w2cs.shape[:2]
            target_c2ws = closed_form_inverse_se3(target_w2cs.flatten(0, 1))
            target_c2ws = self.pose_adjust(target_c2ws, target_time.long().flatten())
            target_w2cs = closed_form_inverse_se3(target_c2ws).reshape(b, t, 4, 4)
        render_inputs = {
            "camtoworlds": camtoworlds,
            "Ks": Ks,
            "width": width,
            "height": height,
            "near_plane": cfg.near_plane,
            "far_plane": cfg.far_plane,
            "render_mode": "RGB+ED",
            # "backgrounds": cfg.background_color,
        }
        query_inputs = {
            "camtoworlds": camtoworlds,
            "query_time": query_time,
            "sh_degree": sh_degree_to_use,
            "target_ts": target_time,
            "target_w2cs": target_w2cs,
        }
        M = len(self.dynamic_splats["means"])
        N = len(self.static_splats["means"])
        if step < cfg.init_steps:
            static_gaussians = self.get_splats(
                mode="static",
                **query_inputs
            )
            dynamic_gaussians = self.get_splats(
                mode="dynamic",
                features=["tf", "vf", "m", "t", "d"],
                **query_inputs
            )
            renders_static = self.rasterize_splats(
                gaussians=static_gaussians,
                **render_inputs,
            )
            renders_dynamic = self.rasterize_splats(
                gaussians=dynamic_gaussians,
                **render_inputs
            )
            loss_static, loss_dict_static, info_static = self.compute_static_loss(data, renders_static, step)
            loss_dynamic, loss_dict_dynamic, info_dynamic = self.compute_dynamic_loss(data, renders_dynamic, step)
            loss = loss_static + loss_dynamic 
            # loss = loss_static
            loss.backward()
            info_static = squeeze_info(info_static, self.key_for_gradient)
            info_dynamic = squeeze_info(info_dynamic, self.key_for_gradient)
            if self.transient_strategy is not None:
                L = len(self.transient_splats["means"])
                info_dynamic, info_transient = split_info(info_dynamic, [M, L], 1, self.key_for_gradient)
            loss_dict = {**loss_dict_static, **loss_dict_dynamic}
            # loss_dict = loss_dict_static
            self.pbar_log(pbar, loss_dict, step)
            self.tb_log(step, data, renders_static, loss_dict_static, stage="train/static")
            self.tb_log(step, data, renders_dynamic, loss_dict_dynamic, stage="train/dynamic")
        else:
            # import pdb; pdb.set_trace()
            gaussians = self.get_splats(
                mode="fused",
                features=["tf", "vf", "m", "t", "d"],
                **query_inputs
            )
            renders = self.rasterize_splats(
                gaussians=gaussians,
                **render_inputs
            )
            # import pdb; pdb.set_trace()
            loss, loss_dict, info = self.compute_loss(data, renders, gaussians, step)
            loss.backward()
            info = squeeze_info(info, self.key_for_gradient)
            if self.transient_strategy is not None:
                L = len(self.transient_splats["means"])
                info_static, info_dynamic, info_transient = split_info(info, [N, M, L], 1, self.key_for_gradient)
            else:
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
        if self.transient_strategy is not None:
            self.transient_strategy.step_post_backward(
                params=self.transient_splats,
                optimizers=self.transient_optimizers,
                state=self.transient_strategy_state,
                step=step,
                info=info_transient,
            )

        if (not cfg.no_transient_gaussian 
            and step >= cfg.transient_start_step
            and step <= cfg.transient_end_step
            and step % cfg.transient_every == 0):
            self.add_transient_gaussians(step)
        # optimize
        for optimizer in self.static_optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.dynamic_optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.motion_base_optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if self.transient_optimizers is not None:
            for optimizer in self.transient_optimizers.values():
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

    @torch.no_grad()
    def add_transient_gaussians(self, step:int):
        print(f"Step {step}")
        print(f"Rigid Gaussians {len(self.dynamic_splats['means'])}")
        print(f"Non-Rigid Gaussians {len(self.transient_splats['means']) if self.transient_splats is not None else 0}")
        # import pdb; pdb.set_trace()
        gaussians, is_transient = make_transient(
            self.dynamic_splats,
            self.motion_bases,
            self.cfg.lifespan_thres,
            self.cfg.lifespan_range,
            vis=self.cfg.vis_lifespan,
            save_path=os.path.join(self.stats_dir, f"lifespan_{step}.png"),
            init_opacity=self.cfg.prune_opa * 2.0,
        )
        if gaussians is not None and is_transient is not None:
            print(f"Adding {len(gaussians['means'])} transient Gaussians.")
            if self.transient_splats is None:
                self.init_transient_gaussians(gaussians, step)
            else:
                add_new_gs(gaussians, self.transient_splats, self.transient_optimizers, self.transient_strategy_state)
            # import pdb; pdb.set_trace()
            # is_prune = is_transient | (torch.sigmoid(self.dynamic_splats["opacities"].flatten()) < self.cfg.prune_opa)
            is_prune = is_transient
            print(f"Removing {is_prune.sum().item()} rigid Gaussians.")
            remove(self.dynamic_splats, self.dynamic_optimizers, self.dynamic_strategy_state, is_prune)
            # reset_opa(
            #     params=self.dynamic_splats,
            #     optimizers=self.dynamic_optimizers,
            #     state=self.dynamic_strategy_state,
            #     value=self.cfg.prune_opa * 2.0,
            # )
            print(f"After removal, Rigid Gaussians {len(self.dynamic_splats['means'])}")
            print(f"After addition, Non-Rigid Gaussians {len(self.transient_splats['means']) if self.transient_splats is not None else 0}")
        else:
            print("No transient Gaussians added.")
    
    def init_transient_gaussians(self, gaussians, step:int):
        
        self.transient_splats = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(value) for name, value in gaussians.items()}
        ).to(self.device)
        lr_dict = {
            "means" :  1.6e-4 * self.parser.d_scale,
            "scales":  5e-4,
            "quats":  1e-3,
            "opacities": 5e-2,
            "temporal_center": 1e-4,
            "lifespan_p": 5e-2, 
            "lifespan_m": 5e-2,
            "velocity_p": 1e-3,
            "velocity_m": 1e-3,
            "colors": 1e-2,
        }
        self.transient_optimizers = {
            name: torch.optim.Adam(
                [{"params": self.transient_splats[name], "lr": lr * math.sqrt(self.cfg.batch_size)}],
                eps=1e-15 / math.sqrt(self.cfg.batch_size),
                betas=(1 - self.cfg.batch_size * (1 - 0.9), 1 - self.cfg.batch_size * (1 - 0.999)),
            ) for name, lr in lr_dict.items()
        }
        self.transient_strategy = DefaultStrategy(
            verbose=True,
            prune_opa=self.cfg.prune_opa,
            grow_grad2d=self.cfg.grow_grad2d,
            grow_scale3d=self.cfg.grow_scale3d,
            prune_scale3d=self.cfg.prune_scale3d,
            refine_start_iter=self.cfg.transient_start_step,
            refine_stop_iter=self.cfg.transient_end_step,
            reset_every=self.cfg.reset_every,
            refine_every=self.cfg.refine_every,
            key_for_gradient=self.key_for_gradient,
        )
        self.transient_strategy.check_sanity(self.transient_splats, self.transient_optimizers)
        self.transient_strategy_state = self.transient_strategy.initialize_state(self.parser.d_scale)
        self.schedulers.append(
            torch.optim.lr_scheduler.ExponentialLR(
                self.transient_optimizers["scales"], gamma=0.1 ** (1.0 / (self.cfg.max_steps - self.cfg.transient_start_step))
            )
        )

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
            backgrounds=cfg.background_color
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
            "lpips" : self.lpips(colors, pixels, normalize=True),
            "mpsnr": self.mpsnr(colors, pixels, masks),
            "mssim": self.mssim(colors, pixels, masks),
            "mlpips": self.mlpips(colors, pixels, masks, normalize=True),

            "inv_mpsnr": self.mpsnr(colors, pixels, ~masks),
            "inv_mssim": self.mssim(colors, pixels, ~masks),
            "inv_mlpips": self.mlpips(colors, pixels, ~masks, normalize=True),

            "ellipse_time": ellipse_time
        }
        
        return metrics

    def test_time_pose_align(self, dataloader, step:int):
        cfg = self.cfg
        # freeze all gaussians
        self.static_splats.requires_grad_(False)
        self.dynamic_splats.requires_grad_(False)
        self.motion_bases.requires_grad_(False)
        if self.transient_splats is not None:
            self.transient_splats.requires_grad_(False)
        print("Start Test-Time Pose Alignment...")
        aligned_pose = []
        inds = []
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
            data = self.copy_data_to_device(data)
            pixels = data["pixels"]
            camtoworlds = data["camtoworlds"]
            height, width = pixels.shape[1:3]
            Ks = data["Ks"]
            image_ids = data["image_ids"]
            masks = data["masks"]
            cam_ids = data["cam_ids"]
            masks = {
                "psnr": torch.ones_like(masks),
                "mpsnr": masks, 
                "inv_psnr": ~masks
            }[cfg.test_time_psnr]
            assert len(image_ids) == 1, "Only support batch size 1 for test-time pose alignment."
            image_ids = image_ids[0]
            cam_ids = cam_ids[0] - 1
            cam_initialized = self.tt_pose_align_init[cam_ids]
            query_time = data["query_time"]
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            tt_pose_align = self.tt_pose_align[image_ids]
            inputs = {
                # "camtoworlds": camtoworlds,
                "Ks": Ks,
                "width": width,
                "height": height,
                "near_plane": cfg.near_plane,
                "far_plane": cfg.far_plane,
                "backgrounds": cfg.background_color,
            }
            if not cam_initialized:
                lr = cfg.test_time_pose_opt_lr
                lr_final = cfg.test_time_pose_opt_lr_final
                total_steps = cfg.test_time_pose_steps_each
                decay_start = cfg.test_time_decay_start
                self.tt_pose_align_init[cam_ids] = True
            else:
                lr = cfg.test_time_pose_refine_lr
                lr_final = cfg.test_time_pose_refine_lr_final
                total_steps = cfg.test_time_pose_refine_steps_each
                decay_start = cfg.test_time_pose_refine_decay_start
                # If already tt aligned before, no need to init from prev cam. use its own pose
                if not self.tt_init:
                    last_cam_weight = self.tt_pose_align[image_ids - 1].embeds.weight.data.detach().clone()
                    tt_pose_align.embeds.weight.data.copy_(last_cam_weight)

            pose_optimizer = torch.optim.Adam( 
                tt_pose_align.parameters(),
                lr=lr
                # weight_decay=cfg.test_time_pose_opt_reg,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                pose_optimizer,
                T_max=total_steps - decay_start,
                eta_min=lr_final
            )
            with torch.no_grad():
                gaussians = self.get_splats(
                    camtoworlds=camtoworlds, # sh??
                    query_time=query_time,
                    sh_degree=sh_degree_to_use,
                    mode="fused",
                    features=[]
                )
            max_psnr = 0.0
            cur_c2w = None
            for j in range(total_steps):
                c2ws = tt_pose_align(camtoworlds, torch.tensor([0], device=self.device, dtype=torch.long))
                renders = self.rasterize_splats(
                    gaussians=gaussians,
                    render_mode="RGB",
                    camtoworlds=c2ws,
                    **inputs
                )
                rgb_pred = renders[0]
                mse = F.mse_loss(rgb_pred[masks], pixels[masks])
                loss = - 10 * torch.log10(1 / mse)
                psnr = - loss.item()
                # loss = mse
                # psnr = 10 * torch.log10(1 / mse).item()
                loss.backward()
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)  
                if j >= decay_start:
                    scheduler.step()
                if j % 10 == 0 or j == cfg.test_time_pose_steps_each - 1:
                    print(f"{i}th frame Test-Time Pose Align Step {j}: PSNR: {psnr:.2f}")
                if psnr > max_psnr:
                    max_psnr = psnr
                    cur_c2w = c2ws[0].detach().cpu().numpy()
            aligned_pose.append(cur_c2w)
            inds.append(image_ids.item())
        aligned_pose = np.stack(aligned_pose, axis=0)
        inds = np.array(inds)
        if self.parser.normalize_transform is not None and self.parser.normalize_scale is not None:
            aligned_pose[:, :3, 3] = aligned_pose[:, :3, 3] * self.parser.normalize_scale
            aligned_pose = np.einsum("ij, tjk -> tik", np.linalg.inv(self.parser.normalize_transform), aligned_pose)

        save_path = os.path.join(self.ckpt_dir, f"pose_{step}.npz")
        np.savez(
            save_path,
            data=aligned_pose,
            inds=inds
        )
        self.static_splats.requires_grad_(True)
        self.dynamic_splats.requires_grad_(True)
        self.motion_bases.requires_grad_(True)
        if self.transient_splats is not None:
            self.transient_splats.requires_grad_(True)
        self.tt_init = True
    
    def refine_mask(self, dataloader, step:int):
        raise NotImplementedError("Refine mask not implemented yet.")
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
        target_time = kwargs.pop("target_ts", None)
        target_w2cs = kwargs.pop("target_w2cs", None)
        assert mode in ["fused", "dynamic", "static", "rigid", "transient"], f"Unknown mode {mode} for get_splats."
        enable_tracking = target_time is not None and target_w2cs is not None and "tf" in features
        if mode in ["fused", "static"]:
            static_gaussians = activate_gaussians(
                self.static_splats,
                self.scale_factor,
                0.1,
                self.cfg.lifespan_range,
                self.cfg.lifespan_activation, 
                self.cfg.color_activation
            )
            static_gaussians = query_gaussian_t(
                static_gaussians,
                query_time=query_time,
                sh_degree=sh_degree,
                camera_pose=kwargs.get("camtoworlds", None),
                opacity_multiplier=self.cfg.opacity_multiplier,
            )
            static_gaussians["m"] = torch.zeros_like(static_gaussians["opacities"]).unsqueeze(-1)
            static_gaussians["t"] = torch.zeros_like(static_gaussians["opacities"]).unsqueeze(-1)
            static_gaussians["d"] = torch.zeros_like(static_gaussians["opacities"]).unsqueeze(-1)
            if enable_tracking:
                static_gaussians["tf"] = compute_tracking_feature(self.static_splats["means"], None, self.motion_bases, target_time, target_w2cs)
            else:
                static_gaussians["tf"] = None

        if mode in ["fused", "dynamic", "transient"]:
            if self.transient_splats is not None:
                transient_gaussians = activate_gaussians(self.transient_splats, self.scale_factor, 0.1, self.cfg.lifespan_range, self.cfg.lifespan_activation, self.cfg.color_activation)
                transient_gaussians = query_gaussian_t(
                    transient_gaussians,
                    query_time=query_time,
                    sh_degree=sh_degree,
                    camera_pose=kwargs.get("camtoworlds", None),
                    opacity_multiplier=self.cfg.opacity_multiplier,
                )
                if enable_tracking:
                    transient_gaussians["tf"] = torch.zeros(
                        (query_time.shape[0], len(self.transient_splats["means"]), target_time.shape[1] * 3),
                        device=self.device, dtype=torch.float32
                    )
                else:
                    transient_gaussians["tf"] = None
                transient_gaussians["m"] = torch.ones_like(transient_gaussians["opacities"]).unsqueeze(-1)
                transient_gaussians["t"] = torch.ones_like(transient_gaussians["opacities"]).unsqueeze(-1)
                transient_gaussians["d"] = torch.zeros_like(transient_gaussians["opacities"]).unsqueeze(-1)
            else:
                transient_gaussians = None

        if mode in ["fused", "dynamic", "rigid"]:
            # import pdb; pdb.set_trace()
            rigid_gaussians = activate_gaussians(
                self.dynamic_splats,
                self.scale_factor,
                0.1,
                self.cfg.lifespan_range,
                self.cfg.lifespan_activation,
                self.cfg.color_activation
            )
            motion_coefs = rigid_gaussians["motion_coefs"]
            (
                rigid_gaussians["means"],
                rigid_gaussians["quats"]
            ) = transform_gaussians(
                rigid_gaussians["means"],
                rigid_gaussians["quats"],
                motion_coefs,
                self.motion_bases,
                query_time
            )
            if not self.cfg.no_decay_gaussians:
                rigid_gaussians["opacities"] = decay_gaussians(
                    rigid_gaussians["opacities"],
                    rigid_gaussians["temporal_center"],
                    query_time,
                    rigid_gaussians["lifespan"],
                    self.parser.num_frames,
                    sharpness=self.cfg.decay_sharpness,
                    method=self.cfg.decay_method,
                    opacity_multiplier=self.cfg.opacity_multiplier,
                )
            else:
                rigid_gaussians["opacities"] = repeat(rigid_gaussians["opacities"], "p -> b p", b=query_time.shape[0])

            for key in ["scales", "colors"]:
                rigid_gaussians[key] = repeat(rigid_gaussians[key], "p ... -> b p ...", b=query_time.shape[0])
            
            rigid_gaussians["m"] = torch.ones_like(rigid_gaussians["opacities"]).unsqueeze(-1) # the same as alpha
            rigid_gaussians["d"] = torch.ones_like(rigid_gaussians["opacities"]).unsqueeze(-1)
            rigid_gaussians["t"] = torch.zeros_like(rigid_gaussians["opacities"]).unsqueeze(-1) # t for transient
            if enable_tracking:
                rigid_gaussians["tf"] = compute_tracking_feature(self.dynamic_splats["means"], motion_coefs, self.motion_bases, target_time, target_w2cs)
            else:
                rigid_gaussians["tf"] = None
            if "vf" in features:
                rigid_gaussians["vf"] = compute_velocity_feature(self.dynamic_splats["means"], motion_coefs, self.motion_bases, query_time)
            else:
                rigid_gaussians["vf"] = None

        if mode in ["fused", "dynamic"]:
            dynamic_gaussians = {}
            if transient_gaussians is not None:
                keys = ["m", "t", "d", "tf", "vf", "means", "quats", "scales", "opacities", "colors"]
                for k in keys:
                    if transient_gaussians[k] is not None and rigid_gaussians[k] is not None:
                        dynamic_gaussians[k] = torch.cat([rigid_gaussians[k], transient_gaussians[k]], dim=1)
                    else:
                        dynamic_gaussians[k] = None
            else:
                dynamic_gaussians = rigid_gaussians

        gaussians = {}
        if mode == "fused":
            keys = ["m", "t", "d", "tf", "vf", "means", "quats", "scales", "opacities", "colors"]
            for k in keys:
                if static_gaussians[k] is not None and dynamic_gaussians[k] is not None:
                    gaussians[k] = torch.cat([static_gaussians[k], dynamic_gaussians[k]], dim=1)
        elif mode == "dynamic":
            gaussians = dynamic_gaussians
        elif mode == "transient":
            gaussians = transient_gaussians
        elif mode == "rigid":
            gaussians = rigid_gaussians
        else:
            gaussians = static_gaussians
        if gaussians is not None:
            f = gaussians["colors"]
            # import pdb; pdb.set_trace()
            for feature in features:
                f = torch.cat([f, gaussians[feature]], dim=-1)
            gaussians["colors"] = f
        return gaussians
    
    def compute_dynamic_loss(self, data:Dict[str, torch.Tensor], renders:Dict[str, torch.Tensor], step:int):
        loss_inputs, render_info = self.parse_data_renders(data, renders)
        loss, loss_dict = self.multitask_loss(target_mask=loss_inputs["dynamic_mask"], step=step, **loss_inputs)
        # scale_loss = self.cfg.scale_lambda * torch.var(torch.exp(self.dynamic_splats["scales"][:, :2]), dim=-1).mean()
        # loss += scale_loss
        # loss_dict["scale_loss"] = scale_loss.item()
        # if step % 500 == 0:
            # import pdb; pdb.set_trace()
        new_loss_dict = {} 
        for key in loss_dict:
            new_loss_dict["dynamic/" + key] = loss_dict[key]
        # import pdb; pdb.set_trace()
        return loss, new_loss_dict, render_info

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

        ssimloss = (1.0 - compute_ssim(
            (colors * static_mask.float().unsqueeze(-1)).permute(0, 3, 1, 2),
            (pixels * static_mask.float().unsqueeze(-1)).permute(0, 3, 1, 2))) * cfg.ssim_lambda
        loss = loss + ssimloss

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

        if self.cfg.model_type == "2dgs":
            scales = torch.exp(self.static_splats["scales"][:, :2])
            scale_loss = self.cfg.scale_lambda * F.mse_loss(scales[:, 0], scales[:, 1])
        else:
            scale_loss = self.cfg.scale_lambda * torch.var(torch.exp(self.static_splats["scales"]), dim=-1).mean()
        loss += scale_loss

        if torch.any(torch.isnan(loss)):
            import pdb; pdb.set_trace()
        loss_dict = {
            "static/loss": loss.item(),
            "static/rgb_loss": rgb_loss.item(),
            "static/depth_loss": depthloss.item(),
            "static/normal_loss": normal_loss.item(),
            # "static/scale_loss": scale_loss.item(),
            # "static/ssim_loss": ssimloss.item(),
        }
        return loss, loss_dict, info
    
    def compute_loss(self, data:Dict[str, torch.Tensor], renders:Dict[str, torch.Tensor], gaussians:Dict[str, torch.Tensor], step:int):
        loss_inputs, render_info = self.parse_data_renders(data, renders)
        loss, loss_dict = self.multitask_loss(target_mask=torch.ones_like(loss_inputs["dynamic_mask"]), step=step, **loss_inputs)
        if self.cfg.model_type == "2dgs":
            scales = torch.exp(self.static_splats["scales"][:, :2])
            scale_loss = self.cfg.scale_lambda * F.mse_loss(scales[:, 0], scales[:, 1])
        else:
            scale_loss = self.cfg.scale_lambda * torch.var(torch.exp(self.static_splats["scales"]), dim=-1).mean()
        # scale_loss_d = self.cfg.scale_lambda * torch.var(torch.exp(self.dynamic_splats["scales"][:, :2]), dim=-1).mean()
        # scale_loss_s = self.cfg.scale_lambda * torch.var(torch.exp(self.static_splats["scales"][:, :2]), dim=-1).mean()
        # scale_loss = self.cfg.scale_lambda * (scale_loss_d + scale_loss_s)
        loss += scale_loss
        loss_dict["scale_loss"] = scale_loss.item()
        # import pdb; pdb.set_trace()
        new_loss_dict = {} 
        for key in loss_dict:
            new_loss_dict["fused/" + key] = loss_dict[key]
        return loss, new_loss_dict, render_info

    def parse_data_renders(self, data: Dict[str, torch.Tensor], renders:Dict[str, torch.Tensor]):   
        (
            ids,
            depth_valid,
            fwd_velocity_gt, fwd_velocity_mask,
            bwd_velocity_gt, bwd_velocity_mask,
            pixels,
            depths_gt,
            query_tracks_2d,
            target_ts,
            # target_w2cs,
            target_Ks,
            target_tracks_2d,
            target_visibles,
            # target_invisibles,
            target_confidences,
            target_track_depths,
            target_track_depth_valid,
            ts,
        ) = (
            data["image_ids"],
            data["depth_valid"],
            data["fwd_velocity"], data["fwd_velocity_mask"],
            data["bwd_velocity"], data["bwd_velocity_mask"],
            data["pixels"],
            data["depths_gt"],
            data["query_tracks_2d"],
            data["target_ts"],
            # data["target_w2cs"],
            data["target_Ks"],
            data["target_tracks_2d"],
            data["target_visibles"],
            # data["target_invisibles"],
            data["target_confidences"],
            data["target_track_depths"],
            data["target_track_depth_valid"],
            data["query_time"]
        )
        cfg = self.cfg
        N = cfg.num_targets_per_frame
        tf_dim = N * 3
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
            tracking_feature,
            velocity_feature,
            motion,
            transience,
            dynamism,
            depths
        ) = feature[..., 0:3], feature[..., 3:3+tf_dim], feature[..., 3+tf_dim:-4], feature[..., -4], feature[..., -3], feature[..., -2], feature[..., -1]
        fwd_velocity, bwd_velocity = velocity_feature[..., :3], velocity_feature[..., 3:]
        pred_tracks_3d = rearrange(tracking_feature, "b h w (n f) -> (b n) (h w) f", n=N)
        pred_tracks_2d = torch.einsum(
            "bij,bpj->bpi", target_Ks.flatten(0, 1), pred_tracks_3d
        )
        # (B * N, H * W, 1).
        mapped_depth = pred_tracks_2d[..., 2:]
        # (B * N, H * W, 2).
        pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(mapped_depth, 1e-6)
        target_ts_vec = target_ts.flatten()
        frame_intervals = (ts.repeat_interleave(N) - target_ts_vec).abs()
        num_frames = self.parser.num_frames
        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=self.device)
            colors = colors + bkgd * (1.0 - alphas)
        info[self.key_for_gradient].retain_grad()

        w_interval = torch.exp(-2 * frame_intervals / num_frames)

        # (P_all, 2).
        tracks_2d = torch.cat([x.reshape(-1, 2) for x in target_tracks_2d], dim=0)
        # (P_all,)
        visibles = torch.cat([x.reshape(-1) for x in target_visibles], dim=0)
        # (P_all,)
        confidences = torch.cat([x.reshape(-1) * w_interval[i] for i, x in enumerate(target_confidences)], dim=0)
        # (P_all,)
        mapped_depth_gt = torch.cat([x.reshape(-1) for x in target_track_depths], dim=0)
        # (P_all,)
        track_depth_valid = torch.cat([x.reshape(-1) for x in target_track_depth_valid], dim=0)

        # w_track_loss = min(1, (self.max_steps - self.global_step) / 6000)
        track_weights = confidences[..., None] 
        track_depth_valid = track_depth_valid[..., None]
        masks_flatten = torch.zeros_like(depth_valid)
        B, H, W = masks_flatten.shape

        # import pdb; pdb.set_trace()
        for i in range(B):
            # This takes advantage of the fact that the query 2D tracks are
            # always on the grid.
            query_pixels = query_tracks_2d[i].to(torch.int64)
            masks_flatten[i, query_pixels[:, 1], query_pixels[:, 0]] = 1.0
        # (B * N, H * W).
        masks_flatten = (
            masks_flatten.reshape(-1, H * W).tile(1, N).reshape(-1, H * W) > 0.5
        )
        return dict(
            colors=colors,
            pixels=pixels,
            # target mask
            dynamic_mask=self.dynamic_masks[ids].to(self.device),
            depths=depths,
            depths_gt=depths_gt,
            depth_valid=depth_valid,
            fwd_velocity=fwd_velocity,
            bwd_velocity=bwd_velocity,
            fwd_velocity_gt=fwd_velocity_gt, fwd_velocity_mask=fwd_velocity_mask,
            bwd_velocity_gt=bwd_velocity_gt, bwd_velocity_mask=bwd_velocity_mask,
            motion=motion,
            transience=transience,
            ts=ts,
            pred_tracks_2d=pred_tracks_2d,
            tracks_2d=tracks_2d,
            masks_flatten=masks_flatten,
            visibles=visibles,
            track_weights=track_weights,
            mapped_depth=mapped_depth,
            mapped_depth_gt=mapped_depth_gt,
            track_depth_valid=track_depth_valid,
            normals_from_depth=normals_from_depth,
            normals=normals,
            alphas=alphas,
            H=H,W=W,
            num_frames=num_frames,
            # step
        ), info

    def multitask_loss(
        self,
        colors: torch.Tensor,
        pixels: torch.Tensor,
        target_mask: torch.Tensor,
        dynamic_mask: torch.Tensor,
        depths: torch.Tensor,
        depths_gt: torch.Tensor,
        depth_valid: torch.Tensor,
        fwd_velocity: torch.Tensor,
        bwd_velocity: torch.Tensor,
        fwd_velocity_gt: torch.Tensor, fwd_velocity_mask: torch.Tensor,
        bwd_velocity_gt: torch.Tensor, bwd_velocity_mask: torch.Tensor,
        motion: torch.Tensor,
        transience: torch.Tensor,
        ts: torch.Tensor,
        pred_tracks_2d: torch.Tensor,
        tracks_2d: torch.Tensor,
        masks_flatten: torch.Tensor,
        visibles: torch.Tensor,
        track_weights: torch.Tensor,
        mapped_depth: torch.Tensor,
        mapped_depth_gt: torch.Tensor,
        track_depth_valid: torch.Tensor,
        normals_from_depth: torch.Tensor,
        normals: torch.Tensor,
        alphas: torch.Tensor,
        H: int,
        W: int,
        num_frames: int,
        step:int,
    ):
        cfg = self.cfg

        loss = 0.0
        rgb_loss = F.l1_loss(colors[target_mask], pixels[target_mask]) * (1.0 - cfg.ssim_lambda)
        loss = loss + rgb_loss
        
        ssimloss = (1.0 - compute_ssim(
            (colors * target_mask.float().unsqueeze(-1)).permute(0, 3, 1, 2),
            (pixels * target_mask.float().unsqueeze(-1)).permute(0, 3, 1, 2))) * cfg.ssim_lambda
        loss = loss + ssimloss

        point_mask = target_mask & depth_valid
        depthloss, _, _, _ = compute_depth_loss(depths, depths_gt, point_mask)
        depthloss = depthloss * cfg.depth_lambda
        depthloss = torch.nan_to_num(depthloss, nan=0.0, posinf=0.0, neginf=0.0)
        loss = loss + depthloss 

        # normal consistency loss
        if cfg.model_type == "2dgs" and step > cfg.normal_start_iter:
            normal_loss = compute_normal_consistency_loss(normals_from_depth, normals, alphas, point_mask) * cfg.normal_lambda
        else:
            normal_loss = torch.tensor(0.0, device=depthloss.device)
        loss = loss + normal_loss
        if self.cfg.alpha_loss_fn == "bce":
            motion_safe = torch.clamp(motion, 1e-6, 1.0 - 1e-6)
            alpha_loss = - (dynamic_mask.float() * torch.log(motion_safe) + (1.0 - dynamic_mask.float()) * torch.log(1 - motion_safe)).mean() * cfg.alpha_lambda
        else:
            alpha_loss = F.l1_loss(motion, dynamic_mask.float()) * cfg.alpha_lambda
        loss = loss + alpha_loss

        motion_mask = (motion.detach() > 0.5).repeat_interleave(cfg.num_targets_per_frame, dim=0).flatten(-2)
        transient_mask = (transience.detach() > 0.5).repeat_interleave(cfg.num_targets_per_frame, dim=0).flatten(-2)
        weights = (track_weights[visibles] * track_depth_valid[visibles] * motion_mask[masks_flatten][visibles] * (~transient_mask[masks_flatten][visibles])).detach()
        track_2d_loss = masked_l1_loss(
            pred_tracks_2d[masks_flatten][visibles],
            tracks_2d[visibles],
            weights=weights,
            quantile=0.98,
        ) * (cfg.track_lambda / max(H, W))
        loss += track_2d_loss 

        # scale problem
        mapped_depth = mapped_depth[masks_flatten][visibles]
        if mapped_depth.numel() > 10:
            mapped_depth_loss = F.l1_loss(
                mapped_depth,
                mapped_depth_gt[visibles, None],
                reduction="none",
            ) * weights
            mapped_depth_loss = torch.mean(mapped_depth_loss) * (cfg.track_depth_lambda / self.parser.scene_scale)
        else:
            mapped_depth_loss = torch.tensor(0.0, device=depthloss.device)
        loss += mapped_depth_loss 

        small_accel_loss = compute_se3_smoothness_loss(
            self.motion_bases.params["rots"],
            self.motion_bases.params["transls"],
            weight_transl=cfg.translation_lambda / self.parser.scene_scale
        )* cfg.smooth_base_lambda
        small_accel_loss = torch.nan_to_num(small_accel_loss, nan=0.0, posinf=0.0, neginf=0.0)
        loss += small_accel_loss 

        ts = torch.clamp(ts, min=1, max=num_frames - 2)
        ts_neighbors = torch.cat((ts - 1, ts, ts + 1))
        transfms_nbs = self.motion_bases.compute_transforms(ts_neighbors, F.softmax(self.dynamic_splats["motion_coefs"], dim=-1))  # (G, 3n, 3, 4)
        means_fg_nbs = torch.einsum(
            "pnij,pj->pni",
            transfms_nbs,
            F.pad(self.dynamic_splats["means"], (0, 1), value=1.0),
        )
        means_fg_nbs = means_fg_nbs.reshape(
            means_fg_nbs.shape[0], 3, -1, 3
        )  # [G, 3, n, 3]
        small_accel_loss_tracks = 0.5 * (
            (2 * means_fg_nbs[:, 1:-1] - means_fg_nbs[:, :-2] - means_fg_nbs[:, 2:])
            .norm(dim=-1)
            .mean()
        )  * (cfg.smooth_track_lambda / self.parser.scene_scale)
        small_accel_loss_tracks = torch.nan_to_num(small_accel_loss_tracks, nan=0.0, posinf=0.0, neginf=0.0)
        loss += small_accel_loss_tracks 
        # import pdb; pdb.set_trace()
        fwd_velocity_loss = quantile_loss(fwd_velocity, fwd_velocity_gt, fwd_velocity_mask & depth_valid, 0.90)
        bwd_velocity_loss = quantile_loss(bwd_velocity, bwd_velocity_gt, bwd_velocity_mask & depth_valid, 0.90)
        velocity_loss = (fwd_velocity_loss + bwd_velocity_loss) * (cfg.velocity_lambda / self.parser.scene_scale)
        loss += velocity_loss

        lifespan_loss = torch.mean(1.0 / F.softplus(self.dynamic_splats["lifespan"])) * cfg.lifespan_lambda
        lifespan_loss = torch.nan_to_num(lifespan_loss, nan=0.0, posinf=0.0, neginf=0.0)
        loss += lifespan_loss

        # locality_loss = compute_grad_loss(fwd_velocity, dynamic_mask) + compute_grad_loss(bwd_velocity, dynamic_mask)
        # locality_loss = locality_loss * cfg.locality_lambda
        # loss += locality_loss

        if torch.any(torch.isnan(loss)):
            import pdb; pdb.set_trace()
        
        loss_dict = {
            "loss": loss.item(),
            "rgb_loss": rgb_loss.item(),
            "depth_loss": depthloss.item(),
            "normal_loss": normal_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "track_2d_loss": track_2d_loss.item(),
            "mapped_depth_loss": mapped_depth_loss.item(),
            "small_accel_loss": small_accel_loss.item(),
            "small_accel_loss_tracks": small_accel_loss_tracks.item(),
            "ssim_loss": ssimloss.item(),
            "velocity_loss": velocity_loss.item(),
            # "lifespan_loss": lifespan_loss.item(),
        }
        
        return loss, loss_dict
    
    def pbar_log(self, pbar, loss_dict, step):
        desc = ""
        for key in ["dynamic/loss", "static/loss", "fused/loss"]:
            if key in loss_dict:
                desc += f"{key}={loss_dict[key]:.3f}| "
        pbar.set_description(desc)
    
    def save_checkpoint(self, step, meta = None):
        meta["num_static_GS"] = len(self.static_splats["means"])
        meta["num_dynamic_GS"] = len(self.dynamic_splats["means"])
        meta["num_transient_GS"] = len(self.transient_splats["means"]) if self.transient_splats is not None else 0
        print("Step: ", step, meta)
        with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
            json.dump(meta, f)
        # adjust scale
        state_dict_s = self.static_splats.state_dict()
        state_dict_d = self.dynamic_splats.state_dict()
        state_dict_s["scales"] = state_dict_s["scales"] + math.log(self.scale_factor)
        state_dict_d["scales"] = state_dict_d["scales"] + math.log(self.scale_factor)
        state_dict = {
            "static_splats": state_dict_s,
            "dynamic_splats": state_dict_d,
            "motion_bases": self.motion_bases.state_dict(),
        }
        if self.transient_splats is not None:
            state_dict_t = self.transient_splats.state_dict()
            state_dict_t["scales"] = state_dict_t["scales"] + math.log(self.scale_factor)
            state_dict["transient_splats"] = state_dict_t

        torch.save(
            state_dict,  
            f"{self.ckpt_dir}/ckpt_{step}.pt",
        )
        
        # # Save static_mask and dynamic_mask as images
        # mask_dir = os.path.join(self.ckpt_dir, f"mask_{step}")
        # os.makedirs(mask_dir, exist_ok=True)
        
        # # Save each frame's mask as an image
        # for i in range(len(self.static_masks)):
        #     static_mask_img = (self.static_masks[i].float() * 255).cpu().numpy().astype(np.uint8)
        #     dynamic_mask_img = (self.dynamic_masks[i].float() * 255).cpu().numpy().astype(np.uint8)
            
        #     # Save static mask
        #     Image.fromarray(static_mask_img).save(
        #         os.path.join(mask_dir, f"static_mask_{i:04d}.png")
        #     )
        #     # Save dynamic mask  
        #     Image.fromarray(dynamic_mask_img).save(
        #         os.path.join(mask_dir, f"dynamic_mask_{i:04d}.png")
        #     )

    def load_checkpoint(self, checkpoint):
        if self.static_splats is None:
            sh_degree = self.cfg.sh_degree
            color_dict = {
                "colors": torch.nn.Parameter(torch.empty(0, 3)),
            } if sh_degree == 0 else {
                "sh0": torch.nn.Parameter(torch.empty(0, 1, 3)),
                "shN": torch.nn.Parameter(torch.empty(0, (sh_degree+1)**2 - 1, 3)),
            }
            self.static_splats = torch.nn.ParameterDict({
                "means": torch.nn.Parameter(torch.empty(0, 3)),
                "quats": torch.nn.Parameter(torch.empty(0, 4)),
                "scales": torch.nn.Parameter(torch.empty(0, 3)),
                "opacities": torch.nn.Parameter(torch.empty(0)),
                **color_dict,
            })
        if self.dynamic_splats is None:
            self.dynamic_splats = torch.nn.ParameterDict({
                "means": torch.nn.Parameter(torch.empty(0, 3)),
                "quats": torch.nn.Parameter(torch.empty(0, 4)),
                "scales": torch.nn.Parameter(torch.empty(0, 3)),
                "opacities": torch.nn.Parameter(torch.empty(0)),
                "colors": torch.nn.Parameter(torch.empty(0, 3)),
                "motion_coefs": torch.nn.Parameter(torch.empty(0, self.motion_bases.num_bases)),
                "temporal_center": torch.nn.Parameter(torch.empty(0)),
                "lifespan": torch.nn.Parameter(torch.empty(0)),
            })
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=self.device)
        # import pdb; pdb.set_trace()
        static_matched_keys = []
        for k in self.static_splats.keys():
            if k in checkpoint["static_splats"]:
                self.static_splats[k].data = checkpoint["static_splats"][k]
                static_matched_keys.append(k)
        static_unexpected_keys = set(checkpoint["static_splats"].keys()).difference(set(static_matched_keys))
        static_missed_keys = set(self.static_splats.keys()).difference(set(static_matched_keys))

        dynamic_matched_keys = []
        for k in self.dynamic_splats.keys():
            if k in checkpoint["dynamic_splats"]:
                self.dynamic_splats[k].data = checkpoint["dynamic_splats"][k]
                dynamic_matched_keys.append(k)
        self.motion_bases = MotionBases.init_from_state_dict(checkpoint["motion_bases"])
        dynamic_unexpected_keys = set(checkpoint["dynamic_splats"].keys()).difference(set(dynamic_matched_keys))
        dynamic_missed_keys = set(self.dynamic_splats.keys()).difference(set(dynamic_matched_keys))

        if "transient_splats" in checkpoint:
            if self.transient_splats is None:
                self.transient_splats = torch.nn.ParameterDict({
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
            transient_matched_keys = []
            for k in self.transient_splats.keys():
                if k in checkpoint["transient_splats"]:
                    self.transient_splats[k].data = checkpoint["transient_splats"][k]
                    transient_matched_keys.append(k)
            transient_unexpected_keys = set(checkpoint["transient_splats"].keys()).difference(set(transient_matched_keys))
            transient_missed_keys = set(self.transient_splats.keys()).difference(set(transient_matched_keys))
        else:
            transient_missed_keys, transient_unexpected_keys = set(), set()

        return static_missed_keys, static_unexpected_keys, dynamic_missed_keys, dynamic_unexpected_keys, transient_missed_keys, transient_unexpected_keys

    def tb_log(self, step, data, renders, loss_dict, stage):
        cfg = self.cfg
        if cfg.tb_every > 0 and (step % cfg.tb_every == 0 or stage=="val"):
            feature, _, normals, normals_from_depth, _, _, info = renders
            if feature.shape[-1] == 4:
                colors, depths = feature[..., 0:3], feature[..., -1]
                transience, motion, dynamism, tracking_feature, velocity_feature = None, None, None, None, None
            else:
                N = cfg.num_targets_per_frame
                tf_dim = N * 3
                (
                    colors,
                    tracking_feature,
                    velocity_feature,
                    motion,
                    transience,
                    dynamism,
                    depths
                ) = feature[..., 0:3], feature[..., 3:3+tf_dim], feature[..., 3+tf_dim:-4], feature[..., -4], feature[..., -3], feature[..., -2], feature[..., -1]

            pixels = data["pixels"]
            mem = torch.cuda.max_memory_allocated() / 1024**3
            self.writer.add_scalar(f"{stage}/mem", mem, step)
            self.writer.add_scalar(f"{stage}/static_num_GS", len(self.static_splats["means"]), step)
            self.writer.add_scalar(f"{stage}/dynamic_num_GS", len(self.dynamic_splats["means"]), step)
            self.writer.add_scalar(f"{stage}/transient_num_GS", len(self.transient_splats["means"]) if self.transient_splats is not None else 0, step) 
            if loss_dict is not None:
                for key in loss_dict:
                    self.writer.add_scalar(f"{stage}/{key}", loss_dict[key], step)

            rgb_vis = torch.cat([pixels, colors], dim=1).permute(0, 3, 1, 2).detach().cpu() # [B, C, H, W]
            canvas = rgb_vis
            if stage != "val":
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

                if velocity_feature is not None:
                    image_ids = data["image_ids"]
                    dynamic_mask = self.dynamic_masks[image_ids].to(self.device)
                    fwd_v_gt = data["fwd_velocity"]
                    bwd_v_gt = data["bwd_velocity"]
                    fwd_v = velocity_feature[..., :3]
                    bwd_v = velocity_feature[..., 3:]

                    fwd_gt_vis = fwd_v_gt / (torch.norm(fwd_v_gt, dim=-1, keepdim=True) + 1e-5)
                    bwd_gt_vis = bwd_v_gt / (torch.norm(bwd_v_gt, dim=-1, keepdim=True) + 1e-5)
                    fwd_vis = fwd_v / (torch.norm(fwd_v, dim=-1, keepdim=True) + 1e-5)
                    bwd_vis = bwd_v / (torch.norm(bwd_v, dim=-1, keepdim=True) + 1e-5)
                    # import pdb; pdb.set_trace()

                    fwd_vis = torch.cat([fwd_gt_vis * dynamic_mask.unsqueeze(-1), fwd_vis * dynamic_mask.unsqueeze(-1)], dim=1) / 2 + 0.5
                    bwd_vis = torch.cat([bwd_gt_vis * dynamic_mask.unsqueeze(-1), bwd_vis * dynamic_mask.unsqueeze(-1)], dim=1) / 2 + 0.5
                    velocity_vis = torch.cat([fwd_vis, bwd_vis], dim=2)

                    velocity_vis = rearrange(velocity_vis, "b h w c -> b c h w").detach().cpu()
                    canvas = torch.cat([canvas, velocity_vis], dim=-1)
                if motion is not None:
                    image_ids = data["image_ids"]
                    dynamic_mask = self.dynamic_masks[image_ids].to(self.device)
                    motion_vis = repeat(motion, "b h w -> b c h w", c=3)
                    dyn_mask = repeat(dynamic_mask, "b h w -> b c h w", c=3)
                    motion_vis = torch.cat([dyn_mask, motion_vis], dim=-2).detach().cpu()
                    canvas = torch.cat([canvas, motion_vis], dim=-1)
                if transience is not None:
                    # image_ids = data["image_ids"]
                    # dynamic_mask = self.dynamic_masks[image_ids].to(self.device)
                    # dyn_mask = repeat(dynamic_mask, "b h w -> b c h w", c=3)
                    dynamism = repeat(dynamism, "b h w -> b c h w", c=3)
                    transience = repeat(transience, "b h w -> b c h w", c=3)
                    transience_vis = torch.cat([dynamism, transience], dim=-2).detach().cpu()
                    canvas = torch.cat([canvas, transience_vis], dim=-1)

            canvas = make_grid(canvas, nrow=1)
            canvas = torch.clamp(canvas, 0.0, 1.0)
            self.writer.add_image(f"{stage}/render", canvas, step)
            if stage != "val":
                save_path = os.path.join(self.render_dir, f"{stage.replace('/', '_')}_{step}.png")
            else:
                save_dir = os.path.join(self.render_dir, f"val_{step}")
                os.makedirs(save_dir, exist_ok=True)
                cam_ids = int(data["cam_ids"][0].item())
                query_time = int(data["query_time"][0].item())
                save_path = os.path.join(save_dir, f"{cam_ids}_{query_time:05d}.png")

            Image.fromarray(
                (canvas.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
            ).save(save_path)
            self.writer.flush()

            if stage.startswith("train"):
                color_activation_fn = {
                    "sigmoid": torch.sigmoid,
                    "sh_relu": lambda x: torch.clamp_min(SH2RGB(x), 0.0),
                }[self.cfg.color_activation]
                if "colors" in self.static_splats:
                    # colors_static = torch.sigmoid(self.static_splats["colors"]).detach()
                    colors_static = color_activation_fn(self.static_splats["colors"].detach())
                else:
                    colors_static = SH2RGB(self.static_splats["sh0"][:, 0]).detach()

                export_pointcloud(
                    self.static_splats["means"].detach(),
                    colors_static,
                    os.path.join(self.ply_dir, f"static_{step}.ply")
                )
                export_pointcloud(
                    self.dynamic_splats["means"].detach(),
                    color_activation_fn(self.dynamic_splats["colors"].detach()),
                    os.path.join(self.ply_dir, f"dynamic_{step}.ply")
                )
                if self.transient_splats is not None:
                    export_pointcloud(
                        self.transient_splats["means"].detach(),
                        color_activation_fn(self.transient_splats["colors"].detach()),
                        os.path.join(self.ply_dir, f"transient_{step}.ply")
                    )

    def render_step(self, data, step:int, name=""):
        cfg = self.cfg
        gaussian_types = cfg.render_types
        if "transient" in gaussian_types and self.transient_splats is None:
            gaussian_types.remove("transient")
        w2cs = data["w2cs"]
        Ks = data["Ks"]
        width = data["width"]   
        height = data["height"]
        for mode in gaussian_types:
            output_path = os.path.join(self.render_dir, f"{name}_{step}_{mode}.mp4")
            frames = []
            for i in range(len(w2cs)):
                query_time = torch.tensor([i], device=self.device, dtype=torch.float32)
                camtoworlds = torch.linalg.inv(w2cs[i:i+1])
                K = Ks[i:i+1]
                gaussians = self.get_splats(
                    camtoworlds=camtoworlds,
                    query_time=query_time,
                    sh_degree=cfg.sh_degree,
                    mode=mode
                )
                renders = self.rasterize_splats(
                    gaussians=gaussians,
                    camtoworlds=camtoworlds,
                    Ks=K,
                    width=width,
                    height=height,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    render_mode="RGB",
                    backgrounds=cfg.background_color
                )
                colors = renders[0][0, ..., :3]
                frames.append(((colors.detach().cpu().numpy()) * 255.0).astype(np.uint8))
            frames = np.stack(frames, axis=0)
            iio.imwrite(output_path, frames, fps=cfg.render_fps, macro_block_size=1)
            
    def point_step(self, data, step:int, name=""):
        from point_rasterization import render_cuda
        cfg = self.cfg
        pixels = data["pixels"]
        camtoworlds = data["camtoworlds"]
        height, width = pixels.shape[1:3]
        Ks = data["Ks"]
        query_time = data["query_time"]
        image_ids = data["image_ids"]
        assert len(image_ids) == 1, "Only support batch size 1 for test-time pose alignment."
        image_ids = image_ids[0]
        transient_gaussians = self.get_splats(
            camtoworlds=camtoworlds,
            query_time=query_time,
            sh_degree=cfg.sh_degree,
            mode="transient"
        )
        rigid_gaussians = self.get_splats(
            camtoworlds=camtoworlds,
            query_time=query_time,
            sh_degree=cfg.sh_degree,
            mode="rigid"
        )
        lifespan_data = self.dynamic_splats["lifespan"].detach()
        gmm, means, stds, weights = fit_gaussian_mixture(lifespan_data.cpu().numpy())
        x, _ = find_local_minimum_between_means(means, stds, weights)

        opacities_r = torch.sigmoid(self.dynamic_splats["opacities"])
        valid_mask = (opacities_r > 0.1)
        transient_mask = (valid_mask & (lifespan_data < x / 2)).unsqueeze(0)
        rigid_mask = valid_mask & (lifespan_data >= x / 2).unsqueeze(0)
        pts = rigid_gaussians["means"].detach()
        pts_r = pts[rigid_mask]

        if transient_gaussians is None:
            pts_t = pts[transient_mask]
        else:
            valid_mask = transient_gaussians["opacities"] > 0.1
            pts_t = torch.cat([pts[transient_mask], transient_gaussians["means"][valid_mask]], dim=0).detach()
        clrs_r = torch.zeros_like(pts_r)
        clrs_r[..., 2] = 1.0
        clrs_t = torch.zeros_like(pts_t)
        clrs_t[..., 0] = 1.0
        w2c = torch.linalg.inv(camtoworlds)
        Ks_flatten = torch.stack([Ks[:, 0, 2], Ks[:, 1, 2], Ks[:, 0, 0], Ks[:, 1, 1]], dim=-1)
        height, width = pixels.shape[1:3]
        colors, _ = render_cuda(
            torch.cat([pts_t, pts_r], dim=0).unsqueeze(0),
            torch.cat([clrs_t, clrs_r], dim=0).unsqueeze(0),
            w2c,
            Ks_flatten,
            width,
            height,
            point_size=cfg.point_size
        )
        output_points_path = os.path.join(self.render_dir, f"{step:05d}_{name}_points.png")
        output_pixels_path = os.path.join(self.render_dir, f"{step:05d}_{name}_colors.png")
        Image.fromarray((colors[0].detach().cpu().numpy() * 255.0).astype(np.uint8)).save(output_points_path)
        Image.fromarray((pixels[0].detach().cpu().numpy() * 255.0).astype(np.uint8)).save(output_pixels_path)



            
