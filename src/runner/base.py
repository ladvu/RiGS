import json
import math
import os
import time
from typing import Dict, Optional, Tuple
import torch
import numpy as np
import torch.nn.functional as F
import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    CameraOptModule,
    set_random_seed,
    batch_normalize,
    export_pointcloud,
    export_video,
    query_gaussian_t,
    activate_gaussians,
    closed_form_inverse_se3,
    mPSNR,
    mSSIM,
    mLPIPS
)
# from gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState
# from nerfview import CameraState, RenderTabState, apply_float_colormap
# import nerfview
from einops import repeat
from scene.dataset import Dataset, Parser
from gsplat.rendering import rasterization_2dgs, rasterization
from config import Config
from scene.utils import *
from abc import abstractmethod
from gsplat.strategy.ops import remove
import lpips
import gc
from flow3d.trajectories import (
    get_arc_w2cs,
    get_avg_w2c,
    get_lemniscate_w2cs,
    get_lookat,
    get_spiral_w2cs,
    get_wander_w2cs,
    get_fixed_w2cs
)

def get_trajectory_w2cs(name, **kwargs):
    return {
        "arc": get_arc_w2cs,
        "lemniscate": get_lemniscate_w2cs,
        "spiral": get_spiral_w2cs,
        "wander": get_wander_w2cs,
        "fixed": get_fixed_w2cs,
    }[name](**kwargs)

class BaseRunner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"
        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)
        exp_dir = os.path.join(cfg.result_dir, cfg.exp_name)

        # Setup output directories.
        self.ckpt_dir = f"{exp_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{exp_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{exp_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{exp_dir}/plys"
        os.makedirs(self.ply_dir, exist_ok=True)
        self.visual_dir = f"{exp_dir}/visuals"
        os.makedirs(self.visual_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{exp_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            name=cfg.data_name,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            use_gpu=True,
            cache_dir=cfg.cache_dir,
            val_dir=cfg.val_dir,
            depth_thres=cfg.depth_thres,
            depth_min=cfg.depth_min,
            depth_max=cfg.depth_max,
            cache_force_reload=cfg.cache_force_reload,
            mask_type=cfg.mask_type,
            depth_type=cfg.depth_type,
            depth_noise_scale=cfg.depth_noise_scale
        )
        if not self.parser.loaded:
            self.parser.init_points(cfg.sample_rate, cfg.voxel_size, use_gpu=True, mask_kernel_size=cfg.mask_kernel_size, maximum=cfg.max_points)
            if cfg.normalize_world_space:
                self.parser.normalize()
            if cfg.cache_dir is not None: 
                print(f"Saving processed data to cache: {self.parser.cache_data_path}")
                self.parser._save_to_cache(self.parser.cache_data_path)

        self.trainset = Dataset(self.parser, split="train", num_targets_per_frame=cfg.num_targets_per_frame)
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale 
        print("Scene scale:", self.scene_scale)

        export_pointcloud(
            self.parser.dynamic_points.detach(),
            self.parser.dynamic_points_rgb.detach(),
            os.path.join(self.ply_dir, "init_dynamic.ply")
        )
        export_pointcloud(
            self.parser.static_points.detach(),
            self.parser.static_points_rgb.detach(), 
            os.path.join(self.ply_dir, "init_static.ply"),
        )
        export_video(
            os.path.join(self.visual_dir, "dynamic_masks.mp4"),
            (repeat(self.parser.dynamic_masks, "s h w -> s h w 3") * self.parser.images * 255.0).to(torch.uint8),
            fps=8
        )
        export_video(
            os.path.join(self.visual_dir, "static_masks.mp4"),
            (repeat(self.parser.static_masks, "s h w -> s h w 3") * self.parser.images * 255.0).to(torch.uint8),
            fps=8
        )
        export_video(
            os.path.join(self.visual_dir, "depths_valid.mp4"),
            (repeat(self.parser.depths_valid, "s h w -> s h w 3") * self.parser.images * 255.0).to(torch.uint8),
            fps=8
        )
        export_video(
            os.path.join(self.visual_dir, "fwd_velocity_masks.mp4"),
            (repeat(self.parser.fwd_velocity_masks, "s h w -> s h w 3") * self.parser.images * 255.0).to(torch.uint8),
            fps=8
        )
        export_video(
            os.path.join(self.visual_dir, "bwd_velocity_masks.mp4"),
            (repeat(self.parser.bwd_velocity_masks, "s h w -> s h w 3") * self.parser.images * 255.0).to(torch.uint8),
            fps=8
        )

        depth_vis = repeat(
            batch_normalize(self.parser.depths) ,
            "s h w -> s h w 3"
        )
        export_video(
            os.path.join(self.visual_dir, "depths.mp4"),
            (depth_vis * 255.0).to(torch.uint8),
            fps=8
        )
        velocity_norm = torch.norm(self.parser.fwd_velocity_map, dim=-1, keepdim=True) + 1e-6
        velocity_vis = self.parser.fwd_velocity_map / velocity_norm
        velocity_vis = velocity_vis / 2 + 0.5
        export_video(
            os.path.join(self.visual_dir, "fwd_velocity.mp4"),
            (velocity_vis * 255.0).to(torch.uint8),
            fps=8
        )
        velocity_norm = torch.norm(self.parser.bwd_velocity_map, dim=-1, keepdim=True) + 1e-6
        velocity_vis = self.parser.bwd_velocity_map / velocity_norm
        velocity_vis = velocity_vis / 2 + 0.5
        export_video(
            os.path.join(self.visual_dir, "bwd_velocity.mp4"),
            (velocity_vis * 255.0).to(torch.uint8),
            fps=8
        )
        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = lpips.LPIPS(net='alex').to(self.device)

        self.mpsnr = mPSNR(data_range=1.0)
        self.mssim = mSSIM(data_range=1.0)
        self.mlpips = mLPIPS(self.device)
        self.key_for_gradient = "gradient_2dgs" if cfg.model_type == "2dgs" else "means2d"
        self.setup_splats(cfg)

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            self.pose_schedulers = [
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / (cfg.max_steps - cfg.pose_opt_start_iter))
                )
            ]
        else:
            self.pose_adjust = lambda x, y : x 
            self.pose_optimizers = []
            self.pose_schedulers = []
        
        if cfg.test_time_pose_opt or cfg.pose_opt:
            self.tt_pose_align = torch.nn.ModuleList([CameraOptModule(1) for _ in range(len(self.valset))]).to(self.device)
            self.tt_pose_align_init = [False for _ in range(self.parser.num_val_cams)]
            self.tt_init = False
            for pose_align in self.tt_pose_align:
                pose_align.zero_init()

    def train(self):
        cfg = self.cfg
        # Dump cfg.
        with open(os.path.join(cfg.result_dir, cfg.exp_name, "config.json"), "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=self.trainset.collate_fn
        )

        trainloader_iter = iter(trainloader)
        refineloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=self.trainset.collate_fn
        )

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar: 

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            data = self.copy_data_to_device(data)
            self.train_step(data, step, pbar)

            # if step % cfg.refine_mask_every == 0 and step > cfg.refine_mask_start_iter and step < cfg.refine_mask_stop_iter:
            #     print("Refining mask")
            #     refine_iter = iter(refineloader)
            #     with torch.no_grad():
            #         self.refine_mask(refine_iter, step)
            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                }
                self.save_checkpoint(step, stats)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                if self.cfg.test_time_pose_opt:
                    self.pose_align(step)
                self.eval(step)
                gc.collect()
    
    def pose_align(self, step:int):
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1, collate_fn=self.valset.collate_fn
        )
        self.test_time_pose_align(valloader, step)

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1, collate_fn=self.valset.collate_fn
        )
        
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": [], "mpsnr": [], "mssim": [], "mlpips": [],
                   "inv_mpsnr": [], "inv_mssim": [], "inv_mlpips": []}
        
        for i, data in enumerate(valloader):
            new_data = self.copy_data_to_device(data)
            metric = self.val_step(new_data, step, None)
            ellipse_time += metric["ellipse_time"]
            metrics["psnr"].append(metric["psnr"].item())
            metrics["ssim"].append(metric["ssim"].item())
            metrics["lpips"].append(metric["lpips"].item())
            metrics["mpsnr"].append(metric["mpsnr"].item())
            metrics["mssim"].append(metric["mssim"].item())
            metrics["mlpips"].append(metric["mlpips"].item())
            metrics["inv_mpsnr"].append(metric["inv_mpsnr"].item())
            metrics["inv_mssim"].append(metric["inv_mssim"].item())
            metrics["inv_mlpips"].append(metric["inv_mlpips"].item())

        ellipse_time /= len(valloader)
        for key in metrics:
            metrics[key] = np.array(metrics[key]).mean()
        message = f"Eval at step {step}: "
        for key in metrics:
            message += f"{key}: {metrics[key]:.4f} "
        print(
            message +
            f"Time: {ellipse_time:.3f}s/image "
        )
        # save stats as json
        stats = {
            **metrics,
            "ellipse_time": ellipse_time,
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()
    
    @torch.no_grad()
    def render_point(self):
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=1, shuffle=False, num_workers=1, collate_fn=self.trainset.collate_fn
        )
        for i, data in enumerate(trainloader):
            new_data = self.copy_data_to_device(data)
            self.point_step(new_data, i, "point")

    @torch.no_grad() 
    def render_video(self):
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1, collate_fn=self.valset.collate_fn
        )
        train_w2cs = self.parser.get_w2cs().to(self.device)
        avg_w2c = get_avg_w2c(train_w2cs)
        # avg_w2c = train_w2cs[0]
        train_c2ws = torch.linalg.inv(train_w2cs)
        # lookat = get_lookat(train_c2ws[:, :3, -1], train_c2ws[:, :3, 2])
        lookat = torch.mean(self.dynamic_splats["means"].detach(), dim=0)
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        K = self.parser.get_Ks()[0].to(self.device)
        img_wh = self.parser.get_img_wh()

        # get the radius of the bounding sphere of training cameras
        rc_train_c2ws = torch.einsum("ij,njk->nik", avg_w2c, train_c2ws)
        rc_pos = rc_train_c2ws[:, :3, -1]
        rads = (rc_pos.amax(0) - rc_pos.amin(0)) * 0.5
        render_steps = [int(x * self.parser.num_frames) for x in self.cfg.reference_time]
        
        for i in range(self.parser.num_frames):
            if i not in render_steps:
                continue
            print(f"Rendering video for frame {i}")
            for name in self.cfg.render_trajectories:
                w2cs = get_trajectory_w2cs(
                    name=name,
                    num_frames=self.parser.num_frames,
                    ref_w2c=train_w2cs[i],
                    lookat=lookat,
                    up=up,
                    focal_length=K[0, 0].item(),
                    rads=rads,
                    zrate=0.2,
                    rots=1, 
                    degree=20.0
                )
                data = dict(w2cs=w2cs, Ks=repeat(K, "h w -> t h w", t=w2cs.shape[0]), width=img_wh[0], height=img_wh[1])
                self.render_step(data, i, name)
    
    def get_splats(self, query_time, sh_degree, camtoworlds, *args, **kwargs):
        gaussians = activate_gaussians(self.splats)
        gaussians = query_gaussian_t(
            gaussians,
            query_time=query_time,
            sh_degree=sh_degree,
            camera_pose=camtoworlds,
            opacity_multiplier=self.cfg.opacity_multiplier,
        )

        return gaussians

    def rasterize_splats(
        self,
        gaussians:Dict[str, Tensor],
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        backgrounds: List[float] | None = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        if backgrounds is not None:
            color_dim = gaussians["colors"].shape[-1]
            pad_size = color_dim - len(backgrounds)
            backgrounds = torch.tensor(backgrounds + [0.0] * pad_size, device=self.device, dtype=torch.float32)
            backgrounds = repeat(backgrounds, "c -> t k c", t=camtoworlds.shape[0], k=1)

        if self.cfg.model_type == "2dgs":
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                info,
            ) = rasterization_2dgs(
                means=gaussians["means"],
                quats=gaussians["quats"],
                scales=gaussians["scales"],
                opacities=gaussians["opacities"],
                colors=gaussians["colors"].unsqueeze(1), # T M 6
                viewmats=closed_form_inverse_se3(camtoworlds).unsqueeze(1),  # [T, 1, 4, 4]
                Ks=Ks.unsqueeze(1),  # [T, 1, 3, 3]
                width=width,
                height=height,
                backgrounds=backgrounds,
                **kwargs,
            )

            return (
                render_colors[:, 0],
                render_alphas[:, 0],
                render_normals[:, 0] if isinstance(render_normals, torch.Tensor) else render_normals,
                normals_from_depth[:, 0] if isinstance(normals_from_depth, torch.Tensor) and normals_from_depth.ndim == 5 else normals_from_depth,
                render_distort[:, 0] if isinstance(render_distort, torch.Tensor) else render_distort,
                render_median[:, 0] if isinstance(render_median, torch.Tensor) else render_median,
                info
            )
        else:
            (
                render_colors,
                render_alphas,
                info,
            ) = rasterization(
                means=gaussians["means"],
                quats=gaussians["quats"],
                scales=gaussians["scales"],
                opacities=gaussians["opacities"],
                colors=gaussians["colors"].unsqueeze(1), # T M 6
                viewmats=closed_form_inverse_se3(camtoworlds).unsqueeze(1),  # [T, 1, 4, 4]
                Ks=Ks.unsqueeze(1),  # [T, 1, 3, 3]
                width=width,
                height=height,
                packed=False,
                backgrounds=backgrounds,
                **kwargs,
            )

            return (
                render_colors[:, 0],
                render_alphas[:, 0],
                None,
                None,
                None,
                None,
                info
            )
    
    def copy_data_to_device(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        device = self.device
        new_data = {}
        for key in data:
            if isinstance(data[key], torch.Tensor):
                new_data[key] = data[key].to(device)
            elif isinstance(data[key], list):
                new_data[key] = [d.to(device) for d in data[key]]
        new_data["image_ids"] = new_data["image_ids"].cpu()
        if "cam_ids" in new_data:
            new_data["cam_ids"] = new_data["cam_ids"].cpu()
        return new_data


    @abstractmethod
    def test_time_pose_align(self, dataloader, step:int):
        pass

    @abstractmethod
    def refine_mask(self, dataloader, step:int):
        pass

    @abstractmethod    
    def compute_loss(
        self,
        data: Dict[str, Tensor],
        renders: Dict[str, Tensor],
        gaussians: Dict[str, Tensor],
        step:int,
    ):
        pass

    @abstractmethod
    def tb_log(
        self,
        step: int,
        data: Dict[str, Tensor],
        renders: Dict[str, Tensor],
        loss_dict: Dict[str, float],
    ):
        pass

    @abstractmethod
    def pbar_log(
        self,
        pbar,
        loss_dict: Dict[str, float],
        step: int,
    ):
        pass

    @abstractmethod
    def train_step(
        self, 
        data: Dict[str, Tensor],
        step:int,
        pbar
    ):
        pass

    @abstractmethod
    def val_step(
        self,
        data: Dict[str, Tensor],
        step:int,
        pbar
    ):
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        step:int,
        meta:Optional[Dict]=None
    ):
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        checkpoint
    ):
        pass

    @abstractmethod 
    def setup_splats(self, cfg:Config):
        pass

    @abstractmethod
    def render_step(self, data, step, name=""):
        pass

    @abstractmethod
    def point_step(self, data, step, name=""):
        pass