from typing import *
from dataclasses import dataclass, field    

@dataclass
class Config:
    # Disable viewer
    # disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None
    # render video
    render_video: bool = False
    # render videos at these times (normalized to [0,1])
    reference_time: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    # render_trajectory
    render_trajectories: List[str] = field(default_factory=lambda: ["arc", "lemniscate", "spiral", "wander", "fixed"])
    # render_types
    render_types: List[str] = field(default_factory=lambda: ["fused", "static", "dynamic", "transient", "rigid"])
    # render_fps
    render_fps:int = 30
    # render point
    render_point: bool = False
    # point size
    point_size: int = 1

    # eval step
    eval_step: Optional[int] = 30_000
    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "../data/nvidia_processed"
    # data_dir: str = "../data/davis/train"
    # data name
    data_name: str = "Balloon1"
    # Downsample factor for the dataset
    data_factor: int = 2
    # Directory to save results
    mask_type:str = "default"
    depth_type: Literal["default", "zoedepth", "depthpro", "moge", "unidepth"] = "default"
    cache_dir: str = "../cache"
    cache_force_reload: bool = False
    #
    seed: int = 114514
    #val
    val_dir: Optional[str] = None

    # val_dir: Optional[str] = None

    result_dir: str = "../results"
    # exp name
    exp_name: str = "train"
    # Every N images there is a test image
    test_every: int = 1

    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True

    depth_thres: float = 0.5
    depth_min: float = 2e-3
    depth_max: float = 1000.0

    max_points: int = 1_000_000
    # sample rate
    sample_rate: int = 1
    # voxel size for point cloud preprocessing
    voxel_size: float = 0.01
    # op
    opacity_multiplier: float = 0.05

    # Port for the viewer server
    vis: bool = True
    port: int = 8881

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [15_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [15_000, 30_000])

    # Degree of spherical harmonics
    sh_degree: int = 0
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 600
    # color activation
    color_activation: Literal["sigmoid", "sh_relu"] = "sigmoid"
    lifespan_activation: Literal["sigmoid", "softplus"] = "sigmoid"
    # Initial opacity of GS
    init_opa: float = 0.7
    # Initial scale of GS
    init_scale: float = 1.0
    # number of foreground gaussians
    num_fg: int = 40_000
    num_motion_bases: int = 10
    num_targets_per_frame: int = 4

    # init lifespan
    # init_lifespan: float = 1.0
    lifespan_thres: float = 3.0
    lifespan_range: float = 1.5
    rescale_gaussian: bool = False
    no_decay_gaussians: bool = False
    decay_method: str = "sigmoid" # "sigmoid" or "exp"
    no_transient_gaussian: bool = False
    transient_start_step: int = 10_000
    transient_every: int = 3000
    transient_end_step: int = 20_000
    vis_lifespan: bool = True
    
    # sharpness
    decay_sharpness: float = 3.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.1

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 200
    # Background color for rendering
    background_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.05
    # prune lifespan
    prune_lifespan: float = 0.1
    # prine velocity, prune the gaussians with velocity norm > prune_velocity
    prune_velocity: float = 0.5
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 1_000
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 10_000
    # Reset opacities every this steps
    reset_every: int = 3_000
    # Refine GSs every this steps
    refine_every: int = 100
    # do prune last iter
    refine_mask_start_iter: int = 100_000
    refine_mask_stop_iter: int = 110_000
    refine_mask_every: int = 3000

    mask_kernel_size: int = 5

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_opt_start_iter: int = 8000
    # Test-Time Pose Alignment
    test_time_pose_opt: bool = False # Test Time Pose alignment may cost a lot of time if many frames, but needed for iphone dataset
    test_time_pose_steps_each: int = 250
    test_time_decay_start:int = 200
    test_time_pose_opt_lr: float = 6e-3
    test_time_pose_opt_lr_final:float = 3e-4
    test_time_pose_refine_lr: float = 3e-4
    test_time_pose_refine_lr_final: float = 6e-5
    test_time_pose_refine_steps_each: int = 30
    test_time_pose_refine_decay_start: int = 20
    test_time_psnr: str = "psnr"
    # test_time_pose_opt_reg: float = 1e-6

    # loss lambdas
    depth_lambda: float = 0.05
    track_depth_lambda: float = 0.05
    smooth_base_lambda: float = 0.1
    translation_lambda: float = 2.0
    smooth_track_lambda: float = 2.0
    normal_lambda: float = 5e-2 # 0.0
    alpha_loss_fn: Literal["l1", "bce"] = "bce"
    alpha_lambda: float = 0.1
    track_lambda: float = 2.0
    velocity_lambda: float = 0.01
    lifespan_lambda: float = 0.5
    scale_lambda: float = 0.5

    depth_noise_scale: float = 0.0
    # locality_lambda: float = 0.01
    # velocity_reg_lambda: float = 0.0 #0.1
    # velocity_reg_thres: float = 0.0 #0.1
    # iso_lambda: float = 1e-2
    # conf_lambda: float = 1e-2
    # scale_reg_lambda: float = 1.0
    # scale_reg_lambda: float = 0.0
    normal_start_iter: int = 7_000

    # two-stage training 
    init_steps: int = 5_000

    # Model for splatting.
    model_type: Literal["2dgs", "3dgs"] = "2dgs"
    # strategy
    strategy: Literal["default", "mcmc"] = "default"

    # Dump information to tensorboard every this steps
    tb_every: int = 1000
    # Save training images to tensorboard
    tb_save_image: bool = True


    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)
        self.init_steps = int(self.init_steps * factor)
        self.normal_start_iter = int(self.normal_start_iter * factor)
        self.pose_opt_start_iter = int(self.pose_opt_start_iter * factor)
        self.transient_end_step = int(self.transient_end_step * factor)
        self.transient_start_step = int(self.transient_start_step * factor)
        self.transient_every = int(self.transient_every * factor)

        self.refine_mask_start_iter = int(self.refine_mask_start_iter * factor)
        self.refine_mask_stop_iter = int(self.refine_mask_stop_iter * factor)
        self.refine_mask_every = int(self.refine_mask_every * factor)

