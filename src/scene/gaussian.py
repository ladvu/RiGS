import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import *
from torch.nn import functional as F
from utils import *
from einops import rearrange, repeat
from flow3d.params import MotionBases
from utils import (
    knn,
    RGB2SH,
)


class GaussianModel(nn.Module, ABC):
    def __init__(
        self,
    ):
        self.params = None
        super().__init__()
        
    @staticmethod
    def init_from_parser(parser):
        pass

    @abstractmethod
    def get_gaussian(self, *args, **kwargs):
        pass

    def load_checkpoint(self, checkpoint: str):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location="cpu")
        matched_keys = []
        for key in self.params:
            if key in checkpoint:
                self.params[key].data = checkpoint[key].data
                matched_keys.append(key)
        if "sh0" in matched_keys:
            self.use_sh = True
        else:
            self.use_sh = False
        matched_keys = set(matched_keys) 
        missed_keys = set(self.params.keys()) - matched_keys
        extra_keys = set(checkpoint.keys()) - matched_keys
        return missed_keys, extra_keys

    def set_optimizers(
        self,
        lr_dict,
        max_steps
    ):
        optimizers = { }
        for key in lr_dict:
            if hasattr(self, 'params') and key in self.params:
                param = self.params[key]
            else:
                param = getattr(self, key)
            optimizers[key] =  torch.optim.Adam(
                [{"params": param, "lr": lr_dict[key]}],
                eps=1e-15,
                betas=(0.9, 0.999),
            )
        optimizers = [optimizers]
        
        # Use means parameter for scheduler
        means_param = self.params['means'] if hasattr(self, 'params') and 'means' in self.params else self.means
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                means_param, gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        return optimizers, schedulers
        
    @abstractmethod
    def render(
        self, render_fn: Callable, **kwargs
    ):
        pass
    
    def forward(self, *args, **kwargs):
        self.render(*args, **kwargs)


class StaticGaussianModel(GaussianModel):
    def __init__(
        self,
        use_sh = False,
    ):
        super().__init__()
        self.params = nn.ParameterDict({
            'means': nn.Parameter(torch.empty(0)),
            'scales': nn.Parameter(torch.empty(0)),
            'quats': nn.Parameter(torch.empty(0)),
            'opacities': nn.Parameter(torch.empty(0)),
            'colors': nn.Parameter(torch.empty(0)),
            # 'sh0': nn.Parameter(torch.empty(0)),
            # 'shN': nn.Parameter(torch.empty(0)),
            # "nonrigidity": torch.nn.Parameter(torch.empty(0)),
        })
        self.use_sh = use_sh
    
    @staticmethod 
    def init_from_parser(
        parser,
        init_scale: float = 1.0,
        init_opacity: float = 0.1,
        sh_degree: int = None,
    ):
        static_points = parser.static_points    
        static_points_rgb = parser.static_points_rgb
        N = static_points.shape[0]
        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(static_points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
        
        use_sh = True
        if sh_degree is not None:
            static_colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
            static_colors[:, 0, :] = RGB2SH(static_points_rgb)
        else:
            static_colors = RGB2SH(static_points_rgb)
            use_sh = False

        model = StaticGaussianModel(use_sh)
        model.params['means'] = torch.nn.Parameter(static_points)
        model.params['scales'] = torch.nn.Parameter(scales)
        model.params['quats'] = torch.nn.Parameter(quats)
        model.params['opacities'] = torch.nn.Parameter(opacities)
        
        if sh_degree is not None:
            model.params['sh0'] = torch.nn.Parameter(static_colors[:, :1, :])
            model.params['shN'] = torch.nn.Parameter(static_colors[:, 1:, :])
        else:
            model.params['colors'] = torch.nn.Parameter(static_colors)

        return model

    def get_gaussian(
        self,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Query the Gaussian parameters at specific times.

        Args:
            gaussians (Dict[str, torch.Tensor]): Dictionary containing Gaussian parameters. [b m c]
            query_time (torch.Tensor): Tensor of shape [b t] representing the query times.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the deformed parameters at the queried times:
                - 'mean_t_q': [b t m 3]
                - 'opacity_t_q': [b t m 1]
                - 'quat_t_q': [b t m 4]   
        """
        query_time: torch.Tensor = kwargs.pop("query_time", None)
        if query_time is None:
            cs = 1
        else:
            cs = query_time.shape[-1]
        mean_t_q = rearrange(self.params['means'], "... m c -> ... 1 m c")
        quat_t_q = rearrange(self.params['quats'], "... m c -> ... 1 m c")
        opacity_t_q = rearrange(torch.sigmoid(self.params['opacities']), "... m -> ... 1 m 1")

        opacity_t_q = repeat(opacity_t_q, "... 1 m 1 -> ... t m 1", t=cs)
        mean_t_q = repeat(mean_t_q, "... 1 m c -> ... t m c", t=cs)
        quat_t_q = repeat(quat_t_q, "... 1 m c -> ... t m c", t=cs)
        feature = kwargs.pop("feature", None)
        if feature is None:
            if self.use_sh:
                camera_pose = kwargs.pop("camera_pose")
                sh_degree = kwargs.pop("sh_degree")
                color_t_q = compute_rgb_from_sh(camera_pose, sh_degree, self.params['sh0'], self.params['shN'], self.params['means'])
            else:
                color_t_q = repeat(torch.sigmoid(self.params['colors']), "... m c -> ... t m c", t = cs)
        else:
            if feature in self.params.keys():
                f:torch.Tensor = self.params[feature]
                if f.ndim == 1:
                    f = f.unsqueeze(-1)
                color_t_q = repeat(f, "... m c -> ... t m c", t = cs)
            else:
                color_t_q = torch.zeros_like(opacity_t_q)

        scale_t_q = repeat(torch.exp(self.params['scales']), "... m c -> ... t m c", t = cs)
        return {
            "scales": scale_t_q,
            "colors": color_t_q,
            "means": mean_t_q,
            "opacities": opacity_t_q.squeeze(-1),
            "quats": quat_t_q,
            # "velocity_p": torch.zeros_like(mean_t_q),
            # "velocity_m": torch.zeros_like(mean_t_q),
        }
       

        
    def render(
        self, render_fn, **kwargs
    ):
        gaussian_params = self.get_gaussian(**kwargs)
        inputs = {
            **gaussian_params,
            **kwargs
        }
        return render_fn(**inputs)


class DynamicGaussianModel(GaussianModel):
    def __init__(
        self,
        sharpness: float = 3.0,
        use_sh: bool = False,

    ):
        super().__init__()
        self.params = nn.ParameterDict({
            'means': nn.Parameter(torch.empty(0)),
            'scales': nn.Parameter(torch.empty(0)),
            'quats': nn.Parameter(torch.empty(0)),
            'opacities': nn.Parameter(torch.empty(0)),
            'lifespan': nn.Parameter(torch.empty(0)),
            'temporal_center': nn.Parameter(torch.empty(0)),
            'motion_coefs': nn.Parameter(torch.empty(0)),
            'colors': nn.Parameter(torch.empty(0)),
            # 'sh0': nn.Parameter(torch.empty(0)),
            # 'shN': nn.Parameter(torch.empty(0)),
        })
        self.use_sh = use_sh
        # self.v_deg = v_deg
        self.sharpness = sharpness

    @staticmethod
    def init_from_parser(
        parser,
        init_scale: float = 1.0,
        init_opacity: float = 0.1,
        sh_degree: int = None,
        velocity_degree: int = 3
    ):
        raise NotImplementedError("DynamicGaussianModel init_from_parser is not implemented yet.")
    
    def get_gaussian(
        self,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        query_time: torch.Tensor = kwargs.pop("query_time")
        motion_bases = kwargs.pop("motion_bases")
        cs = query_time.shape[-1]
        dynamic_gaussians = activate_gaussians(self.params, 1.0, 0.1, color_activation="sigmoid")
        motion_coefs = dynamic_gaussians["motion_coefs"]
        (
            dynamic_gaussians["means"],
            dynamic_gaussians["quats"]
        ) = transform_gaussians(
            dynamic_gaussians["means"],
            dynamic_gaussians["quats"],
            motion_coefs,
            motion_bases,
            query_time
        )
        dynamic_gaussians["opacities"] = decay_gaussians(
            dynamic_gaussians["opacities"],
            dynamic_gaussians["temporal_center"],
            query_time,
            dynamic_gaussians["lifespan"],
            None,
            sharpness=self.sharpness
        )
        for key in ["scales", "colors"]:
            dynamic_gaussians[key] = repeat(dynamic_gaussians[key], "p ... -> b p ...", b=query_time.shape[0])
        feature = kwargs.pop("feature", None)
        if feature is None:
            color_t_q = dynamic_gaussians["colors"]
            # color_t_q = repeat(self.params['colors'], "... m c -> ... t m c", t = cs)
        else:
            if feature in self.params.keys():
                f:torch.Tensor = self.params[feature]
                if f.ndim == 1:
                    f = f.unsqueeze(-1)
                color_t_q = repeat(f, "... m c -> ... t m c", t = cs)
            else:
                color_t_q = torch.zeros_like(dynamic_gaussians["opacities"])

        return {
            "scales": dynamic_gaussians["scales"],
            "colors": color_t_q,
            "means": dynamic_gaussians["means"],
            "opacities": dynamic_gaussians["opacities"],
            "quats": dynamic_gaussians["quats"],
        }

    def render(
        self, render_fn, **kwargs
    ):
        gaussian_params = self.get_gaussian(**kwargs)
        inputs = {
            **gaussian_params,
            **kwargs
        }
        return render_fn(**inputs)

class TransientGaussianModel(GaussianModel):
    def __init__(
        self,
        opacity_multiplier: float = 0.05,
        sharpness: float = 3.0,
        # v_deg: int = 3,
        use_sh: bool = False,

    ):
        super().__init__()
        self.params = nn.ParameterDict({
            "means": nn.Parameter(torch.empty(0)),
            "scales": nn.Parameter(torch.empty(0)),
            "quats": nn.Parameter(torch.empty(0)),
            "opacities": nn.Parameter(torch.empty(0)),
            "temporal_center": nn.Parameter(torch.empty(0)),
            "lifespan_p": nn.Parameter(torch.empty(0)),
            "lifespan_m": nn.Parameter(torch.empty(0)),
            "velocity_p": nn.Parameter(torch.empty(0)),
            "velocity_m": nn.Parameter(torch.empty(0)),
            "colors": nn.Parameter(torch.empty(0)),
        })

        self.use_sh = use_sh
        self.opacity_multiplier = opacity_multiplier

    @staticmethod
    def init_from_parser(
        parser,
        init_scale: float = 1.0,
        init_opacity: float = 0.1,
        sh_degree: int = None,
        velocity_degree: int = 3
    ):
        raise NotImplementedError("DynamicGaussianModel init_from_parser is not implemented yet.")
    
    def get_gaussian(
        self,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Query the Gaussian parameters at specific times.

        Args:
            gaussians (Dict[str, torch.Tensor]): Dictionary containing Gaussian parameters. [b m c]
            query_time (torch.Tensor): Tensor of shape [b t] representing the query times.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the deformed parameters at the queried times:
                - 'mean_t_q': [b t m 3]
                - 'opacity_t_q': [b t m 1]
                - 'quat_t_q': [b t m 4]   
        """
        query_time: torch.Tensor = kwargs.pop("query_time")
        cs = query_time.shape[-1]
        transient_gaussians = activate_gaussians(self.params, 1.0, 0.1, color_activation="sigmoid")
        transient_gaussians = query_gaussian_t(
            transient_gaussians,
            query_time=query_time,
            opacity_multiplier=self.opacity_multiplier,
        )
        feature = kwargs.pop("feature", None)
        if feature is None:
            color_t_q = transient_gaussians["colors"]
        else:
            if feature in self.params.keys():
                f:torch.Tensor = self.params[feature]
                if f.ndim == 1:
                    f = f.unsqueeze(-1)
                color_t_q = repeat(f, "... m c -> ... t m c", t = cs)
            else:
                color_t_q = torch.zeros_like(transient_gaussians["opacities"])

        return {
            "scales": transient_gaussians["scales"],
            "colors": color_t_q,
            "means": transient_gaussians["means"],
            "opacities": transient_gaussians["opacities"],
            "quats": transient_gaussians["quats"],
        }

    def render(
        self, render_fn, **kwargs
    ):
        gaussian_params = self.get_gaussian(**kwargs)
        inputs = {
            **gaussian_params,
            **kwargs
        }
        return render_fn(**inputs)


class SceneGaussianModel(nn.Module):

    def __init__(self, static_use_sh: bool = False, dynamic_use_sh: bool = False, transient_use_sh: bool = False):
        super().__init__()
        self.static_splats = StaticGaussianModel(use_sh=static_use_sh)
        self.dynamic_splats = DynamicGaussianModel(use_sh=dynamic_use_sh)
        self.transient_splats = TransientGaussianModel(use_sh=transient_use_sh)
        self.motion_bases = None
        self.has_static = False
        self.has_dynamic = False
        self.has_transient = False
    
    @staticmethod
    def init_from_parser(
        parser,
        init_scale: float = 1.0,
        init_opacity: float = 0.1,
        sh_degree: int = None,
    ):
        raise NotImplementedError("SceneGaussianModel init_from_parser is not implemented yet.")

    def set_optimizers(
        self,
        lr_dict_static,
        lr_dict_dynamic,
        max_steps
    ):
        raise NotImplementedError("SceneGaussianModel set_optimizers is not implemented yet.")
    
    def load_checkpoint(self, checkpoint: str):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location="cpu")
        if "static_splats" in checkpoint:
            missed_keys_static, extra_keys_static = self.static_splats.load_checkpoint(checkpoint["static_splats"])
            self.has_static = True
        else:
            missed_keys_static, extra_keys_static = set(self.static_splats.params.keys()), set()
        if "dynamic_splats" in checkpoint:
            missed_keys_dynamic, extra_keys_dynamic = self.dynamic_splats.load_checkpoint(checkpoint["dynamic_splats"])
            self.has_dynamic = True
        else:
            missed_keys_dynamic, extra_keys_dynamic = set(self.dynamic_splats.params.keys()), set()
        if "transient_splats" in checkpoint:
            missed_keys_transient, extra_keys_transient = self.transient_splats.load_checkpoint(checkpoint["transient_splats"])   
            self.has_transient = True
        else:
            missed_keys_transient, extra_keys_transient = set(self.transient_splats.params.keys()), set()
            
        if "motion_bases" in checkpoint:
            self.motion_bases = MotionBases.init_from_state_dict(checkpoint["motion_bases"])
        return {
            "static": (missed_keys_static, extra_keys_static),
            "dynamic": (missed_keys_dynamic, extra_keys_dynamic),
            "transient": (missed_keys_transient, extra_keys_transient),
        }

    def get_gaussian(self, *args, **kwargs):
        mode = kwargs.pop("mode", "all")
        gaussian_list = []
        if mode == "static" or mode == 'all':
            static_gaussians = self.static_splats.get_gaussian(**kwargs)
            if mode == "static":
                return static_gaussians
            else:
                gaussian_list.append(static_gaussians)
        if mode in ["rigid", "dynamic", "all"]:
            dynamic_gaussians = self.dynamic_splats.get_gaussian(motion_bases=self.motion_bases, **kwargs)
            if mode == "rigid":
                return dynamic_gaussians
            else:
                gaussian_list.append(dynamic_gaussians)
        if mode in ["transient", "dynamic", "all"] and self.has_transient:
            transient_gaussians = self.transient_splats.get_gaussian(**kwargs)
            if mode == "transient":
                return transient_gaussians
            else:
                gaussian_list.append(transient_gaussians)
        elif not self.has_transient and mode == "transient":
            return self.dynamic_splats.get_gaussian(motion_bases=self.motion_bases, **kwargs)
        f_dim = max([x["colors"].shape[-1] for x in gaussian_list])
        for i in range(len(gaussian_list)):
            if gaussian_list[i]["colors"].shape[-1] != f_dim:
                f_s = gaussian_list[i]["colors"].shape[-1]
                assert f_s == 1, f"Only support one of the feature to be 1, {i}th feature has dim {f_s}"
                gaussian_list[i]["colors"] = gaussian_list[i]["colors"].repeat(1,1, f_dim)
        gaussians = {
            k : torch.cat([x[k] for x in gaussian_list], dim=1)
            for k in gaussian_list[0]
        }
        return gaussians

    
    def render(self, render_fn, mode="all", **kwargs):
        gaussians = self.get_gaussian(mode=mode, **kwargs)
        outputs = render_fn(
            **gaussians,
            **kwargs
        )
        # TODO, split info ?
        return outputs