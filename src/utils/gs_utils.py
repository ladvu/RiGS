import torch
from typing import Dict, Callable, Tuple, Union, Optional, Literal, List
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from functools import partial
from gsplat.rendering import rasterization_2dgs
from einops import einsum, repeat, rearrange
from gsplat.exporter import export_splats
from torch import Tensor
import roma
from utils.math_utils import fit_gaussian_mixture,  find_local_minimum_between_means
import matplotlib.pyplot as plt
import os
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import fsolve
import numpy as np
from utils.misc_utils import knn

def activate_gaussians(
    gaussians: Dict[str, torch.Tensor],
    scale_factor: float = 1.0,
    lifespan_min: float = 0.1,
    lifespan_range: float = 3.0,
    lifespan_activation: Literal["sigmoid", "softplus"] = "softplus",
    color_activation: Literal["sigmoid", "sh_relu"] = "sigmoid"
) -> Dict[str, torch.Tensor]:   
    activated_gaussians = {}
    activated_gaussians["scales"] = torch.exp(gaussians["scales"]) * scale_factor # [m 3]
    activated_gaussians["opacities"] = torch.sigmoid(gaussians["opacities"])  # [m]
    activated_gaussians["means"] = gaussians["means"]  # [m 3]
    activated_gaussians["quats"] = torch.nn.functional.normalize(gaussians["quats"], dim=-1)  # [m 4]
    
    if "temporal_center" in gaussians: # dynamic gaussians
        activated_gaussians["temporal_center"] = gaussians["temporal_center"]  # [m]
    if "lifespan" in gaussians:
        activated_gaussians["lifespan"] = torch.clamp(gaussians["lifespan"], lifespan_min) #[m, ]
    if "lifespan_p" in gaussians:
        if lifespan_activation == "sigmoid":
            activated_gaussians["lifespan_p"] = torch.sigmoid(gaussians["lifespan_p"]) * lifespan_range
            activated_gaussians["lifespan_m"] = torch.sigmoid(gaussians["lifespan_m"]) * lifespan_range
        else:
            activated_gaussians["lifespan_p"] = torch.nn.functional.softplus(gaussians["lifespan_p"])
            activated_gaussians["lifespan_m"] = torch.nn.functional.softplus(gaussians["lifespan_m"])
        # activated_gaussians["lifespan_p"] = torch.sigmoid(gaussians["lifespan_p"]) * lifespan_range
        # activated_gaussians["lifespan_m"] = torch.sigmoid(gaussians["lifespan_m"]) * lifespan_range
        # activated_gaussians["lifespan_p"] = torch.clamp(gaussians["lifespan_p"], lifespan_min) #[m, ]
        # activated_gaussians["lifespan_m"] = torch.clamp(gaussians["lifespan_m"], lifespan_min) #[m, ]
        activated_gaussians["velocity_p"] = gaussians["velocity_p"]  # [m 3]
        activated_gaussians["velocity_m"] = gaussians["velocity_m"]  # [m 3]

    if "sh0" in gaussians:
        activated_gaussians["sh0"] = gaussians["sh0"]  # [m 3]
        activated_gaussians["shN"] = gaussians["shN"]  # [m 3]
    else:
        if color_activation == "sigmoid":
            activated_gaussians["colors"] = torch.sigmoid(gaussians["colors"])  # [m 3]
        else:
            activated_gaussians["colors"] = torch.clamp_min(SH2RGB(gaussians["colors"]), 0.0)  # [m 3]
    
    if "motion_coefs" in gaussians:
        activated_gaussians["motion_coefs"] = F.softmax(gaussians["motion_coefs"], dim=-1)  # [m k]

    return activated_gaussians

def query_gaussian_t(
    gaussians: Dict[str, torch.Tensor],
    query_time: torch.Tensor,
    opacity_multiplier: float = 0.05,
    camera_pose: torch.Tensor = None,
    sh_degree: int = 2, # [m 3]
) -> Dict[str, torch.Tensor]:
    """Query the Gaussian parameters at specific times.
    """
    cs = query_time.shape[-1]
    t_q = rearrange(query_time, "... -> ... 1 1")
    opacity_t_q = rearrange(gaussians['opacities'], "... m -> ... 1 m 1")
    mean_t_q = rearrange(gaussians['means'], "... m c -> ... 1 m c")
    quat_t_q = rearrange(gaussians['quats'], "... m c -> ... 1 m c")
    if "temporal_center" in gaussians: # render dynamic gaussians
        temporal_center = rearrange(gaussians["temporal_center"], "... m -> ... 1 m 1")
        v_p = rearrange(gaussians["velocity_p"], "... m c -> ... 1 m c")
        v_m = rearrange(gaussians["velocity_m"], "... m c -> ... 1 m c")
        p_ind = (t_q > temporal_center).squeeze(-1)
        m_ind = ~p_ind
        lifespan_p = rearrange(gaussians["lifespan_p"], "... m -> ... 1 m 1")
        lifespan_m = rearrange(gaussians["lifespan_m"], "... m -> ... 1 m 1")
        temporal_diff = t_q - temporal_center

        mean_adjustment = torch.zeros_like(mean_t_q)
        mean_adjustment[p_ind] = temporal_diff[p_ind] * v_p[p_ind]
        mean_adjustment[m_ind] = temporal_diff[m_ind] * v_m[m_ind]
        mean_t_q = mean_t_q + mean_adjustment

        alpha_p = temporal_diff[p_ind] / lifespan_p[p_ind]
        alpha_m = temporal_diff[m_ind] / lifespan_m[m_ind]

        opacity_multiplier_tensor = torch.ones_like(opacity_t_q)
        opacity_multiplier_tensor[p_ind] = opacity_multiplier ** ((alpha_p) ** 2)
        opacity_multiplier_tensor[m_ind] = opacity_multiplier ** ((alpha_m) ** 2)
        opacity_t_q = opacity_t_q * opacity_multiplier_tensor

    else: # render static gaussians
        opacity_t_q = repeat(opacity_t_q, "... 1 m 1 -> ... t m 1", t=cs)
        mean_t_q = repeat(mean_t_q, "... 1 m c -> ... t m c", t=cs)
        v_p = torch.zeros_like(mean_t_q)
        v_m = torch.zeros_like(mean_t_q)

    quat_t_q = repeat(quat_t_q, "... 1 m c -> ... t m c", t=cs)
    scale_t_q = repeat(gaussians["scales"], "... m c -> ... t m c", t = cs)
    if scale_t_q.shape[-1] != 3:
        scale_t_q = torch.cat([scale_t_q, torch.ones_like(opacity_t_q)], dim=-1)

    if "sh0" in gaussians:
        shs = torch.cat([gaussians["sh0"], gaussians["shN"]], dim=-2).permute(0, 2, 1)
        camera_center = camera_pose[..., :3, 3]  # [..., t 3]
        view_dir = mean_t_q - camera_center.unsqueeze(-2) # [..., t m 3]
        view_dir = F.normalize(view_dir, dim=-1)
        color_t_q = eval_sh(sh_degree, shs.unsqueeze(-4), view_dir)  # [t m 3]
    else:
        color_t_q = repeat(gaussians["colors"], "... m c -> ... t m c", t = cs)
    
    
    return {
        "scales": scale_t_q,
        "colors": color_t_q,
        "means": mean_t_q,
        "opacities": opacity_t_q.squeeze(-1),
        "quats": quat_t_q,
        "vf": torch.cat([v_p, v_m], dim=-1),
    }

def compute_tracking_feature(
    means: torch.Tensor,
    motion_coefs: torch.Tensor | None,
    motion_base: nn.Module,
    ts: torch.Tensor,
    w2cs: torch.Tensor
) -> Dict[str, torch.Tensor]:
    if motion_coefs is not None:
        transfms = motion_base.compute_transforms(ts.flatten(), motion_coefs)  # (G, B*N, 3, 4)
        target_means = torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(means, (0, 1), value=1.0),
        )
    else:
        target_means = repeat(means.detach(), "p i -> p n i", n=w2cs.shape[0] * w2cs.shape[1]) # no tracking on static gaussians
    target_means = rearrange(target_means, "p (b n) i -> b n p i", b=w2cs.shape[0])
    target_means = torch.einsum(
        "bnij,bnpj->bnpi",
        w2cs[:, :, :3],
        F.pad(target_means, (0, 1), value=1.0),
    )
    target_means = rearrange(target_means, "b n p i -> b p (n i)")
    return target_means

def compute_velocity_feature(
    means: torch.Tensor,
    motion_coefs: torch.Tensor | None,
    motion_base: nn.Module,
    ts: torch.Tensor,
):
    if motion_coefs is not None:
        # ts = torch.clamp(ts, min=1, max=num_frames - 2)
        ts_neighbors = torch.cat((ts - 1, ts, ts + 1))
        ts_neighbors = torch.clamp(ts_neighbors, min=0, max=motion_base.num_frames - 1)

        transfms_nbs = motion_base.compute_transforms(ts_neighbors, motion_coefs)  # (G, 3b, 3, 4)

        means_fg_nbs = torch.einsum(
            "pnij,pj->pni",
            transfms_nbs,
            F.pad(means, (0, 1), value=1.0),
        )
        means_fg_nbs = rearrange(means_fg_nbs, "p (b k) i -> b p k i", k=3)
        fwd_velocity = means_fg_nbs[:, :, 2] - means_fg_nbs[:, :, 1]  # [n, G, 3]
        bwd_velocity = means_fg_nbs[:, :, 1] - means_fg_nbs[:, :, 0]  # [n, G, 3]
        velocity_feature = torch.cat([fwd_velocity, bwd_velocity], dim=-1)  # [n, G, 6]
        # means_fg_nbs = means_fg_nbs.reshape(
            # means_fg_nbs.shape[0], 3, -1, 3
        # )  # [G, 3, n, 3]
    else:
        velocity_feature = torch.zeros(
            ts.shape[0], means.shape[0], 6, device=means.device
        )
    return velocity_feature

def transform_gaussians(
    # gaussians: Dict[str, torch.Tensor], # should be activated
    means: torch.Tensor,
    quats: torch.Tensor,
    motion_coefs: torch.Tensor | None,
    motion_base: nn.Module,
    ts: torch.Tensor,
    inds: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    coefs = motion_coefs
    if inds is not None:
        means = means[inds]
        quats = quats[inds]
        coefs = coefs[inds]
    transfms = motion_base.compute_transforms(ts, coefs)  # (G, B, 3, 4)
    means = torch.einsum(
        "pnij,pj->pni",
        transfms,
        F.pad(means, (0, 1), value=1.0),
    )
    quats = roma.quat_xyzw_to_wxyz(
        (
            roma.quat_product(
                roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                roma.quat_wxyz_to_xyzw(quats[:, None]),
            )
        )
    )
    quats = F.normalize(quats, p=2, dim=-1)

    means = rearrange(means, "p b c -> b p c")
    quats = rearrange(quats, "p b c -> b p c")
    return means, quats

def decay_gaussians(
    opacity: torch.Tensor,
    temporal_center: torch.Tensor,
    query_time: torch.Tensor,
    lifespan: torch.Tensor,
    num_frames: float,
    sharpness: float = 3.0,
    opacity_multiplier: float=0.05,
    method:str = "sigmoid",
) -> torch.Tensor:
    # import pdb; pdb.set_trace()
    t_diff = torch.abs(query_time.unsqueeze(-1) - temporal_center.unsqueeze(0))  # [t m]
    tau = lifespan.unsqueeze(0) 
    if method == "sigmoid":
        decay = torch.sigmoid(
            sharpness * (tau - t_diff)
        )
    elif method == "exp":
        decay = opacity_multiplier**((t_diff / tau)**2)
    else:
        raise NotImplementedError(f"Decay method {method} not implemented.")

    opacity = opacity.unsqueeze(0) * decay
    return opacity
@torch.no_grad()
def make_transient(
    gaussians: Dict[str, torch.Tensor],
    motion_bases,
    lifespan_thres: float = 10.0,
    lifespan_range: float = 1.5,
    vis: bool = True,
    save_path:str = None,
    init_opacity: float = 0.2
):

    lifespan_data = gaussians["lifespan"].cpu().numpy()
    gmm, means, stds, weights = fit_gaussian_mixture(lifespan_data)
    x, y = find_local_minimum_between_means(means, stds, weights)
    if vis:
        # Clear any existing plots to avoid memory issues and overlapping plots
        plt.clf()
        plt.close('all')
        
        plt.figure(figsize=(12, 8))

        counts, bin_edges, patches = plt.hist(lifespan_data, bins=50, density=True, 
                                             alpha=0.6, color='lightblue', edgecolor='black', 
                                             label='Lifespan Distribution')

        x_range = np.linspace(lifespan_data.min(), lifespan_data.max(), 1000)

        gmm_pdf = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
        plt.plot(x_range, gmm_pdf, 'r-', linewidth=3, label='GMM Fit', alpha=0.8)

        component1_pdf = weights[0] * norm.pdf(x_range, means[0], stds[0])
        component2_pdf = weights[1] * norm.pdf(x_range, means[1], stds[1])

        plt.plot(x_range, component1_pdf, '--', linewidth=2, color='green', 
                 label=f'Component 1: μ={means[0]:.3f}, σ={stds[0]:.3f}')
        plt.plot(x_range, component2_pdf, '--', linewidth=2, color='orange', 
                 label=f'Component 2: μ={means[1]:.3f}, σ={stds[1]:.3f}')

        # plot valley (local minimum) as the threshold
        if x is not None:
            plt.plot(x, y, 'ks', markersize=8, label=f'Local minimum threshold: x={x:.3f}')
            plt.axvline(x=x, color='black', linestyle='--', alpha=0.8)
            plt.annotate(f'x={x:.3f}', xy=(x, y), xytext=(10, -30),
                         textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.xlabel('Lifespan Value')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Visualization on lifespan saved to: {save_path}")
        
        # Close the plot to free memory
        plt.close()

    mask = gaussians["lifespan"] < lifespan_thres
    if mask.sum() < 10:
        print("No transient gaussians found.")
        return None, None

    mu_t = gaussians["temporal_center"][mask]
    mu_tp1 = mu_t + 1
    mu_tm1 = mu_t - 1

    coefs = F.softmax(gaussians["motion_coefs"][mask], dim=-1)
    transform_t = motion_bases.compute_transforms_continuous(mu_t, coefs)
    transform_tp1 = motion_bases.compute_transforms_continuous(mu_tp1, coefs)
    transform_tm1 = motion_bases.compute_transforms_continuous(mu_tm1, coefs)
    mean_homo = F.pad(gaussians["means"][mask], (0, 1), value=1.0)
    mean_t = torch.einsum(
        "pij,pj->pi",
        transform_t,
        mean_homo
    )
    mean_tp1 = torch.einsum(
        "pij,pj->pi",
        transform_tp1,
        mean_homo
    )
    mean_tm1 = torch.einsum(
        "pij,pj->pi",
        transform_tm1,
        mean_homo
    )
    fwd_v = mean_tp1 - mean_t
    bwd_v = mean_t - mean_tm1
    # lifespan = torch.logit(
    #     torch.clamp(gaussians["lifespan"][mask], 0.1, lifespan_range - 0.1) / lifespan_range
    # )
    lifespan = torch.clamp(gaussians["lifespan"][mask], 0.1)
    N = mean_t.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(mean_t, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    # scale_dim = 2 if model_type == "2dgs" else 3
    scale_dim = 3
    scales = torch.log(dist_avg * 0.5).unsqueeze(-1).repeat(1, scale_dim).to(mean_t.device)  # [N, 3]
    quats = torch.rand((N, 4)).to(mean_t.device)  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity)).to(mean_t.device)  # [N,]

    transient_gaussians = {
        "means" : mean_t,
        # "scales" : gaussians["scales"][mask],
        # "opacities" : gaussians["opacities"][mask],
        # "quats" : gaussians["quats"][mask],
        "scales": scales,
        "quats": quats,
        "opacities": opacities,

        "colors" : gaussians["colors"][mask],
        "lifespan_p" : lifespan,
        "lifespan_m": lifespan,
        "temporal_center" : gaussians["temporal_center"][mask],
        "velocity_p": fwd_v,
        "velocity_m": bwd_v
    }
    

    return transient_gaussians, mask

def check_nan(tensor: torch.Tensor, name: str):
    if torch.isnan(tensor).any():
        assert False, f"NaN values found in {name}"
    else:
        print(f"No NaN values found in {name}")




@torch.no_grad()
def export_gaussians_3d(gaussians: Dict[str, torch.Tensor], times: torch.Tensor, target_dir: str, prefix: str = "gaussian_"):
    """ export the gaussians to a file.
        Args:
            gaussians (Dict[str, torch.Tensor]): Dictionary containing Gaussian parameters.
            Format should be the same as the output from build_gaussian.
            times (torch.Tensor): Tensor of shape [b t] representing the query times.
    """
    b = gaussians["mean"].shape[0]
    t = times.shape[1]
    for i in range(t):
        gaussian_t = query_gaussian_t(gaussians, times)
        for j in range(b):
            sh0 = RGB2SH(gaussian_t["color_t_q"][j, i]).unsqueeze(0),  # n 1 3
            export_splats(
                means=gaussian_t["mean_t_q"][j, i],
                scales=gaussian_t["scale_t_q"][j, i],
                quats=gaussian_t["quat_t_q"][j, i],
                opacities=gaussian_t["opacity_t_q"][j, i].squeeze(-1),
                sh0=sh0,
                shN=torch.empty([sh0.shape[0], 0, 3], device=sh0.device),
                save_to=f"{target_dir}/{prefix}{j}_{i}.ply",
            )

class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def compute_rgb_from_sh(camera_pose, sh_degree, sh0, shN, mean):
    shs = torch.cat([sh0, shN], dim=-2).permute(0, 2, 1)
    camera_center = camera_pose[..., :3, 3]  # [t 3]
    mean = rearrange(mean, "... m c -> ... 1 m c")
    view_dir = mean - camera_center.unsqueeze(-2) # [..., t m 3]
    view_dir = F.normalize(view_dir, dim=-1)
    color = eval_sh(sh_degree, shs.unsqueeze(-4), view_dir)  # [t m 3]
    return color

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5