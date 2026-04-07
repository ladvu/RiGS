import viser
from pathlib import Path
from typing import Tuple, Callable, Literal
from nerfview import Viewer, RenderTabState
import argparse
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
import viser
from pathlib import Path
from gsplat.distributed import cli
from einops import rearrange
from nerfview import CameraState 
from utils.misc_utils import apply_float_colormap, batch_normalize
from viewer.viewer import GsplatRenderTabState, GsplatViewer
from scene.gaussian import SceneGaussianModel
from scene.renderer import GaussianRenderer

def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)
    gaussians = SceneGaussianModel(False, False)
    renderer = GaussianRenderer(~args.no_2dgs)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    err = gaussians.load_checkpoint(checkpoint)
    print(err)
    gaussians.to(device)
    # max_time = gaussians.dynamic_splats.params["temporal_center"].max() + 1.0 if gaussians.has_dynamic else torch.tensor(0.0)
    max_time = checkpoint["motion_bases"]["params.rots"].shape[1] - 1.0
    print("Number of Dynamic Gaussians:", gaussians.dynamic_splats.params["means"].shape[0])
    print("Number of Static Gaussians:", gaussians.static_splats.params["means"].shape[0])
    print("Number of Transient Gaussians", gaussians.transient_splats.params["means"].shape[0] if gaussians.has_transient else 0)
    
    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: GsplatRenderTabState):
        nonlocal renderer
        assert isinstance(render_tab_state, GsplatRenderTabState)
        width = render_tab_state.render_width
        height = render_tab_state.render_height
        poses = camera_state.c2w
        poses = torch.from_numpy(poses).float().to(device)
        poses = poses[None,:]
        extrinsics = torch.linalg.inv(poses)
        intrinsics = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)[None,:]
        # viewmat = torch.eye(4, device=device)
        # intrinsics = normalized_intrinsics.clone()
        # intrinsics[0, 0] = intrinsics[0, 0] * width
        # intrinsics[0, 1] = intrinsics[0, 1] * height

        current_time = render_tab_state.preview_time if render_tab_state.preview_render else render_tab_state.timestamp
        query_time = torch.tensor([current_time], device=device)

        render_mode, feature = {
            "rgb": ("RGB", None),
            "depth(accumulated)": ("D", None),
            "depth(expected)": ("ED", None),
            "depth(median)": ("D", None),
            "alpha": ("D", None),
            "normal(surface)": ("RGB+D", None),
            "normal": ("RGB+D", None),
            "velocity": ("RGB", "velocity"),
            "velocity(norm)": ("RGB", "velocity"),
            "lifespan": ("RGB", "lifespan"),
            # "conf": ("RGB", "conf"),
            # "nonrigidity": ("RGB", "nonrigidity"),
        }[render_tab_state.render_mode]
        backgrounds = {
            "rgb": [1.0, 1.0, 1.0],
            "depth(accumulated)": [1.0, 1.0, 1.0],
            "depth(expected)": [1.0, 1.0, 1.0],
            "depth(median)": [1.0, 1.0, 1.0],
            "alpha": [0.0, 0.0, 0.0],
            "normal(surface)": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "velocity(norm)": [0.0, 0.0, 0.0],
            "lifespan": [0.0, 0.0, 0.0],
        }[render_tab_state.render_mode]

        rendered = gaussians.render(
            render_fn=renderer,
            mode=render_tab_state.mode,
            query_time=query_time,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            camera_pose=poses,
            width=width,
            height=height,
            sh_degree=3,
            render_mode=render_mode,
            feature=feature,
            backgrounds=backgrounds
        )


        render_tab_state.total_gs_count = gaussians.dynamic_splats.params["means"].shape[0] + gaussians.static_splats.params["means"].shape[0]
        render_tab_state.rendered_gs_count = (rendered["info"]["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)", "depth(median)"]:
            if render_tab_state.render_mode == "depth(median)":
                depth = rendered["median_depth"]
            else:
                depth = rendered["colors"]
            # normalize depth to [0, 1]
            depth_norm = batch_normalize(depth)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm[0, 0], render_tab_state.colormap)
                .cpu()
                .numpy()
            )

        elif render_tab_state.render_mode in ["normal(surface)", "normal"]:
            if render_tab_state.render_mode == "normal(surface)":
                normal = rendered["normals_from_depth"][0]
            else:
                normal = rendered["normals"][0, 0]
            normal = normal * 0.5 + 0.5  # normalize to [0, 1]
            renders = normal.cpu().numpy()

        elif render_tab_state.render_mode == "alpha":
            alpha = rendered["alphas"][0, 0]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        else:
            render_colors = rendered["colors"]
            if feature == "velocity":
                velocity_norm = batch_normalize(torch.norm(render_colors, dim=-1, keepdim=True))
                if render_tab_state.render_mode == "velocity(norm)":
                    render_colors = velocity_norm
                else:
                    render_colors = render_colors / (velocity_norm + 1e-10) * 0.5 + 0.5
            # elif feature in ["lifespan", "nonrigidity"]:
            elif feature == "lifespan":
                render_colors = batch_normalize(render_colors)
            render_colors = render_colors[0, 0].clamp(0.0, 1.0)
            if render_colors.shape[-1] == 1:
                render_colors = apply_float_colormap(render_colors, render_tab_state.colormap)
            renders = render_colors.cpu().numpy()
        return renders

    server = viser.ViserServer(port=args.port, verbose=False, host="0.0.0.0")
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.0, 0.0, 0.0)
        client.camera.look_at = (0.0, 0.0 ,1.0)
        client.camera.up_direction = (0.0, -1.0, 0.0)

    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
        max_time=max_time,
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, help="path to the .pt file"
    )
    parser.add_argument(
        "--no_2dgs", action="store_true", help="whether to use 2dgs renderer"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--output_dir", type=str, default="misc", help="where to dump outputs"
    )
    
    args = parser.parse_args()

    cli(main, args, verbose=True)