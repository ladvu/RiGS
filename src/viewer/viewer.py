import viser
from pathlib import Path
from typing import Tuple, Callable, Literal
from nerfview import Viewer, RenderTabState
import viser
from pathlib import Path
from typing import Dict, List, Optional
from utils import populate_general_render_tab
import dataclasses

@dataclasses.dataclass
class RenderTabState:
    """Useful GUI handles exposed by the render tab."""

    num_train_rays_per_sec: Optional[float] = None
    num_view_rays_per_sec: float = 100000.0
    preview_render: bool = False
    preview_fov: float = 0.0
    preview_time: float = 0.0
    preview_aspect: float = 1.0
    viewer_res: int = 518
    render_width: int = 518
    render_height: int = 518


class GsplatRenderTabState(RenderTabState):
    # non-controlable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # controlable parameters
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "rgb",
        "depth(accumulated)",
        "depth(expected)",
        "depth(median)",
        "alpha",
        "normal(surface)",
        "normal",
        "velocity",
        "velocity(norm)",
        "lifespan",
        # "conf",
        # "nonrigidity"
    ] = "rgb"
    inverse: bool = False
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    timestamp: float = 1.0
    mode:Literal["all", "static", "rigid", "transient", "dynamic"] = "all"


class GsplatViewer(Viewer):
    """
    Viewer for gsplat 2dgs.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        max_time: float = 1.0,
    ):
        self.max_time = max_time
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat 2dgs viewer")

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")
    
    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            with server.gui.add_folder("Gaussian"):
                total_gs_count_number = server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                rendered_gs_count_number = server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )

                timestamp_number = server.gui.add_slider(
                    "Time",
                    min=0.0,
                    max=self.max_time,
                    initial_value=0.0,
                    step=1.0,
                )

                @timestamp_number.on_update
                def _(_) -> None:
                    self.render_tab_state.timestamp = timestamp_number.value
                    self.rerender(_)

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    ("rgb",
                     "depth(accumulated)",
                     "depth(expected)",
                     "depth(median)",
                     "alpha",
                     "normal(surface)",
                     "normal",
                     "velocity",
                     "velocity(norm)",
                     "lifespan",
                    #  "conf"
                    #  "nonrigidity"
                    ),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)


                inverse_checkbox = server.gui.add_checkbox(
                    "Inverse",
                    initial_value=self.render_tab_state.inverse,
                    disabled=True,
                    hint="Inverse the depth.",
                )

                @inverse_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.inverse = inverse_checkbox.value
                    self.rerender(_)

                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for rendering depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
                    self.rerender(_)

                mode_button = server.gui.add_dropdown(
                    "mode",
                    ("all", "static", "rigid", "transient", "dynamic"),
                    initial_value=self.render_tab_state.mode,
                    hint="mode",
                )

                @mode_button.on_update
                def _(_) -> None:
                    self.render_tab_state.mode = mode_button.value
                    self.rerender(_)

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "render_mode_dropdown": render_mode_dropdown,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
                "timestamp_number": timestamp_number,
            }
        )

        # training tab handles should also be disabled during dumping video.
        extra_handles = self._rendering_tab_handles.copy()
        if self.mode == "training":
            extra_handles.update(self._training_tab_handles)
        handles = populate_general_render_tab(
            self.server,
            output_dir=self.output_dir,
            folder=self._rendering_folder,
            render_tab_state=self.render_tab_state,
            extra_handles=extra_handles,
            time_enabled=True,
        )
        self._rendering_tab_handles.update(handles)

    def _after_render(self):
        # Update the GUI elements with current values
        self._rendering_tab_handles[
            "total_gs_count_number"
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            "rendered_gs_count_number"
        ].value = self.render_tab_state.rendered_gs_count