import warnings
from datetime import datetime
from typing import NamedTuple

import mujoco as mj
import numpy as np
from mujoco import viewer

from mjinx.typing import ndarray

try:
    import mediapy
    from dm_control import mjcf
except ImportError as e:
    raise ImportError("visualization is not supported, please install the mjinx[visual]") from e


class MarkerData(NamedTuple):
    """
    A named tuple for storing marker data.

    :param size: The size of the marker.
    :param rgba: The RGBA color values of the marker.
    """

    size: float
    rgba: np.ndarray


class BatchVisualizer:
    def __init__(
        self,
        model_path: str,
        n_models: int,
        geom_group: int = 2,
        alpha: float = 0.5,
        record: bool = False,
        filename: str = "",
        record_res: tuple[int, int] = (1024, 1024),
    ):
        """
        A class for batch visualization of multiple models using MuJoCo.

        This class allows for the visualization of multiple instances of a given model,
        with customizable transparency and marker options. It also supports recording
        the visualization as a video.

        :param model_path: Path to the MuJoCo model file.
        :param n_models: Number of model instances to visualize.
        :param geom_group: Geometry group to render, defaults to 2.
        :param alpha: Transparency value for the models, defaults to 0.5.
        :param record: if True, records and saves mp4 scene recording, defaults to False.
        :param filename: name of the file to save the file without extension, defaults to datetime.
        :param record_res: resolution of recorded video (width, height), defaults to (1024, 1024).
        """
        self.n_models = n_models

        # Generate the model, by stacking several provided models
        self.mj_model = self._generate_mj_model(model_path, n_models, geom_group, alpha, record_res)
        self.mj_data = mj.MjData(self.mj_model)

        # Initializing visualization
        self.mj_viewer = viewer.launch_passive(
            self.mj_model,
            self.mj_data,
            show_left_ui=False,
            show_right_ui=False,
        )
        # For markers
        self.n_markers: int = 0
        self.markers_data: list[MarkerData] = []

        # Recording the visualization
        self.record = record
        self.filename = filename
        self.frames: list = []
        if self.record:
            self.mj_renderer = mj.Renderer(self.mj_model, width=record_res[0], height=record_res[1])

    def _generate_mj_model(
        self,
        model_path: str,
        n_models: int,
        geom_group: int,
        alpha: float,
        off_res: tuple[int, int],
    ) -> mj.MjModel:
        """
        Generate a combined MuJoCo model from multiple instances of the given model.

        :param model_path: Path to the MuJoCo model file.
        :param n_models: Number of model instances to combine.
        :param geom_group: Geometry group to render.
        :param alpha: Transparency value for the models.
        :param off_res: Resolution (width, height,) for the rendering.
        :return: The generated MuJoCo model.
        """

        mjcf_model = mjcf.RootElement()

        # Add white sky
        skybox = mjcf_model.asset.add("texture")
        skybox.name = "skybox"
        skybox.type = "skybox"
        skybox.width = 512
        skybox.height = 3072
        skybox.rgb1 = np.ones(3)
        skybox.rgb2 = np.ones(3)
        skybox.builtin = "flat"

        # Attach all models together
        for i in range(n_models):
            # Compute model prefix
            prefix = self.get_prefix(i)

            # Load the model
            attached_mjcf_model = mjcf.from_path(model_path)
            attached_mjcf_model.model = prefix
            if i > 0:
                for light in attached_mjcf_model.find_all("light"):
                    light.remove()
            # Attach the model
            site = mjcf_model.worldbody.add("site")
            site.attach(attached_mjcf_model)

        # Change color in all material settings
        for material in mjcf_model.find_all("material"):
            if material.rgba is not None:
                material.rgba[3] *= alpha

        # Change color and collision properties for all geometries
        for g in mjcf_model.find_all("geom"):
            # Removes geometries not from the provided geometry group
            # Discards collision geometries etc.

            # Determine the geometry group
            g_group = g.group

            if g_group is None and g.dclass is not None:
                g_group = g.dclass.geom.group

            # Delete the geometry, if it belongs to another group
            # Keep the geometry, if group is not specified
            if g_group is not None and g_group != geom_group:
                g.remove()
                continue

            # Disable collision for all present geometries
            g.contype = 0
            g.conaffinity = 0

            # Reduce transparency of the original model
            if g.rgba is not None:
                g.rgba[3] *= alpha
            elif g.dclass is not None and g.dclass.geom.rgba is not None:
                g.dclass.geom.rgba[3] *= alpha

        # Removing all existing keyframes, since they are invalid
        keyframe = mjcf_model.keyframe
        for child in keyframe.all_children():
            keyframe.remove(child)

        # Remove all exclude contact pairs
        mjcf_model.contact.remove(True)
        mjcf_model.visual.__getattr__("global").offwidth = off_res[0]
        mjcf_model.visual.__getattr__("global").offheight = off_res[1]

        # Build and return mujoco model
        return mjcf.Physics.from_mjcf_model(mjcf_model).model._model

    def add_markers(
        self, size: float, marker_alpha: float, color_begin: np.ndarray, color_end: np.ndarray, n_markers: int = 0
    ):
        """
        Add markers to the visualization.

        :param size: Size of the markers.
        :param marker_alpha: Transparency of the markers.
        :param color_begin: Starting color for marker interpolation.
        :param color_end: Ending color for marker interpolation.
        :param n_markers: Amount of markers to add. Defaults to number of the models in the batch.
        """
        if n_markers < 1:
            n_markers = self.n_models
        self.mj_renderer.scene.ngeom += n_markers
        self.mj_viewer.user_scn.ngeom += n_markers

        for interp_coef in np.linspace(0, 1, n_markers):
            # Interpolate the color
            color = interp_coef * color_begin + (1 - interp_coef) * color_end

            self.markers_data.append(
                MarkerData(
                    size=size,
                    rgba=np.array([*color, marker_alpha]),
                )
            )
        self.n_markers += n_markers

    def get_prefix(self, i: int) -> str:
        """
        Generate a prefix for the i-th model instance.

        :param i: Index of the model instance.
        :return: Prefix string for the model.
        """
        return f"manip{i}"

    def update(self, q: ndarray):
        """
        Update the model positions and record frame if enabled.

        :param q: Array of joint positions for all model instances.
        """
        q_raveled = q.ravel()

        self.mj_data.qpos = q_raveled
        mj.mj_fwdPosition(self.mj_model, self.mj_data)

    def _draw_markers(self, scene: mj.MjvScene, markers: ndarray):
        """
        Draw markers on the given scene.

        :param scene: The MjvScene to draw markers on.
        :param markers: Array of marker positions.
        """
        for i in range(len(markers)):
            size, rgba = self.markers_data[i].size, self.markers_data[i].rgba
            mj.mjv_initGeom(
                scene.geoms[scene.ngeom - self.n_markers + i],
                mj.mjtGeom.mjGEOM_SPHERE,
                size * np.ones(3),
                markers[i],
                np.eye(3).flatten(),
                rgba,
            )

    def visualize(self, markers: ndarray | None = None):
        """
        Visualize the current state of the models and markers.

        :param markers: Array of marker positions. If None, no markers are displayed.
        """
        if markers is not None:
            if markers.ndim == 1:
                markers = markers.reshape(1, -1)
            self._draw_markers(self.mj_viewer.user_scn, markers)

        if self.record:
            self.mj_renderer.update_scene(self.mj_data, scene_option=self.mj_viewer._opt, camera=self.mj_viewer._cam)
            if markers is not None:
                self._draw_markers(self.mj_renderer.scene, markers)

            rendered_frame = self.mj_renderer.render()
            self.frames.append(rendered_frame)

        self.mj_viewer.sync()

    def save_video(self, fps: float):
        """
        Save the recorded frames as an MP4 video.

        :param fps: Frames per second for the output video.
        """
        if not self.record:
            warnings.warn("failed to save the video, it was not recorded", stacklevel=2)
            return
        filename = (
            self.filename + ".mp4"
            if len(self.filename) != 0
            else "{}.mp4".format(datetime.now().strftime("%H-%M_%d-%m-%Y"))
        )

        mediapy.write_video(
            filename,
            self.frames,
            fps=fps,
        )

    def close(self):
        """
        Close the viewer and clean up resources.
        """
        self.mj_viewer.close()
        if self.record:
            del self.frames
            self.mj_renderer.close()
