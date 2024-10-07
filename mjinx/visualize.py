from collections.abc import Sequence
from typing import NamedTuple

import mujoco as mj
import numpy as np
from dm_control import mjcf
from mujoco import viewer

from mjinx.typing import ndarray


class MarkerData(NamedTuple):
    """
    A named tuple for storing marker data.

    :param size: The size of the marker.
    :param rgba: The RGBA color values of the marker.
    """

    size: float
    rgba: np.ndarray


class BatchVisualizer:
    def __init__(self, model_path: str, n_models: int, geom_group: int = 2, alpha: float = 0.5):
        """
        A class for batch visualization of multiple models using MuJoCo.

        This class allows for the visualization of multiple instances of a given model,
        with customizable transparency and marker options.

        :param model_path: Path to the MuJoCo model file.
        :param n_models: Number of model instances to visualize.
        :param geom_group: Geometry group to render, defaults to 2.
        :param alpha: Transparency value for the models, defaults to 0.5.
        """
        self.n_models = n_models

        # Generate the model, by stacking several provided models
        self.mj_model = self._generate_mj_model(model_path, n_models, geom_group, alpha)
        self.mj_data = mj.MjData(self.mj_model)

        # Initializing visualization
        self.renderer = mj.Renderer(self.mj_model)
        self.mj_viewer = viewer.launch_passive(
            self.mj_model,
            self.mj_data,
            show_left_ui=False,
            show_right_ui=False,
        )
        # For markers
        self.n_markers: int = 0
        self.markers_data: list[MarkerData] = []

    def _generate_mj_model(self, model_path: str, n_models: int, geom_group: int, alpha: float) -> mj.MjModel:
        """
        Generate a combined MuJoCo model from multiple instances of the given model.

        :param model_path: Path to the MuJoCo model file.
        :param n_models: Number of model instances to combine.
        :param geom_group: Geometry group to render.
        :param alpha: Transparency value for the models.
        :return: The generated MuJoCo model.
        """

        mjcf_model = mjcf.RootElement()
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
        self.renderer.scene.ngeom += n_markers
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
        Update the model positions.

        :param q: Array of joint positions for all model instances.
        """
        q_raveled = q.ravel()

        self.mj_data.qpos = q_raveled
        mj.mj_fwdPosition(self.mj_model, self.mj_data)

    def visualize(self, markers: ndarray | None = None):
        """
        Visualize the current state of the models and markers.

        :param markers: Array of marker positions. If None, no markers are displayed.
        """
        if markers is not None:
            if markers.ndim == 1:
                markers = markers.reshape(1, -1)
            for i in range(len(markers)):
                size, rgba = self.markers_data[i].size, self.markers_data[i].rgba
                mj.mjv_initGeom(
                    self.mj_viewer.user_scn.geoms[i],
                    mj.mjtGeom.mjGEOM_SPHERE,
                    size * np.ones(3),
                    markers[i],
                    np.eye(3).flatten(),
                    rgba,
                )
        self.mj_viewer.sync()
