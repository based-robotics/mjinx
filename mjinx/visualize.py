import os
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import mujoco as mj
import numpy as np
from mujoco import viewer

from mjinx.typing import ArrayOrFloat, ndarray

try:
    import mediapy
    from dm_control import mjcf
except ImportError as e:
    raise ImportError("visualization is not supported, please install the mjinx[visual]") from e


@dataclass
class MarkerData:
    """
    A dataclass for storing marker data.

    :param name: The name of the marker.
    :param id: The unique identifier of the marker.
    :param type: The type of geometry for the marker.
    :param size: The size of the marker.
    :param pos: The position of the marker in 3D space. Defaults to [0, 0, 0].
    :param rot: The rotation of the marker. Defaults to [1, 0, 0, 0] (identity quaternion).
    :param rgba: The RGBA color values of the marker. Defaults to [0.5, 0.5, 0.5, 0.3].
    """

    name: str
    id: int
    type: mj.mjtGeom
    size: np.ndarray
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    rgba: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.3]))

    @property
    def rot_matrix(self) -> np.ndarray:
        """
        Returns a raveled rotation matrix, generated from orientation.

        It checks the ndim of the rot field. If ndim=1, it assumes quaternion
        (scalar first) is passed, if ndim=2, it assumes that rotation matrix is
        passed.

        :return: Raveled rotation matrix with shape (9,).

        :raises ValueError: If the rotation data has invalid dimensions or length.
        """

        match self.rot.ndim:
            case 1:
                if len(self.rot) != 4:
                    raise ValueError(
                        "invalid length of 1D marker rotation: "
                        f"expected scalar-first quaternion with shape (4, ), got {len(self.rot)} "
                    )
                w, x, y, z = self.rot
                rot_matrix = np.array(
                    [
                        [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                        [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
                        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
                    ]
                )
                return rot_matrix.ravel()
            case 2:
                return self.rot.ravel()

            case _:
                raise ValueError(f"wrong ndim of the self.rot, expected 1 <= self.rot.ndim <= 2, got {self.rot.ndim}")

        return np.array(0)


class BatchVisualizer:
    """
    A class for batch visualization of multiple MuJoCo model instances.

    This class allows for the visualization of multiple instances of a given model,
    with customizable transparency and marker options. It also supports recording
    the visualization as a video.

    :param model_path: Path to the MuJoCo model file.
    :param n_models: Number of model instances to visualize.
    :param geom_group: Geometry group to render, defaults to 2.
    :param alpha: Transparency value for the models, defaults to 0.5.
    :param record: If True, records and saves mp4 scene recording, defaults to False.
    :param filename: Name of the file to save without extension, defaults to current datetime.
    :param record_res: Resolution of recorded video (width, height), defaults to (1024, 1024).
    """

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
        self.marker_data: dict[str, MarkerData] = {}

        # Recording the visualization
        self.record = record
        self.filename = filename
        self.frames: list = []
        if self.record:
            self.mj_renderer = mj.Renderer(self.mj_model, width=record_res[0], height=record_res[1])

    def __find_asset(self, asset_root: str, asset_name: str) -> str:
        """
        Find the full path of an asset file within a given root directory.

        :param asset_root: The root directory to search in.
        :param asset_name: The name of the asset file to find.
        :return: The full path to the asset file.
        :raises ValueError: If the asset is not found in the given root directory.
        """

        for root, _, files in os.walk(asset_root):
            if asset_name in files:
                return os.path.join(root, asset_name)
        raise ValueError(f"asset {asset_name} not found in {asset_root}")

    def remove_high_level_body_tags(self, mjcf_str: str, model_directory: str) -> tuple[str, dict[str, bytes]]:
        """
        Remove high-level body tags from the MJCF XML string and process mesh assets.

        This method modifies the XML structure by moving children of high-level body tags
        directly under the worldbody tag. It also processes mesh elements, updating their
        file attributes and collecting asset data.

        Note that this solution is obviously very hacky and dirty, however the author
        considers this easier and (barely) enough to complete the required task.
        The attempts to solve this using dm_control failed, see: https://github.com/google-deepmind/dm_control/issues/407

        :param mjcf_str: The MJCF XML string to process.
        :param model_directory: The directory containing the model and its assets.
        :return: A tuple containing the modified XML string and a dictionary of asset data.
        """
        # Parse the XML string
        root = ET.fromstring(mjcf_str)

        # Find the worldbody element
        worldbody = root.find(".//worldbody")

        if worldbody is not None:
            # Iterate through direct children of worldbody
            for child in list(worldbody):
                if child.tag == "body":
                    # This is a high-level body tag
                    # Move its children to worldbody and remove it
                    for subchild in list(child):
                        worldbody.append(subchild)
                    worldbody.remove(child)

        # Try to find and load meshes from the xml file directory
        mesh_elements = root.findall(".//mesh")
        assets: dict[str, bytes] = {}
        for mesh in mesh_elements:
            file_attr = mesh.get("file")
            if file_attr:
                # Remove the hash from the file attribute
                new_file_attr = file_attr[: file_attr.find("-")] + file_attr[file_attr.rfind(".") :]

                asset_path = self.__find_asset(model_directory, new_file_attr)
                with open(asset_path, "rb") as f:
                    assets[new_file_attr] = f.read()

                mesh.set("file", new_file_attr)
        # Convert the modified XML tree back to a string
        modified_xml = ET.tostring(root, encoding="unicode")

        return modified_xml, assets

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
                for camera in attached_mjcf_model.find_all("camera"):
                    camera.remove()
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

        edited_xml, assets = self.remove_high_level_body_tags(
            mjcf_model.to_xml_string(),
            os.path.split(model_path)[0],
        )
        # Build and return mujoco model
        return mj.MjModel.from_xml_string(edited_xml, assets)

    def add_markers(
        self,
        name: str | Sequence[str],
        size: ArrayOrFloat,
        marker_alpha: float,
        color_begin: np.ndarray,
        color_end: np.ndarray | None = None,
        marker_type: mj.mjtGeom = mj.mjtGeom.mjGEOM_SPHERE,
        n_markers: int = 1,
    ):
        """
        Add markers to the visualization.

        :param name: Name or sequence of names for the markers.
        :param size: Size of the markers. Can be a single float or an array of size 3.
        :param marker_alpha: Transparency of the markers.
        :param color_begin: Starting color for marker interpolation.
        :param color_end: Ending color for marker interpolation. If None, uses `color_begin`.
        :param marker_type: Type of marker geometry, defaults to sphere.
        :param n_markers: Amount of markers to add. Defaults to 1.
        """
        if n_markers > 1 and (isinstance(name, str) or len(name) != n_markers):
            raise ValueError(f"list of n_marker ({n_markers}) names is required.")

        if color_end is None:
            color_end = color_begin

        size_array = size if isinstance(size, np.ndarray) else np.ones(3) * size

        for i, interp_coef in enumerate(np.linspace(0, 1, n_markers)):
            # Interpolate the color
            name_i = name if isinstance(name, str) else name[i]
            color = interp_coef * color_begin + (1 - interp_coef) * color_end

            self.marker_data[name_i] = MarkerData(
                name=name_i,
                id=self.n_markers + i,
                type=marker_type,
                size=size_array,
                rgba=np.array([*color, marker_alpha]),
            )
        self.n_markers += n_markers
        self.mj_viewer.user_scn.ngeom += n_markers
        if self.record:
            self.mj_renderer.scene.ngeom += n_markers

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

        self._draw_markers(self.mj_viewer.user_scn)

        if self.record:
            self.mj_renderer.update_scene(self.mj_data, scene_option=self.mj_viewer._opt, camera=self.mj_viewer._cam)
            self._draw_markers(self.mj_renderer.scene)

            rendered_frame = self.mj_renderer.render()
            self.frames.append(rendered_frame)

        self.mj_viewer.sync()

    def _draw_markers(self, scene: mj.MjvScene):
        """
        Draw markers on the given scene.

        :param scene: The MjvScene to draw markers on.
        :param markers: Array of marker positions.
        """
        for marker in self.marker_data.values():
            mj.mjv_initGeom(
                scene.geoms[scene.ngeom - self.n_markers + marker.id],
                marker.type,
                marker.size,
                marker.pos,
                marker.rot_matrix,
                marker.rgba,
            )

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
