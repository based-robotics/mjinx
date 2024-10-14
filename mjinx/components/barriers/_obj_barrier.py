"""Frame task implementation."""

from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

import jax.numpy as jnp  # noqa: F401
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
from jaxlie import SE3, SO3

from mjinx.components.barriers._base import Barrier, JaxBarrier
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxObjBarrier(JaxBarrier):
    """
    A JAX implementation of an object-specific barrier function.

    This class extends JaxBarrier to provide barrier functions that are
    specific to a particular object (body, geometry, or site) in the robot model.

    :param obj_id: The ID of the object (body, geometry, or site) to which this barrier applies.
    :param obj_type: The type of the object (mjOBJ_BODY, mjOBJ_GEOM, or mjOBJ_SITE).
    """

    obj_id: jdc.Static[int]
    obj_type: jdc.Static[mj.mjtObj]

    def get_pos(self, data: mjx.Data) -> jnp.ndarray:
        """
        Get the position of the object in the world frame.

        This method returns the position based on the object type (body, geom, or site).

        :param data: The MuJoCo simulation data.
        :return: A 3D vector representing the object's position in the world frame.
        """
        match self.obj_type:
            case mj.mjtObj.mjOBJ_GEOM:
                return data.geom_xpos[self.obj_id]
            case mj.mjtObj.mjOBJ_SITE:
                return data.site_xpos[self.obj_id]
            case _:  # default -- mjOBJ_BODY:
                return data.xpos[self.obj_id]

    def get_rotation(self, data: mjx.Data) -> SO3:
        """
        Get the rotation of the object in the world frame.

        This method returns the rotation based on the object type (body, geom, or site).

        :param data: The MuJoCo simulation data.
        :return: An SO3 object representing the object's rotation in the world frame.
        """
        match self.obj_type:
            case mj.mjtObj.mjOBJ_GEOM:
                return SO3.from_matrix(data.geom_xmat[self.obj_id])
            case mj.mjtObj.mjOBJ_SITE:
                return SO3.from_matrix(data.site_xmat[self.obj_id])
            case _:  # default -- mjOBJ_BODY:
                return SO3.from_matrix(data.xmat[self.obj_id])

    def get_frame(self, data: mjx.Data) -> SE3:
        """
        Get the full pose (position and rotation) of the object in the world frame.

        This method combines the position and rotation to return a complete pose.

        :param data: The MuJoCo simulation data.
        :return: An SE3 object representing the object's pose in the world frame.
        """
        return SE3.from_rotation_and_translation(self.get_rotation(data), self.get_pos(data))


AtomicObjBarrierType = TypeVar("AtomicObjBarrierType", bound=JaxObjBarrier)


class ObjBarrier(Generic[AtomicObjBarrierType], Barrier[AtomicObjBarrierType]):
    """
    A generic object barrier class that wraps atomic object barrier implementations.

    This class provides a high-level interface for object-specific barrier functions,
    which can be applied to bodies, geometries, or sites in the robot model.

    :param name: The name of the barrier.
    :param gain: The gain for the barrier function.
    :param obj_name: The name of the object (body, geometry, or site) to which this barrier applies.
    :param obj_type: The type of the object (mjOBJ_BODY, mjOBJ_GEOM, or mjOBJ_SITE).
    :param gain_fn: A function to compute the gain dynamically.
    :param safe_displacement_gain: The gain for computing safe displacements.
    :param mask: A sequence of integers to mask certain dimensions.
    """

    _obj_name: str
    _obj_id: int
    _obj_type: mj.mjtObj

    def __init__(
        self,
        name: str,
        gain: ArrayOrFloat,
        obj_name: str,
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        gain_fn: Callable[[float], float] | None = None,
        safe_displacement_gain: float = 0,
        mask: Sequence[int] | None = None,
    ):
        """
        Initialize the ObjBarrier object.

        :param name: The name of the barrier.
        :param gain: The gain for the barrier function.
        :param obj_name: The name of the object to which this barrier applies.
        :param gain_fn: A function to compute the gain dynamically.
        :param safe_displacement_gain: The gain for computing safe displacements.
        :param mask: A sequence of integers to mask certain dimensions.
        """
        super().__init__(name, gain, gain_fn, safe_displacement_gain, mask)
        self._obj_name = obj_name
        self._obj_type = obj_type
        self._obj_id = -1

    @property
    def obj_name(self) -> str:
        """
        Get the name of the object to which this barrier applies.

        :return: The name of the object.
        """
        return self._obj_name

    @property
    def obj_id(self) -> int:
        """
        Get the ID of the body to which this barrier applies.

        :return: The ID of the body.
        :raises ValueError: If the body ID is not available.
        """

        if self._obj_id == -1:
            raise ValueError("body_id is not available until model is provided.")
        return self._obj_id

    @property
    def obj_type(self) -> mj.mjtObj:
        """
        Get the type of the object associated with the barrier..

        :return: The MuJoCo object type (mjOBJ_BODY, mjOBJ_GEOM, or mjOBJ_SITE).
        """
        return self._obj_type

    def update_model(self, model: mjx.Model):
        """
        Update the model and retrieve the object ID.

        :param model: The MuJoCo model.
        :return: The updated model.
        :raises ValueError: If the object with the specified name is not found.
        """

        self._obj_id = mjx.name2id(
            model,
            self._obj_type,
            self._obj_name,
        )
        if self._obj_id == -1:
            raise ValueError(f"object with type {self._obj_type} and name {self._obj_name} is not found.")

        return super().update_model(model)
