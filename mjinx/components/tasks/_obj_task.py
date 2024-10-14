"""Frame task implementation."""

from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx
from jaxlie import SE3, SO3

from mjinx.components.tasks._base import JaxTask, Task
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxObjTask(JaxTask):
    """
    A JAX-based implementation of an object task for inverse kinematics.

    This class serves as a base for tasks that are applied to specific objects
    (bodies, geometries, or sites) in the robot model.

    :param obj_id: The ID of the object (body, geometry, or site) to which the task is applied.
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


AtomicObjTaskType = TypeVar("AtomicObjTaskType", bound=JaxObjTask)


class ObjTask(Generic[AtomicObjTaskType], Task[AtomicObjTaskType]):
    """
    A high-level representation of an object task for inverse kinematics.

    This class provides an interface for creating and manipulating tasks
    that are applied to specific objects (bodies, geometries, or sites) in the robot model.

    :param name: The name of the task.
    :param cost: The cost associated with the task.
    :param gain: The gain for the task.
    :param obj_name: The name of the object (body, geometry, or site) to which the task is applied.
    :param obj_type: The type of the object (mjOBJ_BODY, mjOBJ_GEOM, or mjOBJ_SITE).
    :param gain_fn: A function to compute the gain dynamically.
    :param lm_damping: The Levenberg-Marquardt damping factor.
    :param mask: A sequence of integers to mask certain dimensions of the task.
    """

    JaxComponentType: type = JaxObjTask
    _obj_name: str
    _obj_id: int
    _obj_type: mj.mjtObj

    def __init__(
        self,
        name: str,
        cost: ArrayOrFloat,
        gain: ArrayOrFloat,
        obj_name: str,
        obj_type: mj.mjtObj = mj.mjtObj.mjOBJ_BODY,
        gain_fn: Callable[[float], float] | None = None,
        lm_damping: float = 0,
        mask: Sequence[int] | None = None,
    ):
        super().__init__(name, cost, gain, gain_fn, lm_damping, mask)
        self._obj_name = obj_name
        self._obj_type = obj_type
        self._obj_id = -1

    @property
    def obj_name(self) -> str:
        """
        Get the name of the object to which the task is applied.

        :return: The name of the object.
        """
        return self._obj_name

    @property
    def obj_id(self) -> int:
        """
        Get the ID of the object to which the task is applied.

        :return: The ID of the object.
        """
        return self._obj_id

    @property
    def obj_type(self) -> mj.mjtObj:
        """
        Get the type of the object associated with the task.

        :return: The MuJoCo object type (mjOBJ_BODY, mjOBJ_GEOM, or mjOBJ_SITE).
        """
        return self._obj_type

    def update_model(self, model: mjx.Model):
        """
        Update the MuJoCo model and set the object ID for the task.

        This method is called when the model is updated or when the task
        is first added to the problem.

        :param model: The new MuJoCo model.
        :raises ValueError: If the object with the specified name is not found in the model.
        """
        self._obj_id = mjx.name2id(
            model,
            self._obj_type,
            self._obj_name,
        )
        if self._obj_id == -1:
            raise ValueError(f"object with type {self._obj_type} and name {self._obj_name} is not found.")

        return super().update_model(model)
