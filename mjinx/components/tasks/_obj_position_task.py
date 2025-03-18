"""Frame task implementation."""

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import jax_dataclasses as jdc
import mujoco as mj
import mujoco.mjx as mjx

from mjinx.components.tasks._obj_task import JaxObjTask, ObjTask
from mjinx.configuration._model import get_frame_jacobian_world_aligned
from mjinx.typing import ArrayOrFloat


@jdc.pytree_dataclass
class JaxPositionTask(JaxObjTask):
    """
    A JAX-based implementation of a position task for inverse kinematics.

    This class represents a task that aims to achieve a specific target position
    for an object (body, geometry, or site) in the robot model.

    The task function maps joint positions to the object's position in the world frame:

    .. math::

        f(q) = p(q)

    where :math:`p(q)` is the position of the object in world coordinates.

    The error is computed as the difference between the current and target positions:

    .. math::

        e(q) = p(q) - p_{target}

    The Jacobian of this task is the object's position Jacobian, which relates changes in
    joint positions to changes in the object's position.

    :param target_pos: The target position to be achieved.
    """

    target_pos: jnp.ndarray

    def __call__(self, data: mjx.Data) -> jnp.ndarray:
        """
        Compute the error between the current position and the target position.

        The error is given by:

        .. math::

            e(q) = p(q) - p_{target}

        :param data: The MuJoCo simulation data.
        :return: The error vector representing the difference between the current and target positions.
        """
        return self.get_pos(data)[self.mask_idxs,] - self.target_pos

    def compute_jacobian(self, data):
        return get_frame_jacobian_world_aligned(self.model, data, self.obj_id, self.obj_type).T[self.mask_idxs,]


class PositionTask(ObjTask[JaxPositionTask]):
    """
    A high-level representation of a position task for inverse kinematics.

    This class provides an interface for creating and manipulating position tasks,
    which aim to achieve a specific target position for an object in the robot model.

    :param name: The name of the task.
    :param cost: The cost associated with the task.
    :param gain: The gain for the task.
    :param obj_name: The name of the object to which the task is applied.
    :param gain_fn: A function to compute the gain dynamically.
    :param lm_damping: The Levenberg-Marquardt damping factor.
    :param mask: A sequence of integers to mask certain dimensions of the task.
    """

    JaxComponentType: type = JaxPositionTask
    _target_pos: jnp.ndarray

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
        super().__init__(name, cost, gain, obj_name, obj_type, gain_fn, lm_damping, mask)
        self._dim = 3 if mask is None else len(self.mask_idxs)
        self._target_pos = jnp.zeros(self._dim)

    @property
    def target_pos(self) -> jnp.ndarray:
        """
        Get the current target position for the task.

        :return: The current target position as a numpy array.
        """
        return self._target_pos

    @target_pos.setter
    def target_pos(self, value: Sequence):
        """
        Set the target position for the task.

        :param value: The new target position as a sequence of values.
        """
        self.update_target_pos(value)
        self._jax_component = jdc.replace(self._jax_component, target_pos=self._target_pos)

    def update_target_pos(self, target_pos: Sequence):
        """
        Update the target position for the task.

        This method allows setting the target position using a sequence of values.

        :param target_pos: The new target position as a sequence of values.
        :raises ValueError: If the provided sequence doesn't have the correct length.
        """
        target_pos_array = jnp.array(target_pos)
        if target_pos_array.shape[-1] != self._dim:
            raise ValueError(
                f"Invalid dimension of the target position: expected {self._dim}, got {target_pos_array.shape[-1]}"
            )
        self._target_pos = target_pos_array
